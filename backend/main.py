from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.screener import run_quant_scanner
from backend.database import init_db, load_stocks_from_db
import threading
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)  # <--- 여기 괄호 모양 수정 완료!

CACHE = {"data": []}

def background_scanner():
    """백그라운드에서 주기적으로 안전하게 DB를 최신화"""
    while True:
        try:
            print("[SYSTEM] Background DB Sync Started...")
            run_quant_scanner()
            print("[SYSTEM] Background DB Sync Completed.")
        except Exception as e:
            print(f"[SYSTEM] Background Sync Error: {e}")
        # 3시간마다 갱신
        time.sleep(10800)

@app.on_event("startup")
def startup_event():
    init_db()
    existing = load_stocks_from_db()
    if not existing:
        print("[SYSTEM] Local DB is empty. Running initial scan...")
        threading.Thread(target=run_quant_scanner, daemon=True).start()
    
    thread = threading.Thread(target=background_scanner, daemon=True)
    thread.start()

@app.get("/")
async def root():
    return {"message": "Elite Terminal Database-Driven Backend is Running."}

@app.get("/api/scan")
async def get_scan_results():
    results = load_stocks_from_db()
    return {"results": results}
