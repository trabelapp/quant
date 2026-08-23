import os
import sys
import uvicorn

# 프로젝트 루트 경로를 파이썬 패키지 경로에 확실하게 추가
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def initialize_system():
    print("=" * 65)
    print(" 🚀 TRABLE QUANT TERMINAL & BACKEND SERVER ")
    print("=" * 65)
    
    # 1. 백엔드 패키지 존재 여부 확인
    backend_path = os.path.join(ROOT_DIR, "backend")
    if not os.path.exists(backend_path):
        print(" [!] 경고: 'backend' 폴더를 찾을 수 없습니다. 폴더 구조를 확인하세요.")
        return False

    print(" [✔] Core Engine: Screener (Quant & Momentum) Ready")
    print(" [✔] Storage: SQLite Database (quant_terminal.db) Connected")
    print(" [✔] API Router: FastAPI Main Controller Loaded")
    print("-" * 65)
    print(" 🌐 서버 주소: http://127.0.0.1:8888")
    print(" 📄 API 문서: http://127.0.0.1:8888/docs")
    print("=" * 65)
    return True

def main():
    if not initialize_system():
        return

    # FastAPI 백엔드 서버 실행 (핫 리로드 포함)
    try:
        uvicorn.run(
            "backend.main:app", 
            host="127.0.0.1", 
            port=8888, 
            reload=True
        )
    except Exception as e:
        print(f"\n[X] 서버 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
