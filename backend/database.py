import sqlite3
import json
import os

DB_PATH = "backend/quant_terminal.db"

def init_db():
    os.makedirs("backend", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alpha_stocks (
            ticker TEXT PRIMARY KEY,
            price REAL,
            change REAL,
            mom_score REAL,
            value_score REAL,
            total_score REAL,
            history TEXT,
            news TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_stocks_to_db(stocks: list):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for s in stocks:
        cursor.execute("""
            INSERT OR REPLACE INTO alpha_stocks 
            (ticker, price, change, mom_score, value_score, total_score, history, news)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            s['ticker'],
            s['price'],
            s['change'],
            s['mom_score'],
            s['value_score'],
            s['total_score'],
            json.dumps(s['history']),
            json.dumps(s['news'])
        ))
    conn.commit()
    conn.close()

def load_stocks_from_db() -> list:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, price, change, mom_score, value_score, total_score, history, news FROM alpha_stocks ORDER BY total_score DESC")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        results.append({
            "ticker": r[0],
            "price": r[1],
            "change": r[2],
            "mom_score": r[3],
            "value_score": r[4],
            "total_score": r[5],
            "history": json.loads(r[6]),
            "news": json.loads(r[7])
        })
    return results
