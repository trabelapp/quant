import pandas as pd
import yfinance as yf
import time
from backend.database import save_stocks_to_db

def run_quant_scanner(mode: str = "value_momentum") -> list:
    # 밴 방지를 위해 엄선된 핵심 우량주 풀을 안전한 간격으로 스캔
    tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD", "NFLX", "PLTR", "AVGO", "COST", "JPM", "DIS", "V", "MA", "PG", "JNJ", "HD", "UNH"]
    results = []
    
    print(f"[SCREENER] Starting safe sequential scan for {len(tickers)} elite tickers...")
    
    for ticker in tickers:
        try:
            time.sleep(0.4) # 야후 파이낸스 방화벽 우회용 안전 딜레이
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")
            if df.empty or len(df) < 30:
                continue
                
            price_start = float(df['Close'].iloc[0])
            price_end = float(df['Close'].iloc[-1])
            if price_end < 5.0:
                continue
                
            mom_score = (price_end - price_start) / price_start * 100
            
            info = stock.info
            pe = info.get('trailingPE') or 20
            val_score = max(0, 50 - pe)
            
            if mode == "momentum":
                total_score = mom_score * 1.5
            elif mode == "value":
                total_score = val_score * 2.0
            else:
                total_score = (mom_score * 0.6) + (val_score * 0.4)

            history = [{"time": str(d.date()), "close": float(c)} for d, c in df['Close'].items()]
            
            news_list = [
                {"title": f"{ticker} Institutional Market Wire & Update", "publisher": "YAHOO FINANCE", "link": f"https://finance.yahoo.com/quote/{ticker}"},
                {"title": f"Quantitative Alpha Analysis: {ticker} Momentum Metrics", "publisher": "ELITE TERMINAL", "link": f"https://finance.yahoo.com/quote/{ticker}"}
            ]

            results.append({
                "ticker": ticker.upper(),
                "price": float(round(price_end, 2)),
                "change": float(round(mom_score / 4, 2)),
                "mom_score": float(round(mom_score, 2)),
                "value_score": float(round(val_score, 2)),
                "total_score": float(round(total_score, 2)),
                "history": history,
                "news": news_list
            })
        except Exception as e:
            print(f"[SCREENER] Error analyzing {ticker}: {e}")
            continue

    results.sort(key=lambda x: x['total_score'], reverse=True)
    
    if results:
        save_stocks_to_db(results)
        print(f"[SCREENER] Successfully scanned and saved {len(results)} stocks to SQLite DB.")
        
    return results
