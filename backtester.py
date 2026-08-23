import time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import bs4

# ---------------------------------------------------------------------------
# 1. Dynamic Universe Scraper (위키피디아 S&P 500 + Nasdaq 100 최신 동기화)
# ---------------------------------------------------------------------------
def _scrape_wiki_tickers(url, table_id=None):
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": table_id}) if table_id else None
        if table is None:
            for tbl in soup.find_all("table", {"class": "wikitable"}):
                headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
                if any(h in ("ticker", "symbol", "ticker symbol") for h in headers):
                    table = tbl
                    break
        if table is None:
            return []
        rows = table.find_all("tr")
        header_cells = [th.get_text(strip=True).lower() for th in rows[0].find_all("th")]
        col_idx = 0
        for i, h in enumerate(header_cells):
            if "ticker" in h or "symbol" in h:
                col_idx = i
                break
        tickers = []
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) > col_idx:
                ticker = cols[col_idx].get_text(strip=True).replace(".", "-")
                if ticker and len(ticker) <= 6:
                    tickers.append(ticker)
        return list(dict.fromkeys(tickers))
    except Exception as e:
        print(f"[Wiki Scrape Error] {url}: {e}")
        return []

def get_trading_universe():
    print("[Universe] Fetching latest S&P 500 & Nasdaq 100 constituents dynamically...")
    sp500 = _scrape_wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", table_id="constituents")
    nasdaq100 = _scrape_wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100")
    combined = list(dict.fromkeys(sp500 + nasdaq100))
    if not combined:
        combined = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "JPM"]
    print(f"[Universe] Total dynamically fetched tickers: {len(combined)}")
    return combined

# ---------------------------------------------------------------------------
# 2. Rate-Limit Safe Batch Download (최근 15년 고정: 2011 ~ Present)
# ---------------------------------------------------------------------------
BATCH_CHUNK_SIZE = 40
BATCH_PAUSE_SECONDS = 1.5

def fetch_historical_data():
    universe = get_trading_universe()
    tickers_to_fetch = universe + ["SPY"]
    print(f"[Data Pipeline] Downloading data from 2011-01-01 to Present for {len(tickers_to_fetch)} tickers...")
    
    data_dict = {}
    for i in range(0, len(tickers_to_fetch), BATCH_CHUNK_SIZE):
        chunk = tickers_to_fetch[i:i + BATCH_CHUNK_SIZE]
        try:
            df_chunk = yf.download(
                tickers=chunk,
                start="2011-01-01",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True
            )
            if len(chunk) == 1:
                t = chunk[0]
                if df_chunk is not None and not df_chunk.empty:
                    data_dict[t] = df_chunk.dropna(how="all")
            else:
                for t in chunk:
                    try:
                        df_t = df_chunk[t].dropna(how="all")
                        if len(df_t) > 250:  # 12개월 데이터 필수
                            data_dict[t] = df_t
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Error] Batch download failed for chunk {chunk}: {e}")
        time.sleep(BATCH_PAUSE_SECONDS)
        
    spy_df = data_dict.pop("SPY", None)
    print(f"[Data Pipeline] Successfully loaded data for {len(data_dict)} stocks.")
    return data_dict, spy_df

# ---------------------------------------------------------------------------
# 3. Indicator Calculation (12개월 수익률 및 1주일 고점 대비 5~7% 눌림목)
# ---------------------------------------------------------------------------
def calculate_indicators(df):
    close = df['Close']
    volume = df['Volume']

    df['Dollar_Volume'] = close * volume
    df['Return_12M'] = close / close.shift(252) - 1.0

    # 최근 1주일(5거래일) 기준 최고가 및 5%~7% 조정(눌림목) 조건
    df['High_1wk'] = close.rolling(window=5).max()
    df['Dip_Buy_Signal'] = (close <= df['High_1wk'] * 0.95) & (close >= df['High_1wk'] * 0.93)

    return df.dropna()

# ---------------------------------------------------------------------------
# 4. Wall Street Style Momentum Backtest Engine (Top 25%, SL:-30%, TP:+60%)
# ---------------------------------------------------------------------------
def run_wallstreet_momentum_backtest(data_dict, spy_df):
    initial_capital = 100000.0
    cash = initial_capital
    
    if spy_df is not None and not spy_df.empty:
        spy_df['SMA_200'] = spy_df['Close'].rolling(window=200).mean()
    
    # 🔥 사장님 지시사항 적용: 상위 25%, 손절 -30%, 익절 +60%
    TOP_PERCENTILE = 0.25
    STOP_LOSS_PCT = 0.30
    TAKE_PROFIT_PCT = 0.60
    
    SLIPPAGE = 0.002
    MIN_DOLLAR_VOLUME = 50_000_000
    
    processed_data = {}
    for t, df in data_dict.items():
        processed_data[t] = calculate_indicators(df)
        
    all_dates = sorted(list(set().union(*(df.index for df in processed_data.values()))))
    
    all_trades = []
    active_positions = []
    current_top_universe = set()
    last_rebalance_month = -1

    print("[Backtest] Running Final Optimized Momentum & Dip-Buy simulation...")
    
    for date in all_dates:
        # A. 월초 리밸런싱: 12개월 기준 상위 25% 엘리트 종목 선정
        if date.month != last_rebalance_month:
            last_rebalance_month = date.month
            cross_scores = {}
            for t, df in processed_data.items():
                if date in df.index:
                    score = df.loc[date, 'Return_12M']
                    if not np.isnan(score):
                        cross_scores[t] = score
            
            if cross_scores:
                sorted_stocks = sorted(cross_scores.items(), key=lambda x: x[1], reverse=True)
                top_n = max(1, int(len(sorted_stocks) * TOP_PERCENTILE))
                current_top_universe = {item[0] for item in sorted_stocks[:top_n]}

        # 시장 체제 필터 (SPY 200일선)
        is_bull_market = True
        if spy_df is not None and date in spy_df.index:
            spy_row = spy_df.loc[date]
            if spy_row['Close'] < spy_row['SMA_200']:
                is_bull_market = False

        # B. 포지션 청산 관리 (익절 / 손절 체크)
        remaining_positions = []
        for pos in active_positions:
            t = pos['ticker']
            if date not in processed_data[t].index:
                remaining_positions.append(pos)
                continue
                
            row = processed_data[t].loc[date]
            high = float(row['High'])
            low = float(row['Low'])
            
            hit_tp = high >= pos['tp_price']
            hit_sl = low <= pos['sl_price']
            
            if hit_tp or hit_sl:
                exit_price = pos['tp_price'] * (1 - SLIPPAGE) if hit_tp and not hit_sl else pos['sl_price'] * (1 - SLIPPAGE)
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                cash += (pos['shares'] * exit_price)
                
                trade_return = (exit_price - pos['entry_price']) / pos['entry_price']
                all_trades.append({
                    'ticker': t,
                    'return': trade_return,
                    'win': 1 if trade_return > 0 else 0,
                    'pnl': pnl
                })
            else:
                remaining_positions.append(pos)
        active_positions = remaining_positions

        # C. 신규 진입 스캔
        if is_bull_market:
            for t in current_top_universe:
                if t not in processed_data:
                    continue
                df = processed_data[t]
                if date not in df.index:
                    continue
                if any(p['ticker'] == t for p in active_positions):
                    continue
                    
                row = df.loc[date]
                
                if row['Dollar_Volume'] < MIN_DOLLAR_VOLUME:
                    continue
                
                if row['Dip_Buy_Signal']:
                    p = float(row['Close']) * (1 + SLIPPAGE)
                    risk_budget = cash * 0.01  
                    dollar_risk_per_share = p * STOP_LOSS_PCT
                    shares = int(risk_budget / dollar_risk_per_share)
                    
                    cost = shares * p
                    if shares > 0 and cash >= cost:
                        cash -= cost
                        active_positions.append({
                            'ticker': t,
                            'entry_price': p,
                            'shares': shares,
                            'sl_price': p * (1 - STOP_LOSS_PCT),
                            'tp_price': p * (1 + TAKE_PROFIT_PCT)
                        })

    # 🔥 사장님 지시사항 적용: 백테스트 종료 시점(마지막 날)에 모든 남은 포지션 강제 청산
    final_date = all_dates[-1]
    for pos in active_positions:
        t = pos['ticker']
        if final_date in processed_data[t].index:
            row = processed_data[t].loc[final_date]
            exit_price = float(row['Close']) * (1 - SLIPPAGE)
            pnl = (exit_price - pos['entry_price']) * pos['shares']
            cash += (pos['shares'] * exit_price)
            
            trade_return = (exit_price - pos['entry_price']) / pos['entry_price']
            all_trades.append({
                'ticker': t,
                'return': trade_return,
                'win': 1 if trade_return > 0 else 0,
                'pnl': pnl
            })

    return all_trades, initial_capital, cash

# ---------------------------------------------------------------------------
# 5. Performance Report
# ---------------------------------------------------------------------------
def evaluate_performance(trades, initial_capital, final_cash, all_dates):
    if not trades:
        print("\n[Result] 조건에 맞는 체결 거래가 없습니다.")
        return

    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)
    win_trades = df_trades['win'].sum()
    win_rate = (win_trades / total_trades) * 100
    
    final_capital = final_cash
    total_pnl = final_capital - initial_capital
    cumulative_return = (total_pnl / initial_capital) * 100
    
    returns = df_trades['return']
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252) if len(returns) > 1 else 0.0
    
    equity_curve = initial_capital + df_trades['pnl'].cumsum()
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    mdd = drawdown.min() * 100

    print("\n" + "="*65)
    print(" 🏛️ FINAL MOMENTUM REPORT (Top 25%, SL:-30%, TP:+60%)")
    print("="*65)
    print(f" 백테스트 기간   : {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")
    print(f" 초기 자본금     : ${initial_capital:,.2f}")
    print(f" 최종 자산 평가액 : ${final_capital:,.2f}")
    print(f" 총 누적 수익률   : {cumulative_return:.2f}%")
    print(f" 총 거래 횟수     : {total_trades} 회")
    print(f" 승률 (Win Rate)  : {win_rate:.2f}%")
    print(f" 샤프 지수 (Sharpe): {sharpe_ratio:.2f}")
    print(f" 최대 낙폭 (MDD)  : {mdd:.2f}%")
    print("="*65)

if __name__ == "__main__":
    data_dict, spy_df = fetch_historical_data()
    all_dates = sorted(list(set().union(*(df.index for df in data_dict.values()))))
    trades, init_cap, final_cash = run_wallstreet_momentum_backtest(data_dict, spy_df)
    evaluate_performance(trades, init_cap, final_cash, all_dates)
