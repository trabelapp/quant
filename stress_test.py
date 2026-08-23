import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_master_universe():
    # 과거부터 현재까지 아우르는 거대한 마스터 풀 (시간이 흐르며 자연스럽게 합류/탈락)
    tickers = [
        # Legacy Blue-Chips (2011~ )
        "IBM", "INTC", "CSCO", "ORCL", "MSFT", "AAPL", "HPQ", "VZ", "T",
        "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC",
        "JNJ", "PFE", "MRK", "ABT", "BMY", "LLY", "AMGN", "UNH", "CVS",
        "XOM", "CVX", "COP", "SLB", "OXY", "DUK", "SO", "D",
        "CAT", "DE", "HON", "UNP", "UPS", "FDX", "LMT", "BA", "GE", "MMM",
        "WMT", "PG", "KO", "PEP", "MCD", "DIS", "NKE", "GIS", "CL", "KMB",
        # Growth & Tech Giants (Later additions)
        "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "AMD", "AVGO",
        "QCOM", "TXN", "ADBE", "CRM", "NOW", "PYPL", "SBUX", "COST", "TGT",
        "LULU", "ISRG", "REGN", "V", "MA", "SPGI", "BLK", "PLD", "AMT"
    ]
    return list(set(tickers))

def download_master_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_dir, "deep_history_cache.csv")
    
    tickers = get_master_universe()
    print(f"[*] Downloading MASTER UNIVERSE HISTORY for {len(tickers)} assets (Dynamic Test)...")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=15*365 + 100)
    
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=['Date'])

    data_list = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            df = df.reset_index()
            df['Ticker'] = ticker
            data_list.append(df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])
        except Exception as e:
            continue
            
    if data_list:
        df_all = pd.concat(data_list, ignore_index=True)
        df_all.to_csv(cache_path, index=False)
        return df_all
    return None

def run_stress_test():
    df_all = download_master_data()
    if df_all is None: return

    df_all['Date'] = pd.to_datetime(df_all['Date'])
    max_date = df_all['Date'].max()
    min_date = max_date - pd.DateOffset(years=15)
    df_all = df_all[df_all['Date'] >= min_date]
    
    initial_capital = 10000.0
    current_cash = initial_capital
    
    tickers = df_all['Ticker'].unique()
    print(f"[*] Running TRUE DYNAMIC 15-YEAR STRESS TEST across {len(tickers)} assets...")

    processed_dfs = {}
    for ticker in tickers:
        df = df_all[df_all['Ticker'] == ticker].copy().set_index('Date').sort_index()
        if len(df) < 250: continue
        
        prices = df['Close']
        volumes = df['Volume']
        
        df['High_252'] = prices.rolling(252).max()
        df['Value_Score'] = (df['High_252'] - prices) / df['High_252']
        df['Mom_Score'] = prices.shift(20) / prices.shift(252) - 1.0
        
        df['ATR'] = (prices.rolling(14).max() - prices.rolling(14).min()) / 14
        df['Vol_MA'] = volumes.rolling(20).mean()
        df['Swing_Low_20'] = prices.rolling(20).min()
        
        processed_dfs[ticker] = df.dropna()

    df_dates = sorted(df_all['Date'].unique())
    monthly_dates = [df_dates[i] for i in range(len(df_dates)) if i == 0 or df_dates[i].month != df_dates[i-1].month]

    active_positions = []
    total_trades = 0
    wins = 0
    
    portfolio_history = []
    commission_rate = 0.0005

    for current_date in df_dates:
        remaining_positions = []
        for pos in active_positions:
            ticker = pos['ticker']
            df_ticker = processed_dfs.get(ticker)
            if df_ticker is None or current_date not in df_ticker.index:
                remaining_positions.append(pos)
                continue
                
            row = df_ticker.loc[current_date]
            hit_high = row['High'] >= pos['target']
            hit_low = row['Low'] <= pos['stop']
            
            if hit_high and hit_low:
                ret = (pos['stop'] - pos['entry']) / pos['entry'] - commission_rate
                current_cash += pos['allocation'] * (1 + ret)
                total_trades += 1
            elif hit_low:
                ret = (pos['stop'] - pos['entry']) / pos['entry'] - commission_rate
                current_cash += pos['allocation'] * (1 + ret)
                total_trades += 1
            elif hit_high:
                ret = (pos['target'] - pos['entry']) / pos['entry'] - commission_rate
                current_cash += pos['allocation'] * (1 + ret)
                total_trades += 1
                wins += 1
            else:
                remaining_positions.append(pos)
                
        active_positions = remaining_positions

        active_value = sum([p['allocation'] for p in active_positions])
        total_portfolio_value = current_cash + active_value
        portfolio_history.append({'Date': current_date, 'Value': total_portfolio_value})

        if current_date in monthly_dates and len(active_positions) < 5:
            scored_universe = []
            held_tickers = [p['ticker'] for p in active_positions]
            
            for ticker, df in processed_dfs.items():
                if ticker in held_tickers: continue
                
                # 핵심: 현재 날짜 기준으로 데이터가 존재하는 종목만 동적으로 참여 (과거로 돌아갔을 때 상장 전인 종목 배제)
                sub_df = df[df.index <= current_date]
                if len(sub_df) < 252: continue 
                
                row = sub_df.iloc[-1]
                
                if row['Volume'] < row['Vol_MA']: continue
                
                v_score = row['Value_Score']
                m_score = row['Mom_Score']
                
                if m_score > 0:
                    combined_score = (v_score * 0.5) + (m_score * 0.5)
                    scored_universe.append((ticker, combined_score, row))
            
            if scored_universe:
                scored_universe = sorted(scored_universe, key=lambda x: x[1], reverse=True)
                slots_available = 5 - len(active_positions)
                top_picks = scored_universe[:slots_available]
                
                if current_cash > 0 and top_picks:
                    num_picks = len(top_picks)
                    allocation_per_slot = current_cash / num_picks
                    current_cash -= (allocation_per_slot * num_picks)
                    
                    for ticker, score, row in top_picks:
                        entry = row['Close'] * 1.005 * (1 + commission_rate)
                        target = entry * 1.50
                        stop = row['Swing_Low_20'] - (row['ATR'] * 0.8)
                        
                        active_positions.append({
                            'ticker': ticker,
                            'entry': entry,
                            'target': target,
                            'stop': stop,
                            'allocation': allocation_per_slot
                        })

    for pos in active_positions:
        ticker = pos['ticker']
        df_ticker = processed_dfs.get(ticker)
        if df_ticker is not None and not df_ticker.empty:
            final_close = df_ticker.iloc[-1]['Close'] * (1 - commission_rate)
            ret = (final_close - pos['entry']) / pos['entry']
            current_cash += pos['allocation'] * (1 + ret)
            total_trades += 1
            if ret > 0: wins += 1

    df_hist = pd.DataFrame(portfolio_history).set_index('Date')
    df_hist['Peak'] = df_hist['Value'].cummax()
    df_hist['Drawdown'] = (df_hist['Value'] - df_hist['Peak']) / df_hist['Peak']
    max_dd = df_hist['Drawdown'].min() * 100

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return = ((current_cash - initial_capital) / initial_capital) * 100

    print("\n" + "="*58)
    print(" 🚀 TRUE DYNAMIC 15-YEAR STRESS TEST RESULTS")
    print("="*58)
    print(f" Initial Capital : ${initial_capital:,.2f}")
    print(f" Final Portfolio : ${current_cash:,.2f}")
    print(f" Total Trades    : {total_trades}")
    print(f" Win Rate        : {win_rate:.2f}%")
    print(f" Total Return    : {total_return:.2f}%")
    print(f" Maximum Drawdown: {max_dd:.2f}%")
    print("="*58)

if __name__ == "__main__":
    run_stress_test()
