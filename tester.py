import os
import pandas as pd
import numpy as np

def run_backtest():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    targets = ["sp500_cache.csv", "nasdaq_cache.csv"]
    
    dfs = []
    for t in targets:
        path = os.path.join(base_dir, t)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, parse_dates=['Date']))
    
    if not dfs:
        print("❌ Error: Cache files not found in directory.")
        return

    df_all = pd.concat(dfs).drop_duplicates(subset=['Date', 'Ticker']).sort_values('Date')
    
    max_date = df_all['Date'].max()
    min_date = max_date - pd.DateOffset(years=5)
    df_all = df_all[df_all['Date'] >= min_date]
    
    initial_capital = 10000.0
    current_cash = initial_capital
    total_trades = 0
    wins = 0
    trade_returns = []
    
    tickers = df_all['Ticker'].unique()
    print(f"[*] Running PULLBACK MOMENTUM Backtest (High Risk/Reward Strategy) across {len(tickers)} assets...")

    processed_dfs = {}
    for ticker in tickers:
        df = df_all[df_all['Ticker'] == ticker].copy().set_index('Date').sort_index()
        if len(df) < 100: continue
        prices = df['Close']
        volumes = df['Volume']
        delta = prices.diff()
        
        df['LSR'] = (volumes * (delta > 0).astype(int)).rolling(14).sum() / ((volumes * (delta < 0).astype(int)) + 1e-6).rolling(14).sum()
        df['SMA20'] = prices.rolling(20).mean()
        df['SMA50'] = prices.rolling(50).mean()
        df['ATR'] = (prices.rolling(14).max() - prices.rolling(14).min()) / 14
        df['Vol_MA'] = volumes.rolling(20).mean()
        
        processed_dfs[ticker] = df.dropna()

    events = []
    for ticker, df in processed_dfs.items():
        for i in range(50, len(df) - 30):
            row = df.iloc[i]
            date = df.index[i]
            
            # [Pullback Strategy] 50일선 위 상승 추세 중, 20일 이평선 근처로 살짝 눌렸을 때 거래량 급증 + 매수세 유입
            is_uptrend = row['Close'] > row['SMA50']
            is_near_support = (row['Low'] <= row['SMA20'] * 1.02) and (row['Close'] >= row['SMA20'] * 0.98)
            is_volume_surge = row['Volume'] >= (row['Vol_MA'] * 1.5)
            
            if is_uptrend and is_near_support and is_volume_surge and (row['LSR'] > 1.1):
                entry = row['Close'] * 1.002
                atr = row['ATR']
                
                # 손익비 2.5 : 1 설계
                target = entry + (atr * 3.5)
                stop = entry - (atr * 1.4)
                
                events.append({
                    'date': date,
                    'entry': entry,
                    'target': target,
                    'stop': stop,
                    'df': df,
                    'idx': i
                })

    events = sorted(events, key=lambda x: x['date'])

    for event in events:
        entry = event['entry']
        target = event['target']
        stop = event['stop']
        df = event['df']
        idx = event['idx']
        
        future_df = df.iloc[idx+1 : idx+31]
        resolved = False
        won = False
        outcome_return = 0.0
        
        for _, f_row in future_df.iterrows():
            hit_high = f_row['High'] >= target
            hit_low = f_row['Low'] <= stop
            
            if hit_low and hit_high:
                outcome_return = (stop - entry) / entry - 0.001
                won = False
                resolved = True
                break
            elif hit_low:
                outcome_return = (stop - entry) / entry - 0.001
                won = False
                resolved = True
                break
            elif hit_high:
                outcome_return = (target - entry) / entry - 0.001
                won = True
                resolved = True
                break
        
        if not resolved and not future_df.empty:
            final_close = future_df.iloc[-1]['Close']
            outcome_return = (final_close - entry) / entry - 0.001
            won = outcome_return > 0
            resolved = True

        if resolved:
            total_trades += 1
            if won: wins += 1
            trade_returns.append(outcome_return)
            current_cash *= (1 + outcome_return * 0.2)
            if current_cash < 100:
                current_cash = 0
                break

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return = ((current_cash - initial_capital) / initial_capital) * 100

    print("\n" + "="*56)
    print(" 📈 PULLBACK MOMENTUM BACKTEST RESULTS (tester.py)")
    print("="*56)
    print(f" Initial Capital : ${initial_capital:,.2f}")
    print(f" Final Portfolio : ${current_cash:,.2f}")
    print(f" Total Trades    : {total_trades}")
    print(f" Win Rate        : {win_rate:.2f}%")
    print(f" Total Return    : {total_return:.2f}%")
    print("="*56)

if __name__ == "__main__":
    run_backtest()
