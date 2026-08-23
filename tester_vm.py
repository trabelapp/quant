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
    print(f"[*] Running CONTRARIAN VALUE-MOMENTUM Backtest across {len(tickers)} assets...")

    processed_dfs = {}
    for ticker in tickers:
        df = df_all[df_all['Ticker'] == ticker].copy().set_index('Date').sort_index()
        if len(df) < 250: continue  # 252일치 데이터 필수
        
        prices = df['Close']
        volumes = df['Volume']
        
        df['High_252'] = prices.rolling(252).max()
        df['SMA200'] = prices.rolling(200).mean()
        df['SMA20'] = prices.rolling(20).mean()
        df['Swing_High_20'] = prices.rolling(20).max()
        df['Swing_Low_20'] = prices.rolling(20).min()
        df['ATR'] = (prices.rolling(14).max() - prices.rolling(14).min()) / 14
        df['Vol_MA'] = volumes.rolling(20).mean()
        
        processed_dfs[ticker] = df.dropna()

    events = []
    for ticker, df in processed_dfs.items():
        for i in range(200, len(df) - 30):
            row = df.iloc[i]
            date = df.index[i]
            
            # 1. Contrarian Value Filter: 고점 대비 20% ~ 45% 하락 (단기 악재로 바닥에 처박힘)
            # 단, 200일선 위에 있어 장기 체급(우량주)은 유지됨
            drawdown = (row['Close'] - row['High_252']) / row['High_252']
            is_beaten_down = (-0.45 <= drawdown <= -0.20)
            is_quality_alive = row['Close'] >= row['SMA200']
            
            # 2. Momentum Trigger: 거래량 터지며 20일 고점 돌파 (바닥 탈출 시그널)
            is_volume_surge = row['Volume'] >= (row['Vol_MA'] * 1.8)
            is_breakout = row['Close'] >= row['Swing_High_20'] * 0.99
            
            if is_beaten_down and is_quality_alive and is_volume_surge and is_breakout:
                entry = row['Close'] * 1.002  # 슬리피지 반영
                atr = row['ATR']
                stop = row['Swing_Low_20'] - (atr * 0.5)
                target = entry + (atr * 3.5)  # 손익비 2.5 ~ 3 이상 설계
                
                risk = entry - stop
                reward = target - entry
                if risk <= 0 or (reward / risk) < 2.5:
                    continue
                
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

    print("\n" + "="*58)
    print(" 💎 CONTRARIAN VALUE-MOMENTUM BACKTEST RESULTS")
    print("="*58)
    print(f" Initial Capital : ${initial_capital:,.2f}")
    print(f" Final Portfolio : ${current_cash:,.2f}")
    print(f" Total Trades    : {total_trades}")
    print(f" Win Rate        : {win_rate:.2f}%")
    print(f" Total Return    : {total_return:.2f}%")
    print("="*58)

if __name__ == "__main__":
    run_backtest()
