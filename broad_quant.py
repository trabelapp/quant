import os
import pandas as pd
import numpy as np

def run_backtest():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_dir, "broad_market_cache.csv")
    
    if not os.path.exists(cache_path):
        print("❌ Error: broad_market_cache.csv not found.")
        return

    df_all = pd.read_csv(cache_path, parse_dates=['Date'])
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    
    max_date = df_all['Date'].max()
    min_date = max_date - pd.DateOffset(years=5)
    df_all = df_all[df_all['Date'] >= min_date]
    
    initial_capital = 10000.0
    current_cash = initial_capital
    
    tickers = df_all['Ticker'].unique()
    print(f"[*] Running REALITY-CHECKED DRAGON HATCHING Backtest across {len(tickers)} assets...")

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
    commission_rate = 0.0005  # 0.05% 브로커 수수료

    for current_date in df_dates:
        # 1. 포지션 정산 체크
        remaining_positions = []
        for pos in active_positions:
            ticker = pos['ticker']
            df_ticker = processed_dfs[ticker]
            
            if current_date not in df_ticker.index:
                remaining_positions.append(pos)
                continue
                
            row = df_ticker.loc[current_date]
            hit_high = row['High'] >= pos['target']
            hit_low = row['Low'] <= pos['stop']
            
            if hit_high and hit_low:
                # 손절 처리 (수수료 반영)
                ret = (pos['stop'] - pos['entry']) / pos['entry'] - commission_rate
                current_cash += pos['allocation'] * (1 + ret)
                total_trades += 1
            elif hit_low:
                ret = (pos['stop'] - pos['entry']) / pos['entry'] - commission_rate
                current_cash += pos['allocation'] * (1 + ret)
                total_trades += 1
            elif hit_high:
                # 익절 처리 (수수료 반영)
                ret = (pos['target'] - pos['entry']) / pos['entry'] - commission_rate
                current_cash += pos['allocation'] * (1 + ret)
                total_trades += 1
                wins += 1
            else:
                remaining_positions.append(pos)
                
        active_positions = remaining_positions

        # 현재 총 포트폴리오 가치 계산 (현금 + 보유 중인 주식들의 매입가 기준 추정치)
        active_value = sum([p['allocation'] for p in active_positions])
        total_portfolio_value = current_cash + active_value
        portfolio_history.append({'Date': current_date, 'Value': total_portfolio_value})

        # 2. 월말 신규 진입
        if current_date in monthly_dates and len(active_positions) < 5:
            scored_universe = []
            held_tickers = [p['ticker'] for p in active_positions]
            
            for ticker, df in processed_dfs.items():
                if ticker in held_tickers: continue
                sub_df = df[df.index <= current_date]
                if sub_df.empty: continue
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
                    allocation_per_slot = current_cash / len(top_picks)
                    current_cash -= allocation_per_slot
                    
                    for ticker, score, row in top_picks:
                        # 슬리피지를 0.5%로 강화 + 매수 수수료 반영
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

    # 최종 정산
    for pos in active_positions:
        ticker = pos['ticker']
        df_ticker = processed_dfs[ticker]
        final_close = df_ticker.iloc[-1]['Close'] * (1 - commission_rate)
        ret = (final_close - pos['entry']) / pos['entry']
        current_cash += pos['allocation'] * (1 + ret)
        total_trades += 1
        if ret > 0: wins += 1

    # MDD(Maximum Drawdown) 계산
    df_hist = pd.DataFrame(portfolio_history).set_index('Date')
    df_hist['Peak'] = df_hist['Value'].cummax()
    df_hist['Drawdown'] = (df_hist['Value'] - df_hist['Peak']) / df_hist['Peak']
    max_dd = df_hist['Drawdown'].min() * 100

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return = ((current_cash - initial_capital) / initial_capital) * 100

    print("\n" + "="*58)
    print(" 🛡️ REALITY-CHECKED DRAGON BACKTEST RESULTS")
    print("="*58)
    print(f" Initial Capital : ${initial_capital:,.2f}")
    print(f" Final Portfolio : ${current_cash:,.2f}")
    print(f" Total Trades    : {total_trades}")
    print(f" Win Rate        : {win_rate:.2f}%")
    print(f" Total Return    : {total_return:.2f}%")
    print(f" Maximum Drawdown: {max_dd:.2f}% (최대 낙폭)")
    print("="*58)

if __name__ == "__main__":
    run_backtest()
