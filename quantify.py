import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests
from datetime import date

# 1. Page Config & Professional Dark Theme
st.set_page_config(page_title="Master Quantify Terminal", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #c9d1d9; font-family: -apple-system, sans-serif; }
    .brand-title { font-size: 26px; font-weight: 900; color: #ffffff !important; letter-spacing: -0.5px; }
    .pro-badge { background: #1f6feb; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; color: white; vertical-align: middle; }
    .ai-box { background-color: #11151c; padding: 20px; border-radius: 8px; border: 1px solid #1f6feb; margin-top: 15px; line-height: 1.6; color: #d2d6dc; font-size: 14px; }
    .stButton > button { font-weight: 700; background-color: #1f6feb; color: white; border: none; }
    .stButton > button:hover { background-color: #388bfd; color: white; }
    </style>
""", unsafe_allow_html=True)

# 2. Core Screener Engine (Strict Close-Price Entry)
def run_master_screener(mode, index_choice, liq_enabled):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_map = {"S&P 500": "sp500_cache.csv", "Nasdaq 100": "nasdaq_cache.csv"}
    targets = ["sp500_cache.csv", "nasdaq_cache.csv"] if "Combined" in index_choice else [file_map.get(index_choice)]

    dfs = []
    for t in targets:
        path = os.path.join(base_dir, t)
        if os.path.exists(path): dfs.append(pd.read_csv(path, parse_dates=['Date']))

    if not dfs: return {}
    df_all = pd.concat(dfs).drop_duplicates(subset=['Date', 'Ticker'])
    signals = {}

    for ticker in df_all['Ticker'].unique():
        try:
            df = df_all[df_all['Ticker'] == ticker].copy().set_index('Date').sort_index()
            if len(df) < 60: continue

            prices = df['Close']
            volumes = df['Volume']
            delta = prices.diff()
            df['LSR'] = (volumes * (delta > 0).astype(int)).rolling(14).sum() / ((volumes * (delta < 0).astype(int)) + 1e-6).rolling(14).sum()
            df['Swing_High'] = prices.rolling(20).max()
            df['Swing_Low'] = prices.rolling(20).min()

            row = df.iloc[-1]
            if liq_enabled:
                if mode == "AGGRESSIVE" and not (row['LSR'] < 0.9 and row['Close'] >= row['Swing_High'] * 0.98): continue
                if mode == "PRECISION" and not (row['LSR'] > 1.2 and row['Close'] <= row['Swing_Low'] * 1.02): continue

            current_close = round(row['Close'], 2)
            signals[ticker] = {
                "entry": current_close,
                "target": round(current_close * 1.15, 2),
                "stop": round(current_close * 0.92, 2),
                "lsr": round(row.get('LSR', 1.0), 2),
                "df": df
            }
        except: continue
    return signals

# 3. Llama AI Analysis Module (Fixed Cache Key: Ticker + Date Only)
@st.cache_data(show_spinner=False)
def get_llama_analysis(ticker, entry, target, stop, lsr, scan_date):
    prompt = f"Analyze ticker [{ticker}] professionally based on Liquidation & Squeeze Mapping. Entry: ${entry}, Target: ${target}, Stop: ${stop}, LSR: {lsr}. Provide concise institutional analysis in English: 1. Catalyst, 2. Squeeze Potential, 3. Risk. Keep it under 150 words."
    try:
        res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=60)
        if res.status_code == 200:
            return res.json().get('response', 'No response generated.')
    except Exception as e:
        return f"AI Analysis Unavailable (Ollama offline: {e})"
    return "Failed to fetch AI analysis."

# 4. Streamlit UI Layout (Terminal View)
def main():
    st.markdown('<p class="brand-title">MASTER QUANTIFY TERMINAL <span class="pro-badge">PRO v2.6</span></p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar Controls
    st.sidebar.header("⚙️ ENGINE CONFIG")
    mode = st.sidebar.selectbox("Execution Mode", ["AGGRESSIVE", "PRECISION"])
    index_choice = st.sidebar.selectbox("Target Universe", ["S&P 500", "Nasdaq 100", "S&P 500 + Nasdaq 100 (Combined)"])
    liq_enabled = st.sidebar.checkbox("Enable Liquidity Filter (LSR)", value=True)
    
    scan_date = str(date.today())

    if st.sidebar.button("🚀 RUN QUANT SCREENER"):
        st.session_state['run_scan'] = True

    if st.session_state.get('run_scan', False):
        with st.spinner("Executing mathematical screening & scanning market liquidation..."):
            signals = run_master_screener(mode, index_choice, liq_enabled)
            st.session_state['signals'] = signals

    signals = st.session_state.get('signals', {})

    if not signals:
        st.info("👈 사이드바에서 설정을 확인하고 [RUN QUANT SCREENER] 버튼을 클릭하세요.")
        return

    col1, col2 = st.columns([7, 5])

    with col1:
        st.subheader("🔥 Active Signals")
        signal_list = []
        for ticker, data in signals.items():
            signal_list.append({
                "Ticker": ticker,
                "Entry ($)": data['entry'],
                "Target ($)": data['target'],
                "Stop ($)": data['stop'],
                "LSR": data['lsr']
            })
        df_signals = pd.DataFrame(signal_list)
        
        selected_ticker = st.selectbox("Select Ticker for Deep Dive", list(signals.keys()))
        st.dataframe(df_signals, use_container_width=True)

    if selected_ticker and selected_ticker in signals:
        sel_data = signals[selected_ticker]
        with col2:
            st.subheader(f"📊 Deep Dive: {selected_ticker}")
            
            # Plotly Candlestick Chart
            fig = go.Figure()
            # DataFrame에 Open, High, Low 컬럼이 없을 경우를 대비한 안전 장치
            df_plot = sel_data['df']
            op = df_plot['Open'] if 'Open' in df_plot else df_plot['Close']
            hi = df_plot['High'] if 'High' in df_plot else df_plot['Close']
            lo = df_plot['Low'] if 'Low' in df_plot else df_plot['Close']
            
            fig.add_trace(go.Candlestick(
                x=df_plot.index[-60:],
                open=op[-60:],
                high=hi[-60:],
                low=lo[-60:],
                close=df_plot['Close'][-60:],
                name='Price'
            ))
            fig.update_layout(
                template='plotly_dark',
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='#0b0e14',
                plot_bgcolor='#0b0e14'
            )
            st.plotly_chart(fig, use_container_width=True)

            # AI Analysis Box
            st.markdown("### 🤖 Llama 3 Institutional Briefing")
            with st.spinner("Generating AI market analysis via Ollama..."):
                ai_text = get_llama_analysis(
                    selected_ticker, 
                    sel_data['entry'], 
                    sel_data['target'], 
                    sel_data['stop'], 
                    sel_data['lsr'], 
                    scan_date
                )
            st.markdown(f'<div class="ai-box">{ai_text}</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
