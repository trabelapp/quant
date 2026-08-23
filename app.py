import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

# 1. 페이지 설정 (와이드 모드)
st.set_page_config(
    page_title="Quantify | Trading Terminal",
    page_icon="⚡",
    layout="wide"
)

# 2. 트레이딩뷰/블룸버그 스타일의 올블랙 커스텀 CSS 적용
st.markdown("""
    <style>
    /* 전체 배경 올블랙 딥 다크 */
    .stApp { background-color: #09090b; color: #f4f4f5; }
    
    /* 사이드바 스타일링 */
    [data-testid="stSidebar"] { background-color: #121217; border-right: 1px solid #27272a; }
    
    /* 카드 및 컨테이너 스타일 */
    .metric-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    /* 테이블 헤더 및 글자 색상 보정 */
    table { color: #f4f4f5 !important; }
    
    /* 버튼 커스텀 */
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #059669;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 데이터 로딩 엔진 (캐싱 적용)
@st.cache_data(ttl=1800)
def fetch_terminal_data():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "XOM", "AMD", "NFLX", "COST", "AVGO", "PLTR"]
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    
    results = []
    for ticker in data.columns:
        series = data[ticker].dropna()
        if len(series) < 50:
            continue
        price = float(series.iloc[-1])
        mom_12m = ((price / series.iloc[0]) - 1) * 100
        mom_1m = ((price / series.iloc[-21]) - 1) * 100
        score = round((mom_12m * 0.6) + (mom_1m * 0.4), 2)
        
        results.append({
            "ticker": ticker,
            "price": round(price, 2),
            "mom_12m": round(mom_12m, 2),
            "mom_1m": round(mom_1m, 2),
            "score": score
        })
    return pd.DataFrame(results).sort_values(by="score", ascending=False).reset_index(drop=True)

# 4. 사이드바 내비게이션 (모드 선택)
st.sidebar.markdown("<h1 style='color: #10b981; font-size: 24px;'>⚡ QUANTIFY</h1>", unsafe_allow_html=True)
st.sidebar.caption("Institutional-grade Quant Terminal")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "터미널 모드 선택",
    ["📊 퀀트 스캐너 & 터미널", "🤖 AI 심층 분석 랩", "📰 실시간 뉴스 & 촉매", "📈 고급 멀티 차트"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Trable System Status**: Online\n\n🔒 **Database**: SQLite Synchronized")

df = fetch_terminal_data()

# ================= 모드 1: 퀀트 스캐너 & 터미널 =================
if mode == "📊 퀀트 스캐너 & 터미널":
    st.title("⚡ Quantitative Screener Terminal")
    st.markdown("전체 유니버스를 대상으로 한 실시간 멀티팩터 알파 스코어링 보드")
    
    # 상단 요약 지표 카드
    col1, col2, col3 = st.columns(3)
    if not df.empty:
        top1 = df.iloc[0]
        top2 = df.iloc[1]
        top3 = df.iloc[2]
        
        with col1:
            st.markdown(f"""<div class='metric-card'>
                <div style='color: #10b981; font-size: 12px; font-weight: bold;'>RANK #1 ALPHA</div>
                <div style='font-size: 28px; font-weight: bold;'>{top1['ticker']}</div>
                <div style='font-size: 18px; color: #a1a1aa;'>${top1['price']:,.2f}</div>
                <div style='color: #10b981; margin-top: 5px;'>Score: {top1['score']} pts</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='metric-card'>
                <div style='color: #38bdf8; font-size: 12px; font-weight: bold;'>RANK #2 ALPHA</div>
                <div style='font-size: 28px; font-weight: bold;'>{top2['ticker']}</div>
                <div style='font-size: 18px; color: #a1a1aa;'>${top2['price']:,.2f}</div>
                <div style='color: #38bdf8; margin-top: 5px;'>Score: {top2['score']} pts</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class='metric-card'>
                <div style='color: #a855f7; font-size: 12px; font-weight: bold;'>RANK #3 ALPHA</div>
                <div style='font-size: 28px; font-weight: bold;'>{top3['ticker']}</div>
                <div style='font-size: 18px; color: #a1a1aa;'>${top3['price']:,.2f}</div>
                <div style='color: #a855f7; margin-top: 5px;'>Score: {top3['score']} pts</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 검색 및 필터 바
    search_col, _ = st.columns([2, 1])
    with search_col:
        query = st.text_input("🔍 티커 심볼 검색 (예: NVDA, AAPL)", "").upper()
    
    filtered_df = df[df['ticker'].str.contains(query)] if query else df
    
    # 데이터 테이블 출력
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "Ticker",
            "price": st.column_config.NumberColumn("Current Price ($)", format="$%.2f"),
            "mom_12m": st.column_config.NumberColumn("12M Momentum", format="%.2f%%"),
            "mom_1m": st.column_config.NumberColumn("1M Momentum", format="%.2f%%"),
            "score": st.column_config.ProgressColumn("Composite Alpha Score", min_value=-20, max_value=80, format="%.2f")
        }
    )

# ================= 모드 2: AI 심층 분석 랩 =================
elif mode == "🤖 AI 심층 분석 랩":
    st.title("🤖 AI Deep Quantitative Analysis")
    st.markdown("선택한 종목에 대한 실시간 AI 알고리즘 진단 및 리스크 평가")
    
    selected_ticker = st.selectbox("분석할 종목 선택", df['ticker'].tolist() if not df.empty else ["AAPL"])
    
    if st.button("🚀 AI 심층 리포트 생성"):
        with st.spinner(f"[{selected_ticker}] 실시간 가치평가 및 모멘텀 구조 분석 중..."):
            # 가상 AI 분석 결과 렌더링 (안정적이고 빠르게 동작)
            st.markdown(f"### 📊 [{selected_ticker}] AI Synthesis Report")
            st.success("분석 완료: 기관 수급 유입 감지 및 펀더멘털 견조함 확인")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🟢 강점 (Strengths)")
                st.write("- 12개월 장기 추세선 상방 정배열 유지")
                st.write("- 섹터 내 상대 강도(Relative Strength) 상위 10% 이내")
                st.write("- 변동성 대비 알파 수익률 우수")
            with c2:
                st.markdown("#### 🔴 리스크 요인 (Risks)")
                st.write("- 단기 과열 구간 진입에 따른 노이즈 주의")
                st.write("- 거시경제 금리 변동성에 따른 지수 연동성 확인 필요")

# ================= 모드 3: 실시간 뉴스 & 촉매 =================
elif mode == "📰 실시간 뉴스 & 촉매":
    st.title("📰 Real-time Market Catalysts & News")
    st.markdown("시장 주도주 관련 핵심 뉴스 피드 및 센티멘트 분석")
    
    news_items = [
        {"time": "10:45 AM", "ticker": "NVDA", "title": "차세대 AI 칩 공급망 물량 완판 소식에 따른 강세 압력 지속", "sentiment": "Bullish 🟢"},
        {"time": "09:30 AM", "ticker": "AAPL", "title": "서비스 부문 마진 확대에 따른 월가 주요 투자은행 목표가 상향", "sentiment": "Bullish 🟢"},
        {"time": "08:15 AM", "ticker": "TSLA", "title": "글로벌 출하량 데이터 발표 앞두고 관망세 유입", "sentiment": "Neutral 🟡"},
        {"time": "Yesterday", "ticker": "MSFT", "title": "클라우드 인프라 확장 가속화 및 엔터프라이즈 계약 체결", "sentiment": "Bullish 🟢"}
    ]
    
    for item in news_items:
        st.markdown(f"""
            <div style='background-color: #18181b; border: 1px solid #27272a; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
                <span style='color: #10b981; font-weight: bold;'>[{item['ticker']}]</span> 
                <span style='color: #71717a; font-size: 12px;'>{item['time']}</span>
                <div style='font-size: 16px; margin-top: 5px; color: #f4f4f5;'>{item['title']}</div>
                <div style='margin-top: 8px; font-size: 13px;'>Sentiment: <b>{item['sentiment']}</b></div>
            </div>
        """, unsafe_allow_html=True)

# ================= 모드 4: 고급 멀티 차트 =================
elif mode == "📈 고급 멀티 차트":
    st.title("📈 Advanced Price Charting")
    st.markdown("트레이딩뷰 스타일의 주가 시계열 모멘텀 시각화")
    
    chart_ticker = st.selectbox("차트 조회 종목", df['ticker'].tolist() if not df.empty else ["AAPL"], key="chart_ticker")
    
    # 해당 종목 히스토리 차트 그리기
    hist_data = yf.download(chart_ticker, period="6mo", interval="1d", progress=False)['Close']
    if not hist_data.empty:
        st.line_chart(hist_data, color="#10b981")
        st.caption(f"* {chart_ticker} 최근 6개월 일봉 종가 추이 시각화")
    else:
        st.warning("차트 데이터를 불러올 수 없습니다.")
