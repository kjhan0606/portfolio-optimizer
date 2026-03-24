"""종목 선택 및 주가 데이터 수집"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="① 종목선택·데이터", page_icon="📊", layout="wide")
st.title("📊 ① 종목 선택 및 데이터 수집")

# ── 인기 종목 레퍼런스 ──
POPULAR_STOCKS = {
    '삼성전자': '005930', 'SK하이닉스': '000660', 'LG에너지솔루션': '373220',
    '삼성바이오로직스': '207940', '현대차': '005380', 'NAVER': '035420',
    '카카오': '035720', 'LG화학': '051910', 'POSCO홀딩스': '005490',
    '삼성SDI': '006400', '기아': '000270', 'KB금융': '105560',
    '셀트리온': '068270', '신한지주': '055550', '하나금융지주': '086790',
    '현대모비스': '012330', 'LG전자': '066570', 'SK이노베이션': '096770',
    '한국전력': '015760', 'KT&G': '033780', '삼성물산': '028260',
    '삼성생명': '032830', 'HMM': '011200', '두산에너빌리티': '034020',
    'SK텔레콤': '017670', '한화에어로스페이스': '012450', 'LG': '003550',
    '넷마블': '251270', '크래프톤': '259960', '엔씨소프트': '036570',
}

POPULAR_ETFS = {
    'KODEX 200': '069500', 'KODEX 코스닥150': '229200',
    'TIGER 200': '102110', 'KODEX 삼성그룹': '102780',
    'KODEX 레버리지': '122630', 'KODEX 인버스': '114800',
    'TIGER 차이나전기차': '371460', 'KODEX 2차전지산업': '305720',
    'KODEX 반도체': '091160', 'TIGER 미국S&P500': '360750',
    'KODEX 미국나스닥100': '379810', 'TIGER 미국나스닥100': '133690',
    'KODEX 배당가치': '290080', 'TIGER 200 IT': '139260',
    'KODEX 고배당': '279530', 'KODEX 금융': '102970',
    'TIGER KRX바이오K-뉴딜': '364970', 'KODEX 철강': '117700',
    'KODEX 자동차': '091170', 'KODEX 건설': '117680',
}

def get_yf_ticker(code):
    """한국 종목코드를 yfinance 티커로 변환"""
    code = str(code).strip().zfill(6)
    return f"{code}.KS"

@st.cache_data(ttl=3600, show_spinner="주가 데이터 수집 중...")
def fetch_stock_data(tickers_dict, years=5):
    """여러 종목의 주가 데이터 수집"""
    end = datetime.now()
    start = end - timedelta(days=years*365)
    
    all_data = {}
    errors = []
    
    for name, code in tickers_dict.items():
        try:
            yf_ticker = get_yf_ticker(code)
            data = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data is not None and len(data) > 60:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                all_data[name] = data['Close']
            else:
                # KOSDAQ 시도
                yf_ticker_kq = f"{str(code).zfill(6)}.KQ"
                data = yf.download(yf_ticker_kq, start=start, end=end, progress=False, auto_adjust=True)
                if data is not None and len(data) > 60:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    all_data[name] = data['Close']
                else:
                    errors.append(f"{name}({code})")
        except Exception as e:
            errors.append(f"{name}({code}): {str(e)[:50]}")
    
    if all_data:
        price_df = pd.DataFrame(all_data)
        price_df = price_df.dropna(how='all').ffill()
        return price_df, errors
    return None, errors

# ═══════════════════════════════════════
# UI
# ═══════════════════════════════════════
st.markdown("### 📌 종목 선택")

tab1, tab2, tab3 = st.tabs(["🔢 직접 입력", "⭐ 인기 종목 선택", "📋 사용 가이드"])

with tab1:
    st.markdown("**종목코드를 직접 입력하세요** (한 줄에 하나씩: `종목명,코드`)")
    default_text = """삼성전자,005930
SK하이닉스,000660
NAVER,035420
카카오,035720
현대차,005380
LG에너지솔루션,373220
KB금융,105560
KODEX 200,069500
KODEX 반도체,091160
TIGER 미국S&P500,360750"""
    
    text_input = st.text_area("종목 입력 (종목명,코드)", value=default_text, height=250)

with tab2:
    st.markdown("**개별 종목 선택:**")
    cols = st.columns(5)
    selected_stocks = {}
    for i, (name, code) in enumerate(POPULAR_STOCKS.items()):
        with cols[i % 5]:
            if st.checkbox(name, key=f"stock_{code}"):
                selected_stocks[name] = code
    
    st.markdown("**ETF 선택:**")
    cols2 = st.columns(5)
    for i, (name, code) in enumerate(POPULAR_ETFS.items()):
        with cols2[i % 5]:
            if st.checkbox(name, key=f"etf_{code}"):
                selected_stocks[name] = code

with tab3:
    st.markdown("""
    ### 사용 가이드
    
    **종목코드 찾는 방법:**
    - [네이버 금융](https://finance.naver.com) → 종목 검색 → 6자리 코드 확인
    - 예: 삼성전자 = 005930, KODEX 200 = 069500
    
    **권장 포트폴리오 구성:**
    - 개별 종목 7~20개 + ETF 3~10개
    - 다양한 업종에 분산
    - 대형주 위주 + 일부 중소형
    
    **데이터 기간:**
    - 5년 데이터로 장기 변동성과 상관관계를 정확히 추정
    - 최소 1년 이상 상장된 종목만 사용 가능
    """)

st.markdown("---")

# 데이터 수집 설정
col1, col2 = st.columns(2)
with col1:
    years = st.slider("데이터 기간 (년)", 1, 5, 5)
with col2:
    st.metric("개별 종목 비중 상한", "30%")

# 종목 파싱
tickers = {}

# 직접 입력에서 파싱
if text_input.strip():
    for line in text_input.strip().split('\n'):
        line = line.strip()
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                name = parts[0].strip()
                code = parts[1].strip()
                if name and code:
                    tickers[name] = code

# 체크박스 선택 추가
tickers.update(selected_stocks)

st.info(f"📋 선택된 종목: **{len(tickers)}개** — {', '.join(tickers.keys())}")

if st.button("🚀 데이터 수집 시작", type="primary", use_container_width=True):
    if len(tickers) < 2:
        st.error("최소 2개 이상의 종목을 선택해 주세요.")
    elif len(tickers) > 30:
        st.warning("30개 이하로 선택해 주세요.")
    else:
        with st.spinner(f"{len(tickers)}개 종목 데이터 수집 중..."):
            price_df, errors = fetch_stock_data(tickers, years)
        
        if errors:
            st.warning(f"⚠️ 데이터 수집 실패 종목: {', '.join(errors)}")
        
        if price_df is not None and len(price_df.columns) >= 2:
            st.session_state['price_df'] = price_df
            st.session_state['tickers'] = {k: v for k, v in tickers.items() if k in price_df.columns}
            st.session_state['years'] = years
            
            st.success(f"✅ {len(price_df.columns)}개 종목, {len(price_df)}거래일 데이터 수집 완료!")
            
            # 기본 정보
            st.markdown("### 📈 수집된 데이터 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("수집 종목", f"{len(price_df.columns)}개")
            col2.metric("데이터 기간", f"{price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
            col3.metric("거래일 수", f"{len(price_df):,}일")
            col4.metric("비중 상한", "30%")
            
            # 최근 주가 표
            st.markdown("### 💰 최근 주가")
            recent = price_df.tail(5).T
            recent.columns = [d.strftime('%m/%d') for d in recent.columns]
            st.dataframe(recent.style.format("{:,.0f}"), use_container_width=True)
            
            # 수익률 계산
            returns = price_df.pct_change().dropna()
            st.session_state['returns'] = returns
            
            # 주가 차트
            st.markdown("### 📊 주가 추이 (정규화)")
            norm = price_df / price_df.iloc[0] * 100
            
            fig = go.Figure()
            for col in norm.columns:
                fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col, mode='lines'))
            fig.update_layout(
                title="종목별 주가 추이 (시작일 = 100)",
                yaxis_title="정규화 주가",
                xaxis_title="날짜",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 수익률 통계
            st.markdown("### 📊 일별 수익률 기초통계")
            ret_stats = returns.describe().T
            ret_stats['연간수익률(%)'] = (returns.mean() * 252 * 100).round(2)
            ret_stats['연간변동성(%)'] = (returns.std() * np.sqrt(252) * 100).round(2)
            ret_stats['샤프비율'] = ((returns.mean() * 252) / (returns.std() * np.sqrt(252))).round(3)
            st.dataframe(ret_stats[['연간수익률(%)','연간변동성(%)','샤프비율','mean','std','min','max']].style.format("{:.4f}"),
                        use_container_width=True)
        else:
            st.error("❌ 유효한 데이터를 수집하지 못했습니다. 종목코드를 확인해 주세요.")

# 기존 데이터 표시
if 'price_df' in st.session_state:
    st.sidebar.success(f"✅ {len(st.session_state['price_df'].columns)}개 종목 데이터 로드됨")
    st.sidebar.write("종목:", ', '.join(st.session_state['price_df'].columns))
