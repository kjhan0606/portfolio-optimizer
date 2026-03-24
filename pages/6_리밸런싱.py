"""리밸런싱 시뮬레이션 및 안내"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="⑥ 리밸런싱", page_icon="🔄", layout="wide")
st.title("🔄 ⑥ 리밸런싱 시뮬레이션")

if 'returns' not in st.session_state:
    st.warning("⚠️ 먼저 '① 종목선택·데이터'에서 데이터를 수집해 주세요.")
    st.stop()

returns = st.session_state['returns']
price_df = st.session_state['price_df']
n = len(returns.columns)
names = list(returns.columns)

# ── 현재 최적 비중 계산 ──
def get_optimal_weights(strategy, max_w=0.3):
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    w0 = np.ones(n) / n
    bounds = [(0, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    if strategy == '최소분산(MVP)':
        def vol(w): return np.sqrt(w @ cov @ w)
        res = minimize(vol, w0, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x if res.success else w0
    elif strategy == '리스크패리티':
        def rp_obj(w):
            pv = np.sqrt(w @ cov @ w)
            mr = cov @ w
            rc = w * mr / pv
            return np.sum((rc - pv/n)**2)
        res = minimize(rp_obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x if res.success else w0
    elif strategy == '최대샤프비율':
        def neg_sharpe(w):
            return -(w @ mu - 0.035) / np.sqrt(w @ cov @ w)
        res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x if res.success else w0
    else:
        return w0

# ── UI ──
st.markdown("### ⚙️ 리밸런싱 설정")
col1, col2, col3 = st.columns(3)
with col1:
    strategy = st.selectbox("최적화 전략", ['최소분산(MVP)', '리스크패리티', '최대샤프비율', '균등배분'])
with col2:
    rebal_freq = st.selectbox("리밸런싱 주기", ['월간 (매월)', '분기 (3개월)'])
with col3:
    max_weight = st.slider("비중 상한(%)", 10, 50, 30) / 100

total_invest = st.number_input("총 투자금 (만원)", value=10000, step=1000)

if st.button("🔄 리밸런싱 분석 실행", type="primary", use_container_width=True):
    optimal_w = get_optimal_weights(strategy, max_weight)
    
    st.markdown("---")
    st.markdown(f"### 📊 {strategy} — 현재 최적 포트폴리오")
    
    # 비중 테이블
    weight_df = pd.DataFrame({
        '종목': names,
        '비중(%)': np.round(optimal_w * 100, 2),
        '투자금(만원)': np.round(optimal_w * total_invest, 0).astype(int),
    }).sort_values('비중(%)', ascending=False)
    weight_df = weight_df[weight_df['비중(%)'] > 0.1].reset_index(drop=True)
    
    # 현재 주가 기준 매수 수량
    last_prices = price_df.iloc[-1]
    weight_df['현재가(원)'] = weight_df['종목'].map(lambda x: int(last_prices[x]) if x in last_prices.index else 0)
    weight_df['매수수량(주)'] = (weight_df['투자금(만원)'] * 10000 / weight_df['현재가(원)']).fillna(0).astype(int)
    weight_df['실투자금(만원)'] = (weight_df['매수수량(주)'] * weight_df['현재가(원)'] / 10000).round(1)
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.dataframe(weight_df, use_container_width=True, height=500)
    with col2:
        fig = go.Figure(data=[go.Pie(labels=weight_df['종목'], values=weight_df['비중(%)'], hole=0.35)])
        fig.update_layout(title=f"{strategy} 비중", height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # 요약
    actual_invest = weight_df['실투자금(만원)'].sum()
    remaining = total_invest - actual_invest
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 투자금", f"{total_invest:,}만원")
    col2.metric("실 투자금", f"{actual_invest:,.1f}만원")
    col3.metric("잔여 현금", f"{remaining:,.1f}만원")
    col4.metric("편입 종목 수", f"{len(weight_df)}개")
    
    # ── 리밸런싱 시뮬레이션 (리밸런싱 O vs X) ──
    st.markdown("---")
    st.markdown("### 📊 리밸런싱 효과 비교 (리밸런싱 O vs X)")
    
    rebal_days = 21 if '월간' in rebal_freq else 63
    lb_days = 252
    start_idx = lb_days
    
    if start_idx < len(returns):
        dates = returns.index[start_idx:]
        
        # 리밸런싱 O
        port_rebal = [1.0]
        w_current = optimal_w.copy()
        last_r = 0
        
        # 리밸런싱 X (Buy & Hold)
        port_hold = [1.0]
        w_hold = optimal_w.copy()
        
        for i, date in enumerate(dates):
            idx = start_idx + i
            daily_ret = returns.iloc[idx].values
            
            # 리밸런싱 O
            if i - last_r >= rebal_days or i == 0:
                train = returns.iloc[max(0, idx-lb_days):idx]
                w_current = get_optimal_weights(strategy, max_weight)
                last_r = i
            
            pr = w_current @ daily_ret
            port_rebal.append(port_rebal[-1] * (1 + pr))
            
            # 리밸런싱 X — 비중이 시장에 따라 변동
            hr = w_hold @ daily_ret
            port_hold.append(port_hold[-1] * (1 + hr))
            # 비중 드리프트
            w_hold = w_hold * (1 + daily_ret)
            w_hold = w_hold / w_hold.sum()
        
        rebal_s = pd.Series(port_rebal[1:], index=dates)
        hold_s = pd.Series(port_hold[1:], index=dates)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=(rebal_s-1)*100, name=f'리밸런싱 O ({rebal_freq})',
                                  line=dict(color='#E53935', width=2.5)))
        fig2.add_trace(go.Scatter(x=dates, y=(hold_s-1)*100, name='리밸런싱 X (Buy&Hold)',
                                  line=dict(color='#1E88E5', width=2, dash='dash')))
        fig2.update_layout(title="리밸런싱 효과 비교", yaxis_title="누적수익률(%)", height=450)
        st.plotly_chart(fig2, use_container_width=True)
        
        # 비교 지표
        def quick_metrics(s):
            d = s.pct_change().dropna()
            yrs = len(d)/252
            ann_r = ((s.iloc[-1])**(1/yrs)-1)*100 if yrs>0 else 0
            ann_v = d.std()*np.sqrt(252)*100
            mdd = ((s-s.cummax())/s.cummax()).min()*100
            sr = (ann_r/100-0.035)/(ann_v/100) if ann_v>0 else 0
            return {'연환산수익률(%)': round(ann_r,2), '연간변동성(%)': round(ann_v,2),
                    '샤프비율': round(sr,3), 'MDD(%)': round(mdd,2)}
        
        cmp = pd.DataFrame({
            f'리밸런싱 O ({rebal_freq})': quick_metrics(rebal_s),
            '리밸런싱 X (Buy&Hold)': quick_metrics(hold_s)
        })
        st.dataframe(cmp.T.style.format("{:.2f}"), use_container_width=True)
    
    # ── 리밸런싱 체크리스트 ──
    st.markdown("---")
    st.markdown("### 📋 리밸런싱 실행 체크리스트")
    
    freq_text = "매월 첫 영업일" if '월간' in rebal_freq else "매 분기 첫 영업일 (1월, 4월, 7월, 10월)"
    
    st.markdown(f"""
    **리밸런싱 주기:** {freq_text}
    
    **실행 절차:**
    1. ☐ 현재 보유 종목 평가금액 확인
    2. ☐ 이 프로그램에서 최적 비중 재산출
    3. ☐ 목표 비중과 현재 비중의 차이 계산
    4. ☐ 비중 초과 종목 → 매도
    5. ☐ 비중 부족 종목 → 매수
    6. ☐ 거래 비용(수수료+세금) 고려: 차이 2% 미만이면 유보
    7. ☐ 거래 완료 후 실제 비중 재확인
    
    **💡 팁:**
    - 리밸런싱 비용을 줄이려면, 추가 입금 시 부족한 종목 위주로 매수
    - 비중 차이가 작은 종목은 건너뛰어 거래비용 절감
    - 세금 매도 시 양도소득세 고려 (대주주 기준, ETF 과세)
    """)

st.sidebar.info("리밸런싱:\n주기적으로 비중을 재조정하여\n리스크를 관리합니다.")
