"""포트폴리오 백테스트"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="⑤ 백테스트", page_icon="🔙", layout="wide")
st.title("🔙 ⑤ 포트폴리오 백테스트")

if 'returns' not in st.session_state:
    st.warning("⚠️ 먼저 '① 종목선택·데이터'에서 데이터를 수집해 주세요.")
    st.stop()

returns = st.session_state['returns']
price_df = st.session_state['price_df']
n = len(returns.columns)
names = list(returns.columns)

# ── 최적화 함수 ──
def optimize_mvp(ret_data, max_w=0.3):
    n_assets = ret_data.shape[1]
    cov = ret_data.cov().values * 252
    bounds = [(0, max_w)] * n_assets
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    w0 = np.ones(n_assets) / n_assets
    
    def vol(w):
        return np.sqrt(w @ cov @ w)
    
    try:
        res = minimize(vol, w0, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x if res.success else w0
    except:
        return w0

# ── 설정 ──
st.markdown("### ⚙️ 백테스트 설정")
col1, col2, col3, col4 = st.columns(4)
with col1:
    lookback = st.selectbox("학습 기간", ['6개월','1년','2년','3년'], index=1)
    lb_map = {'6개월':126, '1년':252, '2년':504, '3년':756}
    lb_days = lb_map[lookback]
with col2:
    rebal = st.selectbox("리밸런싱 주기", ['월간','분기'])
    rebal_days = 21 if rebal == '월간' else 63
with col3:
    max_weight = st.slider("비중 상한(%)", 10, 50, 30) / 100
with col4:
    initial_capital = st.number_input("초기 투자금(만원)", value=10000, step=1000)

strategy = st.selectbox("최적화 전략", ['최소분산(MVP)', '균등배분', '시가총액 가중(근사)'])

if st.button("🚀 백테스트 실행", type="primary", use_container_width=True):
    with st.spinner("백테스트 실행 중..."):
        # 백테스트 시작점
        start_idx = lb_days
        if start_idx >= len(returns):
            st.error("데이터가 부족합니다.")
            st.stop()
        
        dates = returns.index[start_idx:]
        port_values = [1.0]
        equal_values = [1.0]
        weights_history = []
        rebal_dates = []
        
        current_weights = np.ones(n) / n
        last_rebal = 0
        
        for i, date in enumerate(dates):
            idx = start_idx + i
            
            # 리밸런싱 체크
            if i - last_rebal >= rebal_days or i == 0:
                train_data = returns.iloc[max(0, idx-lb_days):idx]
                
                if strategy == '최소분산(MVP)':
                    current_weights = optimize_mvp(train_data, max_weight)
                elif strategy == '균등배분':
                    current_weights = np.ones(n) / n
                else:  # 시가총액 근사 (가격 기반)
                    prices = price_df.iloc[idx-1].values
                    w = prices / prices.sum()
                    w = np.minimum(w, max_weight)
                    current_weights = w / w.sum()
                
                last_rebal = i
                rebal_dates.append(date)
                weights_history.append({'date': date, 'weights': current_weights.copy()})
            
            # 일별 수익률 적용
            daily_ret = returns.iloc[idx].values
            port_ret = current_weights @ daily_ret
            equal_ret = np.ones(n) / n @ daily_ret
            
            port_values.append(port_values[-1] * (1 + port_ret))
            equal_values.append(equal_values[-1] * (1 + equal_ret))
        
        # KOSPI 벤치마크 (가능하면)
        try:
            import yfinance as yf
            kospi = yf.download('^KS11', start=dates[0], end=dates[-1], progress=False, auto_adjust=True)
            if isinstance(kospi.columns, pd.MultiIndex):
                kospi.columns = kospi.columns.get_level_values(0)
            kospi_ret = kospi['Close'].pct_change().dropna()
            kospi_values = [1.0]
            for r in kospi_ret:
                kospi_values.append(kospi_values[-1] * (1 + r))
            has_kospi = True
        except:
            has_kospi = False
        
        port_series = pd.Series(port_values[1:], index=dates)
        equal_series = pd.Series(equal_values[1:], index=dates)
    
    st.success("✅ 백테스트 완료!")
    
    # ── 누적 수익률 ──
    st.markdown("### 📊 누적 수익률")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=(port_series-1)*100, name=f'{strategy}',
                             line=dict(color='#E53935', width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=(equal_series-1)*100, name='균등배분',
                             line=dict(color='#1E88E5', width=1.5, dash='dot')))
    if has_kospi:
        kospi_idx = pd.Series(kospi_values[1:len(dates)+1], index=dates[:len(kospi_values)-1])
        fig.add_trace(go.Scatter(x=kospi_idx.index, y=(kospi_idx-1)*100, name='KOSPI',
                                 line=dict(color='gray', width=1.5, dash='dash')))
    
    # 리밸런싱 포인트
    for rd in rebal_dates[:50]:  # 최대 50개
        fig.add_vline(x=rd, line_dash="dot", line_color="lightgray", opacity=0.3)
    
    fig.update_layout(title=f"백테스트 누적수익률 ({strategy}, {rebal} 리밸런싱)",
                      xaxis_title="날짜", yaxis_title="누적수익률(%)", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # ── 성과 지표 ──
    st.markdown("### 📊 성과 지표")
    
    port_daily = port_series.pct_change().dropna()
    eq_daily = equal_series.pct_change().dropna()
    
    def calc_metrics(values_series, daily_returns):
        total_ret = (values_series.iloc[-1] - 1) * 100
        years = len(daily_returns) / 252
        ann_ret = ((values_series.iloc[-1]) ** (1/years) - 1) * 100 if years > 0 else 0
        ann_vol = daily_returns.std() * np.sqrt(252) * 100
        sharpe = (ann_ret/100 - 0.035) / (ann_vol/100) if ann_vol > 0 else 0
        
        cummax = values_series.cummax()
        drawdown = (values_series - cummax) / cummax
        mdd = drawdown.min() * 100
        
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        
        return {
            '총수익률(%)': round(total_ret, 2),
            '연환산수익률(%)': round(ann_ret, 2),
            '연간변동성(%)': round(ann_vol, 2),
            '샤프비율': round(sharpe, 3),
            'MDD(%)': round(mdd, 2),
            '칼마비율': round(calmar, 3),
        }
    
    port_metrics = calc_metrics(port_series, port_daily)
    eq_metrics = calc_metrics(equal_series, eq_daily)
    
    metrics_df = pd.DataFrame({strategy: port_metrics, '균등배분': eq_metrics})
    
    if has_kospi and len(kospi_idx) > 10:
        kospi_daily = kospi_idx.pct_change().dropna()
        kospi_metrics = calc_metrics(kospi_idx, kospi_daily)
        metrics_df['KOSPI'] = kospi_metrics
    
    st.dataframe(metrics_df.T.style.format("{:.2f}"), use_container_width=True)
    
    # 투자금 기준 결과
    st.markdown(f"### 💰 투자금 {initial_capital:,}만원 기준")
    final_port = initial_capital * port_series.iloc[-1]
    final_eq = initial_capital * equal_series.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{strategy}", f"{final_port:,.0f}만원",
                f"{final_port - initial_capital:+,.0f}만원")
    col2.metric("균등배분", f"{final_eq:,.0f}만원",
                f"{final_eq - initial_capital:+,.0f}만원")
    if has_kospi and len(kospi_idx) > 1:
        final_k = initial_capital * kospi_idx.iloc[-1]
        col3.metric("KOSPI", f"{final_k:,.0f}만원",
                    f"{final_k - initial_capital:+,.0f}만원")
    
    # ── 낙폭 차트 ──
    st.markdown("### 📉 낙폭 (Drawdown)")
    cummax_p = port_series.cummax()
    dd_p = (port_series - cummax_p) / cummax_p * 100
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dates, y=dd_p, fill='tozeroy', name=strategy,
                                fillcolor='rgba(229,57,53,0.3)', line=dict(color='#E53935')))
    fig_dd.update_layout(title="포트폴리오 낙폭", yaxis_title="낙폭(%)", height=350)
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # ── 월별 수익률 히트맵 ──
    st.markdown("### 📊 월별 수익률 히트맵")
    monthly = port_series.resample('ME').last().pct_change().dropna() * 100
    monthly_df = monthly.to_frame('수익률')
    monthly_df['연도'] = monthly_df.index.year
    monthly_df['월'] = monthly_df.index.month
    pivot = monthly_df.pivot_table(values='수익률', index='연도', columns='월', aggfunc='first')
    pivot.columns = [f'{m}월' for m in pivot.columns]
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
        colorscale='RdYlGn', text=pivot.values.round(1),
        texttemplate='%{text}%', zmin=-10, zmax=10
    ))
    fig_heat.update_layout(title="월별 수익률 (%)", height=300)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # 세션 저장
    st.session_state['backtest'] = {
        'port_series': port_series,
        'equal_series': equal_series,
        'weights_history': weights_history,
        'metrics': port_metrics,
        'strategy': strategy,
        'rebal': rebal,
    }
