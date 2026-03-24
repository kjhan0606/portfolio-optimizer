"""4가지 포트폴리오 최적화 모형"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="③ 포트폴리오최적화", page_icon="🎯", layout="wide")
st.title("🎯 ③ 포트폴리오 최적화")

if 'returns' not in st.session_state:
    st.warning("⚠️ 먼저 '① 종목선택·데이터'에서 데이터를 수집해 주세요.")
    st.stop()

returns = st.session_state['returns']
n = len(returns.columns)
names = list(returns.columns)

# 연간화
mu = returns.mean().values * 252
cov = returns.cov().values * 252
rf = 0.035  # 무위험이자율 3.5%

# ── 최적화 함수들 ──
def portfolio_return(w):
    return w @ mu

def portfolio_vol(w):
    return np.sqrt(w @ cov @ w)

def portfolio_sharpe(w):
    return -(portfolio_return(w) - rf) / portfolio_vol(w)

def portfolio_var(w):
    """포트폴리오 VaR (parametric, 95%)"""
    vol = portfolio_vol(w)
    ret = portfolio_return(w)
    return -(ret - 1.645 * vol)

def portfolio_cvar_historical(w, confidence=0.95):
    """Historical CVaR"""
    port_returns = (returns.values @ w)
    var_threshold = np.percentile(port_returns, (1-confidence)*100)
    return -port_returns[port_returns <= var_threshold].mean() * 252

def risk_parity_objective(w):
    """리스크 패리티 목적함수: 각 종목의 리스크 기여도 균등화"""
    port_vol = portfolio_vol(w)
    marginal_risk = cov @ w
    risk_contribution = w * marginal_risk / port_vol
    target = port_vol / n
    return np.sum((risk_contribution - target)**2)

# ── 설정 ──
st.markdown("### ⚙️ 최적화 설정")
col1, col2, col3 = st.columns(3)
with col1:
    max_weight = st.slider("개별 종목 비중 상한(%)", 10, 50, 30) / 100
with col2:
    min_weight = st.slider("개별 종목 비중 하한(%)", 0, 10, 0) / 100
with col3:
    rf_input = st.number_input("무위험이자율(%)", value=3.5, step=0.1) / 100
    rf = rf_input

# 제약조건
bounds = [(min_weight, max_weight)] * n
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
w0 = np.ones(n) / n  # 균등 배분 초기값

# ── 최적화 실행 ──
if st.button("🚀 포트폴리오 최적화 실행", type="primary", use_container_width=True):
    results = {}
    
    with st.spinner("최적화 중..."):
        # 1) 최소분산 포트폴리오 (MVP)
        try:
            res_mvp = minimize(portfolio_vol, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            if res_mvp.success:
                results['최소분산(MVP)'] = res_mvp.x
        except: pass
        
        # 2) 최대 샤프비율
        try:
            res_sharpe = minimize(portfolio_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            if res_sharpe.success:
                results['최대샤프비율'] = res_sharpe.x
        except: pass
        
        # 3) 리스크 패리티
        try:
            res_rp = minimize(risk_parity_objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            if res_rp.success:
                results['리스크패리티'] = res_rp.x
        except: pass
        
        # 4) CVaR 최소화
        try:
            res_cvar = minimize(lambda w: portfolio_cvar_historical(w), w0, method='SLSQP', 
                               bounds=bounds, constraints=constraints)
            if res_cvar.success:
                results['CVaR최소'] = res_cvar.x
        except: pass
        
        # 5) 균등배분 (벤치마크)
        results['균등배분'] = w0
    
    if not results:
        st.error("최적화에 실패했습니다.")
        st.stop()
    
    st.session_state['opt_results'] = results
    st.success(f"✅ {len(results)}개 모형 최적화 완료!")
    
    # ── 결과 비교 ──
    st.markdown("### 📊 최적화 결과 비교")
    
    compare_data = []
    for model, w in results.items():
        ret = portfolio_return(w)
        vol = portfolio_vol(w)
        sharpe = (ret - rf) / vol if vol > 0 else 0
        port_rets = returns.values @ w
        var_95 = np.percentile(port_rets, 5)
        cvar_95 = port_rets[port_rets <= var_95].mean()
        
        compare_data.append({
            '모형': model,
            '연간수익률(%)': round(ret*100, 2),
            '연간변동성(%)': round(vol*100, 2),
            '샤프비율': round(sharpe, 3),
            'VaR(95%)': round(var_95*100, 2),
            'CVaR(95%)': round(cvar_95*100, 2),
            '최대비중(%)': round(w.max()*100, 1),
            '종목수(>1%)': int(np.sum(w > 0.01)),
        })
    
    compare_df = pd.DataFrame(compare_data)
    st.dataframe(
        compare_df.style.background_gradient(subset=['연간변동성(%)'], cmap='Reds_r')
                       .background_gradient(subset=['샤프비율'], cmap='Greens'),
        use_container_width=True
    )
    
    # ── 수익률-변동성 산점도 ──
    st.markdown("### 📊 모형별 위험-수익 위치")
    fig = go.Figure()
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    for i, (model, w) in enumerate(results.items()):
        ret = portfolio_return(w) * 100
        vol = portfolio_vol(w) * 100
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode='markers+text',
            name=model, text=[model], textposition='top center',
            marker=dict(size=20, color=colors[i % len(colors)]),
        ))
    fig.update_layout(title="모형별 위험-수익 위치", xaxis_title="연간 변동성(%)",
                      yaxis_title="연간 수익률(%)", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # ── 각 모형별 비중 ──
    st.markdown("### 📊 모형별 종목 비중")
    
    tabs = st.tabs(list(results.keys()))
    for tab, (model, w) in zip(tabs, results.items()):
        with tab:
            weight_df = pd.DataFrame({
                '종목': names,
                '비중(%)': np.round(w * 100, 2)
            }).sort_values('비중(%)', ascending=False)
            weight_df = weight_df[weight_df['비중(%)'] > 0.1]
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(weight_df.reset_index(drop=True), use_container_width=True)
            with col2:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=weight_df['종목'], values=weight_df['비중(%)'],
                    textinfo='label+percent', hole=0.3
                )])
                fig_pie.update_layout(title=f"{model} — 종목 비중", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # ── 리스크 기여도 (MVP) ──
    if '최소분산(MVP)' in results:
        st.markdown("### 📊 최소분산 포트폴리오 — 리스크 기여도")
        w_mvp = results['최소분산(MVP)']
        port_vol = portfolio_vol(w_mvp)
        marginal = cov @ w_mvp
        risk_contrib = w_mvp * marginal / port_vol
        
        rc_df = pd.DataFrame({
            '종목': names,
            '비중(%)': np.round(w_mvp * 100, 2),
            '리스크기여도(%)': np.round(risk_contrib / risk_contrib.sum() * 100, 2)
        }).sort_values('리스크기여도(%)', ascending=False)
        rc_df = rc_df[rc_df['비중(%)'] > 0.1]
        
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(name='비중', x=rc_df['종목'], y=rc_df['비중(%)'], marker_color='#1E88E5'))
        fig_rc.add_trace(go.Bar(name='리스크기여도', x=rc_df['종목'], y=rc_df['리스크기여도(%)'], marker_color='#E53935'))
        fig_rc.update_layout(barmode='group', title="MVP: 비중 vs 리스크 기여도", height=400)
        st.plotly_chart(fig_rc, use_container_width=True)

st.sidebar.info("최적화 모형:\n- 최소분산(MVP)\n- 최대샤프비율\n- 리스크패리티\n- CVaR 최소")
