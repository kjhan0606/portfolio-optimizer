"""마코위츠 효율적 프론티어 시각화"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config(page_title="④ 효율적프론티어", page_icon="📈", layout="wide")
st.title("📈 ④ 효율적 프론티어 (Efficient Frontier)")

if 'returns' not in st.session_state:
    st.warning("⚠️ 먼저 '① 종목선택·데이터'에서 데이터를 수집해 주세요.")
    st.stop()

returns = st.session_state['returns']
n = len(returns.columns)
names = list(returns.columns)
mu = returns.mean().values * 252
cov_mat = returns.cov().values * 252
rf = 0.035

col1, col2 = st.columns(2)
with col1:
    max_weight = st.slider("개별 비중 상한(%)", 10, 50, 30) / 100
with col2:
    n_points = st.slider("프론티어 포인트 수", 30, 200, 80)

bounds = [(0, max_weight)] * n
constraints_eq = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

def port_vol(w):
    return np.sqrt(w @ cov_mat @ w)

def port_ret(w):
    return w @ mu

if st.button("🚀 효율적 프론티어 산출", type="primary", use_container_width=True):
    with st.spinner("효율적 프론티어 계산 중..."):
        # 최소분산
        w0 = np.ones(n) / n
        res_min = minimize(port_vol, w0, method='SLSQP', bounds=bounds, constraints=constraints_eq)
        min_vol = port_vol(res_min.x)
        min_ret = port_ret(res_min.x)
        
        # 최대수익
        res_max = minimize(lambda w: -port_ret(w), w0, method='SLSQP', bounds=bounds, constraints=constraints_eq)
        max_ret = port_ret(res_max.x)
        
        # 프론티어 포인트
        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_vols = []
        frontier_rets = []
        frontier_weights = []
        
        for target in target_returns:
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, t=target: port_ret(w) - t}
            ]
            try:
                res = minimize(port_vol, w0, method='SLSQP', bounds=bounds, constraints=cons)
                if res.success:
                    frontier_vols.append(port_vol(res.x) * 100)
                    frontier_rets.append(port_ret(res.x) * 100)
                    frontier_weights.append(res.x)
            except:
                pass
        
        # 최대 샤프
        def neg_sharpe(w):
            return -(port_ret(w) - rf) / port_vol(w)
        res_sharpe = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints_eq)
        sharpe_vol = port_vol(res_sharpe.x) * 100
        sharpe_ret = port_ret(res_sharpe.x) * 100
        
        # 랜덤 포트폴리오 (참고용)
        rand_vols, rand_rets, rand_sharpes = [], [], []
        for _ in range(3000):
            rw = np.random.random(n)
            rw = np.minimum(rw, max_weight)
            rw = rw / rw.sum()
            rv = port_vol(rw) * 100
            rr = port_ret(rw) * 100
            rand_vols.append(rv)
            rand_rets.append(rr)
            rand_sharpes.append((rr/100 - rf) / (rv/100))
    
    # ── 프론티어 시각화 ──
    st.markdown("### 📊 마코위츠 효율적 프론티어")
    
    fig = go.Figure()
    
    # 랜덤 포트폴리오
    fig.add_trace(go.Scatter(
        x=rand_vols, y=rand_rets, mode='markers', name='랜덤 포트폴리오',
        marker=dict(size=3, color=rand_sharpes, colorscale='Viridis',
                    showscale=True, colorbar=dict(title='샤프비율')),
        opacity=0.5
    ))
    
    # 효율적 프론티어
    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets, mode='lines',
        name='효율적 프론티어', line=dict(color='red', width=3)
    ))
    
    # 최소분산 포인트
    fig.add_trace(go.Scatter(
        x=[min_vol*100], y=[min_ret*100], mode='markers+text',
        name='최소분산(MVP)', text=['MVP'],
        marker=dict(size=18, color='blue', symbol='star'),
        textposition='top right'
    ))
    
    # 최대 샤프 포인트
    fig.add_trace(go.Scatter(
        x=[sharpe_vol], y=[sharpe_ret], mode='markers+text',
        name='최대샤프비율', text=['Max Sharpe'],
        marker=dict(size=18, color='green', symbol='diamond'),
        textposition='top right'
    ))
    
    # 개별 종목
    for i, name in enumerate(names):
        vol_i = np.sqrt(cov_mat[i,i]) * 100
        ret_i = mu[i] * 100
        fig.add_trace(go.Scatter(
            x=[vol_i], y=[ret_i], mode='markers+text',
            name=name, text=[name], textposition='top center',
            marker=dict(size=10, symbol='circle'),
            showlegend=False
        ))
    
    # CML (Capital Market Line)
    cml_x = np.linspace(0, max(frontier_vols)*1.1, 100)
    sharpe_ratio = (sharpe_ret/100 - rf) / (sharpe_vol/100)
    cml_y = (rf + sharpe_ratio * cml_x/100) * 100
    fig.add_trace(go.Scatter(
        x=cml_x, y=cml_y, mode='lines', name='CML(자본시장선)',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="효율적 프론티어 & 자본시장선 (CML)",
        xaxis_title="연간 변동성 (%)", yaxis_title="연간 수익률 (%)",
        height=700, hovermode='closest',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 주요 포인트 정보
    st.markdown("### 📊 핵심 포트폴리오 요약")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔵 최소분산(MVP)", f"변동성 {min_vol*100:.2f}%", f"수익률 {min_ret*100:.2f}%")
    with col2:
        st.metric("🟢 최대샤프비율", f"변동성 {sharpe_vol:.2f}%", f"수익률 {sharpe_ret:.2f}%")
    with col3:
        st.metric("📊 샤프비율", f"{sharpe_ratio:.3f}", f"무위험이자율 {rf*100:.1f}%")
    
    # 프론티어 상 포트폴리오 선택
    st.markdown("### 🎚️ 프론티어 상 포트폴리오 탐색")
    if frontier_weights:
        idx = st.slider("프론티어 위치 선택", 0, len(frontier_weights)-1, 0)
        w_sel = frontier_weights[idx]
        
        sel_df = pd.DataFrame({
            '종목': names,
            '비중(%)': np.round(w_sel * 100, 2)
        }).sort_values('비중(%)', ascending=False)
        sel_df = sel_df[sel_df['비중(%)'] > 0.1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**수익률:** {frontier_rets[idx]:.2f}% | **변동성:** {frontier_vols[idx]:.2f}%")
            st.dataframe(sel_df.reset_index(drop=True), use_container_width=True)
        with col2:
            fig_pie = go.Figure(data=[go.Pie(labels=sel_df['종목'], values=sel_df['비중(%)'], hole=0.3)])
            fig_pie.update_layout(title="선택 포트폴리오 비중", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
