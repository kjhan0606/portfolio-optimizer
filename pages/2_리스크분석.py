"""개별 종목 리스크 분석"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="② 리스크분석", page_icon="⚠️", layout="wide")
st.title("⚠️ ② 개별 종목 리스크 분석")

if 'returns' not in st.session_state:
    st.warning("⚠️ 먼저 '① 종목선택·데이터'에서 데이터를 수집해 주세요.")
    st.stop()

returns = st.session_state['returns']
price_df = st.session_state['price_df']

# ── 리스크 지표 계산 함수 ──
def calc_var(returns_series, confidence=0.95):
    """Value at Risk (Historical)"""
    return np.percentile(returns_series.dropna(), (1-confidence)*100)

def calc_cvar(returns_series, confidence=0.95):
    """Conditional VaR (Expected Shortfall)"""
    var = calc_var(returns_series, confidence)
    return returns_series[returns_series <= var].mean()

def calc_mdd(price_series):
    """Maximum Drawdown"""
    cummax = price_series.cummax()
    drawdown = (price_series - cummax) / cummax
    return drawdown.min()

def calc_beta(stock_returns, market_returns):
    """Beta 계산"""
    cov = np.cov(stock_returns.dropna(), market_returns.dropna())
    if cov.shape == (2,2) and cov[1,1] != 0:
        return cov[0,1] / cov[1,1]
    return np.nan

# ── 리스크 종합 테이블 ──
st.markdown("### 📊 종목별 리스크 지표 종합")

confidence = st.slider("VaR/CVaR 신뢰수준", 0.90, 0.99, 0.95, 0.01)

risk_data = []
for col in returns.columns:
    r = returns[col].dropna()
    p = price_df[col].dropna()
    
    ann_vol = r.std() * np.sqrt(252)
    ann_ret = r.mean() * 252
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    var_val = calc_var(r, confidence)
    cvar_val = calc_cvar(r, confidence)
    mdd = calc_mdd(p)
    skew = r.skew()
    kurt = r.kurtosis()
    
    risk_data.append({
        '종목': col,
        '연간수익률(%)': round(ann_ret*100, 2),
        '연간변동성(%)': round(ann_vol*100, 2),
        '샤프비율': round(sharpe, 3),
        f'VaR({confidence:.0%})': round(var_val*100, 2),
        f'CVaR({confidence:.0%})': round(cvar_val*100, 2),
        'MDD(%)': round(mdd*100, 2),
        '왜도(Skew)': round(skew, 3),
        '첨도(Kurt)': round(kurt, 3),
    })

risk_df = pd.DataFrame(risk_data)
st.dataframe(
    risk_df.style.background_gradient(subset=['연간변동성(%)'], cmap='Reds')
               .background_gradient(subset=['MDD(%)'], cmap='Reds')
               .background_gradient(subset=['샤프비율'], cmap='Greens'),
    use_container_width=True, height=400
)

# ── 변동성 비교 차트 ──
st.markdown("### 📊 종목별 연간 변동성 비교")
risk_sorted = risk_df.sort_values('연간변동성(%)')
fig = go.Figure()
fig.add_trace(go.Bar(
    y=risk_sorted['종목'], x=risk_sorted['연간변동성(%)'],
    orientation='h',
    marker_color=px.colors.sequential.Reds_r[:len(risk_sorted)]*3,
    text=risk_sorted['연간변동성(%)'].apply(lambda x: f'{x:.1f}%'),
    textposition='outside'
))
fig.update_layout(title="종목별 연간 변동성 (%)", xaxis_title="변동성(%)", height=max(400, len(risk_sorted)*25))
st.plotly_chart(fig, use_container_width=True)

# ── 수익률-변동성 산점도 ──
st.markdown("### 📊 수익률 vs 변동성 (리스크-리턴 맵)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=risk_df['연간변동성(%)'], y=risk_df['연간수익률(%)'],
    mode='markers+text', text=risk_df['종목'],
    textposition='top center', textfont=dict(size=10),
    marker=dict(size=15, color=risk_df['샤프비율'], colorscale='RdYlGn',
                showscale=True, colorbar=dict(title='샤프비율')),
))
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(title="수익률-변동성 맵 (색상=샤프비율)",
                   xaxis_title="연간 변동성(%)", yaxis_title="연간 수익률(%)", height=500)
st.plotly_chart(fig2, use_container_width=True)

# ── MDD 차트 ──
st.markdown("### 📉 종목별 최대 낙폭 (MDD)")
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    y=risk_df.sort_values('MDD(%)')['종목'],
    x=risk_df.sort_values('MDD(%)')['MDD(%)'],
    orientation='h', marker_color='#E53935',
    text=risk_df.sort_values('MDD(%)')['MDD(%)'].apply(lambda x: f'{x:.1f}%'),
    textposition='outside'
))
fig3.update_layout(title="종목별 MDD (%)", xaxis_title="MDD(%)", height=max(400, len(risk_df)*25))
st.plotly_chart(fig3, use_container_width=True)

# ── 상관행렬 ──
st.markdown("### 🔗 종목 간 상관행렬")
corr = returns.corr()
st.session_state['corr_matrix'] = corr

fig4 = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.index,
    colorscale='RdBu_r', zmin=-1, zmax=1,
    text=corr.values.round(2), texttemplate='%{text}',
    textfont=dict(size=9),
))
fig4.update_layout(title="수익률 상관행렬 (Pearson)", height=max(500, len(corr)*30))
st.plotly_chart(fig4, use_container_width=True)

# ── 공분산행렬 ──
st.markdown("### 📊 연간 공분산행렬")
cov = returns.cov() * 252
st.dataframe(cov.style.format("{:.6f}").background_gradient(cmap='Blues'), use_container_width=True)
st.session_state['cov_matrix'] = cov

# ── VaR/CVaR 분포 ──
st.markdown("### 📊 수익률 분포 및 VaR/CVaR")
selected = st.selectbox("종목 선택", returns.columns)
r_sel = returns[selected].dropna()
var_sel = calc_var(r_sel, confidence)
cvar_sel = calc_cvar(r_sel, confidence)

fig5 = go.Figure()
fig5.add_trace(go.Histogram(x=r_sel*100, nbinsx=80, name='일별 수익률',
                             marker_color='#1E88E5', opacity=0.7))
fig5.add_vline(x=var_sel*100, line_color='red', line_dash='dash',
               annotation_text=f'VaR({confidence:.0%}): {var_sel*100:.2f}%')
fig5.add_vline(x=cvar_sel*100, line_color='darkred', line_dash='dot',
               annotation_text=f'CVaR: {cvar_sel*100:.2f}%')
fig5.update_layout(title=f"{selected} 일별 수익률 분포", xaxis_title="수익률(%)",
                   yaxis_title="빈도", height=400)
st.plotly_chart(fig5, use_container_width=True)

# 리스크 데이터 세션 저장
st.session_state['risk_df'] = risk_df
st.sidebar.success("✅ 리스크 분석 완료")
