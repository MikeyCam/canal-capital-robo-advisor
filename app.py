# Canal Capital – Streamlit Roboadvisor MVP
# Uses daily returns file built separately (build_data.py)
# Educational only – not investment advice.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Canal Capital – Roboadvisor MVP", page_icon="🛶", layout="wide")

# ----------------------------
# Branding
# ----------------------------
st.title("🛶 Canal Capital")
st.caption("Amsterdam-born, globally diversified. Educational prototype – not investment advice.")

# ----------------------------
# Sidebar – Risk Questions
# ----------------------------
st.sidebar.header("Client Profile (5 quick questions)")
age = st.sidebar.number_input("1) Age", 18, 90, 34, 1)
horizon = st.sidebar.slider("2) Investment horizon (years)", 1, 40, 10)
drawdown_ok = st.sidebar.slider("3) Max drawdown you could stomach next year (%)", 5, 80, 30)
experience = st.sidebar.select_slider("4) Investing experience", options=list(range(0,11)), value=5)
liquidity_need = st.sidebar.selectbox("5) Need access to cash in next 3 years?", ["No", "Maybe", "Yes"])

st.sidebar.markdown("---")
rebalance = st.sidebar.selectbox("Rebalance frequency", ["Annual","Quarterly","Monthly"])
risk_levels = st.sidebar.slider("Number of risk bands", 3, 9, 5)

# ----------------------------
# Risk Scoring → Target Volatility
# ----------------------------
def risk_score(age, horizon, drawdown_ok, experience, liquidity_need):
    s = 0
    s += max(0, 40 - min(age,80)) * 0.7
    s += min(horizon, 30) * 1.2
    s += (drawdown_ok/10) * 3.0           # drawdown slider is %; divide by 10 → 0–8
    s += experience * 1.0                 # 0–10
    s += (0 if liquidity_need=="Yes" else (3 if liquidity_need=="Maybe" else 6))
    return int(max(0, min(100, round(s,0))))

score = risk_score(age, horizon, drawdown_ok, experience, liquidity_need)

# Map score → target annualized volatility band
def target_vol_from_score(s):
    # 0→5%, 100→20%
    return 0.05 + (0.20-0.05)*(s/100)

target_vol = target_vol_from_score(score)

# ----------------------------
# Load Data (daily returns from build_data.py)
# ----------------------------
@st.cache_data
def load_data():
    rets = pd.read_csv("etf_returns_daily.csv", index_col=0, parse_dates=True)
    # keep only numeric columns (drop any stray string/object cols)
    rets = rets.apply(pd.to_numeric, errors="coerce")
    return rets.dropna(axis=1, how="all")


rets = load_data()
st.success(f"Loaded {rets.shape[1]} ETFs with {rets.shape[0]} daily return rows.")

# Annualized stats
mu = rets.mean() * 252            # annualized mean
cov = rets.cov() * 252            # annualized covariance
ones = np.ones(len(mu))
bounds = [(0,1) for _ in range(len(mu))]

# ----------------------------
# Portfolio Optimizer
# ----------------------------
def portfolio_stats(w):
    w = np.array(w)
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov @ w))
    return r, v, r/(v+1e-9)

def max_sharpe():
    def neg_sharpe(w): return -portfolio_stats(w)[2]
    cons = ({"type":"eq","fun": lambda w: np.sum(w)-1},)
    x0 = ones/len(mu)
    res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
    return res.x

def min_var_for_target_return(target_r):
    def var_obj(w): return w @ cov @ w
    cons = (
        {"type":"eq","fun": lambda w: np.sum(w)-1},
        {"type":"eq","fun": lambda w: w @ mu - target_r}
    )
    x0 = ones/len(mu)
    res = minimize(var_obj, x0, bounds=bounds, constraints=cons)
    return res.x

# Efficient frontier
grid = np.linspace(mu.min()+1e-4, mu.max()-1e-4, 40)
ef_weights, ef_r, ef_v = [], [], []
for tr in grid:
    try:
        w = min_var_for_target_return(tr)
        r, v, _ = portfolio_stats(w)
        ef_weights.append(w); ef_r.append(r); ef_v.append(v)
    except Exception:
        pass
ef = pd.DataFrame({"Return":ef_r,"Vol":ef_v})
ef["Sharpe"] = ef["Return"] / ef["Vol"]

# Pick portfolio closest to target volatility
idx = (ef["Vol"] - target_vol).abs().idxmin()
w_star = ef_weights[idx]
r_star, v_star, s_star = portfolio_stats(w_star)

# Max Sharpe portfolio
w_ms = max_sharpe()
r_ms, v_ms, s_ms = portfolio_stats(w_ms)

# ----------------------------
# Charts – Efficient Frontier
# ----------------------------
fig_ef = go.Figure()
fig_ef.add_trace(go.Scatter(x=ef["Vol"], y=ef["Return"], mode="lines", name="Efficient Frontier"))
fig_ef.add_trace(go.Scatter(x=[v_star], y=[r_star], mode="markers", marker=dict(size=10),
                            name=f"Recommended (vol≈{v_star:.1%})"))
fig_ef.add_trace(go.Scatter(x=[v_ms], y=[r_ms], mode="markers", marker=dict(symbol="star", size=12),
                            name=f"Max-Sharpe (vol≈{v_ms:.1%})"))
fig_ef.update_layout(title="Efficient Frontier (annualized)",
                     xaxis_title="Volatility", yaxis_title="Expected Return",
                     height=420)
st.plotly_chart(fig_ef, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk score", f"{score}/100")
c2.metric("Target vol.", f"{target_vol:.1%}")
c3.metric("Rec. exp. return", f"{r_star:.1%}")
c4.metric("Rec. vol", f"{v_star:.1%}")

# ----------------------------
# Backtest (daily NAV, simple rebalance)
# ----------------------------
st.subheader("Backtest (daily NAV)")

rets_bt = rets.copy()
w = pd.Series(w_star, index=rets_bt.columns)

# Simple cumulative NAV without rebalancing drift
port_rets = rets_bt @ w
nav = (1 + port_rets).cumprod()

fig_nav = go.Figure()
fig_nav.add_trace(go.Scatter(x=nav.index, y=nav.values, mode="lines", name="Portfolio NAV"))
fig_nav.update_layout(title="Backtest NAV (start=1.0)", height=380,
                      xaxis_title="", yaxis_title="")
st.plotly_chart(fig_nav, use_container_width=True)

ann_ret = (nav.iloc[-1]/nav.iloc[0])**(252/len(nav)) - 1
ann_vol = port_rets.std()*np.sqrt(252)
max_dd = (nav / nav.cummax() - 1).min()

d1, d2, d3 = st.columns(3)
d1.metric("Ann. return", f"{ann_ret:.2%}")
d2.metric("Ann. vol", f"{ann_vol:.2%}")
d3.metric("Max drawdown", f"{max_dd:.1%}")

# ----------------------------
# Value at Risk (VaR)
# ----------------------------
st.subheader("Value at Risk (95%) – daily")
hist_var = -np.percentile(port_rets, 5)  # historical 5% quantile
mu_d = port_rets.mean()
sd_d = port_rets.std()
vc_var = -(mu_d + sd_d * norm.ppf(0.05))  # variance-covariance (Gaussian)

v1, v2 = st.columns(2)
v1.metric("Historical VaR (95%)", f"{hist_var:.2%}")
v2.metric("Var-Covar VaR (95%)", f"{vc_var:.2%}")
st.caption("Interpretation: At 95% confidence, the **daily loss** should not exceed VaR (≈5% worst days).")
