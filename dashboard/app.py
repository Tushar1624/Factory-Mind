import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FactoryMind AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ───────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #080E1A;
    color: #B8C4D4;
}

/* Main container */
.main .block-container {
    padding: 0 2.5rem 4rem 2.5rem;
    max-width: 1440px;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0C1525 !important;
    border-right: 1px solid #172236 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.8rem 1.2rem 2rem 1.2rem;
}

/* Sidebar brand */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.8rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #172236;
}
.sb-brand-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #0EA5E9, #7C3AED);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.sb-brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.1;
}
.sb-brand-sub {
    font-size: 0.65rem;
    color: #4A6080;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Sidebar section label */
.sb-section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.62rem;
    color: #3A5070;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.2rem 0 0.5rem 0;
}

/* Slider labels */
[data-testid="stSidebar"] .stSlider > label {
    color: #7A90A8 !important;
    font-size: 0.78rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] {
    padding: 0 2px;
}

/* Sidebar reset buttons */
.sb-btn-wrap .stButton > button {
    width: 100%;
    background: transparent !important;
    border: 1px solid #172236 !important;
    color: #4A6080 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.45rem 0.8rem !important;
    border-radius: 7px !important;
    transition: all 0.18s ease !important;
    margin-bottom: 0.4rem !important;
    text-align: left !important;
}
.sb-btn-wrap .stButton > button:hover {
    border-color: #0EA5E9 !important;
    color: #0EA5E9 !important;
    background: rgba(14, 165, 233, 0.05) !important;
}

/* ── Page header ──────────────────────────── */
.page-header {
    padding: 1.8rem 0 1.4rem 0;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #172236;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
}
.ph-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #EEF2F7;
    margin: 0;
    letter-spacing: -0.025em;
}
.ph-sub {
    font-size: 0.75rem;
    color: #3A5070;
    margin: 0.15rem 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.28rem 0.8rem;
    border-radius: 100px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    white-space: nowrap;
}
.status-chip.live {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10B981;
}
.status-chip.idle {
    background: rgba(71, 85, 105, 0.12);
    border: 1px solid #1E2D45;
    color: #4A6080;
}
.chip-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: currentColor;
}
.chip-dot.live { animation: blink 1.6s ease-in-out infinite; }
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.25; }
}

/* ── KPI grid ─────────────────────────────── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin: 1.4rem 0;
}
.kpi-card {
    background: #0C1525;
    border: 1px solid #172236;
    border-radius: 14px;
    padding: 1.15rem 1.3rem 1rem 1.3rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    border-radius: 14px 14px 0 0;
    background: var(--kpi-accent, #172236);
}
.kpi-card.ok   { --kpi-accent: #10B981; }
.kpi-card.warn { --kpi-accent: #F59E0B; }
.kpi-card.crit { --kpi-accent: #EF4444; }
.kpi-card.info { --kpi-accent: #0EA5E9; }

.kpi-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.63rem;
    color: #3A5070;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 600;
    line-height: 1;
    color: #EEF2F7;
    margin-bottom: 0.3rem;
}
.kpi-val .unit {
    font-size: 1rem;
    color: #3A5070;
    font-weight: 400;
}
.kpi-val.ok   { color: #10B981; }
.kpi-val.warn { color: #F59E0B; }
.kpi-val.crit { color: #EF4444; }
.kpi-val.info { color: #0EA5E9; }

.kpi-sub {
    font-size: 0.7rem;
    color: #3A5070;
    line-height: 1.3;
}

/* ── Section divider ──────────────────────── */
.section-div {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.8rem 0 1.1rem 0;
}
.section-div-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.63rem;
    color: #3A5070;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    white-space: nowrap;
}
.section-div-line {
    flex: 1;
    height: 1px;
    background: #172236;
}

/* ── Alert banner ─────────────────────────── */
.alert-box {
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}
.alert-box.ok {
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.alert-box.crit {
    background: rgba(239, 68, 68, 0.07);
    border: 1px solid rgba(239, 68, 68, 0.25);
}
.alert-icon {
    font-size: 1rem;
    margin-top: 1px;
    flex-shrink: 0;
}
.alert-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #EEF2F7;
    margin-bottom: 0.15rem;
}
.alert-desc {
    font-size: 0.73rem;
    color: #4A6080;
    line-height: 1.4;
}

/* ── Parameter table ─────────────────────── */
.param-table {
    background: #0C1525;
    border: 1px solid #172236;
    border-radius: 10px;
    overflow: hidden;
}
.param-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #111D2E;
}
.param-row:last-child { border-bottom: none; }
.param-name {
    font-size: 0.7rem;
    color: #3A5070;
    font-family: 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.param-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #B8C4D4;
}

/* ── Simulation buttons ───────────────────── */
.sim-start-btn .stButton > button {
    background: linear-gradient(135deg, #0EA5E9, #0284C7) !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.6rem !important;
    border-radius: 9px !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 4px 16px rgba(14, 165, 233, 0.2) !important;
}
.sim-start-btn .stButton > button:hover {
    background: linear-gradient(135deg, #38BDF8, #0EA5E9) !important;
    box-shadow: 0 4px 24px rgba(14, 165, 233, 0.35) !important;
    transform: translateY(-1px) !important;
}
.sim-stop-btn .stButton > button {
    background: transparent !important;
    color: #EF4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.6rem !important;
    border-radius: 9px !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
}
.sim-stop-btn .stButton > button:hover {
    background: rgba(239, 68, 68, 0.08) !important;
    border-color: #EF4444 !important;
}

/* ── Fleet cards ──────────────────────────── */
.fleet-card {
    background: #0C1525;
    border: 1px solid #172236;
    border-radius: 13px;
    padding: 1.1rem 1.2rem 1rem;
    transition: border-color 0.3s ease;
}
.fleet-card.ok   { border-color: rgba(16, 185, 129, 0.25); }
.fleet-card.warn { border-color: rgba(245, 158, 11, 0.28); }
.fleet-card.crit { border-color: rgba(239, 68, 68, 0.28); }

.fleet-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.85rem;
}
.fleet-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: #C8D5E4;
}
.fleet-badge {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.55rem;
    border-radius: 100px;
}
.fleet-badge.ok   { background: rgba(16,185,129,0.1);  color: #10B981; border: 1px solid rgba(16,185,129,0.25); }
.fleet-badge.warn { background: rgba(245,158,11,0.1);  color: #F59E0B; border: 1px solid rgba(245,158,11,0.25); }
.fleet-badge.crit { background: rgba(239,68,68,0.1);   color: #EF4444; border: 1px solid rgba(239,68,68,0.25); }

.health-track {
    height: 5px;
    background: #111D2E;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 0.45rem;
}
.health-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.health-fill.ok   { background: linear-gradient(90deg, #059669, #10B981); }
.health-fill.warn { background: linear-gradient(90deg, #B45309, #F59E0B); }
.health-fill.crit { background: linear-gradient(90deg, #B91C1C, #EF4444); }

.fleet-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #3A5070;
    text-align: right;
}

/* ── History table ────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid #172236 !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table {
    background: #0C1525 !important;
}
[data-testid="stDataFrame"] th {
    background: #080E1A !important;
    color: #3A5070 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    border-bottom: 1px solid #172236 !important;
    padding: 0.7rem 0.9rem !important;
}
[data-testid="stDataFrame"] td {
    color: #8A9AB0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
    border-bottom: 1px solid #111D2E !important;
    padding: 0.55rem 0.9rem !important;
}

/* ── Download button ──────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #172236 !important;
    color: #4A6080 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.73rem !important;
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.18s ease !important;
}
.stDownloadButton > button:hover {
    border-color: #0EA5E9 !important;
    color: #0EA5E9 !important;
    background: rgba(14, 165, 233, 0.05) !important;
}

/* ── Plotly transparent bg ────────────────── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── Hide Streamlit chrome ────────────────── */
#MainMenu        { visibility: hidden; }
footer           { visibility: hidden; }
[data-testid="stHeader"]    { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* ── Hide default st.metric ───────────────── */
[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
model  = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
_defaults = {
    "running":        False,
    "sensor_history": [],
    "records":        [],
    "last_input":     None,
    "machine_health": {"Machine A": 95, "Machine B": 88, "Machine C": 70},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-icon">⚙</div>
      <div>
        <div class="sb-brand-name">FactoryMind</div>
        <div class="sb-brand-sub">AI Monitoring</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Machine Parameters</div>', unsafe_allow_html=True)
    temperature  = st.slider("Air Temperature (K)",     250, 350, 300)
    process_temp = st.slider("Process Temperature (K)", 250, 350, 310)
    speed        = st.slider("Rotational Speed (rpm)", 1000, 2500, 1500)
    torque       = st.slider("Torque (Nm)",               0, 100,  40)
    wear         = st.slider("Tool Wear (min)",            0, 300,  50)

    st.markdown('<div class="sb-section-label" style="margin-top:1.6rem">System</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="sb-btn-wrap">', unsafe_allow_html=True)
        if st.button("↺  Reset Dashboard"):
            st.session_state.sensor_history = []
            st.session_state.records        = []
            st.session_state.last_input     = None
            st.rerun()
        if st.button("↺  Reset Simulation"):
            st.session_state.sensor_history = []
            st.session_state.records        = []
            st.session_state.running        = False
            st.session_state.machine_health = {"Machine A": 95, "Machine B": 88, "Machine C": 70}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Manual prediction
# ─────────────────────────────────────────────────────────────
input_arr    = np.array([[temperature, process_temp, speed, torque, wear]])
scaled_arr   = scaler.transform(input_arr)
prediction   = model.predict(scaled_arr)
probability  = model.predict_proba(scaled_arr)
failure_risk = probability[0][1] * 100
health       = 100 - failure_risk

wear_factor    = wear / 300
temp_factor    = (temperature - 250) / 100
risk_factor    = wear_factor * 0.6 + temp_factor * 0.4
remaining_days = max(0, int(60 - risk_factor * 60))

current_input = (temperature, process_temp, speed, torque, wear)
if st.session_state.last_input != current_input:
    st.session_state.records.append({
        "Type":             "Manual",
        "Temperature (K)":  temperature,
        "Process Temp (K)": process_temp,
        "Speed (rpm)":      speed,
        "Torque (Nm)":      torque,
        "Wear (min)":       wear,
        "Failure Risk %":   round(failure_risk, 2),
        "Health %":         round(health, 2),
        "Remaining Days":   remaining_days,
    })
    st.session_state.last_input = current_input


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def risk_tier(r):
    if r >= 70: return "crit"
    if r >= 40: return "warn"
    return "ok"

def health_tier(h):
    if h >= 70: return "ok"
    if h >= 40: return "warn"
    return "crit"

def section(label):
    st.markdown(f"""
    <div class="section-div">
      <span class="section-div-label">{label}</span>
      <div class="section-div-line"></div>
    </div>""", unsafe_allow_html=True)

rt = risk_tier(failure_risk)
ht = health_tier(health)

# Plotly shared layout
_chart_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#3A5070", size=10),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", size=10, color="#4A6080"),
        orientation="h", y=1.13, x=0,
    ),
    xaxis=dict(showgrid=False, color="#2A3A50", zeroline=False,
               showline=True, linecolor="#172236", tickfont=dict(size=9)),
    yaxis=dict(gridcolor="#111D2E", color="#2A3A50", zeroline=False,
               showline=False, tickfont=dict(size=9)),
    margin=dict(t=36, b=20, l=0, r=0),
    height=210,
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0C1525", bordercolor="#172236",
        font=dict(family="JetBrains Mono", size=11, color="#B8C4D4"),
    ),
)


# ─────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────
is_live     = st.session_state.running
chip_class  = "live" if is_live else "idle"
chip_label  = "Live · Monitoring" if is_live else "Idle"
dot_class   = "live" if is_live else ""

st.markdown(f"""
<div class="page-header">
  <div>
    <div class="ph-title">Smart Factory AI</div>
    <div class="ph-sub">Predictive Maintenance · Real-time Intelligence</div>
  </div>
  <span class="status-chip {chip_class}">
    <span class="chip-dot {dot_class}"></span>
    {chip_label}
  </span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────
risk_sub = {
    "ok":   "Within safe range",
    "warn": "Elevated · monitor closely",
    "crit": "Critical — schedule inspection",
}[rt]

health_sub = {
    "ok":   "All systems nominal",
    "warn": "Degrading — review parameters",
    "crit": "Poor — maintenance required",
}[ht]

ai_cls    = "crit" if prediction[0] == 1 else "ok"
ai_icon   = "⚠" if prediction[0] == 1 else "✓"
ai_label  = "Warning" if prediction[0] == 1 else "Healthy"
ai_sub    = "Immediate attention required" if prediction[0] == 1 else "No anomaly detected"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card {rt}">
    <div class="kpi-label">Failure Risk</div>
    <div class="kpi-val {rt}">{failure_risk:.1f}<span class="unit">%</span></div>
    <div class="kpi-sub">{risk_sub}</div>
  </div>
  <div class="kpi-card {ht}">
    <div class="kpi-label">Machine Health</div>
    <div class="kpi-val {ht}">{health:.1f}<span class="unit">%</span></div>
    <div class="kpi-sub">{health_sub}</div>
  </div>
  <div class="kpi-card info">
    <div class="kpi-label">Remaining Life</div>
    <div class="kpi-val info">{remaining_days}<span class="unit"> d</span></div>
    <div class="kpi-sub">Estimated before maintenance</div>
  </div>
  <div class="kpi-card {ai_cls}">
    <div class="kpi-label">AI Verdict</div>
    <div class="kpi-val {ai_cls}" style="font-size:1.45rem; padding-top:0.25rem">{ai_icon} {ai_label}</div>
    <div class="kpi-sub">{ai_sub}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Health gauge + alert + parameter summary
# ─────────────────────────────────────────────────────────────
section("Manual Assessment")

g_col, a_col = st.columns([1.1, 1])

with g_col:
    g_color = {"ok": "#10B981", "warn": "#F59E0B", "crit": "#EF4444"}[ht]
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health,
        number={"suffix": "%", "font": {"color": g_color, "family": "JetBrains Mono", "size": 34}},
        title={"text": "MACHINE HEALTH", "font": {"color": "#3A5070", "family": "Space Grotesk", "size": 10}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickfont": {"color": "#2A3A50", "size": 9, "family": "JetBrains Mono"},
                "tickcolor": "#172236",
            },
            "bar":       {"color": g_color, "thickness": 0.22},
            "bgcolor":   "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(239,68,68,0.07)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.06)"},
                {"range": [70,100], "color": "rgba(16,185,129,0.06)"},
            ],
            "threshold": {
                "line":      {"color": "#EF4444", "width": 2},
                "thickness": 0.72,
                "value":     30,
            },
        },
    ))
    fig_g.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=230,
        margin=dict(t=40, b=10, l=20, r=20),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

with a_col:
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.markdown("""
        <div class="alert-box crit">
          <div class="alert-icon">⚠</div>
          <div>
            <div class="alert-title">Failure Risk Detected</div>
            <div class="alert-desc">AI model flagged abnormal operating conditions. Schedule an inspection before continuing production.</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box ok">
          <div class="alert-icon">✓</div>
          <div>
            <div class="alert-title">Operating Normally</div>
            <div class="alert-desc">All parameters are within expected bounds. No corrective action required at this time.</div>
          </div>
        </div>""", unsafe_allow_html=True)

    params = [
        ("Air Temperature", f"{temperature} K"),
        ("Process Temp",    f"{process_temp} K"),
        ("Rotational Speed", f"{speed} rpm"),
        ("Torque",          f"{torque} Nm"),
        ("Tool Wear",       f"{wear} min"),
    ]
    rows_html = "".join(
        f'<div class="param-row"><span class="param-name">{k}</span><span class="param-val">{v}</span></div>'
        for k, v in params
    )
    st.markdown(f'<div class="param-table">{rows_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Live simulation controls
# ─────────────────────────────────────────────────────────────
section("Live Factory Monitoring")

b1, b2, _ = st.columns([1.2, 1.2, 4])
with b1:
    st.markdown('<div class="sim-start-btn">', unsafe_allow_html=True)
    if st.button("▶  Start Simulation", key="btn_start"):
        st.session_state.running = True
    st.markdown('</div>', unsafe_allow_html=True)
with b2:
    st.markdown('<div class="sim-stop-btn">', unsafe_allow_html=True)
    if st.button("⏹  Stop Simulation", key="btn_stop"):
        st.session_state.running = False
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Generate ONE live reading per rerun (while running)
# ─────────────────────────────────────────────────────────────
if st.session_state.running:
    t_live   = np.random.randint(290, 345)
    v_live   = round(np.random.uniform(1, 10), 2)
    rpm_live = np.random.randint(1200, 2300)
    tq_live  = np.random.randint(30, 90)
    w_live   = np.random.randint(0, 300)

    s_arr    = np.array([[t_live, t_live + 10, rpm_live, tq_live, w_live]])
    s_scaled = scaler.transform(s_arr)
    risk_l   = model.predict_proba(s_scaled)[0][1] * 100

    st.session_state.sensor_history.append(
        [t_live, v_live, rpm_live, tq_live, risk_l]
    )
    st.session_state.records.append({
        "Type":             "Live",
        "Temperature (K)":  t_live,
        "Process Temp (K)": t_live + 10,
        "Speed (rpm)":      rpm_live,
        "Torque (Nm)":      tq_live,
        "Wear (min)":       w_live,
        "Failure Risk %":   round(risk_l, 2),
        "Health %":         round(100 - risk_l, 2),
        "Remaining Days":   "",
    })

    if risk_l > 70:
        st.session_state.machine_health["Machine C"] = max(
            0, st.session_state.machine_health["Machine C"] - 2
        )
    elif risk_l > 40:
        st.session_state.machine_health["Machine B"] = max(
            0, st.session_state.machine_health["Machine B"] - 1
        )


# ─────────────────────────────────────────────────────────────
# Live charts
# ─────────────────────────────────────────────────────────────
if len(st.session_state.sensor_history) > 0:
    df_graph = pd.DataFrame(
        st.session_state.sensor_history,
        columns=["Temperature", "Vibration", "Speed", "Torque", "Failure Risk"],
    )
    idx = list(range(len(df_graph)))

    lc1, lc2 = st.columns(2)

    with lc1:
        fig_s = go.Figure(layout=_chart_layout)
        _palette = ["#0EA5E9", "#10B981", "#F59E0B", "#A78BFA"]
        for col, clr in zip(["Temperature", "Speed", "Torque", "Vibration"], _palette):
            fig_s.add_trace(go.Scatter(
                x=idx, y=df_graph[col], name=col,
                line=dict(color=clr, width=1.6),
                mode="lines",
            ))
        fig_s.update_layout(title=dict(
            text="SENSOR TELEMETRY",
            font=dict(family="Space Grotesk", size=9, color="#3A5070"),
            x=0, y=0.98,
        ))
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    with lc2:
        fig_r = go.Figure(layout=_chart_layout)
        fig_r.add_trace(go.Scatter(
            x=idx, y=df_graph["Failure Risk"],
            name="Failure Risk %",
            line=dict(color="#EF4444", width=1.8),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.07)",
        ))
        fig_r.add_hline(y=70, line_dash="dash",
                        line_color="rgba(239,68,68,0.35)",
                        annotation_text="Critical",
                        annotation_font=dict(color="#EF4444", size=9))
        fig_r.add_hline(y=40, line_dash="dash",
                        line_color="rgba(245,158,11,0.35)",
                        annotation_text="Warning",
                        annotation_font=dict(color="#F59E0B", size=9))
        fig_r.update_layout(
            yaxis=dict(range=[0, 108], **{k: v for k, v in _chart_layout["yaxis"].items()}),
            title=dict(
                text="FAILURE RISK TREND",
                font=dict(family="Space Grotesk", size=9, color="#3A5070"),
                x=0, y=0.98,
            ),
        )
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# Fleet status
# ─────────────────────────────────────────────────────────────
section("Machine Fleet Status")

f_cols = st.columns(3)
_badge = {"ok": "Running", "warn": "Warning", "crit": "Maintenance"}

for col, (machine, value) in zip(f_cols, st.session_state.machine_health.items()):
    tier = health_tier(value)
    with col:
        st.markdown(f"""
        <div class="fleet-card {tier}">
          <div class="fleet-header">
            <div class="fleet-name">{machine}</div>
            <span class="fleet-badge {tier}">{_badge[tier]}</span>
          </div>
          <div class="health-track">
            <div class="health-fill {tier}" style="width:{value}%"></div>
          </div>
          <div class="fleet-pct">{value}% health</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Prediction history
# ─────────────────────────────────────────────────────────────
section("Prediction History")

hist_df = pd.DataFrame(st.session_state.records)

if not hist_df.empty:
    st.dataframe(hist_df, use_container_width=True, height=270)
    csv_data = hist_df.to_csv(index=False)
    dl_col, _ = st.columns([1.2, 5])
    with dl_col:
        st.download_button(
            "↓  Export CSV",
            data=csv_data,
            file_name="machine_report.csv",
            mime="text/csv",
            key="download_report",
        )
else:
    st.markdown(
        '<p style="color:#3A5070;font-size:0.8rem;font-family:Space Grotesk,sans-serif;'
        'padding:1.2rem 0">Adjust a slider or start the live simulation to generate prediction records.</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Auto-rerun — one reading per 2-second tick
# ─────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(2)
    st.rerun()