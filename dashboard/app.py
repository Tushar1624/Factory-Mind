import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time
import plotly.graph_objects as go


# -----------------------------
# Page Setup
# -----------------------------

st.set_page_config(
    page_title="Smart Factory AI",
    layout="wide"
)


# -----------------------------
# Load Model
# -----------------------------

model = joblib.load(
    "models/model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# -----------------------------
# Session State Init
# -----------------------------

if "running" not in st.session_state:
    st.session_state.running = False

if "sensor_history" not in st.session_state:
    st.session_state.sensor_history = []

if "records" not in st.session_state:
    st.session_state.records = []

if "last_input" not in st.session_state:
    st.session_state.last_input = None

if "machine_health" not in st.session_state:
    st.session_state.machine_health = {
        "Machine A": 95,
        "Machine B": 88,
        "Machine C": 70
    }


# -----------------------------
# Sidebar Reset Buttons
# -----------------------------

if st.sidebar.button("Reset Dashboard"):
    st.session_state.sensor_history = []
    st.session_state.records = []
    st.session_state.last_input = None
    st.rerun()

if st.sidebar.button("🔄 Reset Simulation"):
    st.session_state.sensor_history = []
    st.session_state.records = []
    st.session_state.running = False
    st.session_state.machine_health = {
        "Machine A": 95,
        "Machine B": 88,
        "Machine C": 70
    }
    st.rerun()


# -----------------------------
# Title
# -----------------------------

st.title("🏭 Smart Factory AI Monitoring System")
st.write("AI Based Preemptive Maintenance Dashboard")


# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.header("Machine Parameters")

temperature = st.sidebar.slider("Air Temperature (K)", 250, 350, 300)
process_temp = st.sidebar.slider("Process Temperature (K)", 250, 350, 310)
speed = st.sidebar.slider("Rotational Speed (rpm)", 1000, 2500, 1500)
torque = st.sidebar.slider("Torque (Nm)", 0, 100, 40)
wear = st.sidebar.slider("Tool Wear (min)", 0, 300, 50)


# -----------------------------
# Manual AI Prediction
# -----------------------------

input_data = np.array([[temperature, process_temp, speed, torque, wear]])
scaled = scaler.transform(input_data)
prediction = model.predict(scaled)
probability = model.predict_proba(scaled)
failure_risk = probability[0][1] * 100
health = 100 - failure_risk


# -----------------------------
# Remaining Life Estimate
# -----------------------------

wear_factor = wear / 300
temp_factor = (temperature - 250) / 100
risk_factor = (wear_factor * 0.6 + temp_factor * 0.4)
remaining_days = int(60 - (risk_factor * 60))

if remaining_days < 0:
    remaining_days = 0


# -----------------------------
# Save Manual Prediction (only on slider change)
# -----------------------------

current_input = (temperature, process_temp, speed, torque, wear)

if st.session_state.last_input != current_input:
    st.session_state.records.append({
        "Type": "Manual Input",
        "Temperature": temperature,
        "Process Temperature": process_temp,
        "Speed": speed,
        "Torque": torque,
        "Wear": wear,
        "Failure Risk": round(failure_risk, 2),
        "Health": round(health, 2),
        "Remaining Days": remaining_days
    })
    st.session_state.last_input = current_input


# -----------------------------
# Dashboard Cards
# -----------------------------

st.subheader("Machine Status")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Failure Risk", f"{failure_risk:.2f}%")

with c2:
    st.metric("Health", f"{health:.2f}%")

with c3:
    st.metric("Remaining Life", f"{remaining_days} Days")

with c4:
    if prediction[0] == 1:
        st.metric("Status", "⚠ Warning")
    else:
        st.metric("Status", "✅ Healthy")


# -----------------------------
# Gauge
# -----------------------------

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=health,
        title={"text": "Machine Health"},
        gauge={"axis": {"range": [0, 100]}}
    )
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Alert
# -----------------------------

if prediction[0] == 1:
    st.error("⚠ Machine Failure Risk Detected")
else:
    st.success("Machine Operating Normally")


# -----------------------------
# Simulation Controls
# -----------------------------

st.divider()
st.subheader("📡 Live Factory Monitoring")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Simulation"):
        st.session_state.running = True

with col2:
    if st.button("⏹ Stop Simulation"):
        st.session_state.running = False


# -----------------------------
# Generate ONE reading per rerun (while running)
# This is the correct Streamlit pattern: one tick per
# script execution, driven by st.rerun() at the bottom.
# The Stop button registers on the NEXT script run,
# which sets running=False before this block executes.
# -----------------------------

if st.session_state.running:

    temp        = np.random.randint(290, 345)
    vibration   = round(np.random.uniform(1, 10), 2)
    rpm         = np.random.randint(1200, 2300)
    torque_live = np.random.randint(30, 90)

    sensor = np.array([[
        temp,
        temp + 10,
        rpm,
        torque_live,
        np.random.randint(0, 300)
    ]])

    sensor_scaled = scaler.transform(sensor)
    risk = model.predict_proba(sensor_scaled)[0][1] * 100

    # Append to graph history
    st.session_state.sensor_history.append([
        temp, vibration, rpm, torque_live, risk
    ])

    # Append to prediction history table
    st.session_state.records.append({
        "Type":         "Live Sensor",
        "Temperature":  temp,
        "Vibration":    vibration,
        "Speed":        rpm,
        "Torque":       torque_live,
        "Failure Risk": round(risk, 2),
        "Health":       round(100 - risk, 2)
    })

    # Update machine health
    if risk > 70:
        st.session_state.machine_health["Machine C"] -= 2
    elif risk > 40:
        st.session_state.machine_health["Machine B"] -= 1

    for m in st.session_state.machine_health:
        if st.session_state.machine_health[m] < 0:
            st.session_state.machine_health[m] = 0


# -----------------------------
# Live Charts — always rendered from session state,
# so they update every rerun whether running or not.
# -----------------------------

if len(st.session_state.sensor_history) > 0:

    graph = pd.DataFrame(
        st.session_state.sensor_history,
        columns=["Temperature", "Vibration", "Speed", "Torque", "Failure Risk"]
    )

    st.subheader("📈 Live Sensor Data")
    st.line_chart(graph[["Temperature", "Vibration", "Speed", "Torque"]])

    st.subheader("⚠ Failure Risk Trend")
    st.line_chart(graph["Failure Risk"])


# -----------------------------
# Factory Machines — always rendered from session state
# -----------------------------

st.divider()
st.subheader("🏭 Factory Machines")

cols = st.columns(3)

for col, machine in zip(cols, st.session_state.machine_health.keys()):
    value = st.session_state.machine_health[machine]
    with col:
        if value > 70:
            st.success(f"{machine}\n\nStatus: Running\n\nHealth: {value}%")
        elif value > 40:
            st.warning(f"{machine}\n\nStatus: Warning\n\nHealth: {value}%")
        else:
            st.error(f"{machine}\n\nStatus: Maintenance\n\nHealth: {value}%")


# -----------------------------
# Prediction History — always rendered from session state
# -----------------------------

st.divider()
st.subheader("📋 Prediction History")

history_df = pd.DataFrame(st.session_state.records)

st.dataframe(history_df, use_container_width=True)

csv = history_df.to_csv(index=False)

st.download_button(
    "⬇ Download Report",
    data=csv,
    file_name="machine_report.csv",
    mime="text/csv",
    key="download_report"
)


# -----------------------------
# Auto-rerun every 2 seconds while simulation is active.
# This is what drives the "live" effect — each rerun
# generates one new reading and re-renders the whole page.
# The Stop button click is captured at the TOP of the
# next rerun, setting running=False before the data
# generation block, so it stops cleanly.
# -----------------------------

if st.session_state.running:
    time.sleep(2)
    st.rerun()