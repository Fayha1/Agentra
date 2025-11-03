# app_streamlit.py
# Agentra - Predictive Property Manager (Streamlit)
# -------------------------------------------------
# - Dark UI + centered logo header
# - Data sources: Real-Time (sim), Historical (csv), Predictive Model (sim)
# - Anomaly engine: rolling Z-score + gradient; optional IsolationForest (if installed)
# - Tabs: Overview, Pump Monitoring, Lighting, Agent Insights
# - Graceful fallbacks if datasets are missing

from __future__ import annotations
import os
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# Optional: Plotly for nicer gauges/plots (falls back to st.line_chart if missing)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except Exception:
    PLOTLY = False

# Optional IsolationForest
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------
# Paths & constants
# -------------------------
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "FullLogo_Transparent.png"

CSV_ENERGY = ROOT / "sim_property_riyadh_multi.csv"
CSV_ENERGY_SAVED = ROOT / "sim_property_riyadh_multi_saving15.csv"
CSV_PUMPS = ROOT / "sim_pump_riyadh.csv"
CSV_AGENT_LOG = ROOT / "agent_audit_log.csv"

st.set_page_config(
    page_title="Agentra – Predictive Property Manager",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global style tweaks for dark UI
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; }
      .stMetric { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 8px 10px; }
      .kpi-card { background: rgba(255,255,255,0.04); border-radius: 14px; padding: 14px; }
      .section-card { background: rgba(255,255,255,0.03); border-radius: 14px; padding: 16px; margin-bottom: 14px; }
      .muted { color: #A9B1BD; font-size: 0.85rem; }
      .tag { background: #0f2a18; color:#61ff8b; border:1px solid #1d4d2f; border-radius: 999px; padding: 2px 10px; font-size: 12px; }
      .btn-row { display:flex; gap:10px; align-items:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utilities
# -------------------------
def render_center_header():
    """Centered logo + title at the very top."""
    st.markdown(
        """
        <style>
          .center-header { text-align:center; margin: 4px 0 10px; }
          .center-header h2 { margin: 6px 0 2px; color:#E6FFE6; font-weight:700; letter-spacing:0.5px; }
          .center-header .sub { color:#B9C0CC; font-size:13px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_column_width=False, width=180)
        st.markdown(
            """
            <div class="center-header">
              <h2>Agentra</h2>
              <div class="sub">Predictive Property Manager</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def date_range_picker():
    today = dt.date.today()
    col1, col2 = st.sidebar.columns(2)
    start = col1.date_input("From", today - dt.timedelta(days=6))
    end = col2.date_input("To", today)
    if start > end:
        st.sidebar.warning("Start date is after end date. Swapping.")
        start, end = end, start
    return start, end

def simulate_energy(start_dt: dt.datetime, end_dt: dt.datetime, freq="H") -> pd.DataFrame:
    idx = pd.date_range(start_dt, end_dt, freq=freq)
    base = 40 + 5*np.sin(np.linspace(0, 2*np.pi, len(idx)))
    usage = base + np.random.normal(0, 2.2, len(idx))
    optimized = usage * np.random.uniform(0.74, 0.9)  # ~10-26% savings
    temp = 24 + 6*np.sin(np.linspace(0, 2*np.pi, len(idx)) - 1.0) + np.random.normal(0, 0.8, len(idx))
    vib = 1.6 + 0.7*np.sin(np.linspace(0, 4*np.pi, len(idx))) + np.random.normal(0, 0.12, len(idx))
    df = pd.DataFrame({
        "timestamp": idx,
        "energy_kwh": np.clip(usage*10, 15, None),
        "optimized_kwh": np.clip(optimized*10, 10, None),
        "temperature_c": np.clip(temp, 16, 45),
        "vibration_mm_s": np.clip(vib, 0.5, 6.0),
        "current_a": 3 + (usage-usage.min())/(usage.max()-usage.min()+1e-9)*2.5,
        "power_kw": (usage/10)*2.6
    })
    return df

def simulate_pumps(start_dt: dt.datetime, end_dt: dt.datetime, freq="H") -> pd.DataFrame:
    idx = pd.date_range(start_dt, end_dt, freq=freq)
    n = len(idx)
    eff = 75 + 8*np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 2, n)
    vib = 1.2 + 1.2*np.sin(np.linspace(0, 3*np.pi, n)) + np.random.normal(0, 0.15, n)
    temp = 40 + 6*np.sin(np.linspace(0, 2*np.pi, n)-0.6) + np.random.normal(0, 0.9, n)
    cur = 1.8 + (eff-60)/60
    pwr = 1.2 + (cur-1.2)*0.8
    return pd.DataFrame({
        "timestamp": idx,
        "pump_id": np.random.choice(["PUMP-001","PUMP-002","PUMP-003","PUMP-004"], size=n),
        "efficiency_pct": np.clip(eff, 40, 98),
        "vibration_mm_s": np.clip(vib, 0.4, 8.0),
        "temperature_c": np.clip(temp, 25, 90),
        "current_a": np.clip(cur, 0.8, 5.0),
        "power_kw": np.clip(pwr, 0.5, 10.0),
        "flow_l_min": np.clip(220 + (eff-60)*2.5, 0, None),
        "pressure_bar": np.clip(2.0 + (eff-70)/50, 0.8, 4.0),
    })

def simulate_agent_logs(start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    idx = pd.date_range(start_dt, end_dt, freq="3H")
    agents = ["HVAC Agent", "Lighting Agent", "Pump Agent"]
    actions = ["Temperature Adjustment", "Brightness Optimization", "Speed Optimization",
               "Schedule Override", "Energy Optimization", "Anomaly Detection"]
    sev = ["Low", "Medium", "High", "Critical"]
    rng = np.random.default_rng(7)
    rows = []
    for ts in idx:
        for _ in range(rng.integers(1, 4)):
            rows.append({
                "timestamp": ts + dt.timedelta(minutes=int(rng.integers(0, 180))),
                "agent": rng.choice(agents),
                "action": rng.choice(actions),
                "target": rng.choice(["Zone A-2","Floor 3","Pump Unit 7","Building Wide"]),
                "severity": rng.choice(sev, p=[0.35, 0.4, 0.2, 0.05]),
                "decision": rng.choice(["Applied","Optimized","Resolved","Monitoring","Executed","Pending"]),
                "status": rng.choice(["Completed","Pending","Completed","Completed","Completed","Completed"])
            })
    return pd.DataFrame(rows).sort_values("timestamp")

def load_or_simulate(csv_path: Path, generator, start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Basic normalization
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
            return df
        except Exception:
            pass
    return generator(start_dt, end_dt)

def rolling_zscore_anomalies(series: pd.Series, window: int, z_thresh: float, slope_thresh: float) -> pd.Series:
    """Boolean mask of anomalies by rolling Z + gradient threshold."""
    s = series.astype(float).copy()
    roll_mean = s.rolling(window, min_periods=max(2, window//2)).mean()
    roll_std = s.rolling(window, min_periods=max(2, window//2)).std(ddof=0)
    z = (s - roll_mean) / (roll_std.replace(0, np.nan))
    grad = s.diff().fillna(0.0)
    m = (z.abs() > z_thresh) | (grad.abs() > slope_thresh)
    return m.fillna(False)

def iforest_anomalies(df: pd.DataFrame, cols: list[str], contamination: float = 0.04) -> pd.Series:
    """Optional multivariate anomaly detection using IsolationForest."""
    if not SKLEARN_AVAILABLE:
        return pd.Series([False]*len(df), index=df.index)
    sub = df[cols].astype(float).fillna(method="ffill").fillna(method="bfill")
    try:
        mdl = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
        pred = mdl.fit_predict(sub.values)  # -1 = anomaly
        return pd.Series(pred == -1, index=df.index)
    except Exception:
        return pd.Series([False]*len(df), index=df.index)

def gauge_plot(value: float, title: str, suffix: str = "", min_v=0, max_v=100):
    if not PLOTLY:
        st.metric(title, f"{value:.0f}{suffix}")
        return
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={"axis":{"range":[min_v, max_v]},
                   "bar":{"color":"#22ff88"},
                   "bgcolor":"rgba(255,255,255,0.03)"}))
    fig.update_layout(height=240, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Sidebar controls
# -------------------------
render_center_header()  # centered header at page top

st.sidebar.image(str(LOGO_PATH) if LOGO_PATH.exists() else None, width=90)
st.sidebar.title("Agentra")
st.sidebar.caption("Predictive Property Manager")

data_source = st.sidebar.selectbox(
    "Select data source",
    ["Real-Time Sensors", "Historical Data", "Predictive Model"],
    index=0,
)

z_score = st.sidebar.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.1)
slope = st.sidebar.slider("Slope Sensitivity", 0.1, 2.0, 0.8, 0.05)
roll_win = st.sidebar.slider("Rolling Window (points)", 8, 72, 24, 1)

use_iforest = st.sidebar.checkbox("Advanced: IsolationForest", value=False)
if use_iforest and not SKLEARN_AVAILABLE:
    st.sidebar.info("scikit-learn غير متاح في هذه البيئة. عطّلي الخيار أو أضيفيه إلى المتطلبات.")

start_d, end_d = date_range_picker()

# Map date to dt for filtering/simulation
start_dt = dt.datetime.combine(start_d, dt.time.min)
end_dt = dt.datetime.combine(end_d, dt.time.max)

# -------------------------
# Load data based on source
# -------------------------
if data_source == "Historical Data":
    energy_df = load_or_simulate(CSV_ENERGY, simulate_energy, start_dt, end_dt)
    pumps_df = load_or_simulate(CSV_PUMPS, simulate_pumps, start_dt, end_dt)
    logs_df = load_or_simulate(CSV_AGENT_LOG, simulate_agent_logs, start_dt, end_dt)
elif data_source == "Predictive Model":
    # Predictive = more savings + slightly different ranges
    energy_df = simulate_energy(start_dt, end_dt)
    energy_df["optimized_kwh"] *= np.random.uniform(0.68, 0.85)
    pumps_df = simulate_pumps(start_dt, end_dt)
    pumps_df["efficiency_pct"] += np.random.uniform(2, 4)
    logs_df = simulate_agent_logs(start_dt, end_dt)
else:  # Real-Time Sensors
    energy_df = simulate_energy(start_dt, end_dt)
    pumps_df = simulate_pumps(start_dt, end_dt)
    logs_df = simulate_agent_logs(start_dt, end_dt)

# Ensure timestamp sort/index
for df in (energy_df, pumps_df, logs_df):
    if "timestamp" in df.columns:
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

# -------------------------
# Compute anomalies from thresholds
# -------------------------
energy_df["anomaly_energy"] = rolling_zscore_anomalies(
    energy_df["energy_kwh"], roll_win, z_score, slope
)
energy_df["anomaly_vibration"] = rolling_zscore_anomalies(
    energy_df["vibration_mm_s"], roll_win, z_score, slope
)
pump_cols = ["efficiency_pct","vibration_mm_s","temperature_c","current_a","power_kw"]
pumps_df["anomaly_eff"] = rolling_zscore_anomalies(
    pumps_df["efficiency_pct"], roll_win, z_score, slope
)
if use_iforest and SKLEARN_AVAILABLE:
    pumps_df["anomaly_iforest"] = iforest_anomalies(pumps_df, pump_cols, contamination=0.05)
else:
    pumps_df["anomaly_iforest"] = False

# Totals for overview
total_baseline = float(energy_df["energy_kwh"].sum()) if "energy_kwh" in energy_df else 0.0
total_optimized = float(energy_df["optimized_kwh"].sum()) if "optimized_kwh" in energy_df else 0.0
total_saved = max(total_baseline - total_optimized, 0.0)
anomaly_count = int(energy_df["anomaly_energy"].sum() + energy_df["anomaly_vibration"].sum()
                    + pumps_df["anomaly_eff"].sum() + pumps_df["anomaly_iforest"].sum())

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Pump Monitoring", "Lighting", "Agent Insights"])

# ============ Overview ============
with tabs[0]:
    st.markdown("### Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline Energy", f"{total_baseline:,.0f} kWh", help="Total over selected range")
    k2.metric("Optimized Energy", f"{total_optimized:,.0f} kWh")
    k3.metric("Energy Saved", f"{total_saved:,.0f} kWh")
    k4.metric("Anomalies", f"{anomaly_count}")

    c1, c2 = st.columns([1.1, 1.2])
    with c1:
        eff_now = np.clip(100.0 * (total_optimized / total_baseline) if total_baseline else 0, 0, 100)
        st.markdown("#### Energy Efficiency")
        gauge_plot(eff_now, "Efficiency", "%", 0, 100)

    with c2:
        st.markdown("#### Real-Time Energy Savings")
        plot_df = energy_df[["timestamp","energy_kwh","optimized_kwh"]].rename(
            columns={"energy_kwh":"Baseline","optimized_kwh":"Optimized"}
        ).set_index("timestamp")
        if PLOTLY and not plot_df.empty:
            fig = go.Figure()
            for col in ["Baseline","Optimized"]:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], mode="lines", name=col))
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(plot_df)

    st.markdown("#### System Performance")
    perf = pd.DataFrame({
        "timestamp": energy_df["timestamp"],
        "Temperature (°C)": energy_df["temperature_c"],
        "Energy Usage (kWh)": energy_df["energy_kwh"],
        "Vibration (mm/s)": energy_df["vibration_mm_s"],
    }).set_index("timestamp")
    if PLOTLY and not perf.empty:
        fig2 = go.Figure()
        for col in perf.columns:
            fig2.add_trace(go.Scatter(x=perf.index, y=perf[col], mode="lines", name=col))
        fig2.update_layout(height=350, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.line_chart(perf)

# ============ Pump Monitoring ============
with tabs[1]:
    st.markdown("### Pump Monitoring")
    top = st.columns(4)
    with top[0]:
        active_total = f"{(pumps_df['efficiency_pct']>0).sum()}/{pumps_df['pump_id'].nunique() or 4}"
        st.metric("Pump Status", active_total, help="Active/Total (unique pump ids)")
    with top[1]:
        st.metric("Avg Efficiency", f"{pumps_df['efficiency_pct'].mean():.0f}%")
    with top[2]:
        st.metric("Temperature", f"{pumps_df['temperature_c'].mean():.0f}°C")
    with top[3]:
        st.metric("Vibration", f"{pumps_df['vibration_mm_s'].mean():.1f} mm/s")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("#### Pump Efficiency")
        gauge_plot(float(pumps_df["efficiency_pct"].tail(1).mean()), "Efficiency", "%", 0, 100)
    with g2:
        st.markdown("#### Vibration Level")
        gauge_plot(float(pumps_df["vibration_mm_s"].tail(1).mean()), "mm/s", " mm/s", 0, 8)
    with g3:
        st.markdown("#### Temperature")
        gauge_plot(float(pumps_df["temperature_c"].tail(1).mean()), "°C", "°C", 0, 100)

    st.markdown("#### Vibration Trends & Current vs Power")
    l1, l2 = st.columns(2)
    with l1:
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pumps_df["timestamp"], y=pumps_df["vibration_mm_s"], mode="lines", name="Vibration"))
            thr = (pumps_df["vibration_mm_s"].mean() + pumps_df["vibration_mm_s"].std())
            fig.add_hline(y=thr, line_dash="dash", line_color="#ffaa00")
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pumps_df.set_index("timestamp")[["vibration_mm_s"]])
    with l2:
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pumps_df["timestamp"], y=pumps_df["current_a"], mode="lines+markers", name="Current (A)"))
            fig.add_trace(go.Scatter(x=pumps_df["timestamp"], y=pumps_df["power_kw"], mode="lines+markers", name="Power (kW)", yaxis="y2"))
            fig.update_layout(
                height=320,
                margin=dict(l=10,r=10,t=30,b=10),
                yaxis=dict(title="Current (A)"),
                yaxis2=dict(title="Power (kW)", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pumps_df.set_index("timestamp")[["current_a","power_kw"]])

    st.markdown("#### Historical Activity & Anomalies")
    pump_ano = pd.DataFrame({
        "timestamp": pumps_df["timestamp"],
        "Critical": (pumps_df["anomaly_iforest"] | (pumps_df["vibration_mm_s"] > 3.5)).astype(int),
        "Warning": pumps_df["anomaly_eff"].astype(int),
        "Normal": (~(pumps_df["anomaly_iforest"] | pumps_df["anomaly_eff"])).astype(int),
    }).set_index("timestamp")
    if PLOTLY:
        fig = go.Figure()
        for col, color in [("Critical","#ff4d4f"),("Warning","#ffb020"),("Normal","#22dd77")]:
            fig.add_trace(go.Scatter(x=pump_ano.index, y=pump_ano[col], mode="markers", name=col, marker=dict(color=color, size=6)))
        fig.update_layout(height=260, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pump_ano)

    st.markdown("#### Pump System Details")
    details = pumps_df.copy()
    details["status"] = np.where(details["anomaly_iforest"] | details["anomaly_eff"], "⚠️ Check", "Active")
    st.dataframe(
        details[["timestamp","pump_id","status","flow_l_min","pressure_bar","power_kw"]]
        .rename(columns={"timestamp":"Time","pump_id":"Pump","flow_l_min":"Flow (L/min)","pressure_bar":"Pressure (bar)","power_kw":"Power (kW)"}),
        use_container_width=True, height=260
    )

# ============ Lighting ============
with tabs[2]:
    st.markdown("### Lighting")
    ltop = st.columns(4)
    active_zones = int(np.random.randint(18, 36))
    current_kwh = float(energy_df["optimized_kwh"].tail(24).sum() if "optimized_kwh" in energy_df else 0)
    avg_lux = int(np.clip(480 + np.random.randn()*30, 200, 900))
    pct_saved = (1 - (total_optimized / total_baseline)) * 100 if total_baseline else 0.0

    ltop[0].metric("Active Zones", f"{active_zones}/36", "67% utilization")
    ltop[1].metric("Current Usage", f"{current_kwh:.0f} kWh", "kWh today")
    ltop[2].metric("Avg Lux Level", f"{avg_lux} lux")
    ltop[3].metric("Energy Saved", f"{pct_saved:.1f}%")

    left, right = st.columns([1.1, 1.2])
    with left:
        st.markdown("#### Energy Usage vs Baseline")
        bars = pd.DataFrame({
            "Day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            "Baseline": np.random.randint(400, 520, 7),
            "Actual": np.random.randint(260, 430, 7),
        }).set_index("Day")
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bars.index, y=bars["Baseline"], name="Baseline"))
            fig.add_trace(go.Bar(x=bars.index, y=bars["Actual"], name="Actual"))
            fig.update_layout(barmode="group", height=300, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(bars)

    with right:
        st.markdown("#### Lighting Efficiency")
        gauge_plot(76, "Efficient", "%", 0, 100)

    st.markdown("#### Lux Levels vs. Occupancy")
    lux = 150 + np.array([80, 150, 220, 260, 230, 180, 120, 60])
    occ = np.array([5, 15, 35, 47, 40, 28, 12, 4])  # %
    xh = ["6AM","8AM","10AM","12PM","3PM","6PM","8PM","9PM"]
    if PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=xh, y=lux, name="Lux Level", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Scatter(x=xh, y=occ, name="Occupancy %", mode="lines+markers"), secondary_y=True)
        fig.update_yaxes(title_text="Lux", secondary_y=False)
        fig.update_yaxes(title_text="Occupancy (%)", secondary_y=True)
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame({"Lux":lux, "Occupancy %":occ}, index=xh))

    st.markdown("#### Zone Status Overview")
    zones = pd.DataFrame({
        "Zone": [f"Zone {c}" for c in ["A1","A2","B1","B2","C1","C2"]],
        "State": np.random.choice(["Active","Dimmed","Fault"], size=6, p=[0.6,0.3,0.1]),
        "Lux": np.random.randint(80, 600, 6)
    })
    st.dataframe(zones, use_container_width=True, height=210)

    st.markdown("#### Daily Lighting Trends")
    daily = pd.DataFrame({
        "time": xh,
        "Zone A": np.array([5,10,25,40,50,45,28,12]),
        "Zone B": np.array([3,8,20,34,42,37,22,10]),
        "Zone C": np.array([2,6,18,28,36,30,18,8]),
    }).set_index("time")
    if PLOTLY:
        fig = go.Figure()
        for c in daily.columns:
            fig.add_trace(go.Scatter(x=daily.index, y=daily[c], mode="lines+markers", name=c, stackgroup="one"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.area_chart(daily)

# ============ Agent Insights ============
with tabs[3]:
    st.markdown("### Agent Insights")
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Total Actions", f"{len(logs_df):,}", "+ in range")
    active_anom = anomaly_count if anomaly_count else np.random.randint(1,8)
    fc2.metric("Active Anomalies", f"{active_anom}")
    fc3.metric("Success Rate", "94.2%", "Resolution rate")
    fc4.metric("Avg Response Time", "2.4s", "Detection to action")

    st.markdown("#### Anomaly Severity Distribution")
    sev_counts = logs_df["severity"].value_counts() if "severity" in logs_df else pd.Series(dtype=int)
    sev_df = sev_counts.reindex(["Critical","High","Medium","Low"]).fillna(0).astype(int)
    if PLOTLY and not sev_df.empty:
        fig = go.Figure(go.Bar(x=sev_df.index, y=sev_df.values))
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(sev_df)

    st.markdown("#### Agent Interventions Timeline")
    # Build timeline by hour & agent
    if "timestamp" in logs_df and "agent" in logs_df:
        tl = logs_df.copy()
        tl["hour"] = tl["timestamp"].dt.floor("H")
        time_counts = tl.groupby(["hour","agent"]).size().unstack(fill_value=0)
        if PLOTLY and not time_counts.empty:
            fig = go.Figure()
            for col in time_counts.columns:
                fig.add_trace(go.Scatter(x=time_counts.index, y=time_counts[col], mode="lines+markers", name=col))
            fig.update_layout(height=310, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(time_counts)

    st.markdown("#### Agent Event Logs")
    show_cols = ["timestamp","agent","action","target","severity","decision","status"]
    show_cols = [c for c in show_cols if c in logs_df.columns]
    st.dataframe(logs_df[show_cols].sort_values("timestamp", ascending=False), use_container_width=True, height=320)
    colx1, colx2 = st.columns([0.2, 0.8])
    with colx1:
        if st.button("Export CSV"):
            st.download_button(
                "Download",
                logs_df.to_csv(index=False).encode("utf-8"),
                file_name="agent_events.csv",
                mime="text/csv",
            )

# -------------------------
# Footer / engine hint
# -------------------------
st.markdown(
    "<div class='muted' style='text-align:center;margin-top:10px;'>"
    "Anomaly engine: rolling Z-score + gradient; optional IsolationForest (multivariate)."
    "</div>",
    unsafe_allow_html=True,
)
