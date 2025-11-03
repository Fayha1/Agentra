# app_streamlit.py
# Agentra — Predictive Property Manager (Streamlit)
# -------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -------------------------
# Paths & Constants
# -------------------------
ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "FullLogo_Transparent.png"

# Your CSVs (rename here if your filenames differ)
CSV_OVERALL = ROOT / "sim_property_riyadh_multi.csv"                # overall (baseline/actual/temp/vibration)
CSV_OVERALL_SAVING = ROOT / "sim_property_riyadh_multi_saving15.csv" # optimized/energy saving variant (optional)
CSV_PUMP = ROOT / "sim_pump_riyadh.csv"
CSV_AGENT_LOG = ROOT / "agent_audit_log.csv"

# -------------------------
# Optional: IsolationForest
# -------------------------
SKLEARN_OK = False
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# -------------------------
# Page Config + Dark Styling
# -------------------------
st.set_page_config(
    page_title="Agentra · Predictive Property Manager",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple dark theme tweaks (keeps Streamlit Dark but adjusts a few tones)
st.markdown(
    """
    <style>
      .center-header { text-align:center; margin: 8px 0 6px; }
      .center-header h2 { margin: 6px 0 2px; color:#E6FFE6; font-weight:700; letter-spacing:0.3px; }
      .center-header .sub { color:#B9C0CC; font-size:13px; }
      .css-1v0mbdj, .stTabs [data-baseweb="tab-list"] { gap: 2px !important; }
      .kpi-card { padding:16px; border-radius:12px; background: #0e1117; border:1px solid rgba(255,255,255,0.05); }
      .kpi-title { color:#9fb4a6; font-size:12px; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:6px;}
      .kpi-value { color:#9eff9e; font-size:28px; font-weight:700; line-height:1.1; }
      .kpi-sub { color:#7d8790; font-size:12px; }
      .note { font-size:12px; color:#c2c8d0; background:#0e1117; padding:10px 12px; border-radius:8px; border:1px solid rgba(255,255,255,.06);}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utilities
# -------------------------
@st.cache_data(show_spinner=False)
def load_or_simulate_overall():
    """
    Returns overall df with columns:
    timestamp, energy_kwh, optimized_kwh, temp_c, vibration_mm_s, current_a, power_kw
    If files missing, simulate.
    """
    if CSV_OVERALL.exists():
        df = pd.read_csv(CSV_OVERALL)
    else:
        # simulate
        ts = pd.date_range(end=datetime.now(), periods=24, freq="H")
        energy = np.linspace(320, 160, len(ts)) + np.random.normal(0, 10, len(ts))
        temp = 24 + 4*np.sin(np.linspace(0, 3.14, len(ts))) + np.random.normal(0, .2, len(ts))
        vib = 1.8 + 0.6*np.sin(np.linspace(0, 4, len(ts))) + np.random.normal(0, .05, len(ts))
        current = 4.0 + 0.8*np.sin(np.linspace(0, 5, len(ts)))
        power = current * 0.55 + np.random.normal(0, .05, len(ts))
        df = pd.DataFrame({
            "timestamp": ts,
            "energy_kwh": energy,
            "temp_c": temp,
            "vibration_mm_s": vib,
            "current_a": current,
            "power_kw": power
        })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # optimized energy: prefer a dedicated file if exists; else assume 20–30% reduction
    if CSV_OVERALL_SAVING.exists():
        sav = pd.read_csv(CSV_OVERALL_SAVING)
        if "optimized_kwh" in sav.columns:
            sav["timestamp"] = pd.to_datetime(sav["timestamp"])
            df = pd.merge_asof(df.sort_values("timestamp"),
                               sav[["timestamp", "optimized_kwh"]].sort_values("timestamp"),
                               on="timestamp", direction="nearest")
        else:
            df["optimized_kwh"] = df["energy_kwh"] * 0.78
    else:
        if "optimized_kwh" not in df.columns:
            df["optimized_kwh"] = df["energy_kwh"] * 0.78
    # ensure rest cols
    for col, base in [("current_a", 4.3), ("power_kw", 2.1)]:
        if col not in df.columns:
            df[col] = base + np.random.normal(0, .1, len(df))
    return df

@st.cache_data(show_spinner=False)
def load_or_simulate_pump():
    """
    Returns pump df with:
    timestamp, pump_id, status, efficiency, temperature, vibration, flow_lpm, pressure_bar, power_kw
    """
    if CSV_PUMP.exists():
        dfp = pd.read_csv(CSV_PUMP)
        if "timestamp" in dfp.columns:
            dfp["timestamp"] = pd.to_datetime(dfp["timestamp"])
        else:
            # fabricate time if absent
            dfp["timestamp"] = pd.date_range(end=datetime.now(), periods=len(dfp), freq="H")
    else:
        ts = pd.date_range(end=datetime.now(), periods=24, freq="H")
        rows = []
        pump_ids = ["PUMP-001", "PUMP-002", "PUMP-003"]
        for pid in pump_ids:
            eff = 75 + 10*np.sin(np.linspace(0, 1.2*np.pi, len(ts))) + np.random.normal(0, 2, len(ts))
            vib = 1.0 + 1.2*np.abs(np.sin(np.linspace(0, 2*np.pi, len(ts)))) + np.random.normal(0, 0.1, len(ts))
            temp = 38 + 7*np.sin(np.linspace(0, 1.6*np.pi, len(ts))) + np.random.normal(0, .5, len(ts))
            flow = 240 + 30*np.sin(np.linspace(0, 1.8*np.pi, len(ts)))
            pres = 2.9 + 0.3*np.sin(np.linspace(0, 2.5*np.pi, len(ts)))
            pwr = 12 + 3*np.sin(np.linspace(0, 2*np.pi, len(ts)))
            for i, t in enumerate(ts):
                rows.append([t, pid,
                             "Active" if i % 11 != 0 else "Warning",
                             eff[i], temp[i], vib[i], flow[i], pres[i], pwr[i]])
        dfp = pd.DataFrame(rows, columns=[
            "timestamp","pump_id","status","efficiency","temperature","vibration",
            "flow_lpm","pressure_bar","power_kw"
        ])
    return dfp

@st.cache_data(show_spinner=False)
def load_or_simulate_agent_log():
    """
    Agent actions log with:
    timestamp, agent_type, action, target, severity, decision, status
    """
    if CSV_AGENT_LOG.exists():
        log = pd.read_csv(CSV_AGENT_LOG)
        if "timestamp" in log.columns:
            log["timestamp"] = pd.to_datetime(log["timestamp"])
        else:
            log["timestamp"] = pd.date_range(end=datetime.now(), periods=len(log), freq="H")
    else:
        # demo log
        ts = pd.date_range(end=datetime.now(), periods=40, freq="H")
        agent_types = ["HVAC Agent", "Lighting Agent", "Pump Agent"]
        actions = ["Temperature Adjustment", "Brightness Optimization", "Speed Optimization",
                   "Anomaly Detection", "Schedule Override", "Energy Optimization"]
        targets = ["Zone A-203", "Floor 3 East", "Pump Unit 7", "Building Wide", "Zone B-105"]
        severities = ["Low","Medium","High","Critical"]
        decisions = ["Optimized","Applied","Resolved","Monitoring","Executed","Reduce by 3°C",
                     "Schedule Maintenance", "Dim to 75%"]
        statuses = ["Completed","Pending","Completed","Completed","Completed"]
        rng = np.random.default_rng(42)
        rows = []
        for t in ts:
            rows.append([
                t,
                rng.choice(agent_types),
                rng.choice(actions),
                rng.choice(targets),
                rng.choice(severities, p=[0.45,0.35,0.15,0.05]),
                rng.choice(decisions),
                rng.choice(statuses, p=[0.65,0.1,0.25,0.0,0.0]),
            ])
        log = pd.DataFrame(rows, columns=["timestamp","agent_type","action","target",
                                          "severity","decision","status"])
    return log

def compute_anomalies(df: pd.DataFrame, z_thresh: float, slope_sens: float, window_pts: int,
                      use_iforest: bool, features=("energy_kwh","temp_c","vibration_mm_s")):
    """
    Returns df with columns:
    z_score, grad, anomaly_z, anomaly_grad, anomaly_iforest (opt), anomaly (final flag)
    """
    data = df.copy()
    if "timestamp" in data.columns:
        data = data.sort_values("timestamp").reset_index(drop=True)

    # Rolling Z-score on the first feature (energy_kwh) if exists
    key = features[0] if features[0] in data.columns else data.select_dtypes(float).columns[0]
    x = data[key].astype(float)
    roll_mean = x.rolling(window_pts, min_periods=max(3, window_pts//3)).mean()
    roll_std = x.rolling(window_pts, min_periods=max(3, window_pts//3)).std(ddof=0)
    z = (x - roll_mean) / (roll_std.replace(0, np.nan))
    data["z_score"] = z.fillna(0)

    # Gradient magnitude across selected features (simple finite diff norm)
    grads = []
    feat_used = [c for c in features if c in data.columns]
    if len(feat_used) == 0:
        feat_used = [key]
    for i in range(len(data)):
        if i == 0:
            grads.append(0.0)
        else:
            vec = []
            for c in feat_used:
                vec.append((data.loc[i,c] - data.loc[i-1,c]))
            grads.append(float(np.linalg.norm(vec)))
    data["grad"] = pd.Series(grads, index=data.index)

    # Thresholds
    data["anomaly_z"] = (data["z_score"].abs() > z_thresh).astype(int)
    # slope threshold: use percentile-based sensitivity (lower sens -> detect smaller changes)
    if data["grad"].std() == 0 or np.isnan(data["grad"].std()):
        grad_thr = data["grad"].mean() + 3e-6  # tiny
    else:
        # convert slope_sens (0.1..2.0) to a percentile 65..95 roughly
        p = min(99, max(50, 60 + slope_sens*18))
        grad_thr = np.percentile(data["grad"], p)
    data["anomaly_grad"] = (data["grad"] > grad_thr).astype(int)

    # Optional IsolationForest (multivariate)
    if use_iforest and SKLEARN_OK and len(feat_used) >= 1:
        X = data[feat_used].fillna(method="ffill").fillna(method="bfill")
        try:
            clf = IsolationForest(
                n_estimators=128,
                contamination=0.05,
                random_state=42,
            )
            pred = clf.fit_predict(X)  # -1 anomaly, 1 normal
            data["anomaly_iforest"] = (pred == -1).astype(int)
        except Exception:
            data["anomaly_iforest"] = 0
    else:
        data["anomaly_iforest"] = 0

    # Final anomaly if any method flags
    data["anomaly"] = ((data["anomaly_z"] + data["anomaly_grad"] + data["anomaly_iforest"]) > 0).astype(int)
    return data

def kpi_card(title, value, sub=None, color="#9eff9e"):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value" style="color:{color}">{value}</div>
          <div class="kpi-sub">{'' if sub is None else sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_center_header():
    """Centered logo + title"""
    c1, c2, c3 = st.columns([1,2,1])
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

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.image(str(LOGO_PATH), use_column_width=False, width=72) if LOGO_PATH.exists() else None
st.sidebar.title("Agentra")

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Real-Time Sensors", "Historical Data", "Predictive Model"],
    index=0
)

z_thresh = st.sidebar.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.1)
slope_sens = st.sidebar.slider("Slope Sensitivity", 0.1, 2.0, 0.8, 0.05)
roll_win = st.sidebar.slider("Rolling Window (points)", 8, 72, 24, 1)

use_iforest = st.sidebar.checkbox("Advanced: IsolationForest", value=True)
if use_iforest and not SKLEARN_OK:
    st.sidebar.info("scikit-learn غير متاح في هذه البيئة. عطّل الخيار أو أضِفه إلى المتطلبات.", icon="⚠️")

date_from = st.sidebar.date_input("From", datetime.now().date() - timedelta(days=2))
date_to = st.sidebar.date_input("To", datetime.now().date())

# -------------------------
# Center Header
# -------------------------
render_center_header()

# -------------------------
# Load Data
# -------------------------
df_overall = load_or_simulate_overall()
df_pump = load_or_simulate_pump()
df_log = load_or_simulate_agent_log()

# Apply data source notion (for demo we only subset / resample differently)
if data_source == "Historical Data":
    # show last 7 days if exist
    df_overall = df_overall.set_index("timestamp").resample("1H").mean().iloc[-24*7:].reset_index()
elif data_source == "Predictive Model":
    # emulate a predicted path with mild reduction
    df_overall = df_overall.copy()
    df_overall["optimized_kwh"] = df_overall["energy_kwh"] * 0.72

# Date filter
def date_filter(df, col="timestamp"):
    return df[(df[col] >= pd.Timestamp(date_from)) & (df[col] < pd.Timestamp(date_to) + pd.Timedelta(days=1))]

df_overall_f = date_filter(df_overall, "timestamp")
df_pump_f = date_filter(df_pump, "timestamp")
df_log_f = date_filter(df_log, "timestamp")

# Anomaly compute
df_ano = compute_anomalies(
    df_overall_f,
    z_thresh=z_thresh,
    slope_sens=slope_sens,
    window_pts=roll_win,
    use_iforest=use_iforest,
    features=("energy_kwh","temp_c","vibration_mm_s")
)

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_pump, tab_light, tab_agents = st.tabs(["Overview", "Pump Monitoring", "Lighting", "Agent Insights"])

# -------------------------
# Overview
# -------------------------
with tab_overview:
    if len(df_overall_f) == 0:
        st.warning("لا توجد بيانات ضمن النطاق الزمني المحدد. غيّر التاريخ أو تأكد من ربط الداتا.")
    else:
        baseline = float(df_overall_f["energy_kwh"].iloc[-1]) if "energy_kwh" in df_overall_f else 0.0
        optimized = float(df_overall_f["optimized_kwh"].iloc[-1]) if "optimized_kwh" in df_overall_f else baseline*0.8
        saved = max(0.0, baseline - optimized)
        anomalies_cnt = int(df_ano["anomaly"].sum())

        cK1, cK2, cK3, cK4 = st.columns(4)
        with cK1: kpi_card("Baseline Energy", f"{baseline:,.0f} kWh")
        with cK2: kpi_card("Agent Optimized", f"{optimized:,.0f} kWh")
        with cK3: kpi_card("Energy Saved", f"{saved:,.0f} kWh", sub=f"{(saved/(baseline+1e-9))*100:.1f}% vs baseline")
        with cK4: kpi_card("Anomalies", f"{anomalies_cnt}", sub="in range", color="#ffd166")

        c1, c2 = st.columns([1,1])
        with c1:
            # Energy efficiency gauge (fake ratio from baseline/optimized)
            eff = (optimized/(baseline+1e-9))
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round((1-eff)*100,1),
                number={'suffix': "%"},
                gauge={'axis':{'range':[0,100]},
                       'bar':{'color':"#4ade80"},
                       'bgcolor':"rgba(255,255,255,0.05)"}},
            )
            gauge.update_layout(height=320, margin=dict(l=20,r=20,b=10,t=10))
            st.plotly_chart(gauge, use_container_width=True, theme=None)
        with c2:
            if {"timestamp","optimized_kwh","energy_kwh"} <= set(df_overall_f.columns):
                tmp = df_overall_f[["timestamp","energy_kwh","optimized_kwh"]].melt("timestamp",
                           var_name="Series", value_name="kWh")
                st.plotly_chart(px.line(tmp, x="timestamp", y="kWh", color="Series",
                                        template="plotly_dark", height=320), use_container_width=True)
        st.markdown("#### System Performance")
        perf_cols = [c for c in ["temp_c","energy_kwh","vibration_mm_s"] if c in df_overall_f.columns]
        if len(perf_cols) >= 1:
            perf_m = df_overall_f.melt("timestamp", value_vars=perf_cols, var_name="Metric", value_name="Value")
            figp = px.line(perf_m, x="timestamp", y="Value", color="Metric", template="plotly_dark", height=340)
            st.plotly_chart(figp, use_container_width=True)

# -------------------------
# Pump Monitoring
# -------------------------
with tab_pump:
    if len(df_pump_f) == 0:
        st.warning("لا توجد بيانات للمضخات ضمن النطاق المحدد.")
    else:
        # KPI row (aggregate)
        active_ratio = (df_pump_f["status"].eq("Active").mean() if "status" in df_pump_f else 0.75)
        avg_eff = df_pump_f["efficiency"].mean() if "efficiency" in df_pump_f else 80
        avg_temp = df_pump_f["temperature"].mean() if "temperature" in df_pump_f else 42
        avg_vib = df_pump_f["vibration"].mean() if "vibration" in df_pump_f else 1.8

        cK1, cK2, cK3, cK4 = st.columns(4)
        with cK1: kpi_card("Pump Status", f"{round(active_ratio*len(df_pump_f['pump_id'].unique()))}/{len(df_pump_f['pump_id'].unique())}")
        with cK2: kpi_card("Efficiency", f"{avg_eff:,.0f}%", sub="Average")
        with cK3: kpi_card("Temperature", f"{avg_temp:,.0f}°C", color="#9ecbff")
        with cK4: kpi_card("Vibration", f"{avg_vib:,.1f} mm/s", color="#ffa8a8")

        c1, c2 = st.columns([1,1])
        with c1:
            if {"timestamp","vibration"} <= set(df_pump_f.columns):
                st.plotly_chart(px.line(df_pump_f, x="timestamp", y="vibration", color="pump_id",
                                        template="plotly_dark", height=320,
                                        title="Vibration Trends"), use_container_width=True)
        with c2:
            if {"timestamp","current_a","power_kw"} <= set(df_overall_f.columns):
                fig = px.line(df_overall_f, x="timestamp", y=["current_a","power_kw"],
                              template="plotly_dark", height=320, title="Current & Power")
                st.plotly_chart(fig, use_container_width=True)

        # Scatter of anomalies history (use df_ano)
        st.markdown("#### Historical Activity & Anomalies")
        if "anomaly" in df_ano.columns:
            dots = df_ano.copy()
            dots["Level"] = np.where(dots["anomaly"]==1, "Anomaly", "Normal")
            if {"timestamp","anomaly"} <= set(dots.columns):
                st.plotly_chart(px.scatter(dots, x="timestamp", y="z_score", color="Level",
                                           template="plotly_dark", height=280), use_container_width=True)

        # Pump detail table (latest by pump)
        st.markdown("#### Pump System Details")
        latest = (df_pump_f.sort_values("timestamp")
                  .groupby("pump_id").tail(1)
                  .sort_values("pump_id"))
        st.dataframe(latest[["pump_id","status","efficiency","flow_lpm","pressure_bar","power_kw"]],
                     use_container_width=True)

# -------------------------
# Lighting
# -------------------------
with tab_light:
    # We will emulate zones from df_overall_f occupancy/lux if absent
    zones = ["Zone A","Zone B","Zone C","Zone D"]
    active_zones = 18
    current_usage = df_overall_f["optimized_kwh"].iloc[-1] if "optimized_kwh" in df_overall_f else df_overall_f["energy_kwh"].iloc[-1]
    avg_lux = 485
    saved_pct = 100*(1 - (current_usage/(df_overall_f["energy_kwh"].iloc[-1]+1e-9))) if "energy_kwh" in df_overall_f else 18.0

    cK1, cK2, cK3, cK4 = st.columns(4)
    with cK1: kpi_card("Active Zones", f"{active_zones}", sub="of 24 zones")
    with cK2: kpi_card("Current Usage", f"{current_usage:,.0f} kWh today")
    with cK3: kpi_card("Avg Lux Level", f"{avg_lux}", color="#faca72")
    with cK4: kpi_card("Energy Saved", f"{saved_pct:,.1f}%", color="#9eff9e")

    c1, c2 = st.columns([1,1])
    with c1:
        # Energy Usage vs Baseline (bar)
        if {"timestamp","energy_kwh"} <= set(df_overall_f.columns):
            tmp = df_overall_f.copy()
            tmp["Baseline"] = tmp["energy_kwh"]
            tmp["Optimized"] = tmp["optimized_kwh"] if "optimized_kwh" in tmp else tmp["energy_kwh"]*0.8
            tmp = tmp.melt("timestamp", value_vars=["Baseline","Optimized"], var_name="Series", value_name="kWh")
            st.plotly_chart(px.bar(tmp, x="timestamp", y="kWh", color="Series",
                                   barmode="group", template="plotly_dark", height=320), use_container_width=True)
    with c2:
        # Lighting Efficiency gauge (just reuse saved_pct)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max(0.0, min(100.0, saved_pct if saved_pct>0 else 76.0)),
            number={'suffix': "%"},
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#4ade80"},
                   'bgcolor':"rgba(255,255,255,0.05)"}
        ))
        fig.update_layout(height=320, margin=dict(l=20,r=20,b=10,t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Lux vs Occupancy")
    # fabricate occupancy from energy curve
    if "timestamp" in df_overall_f:
        occ = df_overall_f[["timestamp"]].copy()
        occ["Lux"] = np.interp(range(len(occ)), [0,len(occ)//2,len(occ)-1], [100, 550, 120]) + np.random.normal(0, 20, len(occ))
        occ["Occupancy %"] = np.clip(np.interp(range(len(occ)), [0,len(occ)//2,len(occ)-1], [10, 85, 15])
                                     + np.random.normal(0, 3, len(occ)), 0, 100)
        fig2 = px.line(occ, x="timestamp", y=["Lux","Occupancy %"], template="plotly_dark", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Zone Status Overview (fake snapshot)
    st.markdown("#### Zone Status Overview")
    zone_rows = []
    rng = np.random.default_rng(1)
    for z in ["Zone A1","Zone A2","Zone B1","Zone B2","Zone C1","Zone C2"]:
        state = rng.choice(["Active","Dimmed","Fault"], p=[0.6,0.35,0.05])
        lux = int(rng.normal(450 if state!="Dimmed" else 210, 25))
        zone_rows.append([z, state, lux])
    st.dataframe(pd.DataFrame(zone_rows, columns=["Zone","State","Lux"]), use_container_width=True)

    # Daily Lighting Trends
    st.markdown("#### Daily Lighting Trends")
    if "timestamp" in df_overall_f.columns:
        zA = np.clip(np.linspace(5, 60, len(df_overall_f)) + np.random.normal(0, 2, len(df_overall_f)), 0, None)
        zB = zA*0.8 + np.random.normal(0, 1.5, len(zA))
        zC = zA*0.65 + np.random.normal(0, 1.2, len(zA))
        trend = pd.DataFrame({"timestamp": df_overall_f["timestamp"], "Zone A": zA, "Zone B": zB, "Zone C": zC})
        st.plotly_chart(px.line(trend, x="timestamp", y=["Zone A","Zone B","Zone C"], template="plotly_dark", height=340),
                        use_container_width=True)

# -------------------------
# Agent Insights
# -------------------------
with tab_agents:
    if len(df_log_f) == 0:
        st.info("لا يوجد سجلات للوكيل ضمن النطاق. (ربما الداتا غير مربوطة —
