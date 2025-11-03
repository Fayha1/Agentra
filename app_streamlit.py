# app_streamlit.py  (Agentra – high-fidelity UI)
from __future__ import annotations
import os
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Plotly ----------
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except Exception:
    PLOTLY = False

# ---------- Optional sklearn ----------
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "FullLogo_Transparent.png"  # انتبهي لحالة الأحرف

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

# ---------- Global style ----------
st.markdown("""
<style>
  .block-container{padding-top:0.8rem;padding-bottom:0.4rem;}
  h1,h2,h3,h4{letter-spacing:.3px}
  .kpi{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
       border-radius:16px;padding:14px 16px}
  .card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.05);
       border-radius:18px;padding:16px;margin-bottom:14px}
  .muted{color:#aab3c1;font-size:.86rem}
  .center-header{text-align:center;margin:8px 0 10px}
  .center-header h2{margin:6px 0 2px;color:#eaffea;font-weight:800}
  .center-header .sub{color:#b9c0cc;font-size:13px}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def render_center_header():
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_column_width=False, width=170)
        st.markdown('<div class="center-header"><h2>Agentra</h2>'
                    '<div class="sub">Predictive Property Manager</div></div>',
                    unsafe_allow_html=True)

def _plotly_layout(fig, h=320):
    fig.update_layout(
        template="plotly_dark",
        height=h,
        margin=dict(l=10,r=10,t=40,b=10),
        plot_bgcolor="rgba(18,18,18,0)",
        paper_bgcolor="rgba(18,18,18,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def gauge(value, title, min_v=0, max_v=100, unit=""):
    """High quality gauge."""
    if not PLOTLY:
        st.metric(title, f"{value:.1f}{unit}")
        return
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={'suffix': unit, 'font': {'size': 26}},
        title={'text': f"<b>{title}</b>", 'font': {'size': 14}},
        gauge=dict(
            axis={'range':[min_v,max_v], 'tickwidth':1, 'tickcolor':'#7a7a7a'},
            bar={'color': '#25ff88'},
            bgcolor="rgba(255,255,255,.02)",
            bordercolor="rgba(255,255,255,.08)",
            borderwidth=1,
            steps=[
                {'range': [min_v, (min_v+max_v)*0.6], 'color': "rgba(37,255,136,.15)"},
                {'range': [(min_v+max_v)*0.6, (min_v+max_v)*0.85], 'color': "rgba(255,176,32,.15)"},
                {'range': [(min_v+max_v)*0.85, max_v], 'color': "rgba(255,77,79,.15)"},
            ],
        )))
    _plotly_layout(fig, h=250)
    st.plotly_chart(fig, use_container_width=True)

def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists(): return None
    for enc in ("utf-8","utf-8-sig","cp1256"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    try: return pd.read_csv(path)
    except Exception: return None

def simulate_energy(start_dt, end_dt, freq="H"):
    idx = pd.date_range(start_dt, end_dt, freq=freq)
    base = 40 + 5*np.sin(np.linspace(0,2*np.pi,len(idx)))
    usage = base + np.random.normal(0,2.1,len(idx))
    optimized = usage * np.random.uniform(0.74,0.9)
    temp = 24 + 6*np.sin(np.linspace(0,2*np.pi,len(idx))-1) + np.random.normal(0,0.8,len(idx))
    vib = 1.6 + 0.7*np.sin(np.linspace(0,4*np.pi,len(idx))) + np.random.normal(0,0.12,len(idx))
    return pd.DataFrame({
        "timestamp": idx,
        "energy_kwh": np.clip(usage*10, 15, None),
        "optimized_kwh": np.clip(optimized*10, 10, None),
        "temperature_c": np.clip(temp, 16, 45),
        "vibration_mm_s": np.clip(vib, 0.3, 6.5),
        "current_a": 3 + (usage-usage.min())/(usage.max()-usage.min()+1e-9)*2.5,
        "power_kw": (usage/10)*2.6
    })

def simulate_pumps(start_dt, end_dt, freq="H"):
    idx = pd.date_range(start_dt, end_dt, freq=freq)
    n = len(idx)
    eff = 75 + 8*np.sin(np.linspace(0,2*np.pi,n)) + np.random.normal(0,2,n)
    vib = 1.2 + 1.2*np.sin(np.linspace(0,3*np.pi,n)) + np.random.normal(0,0.15,n)
    temp = 40 + 6*np.sin(np.linspace(0,2*np.pi,n)-.6) + np.random.normal(0,.9,n)
    cur = 1.8 + (eff-60)/60
    pwr = 1.2 + (cur-1.2)*0.8
    return pd.DataFrame({
        "timestamp": idx,
        "pump_id": np.random.choice(["PUMP-001","PUMP-002","PUMP-003","PUMP-004"], size=n),
        "efficiency_pct": np.clip(eff, 40, 98),
        "vibration_mm_s": np.clip(vib, .4, 8.0),
        "temperature_c": np.clip(temp, 25, 90),
        "current_a": np.clip(cur, .8, 5.0),
        "power_kw": np.clip(pwr, .5, 10.0),
        "flow_l_min": np.clip(220 + (eff-60)*2.5, 0, None),
        "pressure_bar": np.clip(2.0 + (eff-70)/50, 0.8, 4.0),
    })

def simulate_logs(start_dt, end_dt):
    idx = pd.date_range(start_dt, end_dt, freq="3H")
    agents = ["HVAC Agent","Lighting Agent","Pump Agent"]
    actions = ["Temperature Adjustment","Brightness Optimization","Speed Optimization",
               "Schedule Override","Energy Optimization","Anomaly Detection"]
    sev = ["Low","Medium","High","Critical"]
    rng = np.random.default_rng(7)
    rows=[]
    for ts in idx:
        for _ in range(rng.integers(1,4)):
            rows.append({
                "timestamp": ts + dt.timedelta(minutes=int(rng.integers(0,180))),
                "agent": rng.choice(agents),
                "action": rng.choice(actions),
                "target": rng.choice(["Zone A-2","Floor 3","Pump Unit 7","Building Wide"]),
                "severity": rng.choice(sev, p=[.35,.4,.2,.05]),
                "decision": rng.choice(["Applied","Optimized","Resolved","Monitoring","Executed","Pending"]),
                "status": rng.choice(["Completed","Pending","Completed","Completed","Completed","Completed"])
            })
    return pd.DataFrame(rows).sort_values("timestamp")

def load_or_sim(csv_path, gen, start_dt, end_dt):
    df = read_csv_safe(csv_path)
    if df is None or df.empty:
        return gen(start_dt, end_dt)
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df[(df["timestamp"]>=start_dt)&(df["timestamp"]<=end_dt)]
    return df

def rolling_anoms(series, window, z, slope):
    s = series.astype(float).copy()
    mean = s.rolling(window, min_periods=max(2,window//2)).mean()
    std = s.rolling(window, min_periods=max(2,window//2)).std(ddof=0)
    zscore = (s-mean)/(std.replace(0,np.nan))
    grad = s.diff().fillna(0.0)
    return ((zscore.abs()>z) | (grad.abs()>slope)).fillna(False)

def iforest_anoms(df, cols, contamination=0.04):
    if not SKLEARN_AVAILABLE:  # حماية
        return pd.Series([False]*len(df), index=df.index)
    sub = df[cols].astype(float).fillna(method="ffill").fillna(method="bfill")
    try:
        mdl = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
        pred = mdl.fit_predict(sub.values)
        return pd.Series(pred==-1, index=df.index)
    except Exception:
        return pd.Series([False]*len(df), index=df.index)

# ---------- Header ----------
render_center_header()

# ---------- Sidebar ----------
st.sidebar.title("Agentra")
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width=90)
else:
    st.sidebar.caption("Place logo at: assets/FullLogo_Transparent.png")

data_source = st.sidebar.selectbox("Select data source",
                                   ["Real-Time Sensors","Historical Data","Predictive Model"], index=0)

z_score = st.sidebar.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.1)
slope = st.sidebar.slider("Slope Sensitivity", 0.1, 2.0, 0.8, 0.05)
roll_win = st.sidebar.slider("Rolling Window (points)", 8, 72, 24, 1)

# تعطيل Advanced إن لم تتوفر sklearn (بدون أخطاء عند الضغط)
use_iforest = st.sidebar.checkbox("Advanced: IsolationForest",
                                  value=False, disabled=not SKLEARN_AVAILABLE,
                                  help=None if SKLEARN_AVAILABLE else "scikit-learn غير متاح هنا.")

def date_range():
    today = dt.date.today()
    c1, c2 = st.sidebar.columns(2)
    s = c1.date_input("From", today - dt.timedelta(days=6))
    e = c2.date_input("To", today)
    if s > e: s, e = e, s
    return dt.datetime.combine(s, dt.time.min), dt.datetime.combine(e, dt.time.max)

start_dt, end_dt = date_range()

# ---------- Load data ----------
if data_source == "Historical Data":
    energy = load_or_sim(CSV_ENERGY, simulate_energy, start_dt, end_dt)
    saved = read_csv_safe(CSV_ENERGY_SAVED)
    if saved is not None and "timestamp" in saved and "optimized_kwh" in saved:
        saved["timestamp"]=pd.to_datetime(saved["timestamp"],errors="coerce")
        saved=saved.dropna(subset=["timestamp"])
        energy = energy.merge(saved[["timestamp","optimized_kwh"]], on="timestamp", how="left", suffixes=("","_alt"))
        energy["optimized_kwh"] = energy["optimized_kwh_alt"].fillna(energy["optimized_kwh"])
        energy.drop(columns=[c for c in energy.columns if c.endswith("_alt")], inplace=True)
    pumps = load_or_sim(CSV_PUMPS, simulate_pumps, start_dt, end_dt)
    logs = load_or_sim(CSV_AGENT_LOG, simulate_logs, start_dt, end_dt)
elif data_source == "Predictive Model":
    energy = simulate_energy(start_dt, end_dt); energy["optimized_kwh"] *= np.random.uniform(0.68,0.85)
    pumps = simulate_pumps(start_dt, end_dt);  pumps["efficiency_pct"] += np.random.uniform(2,4)
    logs  = simulate_logs(start_dt, end_dt)
else:
    energy = simulate_energy(start_dt, end_dt)
    pumps  = simulate_pumps(start_dt, end_dt)
    logs   = simulate_logs(start_dt, end_dt)

for df in (energy, pumps, logs):
    if "timestamp" in df: df.sort_values("timestamp", inplace=True); df.reset_index(drop=True, inplace=True)

# ---------- Anomalies ----------
energy["anomaly_energy"] = rolling_anoms(energy["energy_kwh"], roll_win, z_score, slope)
energy["anomaly_vibration"] = rolling_anoms(energy["vibration_mm_s"], roll_win, z_score, slope)
pumps["anomaly_eff"] = rolling_anoms(pumps["efficiency_pct"], roll_win, z_score, slope)
pumps["anomaly_iforest"] = iforest_anoms(pumps, ["efficiency_pct","vibration_mm_s","temperature_c","current_a","power_kw"], .05) if use_iforest else False

total_baseline = float(energy["energy_kwh"].sum())
total_optimized = float(energy["optimized_kwh"].sum())
total_saved = max(total_baseline - total_optimized, 0.0)
anomaly_count = int(energy["anomaly_energy"].sum() + energy["anomaly_vibration"].sum()
                    + pumps["anomaly_eff"].sum() + (pumps["anomaly_iforest"].sum() if isinstance(pumps["anomaly_iforest"], pd.Series) else 0))

# ---------- Tabs ----------
tabs = st.tabs(["Overview","Pump Monitoring","Lighting","Agent Insights"])

# ===== Overview =====
with tabs[0]:
    st.markdown("### Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.container().markdown(f'<div class="kpi"><h4>Baseline Energy</h4><h3>{total_baseline:,.0f} kWh</h3></div>', unsafe_allow_html=True)
    c2.container().markdown(f'<div class="kpi"><h4>Optimized Energy</h4><h3>{total_optimized:,.0f} kWh</h3></div>', unsafe_allow_html=True)
    c3.container().markdown(f'<div class="kpi"><h4>Energy Saved</h4><h3>{total_saved:,.0f} kWh</h3></div>', unsafe_allow_html=True)
    c4.container().markdown(f'<div class="kpi"><h4>Anomalies</h4><h3>{anomaly_count}</h3></div>', unsafe_allow_html=True)

    l, r = st.columns([1.1,1.2])
    with l:
        st.markdown("#### Energy Efficiency")
        eff_now = np.clip(100.0 * (total_optimized/total_baseline) if total_baseline else 0, 0, 100)
        gauge(eff_now, "Efficiency", 0, 100, "%")
    with r:
        st.markdown("#### Real-Time Energy Savings")
        df = energy[["timestamp","energy_kwh","optimized_kwh"]].rename(columns={"energy_kwh":"Baseline","optimized_kwh":"Optimized"})
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["Baseline"], name="Baseline",
                                     mode="lines+markers", line=dict(width=3), marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["Optimized"], name="Optimized",
                                     mode="lines+markers", line=dict(width=3), marker=dict(size=5)))
            _plotly_layout(fig, h=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df.set_index("timestamp")[["Baseline","Optimized"]])

    st.markdown("#### System Performance")
    perf = pd.DataFrame({
        "timestamp": energy["timestamp"],
        "Temperature (°C)": energy["temperature_c"],
        "Energy Usage (kWh)": energy["energy_kwh"],
        "Vibration (mm/s)": energy["vibration_mm_s"],
    })
    if PLOTLY:
        fig = go.Figure()
        for col in ["Temperature (°C)","Energy Usage (kWh)","Vibration (mm/s)"]:
            fig.add_trace(go.Scatter(x=perf["timestamp"], y=perf[col], name=col,
                                     mode="lines", line=dict(width=3)))
        _plotly_layout(fig, h=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(perf.set_index("timestamp"))

# ===== Pump Monitoring =====
with tabs[1]:
    st.markdown("### Pump Monitoring")
    t1,t2,t3,t4 = st.columns(4)
    t1.markdown(f'<div class="kpi"><h4>Pump Status</h4><h3>{(pumps["efficiency_pct"]>0).sum()}/{pumps["pump_id"].nunique()}</h3></div>', unsafe_allow_html=True)
    t2.markdown(f'<div class="kpi"><h4>Avg Efficiency</h4><h3>{pumps["efficiency_pct"].mean():.0f}%</h3></div>', unsafe_allow_html=True)
    t3.markdown(f'<div class="kpi"><h4>Temperature</h4><h3>{pumps["temperature_c"].mean():.0f}°C</h3></div>', unsafe_allow_html=True)
    t4.markdown(f'<div class="kpi"><h4>Vibration</h4><h3>{pumps["vibration_mm_s"].mean():.1f} mm/s</h3></div>', unsafe_allow_html=True)

    g1,g2,g3 = st.columns(3)
    with g1: st.markdown("#### Pump Efficiency"); gauge(pumps["efficiency_pct"].tail(1).mean(), "Efficiency", 0, 100, "%")
    with g2: st.markdown("#### Vibration Level");  gauge(pumps["vibration_mm_s"].tail(1).mean(), "mm/s", 0, 8, " mm/s")
    with g3: st.markdown("#### Temperature");      gauge(pumps["temperature_c"].tail(1).mean(), "°C", 0, 100, "°C")

    st.markdown("#### Vibration Trends & Current vs Power")
    a,b = st.columns(2)
    with a:
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pumps["timestamp"], y=pumps["vibration_mm_s"],
                                     mode="lines+markers", name="Vibration",
                                     line=dict(width=3), marker=dict(size=5)))
            thr = pumps["vibration_mm_s"].mean() + pumps["vibration_mm_s"].std()
            fig.add_hline(y=float(thr), line_dash="dash", line_color="#ffb020")
            _plotly_layout(fig, h=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pumps.set_index("timestamp")[["vibration_mm_s"]])
    with b:
        if PLOTLY:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
    go.Scatter(
        x=pumps["timestamp"],
        y=pumps["current_a"],
        name="Current (A)",
        mode="lines+markers",
        line=dict(width=3),
        marker=dict(size=5)
    ),
    secondary_y=False
)

            fig.update_yaxes(title_text="Current (A)", secondary_y=False)
            fig.update_yaxes(title_text="Power (kW)", secondary_y=True)
            _plotly_layout(fig, h=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pumps.set_index("timestamp")[["current_a","power_kw"]])

    st.markdown("#### Historical Activity & Anomalies")
    pump_ano = pd.DataFrame({
        "timestamp": pumps["timestamp"],
        "Critical": ( (pumps["vibration_mm_s"] > 3.5) | (isinstance(pumps["anomaly_iforest"], pd.Series) & pumps["anomaly_iforest"]) ).astype(int),
        "Warning": pumps["anomaly_eff"].astype(int),
        "Normal": (~(pumps["anomaly_eff"] | (isinstance(pumps["anomaly_iforest"], pd.Series) & pumps["anomaly_iforest"]))).astype(int),
    })
    if PLOTLY:
        fig = go.Figure()
        for col, color in [("Critical","#ff4d4f"),("Warning","#ffb020"),("Normal","#22dd77")]:
            fig.add_trace(go.Scatter(x=pump_ano["timestamp"], y=pump_ano[col], mode="markers",
                                     name=col, marker=dict(color=color, size=7, opacity=.9)))
        _plotly_layout(fig, h=260)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pump_ano.set_index("timestamp")[["Critical","Warning","Normal"]])

    st.markdown("#### Pump System Details")
    details = pumps.copy()
    if isinstance(details["anomaly_iforest"], pd.Series):
        details["status"] = np.where(details["anomaly_iforest"] | details["anomaly_eff"], "⚠️ Check", "Active")
    else:
        details["status"] = np.where(details["anomaly_eff"], "⚠️ Check", "Active")
    st.dataframe(
        details[["timestamp","pump_id","status","flow_l_min","pressure_bar","power_kw"]]
        .rename(columns={"timestamp":"Time","pump_id":"Pump","flow_l_min":"Flow (L/min)","pressure_bar":"Pressure (bar)","power_kw":"Power (kW)"}),
        use_container_width=True, height=260
    )

# ===== Lighting =====
with tabs[2]:
    st.markdown("### Lighting")
    t = st.columns(4)
    active_zones = int(np.random.randint(18,36))
    curr_kwh = float(energy["optimized_kwh"].tail(24).sum())
    avg_lux = int(np.clip(480 + np.random.randn()*30, 200, 900))
    pct_saved = (1 - (total_optimized/total_baseline))*100 if total_baseline else 0.0
    t[0].markdown(f'<div class="kpi"><h4>Active Zones</h4><h3>{active_zones}/36</h3></div>', unsafe_allow_html=True)
    t[1].markdown(f'<div class="kpi"><h4>Current Usage</h4><h3>{curr_kwh:.0f} kWh</h3></div>', unsafe_allow_html=True)
    t[2].markdown(f'<div class="kpi"><h4>Avg Lux Level</h4><h3>{avg_lux} lux</h3></div>', unsafe_allow_html=True)
    t[3].markdown(f'<div class="kpi"><h4>Energy Saved</h4><h3>{pct_saved:.1f}%</h3></div>', unsafe_allow_html=True)

    l, r = st.columns([1.1,1.2])
    with l:
        st.markdown("#### Energy Usage vs Baseline")
        bars = pd.DataFrame({"Day":["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                             "Baseline": np.random.randint(400,520,7),
                             "Actual":   np.random.randint(260,430,7)})
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bars["Day"], y=bars["Baseline"], name="Baseline"))
            fig.add_trace(go.Bar(x=bars["Day"], y=bars["Actual"],   name="Actual"))
            fig.update_layout(barmode="group")
            _plotly_layout(fig, h=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(bars.set_index("Day"))
    with r:
        st.markdown("#### Lighting Efficiency")
        gauge(76, "Efficient", 0, 100, "%")

    st.markdown("#### Lux Levels vs. Occupancy")
    lux = 150 + np.array([80,150,220,260,230,180,120,60])
    occ = np.array([5,15,35,47,40,28,12,4])
    xh = ["6AM","8AM","10AM","12PM","3PM","6PM","8PM","9PM"]
    if PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=xh, y=lux, name="Lux Level",
                                 mode="lines+markers", line=dict(width=3), marker=dict(size=5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=xh, y=occ, name="Occupancy %",
                                 mode="lines+markers", line=dict(width=3), marker=dict(size=5)), secondary_y=True)
        fig.update_yaxes(title_text="Lux", secondary_y=False)
        fig.update_yaxes(title_text="Occupancy (%)", secondary_y=True)
        _plotly_layout(fig, h=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame({"Lux":lux,"Occupancy %":occ}, index=xh))

# ===== Agent Insights =====
with tabs[3]:
    st.markdown("### Agent Insights")
    fc1,fc2,fc3,fc4 = st.columns(4)
    fc1.markdown(f'<div class="kpi"><h4>Total Actions</h4><h3>{len(logs):,}</h3></div>', unsafe_allow_html=True)
    active_anom = anomaly_count if anomaly_count else np.random.randint(1,8)
    fc2.markdown(f'<div class="kpi"><h4>Active Anomalies</h4><h3>{active_anom}</h3></div>', unsafe_allow_html=True)
    fc3.markdown(f'<div class="kpi"><h4>Success Rate</h4><h3>94.2%</h3></div>', unsafe_allow_html=True)
    fc4.markdown(f'<div class="kpi"><h4>Avg Response Time</h4><h3>2.4s</h3></div>', unsafe_allow_html=True)

    st.markdown("#### Anomaly Severity Distribution")
    sev = logs["severity"].value_counts().reindex(["Critical","High","Medium","Low"]).fillna(0)
    if PLOTLY:
        fig = go.Figure(go.Bar(x=sev.index, y=sev.values, marker_color=["#ff4d4f","#ffb020","#ffd24a","#22dd77"]))
        _plotly_layout(fig, h=280)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(sev)

    st.markdown("#### Agent Interventions Timeline")
    tl = logs.copy()
    tl["hour"] = tl["timestamp"].dt.floor("H")
    time_counts = tl.groupby(["hour","agent"]).size().unstack(fill_value=0)
    if PLOTLY:
        fig = go.Figure()
        for col in time_counts.columns:
            fig.add_trace(go.Scatter(x=time_counts.index, y=time_counts[col], name=col,
                                     mode="lines+markers", line=dict(width=3), marker=dict(size=5)))
        _plotly_layout(fig, h=310)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(time_counts)

    st.markdown("#### Agent Event Logs")
    show = ["timestamp","agent","action","target","severity","decision","status"]
    show = [c for c in show if c in logs.columns]
    st.dataframe(logs[show].sort_values("timestamp", ascending=False),
                 use_container_width=True, height=320)

st.markdown("<div class='muted' style='text-align:center;margin-top:10px'>"
            "Anomaly engine: rolling Z-score + gradient; optional IsolationForest."
            "</div>", unsafe_allow_html=True)
