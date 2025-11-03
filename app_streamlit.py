# app_streamlit.py
# -----------------------------------------
# Agentra - Predictive Property Manager (Streamlit)
# Tabs: Overview, Pump Monitoring, Lighting, Agent Insights
# Data: sim_property_riyadh_multi.csv, sim_property_riyadh_multi_saving15.csv,
#       sim_pump_riyadh.csv, agent_audit_log.csv
# Logo: assets/FullLogo_Transparent.png
# -----------------------------------------

from __future__ import annotations
import os
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Config & Theme
# -------------------------
st.set_page_config(
    page_title="Agentra – Predictive Property Manager",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject dark styling aligned with brand
st.markdown(
    """
    <style>
        :root {
          --brand:#27E36A;   /* اللمسة الخضراء */
          --bg:#0B0B0B;
          --panel:#121418;
          --muted:#B9C0CC;
          --danger:#FF4D4F;
          --warn:#FFA500;
          --ok:#27E36A;
        }
        body { background-color: var(--bg); }
        [data-testid="stAppViewContainer"] { background-color: var(--bg); }
        [data-testid="stSidebar"] { background: #000; border-right: 1px solid #1C1F26; }
        .stMetric label, .stMarkdown, .stText, .stNumberInput, .stDateInput, .stCheckbox, .stSelectbox, .stSlider {
          color: #E6E8EE !important;
        }
        h1,h2,h3,h4,h5,h6 { color:#E6FFE6; }
        .kpi-card {
            background: var(--panel);
            padding: 16px 18px;
            border-radius: 14px;
            border: 1px solid #1C1F26;
        }
        .section {
            background: var(--panel);
            padding: 14px 16px;
            border-radius: 14px;
            border: 1px solid #1C1F26;
        }
        .pill {
            display:inline-block;
            padding: 2px 10px;
            border-radius:999px;
            font-size: 12px;
            margin-left:6px;
        }
        .pill.ok{ background:rgba(39,227,106,0.15); color:#7CFFA9; }
        .pill.warn{ background:rgba(255,165,0,0.15); color:#FFC879; }
        .pill.danger{ background:rgba(255,77,79,0.15); color:#FFA7A8; }
        .muted { color: var(--muted); font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Paths & Helpers
# -------------------------
ROOT = pathlib.Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "FullLogo_Transparent.png"

DATA_PROPERTY = ROOT / "sim_property_riyadh_multi.csv"
DATA_PROPERTY_SAVING = ROOT / "sim_property_riyadh_multi_saving15.csv"
DATA_PUMP = ROOT / "sim_pump_riyadh.csv"
DATA_AGENT_LOG = ROOT / "agent_audit_log.csv"


@st.cache_data(show_spinner=False)
def load_csv(path: pathlib.Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path)
            if parse_dates:
                for c in parse_dates:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
            return df
        except Exception as e:
            st.warning(f"تعذّر قراءة الملف: {path.name} ({e}). سيتم توليد بيانات بديلة.")
    return pd.DataFrame()


def fake_property_data(n_days: int = 7) -> pd.DataFrame:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    idx = pd.date_range(now - timedelta(days=n_days-1), periods=n_days, freq="D")
    base = np.random.randint(2500, 3200, size=n_days)  # kWh baseline
    optimized = (base * np.random.uniform(0.7, 0.9, size=n_days)).astype(int)
    saved = base - optimized
    eff = np.clip(np.random.normal(0.82, 0.05, n_days), 0.55, 0.98)
    temp = np.clip(np.random.normal(26, 3, n_days), 18, 40)
    vibr = np.clip(np.random.normal(2.0, 0.4, n_days), 1.0, 3.8)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "baseline_kwh": base,
            "optimized_kwh": optimized,
            "saved_kwh": saved,
            "efficiency": eff,
            "temperature_c": temp,
            "vibration_mm_s": vibr,
        }
    )


def fake_pump_data(n_hours: int = 24) -> pd.DataFrame:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    idx = pd.date_range(now - timedelta(hours=n_hours - 1), periods=n_hours, freq="H")
    curr = np.clip(np.random.normal(4.5, 0.4, n_hours), 3.8, 5.2)  # Amps
    power = np.clip(curr * np.random.uniform(0.45, 0.55, n_hours), 1.8, 2.6)  # kW
    vib = np.clip(np.random.normal(1.8, 0.5, n_hours), 0.8, 3.6)  # mm/s
    temp = np.clip(np.random.normal(48, 4, n_hours), 36, 65)  # °C
    eff = np.clip(np.random.normal(0.84, 0.07, n_hours), 0.55, 0.98)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "current_a": curr,
            "power_kw": power,
            "vibration_mm_s": vib,
            "temperature_c": temp,
            "efficiency": eff,
        }
    )


def fake_agent_log(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_time = datetime.now() - timedelta(hours=12)
    agents = ["HVAC Agent", "Lighting Agent", "Pump Agent"]
    actions = ["Temperature Adjustment", "Brightness Optimization", "Speed Optimization", "Anomaly Detection", "Schedule Override"]
    severity_vals = ["Low", "Medium", "High", "Critical"]
    decision_vals = ["Optimized", "Applied", "Resolved", "Monitoring", "Executed", "Pending", "Reduced by 3°C", "Schedule Maintenance", "Optimize Schedule"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": base_time + timedelta(minutes=15*i),
                "agent": rng.choice(agents),
                "action": rng.choice(actions),
                "target": rng.choice(["Zone A-203", "Pump-001", "Floor 2 East", "Building Wide", "Conference Room A"]),
                "severity": rng.choice(severity_vals, p=[0.35, 0.33, 0.22, 0.10]),
                "decision": rng.choice(decision_vals),
                "status": rng.choice(["Completed", "Pending", "In-Progress"]),
            }
        )
    return pd.DataFrame(rows)


def coalesce(df: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    return df if not df.empty else fallback


# -------------------------
# Load data
# -------------------------
property_df = load_csv(DATA_PROPERTY, parse_dates=["timestamp"])
property_saving_df = load_csv(DATA_PROPERTY_SAVING, parse_dates=["timestamp"])
pump_df = load_csv(DATA_PUMP, parse_dates=["timestamp"])
agent_df = load_csv(DATA_AGENT_LOG, parse_dates=["timestamp"])

property_df = coalesce(property_df, fake_property_data(14))
property_saving_df = coalesce(property_saving_df, property_df.copy())
pump_df = coalesce(pump_df, fake_pump_data(48))
agent_df = coalesce(agent_df, fake_agent_log(120))

# Normalize columns if needed
def safe_col(df: pd.DataFrame, name: str, default):
    if name not in df.columns:
        df[name] = default
    return df

for df in [property_df, property_saving_df]:
    df = safe_col(df, "baseline_kwh", np.random.randint(2500, 3200))
    df = safe_col(df, "optimized_kwh", df["baseline_kwh"] * 0.8)
    df = safe_col(df, "saved_kwh", df["baseline_kwh"] - df["optimized_kwh"])
    df = safe_col(df, "efficiency", 0.8)

pump_df = safe_col(pump_df, "efficiency", 0.82)
pump_df = safe_col(pump_df, "vibration_mm_s", 1.8)
pump_df = safe_col(pump_df, "temperature_c", 48.0)
pump_df = safe_col(pump_df, "current_a", 4.5)
pump_df = safe_col(pump_df, "power_kw", 2.2)

# -------------------------
# Sidebar: Logo & Controls
# -------------------------
if LOGO_PATH.exists():
    try:
        logo_img = Image.open(LOGO_PATH)
        st.sidebar.image(logo_img, caption="", use_column_width=True)
    except Exception:
        st.sidebar.markdown("### Agentra")

st.sidebar.markdown("#### Data Source")
data_source = st.sidebar.selectbox(
    "Select data source",
    ["Real-Time Sensors", "Historical Data", "Predictive Model"],
    index=0,
)

st.sidebar.markdown("#### Anomaly Thresholds")
z_score = st.sidebar.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.1)
slope_sens = st.sidebar.slider("Slope Sensitivity", 0.1, 2.0, 0.8, 0.1)

st.sidebar.markdown("#### Date Range")
start_date = st.sidebar.date_input("From", value=(datetime.now() - timedelta(days=6)).date())
end_date = st.sidebar.date_input("To", value=datetime.now().date())

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="muted">© 2025 Agentra • Streamlit + Plotly</span>',
    unsafe_allow_html=True,
)

# Filter data by range
def filter_by_date(df: pd.DataFrame, start, end):
    if "timestamp" not in df.columns:
        return df
    mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
    return df.loc[mask].reset_index(drop=True)

property_view = filter_by_date(property_df.copy(), start_date, end_date)
saving_view = filter_by_date(property_saving_df.copy(), start_date, end_date)
pump_view = filter_by_date(pump_df.copy(), start_date, end_date)
agent_view = filter_by_date(agent_df.copy(), start_date, end_date)

# -------------------------
# Reusable visuals
# -------------------------
def kpi_card(label: str, value: str, sublabel: str | None = None, color: str = "ok"):
    pill = f'<span class="pill {color}">{sublabel}</span>' if sublabel else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div style="color:#9CA3AF; font-size:12px;">{label}</div>
            <div style="font-size:28px; font-weight:700; color:#FFFFFF; margin:4px 0 6px 0;">{value}</div>
            <div>{pill}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def gauge(title: str, val: float, suffix: str = "%", min_v: float = 0, max_v: float = 100):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(val),
            number={"suffix": suffix, "font": {"color": "#E6E8EE"}},
            title={"text": title, "font": {"color": "#E6E8EE"}},
            gauge={
                "axis": {"range": [min_v, max_v], "tickcolor": "#7B8190", "tickwidth": 1},
                "bar": {"color": "#27E36A"},
                "bgcolor": "#121418",
                "borderwidth": 1,
                "bordercolor": "#1C1F26",
                "steps": [
                    {"range": [min_v, (min_v+max_v)*0.55], "color": "#1B1E24"},
                    {"range": [(min_v+max_v)*0.55, (min_v+max_v)*0.8], "color": "#18221B"},
                    {"range": [(min_v+max_v)*0.8, max_v], "color": "#152418"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="#121418")
    st.plotly_chart(fig, use_container_width=True, theme=None)

def line_chart(df: pd.DataFrame, x: str, y: str | list[str], title: str, yaxis_title: str = ""):
    fig = px.line(df, x=x, y=y, markers=False, title=title)
    fig.update_layout(
        height=300,
        paper_bgcolor="#121418",
        plot_bgcolor="#121418",
        font_color="#E6E8EE",
        title_font_color="#E6E8EE",
        margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="#232832")
    fig.update_yaxes(gridcolor="#232832", title=yaxis_title)
    st.plotly_chart(fig, use_container_width=True, theme=None)

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.bar(df, x=x, y=y, title=title)
    fig.update_layout(
        height=300,
        paper_bgcolor="#121418",
        plot_bgcolor="#121418",
        font_color="#E6E8EE",
        title_font_color="#E6E8EE",
        margin=dict(l=10,r=10,t=40,b=10),
    )
    fig.update_xaxes(gridcolor="#232832")
    fig.update_yaxes(gridcolor="#232832")
    st.plotly_chart(fig, use_container_width=True, theme=None)

# -------------------------
# Header with brand wordmark at top-left
# -------------------------
c1, c2 = st.columns([0.7, 0.3])
with c1:
    st.markdown("<h2 style='margin-top:6px'>Agentra</h2><div class='muted'>Predictive Property Manager</div>", unsafe_allow_html=True)

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_pump, tab_light, tab_agent = st.tabs(
    ["Overview", "Pump Monitoring", "Lighting", "Agent Insights"]
)

# =========================
# OVERVIEW TAB
# =========================
with tab_overview:
    ov = property_view.copy()
    if ov.empty:
        st.info("لا يوجد بيانات ضمن النطاق الزمني المحدد.")
    else:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            kpi_card("Baseline Energy", f"{int(ov['baseline_kwh'].iloc[-1]):,} kWh", "This month", "ok")
        with kpi2:
            kpi_card("Optimized Energy", f"{int(ov['optimized_kwh'].iloc[-1]):,} kWh", f"{int(100*(1-ov['optimized_kwh'].iloc[-1]/ov['baseline_kwh'].iloc[-1]))}% vs baseline", "ok")
        with kpi3:
            kpi_card("Energy Saved", f"{int(ov['saved_kwh'].iloc[-1]):,} kWh", "$≈ cost saved today", "ok")
        with kpi4:
            active_anoms = np.random.randint(2, 9)  # demo
            resolved = np.random.randint(1, active_anoms)
            kpi_card("Anomalies", f"{active_anoms}", f"{resolved} resolved", "warn" if active_anoms <= 6 else "danger")

        # Efficiency gauge + Savings line
        colA, colB = st.columns([0.45, 0.55])
        with colA:
            eff_perc = float(np.clip(100 * ov["efficiency"].iloc[-1], 1, 100))
            gauge("Energy Efficiency", eff_perc, "%", 0, 100)
        with colB:
            line_chart(ov, x="timestamp", y="saved_kwh", title="Real-Time Energy Savings", yaxis_title="kWh Saved")

        # System performance (temperature, energy, vibration)
        st.markdown("#### System Performance")
        perf = ov[["timestamp", "temperature_c", "baseline_kwh", "vibration_mm_s"]].rename(
            columns={"baseline_kwh": "energy_kwh"}
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf["timestamp"], y=perf["temperature_c"], name="Temperature (°C)", mode="lines"))
        fig.add_trace(go.Scatter(x=perf["timestamp"], y=perf["energy_kwh"], name="Energy (kWh)", mode="lines", yaxis="y2"))
        fig.add_trace(go.Scatter(x=perf["timestamp"], y=perf["vibration_mm_s"], name="Vibration (mm/s)", mode="lines", yaxis="y3"))

        fig.update_layout(
            height=360,
            paper_bgcolor="#121418",
            plot_bgcolor="#121418",
            font_color="#E6E8EE",
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(gridcolor="#232832"),
            yaxis=dict(title="Temp (°C)", gridcolor="#232832"),
            yaxis2=dict(title="Energy (kWh)", overlaying="y", side="right"),
            yaxis3=dict(title="Vibration (mm/s)", overlaying="y", side="right", position=0.95),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

# =========================
# PUMP MONITORING TAB
# =========================
with tab_pump:
    pm = pump_view.copy()
    if pm.empty:
        st.info("لا يوجد بيانات مضخات ضمن النطاق.")
    else:
        # Top row KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            active = np.random.randint(2, 5)
            total = 4
            kpi_card("Pump Status", f"{active}/{total}", "Active/Total", "ok" if active >= 3 else "warn")
        with k2:
            avg_eff = 100 * pm["efficiency"].tail(12).mean()
            kpi_card("Avg Efficiency", f"{avg_eff:.0f}%", "Below target" if avg_eff < 85 else "On target", "warn" if avg_eff < 85 else "ok")
        with k3:
            t_now = pm["temperature_c"].iloc[-1]
            kpi_card("Temperature", f"{t_now:.0f}°C", "Normal range" if t_now <= 70 else "High", "ok" if t_now <= 70 else "danger")
        with k4:
            v_now = pm["vibration_mm_s"].iloc[-1]
            sev = "Critical" if v_now >= 3.2 else ("Warning" if v_now >= 2.5 else "Low")
            kpi_card("Vibration", f"{v_now:.1f} mm/s", sev, "danger" if sev == "Critical" else ("warn" if sev == "Warning" else "ok"))

        c1, c2, c3 = st.columns(3)
        with c1:
            gauge("Pump Efficiency", float(np.clip(pm["efficiency"].iloc[-1] * 100, 0, 100)))
        with c2:
            gauge("Vibration Level (mm/s)", float(pm["vibration_mm_s"].iloc[-1]), suffix=" mm/s", max_v=6)
        with c3:
            gauge("Temperature (°C)", float(pm["temperature_c"].iloc[-1]), suffix="°C", max_v=80)

        # Trends
        colL, colR = st.columns(2)
        with colL:
            line_chart(pm, "timestamp", "vibration_mm_s", "Vibration Trends", "mm/s")
        with colR:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pm["timestamp"], y=pm["current_a"], name="Current (A)", mode="lines"))
            fig.add_trace(go.Scatter(x=pm["timestamp"], y=pm["power_kw"], name="Power (kW)", mode="lines", yaxis="y2"))
            fig.update_layout(
                height=300,
                paper_bgcolor="#121418",
                plot_bgcolor="#121418",
                font_color="#E6E8EE",
                margin=dict(l=10,r=10,t=40,b=10),
                xaxis=dict(gridcolor="#232832"),
                yaxis=dict(title="Current (A)", gridcolor="#232832"),
                yaxis2=dict(title="Power (kW)", overlaying="y", side="right"),
                title="Current & Power",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("#### Historical Activity & Anomalies")
        dots = pm.copy()
        dots["level"] = pd.cut(
            dots["vibration_mm_s"],
            bins=[-np.inf, 1.5, 2.5, 3.0, np.inf],
            labels=["Normal", "Warning", "High", "Critical"],
        )
        fig = px.scatter(
            dots,
            x="timestamp",
            y="temperature_c",
            color="level",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Vibration (level) vs Temperature",
        )
        fig.update_layout(
            height=320,
            paper_bgcolor="#121418",
            plot_bgcolor="#121418",
            font_color="#E6E8EE",
            margin=dict(l=10,r=10,t=40,b=10),
        )
        fig.update_xaxes(gridcolor="#232832")
        fig.update_yaxes(gridcolor="#232832", title="Temp (°C)")
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("#### Pump System Details")
        # Small demo table
        detail = pd.DataFrame(
            {
                "Pump ID": ["PUMP-001", "PUMP-002", "PUMP-003"],
                "Status": ["Active", "Warning", "Active"],
                "Flow Rate": ["245 L/min", "198 L/min", "210 L/min"],
                "Pressure": ["3.2 bar", "2.8 bar", "2.9 bar"],
                "Power": ["15.8 kW", "12.9 kW", "13.4 kW"],
                "Last Maintenance": ["Oct 15, 2024", "Sep 28, 2024", "Aug 07, 2024"],
            }
        )
        st.dataframe(detail, use_container_width=True, hide_index=True)

# =========================
# LIGHTING TAB
# =========================
with tab_light:
    lv = saving_view.copy()
    if lv.empty:
        st.info("لا يوجد بيانات إنارة ضمن النطاق.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Active Zones", f"{np.random.randint(18, 32)}", "of 36 zones", "ok")
        with c2:
            kpi_card("Current Usage", f"{int(lv['optimized_kwh'].iloc[-1]):,} kWh", "kWh today", "ok")
        with c3:
            kpi_card("Avg Lux Level", f"{np.random.randint(420, 520)}", "lux", "ok")
        with c4:
            perc = 100 * (lv["saved_kwh"].sum() / lv["baseline_kwh"].sum())
            kpi_card("Energy Saved", f"{perc:.1f}%", "vs baseline", "ok")

        col1, col2, col3 = st.columns([0.42, 0.29, 0.29])
        with col1:
            # Energy vs Baseline
            df_bar = pd.DataFrame(
                {
                    "timestamp": lv["timestamp"],
                    "Baseline": lv["baseline_kwh"],
                    "Actual": lv["optimized_kwh"],
                }
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_bar["timestamp"], y=df_bar["Baseline"], name="Baseline"))
            fig.add_trace(go.Bar(x=df_bar["timestamp"], y=df_bar["Actual"], name="Actual Usage"))
            fig.update_layout(
                barmode="group",
                title="Energy Usage vs. Baseline",
                height=300,
                paper_bgcolor="#121418",
                plot_bgcolor="#121418",
                font_color="#E6E8EE",
                margin=dict(l=10,r=10,t=40,b=10),
            )
            fig.update_xaxes(gridcolor="#232832")
            fig.update_yaxes(gridcolor="#232832", title="kWh")
            st.plotly_chart(fig, use_container_width=True, theme=None)
        with col2:
            gauge("Lighting Efficiency", float(np.clip(lv["efficiency"].iloc[-1] * 100, 0, 100)))
        with col3:
            # Lux vs Occupancy (synthetic)
            hrs = np.arange(6, 22, 1)
            lux = np.clip(120 + 380 * np.sin((hrs-6)/16*np.pi), 80, 600)
            occ = np.clip(5 + 95 * np.sin((hrs-6)/16*np.pi), 0, 100)
            df_lux = pd.DataFrame({"hour": hrs, "Lux Level": lux, "Occupancy (%)": occ})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_lux["hour"], y=df_lux["Lux Level"], name="Lux Level", mode="lines"))
            fig.add_trace(go.Scatter(x=df_lux["hour"], y=df_lux["Occupancy (%)"], name="Occupancy", mode="lines", yaxis="y2"))
            fig.update_layout(
                height=300,
                paper_bgcolor="#121418",
                plot_bgcolor="#121418",
                font_color="#E6E8EE",
                margin=dict(l=10,r=10,t=40,b=10),
                xaxis=dict(title="Hour", gridcolor="#232832"),
                yaxis=dict(title="Lux Level", gridcolor="#232832"),
                yaxis2=dict(title="Occupancy (%)", overlaying="y", side="right"),
                title="Lux vs Occupancy",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("#### Zone Status Overview")
        zones = pd.DataFrame(
            {
                "Zone": [f"Zone {c}" for c in ["A1","A2","B1","B2","C1","C2"]],
                "State": ["Active","Dimmed","Active","Fault","Active","Dimmed"],
                "Lux": [450,180,520,0,390,210],
            }
        )
        st.dataframe(zones, use_container_width=True, hide_index=True)

        st.markdown("#### Daily Lighting Trends")
        zoneA = 10 + 50*np.sin(np.linspace(0, np.pi, 10)) + 30
        zoneB = 8 + 44*np.sin(np.linspace(0, np.pi, 10)) + 25
        zoneC = 6 + 36*np.sin(np.linspace(0, np.pi, 10)) + 20
        hours = np.linspace(8, 22, 10)
        df_tr = pd.DataFrame({"hour": hours, "Zone A": zoneA, "Zone B": zoneB, "Zone C": zoneC})
        line_chart(df_tr, "hour", ["Zone A", "Zone B", "Zone C"], "Daily Lighting Trends", "Energy Usage (kWh)")

# =========================
# AGENT INSIGHTS TAB
# =========================
with tab_agent:
    ag = agent_view.copy()
    if ag.empty:
        st.info("لا يوجد سجلات للوكيل ضمن النطاق.")
    else:
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            total_actions = len(ag)
            kpi_card("Total Actions", f"{total_actions:,}", "Last 24 hours", "ok")
        with a2:
            active_anom = (ag["severity"].isin(["Critical", "High"])).sum()
            kpi_card("Active Anomalies", f"{active_anom}", f"{(ag['severity']=='Critical').sum()} critical, {(ag['severity']=='High').sum()} high", "warn" if active_anom else "ok")
        with a3:
            success_rate = np.clip(94.2 + np.random.normal(0, 0.7), 90, 99)
            kpi_card("Success Rate", f"{success_rate:.1f}%", "Resolution rate", "ok")
        with a4:
            avg_resp = np.clip(np.random.normal(2.4, 0.35), 1.6, 4.0)
            kpi_card("Avg Response Time", f"{avg_resp:.1f}s", "Detection to action", "ok")

        colL, colR = st.columns(2)
        with colL:
            sev_counts = ag["severity"].value_counts().reindex(["Critical","High","Medium","Low"]).fillna(0).reset_index()
            sev_counts.columns = ["Severity", "Count"]
            bar_chart(sev_counts, "Severity", "Count", "Anomaly Severity Distribution")
        with colR:
            # Interventions timeline by agent
            ag2 = ag.copy()
            ag2["hour"] = ag2["timestamp"].dt.floor("1H")
            series = ag2.groupby(["hour", "agent"]).size().reset_index(name="Interventions")
            fig = px.line(series, x="hour", y="Interventions", color="agent", title="Agent Interventions Timeline")
            fig.update_layout(
                height=300,
                paper_bgcolor="#121418",
                plot_bgcolor="#121418",
                font_color="#E6E8EE",
                margin=dict(l=10,r=10,t=40,b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_xaxes(gridcolor="#232832")
            fig.update_yaxes(gridcolor="#232832")
            st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("#### Agent Event Logs")
        show_cols = ["timestamp","agent","action","target","severity","decision","status"]
        if not set(show_cols).issubset(ag.columns):
            ag = fake_agent_log(80)
        st.dataframe(
            ag[show_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# -------------------------
# Footer note
# -------------------------
st.markdown("<div class='muted' style='text-align:center; margin-top:18px'>Built for demo purposes • Thresholds (Z-score / Slope) affect anomaly sensitivity conceptually</div>", unsafe_allow_html=True)
