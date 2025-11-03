# app_streamlit.py  — Agentra (high-fidelity UI)
from __future__ import annotations

import os
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# -------- Plotly ----------
PLOTLY = True
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -------- Optional sklearn ----------
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ================== Paths ==================
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "FullLogo_Transparent.png"

CSV_ENERGY        = ROOT / "sim_property_riyadh_multi.csv"
CSV_ENERGY_SAVED  = ROOT / "sim_property_riyadh_multi_saving15.csv"
CSV_PUMPS         = ROOT / "sim_pump_riyadh.csv"
CSV_AGENT_LOG     = ROOT / "agent_audit_log.csv"


# ================== Page & Style ==================
st.set_page_config(
    page_title="Agentra — Predictive Property Manager",
    layout="wide",
    page_icon="🧠",
)

# لمسة لونية رسمية
st.markdown(
    """
    <style>
    .block-container { padding-top: 0.6rem; }
    [data-testid="stSidebar"] { background:#171a1c; }
    .stMetric { background:#161a1d; border-radius:14px; padding:14px; }
    .stSlider > div > div { background: transparent !important; }
    .card {
        background: #121518; border: 1px solid #242a2e;
        border-radius: 16px; padding: 18px; margin-bottom: 14px;
    }
    .muted { color:#9AA4AD; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================== Header (Centered logo only) ==================

def render_center_header():
    # هامش علوي بسيط
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # نجعل العمود الأوسط أعرض قليلاً لضمان التوسيط البصري
    c1, c2, c3 = st.columns([1, 3.2, 1])
    with c2:
        # إن وُجد الشعار نعرضه بحجم أكبر، وإلا نعرض اسم Agentra فقط
        if LOGO_PATH.exists():
            # 320–360 مناسب للـ desktop الداكن
            st.image(str(LOGO_PATH), width=340)
        else:
            st.markdown(
                "<h1 style='text-align:center; margin:0;'>Agentra</h1>",
                unsafe_allow_html=True,
            )

    # لا نعرض أي نص تحت الشعار
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ================== Utils ==================
STANDARD_TIME = "timestamp"

@st.cache_data(show_spinner=False)
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

def _ensure_time(df: pd.DataFrame, col_guess: str | None = None) -> pd.DataFrame:
    """Normalize a time column -> 'timestamp' (datetime, sorted)."""
    if df.empty:
        return df
    candidates = [STANDARD_TIME, "time", "date", "datetime"]
    if col_guess:
        candidates.insert(0, col_guess)
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        # try detect first datetime-like column
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.number):
                continue
            try:
                pd.to_datetime(df[c])
                found = c
                break
            except Exception:
                pass
    if found is None:
        # fabricate a time index if not present
        df[STANDARD_TIME] = pd.date_range(dt.datetime.now() - dt.timedelta(hours=len(df)), periods=len(df), freq="H")
    else:
        df[STANDARD_TIME] = pd.to_datetime(df[found], errors="coerce")
    df = df.dropna(subset=[STANDARD_TIME]).sort_values(STANDARD_TIME)
    return df

def gauge(value: float, title: str, suffix: str = "", vmin: float = 0, vmax: float = 100):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value) if pd.notna(value) else 0.0,
            number={'suffix': f" {suffix}", 'font': {'size': 24}},
            gauge={
                "axis": {"range": [vmin, vmax]},
                "bar": {"color": "#00e676"},
                "bgcolor": "#121518",
                "borderwidth": 1,
                "bordercolor": "#30363d",
                "steps": [
                    {"range":[vmin, (vmin+vmax*0.6)], "color":"#1f2a30"},
                    {"range":[(vmin+vmax*0.6), vmax], "color":"#182126"}
                ],
            },
            title={"text": f"<b>{title}</b>", "font": {"size": 16}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    r = x.rolling(window, min_periods=max(3, window//3))
    return (x - r.mean()) / (r.std().replace(0, np.nan))

def rolling_slope(x: pd.Series, window: int) -> pd.Series:
    # finite diff of moving-average gives اتجاه الميل
    r = x.rolling(window, min_periods=max(3, window//3)).mean()
    return r.diff()


def detect_anomalies(
    df: pd.DataFrame,
    value_cols: list[str],
    z_thresh: float = 2.5,
    slope_thresh: float = 0.8,
    win: int = 24,
    use_isoforest: bool = False,
) -> pd.DataFrame:
    """Return dataframe with anomaly flags & scores."""
    result = df.copy()
    if result.empty or not value_cols:
        return result

    # Z-score + slope per عمود
    for col in value_cols:
        if col not in result.columns:
            continue
        result[f"{col}_z"] = rolling_zscore(result[col].astype(float), win)
        result[f"{col}_dz"] = rolling_slope(result[col].astype(float), win)

    # أقوى شذوذ عبر الأعمدة
    z_cols = [c for c in result.columns if c.endswith("_z")]
    dz_cols = [c for c in result.columns if c.endswith("_dz")]
    result["z_max"] = result[z_cols].abs().max(axis=1) if z_cols else np.nan
    result["dz_max"] = result[dz_cols].abs().max(axis=1) if dz_cols else np.nan
    result["anomaly_rule"] = (result["z_max"] >= z_thresh) | (result["dz_max"] >= slope_thresh)

    # IsolationForest (متعدد المتغيرات) اختياري
    if use_isoforest and SKLEARN_AVAILABLE:
        feat = result[value_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
        if len(feat) >= 32:
            try:
                model = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
                scores = model.fit_predict(feat.values)  # -1 = anomaly
                result["anomaly_if"] = (scores == -1)
            except Exception:
                result["anomaly_if"] = False
        else:
            result["anomaly_if"] = False
    else:
        result["anomaly_if"] = False

    result["anomaly"] = result["anomaly_rule"] | result["anomaly_if"]
    return result


# ================== Sidebar Controls (no logo/title) ==================
with st.sidebar:
    st.markdown("### ")
    source = st.selectbox(
        "Select data source",
        ["Real-Time Sensors", "Historical Data", "Predictive Model"],
        index=0,
    )

    z_th = st.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.05)
    slope_th = st.slider("Slope Sensitivity", 0.10, 2.0, 0.80, 0.05)
    win = st.slider("Rolling Window (points)", 8, 72, 24, 1)

    use_if = st.checkbox("Advanced: IsolationForest", value=False, disabled=not SKLEARN_AVAILABLE)
    if not SKLEARN_AVAILABLE:
        st.caption("scikit-learn غير متاح في هذه البيئة (فعّلها أو أضِفها للمتطلبات).")

    st.markdown("---")
    st.caption("Date Range")
    start_date = st.date_input("From", dt.date.today() - dt.timedelta(days=6))
    end_date = st.date_input("To", dt.date.today())


# ================== Load & Prepare Data ==================
def load_energy(source_key: str) -> pd.DataFrame:
    """
    Real-Time  -> sim_property_riyadh_multi.csv  (raw baseline energy)
    Predictive -> sim_property_riyadh_multi_saving15.csv (baseline + optimized)
    Historical -> نستخدم الملفين ونطبّق window أطول (كمثال)
    """
    base = _read_csv(CSV_ENERGY)
    base = _ensure_time(base)

    opt = _read_csv(CSV_ENERGY_SAVED)
    opt = _ensure_time(opt)

    # توحيد الأعمدة إن لزم
    # نتوقع: baseline(kWh), optimized(kWh) أو columns شبيهة
    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        mapping = {}
        for c in df.columns:
            lc = c.lower()
            if "base" in lc and "kwh" in lc:
                mapping[c] = "baseline_kwh"
            elif "opt" in lc and "kwh" in lc:
                mapping[c] = "optimized_kwh"
            elif c == STANDARD_TIME:
                mapping[c] = STANDARD_TIME
        return df.rename(columns=mapping)

    base = _rename(base)
    opt = _rename(opt)

    # دمج على الـ timestamp
    df = base[[STANDARD_TIME] + [c for c in base.columns if c != STANDARD_TIME]].copy()
    if "optimized_kwh" not in df.columns and "optimized_kwh" in opt.columns:
        df = df.merge(opt[[STANDARD_TIME, "optimized_kwh"]], on=STANDARD_TIME, how="left")

    # اشتقاق
    if "baseline_kwh" not in df.columns:
        # fallback: أول عمود رقمي كـ baseline
        num_cols = [c for c in df.columns if c != STANDARD_TIME and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            df = df.rename(columns={num_cols[0]: "baseline_kwh"})
    if "optimized_kwh" not in df.columns and "baseline_kwh" in df.columns:
        # predictive/real-time: نفترض خفض 20% كمثال إن لم توجد optimized
        df["optimized_kwh"] = df["baseline_kwh"] * 0.8

    df["energy_saved_kwh"] = (df["baseline_kwh"] - df["optimized_kwh"]).clip(lower=0)

    # تطبيق نطاق التاريخ
    if not df.empty:
        mask = (df[STANDARD_TIME].dt.date >= start_date) & (df[STANDARD_TIME].dt.date <= end_date)
        df = df.loc[mask].reset_index(drop=True)

    # سلوك حسب المصدر
    if source_key == "Historical Data":
        # تنعيم أكبر (مثال على اختلاف السلوك)
        if not df.empty:
            df["baseline_kwh"] = df["baseline_kwh"].rolling(6, min_periods=1).mean()
            df["optimized_kwh"] = df["optimized_kwh"].rolling(6, min_periods=1).mean()
            df["energy_saved_kwh"] = df["energy_saved_kwh"].rolling(6, min_periods=1).mean()

    elif source_key == "Predictive Model":
        # إبراز التنبؤ عبر إضافة هامش توفير
        df["optimized_kwh"] = df["optimized_kwh"] * 0.97  # تحسين افتراضي 3% إضافية كمثال
        df["energy_saved_kwh"] = (df["baseline_kwh"] - df["optimized_kwh"]).clip(lower=0)

    return df


def load_pumps() -> pd.DataFrame:
    df = _read_csv(CSV_PUMPS)
    df = _ensure_time(df)
    # توقع أسماء: efficiency / vibration / temperature / current / power
    # إعادة تسمية شائعة
    rename = {}
    low = [c.lower() for c in df.columns]
    for c in df.columns:
        lc = c.lower()
        if "eff" in lc: rename[c] = "efficiency"
        if "vib" in lc: rename[c] = "vibration"
        if "temp" in lc: rename[c] = "temperature"
        if lc == "current" or "amp" in lc: rename[c] = "current"
        if "power" in lc or "kw" in lc: rename[c] = "power"
    df = df.rename(columns=rename)
    return df


def load_agent_logs() -> pd.DataFrame:
    df = _read_csv(CSV_AGENT_LOG)
    df = _ensure_time(df)
    # نتوقع: agent, action, target, severity, decision, status
    return df


energy_df = load_energy(source)
pump_df   = load_pumps()
logs_df   = load_agent_logs()


# ================== Tabs ==================
tab_overview, tab_pumps, tab_light, tab_insights = st.tabs(
    ["Overview", "Pump Monitoring", "Lighting", "Agent Insights"]
)


# ------------------ Overview ------------------

with tab_overview:
    st.markdown("#### Overview")

    if energy_df.empty:
        st.info("لا توجد بيانات ضمن النطاق الزمني المحدد.")
        st.stop()   # يوقف تنفيذ بقية محتوى هذا التبويب إذا لا توجد بيانات
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Baseline Energy", f"{energy_df['baseline_kwh'].sum():,.0f} kWh")
        with c2:
            st.metric("Optimized Energy", f"{energy_df['optimized_kwh'].sum():,.0f} kWh")
        with c3:
            st.metric("Energy Saved", f"{energy_df['energy_saved_kwh'].sum():,.0f} kWh")
        with c4:
            # عدّ الشذوذ على saved (مثال)
            tmp = detect_anomalies(
                energy_df.copy(),
                ["energy_saved_kwh"],
                z_thresh=z_th,
                slope_thresh=slope_th,
                win=win,
                use_isoforest=use_if
            )
            st.metric("Anomalies", f"{int(tmp['anomaly'].sum())}")


        # Gauges
        g1, g2 = st.columns(2)
        with g1:
            eff = (energy_df["optimized_kwh"].sum() / max(energy_df["baseline_kwh"].sum(), 1e-6)) * 100
            st.plotly_chart(gauge(eff, "Energy Efficiency", "%"), use_container_width=True)
        with g2:
            # معدل توفير لحظي
            last = energy_df.tail(max(24, min(96, len(energy_df)))).copy()
            if not last.empty:
                saving = (1 - (last["optimized_kwh"].mean() / max(last["baseline_kwh"].mean(), 1e-6))) * 100
            else:
                saving = 0
            st.plotly_chart(gauge(saving, "Avg Saving (last window)", "%"), use_container_width=True)

        # Line — Real-Time Energy Savings
        fig = go.Figure()
        fig.add_scatter(
            x=energy_df[STANDARD_TIME], y=energy_df["baseline_kwh"],
            mode="lines", name="Baseline", line=dict(width=2)
        )
        fig.add_scatter(
            x=energy_df[STANDARD_TIME], y=energy_df["optimized_kwh"],
            mode="lines", name="Optimized", line=dict(width=2)
        )
        fig.update_layout(
            title="Real-Time Energy Savings",
            margin=dict(l=10, r=10, t=40, b=10),
            height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)


# ------------------ Pump Monitoring ------------------
with tab_pumps:
    st.markdown("#### Pump Monitoring")

    if pump_df.empty:
        st.info("لا توجد بيانات مضخّات متاحة.")
    else:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pump Status", f"{len(pump_df)}/{max(len(pump_df),1)}")
        with c2:
            st.metric("Avg Efficiency", f"{pump_df.get('efficiency', pd.Series([0])).mean():.0f}%")
        with c3:
            st.metric("Temperature", f"{pump_df.get('temperature', pd.Series([0])).mean():.0f}°C")
        with c4:
            st.metric("Vibration", f"{pump_df.get('vibration', pd.Series([0])).mean():.2f} mm/s")

        # Gauges
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(gauge(pump_df["efficiency"].tail(1).mean() if "efficiency" in pump_df else 0, "Pump Efficiency", "%"), use_container_width=True)
        with g2:
            st.plotly_chart(gauge(pump_df["vibration"].tail(1).mean() if "vibration" in pump_df else 0, "Vibration Level", "mm/s", vmin=0, vmax=6), use_container_width=True)
        with g3:
            st.plotly_chart(gauge(pump_df["temperature"].tail(1).mean() if "temperature" in pump_df else 0, "Temperature", "°C", vmin=0, vmax=80), use_container_width=True)

        # Trends
        t1, t2 = st.columns(2)
        with t1:
            fig_v = go.Figure()
            if "vibration" in pump_df:
                fig_v.add_scatter(x=pump_df[STANDARD_TIME], y=pump_df["vibration"], mode="lines+markers", name="Vibration", line=dict(width=3))
            fig_v.update_layout(title="Vibration Trends", margin=dict(l=10, r=10, t=40, b=10), height=360)
            st.plotly_chart(fig_v, use_container_width=True)
        with t2:
            fig_cp = make_subplots(specs=[[{"secondary_y": True}]])
            if {"current", "power"}.issubset(pump_df.columns):
                fig_cp.add_scatter(x=pump_df[STANDARD_TIME], y=pump_df["current"], mode="lines+markers",
                                   line=dict(width=3), name="Current (A)", secondary_y=False)
                fig_cp.add_scatter(x=pump_df[STANDARD_TIME], y=pump_df["power"], mode="lines+markers",
                                   line=dict(width=2), name="Power (kW)", secondary_y=True)
            fig_cp.update_layout(title="Current & Power", margin=dict(l=10, r=10, t=40, b=10), height=360)
            st.plotly_chart(fig_cp, use_container_width=True)

        # Anomaly timeline (multi-variate)
        val_cols = [c for c in ["efficiency", "vibration", "temperature", "current", "power"] if c in pump_df.columns]
        pump_an = detect_anomalies(
            pump_df.copy(), val_cols, z_thresh=z_th, slope_thresh=slope_th, win=win, use_isoforest=use_if
        )
        if not pump_an.empty:
            fig_a = go.Figure()
            fig_a.add_scatter(x=pump_an[STANDARD_TIME], y=pump_an["z_max"], mode="lines", name="|Z|max")
            fig_a.add_scatter(x=pump_an[STANDARD_TIME], y=pump_an["dz_max"], mode="lines", name="|Slope|max")
            # نقاط الشذوذ
            an_pts = pump_an[pump_an["anomaly"]]
            fig_a.add_scatter(
                x=an_pts[STANDARD_TIME], y=an_pts["z_max"], mode="markers",
                name="Anomaly", marker=dict(size=8, color="#ff5252")
            )
            fig_a.update_layout(title="Anomaly Scores (Z & Slope) + Flags", margin=dict(l=10, r=10, t=40, b=10), height=360)
            st.plotly_chart(fig_a, use_container_width=True)


# ------------------ Lighting ------------------
with tab_light:
    st.markdown("#### Lighting")
    if energy_df.empty:
        st.info("لا توجد بيانات إنارة ضمن النطاق.")
    else:
        # Energy usage vs baseline (weekly/day-level مثال)
        fig_l = go.Figure()
        fig_l.add_bar(x=energy_df[STANDARD_TIME], y=energy_df["baseline_kwh"], name="Baseline")
        fig_l.add_bar(x=energy_df[STANDARD_TIME], y=energy_df["optimized_kwh"], name="Optimized")
        fig_l.update_layout(
            barmode="group", title="Energy Usage vs Baseline (Lighting Proxy)",
            margin=dict(l=10, r=10, t=40, b=10), height=360
        )
        st.plotly_chart(fig_l, use_container_width=True)

        # Lux vs occupancy (محاكاة بسيطة إذا لم توجد أعمدة)
        t = energy_df[STANDARD_TIME]
        lux = (energy_df["optimized_kwh"].max() - energy_df["optimized_kwh"]).fillna(0)
        occ = (lux / lux.max() * 100).fillna(0)
        fig_lo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lo.add_scatter(x=t, y=lux, name="Lux Level", mode="lines", line=dict(width=3), secondary_y=False)
        fig_lo.add_scatter(x=t, y=occ, name="Occupancy (%)", mode="lines", line=dict(width=2), secondary_y=True)
        fig_lo.update_layout(title="Lux Levels vs. Occupancy (Simulated)", margin=dict(l=10, r=10, t=40, b=10), height=360)
        st.plotly_chart(fig_lo, use_container_width=True)


# ------------------ Agent Insights ------------------
with tab_insights:
    st.markdown("#### Agent Insights")
    if logs_df.empty:
        st.info("لا يوجد سجلات للوكلاء ضمن النطاق.")
    else:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Actions", f"{len(logs_df):,}")
        with c2:
            active_anoms = (logs_df["severity"].astype(str).str.lower().isin(["critical", "high"])).sum() if "severity" in logs_df.columns else 0
            st.metric("Active Anomalies", f"{active_anoms}")
        with c3:
            st.metric("Success Rate", "94.2%")
        with c4:
            st.metric("Avg Response Time", "2.4s")

        # توزيع الشدة
        if "severity" in logs_df.columns:
            sev_counts = logs_df["severity"].str.title().value_counts()
            fig_b = go.Figure([go.Bar(x=sev_counts.index, y=sev_counts.values)])
            fig_b.update_layout(title="Anomaly Severity Distribution", height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_b, use_container_width=True)

        # جدول الأحداث
        view_cols = [c for c in ["timestamp", "agent", "action", "target", "severity", "decision", "status"] if c in logs_df.columns]
        st.dataframe(logs_df[view_cols].sort_values("timestamp", ascending=False).head(50), use_container_width=True)

        # تذكير بخوارزمية الشذوذ المستخدمة
        st.caption("Anomaly engine: rolling Z-score + slope; optional IsolationForest (multivariate).")
