# app_streamlit.py  — Agentra (clean, resilient)
# -------------------------------------------------------------
# • Streamlit + Plotly
# • Robust CSV loading + safe numeric casting
# • Optional synthetic fallback if CSVs are missing
# • Rolling Z-score + slope anomaly detection (with toggles)
# • Centered logo (from assets/FullLogo_Transparent.png)
# -------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import datetime as dt
import math

import numpy as np
import pandas as pd
import streamlit as st

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================== Paths ==============================
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "FullLogo_Transparent.png"

CSV_ENERGY = ROOT / "sim_property_riyadh_multi.csv"
CSV_ENERGY_SAVED = ROOT / "sim_property_riyadh_multi_saving15.csv"  # اختياري
CSV_PUMPS = ROOT / "sim_pump_riyadh.csv"                             # اختياري


# ========================== UI helpers =============================
def render_center_header():
    """Title + centered logo."""
    st.markdown(
        """
        <div style="text-align:center; margin-top:8px;">
            <img src="app://assets/FullLogo_Transparent.png" style="height:84px;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='text-align:center; margin: 4px 0 0 0;'>Agentra</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center; opacity:.8; margin-top:4px;'>Predictive Property Manager</div>",
        unsafe_allow_html=True,
    )


def ensure_logo_served():
    """
    Streamlit لا يقرأ مسارات القرص داخل <img> مباشرة.
    نعرّف مخطط URI بسيط app://assets/... ونحوّله إلى static
    عبر st.logo كاختصار: سنعرضه مرّة (مخفيًا) ليضيفه للـ static.
    """
    if LOGO_PATH.exists():
        try:
            # نعرضه صغيرًا (غير مهم)، فقط ليتوفر داخل static
            st.logo(str(LOGO_PATH), icon_image=str(LOGO_PATH))
        except Exception:
            pass


# ======================== Data Utilities ===========================
def _infer_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    يضمن وجود عمود timestamp باسم "timestamp" ويكون datetime.
    يبحث عن أعمدة محتملة ويعيد تسميتها إن لزم.
    """
    if df.empty:
        return df

    candidates = ["timestamp", "time", "date", "datetime"]
    found = None
    for c in df.columns:
        if c.strip().lower() in candidates:
            found = c
            break
    if found is None:
        # إذا لا يوجد، نولّد واحدًا متسلسلًا
        df = df.copy()
        df["timestamp"] = pd.date_range(
            end=dt.datetime.now(), periods=len(df), freq="H"
        )
    else:
        df = df.rename(columns={found: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # إسقاط الصفوف بدون وقت
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    تحويل آمن للأعمدة إلى قيم عددية مع تحذير إن غاب عمود.
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            # إزالة رموز شائعة تمنع التحويل (%, °C, فواصل…)
            if df[c].dtype == object:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace("°C", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace(" ", "", regex=False)
                )
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            st.warning(f"⚠️ العمود '{c}' غير موجود في البيانات.")
    return df


def _nice_int(x: float | int) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return 0


def _smart_resample(df: pd.DataFrame, rule: str = "1H") -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return df
    return (
        df.set_index("timestamp")
        .resample(rule)
        .mean(numeric_only=True)
        .reset_index()
    )


# ========================= Loaders (CSV) ===========================
def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    return _infer_timestamp(df)


def load_energy() -> pd.DataFrame:
    """
    يحمّل بيانات الطاقة. يتوقع أعمدة:
      baseline_kwh, optimized_kwh  (وسنحسب saved)
    """
    df = _load_csv(CSV_ENERGY)
    if df.empty:
        return df

    # أسماء أعمدة مرنة
    # إن لم توجد الأعمدة القياسية، نحاول إيجاد بدائل
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    if "baseline_kwh" not in df.columns:
        for key in ["baseline", "baseline_kwh", "kwh_baseline"]:
            if key in cols_lower:
                rename_map[cols_lower[key]] = "baseline_kwh"
                break
    if "optimized_kwh" not in df.columns:
        for key in ["optimized", "optimized_kwh", "kwh_optimized"]:
            if key in cols_lower:
                rename_map[cols_lower[key]] = "optimized_kwh"
                break

    df = df.rename(columns=rename_map)
    df = to_numeric_safe(df, ["baseline_kwh", "optimized_kwh"])
    if "energy_saved_kwh" not in df.columns:
        if "baseline_kwh" in df.columns and "optimized_kwh" in df.columns:
            df["energy_saved_kwh"] = df["baseline_kwh"] - df["optimized_kwh"]
        else:
            df["energy_saved_kwh"] = np.nan

    return df


def load_pumps() -> pd.DataFrame:
    """
    يحمّل بيانات المضخات. يتوقع أعمدة:
      efficiency, vibration_mm_s, temp_c, current_a, power_kw
    """
    df = _load_csv(CSV_PUMPS)
    if df.empty:
        return df
    wanted = ["efficiency", "vibration_mm_s", "temp_c", "current_a", "power_kw"]
    df = to_numeric_safe(df, wanted)
    return df


# ================ Synthetic fallback (optional) ====================
def synthetic_energy(n_hours: int = 72) -> pd.DataFrame:
    idx = pd.date_range(end=dt.datetime.now(), periods=n_hours, freq="H")
    base = 400 + 50 * np.sin(np.linspace(0, 4 * math.pi, n_hours)) + np.random.normal(0, 8, n_hours)
    opt = base * (0.8 + 0.03 * np.sin(np.linspace(0, 2 * math.pi, n_hours))) - 5
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "baseline_kwh": np.clip(base, 150, None),
            "optimized_kwh": np.clip(opt, 120, None),
        }
    )
    df["energy_saved_kwh"] = df["baseline_kwh"] - df["optimized_kwh"]
    return df


def synthetic_pumps(n_hours: int = 72) -> pd.DataFrame:
    idx = pd.date_range(end=dt.datetime.now(), periods=n_hours, freq="H")
    # موجات هادئة + قليلاً من الضوضاء
    vib = 1.2 + 0.4 * np.sin(np.linspace(0, 3 * math.pi, n_hours)) + np.random.normal(0, 0.05, n_hours)
    temp = 37 + 2.5 * np.sin(np.linspace(0, 2 * math.pi, n_hours + 10)[:n_hours]) + np.random.normal(0, 0.2, n_hours)
    eff = 74 + 5 * np.cos(np.linspace(0, 2 * math.pi, n_hours)) + np.random.normal(0, 0.4, n_hours)
    cur = 4 + 0.3 * np.sin(np.linspace(0, 2 * math.pi, n_hours))
    pwr = 2 + 0.25 * np.sin(np.linspace(0, 2 * math.pi, n_hours))

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "efficiency": eff,
            "vibration_mm_s": vib,
            "temp_c": temp,
            "current_a": cur,
            "power_kw": pwr,
        }
    )
    return df


# ==================== Anomaly detection ============================
def detect_anomalies(
    df: pd.DataFrame,
    cols: list[str],
    z_thresh: float = 2.5,
    slope_thresh: float = 0.8,
    win: int = 24,
) -> pd.DataFrame:
    """
    كشف شذوذ مبسّط:
    • rolling mean/std => Z-score
    • slope (التغير) على نافذة متحركة
    """
    if df.empty:
        return df

    df = df.copy()
    if "timestamp" not in df.columns:
        return df

    df = df.sort_values("timestamp")
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")

        roll_mean = s.rolling(win, min_periods=max(3, win // 4)).mean()
        roll_std = s.rolling(win, min_periods=max(3, win // 4)).std()
        z = (s - roll_mean) / (roll_std.replace(0, np.nan))

        # slope: فرق القيمة الحالية عن قيمة قبل win//4 (تقريبًا)
        step = max(1, win // 4)
        slope = s.diff(step) / step

        df[f"{c}_z"] = z
        df[f"{c}_slope"] = slope

        # شذوذ لكل عمود على حدة
        df[f"{c}_anomaly"] = (z.abs() > z_thresh) | (slope.abs() > slope_thresh)

    # ملخّص شذوذ شامل (OR على كل الأعمدة)
    anomaly_cols = [f"{c}_anomaly" for c in cols if f"{c}_anomaly" in df.columns]
    if anomaly_cols:
        df["anomaly"] = df[anomaly_cols].any(axis=1)
    else:
        df["anomaly"] = False

    return df


# ========================= Plot helpers ============================
def gauge(title: str, value: float, suffix: str = "", vmin: float = 0, vmax: float = 100):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": f" {suffix}".strip()},
            title={"text": title},
            gauge={"axis": {"range": [vmin, vmax]}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def line_xy(df: pd.DataFrame, x: str, y: list[str], names: list[str] | None = None, title: str = ""):
    fig = go.Figure()
    for i, col in enumerate(y):
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df[x], y=df[col],
                mode="lines",
                name=(names[i] if names and i < len(names) else col),
                line=dict(width=2.2),
            )
        )
    fig.update_layout(
        height=320,
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time",
        yaxis_title="",
    )
    return fig


# ============================== App ================================
st.set_page_config(page_title="Agentra — Predictive Property Manager", layout="wide")
ensure_logo_served()
render_center_header()

# ---------------- Sidebar controls ----------------
st.sidebar.markdown("### Select data source")
data_source = st.sidebar.selectbox(
    "Select data source",
    ["Real-Time Sensors", "Historical Data", "Predictive Model"],
    index=0,
    label_visibility="collapsed",
)

z_th = st.sidebar.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.1)
slope_th = st.sidebar.slider("Slope Sensitivity", 0.1, 2.0, 0.8, 0.05)
win = st.sidebar.slider("Rolling Window (points)", 8, 72, 24, 1)

st.sidebar.markdown("---")
st.sidebar.checkbox("Advanced: IsolationForest", value=False, disabled=True)
st.sidebar.caption("scikit-learn غير مُتاح في هذا التشغيل. فعّليه لاحقًا بإضافته إلى المتطلبات.")

# نطاق التاريخ (اختياري)
st.sidebar.markdown("---")
st.sidebar.markdown("#### Date Range")
d_from = st.sidebar.date_input("From", value=dt.date.today() - dt.timedelta(days=3))
d_to = st.sidebar.date_input("To", value=dt.date.today())

date_min = dt.datetime.combine(d_from, dt.time.min)
date_max = dt.datetime.combine(d_to, dt.time.max)

# ---------------- Load data according to source ----------------
energy_df = load_energy()
pumps_df = load_pumps()

if energy_df.empty:
    energy_df = synthetic_energy(96)
if pumps_df.empty:
    pumps_df = synthetic_pumps(96)

# قصّ حسب التاريخ
energy_df = energy_df[(energy_df["timestamp"] >= date_min) & (energy_df["timestamp"] <= date_max)].reset_index(drop=True)
pumps_df = pumps_df[(pumps_df["timestamp"] >= date_min) & (pumps_df["timestamp"] <= date_max)].reset_index(drop=True)

# مصدر البيانات (للتوضيح فقط—حالياً نعرض نفس الداتا مع تسمية مختلفة)
source_label_map = {
    "Real-Time Sensors": "Real-Time feed (simulated)",
    "Historical Data": "Historical CSV",
    "Predictive Model": "What-if / Predictive",
}
st.caption(f"**Data Source:** {source_label_map.get(data_source, data_source)}")

# ---------------- Tabs ----------------
tab_overview, tab_pump, tab_light, tab_agent = st.tabs(["Overview", "Pump Monitoring", "Lighting", "Agent Insights"])

# ------------------ Overview ------------------
with tab_overview:
    st.markdown("#### Overview")

    if energy_df.empty:
        st.info("لا توجد بيانات ضمن النطاق الزمني المحدد.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Baseline Energy", f"{_nice_int(energy_df['baseline_kwh'].sum()):,} kWh")
        with c2:
            st.metric("Optimized Energy", f"{_nice_int(energy_df['optimized_kwh'].sum()):,} kWh")
        with c3:
            st.metric("Energy Saved", f"{_nice_int(energy_df['energy_saved_kwh'].sum()):,} kWh")
        with c4:
            tmp = detect_anomalies(
                energy_df.copy(),
                ["energy_saved_kwh"],
                z_thresh=z_th, slope_thresh=slope_th, win=win
            )
            st.metric("Anomalies", f"{int(tmp['anomaly'].sum())}")

        # رسومات: كفاءة (Gauge) + وفورات لحظية
        c1, c2 = st.columns(2)
        with c1:
            eff_now = 100.0 * (
                1.0 - energy_df["optimized_kwh"].sum() / max(energy_df["baseline_kwh"].sum(), 1e-6)
            )
            st.plotly_chart(gauge("Energy Efficiency", max(0.0, eff_now), suffix="%", vmin=0, vmax=100), use_container_width=True)

        with c2:
            show_df = energy_df.copy()
            show_df = _smart_resample(show_df, "1H")
            fig = line_xy(
                show_df,
                "timestamp",
                ["baseline_kwh", "optimized_kwh"],
                names=["Baseline", "Optimized"],
                title="Real-Time Energy (kWh)",
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Pump Monitoring ----------------
with tab_pump:
    st.markdown("#### Pump Monitoring")

    if pumps_df.empty:
        st.info("لا توجد بيانات مضخات ضمن النطاق الزمني المحدد.")
    else:
        # مقاييس عامة
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pump Status", f"{len(pumps_df):,}/{len(pumps_df):,}")
        with c2:
            st.metric("Avg Efficiency", f"{pumps_df['efficiency'].mean():.1f}%")
        with c3:
            st.metric("Temperature", f"{pumps_df['temp_c'].mean():.1f}°C")
        with c4:
            st.metric("Vibration", f"{pumps_df['vibration_mm_s'].mean():.2f} mm/s")

        # جداول العدّادات
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(gauge("Efficiency", max(0.0, float(pumps_df["efficiency"].tail(1).mean())), suffix="%", vmin=0, vmax=100), use_container_width=True)
        with c2:
            st.plotly_chart(gauge("Vibration Level", float(pumps_df["vibration_mm_s"].tail(1).mean()), suffix="mm/s", vmin=0, vmax=6), use_container_width=True)
        with c3:
            st.plotly_chart(gauge("Temperature", float(pumps_df["temp_c"].tail(1).mean()), suffix="°C", vmin=0, vmax=80), use_container_width=True)

        # كشف الشذوذ
        cols_for_anom = ["vibration_mm_s", "temp_c", "efficiency"]
        pump_an = detect_anomalies(pumps_df.copy(), cols_for_anom, z_thresh=z_th, slope_thresh=slope_th, win=win)

        st.markdown("##### Trends")
        c1, c2 = st.columns(2)
        with c1:
            fig = line_xy(pump_an, "timestamp", ["vibration_mm_s"], names=["Vibration"], title="Vibration Trends (mm/s)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = line_xy(pump_an, "timestamp", ["current_a", "power_kw"], names=["Current (A)", "Power (kW)"], title="Current vs Power")
            st.plotly_chart(fig, use_container_width=True)

        # عدد الشذوذ وزمنها
        n_an = int(pump_an["anomaly"].sum())
        st.info(f"Anomalies detected: **{n_an}** (Z>{z_th} أو slope>{slope_th})")

# ---------------- Lighting (placeholder, متوافق مع التصميم) -----
with tab_light:
    st.markdown("#### Lighting")
    st.write("نموذج إرشادي لعرض كفاءة الإضاءة واستهلاكها. (يمكن ربطه لاحقًا بملفات CSV الخاصة بالإضاءة).")
    # مثال رسومي باستخدام طاقة baseline/optimized كمُماثل:
    show_df = energy_df.copy()
    show_df = _smart_resample(show_df, "1H")
    fig = line_xy(show_df, "timestamp", ["baseline_kwh", "optimized_kwh"], names=["Baseline", "Optimized"], title="Energy Usage vs Baseline")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Agent Insights (placeholder) ---------------------
with tab_agent:
    st.markdown("#### Agent Insights")
    st.write("سجلات/إحصائيات الوكلاء (HVAC/Lighting/Pump). أضف مصدرك لاحقًا (CSV أو API) لعرض الأحداث الحقيقية.")
    # جدول بسيط تلخيصي
    demo = pd.DataFrame(
        {
            "Agent": ["HVAC-001", "LIGHT-002", "PUMP-003"],
            "Last Action": ["Temp -3°C", "Dim to 75%", "Vibration Alert"],
            "Severity": ["Critical", "Medium", "High"],
            "Status": ["Completed", "Completed", "Pending"],
            "Timestamp": [dt.datetime.now() - dt.timedelta(hours=h) for h in (1, 2, 3)],
        }
    )
    st.dataframe(demo, use_container_width=True)
