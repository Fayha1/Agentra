# app_streamlit.py  (Agentra — high-fidelity UI)
from __future__ import annotations

from pathlib import Path
import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Plotly (لرسوم عالية الجودة) ----------
PLOTLY = True
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    PLOTLY = False

# ---------- scikit-learn (اختياري لعزل الشذوذ) ----------
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ================== ثوابت ومسارات ==================
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "FullLogo_Transparent.png"

CSV_ENERGY_CANDIDATES = [
    ROOT / "sim_property_riyadh_multi.csv",
    ROOT / "sim_property_riyadh_multi_saving15.csv",
]
CSV_PUMPS = ROOT / "sim_pump_riyadh.csv"
CSV_AGENT_LOG = ROOT / "agent_audit_log.csv"

# ======== إعداد الصفحة والثيم الداكن الهادئ ========
st.set_page_config(page_title="Agentra — Predictive Property Manager", layout="wide")
st.markdown(
    """
    <style>
      /* هوامش ألطف وتكبير خفيف للعناوين */
      .block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
      h1, h2, h3, h4 {letter-spacing: .3px}
      /* شرائط جانبية أغمق قليلاً */
      section[data-testid="stSidebar"] {background: #25262b;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== دوال مساعدة ==================
def _find_first(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def _ensure_timestamp(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    return df.sort_values(col)

def _smart_resample(df: pd.DataFrame, ts: str, rule: str = "1H") -> pd.DataFrame:
    if ts not in df.columns or df.empty:
        return df
    out = (
        df.set_index(ts)
          .sort_index()
          .resample(rule)
          .mean(numeric_only=True)
          .reset_index()
    )
    return out

def _safe_sum(s: pd.Series) -> float:
    try:
        return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())
    except Exception:
        return 0.0

def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_energy() -> pd.DataFrame:
    """يحاول تحميل بيانات الطاقة من أول ملف متاح، ويُطوّع الأعمدة."""
    p = _find_first(CSV_ENERGY_CANDIDATES)
    if p is None:
        return pd.DataFrame()

    df = pd.read_csv(p)
    # محاولات تلقائية لاكتشاف أسماء الأعمدة
    # توقعات: timestamp, baseline_kwh, optimized_kwh
    # إن لم توجد optimized_kwh سنولدها تقريبيًا لتعمل الواجهة
    ts_candidates = ["timestamp", "time", "date", "datetime"]
    ts = next((c for c in ts_candidates if c in df.columns), None)
    if ts is None:
        # لو لم نجد طابع زمني سنفترض عمود أول كفهرس زمني
        df["timestamp"] = pd.date_range("2025-10-28", periods=len(df), freq="H")
        ts = "timestamp"

    df = _ensure_timestamp(df, ts)
    # توحيد الأسماء
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if "baseline" in lc and "kwh" in lc:
            rename_map[c] = "baseline_kwh"
        if ("opt" in lc or "optimized" in lc) and "kwh" in lc:
            rename_map[c] = "optimized_kwh"
        if ("saved" in lc or "saving" in lc) and "kwh" in lc:
            rename_map[c] = "energy_saved_kwh"
    df = df.rename(columns=rename_map)

    # توليد optimized إن لم يوجد (خفض 20% مثلا) فقط لتعمل الواجهة
    if "optimized_kwh" not in df.columns and "baseline_kwh" in df.columns:
        df["optimized_kwh"] = df["baseline_kwh"] * 0.8

    # توليد saved
    if "energy_saved_kwh" not in df.columns and \
       {"baseline_kwh", "optimized_kwh"}.issubset(df.columns):
        df["energy_saved_kwh"] = df["baseline_kwh"] - df["optimized_kwh"]

    df = _numeric(df, ["baseline_kwh", "optimized_kwh", "energy_saved_kwh"])
    return _smart_resample(df, ts), ts

def load_pumps() -> pd.DataFrame:
    if not CSV_PUMPS.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(CSV_PUMPS)
    ts_candidates = ["timestamp", "time", "date", "datetime"]
    ts = next((c for c in ts_candidates if c in df.columns), None)
    if ts is None:
        df["timestamp"] = pd.date_range("2025-10-28", periods=len(df), freq="H")
        ts = "timestamp"
    df = _ensure_timestamp(df, ts)
    # الأعمدة المتوقعة
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if "eff" in lc: rename_map[c] = "efficiency"
        if "vib" in lc: rename_map[c] = "vibration_mm_s"
        if ("temp" in lc) or ("°c" in lc): rename_map[c] = "temp_c"
        if "current" in lc or lc == "a": rename_map[c] = "current_a"
        if "power" in lc or lc == "kw": rename_map[c] = "power_kw"
    df = df.rename(columns=rename_map)
    df = _numeric(df, ["efficiency", "vibration_mm_s", "temp_c", "current_a", "power_kw"])
    return _smart_resample(df, ts), ts

def load_agent_log() -> pd.DataFrame:
    if not CSV_AGENT_LOG.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(CSV_AGENT_LOG)
    ts_candidates = ["timestamp", "time", "date", "datetime"]
    ts = next((c for c in ts_candidates if c in df.columns), None)
    if ts is None:
        df["timestamp"] = pd.date_range("2025-10-28", periods=len(df), freq="H")
        ts = "timestamp"
    df = _ensure_timestamp(df, ts)
    return df, ts

def detect_anomalies(
    df: pd.DataFrame,
    value_cols: list[str],
    z_thresh: float = 2.5,
    slope_thresh: float = 0.8,
    win: int = 24,
    use_isoforest: bool = False,
) -> pd.DataFrame:
    """محرك كشف الشذوذ: Z-score متحرك + ميل قياسي، واختياري IsolationForest متعدد المتغيرات."""
    if df.empty or not value_cols:
        df["anomaly"] = 0
        return df

    work = df.copy()
    # نضمن الأعداديات
    work = _numeric(work, value_cols)

    # Z-score متحرك
    z_flags = []
    for c in value_cols:
        roll = work[c].rolling(win, min_periods=max(3, win//3))
        mu = roll.mean()
        sd = roll.std(ddof=0).replace(0, np.nan)
        z = (work[c] - mu) / sd
        z_flag = (np.abs(z) >= z_thresh).astype(int).fillna(0)
        z_flags.append(z_flag)

        # ميل نسبي (مُطبّع على الانحراف المعياري)
        grad = work[c].diff()
        grad_sd = pd.Series(grad).rolling(win, min_periods=max(3, win//3)).std(ddof=0)
        slope_norm = np.abs(grad) / (grad_sd.replace(0, np.nan))
        slope_flag = (slope_norm >= slope_thresh).astype(int).fillna(0)

        # تجميع بعلم مؤقت
        if "anomaly_tmp" not in work:
            work["anomaly_tmp"] = 0
        work["anomaly_tmp"] = np.maximum(work["anomaly_tmp"], np.maximum(z_flag, slope_flag))

    # IsolationForest (اختياري)
    if use_isoforest and SKLEARN_AVAILABLE and len(value_cols) >= 1:
        X = work[value_cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        try:
            iso = IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=42,
            )
            iso.fit(X)
            pred = iso.predict(X)  # -1 شاذ
            iso_flag = (pred == -1).astype(int)
            work["anomaly"] = np.maximum(work["anomaly_tmp"], iso_flag)
        except Exception:
            work["anomaly"] = work["anomaly_tmp"]
    else:
        work["anomaly"] = work["anomaly_tmp"]

    return work.drop(columns=[c for c in ["anomaly_tmp"] if c in work.columns])

def gauge(title: str, value: float, suffix: str = "", min_v=0, max_v=100) -> "go.Figure|None":
    if not PLOTLY:
        return None
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={"text": title},
            number={"suffix": suffix},
            gauge={
                "axis": {"range": [min_v, max_v]},
                "bar": {"thickness": 0.25},
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=0))
    return fig

def _plotly_layout(fig, title=""):
    if not PLOTLY:
        return fig
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig

def render_center_header():
    # يعرض اللوغو في المنتصف بحجم أكبر
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=220)
        st.markdown(
            "<h1 style='text-align:center; margin-top: 0.25rem;'>Agentra</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align:center; opacity:.85;'>Predictive Property Manager</div>",
            unsafe_allow_html=True,
        )

# ================== رأس الصفحة ==================
render_center_header()

# ================== الشريط الجانبي ==================
st.sidebar.header(" ")
st.sidebar.write("Select data source")

data_src = st.sidebar.selectbox(
    "Select data source",
    ["Real-Time Sensors", "Historical Data", "Predictive Model"],
    index=0,
)

z_th = float(st.sidebar.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.01))
slope_th = float(st.sidebar.slider("Slope Sensitivity", 0.10, 2.0, 0.80, 0.01))
win = int(st.sidebar.slider("Rolling Window (points)", 8, 72, 24, 1))

use_if = st.sidebar.checkbox("Advanced: IsolationForest", value=False, disabled=not SKLEARN_AVAILABLE)
if use_if and not SKLEARN_AVAILABLE:
    st.sidebar.warning("scikit-learn غير متاح في هذه البيئة. عطِّله أو أضِفه إلى المتطلبات.")

# تحميل الداتا
energy_df, energy_ts = load_energy()
pump_df, pump_ts = load_pumps()
agent_df, agent_ts = load_agent_log()

# نطاق التاريخ
def _bounds(ts_col: str | None, df: pd.DataFrame):
    if not ts_col or df.empty:
        return (dt.date(2025, 10, 28), dt.date(2025, 11, 3))
    mn = df[ts_col].min().date()
    mx = df[ts_col].max().date()
    return (mn, mx)

from_d, to_d = _bounds(energy_ts, energy_df)
sd_from = st.sidebar.date_input("From", from_d)
sd_to = st.sidebar.date_input("To", to_d)

# فلترة بالتاريخ
def _filter_by_date(df: pd.DataFrame, ts_col: str | None) -> pd.DataFrame:
    if df.empty or not ts_col:
        return df
    low = pd.to_datetime(dt.datetime.combine(sd_from, dt.time.min))
    high = pd.to_datetime(dt.datetime.combine(sd_to, dt.time.max))
    return df[(df[ts_col] >= low) & (df[ts_col] <= high)].copy()

energy_df = _filter_by_date(energy_df, energy_ts)
pump_df = _filter_by_date(pump_df, pump_ts)
agent_df = _filter_by_date(agent_df, agent_ts)

# تبديل مصدر البيانات (رمزي — يغير فقط وسم/رسالة)
src_badge = {
    "Real-Time Sensors": "حالي (Streaming)",
    "Historical Data": "تاريخي (من ملفات)",
    "Predictive Model": "نموذج تنبؤي (Synthetic)",
}[data_src]
st.caption(f"**Data Source:** {src_badge}")

# ================== التبويبات ==================
tab_overview, tab_pumps, tab_light, tab_agents = st.tabs(
    ["Overview", "Pump Monitoring", "Lighting", "Agent Insights"]
)

# ------------------ Overview ------------------
with tab_overview:
    st.markdown("#### Overview")

    if energy_df.empty:
        st.info("لا توجد بيانات ضمن النطاق الزمني المحدد.")
        st.stop()
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Baseline Energy", f"{_safe_sum(energy_df.get('baseline_kwh', pd.Series())):,.0f} kWh")
        with c2:
            st.metric("Optimized Energy", f"{_safe_sum(energy_df.get('optimized_kwh', pd.Series())):,.0f} kWh")
        with c3:
            st.metric("Energy Saved", f"{_safe_sum(energy_df.get('energy_saved_kwh', pd.Series())):,.0f} kWh")
        with c4:
            tmp = detect_anomalies(
                energy_df.copy(),
                ["energy_saved_kwh"] if "energy_saved_kwh" in energy_df.columns else [],
                z_thresh=z_th, slope_thresh=slope_th, win=win, use_isoforest=use_if
            )
            st.metric("Anomalies", f"{int(tmp.get('anomaly', pd.Series(dtype=int)).sum() if 'anomaly' in tmp else 0)}")

        st.divider()
        cA, cB = st.columns([1.2, 1.0])

        # Gauge: Efficiency = optimized / baseline
        with cA:
            eff = 0.0
            if {"baseline_kwh", "optimized_kwh"}.issubset(energy_df.columns):
                base = _safe_sum(energy_df["baseline_kwh"])
                opt = _safe_sum(energy_df["optimized_kwh"])
                eff = (opt / base * 100.0) if base > 0 else 0.0
            fig_g = gauge("Energy Efficiency", eff, suffix="%", min_v=0, max_v=100)
            if fig_g is not None:
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.write(f"Energy Efficiency: {eff:.1f}%")

        # خطي: Baseline vs Optimized
        with cB:
            if PLOTLY and energy_ts and {"baseline_kwh", "optimized_kwh"}.issubset(energy_df.columns):
                fig = go.Figure()
                fig.add_scatter(x=energy_df[energy_ts], y=energy_df["baseline_kwh"], name="Baseline", mode="lines")
                fig.add_scatter(x=energy_df[energy_ts], y=energy_df["optimized_kwh"], name="Optimized", mode="lines")
                _plotly_layout(fig, "Real-Time Energy Savings")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(energy_df.head(50))

        st.divider()
        # أداء النظام (مثال: من مضخة)
        if not pump_df.empty and PLOTLY and pump_ts:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            if "temp_c" in pump_df.columns:
                fig2.add_scatter(
                    x=pump_df[pump_ts], y=pump_df["temp_c"], name="Temperature (°C)",
                    mode="lines", line=dict(width=3), secondary_y=False
                )
            if "vibration_mm_s" in pump_df.columns:
                fig2.add_scatter(
                    x=pump_df[pump_ts], y=pump_df["vibration_mm_s"], name="Vibration (mm/s)",
                    mode="lines", line=dict(width=2, dash="dot"), secondary_y=True
                )
            _plotly_layout(fig2, "System Performance")
            st.plotly_chart(fig2, use_container_width=True)

# ------------------ Pump Monitoring ------------------
with tab_pumps:
    st.markdown("#### Pump Monitoring")
    if pump_df.empty:
        st.info("لا توجد بيانات مضخات ضمن النطاق.")
    else:
        # بطاقات ملخص
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pump Status", f"{len(pump_df):,}")
        with c2:
            avg_eff = float(pd.to_numeric(pump_df.get("efficiency", pd.Series()), errors="coerce").mean() or 0.0)
            st.metric("Avg Efficiency", f"{avg_eff:.0f}%")
        with c3:
            avg_t = float(pd.to_numeric(pump_df.get("temp_c", pd.Series()), errors="coerce").mean() or 0.0)
            st.metric("Temperature", f"{avg_t:.0f}°C")
        with c4:
            avg_v = float(pd.to_numeric(pump_df.get("vibration_mm_s", pd.Series()), errors="coerce").mean() or 0.0)
            st.metric("Vibration", f"{avg_v:.2f} mm/s")

        st.divider()
        cA, cB, cC = st.columns(3)
        if PLOTLY:
            with cA:
                val = avg_eff
                fig = gauge("Pump Efficiency", val, suffix="%", min_v=0, max_v=100)
                st.plotly_chart(fig, use_container_width=True)
            with cB:
                fig = gauge("Vibration Level", avg_v, suffix=" mm/s", min_v=0, max_v=max(6, avg_v * 2))
                st.plotly_chart(fig, use_container_width=True)
            with cC:
                fig = gauge("Temperature", avg_t, suffix=" °C", min_v=0, max_v=max(80, avg_t * 2))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        if PLOTLY and pump_ts:
            cX, cY = st.columns(2)
            with cX:
                if "vibration_mm_s" in pump_df.columns:
                    fig = go.Figure()
                    fig.add_scatter(x=pump_df[pump_ts], y=pump_df["vibration_mm_s"], name="Vibration", mode="lines+markers",
                                    line=dict(width=2))
                    _plotly_layout(fig, "Vibration Trends")
                    st.plotly_chart(fig, use_container_width=True)
            with cY:
                if {"current_a", "power_kw"}.issubset(pump_df.columns):
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_scatter(x=pump_df[pump_ts], y=pump_df["current_a"], name="Current (A)",
                                    mode="lines+markers", line=dict(width=2), secondary_y=False)
                    fig.add_scatter(x=pump_df[pump_ts], y=pump_df["power_kw"], name="Power (kW)",
                                    mode="lines+markers", line=dict(width=2, dash="dot"), secondary_y=True)
                    _plotly_layout(fig, "Current & Power")
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        # نشاط تاريخي وشذوذ من الإهتزاز (مثال)
        if "vibration_mm_s" in pump_df.columns:
            pump_an = detect_anomalies(
                pump_df[[pump_ts, "vibration_mm_s"]].rename(columns={pump_ts: "ts"}).copy(),
                ["vibration_mm_s"], z_thresh=z_th, slope_thresh=slope_th, win=win, use_isoforest=use_if
            )
            if PLOTLY:
                fig = go.Figure()
                fig.add_scatter(x=pump_an["ts"], y=pump_an["vibration_mm_s"], name="Vibration", mode="lines")
                if "anomaly" in pump_an:
                    pts = pump_an[pump_an["anomaly"] == 1]
                    fig.add_scatter(x=pts["ts"], y=pts["vibration_mm_s"], name="Anomaly", mode="markers")
                _plotly_layout(fig, "Historical Activity & Anomalies")
                st.plotly_chart(fig, use_container_width=True)

# ------------------ Lighting (مبسّطة) ------------------
with tab_light:
    st.markdown("#### Lighting")
    if energy_df.empty:
        st.info("لا توجد بيانات طاقة لإظهار الإضاءة.")
    else:
        c1, c2 = st.columns([1.2, 1.0])
        with c1:
            # أعمدة: baseline vs optimized لكل يوم
            if PLOTLY and energy_ts and {"baseline_kwh", "optimized_kwh"}.issubset(energy_df.columns):
                d = energy_df[[energy_ts, "baseline_kwh", "optimized_kwh"]].copy()
                d["day"] = d[energy_ts].dt.date
                g = d.groupby("day", as_index=False).sum(numeric_only=True)
                fig = go.Figure(data=[
                    go.Bar(name="Baseline", x=g["day"], y=g["baseline_kwh"]),
                    go.Bar(name="Optimized", x=g["day"], y=g["optimized_kwh"]),
                ])
                fig.update_layout(barmode="group")
                _plotly_layout(fig, "Energy Usage vs Baseline (Daily)")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            # كفاءة الإضاءة = 1 - (optimized/baseline)
            eff = 0.0
            base = _safe_sum(energy_df.get("baseline_kwh", pd.Series()))
            opt = _safe_sum(energy_df.get("optimized_kwh", pd.Series()))
            if base > 0:
                eff = (1.0 - (opt / base)) * 100.0
            fig = gauge("Lighting Efficiency", eff, suffix="%", min_v=0, max_v=100)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

# ------------------ Agent Insights ------------------
with tab_agents:
    st.markdown("#### Agent Insights")
    if agent_df.empty:
        st.info("لا توجد سجلات للوكلاء ضمن النطاق.")
    else:
        # بطاقات
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Actions", f"{len(agent_df):,}")
        with c2:
            # تقدير: anomalies = عدد الأحداث ذات severity High/Critical إن وُجدت
            sev_col = next((c for c in agent_df.columns if c.lower() in ["severity", "level"]), None)
            an_count = 0
            if sev_col:
                an_count = int(agent_df[sev_col].astype(str).str.lower().isin(["high", "critical"]).sum())
            st.metric("Active Anomalies", f"{an_count}")
        with c3:
            # success rate من status=Completed
            st_col = next((c for c in agent_df.columns if c.lower() in ["status", "result"]), None)
            succ = 0.0
            if st_col:
                n = len(agent_df)
                k = agent_df[st_col].astype(str).str.lower().isin(["done", "completed", "executed", "resolved"]).sum()
                succ = (k / n * 100.0) if n else 0.0
            st.metric("Success Rate", f"{succ:.1f}%")
        with c4:
            # متوسط زمن الاستجابة (إن توفر)
            tcol = agent_ts
            # بدون بيانات زمن فعلية سنعرض '—'
            st.metric("Avg Response Time", "—")

        st.divider()
        # توزيع الشدة + خط زمني للتدخلات
        if PLOTLY:
            cA, cB = st.columns(2)
            with cA:
                if sev_col:
                    g = agent_df[sev_col].astype(str).str.title().value_counts().reset_index()
                    g.columns = ["Severity", "Count"]
                    fig = go.Figure()
                    fig.add_bar(x=g["Severity"], y=g["Count"], name="Severity")
                    _plotly_layout(fig, "Anomaly Severity Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            with cB:
                if agent_ts:
                    g = agent_df.groupby(pd.to_datetime(agent_df[agent_ts]).dt.hour).size().reset_index(name="count")
                    fig = go.Figure()
                    fig.add_scatter(x=g[agent_ts], y=g["count"], mode="lines+markers", name="Interventions")
                    _plotly_layout(fig, "Agent Interventions Timeline (per hour)")
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.write("##### Agent Event Logs")
        st.dataframe(agent_df.tail(200), use_container_width=True)
