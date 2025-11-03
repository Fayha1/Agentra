# app_streamlit.py  — Agentra (stable, alias-friendly, Arabic-friendly)

from __future__ import annotations
from pathlib import Path
import os
import math
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# -------- Plotly (optional but preferred) --------
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except Exception:
    PLOTLY = False

# -------- Optional sklearn for IsolationForest --------
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ================== Paths & Theme ==================
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "FullLogo_Transparent.png"

CSV_ENERGY = ROOT / "sim_property_riyadh_multi.csv"
CSV_ENERGY_SAVED = ROOT / "sim_property_riyadh_multi_saving15.csv"  # اختياري للمقارنة
CSV_PUMPS  = ROOT / "sim_pump_riyadh.csv"
CSV_AGENT_LOG = ROOT / "agent_audit_log.csv"

st.set_page_config(
    page_title="Agentra — Predictive Property Manager",
    layout="wide",
)

# ================ Utility ================

# خريطة الأسماء البديلة للأعمدة
ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date", "created_at"],
    "baseline_kwh": ["baseline_kwh", "baseline", "baseline_energy", "kwh_baseline"],
    "optimized_kwh": ["optimized_kwh", "optimized", "kwh_optimized", "agent_kwh"],
    "energy_saved_kwh": ["energy_saved_kwh", "saving_kwh", "kwh_saved", "energy_saved"],
    "efficiency": ["efficiency", "energy_efficiency", "eff"],
    "vibration_mm_s": ["vibration_mm_s", "vibration", "vibration_level", "vib_mm_s"],
    "temp_c": ["temp_c", "temperature_c", "temp", "temperature"],
    "current_a": ["current_a", "current", "ampere", "amps"],
    "power_kw": ["power_kw", "power", "kw", "pwr_kw"],
    "zone": ["zone", "area", "section"],
}

def _find_first(cols: list[str], options: list[str]) -> str | None:
    low = [c.lower().strip() for c in cols]
    for name in options:
        if name.lower() in low:
            return cols[low.index(name.lower())]
    return None

def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # حدد عمود الوقت بأي اسم بديل
    if df.empty:
        return df
    ts_col = _find_first(df.columns.tolist(), ALIASES["timestamp"])
    if ts_col is None:
        # لو لا يوجد وقت سنصنع واحداً متتابعاً
        df = df.copy()
        df["timestamp"] = pd.date_range(
            start=dt.datetime.now() - dt.timedelta(days=len(df)),
            periods=len(df),
            freq="H",
        )
        return df
    df = df.rename(columns={ts_col: "timestamp"}).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    return df

def _normalize_energy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _ensure_timestamp(df)

    # طاقة
    for canonical in ["baseline_kwh", "optimized_kwh", "energy_saved_kwh"]:
        cand = _find_first(df.columns.tolist(), ALIASES[canonical])
        if cand and cand != canonical:
            df = df.rename(columns={cand: canonical})

    # قياسات مضخة (احتمال تكون في نفس الملف)
    for canonical in ["efficiency", "vibration_mm_s", "temp_c", "current_a", "power_kw", "zone"]:
        cand = _find_first(df.columns.tolist(), ALIASES[canonical])
        if cand and cand != canonical:
            df = df.rename(columns={cand: canonical})

    # تحويل أرقام
    for c in ["baseline_kwh", "optimized_kwh", "energy_saved_kwh",
              "efficiency", "vibration_mm_s", "temp_c", "current_a", "power_kw"]:
        if c in df.columns:
            df[c] = _coerce_num(df[c])

    # إن لم يوجد energy_saved_kwh نحسبه عند توفر baseline & optimized
    if "energy_saved_kwh" not in df.columns and {"baseline_kwh", "optimized_kwh"} <= set(df.columns):
        df["energy_saved_kwh"] = (df["baseline_kwh"] - df["optimized_kwh"]).clip(lower=0)

    return df

def _smart_read(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()
    return df

def load_energy(source: str) -> pd.DataFrame:
    """
    source in {"Real-Time Sensors", "Historical Data", "Predictive Model"}
    كل الخيارات ترجع DataFrame مطبّعة الأعمدة.
    """
    # الافتراضي ملف sim_property_riyadh_multi.csv
    df = _smart_read(CSV_ENERGY)

    # إن وُجد ملف savings للمقارنة ندمجه بالعمود المناسب
    if CSV_ENERGY_SAVED.exists():
        df2 = _smart_read(CSV_ENERGY_SAVED)
        if not df2.empty:
            # نطبّع الاثنين ثم ندمج على timestamp
            df = _normalize_energy(df)
            df2 = _normalize_energy(df2)
            # سنأخذ optimized_kwh من df2 إن وُجد أعلى جودة
            use_cols = [c for c in ["optimized_kwh", "energy_saved_kwh"] if c in df2.columns]
            if "timestamp" in df.columns and "timestamp" in df2.columns and use_cols:
                df = df.merge(df2[["timestamp"] + use_cols], on="timestamp", how="left", suffixes=("", "_v2"))
                # لو جاءت أعمدة _v2 نستخدمها كبديل عند وجودها
                if "optimized_kwh_v2" in df.columns:
                    df["optimized_kwh"] = df["optimized_kwh_v2"].combine_first(df.get("optimized_kwh"))
                if "energy_saved_kwh_v2" in df.columns:
                    df["energy_saved_kwh"] = df["energy_saved_kwh_v2"].combine_first(df.get("energy_saved_kwh"))
                df = df.drop(columns=[c for c in df.columns if c.endswith("_v2")], errors="ignore")

    df = _normalize_energy(df)

    # لو وضع المستخدم "Predictive Model" نُضيف خَفضاً اصطناعياً % و jitter بسيط
    if source == "Predictive Model" and not df.empty:
        if "optimized_kwh" in df.columns:
            baseline = df.get("baseline_kwh")
            if baseline is not None:
                # افتراض خفض 20% ± 5%
                factor = 0.80 + np.random.normal(0, 0.03, size=len(df))
                df["optimized_kwh"] = (baseline * factor).clip(lower=0)
                df["energy_saved_kwh"] = (baseline - df["optimized_kwh"]).clip(lower=0)

    return df

def load_pumps() -> pd.DataFrame:
    df = _smart_read(CSV_PUMPS)
    df = _normalize_energy(df)
    return df

# =============== Anomaly detection ===============

def rolling_zscore(x: pd.Series, win: int = 24) -> pd.Series:
    mu = x.rolling(win, min_periods=max(3, win//3)).mean()
    sd = x.rolling(win, min_periods=max(3, win//3)).std()
    return (x - mu) / (sd.replace(0, np.nan))

def detect_anomalies(
    df: pd.DataFrame,
    cols: list[str],
    z_thresh: float = 2.5,
    slope_thresh: float = 0.8,
    win: int = 24,
    use_isoforest: bool = False,
) -> pd.DataFrame:
    """
    يشير العمود 'anomaly' إلى 1 عند تحقق أي من:
    - |rolling Z| >= z_thresh
    - |gradient| >= slope_thresh (بعد تطبيع x إلى [0,1])
    - (اختياري) IsolationForest درجة أقل من العتبة (outlier)
    """
    if df.empty or not cols:
        return df.assign(anomaly=0)

    out = df.copy()
    anom = pd.Series(0, index=out.index, dtype=int)

    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")

        # z-score
        z = rolling_zscore(s, win=win).abs()
        anom_z = (z >= z_thresh).astype(int)

        # slope (gradient على نسخة مطبعة 0..1)
        rng = s.max() - s.min()
        s_norm = (s - s.min()) / rng if rng and not pd.isna(rng) else s * 0
        grad = np.gradient(s_norm.fillna(0).values)
        anom_slope = (np.abs(grad) >= slope_thresh).astype(int)

        anom = np.maximum(anom.values, np.maximum(anom_z.values, anom_slope))

    # IsolationForest (متعدد المتغيرات)
    if use_isoforest and SKLEARN_AVAILABLE:
        num_cols = [c for c in cols if c in out.columns]
        X = out[num_cols].apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill")
        try:
            if not X.empty:
                iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
                lab = iso.fit_predict(X)  # -1 شاذ
                anom = np.maximum(anom, (lab == -1).astype(int))
        except Exception:
            pass

    out["anomaly"] = anom
    return out

# =============== Layout helpers ===============

PRIMARY = "#22c55e"      # أخضر رسمي
ACCENT  = "#a3e635"      # أخضر فاتح
BG_DARK = "#0f1117"

def center_header():
    # شعار في الوسط فقط
    st.markdown(
        """
        <style>
          .block-container{padding-top:1rem;}
          header {visibility: hidden;}
          .agentra-logo {display:flex; align-items:center; justify-content:center; margin-top:4px;}
          .agentra-title{ text-align:center; margin-top:6px; color:#e7ffe7; letter-spacing:0.5px; }
          .agentra-sub{ text-align:center; color:#94a3b8; margin-top:4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col = st.columns([1,6,1])[1]
    with col:
        if LOGO_PATH.exists():
            from PIL import Image
            img = Image.open(LOGO_PATH)
            st.image(img, width=140)
        st.markdown("<h1 class='agentra-title'>Agentra</h1>", unsafe_allow_html=True)
        st.markdown("<div class='agentra-sub'>Predictive Property Manager</div>", unsafe_allow_html=True)

def metric_row(metrics: list[tuple[str, str]]):
    cols = st.columns(len(metrics))
    for (label, value), c in zip(metrics, cols):
        with c:
            st.metric(label, value)

def gauge(label: str, value: float, vmin=0, vmax=100, suffix=""):
    if not PLOTLY:
        st.progress(min(max(int(100*(value-vmin)/(vmax-vmin)), 0), 100), text=f"{label}: {value:.1f}{suffix}")
        return
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value) if pd.notna(value) else 0.0,
            number={"suffix": suffix},
            title={"text": label},
            gauge={
                "axis": {"range": [vmin, vmax]},
                "bar": {"color": PRIMARY},
                "bordercolor": "#222",
                "bgcolor": "#0b0f16",
                "steps":[{"range":[vmin,vmax], "color":"#162227"}],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def line_xy(df, x, y, name):
    if not PLOTLY:
        st.line_chart(df.set_index(x)[y].rename(name))
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=name, line=dict(width=2)))
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ==================== Sidebar ====================
def sidebar() -> tuple[str, float, float, int, bool, tuple[pd.Timestamp|None, pd.Timestamp|None]]:
    st.sidebar.markdown("### Select data source")
    source = st.sidebar.selectbox(
        "Select data source",
        ["Real-Time Sensors", "Historical Data", "Predictive Model"],
        index=0
    )
    z_th = st.sidebar.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.05)
    slope_th = st.sidebar.slider("Slope Sensitivity", 0.1, 2.0, 0.8, 0.05)
    win = st.sidebar.slider("Rolling Window (points)", 8, 72, 24, 1)

    use_if = st.sidebar.checkbox("Advanced: IsolationForest", value=False, help="يتطلب scikit-learn")
    if use_if and not SKLEARN_AVAILABLE:
        st.sidebar.warning("scikit-learn غير مُثبّت في هذه البيئة. عطّل الخيار أو أضِفْه في المتطلبات.")

    st.sidebar.markdown("### Date Range")
    today = pd.Timestamp.now().normalize()
    start = today - pd.Timedelta(days=7)
    date_min = st.sidebar.date_input("From", start).strftime("%Y-%m-%d")
    date_max = st.sidebar.date_input("To", today).strftime("%Y-%m-%d")

    dmin = pd.to_datetime(date_min, errors="coerce")
    dmax = pd.to_datetime(date_max, errors="coerce") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    return source, z_th, slope_th, win, use_if, (dmin, dmax)

# ===================== App ======================
def main():
    center_header()

    source, z_th, slope_th, win, use_if, (dmin, dmax) = sidebar()

    tab_overview, tab_pump, tab_light, tab_agent = st.tabs(["Overview", "Pump Monitoring", "Lighting", "Agent Insights"])

    # --------- Load data safely ---------
    energy_df = load_energy(source)
    pumps_df  = load_pumps()

    # فلترة زمنية آمنة إن وُجد timestamp
    if not energy_df.empty and "timestamp" in energy_df.columns:
        energy_df = energy_df[
            (energy_df["timestamp"] >= dmin) &
            (energy_df["timestamp"] <= dmax)
        ].reset_index(drop=True)

    # ================= Overview =================
    with tab_overview:
        st.markdown("#### Overview")

        if energy_df.empty:
            st.info("لا توجد بيانات ضمن النطاق الزمني المحدد.")
        else:
            # حساب المجاميع/المؤشرات
            baseline_sum = energy_df.get("baseline_kwh")
            optimized_sum = energy_df.get("optimized_kwh")
            saved_sum = energy_df.get("energy_saved_kwh")

            metric_row([
                ("Baseline Energy", f"{baseline_sum.sum():,.0f} kWh" if baseline_sum is not None else "—"),
                ("Optimized Energy", f"{optimized_sum.sum():,.0f} kWh" if optimized_sum is not None else "—"),
                ("Energy Saved", f"{saved_sum.sum():,.0f} kWh" if saved_sum is not None else "—"),
                ("Anomalies", "…"),
            ])

            # عدّ الشذوذ على saved (لو موجود وإلا على optimized أو baseline)
            anomaly_source = "energy_saved_kwh" if "energy_saved_kwh" in energy_df.columns else \
                             ("optimized_kwh" if "optimized_kwh" in energy_df.columns else "baseline_kwh")
            adf = detect_anomalies(
                energy_df.copy(),
                [anomaly_source] if anomaly_source in energy_df.columns else [],
                z_thresh=z_th, slope_thresh=slope_th, win=win, use_isoforest=use_if
            )
            total_anom = int(adf["anomaly"].sum()) if "anomaly" in adf.columns else 0
            st.experimental_rerun if False else None  # placeholder to avoid linter

            # إعادة كتابة صف المقاييس لإظهار عدد الشذوذ الصحيح في العمود الرابع
            st.divider()
            metric_row([
                ("Baseline Energy", f"{baseline_sum.sum():,.0f} kWh" if baseline_sum is not None else "—"),
                ("Optimized Energy", f"{optimized_sum.sum():,.0f} kWh" if optimized_sum is not None else "—"),
                ("Energy Saved", f"{saved_sum.sum():,.0f} kWh" if saved_sum is not None else "—"),
                ("Anomalies", f"{total_anom}"),
            ])

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                # كفاءة (إن وُجدت) — وإلا نُقدّرها من saved/baseline
                if "efficiency" in energy_df.columns:
                    gauge("Energy Efficiency", float(energy_df["efficiency"].mean()), 0, 100, "%")
                elif ("baseline_kwh" in energy_df.columns) and ("optimized_kwh" in energy_df.columns):
                    eff = 100.0 * (1.0 - (energy_df["optimized_kwh"].sum() / (energy_df["baseline_kwh"].sum() + 1e-6)))
                    eff = np.clip(eff, 0, 100)
                    gauge("Energy Efficiency (est.)", float(eff), 0, 100, "%")
                else:
                    st.info("لا يوجد عمود 'efficiency' ولا يمكن تقدير الكفاءة.")

            with c2:
                if "energy_saved_kwh" in energy_df.columns:
                    line_xy(energy_df, "timestamp", "energy_saved_kwh", "Energy Savings (kWh)")
                elif "optimized_kwh" in energy_df.columns and "baseline_kwh" in energy_df.columns:
                    tmp = energy_df.copy()
                    tmp["energy_saved_kwh"] = (tmp["baseline_kwh"] - tmp["optimized_kwh"]).clip(lower=0)
                    line_xy(tmp, "timestamp", "energy_saved_kwh", "Energy Savings (kWh)")
                else:
                    st.info("لا تتوفر أعمدة كافية لعرض منحنى التوفير.")

    # ================= Pump Monitoring =================
    with tab_pump:
        st.markdown("#### Pump Monitoring")
        if pumps_df.empty:
            st.info("لا توجد بيانات مضخات (ملف sim_pump_riyadh.csv غير متاح).")
        else:
            # مقاييس عامة
            eff = pumps_df.get("efficiency")
            vib = pumps_df.get("vibration_mm_s")
            temp = pumps_df.get("temp_c")
            metric_row([
                ("Avg Efficiency", f"{eff.mean():.0f}%" if eff is not None else "—"),
                ("Vibration", f"{vib.mean():.2f} mm/s" if vib is not None else "—"),
                ("Temperature", f"{temp.mean():.1f} °C" if temp is not None else "—"),
                ("Samples", f"{len(pumps_df):,}")
            ])
            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                if "efficiency" in pumps_df.columns:
                    gauge("Pump Efficiency", float(eff.mean()), 0, 100, "%")
            with c2:
                if "vibration_mm_s" in pumps_df.columns:
                    gauge("Vibration Level", float(vib.mean()), 0, 6, " mm/s")
            with c3:
                if "temp_c" in pumps_df.columns:
                    gauge("Temperature", float(temp.mean()), 0, 80, " °C")

            st.divider()
            # خطوط زمنية
            if "timestamp" in pumps_df.columns:
                c1, c2 = st.columns(2)
                with c1:
                    if "vibration_mm_s" in pumps_df.columns:
                        line_xy(pumps_df, "timestamp", "vibration_mm_s", "Vibration")
                with c2:
                    if "power_kw" in pumps_df.columns:
                        line_xy(pumps_df, "timestamp", "power_kw", "Power (kW)")

    # ================= Lighting =================
    with tab_light:
        st.markdown("#### Lighting")
        if energy_df.empty:
            st.info("لا توجد بيانات.")
        else:
            # رسم مقارنة baseline vs optimized إن وُجدا
            if PLOTLY and {"timestamp","baseline_kwh","optimized_kwh"} <= set(energy_df.columns):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=energy_df["timestamp"], y=energy_df["baseline_kwh"],
                                         mode="lines", name="Baseline", line=dict(width=2)))
                fig.add_trace(go.Scatter(x=energy_df["timestamp"], y=energy_df["optimized_kwh"],
                                         mode="lines", name="Optimized", line=dict(width=2)))
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("تحتاج الأعمدة baseline_kwh و optimized_kwh لإظهار مقارنة الإضاءة.")

    # ================= Agent Insights =================
    with tab_agent:
        st.markdown("#### Agent Insights")
        if not CSV_AGENT_LOG.exists():
            st.info("لا توجد سجلات للوكلاء (agent_audit_log.csv غير موجود).")
        else:
            logs = _smart_read(CSV_AGENT_LOG)
            if logs.empty:
                st.info("لا توجد سجلات ضمن النطاق الزمني المحدد.")
            else:
                # عرض مبسط
                st.dataframe(logs.head(500), use_container_width=True)

# --------------- Run ---------------
if __name__ == "__main__":
    main()
