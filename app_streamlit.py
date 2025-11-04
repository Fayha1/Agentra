# app_streamlit.py  —  Agentra (Streamlit, Plotly, robust loaders + anomaly engine)
from __future__ import annotations

from pathlib import Path
import os
import math
import json
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------ Plotly (مع تجميل افتراضي) ------------------
PLOTLY = True
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    PLOTLY = False

# ------------------ IsolationForest (اختياري) ------------------
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# =================================================================
#                              ثوابت
# =================================================================
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
LOGO_PATH = ASSETS / "FullLogo_Transparent.png"

# مسارات الملفات الافتراضية (عدّل الأسماء لو تغيّرت عندك)
CSV_ENERGY = ROOT / "sim_property_riyadh_multi_saving15.csv"   # baseline/optimized/saved + (ربما lux/occupancy)
CSV_ENERGY_SAVED = ROOT / "sim_property_riyadh_multi.csv"      # نسخة بديلة للطاقة
CSV_PUMPS = ROOT / "sim_pump_riyadh.csv"                       # مضخات
CSV_AGENT_LOG = ROOT / "agent_audit_log.csv"                   # سجلات الوكيل

# =================================================================
#                          أدوات تنسيق/مساعدة
# =================================================================
PRIMARY = "#7DF28B"          # أخضر رسمي
ACCENT  = "#2ee69e"
BG_CARD = "rgba(255,255,255,0.04)"
RED     = "#ff6b6b"
AMBER   = "#ffb86b"
BLUE    = "#58a6ff"
GRAY    = "#8b949e"

st.set_page_config(page_title="Agentra — Predictive Property Manager", layout="wide")

def _center_logo_and_title():
    """شعار كبير في المنتصف + عنوان، بدون تكراره في الشريط الجانبي."""
    logo_html = ""
    if LOGO_PATH.exists():
        # أكبر قليلًا من الكلمة
        logo_html = f"""
        <div style="display:flex;justify-content:center;margin-top:8px">
          <img src="app://{LOGO_PATH.as_posix()}" style="width:140px;opacity:.95" />
        </div>
        """
    title_html = f"""
      <h1 style="text-align:center;margin:8px 0 4px 0;color:{PRIMARY}">Agentra</h1>
      <div style="text-align:center;color:{GRAY};margin-bottom:10px">Predictive Property Manager</div>
    """
    st.markdown(logo_html + title_html, unsafe_allow_html=True)

def _nice_int(x: float | int) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"

def _to_dt(s) -> pd.Timestamp:
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """يحاول العثور/إنشاء عمود timestamp بأسماء بديلة أو من index."""
    candidates = ["timestamp","time","datetime","date_time","ts","Date","DATE","date"]
    col = next((c for c in candidates if c in df.columns), None)
    if col:
        df["timestamp"] = pd.to_datetime(df[col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = df.index.tz_localize(None)
    else:
        # بدون وقت؟ اصنع تسلسل زمني وهمي بالثواني
        df["timestamp"] = pd.date_range("2025-08-01", periods=len(df), freq="5min")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df

def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    return None

def _warn_missing(colname: str):
    st.warning(f"⚠️ العمود '{colname}' غير موجود في البيانات.", icon="⚠️")

# =================================================================
#                 تحميل البيانات وتطبيع أسماء الأعمدة
# =================================================================
def load_energy() -> pd.DataFrame:
    """
    يعيد DataFrame بعمود timestamp وأعمدة:
      baseline_kwh, optimized_kwh, energy_saved_kwh (ويحاول استنتاجها أو توليدها)
      وقد يحوي lux, occupancy إن توفّرت.
    """
    df = None
    paths = [CSV_ENERGY, CSV_ENERGY_SAVED]
    for p in paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                break
            except Exception:
                continue
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_timestamp(df)

    # أعمدة الطاقة
    c_base = _find_col(df, ["baseline_kwh","baseline","Base","baselineEnergy","baseline_kW","baseline_kwh_today"])
    c_opt  = _find_col(df, ["optimized_kwh","optimized","opt","optimizedEnergy"])
    c_save = _find_col(df, ["energy_saved_kwh","saved","saving_kwh","kwh_saved","energy_savings"])

    # لو ناقصة، أنشئها (افتراضًا توفير 15%)
    if not c_base:
        # حاول استنتاج baseline من مجموع أي عمود طاقة إن وُجد
        c_base = "baseline_kwh"
        df[c_base] = pd.to_numeric(df.select_dtypes(include=[np.number]).sum(axis=1), errors="coerce")
        df[c_base] = df[c_base].replace([np.inf,-np.inf], np.nan).fillna(method="ffill").fillna(0)

    if not c_opt and c_base:
        c_opt = "optimized_kwh"
        df[c_opt] = pd.to_numeric(df[c_base], errors="coerce") * 0.85

    if not c_save:
        c_save = "energy_saved_kwh"
        df[c_save] = pd.to_numeric(df[c_base], errors="coerce") - pd.to_numeric(df[c_opt], errors="coerce")

    df = _coerce_numeric(df, [c_base, c_opt, c_save])

    # لو موجودة بيانات الإضاءة
    c_lux = _find_col(df, ["lux","avg_lux","light_lux"])
    c_occ = _find_col(df, ["occupancy","occ","occupancy_pct"])
    # اعادة تسمية موحدة
    rename = {}
    rename[c_base] = "baseline_kwh"
    rename[c_opt]  = "optimized_kwh"
    rename[c_save] = "energy_saved_kwh"
    if c_lux: rename[c_lux] = "lux"
    if c_occ: rename[c_occ] = "occupancy"
    df = df.rename(columns=rename)

    return df[["timestamp","baseline_kwh","optimized_kwh","energy_saved_kwh"] + ([c for c in ["lux","occupancy"] if c in df.columns])].copy()

def load_pumps() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    يعيد (pump_df, pump_ts):
      pump_df: صفوف قياسات لحظية للمضخات
      pump_ts: تجميع زمني للدوال البيانية
    يستوعب أسماء بديلة للأعمدة.
    """
    if not CSV_PUMPS.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(CSV_PUMPS)
    df = _ensure_timestamp(df)

    c_eff  = _find_col(df, ["efficiency","pump_efficiency","eff"])
    c_vib  = _find_col(df, ["vibration_mm_s","vibration","vib_mm_s","vib_g"])
    c_tmp  = _find_col(df, ["temp_c","temperature","temp","temp_C"])
    c_cur  = _find_col(df, ["current_a","current","amps"])
    c_pow  = _find_col(df, ["power_kw","power","kW"])

    # كوّن ما يلزم إن ناقص
    if not c_eff:
        c_eff = "efficiency"
        df[c_eff] = 70 + 10*np.sin(np.linspace(0, 20, len(df)))
    if not c_vib:
        c_vib = "vibration_mm_s"
        df[c_vib] = np.abs(np.random.normal(1.5, 0.4, len(df)))
    if not c_tmp:
        c_tmp = "temp_c"
        df[c_tmp] = 35 + 4*np.sin(np.linspace(0, 10, len(df)))
    if not c_cur:
        c_cur = "current_a"
        df[c_cur] = 12 + 3*np.sin(np.linspace(0, 14, len(df)))
    if not c_pow:
        c_pow = "power_kw"
        df[c_pow] = df[c_cur] * 0.75

    df = _coerce_numeric(df, [c_eff,c_vib,c_tmp,c_cur,c_pow])

    df = df.rename(columns={
        c_eff:"efficiency",
        c_vib:"vibration_mm_s",
        c_tmp:"temp_c",
        c_cur:"current_a",
        c_pow:"power_kw",
    })

    # تجميع لكل ساعة (أو حسب الدقة الموجودة)
    pump_ts = (df
               .set_index("timestamp")
               .resample("1H")[["efficiency","vibration_mm_s","temp_c","current_a","power_kw"]]
               .mean()
               .reset_index())
    return df, pump_ts

def load_agent_log() -> pd.DataFrame:
    if not CSV_AGENT_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(CSV_AGENT_LOG)
    df = _ensure_timestamp(df)
    # أعمدة متوقعة: timestamp, agent, action, target, severity, decision_basis
    # جرّب إعادة تسمية إن لزم
    mapping = {}
    for k, aliases in {
        "agent": ["agent","agent_id","source"],
        "action": ["action","event","decision"],
        "target": ["target","asset","pump","zone"],
        "severity": ["severity","level","prio"],
        "decision_basis": ["decision_basis","reason","basis","explain"],
    }.items():
        c = _find_col(df, aliases)
        if c and c != k:
            mapping[c] = k
    if mapping:
        df = df.rename(columns=mapping)
    for c in ["agent","action","target","severity","decision_basis"]:
        if c not in df.columns:
            df[c] = ""
    return df[["timestamp","agent","action","target","severity","decision_basis"]].copy()

# =================================================================
#                     محرّك اكتشاف الشذوذ
# =================================================================
def rolling_z_anomaly(
    s: pd.Series,
    win: int = 24,
    z_thresh: float = 2.5,
    slope_thresh: float = 0.8,
) -> pd.Series:
    """يرجع Series منطقية للشذوذ عبر |Z| و/أو ميل مفاجئ مقابل انحراف معياري متحرك."""
    if len(s) < max(8, win):
        return pd.Series([False]*len(s), index=s.index)

    s = pd.to_numeric(s, errors="coerce")
    mu = s.rolling(win, min_periods=8).mean()
    sd = s.rolling(win, min_periods=8).std().replace(0, np.nan)
    z = (s - mu) / sd

    # ميل (gradient) على نافذة قصيرة (5 نقاط) كنسبة من SD
    grad = s.diff().rolling(5, min_periods=2).mean()
    grad_norm = np.abs(grad) / (sd + 1e-9)

    z_flag = z.abs() >= z_thresh
    slope_flag = grad_norm >= slope_thresh
    return (z_flag | slope_flag).fillna(False)

def detect_anomalies(
    df: pd.DataFrame,
    cols: list[str],
    z_thresh: float,
    slope_thresh: float,
    win: int,
    use_isoforest: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    out["anomaly"] = False

    # Rolling Z لكل عمود
    for c in cols:
        if c not in out.columns:
            continue
        try:
            flags = rolling_z_anomaly(out[c], win=win, z_thresh=z_thresh, slope_thresh=slope_thresh)
            out["anomaly"] = out["anomaly"] | flags
        except Exception:
            pass

    # IsolationForest (متعدد المتغيرات) إن توفر
    if use_isoforest and SKLEARN_AVAILABLE:
        feats = [c for c in cols if c in out.columns]
        if len(feats) >= 2:
            X = out[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
            try:
                iso = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
                pred = iso.fit_predict(X.values)  # -1 شاذ
                out["anomaly"] = out["anomaly"] | (pred == -1)
            except Exception:
                pass

    return out

# =================================================================
#                          واجهة المستخدم
# =================================================================
def sidebar_controls():
    st.sidebar.markdown("### Select data source")
    src = st.sidebar.selectbox("",
        options=["Real-Time Sensors","Historical Data","Predictive Model"],
        index=0
    )

    z_th = st.sidebar.slider("Z-Score Sensitivity", 1.0, 5.0, 2.5, 0.05)
    slope_th = st.sidebar.slider("Slope Sensitivity", 0.10, 2.0, 0.80, 0.05)
    win = st.sidebar.slider("Rolling Window (points)", 8, 72, 24, 1)

    use_if = st.sidebar.checkbox("Advanced: IsolationForest")
    if use_if and not SKLEARN_AVAILABLE:
        st.sidebar.info("scikit-learn غير متاح في هذه البيئة. فعِّله لاحقًا أو أضِفه إلى المتطلبات.")

    # التواريخ
    st.sidebar.markdown("### Date Range")
    today = dt.datetime.now().date()
    start_default = today - dt.timedelta(days=7)
    date_min = st.sidebar.date_input("From", value=start_default)
    date_max = st.sidebar.date_input("To", value=today)

    # لقطة شاشة
    st.sidebar.button("Screenshot")

    return src, z_th, slope_th, win, use_if, date_min, date_max

def filter_by_date(df: pd.DataFrame, date_min: dt.date, date_max: dt.date) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "timestamp" not in df.columns:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    # اجعل نهاية اليوم شاملة
    dt_min = pd.Timestamp(dt.datetime.combine(date_min, dt.time.min))
    dt_max = pd.Timestamp(dt.datetime.combine(date_max, dt.time.max))
    return df[(df["timestamp"] >= dt_min) & (df["timestamp"] <= dt_max)].reset_index(drop=True)

def gauge(value: float, title: str, suffix: str = "", vmin: float = 0, vmax: float = 100):
    if not PLOTLY:
        st.metric(title, f"{_nice_int(value)}{suffix}")
        return
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value) if value is not None else 0.0,
        number={"font":{"color":"#e6edf3"}},
        gauge={
            "axis":{"range":[vmin, vmax],"tickcolor":GRAY},
            "bar":{"color":PRIMARY},
            "bgcolor":"rgba(255,255,255,0.02)",
            "borderwidth":0,
            "steps":[
                {"range":[vmin, (vmin+vmax)/2], "color":"rgba(125,242,139,0.20)"},
                {"range":[(vmin+vmax)/2, vmax], "color":"rgba(125,242,139,0.05)"},
            ],
        },
        title={"text":title, "font":{"color":GRAY}}
    ))
    fig.update_layout(height=220, margin=dict(l=10,r=10,t=20,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

def line(fig_title: str, ts_df: pd.DataFrame, y_cols: list[str], y_names: list[str], y2_cols: list[str] | None = None, y2_names: list[str] | None = None):
    if not PLOTLY or ts_df.empty:
        st.info("لا توجد بيانات للرسم.")
        return

    if y2_cols:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    for c, name in zip(y_cols, y_names):
        if c in ts_df.columns:
            fig.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df[c],
                                     mode="lines", name=name,
                                     line=dict(width=2), hovertemplate="%{y:.2f}<extra></extra>"),
                          secondary_y=False if y2_cols else None)
    if y2_cols:
        for c, name in zip(y2_cols, y2_names):
            if c in ts_df.columns:
                fig.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df[c],
                                         mode="lines", name=name,
                                         line=dict(width=2, dash="dot"),
                                         hovertemplate="%{y:.2f}<extra></extra>"),
                              secondary_y=True)

    fig.update_layout(
        title=fig_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340, margin=dict(l=10,r=10,t=40,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3")
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig, use_container_width=True)

# =================================================================
#                               التطبيق
# =================================================================
def main():
    _center_logo_and_title()

    src, z_th, slope_th, win, use_if, date_min, date_max = sidebar_controls()

    # تحميل
    energy_df = load_energy()
    pump_df, pump_ts = load_pumps()
    agent_df = load_agent_log()

    # اختيار المصدر (تأثيره هنا فقط للتسمية/الفلترة؛ الربط الحقيقي يتم لاحقًا)
    # Real-Time: نعرض آخر 24–48 ساعة، Historical: المدى الزمني، Predictive: نستخدم ملفات الطاقة + النمذجة
    if src == "Real-Time Sensors":
        # لا تغيّر المسارات، فقط فلترة زمنية قصيرة
        energy_df = filter_by_date(energy_df, date_min, date_max)
        pump_df   = filter_by_date(pump_df, date_min, date_max)
        pump_ts   = filter_by_date(pump_ts, date_min, date_max)
        agent_df  = filter_by_date(agent_df, date_min, date_max)
    elif src == "Historical Data":
        energy_df = filter_by_date(energy_df, date_min, date_max)
        pump_df   = filter_by_date(pump_df, date_min, date_max)
        pump_ts   = filter_by_date(pump_ts, date_min, date_max)
        agent_df  = filter_by_date(agent_df, date_min, date_max)
    else:
        # Predictive Model: نفس البيانات ولكن سنستخدم أعمدة الطاقة كناتج "محسّن" ونعمل كشف شذوذ عليها
        energy_df = filter_by_date(energy_df, date_min, date_max)
        pump_df   = filter_by_date(pump_df, date_min, date_max)
        pump_ts   = filter_by_date(pump_ts, date_min, date_max)
        agent_df  = filter_by_date(agent_df, date_min, date_max)

    # تبويبات
    tab_overview, tab_pump, tab_light, tab_agent = st.tabs(["Overview","Pump Monitoring","Lighting","Agent Insights"])

    # ------------------ Overview ------------------
    with tab_overview:
        st.markdown("#### Overview")
        if energy_df.empty:
            st.info("لا توجد بيانات ضمن النطاق الزمني المحدد.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Baseline Energy", f"{_nice_int(energy_df['baseline_kwh'].sum())} kWh")
            with c2:
                st.metric("Optimized Energy", f"{_nice_int(energy_df['optimized_kwh'].sum())} kWh")
            with c3:
                st.metric("Energy Saved", f"{_nice_int(energy_df['energy_saved_kwh'].sum())} kWh")
            with c4:
                tmp = detect_anomalies(
                    energy_df.copy(),
                    ["energy_saved_kwh"],
                    z_thresh=z_th, slope_thresh=slope_th, win=win, use_isoforest=use_if
                )
                st.metric("Anomalies", f"{int(tmp['anomaly'].sum())}")

            # كفاءة الطاقة (Gauge = optimized / baseline)
            try:
                eff = 100.0 * (energy_df["optimized_kwh"].sum() / max(1.0, energy_df["baseline_kwh"].sum()))
                gauge(100.0*eff, "Energy Efficiency (%)")  # eff كان ratio، نضرب مرّة ثانية غير صحيح
            except Exception:
                gauge(0, "Energy Efficiency (%)")

            # سلسلة زمنية للتوفير الفعلي
            ts = (energy_df
                  .set_index("timestamp")[["baseline_kwh","optimized_kwh","energy_saved_kwh"]]
                  .resample("1H").sum().reset_index())
            line("Real-Time Energy Savings", ts,
                 ["baseline_kwh","optimized_kwh","energy_saved_kwh"],
                 ["Baseline","Optimized","Saved"])

    # ------------------ Pump Monitoring ------------------
    with tab_pump:
        st.markdown("#### Pump Monitoring")
        if pump_ts.empty:
            st.info("لا توجد بيانات مضخات ضمن النطاق.")
        else:
            k1,k2,k3,k4 = st.columns([1,1,1,1])
            with k1:
                try:
                    gauge(float(pump_ts["efficiency"].mean()), "Avg Efficiency (%)", vmin=0, vmax=100)
                except Exception:
                    gauge(0, "Avg Efficiency (%)")
            with k2:
                try:
                    gauge(float(pump_ts["vibration_mm_s"].mean()), "Vibration (mm/s)", vmin=0, vmax=6)
                except Exception:
                    gauge(0, "Vibration (mm/s)", vmin=0, vmax=6)
            with k3:
                try:
                    gauge(float(pump_ts["temp_c"].mean()), "Temperature (°C)", vmin=20, vmax=80)
                except Exception:
                    gauge(0, "Temperature (°C)", vmin=20, vmax=80)
            with k4:
                st.metric("Samples", f"{_nice_int(len(pump_df))}")

            # اتجاهات
            line("Vibration / Temperature Trends", pump_ts,
                 ["vibration_mm_s","temp_c"],
                 ["Vibration","Temperature"])

            # Current vs Power على محورين
            line("Current vs Power", pump_ts,
                 ["current_a"], ["Current (A)"],
                 y2_cols=["power_kw"], y2_names=["Power (kW)"])

    # ------------------ Lighting ------------------
    with tab_light:
        st.markdown("#### Lighting")
        if ("lux" not in energy_df.columns) or energy_df.empty:
            st.info("لا توجد بيانات إضاءة (lux/occupancy) في الملف الحالي.")
        else:
            # تراكب lux vs occupancy
            ts_l = (energy_df.set_index("timestamp")[["lux","occupancy"]]
                    .resample("1H").mean().reset_index())
            line("Lux vs Occupancy", ts_l, ["lux"], ["Lux"], y2_cols=["occupancy"], y2_names=["Occupancy (%)"])

            # كفاءة الإضاءة: وفّرنا الطاقة عبر التحكم بالإشغال
            try:
                saved = energy_df["energy_saved_kwh"].sum()
                base  = energy_df["baseline_kwh"].sum()
                eff_l = 100.0 * (saved / max(1.0, base))
                gauge(eff_l, "Lighting Efficiency (%)", vmin=0, vmax=100)
            except Exception:
                gauge(0, "Lighting Efficiency (%)")

    # ------------------ Agent Insights ------------------
    with tab_agent:
        st.markdown("#### Agent Insights")
        if agent_df.empty:
            st.info("لا يوجد سجلات للوكيل ضمن النطاق.")
        else:
            # فلترة حسب الشدّة/الزمن يمكن إضافتها لاحقًا
            st.dataframe(agent_df.sort_values("timestamp").reset_index(drop=True), use_container_width=True)

    # توضيح محرّك الشذوذ
    st.caption("Anomaly engine: rolling Z-score + gradient; optional IsolationForest (multivariate).")

if __name__ == "__main__":
    main()
