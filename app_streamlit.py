# ---------------------------------------------
# Agentra — Predictive Property Manager (Hardened)
# Streamlit + Plotly | Robust to schema drift, dtype issues, and rolling/resample errors
# ---------------------------------------------

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============ Page ============
st.set_page_config(page_title="Agentra — Predictive Property Manager", layout="wide")
st.markdown(
    """
    <div style="text-align:center; margin-top:8px; margin-bottom:4px">
        <h1 style="font-size:40px; margin:0">Agentra — Predictive Property Manager</h1>
        <div style="opacity:.8; margin-top:6px">
            AI for predictive maintenance & energy optimization
        </div>
    </div>
    <hr style="margin:12px 0"/>
    """,
    unsafe_allow_html=True,
)

# ============ Helpers: schema ============
STANDARD_COLS = {
    # timestamps
    "timestamp": ["timestamp", "time", "datetime", "date"],
    # energy (Wh)
    "light_wh_base":  ["light_wh_base","light_wh_baseline","light_wh_b","energy_wh_base"],
    "light_wh_agent": ["light_wh_agent","light_wh_ai","light_wh_ag","energy_wh_agent","light_wh"],
    # power (W) -> fallback to compute Wh
    "light_w_base":  ["light_w_base","light_w_baseline","light_watts_base","light_w_b"],
    "light_w_agent": ["light_w_agent","light_w_ai","light_watts_agent"],
    # pump
    "pump_vib_rms_g": ["pump_vib_rms_g","pump_vibration_rms_g","pump_vib_g","vibration_g","vib_g","vibration","pump_vib_ms_g"],
    "pump_current_a": ["pump_current_a","current_a","pump_i"],
    "pump_temp_c":    ["pump_temp_c","pump_temperature_c","temp_c"],
    # labels / model outputs
    "status":        ["status"],
    "anomaly_label": ["anomaly_label","label"],
    "pred_anom":     ["pred_anom","pred_anomaly","is_anom","predicted_anomaly"],
    "anom_score":    ["anom_score","anomaly_score","score"],
    # lighting context
    "occ": ["occ","occupancy"],
    "lux": ["lux","illuminance_lux"],
    # convenience
    "day":  ["day","day_index"],
    "hour": ["hour"],
}

def _find_first(cols, candidates):
    cl = [c.lower() for c in cols]
    for c in candidates:
        if c.lower() in cl:
            return c.lower()
    return None

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for std, variants in STANDARD_COLS.items():
        found = _find_first(df.columns, variants)
        if found and found != std and std not in df.columns:
            df = df.rename(columns={found: std})
    return df

def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    for c in STANDARD_COLS["timestamp"]:
        c = c.lower()
        if c in df.columns:
            df["timestamp"] = pd.to_datetime(df[c], errors="coerce")
            break
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include a timestamp column (e.g., 'timestamp'/'time'/'datetime').")
    if df["timestamp"].isna().all():
        raise ValueError("Failed to parse any timestamps. Please check the date format.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def _to_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _infer_energy_from_power(df: pd.DataFrame) -> pd.DataFrame:
    """If Wh columns are missing but power (W) exist, compute per-row Wh using delta time in hours."""
    df = df.copy()
    has_wh = ("light_wh_base" in df.columns) and ("light_wh_agent" in df.columns)
    has_w  = ("light_w_base" in df.columns) or ("light_w_agent" in df.columns)
    if (not has_wh) and has_w:
        # compute dt (hours) safely
        dt_h = df["timestamp"].diff().dt.total_seconds() / 3600.0
        # fallback: fill 0/NaN with median cadence if available, else 1 minute
        cadence = np.nanmedian(dt_h) if np.isfinite(np.nanmedian(dt_h)) else (1.0/60.0)
        dt_h = dt_h.replace(0, np.nan).fillna(cadence)
        if "light_w_base" in df.columns and "light_wh_base" not in df.columns:
            df["light_wh_base"]  = pd.to_numeric(df["light_w_base"],  errors="coerce") * dt_h
        if "light_w_agent" in df.columns and "light_wh_agent" not in df.columns:
            df["light_wh_agent"] = pd.to_numeric(df["light_w_agent"], errors="coerce") * dt_h
    return df

def _rolling_stats_time(df: pd.DataFrame, value_col: str, window_minutes: int) -> pd.DataFrame:
    """Time-based rolling mean/std using '<N>min' window. Robust to dtype/index issues."""
    if value_col not in df.columns:
        return pd.DataFrame(columns=["timestamp", value_col, "roll_mean", "roll_std"])
    x = df[["timestamp", value_col]].copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x[value_col]   = pd.to_numeric(x[value_col], errors="coerce")
    x = x.dropna(subset=["timestamp", value_col]).sort_values("timestamp")
    if x.empty:
        return pd.DataFrame(columns=["timestamp", value_col, "roll_mean", "roll_std"])
    x = x.set_index("timestamp")
    try:
        rm = x[value_col].rolling(f"{int(window_minutes)}min")
        x["roll_mean"] = rm.mean()
        x["roll_std"]  = rm.std()
    except Exception:
        # last-resort: fixed-size window (not time-based)
        x["roll_mean"] = x[value_col].rolling(max(2, int(window_minutes))).mean()
        x["roll_std"]  = x[value_col].rolling(max(2, int(window_minutes))).std()
    x = x.reset_index()
    return x

def _smart_resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Robust resampling:
      - 'light_wh*' columns: SUM (preserve energy totals)
      - other numeric columns: MEAN
      - non-numeric columns: FIRST
    Only includes columns that exist to avoid agg errors.
    """
    if rule == "OFF":
        out = df.copy()
        if "day" not in out.columns:  out["day"]  = out["timestamp"].dt.date
        if "hour" not in out.columns: out["hour"] = out["timestamp"].dt.hour
        return out

    # attempt to coerce typical numeric-like columns
    maybe_numeric = [
        "light_wh_base","light_wh_agent","light_w_base","light_w_agent",
        "pump_vib_rms_g","pump_current_a","pump_temp_c","anom_score","occ","lux"
    ]
    df = _to_numeric(df, maybe_numeric)

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    agg = {}
    for c in df.columns:
        if c == "timestamp":
            continue
        if c in num_cols:
            agg[c] = "sum" if c.startswith("light_wh") else "mean"
        else:
            agg[c] = "first"

    # keep only existing columns in agg
    agg = {k: v for k, v in agg.items() if k in df.columns}

    out = (
        df.set_index("timestamp")
          .resample(rule)
          .agg(agg)
          .reset_index()
    )
    out["day"]  = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    return out

def _safe_sum(df, col):
    return float(df[col].sum()) if col in df.columns and df[col].notna().any() else np.nan

def _nice_int(x):
    return f"{int(round(float(x))):,}" if (isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))) else "—"

def _load_optional_csv(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        d = pd.read_csv(path)
        d.columns = [c.strip().lower() for c in d.columns]
        if "timestamp" in d.columns:
            d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        return d
    except Exception:
        return None

# ============ Sidebar ============
with st.sidebar:
    st.header("Settings")
    csv_path   = st.text_input("CSV file path", value="sim_property_riyadh_multi_saving15.csv")
    audit_path = st.text_input("Agent audit log (optional)", value="")
    use_range  = st.checkbox("Enable date range filter", value=False)

    st.divider()
    st.subheader("Anomaly detection (fallback if no 'pred_anom')")
    roll_minutes = st.slider("Rolling window (minutes)", 10, 240, 60, step=10)
    z_thresh     = st.slider("Z-score threshold", 1.0, 6.0, 2.5, step=0.1)
    slope_thresh = st.slider("Slope threshold (Δg per hour)", 0.0, 2.0, 0.25, step=0.05)
    st.caption("An anomaly is flagged if |Z| > threshold OR rolling slope exceeds threshold.")

    st.divider()
    st.subheader("Charts")
    resample_rule = st.selectbox("Resample (smoothing/aggregation)", ["OFF", "5min", "15min", "30min", "1H"], index=2)

# ============ Load data ============
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_cols(df)
    df = _ensure_timestamp(df)
    df = _infer_energy_from_power(df)
    return df

try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

audit_df = _load_optional_csv(audit_path)

# ============ Optional date range ============
if use_range:
    min_d = df["timestamp"].min().date()
    max_d = df["timestamp"].max().date()
    c1, c2 = st.columns(2)
    start = c1.date_input("Start date", min_d, min_value=min_d, max_value=max_d)
    end   = c2.date_input("End date", max_d, min_value=min_d, max_value=max_d)
    if start > end:
        st.warning("Start date must be before end date.")
    else:
        mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
        subset = df.loc[mask]
        if subset.empty:
            st.info("No data for the selected date range — showing full dataset.")
        else:
            df = subset

# ============ Derivatives & resample ============
df["day"]  = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df = _smart_resample(df, resample_rule)

# ============ Energy KPIs ============
baseline_wh = _safe_sum(df, "light_wh_base")
agent_wh    = _safe_sum(df, "light_wh_agent")
if math.isfinite(baseline_wh) and baseline_wh > 0 and math.isfinite(agent_wh):
    saved_wh  = baseline_wh - agent_wh
    saved_pct = max(0.0, (saved_wh / baseline_wh) * 100.0)
else:
    saved_wh, saved_pct = np.nan, 0.0

# ============ Anomaly Detection ============
anom_df = pd.DataFrame()
anomaly_count = 0

# Prefer model output if available
if "pred_anom" in df.columns and df["pred_anom"].astype(str).str.lower().isin(["1","true","yes"]).any():
    tmp = df[df["pred_anom"].astype(str).str.lower().isin(["1","true","yes"])].copy()
    ycol = "pump_vib_rms_g" if "pump_vib_rms_g" in tmp.columns else None
    keep = ["timestamp"]
    if ycol: keep.append(ycol)
    for c in ("anom_score","pump_current_a","pump_temp_c"):
        if c in tmp.columns: keep.append(c)
    anom_df = tmp[keep].copy()
    # severity from score if possible
    if "anom_score" in tmp.columns and tmp["anom_score"].notna().sum() >= 3:
        anom_df["severity"] = pd.qcut(tmp["anom_score"].fillna(tmp["anom_score"].median()), q=3, labels=["medium","high","critical"])
    else:
        anom_df["severity"] = "high"
    anom_df["agent"]  = "maintenance-agent"
    anom_df["action"] = "OPEN_TICKET"
    anom_df["target"] = "Pump-1"
    if ycol:
        anom_df["decision_basis"] = anom_df.apply(lambda r: f"pred_anom=1, vib={r[ycol]:.3f} g" if pd.notna(r.get(ycol, np.nan)) else "pred_anom=1", axis=1)
    else:
        anom_df["decision_basis"] = "pred_anom=1"
    anomaly_count = len(anom_df)

# Fallback heuristic if no pred_anom output
if (anom_df is None or anom_df.empty) and "pump_vib_rms_g" in df.columns:
    # rolling stats
    rs = _rolling_stats_time(df, "pump_vib_rms_g", int(roll_minutes))
    if not rs.empty:
        # Z-score
        rs["z"] = (rs["pump_vib_rms_g"] - rs["roll_mean"]) / rs["roll_std"]
        # slope ≈ Δ(roll_mean) / Δtime (hours)
        dt_h = pd.to_datetime(rs["timestamp"]).diff().dt.total_seconds() / 3600.0
        dt_h = dt_h.replace(0, np.nan)
        rs["slope"] = rs["roll_mean"].diff() / dt_h
        rs["slope"] = rs["slope"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # decision
        rs["is_anom"] = (rs["z"].abs() > z_thresh) | (rs["slope"].abs() > slope_thresh)
        out = rs.loc[rs["is_anom"], ["timestamp","pump_vib_rms_g","z","slope"]].copy()

        def _sev(row):
            if (pd.notna(row.get("z")) and abs(row["z"]) >= z_thresh + 1.5) or (pd.notna(row.get("slope")) and abs(row["slope"]) >= slope_thresh*2):
                return "critical"
            if (pd.notna(row.get("z")) and abs(row["z"]) >= z_thresh + 0.5) or (pd.notna(row.get("slope")) and abs(row["slope"]) >= slope_thresh*1.2):
                return "high"
            return "medium"

        if not out.empty:
            out["severity"] = out.apply(_sev, axis=1)
            out["agent"]  = "maintenance-agent"
            out["action"] = "OPEN_TICKET"
            out["target"] = "Pump-1"
            out["decision_basis"] = out.apply(lambda r: f"z={r['z']:.2f}, slope={r['slope']:.2f}, vib={r['pump_vib_rms_g']:.3f} g", axis=1)
            anom_df = out.copy()
            anomaly_count = len(anom_df)

# Merge optional audit log
if audit_df is not None and not audit_df.empty:
    keep = [c for c in ["timestamp","agent","action","target","severity","decision_basis"] if c in audit_df.columns]
    if keep:
        merged = audit_df[keep].copy()
        if anom_df is not None and not anom_df.empty and all(k in anom_df.columns for k in keep):
            anom_df = pd.concat([anom_df[keep], merged], ignore_index=True).drop_duplicates()
        else:
            anom_df = merged
        anomaly_count = len(anom_df)

# ============ Tabs ============
tab_overview, tab_pump, tab_light, tab_agent = st.tabs(["Overview", "Pump Monitoring", "Lighting", "Agent Insights"])

# ---- Overview ----
with tab_overview:
    st.subheader("Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline Energy (Wh)", _nice_int(baseline_wh))
    k2.metric("Agent Energy (Wh)", _nice_int(agent_wh))
    k3.metric("Energy Saved (%)", f"{saved_pct:.2f}%")
    k4.metric("Anomalies Detected", f"{anomaly_count}")

    st.markdown("### Overall Energy Saving Gauge")
    gval = max(0.0, min(25.0, saved_pct))
    fig_g = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gval,
            number={"suffix": "%"},
            title={"text": "Projected Saving vs Baseline"},
            gauge={
                "axis": {"range": [0, 25]},
                "bar": {"thickness": 0.35},
                "steps": [
                    {"range": [0, 10], "color": "#2e7d32"},
                    {"range": [10, 20], "color": "#558b2f"},
                    {"range": [20, 25], "color": "#9e9d24"},
                ],
            },
        )
    )
    st.plotly_chart(fig_g, use_container_width=True)

    if "light_wh_agent" in df.columns and df["light_wh_agent"].notna().any():
        st.markdown("### Trends — Average Daily Agent Energy (Wh)")
        daily = df.groupby("day", as_index=False)["light_wh_agent"].mean(numeric_only=True)
        st.plotly_chart(px.line(daily, x="day", y="light_wh_agent"), use_container_width=True)
    else:
        st.info("No agent lighting energy data found.")

# ---- Pump Monitoring ----
with tab_pump:
    st.subheader("Pump Vibration Monitoring")
    if "pump_vib_rms_g" not in df.columns or df["pump_vib_rms_g"].dropna().empty:
        st.info("Column 'pump_vib_rms_g' not found or empty.")
    else:
        try:
            st.markdown("#### Pump Vibration (g RMS)")
            st.plotly_chart(px.line(df, x="timestamp", y="pump_vib_rms_g").update_traces(line=dict(width=2)),
                            use_container_width=True)
        except Exception:
            st.warning("Could not plot pump vibration line chart.")

        # Rolling average display
        rs = _rolling_stats_time(df, "pump_vib_rms_g", int(roll_minutes))
        if not rs.empty and rs["roll_mean"].notna().any():
            st.markdown(f"#### Rolling Average of Vibration (window={roll_minutes} min)")
            st.plotly_chart(px.line(rs, x="timestamp", y="roll_mean", labels={"roll_mean": "roll_mean (g)"}),
                            use_container_width=True)

        # anomalies points
        if anom_df is not None and not anom_df.empty:
            st.markdown("#### Detected Anomalies")
            ycol = "pump_vib_rms_g" if "pump_vib_rms_g" in anom_df.columns else None
            try:
                st.plotly_chart(
                    px.scatter(anom_df, x="timestamp", y=ycol, color="severity",
                               hover_data=[c for c in ["z","slope","decision_basis"] if c in anom_df.columns]),
                    use_container_width=True
                )
            except Exception:
                st.caption("Anomalies available but could not be plotted (missing numeric y).")

# ---- Lighting ----
with tab_light:
    st.subheader("Cumulative Lighting Energy (Wh)")
    if {"light_wh_base","light_wh_agent"}.issubset(df.columns):
        try:
            fig_l = px.area(df, x="timestamp", y=["light_wh_base","light_wh_agent"],
                            labels={"value": "Energy (Wh)", "variable": "Source"})
            st.plotly_chart(fig_l, use_container_width=True)
        except Exception:
            st.warning("Could not draw area chart for lighting energy.")

        try:
            by_hour = df.groupby("hour", as_index=False)["light_wh_agent"].mean(numeric_only=True)
            st.markdown("#### Average Agent Energy by Hour")
            st.plotly_chart(px.bar(by_hour, x="hour", y="light_wh_agent"), use_container_width=True)
        except Exception:
            st.caption("Could not compute hourly average.")
    else:
        st.info("Expected lighting energy columns 'light_wh_base' & 'light_wh_agent' not found. "
                "If only W columns exist, they were auto-converted to Wh when possible.")

# ---- Agent Insights ----
with tab_agent:
    st.subheader("Agent Action Log")
    if anom_df is not None and not anom_df.empty:
        show_cols = [c for c in ["timestamp","agent","action","target","severity","decision_basis"] if c in anom_df.columns]
        if not show_cols:
            show_cols = [c for c in anom_df.columns if c != "index"]
        try:
            st.dataframe(anom_df[show_cols].sort_values("timestamp", ascending=False),
                         use_container_width=True, height=360)
        except Exception:
            st.dataframe(anom_df, use_container_width=True, height=360)

        # Severity distribution
        if "severity" in anom_df.columns:
            try:
                sev_cnt = anom_df.groupby(["severity"], as_index=False).size()
                st.markdown("#### Severity Distribution")
                st.plotly_chart(px.bar(sev_cnt, x="severity", y="size"), use_container_width=True)
            except Exception:
                st.caption("Could not compute severity distribution.")
    else:
        st.info("No agent actions logged yet (no anomalies with current thresholds or audit file).")

st.markdown(
    """
    <hr/>
    <div style="text-align:center; opacity:.7">
        © 2025 Agentra • Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
