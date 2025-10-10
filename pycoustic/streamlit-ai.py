# Python 3.12
# Streamlit app entrypoint for PyCoustic-like workflow

from __future__ import annotations

import io
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Plotting configuration
# -----------------------------
COLOURS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
TEMPLATE: str = "plotly_white"


# -----------------------------
# Helpers
# -----------------------------
def _try_read_table(upload) -> pd.DataFrame:
    # Try CSV first, then Excel
    name = getattr(upload, "name", "uploaded")
    data = upload.read() if hasattr(upload, "read") else upload.getvalue()
    # ensure the buffer can be reused for Excel attempt
    buf = io.BytesIO(data)

    # CSV attempt
    try:
        df_csv = pd.read_csv(io.BytesIO(data))
        if not df_csv.empty:
            df_csv.attrs["__source_name__"] = name
            return df_csv
    except Exception:
        pass

    # Excel attempt
    buf.seek(0)
    try:
        df_xls = pd.read_excel(buf)
        if not df_xls.empty:
            df_xls.attrs["__source_name__"] = name
            return df_xls
    except Exception:
        pass

    raise ValueError(f"Could not parse file: {name}")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten MultiIndex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        flat = [" | ".join([str(x) for x in tup if x is not None]) for tup in df.columns]
        df = df.copy()
        df.columns = flat
    return df


def _maybe_parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # Heuristic: try common datetime column names and parse them
    dt_candidates = ["Datetime", "DateTime", "Timestamp", "Time", "Date", "Date_Time", "datetime", "time", "timestamp"]
    for col in df.columns:
        if col in dt_candidates or re.search(r"time|date|stamp", str(col), re.IGNORECASE):
            try:
                parsed = pd.to_datetime(df[col], errors="raise", utc=False, infer_datetime_format=True)
                out = df.copy()
                out[col] = parsed
                return out
            except Exception:
                continue
    return df


def _detect_position_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Position", "Pos", "Mic", "Channel", "Location", "Site"]
    for c in candidates:
        if c in df.columns:
            return c
    # also try case-insensitive exact matches
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _metric_patterns() -> Dict[str, re.Pattern]:
    # Detect wide spectral columns, e.g., "Leq_31.5", "Lmax_1000", "Leq 4k", "Lmax 1 kHz", "63 Hz"
    # Strategy:
    # - Either prefixed by a metric (Leq/Lmax) and then frequency
    # - Or pure frequency with "Hz" and a separate metric column naming is handled by selection
    def freq_part():
        # numbers like 31.5, 1000, 1k, 2 kHz, etc.
        return r"(?P<freq>(\d+(\.\d+)?)(\s*k(hz)?)?)"

    # metric-first naming: "Leq_31.5", "Lmax 1000", "Leq-1k", "Leq 1 kHz"
    metric_first = rf"^(?P<metric>Leq|Lmax)[\s_\-]*{freq_part()}(hz)?$"
    # freq-first naming: "31.5", "63 Hz", "1k", with an optional suffix metric after a sep: "63Hz_Leq"
    freq_first = rf"^{freq_part()}(hz)?[\s_\-]*(?P<metric>Leq|Lmax)?$"
    return {
        "metric_first": re.compile(metric_first, re.IGNORECASE),
        "freq_first": re.compile(freq_first, re.IGNORECASE),
    }


def _parse_freq_to_hz(freq_str: str) -> Optional[float]:
    if freq_str is None:
        return None
    s = str(freq_str).strip().lower().replace(" ", "")
    s = s.replace("khz", "k").replace("hz", "")
    # handle "1k" or "1.0k"
    m = re.match(r"^(\d+(\.\d+)?)k$", s)
    if m:
        return float(m.group(1)) * 1000.0
    try:
        return float(s)
    except Exception:
        return None


def spectra_to_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide spectral columns into a tidy long form:
    Columns like "Leq_31.5", "Lmax 63", "63 Hz Leq" -> rows with Frequency (Hz), Metric, Value.
    Non-spectral columns are carried along.
    """
    df = _flatten_columns(df)
    patterns = _metric_patterns()

    spectral_cols: List[Tuple[str, str, float]] = []  # (original_col, metric, freq_hz)
    for col in df.columns:
        col_str = str(col)
        matched = False

        # metric first
        m1 = patterns["metric_first"].match(col_str)
        if m1:
            metric = m1.group("metric").upper()
            freq_hz = _parse_freq_to_hz(m1.group("freq"))
            if metric in ("LEQ", "LMAX") and freq_hz is not None:
                spectral_cols.append((col, metric, freq_hz))
                matched = True

        if matched:
            continue

        # frequency first
        m2 = patterns["freq_first"].match(col_str)
        if m2:
            metric = m2.group("metric")
            metric = metric.upper() if metric else None
            freq_hz = _parse_freq_to_hz(m2.group("freq"))
            if freq_hz is not None:
                # If metric is not embedded, we will treat it as "LEQ" by default for plotting,
                # but also keep the column name when we pivot.
                spectral_cols.append((col, metric or "LEQ", freq_hz))

    if not spectral_cols:
        return pd.DataFrame(columns=["Frequency_Hz", "Metric", "Value"])

    # Build tidy rows
    id_cols = [c for c in df.columns if c not in [c0 for (c0, _, _) in spectral_cols]]
    tidies: List[pd.DataFrame] = []
    for (col, metric, f_hz) in spectral_cols:
        block = pd.DataFrame(
            {
                "Frequency_Hz": f_hz,
                "Metric": metric,
                "Value": df[col].astype("float64").values,
            }
        )
        # Attach IDs if present
        if id_cols:
            block = pd.concat([df[id_cols].reset_index(drop=True), block], axis=1)
        tidies.append(block)

    tidy = pd.concat(tidies, axis=0, ignore_index=True)
    # Sort by frequency numeric
    tidy = tidy.sort_values(["Metric", "Frequency_Hz"]).reset_index(drop=True)
    return tidy


def _resample_if_possible(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """
    how: '', '1min', '5min', '1H', '1D'
    """
    if not how:
        return df

    # find a datetime column
    dt_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_col = c
            break

    if dt_col is None:
        return df  # nothing to resample on

    df_sorted = df.sort_values(dt_col)
    df_sorted = df_sorted.set_index(dt_col)

    # numeric only for resample
    numeric_cols = df_sorted.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return df

    grouped = df_sorted[numeric_cols].resample(how).mean().reset_index()
    # put back other columns in a sensible way (drop or take first)
    return grouped


def _build_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if not group_cols:
        # simple numeric summary
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        s = df[numeric_cols].agg(["count", "mean", "std", "min", "max"]).T.reset_index()
        s.columns = ["Metric"] + list(s.columns[1:])
        return s

    # groupby summary on numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return pd.DataFrame()

    g = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["count", "mean", "std", "min", "max"])
    g = g.reset_index()

    # flatten resulting MultiIndex columns
    out_cols = []
    for tup in g.columns:
        if isinstance(tup, tuple):
            lvl0, lvl1 = tup
            if lvl1 == "":
                out_cols.append(str(lvl0))
            elif lvl0 in group_cols:
                out_cols.append(str(lvl0))
            else:
                out_cols.append(f"{lvl0}__{lvl1}")
        else:
            out_cols.append(str(tup))
    g.columns = out_cols
    return g


def _guess_resample_options(df: pd.DataFrame) -> List[Tuple[str, str]]:
    # Label, pandas rule
    has_dt = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
    if not has_dt:
        return [("None", "")]
    return [
        ("None", ""),
        ("1 minute", "1min"),
        ("5 minutes", "5min"),
        ("15 minutes", "15min"),
        ("Hourly", "1H"),
        ("Daily", "1D"),
    ]


def _plot_spectra(tidy_spec: pd.DataFrame, color_by: Optional[str]) -> go.Figure:
    fig = go.Figure()
    if tidy_spec.empty:
        fig.update_layout(template=TEMPLATE)
        return fig

    # X is frequency Hz (log10)
    x = tidy_spec["Frequency_Hz"].to_numpy(dtype=float)

    # determine trace grouping
    if color_by and color_by in tidy_spec.columns:
        groups = list(tidy_spec[color_by].astype(str).unique())
        for i, key in enumerate(groups):
            sub = tidy_spec[tidy_spec[color_by].astype(str) == str(key)]
            # Keep metric separated as line style if available
            if "Metric" in sub.columns and sub["Metric"].nunique() > 1:
                for metric in sorted(sub["Metric"].unique()):
                    subm = sub[sub["Metric"] == metric]
                    fig.add_trace(
                        go.Scatter(
                            x=subm["Frequency_Hz"],
                            y=subm["Value"],
                            mode="lines+markers",
                            name=f"{key} – {metric}",
                            line=dict(color=COLOURS[i % len(COLOURS)], dash="solid" if metric == "LEQ" else "dash"),
                            marker=dict(size=6),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=sub["Frequency_Hz"],
                        y=sub["Value"],
                        mode="lines+markers",
                        name=str(key),
                        line=dict(color=COLOURS[i % len(COLOURS)]),
                        marker=dict(size=6),
                    )
                )
    else:
        # single trace per metric
        if "Metric" in tidy_spec.columns:
            for i, metric in enumerate(sorted(tidy_spec["Metric"].unique())):
                sub = tidy_spec[tidy_spec["Metric"] == metric]
                fig.add_trace(
                    go.Scatter(
                        x=sub["Frequency_Hz"],
                        y=sub["Value"],
                        mode="lines+markers",
                        name=str(metric),
                        line=dict(color=COLOURS[i % len(COLOURS)]),
                        marker=dict(size=6),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=tidy_spec["Value"],
                    mode="lines+markers",
                    name="Spectrum",
                    line=dict(color=COLOURS[0]),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        template=TEMPLATE,
        xaxis=dict(
            type="log",
            title="Frequency (Hz)",
            tickvals=[31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000],
            ticktext=["31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k"],
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(title="Level (dB)", gridcolor="rgba(0,0,0,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=40, b=50),
    )
    return fig


def _download_csv_button(label: str, df: pd.DataFrame, key: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=f"{key}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------
# UI
# -----------------------------
def main():
    st.set_page_config(page_title="PyCoustic – Streamlit", layout="wide")

    st.title("PyCoustic – Streamlit App")
    st.caption("Upload measurement logs, explore summaries and spectra, and export results.")

    with st.sidebar:
        st.header("Inputs")
        uploads = st.file_uploader(
            "Upload one or more files (CSV or Excel)",
            type=["csv", "txt", "xlsx", "xls"],
            accept_multiple_files=True,
        )

        st.markdown("---")
        st.subheader("Options")

        # These options approximate a typical workflow
        resample_label = "Resample period"
        resample_options: List[Tuple[str, str]] = [("None", "")]
        # temporarily show None; we'll refine after reading a file

        # placeholder UI to avoid reflow
        resample_choice_label = st.selectbox(resample_label, [x[0] for x in resample_options], index=0, key="resample_placeholder")

        group_hint = st.text_input("Optional Group Column (e.g., Position/Location)", value="")

        st.markdown("---")
        st.subheader("Spectra")
        colour_by = st.text_input("Colour lines by column (e.g., Position or source)", value="source")
        show_markers = st.checkbox("Show markers", value=True)

        st.markdown("---")
        st.subheader("Display")
        show_raw_preview = st.checkbox("Show raw preview table", value=False)

    if not uploads:
        st.info("Upload files to begin.")
        return

    # Read and combine
    logs: List[str] = []
    frames: List[pd.DataFrame] = []
    for uf in uploads:
        try:
            df = _try_read_table(uf)
            df = _flatten_columns(df)
            df = _maybe_parse_datetime(df)
            df["source"] = getattr(uf, "name", df.attrs.get("__source_name__", "uploaded"))
            frames.append(df)
            logs.append(f"Loaded: {df['source'].iloc[0]} (rows={len(df)}, cols={len(df.columns)})")
        except Exception as e:
            logs.append(f"Error reading {getattr(uf, 'name', '?')}: {e}")

    if not frames:
        st.error("No readable files.")
        st.text_area("Logs", value="\n".join(logs), height=160)
        return

    raw_master = pd.concat(frames, axis=0, ignore_index=True)
    # Now that we have data, rebuild resample options
    resample_options = _guess_resample_options(raw_master)
    resample_choice_label = st.sidebar.selectbox(
        "Resample period",
        [x[0] for x in resample_options],
        index=0,
        key="resample_choice",
    )
    resample_rule = dict(resample_options)[resample_choice_label]

    # Optional resampling
    df_used = _resample_if_possible(raw_master, resample_rule) if resample_rule else raw_master

    # Try to determine a reasonable group column
    detected_group_col = _detect_position_col(df_used)
    group_col = (st.sidebar.text_input("Detected Group Column", value=detected_group_col or "") or "").strip()
    if not group_col and group_hint.strip():
        group_col = group_hint.strip()
    if group_col and group_col not in df_used.columns:
        st.warning(f"Group column '{group_col}' not found. It will be ignored.")
        group_col = ""

    # Build spectra in tidy form
    tidy_spec = spectra_to_rows(df_used)

    tabs = st.tabs(["Summary", "Spectra", "Raw Data", "Logs"])

    # Summary tab
    with tabs[0]:
        st.subheader("Summary")
        group_cols: List[str] = []
        if group_col:
            group_cols.append(group_col)
        if "source" in df_used.columns:
            group_cols.append("source")

        summary_df = _build_summary(df_used, group_cols)
        if summary_df.empty:
            st.info("No numeric data to summarize.")
        else:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            _download_csv_button("Download summary CSV", summary_df, "summary")

    # Spectra tab
    with tabs[1]:
        st.subheader("Spectra")

        # Allow user to filter a group (optional)
        filters_cols = []
        if group_col:
            filters_cols.append(group_col)
        if "source" in tidy_spec.columns:
            filters_cols.append("source")

        sub = tidy_spec.copy()
        # Dynamic filters
        if not sub.empty and filters_cols:
            cols = st.columns(len(filters_cols))
            for i, colname in enumerate(filters_cols):
                with cols[i]:
                    uniq = sorted([str(x) for x in sub[colname].dropna().unique()])
                    if len(uniq) <= 1:
                        continue
                    selected = st.multiselect(f"Filter {colname}", options=uniq, default=uniq)
                    sub = sub[sub[colname].astype(str).isin(selected)]

        # Plot
        if sub.empty:
            st.info("No spectral data detected in the uploaded files.")
        else:
            fig = _plot_spectra(sub, color_by=(colour_by if colour_by in sub.columns else None))
            if not show_markers:
                for tr in fig.data:
                    tr.mode = "lines"
            st.plotly_chart(fig, use_container_width=True)

            # Download tidy spectra
            _download_csv_button("Download tidy spectra CSV", sub, "spectra_tidy")

    # Raw Data tab
    with tabs[2]:
        st.subheader("Raw Data")
        if show_raw_preview:
            st.dataframe(raw_master, use_container_width=True, hide_index=True)
        else:
            st.caption("Enable 'Show raw preview table' in the sidebar to render the full table.")
        _download_csv_button("Download combined raw CSV", raw_master, "raw_combined")

    # Logs tab
    with tabs[3]:
        st.subheader("Logs")
        st.text_area("Ingestion log", value="\n".join(logs), height=240, label_visibility="collapsed")


if __name__ == "__main__":
    main()
