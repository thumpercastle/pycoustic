# streamlit-ai.py
# Streamlit UI for inspecting noise survey data and plots

from __future__ import annotations

import os
import io
import re
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pycoustic import Log  # expects a Log class exposing a .df with a DateTimeIndex

# Plot styling
COLOURS: List[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
TEMPLATE: str = "plotly_white"


def _try_read_table(
    file_like_or_bytes: io.BytesIO | bytes | str,
    filename: Optional[str] = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Attempt to read a tabular file (CSV preferred; Excel fallback).
    - file_like_or_bytes may be a path, raw bytes, or a BytesIO-like object.
    - filename helps determine the extension when not a path.
    """
    def _ensure_buffer(obj) -> io.BytesIO:
        if isinstance(obj, (bytes, bytearray)):
            return io.BytesIO(obj)
        if hasattr(obj, "read"):
            return obj  # assume file-like opened in binary mode
        if isinstance(obj, str):
            # Path-like; will be handled by pandas directly, so we return None
            return None
        raise TypeError("Unsupported input type for _try_read_table")

    ext = None
    if isinstance(file_like_or_bytes, str):
        lower = file_like_or_bytes.lower()
        if lower.endswith(".csv"):
            ext = ".csv"
        elif lower.endswith((".xlsx", ".xlsm", ".xlsb", ".xls")):
            ext = ".xlsx"
    elif filename:
        lower = filename.lower()
        if lower.endswith(".csv"):
            ext = ".csv"
        elif lower.endswith((".xlsx", ".xlsm", ".xlsb", ".xls")):
            ext = ".xlsx"

    # Prefer CSV
    if ext in (None, ".csv"):
        try:
            if isinstance(file_like_or_bytes, str):
                df = pd.read_csv(file_like_or_bytes, encoding=encoding)
            else:
                buf = _ensure_buffer(file_like_or_bytes)
                if buf is None:  # pragma: no cover
                    raise ValueError("Failed to open buffer for CSV")
                df = pd.read_csv(buf, encoding=encoding)
            return _flatten_columns(df)
        except Exception:
            # Try Excel as fallback
            pass

    # Excel fallback
    if isinstance(file_like_or_bytes, str):
        df = pd.read_excel(file_like_or_bytes)
    else:
        buf = _ensure_buffer(file_like_or_bytes)
        if buf is None:  # pragma: no cover
            raise ValueError("Failed to open buffer for Excel")
        df = pd.read_excel(buf)
    return _flatten_columns(df)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns into simple strings.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join([str(p) for p in tup if p is not None]).strip() for tup in df.columns.values]
    else:
        # Ensure all columns are strings, stripped
        df = df.rename(columns=lambda c: str(c).strip())
    return df


def _maybe_parse_datetime(
    df: pd.DataFrame,
    candidates: Iterable[str] = ("timestamp", "time", "date", "datetime"),
    utc: bool = False,
) -> pd.DataFrame:
    """
    If a plausible timestamp column exists, convert to DatetimeIndex.
    Does nothing if index is already DatetimeIndex.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.copy()
    lower_cols = {str(c).lower(): c for c in df.columns}
    for key in candidates:
        if key in lower_cols:
            col = lower_cols[key]
            try:
                ts = pd.to_datetime(df[col], utc=utc, errors="raise")
                df = df.set_index(ts)
                df.index.name = "timestamp"
                return df.drop(columns=[col])
            except Exception:
                continue
    return df


def _detect_position_col(df: pd.DataFrame) -> Optional[str]:
    """
    Attempt to find a 'position' or 'location' column.
    """
    patterns = [
        re.compile(r"\bposition\b", re.I),
        re.compile(r"\bpos\b", re.I),
        re.compile(r"\blocation\b", re.I),
        re.compile(r"\bsite\b", re.I),
    ]
    for c in df.columns:
        name = str(c)
        for pat in patterns:
            if pat.search(name):
                return c
    return None


def _metric_patterns() -> Dict[str, re.Pattern]:
    """
    Patterns for commonly used acoustic metrics.
    """
    # Matches LAeq, Leq A, Leq(A), etc.
    return {
        "Leq": re.compile(r"\bL\s*eq\b(?:\s*\(?\s*A\s*\)?)?", re.I),
        "Lmax": re.compile(r"\bL\s*max\b(?:\s*\(?\s*A\s*\)?)?", re.I),
        "L90": re.compile(r"\bL\s*90\b(?:\s*\(?\s*A\s*\)?)?", re.I),
    }


def _parse_freq_to_hz(label: str) -> Optional[float]:
    """
    Parse frequency-like column labels such as '63 Hz', '1k', '1 kHz', etc.
    """
    s = str(label).strip().lower()
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(k|khz|hz)?\s*$", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if not unit or unit == "hz":
        return val
    return val * 1000.0


def spectra_to_rows(
    df: pd.DataFrame,
    value_col_name: str = "Level (dB)",
) -> Optional[pd.DataFrame]:
    """
    Convert wide spectra columns (e.g., '63 Hz', '1 kHz') into (freq_hz, value) rows.
    Returns None if no spectra-like columns are detected.
    """
    freq_cols: List[Tuple[str, float]] = []
    for c in df.columns:
        hz = _parse_freq_to_hz(c)
        if hz is not None:
            freq_cols.append((c, hz))

    if not freq_cols:
        return None

    freq_cols_sorted = sorted(freq_cols, key=lambda x: x[1])
    melted = df[fname_list := [c for c, _ in freq_cols_sorted]].copy()
    melted.columns = [f"{hz:.6g}" for _, hz in freq_cols_sorted]
    out = melted.melt(var_name="freq_hz", value_name=value_col_name)
    out["freq_hz"] = out["freq_hz"].astype(float)
    return out.sort_values("freq_hz").reset_index(drop=True)


def _resample_if_possible(df: pd.DataFrame, rule: str = "1S") -> pd.DataFrame:
    """
    Downsample a DateTimeIndex DataFrame for responsiveness.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        return df
    return df[numeric_cols].resample(rule).median().dropna(how="all")


def _build_summary(
    df: pd.DataFrame,
    position_col: Optional[str] = None,
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Build simple summary stats for selected metrics, optionally grouped by position.
    """
    if metrics is None:
        metrics = ["Leq", "Lmax", "L90"]

    pats = _metric_patterns()
    metric_cols: Dict[str, List[str]] = {m: [] for m in metrics}
    for c in df.columns:
        for m in metrics:
            if pats.get(m) and pats[m].search(str(c)):
                metric_cols[m].append(c)

    work = {}
    for m, cols in metric_cols.items():
        sel = df[cols].select_dtypes(include=[np.number]) if cols else pd.DataFrame()
        if not sel.empty:
            work[m] = sel.mean(axis=1)

    if not work:
        return pd.DataFrame()

    agg_df = pd.DataFrame(work)
    if position_col and position_col in df.columns:
        agg_df[position_col] = df[position_col].values[: len(agg_df)]
        g = agg_df.groupby(position_col, dropna=False).agg(["mean", "min", "max", "count"])
        # Flatten columns
        g.columns = [" ".join([str(p) for p in col if p]).strip() for col in g.columns.values]
        return g.reset_index()
    else:
        return agg_df.agg(["mean", "min", "max", "count"]).T.reset_index(names=["Metric"])


def _guess_resample_options(n: int) -> str:
    """
    Heuristically pick a resampling rule based on number of points.
    """
    if n > 200_000:
        return "10S"
    if n > 50_000:
        return "5S"
    if n > 10_000:
        return "2S"
    return "1S"


def _plot_spectra(
    df_rows: pd.DataFrame,
    title: str = "Spectral Levels",
    value_col_name: str = "Level (dB)",
) -> go.Figure:
    """
    Plot spectra contained in a (freq_hz, value) rows dataframe.
    """
    fig = go.Figure()
    if {"freq_hz", value_col_name}.issubset(df_rows.columns):
        fig.add_trace(
            go.Scatter(
                x=df_rows["freq_hz"],
                y=df_rows[value_col_name],
                mode="lines+markers",
                line=dict(color=COLOURS[0]),
                name=value_col_name,
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title="Frequency (Hz)",
        yaxis_title=value_col_name,
        hovermode="x",
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
    )
    fig.update_xaxes(type="log", tickformat=".0f")
    return fig


def _download_csv_button(label: str, df: pd.DataFrame, file_name: str) -> None:
    """
    Render a download button for a dataframe as CSV.
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
    )


# === New helpers for DateTimeIndex-based Log flow ===

@st.cache_data(show_spinner=False)
def _load_log_df_from_bytes(file_bytes: bytes, file_suffix: str = ".csv") -> pd.DataFrame:
    """
    Persist uploaded bytes to a temp file and create a Log to obtain a DataFrame.
    Requires Log().df to have a DateTimeIndex.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        log = Log(path=tmp_path)
        if hasattr(log, "df") and isinstance(log.df, pd.DataFrame):
            df = log.df.copy()
        else:
            raise ValueError("Log() did not expose a DataFrame on .df")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected Log().df to have a DateTimeIndex for timestamps.")
    return df


def _find_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Locate the canonical metric columns using case-insensitive matching.
    Returns a mapping from UI label to actual column name.
    """
    targets: List[str] = ["Leq A", "Lmax A", "L90 A"]
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    found: Dict[str, str] = {}
    for label in targets:
        key = label.lower()
        if key in norm_map:
            found[label] = norm_map[key]
    return found


def _build_time_history_figure(df: pd.DataFrame, series_map: Dict[str, str]) -> go.Figure:
    """
    Build a Plotly figure for time history lines using the DataFrame's DateTimeIndex.
    """
    fig = go.Figure()
    for i, (label, col) in enumerate(series_map.items()):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=label,
                line=dict(color=COLOURS[i % len(COLOURS)]),
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        xaxis_title="Timestamp",
        yaxis_title="Level (dB)",
        hovermode="x unified",
        legend_title_text="Series",
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="pycoustic - Noise survey toolkit", layout="wide")
    st.title("Noise survey toolkit")

    st.caption("Upload logs to visualize time histories. CSV files are supported.")

    # === Time history plots based on Log().df with a DateTimeIndex ===
    with st.expander("Time history plots (Leq A, Lmax A, L90 A)", expanded=True):
        uploaded = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="time_history_logs_uploader",
        )

        if uploaded:
            # User options
            max_points = st.number_input("Max points per series (downsampling)", 1_000, 1_000_000, value=10_000, step=1_000)
            for file in uploaded:
                st.markdown(f"**File:** {file.name}")

                # Load via Log() and enforce DateTimeIndex
                try:
                    df = _load_log_df_from_bytes(file.getbuffer(), file_suffix=".csv")
                except Exception as e:
                    st.error(f"Could not load Log() from file: {e}")
                    continue

                if df.empty:
                    st.warning("No data available.")
                    continue

                # Ensure sorted index
                df = df.sort_index()

                # Identify standard metric columns
                available = _find_metric_columns(df)
                missing = [label for label in ("Leq A", "Lmax A", "L90 A") if label not in available]

                if missing:
                    st.info(f"Missing columns: {', '.join(missing)}")

                if not available:
                    st.warning("None of the required columns were found. Expected any of: Leq A, Lmax A, L90 A.")
                    continue

                # Optional downsampling for responsiveness
                if len(df) > max_points:
                    step = max(1, len(df) // int(max_points))
                    df_plot = df.iloc[::step].copy()
                else:
                    df_plot = df

                # Plot
                fig = _build_time_history_figure(df_plot, available)
                st.plotly_chart(fig, use_container_width=True)

                # Simple summary table per visible series
                with st.expander("Summary (visible series)"):
                    numeric_summary = df_plot[list(available.values())].describe().T.reset_index(names=["Series"])
                    _download_csv_button("Download summary CSV", numeric_summary, f"{os.path.splitext(file.name)[0]}_summary.csv")
                    st.dataframe(numeric_summary, use_container_width=True)

    # Placeholder for additional tools/sections (spectra, etc.) can go below.
    # Existing/legacy time-history upload/plotting sections should be removed to avoid duplication.


if __name__ == "__main__":
    main()