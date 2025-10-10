import os
import tempfile
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import re

# Import from the package so relative imports inside submodules resolve
try:
    from pycoustic.survey import *
except ImportError:
    # Fallback for local runs
    from survey import *

try:
    from pycoustic.log import *
except ImportError:
    from log import *

# Streamlit app config
st.set_page_config(page_title="Pycoustic Acoustic Survey Explorer", layout="wide")

# Graph colour palette config
COLOURS = {
    "Leq A": "#FBAE18",   # light grey
    "L90 A": "#4d4d4d",   # dark grey
    "Lmax A": "#fc2c2c",  # red
}
# Graph template config
TEMPLATE = "plotly"

if "apply_agg" not in st.session_state:
    st.session_state["apply_agg"] = False
if "period_last" not in st.session_state:
    st.session_state["period_last"] = ""

with st.sidebar:
    # File Upload in expander container
    with st.expander("File Upload", expanded=True):
        files = st.file_uploader(
            "Select one or more CSV files",
            type="csv",
            accept_multiple_files=True,
        )
        if not files:
            st.stop()
    # Integration period entry in expander container
    with st.expander("Integration Period", expanded=True):
        int_period = st.number_input(
            "Insert new integration period (must be larger than data)",
            step=1,
            value=15,
        )
        period_select = st.selectbox(
            "Please select time period",
            ("second(s)", "minute(s)", "hour(s)"),
            index=1,
        )

        # Build the period string
        suffix_map = {"second(s)": "s", "minute(s)": "min", "hour(s)": "h"}
        period = f"{int_period}{suffix_map.get(period_select, '')}"

        # If the period changed since last time, reset the "apply_agg" flag
        if st.session_state["period_last"] != period:
            st.session_state["apply_agg"] = False
            st.session_state["period_last"] = period

        # Button to trigger aggregation for ALL positions
        apply_agg_btn = st.button("Apply Integration Period")
        if apply_agg_btn:
            st.session_state["apply_agg"] = True

# Main Window / Data Load
with st.spinner("Processing Data...", show_time=True):
    # Load each uploaded CSV into a pycoustic Log
    logs: Dict[str, Log] = {}
    for upload_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(upload_file.getbuffer())
            path = tmp.name
        try:
            logs[upload_file.name] = Log(path)
        except Exception as err:
            st.error(f"Failed to load `{upload_file.name}` into Pycoustic: {err}")
        finally:
            os.unlink(path)

    # Build Survey and pull summary + spectra
    summary_df = leq_spec_df = lmax_spec_df = None
    summary_error = ""
    if logs:
        try:
            survey = Survey()
            if callable(getattr(survey, "add_log", None)):
                for name, lg in logs.items():
                    survey.add_log(lg, name=name)
            elif hasattr(survey, "_logs"):
                survey._logs = logs

            summary_df = survey.resi_summary()
            leq_spec_df = getattr(survey, "typical_leq_spectra", lambda: None)()
            lmax_spec_df = getattr(survey, "lmax_spectra", lambda: None)()
        except Exception as err:
            summary_error = str(err)
    else:
        summary_error = "No valid logs loaded."

    # Helper list of “position” names (i.e. filenames)
    pos_list = list(logs.keys())

    # Helper: turn a “spectra” DataFrame into a long‐format table for plotting
    def spectra_to_rows(df: pd.DataFrame, pos_names: List[str]) -> pd.DataFrame | None:
        if df is None:
            return None
        if not isinstance(df.columns, pd.MultiIndex):
            tidy = df.reset_index().rename(columns={df.index.name or "index": "Period"})
            if "Position" not in tidy.columns:
                tidy.insert(0, "Position", pos_names[0] if pos_names else "Pos1")
            return tidy

        # If there is a MultiIndex
        bands = [band for _, band in df.columns][: len({band for _, band in df.columns})]
        set_len = len(bands)
        blocks = []
        for i, pos in enumerate(pos_names):
            start, end = i * set_len, (i + 1) * set_len
            if end > df.shape[1]:
                break
            sub = df.iloc[:, start:end].copy()
            sub.columns = [str(b) for b in bands]
            sub = sub.reset_index().rename(columns={df.index.names[-1] or "index": "Period"})
            if "Position" not in sub.columns:
                sub.insert(0, "Position", pos)
            blocks.append(sub)
        return pd.concat(blocks, ignore_index=True)

    #Create tabs
    ui_tabs = st.tabs(["Summary"] + pos_list)

    #Summary tab
    with ui_tabs[0]:
        st.subheader("Broadband Summary")
        if summary_df is not None:
            st.dataframe(summary_df)
        else:
            st.warning(f"Summary unavailable: {summary_error}")

        # Plot “Typical Leq Spectra” and “Lmax Spectra”, if available
        for title, df_data in (
            ("Typical Leq Spectra", leq_spec_df),
            ("Lmax Spectra", lmax_spec_df),
        ):
            tidy = spectra_to_rows(df_data, pos_list)
            if tidy is None:
                continue

            freq_cols = [c for c in tidy.columns if c not in ("Position", "Period", "A")]
            if freq_cols:
                fig = go.Figure()
                for pos in pos_list:
                    subset = tidy[tidy["Position"] == pos]
                    for _, row in subset.iterrows():
                        period_label = row["Period"]
                        # Cast to string so .lower() is safe
                        period_label_str = str(period_label)
                        mode = (
                            "lines+markers"
                            if period_label_str.lower().startswith("day")
                            else "lines"
                        )
                        label = (
                            f"{pos} {period_label_str}"
                            if len(pos_list) > 1
                            else period_label_str
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=freq_cols,
                                y=row[freq_cols],
                                mode=mode,
                                name=label,
                            )
                        )
                fig.update_layout(
                    template=TEMPLATE,
                    title=f"{title} - Day & Night",
                    xaxis_title="Octave band (Hz)",
                    yaxis_title="dB",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No frequency columns found for `{title}`.")

    # Position‐Specific Tabs
    for tab, uf in zip(ui_tabs[1:], files):
        with tab:
            log = logs.get(uf.name)
            if log is None:
                st.error(f"Log for `{uf.name}` not found.")
                continue

            # Decide whether to show raw or aggregated data
            if st.session_state["apply_agg"]:
                # 1) Re-aggregate / resample using the chosen period
                try:
                    df_used = log.as_interval(t=period)
                    df_used = df_used.reset_index().rename(
                        columns={df_used.index.name or "index": "Timestamp"}
                    )
                    subheader = "Integrated Survey Data"
                except Exception as e:
                    st.error(f"Failed to apply integration period for `{uf.name}`: {e}")
                    continue
            else:
                # 2) Show the raw data (from log._master) if available
                try:
                    raw_master = log._master  # original DataFrame, indexed by Timestamp
                    df_used = raw_master.reset_index().rename(columns={"Time": "Timestamp"})
                    subheader = "Raw Survey Data"
                except Exception as e:
                    st.error(f"Failed to load raw data for `{uf.name}`: {e}")
                    continue

            # Prepare a flattened‐column header copy JUST FOR PLOTTING
            df_plot = df_used.copy()
            if isinstance(df_plot.columns, pd.MultiIndex):
                flattened_cols = []
                for lvl0, lvl1 in df_plot.columns:
                    lvl0_str = str(lvl0)
                    lvl1_str = str(lvl1) if lvl1 is not None else ""
                    flattened_cols.append(f"{lvl0_str} {lvl1_str}".strip())
                df_plot.columns = flattened_cols

            #  Time‐history Graph (Leq A, L90 A, Lmax A) using df_plot 
            required_cols = {"Leq A", "L90 A", "Lmax A"}
            if required_cols.issubset(set(df_plot.columns)):
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["Timestamp"],
                        y=df_plot["Leq A"],
                        name="Leq A",
                        mode="lines",
                        line=dict(color=COLOURS["Leq A"], width=1),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["Timestamp"],
                        y=df_plot["L90 A"],
                        name="L90 A",
                        mode="lines",
                        line=dict(color=COLOURS["L90 A"], width=1),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["Timestamp"],
                        y=df_plot["Lmax A"],
                        name="Lmax A",
                        mode="markers",
                        marker=dict(color=COLOURS["Lmax A"], size=3),
                    )
                )
                fig.update_layout(
                    template=TEMPLATE,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(
                        title="Time & Date (hh:mm & dd/mm/yyyy)",
                        type="date",
                        tickformat="%H:%M<br>%d/%m/%Y",
                        tickangle=0,
                    ),
                    yaxis_title="Measured Sound Pressure Level dB(A)",
                    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Required columns {required_cols} missing in {subheader}.")

            # --- Finally, display the TABLE with MultiIndex intact ---
            st.subheader(subheader)
            st.dataframe(df_used, hide_index=True)

# Python
# --- Summary tab: show Lmax spectra table ---
# Assumes the first tab in `ui_tabs` is "Summary"
try:
    summary_tab = ui_tabs[0]
except Exception:
    summary_tab = None

if summary_tab is not None:
    with summary_tab:
        st.markdown("### Lmax Spectra")
        try:
            # Prefer a precomputed dataframe if your app already produces it
            df_lmax = None
            if "lmax_spec_df" in globals() and isinstance(lmax_spec_df, pd.DataFrame):
                df_lmax = lmax_spec_df
            elif "survey" in globals() and survey is not None:
                df_lmax = survey.lmax_spectra()

            if df_lmax is not None and not df_lmax.empty:
                st.dataframe(df_lmax, use_container_width=True)
            else:
                st.info("No Lmax spectra available. Load logs and compute the survey summary first.")
        except Exception as e:
            st.warning(f"Unable to display Lmax spectra table: {e}")

        # Leq spectra table
        st.subheader("Leq spectra")
        try:
            # Prefer a cached value if you already store it
            leq_spec_df = st.session_state.get("leq_spec_df")
            if leq_spec_df is None:
                leq_spec_df = survey.leq_spectra()
                # Cache for reuse elsewhere in the UI
                st.session_state["leq_spec_df"] = leq_spec_df

            if leq_spec_df is not None and hasattr(leq_spec_df, "empty") and not leq_spec_df.empty:
                st.dataframe(leq_spec_df, use_container_width=True)
            else:
                st.info("No Leq spectra available to display.")
        except Exception as e:
            st.warning(f"Unable to display Leq spectra: {e}")

    # --- Modal table (similar to "Leq spectra") ---
        st.subheader("Modal")

        # --- Modal duration overrides (in minutes, optional) ---
        with st.expander("Modal durations (optional overrides)", expanded=False):
            st.caption('Enter values like "60min" or "15min". Leave blank to use library defaults.')
            c1, c2, c3 = st.columns(3)
            with c1:
                modal_day_t = st.text_input("day_t", value="", placeholder="60min", key="modal_day_t")
            with c2:
                modal_evening_t = st.text_input("evening_t", value="", placeholder="120min", key="modal_evening_t")
            with c3:
                modal_night_t = st.text_input("night_t", value="", placeholder="180min", key="modal_night_t")


        def _is_valid_minutes(s: str) -> bool:
            if not s:
                return False
            s_norm = s.strip().lower()
            m = re.fullmatch(r"(\d+)\s*min", s_norm)
            if not m:
                return False
            try:
                return int(m.group(1)) > 0
            except ValueError:
                return False


        def _normalize_minutes(s: str) -> str:
            # normalize to "<int>min" without spaces, lowercase
            s_norm = s.strip().lower()
            num = re.fullmatch(r"(\d+)\s*min", s_norm).group(1)
            return f"{int(num)}min"


        # Build kwargs to pass only for valid, non-empty inputs
        modal_kwargs: dict = {}
        if modal_day_t.strip():
            if _is_valid_minutes(modal_day_t):
                modal_kwargs["day_t"] = _normalize_minutes(modal_day_t)
            else:
                st.warning('day_t must be in the format "<number>min", e.g., "60min".')
        if modal_evening_t.strip():
            if _is_valid_minutes(modal_evening_t):
                modal_kwargs["evening_t"] = _normalize_minutes(modal_evening_t)
            else:
                st.warning('evening_t must be in the format "<number>min", e.g., "120min".')
        if modal_night_t.strip():
            if _is_valid_minutes(modal_night_t):
                modal_kwargs["night_t"] = _normalize_minutes(modal_night_t)
            else:
                st.warning('night_t must be in the format "<number>min", e.g., "180min".')

        if "modal_df" not in st.session_state:
            st.session_state.modal_df = None

        if survey is not None:
            try:
                st.session_state.modal_df = survey.modal(**modal_kwargs)
            except Exception as e:
                st.session_state.modal_df = None
                st.warning(f"Could not compute modal results: {e}")

        if st.session_state.modal_df is not None:
            st.dataframe(st.session_state.modal_df, use_container_width=True)
        else:
            st.info("No modal results to display.")
#testing