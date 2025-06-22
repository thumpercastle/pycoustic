import os
import tempfile
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from log import *
from survey import *

# Streamlit app config
st.set_page_config(page_title="Pycoustic Acoustic Survey Explorer", layout="wide")

# Graph colour palette config
COLOURS = {
    "Leq A": "#9e9e9e",   # light grey
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
