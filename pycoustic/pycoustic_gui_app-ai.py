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

# Python
import datetime as _dt
import streamlit as st

def _set_periods_on_survey(survey, day_start, eve_start, night_start):
    """
    Try to set periods on the Survey object in a way that's compatible with your Survey/Log API.
    Accepts HH:MM strings for day/evening/night starts.
    """
    if survey is None:
        return

    # Prefer a Survey-level setter if available
    if hasattr(survey, "set_periods") and callable(getattr(survey, "set_periods")):
        try:
            survey.set_periods(day_start=day_start, eve_start=eve_start, night_start=night_start)
            return
        except Exception as e:
            st.sidebar.warning(f"set_periods failed on Survey: {e}")

    # Next, try a Survey-level set_period_times
    if hasattr(survey, "set_period_times") and callable(getattr(survey, "set_period_times")):
        try:
            survey.set_period_times(day_start, eve_start, night_start)
            return
        except Exception as e:
            st.sidebar.warning(f"set_period_times failed on Survey: {e}")

    # Fallback: set on each Log if available
    try:
        logs = getattr(survey, "_logs", None)
        if isinstance(logs, dict):
            for _k, log in logs.items():
                if hasattr(log, "set_period_times") and callable(getattr(log, "set_period_times")):
                    log.set_period_times(day_start, eve_start, night_start)
        else:
            st.sidebar.info("Survey logs not accessible; cannot set periods per-log.")
    except Exception as e:
        st.sidebar.error(f"Unable to apply periods to logs: {e}")


def render_sidebar_set_periods():
    """
    Sidebar UI to choose Day/Evening/Night boundaries.
    Applies changes to the Survey stored in st.session_state['survey'] so that
    render_resi_summary uses the updated periods.
    """
    st.sidebar.subheader("Assessment Periods")

    # Defaults commonly used: Day 07:00–19:00, Evening 19:00–23:00, Night 23:00–07:00
    default_day = _dt.time(7, 0)
    default_eve = _dt.time(19, 0)
    default_night = _dt.time(23, 0)

    # Use session defaults if already selected earlier
    day_t = st.sidebar.time_input("Day starts", value=st.session_state.get("periods_day", default_day), step=300)
    eve_t = st.sidebar.time_input("Evening starts", value=st.session_state.get("periods_eve", default_eve), step=300)
    night_t = st.sidebar.time_input("Night starts", value=st.session_state.get("periods_night", default_night), step=300)

    # Persist chosen values in session state
    st.session_state["periods_day"] = day_t
    st.session_state["periods_eve"] = eve_t
    st.session_state["periods_night"] = night_t

    # Also apply immediately to the current Survey instance (if present)
    survey = st.session_state.get("survey")
    if survey is not None:
        # Convert time objects to "HH:MM" strings (commonly accepted by survey APIs)
        day_s = f"{day_t.hour:02d}:{day_t.minute:02d}"
        eve_s = f"{eve_t.hour:02d}:{eve_t.minute:02d}"
        night_s = f"{night_t.hour:02d}:{night_t.minute:02d}"

        _set_periods_on_survey(survey, day_s, eve_s, night_s)
        # Store back (not strictly necessary if mutated in-place, but keeps things explicit)
        st.session_state["survey"] = survey

        st.sidebar.caption(f"Applied periods: Day {day_s}, Eve {eve_s}, Night {night_s}")
    else:
        st.sidebar.info("Load a survey to apply period changes.")

# --- helper: sidebar UI to set Survey periods ---
# def render_sidebar_set_periods(survey):
#     # Defaults matching the example: day=07:00, evening=23:00, night=23:00
#     if "sp_times" not in st.session_state:
#         st.session_state["sp_times"] = {"day": (7, 0), "evening": (23, 0), "night": (23, 0)}
#
#     times = st.session_state["sp_times"]
#     day_h0, day_m0 = times["day"]
#     eve_h0, eve_m0 = times["evening"]
#     nig_h0, nig_m0 = times["night"]
#
#     st.sidebar.markdown("### Set periods")
#     st.sidebar.caption("Choose start times for Day, Evening, and Night. Set Evening equal to Night to disable it.")
#
#     hours = list(range(24))
#     minutes = list(range(60))
#
#     # Day
#     st.sidebar.write("Day start")
#     d_h = st.sidebar.selectbox("Hour (Day)", hours, index=day_h0, key="sp_day_h")
#     d_m = st.sidebar.selectbox("Minute (Day)", minutes, index=day_m0, key="sp_day_m")
#
#     # Evening
#     st.sidebar.write("Evening start")
#     e_h = st.sidebar.selectbox("Hour (Evening)", hours, index=eve_h0, key="sp_eve_h")
#     e_m = st.sidebar.selectbox("Minute (Evening)", minutes, index=eve_m0, key="sp_eve_m")
#
#     # Night
#     st.sidebar.write("Night start")
#     n_h = st.sidebar.selectbox("Hour (Night)", hours, index=nig_h0, key="sp_nig_h")
#     n_m = st.sidebar.selectbox("Minute (Night)", minutes, index=nig_m0, key="sp_nig_m")
#
#     # Validation: 'night' hour must be strictly after 'day' hour
#     valid = n_h > d_h
#     if not valid:
#         st.sidebar.error("Night hour must be strictly after Day hour.")
#
#     apply_clicked = st.sidebar.button("Apply periods", use_container_width=True, disabled=not valid)
#
#     # Persist current selections
#     st.session_state["sp_times"] = {
#         "day": (int(d_h), int(d_m)),
#         "evening": (int(e_h), int(e_m)),
#         "night": (int(n_h), int(n_m)),
#     }
#
#     if apply_clicked:
#         times = st.session_state["sp_times"]
#         try:
#             survey.set_periods(times=times)
#             st.sidebar.success(
#                 f"Applied: Day {times['day'][0]:02d}:{times['day'][1]:02d}, "
#                 f"Evening {times['evening'][0]:02d}:{times['evening'][1]:02d}, "
#                 f"Night {times['night'][0]:02d}:{times['night'][1]:02d}"
#             )
#         except Exception as e:
#             st.sidebar.error(f"Failed to set periods: {e}")

# Python
def render_resi_summary(survey):
    """
    Render the Residential Summary (survey.resi_summary) in the Streamlit GUI.
    Includes options for lmax_n, lmax_t and optional advanced inputs for leq_cols/max_cols.
    """
    import ast
    import streamlit as st

    st.header("Broadband Summary")

    if survey is None:
        st.info("No survey loaded.")
        return

    with st.expander("Options", expanded=False):
        lmax_n = st.number_input("Nth-highest Lmax (lmax_n)", min_value=1, max_value=1000, value=10, step=1)
        lmax_t_choice = st.selectbox(
            "Lmax time basis (lmax_t)",
            options=["2min", "1min", "5min", "15min", "60min", "custom"],
            index=0,
            help="Select the time aggregation used to compute Lmax rankings."
        )
        if lmax_t_choice == "custom":
            lmax_t = st.text_input("Custom time basis (e.g., '30s', '10min')", value="2min")
        else:
            lmax_t = lmax_t_choice

        advanced = st.checkbox("Advanced column selection (leq_cols, max_cols)")
        leq_cols = None
        max_cols = None

        if advanced:
            st.caption("Provide lists of tuples. Example: [(\"Leq\",\"A\"), (\"L90\",\"125\")]")
            leq_text = st.text_input("leq_cols", value="")
            max_text = st.text_input("max_cols", value="")
            parse_errors = []

            def _parse_tuple_list(s):
                if not s.strip():
                    return None
                try:
                    val = ast.literal_eval(s)
                    if not isinstance(val, (list, tuple)):
                        raise ValueError("Expected a list/tuple of tuples")
                    # Coerce to list of tuples
                    parsed = []
                    for item in val:
                        if not isinstance(item, (list, tuple)) or len(item) != 2:
                            raise ValueError("Each entry must be a 2-tuple like (name, subname)")
                        parsed.append((str(item[0]), str(item[1])))
                    return parsed
                except Exception as e:
                    parse_errors.append(str(e))
                    return None

            leq_cols = _parse_tuple_list(leq_text)
            max_cols = _parse_tuple_list(max_text)

            if parse_errors:
                st.warning("There were issues parsing advanced inputs: " + "; ".join(parse_errors))

    run = st.button("Compute residential summary", use_container_width=True)

    if run:
        try:
            with st.spinner("Computing summary..."):
                df = survey.resi_summary(
                    leq_cols=leq_cols,
                    max_cols=max_cols,
                    lmax_n=int(lmax_n),
                    lmax_t=str(lmax_t),
                )

            if df is None:
                st.info("No data returned.")
                return

            st.success(f"Summary computed. Rows: {getattr(df, 'shape', ['?','?'])[0]}, Columns: {getattr(df, 'shape', ['?','?'])[1]}")
            st.dataframe(df, use_container_width=True, height=480)

            csv_bytes = df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name="resi_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Failed to compute residential summary: {e}")

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


#test
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
#test
    # Build Survey and pull summary + spectra
    summary_df = leq_spec_df = lmax_spec_df = None
    summary_error = ""
    if logs:
        try:
            survey = Survey()
            st.session_state["survey"] = survey
            if callable(getattr(survey, "add_log", None)):
                for name, lg in logs.items():
                    survey.add_log(lg, name=name)
            elif hasattr(survey, "_logs"):
                survey._logs = logs

            summary_df = survey.resi_summary()
            leq_spec_df = getattr(survey, "typical_leq_spectra", lambda: None)()
            lmax_spec_df = getattr(survey, "lmax_spectra", lambda: None)()
            render_sidebar_set_periods()

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

        survey = st.session_state.get("survey")
        if survey is None:
            st.warning("Survey not found in session_state['survey']. Ensure you set it after loading.")
        render_resi_summary(survey)

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



# --- Summary tab: show Lmax spectra table ---
# Assumes the first tab in `ui_tabs` is "Summary"
try:
    summary_tab = ui_tabs[0]
except Exception:
    summary_tab = None

if summary_tab is not None:
    with summary_tab:
        st.markdown("### Lmax Spectra")
        # Lmax spectra controls and live update
        if "lmax_kwargs" not in st.session_state:
            st.session_state.lmax_kwargs = {"n": 10, "t": "2min", "period": "nights"}

        st.markdown("##Lmax Spectra Parameters")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_val = st.number_input(
                "Nth-highest (n)",
                min_value=1,
                step=1,
                value=int(st.session_state.lmax_kwargs.get("n", 10)),
                key="lmax_n_input",
            )
        with col2:
            t_val = st.text_input(
                "Aggregation period (t)",
                value=st.session_state.lmax_kwargs.get("t", "2min"),
                key="lmax_t_input",
                help='Examples: "1min", "2min", "5min"',
            )
        with col3:
            period_options = ["days", "evenings", "nights"]
            default_period = st.session_state.lmax_kwargs.get("period", "nights")
            try:
                default_idx = period_options.index(default_period)
            except ValueError:
                default_idx = period_options.index("nights")
            period_val = st.selectbox(
                "Time window",
                options=period_options,
                index=default_idx,
                key="lmax_period_select",
            )

        # Update kwargs from UI
        st.session_state.lmax_kwargs = {
            "n": int(n_val),
            "t": t_val.strip(),
            "period": period_val,
        }

        # Compute and store df_lmax so changes reflect immediately in Streamlit Cloud
        df_lmax = None
        try:
            df_lmax = survey.lmax_spectra(**st.session_state.lmax_kwargs)
            st.session_state.df_lmax = df_lmax
            st.success("Lmax spectra updated.")
        except Exception as e:
            st.session_state.df_lmax = None
            st.error(f"Failed to compute Lmax spectra: {e}")

        # Display
        if st.session_state.df_lmax is not None:
            st.dataframe(st.session_state.df_lmax)

        #     # Load previous values if set; otherwise provide sensible defaults
        #     defaults = st.session_state.get("lmax_args", {"n": 10, "t": "2min", "period": "days"})
        #     c1, c2, c3 = st.columns(3)
        #
        #     n_val = c1.number_input("n (integer)", min_value=1, step=1, value=int(defaults["n"]))
        #     t_val = c2.text_input('t (e.g., "2min")', value=str(defaults["t"]))
        #     period_options = ["days", "evenings", "nights"]
        #     period_val = c3.selectbox(
        #         "period",
        #         options=period_options,
        #         index=period_options.index(defaults.get("period", "days")),
        #     )
        #
        #     if st.button("Apply Lmax parameters", type="primary"):
        #         st.session_state["lmax_args"] = {"n": int(n_val), "t": t_val, "period": period_val}
        #         try:
        #             lmax_df = survey.lmax_spectra(n=int(n_val), t=t_val, period=period_val)
        #             st.session_state["lmax_spec_df"] = lmax_df
        #             st.success("Lmax spectra updated.")
        #         except Exception as e:
        #             st.error(f"Error computing Lmax spectra: {e}")
        #
        # try:
        #     # Prefer a precomputed dataframe if your app already produces it
        #     df_lmax = None
        #     if "lmax_spec_df" in globals() and isinstance(lmax_spec_df, pd.DataFrame):
        #         df_lmax = lmax_spec_df
        #     elif "survey" in globals() and survey is not None:
        #         df_lmax = survey.lmax_spectra()
        #
        #     if df_lmax is not None and not df_lmax.empty:
        #         st.dataframe(df_lmax, use_container_width=True)
        #     else:
        #         st.info("No Lmax spectra available. Load logs and compute the survey summary first.")
        # except Exception as e:
        #     st.warning(f"Unable to display Lmax spectra table: {e}")

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