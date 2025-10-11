import os
import tempfile
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import pycoustic classes
from log import *
from survey import *
from weather import *

st.set_page_config(page_title="pycoustic GUI", layout="wide")
st.title("pycoustic Streamlit GUI")

# Initialize session state
ss = st.session_state
ss.setdefault("tmp_paths", [])          # List[str] for cleanup
ss.setdefault("logs", {})               # Dict[str, Log]
ss.setdefault("survey", None)           # Survey or None
ss.setdefault("resi_df", None)          # Cached summary
ss.setdefault("periods_times", {        # Default times for set_periods()
    "day": (7, 0),
    "evening": (23, 0),
    "night": (23, 0),
})
ss.setdefault("lmax_n", 5)
ss.setdefault("lmax_t", 30)
ss.setdefault("extra_kwargs_raw", "{}")


def save_upload_to_tmp(uploaded_file) -> str:
    """Persist an uploaded CSV to a temporary file and return its path."""
    # Create a persistent temporary file (delete later on reset)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


# File Upload in expander container
with st.expander("1) Load CSV data", expanded=True):
    st.write("Upload one or more CSV files to create Log objects for a single Survey.")

    uploaded = st.file_uploader(
        "Select CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Each CSV should match the expected pycoustic format."
    )

    if uploaded:
        st.caption("Assign a position name for each file (defaults to base filename).")

        # Build a list of (file, default_name) for user naming
        pos_names = []
        for idx, f in enumerate(uploaded):
            default_name = f.name.rsplit(".", 1)[0]
            name = st.text_input(
                f"Position name for file {idx + 1}: {f.name}",
                value=default_name,
                key=f"pos_name_{f.name}_{idx}"
            )
            pos_names.append((f, name.strip() or default_name))

        col_l, col_r = st.columns([1, 1])
        replace = col_l.checkbox("Replace existing survey/logs", value=True)
        load_btn = col_r.button("Load CSVs")

        if load_btn:
            if replace:
                # Reset previous state
                for p in ss["tmp_paths"]:
                    try:
                        # Cleanup files on supported OS; not critical if fails
                        import os
                        os.unlink(p)
                    except Exception:
                        pass
                ss["tmp_paths"] = []
                ss["logs"] = {}
                ss["survey"] = None
                ss["resi_df"] = None

            added = 0
            for f, pos_name in pos_names:
                try:
                    tmp_path = save_upload_to_tmp(f)
                    ss["tmp_paths"].append(tmp_path)
                    log_obj = Log(path=tmp_path)
                    ss["logs"][pos_name] = log_obj
                    added += 1
                except Exception as e:
                    st.error(f"Failed to load {f.name}: {e}")

            if added > 0:
                st.success(f"Loaded {added} file(s) into logs.")
            else:
                st.warning("No files loaded. Please check the CSV format and try again.")

    if ss["logs"]:
        st.info(f"Current logs in session: {', '.join(ss['logs'].keys())}")

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
