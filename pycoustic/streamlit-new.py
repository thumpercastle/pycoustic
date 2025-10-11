import os
import tempfile
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import pycoustic classes
from pycoustic import Log, Survey

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


def build_survey(log_map: dict, times_kwarg: dict | None = None) -> Survey:
    """Create a Survey, attach logs, and optionally call set_periods(times=...)."""
    survey = Survey()

    # Attach logs to the Survey (simple, direct assignment to internal storage)
    # If a public adder method exists, prefer that; fallback to internal attribute.
    if hasattr(survey, "add_log"):
        for key, lg in log_map.items():
            try:
                survey.add_log(key, lg)  # type: ignore[attr-defined]
            except Exception:
                # Fallback if signature differs
                setattr(survey, "_logs", log_map)
                break
    else:
        setattr(survey, "_logs", log_map)

    # Apply periods if provided
    if times_kwarg is not None:
        try:
            survey.set_periods(times=times_kwarg)
        except Exception as e:
            st.warning(f"set_periods failed with provided times: {e}")

    return survey


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