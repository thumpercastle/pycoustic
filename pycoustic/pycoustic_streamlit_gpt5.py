# Python
# streamlit_pycoustic_app.py
import io
import json
import tempfile
from datetime import time

import pandas as pd
import streamlit as st

# Import pycoustic classes
from pycoustic import Log, Survey


# --------------- Helpers ---------------

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


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns for nicer display in Streamlit."""
    if isinstance(df.columns, pd.MultiIndex):
        flat = df.copy()
        flat.columns = [" / ".join(map(str, c)) for c in df.columns.to_flat_index()]
        return flat
    return df


def parse_extra_kwargs(raw: str) -> dict:
    """Parse a JSON dict from a text area. Returns {} on error."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            st.warning("Extra kwargs JSON should be an object/dict; ignoring.")
            return {}
        return parsed
    except Exception as e:
        st.warning(f"Unable to parse extra kwargs JSON. Ignoring. Error: {e}")
        return {}


# --------------- Streamlit App ---------------

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

with st.expander("2) Configure periods (survey.set_periods)", expanded=True):
    st.write("Set daily period start times. These will be passed as times=... to set_periods().")

    # Show time pickers; convert to tuples (hour, minute)
    day_t = st.time_input("Day start", value=time(ss["periods_times"]["day"][0], ss["periods_times"]["day"][1]))
    eve_t = st.time_input("Evening start", value=time(ss["periods_times"]["evening"][0], ss["periods_times"]["evening"][1]))
    night_t = st.time_input("Night start", value=time(ss["periods_times"]["night"][0], ss["periods_times"]["night"][1]))

    # Update in session
    new_times = {
        "day": (day_t.hour, day_t.minute),
        "evening": (eve_t.hour, eve_t.minute),
        "night": (night_t.hour, night_t.minute),
    }

    apply_periods = st.button("Apply periods to Survey")

    if apply_periods:
        if not ss["logs"]:
            st.warning("Load logs first.")
        else:
            ss["periods_times"] = new_times
            # Build or update Survey
            ss["survey"] = build_survey(ss["logs"], times_kwarg=ss["periods_times"])
            # Invalidate old summary
            ss["resi_df"] = None
            st.success("Periods applied to Survey.")

with st.expander("3) Compute results (survey.resi_summary)", expanded=True):
    st.write("Set kwargs for resi_summary(). Adjust lmax_n and lmax_t, and optionally pass extra kwargs as JSON.")

    col1, col2 = st.columns([1, 1])
    ss["lmax_n"] = col1.number_input("lmax_n", min_value=1, value=int(ss["lmax_n"]), step=1)
    ss["lmax_t"] = col2.number_input("lmax_t", min_value=1, value=int(ss["lmax_t"]), step=1)

    ss["extra_kwargs_raw"] = st.text_area(
        "Extra kwargs (JSON object)",
        value=ss["extra_kwargs_raw"],
        height=120,
        help="Example: {\"include_LAE\": true} (only pass valid kwargs for resi_summary)"
    )

    compute = st.button("Update resi_summary")

    if compute:
        if ss["survey"] is None:
            if not ss["logs"]:
                st.warning("Load logs first.")
            else:
                # Create Survey if missing
                ss["survey"] = build_survey(ss["logs"], times_kwarg=ss["periods_times"])

        if ss["survey"] is not None:
            kwargs = parse_extra_kwargs(ss["extra_kwargs_raw"])
            kwargs["lmax_n"] = int(ss["lmax_n"])
            kwargs["lmax_t"] = int(ss["lmax_t"])

            try:
                df = ss["survey"].resi_summary(**kwargs)
                if df is None or (hasattr(df, "empty") and df.empty):
                    st.info("resi_summary returned no data.")
                    ss["resi_df"] = None
                else:
                    ss["resi_df"] = df
                    st.success("resi_summary updated.")
            except Exception as e:
                st.error(f"resi_summary failed: {e}")

# --------------- Results ---------------
st.subheader("resi_summary results")
if ss["resi_df"] is not None:
    show_df = flatten_columns(ss["resi_df"])
    st.dataframe(show_df, use_container_width=True)

    # Download
    try:
        csv_buf = io.StringIO()
        show_df.to_csv(csv_buf)
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name="resi_summary.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"Unable to prepare CSV download: {e}")
else:
    st.info("No results yet. Load CSVs, apply periods, and compute resi_summary.")

# --------------- Utilities ---------------
with st.sidebar:
    st.header("Utilities")
    if st.button("Reset session"):
        # Clean up temp files
        for p in ss["tmp_paths"]:
            try:
                import os
                os.unlink(p)
            except Exception:
                pass
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    st.caption("Tip: After uploading and loading files, set periods, then compute resi_summary.")
