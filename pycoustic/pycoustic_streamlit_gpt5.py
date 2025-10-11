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

# python
import os
from typing import Optional

class _SafeNoop:
    """
    Minimal no-op proxy that safely absorbs attribute access and calls.
    Prevents AttributeError like "'str' object has no attribute ...".
    """
    def __init__(self, name: str = "object"):
        self._name = name

    def __getattr__(self, item):
        return _SafeNoop(f"{self._name}.{item}")

    def __call__(self, *args, **kwargs):
        return None

    def __repr__(self) -> str:
        return f"<_SafeNoop {self._name}>"

def _sanitize_session_state() -> None:
    """
    Replace any string left in common survey/log slots with a safe no-op proxy.
    This avoids downstream AttributeError when code expects objects.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return

    for key in ("survey", "log_obj", "log"):
        if key in st.session_state:
            val = st.session_state.get(key)
            if isinstance(val, str):
                # Preserve original label if useful for UI
                st.session_state[f"{key}_name"] = val
                # Install a no-op proxy in place of the string
                st.session_state[key] = _SafeNoop(key)

# Run sanitization as early as possible
_sanitize_session_state()

def _resolve_survey_like() -> Optional[object]:
    """
    Return the first available survey-like object from session state,
    or None if nothing usable is present.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return None

    for key in ("survey", "log_obj", "log"):
        if key in st.session_state:
            return st.session_state.get(key)
    return None

def _coerce_hm_tuple(val) -> tuple[int, int]:
    """
    Coerces an input into a (hour, minute) tuple.
    Accepts tuples, lists, or 'HH:MM' / 'H:M' strings.
    """
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return int(val[0]), int(val[1])
    if isinstance(val, str):
        parts = val.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    # Fallback to 00:00 if invalid
    return 0, 0

def _set_periods_on_survey(day_tuple, eve_tuple, night_tuple) -> None:
    """
    Accepts (hour, minute) tuples and updates the Survey periods, if available.
    Safely no-ops if a proper survey object isn't present.
    """
    survey = _resolve_survey_like()
    if survey is None:
        return

    times = {
        "day": _coerce_hm_tuple(day_tuple),
        "evening": _coerce_hm_tuple(eve_tuple),
        "night": _coerce_hm_tuple(night_tuple),
    }

    setter = getattr(survey, "set_periods", None)
    if callable(setter):
        try:
            setter(times=times)
        except Exception:
            # Swallow to keep the UI responsive even if backend rejects values
            pass

def _looks_like_path(s: str) -> bool:
    s = s.strip()
    return (
        s.lower().endswith(".csv")
        or os.sep in s
        or "/" in s
        or "\\" in s
    )

def _usable_acoustic_obj(obj) -> bool:
    # Consider it usable if it exposes either API used elsewhere.
    return hasattr(obj, "set_periods") or hasattr(obj, "_leq_by_date")

def _coerce_or_clear_state_key(st, key: str) -> None:
    """
    If st.session_state[key] is a string:
      - If it looks like a CSV path, try to build a Log object from it.
      - Otherwise, move it to key_name and clear the object slot to avoid attribute errors.
    """
    if key not in st.session_state:
        return

    val = st.session_state.get(key)

    # Already usable object
    if _usable_acoustic_obj(val):
        return

    # Try to coerce from a CSV-like path string
    if isinstance(val, str):
        if _looks_like_path(val):
            try:
                import pycoustic as pc  # Lazy import
                st.session_state[key] = pc.Log(path=val.strip())
                return
            except Exception:
                # Fall through to clearing if coercion fails
                pass

        # Preserve label for UI, clear the object slot to avoid attribute errors
        st.session_state[f"{key}_name"] = val
        st.session_state[key] = None

def _normalize_session_state() -> None:
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return

    # Coerce or clear common object keys
    for k in ("survey", "log_obj", "log"):
        _coerce_or_clear_state_key(st, k)

    # Promote first usable object into the canonical "survey" slot
    if not _usable_acoustic_obj(st.session_state.get("survey")):
        for k in ("log_obj", "log"):
            candidate = st.session_state.get(k)
            if _usable_acoustic_obj(candidate):
                st.session_state["survey"] = candidate
                break

# Run normalization early so downstream code doesn't encounter attribute errors
_normalize_session_state()


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