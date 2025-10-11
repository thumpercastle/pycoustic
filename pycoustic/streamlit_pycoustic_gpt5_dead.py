# streamlit_pycoustic_app.py
import ast
import datetime as dt
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

import streamlit as st

from pycoustic import Log, Survey


def _parse_kwargs(text: str) -> Dict[str, Any]:
    """
    Safely parse a Python dict literal from text area.
    Returns {} if empty or invalid.
    """
    if not text or not text.strip():
        return {}
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _display_result(obj: Any):
    """
    Display helper to handle common return types.
    """
    # Plotly Figure-like
    if hasattr(obj, "to_plotly_json"):
        st.plotly_chart(obj, use_container_width=True)
        return

    # Pandas DataFrame-like
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        st.dataframe(obj, use_container_width=True)
        return

    # Dict/list -> JSON
    if isinstance(obj, (dict, list)):
        st.json(obj)
        return

    # Fallback
    st.write(obj)


def _ensure_state():
    if "survey" not in st.session_state:
        st.session_state["survey"] = None
    if "periods" not in st.session_state:
        st.session_state["periods"] = {"day": (7, 0), "evening": (19, 0), "night": (23, 0)}


def _write_uploaded_to_temp(uploaded) -> str:
    """
    Persist an UploadedFile to a temporary file and return the path.
    Using a real file path keeps Log(...) happy across environments.
    """
    suffix = Path(uploaded.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def _build_survey_from_files(files) -> Survey:
    """
    Create a Survey and attach Log objects for each uploaded file.
    """
    survey = Survey()
    for f in files:
        # Persist to disk to ensure compatibility with pandas and any path usage in Log
        tmp_path = _write_uploaded_to_temp(f)
        log_obj = Log(path=tmp_path)

        key = Path(f.name).stem
        # Attach Log to survey
        if hasattr(survey, "add_log"):
            try:
                survey.add_log(key, log_obj)
            except TypeError:
                survey.add_log(log_obj, key)
        else:
            # Fallback to internal storage if no public API is available
            survey._logs[key] = log_obj  # noqa: SLF001
    return survey


def _apply_periods_to_all_logs(survey: Survey, times: Dict[str, tuple[int, int]]):
    """
    Apply set_periods to each Log attached to the Survey.
    This avoids calling set_periods on Survey if it doesn't exist.
    """
    logs: Iterable[Log] = getattr(survey, "_logs", {}).values()
    for log in logs:
        if hasattr(log, "set_periods"):
            log.set_periods(times=times)


def _render_period_controls(survey: Survey):
    st.subheader("Assessment Periods")

    # Current periods from session (defaults set in _ensure_state)
    periods = st.session_state["periods"]
    day_h, day_m = periods["day"]
    eve_h, eve_m = periods["evening"]
    night_h, night_m = periods["night"]

    c1, c2, c3 = st.columns(3)
    with c1:
        day_time = st.time_input("Day starts", value=dt.time(day_h, day_m), key="period_day_start")
    with c2:
        eve_time = st.time_input("Evening starts", value=dt.time(eve_h, eve_m), key="period_eve_start")
    with c3:
        night_time = st.time_input("Night starts", value=dt.time(night_h, night_m), key="period_night_start")

    new_periods = {
        "day": (int(day_time.hour), int(day_time.minute)),
        "evening": (int(eve_time.hour), int(eve_time.minute)),
        "night": (int(night_time.hour), int(night_time.minute)),
    }

    if st.button("Apply periods to all logs", key="apply_periods"):
        try:
            _apply_periods_to_all_logs(survey, new_periods)
            st.session_state["periods"] = new_periods
            st.success("Periods applied to all logs.")
        except Exception as e:
            st.warning(f"Could not set periods: {e}")


def _render_method_runner(survey: Survey, method_name: str, help_text: str = ""):
    """
    Generic UI for running a Survey method with kwargs provided via text area.
    """
    with st.expander(method_name, expanded=True):
        if help_text:
            st.caption(help_text)

        kwargs_text = st.text_area(
            "kwargs (Python dict literal)",
            value="{}",
            key=f"kwargs_{method_name}",
            placeholder='Example: {"position": "UA1", "date": "2023-06-01"}',
            height=100,
        )

        kwargs = _parse_kwargs(kwargs_text)
        if st.button(f"Run {method_name}", key=f"run_{method_name}"):
            try:
                fn = getattr(survey, method_name)
                result = fn(**kwargs)
                _display_result(result)
            except AttributeError:
                st.error(f"Survey has no method named '{method_name}'.")
            except Exception as e:
                st.error(f"Error running {method_name}: {e}")


def main():
    st.set_page_config(page_title="pycoustic GUI", layout="wide")
    st.title("pycoustic – Streamlit GUI")

    _ensure_state()

    st.sidebar.header("Load CSV Logs")
    files = st.sidebar.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Each file becomes a Log; all Logs go into one Survey."
    )

    build = st.sidebar.button("Create / Update Survey", type="primary")

    if build and files:
        try:
            survey = _build_survey_from_files(files)
            # Apply default periods to all logs
            _apply_periods_to_all_logs(survey, st.session_state["periods"])
            st.session_state["survey"] = survey
            st.success("Survey created/updated.")
        except Exception as e:
            st.error(f"Unable to create Survey: {e}")

    survey: Survey = st.session_state.get("survey")

    if survey is None:
        st.info("Upload CSV files in the sidebar and click 'Create / Update Survey' to begin.")
        return

    # Period controls
    _render_period_controls(survey)

    st.markdown("---")
    st.header("Survey Outputs")

    _render_method_runner(
        survey,
        "resi_summary",
        help_text="Summary results for residential assessment. Provide any optional kwargs here."
    )
    _render_method_runner(
        survey,
        "modal",
        help_text="Run modal analysis over the survey. Provide any optional kwargs here."
    )
    _render_method_runner(
        survey,
        "leq_spectra",
        help_text="Compute or plot Leq spectra. Provide any optional kwargs here."
    )
    _render_method_runner(
        survey,
        "lmax_spectra",
        help_text="Compute or plot Lmax spectra. Provide any optional kwargs here."
    )

    st.markdown("---")
    with st.expander("Loaded Logs", expanded=False):
        try:
            names = list(getattr(survey, "_logs", {}).keys())
            if names:
                st.write(", ".join(names))
            else:
                st.write("No logs found in survey.")
        except Exception:
            st.write("Unable to list logs.")


if __name__ == "__main__":
    main()