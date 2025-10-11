# streamlit_pycoustic_app.py
import ast
from typing import Any, Dict

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
    try:
        import pandas as pd  # noqa: F401
        import plotly.graph_objects as go  # noqa: F401
    except Exception:
        pass

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


def _ensure_survey():
    if "survey" not in st.session_state:
        st.session_state["survey"] = None


def _build_survey_from_files(files):
    """
    Create a Survey and attach Log objects for each uploaded file.
    """
    survey = Survey()
    for f in files:
        # Ensure stream is at start for each run
        try:
            f.seek(0)
        except Exception:
            pass

        log_obj = Log(path=f)  # pandas.read_csv accepts file-like objects
        name = getattr(f, "name", "log")
        # Prefer a public method if present; otherwise attach to internal store
        if hasattr(survey, "add_log"):
            # If the library exposes an API, use it
            try:
                survey.add_log(name, log_obj)
            except TypeError:
                # Some APIs prefer (log_obj, name)
                survey.add_log(log_obj, name)
        else:
            # Fallback: set into the internal dictionary
            # Note: name collisions will overwrite; user can upload unique filenames
            survey._logs[name] = log_obj  # noqa: SLF001
    return survey


def _render_period_controls(survey: Survey):
    st.subheader("Assessment Periods")

    # Defaults: Day 07:00–19:00, Evening 19:00–23:00, Night 23:00–07:00
    col1, col2, col3 = st.columns(3)
    with col1:
        day_time = st.time_input("Day starts", value=None, key="period_day_start")
    with col2:
        eve_time = st.time_input("Evening starts", value=None, key="period_eve_start")
    with col3:
        night_time = st.time_input("Night starts", value=None, key="period_night_start")

    # If not set yet, pick defaults
    import datetime as _dt
    if day_time is None:
        day_time = _dt.time(7, 0)
    if eve_time is None:
        eve_time = _dt.time(19, 0)
    if night_time is None:
        night_time = _dt.time(23, 0)

    # Convert immediately to (hour, minute) tuples of ints
    times = {
        "day": (int(day_time.hour), int(day_time.minute)),
        "evening": (int(eve_time.hour), int(eve_time.minute)),
        "night": (int(night_time.hour), int(night_time.minute)),
    }

    # Apply to Survey
    try:
        survey.set_periods(times=times)
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
            height=80,
        )

        kwargs = _parse_kwargs(kwargs_text)
        run = st.button(f"Run {method_name}", key=f"run_{method_name}")

        if run:
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

    _ensure_survey()

    st.sidebar.header("Load CSV Logs")
    files = st.sidebar.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Each file will initialise a Log; all Logs are attached to one Survey."
    )

    build = st.sidebar.button("Create / Update Survey", type="primary")

    if build and files:
        try:
            survey = _build_survey_from_files(files)
            # Set sensible default periods immediately
            survey.set_periods(times={"day": (7, 0), "evening": (19, 0), "night": (23, 0)})
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

    # Provide simple runners for the required methods, each with kwargs injection
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
