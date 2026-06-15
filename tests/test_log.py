import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import datetime as dt
from unittest.mock import patch
import requests

from pycoustic.log import Log
from pycoustic.survey import Survey
from pycoustic.weather import WeatherHistory

#pytest tests/test_log.py -v

# --- Fixtures for generating temporary CSV data ---

@pytest.fixture
def nan_csv_path():
    """Generates a CSV with missing values (NaNs)."""
    content = (
        "Time,Leq A,L90 A\n"
        "2024-12-16 12:00,50.0,40.0\n"
        "2024-12-16 12:15,,\n"
        "2024-12-16 12:30,55.0,42.0\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        path = tmp.name
    yield path
    os.remove(path)


@pytest.fixture
def malformed_header_csv_path():
    """Generates a CSV with an unexpected column name (no spaces)."""
    content = (
        "Time,Leq A,BatteryLevel\n"
        "2024-12-16 12:00,50.0,99\n"
        "2024-12-16 12:15,55.0,98\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        path = tmp.name
    yield path
    os.remove(path)


@pytest.fixture
def mismatched_logs_paths():
    """Generates two CSVs with different columns to test heterogeneous surveys."""
    content1 = "Time,Leq A\n2024-12-16 12:00,50.0\n2024-12-16 12:15,55.0\n"
    content2 = "Time,Leq A,Leq 125\n2024-12-16 12:00,45.0,40.0\n2024-12-16 12:15,48.0,42.0\n"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_log1.csv") as tmp1, \
            tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_log2.csv") as tmp2:
        tmp1.write(content1)
        tmp2.write(content2)
        paths = (tmp1.name, tmp2.name)

    yield paths
    os.remove(paths[0])
    os.remove(paths[1])


@pytest.fixture
def sample_csv_path():
    """
    Creates a temporary CSV file with dummy acoustic data to be used in tests.
    Using a temporary file ensures our tests are completely isolated.
    """
    csv_content = """Time,Leq A,L90 A,Lmax A,Leq 125,L90 125,Lmax 125
2024/12/16 22:45,50.0,45.0,60.0,55.0,50.0,65.0
2024/12/16 23:00,48.0,43.0,58.0,53.0,48.0,63.0
2024/12/16 23:15,45.0,40.0,55.0,50.0,45.0,60.0
2024/12/17 00:00,40.0,35.0,50.0,45.0,40.0,55.0
2024/12/17 06:45,42.0,38.0,52.0,48.0,44.0,58.0
2024/12/17 07:00,55.0,50.0,65.0,60.0,55.0,70.0
2024/12/17 07:15,58.0,52.0,68.0,62.0,58.0,72.0
"""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(csv_content)

    yield path
    os.remove(path)


@pytest.fixture
def real_csv_path():
    """Provides the path to a real sample CSV file if it exists."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "UA1_py.csv")
    if os.path.exists(path):
        return path
    return None


@pytest.fixture
def nor140_xlsx_path():
    """Provides the path to the Nor140 overview workbook if it exists."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "Overview140_260525_155527.xlsx")
    if os.path.exists(path):
        return path
    return None

def _normalise_scalar_for_csv_comparison(value):
    if pd.isna(value):
        return ""

    if isinstance(value, dt.time):
        return value.strftime("%H:%M:%S")

    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return dt.datetime.combine(value, dt.time()).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(value, (pd.Timestamp, dt.datetime, np.datetime64)):
        parsed_dt = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed_dt):
            return parsed_dt.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return ""

        if len(stripped) == 8 and stripped.count(":") == 2:
            parsed_time = pd.to_datetime(stripped, format="%H:%M:%S", errors="coerce")
            if not pd.isna(parsed_time):
                return stripped

        parsed_dt = pd.to_datetime(stripped, errors="coerce")
        if not pd.isna(parsed_dt):
            if len(stripped) == 10 and stripped.count("-") == 2:
                return parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
            return parsed_dt.strftime("%Y-%m-%d %H:%M:%S")

        try:
            num = float(stripped)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except ValueError:
            return stripped

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))

    return str(value)

def _normalise_for_csv_comparison(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_normalise_scalar_for_csv_comparison)
    return out

# --- 1. Log Edge Cases ---

def test_log_handles_nan_values(nan_csv_path):
    """Test that missing rows (NaNs) don't crash Leq recomputations."""
    log = Log(nan_csv_path)
    interval_data = log.as_interval(t="1h")

    assert not interval_data.empty
    assert not pd.isna(interval_data[("Leq", "A")].iloc[0])


def test_log_malformed_headers(malformed_header_csv_path):
    """Test that CSVs with columns lacking a space don't trigger IndexError."""
    try:
        log = Log(malformed_header_csv_path)
        assert ("BatteryLevel", "") in log.get_data().columns or ("BatteryLevel", "BatteryLevel") in log.get_data().columns
    except IndexError:
        pytest.fail("Log._assign_header crashed on a column name without spaces.")


def test_log_custom_midnight_boundaries(nan_csv_path):
    """Test period boundaries exactly at midnight."""
    log = Log(nan_csv_path)
    log.set_periods({"day": (7, 0), "evening": (23, 59), "night": (0, 0)})

    times = log.get_period_times()
    assert times[2] == dt.time(0, 0)
    assert ("Night idx", "") in log.get_data().columns


def test_log_initialization(sample_csv_path):
    """Test that the Log class initializes correctly and extracts basic start/end times."""
    log = Log(sample_csv_path)

    assert log.get_start() == pd.to_datetime("2024-12-16 22:45:00")
    assert log.get_end() == pd.to_datetime("2024-12-17 07:15:00")

    day, evening, night = log.get_period_times()
    assert day == dt.time(7, 0)
    assert evening == dt.time(23, 0)
    assert night == dt.time(23, 0)

    assert not log.is_evening()


def test_log_header_assignment(sample_csv_path):
    """Test that headers are properly mapped into a Pandas MultiIndex."""
    log = Log(sample_csv_path)
    data = log.get_data()

    assert isinstance(data.columns, pd.MultiIndex)
    assert ("Leq", "A") in data.columns
    assert ("L90", "A") in data.columns
    assert ("Leq", 125.0) in data.columns


def test_append_night_idx(sample_csv_path):
    """Test that early morning measurements are properly assigned to the previous night."""
    log = Log(sample_csv_path)
    data = log.get_data()

    assert ("Night idx", "") in data.columns

    idx_00_00 = data.loc[pd.to_datetime("2024-12-17 00:00:00"), ("Night idx", "")]
    if isinstance(idx_00_00, pd.Series):
        idx_00_00 = idx_00_00.iloc[0]
    assert idx_00_00 == pd.to_datetime("2024-12-16 00:00:00")

    idx_07_00 = data.loc[pd.to_datetime("2024-12-17 07:00:00"), ("Night idx", "")]
    if isinstance(idx_07_00, pd.Series):
        idx_07_00 = idx_07_00.iloc[0]
    assert idx_07_00 == pd.to_datetime("2024-12-17 07:00:00")


def test_get_period(sample_csv_path):
    """Test data extraction split by configured daily periods."""
    log = Log(sample_csv_path)

    days = log.get_period(period="days")
    nights = log.get_period(period="nights")

    assert len(days) == 3
    assert len(nights) == 4


def test_leq_by_date(sample_csv_path):
    """Test Leq computation for continuous periods."""
    log = Log(sample_csv_path)
    nights = log.get_period(data=log.get_antilogs(), period="nights")

    leq = log.leq_by_date(nights, cols=[("Leq", "A")])

    assert not leq.empty
    assert len(leq) == 1

    val = leq.iloc[0][("Leq", "A")]
    assert isinstance(val, (float, np.floating))
    assert 40.0 <= val <= 50.0


def test_as_interval(sample_csv_path):
    """Test measurement period recalculations via interval resampling."""
    log = Log(sample_csv_path)

    interval_data = log.as_interval(t="1h", leq_cols=[("Leq", "A")], max_pivots=[("Lmax", "A")])

    assert not interval_data.empty
    assert ("Night idx", "") in interval_data.columns
    assert ("Leq", "A") in interval_data.columns
    assert ("Lmax", "A") in interval_data.columns


def test_set_periods(sample_csv_path):
    """Test that modifying the day/evening/night cycle reflects successfully."""
    log = Log(sample_csv_path)
    log.set_periods({"day": (8, 0), "evening": (18, 0), "night": (22, 0)})

    day, evening, night = log.get_period_times()
    assert day == dt.time(8, 0)
    assert evening == dt.time(18, 0)
    assert night == dt.time(22, 0)

    assert log.is_evening()


def test_real_data_compatibility(real_csv_path):
    """Integration test ensuring the Log class functions dynamically with actual target files."""
    if real_csv_path is None:
        pytest.skip("Real CSV sample was not found.")

    log = Log(real_csv_path)
    data = log.get_data()

    assert not data.empty
    assert isinstance(data.columns, pd.MultiIndex)
    assert ("Night idx", "") in data.columns
    assert ("Leq", "A") in data.columns


def test_log_load_nor140_overview_xlsx(nor140_xlsx_path):
    """Integration test for Nor140 overview XLSX parsing."""
    if nor140_xlsx_path is None:
        pytest.skip("Nor140 overview workbook was not found.")

    log = Log(nor140_xlsx_path)
    data = log.get_data()

    assert not data.empty
    assert isinstance(data.columns, pd.MultiIndex)

    assert ("Leq", "A") in data.columns
    assert ("Lmax", "A") in data.columns
    assert ("L90", "A") in data.columns
    assert ("Leq", "Z") in data.columns or ("Lmax", "Z") in data.columns
    assert ("Night idx", "") in data.columns

    spectral_metrics = [col for col in data.columns if col[0] in {"Leq", "Lmax", "L90", "Perc2", "Perc4"} and isinstance(col[1], float)]
    assert len(spectral_metrics) > 0

def _flatten_result_frame(df: pd.DataFrame, section: str) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " | ".join("" if v is None else str(v) for v in col).strip(" |")
            for col in out.columns.to_list()
        ]
    else:
        out.columns = [str(col) for col in out.columns]

    if isinstance(out.index, pd.MultiIndex):
        index_names = [
            name if name is not None else f"index_{i}"
            for i, name in enumerate(out.index.names)
        ]
        out = out.reset_index()
        out.columns = index_names + list(out.columns[len(index_names):])
    else:
        index_name = out.index.name if out.index.name is not None else "index"
        out = out.reset_index().rename(columns={"index": index_name})

    out.insert(0, "section", section)
    return out


def test_output_matches_provided_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    overview_path = os.path.join(base_dir, "Overview140_260525_155527.xlsx")
    ua5_path = os.path.join(base_dir, "UA5_py.csv")
    truth_path = os.path.join(base_dir, "survey_ground_truth.csv")

    if not os.path.exists(overview_path):
        pytest.skip("Nor140 overview workbook was not found.")
    if not os.path.exists(ua5_path):
        pytest.skip("UA5 sample CSV was not found.")
    if not os.path.exists(truth_path):
        pytest.skip("Ground-truth CSV was not found.")

    survey = Survey()
    survey.add_log(Log(overview_path), name="Overview140")
    survey.add_log(Log(ua5_path), name="UA5")
    survey.set_periods()

    periods = pd.DataFrame.from_dict(
        {
            key: {
                "day": value[0].isoformat(),
                "evening": value[1].isoformat(),
                "night": value[2].isoformat(),
            }
            for key, value in survey.get_periods().items()
        },
        orient="index",
    )
    periods.index.name = "position"

    computed_results = {
        "get_periods": _flatten_result_frame(periods, "get_periods"),
        "broadband_summary": _flatten_result_frame(survey.broadband_summary(), "broadband_summary"),
        "modal": _flatten_result_frame(survey.modal(), "modal"),
        "counts": _flatten_result_frame(survey.counts(), "counts"),
        "lmax_spectra": _flatten_result_frame(survey.lmax_spectra(), "lmax_spectra"),
        "leq_spectra": _flatten_result_frame(survey.leq_spectra(), "leq_spectra"),
    }

    reference = pd.read_csv(truth_path)

    for section, computed in computed_results.items():
        expected = reference[reference["section"] == section].copy()

        assert not expected.empty, f"Missing section in ground truth: {section}"

        missing_in_expected = set(computed.columns) - set(expected.columns)

        assert not missing_in_expected, (
            f"Ground truth for {section} is missing columns: {sorted(missing_in_expected)}"
        )

        expected = expected.loc[:, computed.columns]

        computed = _normalise_for_csv_comparison(computed)
        expected = _normalise_for_csv_comparison(expected)

        sort_cols = list(computed.columns)

        computed = computed.sort_values(by=sort_cols).reset_index(drop=True)
        expected = expected.sort_values(by=sort_cols).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            computed,
            expected,
            check_dtype=False,
            check_exact=True,
            obj=f"Mismatch in section {section}",
        )

def test_as_interval_retains_spectral_columns_when_a_weighted_column_missing(tmp_path):
    """Test that as_interval keeps spectral columns even when no A-weighted pivot column exists."""
    csv_content = """Time,Leq 63,L90 4000,Lmax 2000
2024/12/16 12:00,50.0,40.0,60.0
2024/12/16 12:15,52.0,41.0,62.0
2024/12/16 12:30,54.0,42.0,64.0
2024/12/16 12:45,56.0,43.0,66.0
"""
    csv_path = tmp_path / "spectral_only.csv"
    csv_path.write_text(csv_content)

    log = Log(str(csv_path))
    interval_data = log.as_interval(
        t="1h",
        leq_cols=[("Leq", 63.0), ("L90", 4000.0)],
        max_pivots=[("Lmax", "A")],
    )

    assert not interval_data.empty
    assert ("Night idx", "") in interval_data.columns

    assert ("Leq", 63.0) in interval_data.columns
    assert ("L90", 4000.0) in interval_data.columns
    assert ("Lmax", 2000.0) in interval_data.columns

    assert not interval_data[("Leq", 63.0)].isna().all()
    assert not interval_data[("L90", 4000.0)].isna().all()
    assert not interval_data[("Lmax", 2000.0)].isna().all()


# --- as_interval arithmetic averaging tests ---

def test_as_interval_arithmetic_mean_matches_simple_average(tmp_path):
    """Verify arithmetic averaging computes a plain mean of dB values, not an energy average."""
    csv_content = """Time,L90 A
2024/01/01 12:00,40.0
2024/01/01 12:15,42.0
2024/01/01 12:30,44.0
2024/01/01 12:45,46.0
"""
    csv_path = tmp_path / "l90_arith.csv"
    csv_path.write_text(csv_content)

    log = Log(str(csv_path))

    interval_arith = log.as_interval(
        t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="arithmetic"
    )
    interval_log = log.as_interval(
        t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="log"
    )

    arith_val = interval_arith[("L90", "A")].iloc[0]
    log_val = interval_log[("L90", "A")].iloc[0]

    # Arithmetic mean of 40, 42, 44, 46 = 43.0
    assert abs(arith_val - 43.0) < 0.2, f"Expected 43.0, got {arith_val}"

    # Log (energy) mean of the same values is always >= arithmetic mean
    expected_log = round(10 * np.log10(np.mean([10 ** (x / 10) for x in [40.0, 42.0, 44.0, 46.0]])), 1)
    assert abs(log_val - expected_log) < 0.2, f"Expected {expected_log}, got {log_val}"

    # The two modes must differ for non-uniform inputs
    assert log_val != arith_val


def test_as_interval_arithmetic_mean_large_spread(tmp_path):
    """With a large dB spread, energy average should be significantly higher than arithmetic average."""
    csv_content = """Time,L90 A
2024/01/01 12:00,30.0
2024/01/01 12:15,50.0
2024/01/01 12:30,30.0
2024/01/01 12:45,50.0
"""
    csv_path = tmp_path / "l90_spread.csv"
    csv_path.write_text(csv_content)

    log = Log(str(csv_path))

    interval_arith = log.as_interval(
        t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="arithmetic"
    )
    interval_log = log.as_interval(
        t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="log"
    )

    arith_val = interval_arith[("L90", "A")].iloc[0]
    log_val = interval_log[("L90", "A")].iloc[0]

    # Arithmetic mean of 30, 50, 30, 50 = 40.0
    assert abs(arith_val - 40.0) < 0.2, f"Expected 40.0, got {arith_val}"

    # Energy average is dominated by the 50 dB values; should be ~47 dB
    expected_log = round(10 * np.log10(np.mean([10 ** (x / 10) for x in [30.0, 50.0, 30.0, 50.0]])), 1)
    assert abs(log_val - expected_log) < 0.2, f"Expected {expected_log:.1f}, got {log_val}"

    # Log mean must be substantially higher than arithmetic mean
    assert log_val > arith_val + 5.0, (
        f"Log mean ({log_val}) should be > 5 dB above arithmetic mean ({arith_val})"
    )


def test_as_interval_default_averaging_is_log(sample_csv_path):
    """Default averaging mode must be logarithmic for backward compatibility."""
    log = Log(sample_csv_path)

    default_data = log.as_interval(t="1h", leq_cols=[("L90", "A")], max_pivots=[])
    log_data = log.as_interval(t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="log")

    if ("L90", "A") in default_data.columns and ("L90", "A") in log_data.columns:
        pd.testing.assert_series_equal(
            default_data[("L90", "A")].dropna(),
            log_data[("L90", "A")].dropna(),
            check_names=False,
        )


def test_as_interval_arithmetic_multiple_intervals(tmp_path):
    """Arithmetic averaging must produce correct means for each resampled interval independently."""
    csv_content = """Time,L90 A
2024/01/01 12:00,40.0
2024/01/01 12:15,42.0
2024/01/01 12:30,44.0
2024/01/01 12:45,46.0
2024/01/01 13:00,50.0
2024/01/01 13:15,52.0
2024/01/01 13:30,54.0
2024/01/01 13:45,56.0
"""
    csv_path = tmp_path / "l90_multi.csv"
    csv_path.write_text(csv_content)

    log = Log(str(csv_path))

    interval_data = log.as_interval(
        t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="arithmetic"
    )

    assert not interval_data.empty
    assert ("L90", "A") in interval_data.columns

    vals = interval_data[("L90", "A")].dropna()
    assert len(vals) == 2

    # 12:00–12:45 → mean(40, 42, 44, 46) = 43.0
    assert abs(vals.iloc[0] - 43.0) < 0.2, f"First interval: expected 43.0, got {vals.iloc[0]}"
    # 13:00–13:45 → mean(50, 52, 54, 56) = 53.0
    assert abs(vals.iloc[1] - 53.0) < 0.2, f"Second interval: expected 53.0, got {vals.iloc[1]}"


def test_as_interval_arithmetic_with_nan_values(tmp_path):
    """Arithmetic averaging must skip NaN values without crashing."""
    csv_content = """Time,L90 A
2024/01/01 12:00,40.0
2024/01/01 12:15,
2024/01/01 12:30,44.0
2024/01/01 12:45,46.0
"""
    csv_path = tmp_path / "l90_nan.csv"
    csv_path.write_text(csv_content)

    log = Log(str(csv_path))

    interval_data = log.as_interval(
        t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="arithmetic"
    )

    assert not interval_data.empty
    assert ("L90", "A") in interval_data.columns

    val = interval_data[("L90", "A")].dropna().iloc[0]
    # mean(40, 44, 46) ignoring NaN = 43.33...
    assert abs(val - round((40.0 + 44.0 + 46.0) / 3, 1)) < 0.2


def test_as_interval_arithmetic_single_value_equals_value(tmp_path):
    """For a single-row interval, arithmetic average should return that value unchanged."""
    csv_content = """Time,L90 A
2024/01/01 12:00,45.0
"""
    csv_path = tmp_path / "l90_single.csv"
    csv_path.write_text(csv_content)

    log = Log(str(csv_path))

    interval_data = log.as_interval(
        t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="arithmetic"
    )

    assert not interval_data.empty
    val = interval_data[("L90", "A")].dropna().iloc[0]
    assert abs(val - 45.0) < 0.2


def test_as_interval_arithmetic_spectral_l90(tmp_path):
    """Arithmetic averaging must work correctly for spectral L90 columns."""
    csv_content = """Time,L90 125,L90 250,L90 500
2024/01/01 12:00,30.0,35.0,40.0
2024/01/01 12:15,32.0,37.0,42.0
2024/01/01 12:30,34.0,39.0,44.0
2024/01/01 12:45,36.0,41.0,46.0
"""
    csv_path = tmp_path / "l90_spectral.csv"
    csv_path.write_text(csv_content)

    log = Log(str(csv_path))

    interval_data = log.as_interval(
        t="1h",
        leq_cols=[("L90", 125.0), ("L90", 250.0), ("L90", 500.0)],
        max_pivots=[],
        averaging="arithmetic",
    )

    assert not interval_data.empty

    assert abs(interval_data[("L90", 125.0)].iloc[0] - 33.0) < 0.2  # mean(30,32,34,36)
    assert abs(interval_data[("L90", 250.0)].iloc[0] - 38.0) < 0.2  # mean(35,37,39,41)
    assert abs(interval_data[("L90", 500.0)].iloc[0] - 43.0) < 0.2  # mean(40,42,44,46)


def test_as_interval_invalid_averaging_raises_value_error(sample_csv_path):
    """An unrecognised averaging mode must raise ValueError."""
    log = Log(sample_csv_path)

    with pytest.raises(ValueError, match="averaging"):
        log.as_interval(
            t="1h", leq_cols=[("L90", "A")], max_pivots=[], averaging="geometric"
        )


def test_as_interval_arithmetic_preserves_lmax_and_night_idx(tmp_path):
    """Arithmetic averaging must not affect Lmax computation or Night idx column."""
    csv_content = """Time,L90 A,Lmax A
2024/01/01 12:00,40.0,65.0
2024/01/01 12:15,42.0,67.0
2024/01/01 12:30,44.0,70.0
2024/01/01 12:45,46.0,63.0
"""
    csv_path = tmp_path / "l90_lmax.csv"
    csv_path.write_text(csv_content)

    log = Log(str(csv_path))

    interval_data = log.as_interval(
        t="1h",
        leq_cols=[("L90", "A")],
        max_pivots=[("Lmax", "A")],
        averaging="arithmetic",
    )

    assert not interval_data.empty
    assert ("Night idx", "") in interval_data.columns
    assert ("L90", "A") in interval_data.columns
    assert ("Lmax", "A") in interval_data.columns

    # Lmax is still the maximum, not an average
    assert interval_data[("Lmax", "A")].iloc[0] == 70.0


# --- 2. Survey Edge Cases ---

def test_survey_empty_operations():
    """Test that Survey gracefully handles operations when no logs are added."""
    survey = Survey()

    summary = survey.broadband_summary()
    assert summary.empty

    leq_spec = survey.leq_spectra()
    assert leq_spec.empty


def test_survey_heterogeneous_logs(mismatched_logs_paths):
    """Test that Survey can process logs with different column schemas without crashing."""
    survey = Survey()

    log1 = Log(mismatched_logs_paths[0])
    log2 = Log(mismatched_logs_paths[1])

    survey.add_log(log1, name="Pos1")
    survey.add_log(log2, name="Pos2")

    try:
        summary = survey.broadband_summary(leq_cols=[("Leq", "A"), ("Leq", 125.0)])
        assert not summary.empty
        assert ("Daytime", "Leq", 125.0) in summary.columns
        assert ("Night-time", "Leq", 125.0) in summary.columns
    except KeyError:
        pytest.fail("Survey crashed when attempting to process heterogeneous logs with missing columns.")


def test_survey_set_get_periods(sample_csv_path):
    """Test setting and getting daytime/evening/night-time periods across all logs in a survey."""
    survey = Survey()
    log1 = Log(sample_csv_path)
    survey.add_log(log1, name="Pos1")

    survey.set_periods({"day": (8, 0), "evening": (18, 0), "night": (22, 0)})

    periods = survey.get_periods()
    assert "Pos1" in periods

    day, evening, night = periods["Pos1"]
    assert day == dt.time(8, 0)
    assert evening == dt.time(18, 0)
    assert night == dt.time(22, 0)


def test_survey_modal(sample_csv_path):
    """Test calculation of Modal L90 values across the survey."""
    survey = Survey()
    survey.add_log(Log(sample_csv_path), name="Pos1")

    modal_df = survey.modal(cols=[("L90", "A")], by_date=True)
    assert not modal_df.empty

    positions = [idx[0] for idx in modal_df.index]
    assert "Pos1" in positions

    assert "Daytime" in modal_df.columns
    assert "Night-time" in modal_df.columns


def test_survey_modal_handles_logs_with_no_lmax(tmp_path):
    """Test modal handles logs with no Lmax column."""
    csv_content = """Time,L90 A
2024/12/16 23:00,43.0
2024/12/17 00:00,35.0
2024/12/17 06:45,38.0
2024/12/17 07:00,50.0
"""
    csv_path = tmp_path / "no_lmax.csv"
    csv_path.write_text(csv_content)

    survey = Survey()
    survey.add_log(Log(str(csv_path)), name="Pos1")

    try:
        result = survey.modal()
    except Exception as exc:
        pytest.fail(f"modal crashed when Lmax was absent: {exc}")

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_survey_counts(sample_csv_path):
    """Test generating value counts for decibel levels."""
    survey = Survey()
    survey.add_log(Log(sample_csv_path), name="Pos1")

    counts_df = survey.counts(cols=[("L90", "A")])
    assert not counts_df.empty

    assert counts_df.index.name == "dB"
    assert isinstance(counts_df.index[0], (int, np.integer))

    assert "Pos1" in counts_df.columns
    assert ("Pos1", "Daytime") in counts_df.columns
    assert ("Pos1", "Night-time") in counts_df.columns


def test_survey_leq_spectra(sample_csv_path):
    """Test computing an overall continuous Leq over daytime and night-time periods."""
    survey = Survey()
    survey.add_log(Log(sample_csv_path), name="Pos1")

    leq_spec = survey.leq_spectra(leq_cols=[("Leq", "A"), ("Leq", 125.0)])
    assert not leq_spec.empty

    assert "Pos1" in leq_spec.index
    assert ("Daytime", "Leq", "A") in leq_spec.columns
    assert ("Night-time", "Leq", 125.0) in leq_spec.columns


def test_survey_lmax_spectra(sample_csv_path):
    """Test extraction of the highest Lmax values across the survey."""
    survey = Survey()
    survey.add_log(Log(sample_csv_path), name="Pos1")

    lmax_spec = survey.lmax_spectra(n=2, t="15min", period="nights")

    assert not lmax_spec.empty

    positions = [idx[0] for idx in lmax_spec.index]
    assert "Pos1" in positions

    assert ("Lmax", "A") in lmax_spec.columns
    assert ("Time", "") in lmax_spec.columns


def test_survey_broadband_summary_handles_missing_lmax_and_leq_cols(tmp_path):
    """Test broadband_summary handles missing Lmax data and missing requested Leq columns."""
    csv_content = """Time,Leq A
2024/12/16 23:00,48.0
2024/12/17 00:00,40.0
2024/12/17 06:45,42.0
2024/12/17 07:00,55.0
"""
    csv_path = tmp_path / "no_lmax.csv"
    csv_path.write_text(csv_content)

    survey = Survey()
    survey.add_log(Log(str(csv_path)), name="Pos1")

    try:
        summary = survey.broadband_summary(
            leq_cols=[("Leq", 125.0)],
            max_cols=[("Lmax", "A")],
            lmax_n=1,
            lmax_t="15min",
        )
    except Exception as exc:
        pytest.fail(f"broadband_summary crashed when Lmax and requested Leq columns were missing: {exc}")

    assert isinstance(summary, pd.DataFrame)
    assert not summary.empty

    assert ("Daytime", "Leq", 125.0) in summary.columns
    assert ("Night-time", "Leq", 125.0) in summary.columns
    assert ("Night-time", "Lmax", "A") in summary.columns

    pos1 = summary.loc["Pos1"]

    assert pos1[("Daytime", "Leq", 125.0)].isna().all()
    assert pos1[("Night-time", "Leq", 125.0)].isna().all()
    assert pos1[("Night-time", "Lmax", "A")].isna().all()


def test_survey_broadband_summary_handles_logs_with_no_leq_or_lmax(tmp_path):
    """Test broadband_summary handles input logs with no Leq or Lmax columns at all."""
    csv_content = """Time,L90 A
2024/12/16 23:00,43.0
2024/12/17 00:00,35.0
2024/12/17 06:45,38.0
2024/12/17 07:00,50.0
"""
    csv_path = tmp_path / "no_leq_no_lmax.csv"
    csv_path.write_text(csv_content)

    survey = Survey()
    survey.add_log(Log(str(csv_path)), name="Pos1")

    try:
        summary = survey.broadband_summary(
            leq_cols=[("Leq", "A")],
            max_cols=[("Lmax", "A")],
            lmax_n=1,
            lmax_t="15min",
        )
    except Exception as exc:
        pytest.fail(f"broadband_summary crashed when Leq and Lmax columns were entirely absent: {exc}")

    assert isinstance(summary, pd.DataFrame)
    assert not summary.empty

    assert ("Daytime", "Leq", "A") in summary.columns
    assert ("Night-time", "Leq", "A") in summary.columns
    assert ("Night-time", "Lmax", "A") in summary.columns

    pos1 = summary.loc["Pos1"]

    assert pos1[("Daytime", "Leq", "A")].isna().all()
    assert pos1[("Night-time", "Leq", "A")].isna().all()
    assert pos1[("Night-time", "Lmax", "A")].isna().all()




# --- 2c. Log.get_nth_high_low — count, group_by_date ---

@pytest.fixture
def rich_sample_csv_path():
    """Data with multiple rows per date for granular nth_high_low testing."""
    content = """Time,Leq A,Lmax A,Lmax C,Leq 125
2024-12-16 22:45,50.0,60.0,65.0,55.0
2024-12-16 23:00,48.0,58.0,63.0,53.0
2024-12-16 23:15,45.0,55.0,60.0,50.0
2024-12-17 00:00,40.0,50.0,55.0,45.0
2024-12-17 06:45,42.0,52.0,58.0,48.0
2024-12-17 07:00,55.0,65.0,70.0,60.0
2024-12-17 07:15,58.0,68.0,72.0,62.0
"""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    yield path
    os.remove(path)


def test_nth_high_low_default_behaviour(sample_csv_path):
    """Default args (count=1, group_by_date=True) preserve original behaviour — one row per date."""
    log = Log(sample_csv_path)
    result = log.get_nth_high_low(n=1, pivot_col=("Lmax", "A"))
    # One row per date: highest Lmax A per date = 60.0 (12-16), 68.0 (12-17)
    assert len(result) == 2
    dates = set(result.index.date)
    assert dt.date(2024, 12, 16) in dates
    assert dt.date(2024, 12, 17) in dates
    vals = result[("Lmax", "A")].tolist()
    assert 60.0 in vals
    assert 68.0 in vals


def test_nth_high_low_count_multiple_per_date(rich_sample_csv_path):
    """count=3 returns the top 3 ranked rows for each date group."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(n=1, count=3, pivot_col=("Lmax", "A"))
    # 2024-12-16 has 3 rows total: returns 3 rows; 2024-12-17 has 4 rows: returns 3 rows
    assert len(result) == 6  # 3 per date × 2 dates
    per_date = result.groupby(result.index.date)
    assert len(per_date) == 2
    for date, group in per_date:
        assert len(group) == 3
        # Within each date, values should be descending
        vals = group[("Lmax", "A")].tolist()
        assert vals == sorted(vals, reverse=True), f"Values not descending for {date}: {vals}"


def test_nth_high_low_count_smaller_than_available(rich_sample_csv_path):
    """count larger than the number of rows in a date returns all available rows for that date."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(n=1, count=10, pivot_col=("Lmax", "A"))
    # 2024-12-16 has 3 rows, 2024-12-17 has 4 rows — count=10 returns all available
    assert len(result) == 7  # total rows in dataset
    vals = result[("Lmax", "A")].tolist()
    assert vals[0] == 68.0  # highest overall (2024-12-17 rank 1)
    assert vals[-1] == 50.0  # lowest overall (2024-12-17 rank 4)


def test_nth_high_low_group_by_date_false_all_rows(rich_sample_csv_path):
    """group_by_date=False returns overall ranking with no date grouping, count returns global rows."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(n=1, count=3, pivot_col=("Lmax", "A"), group_by_date=False)
    assert len(result) == 3  # top 3 overall
    expected = [68.0, 65.0, 60.0]  # highest 3 Lmax A values globally
    actual = result[("Lmax", "A")].tolist()
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_nth_high_low_group_by_date_false_offset(rich_sample_csv_path):
    """group_by_date=False with n>1 returns lower-ranked rows overall."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(n=3, count=2, pivot_col=("Lmax", "A"), group_by_date=False)
    assert len(result) == 2  # 3rd and 4th overall
    expected = [60.0, 58.0]  # 3rd=60.0 (at 22:45 on 12-16), 4th=58.0 (at 23:00 on 12-16)
    actual = result[("Lmax", "A")].tolist()
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_nth_high_low_high_false_with_count(rich_sample_csv_path):
    """high=False with count returns multiple lowest-ranked rows."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(
        n=1, count=3, pivot_col=("Lmax", "A"), high=False, group_by_date=False
    )
    assert len(result) == 3
    expected = [50.0, 52.0, 55.0]  # 3 lowest Lmax A globally
    actual = result[("Lmax", "A")].tolist()
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_nth_high_low_high_false_grouped_with_count(rich_sample_csv_path):
    """high=False with count and group_by_date=True returns multiple lowest per date."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(n=1, count=2, pivot_col=("Lmax", "A"), high=False)
    # 2 lowest per date: 2024-12-16 → (55.0, 58.0), 2024-12-17 → (50.0, 52.0)
    assert len(result) == 4  # 2 per date × 2 dates
    per_date = result.groupby(result.index.date)
    for date, group in per_date:
        vals = group[("Lmax", "A")].tolist()
        assert vals == sorted(vals), f"Values not ascending for {date}: {vals}"


def test_nth_high_low_pivot_spectral_with_count(rich_sample_csv_path):
    """Spectral pivot column (Leq 125) with count and group_by_date=False returns Leq columns."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(
        n=1, count=2, pivot_col=("Leq", 125.0), group_by_date=False
    )
    assert len(result) == 2
    # Top 2 Leq 125 values overall: 62.0 (07:15 on 12-17), 60.0 (07:00 on 12-17)
    assert result[("Leq", 125.0)].iloc[0] == 62.0
    assert result[("Leq", 125.0)].iloc[1] == 60.0
    # Should return all Leq columns (first level matches "Leq")
    assert ("Leq", "A") in result.columns


def test_nth_high_low_pivot_lmax_c_with_count(rich_sample_csv_path):
    """Lmax C pivot with count returns all Lmax columns from matching timestamps."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(
        n=1, count=3, pivot_col=("Lmax", "C"), group_by_date=True
    )
    # Top 3 per date by Lmax C
    assert len(result) == 6  # 3 per date × 2 dates
    # Should return all Lmax columns
    assert ("Lmax", "A") in result.columns
    assert ("Lmax", "C") in result.columns
    assert ("Time", "") in result.columns


def test_nth_high_low_all_cols_with_group_by_date_false(rich_sample_csv_path):
    """all_cols=True + group_by_date=False returns all data columns."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(
        n=1, count=2, pivot_col=("Lmax", "A"), all_cols=True, group_by_date=False
    )
    assert len(result) == 2
    # Should have all original columns including those not in the pivot family
    assert ("Leq", "A") in result.columns
    assert ("Leq", 125.0) in result.columns
    assert ("Lmax", "A") in result.columns
    assert ("Lmax", "C") in result.columns


def test_nth_high_low_n_greater_than_available_returns_empty(rich_sample_csv_path):
    """n exceeding available rows returns an empty DataFrame."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(n=999, pivot_col=("Lmax", "A"), group_by_date=False)
    assert result.empty


def test_nth_high_low_missing_pivot_returns_empty(rich_sample_csv_path):
    """Missing pivot column returns empty DataFrame regardless of new params."""
    log = Log(rich_sample_csv_path)
    result = log.get_nth_high_low(
        n=1, count=3, pivot_col=("L90", "A"), group_by_date=False
    )
    assert result.empty


def test_nth_high_low_count_returns_correct_values(rich_sample_csv_path):
    """Verify exact values returned for a specific known case."""
    log = Log(rich_sample_csv_path)
    # 2nd and 3rd highest Lmax A per date, grouped
    result = log.get_nth_high_low(n=2, count=2, pivot_col=("Lmax", "A"))
    # 2024-12-16: ranks 2 (58.0) and 3 (55.0) = the last 2 rows for that date
    # 2024-12-17: ranks 2 (65.0) and 3 (52.0) = mid 2 rows
    dec16 = result.loc[result.index.date == dt.date(2024, 12, 16)]
    assert len(dec16) == 2
    assert dec16[("Lmax", "A")].iloc[0] == 58.0
    assert dec16[("Lmax", "A")].iloc[1] == 55.0
    dec17 = result.loc[result.index.date == dt.date(2024, 12, 17)]
    assert len(dec17) == 2
    assert dec17[("Lmax", "A")].iloc[0] == 65.0
    assert dec17[("Lmax", "A")].iloc[1] == 52.0


# --- 3. WeatherHistory Edge Cases ---

@patch("requests.get")
def test_weather_invalid_postcode(mock_get):
    """Test that WeatherHistory raises KeyError for unresolvable postcodes."""
    mock_get.return_value.json.return_value = {"cod": "404", "message": "not found"}
    mock_get.return_value.status_code = 404

    hist = WeatherHistory()
    start_dt = dt.datetime(2024, 1, 1)
    end_dt = dt.datetime(2024, 1, 2)

    with pytest.raises(KeyError):
        hist.reinit(start=start_dt, end=end_dt, api_key="dummy_key", country="GB", postcode="XX99 9XX")


@patch("requests.get")
def test_weather_api_timeout(mock_get):
    """Test that a timeout from the weather API correctly bubbles up."""
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    hist = WeatherHistory()
    start_dt = dt.datetime(2024, 1, 1)
    end_dt = dt.datetime(2024, 1, 2)

    with pytest.raises(requests.exceptions.Timeout):
        hist.reinit(start=start_dt, end=end_dt, api_key="dummy_key", country="GB", postcode="SW1A 1AA")