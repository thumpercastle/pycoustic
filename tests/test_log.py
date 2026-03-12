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
from pycoustic.weather import WeatherHistory  # Importing the updated dataclass version from survey.py


# --- Fixtures for generating temporary CSV data ---

@pytest.fixture
def nan_csv_path():
    """Generates a CSV with missing values (NaNs)."""
    content = (
        "Time,Leq A,L90 A\n"
        "2024-12-16 12:00,50.0,40.0\n"
        "2024-12-16 12:15,,\n"  # Missing data row
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
    with os.fdopen(fd, 'w') as f:
        f.write(csv_content)

    yield path

    # Cleanup temporary file after tests are done
    os.remove(path)


@pytest.fixture
def real_csv_path():
    """Provides the path to a real sample CSV file if it exists."""
    # Assuming text/ directory is at the root of the project
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "UA1_py.csv")
    if os.path.exists(path):
        return path
    return None


# --- 1. Log Edge Cases ---

def test_log_handles_nan_values(nan_csv_path):
    """Test that missing rows (NaNs) don't crash Leq recomputations."""
    log = Log(nan_csv_path)
    # Recomputing over a longer period that includes the NaN row
    interval_data = log.as_interval(t="1h")
    
    # It should compute the Leq of 50.0 and 55.0 smoothly, ignoring the NaN
    assert not interval_data.empty
    assert not pd.isna(interval_data[("Leq", "A")].iloc[0])


def test_log_malformed_headers(malformed_header_csv_path):
    """Test that CSVs with columns lacking a space don't trigger IndexError."""
    # NOTE: This will fail with an IndexError in _assign_header until fixed!
    try:
        log = Log(malformed_header_csv_path)
        assert ("BatteryLevel", "") in log.get_data().columns or ("BatteryLevel", "BatteryLevel") in log.get_data().columns
    except IndexError:
        pytest.fail("Log._assign_header crashed on a column name without spaces (e.g. 'BatteryLevel').")


def test_log_custom_midnight_boundaries(nan_csv_path):
    """Test period boundaries exactly at midnight."""
    log = Log(nan_csv_path)
    # Night starting exactly at midnight
    log.set_periods({"day": (7, 0), "evening": (23, 59), "night": (0, 0)})
    
    times = log.get_period_times()
    assert times[2] == dt.time(0, 0)
    assert ("Night idx", "") in log.get_data().columns


def test_log_initialization(sample_csv_path):
    """Test that the Log class initializes correctly and extracts basic start/end times."""
    log = Log(sample_csv_path)

    assert log.get_start() == pd.to_datetime("2024-12-16 22:45:00")
    assert log.get_end() == pd.to_datetime("2024-12-17 07:15:00")

    # Check default periods
    day, evening, night = log.get_period_times()
    assert day == dt.time(7, 0)
    assert evening == dt.time(23, 0)
    assert night == dt.time(23, 0)

    # Evening is identical to night initially, so it should not be considered "evening"
    assert not log.is_evening()


def test_log_header_assignment(sample_csv_path):
    """Test that headers are properly mapped into a Pandas MultiIndex."""
    log = Log(sample_csv_path)
    data = log.get_data()

    # The columns should be a MultiIndex
    assert isinstance(data.columns, pd.MultiIndex)

    # A-weighted metrics should have super-header and sub-header "A"
    assert ("Leq", "A") in data.columns
    assert ("L90", "A") in data.columns

    # Spectral metrics should have float sub-headers after parsing
    assert ("Leq", 125.0) in data.columns


def test_append_night_idx(sample_csv_path):
    """Test that early morning measurements are properly assigned to the previous night."""
    log = Log(sample_csv_path)
    data = log.get_data()
    
    # In Pandas MultiIndex, adding `data["Night idx"]` often casts it to ("Night idx", "")
    assert ("Night idx", "") in data.columns
    
    # Time at 00:00 should have a night index rolled back to the previous day (2024-12-16)
    idx_00_00 = data.loc[pd.to_datetime("2024-12-17 00:00:00"), ("Night idx", "")]
    if isinstance(idx_00_00, pd.Series):
        idx_00_00 = idx_00_00.iloc[0]
    assert idx_00_00 == pd.to_datetime("2024-12-16 00:00:00")
    
    # Time at 07:00 (Daytime) should strictly remain as its actual date
    idx_07_00 = data.loc[pd.to_datetime("2024-12-17 07:00:00"), ("Night idx", "")]
    if isinstance(idx_07_00, pd.Series):
        idx_07_00 = idx_07_00.iloc[0]
    assert idx_07_00 == pd.to_datetime("2024-12-17 07:00:00")


def test_get_period(sample_csv_path):
    """Test data extraction split by configured daily periods."""
    log = Log(sample_csv_path)

    days = log.get_period(period="days")
    nights = log.get_period(period="nights")

    # Day periods: 07:00 to 23:00 (left inclusive)
    # Based on dummy data: 07:00, 07:15, and 22:45
    assert len(days) == 3

    # Night periods: 23:00 to 07:00 (left inclusive)
    # Based on dummy data: 23:00, 23:15, 00:00, 06:45
    assert len(nights) == 4


def test_leq_by_date(sample_csv_path):
    """Test Leq computation for continuous periods."""
    log = Log(sample_csv_path)
    nights = log.get_period(data=log.get_antilogs(), period="nights")

    leq = log.leq_by_date(nights, cols=[("Leq", "A")])

    # Even though data spans across midnight (16th & 17th), the night idx keeps it as 1 entry
    assert not leq.empty
    assert len(leq) == 1

    # Ensure standard mathematical properties resulting from np.log10 computation
    val = leq.iloc[0][("Leq", "A")]
    assert isinstance(val, (float, np.floating))
    assert 40.0 <= val <= 50.0  # Output should roughly be bound by its max input values


def test_as_interval(sample_csv_path):
    """Test measurement period recalculations via interval resampling."""
    log = Log(sample_csv_path)

    # Recompute using a 1-hour interval
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

    # Evening is now enabled
    assert log.is_evening()


def test_real_data_compatibility(real_csv_path):
    """Integration test ensuring the Log class functions dynamically with actual target files."""
    if real_csv_path is None:
        pytest.skip("Real CSV sample 'UA1_py.csv' was not found in 'text/' directory.")

    log = Log(real_csv_path)
    data = log.get_data()

    assert not data.empty
    assert isinstance(data.columns, pd.MultiIndex)
    assert ("Night idx", "") in data.columns
    assert ("Leq", "A") in data.columns


# --- 2. Survey Edge Cases ---

def test_survey_empty_operations():
    """Test that Survey gracefully handles operations when no logs are added."""
    survey = Survey()
    
    # broadband_summary should safely return an empty DataFrame, not throw an error
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
    
    # Passing leq_cols that exist in Pos2 but NOT in Pos1
    try:
        summary = survey.broadband_summary(leq_cols=[("Leq", "A"), ("Leq", 125.0)])
        assert not summary.empty
        # Pos1 shouldn't have 125Hz data, so its column should be populated with NaNs
        # The columns are now formatted with the Period at level 0: ("Daytime", "Leq", 125.0)
        assert ("Daytime", "Leq", 125.0) in summary.columns
        assert ("Night-time", "Leq", 125.0) in summary.columns
    except KeyError:
        pytest.fail("Survey crashed when attempting to process heterogeneous logs with missing columns.")

def test_survey_set_get_periods(sample_csv_path):
    """Test setting and getting daytime/evening/night-time periods across all logs in a survey."""
    survey = Survey()
    log1 = Log(sample_csv_path)
    survey.add_log(log1, name="Pos1")

    # Set custom periods for the whole survey
    survey.set_periods({"day": (8, 0), "evening": (18, 0), "night": (22, 0)})

    # Retrieve periods and verify they were applied
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

    # Grouped by Date
    modal_df = survey.modal(cols=[("L90", "A")], by_date=True)
    assert not modal_df.empty

    # Ensure MultiIndex structure containing Position
    positions = [idx[0] for idx in modal_df.index]
    assert "Pos1" in positions

    # Ensure Period top-level header is inserted successfully
    assert "Daytime" in modal_df.columns
    assert "Night-time" in modal_df.columns


def test_survey_counts(sample_csv_path):
    """Test generating value counts for decibel levels."""
    survey = Survey()
    survey.add_log(Log(sample_csv_path), name="Pos1")

    counts_df = survey.counts(cols=[("L90", "A")])
    assert not counts_df.empty

    # The counts dataframe renames its index to "dB" and should be numeric
    assert counts_df.index.name == "dB"
    assert isinstance(counts_df.index[0], (int, np.integer))

    # It should have Pos1 as a top-level column with "Daytime" & "Night-time" sub-columns
    assert "Pos1" in counts_df.columns
    assert ("Pos1", "Daytime") in counts_df.columns
    assert ("Pos1", "Night-time") in counts_df.columns


def test_survey_leq_spectra(sample_csv_path):
    """Test computing an overall continuous Leq over daytime and night-time periods."""
    survey = Survey()
    survey.add_log(Log(sample_csv_path), name="Pos1")

    # Compute for both Broadband (A) and 125Hz
    leq_spec = survey.leq_spectra(leq_cols=[("Leq", "A"), ("Leq", 125.0)])
    assert not leq_spec.empty

    # Position should be in the index
    assert "Pos1" in leq_spec.index

    # Columns should be multi-index: Period -> metric -> freq
    assert ("Daytime", "Leq", "A") in leq_spec.columns
    assert ("Night-time", "Leq", 125.0) in leq_spec.columns

def test_survey_lmax_spectra(sample_csv_path):
    """Test extraction of the highest Lmax values across the survey."""
    survey = Survey()
    survey.add_log(Log(sample_csv_path), name="Pos1")

    # Check top 2 lmax spectra (sample only has a few lines of dummy data anyway)
    # Re-evaluate over 15-minute periods
    lmax_spec = survey.lmax_spectra(n=2, t="15min", period="nights")

    assert not lmax_spec.empty

    # Ensure MultiIndex structure containing Position
    positions = [idx[0] for idx in lmax_spec.index]
    assert "Pos1" in positions

    # Columns should contain at least Lmax A and Time
    assert ("Lmax", "A") in lmax_spec.columns
    assert ("Time", "") in lmax_spec.columns

# --- 3. WeatherHistory Edge Cases ---

@patch("requests.get")
def test_weather_invalid_postcode(mock_get):
    """Test that WeatherHistory raises KeyError for unresolvable postcodes."""
    # Mocking the 404 response from OpenWeather
    mock_get.return_value.json.return_value = {"cod": "404", "message": "not found"}
    mock_get.return_value.status_code = 404
    
    hist = WeatherHistory()
    start_dt = dt.datetime(2024, 1, 1)
    end_dt = dt.datetime(2024, 1, 2)
    
    # We expect a KeyError because the resp.json()["lat"] will fail on a 404
    with pytest.raises(KeyError):
        hist.reinit(start=start_dt, end=end_dt, api_key="dummy_key", country="GB", postcode="XX99 9XX")


@patch("requests.get")
def test_weather_api_timeout(mock_get):
    """Test that a timeout from the OpenWeather API correctly bubbles up."""
    # Force requests.get to simulate a timeout
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
    
    hist = WeatherHistory()
    start_dt = dt.datetime(2024, 1, 1)
    end_dt = dt.datetime(2024, 1, 2)
    
    # We expect requests.exceptions.Timeout to be thrown
    with pytest.raises(requests.exceptions.Timeout):
        hist.reinit(start=start_dt, end=end_dt, api_key="dummy_key", country="GB", postcode="SW1A 1AA")