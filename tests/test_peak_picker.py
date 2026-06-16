"""Tests for Survey.peak_picker() method."""

import datetime as dt
import os
import tempfile

import pandas as pd
import pytest

from pycoustic import Log, Survey


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def survey_with_logs():
    """Survey with 2 Logs for peak_picker testing."""
    content_a = """Time,Leq A,Lmax A,Lmax C,Leq 125
2024-12-16 22:45,50.0,60.0,65.0,55.0
2024-12-16 23:00,48.0,58.0,63.0,53.0
2024-12-16 23:15,45.0,55.0,60.0,50.0
2024-12-17 00:00,40.0,50.0,55.0,45.0
2024-12-17 06:45,42.0,52.0,58.0,48.0
2024-12-17 07:00,55.0,65.0,70.0,60.0
2024-12-17 07:15,58.0,68.0,72.0,62.0
"""
    content_b = """Time,Leq A,Lmax A
2024-12-17 08:00,45.0,55.0
2024-12-17 08:15,50.0,60.0
2024-12-17 08:30,48.0,58.0
"""

    fd_a, path_a = tempfile.mkstemp(suffix="_a.csv")
    with os.fdopen(fd_a, "w") as f: f.write(content_a)
    fd_b, path_b = tempfile.mkstemp(suffix="_b.csv")
    with os.fdopen(fd_b, "w") as f: f.write(content_b)

    survey = Survey()
    survey.add_log(Log(path_a), name="PosA")
    survey.add_log(Log(path_b), name="PosB")

    yield survey
    os.remove(path_a)
    os.remove(path_b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_peak_picker_top_k_highest(survey_with_logs):
    """peak_picker returns the K highest pivot values overall with all columns."""
    peaks_df, history_df = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("Lmax", "A"),
        k=3,
        high=True,
    )

    # Top 3 Lmax A overall: 68.0, 65.0, 60.0
    assert len(peaks_df) == 3
    pivot_vals = peaks_df[("Lmax", "A")].tolist()
    assert pivot_vals == [68.0, 65.0, 60.0], f"Expected [68.0, 65.0, 60.0], got {pivot_vals}"

    # All spectral columns should be present
    assert ("Lmax", "C") in peaks_df.columns
    assert ("Leq", "A") in peaks_df.columns
    assert ("Leq", 125.0) in peaks_df.columns
    assert ("Time", "") in peaks_df.columns


def test_peak_picker_top_k_lowest(survey_with_logs):
    """peak_picker returns the K lowest pivot values overall."""
    peaks_df, history_df = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("Lmax", "A"),
        k=3,
        high=False,
    )

    # Bottom 3 Lmax A overall: 50.0, 52.0, 55.0
    assert len(peaks_df) == 3
    pivot_vals = peaks_df[("Lmax", "A")].tolist()
    assert pivot_vals == [50.0, 52.0, 55.0], f"Expected [50.0, 52.0, 55.0], got {pivot_vals}"


def test_peak_picker_k_larger_than_data(survey_with_logs):
    """peak_picker clamps k to the number of available rows."""
    peaks_df, history_df = survey_with_logs.peak_picker(
        log_name="PosB",
        pivot_col=("Lmax", "A"),
        k=100,
        high=True,
    )

    # PosB has 3 rows, so K=100 should return all 3
    assert len(peaks_df) == 3


def test_peak_picker_time_history(survey_with_logs):
    """peak_picker returns the full time history of the pivot column."""
    peaks_df, history_df = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("Lmax", "A"),
        k=2,
        high=True,
    )

    # PosA has 7 rows
    assert len(history_df) == 7
    # History should be a Series with DatetimeIndex
    assert isinstance(history_df, pd.Series)
    assert history_df.name == ("Lmax", "A")


def test_peak_picker_invalid_log(survey_with_logs):
    """peak_picker raises KeyError for unknown log names."""
    with pytest.raises(KeyError):
        survey_with_logs.peak_picker(
            log_name="NonExistent",
            pivot_col=("Lmax", "A"),
            k=3,
        )


def test_peak_picker_invalid_pivot(survey_with_logs):
    """peak_picker returns empty DataFrame for invalid pivot column."""
    peaks_df, history_df = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("L90", "A"),
        k=3,
    )
    assert peaks_df.empty


def test_peak_picker_spectral_pivot(survey_with_logs):
    """peak_picker works with spectral columns as pivot."""
    peaks_df, history_df = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("Leq", 125.0),
        k=2,
        high=True,
    )

    # Top 2 Leq 125 overall: 62.0, 60.0
    assert len(peaks_df) == 2
    pivot_vals = peaks_df[("Leq", 125.0)].tolist()
    assert pivot_vals == [62.0, 60.0], f"Expected [62.0, 60.0], got {pivot_vals}"

    # Should return all Leq columns (first level matches "Leq")
    assert ("Leq", "A") in peaks_df.columns


def test_peak_picker_exclusion_zone_skips_adjacent_peaks(survey_with_logs):
    """peak_picker with exclusion_zone_s skips peaks within the time window
    of a higher-ranked peak, picking the next-best value outside the zone."""
    peaks_df, _ = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("Lmax", "A"),
        k=3,
        high=True,
        exclusion_zone_s=900,  # 15 min — skip rows within 15 min of each peak
    )

    # PosA data has 15-minute intervals. With exclusion_zone_s=900 (15 min):
    #   - Top Lmax A = 68.0 at 2024-12-17 07:15
    #   - Next: 65.0 at 07:00 — within 15 min of 68.0, SKIP
    #   - Next: 60.0 at 2024-12-16 22:45 — outside 15 min of 07:15 (different dates), PICK
    #   - Next: 58.0 at 23:00 — within 15 min of 60.0 (22:45), SKIP
    #   - Next: 55.0 at 23:15 — outside 15 min of 22:45, PICK
    # Result: 3 peaks — 68.0, 60.0, 55.0
    assert len(peaks_df) == 3
    pivot_vals = peaks_df[("Lmax", "A")].tolist()
    assert pivot_vals == [68.0, 60.0, 55.0], (
        f"Expected [68.0, 60.0, 55.0] with exclusion_zone_s=900, got {pivot_vals}"
    )


def test_peak_picker_exclusion_zone_zero_is_noop(survey_with_logs):
    """exclusion_zone_s=0 (default) should behave identically to no exclusion zone."""
    peaks_default, _ = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("Lmax", "A"),
        k=3,
        high=True,
    )
    peaks_explicit, _ = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("Lmax", "A"),
        k=3,
        high=True,
        exclusion_zone_s=0,
    )
    pd.testing.assert_frame_equal(peaks_default, peaks_explicit)


def test_peak_picker_exclusion_zone_lowest(survey_with_logs):
    """exclusion_zone_s works with high=False (lowest)."""
    peaks_df, _ = survey_with_logs.peak_picker(
        log_name="PosA",
        pivot_col=("Lmax", "A"),
        k=2,
        high=False,
        exclusion_zone_s=900,
    )

    # Bottom Lmax A values with 15 min exclusion:
    #   - Lowest: 50.0 at 00:00
    #   - Next: 52.0 at 06:45 — outside 15 min of 00:00, PICK
    assert len(peaks_df) == 2
    assert peaks_df[("Lmax", "A")].iloc[0] == 50.0
    assert peaks_df[("Lmax", "A")].iloc[1] == 52.0
