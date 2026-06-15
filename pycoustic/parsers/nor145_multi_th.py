"""Parser for Norsonic Nor145 XLSX exports containing multiple time histories.

A Nor145 "multiple time history" export contains:
- One overview sheet listing all measurements with names and durations
- Pairs of sheets per measurement: one overview (metadata) and one profile (time history)
- A "Text notes" sheet

Each profile sheet contains 250 ms time-history data with broadband (A/C/Z) and
1/3-octave spectral columns (Leq, Lmax, Lmin), producing one Log per measurement.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseLogParser

# ---------------------------------------------------------------------------
# Broadband column mapping (exact matches)
# ---------------------------------------------------------------------------
DIRECT_BROADBAND_MAP: dict[str, str] = {
    "LAeq [dB]": "Leq A",
    "LAFmax [dB]": "Lmax A",
    "LAFmin [dB]": "Lmin A",
    "LApeak [dB]": "Lpeak A",
    "LCeq [dB]": "Leq C",
    "LCFmax [dB]": "Lmax C",
    "LCFmin [dB]": "Lmin C",
    "LCpeak [dB]": "Lpeak C",
    "LZeq [dB]": "Leq Z",
    "LZFmax [dB]": "Lmax Z",
    "LZFmin [dB]": "Lmin Z",
    "LZpeak [dB]": "Lpeak Z",
}

# ---------------------------------------------------------------------------
# Spectral family mapping: Nor145 metric → pycoustic metric
# ---------------------------------------------------------------------------
SPECTRAL_FAMILY_MAP: dict[str, str] = {
    "Lfeq": "Leq",
    "LfFmax": "Lmax",
    "LfFmin": "Lmin",
}

# Regex for spectral headers like:
#   "Lfeq 20 Hz  (1/3) [dB]"
#   "LfFmax 1 kHz  (1/3) [dB]"
#   "LfFmin 1.25 kHz  (1/3) [dB]"
_SPECTRAL_RE = re.compile(
    r"^(Lfeq|LfFmax|LfFmin)\s+"
    r"(\d+(?:\.\d+)?)\s*"
    r"(Hz|kHz)"
    r".*",
    re.IGNORECASE,
)


class Nor145MultipleTHParser(BaseLogParser):
    """Parser for Nor145 XLSX exports with multiple time histories.

    Each time-history (Profile) sheet is parsed into a flat-column DataFrame
    with pycoustic-compatible column names (e.g. ``"Leq A"``, ``"Lmax 125"``)
    and a ``DatetimeIndex``.

    Usage::

        parser = Nor145MultipleTHParser()
        for name, df in parser.parse_all(path):
            log = Log.from_dataframe(df, filepath=path, name=name)
    """

    @classmethod
    def can_parse(cls, path: str | Path) -> bool:
        file_path = Path(path)
        if file_path.suffix.lower() != ".xlsx":
            return False

        try:
            workbook = pd.ExcelFile(file_path)
        except Exception:
            return False

        profile_count = 0
        for sheet_name in workbook.sheet_names:
            if cls._is_profile_sheet(file_path, sheet_name):
                profile_count += 1
                if profile_count >= 2:
                    return True

        # A single profile sheet is enough — the headers are
        # specific enough to avoid false positives on generic XLSX.
        return profile_count >= 1

    @classmethod
    def _is_profile_sheet(cls, path: str | Path, sheet_name: str) -> bool:
        """Return True if *sheet_name* is a Nor145 time-history profile sheet."""
        try:
            probe = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=3)
        except Exception:
            return False

        if probe.empty or probe.shape[0] < 2:
            return False

        # Row 1 (index 1) should contain LAeq / LCeq / LZeq column headers
        row1_values = [
            str(v).strip()
            for v in probe.iloc[1].tolist()
            if pd.notna(v)
        ]
        return any(
            val in DIRECT_BROADBAND_MAP or val in SPECTRAL_FAMILY_MAP
            for val in row1_values
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, path: str | Path) -> pd.DataFrame:
        """Not implemented — use :meth:`parse_all` for multi-TH files."""
        raise NotImplementedError(
            "Nor145MultipleTHParser produces multiple DataFrames. "
            "Use parse_all() instead."
        )

    def parse_all(self, path: str | Path) -> list[tuple[str, pd.DataFrame]]:
        """Parse all time-history sheets and return ``(name, DataFrame)`` pairs.

        Each DataFrame has:
        - a :class:`~pandas.DatetimeIndex`
        - flat string column names in pycoustic format (e.g. ``"Leq A"``)
        """
        file_path = Path(path)
        results: list[tuple[str, pd.DataFrame]] = []

        workbook = pd.ExcelFile(file_path)
        for sheet_name in workbook.sheet_names:
            if not self._is_profile_sheet(file_path, sheet_name):
                continue

            try:
                name, df = self._parse_single_profile(file_path, sheet_name)
                if name and df is not None and not df.empty:
                    results.append((name, df))
            except Exception:
                # Skip sheets that fail to parse — a single bad sheet
                # shouldn't prevent the rest from loading.
                continue

        if not results:
            raise ValueError(
                f"No profile sheets could be parsed from: {file_path}"
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_single_profile(
        self, path: Path, sheet_name: str
    ) -> tuple[str, pd.DataFrame | None]:
        """Parse one profile sheet into (measurement_name, DataFrame)."""
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

        # Row 0: measurement name (col 0)
        measurement_name = str(raw.iloc[0, 0]).strip()

        # Row 1: column headers
        raw_headers = raw.iloc[1].tolist()

        # Build rename map and identify the timestamp column
        rename_map: dict[int, str] = {}
        time_col_idx: int | None = None

        for i, header in enumerate(raw_headers):
            # --- Timestamp detection ---
            # In Nor145 profile sheets the timestamp is in col 1 and the
            # header above it is usually NaN.  Always treat column 1 as
            # the time column regardless of its header value.
            if i == 1:
                time_col_idx = i
                rename_map[i] = "Time"
                continue

            if not pd.notna(header):
                continue

            header_str = str(header).strip()

            # --- Broadband mapping ---
            if header_str in DIRECT_BROADBAND_MAP:
                rename_map[i] = DIRECT_BROADBAND_MAP[header_str]
                continue

            # --- Spectral mapping ---
            spectral_match = _SPECTRAL_RE.match(header_str)
            if spectral_match:
                metric_key = spectral_match.group(1)
                freq_str = spectral_match.group(2)
                unit = spectral_match.group(3).lower()
                freq_hz = float(freq_str) * (1000 if unit == "khz" else 1)
                # Use int for whole-number frequencies (63, 125, ...) and
                # keep the float for fractional bands like 31.5 Hz.
                if freq_hz == int(freq_hz):
                    freq_hz = int(freq_hz)
                metric = SPECTRAL_FAMILY_MAP.get(metric_key, metric_key)
                rename_map[i] = f"{metric} {freq_hz}"
                continue

            # Unknown columns are skipped (e.g. "Markers", counter column)
            rename_map[i] = f"_skip_{i}"

        if time_col_idx is None:
            raise ValueError(f"No timestamp column found in sheet {sheet_name!r}")

        # --- Build output DataFrame ---
        # Data rows start at row 2
        data = raw.iloc[2:].copy()

        # Rename columns using our map
        data.rename(columns=rename_map, inplace=True)

        # Drop skipped columns
        skip_cols = [c for c in data.columns if str(c).startswith("_skip_")]
        data.drop(columns=skip_cols, inplace=True)

        # Parse timestamp
        data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
        data.dropna(subset=["Time"], inplace=True)
        data.set_index("Time", inplace=True)

        # Coerce remaining columns to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Drop fully-NaN columns and rows
        data = data.dropna(axis=1, how="all").dropna(axis=0, how="all")

        if data.empty:
            return measurement_name, None

        # Finalise via base-class pipeline (sort index, validate not empty)
        data = self.finalize(data)

        return measurement_name, data