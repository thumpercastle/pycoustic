import re
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseLogParser


class Nor140OverviewXlsxParser(BaseLogParser):
    """
    Parser for Norsonic Nor140 overview XLSX exports.

    Expected output:
    - DatetimeIndex
    - flat string column names such as:
        "Leq A", "Lmax A", "L90 A", "Leq 125", "Lmax 1000"
    """

    SPECTRAL_GROUPS = {"Lfeq", "LfFmax", "LfF,Perc2", "LfF,Perc4", "LfF,Perc6"}

    DIRECT_MAP = {
        "LAeq": "Leq A",
        "LAFmax": "Lmax A",
        "Lzeq": "Leq Z",
        "LZFmax": "Lmax Z",
        "LAF,Perc6": "L90 A",
        "LAF,Perc2": "Perc2 A",
        "LAF,Perc4": "Perc4 A",
        "LZF,Perc6": "L90 Z",
        "LZF,Perc2": "Perc2 Z",
        "LZF,Perc4": "Perc4 Z",
    }

    METRIC_MAP = {
        "Lfeq": "Leq",
        "LfFmax": "Lmax",
        "LfF,Perc6": "L90",
        "LfF,Perc2": "Perc2",
        "LfF,Perc4": "Perc4",
    }

    @classmethod
    def can_parse(cls, path: str | Path) -> bool:
        file_path = Path(path)
        if file_path.suffix.lower() != ".xlsx":
            return False

        try:
            workbook = pd.ExcelFile(file_path)
        except Exception:
            return False

        for sheet_name in workbook.sheet_names:
            try:
                probe = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            except Exception:
                continue

            probe = probe.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if probe.empty:
                continue

            for i in range(min(len(probe), 10)):
                row_values = probe.iloc[i].tolist()
                cleaned = [str(v).strip() for v in row_values if pd.notna(v) and str(v).strip()]
                if "File" in cleaned and "Date" in cleaned:
                    return True

        return False

    def parse(self, path: str | Path) -> pd.DataFrame:
        file_path = Path(path)
        workbook = pd.ExcelFile(file_path)
        last_error: Exception | None = None

        for sheet_name in workbook.sheet_names:
            try:
                raw = self._read_sheet(file_path, sheet_name)
                renamed = raw.rename(columns=self._build_rename_map(raw.columns))
                time_index = self._extract_time_index(renamed)

                if time_index is None or time_index.notna().sum() == 0:
                    raise ValueError(f"No usable datetime data found in sheet {sheet_name!r}.")

                renamed["Time"] = time_index
                df = renamed.dropna(subset=["Time"]).set_index("Time")
                df = self._drop_time_like_columns(df)
                df = self._coerce_numeric(df)

                if df.empty:
                    raise ValueError(f"Parsed sheet {sheet_name!r} but no usable data rows remained.")

                return self.finalize(df)

            except Exception as exc:
                last_error = exc

        raise ValueError(
            f"Could not identify a usable Nor140 overview sheet in workbook: {file_path}"
        ) from last_error

    @classmethod
    def _build_rename_map(cls, columns: list[Any] | pd.Index) -> dict[str, str]:
        rename_map: dict[str, str] = {}
        for col in columns:
            mapped = cls._map_column(col)
            if mapped is not None:
                rename_map[str(col)] = mapped
        return rename_map

    @staticmethod
    def _to_float_if_possible(value: str) -> Any:
        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def _normalise_band_token(token: str) -> str:
        token = str(token).strip().replace(",", ".")
        token = re.sub(r"\s+", " ", token)

        if token.lower().endswith("khz"):
            number = token[:-3].strip()
            return str(int(round(float(number) * 1000)))

        if token.lower().endswith("hz"):
            number = token[:-2].strip()
            return str(int(round(float(number))))

        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return str(int(round(float(token))))

        return token

    @classmethod
    def _map_column(cls, column: Any) -> str | None:
        if column is None:
            return None

        text = str(column).strip()
        if not text:
            return None

        compact = re.sub(r"\s+", "", text)

        if compact.lower() in {"time", "datetime", "date/time"}:
            return "Time"

        if compact in cls.DIRECT_MAP:
            return cls.DIRECT_MAP[compact]

        match = re.fullmatch(r"(Lf(?:eq|Fmax|F,Perc[246]))(.+)", compact)
        if not match:
            return text

        prefix, band = match.groups()
        metric = cls.METRIC_MAP.get(prefix)
        if metric is None:
            return text

        band = cls._normalise_band_token(band)
        return f"{metric} {band}"

    @classmethod
    def _looks_like_header_row(cls, values: list[Any]) -> bool:
        cleaned = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
        if not cleaned:
            return False

        compact = [re.sub(r"\s+", "", value) for value in cleaned]
        hits = 0

        for value in compact:
            lower = value.lower()
            if lower in {"time", "date", "datetime", "date/time", "timestamp"}:
                hits += 1
            elif value in cls.DIRECT_MAP:
                hits += 1
            elif re.fullmatch(r"Lf(?:eq|Fmax|F,Perc[246]).*", value):
                hits += 1

        return hits >= 2

    @classmethod
    def _read_sheet(cls, path: str | Path, sheet_name: int | str) -> pd.DataFrame:
        probe = pd.read_excel(path, sheet_name=sheet_name, header=None)
        probe = probe.dropna(axis=0, how="all").dropna(axis=1, how="all")

        header_row = None
        for i in range(len(probe)):
            row_values = probe.iloc[i].tolist()
            cleaned = [str(v).strip() for v in row_values if pd.notna(v) and str(v).strip()]
            if "File" in cleaned and "Date" in cleaned:
                header_row = i
                break
            if cls._looks_like_header_row(row_values):
                header_row = i
                break

        if header_row is None:
            raise ValueError(f"Could not find a Nor140 header row in sheet {sheet_name!r}.")

        subheader_row = header_row + 1 if header_row + 1 < len(probe) else None
        header_values = probe.iloc[header_row].tolist()
        subheader_values = probe.iloc[subheader_row].tolist() if subheader_row is not None else []

        columns: list[str] = []
        current_group = ""

        for i, head in enumerate(header_values):
            head_text = "" if pd.isna(head) else str(head).strip()
            sub_text = ""
            if i < len(subheader_values):
                sub = subheader_values[i]
                sub_text = "" if pd.isna(sub) else str(sub).strip()

            if head_text:
                current_group = head_text

            if current_group in cls.SPECTRAL_GROUPS and sub_text:
                columns.append(f"{current_group} {sub_text}")
            else:
                columns.append(current_group or head_text or f"Unnamed: {i}")

        data_start = header_row + 2
        df = probe.iloc[data_start:].copy()
        df.columns = [str(col).strip() for col in columns]
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return df

    @classmethod
    def _find_time_column(cls, columns: list[Any]) -> str | None:
        normalised = {
            str(column).strip(): re.sub(r"\s+", "", str(column).strip()).lower()
            for column in columns
        }

        for original, compact in normalised.items():
            if compact in {"time", "datetime", "date/time", "timestamp"}:
                return original

        for original, compact in normalised.items():
            if "time" in compact and "run" not in compact:
                return original

        return None

    @classmethod
    def _find_date_column(cls, columns: list[Any]) -> str | None:
        for column in columns:
            compact = re.sub(r"\s+", "", str(column).strip()).lower()
            if compact in {"date", "measurementdate"}:
                return str(column)
        return None

    @staticmethod
    def _coerce_time(series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(r"^\((.*)\)$", r"\1", regex=True)
        )

        parsed = pd.to_datetime(cleaned, errors="coerce", format="%Y/%m/%d %H:%M:%S.%f")
        if parsed.notna().any():
            return parsed

        parsed = pd.to_datetime(cleaned, errors="coerce", dayfirst=True)
        if parsed.notna().any():
            return parsed

        parsed = pd.to_datetime(cleaned, errors="coerce", dayfirst=False)
        if parsed.notna().any():
            return parsed

        raise ValueError("Could not parse Nor140 time column into datetimes.")

    @staticmethod
    def _coerce_date(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.strip().str.replace(r"^\((.*)\)$", r"\1", regex=True)

        parsed = pd.to_datetime(cleaned, errors="coerce", format="%Y/%m/%d %H:%M:%S.%f")
        if parsed.notna().any():
            return parsed.dt.normalize()

        parsed = pd.to_datetime(cleaned, errors="coerce", dayfirst=True)
        if parsed.notna().any():
            return parsed.dt.normalize()

        parsed = pd.to_datetime(cleaned, errors="coerce", dayfirst=False)
        if parsed.notna().any():
            return parsed.dt.normalize()

        raise ValueError("Could not parse Nor140 date column into datetimes.")

    @classmethod
    def _combine_date_time(cls, date_series: pd.Series, time_series: pd.Series) -> pd.Series:
        parsed_date = cls._coerce_date(date_series)
        time_as_str = time_series.astype(str).str.strip()

        combined = pd.to_datetime(
            parsed_date.dt.strftime("%Y-%m-%d") + " " + time_as_str,
            errors="coerce",
            dayfirst=True,
            )
        if combined.notna().any():
            return combined

        time_as_dt = pd.to_datetime(time_series, errors="coerce")
        if time_as_dt.notna().any():
            return parsed_date + (
                    time_as_dt.dt.hour.fillna(0).astype(int).map(lambda x: pd.Timedelta(hours=x))
                    + time_as_dt.dt.minute.fillna(0).astype(int).map(lambda x: pd.Timedelta(minutes=x))
                    + time_as_dt.dt.second.fillna(0).astype(int).map(lambda x: pd.Timedelta(seconds=x))
            )

        raise ValueError("Could not combine Nor140 date and time columns.")

    @classmethod
    def _extract_time_index(cls, df: pd.DataFrame) -> pd.Series | None:
        time_col = cls._find_time_column(list(df.columns))
        date_col = cls._find_date_column(list(df.columns))

        if time_col is not None:
            try:
                parsed = cls._coerce_time(df[time_col])
                if parsed.notna().any():
                    return parsed
            except ValueError:
                pass

        if date_col is not None:
            try:
                parsed = cls._coerce_time(df[date_col])
                if parsed.notna().any():
                    return parsed
            except ValueError:
                pass

        if date_col is not None and time_col is not None:
            try:
                parsed = cls._combine_date_time(df[date_col], df[time_col])
                if parsed.notna().any():
                    return parsed
            except ValueError:
                pass

        for col in list(df.columns)[:5]:
            try:
                parsed = cls._coerce_time(df[col])
                if parsed.notna().any():
                    return parsed
            except ValueError:
                continue

        return None

    @staticmethod
    def _drop_time_like_columns(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols: list[str] = []

        for col in df.columns:
            compact = re.sub(r"\s+", "", str(col).strip()).lower()
            if compact in {
                "date",
                "time",
                "datetime",
                "date/time",
                "timestamp",
                "measurementtime",
                "measurementdate",
                "file",
                "duration",
                "status",
            }:
                drop_cols.append(col)

        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        return df

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(axis=1, how="all").sort_index()