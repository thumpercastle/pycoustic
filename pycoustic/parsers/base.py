from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseLogParser(ABC):
    @classmethod
    @abstractmethod
    def can_parse(cls, path: str | Path) -> bool:
        """Return True if this parser can handle the input file."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, path: str | Path) -> pd.DataFrame:
        """Parse the file and return normalized log data."""
        raise NotImplementedError

    @staticmethod
    def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    @staticmethod
    def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Parsed data must have a DatetimeIndex")
        return df.sort_index()

    @staticmethod
    def ensure_not_empty(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Parsed data is empty")
        return df

    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_empty(df)
        df = self.ensure_not_empty(df)
        df = self.ensure_datetime_index(df)
        return df


@staticmethod
def try_parse_datetime(series: pd.Series, formats: list[str] | None = None) -> pd.Series:
    text = series.astype(str).str.strip()
    if formats:
        for fmt in formats:
            parsed = pd.to_datetime(text, errors="coerce", format=fmt)
            if parsed.notna().any():
                return parsed
    parsed = pd.to_datetime(text, errors="coerce")
    if parsed.notna().any():
        return parsed
    raise ValueError("Could not parse datetime values")


@staticmethod
def suffix(path: str | Path) -> str:
    return Path(path).suffix.lower()