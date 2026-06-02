import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

DECIMALS = 0
DEFAULT_PERIODS = {"day": (7, 0), "evening": (23, 0), "night": (23, 0)}
NIGHT_IDX_LABEL = "Night idx"
NIGHT_IDX_COLUMN = (NIGHT_IDX_LABEL, "")

wb_weighting_dB = [
    -8.4,
    -8.3,
    -8.3,
    -8.1,
    -7.6,
    -6.1,
    -3.5,
    -1.1,
    0.2,
    0.5,
    0.2,
    -0.2,
    -0.9,
    -1.8,
    -3.0,
    -4.5,
    -6.2,
    -8.1,
    -10.1,
    -12.4,
    -15.2,
    -18.8,
    -23.2,
    -28.4,
    -34.0,
    -39.8,
    -45.8,
]

wb_weighting_factors = [
    0.381,
    0.385,
    0.386,
    0.392,
    0.417,
    0.496,
    0.665,
    0.885,
    1.026,
    1.054,
    1.026,
    0.974,
    0.904,
    0.814,
    0.709,
    0.597,
    0.491,
    0.395,
    0.312,
    0.239,
    0.173,
    0.115,
    0.069,
    0.039,
    0.02,
    0.02,
    0.005,
]


class Log:
    def __init__(self, path: str = "") -> None:
        """
        Store measured noise data from one data logger.

        The data must be provided in a CSV with headings in the format
        "Leq A", "L90 125", etc.

        :param path: File path for the CSV noise data.
        """
        self._filepath = path
        self._master = pd.read_csv(
            path,
            index_col="Time",
            parse_dates=["Time"],
            date_format="%Y/%m/%d %H:%M",
        )
        self._master.index = pd.to_datetime(self._master.index)
        self._master = self._master.sort_index(axis=1)
        self._start = self._master.index.min()
        self._end = self._master.index.max()

        self._assign_header()

        self._night_start = None
        self._day_start = None
        self._evening_start = None
        self._init_periods()

        self._antilogs = self._prep_antilogs()
        self._master = self._append_night_idx(data=self._master)
        self._antilogs = self._append_night_idx(data=self._antilogs)

        self._decimals = 1

    @staticmethod
    def _build_time(hour_minute: tuple[int, int]) -> dt.time:
        return dt.time(hour_minute[0], hour_minute[1])

    @staticmethod
    def _db_from_antilog(value: Any, decimals: int) -> Any:
        return np.round((10 * np.log10(value)), decimals)

    @staticmethod
    def _empty_like_columns(columns: Any) -> pd.DataFrame:
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _none_if_zero(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series | None:
        return None if len(df) == 0 else df

    @staticmethod
    def _first_existing_column(df: pd.DataFrame, candidates: list[Any]) -> Any | None:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    @staticmethod
    def _period_alias(period: str) -> str:
        aliases = {
            "day": "days",
            "days": "days",
            "evening": "evenings",
            "evenings": "evenings",
            "night": "nights",
            "nights": "nights",
        }
        return aliases.get(period, period)

    def _assign_header(self) -> None:
        """
        Convert single-string CSV headers into a two-level column index.

        Example:
            "Leq A" -> ("Leq", "A")
            "L90 125" -> ("L90", 125.0)
        """
        csv_headers = self._master.columns.to_list()
        superheaders: list[Any] = []
        subheaders: list[Any] = []

        for item in csv_headers:
            parts = item.split(" ", 1)
            superheaders.append(parts[0])
            subheaders.append(parts[1] if len(parts) > 1 else "")

        for i, value in enumerate(subheaders):
            try:
                subheaders[i] = float(value)
            except ValueError:
                pass

        self._master.columns = [superheaders, subheaders]
        self._master.sort_index(axis=1, level=1, inplace=True)

    def _init_periods(self) -> None:
        """
        Initialise default daytime, evening, and night-time periods.
        """
        self._day_start = self._build_time(DEFAULT_PERIODS["day"])
        self._evening_start = self._build_time(DEFAULT_PERIODS["evening"])
        self._night_start = self._build_time(DEFAULT_PERIODS["night"])

    def _prep_antilogs(self) -> pd.DataFrame:
        """
        Create a copy of the master dataframe with dB values converted to antilogs.
        This dataframe should be used for Leq-type calculations.
        """
        return self._master.copy().apply(lambda x: np.power(10, (x / 10)))

    def _append_night_idx(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Append a 'Night idx' column with early-morning timestamps shifted back by one day.

        This allows night-time periods crossing midnight to be processed as a contiguous block.
        """
        if data is None:
            raise ValueError("No DataFrame provided")

        night_indices = data.index.to_list()
        if self._night_start > self._day_start:
            for i in range(len(night_indices)):
                if night_indices[i].time() < self._day_start:
                    night_indices[i] += dt.timedelta(days=-1)

        data[NIGHT_IDX_LABEL] = night_indices
        return data

    def _return_as_night_idx(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Return the dataframe with the night index set as the active index.
        """
        if data is None:
            raise ValueError("No DataFrame provided")
        if NIGHT_IDX_COLUMN not in data.columns:
            raise Exception("No night indices in current DataFrame")
        return data.set_index(NIGHT_IDX_LABEL)

    def _recompute_leq(
            self,
            data: pd.DataFrame | None = None,
            t: str = "15min",
            cols: list[Any] | None = None,
    ) -> pd.DataFrame | None:
        """
        Recompute shorter Leq-style measurements as longer periods.

        :param data: Input dataframe in antilog format.
        :param t: Desired output period.
        :param cols: Columns to recompute.
        :return: Recomputed dataframe or None.
        """
        if data is None:
            data = self._antilogs
        if cols is None:
            cols = ["Leq", "L90"]

        recomputed = self._empty_like_columns(data.columns)
        for idx in cols:
            if idx in data.columns:
                recomputed[idx] = data[idx].resample(t).mean().apply(
                    lambda x: self._db_from_antilog(x, self._decimals)
                )
        return self._none_if_zero(recomputed)

    def _recompute_night_idx(self, data: pd.DataFrame | None = None, t: str = "15min") -> pd.DataFrame | None:
        """
        Recompute the night index column for resampled data.

        :param data: Input dataframe to be recomputed.
        :param t: Desired measurement period.
        :return: Dataframe with recomputed night index column.
        """
        if data is None:
            raise Exception("No DataFrame provided for night idx")

        if NIGHT_IDX_COLUMN in data.columns:
            data[NIGHT_IDX_LABEL] = data[NIGHT_IDX_LABEL].resample(t).asfreq()
        else:
            data[NIGHT_IDX_LABEL] = self._master[NIGHT_IDX_LABEL].resample(t).asfreq()
            return data
        return data

    def _recompute_max(
            self,
            data: pd.DataFrame | None = None,
            t: str = "15min",
            pivot_cols: list[tuple[Any, Any]] | None = None,
            hold_spectrum: bool = False,
    ) -> pd.DataFrame:
        """
        Recompute maximum readings from shorter to longer periods.

        :param data: Input dataframe, usually self._master.
        :param t: Desired measurement period.
        :param pivot_cols: Column(s) used to determine maxima.
        :param hold_spectrum: If True, use hold maxima across bands; otherwise use event maxima.
        :return: Dataframe with recomputed maxima.
        """
        if pivot_cols is None:
            pivot_cols = [("Lmax", "A")]
        if data is None:
            data = self._master

        combined = self._empty_like_columns(data.columns)

        for col in pivot_cols:
            if col not in data.columns:
                continue

            metric = col[0]

            if hold_spectrum:
                combined[metric] = data.resample(t)[metric].max()
            else:
                combined[metric] = data[metric].groupby(pd.Grouper(freq=t)).max()

        return combined

    def _as_multiindex(
            self,
            df: pd.DataFrame | pd.Series | None = None,
            super: Any = None,
            name1: str = "Date",
            name2: str = "Num",
    ) -> pd.DataFrame:
        """
        Return the input series/dataframe with a two-level row index.
        """
        if df is None:
            raise ValueError("No DataFrame or Series provided")

        subs = df.index.to_list()
        tuples = [(super, sub) for sub in subs]
        idx = pd.MultiIndex.from_tuples(tuples, names=[name1, name2])

        if isinstance(df, pd.Series):
            df = pd.DataFrame(data=df)

        return df.set_index(idx, inplace=False)

    def get_period(
            self,
            data: pd.DataFrame | None = None,
            period: str = "days",
            night_idx: bool = True,
    ) -> pd.DataFrame | None:
        """
        Get data for daytime, evening, or night-time periods.

        :param data: Input data, usually master.
        :param period: One of "days", "evenings", or "nights".
        :param night_idx: If True, uses contiguous night-time periods across midnight.
        :return: Filtered dataframe.
        """
        if data is None:
            data = self._master

        period = self._period_alias(period)

        if period == "days":
            return data.between_time(self._day_start, self._evening_start, inclusive="left")
        if period == "evenings":
            return data.between_time(self._evening_start, self._night_start, inclusive="left")
        if period == "nights":
            if night_idx:
                data = self._return_as_night_idx(data=data)
            return data.between_time(self._night_start, self._day_start, inclusive="left")
        return None

    def leq_by_date(self, data: pd.DataFrame, cols: list[Any] | None = None) -> pd.DataFrame:
        """
        Calculate Leq values grouped by date.

        :param data: Input data in antilog format, usually with night-time index for nights.
        :param cols: Columns to recalculate.
        :return: Dataframe of calculated Leq values grouped by date.
        """
        if cols is None:
            cols = ["Leq"]

        existing_cols = [c for c in cols if c in data.columns]

        if isinstance(cols[0], tuple):
            expected_cols = pd.MultiIndex.from_tuples(cols)
        else:
            expected_cols = cols

        if not existing_cols:
            idx = np.unique(data.index.date) if not data.empty else []
            return pd.DataFrame(index=idx, columns=expected_cols, dtype=float)

        res = data[existing_cols].groupby(data.index.date).mean().apply(
            lambda x: self._db_from_antilog(x, self._decimals)
        )
        return res.reindex(columns=expected_cols)

    def get_data(self) -> pd.DataFrame:
        """
        Return the loaded CSV data as a dataframe.
        """
        return self._master

    def get_antilogs(self) -> pd.DataFrame:
        """
        Return the antilog version of the loaded data.
        """
        return self._antilogs

    def as_interval(
            self,
            data: pd.DataFrame | None = None,
            antilogs: pd.DataFrame | None = None,
            t: str = "15min",
            leq_cols: list[Any] | None = None,
            max_pivots: list[tuple[Any, Any]] | None = None,
            hold_spectrum: bool = False,
    ) -> pd.DataFrame:
        """
        Return the data recomputed as longer time intervals.

        :param data: Input dataframe, usually master.
        :param antilogs: Antilog dataframe used for Leq calculations.
        :param t: Desired output period.
        :param leq_cols: Leq columns to include.
        :param max_pivots: Values used to pivot the Lmax recalculation.
        :param hold_spectrum: True for Lmax hold, False for Lmax event.
        :return: Recalculated dataframe.
        """
        if data is None:
            data = self._master
        if antilogs is None:
            antilogs = self._antilogs
        if leq_cols is None:
            leq_cols = ["Leq", "L90"]
        if max_pivots is None:
            max_pivots = [("Lmax", "A")]

        leq = self._recompute_leq(data=antilogs, t=t, cols=leq_cols)
        maxes = self._recompute_max(data=data, t=t, pivot_cols=max_pivots, hold_spectrum=hold_spectrum)
        conc = pd.concat([leq, maxes], axis=1).sort_index(axis=1).dropna(axis=1, how="all")
        conc = self._append_night_idx(data=conc)
        return conc.dropna(axis=0, how="all")

    def get_nth_high_low(
            self,
            n: int = 10,
            data: pd.DataFrame | None = None,
            pivot_col: tuple[Any, Any] | None = None,
            all_cols: bool = False,
            high: bool = True,
    ) -> pd.DataFrame:
        """
        Return the nth-highest or nth-lowest values for the specified parameter.

        :param n: The rank to return.
        :param data: Input dataframe.
        :param pivot_col: Column used for ranking.
        :param all_cols: If True, return all columns for the selected row.
        :param high: True for highest values, False for lowest.
        :return: Dataframe of nth-ranked values.
        """
        if data is None:
            data = self._master
        if pivot_col is None:
            pivot_col = ("Lmax", "A")

        if pivot_col not in data.columns:
            return pd.DataFrame()

        nth = data.sort_values(by=pivot_col, ascending=not high)
        nth["Time"] = nth.index.time

        if all_cols:
            return nth.groupby(by=nth.index.date).nth(n - 1)
        return nth[[pivot_col[0], "Time"]].groupby(by=nth.index.date).nth(n - 1)

    def get_modal(
            self,
            data: pd.DataFrame | None = None,
            by_date: bool = True,
            cols: list[tuple[Any, Any]] | None = None,
            round_decimals: bool = True,
    ) -> pd.DataFrame | pd.Series:
        """
        Return modal values for the selected columns.

        :param data: Input dataframe, usually master.
        :param by_date: If True, group modal values by date.
        :param cols: Desired columns.
        :param round_decimals: If True, round values before calculating mode.
        :return: Dataframe or series of modal values.
        """
        if data is None:
            data = self._master
        if round_decimals:
            data = data.round()
        if cols is None:
            cols = [("L90", "A")]

        existing_cols = [c for c in cols if c in data.columns]
        if not existing_cols:
            return pd.DataFrame() if by_date else pd.Series(dtype=float)

        if by_date:
            dates = np.unique(data.index.date)
            modes_by_date = pd.DataFrame()
            for date in dates:
                date_str = date.strftime("%Y-%m-%d")
                try:
                    mode_by_date = data[existing_cols].loc[date_str].mode()
                    mode_by_date = self._as_multiindex(df=mode_by_date, super=date_str)
                    modes_by_date = pd.concat([modes_by_date, mode_by_date])
                except KeyError:
                    pass
            return modes_by_date

        return data[existing_cols].mode()

    def counts(
            self,
            data: pd.DataFrame | None = None,
            cols: list[tuple[Any, Any]] | None = None,
            round_decimals: bool = True,
    ) -> pd.Series:
        """
        Return counts for the selected column values.

        Behaviour remains aligned with the original implementation, which effectively
        supports a single selected column for the index flattening step.

        :param data: Input dataframe, usually master.
        :param cols: Desired columns.
        :param round_decimals: If True, round values before counting.
        :return: Series of counts.
        """
        if data is None:
            data = self._master
        if round_decimals:
            data = data.round(decimals=0)
        if cols is None:
            cols = [("L90", "A")]

        existing_cols = [c for c in cols if c in data.columns]
        if not existing_cols:
            return pd.Series(dtype="int64")

        selected_col = self._first_existing_column(data, existing_cols)
        if selected_col is None:
            return pd.Series(dtype="int64")

        df = data[[selected_col]].value_counts()
        df.index = [int(x[0]) for x in df.index]
        return df

    def set_periods(self, times: dict[str, tuple[int, int]] | None = None) -> None:
        """
        Set daytime, evening, and night-time periods.

        To disable evening periods, set evening equal to night.

        :param times: Dictionary with keys "day", "evening", and "night",
                      each mapped to an (hour, minute) tuple.
        """
        if times is None:
            times = DEFAULT_PERIODS

        self._day_start = self._build_time(times["day"])
        self._evening_start = self._build_time(times["evening"])
        self._night_start = self._build_time(times["night"])

        self._master.drop(labels=NIGHT_IDX_LABEL, axis=1, level=0, inplace=True)
        self._antilogs.drop(labels=NIGHT_IDX_LABEL, axis=1, level=0, inplace=True)
        self._master = self._append_night_idx(data=self._master)
        self._antilogs = self._append_night_idx(data=self._antilogs)

    def get_period_times(self) -> tuple[dt.time, dt.time, dt.time]:
        """
        Return the tuples of period start times.
        """
        return self._day_start, self._evening_start, self._night_start

    def is_evening(self) -> bool:
        """
        Check if evening periods are enabled.

        :return: True if evening periods are enabled, otherwise False.
        """
        return self._evening_start != self._night_start

    def get_start(self) -> pd.Timestamp:
        return self._start

    def get_end(self) -> pd.Timestamp:
        return self._end


class VibLog(Log):
    """
    This class is a work in progress. Do not use.
    """

    def __init__(self, path: str, units: str | None = None) -> None:
        super().__init__(path)
        self._units = units
        if self._units is None:
            self._units = "ms2"
        self._decimals = 12
        self._wb_weighted = self.apply_wb_weighting()

    def head(self) -> pd.DataFrame:
        return self._master.head()

    def apply_wb_weighting(self, factors: list[float] = wb_weighting_factors) -> pd.DataFrame:
        """
        Multiply each numeric column in self._master by the corresponding WB weighting factor
        and store the result in self._wb_weighted.

        Notes:
        - Applies only to numeric columns.
        - Non-numeric columns, such as 'Night idx', are left unchanged.
        - The mapping is by column order: factor[i] is applied to the i-th numeric column.
        """
        numeric_cols = self._master.select_dtypes(include="number").columns

        if len(numeric_cols) != len(factors):
            raise ValueError(
                f"WB weighting length mismatch: {len(numeric_cols)} numeric columns in _master "
                f"but {len(factors)} factors were provided."
            )

        weighted = self._master.copy()
        weighted.loc[:, numeric_cols] = weighted.loc[:, numeric_cols].mul(factors, axis=1)
        self._wb_weighted = weighted
        return self._wb_weighted

    def sum_across_bands(self, df: pd.DataFrame | str | None = None) -> pd.DataFrame:
        """
        Sum values across band columns and return a dataframe with:
        - a new column 'Sum' containing one summed figure per row
        - the existing 'Night idx' column preserved
        - the original index preserved
        """
        if df is None:
            df = self._master
        elif df == "wb_weighted":
            df = self._wb_weighted

        night_candidates = [NIGHT_IDX_LABEL, NIGHT_IDX_COLUMN]
        night_col = self._first_existing_column(df, night_candidates)
        if night_col is None:
            raise ValueError(f"Expected column {NIGHT_IDX_LABEL!r} in DataFrame.")

        numeric_cols = df.select_dtypes(include="number").columns
        numeric_cols = [c for c in numeric_cols if c != night_col]

        out = df.copy()
        out["Sum"] = out.loc[:, numeric_cols].sum(axis=1)
        return out

    def evdv_from_rms(
            self,
            df: pd.DataFrame | pd.Series | None = None,
            t: float | int | None = None,
            col: Any = None,
    ) -> float:
        """
        Compute eVDV (estimated Vibration Dose Value) from a time history of RMS acceleration.

        :param df: Series or dataframe containing RMS acceleration values.
        :param t: Assessment interval in seconds.
        :param col: Column to select when df is a dataframe.
        :return: eVDV as a float.
        """
        if t is None:
            raise ValueError("Parameter 't' (measurement interval in seconds) is required.")
        if t <= 0:
            raise ValueError("Parameter 't' must be a positive number of seconds.")

        data = self._master if df is None else df

        if isinstance(data, pd.Series):
            a_rms = data
        else:
            if col is None:
                raise ValueError("When 'df' is a DataFrame, you must provide 'col' to select the a_rms column.")
            a_rms = data[col]

        a_rms = pd.to_numeric(a_rms, errors="coerce").dropna()
        if len(a_rms) == 0:
            raise ValueError("No valid numeric RMS acceleration values available to compute eVDV.")

        a_eq = a_rms * 1.4
        evdv_val = float(((a_eq.pow(4).sum() * float(t)) ** 0.25))
        return evdv_val