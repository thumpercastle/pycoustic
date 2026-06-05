import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import pgeocode as geo
import requests

from .weather import WeatherHistory

DECIMALS = 1
DEFAULT_PERIODS = {"day": (7, 0), "evening": (23, 0), "night": (23, 0)}

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


class Survey:
    """
    Survey Class is an overarching class which takes multiple Log objects and
    processes and summarises them together.

    This should be the main interface for user interaction with their survey data.
    """

    def __init__(self) -> None:
        self._logs: dict[str, Any] = {}
        self._weather = WeatherHistory()

    @staticmethod
    def _none_if_empty_df(df: pd.DataFrame | None) -> pd.DataFrame:
        return pd.DataFrame() if df is None else df

    @staticmethod
    def _db_mean(series: pd.Series) -> float:
        return float(np.round(10 * np.log10(np.mean(series)), DECIMALS))

    @staticmethod
    def _existing_columns(df: pd.DataFrame, cols: list[Any]) -> list[Any]:
        return [c for c in cols if c in df.columns]

    @staticmethod
    def _expected_columns(cols: list[Any]) -> Any:
        if cols and isinstance(cols[0], tuple):
            return pd.MultiIndex.from_tuples(cols)
        return cols

    @staticmethod
    def _concat_period_blocks(
            period_blocks: list[pd.DataFrame | pd.Series],
            period_names: list[str],
    ) -> pd.DataFrame:
        valid_blocks = [obj for obj in period_blocks if obj is not None]
        return pd.concat(objs=valid_blocks, axis=1, keys=period_names, names=["Period"])

    def _insert_multiindex(
            self,
            df: pd.DataFrame | pd.Series | None = None,
            super: Any = None,
            name1: str = "Position",
            name2: str = "Date",
    ) -> pd.DataFrame:
        if df is None:
            raise ValueError("No DataFrame or Series provided")

        subs = df.index.to_list()
        tuples = [(super, sub) for sub in subs]
        idx = pd.MultiIndex.from_tuples(tuples, names=[name1, name2])

        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)

        return df.set_index(idx, inplace=False)

    def _insert_header(
            self,
            df: pd.DataFrame | None = None,
            new_head_list: list[Any] | Any | None = None,
            header_idx: int | None = None,
    ) -> pd.DataFrame | None:
        if df is None or new_head_list is None:
            return df

        ncols = df.shape[1]

        if isinstance(new_head_list, (str, int, float)):
            new_head_list = [new_head_list] * ncols

        if len(new_head_list) != ncols:
            raise ValueError(
                f"new_head_list length ({len(new_head_list)}) must equal "
                f"number of columns ({ncols})."
            )

        cols = list(df.columns)
        if isinstance(df.columns, pd.MultiIndex):
            tuples = cols
        else:
            tuples = [(c,) for c in cols]

        arrays = [list(x) for x in zip(*tuples)]
        arrays.insert(header_idx, list(new_head_list))
        df.columns = pd.MultiIndex.from_arrays(arrays)
        return df

    def set_periods(self, times: dict[str, tuple[int, int]] | None = None) -> None:
        """
        Set the daytime, evening and night-time periods of all Log objects in the Survey.

        To disable evening periods, simply set it the same as night-time.

        :param times: A dictionary with strings as keys and integer tuples as values.
        The first value in the tuple represents the hour of the day that period starts at
        (24hr clock), and the second value represents the minutes past the hour.
        e.g. for daytime from 07:00 to 19:00, evening 19:00 to 23:00 and night-time
        23:00 to 07:00, times = {"day": (7, 0), "evening": (19, 0), "night": (23, 0)}
        :return: None.
        """
        if times is None:
            times = DEFAULT_PERIODS

        for log in self._logs.values():
            log.set_periods(times=times)

    def add_log(self, data: Any = None, name: str = "") -> None:
        """
        Add a Log object to the Survey object.

        :param data: Initialised Log object
        :param name: Name of the position, e.g. "A1"
        :return: None.
        """
        self._logs[name] = data

    def get_periods(self) -> dict[str, Any]:
        """
        Check the currently-set daytime, evening and night-time periods for each Log
        object in the Survey.

        :return: Tuples of start times.
        """
        periods: dict[str, Any] = {}
        for key, log in self._logs.items():
            periods[key] = log.get_period_times()
        return periods

    def broadband_summary(
            self,
            leq_cols: list[Any] | None = None,
            max_cols: list[Any] | None = None,
            lmax_n: int = 10,
            lmax_t: str = "2min",
    ) -> pd.DataFrame:
        """
        Get a dataframe summarising the parameters relevant to assessment of internal
        ambient noise levels in UK residential property assessments.

        Daytime and night-time Leqs, and nth-highest Lmax values are presented in a
        succinct table. These will be summarised as per the daytime, evening and
        night-time periods set.

        :param leq_cols: List of tuples. Columns on which to perform Leq calculations.
        :param max_cols: List of tuples. Columns on which to get nth-highest values.
        :param lmax_n: Nth-highest value for max_cols.
        :param lmax_t: Time period over which to compute nth-highest Lmax values.
        :return: Summary dataframe.
        """
        combi = pd.DataFrame()
        if leq_cols is None:
            leq_cols = [("Leq", "A")]
        if max_cols is None:
            max_cols = [("Lmax", "A")]

        for key, lg in self._logs.items():
            period_blocks: list[pd.DataFrame | pd.Series] = []
            period_names: list[str] = []

            days = lg.leq_by_date(
                lg.get_period(data=lg.get_antilogs(), period="days"),
                cols=leq_cols,
            )
            days.sort_index(inplace=True)
            period_blocks.append(days)
            period_names.append("Daytime")

            if lg.is_evening():
                evenings = lg.leq_by_date(
                    lg.get_period(data=lg.get_antilogs(), period="evenings"),
                    cols=leq_cols,
                )
                evenings.sort_index(inplace=True)
                period_blocks.append(evenings)
                period_names.append("Evening")

            nights = lg.leq_by_date(
                lg.get_period(data=lg.get_antilogs(), period="nights"),
                cols=leq_cols,
            )
            nights.sort_index(inplace=True)
            period_blocks.append(nights)
            period_names.append("Night-time")

            maxes = lg.as_interval(t=lmax_t)
            maxes = lg.get_period(data=maxes, period="nights", night_idx=True)
            max_df = lg.get_nth_high_low(n=lmax_n, data=maxes)

            existing_max_cols = self._existing_columns(max_df, max_cols)
            maxes = max_df[existing_max_cols] if existing_max_cols else pd.DataFrame(index=max_df.index)

            expected_max_cols = self._expected_columns(max_cols)
            maxes = maxes.reindex(columns=expected_max_cols)

            maxes.sort_index(inplace=True)
            try:
                maxes.index = pd.to_datetime(maxes.index)
                maxes.index = maxes.index.date
            except Exception as e:
                print(f"Error converting index to date: {e}")
            maxes.index.name = None
            period_blocks.append(maxes)
            period_names.append("Night-time")

            summary = self._concat_period_blocks(period_blocks=period_blocks, period_names=period_names)
            summary = self._insert_multiindex(df=summary, super=key)
            combi = pd.concat(objs=[combi, summary], axis=0)

        return combi.round(decimals=DECIMALS)

    def modal(
            self,
            cols: list[tuple[Any, Any]] | None = None,
            by_date: bool = False,
            day_t: str = "60min",
            evening_t: str = "60min",
            night_t: str = "15min",
            include_all: bool = False,
            all_t: str = "15min",
            averaging: str = "log",
    ) -> pd.DataFrame:
        """
        Get a dataframe summarising modal values for each time period.

        :param cols: Desired columns.
        :param by_date: If True, group modal values by date.
        :param day_t: Daytime averaging period.
        :param evening_t: Evening averaging period.
        :param night_t: Night-time averaging period.
        :param include_all: If True, add an extra "All" column covering the whole survey.
        :param all_t: Averaging period used for the all-periods calculation.
        :param averaging: ``"log"`` for energy averaging (default); ``"arithmetic"`` for
                          plain dB mean. Applied when resampling to the interval period.
        :return: Dataframe of modal values.
        """
        if cols is None:
            cols = [("L90", "A")]

        combi = pd.DataFrame()

        for key, log in self._logs.items():
            pos_summary = []
            period_headers = ["Daytime"]

            days = log.get_modal(
                data=log.get_period(data=log.as_interval(t=day_t, averaging=averaging), period="days"),
                by_date=by_date,
                cols=cols,
            )
            days.sort_index(inplace=True)
            pos_summary.append(days)

            if log.is_evening():
                period_headers.append("Evening")
                evenings = log.get_modal(
                    data=log.get_period(data=log.as_interval(t=evening_t, averaging=averaging), period="evenings"),
                    by_date=by_date,
                    cols=cols,
                )
                evenings.sort_index(inplace=True)
                pos_summary.append(evenings)

            nights = log.get_modal(
                data=log.get_period(data=log.as_interval(t=night_t, averaging=averaging), period="nights"),
                by_date=by_date,
                cols=cols,
            )
            nights.sort_index(inplace=True)
            pos_summary.append(nights)
            period_headers.append("Night-time")

            if include_all:
                period_headers.append("All")
                all_modal = log.get_modal(
                    data=log.as_interval(t=all_t, averaging=averaging),
                    by_date=by_date,
                    cols=cols,
                )
                all_modal.sort_index(inplace=True)
                pos_summary.append(all_modal)

            pos_df = pd.concat(pos_summary, axis=1)
            pos_df = self._insert_multiindex(pos_df, super=key)
            combi = pd.concat([combi, pos_df], axis=0)

        combi = self._insert_header(df=combi, new_head_list=period_headers, header_idx=0)
        combi.rename_axis(index={"Date": "#"}, inplace=True)
        return combi

    def counts(
            self,
            cols: list[tuple[Any, Any]] | None = None,
            day_t: str = "60min",
            evening_t: str = "60min",
            night_t: str = "15min",
            include_all: bool = False,
            all_t: str = "15min",
            averaging: str = "log",
    ) -> pd.DataFrame:
        """
        Returns counts for each time period.

        :param cols: Which columns to consider. Default (L90, A).
        :param day_t: Daytime averaging period.
        :param evening_t: Evening averaging period.
        :param night_t: Night-time averaging period.
        :param include_all: If True, add an extra "All" column covering the whole survey.
        :param all_t: Averaging period used for the all-periods calculation.
        :param averaging: ``"log"`` for energy averaging (default); ``"arithmetic"`` for
                          plain dB mean. Applied when resampling to the interval period.
        :return: Dataframe of counts.
        """
        if cols is None:
            cols = [("L90", "A")]

        combi = pd.DataFrame()

        for key, log in self._logs.items():
            pos_summary = []

            days = log.counts(data=log.get_period(data=log.as_interval(t=day_t, averaging=averaging), period="days"), cols=cols)
            days.sort_index(inplace=True)
            days.name = "Daytime"
            pos_summary.append(days)

            if log.is_evening():
                evenings = log.counts(
                    data=log.get_period(data=log.as_interval(t=evening_t, averaging=averaging), period="evenings"),
                    cols=cols,
                )
                evenings.sort_index(inplace=True)
                evenings.name = "Evening"
                pos_summary.append(evenings)

            nights = log.counts(data=log.get_period(data=log.as_interval(t=night_t, averaging=averaging), period="nights"), cols=cols)
            nights.sort_index(inplace=True)
            nights.name = "Night-time"
            pos_summary.append(nights)

            if include_all:
                all_counts = log.counts(data=log.as_interval(t=all_t, averaging=averaging), cols=cols)
                all_counts.sort_index(inplace=True)
                all_counts.name = "All"
                pos_summary.append(all_counts)

            pos_df = pd.concat(pos_summary, axis=1)
            pos_df = pos_df.fillna(0).astype("int64")
            pos_df.columns = pd.MultiIndex.from_product([[key], pos_df.columns], names=["Position", "Period"])
            combi = pd.concat([combi, pos_df], axis=1)

        combi = combi.fillna(0).astype("int64")
        combi.sort_index(inplace=True)
        combi.index.name = "dB"
        return combi

    def lmax_spectra(self, n: int = 10, t: str = "2min", period: str = "nights") -> pd.DataFrame:
        """
        Get spectral data for the nth-highest Lmax values during a given time period.

        This computes Lmax Event spectra. Lmax Hold spectra has not yet been implemented.

        :param n: Nth-highest Lmax.
        :param t: Time period over which to compute nth-highest Lmax values.
        :param period: "days", "evenings" or "nights".
        :return: Dataframe of nth-highest Lmax Event spectra.
        """
        combi = pd.DataFrame()

        for key, log in self._logs.items():
            combined_list = []

            max_df = log.get_nth_high_low(n=n, data=log.get_period(data=log.as_interval(t=t), period=period))
            existing_cols = [c for c in ["Lmax", "Time"] if c in max_df.columns]
            maxes = max_df[existing_cols] if existing_cols else pd.DataFrame()

            maxes.sort_index(inplace=True)
            combined_list.append(maxes)
            summary = pd.concat(objs=combined_list, axis=1)
            summary = self._insert_multiindex(df=summary, super=key)
            combi = pd.concat(objs=[combi, summary], axis=0)

        return combi.round(decimals=DECIMALS)

    def leq_spectra(self, leq_cols: list[Any] | None = None) -> pd.DataFrame:
        """
        Compute Leqs over daytime, evening and night-time periods.

        This is an overall Leq and does not group Leqs by date.

        :param leq_cols: List of strings or tuples.
        :return: Dataframe with continuous Leq computation across dates for each time period.
        """
        all_pos = []
        labels = []

        if leq_cols is None:
            leq_cols = ["Leq"]

        for label, log in self._logs.items():
            days = log.get_period(data=log.get_antilogs(), period="days")
            valid_cols = self._existing_columns(days, leq_cols)
            if valid_cols:
                days = days[valid_cols].apply(self._db_mean)
            else:
                days = pd.Series(dtype=float)

            nights = log.get_period(data=log.get_antilogs(), period="nights")
            valid_cols = self._existing_columns(nights, leq_cols)
            if valid_cols:
                nights = nights[valid_cols].apply(self._db_mean)
            else:
                nights = pd.Series(dtype=float)

            if log.is_evening():
                evenings = log.get_period(data=log.get_antilogs(), period="evenings")
                valid_cols = self._existing_columns(evenings, leq_cols)
                if valid_cols:
                    evenings = evenings[valid_cols].apply(self._db_mean)
                else:
                    evenings = pd.Series(dtype=float)
                df = pd.concat([days, evenings, nights], axis=1, keys=["Daytime", "Evening", "Night-time"])
            else:
                df = pd.concat([days, nights], axis=1, keys=["Daytime", "Night-time"])

            all_pos.append(df)
            labels.append(label)

        if not all_pos:
            return pd.DataFrame()

        combi = pd.concat(all_pos, axis=1, keys=labels)
        combi = combi.transpose().unstack(level=1)
        combi.columns = combi.columns.reorder_levels([2, 0, 1])
        combi = combi.sort_index(axis=1)

        return combi.round(decimals=DECIMALS)