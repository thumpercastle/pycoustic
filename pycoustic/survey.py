import requests
import pandas as pd
import numpy as np
import datetime as dt
import pgeocode as geo
# from .weather import WeatherHistory


DECIMALS=1

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#survey.leq_spectra() bug
#TODO: C:\Users\tonyr\PycharmProjects\pycoustic\.venv1\Lib\site-packages\pycoustic\survey.py:287: FutureWarning: The behavior of pd.concat with len(keys) != len(objs) is deprecated. In a future version this will raise instead of truncating to the smaller of the two sequences combi = pd.concat(all_pos, axis=1, keys=["UA1", "UA2"])
#TODO: Survey should make a deep copy of Log objects. Otherwise setting time periods messes it up for other instances.

class Survey:
    """
    Survey Class is an overarching class which takes multiple Log objects and processes and summarises them together.
    This should be the main interface for user interaction with their survey data.
    """

    # ###########################---PRIVATE---######################################

    def __init__(self):
        self._logs = {}
        self._weather = WeatherHistory()

    def _insert_multiindex(self, df=None, super=None, name1="Position", name2="Date"):
        subs = df.index.to_list()   # List of subheaders (dates)
        # Super should be the position name (key from master dictionary)
        tuples = [(super, sub) for sub in subs]
        idx = pd.MultiIndex.from_tuples(tuples, names=[name1, name2])
        return df.set_index(idx, inplace=False)

    def _insert_header(self, df=None, new_head_list=None, header_idx=None):
        if df is None or new_head_list is None:
            return df

        ncols = df.shape[1]

        # Allow scalar broadcast for convenience
        if isinstance(new_head_list, (str, int, float)):
            new_head_list = [new_head_list] * ncols

        if len(new_head_list) != ncols:
            raise ValueError(
                f"new_head_list length ({len(new_head_list)}) must equal number of columns ({ncols})."
            )

        cols = list(df.columns)
        # Normalize columns into list of tuples
        if isinstance(df.columns, pd.MultiIndex):
            tuples = cols  # already list of tuples
        else:
            tuples = [(c,) for c in cols]

        # Build arrays per level and insert new header list
        arrays = [list(x) for x in zip(*tuples)]
        arrays.insert(header_idx, list(new_head_list))
        df.columns = pd.MultiIndex.from_arrays(arrays)
        return df

    # ###########################---PUBLIC---######################################

    def set_periods(self, times=None):
        """
        Set the daytime, evening and night-time periods of all Log objects in the Survey.
        To disable evening periods, simply set it the same as night-time.
        :param times: A dictionary with strings as keys and integer tuples as values.
        The first value in the tuple represents the hour of the day that period starts at (24hr clock), and the
        second value represents the minutes past the hour.
        e.g. for daytime from 07:00 to 19:00, evening 19:00 to 23:00 and night-time 23:00 to 07:00,
        times = {"day": (7, 0), "evening": (19, 0), "night": (23, 0)}
        NOTES:
        Night-time must cross over midnight. (TBC experimentally).
        Evening must be between daytime and night-time. To
        :return: None.
        """
        if times is None:
            times = {"day": (7, 0), "evening": (23, 0), "night": (23, 0)}
        for key in self._logs.keys():
            self._logs[key].set_periods(times=times)

    def add_log(self, data=None, name=""):
        """
        Add a Log object to the Survey object.
        :param data: Initialised Log object
        :param name: Name of the position, e.g. "A1"
        :return: None.
        """
        self._logs[name] = data

    def get_periods(self):
        """
        Check the currently-set daytime, evening and night-time periods for each Log object in the Survey.
        :return: Tuples of start times.
        """
        periods = {}
        for key in self._logs.keys():
            periods[key] = self._logs[key].get_period_times()
        return periods

    def broadband_summary(self, leq_cols=None, max_cols=None, lmax_n=10, lmax_t="2min"):
        """
        Get a dataframe summarising the parameters relevant to assessment of internal ambient noise levels in
        UK residential property assessments. Daytime and night-time Leqs, and nth-highest Lmax values all presented
        in a succinct table. These will be summarised as per the daytime, evening and night-time periods set (default
        daytime 07:00 to 23:00 and night-time 23:00 to 07:00).
        The date of the Lmax values are presented for the night-time period beginning on that date. i.e. an Lmax
        on 20/12/2024 would have occurred in the night-time period starting on that date and ending the following
        morning.
        :param leq_cols: List of tuples. The columns on which to perform Leq calculations. This can include L90
        columns, or spectral values. e.g.  leq_cols = [("Leq", "A"), ("L90", "125")]
        :param max_cols: List of tuples. The columns on which to get the nth-highest values.
        Default max_cols = [("Lmax", "A")]
        :param lmax_n: Int. The nth-highest value for max_cols. Default 10 for 10th-highest.
        :param lmax_t: String. This is the time period over which to compute nth-highest Lmax values.
        e.g. "2min" computes the nth-highest Lmaxes over 2-minute periods. Note that the chosen period must be
        equal to or more than the measurement period. So you cannot measure in 5-minute periods and request 2-minute
        Lmaxes.
        :return: A dataframe presenting a summary of the Leq and Lmax values requested.
        """
        combi = pd.DataFrame()
        if leq_cols is None:
            leq_cols = [("Leq", "A")]
        if max_cols is None:
            max_cols = [("Lmax", "A")]

        for key, lg in self._logs.items():
            period_blocks = []
            period_names = []

            # Day
            days = lg.leq_by_date(lg.get_period(data=lg.get_antilogs(), period="days"), cols=leq_cols)
            days.sort_index(inplace=True)
            period_blocks.append(days)
            period_names.append("Daytime")

            # Evening (optional)
            if lg.is_evening():
                evenings = lg.leq_by_date(lg.get_period(data=lg.get_antilogs(), period="evenings"), cols=leq_cols)
                evenings.sort_index(inplace=True)
                period_blocks.append(evenings)
                period_names.append("Evening")

            # Night Leq
            nights = lg.leq_by_date(lg.get_period(data=lg.get_antilogs(), period="nights"), cols=leq_cols)
            nights.sort_index(inplace=True)
            period_blocks.append(nights)
            period_names.append("Night-time")

            # Night max
            maxes = lg.as_interval(t=lmax_t)
            maxes = lg.get_period(data=maxes, period="nights", night_idx=True)
            maxes = lg.get_nth_high_low(n=lmax_n, data=maxes)[max_cols]
            maxes.sort_index(inplace=True)
            try:
                maxes.index = pd.to_datetime(maxes.index)
                maxes.index = maxes.index.date
            except Exception as e:
                print(f"Error converting index to date: {e}")
            maxes.index.name = None
            period_blocks.append(maxes)
            period_names.append("Night-time")

            # Build Period as a proper column level to avoid manual header length mismatches
            summary = pd.concat(objs=period_blocks, axis=1, keys=period_names, names=["Period"])

            # Add the per-log super level (kept as before)
            summary = self._insert_multiindex(df=summary, super=key)

            # Stack logs by rows; columns remain aligned across logs
            combi = pd.concat(objs=[combi, summary], axis=0)

        # No manual header insertion needed anymore; Period is already a column level
        return combi.round(decimals=DECIMALS)
#test
    def modal(self, cols=None, by_date=False, day_t="60min", evening_t="60min", night_t="15min"):
        """
        Get a dataframe summarising Modal L90 values for each time period, as suggested by BS 4142:2014.
        Currently, this method rounds the values to 0 decimal places by default and there is no alternative
        implementation.
        Note that this function will estimate L90s as a longer value by performing an Leq computation on them.
        The measured data in Logs must be smaller than or equal to the desired period, i.e. you can't measure in 15-
        minute periods and request 5-minute modal values.
        :param cols: List of tuples of the columns desired. This does not have to be L90s, but can be any column.
        :param by_date: Bool. If True, group the modal values by date. If False, present one modal value for each
        period.
        :param day_t:  String. Measurement period T. i.e. daytime measurements will compute modal values of
        L90,60min by default.
        :param evening_t: String. Measurement period T. i.e. evening measurements will compute modal values of
        L90,60min by default, unless evenings are disabled (which they are by default).
        :param night_t: Measurement period T. i.e. night-time measurements will compute modal values of
        L90,15min by default.
        :return: A dataframe of modal values for each time period.
        """
        if cols is None:
            cols = [("L90", "A")]
        combi = pd.DataFrame()
        period_headers = []
        for key in self._logs.keys():
            # Key is the name of the measurement position
            log = self._logs[key]
            pos_summary = []
            # Daytime
            period_headers = ["Daytime"]
            days = log.get_modal(data=log.get_period(data=log.as_interval(t=day_t), period="days"), by_date=by_date, cols=cols)
            days.sort_index(inplace=True)
            pos_summary.append(days)
            # Evening
            if log.is_evening():
                period_headers.append("Evening")
                evenings = log.get_modal(data=log.get_period(data=log.as_interval(t=evening_t), period="evenings"), by_date=by_date, cols=cols)
                evenings.sort_index(inplace=True)
                pos_summary.append(evenings)
            # Night time
            nights = log.get_modal(data=log.get_period(data=log.as_interval(t=night_t), period="nights"), by_date=by_date, cols=cols)
            nights.sort_index(inplace=True)
            pos_summary.append(nights)
            period_headers.append("Night-time")
            pos_df = pd.concat(pos_summary, axis=1)
            pos_df = self._insert_multiindex(pos_df, super=key)
            combi = pd.concat([combi, pos_df], axis=0)
        combi = self._insert_header(df=combi, new_head_list=period_headers, header_idx=0)
        combi.rename_axis(index={'Date': '#'}, inplace=True)
        return combi

    def counts(self, cols=None, day_t="60min", evening_t="60min", night_t="15min"):
        """
        Returns counts for each time period. For example, this can return the number of L90 occurrences at each decibel
        level for daytime and night-time periods.
        :param cols: Which columns to consider. Default (L90, A).
        :param day_t: Daytime averaging period. Default 60min.
        :param evening_t: Evening average period. Default 60min.
        :param night_t: Night-time average period. Default 15min.
        :return: Returns a dataframe of counts for each time period.
        """
        if cols is None:
            cols = [("L90", "A")]
        combi = pd.DataFrame()
        period_headers = []
        for key in self._logs.keys():
            # Key is the name of the measurement position
            log = self._logs[key]
            pos_summary = []
            # Daytime
            # period_headers = ["Daytime"]
            days = log.counts(data=log.get_period(data=log.as_interval(t=day_t), period="days"), cols=cols)
            days.sort_index(inplace=True)
            days.name = "Daytime"
            pos_summary.append(days)
            # Evening
            if log.is_evening():
                # period_headers.append("Evening")
                evenings = log.counts(data=log.get_period(data=log.as_interval(t=evening_t), period="evenings"), cols=cols)
                evenings.sort_index(inplace=True)
                evenings.name = "Evening"
                pos_summary.append(evenings)
            # Night time
            nights = log.counts(data=log.get_period(data=log.as_interval(t=night_t), period="nights"), cols=cols)
            nights.sort_index(inplace=True)
            pos_summary.append(nights)
            nights.name = "Night-time"
            # period_headers.append("Night-time")
            pos_df = pd.concat(pos_summary, axis=1)
            # ensure integer dtype for every period block
            pos_df = pos_df.fillna(0).astype("int64")
            pos_df = self._insert_multiindex(pos_df, super=key)
            combi = pd.concat([combi, pos_df], axis=0)
        # combi = self._insert_header(df=combi, new_head_list=period_headers, header_idx=0)
        combi.sort_index(inplace=True)
        combi.rename_axis(index={'Date': 'dB'}, inplace=True)
        return combi.fillna(0).astype("int64")


    def lmax_spectra(self, n=10, t="2min", period="nights"):
        """
        Get spectral data for the nth-highest Lmax values during a given time period.
        This computes Lmax Event spectra. Lmax Hold spectra has not yet been implemented.
        Assumptions and inputs as per Survey.resi_summary() method.
        IMPORTANT: The dates of the Lmax values are presented for the night-time period beginning on that date.
        This means that for early morning timings, the date is behind by one day.
        e.g. an Lmax presented as occurring at 20/12/2024 at 01:22 would have occurred at 21/12/2024 at 01:22.
        :param n: Int. Nth-highest Lmax. Default 10th-highest.
        :param t: String. This is the time period over which to compute nth-highest Lmax values.
        e.g. "2min" computes the nth-highest Lmaxes over 2-minute periods. Note that the chosen period must be
        equal to or more than the measurement period. So you cannot measure in 5-minute periods and request 2-minute
        Lmaxes.
        :param period: String. "days", "evenings" or "nights"
        :return: Dataframe of nth-highest Lmax Event spectra.
        """
        combi = pd.DataFrame()
        # TODO: The night-time timestamp on this is sometimes out by a minute.
        for key in self._logs.keys():
            log = self._logs[key]
            combined_list = []
            maxes = log.get_nth_high_low(n=n, data=log.get_period(data=log.as_interval(t=t), period=period))[["Lmax", "Time"]]
            maxes.sort_index(inplace=True)
            combined_list.append(maxes)
            summary = pd.concat(objs=combined_list, axis=1)
            summary = self._insert_multiindex(df=summary, super=key)
            combi = pd.concat(objs=[combi, summary], axis=0)
        return combi.round(decimals=DECIMALS)

    def leq_spectra(self, leq_cols=None):
        """
        Compute Leqs over daytime, evening and night-time periods.
        This is an overall Leq, and does not group Leqs by date.
        :param leq_cols: List of strings or List of Tuples.
        For all Leq columns, use ["Leq"]. For specific columns, use list of tuples [("Leq", "A"), ("Leq", 125)]
        :return: A dataframe with a continuous Leq computation across dates, for each time period.
        """
        all_pos = []
        labels = []

        if leq_cols is None:
            leq_cols = ["Leq"]

        for label, log in self._logs.items():
            # Day
            days = log.get_period(data=log.get_antilogs(), period="days")
            days = days[leq_cols].apply(lambda x: np.round(10 * np.log10(np.mean(x)), DECIMALS))

            # Night-time
            nights = log.get_period(data=log.get_antilogs(), period="nights")
            nights = nights[leq_cols].apply(lambda x: np.round(10 * np.log10(np.mean(x)), DECIMALS))

            # Evening (if applicable)
            if log.is_evening():
                evenings = log.get_period(data=log.get_antilogs(), period="evenings")
                evenings = evenings[leq_cols].apply(lambda x: np.round(10 * np.log10(np.mean(x)), DECIMALS))
                df = pd.concat([days, evenings, nights], axis=1, keys=["Daytime", "Evening", "Night-time"])
            else:
                df = pd.concat([days, nights], axis=1, keys=["Daytime", "Night-time"])

            all_pos.append(df)
            labels.append(label)

        if not all_pos:
            return pd.DataFrame()

        # Concatenate across logs; keys match number of objects (no FutureWarning)
        combi = pd.concat(all_pos, axis=1, keys=labels)
        return combi.transpose().round(decimals=DECIMALS)

    def get_start_end(self):
        starts = [self._logs[key].get_start() for key in self._logs.keys()]
        ends = [self._logs[key].get_end() for key in self._logs.keys()]
        return min(starts), max(ends)

    def weather_config(self, **kwargs):
        """
        Configure the embedded WeatherHistory using the Survey's overall start/end
        (earliest log start, latest log end). Additional WeatherHistory kwargs may
        be provided by the caller (e.g. interval_hours, api_key, country, postcode, units, recompute).
        """
        if not self._logs:
            raise ValueError("No logs in Survey. Add Log objects before configuring weather.")
        start, end = self.get_start_end()
        return self._weather.configure(start=start, end=end, **kwargs)

    def weather_compute(self, **kwargs):
        """
        Compute weather history via the embedded WeatherHistory for the Survey's overall
        start/end (earliest log start, latest log end). Caller may pass through any
        WeatherHistory.compute kwargs (e.g. drop_cols, recompute, timeout_s).

        If WeatherHistory is not yet configured, this will configure it first using the
        Survey's start/end plus any kwargs intended for configure(...).
        """
        if not self._logs:
            raise ValueError("No logs in Survey. Add Log objects before computing weather.")
        if self._weather is None:
            raise ValueError("WeatherHistory object is missing from this Survey instance.")

        # If not configured yet, configure first. Split kwargs between configure/compute by name.
        if getattr(self._weather, "_config", None) is None:
            cfg_keys = {"interval_hours", "api_key", "country", "postcode", "units", "recompute"}
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_keys}
            self.weather_config(**cfg_kwargs)

        compute_keys = {"drop_cols", "recompute", "timeout_s"}
        compute_kwargs = {k: v for k, v in kwargs.items() if k in compute_keys}
        return self._weather.compute(**compute_kwargs)

    def get_weather_hist(self):
        """Return cached computed weather dataframe (or None if not computed yet)."""
        return self._weather.get_weather_history()

    def get_weather_raw(self):
        """Return cached raw API responses (or None if not computed yet)."""
        return self._weather.get_raw_responses()


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class WeatherConfig:
    start: dt.datetime
    end: dt.datetime
    interval_hours: int
    api_key: str
    country: str
    postcode: str
    units: str = "metric"


class WeatherHistory:
    """
    Fetches historic weather snapshots from OpenWeather One Call 3.0 Time Machine endpoint
    at a fixed interval between start and end (inclusive-ish; end is an upper bound).

    Design:
    - Configure once with (start, end, interval, api_key, country, postcode, units)
    - Compute once and cache the resulting dataframe
    - Subsequent compute() calls return cached data unless recompute=True
    """

    ONECALL_TIMEMACHINE_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    DEFAULT_DROP_COLS = ("sunrise", "sunset", "feels_like", "dew_point", "visibility")

    def __init__(self) -> None:
        self._config: Optional[WeatherConfig] = None
        self._latlon: Optional[Tuple[float, float]] = None
        self._data: Optional[pd.DataFrame] = None
        self._raw_responses: Optional[List[Dict[str, Any]]] = None

    # ----------------------------
    # Configuration / caching
    # ----------------------------
    def configure(
        self,
        *,
        start: Union[str, dt.datetime],
        end: Union[str, dt.datetime],
        interval_hours: int = 12,
        api_key: Optional[str] = None,
        country: str = "GB",
        postcode: str = "WC1",
        units: str = "metric",
        recompute: bool = False,
    ) -> "WeatherHistory":
        """
        Store config. If config changes, cached data is invalidated automatically.
        If recompute=True, cached data is cleared even if config is unchanged.
        """
        if not api_key:
            raise ValueError("api_key is required")

        start_dt = self._coerce_datetime(start)
        end_dt = self._coerce_datetime(end)

        if end_dt <= start_dt:
            raise ValueError("end must be after start")

        if interval_hours <= 0:
            raise ValueError("interval_hours must be a positive integer")

        new_cfg = WeatherConfig(
            start=start_dt,
            end=end_dt,
            interval_hours=int(interval_hours),
            api_key=str(api_key),
            country=str(country),
            postcode=str(postcode),
            units=str(units),
        )

        if recompute or (self._config != new_cfg):
            self._config = new_cfg
            self._latlon = None
            self._data = None
            self._raw_responses = None
        else:
            self._config = new_cfg

        return self

    def compute(
        self,
        *,
        drop_cols: Optional[List[str]] = None,
        recompute: bool = False,
        timeout_s: int = 30,
    ) -> pd.DataFrame:
        """
        Compute and return a dataframe of weather snapshots.
        Uses cached value unless recompute=True.
        """
        if self._config is None:
            raise ValueError("WeatherHistory is not configured. Call configure(...) first.")

        if self._data is not None and not recompute:
            return self._data

        lat, lon = self._get_latlon(country=self._config.country, postcode=self._config.postcode)

        timestamps = self._construct_timestamps(
            start=self._config.start,
            end=self._config.end,
            interval_hours=self._config.interval_hours,
        )

        rows: List[pd.Series] = []
        raw: List[Dict[str, Any]] = []

        for ts in timestamps:
            payload = self._call_openweather(
                lat=lat,
                lon=lon,
                timestamp=ts,
                api_key=self._config.api_key,
                units=self._config.units,
                timeout_s=timeout_s,
            )
            raw.append(payload)

            row = self._parse_timemachine_payload(payload)
            rows.append(pd.Series(row))

        df = pd.concat(rows, axis=1).T if rows else pd.DataFrame()

        # Convert unix seconds to datetimes where present
        for col in ("dt", "sunrise", "sunset"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: dt.datetime.fromtimestamp(int(x)) if pd.notna(x) else x
                )

        cols_to_drop = list(self.DEFAULT_DROP_COLS) if drop_cols is None else list(drop_cols)
        existing = [c for c in cols_to_drop if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

        self._data = df
        self._raw_responses = raw
        return self._data

    # ----------------------------
    # Accessors
    # ----------------------------
    def get_weather_history(self) -> Optional[pd.DataFrame]:
        """Return cached dataframe (or None if not computed yet)."""
        return self._data

    def get_raw_responses(self) -> Optional[List[Dict[str, Any]]]:
        """Return cached raw API payloads (or None if not computed yet)."""
        return self._raw_responses

    # ----------------------------
    # Internals
    # ----------------------------
    @staticmethod
    def _coerce_datetime(value: Union[str, dt.datetime]) -> dt.datetime:
        if isinstance(value, dt.datetime):
            return value
        if isinstance(value, str):
            # Accept the existing convention used elsewhere in your project
            return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        raise TypeError("start/end must be datetime or 'YYYY-mm-dd HH:MM:SS' string")

    def _get_latlon(self, *, country: str, postcode: str) -> Tuple[float, float]:
        if self._latlon is not None:
            return self._latlon

        nomi = geo.Nominatim(country)
        rec = nomi.query_postal_code(postcode)

        lat = getattr(rec, "latitude", None)
        lon = getattr(rec, "longitude", None)

        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            raise ValueError(f"Could not resolve postcode to lat/lon: {country=} {postcode=}")

        self._latlon = (float(lat), float(lon))
        return self._latlon

    @staticmethod
    def _construct_timestamps(*, start: dt.datetime, end: dt.datetime, interval_hours: int) -> List[int]:
        """
        Build timestamps starting at 'start', stepping by interval_hours, strictly less than end.
        """
        timestamps: List[int] = [int(start.timestamp())]
        next_time = start + dt.timedelta(hours=interval_hours)
        while next_time < end:
            timestamps.append(int(next_time.timestamp()))
            next_time += dt.timedelta(hours=interval_hours)
        return timestamps

    def _call_openweather(
        self,
        *,
        lat: float,
        lon: float,
        timestamp: int,
        api_key: str,
        units: str,
        timeout_s: int,
    ) -> Dict[str, Any]:
        params = {
            "lat": lat,
            "lon": lon,
            "dt": int(timestamp),
            "appid": api_key,
            "units": units,
        }

        resp = requests.get(self.ONECALL_TIMEMACHINE_URL, params=params, timeout=timeout_s)
        # Raise for 4xx/5xx with a helpful message
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            msg = f"OpenWeather request failed: status={resp.status_code} body={resp.text[:500]}"
            raise requests.HTTPError(msg) from e

        return resp.json()

    @staticmethod
    def _parse_timemachine_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenWeather Time Machine returns a payload with a 'data' array.
        We take the first element as the snapshot.

        Keeps 'weather' out by default; flattens some common nested fields.
        """
        data = payload.get("data", [])
        if not data:
            raise ValueError("OpenWeather payload missing 'data' items")

        snap = dict(data[0])  # shallow copy

        # Flatten rain/snow dicts (e.g. {"1h": 0.25}) into rain_1h
        for k in ("rain", "snow"):
            if k in snap and isinstance(snap[k], dict):
                for subk, val in snap[k].items():
                    snap[f"{k}_{subk}"] = val
                del snap[k]

        # Optionally flatten weather[0].main/description/icon into columns
        wx = snap.get("weather")
        if isinstance(wx, list) and wx:
            w0 = wx[0] if isinstance(wx[0], dict) else None
            if w0:
                for key in ("main", "description", "icon"):
                    if key in w0:
                        snap[f"weather_{key}"] = w0[key]
        if "weather" in snap:
            del snap["weather"]

        return snap
