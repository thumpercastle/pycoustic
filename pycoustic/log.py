import pandas as pd
import numpy as np
import datetime as dt

DECIMALS=0
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
   -45.8
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
    0.005
]

class Log:
    def __init__(self, path=""):
        #TODO C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pycoustic\log.py:15: UserWarning:
        #Parsing dates in %Y/%m/%d %H:%M format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.
        """
        The Log class is used to store the measured noise data from one data logger.
        The data must be entered in a .csv file with headings in the specific format "Leq A", "L90 125" etc.
        :param path: the file path for the .csv noise data
        """
        self._filepath = path
        self._master = pd.read_csv(
            path,
            index_col="Time",
            parse_dates=["Time"],
            # date_format="%d/%m/%Y %H:%M",  # Explicit format to avoid the dayfirst warning
            dayfirst=True,  # Optional: include for clarity; default is False
        )
        self._master.index = pd.to_datetime(self._master.index)
        self._master = self._master.sort_index(axis=1)
        self._start = self._master.index.min()
        self._end = self._master.index.max()
        self._assign_header()

        # Assign day, evening, night periods
        self._night_start = None
        self._day_start = None
        self._evening_start = None
        self._init_periods()

        # Prepare night-time indices and antilogs
        self._antilogs = self._prep_antilogs()  # Use the antilogs dataframe as input to Leq calculations
        self._master = self._append_night_idx(data=self._master)
        self._antilogs = self._append_night_idx(data=self._antilogs)

        self._decimals = 1

    def _assign_header(self):
        csv_headers = self._master.columns.to_list()
        superheaders = [item.split(" ")[0] for item in csv_headers]
        subheaders = [item.split(" ")[1] for item in csv_headers]
        # Convert numerical subheaders to ints
        for i in range(len(subheaders)):
            try:
                subheaders[i] = float(subheaders[i])
            except Exception:
                continue
        self._master.columns = [superheaders, subheaders]
        self._master.sort_index(axis=1, level=1, inplace=True)

    def _init_periods(self):
        times = {"day": (7, 0), "evening": (23, 0), "night": (23, 0)}
        self._day_start = dt.time(times["day"][0], times["day"][1])
        self._evening_start = dt.time(times["evening"][0], times["evening"][1])
        self._night_start = dt.time(times["night"][0], times["night"][1])


    def _prep_antilogs(self):
        """
        Private method creates a copy dataframe of master, but with dB sound pressure levels presented as antilogs.
        This antilogs dataframe should be used if you want to undertake calculations of Leqs and similar.
        :return:
        """
        return self._master.copy().apply(lambda x: np.power(10, (x / 10)))

    def _append_night_idx(self, data=None):
        """
        Private method appends an additional column of the measurement date and time, but with the early morning
        dates set to the day before.
        e.g.
        the measurement at 16-12-2024 23:57 would stay as is, but
        the measurement at 17-12-2024 00:02 would have a night index of 16-12-2024 00:02
        The logic behind this is that it allows us to process a night-time as one contiguous period, whereas
        Pandas would otherwise treat the two measurements as separate because of their differing dates.
        :param data:
        :return:
        """
        night_indices = data.index.to_list()
        if self._night_start > self._day_start:
            for i in range(len(night_indices)):
                if night_indices[i].time() < self._day_start:
                    night_indices[i] += dt.timedelta(days=-1)
        data["Night idx"] = night_indices
        return data

    def _return_as_night_idx(self, data=None):
        """
        Private method to set the dataframe index as the night_idx. This is used when undertaking data processing for
        night-time periods.
        :param data:
        :return:
        """
        if ("Night idx", "") not in data.columns:
            raise Exception("No night indices in current DataFrame")
        return data.set_index("Night idx")

    def _none_if_zero(self, df):
        if len(df) == 0:
            return None
        else:
            return df

    def _recompute_leq(self, data=None, t="15min", cols=None):
        """
        Private method to recompute shorter Leq measurements as longer ones.
        :param data: Input data (should be in antilog format)
        :param t: The desired Leq period
        :param cols: Which columns of the input data do you wish to recompute?
        :return:
        """
        # Set default mutable args
        if data is None:
            data = self._antilogs
        if cols is None:
            cols = ["Leq", "L90"]
        # Loop through column superheaders and recompute as a longer Leq
        recomputed = pd.DataFrame(columns=data.columns)
        for idx in cols:
            if idx in data.columns:
                recomputed[idx] = data[idx].resample(t).mean().\
                    apply(lambda x: np.round((10 * np.log10(x)), self._decimals))
        return self._none_if_zero(recomputed)

    def _recompute_night_idx(self, data=None, t="15min"):
        """
        Internal method to recompute night index column.
        :param data: input dataframe to be recomputed
        :param t: desired measurement period
        :return: dataframe with night index column recomputed to the desired period
        """
        if data is None:
            raise Exception("No DataFrame provided for night idx")
        if ("Night idx", "") in data.columns:
            data["Night idx"] = data["Night idx"].resample(t).asfreq()
        else:
            data["Night idx"] = self._master["Night idx"].resample(t).asfreq()
            return data

    def _recompute_max(self, data=None, t="15min", pivot_cols=None, hold_spectrum=False):
        """
        Private method to recompute max readings from shorter to longer periods.
        :param data: input data, usually self._master
        :param t: desired measurement period
        :param pivot_cols: how to choose the highest value - this will usually be "Lmax A". This is especially
        important when you want to get specific octave band data for an Lmax event. If you wanted to recompute maxes
        as the events with the highest values at 500 Hz, you could enter [("Lmax", 500)]. Caution: This functionality
        has not been tested
        :param hold_spectrum: if hold_spectrum, the dataframe returned will contain the highest value at each octave
        band over the new measurement period, i.e. like the Lmax Hold setting on a sound level meter.
        If hold_spectrum=false, the dataframe will contain the spectrum for the highest event around the pivot column,
        i.e. the spectrum for that specific LAmax event
        :return: returns a dataframe with the values recomputed to the desired measurement period.
        """
        # Set default mutable args
        if pivot_cols is None:
            pivot_cols = [("Lmax", "A")]
        if data is None:
            data = self._master
        # Loop through column superheaders and recompute over a longer period
        combined = pd.DataFrame(columns=data.columns)
        if hold_spectrum:   # Hold the highest value, in given period per frequency band
            for col in pivot_cols:
                if col in combined.columns:
                    max_hold = data.resample(t)[col[0]].max()
                    combined[col[0]] = max_hold
        else:   # Event spectrum (octave band data corresponding to the highest A-weighted event)
            for col in pivot_cols:
                if col in combined.columns:
                    idx = data[col[0]].groupby(pd.Grouper(freq=t)).max()
                    combined[col[0]] = idx
        return combined

    def _as_multiindex(self, df=None, super=None, name1="Date", name2="Num"):
        subs = df.index.to_list()   # List of subheaders
        # Super will likely be the date
        tuples = [(super, sub) for sub in subs]
        idx = pd.MultiIndex.from_tuples(tuples, names=[name1, name2])
        if isinstance(df, pd.Series):
            df = pd.DataFrame(data=df)
        return df.set_index(idx, inplace=False)
#test
    def get_period(self, data=None, period="days", night_idx=True):
        """
        Private method to get data for daytime, evening or night-time periods.
        :param data: Input data, usually master
        :param period: string, "days", "evenings" or "nights"
        :param night_idx: Bool. Needs to be True if you want to compute contiguous night-time periods. If False,
        it will consider early morning measurements as part of the following day, i.e. the cut-off becomes midnight.
        :return:
        """
        if data is None:
            data = self._master
        if period == "days":
            return data.between_time(self._day_start, self._evening_start, inclusive="left")
        elif period == "evenings":
            return data.between_time(self._evening_start, self._night_start, inclusive="left")
        elif period == "nights":
            if night_idx:
                data = self._return_as_night_idx(data=data)
            return data.between_time(self._night_start, self._day_start, inclusive="left")

    def leq_by_date(self, data, cols=None):
        """
        Private method to undertake Leq calculations organised by date. For contiguous night-time periods crossing
        over midnight (e.g. from 23:00 to 07:00), the input data needs to have a night-time index.
        This method is normally used for calculating Leq over a specific daytime, evening or night-time period, hence
        it is usually passed the output of _get_period()
        :param data: Input data. Must be antilogs, and usually with night-time index
        :param cols: Which columns do you wish to recalculate? If ["Leq"] it will calculate for all subcolumns within
        that heading, i.e. all frequency bands and A-weighted. If you want an individual column, use [("Leq", "A")] for
        example.
        :return: A dataframe of the calculated Leq for the data, organised by dates
        """
        if cols is None:
            cols = ["Leq"]
        return data[cols].groupby(data.index.date).mean().apply(lambda x: np.round((10 * np.log10(x)), self._decimals))

    # ###########################---PUBLIC---######################################
    # ss++
    def get_data(self):
        """
        # Returns a dataframe of the loaded csv
        """
        return self._master
    #ss--

    def get_antilogs(self):
        return self._antilogs


    def as_interval(self, data=None, antilogs=None, t="15min", leq_cols=None, max_pivots=None,
                    hold_spectrum=False):
        """
        Returns a dataframe recomputed as longer periods. This implements the private leq and max recalculations
        :param data: input dataframe, usually master
        :param antilogs: antilog dataframe, used for leq calcs
        :param t: desired output period
        :param leq_cols: which Leq columns to include
        :param max_pivots: which value to pivot the Lmax recalculation on
        :param hold_spectrum: True will be Lmax hold, False will be Lmax event
        :return: a dataframe recalculated to the desired period, with the desired columns
        """
        # Set defaults for mutable args
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
        conc = self._append_night_idx(data=conc)    # Re-append night indices
        return conc.dropna(axis=0, how="all")

    def get_nth_high_low(self, n=10, data=None, pivot_col=None, all_cols=False, high=True):
        """
        Return a dataframe with the nth-highest or nth-lowest values for the specified parameters.
        This is useful for calculating the 10th-highest or 15th-highest Lmax values, but can be used for other purposes
        :param n: The nth-highest or nth-lowest values to return
        :param data: Input dataframe, usually a night-time dataframe with night-time indices
        :param pivot_col: Tuple of strings,
        Which column to use for the highest-lowest computation. Other columns in the row will follow.
        :param all_cols: Perform this operation over all columns?
        :param high: True for high, False for low
        :return: dataframe with the nth-highest or -lowest values for the specified parameters.
        """
        if data is None:
            data = self._master
        if pivot_col is None:
            pivot_col = ("Lmax", "A")
        nth = None
        if high:
            nth = data.sort_values(by=pivot_col, ascending=False)
        if not high:
            nth = data.sort_values(by=pivot_col, ascending=True)
        nth["Time"] = nth.index.time
        if all_cols:
            return nth.groupby(by=nth.index.date).nth(n-1)
        else:
            return nth[[pivot_col[0], "Time"]].groupby(by=nth.index.date).nth(n-1)

    def get_modal(self, data=None, by_date=True, cols=None, round_decimals=True):
        """
        Return a dataframe with the modal values
        :param data: Input dataframe, usually master
        :param by_date: Bool. Group the modal values by date, as opposed to an overall modal value (currently not
        implemented).
        :param cols: List of tuples of the desired columns. e.g. [("L90", "A"), ("Leq", "A")]
        :param round_decimals: Bool. Round the values to 0 decimal places.
        :return: A dataframe with the modal values for the desired columns, either grouped by date or overall.
        """
        if data is None:
            data = self._master
        if round_decimals:
            data = data.round()
        if cols is None:
            cols = [("L90", "A")]
        if by_date:
            dates = np.unique(data.index.date)
            modes_by_date = pd.DataFrame()
            for date in range(len(dates)):
                date_str = dates[date].strftime("%Y-%m-%d")
                mode_by_date = data[cols].loc[date_str].mode()
                mode_by_date = self._as_multiindex(df=mode_by_date, super=date_str)
                modes_by_date = pd.concat([modes_by_date, mode_by_date])
            return modes_by_date
        else:
            return data[cols].mode()

    def counts(self, data=None, cols=None, round_decimals=True):
        # This does not work with multiple columns
        if data is None:
            data = self._master
        if round_decimals:
            data = data.round(decimals=0)
        if cols is None:
            cols = [("L90", "A")]
        df = data[cols].value_counts()
        df.index = [int(x[0]) for x in df.index]
        return df

    def set_periods(self, times=None):
        """
        Set the daytime, night-time and evening periods. To disable evening periods, simply set it the same
        as night-time.
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
        self._day_start = dt.time(times["day"][0], times["day"][1])
        self._evening_start = dt.time(times["evening"][0], times["evening"][1])
        self._night_start = dt.time(times["night"][0], times["night"][1])
        # Recompute night indices
        self._master.drop(labels="Night idx", axis=1, inplace=True)
        self._antilogs.drop(labels="Night idx", axis=1, inplace=True)
        self._master = self._append_night_idx(data=self._master)
        self._antilogs = self._append_night_idx(data=self._antilogs)

#C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pycoustic\log.py:339: PerformanceWarning:
#dropping on a non-lexsorted multi-index without a level parameter may impact performance.

    def get_period_times(self):
        """
        :return: the tuples of period start times.
        """
        return self._day_start, self._evening_start, self._night_start

    def is_evening(self):
        """
        Check if evening periods are enabled.
        :return: True if evening periods are enabled, False otherwise.
        """
        if self._evening_start == self._night_start:
            return False
        else:
            return True

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end


class VibLog(Log):
    """
    This class is a work in progress. Do not use.
    """
    def __init__(self, path, units=None):
        super().__init__(path)
        self._units = units
        if self._units is None:
            self._units = "ms2"
        self._decimals = 12
        self._wb_weighted = self.apply_wb_weighting()

    def head(self):
        return self._master.head()

    def apply_wb_weighting(self, factors=wb_weighting_factors):
        """
        Multiply each column in self._master by the corresponding WB weighting factor and
        store the result in self._wb_weighted.

        Notes:
        - Applies only to numeric columns (non-numeric, e.g. the "Night idx" column, are left unchanged).
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

    def sum_across_bands(self, df=None):
        """
        Sum values across band columns and return a DataFrame with:
        - a new column "Sum" containing one summed figure per row
        - the existing "Night idx" column preserved (not included in the sum)
        - the original index preserved
        """
        if df is None:
            df = self._master
        elif df == "wb_weighted":
            df = self._wb_weighted

        night_col = "Night idx"
        if night_col not in df.columns:
            raise ValueError(f"Expected column {night_col!r} in DataFrame.")

        # Sum only numeric columns, but explicitly exclude the Night idx column
        numeric_cols = df.select_dtypes(include="number").columns
        numeric_cols = [c for c in numeric_cols if c != night_col]

        out = df.copy()
        out["Sum"] = out.loc[:, numeric_cols].sum(axis=1)
        return out


    def evdv_from_rms(self, df=None, t=None, col=None):
        """
        Compute eVDV (estimated Vibration Dose Value) from a time history of RMS acceleration.

        This implementation first converts RMS to an estimated peak-equivalent value by:
            a_eq = a_rms * 1.4

        Then computes (discrete):
            eVDV = ( sum_i ( a_eq[i]^4 * t ) )^(1/4)

        Parameters
        ----------
        df : pandas.Series or pandas.DataFrame, optional
            If None, uses self._master.
            If Series, treated as the a_rms time history.
            If DataFrame, `col` must identify the column to use.
        t : float, required
            Assessment interval in seconds.
        col : column label, optional
            Column to use when `df` is a DataFrame (e.g. ("Leq","A") style labels if applicable).

        Returns
        -------
        float
            eVDV in the same acceleration units as the input (e.g. m/s^2).
        """
        if t is None:
            raise ValueError("Parameter 't' (measurement interval in seconds) is required.")
        if t <= 0:
            raise ValueError("Parameter 't' must be a positive number of seconds.")

        data = self._master if df is None else df

        # Extract a 1D series of a_rms values
        if isinstance(data, pd.Series):
            a_rms = data
        else:
            if col is None:
                raise ValueError("When 'df' is a DataFrame, you must provide 'col' to select the a_rms column.")
            a_rms = data[col]

        # Ensure numeric and drop missing
        a_rms = pd.to_numeric(a_rms, errors="coerce").dropna()
        if len(a_rms) == 0:
            raise ValueError("No valid numeric RMS acceleration values available to compute eVDV.")

        a_eq = a_rms * 1.4
        evdv_val = float(((a_eq.pow(4).sum() * float(t)) ** 0.25))
        return evdv_val
    #
    # def get_day_evdv(self):
    #     return self.evdv_from_rms(df=self.get_period(self.sum_across_bands(self.), period="days"), t=1, col="Sum")