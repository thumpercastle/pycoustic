import pandas as pd
import numpy as np
import datetime as dt



class Log:
    def __init__(self, path=""):
        """
        The Log class is used to store the measured noise data from one data logger.
        The data must be entered in a .csv file with headings in the specific format "Leq A", "L90 125" etc.
        :param path: the file path for the .csv noise data
        """
        self._filepath = path
        self._master = pd.read_csv(path, index_col="Time", parse_dates=["Time"], dayfirst=True)
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

    def _get_period(self, data=None, period="days", night_idx=True):
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

    def _leq_by_date(self, data, cols=None):
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
        if data is None:
            data = self._master
        if round_decimals:
            data = data.round()
        if cols is None:
            cols = [("L90", "A")]
        return data[cols].value_counts()

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