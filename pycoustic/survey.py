import pandas as pd
import numpy as np
from weather import WeatherHistory


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
        self._weatherhist = None

    def _insert_multiindex(self, df=None, super=None, name1="Position", name2="Date"):
        subs = df.index.to_list()   # List of subheaders (dates)
        # Super should be the position name (key from master dictionary)
        tuples = [(super, sub) for sub in subs]
        idx = pd.MultiIndex.from_tuples(tuples, names=[name1, name2])
        return df.set_index(idx, inplace=False)

    def _insert_header(self, df=None, new_head_list=None, header_idx=None):
        cols = df.columns.to_list()
        new_cols = [list(c) for c in zip(*cols)]
        new_cols.insert(header_idx, new_head_list)
        df.columns = new_cols
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

    def resi_summary(self, leq_cols=None, max_cols=None, lmax_n=10, lmax_t="2min"):
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
        period_headers = []
        if leq_cols is None:
            leq_cols = [("Leq", "A")]
        if max_cols is None:
            max_cols = [("Lmax", "A")]
        for key in self._logs.keys():
            log = self._logs[key]
            combined_list = []
            # Day
            days = log._leq_by_date(log._get_period(data=log.get_antilogs(), period="days"), cols=leq_cols)
            days.sort_index(inplace=True)
            combined_list.append(days)
            period_headers = ["Daytime" for i in range(len(leq_cols))]
            # Evening
            if log.is_evening():
                evenings = log._leq_by_date(log._get_period(data=log.get_antilogs(), period="evenings"), cols=leq_cols)
                evenings.sort_index(inplace=True)
                combined_list.append(evenings)
                for i in range(len(leq_cols)):
                    period_headers.append("Evening")
            # Night Leq
            nights = log._leq_by_date(log._get_period(data=log.get_antilogs(), period="nights"), cols=leq_cols)
            nights.sort_index(inplace=True)
            combined_list.append(nights)
            for i in range(len(leq_cols)):
                period_headers.append("Night-time")
            # Night max
            maxes = log.as_interval(t=lmax_t)
            maxes = log._get_period(data=maxes, period="nights", night_idx=True)
            maxes = log.get_nth_high_low(n=lmax_n, data=maxes)[max_cols]
            maxes.sort_index(inplace=True)
            #  +++
            # SS Feb2025  - Code changed to prevent exception
            #maxes.index = maxes.index.date
            try:
                maxes.index = pd.to_datetime(maxes.index)
                maxes.index = maxes.index.date
            except Exception as e:
                print(f"Error converting index to date: {e}")
            # SSS ---
            maxes.index.name = None
            combined_list.append(maxes)
            for i in range(len(max_cols)):
                period_headers.append("Night-time")
            summary = pd.concat(objs=combined_list, axis=1)
            summary = self._insert_multiindex(df=summary, super=key)
            combi = pd.concat(objs=[combi, summary], axis=0)
        combi = self._insert_header(df=combi, new_head_list=period_headers, header_idx=0)
        return combi

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
            days = log.get_modal(data=log._get_period(data=log.as_interval(t=day_t), period="days"), by_date=by_date, cols=cols)
            days.sort_index(inplace=True)
            pos_summary.append(days)
            # Evening
            if log.is_evening():
                period_headers.append("Evening")
                evenings = log.get_modal(data=log._get_period(data=log.as_interval(t=evening_t), period="evenings"), by_date=by_date, cols=cols)
                evenings.sort_index(inplace=True)
                pos_summary.append(evenings)
            # Night time
            nights = log.get_modal(data=log._get_period(data=log.as_interval(t=night_t), period="nights"), by_date=by_date, cols=cols)
            nights.sort_index(inplace=True)
            pos_summary.append(nights)
            period_headers.append("Night-time")
            pos_df = pd.concat(pos_summary, axis=1)
            pos_df = self._insert_multiindex(pos_df, super=key)
            combi = pd.concat([combi, pos_df], axis=0)
        combi = self._insert_header(df=combi, new_head_list=period_headers, header_idx=0)
        return combi

    def counts(self, cols=None, day_t="60min", evening_t="60min", night_t="15min"):
        #TODO Need to order rows and rename from 'date'
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
            period_headers = ["Daytime"]
            days = log.counts(data=log._get_period(data=log.as_interval(t=day_t), period="days"), cols=cols)
            days.sort_index(inplace=True)
            pos_summary.append(days)
            # Evening
            if log.is_evening():
                period_headers.append("Evening")
                evenings = log.counts(data=log._get_period(data=log.as_interval(t=evening_t), period="evenings"), cols=cols)
                evenings.sort_index(inplace=True)
                pos_summary.append(evenings)
            # Night time
            nights = log.counts(data=log._get_period(data=log.as_interval(t=night_t), period="nights"), cols=cols)
            nights.sort_index(inplace=True)
            pos_summary.append(nights)
            period_headers.append("Night-time")
            pos_df = pd.concat(pos_summary, axis=1)
            pos_df = self._insert_multiindex(pos_df, super=key)
            combi = pd.concat([combi, pos_df], axis=0)
        combi = self._insert_header(df=combi, new_head_list=period_headers, header_idx=0)
        #TODO: This dataframe needs tidying.
        return combi

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
            maxes = log.get_nth_high_low(n=n, data=log._get_period(data=log.as_interval(t=t), period=period))[["Lmax", "Time"]]
            maxes.sort_index(inplace=True)
            combined_list.append(maxes)
            summary = pd.concat(objs=combined_list, axis=1)
            summary = self._insert_multiindex(df=summary, super=key)
            combi = pd.concat(objs=[combi, summary], axis=0)
        return combi

    def leq_spectra(self, leq_cols=None):
        """
        Compute Leqs over daytime, evening and night-time periods.
        This is an overall Leq, and does not group Leqs by date.
        :param leq_cols: List of strings or List of Tuples.
        For all Leq columns, use ["Leq"]. For specific columns, use list of tuples [("Leq", "A"), ("Leq", 125)]
        :return: A dataframe with a continuous Leq computation across dates, for each time period.
        """
        #TODO: C:\Users\tonyr\PycharmProjects\pycoustic\tests.py:674: FutureWarning: The behavior of pd.concat with len(keys) != len(objs) is deprecated. In a future version this will raise instead of truncating to the smaller of the two sequences combi = pd.concat(all_pos, axis=1, keys=["UA1", "UA2"])
        all_pos = []
        if leq_cols is None:
            leq_cols = ["Leq"]
        for key in self._logs.keys():
            log = self._logs[key]
            # Day
            days = log._get_period(data=log.get_antilogs(), period="days")
            days = days[leq_cols].apply(lambda x: np.round(10*np.log10(np.mean(x)), DECIMALS))
            # Night-time
            nights = log._get_period(data=log.get_antilogs(), period="nights")
            nights = nights[leq_cols].apply(lambda x: np.round(10*np.log10(np.mean(x)), DECIMALS))
            df = pd.DataFrame
            # Evening
            if log.is_evening():
                evenings = log._get_period(data=log.get_antilogs(), period="evenings")
                evenings = evenings[leq_cols].apply(lambda x: np.round(10 * np.log10(np.mean(x)), DECIMALS))
                df = pd.concat([days, evenings, nights], axis=1, keys=["Daytime", "Evening", "Night-time"])
            else:
                df = pd.concat([days, nights], axis=1, keys=["Daytime", "Night-time"])
            all_pos.append(df)
        combi = pd.concat(all_pos, axis=1, keys=["UA1", "UA2"])
        combi = combi.transpose()
        return combi

    def get_start_end(self):
        starts = [self._logs[key].get_start() for key in self._logs.keys()]
        ends = [self._logs[key].get_end() for key in self._logs.keys()]
        return min(starts), max(ends)

    def weather(self, interval=6, api_key=None, country="GB", postcode="WC1", tz="", recompute=False,
                drop_cols=None):
        if drop_cols is None:
            drop_cols = ["sunrise", "sunset", "feels_like", "dew_point", "visibility"]
        if self._weatherhist is not None and recompute==False:
            return self._weatherhist
        else:
            if api_key is None:
                raise ValueError("api_key is required")
            start, end = self.get_start_end()
            self._weather.reinit(start=start, end=end, interval=interval, api_key=api_key, country=country,
                                 postcode=postcode, tz=tz, units="metric")
            self._weatherhist = self._weather.compute_weather_history(drop_cols=drop_cols)
        return self._weatherhist

    def weather_summary(self):
        if self._weatherhist is None:
            raise ValueError("No weather history available. Use Survey.weather() first.")
        return pd.DataFrame([self._weatherhist.min(), self._weatherhist.max(), self._weatherhist.mean()],
                            index=["Min", "Max", "Mean"]).drop(columns=["dt"]).round(decimals=1)


# TODO: Fix this bug in weatherhist
# survey.weather(api_key=r"eef3f749e018627b70c2ead1475a1a32", postcode="HA8")
#                     dt   temp pressure humidity clouds wind_speed wind_deg  \
# 0  2025-09-03 08:59:00  17.52      998       97     75       6.69      210
# 1  2025-09-03 14:59:00  19.85      997       84     40       9.26      220
# 2  2025-09-03 20:59:00  16.27   1003.0     90.0   20.0       4.63    240.0
# 3  2025-09-04 02:59:00  14.59   1005.0     91.0   99.0       3.09    230.0
# 4  2025-09-04 08:59:00  15.08     1004       93     40       4.12      200
# 5  2025-09-04 14:59:00  18.73     1007       63     40       8.75      260
# 6  2025-09-04 20:59:00  15.64   1013.0     76.0    0.0        3.6    270.0
# 7  2025-09-05 02:59:00  11.42   1016.0     94.0    0.0       3.09    260.0
# 8  2025-09-05 08:59:00  14.12   1020.0     89.0   20.0       3.09    270.0
# 9  2025-09-05 14:59:00  22.16   1021.0     50.0    0.0       4.12    280.0
# 10 2025-09-05 20:59:00  17.38   1023.0     75.0   75.0       3.09    220.0
# 11 2025-09-06 02:59:00  14.37   1022.0     83.0   99.0       1.78    187.0
# 12 2025-09-06 08:59:00  16.44   1020.0     73.0  100.0       3.48    138.0
# 13 2025-09-06 14:59:00  23.21   1037.0     50.0    0.0       7.72    160.0
# 14 2025-09-06 20:59:00   18.5   1035.0     75.0   93.0        3.6    120.0
# 15 2025-09-07 02:59:00  16.06   1031.0     77.0   84.0       3.09    120.0
# 16 2025-09-07 08:59:00  18.78   1029.0     77.0    0.0       4.63    110.0
# 17 2025-09-07 14:59:00  23.82   1027.0     67.0   75.0       8.75    200.0
# 18 2025-09-07 20:59:00  19.38   1031.0     76.0   72.0       4.63    200.0
# 19 2025-09-08 02:59:00  14.49   1034.0     91.0    4.0       1.54    190.0
# 20 2025-09-08 08:59:00  14.84   1037.0     85.0   20.0       4.12    240.0
#             rain wind_gust   uvi
# 0   {'1h': 0.25}       NaN   NaN
# 1   {'1h': 1.27}     14.92   NaN
# 2            NaN       NaN   NaN
# 3            NaN       NaN   NaN
# 4   {'1h': 1.27}       NaN   NaN
# 5   {'3h': 0.13}       NaN   NaN
# 6            NaN       NaN   NaN
# 7            NaN       NaN   NaN
# 8            NaN       NaN   NaN
# 9            NaN       NaN   NaN
# 10           NaN       NaN   NaN
# 11           NaN      3.31   0.0
# 12           NaN       7.4  0.86
# 13           NaN       NaN  2.96
# 14           NaN       NaN   0.0
# 15           NaN       NaN   0.0
# 16           NaN       NaN   1.1
# 17           NaN       NaN  2.24
# 18           NaN       NaN   0.0
# 19           NaN       NaN   0.0
# 20           NaN       NaN  1.12
# survey.weather_summary()
# Traceback (most recent call last):
#   File "<input>", line 1, in <module>
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pycoustic\survey.py", line 328, in weather_summary
#     return pd.DataFrame([self._weatherhist.min(), self._weatherhist.max(), self._weatherhist.mean()],
#                          ^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\frame.py", line 11643, in min
#     result = super().min(axis, skipna, numeric_only, **kwargs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\generic.py", line 12388, in min
#     return self._stat_function(
#            ^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\generic.py", line 12377, in _stat_function
#     return self._reduce(
#            ^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\frame.py", line 11562, in _reduce
#     res = df._mgr.reduce(blk_func)
#           ^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\internals\managers.py", line 1500, in reduce
#     nbs = blk.reduce(func)
#           ^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\internals\blocks.py", line 404, in reduce
#     result = func(self.values)
#              ^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\frame.py", line 11481, in blk_func
#     return op(values, axis=axis, skipna=skipna, **kwds)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\nanops.py", line 147, in f
#     result = alt(values, axis=axis, skipna=skipna, **kwds)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\nanops.py", line 404, in new_func
#     result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\pandas\core\nanops.py", line 1098, in reduction
#     result = getattr(values, meth)(axis)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\tonyr\PycharmProjects\pycoustic\.venv2\Lib\site-packages\numpy\_core\_methods.py", line 48, in _amin
#     return umr_minimum(a, axis, None, out, keepdims, initial, where)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: '<=' not supported between instances of 'dict' and 'dict'