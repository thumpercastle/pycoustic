import pandas as pd
import numpy as np


class Survey:
    """
    Survey Class is an overarching class which takes multiple Log objects and processes and summarises them together.
    This should be the main interface for user interaction with their survey data.
    """

    # ###########################---PRIVATE---######################################

    def __init__(self):
        self._logs = {}

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

    # TODO: get_lowest_l90

    def leq_spectra(self, leq_cols=None):
        """
        Compute Leqs over daytime, evening and night-time periods.
        This is an overall Leq, and does not group Leqs by date.
        :param leq_cols: List of strings or List of Tuples.
        For all Leq columns, use ["Leq"]. For specific columns, use list of tuples [("Leq", "A"), ("Leq", 125)]
        :return: A dataframe with a continuous Leq computation across dates, for each time period.
        """
        #TODO: C:\Users\tonyr\PycharmProjects\src\tests.py:674: FutureWarning: The behavior of pd.concat with len(keys) != len(objs) is deprecated. In a future version this will raise instead of truncating to the smaller of the two sequences combi = pd.concat(all_pos, axis=1, keys=["UA1", "UA2"])
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

    # def typical_leq_spectra(self, leq_cols=None):
    #     """
    #     DEPRECATED 2025/06/05. Replaced by .leq_spectra() **TT**
    #     Compute Leqs over daytime, evening and night-time periods.
    #     This is an overall Leq, and does not group Leqs by date.
    #     :param leq_cols: List of strings or List of Tuples.
    #     For all Leq columns, use ["Leq"]. For specific columns, use list of tuples [("Leq", "A"), ("Leq", 125)]
    #     :return: A dataframe with a continuous Leq computation across dates, for each time period.
    #     """
    #     combi = pd.DataFrame()
    #     if leq_cols is None:
    #         leq_cols = ["Leq"]
    #     for key in self._logs.keys():
    #         log = self._logs[key]
    #         combined_list = []
    #         # Day
    #         days = log._get_period(data=log.get_antilogs(), period="days")
    #         days = days[leq_cols].apply(lambda x: np.round(10*np.log10(np.mean(x)), DECIMALS))
    #         #days.sort_index(inplace=True)
    #         combined_list.append(days)
    #         period_headers = ["Daytime" for i in range(len(leq_cols))]
    #         # Evening
    #         if log.is_evening():
    #             evenings = log._get_period(data=log.get_antilogs(), period="evenings")
    #             evenings = evenings[leq_cols].apply(lambda x: np.round(10*np.log10(np.mean(x)), DECIMALS))
    #             evenings.sort_index(inplace=True)
    #             combined_list.append(evenings)
    #             for i in range(len(leq_cols)):
    #                 period_headers.append("Evening")
    #         # Night Leq
    #         nights = log._get_period(data=log.get_antilogs(), period="nights")
    #         nights = nights[leq_cols].apply(lambda x: np.round(10*np.log10(np.mean(x)), DECIMALS))
    #         nights.sort_index(inplace=True)
    #         combined_list.append(nights)
    #         for i in range(len(leq_cols)):
    #             period_headers.append("Night-time")
    #         summary = pd.concat(objs=combined_list, axis=1)
    #         summary = self._insert_multiindex(df=summary, super=key)
    #         combi = pd.concat(objs=[combi, summary], axis=0)
    #     new_head_dict = {}
    #     for i in range(len(period_headers)):
    #         new_head_dict[i] = period_headers[i]
    #     combi.rename(columns=new_head_dict, inplace=True)
    #     #combi = combi.transpose()
    #     return combi
