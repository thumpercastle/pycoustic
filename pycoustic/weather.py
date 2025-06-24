import requests
import pandas as pd
import datetime as dt

test=0

appid = ""
with open("tests/openweather_app_id.txt") as f:
    appid = f.readlines()[0]

w_dict = {
    "start": "2022-09-16 12:00:00",
    "end": "2022-09-17 18:00:00",
    "interval": 6,
    "api_key": appid,
    "country": "GB",
    "postcode": "WC1",
    "tz": "GB"
}

def test_weather_obj(weather_test_dict):
    hist = WeatherHistory(start=w_dict["start"], end=w_dict["end"], interval=w_dict["interval"],
                          api_key=w_dict["api_key"], country=w_dict["country"], postcode=w_dict["postcode"],
                          tz=w_dict["tz"])
    hist.compute_weather_history()
    return hist

#TODO: Make this take the start and end times of a Survey object.
#TODO: Implement post codes instead of coordinates
class WeatherHistory:
    def __init__(self):
        return

    def reinit(self, start=None, end=None, interval=6, api_key="", country="GB", postcode="WC1", tz="",
                 units="metric"):
        if api_key==None:
            raise ValueError("API key is missing")
        if type(start) == str:
            self._start = dt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        else:
            self._start = start
        if type(end) == str:
            self._end = dt.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        else:
            self._end = end
        self._interval = interval
        self._api_key = str(api_key)
        self._lat, self._lon = self.get_latlon(api_key=api_key, country=country, postcode=postcode)
        self._hist = None
        self._units = units

    def get_latlon(self, api_key="", country="GB", postcode=""):
        query = str("http://api.openweathermap.org/geo/1.0/zip?zip=" + postcode + "," + country + "&appid=" + api_key)
        resp = requests.get(query)
        return resp.json()["lat"], resp.json()["lon"]

    def _construct_api_call(self, timestamp):
        base = "https://api.openweathermap.org/data/3.0/onecall/timemachine?"
        query = str(base + "lat=" + str(self._lat) + "&" + "lon=" + str(self._lon) + "&" + "units=" + self._units + \
                    "&" + "dt=" + str(timestamp) + "&" + "appid=" + self._api_key)
        print(query)
        return query

    def _construct_timestamps(self):
        next_time = (self._start + dt.timedelta(hours=self._interval))
        timestamps = [int(self._start.timestamp())]
        while next_time < self._end:
            timestamps.append(int(next_time.timestamp()))
            next_time += dt.timedelta(hours=self._interval)
        return timestamps

    def _make_and_parse_api_call(self, query):
        response = requests.get(query)
        print(response.json())
        # This drops some unwanted cols like lat, lon, timezone and tz offset.
        resp_dict = response.json()["data"][0]
        del resp_dict["weather"]    # delete weather key as not useful.
        # TODO: parse 'weather' nested dict.
        return resp_dict

    def compute_weather_history(self):
        # construct timestamps
        timestamps = self._construct_timestamps()
        # make calls to API
        responses = []
        for ts in timestamps:
            print(f"ts: {ts}")
            query = self._construct_api_call(timestamp=ts)
            response_dict = self._make_and_parse_api_call(query=query)
            responses.append(pd.Series(response_dict))
        df = pd.concat(responses, axis=1).transpose()
        for col in ["dt", "sunrise", "sunset"]:
            df[col] = df[col].apply(lambda x: dt.datetime.fromtimestamp(int(x)))  # convert timestamp into datetime
        print(df)
        self._hist = df
        return df

    def get_weather_history(self):
        return self._hist
