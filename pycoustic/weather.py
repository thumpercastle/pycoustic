import datetime as dt
from typing import Any

import pandas as pd
import requests

APP_ID = ""

w_dict = {
    "start": "2022-09-16 12:00:00",
    "end": "2022-09-17 18:00:00",
    "interval": 6,
    "api_key": APP_ID,
    "country": "GB",
    "postcode": "WC1",
    "tz": "GB",
}


def test_weather_obj(weather_test_dict: dict[str, Any]) -> "WeatherHistory":
    hist = WeatherHistory()
    hist.reinit(
        start=w_dict["start"],
        end=w_dict["end"],
        interval=w_dict["interval"],
        api_key=w_dict["api_key"],
        country=w_dict["country"],
        postcode=w_dict["postcode"],
        tz=w_dict["tz"],
    )
    hist.compute_weather_history()
    return hist


class WeatherHistory:
    def __init__(self) -> None:
        return

    @staticmethod
    def _parse_datetime(value: dt.datetime | str | None) -> dt.datetime | None:
        if isinstance(value, str):
            return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return value

    @staticmethod
    def _to_datetime_from_unix(value: Any) -> dt.datetime:
        return dt.datetime.fromtimestamp(int(value))

    def reinit(
            self,
            start: dt.datetime | str | None = None,
            end: dt.datetime | str | None = None,
            interval: int = 6,
            api_key: str | None = "",
            country: str = "GB",
            postcode: str = "WC1",
            tz: str = "",
            units: str = "metric",
    ) -> None:
        if api_key is None:
            raise ValueError("API key is missing")

        self._start = self._parse_datetime(start)
        self._end = self._parse_datetime(end)
        self._interval = interval
        self._api_key = str(api_key)
        self._lat, self._lon = self.get_latlon(api_key=api_key, country=country, postcode=postcode)
        self._hist = None
        self._units = units
        self._tz = tz

    def get_latlon(self, api_key: str = "", country: str = "GB", postcode: str = "") -> tuple[float, float]:
        query = f"http://api.openweathermap.org/geo/1.0/zip?zip={postcode},{country}&appid={api_key}"
        resp = requests.get(query)
        payload = resp.json()
        return payload["lat"], payload["lon"]

    def _construct_api_call(self, timestamp: int) -> str:
        base = "https://api.openweathermap.org/data/3.0/onecall/timemachine?"
        query = (
            f"{base}"
            f"lat={self._lat}&"
            f"lon={self._lon}&"
            f"units={self._units}&"
            f"dt={timestamp}&"
            f"appid={self._api_key}"
        )
        return query

    def _construct_timestamps(self) -> list[int]:
        next_time = self._start + dt.timedelta(hours=self._interval)
        timestamps = [int(self._start.timestamp())]
        while next_time < self._end:
            timestamps.append(int(next_time.timestamp()))
            next_time += dt.timedelta(hours=self._interval)
        return timestamps

    def _make_and_parse_api_call(self, query: str) -> dict[str, Any]:
        response = requests.get(query)
        resp_dict = response.json()["data"][0]
        del resp_dict["weather"]
        return resp_dict

    def compute_weather_history(self, drop_cols: list[str] | None = None) -> pd.DataFrame:
        if drop_cols is None:
            drop_cols = []

        timestamps = self._construct_timestamps()
        responses = []

        for ts in timestamps:
            query = self._construct_api_call(timestamp=ts)
            response_dict = self._make_and_parse_api_call(query=query)
            responses.append(pd.Series(response_dict))

        df = pd.concat(responses, axis=1).transpose()

        for col in ["dt", "sunrise", "sunset"]:
            df[col] = df[col].apply(self._to_datetime_from_unix)

        df.drop(columns=drop_cols, inplace=True)
        return df

    def get_weather_history(self) -> Any:
        return self._hist