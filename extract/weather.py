"""Hourly NYC weather from Open-Meteo's historical archive (keyless).

One point (Central Park) stands in for the whole city: borough-to-borough
weather differences are small next to the hour-to-hour swings that move
taxi demand.

`timezone=America/New_York` returns naive local wall-clock hours, exactly
24 per day even on DST-change days -- the same convention as TLC's naive
local pickup timestamps, so the two join on hour directly with no UTC
conversion.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CENTRAL_PARK = (40.7812, -73.9665)

# Open-Meteo variable -> our column name (units in the suffix)
VARIABLES = {
    "temperature_2m": "temperature_c",
    "precipitation": "precipitation_mm",
    "snowfall": "snowfall_cm",
    "wind_speed_10m": "wind_speed_kmh",
    "relative_humidity_2m": "relative_humidity_pct",
    "weather_code": "weather_code",
}

# The archive lags real time by a few days; asking past this returns nulls.
ARCHIVE_LAG_DAYS = 5


def parse_response(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({"obs_hour": pd.to_datetime(hourly["time"])})
    for api_name, column in VARIABLES.items():
        df[column] = hourly[api_name]
    # Trailing hours inside the archive lag come back as all-null rows.
    return df.dropna(subset=["temperature_c"]).reset_index(drop=True)


def fetch_hourly(
    start: dt.date,
    end: dt.date,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    session = session or requests.Session()
    lat, lon = CENTRAL_PARK
    response = session.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(VARIABLES),
            "timezone": "America/New_York",
        },
        timeout=60,
    )
    response.raise_for_status()
    return parse_response(response.json())
