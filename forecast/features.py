"""Feature engineering for hourly pickup series.

Forecasting is *direct* multi-step over a fixed horizon (default 168h, one
week): every feature for a target hour must be known `horizon` hours
before it. So no lag shorter than the horizon, and rolling windows are
shifted by the horizon too. Using lag-1 or lag-24 would look great in a
backtest and be unusable for an actual week-ahead forecast.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

DEFAULT_HORIZON = 168
LAGS = (168, 336)
ROLLING_WINDOWS = (24, 168)
# Staten Island is deliberately not forecast: real data has ~12 yellow-cab
# pickups a day there (7k in 19 months vs Manhattan's 64M), so its hourly
# series is almost all zeros -- noise, not demand. It still counts in "total".
BOROUGHS = ("Manhattan", "Brooklyn", "Queens", "Bronx")
SERIES_IDS = ("total", *BOROUGHS)
WEATHER_COLUMNS = ("temperature_c", "precipitation_mm", "snowfall_cm", "wind_speed_kmh")
CALENDAR_COLUMNS = (
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
)


def series_from_borough_rows(borough_df: pd.DataFrame, series_id: str) -> pd.Series:
    """Hourly trip counts for one series, zero-filled across the full range.

    `borough_df` has columns pickup_hour, borough, trips (v_hourly_pickups_borough).
    Missing hours mean zero trips -- the fact table only stores hours that
    had any (Staten Island has plenty of real zero hours overnight).
    """
    if borough_df.empty:
        return pd.Series(dtype="float64", name="y")
    if series_id == "total":
        grouped = borough_df.groupby("pickup_hour")["trips"].sum()
    else:
        grouped = borough_df.loc[borough_df["borough"] == series_id].set_index("pickup_hour")[
            "trips"
        ]
    full_index = pd.date_range(
        borough_df["pickup_hour"].min(), borough_df["pickup_hour"].max(), freq="h"
    )
    series = grouped.reindex(full_index, fill_value=0).astype("float64")
    series.index.name = "hour"
    return series.rename("y")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    out = df.copy()
    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    ny_holidays = holidays.US(subdiv="NY", years=sorted(set(idx.year)))
    out["is_holiday"] = np.array([d in ny_holidays for d in idx.date], dtype=int)
    # Cyclical encodings so 23:00 sits next to 00:00 and Dec next to Jan.
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return out


def lag_columns(horizon: int = DEFAULT_HORIZON) -> list[str]:
    lags = [f"lag_{lag}" for lag in LAGS if lag >= horizon]
    rolls = [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    return lags + rolls


def add_lag_features(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    if min(LAGS) < horizon:
        raise ValueError(f"shortest lag {min(LAGS)}h is inside the {horizon}h horizon")
    out = df.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out["y"].shift(lag)
    for window in ROLLING_WINDOWS:
        # Mean of the `window` hours ending `horizon` hours before the target.
        out[f"roll_mean_{window}"] = out["y"].rolling(window).mean().shift(horizon)
    return out


def add_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    weather = weather_df.set_index("obs_hour")[list(WEATHER_COLUMNS)]
    weather = weather[~weather.index.duplicated(keep="last")]
    # Short gaps (a missing archive hour) are forward-filled; longer ones
    # stay NaN and those rows drop out of training.
    joined = df.join(weather, how="left")
    joined[list(WEATHER_COLUMNS)] = joined[list(WEATHER_COLUMNS)].ffill(limit=3)
    return joined


def feature_columns(horizon: int = DEFAULT_HORIZON) -> list[str]:
    return [*CALENDAR_COLUMNS, *lag_columns(horizon), *WEATHER_COLUMNS]


def build_frame(
    series: pd.Series,
    weather_df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """History plus `horizon` future hours (y = NaN) with every feature filled.

    Rows with y present and all features present are trainable; the
    trailing `horizon` rows are what a forward forecast predicts.
    """
    future_index = pd.date_range(
        series.index.max() + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    extended = pd.concat([series, pd.Series(np.nan, index=future_index, name="y")])
    extended.index.name = "hour"
    frame = extended.to_frame()
    frame = add_calendar_features(frame)
    frame = add_lag_features(frame, horizon)
    frame = add_weather(frame, weather_df)
    return frame
