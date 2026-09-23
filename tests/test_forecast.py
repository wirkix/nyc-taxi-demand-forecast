import numpy as np
import pandas as pd
import pytest

from forecast import features
from forecast.backtest import rolling_origin_folds, run_backtest, score
from forecast.models import GradientBoosting, SeasonalNaive


def _synthetic(weeks: int = 8) -> tuple[pd.Series, pd.DataFrame]:
    """A clean daily+weekly pattern the models should nail."""
    index = pd.date_range("2026-01-05", periods=weeks * 168, freq="h")
    daily = 100 + 80 * np.sin(2 * np.pi * (index.hour - 6) / 24)
    weekly = np.where(index.dayofweek >= 5, 0.7, 1.0)
    series = pd.Series(np.round(daily * weekly), index=index, name="y")
    weather = pd.DataFrame(
        {
            "obs_hour": pd.date_range(index[0], periods=len(index) + 168, freq="h"),
            "temperature_c": 5.0,
            "precipitation_mm": 0.0,
            "snowfall_cm": 0.0,
            "wind_speed_kmh": 10.0,
        }
    )
    return series, weather


def test_series_from_borough_rows_zero_fills_missing_hours():
    rows = pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(
                ["2026-07-01 00:00", "2026-07-01 00:00", "2026-07-01 03:00"]
            ),
            "borough": ["Manhattan", "Staten Island", "Manhattan"],
            "trips": [10.0, 1.0, 7.0],
        }
    )
    si = features.series_from_borough_rows(rows, "Staten Island")
    assert si.tolist() == [1.0, 0.0, 0.0, 0.0]
    total = features.series_from_borough_rows(rows, "total")
    assert total.tolist() == [11.0, 0.0, 0.0, 7.0]


def test_lag_features_never_look_inside_the_horizon():
    series, weather = _synthetic()
    frame = features.build_frame(series, weather, horizon=168)
    target = frame.index[400]
    assert frame.loc[target, "lag_168"] == series.loc[target - pd.Timedelta(hours=168)]
    # roll_mean_24 for hour t = mean of the 24h ending at t-168.
    window = series.loc[target - pd.Timedelta(hours=191) : target - pd.Timedelta(hours=168)]
    assert frame.loc[target, "roll_mean_24"] == pytest.approx(window.mean())
    with pytest.raises(ValueError):
        features.add_lag_features(frame[["y"]], horizon=24 * 30)


def test_build_frame_appends_future_horizon_with_features():
    series, weather = _synthetic()
    frame = features.build_frame(series, weather, horizon=168)
    future = frame.loc[frame["y"].isna()]
    assert len(future) == 168
    assert future.index[0] == series.index[-1] + pd.Timedelta(hours=1)
    assert not future[features.feature_columns(168)].isna().any().any()


def test_holiday_flag():
    frame = features.add_calendar_features(
        pd.DataFrame(index=pd.to_datetime(["2026-07-04 12:00", "2026-07-06 12:00"]))
    )
    assert frame["is_holiday"].tolist() == [1, 0]


def test_folds_are_contiguous_non_overlapping_weeks_ending_at_last_observation():
    last = pd.Timestamp("2026-07-31 23:00")
    folds = rolling_origin_folds(last, n_folds=3, horizon=168)
    assert [f.number for f in folds] == [1, 2, 3]
    assert folds[-1].test_end == last
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.test_start == earlier.test_end + pd.Timedelta(hours=1)
    for f in folds:
        assert f.train_end < f.test_start
        assert f.test_end - f.test_start == pd.Timedelta(hours=167)


def test_score():
    m = score(np.array([10.0, 0.0, 10.0]), np.array([12.0, 1.0, 7.0]))
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert m["wape"] == pytest.approx(6 / 20)


def test_backtest_on_a_perfectly_weekly_series():
    series, weather = _synthetic()
    frame = features.build_frame(series, weather, horizon=168)
    factories = {"seasonal_naive": SeasonalNaive, "gradient_boosting": GradientBoosting}
    preds, metrics = run_backtest(frame, factories, n_folds=2, horizon=168)
    assert set(metrics["model"]) == set(factories)
    assert len(preds) == 2 * 2 * 168
    # A pattern that repeats exactly every week is seasonal-naive's best case.
    naive = metrics[metrics["model"] == "seasonal_naive"]
    assert naive["mae"].max() == 0
    assert metrics[metrics["model"] == "gradient_boosting"]["wape"].max() < 0.1
