"""Forecast entrypoint: MySQL -> features -> backtest + forward forecast -> MySQL.

Per series (total + each borough): rolling-origin backtest of every model,
then refit on all history and forecast the `horizon` hours right after the
latest published TLC month. Everything from one invocation shares a
run_id, which the report notebook reads.

    python -m forecast.run
    python -m forecast.run --series total Manhattan --models seasonal_naive gradient_boosting
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import uuid
from functools import partial

import pandas as pd
import sqlalchemy as sa

from forecast.backtest import run_backtest, trainable
from forecast.features import (
    DEFAULT_HORIZON,
    SERIES_IDS,
    build_frame,
    feature_columns,
    series_from_borough_rows,
)
from forecast.models import MODELS
from load.db import apply_schema, get_engine
from load.upsert import upsert_dataframe

log = logging.getLogger("forecast")

DEFAULT_FOLDS = 4


def read_inputs(engine: sa.Engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    with engine.connect() as conn:
        boroughs = pd.read_sql(sa.text("SELECT * FROM v_hourly_pickups_borough"), conn)
        weather = pd.read_sql(sa.text("SELECT * FROM weather_hourly"), conn)
    boroughs["trips"] = boroughs["trips"].astype("float64")
    return boroughs, weather


def forecast_series(
    series_id: str,
    boroughs: pd.DataFrame,
    weather: pd.DataFrame,
    model_names: list[str],
    n_folds: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series = series_from_borough_rows(boroughs, series_id)
    frame = build_frame(series, weather, horizon)
    factories = {name: partial(MODELS[name], horizon=horizon) for name in model_names}

    bt_preds, bt_metrics = run_backtest(frame, factories, n_folds, horizon)

    train = trainable(frame, horizon)
    future = frame.loc[frame["y"].isna()]
    # Fail loudly: HistGradientBoosting accepts NaN features and would
    # silently forecast without weather (Prophet at least raises).
    missing = future[feature_columns(horizon)].isna().any()
    if missing.any():
        raise RuntimeError(
            f"{series_id}: forecast horizon has missing features {list(missing[missing].index)}"
            " -- is weather_hourly loaded past the last pickup hour? Re-run extract.run."
        )
    forward = []
    for name, factory in factories.items():
        yhat = factory().fit(train).predict(future)
        forward.append(pd.DataFrame({"model": name, "target_hour": future.index, "yhat": yhat}))
    forward_df = pd.concat(forward, ignore_index=True)

    for df in (bt_preds, bt_metrics, forward_df):
        df.insert(0, "series_id", series_id)
    summary = bt_metrics.groupby("model")["wape"].mean().round(3).to_dict()
    log.info("%s: mean backtest WAPE %s", series_id, summary)
    return bt_preds, bt_metrics, forward_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="*", default=list(SERIES_IDS))
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine = get_engine()
    apply_schema(engine)
    boroughs, weather = read_inputs(engine)
    if boroughs.empty:
        raise SystemExit("No pickups loaded -- run `python -m extract.run` first")

    run_id = uuid.uuid4().hex
    results = [
        forecast_series(s, boroughs, weather, args.models, args.folds, args.horizon)
        for s in args.series
    ]
    bt_preds, bt_metrics, forward = (
        pd.concat(parts, ignore_index=True) for parts in zip(*results, strict=True)
    )

    run_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "created_at": dt.datetime.now(),
                "train_end": boroughs["pickup_hour"].max(),
                "horizon_hours": args.horizon,
                "n_folds": args.folds,
            }
        ]
    )
    for df in (bt_preds, bt_metrics, forward):
        df.insert(0, "run_id", run_id)
    upsert_dataframe(
        engine,
        "backtest_predictions",
        bt_preds,
        ["run_id", "series_id", "model", "fold", "target_hour"],
    )
    upsert_dataframe(
        engine, "backtest_metrics", bt_metrics, ["run_id", "series_id", "model", "fold"]
    )
    upsert_dataframe(
        engine, "forecast_predictions", forward, ["run_id", "series_id", "model", "target_hour"]
    )
    # Run row last: the report reads the newest run, so a run only becomes
    # visible once all of its rows are written.
    upsert_dataframe(engine, "forecast_runs", run_row, ["run_id"])
    log.info("run %s written", run_id)


if __name__ == "__main__":
    main()
