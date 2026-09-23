"""Rolling-origin backtesting.

Each fold trains on everything up to a cutoff and predicts the next
`horizon` hours; cutoffs step back from the end of the data one horizon
at a time, so the n folds are the n most recent non-overlapping weeks. A
single train/test split would score the model on one arbitrary week; this
shows how stable the error is across several.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast.features import DEFAULT_HORIZON, feature_columns


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: pd.Timestamp  # inclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive


def rolling_origin_folds(
    last_observed: pd.Timestamp,
    n_folds: int,
    horizon: int = DEFAULT_HORIZON,
) -> list[Fold]:
    folds = []
    for k in range(n_folds):
        test_end = last_observed - pd.Timedelta(hours=k * horizon)
        test_start = test_end - pd.Timedelta(hours=horizon - 1)
        train_end = test_start - pd.Timedelta(hours=1)
        folds.append(Fold(0, train_end, test_start, test_end))
    # Oldest first, numbered 1..n.
    return [
        Fold(i + 1, f.train_end, f.test_start, f.test_end) for i, f in enumerate(reversed(folds))
    ]


def score(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and WAPE.

    WAPE (sum|error| / sum|actual|) rather than MAPE: hourly counts hit zero
    (Staten Island at 4am), where MAPE divides by zero.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    errors = yhat - y
    denom = np.abs(y).sum()
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "wape": float(np.abs(errors).sum() / denom) if denom else float("nan"),
    }


def trainable(frame: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    return frame.dropna(subset=["y", *feature_columns(horizon)])


def run_backtest(
    frame: pd.DataFrame,
    model_factories: dict[str, Callable[[], object]],
    n_folds: int,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (predictions, metrics) long-format DataFrames."""
    data = trainable(frame, horizon)
    last_observed = frame["y"].last_valid_index()
    predictions, metrics = [], []
    for fold in rolling_origin_folds(last_observed, n_folds, horizon):
        train = data.loc[: fold.train_end]
        test = data.loc[fold.test_start : fold.test_end]
        if train.empty or test.empty:
            continue
        for name, factory in model_factories.items():
            yhat = factory().fit(train).predict(test)
            predictions.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold.number,
                        "target_hour": test.index,
                        "y": test["y"].to_numpy(),
                        "yhat": yhat,
                    }
                )
            )
            metrics.append({"model": name, "fold": fold.number, **score(test["y"], yhat)})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)
