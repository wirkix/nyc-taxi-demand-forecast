"""Three forecasters behind one fit/predict interface.

- seasonal_naive: same hour one week earlier. The baseline every other
  model has to beat to be worth its complexity.
- gradient_boosting: scikit-learn HistGradientBoosting over calendar, lag
  and weather features, with a Poisson loss (targets are counts).
- prophet: additive daily/weekly/yearly seasonality + US holidays, with
  weather as extra regressors.

All predictions are clipped at 0 -- negative trip counts aren't a forecast.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecast.features import DEFAULT_HORIZON, WEATHER_COLUMNS, feature_columns


class SeasonalNaive:
    name = "seasonal_naive"

    def __init__(self, horizon: int = DEFAULT_HORIZON):
        self.horizon = horizon

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.clip(future["lag_168"].to_numpy(dtype=float), 0, None)


class GradientBoosting:
    name = "gradient_boosting"

    def __init__(self, horizon: int = DEFAULT_HORIZON, random_state: int = 42):
        self.features = feature_columns(horizon)
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=63,
            categorical_features=[self.features.index("dow")],
            random_state=random_state,
        )

    def fit(self, train: pd.DataFrame) -> GradientBoosting:
        self.model.fit(train[self.features], train["y"])
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(future[self.features]), 0, None)


class ProphetModel:
    name = "prophet"
    regressors = ("temperature_c", "precipitation_mm", "snowfall_cm")

    def __init__(self, horizon: int = DEFAULT_HORIZON):
        # Imported lazily: prophet pulls in cmdstanpy and is slow to import.
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            # Taxi demand scales with the level (a busy Friday is busier in
            # absolute terms in peak season), not a fixed additive bump.
            seasonality_mode="multiplicative",
        )
        self.model.add_country_holidays(country_name="US")
        for column in self.regressors:
            assert column in WEATHER_COLUMNS
            self.model.add_regressor(column)

    def _to_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"ds": df.index})
        for column in self.regressors:
            out[column] = df[column].to_numpy()
        return out

    def fit(self, train: pd.DataFrame) -> ProphetModel:
        prophet_df = self._to_prophet(train)
        prophet_df["y"] = train["y"].to_numpy()
        self.model.fit(prophet_df)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        forecast = self.model.predict(self._to_prophet(future))
        return np.clip(forecast["yhat"].to_numpy(), 0, None)


MODELS = {
    SeasonalNaive.name: SeasonalNaive,
    GradientBoosting.name: GradientBoosting,
    ProphetModel.name: ProphetModel,
}
