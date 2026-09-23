# NYC Taxi Demand Forecast

Week-ahead forecasts of hourly yellow-taxi pickups in New York City, per
borough, with three models compared through rolling-origin backtests and
published as a reproducible notebook report that rebuilds itself weekly.

## Architecture

```
NYC TLC monthly Parquet ──download──> DuckDB: clip to month, drop voids,
  (~3.5M trips/month)                 aggregate to hour x zone (~100k rows)
                                                   │
Open-Meteo archive API ──hourly weather────────────┤
                                                   ▼
                                      MySQL (Aiven free tier)
                          dim_zone · fact_hourly_pickups · weather_hourly
                                                   │
                                                   ▼
          features: calendar, US/NY holidays, 168h/336h lags, shifted
          rolling means, temperature/precipitation/snow/wind
                                                   │
               ┌───────────────────┬───────────────┴────────┐
          seasonal naive    gradient boosting (Poisson)    Prophet
               └───────────────────┴───────────────┬────────┘
                    rolling-origin backtests + forward forecast
                           (written back to MySQL per run_id)
                                                   │
                                                   ▼
           notebooks/report.ipynb ─papermill─> nbconvert ─> GitHub Pages
                    (GitHub Actions, every Monday)
```

## Design choices

- **Aggregate before loading.** Raw trips never reach the database: DuckDB
  reduces each monthly Parquet file straight to hour x zone grain, which
  is what keeps a multi-year history inside a 1 GB free MySQL.
- **Direct week-ahead forecasting, no leakage.** Every feature for a target
  hour is known a full week before it: lags start at 168h, and rolling
  windows are shifted by the horizon. Shorter lags would score better in a
  backtest and be unusable for a real week-ahead forecast.
- **A baseline that has to be beaten.** Seasonal naive ("same hour last
  week") is the bar; the backtest table shows where the ML models earn
  their complexity and where they don't.
- **Several folds, not one test week.** Rolling-origin backtesting over
  the most recent non-overlapping weeks shows how stable the error is.
- **WAPE over MAPE.** Hourly counts hit zero (Staten Island at 4am).
- **Idempotent everywhere.** Every write is an upsert keyed on the table's
  primary key, and `extract_log` records only fully loaded months, so any
  run can be retried.

## Setup (local)

Requires Python 3.12 and Docker.

```bash
py -3.12 -m venv .venv          # or python3.12 -m venv .venv
.venv/Scripts/activate          # or source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # defaults match docker-compose.yml
docker compose up -d --wait     # local MySQL 8.4
```

Neither data source needs an API key.

## Running

```bash
python -m extract.run                    # load every published month since BACKFILL_START
python -m extract.run --months 2026-07   # (re)load specific months
python -m forecast.run                   # backtests + forward forecast, all series/models
python -m forecast.run --series total --models seasonal_naive gradient_boosting --folds 2

papermill notebooks/report.ipynb report.executed.ipynb --cwd .
jupyter nbconvert report.executed.ipynb --to html --no-input --output-dir site --output index
```

## Tests

```bash
pytest -q        # offline: fixtures only, no network, no database
ruff check . && ruff format --check .
```

## Deploying (Aiven + GitHub Actions)

1. Create a free **Aiven for MySQL** service at https://aiven.io (no card
   required). From its Overview page, copy host/port/user/password/database
   and download the CA certificate.
2. Add repository secrets: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`,
   `MYSQL_PASSWORD`, `MYSQL_DATABASE`, and `MYSQL_SSL_CA_PEM` (the CA
   certificate's full text).
3. Settings -> Pages -> Source: **GitHub Actions**.
4. Run the **Refresh** workflow manually once (the first run backfills from
   `BACKFILL_START`); after that it runs every Monday.

## Data model

| table | grain |
|---|---|
| `dim_zone` | TLC taxi zone (263) with borough |
| `fact_hourly_pickups` | hour x pickup zone, hours with >= 1 trip |
| `weather_hourly` | hour (Central Park) |
| `v_hourly_pickups_borough` | view: hour x borough |
| `forecast_runs` | one row per `forecast.run` invocation |
| `backtest_predictions` / `backtest_metrics` | run x series x model x fold (x hour) |
| `forecast_predictions` | run x series x model x future hour |

## Known limitations

- Yellow taxis only. Green taxis and Uber/Lyft (FHVHV, ~20M trips/month)
  aren't included, so this is yellow-cab demand, not total ride-hail demand.
- TLC publishes ~2 months late, so the "next week" forecast is the week
  after the latest published month, and it uses observed rather than
  forecast weather (in the backtests too). Real-time use would swap in
  Open-Meteo's forecast endpoint.
- One weather point (Central Park) for the whole city.
