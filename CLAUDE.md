# CLAUDE.md

Project #6 (final) of Alois Wirkes' portfolio roadmap ("Demand
Forecasting Lab"): hourly NYC yellow-taxi demand, forecast a week ahead
per borough. NYC TLC trip Parquet + Open-Meteo weather -> MySQL (Aiven
free tier) -> features -> seasonal-naive / gradient-boosting / Prophet
with rolling-origin backtests -> a papermill-executed Plotly notebook
published to GitHub Pages by a weekly GitHub Actions run.

The roadmap originally said this project would read another portfolio
project's warehouse; the owner chose a new international data source
instead, so every project in the portfolio has its own domain.

## Layout

```
extract/tlc.py        TLC monthly Parquet download + DuckDB hourly-per-zone aggregation
extract/weather.py    Open-Meteo archive client (keyless)
extract/run.py        incremental extract+load entrypoint (python -m extract.run)
load/schema.sql       MySQL DDL (idempotent, applied on every run)
load/db.py            SQLAlchemy/PyMySQL engine; TLS via MYSQL_SSL_CA for Aiven
load/upsert.py        INSERT ... ON DUPLICATE KEY UPDATE bulk upsert
forecast/features.py  calendar/holiday/lag/rolling/weather features
forecast/models.py    SeasonalNaive, GradientBoosting, ProphetModel (fit/predict)
forecast/backtest.py  rolling-origin folds + MAE/RMSE/WAPE
forecast/run.py       backtest + forward forecast entrypoint (python -m forecast.run)
notebooks/report.ipynb  papermill-parameterized report (run_id)
docker-compose.yml    local MySQL 8.4 only
.github/workflows/    ci.yml (ruff + pytest), refresh.yml (weekly pipeline + Pages)
```

## Local dev

- Python 3.12 venv (`py -3.12 -m venv .venv`) -- same as the other roadmap
  repos; this machine's default 3.14 isn't used.
- `docker compose up -d --wait` for MySQL, `cp .env.example .env`, then
  `python -m extract.run` / `python -m forecast.run`.
- Report locally: `papermill notebooks/report.ipynb report.executed.ipynb --cwd .`
  (`--cwd .` matters: the notebook imports `load.db` from the repo root).
- No custom Docker image is built anywhere in this repo on purpose -- it
  sidesteps the Avast pip-inside-container TLS problem the other roadmap
  repos hit (see economic-pulse-lakehouse's CLAUDE.md if one is ever needed).

## Data gotchas

- **TLC files contain rows outside their own month** (July 2026's file has
  pickups dated 2008-12-30 and 2026-08-05). `aggregate_month` clips to the
  file's month; without that, stray rows would upsert over other months'
  real hourly counts. Test covers it.
- TLC publishes ~2 months behind. The "forward forecast" is therefore for
  the week right after the latest published month, not literally next
  week -- the report says so. Weather for that week is already in the
  Open-Meteo archive, so the forecast uses *observed* weather (a "perfect
  weather forecast" assumption, same in the backtests).
- Zones 264/265 (Unknown / Outside of NYC) are dropped; `total_amount <= 0`
  rows (voids/refunds) are dropped.
- Both sources use **naive NYC local time**. Open-Meteo with
  `timezone=America/New_York` returns exactly 24 hours on DST-change days,
  matching TLC's naive timestamps -- join on hour directly, never convert
  one side to UTC.
- The fact table stores only hours with >= 1 trip; `series_from_borough_rows`
  zero-fills. Staten Island has many real zero hours.

## Modeling gotchas

- Direct week-ahead forecasting: **no feature may use data inside the
  horizon**. Lags are 168/336h, rolling means are shifted by the horizon.
  `add_lag_features` raises if a lag shorter than the horizon sneaks in.
- WAPE, not MAPE -- MAPE divides by zero on zero-trip hours.
- Prophet fits are the slow part (~tens of seconds per series per fold).
  `--models`/`--series`/`--folds` flags on `forecast.run` trim a run.

## Aiven / MySQL

- Aiven enforces `sql_require_primary_key=ON`; docker-compose.yml turns
  the same flag on locally and a test asserts every table has a PK.
- Aiven requires TLS: the refresh workflow writes the `MYSQL_SSL_CA_PEM`
  secret to a file and points `MYSQL_SSL_CA` at it.
- Aiven's free tier is 1 GB; hour x zone is ~100k rows/month, so
  `BACKFILL_START` defaults to 2024-01 (~3M rows).
- Free services can be powered off after inactivity -- the weekly refresh
  run is the keep-alive.

## Known gotchas / history

- **First real run (2026-09-23, local MySQL, 2025-01..2026-07):** 2.2M
  hour x zone rows / 73.8M trips, ~6.5 MB per month in MySQL (so a
  2024-01 backfill is ~200 MB). Mean backtest WAPE, 4 folds:
  total -- gradient boosting 9.5%, seasonal naive 14.5%, Prophet 18.1%.
  Gradient boosting beats the baseline on every series; Prophet loses to
  it everywhere except Queens. That's the honest result; don't tune the
  report to hide it.
- **Staten Island isn't forecast**: ~7k yellow-cab pickups in 19 months
  (~12/day), an all-zeros series. Dropped from `SERIES_IDS`; still in
  "total" and in the history chart.
- **Weather must extend past the last pickup hour.** The first real
  forecast run crashed because weather stopped at the TLC data's end and
  the forward week had none. `extract/run.py` now loads weather
  `FORECAST_WEATHER_DAYS` past it, and `forecast/run.py` raises on any NaN
  feature in the horizon -- HistGradientBoosting accepts NaNs and would
  otherwise have silently forecast with no weather.
- **Sort before plotting.** MySQL `GROUP BY` output is unordered and
  `px.line` draws in row order, so the first report render had zig-zag
  lines. Every notebook query that feeds a line chart sorts explicitly.
- Prophet warns about <730 days of history with only 19 months loaded
  locally; the prod default (`BACKFILL_START=2024-01`) gives it 31.
