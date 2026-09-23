-- MySQL schema. Idempotent: safe to run on every pipeline start.
--
-- Every table has an explicit PRIMARY KEY: Aiven for MySQL enforces
-- sql_require_primary_key=ON (so does docker-compose.yml's local MySQL,
-- to catch a missing one before it reaches Aiven).
--
-- All *_hour columns are naive NYC local wall-clock time (see
-- extract/weather.py for why that's consistent across both sources).

CREATE TABLE IF NOT EXISTS dim_zone (
    zone_id       SMALLINT     NOT NULL,
    borough       VARCHAR(32)  NOT NULL,
    zone_name     VARCHAR(64)  NOT NULL,
    service_zone  VARCHAR(32)  NOT NULL,
    PRIMARY KEY (zone_id)
);

-- Grain: one row per (hour, pickup zone) with at least one trip. Hours
-- with zero trips are absent, not stored as 0 -- forecast/features.py
-- fills them back in. ~100k rows/month; FLOAT (not DOUBLE) averages keep
-- a multi-year backfill well inside Aiven's 1 GB free-tier disk.
CREATE TABLE IF NOT EXISTS fact_hourly_pickups (
    pickup_hour       DATETIME  NOT NULL,
    zone_id           SMALLINT  NOT NULL,
    trips             INT       NOT NULL,
    avg_distance_mi   FLOAT     NULL,
    avg_total_amount  FLOAT     NULL,
    PRIMARY KEY (pickup_hour, zone_id),
    KEY idx_zone_hour (zone_id, pickup_hour)
);

CREATE TABLE IF NOT EXISTS weather_hourly (
    obs_hour               DATETIME  NOT NULL,
    temperature_c          FLOAT     NULL,
    precipitation_mm       FLOAT     NULL,
    snowfall_cm            FLOAT     NULL,
    wind_speed_kmh         FLOAT     NULL,
    relative_humidity_pct  FLOAT     NULL,
    weather_code           SMALLINT  NULL,
    PRIMARY KEY (obs_hour)
);

-- Which TLC months are fully loaded, so extract.run only fetches new ones.
CREATE TABLE IF NOT EXISTS extract_log (
    source       VARCHAR(16)  NOT NULL,
    period       VARCHAR(16)  NOT NULL,
    rows_loaded  INT          NOT NULL,
    loaded_at    DATETIME     NOT NULL,
    PRIMARY KEY (source, period)
);

CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id         CHAR(32)  NOT NULL,
    created_at     DATETIME  NOT NULL,
    train_end      DATETIME  NOT NULL,
    horizon_hours  SMALLINT  NOT NULL,
    n_folds        SMALLINT  NOT NULL,
    PRIMARY KEY (run_id)
);

-- Forward forecasts for the horizon right after the latest published TLC
-- month: scoreable against actuals once the next month is published.
CREATE TABLE IF NOT EXISTS forecast_predictions (
    run_id       CHAR(32)     NOT NULL,
    series_id    VARCHAR(32)  NOT NULL,
    model        VARCHAR(32)  NOT NULL,
    target_hour  DATETIME     NOT NULL,
    yhat         FLOAT        NOT NULL,
    PRIMARY KEY (run_id, series_id, model, target_hour)
);

-- Rolling-origin backtest predictions (kept, not just the metrics, so the
-- report can plot forecast-vs-actual per fold).
CREATE TABLE IF NOT EXISTS backtest_predictions (
    run_id       CHAR(32)     NOT NULL,
    series_id    VARCHAR(32)  NOT NULL,
    model        VARCHAR(32)  NOT NULL,
    fold         SMALLINT     NOT NULL,
    target_hour  DATETIME     NOT NULL,
    y            FLOAT        NOT NULL,
    yhat         FLOAT        NOT NULL,
    PRIMARY KEY (run_id, series_id, model, fold, target_hour)
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    run_id     CHAR(32)     NOT NULL,
    series_id  VARCHAR(32)  NOT NULL,
    model      VARCHAR(32)  NOT NULL,
    fold       SMALLINT     NOT NULL,
    mae        FLOAT        NOT NULL,
    rmse       FLOAT        NOT NULL,
    wape       FLOAT        NOT NULL,
    PRIMARY KEY (run_id, series_id, model, fold)
);

-- Borough-level series (what the forecasts actually model). Zero-filling
-- missing hours happens later in pandas, not here.
CREATE OR REPLACE VIEW v_hourly_pickups_borough AS
SELECT f.pickup_hour, z.borough, SUM(f.trips) AS trips
FROM fact_hourly_pickups f
JOIN dim_zone z ON z.zone_id = f.zone_id
GROUP BY f.pickup_hour, z.borough;
