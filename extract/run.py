"""Extract + load entrypoint: TLC months and Open-Meteo weather -> MySQL.

Incremental by default: loads every published TLC month not yet in
`extract_log` (starting at BACKFILL_START on an empty database), then
tops weather up to cover the same range. Re-running is safe; every write
is an upsert.

    python -m extract.run                     # incremental
    python -m extract.run --months 2026-06 2026-07   # force specific months
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path

import pandas as pd
import requests
import sqlalchemy as sa

from extract import tlc, weather
from load.db import apply_schema, get_engine
from load.upsert import upsert_dataframe

log = logging.getLogger("extract")

RAW_DIR = Path("data/raw")
# 2024-01 onward keeps a full backfill (~3M rows) comfortably inside
# Aiven's 1 GB free tier while still giving Prophet 2+ yearly cycles.
DEFAULT_BACKFILL_START = "2024-01"
# Open-Meteo archive requests are chunked by year to keep responses small.
WEATHER_CHUNK_DAYS = 366
# Weather must also cover the forecast horizon *after* the last pickup
# hour: forecast.run's forward week needs its weather features (8 days =
# the 168h horizon plus slack). TLC's ~2-month lag means the archive
# already has it.
FORECAST_WEATHER_DAYS = 8


def loaded_months(engine: sa.Engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT period FROM extract_log WHERE source = 'tlc'"))
        return {r[0] for r in rows}


def months_to_load(engine: sa.Engine, session: requests.Session) -> list[str]:
    latest = tlc.latest_published_month(session=session)
    if latest is None:
        raise RuntimeError("No published TLC month found in the last 6 months")
    start = os.environ.get("BACKFILL_START", DEFAULT_BACKFILL_START)
    done = loaded_months(engine)
    return [m for m in tlc.month_range(start, latest) if m not in done]


def load_zones(engine: sa.Engine) -> None:
    zones = tlc.load_zone_lookup()
    upsert_dataframe(engine, "dim_zone", zones, ["zone_id"])
    log.info("dim_zone: %d zones", len(zones))


def load_month(engine: sa.Engine, month: str, session: requests.Session, keep_raw: bool) -> int:
    path = tlc.download_month(month, RAW_DIR, session)
    hourly = tlc.aggregate_month(path, month)
    n = upsert_dataframe(engine, "fact_hourly_pickups", hourly, ["pickup_hour", "zone_id"])
    log_row = pd.DataFrame(
        [{"source": "tlc", "period": month, "rows_loaded": n, "loaded_at": dt.datetime.now()}]
    )
    # Logged only after the fact rows commit, so a crash mid-month means
    # the month is retried (and upserted over) on the next run.
    upsert_dataframe(engine, "extract_log", log_row, ["source", "period"])
    if not keep_raw:
        path.unlink(missing_ok=True)
    log.info("tlc %s: %d hourly zone rows", month, n)
    return n


def load_weather(engine: sa.Engine, session: requests.Session) -> int:
    with engine.connect() as conn:
        bounds = conn.execute(
            sa.text("SELECT MIN(pickup_hour), MAX(pickup_hour) FROM fact_hourly_pickups")
        ).one()
        latest_weather = conn.execute(sa.text("SELECT MAX(obs_hour) FROM weather_hourly")).scalar()
    if bounds[0] is None:
        log.info("weather: no trips loaded yet, skipping")
        return 0
    # Re-fetch the last loaded day too: the archive's newest hours can be
    # provisional and get revised.
    start = latest_weather.date() if latest_weather else bounds[0].date()
    end = min(
        bounds[1].date() + dt.timedelta(days=FORECAST_WEATHER_DAYS),
        dt.date.today() - dt.timedelta(days=weather.ARCHIVE_LAG_DAYS),
    )
    total = 0
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + dt.timedelta(days=WEATHER_CHUNK_DAYS - 1), end)
        df = weather.fetch_hourly(chunk_start, chunk_end, session)
        total += upsert_dataframe(engine, "weather_hourly", df, ["obs_hour"])
        chunk_start = chunk_end + dt.timedelta(days=1)
    log.info("weather: %d hourly rows (%s -> %s)", total, start, end)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="*", help="YYYY-MM months to (re)load")
    parser.add_argument("--keep-raw", action="store_true", help="keep downloaded Parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    engine = get_engine()
    apply_schema(engine)
    session = requests.Session()

    load_zones(engine)
    months = args.months or months_to_load(engine, session)
    log.info("tlc months to load: %s", months or "none (up to date)")
    for month in months:
        load_month(engine, month, session, keep_raw=args.keep_raw)
    load_weather(engine, session)


if __name__ == "__main__":
    main()
