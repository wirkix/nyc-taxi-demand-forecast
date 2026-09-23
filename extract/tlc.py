"""NYC TLC yellow-taxi trip records -> hourly pickups per zone.

TLC publishes one Parquet file per month (~60 MB, ~3.5M trips) on a public
CloudFront bucket, no key needed, roughly two months behind real time.
Raw files are only an intermediate: DuckDB aggregates each one straight to
(pickup_hour, zone_id) grain (~100k rows/month) so the 3.5M trip rows
never go through pandas or reach MySQL.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
ZONE_LOOKUP_URL = f"{BASE_URL}/misc/taxi_zone_lookup.csv"

# 264 = "Unknown" / 265 = "Outside of NYC" in the zone lookup -- not
# forecastable locations, dropped at aggregation time.
UNKNOWN_ZONE_IDS = (264, 265)


def month_url(month: str) -> str:
    """`month` is 'YYYY-MM'."""
    _parse_month(month)
    return f"{BASE_URL}/trip-data/yellow_tripdata_{month}.parquet"


def _parse_month(month: str) -> dt.date:
    return dt.datetime.strptime(month, "%Y-%m").date()


def next_month(month: str) -> str:
    d = _parse_month(month)
    return f"{d.year + d.month // 12}-{d.month % 12 + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end."""
    months, current = [], start
    while current <= end:
        months.append(current)
        current = next_month(current)
    return months


def is_published(month: str, session: requests.Session | None = None) -> bool:
    session = session or requests.Session()
    response = session.head(month_url(month), timeout=30)
    return response.status_code == 200


def latest_published_month(
    today: dt.date | None = None,
    session: requests.Session | None = None,
    lookback_months: int = 6,
) -> str | None:
    """Probe backwards from the current month for the newest published file."""
    today = today or dt.date.today()
    candidate = dt.date(today.year, today.month, 1)
    for _ in range(lookback_months):
        month = candidate.strftime("%Y-%m")
        if is_published(month, session):
            return month
        candidate = (candidate - dt.timedelta(days=1)).replace(day=1)
    return None


def download_month(month: str, dest_dir: Path, session: requests.Session | None = None) -> Path:
    """Stream one month's Parquet to disk; a complete existing file is reused."""
    session = session or requests.Session()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"yellow_tripdata_{month}.parquet"
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(".part")
    with session.get(month_url(month), stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(path)
    return path


# Clipping to the file's own month matters: real files carry stray rows
# with timestamps years outside it (July 2026's file has pickups dated
# 2008 and August 2026 -- 46 rows). Left in, they'd create phantom hours
# in other months and get overwritten or double-counted on upsert.
# total_amount <= 0 rows are voids/refunds, not real demand.
_AGGREGATE_SQL = """
SELECT
    date_trunc('hour', tpep_pickup_datetime)  AS pickup_hour,
    PULocationID                              AS zone_id,
    count(*)                                  AS trips,
    avg(trip_distance)                        AS avg_distance_mi,
    avg(total_amount)                         AS avg_total_amount
FROM read_parquet($path)
WHERE tpep_pickup_datetime >= $month_start
  AND tpep_pickup_datetime <  $month_end
  AND total_amount > 0
  AND PULocationID IS NOT NULL
  AND PULocationID NOT IN (264, 265)
GROUP BY ALL
ORDER BY pickup_hour, zone_id
"""


def aggregate_month(parquet_path: Path, month: str) -> pd.DataFrame:
    start = _parse_month(month)
    end = _parse_month(next_month(month))
    with duckdb.connect() as con:
        df = con.execute(
            _AGGREGATE_SQL,
            {"path": str(parquet_path), "month_start": start, "month_end": end},
        ).df()
    df["trips"] = df["trips"].astype("int64")
    df["zone_id"] = df["zone_id"].astype("int64")
    return df


def load_zone_lookup(source: str | Path = ZONE_LOOKUP_URL) -> pd.DataFrame:
    df = pd.read_csv(source)
    df = df.rename(
        columns={
            "LocationID": "zone_id",
            "Borough": "borough",
            "Zone": "zone_name",
            "service_zone": "service_zone",
        }
    )
    df = df[~df["zone_id"].isin(UNKNOWN_ZONE_IDS)]
    return df[["zone_id", "borough", "zone_name", "service_zone"]].fillna("Unknown")
