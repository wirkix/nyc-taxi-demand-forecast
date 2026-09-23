import datetime as dt

import pandas as pd
import pytest
import responses

from extract import tlc


def test_month_helpers():
    assert tlc.next_month("2025-12") == "2026-01"
    assert tlc.next_month("2026-07") == "2026-08"
    assert tlc.month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert tlc.month_url("2026-07").endswith("/trip-data/yellow_tripdata_2026-07.parquet")
    with pytest.raises(ValueError):
        tlc.month_url("2026-7x")


def _write_trips(path, rows):
    df = pd.DataFrame(
        rows, columns=["tpep_pickup_datetime", "PULocationID", "trip_distance", "total_amount"]
    )
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df.to_parquet(path)


def test_aggregate_month_clips_to_month_and_drops_voids(tmp_path):
    path = tmp_path / "yellow_tripdata_2026-07.parquet"
    _write_trips(
        path,
        [
            ("2026-07-01 08:05", 161, 1.0, 20.0),
            ("2026-07-01 08:40", 161, 3.0, 30.0),
            ("2026-07-01 08:59", 236, 2.0, 15.0),
            ("2026-07-31 23:59", 161, 1.0, 10.0),
            # Real files carry stray out-of-month timestamps like these:
            ("2008-12-30 23:06", 161, 1.0, 10.0),
            ("2026-08-01 00:00", 161, 1.0, 10.0),
            # Void/refund, and the Unknown/Outside-NYC zones:
            ("2026-07-01 08:10", 161, 1.0, -20.0),
            ("2026-07-01 08:10", 264, 1.0, 20.0),
            ("2026-07-01 08:10", 265, 1.0, 20.0),
        ],
    )
    df = tlc.aggregate_month(path, "2026-07")

    assert df["trips"].sum() == 4
    row = df[(df["pickup_hour"] == "2026-07-01 08:00") & (df["zone_id"] == 161)].iloc[0]
    assert row["trips"] == 2
    assert row["avg_distance_mi"] == pytest.approx(2.0)
    assert row["avg_total_amount"] == pytest.approx(25.0)
    assert df["pickup_hour"].min() >= pd.Timestamp("2026-07-01")
    assert df["pickup_hour"].max() < pd.Timestamp("2026-08-01")
    assert not df["zone_id"].isin([264, 265]).any()


@responses.activate
def test_latest_published_month_probes_backwards():
    for month, status in [("2026-09", 403), ("2026-08", 403), ("2026-07", 200)]:
        responses.add(responses.HEAD, tlc.month_url(month), status=status)
    assert tlc.latest_published_month(today=dt.date(2026, 9, 23)) == "2026-07"


@responses.activate
def test_download_month_reuses_existing_file(tmp_path):
    existing = tmp_path / "yellow_tripdata_2026-07.parquet"
    existing.write_bytes(b"already here")
    # No responses registered: any HTTP call would raise ConnectionError.
    assert tlc.download_month("2026-07", tmp_path) == existing


def test_load_zone_lookup_drops_unknown_zones(tmp_path):
    csv = tmp_path / "zones.csv"
    csv.write_text(
        '"LocationID","Borough","Zone","service_zone"\n'
        '1,"EWR","Newark Airport","EWR"\n'
        '161,"Manhattan","Midtown Center","Yellow Zone"\n'
        '264,"Unknown","N/A","N/A"\n'
        '265,"N/A","Outside of NYC","N/A"\n'
    )
    zones = tlc.load_zone_lookup(csv)
    assert list(zones.columns) == ["zone_id", "borough", "zone_name", "service_zone"]
    assert zones["zone_id"].tolist() == [1, 161]
