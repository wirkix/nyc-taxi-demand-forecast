import datetime as dt

import numpy as np
import pandas as pd

from load.db import SCHEMA_PATH, split_statements
from load.upsert import build_upsert_sql, dataframe_rows


def test_build_upsert_sql_updates_only_non_key_columns():
    sql = build_upsert_sql(
        "fact_hourly_pickups", ["pickup_hour", "zone_id", "trips"], ["pickup_hour", "zone_id"]
    )
    assert sql == (
        "INSERT INTO `fact_hourly_pickups` (`pickup_hour`, `zone_id`, `trips`) "
        "VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE `trips` = VALUES(`trips`)"
    )


def test_build_upsert_sql_all_key_table_gets_noop_update():
    sql = build_upsert_sql("t", ["a", "b"], ["a", "b"])
    assert sql.endswith("ON DUPLICATE KEY UPDATE `a` = `a`")


def test_dataframe_rows_converts_to_plain_python():
    df = pd.DataFrame(
        {
            "hour": pd.to_datetime(["2026-07-01 08:00"]),
            "trips": np.array([5], dtype="int64"),
            "avg": [np.nan],
        }
    )
    [row] = dataframe_rows(df)
    assert row == (dt.datetime(2026, 7, 1, 8), 5, None)
    assert type(row[1]) is int


def test_schema_statements_all_have_primary_keys():
    statements = split_statements(SCHEMA_PATH.read_text(encoding="utf-8"))
    tables = [s for s in statements if s.startswith("CREATE TABLE")]
    assert len(tables) == 8
    # Aiven enforces sql_require_primary_key -- a table without one fails there.
    assert all("PRIMARY KEY" in s for s in tables)
