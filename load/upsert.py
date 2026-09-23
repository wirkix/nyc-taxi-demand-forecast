"""Idempotent bulk upsert via INSERT ... ON DUPLICATE KEY UPDATE.

Re-running any extract (a month TLC republished with corrections, an
overlapping weather window) overwrites rows in place instead of erroring
or duplicating. Uses the DBAPI cursor's executemany: PyMySQL rewrites an
`INSERT ... VALUES (...)` statement into multi-row VALUES batches, one
round trip per chunk rather than one per row.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
from sqlalchemy.engine import Engine

CHUNK_ROWS = 5000


def build_upsert_sql(table: str, columns: Sequence[str], key_columns: Sequence[str]) -> str:
    cols = ", ".join(f"`{c}`" for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    updates = [c for c in columns if c not in key_columns]
    # A table that's all key columns still needs a no-op update clause.
    update_clause = ", ".join(f"`{c}` = VALUES(`{c}`)" for c in updates) or (
        f"`{key_columns[0]}` = `{key_columns[0]}`"
    )
    return (
        f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )


def _clean(value):
    # NaN/NaT -> NULL; numpy scalars -> Python scalars PyMySQL can escape.
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def dataframe_rows(df: pd.DataFrame) -> list[tuple]:
    return [tuple(_clean(v) for v in row) for row in df.itertuples(index=False, name=None)]


def upsert_dataframe(
    engine: Engine,
    table: str,
    df: pd.DataFrame,
    key_columns: Sequence[str],
    chunk_rows: int = CHUNK_ROWS,
) -> int:
    if df.empty:
        return 0
    sql = build_upsert_sql(table, list(df.columns), key_columns)
    rows = dataframe_rows(df)
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        for start in range(0, len(rows), chunk_rows):
            cursor.executemany(sql, rows[start : start + chunk_rows])
            raw.commit()
        cursor.close()
    finally:
        raw.close()
    return len(rows)
