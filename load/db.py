"""MySQL connection (Aiven in prod, docker-compose.yml locally)."""

from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy as sa
from dotenv import load_dotenv
from sqlalchemy.engine import URL, Engine

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_engine() -> Engine:
    # .env for local runs; in GitHub Actions the vars come from secrets and
    # load_dotenv never overrides an already-set variable.
    load_dotenv()
    url = URL.create(
        "mysql+pymysql",
        username=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
    )
    connect_args: dict = {}
    # Aiven only accepts TLS connections; its CA cert is downloaded from the
    # service's Overview page. Unset locally (docker MySQL has no TLS setup).
    ca_path = os.environ.get("MYSQL_SSL_CA")
    if ca_path:
        connect_args["ssl"] = {"ca": ca_path}
    # pool_pre_ping: a connection left idle across a long download/aggregate
    # step may have been dropped server-side -- reconnect, don't fail mid-run.
    return sa.create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def split_statements(sql: str) -> list[str]:
    """Split schema.sql on ';' (it has no procedures or string literals)."""
    lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


def apply_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for statement in split_statements(SCHEMA_PATH.read_text(encoding="utf-8")):
            conn.exec_driver_sql(statement)
