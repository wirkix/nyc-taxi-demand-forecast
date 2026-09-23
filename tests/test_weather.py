import datetime as dt
import json
from pathlib import Path

import responses

from extract import weather

FIXTURE = json.loads(Path(__file__).with_name("fixtures").joinpath("open_meteo.json").read_text())


def test_parse_response_renames_and_drops_lagging_null_hours():
    df = weather.parse_response(FIXTURE)
    assert list(df.columns) == ["obs_hour", *weather.VARIABLES.values()]
    # Fixture has 4 hours, the last one all-null (inside the archive lag).
    assert len(df) == 3
    assert df["obs_hour"].iloc[0].isoformat() == "2026-07-01T00:00:00"
    assert df["precipitation_mm"].tolist() == [0.0, 1.2, 0.4]


@responses.activate
def test_fetch_hourly_requests_local_time_and_all_variables():
    responses.add(responses.GET, weather.ARCHIVE_URL, json=FIXTURE)
    df = weather.fetch_hourly(dt.date(2026, 7, 1), dt.date(2026, 7, 1))
    params = responses.calls[0].request.params
    assert params["timezone"] == "America/New_York"
    assert params["hourly"].split(",") == list(weather.VARIABLES)
    assert len(df) == 3
