from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.date_parser import parse_target_day, parse_trip_dates


def test_date_parser_understands_relative_date_and_days() -> None:
    parsed = parse_trip_dates("明天去杭州玩三天")
    assert parsed.start_date == datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)
    assert parsed.days == 3
    assert parsed.explicit is True


def test_target_day_parser_supports_followup() -> None:
    assert parse_target_day("第二天呢") == 2
    assert parse_target_day("只调整第3天") == 3


def test_target_day_is_not_mistaken_for_trip_length() -> None:
    parsed = parse_trip_dates("把第二天换成亲子景点")
    assert parsed.days is None
