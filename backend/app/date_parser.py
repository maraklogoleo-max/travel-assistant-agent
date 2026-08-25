import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ParsedTripDates:
    start_date: date | None = None
    days: int | None = None
    explicit: bool = False


_DAY_MAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}


def parse_day_count(text: str) -> int | None:
    # “第二天” identifies a target day, not a request to shorten the trip to
    # two days. Exclude the day ordinal prefix (with or without a space).
    match = re.search(r"(?<!第)(?<!第\s)(\d+|[一二三四五六七])\s*(?:天|日)", text)
    if not match:
        return None
    token = match.group(1)
    value = int(token) if token.isdigit() else _DAY_MAP[token]
    return min(7, max(1, value))


def parse_target_day(text: str) -> int | None:
    match = re.search(r"第?([一二三四五六七\d])天", text)
    if not match:
        return None
    token = match.group(1)
    value = int(token) if token.isdigit() else _DAY_MAP.get(token)
    return value if value and 1 <= value <= 7 else None


def parse_trip_dates(text: str, timezone: str = "Asia/Shanghai") -> ParsedTripDates:
    today = datetime.now(ZoneInfo(timezone)).date()
    days = parse_day_count(text)
    explicit = False
    start: date | None = None

    iso = re.search(r"(?<!\d)(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", text)
    if iso:
        try:
            start = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            explicit = True
        except ValueError:
            start = None

    if start is None:
        month_day = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})(?:日|号)?", text)
        if month_day:
            try:
                candidate = date(today.year, int(month_day.group(1)), int(month_day.group(2)))
                if candidate < today:
                    candidate = date(today.year + 1, candidate.month, candidate.day)
                start = candidate
                explicit = True
            except ValueError:
                start = None

    if start is None:
        relative = (("大后天", 3), ("后天", 2), ("明天", 1), ("今天", 0))
        for keyword, offset in relative:
            if keyword in text:
                start = today + timedelta(days=offset)
                explicit = True
                break

    if start is None and re.search(r"(?:本周|这个)?周末", text):
        offset = (5 - today.weekday()) % 7
        start = today + timedelta(days=offset)
        days = days or 2
        explicit = True

    weekday_match = re.search(r"下周([一二三四五六日天])", text)
    if start is None and weekday_match:
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        target = weekday_map[weekday_match.group(1)]
        start = today + timedelta(days=7 - today.weekday() + target)
        explicit = True

    return ParsedTripDates(start_date=start, days=days, explicit=explicit)
