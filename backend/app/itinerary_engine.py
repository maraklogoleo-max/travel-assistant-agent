import math
import re
from datetime import date, timedelta
from typing import ClassVar
from uuid import uuid4

from .models import (
    POI,
    Activity,
    ConstraintWarning,
    DayPlan,
    SourceRecord,
    TripRequest,
    ValidationResult,
    WeatherSnapshot,
)


class ItineraryEngine:
    _slots: ClassVar[dict[str, list[tuple[str, str, str]]]] = {
        "relaxed": [("morning", "09:30", "11:30"), ("afternoon", "14:00", "16:30")],
        "balanced": [("morning", "09:00", "11:30"), ("afternoon", "13:30", "16:00"), ("evening", "18:00", "20:00")],
        "packed": [("morning", "08:30", "10:30"), ("morning", "10:45", "12:15"), ("afternoon", "14:00", "16:00"), ("evening", "18:00", "20:00")],
    }

    @staticmethod
    def _distance(a: POI, b: POI) -> float:
        if None in (a.longitude, a.latitude, b.longitude, b.latitude):
            return float("inf")
        return math.hypot(float(a.longitude) - float(b.longitude), float(a.latitude) - float(b.latitude))

    @staticmethod
    def _score(poi: POI, request: TripRequest) -> int:
        haystack = f"{poi.name} {poi.type} {poi.address}"
        score = sum(8 for interest in request.interests if interest and interest in haystack)
        if re.search(r"风景|景区|博物馆|公园|古镇|乐园|文化", haystack):
            score += 3
        if not poi.location or poi.longitude is None or poi.latitude is None:
            score -= 20
        return score

    def _cluster(self, pois: list[POI], request: TripRequest) -> list[POI]:
        remaining = sorted(pois, key=lambda item: self._score(item, request), reverse=True)
        ordered: list[POI] = []
        while remaining:
            current = remaining.pop(0) if not ordered else min(
                remaining, key=lambda item: (self._distance(ordered[-1], item), -self._score(item, request))
            )
            if current in remaining:
                remaining.remove(current)
            ordered.append(current)
        return ordered

    def build(
        self, request: TripRequest, pois: list[POI], weather: list[WeatherSnapshot], start: date,
    ) -> tuple[list[DayPlan], list[ConstraintWarning]]:
        unique: dict[str, POI] = {}
        for poi in pois:
            if poi.id and poi.id not in unique and poi.longitude is not None and poi.latitude is not None:
                unique[poi.id] = poi
        ordered = self._cluster(list(unique.values()), request)
        slots = self._slots[request.pace]
        warnings: list[ConstraintWarning] = []
        days: list[DayPlan] = []

        for day_index in range(request.days):
            day_date = start + timedelta(days=day_index)
            chosen = ordered[day_index * len(slots):(day_index + 1) * len(slots)]
            activities: list[Activity] = []
            for poi, (period, start_time, end_time) in zip(chosen, slots):
                indoor = bool(re.search(r"博物馆|展览|商场|餐厅|室内|科技馆", poi.type + poi.name))
                source = SourceRecord(
                    kind="地点", location=poi.name, reporttime="高德查询结果",
                    resource_id=poi.id, detail=poi.address or poi.type,
                )
                activities.append(Activity(
                    id=str(uuid4()), date=day_date, period=period, poi=poi,
                    start_time=start_time, end_time=end_time,
                    duration_minutes=round((int(end_time[:2]) * 60 + int(end_time[3:]) - int(start_time[:2]) * 60 - int(start_time[3:])) / 30) * 30,
                    indoor=indoor, reason="符合你的兴趣、行程节奏和相邻区域安排", sources=[source],
                ))
            snapshot = next((item for item in weather if item.date == day_date.isoformat()), None)
            weather_sources = []
            if snapshot:
                weather_sources.append(SourceRecord(
                    kind="预报", location=snapshot.location.display_name,
                    reporttime=snapshot.reporttime, resource_id=snapshot.location.adcode,
                    detail=snapshot.date or snapshot.weather,
                ))
            days.append(DayPlan(
                date=day_date, weather_summary=snapshot.weather if snapshot else "天气待临近日期刷新",
                activities=activities, sources=weather_sources,
            ))

        missing_days = [index + 1 for index, item in enumerate(days) if not item.activities]
        if missing_days:
            warnings.append(ConstraintWarning(
                type="insufficient_places", severity="warning",
                message=f"第 {'、'.join(map(str, missing_days))} 天缺少已验证的候选地点。",
                suggestion="可以补充兴趣或缩小目的地区域后继续搜索。",
            ))
        return days, warnings

    def validate(self, days: list[DayPlan]) -> ValidationResult:
        warnings: list[ConstraintWarning] = []
        affected: list[int] = []
        for day_index, day in enumerate(days, start=1):
            seen: set[str] = set()
            for activity in day.activities:
                if activity.poi.id in seen:
                    warnings.append(ConstraintWarning(type="duplicate", message=f"{day.date} 出现重复活动：{activity.poi.name}"))
                    affected.append(day_index)
                seen.add(activity.poi.id)
            if len(day.activities) >= 4:
                warnings.append(ConstraintWarning(type="pace", message=f"{day.date} 活动较多，节奏可能偏紧。", suggestion="可以减少一个活动。"))
            if re.search(r"雨|雪|雷", day.weather_summary) and any(not item.indoor for item in day.activities):
                warnings.append(ConstraintWarning(type="weather_conflict", message=f"{day.date} 有降水且包含户外活动。", suggestion="可以替换为室内活动。"))
                affected.append(day_index)
            for previous, current in zip(day.activities, day.activities[1:]):
                if (
                    current.route_from_previous
                    and current.route_from_previous.duration_s
                    and current.route_from_previous.duration_s > 2 * 60 * 60
                ):
                    warnings.append(ConstraintWarning(type="long_route", message=f"{previous.poi.name}到{current.poi.name}通勤超过两小时。", suggestion="建议换成同一区域活动。"))
                    affected.append(day_index)
        return ValidationResult(valid=not any(item.severity == "error" for item in warnings), warnings=warnings, retryable=bool(affected), affected_days=sorted(set(affected)))
