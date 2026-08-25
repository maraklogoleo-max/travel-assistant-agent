from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..amap import AmapClient, AmapError
from ..models import POI, ResolvedLocation, SourceRecord, ToolResult


class TravelToolRegistry:
    """Typed boundary around external travel facts.

    The agent can decide which tool to use, while this registry enforces that
    weather, place and route claims originate from Amap responses.
    """

    def __init__(self, amap: AmapClient, timezone: str = "Asia/Shanghai") -> None:
        self.amap = amap
        self.timezone = ZoneInfo(timezone)

    def _now(self) -> str:
        return datetime.now(self.timezone).isoformat(timespec="seconds")

    def resolve_location(self, query: str) -> ToolResult:
        try:
            candidates = self.amap.resolve_location(query)
        except AmapError as exc:
            return ToolResult(tool="resolve_location", success=False, error_code=exc.code, user_message=str(exc), retryable=True)
        if not candidates:
            return ToolResult(tool="resolve_location", success=False, error_code="LOCATION_NOT_FOUND", user_message=f"没有找到“{query}”，请补充省份或城市。")
        sources = [
            SourceRecord(
                kind="地点", location=item.display_name, reporttime=self._now(),
                query_time=self._now(), resource_id=item.adcode, detail=item.query,
            )
            for item in candidates
        ]
        if len(candidates) > 1:
            return ToolResult(
                tool="resolve_location", success=False,
                data=[item.model_dump(mode="json") for item in candidates], sources=sources,
                error_code="LOCATION_AMBIGUOUS", user_message=f"“{query}”对应多个地点，请先选择具体地区。",
            )
        return ToolResult(tool="resolve_location", success=True, data=candidates[0].model_dump(mode="json"), sources=sources)

    def weather(self, location: ResolvedLocation, *, forecast: bool = True) -> ToolResult:
        try:
            snapshots = self.amap.get_forecast_weather(location) if forecast else self.amap.get_current_weather(location)
        except AmapError as exc:
            return ToolResult(tool="weather", success=False, error_code=exc.code, user_message=str(exc), retryable=True)
        kind = "预报" if forecast else "实时"
        sources = [
            SourceRecord(
                kind=kind, location=item.location.display_name, reporttime=item.reporttime,
                query_time=self._now(), resource_id=item.location.adcode,
                detail=item.date or item.weather,
            )
            for item in snapshots
        ]
        return ToolResult(tool="weather", success=True, data=[item.model_dump(mode="json") for item in snapshots], sources=sources)

    def places(self, keywords: str, *, city: str) -> ToolResult:
        try:
            places = self.amap.search_places(keywords, city=city)
        except AmapError as exc:
            return ToolResult(tool="places", success=False, error_code=exc.code, user_message=str(exc), retryable=True)
        sources = [
            SourceRecord(
                kind="地点", location=item.name, reporttime=self._now(), query_time=self._now(),
                resource_id=item.id, detail=item.address or item.type,
            )
            for item in places
        ]
        return ToolResult(tool="places", success=True, data=[item.model_dump(mode="json") for item in places], sources=sources)

    def route(self, origin: POI, destination: POI, mode: str) -> ToolResult:
        try:
            route = self.amap.plan_route(origin, destination, mode)
        except AmapError as exc:
            return ToolResult(tool="routes", success=False, error_code=exc.code, user_message=str(exc), retryable=True)
        route.query_time = self._now()
        source = SourceRecord(
            kind="路线", location=f"{origin.name} → {destination.name}", reporttime=route.query_time,
            query_time=route.query_time, resource_id=f"{origin.id}:{destination.id}",
            detail=f"{route.mode} · {route.distance_m or 0}米 · {route.duration_s or 0}秒",
        )
        return ToolResult(tool="routes", success=True, data=route.model_dump(mode="json"), sources=[source])

    def execute(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        if tool == "resolve_location":
            return self.resolve_location(str(arguments.get("query", "")))
        return ToolResult(tool=tool, success=False, error_code="TOOL_NOT_SUPPORTED", user_message=f"不支持的工具：{tool}")
