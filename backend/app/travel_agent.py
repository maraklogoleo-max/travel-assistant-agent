import re
import time
from collections.abc import Callable
from contextvars import ContextVar
from datetime import date, datetime, timedelta
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .amap import AmapClient, AmapError
from .config import Settings
from .db import ProfileRepository
from .models import (
    Activity, ConstraintWarning, DayPlan, POI, RouteLeg, TravelProfile, TripPlan,
    TripRequest, WeatherSnapshot,
)

EventEmitter = Callable[[dict[str, Any]], None]
_emitter: ContextVar[EventEmitter | None] = ContextVar("travel_event_emitter", default=None)


def emit(event: dict[str, Any]) -> None:
    callback = _emitter.get()
    if callback:
        callback(event)


def step(name: str, status: str, label: str) -> None:
    emit({"type": "step", "step": name, "status": status, "label": label})


def tool(name: str, status: str, label: str, **extra: Any) -> None:
    emit({"type": "tool_start" if status == "running" else "tool_result", "tool": name, "status": status, "label": label, **extra})


class TravelState(TypedDict, total=False):
    message: str
    trip: dict[str, Any]
    request: dict[str, Any]
    pois: list[dict[str, Any]]
    weather: list[dict[str, Any]]
    days: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    clarification: str | None
    answer: str
    sources: list[dict[str, Any]]
    replan_count: int
    replan_required: bool


class TravelAgent:
    """Multi-skill travel planner. Facts come from Amap; composition is deterministic."""

    def __init__(self, settings: Settings, profiles: ProfileRepository, amap: AmapClient | None = None) -> None:
        self.settings = settings
        self.profiles = profiles
        self.amap = amap or AmapClient(settings.amap_api_key, timeout_seconds=settings.amap_timeout_seconds)
        self.timezone = settings.timezone
        self.graph = self._build_graph()

    def close(self) -> None:
        self.amap.close()

    def _build_graph(self):
        builder = StateGraph(TravelState)
        builder.add_node("collect", self._collect)
        builder.add_node("discover", self._discover)
        builder.add_node("weather", self._weather)
        builder.add_node("routes", self._routes)
        builder.add_node("compose", self._compose)
        builder.add_node("validate", self._validate)
        builder.add_node("replan", self._replan)
        builder.add_node("respond", self._respond)
        builder.add_edge(START, "collect")
        builder.add_conditional_edges("collect", lambda state: "respond" if state.get("clarification") else "discover")
        builder.add_edge("discover", "weather")
        builder.add_edge("weather", "compose")
        builder.add_edge("compose", "routes")
        builder.add_edge("routes", "validate")
        builder.add_conditional_edges(
            "validate",
            lambda state: "replan" if state.get("replan_required") and state.get("replan_count", 0) < 2 else "respond",
        )
        builder.add_edge("replan", "routes")
        builder.add_edge("respond", END)
        return builder.compile()

    def _collect(self, state: TravelState) -> TravelState:
        step("requirements", "running", "理解旅行需求并补齐必要条件")
        text = state["message"]
        current = TripPlan.model_validate(state["trip"])
        old = current.request
        destination = "" if old.destination == "待定目的地" else old.destination
        match = re.search(r"(?:去|到|前往|目的地(?:是|为)?)[\s:：]*([\u4e00-\u9fff]{2,12})", text)
        if match:
            destination = re.split(r"(?:玩|旅游|旅行|[，,。；;\s]|\d+天)", match.group(1))[0]
        origin = old.origin
        match = re.search(r"从([\u4e00-\u9fff]{2,12})到", text)
        if match:
            origin = match.group(1)
        days = old.days
        match = re.search(r"(\d+)\s*(?:天|日)", text)
        if match:
            days = min(7, max(1, int(match.group(1))))
        pace = old.pace
        if re.search(r"轻松|慢节奏|不赶", text):
            pace = "relaxed"
        elif re.search(r"紧凑|多玩|特种兵", text):
            pace = "packed"
        interests = list(dict.fromkeys(old.interests + [key for key in ("自然风景", "博物馆", "亲子", "美食", "古镇", "购物") if key in text]))
        transport = old.transport_mode
        if "自驾" in text or "开车" in text:
            transport = "driving"
        elif "步行" in text:
            transport = "walking"
        request = TripRequest(
            destination=destination, origin=origin, start_date=old.start_date,
            days=days, travelers=old.travelers, budget_level=old.budget_level,
            pace=pace, interests=interests[:10], transport_mode=transport,
            dietary_restrictions=old.dietary_restrictions, special_needs=old.special_needs,
        )
        if not destination:
            return {"request": request.model_dump(mode="json"), "clarification": "你想去哪里旅行？还可以告诉我出发地、日期和游玩天数。"}
        plan = TripPlan.model_validate(state["trip"])
        plan.request = request
        emit({"type": "plan", "task_type": "trip_planning", "title": "正在准备你的行程", "steps": ["整理旅行需求", "查找合适的景点和餐饮", "了解天气和路程", "安排每天的活动", "检查行程是否合理"]})
        step("requirements", "complete", "旅行需求已整理")
        return {"request": request.model_dump(mode="json"), "trip": plan.model_dump(mode="json"), "clarification": None}

    def _discover(self, state: TravelState) -> TravelState:
        request = TripRequest.model_validate(state["request"])
        step("places", "running", "搜索景点、餐饮和备选活动")
        tool("search_places", "running", "正在查找景点和餐饮")
        try:
            locations = self.amap.resolve_location(request.destination)
            if not locations:
                return {"clarification": f"没有找到“{request.destination}”，请补充省份或城市。", "pois": []}
            location = locations[0]
            city = location.adcode
            keywords = request.interests or ["景点", "博物馆", "餐饮"]
            if "博物馆" not in keywords:
                keywords.append("博物馆")
            if "亲子" in state["message"] and "亲子" not in keywords:
                keywords.append("亲子")
            found: list[POI] = []
            for keyword in keywords[:4]:
                found.extend(self.amap.search_places(keyword, city=city))
            unique: dict[str, POI] = {item.id: item for item in found}
            pois = list(unique.values())[:30]
            tool("search_places", "complete", f"高德返回 {len(pois)} 个候选地点", count=len(pois))
            step("places", "complete", "已找到候选地点")
            return {"pois": [item.model_dump(mode="json") for item in pois]}
        except AmapError as exc:
            tool("search_places", "error", str(exc))
            step("places", "error", "景点搜索失败")
            return {"warnings": [ConstraintWarning(type="data", severity="error", message=str(exc)).model_dump()]}

    def _weather(self, state: TravelState) -> TravelState:
        request = TripRequest.model_validate(state["request"])
        step("weather", "running", "查询行程日期天气")
        tool("get_weather", "running", "正在查询行程天气")
        if request.days > 3:
            tool("get_weather", "complete", "超过三天的天气将在临近出发时刷新")
            step("weather", "complete", "已标记待刷新天气")
            return {"weather": [], "warnings": [ConstraintWarning(type="weather", severity="info", message="高德基础预报约覆盖三天，后续日期天气待临近出发时刷新。", suggestion="出发前再次查询天气并调整行程。" ).model_dump()]}
        try:
            location = self.amap.resolve_location(request.destination)[0]
            weather = self.amap.get_forecast_weather(location)
            tool("get_weather", "complete", f"已取得 {len(weather)} 条天气数据")
            step("weather", "complete", "天气数据已取得")
            return {"weather": [item.model_dump(mode="json") for item in weather]}
        except (AmapError, IndexError) as exc:
            tool("get_weather", "error", str(exc))
            return {"weather": [], "warnings": [ConstraintWarning(type="weather", message=str(exc)).model_dump()]}

    def _routes(self, state: TravelState) -> TravelState:
        days = [DayPlan.model_validate(item) for item in state.get("days", [])]
        request = TripRequest.model_validate(state["request"])
        step("routes", "running", "计算活动之间的通勤路线")
        tool("plan_routes", "running", "正在计算活动之间的路程")
        routes: list[RouteLeg] = []
        failures = 0
        for day in days:
            for previous, current in zip(day.activities, day.activities[1:]):
                try:
                    route = self.amap.plan_route(previous.poi, current.poi, request.transport_mode)
                    current.route_from_previous = route
                    routes.append(route)
                except AmapError:
                    failures += 1
        if failures:
            tool("plan_routes", "complete", f"已计算 {len(routes)} 段路线，{failures} 段暂不可用")
        else:
            tool("plan_routes", "complete", f"已计算 {len(routes)} 段路线")
        step("routes", "complete", "路线数据已取得")
        result: TravelState = {
            "routes": [item.model_dump(mode="json") for item in routes],
            "days": [item.model_dump(mode="json") for item in days],
        }
        if failures:
            result["warnings"] = [*state.get("warnings", []), ConstraintWarning(type="route", severity="info", message="部分活动之间暂未取得高德路线耗时，出发前请再次确认。", suggestion="可以切换公交、驾车或减少跨区活动。 ").model_dump()]
        return result

    def _compose(self, state: TravelState) -> TravelState:
        request = TripRequest.model_validate(state["request"])
        pois = [POI.model_validate(item) for item in state.get("pois", [])]
        start = request.start_date or datetime.now().date()
        per_day = {"relaxed": 2, "balanced": 3, "packed": 4}[request.pace]
        existing = TripPlan.model_validate(state["trip"]).days
        update_request = bool(existing and re.search(r"换|调整|改成|减少|增加|放慢|雨天|第?[一二三四五六七\d]天", state["message"]))
        if update_request:
            days = [item.model_copy(deep=True) for item in existing]
            match = re.search(r"第?([一二三四五六七\d])天", state["message"])
            mapping = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
            token = match.group(1) if match else "一"
            target = int(token) if token.isdigit() else mapping.get(token, 1)
            index = min(len(days) - 1, max(0, target - 1))
            day = days[index]
            if re.search(r"放慢|减少", state["message"]):
                day.activities = day.activities[: max(1, len(day.activities) - 1)]
            if "亲子" in state["message"]:
                candidate = next((item for item in pois if re.search(r"亲子|乐园|动物|科技", item.type + item.name)), None)
                if candidate and day.activities:
                    day.activities[0].poi = candidate
                    day.activities[0].reason = "根据亲子偏好局部替换"
            if re.search(r"雨|室内", state["message"]):
                indoor = [item for item in pois if re.search(r"博物馆|展览|室内|商场", item.type + item.name)]
                for activity, candidate in zip([item for item in day.activities if not item.indoor], indoor):
                    activity.poi = candidate
                    activity.indoor = True
                    activity.reason = "根据雨天需求局部替换"
            step("itinerary", "complete", f"已只调整第 {index + 1} 天")
            return {"days": [item.model_dump(mode="json") for item in days]}
        if re.search(r"第?二天.*雨|下雨.*第二天", state["message"]):
            indoor = [item for item in pois if re.search(r"博物馆|展览|室内|商场", item.type + item.name)]
            pois = indoor + [item for item in pois if item not in indoor]
        if "亲子" in state["message"]:
            child = [item for item in pois if re.search(r"亲子|乐园|动物|科技", item.type + item.name)]
            pois = child + [item for item in pois if item not in child]
        days: list[DayPlan] = []
        weather = [WeatherSnapshot.model_validate(item) for item in state.get("weather", [])]
        for day_index in range(request.days):
            day = start + timedelta(days=day_index)
            chosen = pois[day_index * per_day:(day_index + 1) * per_day]
            activities: list[Activity] = []
            for index, poi in enumerate(chosen):
                period = ("morning", "afternoon", "evening")[min(index, 2)]
                indoor = bool(re.search(r"博物馆|展览|商场|餐厅|室内", poi.type + poi.name))
                activities.append(Activity(id=str(uuid4()), date=day, period=period, poi=poi, indoor=indoor, reason="符合你的兴趣与行程节奏"))
            day_weather = next((item for item in weather if item.date == day.isoformat()), None)
            summary = day_weather.weather if day_weather else "天气待临近日期刷新"
            days.append(DayPlan(date=day, weather_summary=summary, activities=activities))
        step("itinerary", "complete", "已生成每日行程")
        return {"days": [item.model_dump(mode="json") for item in days]}

    def _validate(self, state: TravelState) -> TravelState:
        warnings = [ConstraintWarning.model_validate(item) for item in state.get("warnings", []) if item.get("type") != "weather_conflict"]
        days = [DayPlan.model_validate(item) for item in state.get("days", [])]
        replan_required = False
        for day in days:
            if len(day.activities) >= 4:
                warnings.append(ConstraintWarning(type="pace", severity="warning", message=f"{day.date} 安排了较多活动，可能比较紧凑。", suggestion="可以说“放慢节奏”让我减少活动。"))
            if day.weather_summary != "天气待临近日期刷新" and re.search(r"雨|雪|雷", day.weather_summary):
                outdoor = sum(not item.indoor for item in day.activities)
                if outdoor:
                    warnings.append(ConstraintWarning(type="weather_conflict", severity="warning", message=f"{day.date} 有降水，当前仍有 {outdoor} 个室外活动。", suggestion="正在尝试替换为室内活动。"))
                    replan_required = True
        step("validate", "complete", "已完成行程冲突检查")
        return {"warnings": [item.model_dump(mode="json") for item in warnings], "replan_required": replan_required}

    def _replan(self, state: TravelState) -> TravelState:
        attempt = state.get("replan_count", 0) + 1
        step("replan", "running", "正在根据天气优化安排")
        days = [DayPlan.model_validate(item) for item in state.get("days", [])]
        pois = [POI.model_validate(item) for item in state.get("pois", [])]
        indoor = [item for item in pois if re.search(r"博物馆|展览|室内|商场", item.type + item.name)]
        used = {activity.poi.id for day in days for activity in day.activities}
        candidates = [item for item in indoor if item.id not in used] or indoor
        changed = 0
        for day in days:
            if not re.search(r"雨|雪|雷", day.weather_summary):
                continue
            for activity in day.activities:
                if activity.indoor or not candidates:
                    continue
                replacement = candidates.pop(0)
                old_name = activity.poi.name
                activity.poi = replacement
                activity.indoor = True
                activity.reason = f"考虑到{day.weather_summary}，将{old_name}换成室内活动"
                changed += 1
        step("replan", "complete", f"已调整 {changed} 个受天气影响的活动" if changed else "暂时没有找到合适的室内替代")
        return {
            "days": [item.model_dump(mode="json") for item in days],
            "warnings": [item for item in state.get("warnings", []) if item.get("type") != "weather_conflict"],
            "replan_count": attempt,
            "replan_required": bool(changed),
        }

    def _respond(self, state: TravelState) -> TravelState:
        if state.get("clarification"):
            answer = state["clarification"]
            return {"answer": answer}
        request = TripRequest.model_validate(state["request"])
        days = [DayPlan.model_validate(item) for item in state.get("days", [])]
        warnings = [ConstraintWarning.model_validate(item) for item in state.get("warnings", [])]
        pace_label = {"relaxed": "轻松", "balanced": "适中", "packed": "充实"}[request.pace]
        period_label = {"morning": "上午", "afternoon": "下午", "evening": "晚上"}
        lines = [f"我为你安排了一份 {request.destination} {request.days} 天行程，整体节奏{pace_label}："]
        for day in days:
            lines.append(f"\n{day.date}｜天气：{day.weather_summary}")
            for activity in day.activities:
                route = f"（从上一站过来约 {activity.route_from_previous.duration_s // 60} 分钟）" if activity.route_from_previous and activity.route_from_previous.duration_s else ""
                lines.append(f"- {period_label[activity.period]}：{activity.poi.name}{route}。{activity.reason}")
        lines.append("\n如果想修改，可以直接说“第二天安排室内活动”“换成亲子景点”或“行程轻松一点”。")
        answer = "\n".join(lines)
        for index in range(0, len(answer), 8):
            emit({"type": "token", "delta": answer[index:index + 8]})
            time.sleep(0.012)
        emit({"type": "itinerary_patch", "days": [item.model_dump(mode="json") for item in days], "warnings": [item.model_dump(mode="json") for item in warnings]})
        step("answer", "complete", "行程建议已生成")
        return {"answer": answer}

    def run(
        self, trip_id: str, message: str, emit_callback: EventEmitter | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        token = _emitter.set(emit_callback)
        try:
            trip = self.profiles.get_trip(trip_id)
            if not trip:
                raise ValueError("旅行项目不存在")
            result = self.graph.invoke({"message": message, "trip": trip.model_dump(mode="json"), "request": trip.request.model_dump(mode="json"), "pois": [], "weather": [], "days": [], "warnings": [], "replan_count": 0, "replan_required": False})
            updated = trip.model_copy(update={
                "request": TripRequest.model_validate(result.get("request", trip.request.model_dump(mode="json"))),
                "days": [DayPlan.model_validate(item) for item in result.get("days", [])] or trip.days,
                "warnings": [ConstraintWarning.model_validate(item) for item in result.get("warnings", [])],
                "status": "needs_input" if result.get("clarification") else "ready",
            })
            if not result.get("clarification"):
                self.profiles.save_trip(updated, expected_version=expected_version)
            return {"answer": result.get("answer", ""), "trip": updated.model_dump(mode="json"), "sources": []}
        finally:
            _emitter.reset(token)
