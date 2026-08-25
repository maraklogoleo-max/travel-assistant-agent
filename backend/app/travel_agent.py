import json
import re
import time
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .amap import AmapClient, AmapError
from .config import Settings
from .date_parser import parse_target_day, parse_trip_dates
from .db import ProfileRepository
from .itinerary_engine import ItineraryEngine
from .models import (
    POI,
    ConstraintWarning,
    DayPlan,
    ResolvedLocation,
    RouteLeg,
    SourceRecord,
    TripPlan,
    TripRequest,
    WeatherSnapshot,
)
from .tools import TravelToolRegistry

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
    routes: list[dict[str, Any]]
    updated_day: int
    update_notice: str
    keep_outdoor: bool
    outdoor_action: str
    outdoor_change_count: int
    no_trip_change: bool


class TravelAgent:
    """Multi-skill travel planner. Facts come from Amap; composition is deterministic."""

    def __init__(self, settings: Settings, profiles: ProfileRepository, amap: AmapClient | None = None) -> None:
        self.settings = settings
        self.profiles = profiles
        self.amap = amap or AmapClient(settings.amap_api_key, timeout_seconds=settings.amap_timeout_seconds)
        self.timezone = settings.timezone
        self.tools = TravelToolRegistry(self.amap, settings.timezone)
        self.itinerary = ItineraryEngine()
        self.llm: ChatOpenAI | None = None
        if settings.deepseek_api_key:
            self.llm = ChatOpenAI(
                api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
                model=settings.deepseek_model, temperature=0.35, timeout=25, max_retries=1,
            )
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
        parsed_dates = parse_trip_dates(text, self.timezone)
        days = parsed_dates.days or old.days
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
            destination=destination, origin=origin, start_date=parsed_dates.start_date or old.start_date,
            days=days, travelers=old.travelers, budget_level=old.budget_level,
            pace=pace, interests=interests[:10], transport_mode=transport,
            transport_preference=("高铁" if "高铁" in text else old.transport_preference),
            accommodation_preference=("民宿" if "民宿" in text else old.accommodation_preference),
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
        state_sources = list(state.get("sources", []))
        step("places", "running", "搜索景点、餐饮和备选活动")
        tool("search_places", "running", "正在查找景点和餐饮")
        try:
            resolved = self.tools.resolve_location(request.destination)
            if not resolved.success:
                if resolved.error_code == "LOCATION_AMBIGUOUS":
                    candidates = [item.get("province", "") + item.get("city", "") + item.get("district", "") for item in (resolved.data or [])]
                    choices = "、".join(item for item in candidates if item)
                    return {"clarification": f"“{request.destination}”对应多个地点：{choices}。请告诉我具体地区。", "pois": [], "sources": [item.model_dump(mode="json") for item in resolved.sources]}
                return {"clarification": resolved.user_message or f"没有找到“{request.destination}”。", "pois": [], "sources": [item.model_dump(mode="json") for item in resolved.sources]}
            location = ResolvedLocation.model_validate(resolved.data)
            city = location.adcode
            keywords = request.interests or ["景点", "博物馆", "餐饮"]
            if "亲子" in state["message"]:
                keywords = ["亲子乐园", "动物园", "科技馆", *keywords]
            if re.search(r"户外|室外|不怕雨|不介意下雨|坚持户外", state["message"]):
                keywords = ["景点", "公园", "自然风景", *keywords]
            if "博物馆" not in keywords:
                keywords.append("博物馆")
            found: list[POI] = []
            for keyword in keywords[:4]:
                result = self.tools.places(keyword, city=city)
                if result.success:
                    found.extend(POI.model_validate(item) for item in result.data)
                    state_sources.extend(item.model_dump(mode="json") for item in result.sources)
            # Search results commonly include ticket booths, visitor centres
            # and transport facilities beside a scenic spot.  They are useful
            # map results but poor primary travel activities.
            service_poi = re.compile(r"售票|检票|游客中心|服务中心|停车|入口|出口|乘车|观光车|警务|医务|厕所|卫生间|站点?$")
            food_poi = re.compile(r"麦当劳|肯德基|快餐|餐饮|餐厅|饭店|咖啡|奶茶|小吃|烧烤|火锅|美食")
            include_food = any(item in request.interests for item in ("美食", "餐饮"))
            found = [
                item for item in found
                if not service_poi.search(item.name)
                and (include_food or not food_poi.search(f"{item.name} {item.type}"))
            ]
            unique: dict[str, POI] = {item.id: item for item in found}
            pois = list(unique.values())[:30]
            tool("search_places", "complete", f"高德返回 {len(pois)} 个候选地点", count=len(pois))
            step("places", "complete", "已找到候选地点")
            return {"pois": [item.model_dump(mode="json") for item in pois], "sources": state_sources}
        except AmapError as exc:
            tool("search_places", "error", str(exc))
            step("places", "error", "景点搜索失败")
            return {"warnings": [ConstraintWarning(type="data", severity="error", message=str(exc)).model_dump()]}

    def _weather(self, state: TravelState) -> TravelState:
        request = TripRequest.model_validate(state["request"])
        sources = list(state.get("sources", []))
        warnings = list(state.get("warnings", []))
        step("weather", "running", "查询行程日期天气")
        tool("get_weather", "running", "正在查询行程天气")
        resolved = self.tools.resolve_location(request.destination)
        if not resolved.success:
            tool("get_weather", "error", resolved.user_message or "地点解析失败")
            warnings.append(ConstraintWarning(type="weather", message=resolved.user_message or "地点解析失败").model_dump())
            return {"weather": [], "warnings": warnings, "sources": sources}
        result = self.tools.weather(ResolvedLocation.model_validate(resolved.data), forecast=True)
        if not result.success:
            tool("get_weather", "error", result.user_message or "天气查询失败")
            warnings.append(ConstraintWarning(type="weather", message=result.user_message or "天气查询失败").model_dump())
            return {"weather": [], "warnings": warnings, "sources": sources}
        weather = [WeatherSnapshot.model_validate(item) for item in result.data]
        sources.extend(item.model_dump(mode="json") for item in result.sources)
        if request.days > len(weather):
            warnings.append(ConstraintWarning(
                type="weather", severity="info",
                message=f"当前取得 {len(weather)} 天高德预报，其余日期暂不填入天气结论。",
            ).model_dump())
        tool("get_weather", "complete", f"已取得 {len(weather)} 条天气数据")
        step("weather", "complete", "天气数据已取得")
        return {"weather": [item.model_dump(mode="json") for item in weather], "warnings": warnings, "sources": sources}

    def _routes(self, state: TravelState) -> TravelState:
        days = [DayPlan.model_validate(item) for item in state.get("days", [])]
        request = TripRequest.model_validate(state["request"])
        step("routes", "running", "计算活动之间的通勤路线")
        tool("plan_routes", "running", "正在计算活动之间的路程")
        routes: list[RouteLeg] = []
        sources = list(state.get("sources", []))
        failures = 0
        for day in days:
            for previous, current in zip(day.activities, day.activities[1:]):
                result = self.tools.route(previous.poi, current.poi, request.transport_mode)
                if result.success:
                    route = RouteLeg.model_validate(result.data)
                    current.route_from_previous = route
                    routes.append(route)
                    current.sources.extend(result.sources)
                    sources.extend(item.model_dump(mode="json") for item in result.sources)
                else:
                    failures += 1
        if failures:
            tool("plan_routes", "complete", f"已计算 {len(routes)} 段路线，{failures} 段暂不可用")
        else:
            tool("plan_routes", "complete", f"已计算 {len(routes)} 段路线")
        step("routes", "complete", "路线数据已取得")
        result: TravelState = {
            "routes": [item.model_dump(mode="json") for item in routes],
            "days": [item.model_dump(mode="json") for item in days],
            "sources": sources,
        }
        if failures:
            result["warnings"] = [*state.get("warnings", []), ConstraintWarning(type="route", severity="info", message="部分活动之间暂未取得高德路线耗时，因此没有写入通勤结论。", suggestion="可以切换交通方式或减少跨区活动。").model_dump()]
        return result

    def _compose(self, state: TravelState) -> TravelState:
        request = TripRequest.model_validate(state["request"])
        pois = [POI.model_validate(item) for item in state.get("pois", [])]
        start = request.start_date or datetime.now(ZoneInfo(self.timezone)).date()
        existing = TripPlan.model_validate(state["trip"]).days
        update_request = bool(existing and re.search(
            r"换|调整|改成|减少|增加|放慢|雨天|户外|室外|多一些|更多|无所谓|不怕雨|不介意下雨|坚持户外|第?[一二三四五六七\d]天",
            state["message"],
        ))
        if update_request:
            days = [item.model_copy(deep=True) for item in existing]
            switch_to_outdoor = bool(re.search(
                r"(?:室内|博物馆).{0,12}(?:换成|改成|改为|替换(?:成)?).{0,12}(?:户外|室外)|"
                r"(?:换成|改成|改为|替换(?:成)?).{0,8}(?:户外|室外)",
                state["message"],
            ))
            keep_outdoor = bool(re.search(
                r"(?:下雨|雨天|降雨).{0,10}(?:也要|仍要|还要|照样|无所谓|没关系|不影响).{0,8}(?:户外|室外)|"
                r"(?:户外|室外).{0,10}(?:也要|仍要|还要|照样|多一些|不减少)|"
                r"(?:多一些|更多).{0,6}(?:户外|室外)|(?:不怕雨|不介意下雨|坚持户外)",
                state["message"],
            ))
            if keep_outdoor or switch_to_outdoor:
                candidates = [
                    item for item in pois
                    if re.search(r"风景|公园|湖|森林|自然|景区|山|瀑布|海|沟|景点", item.type + item.name)
                    and not re.search(r"博物馆|展览|室内|商场|酒店|餐饮|检票|游客中心|服务|售票|乘车|广场|警务|医务|停车|入口|出口", item.type + item.name)
                ]
                used = {activity.poi.id for day in days for activity in day.activities if not activity.indoor}
                changed = 0
                for day in days:
                    if keep_outdoor and not switch_to_outdoor and not re.search(r"雨|雪|雷", day.weather_summary):
                        continue
                    for activity in day.activities:
                        if not activity.indoor:
                            continue
                        candidate = next((item for item in candidates if item.id not in used), None)
                        if not candidate:
                            break
                        activity.poi = candidate
                        activity.indoor = False
                        activity.reason = "按你的要求保留户外活动，即使有降雨也不自动改为室内"
                        activity.sources = []
                        used.add(candidate.id)
                        changed += 1
                if switch_to_outdoor:
                    label = f"已将 {changed} 个室内活动换为户外活动" if changed else "没有找到可替换的室内活动"
                    notice = "已按你的要求将室内活动换为高德返回的户外地点；雨天也不会自动改回室内。" if changed else "当前没有可替换的室内活动，原行程保持不变。"
                else:
                    label = f"已按要求保留 {changed} 个户外活动" if changed else "已确认现有活动保持户外"
                    notice = "雨天也按你的要求保留户外活动；请带好雨具并避开雨势最强时段。"
                step("itinerary", "complete", label)
                return {
                    "days": [item.model_dump(mode="json") for item in days],
                    "updated_day": 0,
                    "update_notice": notice,
                    "keep_outdoor": True,
                    "outdoor_action": "switch" if switch_to_outdoor else "keep",
                    "outdoor_change_count": changed,
                    "no_trip_change": changed == 0,
                }
            target = parse_target_day(state["message"]) or 1
            index = min(len(days) - 1, max(0, target - 1))
            day = days[index]
            update_notice = ""
            if re.search(r"放慢|减少", state["message"]):
                day.activities = day.activities[: max(1, len(day.activities) - 1)]
            if "亲子" in state["message"]:
                candidate = next((item for item in pois if re.search(r"亲子|乐园|动物|科技", item.type + item.name)), None)
                if candidate and day.activities:
                    day.activities[0].poi = candidate
                    day.activities[0].reason = "根据亲子偏好局部替换"
                elif not candidate:
                    update_notice = "高德暂未返回明确的亲子景点，因此这一天暂时保留原安排。"
            if re.search(r"雨|室内", state["message"]):
                indoor = [item for item in pois if re.search(r"博物馆|展览|室内|商场", item.type + item.name)]
                for activity, candidate in zip([item for item in day.activities if not item.indoor], indoor):
                    activity.poi = candidate
                    activity.indoor = True
                    activity.reason = "根据雨天需求局部替换"
            step("itinerary", "complete", f"已只调整第 {index + 1} 天")
            return {
                "days": [item.model_dump(mode="json") for item in days],
                "updated_day": index + 1,
                "update_notice": update_notice,
            }
        if re.search(r"第?二天.*雨|下雨.*第二天", state["message"]):
            indoor = [item for item in pois if re.search(r"博物馆|展览|室内|商场", item.type + item.name)]
            pois = indoor + [item for item in pois if item not in indoor]
        if "亲子" in state["message"]:
            child = [item for item in pois if re.search(r"亲子|乐园|动物|科技", item.type + item.name)]
            pois = child + [item for item in pois if item not in child]
        weather = [WeatherSnapshot.model_validate(item) for item in state.get("weather", [])]
        days, engine_warnings = self.itinerary.build(request, pois, weather, start)
        source_by_id = {
            item.resource_id: item
            for item in (SourceRecord.model_validate(raw) for raw in state.get("sources", []))
            if item.kind == "地点" and item.resource_id
        }
        for day in days:
            for activity in day.activities:
                if activity.poi.id in source_by_id:
                    activity.sources = [source_by_id[activity.poi.id]]
        step("itinerary", "complete", "已生成每日行程")
        return {
            "days": [item.model_dump(mode="json") for item in days],
            "warnings": [*state.get("warnings", []), *[item.model_dump(mode="json") for item in engine_warnings]],
        }

    def _validate(self, state: TravelState) -> TravelState:
        warnings = [ConstraintWarning.model_validate(item) for item in state.get("warnings", []) if item.get("type") != "weather_conflict"]
        days = [DayPlan.model_validate(item) for item in state.get("days", [])]
        validation = self.itinerary.validate(days)
        if not state.get("keep_outdoor", False):
            warnings.extend(validation.warnings)
        replan_required = not state.get("keep_outdoor", False) and any(item.type == "weather_conflict" for item in validation.warnings)
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
        updated_day = state.get("updated_day")
        update_notice = state.get("update_notice")
        if updated_day == 0:
            outdoor_action = state.get("outdoor_action")
            outdoor_change_count = state.get("outdoor_change_count", 0)
            if outdoor_change_count:
                lines = ["已按你的要求调整行程，雨天也保留户外活动："]
            elif outdoor_action == "switch":
                lines = ["当前没有需要替换的室内活动，行程保持不变："]
            else:
                lines = ["当前行程已经保留户外活动："]
            display_days = days
        elif updated_day:
            lines = [f"已调整第 {updated_day} 天，其他日期保持不变："]
            display_days = [days[updated_day - 1]] if updated_day <= len(days) else []
        else:
            lines = [f"好呀，这趟 {request.destination} {request.days} 天行程已经为你排好，整体会保持{pace_label}的节奏。"]
            if request.origin:
                transport_label = {"transit": "公共交通", "driving": "自驾", "walking": "步行"}[request.transport_mode]
                transport_detail = request.transport_preference or transport_label
                stay_detail = f"，住宿按{request.accommodation_preference}偏好考虑" if request.accommodation_preference else ""
                lines.append(f"从{request.origin}出发，长途交通优先按{transport_detail}考虑，抵达后以{transport_label}衔接{stay_detail}；不用把每天排得太满，把时间留给沿途的风景。")
            display_days = days
        themes = (
            "第一天先把脚步放慢，熟悉环境，也给旅途留一点余裕。",
            "这一天把重心放在体验上，慢慢走、慢慢看会更舒服。",
            "最后一天安排得从容一些，让这段旅程好好收尾。",
        )
        previous_weather = None
        for day_index, day in enumerate(display_days, start=1):
            weather_label = day.weather_summary if day.weather_summary != previous_weather else "天气与上一天相近"
            lines.append(f"\n第 {day_index if not updated_day else updated_day} 天 · {day.date}｜天气：{weather_label}")
            previous_weather = day.weather_summary
            if not updated_day:
                lines.append(themes[min(day_index - 1, len(themes) - 1)])
            if not day.activities:
                lines.append("- 暂未找到适合的已验证活动。")
            for activity in day.activities:
                route = f"（从上一站过来约 {activity.route_from_previous.duration_s // 60} 分钟）" if activity.route_from_previous and activity.route_from_previous.duration_s else ""
                time_range = f"{activity.start_time}–{activity.end_time} · " if activity.start_time and activity.end_time else ""
                lines.append(f"- {period_label[activity.period]}：{time_range}{activity.poi.name}{route}")
        if update_notice:
            lines.append(f"\n{update_notice}")
        answer = "\n".join(lines)
        narrated = self._llm_narrative(state, answer)
        if narrated:
            answer = narrated
        for index in range(0, len(answer), 8):
            emit({"type": "token", "delta": answer[index:index + 8]})
            time.sleep(0.012)
        emit({"type": "itinerary_patch", "days": [item.model_dump(mode="json") for item in days], "warnings": [item.model_dump(mode="json") for item in warnings]})
        step("answer", "complete", "行程建议已生成")
        return {"answer": answer}

    def _llm_narrative(self, state: TravelState, fallback: str) -> str | None:
        """Explain verified itinerary facts naturally without inventing facts."""
        if not self.llm or state.get("clarification"):
            return None
        request = TripRequest.model_validate(state["request"])
        days = [DayPlan.model_validate(item) for item in state.get("days", [])]
        payload = {
            "request": {
                "destination": request.destination, "origin": request.origin,
                "days": request.days, "pace": request.pace, "interests": request.interests,
                "transport": request.transport_preference or request.transport_mode,
                "accommodation": request.accommodation_preference,
            },
            "days": [
                {
                    "date": item.date.isoformat(), "weather": item.weather_summary,
                    "activities": [
                        {
                            "period": activity.period,
                            "time": f"{activity.start_time or ''}-{activity.end_time or ''}",
                            "name": activity.poi.name, "indoor": activity.indoor,
                            "route_minutes": round(activity.route_from_previous.duration_s / 60)
                            if activity.route_from_previous and activity.route_from_previous.duration_s else None,
                        }
                        for activity in item.activities
                    ],
                }
                for item in days
            ],
        }
        prompt = (
            "你是一个有判断力、有温度的中文旅行助手。请根据已验证的行程 JSON 写最终答复。"
            "只能使用 JSON 中出现的地点、日期、天气、时间和路线事实，不能补造景点、价格、营业时间或交通耗时。"
            "先用一两句说明整体取舍，再按天写安排和理由；每天只写一次天气，避免重复。"
            "不要提模型、工具、自动规划、思维过程或内部失败，不要加入费用和固定的出发前提醒。"
            "天气未知时只说‘天气待临近日期刷新’。请使用自然、具体、有画面感的中文。"
            f"已验证 JSON：{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            response = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["message"])])
            content = response.content if isinstance(response.content, str) else ""
            content = content.strip()
            if len(content) < 80 or "麦当劳" in content or "肯德基" in content:
                return None
            return content
        except Exception:  # noqa: BLE001 - retain verified deterministic response
            return None

    def run(
        self, trip_id: str, message: str, emit_callback: EventEmitter | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        token = _emitter.set(emit_callback)
        try:
            trip = self.profiles.get_trip(trip_id)
            if not trip:
                raise ValueError("旅行项目不存在")
            result = self.graph.invoke({
                "message": message, "trip": trip.model_dump(mode="json"),
                "request": trip.request.model_dump(mode="json"), "pois": [],
                "weather": [], "days": [], "warnings": [], "sources": [],
                "routes": [], "replan_count": 0, "replan_required": False,
            })
            updated = trip.model_copy(update={
                "request": TripRequest.model_validate(result.get("request", trip.request.model_dump(mode="json"))),
                "days": [DayPlan.model_validate(item) for item in result.get("days", [])] or trip.days,
                "warnings": [ConstraintWarning.model_validate(item) for item in result.get("warnings", [])],
                "status": "needs_input" if result.get("clarification") else "ready",
            })
            if not result.get("clarification") and not result.get("no_trip_change"):
                self.profiles.save_trip(updated, expected_version=expected_version)
            elif result.get("no_trip_change"):
                updated = trip
            unique_sources: dict[tuple[str, str | None, str, str], dict[str, Any]] = {}
            for raw in result.get("sources", []):
                source = SourceRecord.model_validate(raw)
                key = (source.kind, source.resource_id, source.location, source.reporttime)
                unique_sources[key] = source.model_dump(mode="json")
            return {
                "answer": result.get("answer", ""),
                "trip": updated.model_dump(mode="json"),
                "sources": list(unique_sources.values()),
            }
        finally:
            _emitter.reset(token)
