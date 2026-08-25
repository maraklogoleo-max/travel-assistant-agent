import re
import time
from contextvars import ContextVar
from datetime import datetime
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .agent import WeatherAgent
from .amap import AmapError
from .config import Settings
from .db import ProfileRepository, TripVersionConflict
from .models import (
    AgentPlan, ConstraintWarning, POI, SourceRecord, TravelProfile, TripChangeProposal, TripPlan,
    TripRequest, TripSummary,
)
from .travel_agent import TravelAgent


Emitter = Any
_emitter: ContextVar[Emitter | None] = ContextVar("unified_agent_emitter", default=None)


def emit(payload: dict[str, Any]) -> None:
    callback = _emitter.get()
    if callback:
        callback(payload)


class UnifiedState(TypedDict, total=False):
    message: str
    trip_id: str | None
    conversation_id: str
    expected_version: int | None
    trip: dict[str, Any] | None
    profile: dict[str, Any]
    messages: list[dict[str, Any]]
    summary: dict[str, Any]
    plan: dict[str, Any]
    result: dict[str, Any]


class TravelAssistantAgent:
    def __init__(
        self, settings: Settings, profiles: ProfileRepository,
        weather_agent: WeatherAgent, travel_agent: TravelAgent,
    ) -> None:
        self.settings = settings
        self.profiles = profiles
        self.weather_agent = weather_agent
        self.travel_agent = travel_agent
        self.llm: ChatOpenAI | None = None
        if settings.deepseek_api_key:
            self.llm = ChatOpenAI(
                api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
                model=settings.deepseek_model, temperature=0, timeout=25, max_retries=1,
            )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(UnifiedState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("plan", self._plan)
        builder.add_node("execute", self._execute)
        builder.add_node("persist_context", self._persist_context)
        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "plan")
        builder.add_edge("plan", "execute")
        builder.add_edge("execute", "persist_context")
        builder.add_edge("persist_context", END)
        return builder.compile()

    def _load_context(self, state: UnifiedState) -> UnifiedState:
        trip = self.profiles.get_trip(state.get("trip_id") or "") if state.get("trip_id") else None
        if state.get("trip_id") and trip is None:
            raise ValueError("旅行项目不存在")
        if trip and state.get("expected_version") is not None and trip.version != state["expected_version"]:
            raise TripVersionConflict(trip.version)
        messages = self.profiles.get_trip_messages(trip.trip_id, 40) if trip else []
        summary = self.profiles.get_trip_summary(trip.trip_id) if trip else None
        emit({"type": "step", "step": "memory", "status": "complete", "label": "已读取偏好与行程上下文"})
        return {
            "trip": trip.model_dump(mode="json") if trip else None,
            "profile": self.profiles.get_travel_profile().model_dump(mode="json"),
            "messages": [item.model_dump(mode="json") for item in messages],
            "summary": summary.model_dump(mode="json") if summary else {},
        }

    def _fallback_plan(self, text: str, has_trip: bool) -> AgentPlan:
        memory = bool(re.search(r"记住|以后|今后", text))
        weather = bool(re.search(r"天气|气温|温度|下雨|降雨|晴|带伞|大风|冷不冷|热不热|明天|后天", text))
        trip_create = bool(re.search(
            r"(?:去|到|前往).{1,16}(?:玩|旅游|旅行|行程)|"
            r"(?:安排|规划|制定|做).{0,18}(?:旅行|行程)|"
            r"(?:\d+|[一二三四五六七])\s*天(?:的)?.{1,12}(?:旅行|行程)",
            text,
        ))
        update = bool(re.search(r"换|调整|改成|改为|减少|增加|延长|缩短|放慢|紧凑", text))
        day_match = re.search(r"第([一二三四五六七\d])天", text)
        day_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
        day_token = day_match.group(1) if day_match else None
        target_day = (int(day_token) if day_token and day_token.isdigit() else day_map.get(day_token)) if day_token else None
        if memory:
            intent = "memory_update"
            tools = ["memory"]
        elif trip_create:
            intent = "trip_create"
            tools = ["resolve_location", "places", "weather", "routes", "trip"]
        elif has_trip and weather:
            intent = "trip_weather_assessment"
            tools = ["weather", "places", "trip"]
        elif has_trip and update:
            intent = "trip_update"
            tools = ["places", "routes", "trip"]
        elif has_trip:
            intent = "trip_query"
            tools = ["trip"]
        elif weather:
            intent = "weather_query"
            tools = ["resolve_location", "weather"]
        else:
            intent = "clarification"
            tools = []
        return AgentPlan(
            intent=intent, objective=text, tools=tools, target_day=target_day,
            requires_confirmation=intent == "trip_weather_assessment",
        )

    def _llm_plan(self, text: str, has_trip: bool, messages: list[dict[str, Any]], summary: dict[str, Any]) -> AgentPlan | None:
        if not self.llm:
            return None
        prompt = (
            "你是旅游出行助手的任务规划器，只输出结构化 AgentPlan。"
            "天气事实必须调用 weather；旅行地点调用 places；最终活动确定后调用 routes。"
            "有当前行程时，天气问题是 trip_weather_assessment，修改请求是 trip_update。"
            "只有用户明确说记住/以后时才是 memory_update。"
            f"当前是否有行程：{has_trip}。结构化行程摘要：{summary}。最近上下文：{messages[-6:]}"
        )
        try:
            return self.llm.with_structured_output(AgentPlan, method="function_calling").invoke([
                SystemMessage(content=prompt), HumanMessage(content=text),
            ])
        except Exception:
            return None

    def _plan(self, state: UnifiedState) -> UnifiedState:
        text = state["message"]
        fallback = self._fallback_plan(text, bool(state.get("trip")))
        plan = self._llm_plan(text, bool(state.get("trip")), state.get("messages", []), state.get("summary", {})) or fallback
        # Safety-critical routing remains deterministic even when the model omits trip context.
        if state.get("trip") and fallback.intent in {"trip_create", "trip_update", "trip_weather_assessment"}:
            plan = fallback
        if state.get("trip") and fallback.intent == "trip_query" and fallback.target_day:
            plan = fallback
        step_labels = {
            "resolve_location": "确认具体地点", "weather": "查询最新天气",
            "places": "查找合适的地点", "routes": "计算活动之间的路程",
            "memory": "更新你的旅行偏好", "trip": "整理当前行程",
        }
        emit({
            "type": "plan", "task_type": plan.intent, "title": "我会这样帮你",
            "steps": ["了解你的需求", *[step_labels[name] for name in plan.tools], "整理结果和建议"],
        })
        return {"plan": plan.model_dump(mode="json")}

    def _update_memory(self, text: str) -> dict[str, Any]:
        profile = self.profiles.get_travel_profile()
        saved: list[str] = []
        home = re.search(r"记住我(?:常住|住在|从)([\u4e00-\u9fff]{2,10})", text)
        if home:
            profile.home_city = home.group(1)
            saved.append(f"常住地 {profile.home_city}")
        for interest in ("自然风景", "博物馆", "亲子", "美食", "古镇", "购物"):
            if interest in text and interest not in profile.interests:
                profile.interests.append(interest)
                saved.append(f"喜欢{interest}")
        if re.search(r"轻松|慢节奏|不赶", text):
            profile.pace = "relaxed"
            saved.append("偏好轻松节奏")
        if re.search(r"不吃辣|忌辣", text) and "不吃辣" not in profile.dietary_restrictions:
            profile.dietary_restrictions.append("不吃辣")
            saved.append("不吃辣")
        self.profiles.save_travel_profile(profile)
        answer = "已记住：" + "；".join(saved) if saved else "我没有识别出需要长期保存的旅行偏好，请用“记住我……”说明。"
        self._stream(answer)
        return {"answer": answer, "travel_profile": profile.model_dump(mode="json"), "sources": []}

    def _create_trip_from_text(self, text: str, profile: TravelProfile) -> TripPlan:
        destination_match = re.search(r"(?:去|到|前往)[\s:：]*([\u4e00-\u9fff]{2,12})", text)
        if not destination_match:
            destination_match = re.search(r"(?:\d+|[一二三四五六七])\s*天(?:的)?([\u4e00-\u9fff]{2,10})(?:行程|旅行|旅游)", text)
        if not destination_match:
            destination_match = re.search(r"(?:安排|规划|制定|做)(?:一份|一个)?([\u4e00-\u9fff]{2,10})(?:行程|旅行|旅游)", text)
        destination = re.split(r"(?:玩|旅游|旅行|行程|[，,。；;\s]|\d+天)", destination_match.group(1))[0] if destination_match else "待定目的地"
        days_match = re.search(r"(\d+|[一二三四五六七])\s*(?:天|日)", text)
        day_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
        raw_days = days_match.group(1) if days_match else "1"
        days = min(7, max(1, int(raw_days) if raw_days.isdigit() else day_map[raw_days]))
        interests = list(dict.fromkeys(profile.interests + [item for item in ("自然风景", "博物馆", "亲子", "美食", "古镇", "购物") if item in text]))
        request = TripRequest(
            destination=destination, origin=profile.home_city, days=days, pace=profile.pace,
            budget_level=profile.budget_level, interests=interests,
            transport_mode=profile.transport_modes[0] if profile.transport_modes else "transit",
            dietary_restrictions=profile.dietary_restrictions, special_needs=profile.accessibility_needs,
        )
        trip = TripPlan(trip_id=str(uuid4()), name=f"{destination}之旅", request=request)
        return self.profiles.create_trip(trip.trip_id, request, trip)

    def _answer_trip_day(self, trip: TripPlan, target_day: int) -> dict[str, Any]:
        if target_day > len(trip.days):
            answer = (
                f"这份行程目前只有 {len(trip.days)} 天，还没有第 {target_day} 天的安排。"
                f"如果需要，可以说“重新安排 {target_day} 天的{trip.request.destination}行程”。"
            )
            self._stream(answer)
            return {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": []}
        day = trip.days[target_day - 1]
        period_label = {"morning": "上午", "afternoon": "下午", "evening": "晚上"}
        lines = [f"第 {target_day} 天（{day.date}）｜天气：{day.weather_summary}"]
        if not day.activities:
            lines.append("这一天还没有安排具体活动。")
        for activity in day.activities:
            route = ""
            if activity.route_from_previous and activity.route_from_previous.duration_s:
                route = f"，从上一站过来约 {activity.route_from_previous.duration_s // 60} 分钟"
            lines.append(f"- {period_label[activity.period]}：{activity.poi.name}{route}。{activity.reason}")
        answer = "\n".join(lines)
        self._stream(answer)
        return {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": []}

    def _assess_trip_weather(self, trip: TripPlan, text: str) -> dict[str, Any]:
        emit({"type": "tool_start", "tool": "weather", "status": "running", "label": "正在查看行程日期的天气"})
        location = self.travel_agent.amap.resolve_location(trip.request.destination)[0]
        forecasts = self.travel_agent.amap.get_forecast_weather(location)
        by_date = {item.date: item for item in forecasts if item.date}
        proposed = trip.model_copy(deep=True)
        changes: list[str] = []
        for day in proposed.days:
            forecast = by_date.get(day.date.isoformat())
            if not forecast:
                continue
            day.weather_summary = forecast.weather
            if re.search(r"雨|雪|雷", forecast.weather):
                outdoor = next((item for item in day.activities if not item.indoor), None)
                if outdoor:
                    try:
                        indoor_candidates = self.travel_agent.amap.search_places("博物馆", city=location.adcode)
                    except AmapError:
                        indoor_candidates = []
                    if indoor_candidates:
                        replacement = indoor_candidates[0]
                        old_name = outdoor.poi.name
                        outdoor.poi = replacement
                        outdoor.indoor = True
                        outdoor.reason = f"根据{forecast.weather}调整为室内活动"
                        changes.append(f"{day.date}：{old_name} → {replacement.name}")
        if changes:
            emit({"type": "tool_start", "tool": "routes", "status": "running", "label": "正在更新调整后的路程"})
            route_count = 0
            for day in proposed.days:
                for index, activity in enumerate(day.activities):
                    activity.route_from_previous = None
                    if index == 0:
                        continue
                    previous = day.activities[index - 1]
                    if not all((previous.poi.longitude, previous.poi.latitude, activity.poi.longitude, activity.poi.latitude)):
                        continue
                    try:
                        activity.route_from_previous = self.travel_agent.amap.plan_route(
                            previous.poi, activity.poi, proposed.request.transport_mode,
                        )
                        route_count += 1
                    except AmapError:
                        day.warnings.append(ConstraintWarning(
                            type="route_unavailable", severity="warning",
                            message=f"{previous.poi.name}到{activity.poi.name}的路线查询失败",
                            suggestion="出发前在高德地图重新确认",
                        ))
            emit({"type": "tool_result", "tool": "routes", "status": "complete", "label": f"已更新 {route_count} 段真实路线"})
        sources = [SourceRecord(location=location.display_name, reporttime=item.reporttime, kind="预报").model_dump() for item in forecasts[:1]]
        emit({"type": "tool_result", "tool": "weather", "status": "complete", "label": f"取得 {len(forecasts)} 天预报"})
        if changes:
            proposal = TripChangeProposal(
                proposal_id=str(uuid4()), trip_id=trip.trip_id, based_on_version=trip.version,
                title="雨天行程建议", description="这几天可能有降水，可以把部分户外活动换成室内活动。",
                changes=changes, proposed_plan=proposed,
            )
            self.profiles.save_change_proposal(proposal)
            emit({"type": "change_proposal", "proposal": proposal.model_dump(mode="json")})
            answer = "我看了行程日期的天气，部分户外活动可能会受降雨影响。下面是建议调整，确认前不会改动你的行程：\n\n" + "\n".join(f"- {item}" for item in changes)
        else:
            summaries = [f"{item.date}：{item.weather}" for item in forecasts]
            answer = "我看了行程日期的天气，目前不需要更换活动：\n\n" + "\n".join(summaries)
        self._stream(answer)
        return {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": sources}

    def _stream(self, answer: str) -> None:
        for index in range(0, len(answer), 8):
            emit({"type": "token", "delta": answer[index:index + 8]})
            time.sleep(0.01)

    def _execute(self, state: UnifiedState) -> UnifiedState:
        plan = AgentPlan.model_validate(state["plan"])
        text = state["message"]
        trip = TripPlan.model_validate(state["trip"]) if state.get("trip") else None
        if plan.intent == "memory_update":
            result = self._update_memory(text)
        elif plan.intent == "trip_create":
            trip = self._create_trip_from_text(text, TravelProfile.model_validate(state["profile"]))
            result = self.travel_agent.run(trip.trip_id, text, emit, expected_version=trip.version)
        elif plan.intent == "trip_weather_assessment" and trip:
            result = self._assess_trip_weather(trip, text)
        elif plan.intent in {"trip_update", "trip_query"} and trip:
            if plan.intent == "trip_query" and plan.target_day:
                result = self._answer_trip_day(trip, plan.target_day)
            elif plan.intent == "trip_query" and not re.search(r"换|调整|改|增加|减少|天气|路线", text):
                answer = f"你正在查看“{trip.name}”，一共 {len(trip.days)} 天。你可以继续问某一天的安排、天气是否有影响，或者告诉我想改哪里。"
                self._stream(answer)
                result = {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": []}
            else:
                result = self.travel_agent.run(trip.trip_id, text, emit, expected_version=state.get("expected_version"))
        elif plan.intent == "weather_query":
            result = self.weather_agent.run(state["conversation_id"], text, emit)
            if result.get("error_code"):
                raise AmapError(result.get("answer", "高德天气查询失败"), code=result["error_code"])
        else:
            answer = "你可以直接问天气，或告诉我目的地、天数和偏好来创建旅行计划。"
            self._stream(answer)
            result = {"answer": answer, "sources": []}
        return {"result": result, "trip_id": result.get("trip", {}).get("trip_id") or state.get("trip_id")}

    def _persist_context(self, state: UnifiedState) -> UnifiedState:
        result = state.get("result", {})
        trip_id = state.get("trip_id")
        if trip_id:
            self.profiles.add_trip_message(trip_id, "user", state["message"])
            self.profiles.add_trip_message(trip_id, "assistant", result.get("answer", ""), AgentPlan.model_validate(state["plan"]).intent)
            trip = self.profiles.get_trip(trip_id)
            if trip:
                summary = self.profiles.get_trip_summary(trip_id)
                summary.confirmed_requirements = [f"目的地：{trip.request.destination}", f"天数：{trip.request.days}", f"节奏：{trip.request.pace}"]
                summary.recent_changes = (summary.recent_changes + [state["message"]])[-10:]
                self.profiles.save_trip_summary(summary)
                result["trip"] = trip.model_dump(mode="json")
        return {"result": result}

    def run(
        self, message: str, *, conversation_id: str, trip_id: str | None = None,
        expected_version: int | None = None, emit_callback: Any = None,
    ) -> dict[str, Any]:
        token = _emitter.set(emit_callback)
        try:
            result = self.graph.invoke({
                "message": message, "trip_id": trip_id, "conversation_id": conversation_id,
                "expected_version": expected_version, "trip": None, "profile": {}, "messages": [], "summary": {}, "plan": {}, "result": {},
            })
            return result["result"]
        finally:
            _emitter.reset(token)
