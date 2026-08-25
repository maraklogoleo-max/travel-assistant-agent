import json
import re
import sqlite3
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .agent import WeatherAgent
from .amap import AmapError
from .config import Settings
from .date_parser import parse_day_count, parse_target_day, parse_trip_dates
from .db import ProfileRepository, TripVersionConflict
from .llm_json import invoke_json_model
from .models import (
    POI,
    AgentAction,
    AgentObservation,
    AgentPlan,
    AgentRunRecord,
    ConstraintWarning,
    ResolvedLocation,
    RouteLeg,
    SourceRecord,
    ToolResult,
    TravelProfile,
    TripChangeProposal,
    TripPlan,
    TripRequest,
    WeatherSnapshot,
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
    observations: list[dict[str, Any]]
    next_action: dict[str, Any]
    action_count: int
    done: bool
    run_id: str
    result: dict[str, Any]


class TravelAssistantAgent:
    """Goal-oriented, bounded travel agent with a plan/action/observation loop."""

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
        checkpoint_path = Path(settings.database_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.checkpointer.setup()
        self.graph = self._build_graph()

    def close(self) -> None:
        self._checkpoint_connection.close()

    def delete_thread(self, conversation_id: str) -> None:
        self.checkpointer.delete_thread(f"assistant:{conversation_id}")

    def _build_graph(self):
        builder = StateGraph(UnifiedState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("plan", self._plan)
        builder.add_node("decide", self._decide)
        builder.add_node("act", self._act)
        builder.add_node("observe", self._observe)
        builder.add_node("persist_context", self._persist_context)
        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "plan")
        builder.add_edge("plan", "decide")
        builder.add_conditional_edges(
            "decide", lambda state: "persist_context" if AgentAction.model_validate(state["next_action"]).tool == "finish" else "act",
        )
        builder.add_edge("act", "observe")
        builder.add_conditional_edges("observe", lambda state: "persist_context" if state.get("done") else "decide")
        builder.add_edge("persist_context", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _load_context(self, state: UnifiedState) -> UnifiedState:
        trip = self.profiles.get_trip(state.get("trip_id") or "") if state.get("trip_id") else None
        if state.get("trip_id") and trip is None:
            raise ValueError("旅行项目不存在")
        if trip and state.get("expected_version") is not None and trip.version != state["expected_version"]:
            raise TripVersionConflict(trip.version)
        conversation_messages = self.profiles.get_conversation_messages(state["conversation_id"], 20)
        trip_messages = self.profiles.get_trip_messages(trip.trip_id, 20) if trip else []
        messages = [item.model_dump(mode="json") for item in conversation_messages]
        known = {(item["role"], item["content"]) for item in messages}
        messages.extend(item.model_dump(mode="json") for item in trip_messages if (item.role, item.content) not in known)
        conversation_summary = self.profiles.get_conversation_summary(state["conversation_id"])
        trip_summary = self.profiles.get_trip_summary(trip.trip_id) if trip else None
        emit({"type": "step", "step": "memory", "status": "complete", "label": "已读取对话、偏好和行程上下文"})
        return {
            "trip": trip.model_dump(mode="json") if trip else None,
            "profile": self.profiles.get_travel_profile().model_dump(mode="json"),
            "messages": messages[-40:],
            "summary": {
                "conversation": conversation_summary.model_dump(mode="json"),
                "trip": trip_summary.model_dump(mode="json") if trip_summary else {},
            },
            "observations": [], "action_count": 0, "done": False, "result": {},
        }

    @staticmethod
    def _extract_destination(text: str) -> str | None:
        patterns = (
            # Stop at common travel-detail markers.  Without this boundary,
            # “去九寨沟从无锡出发住民宿” used to become one fictitious place.
            r"(?:从[\u4e00-\u9fff]{2,12})?(?:去|到|前往)[\s:：]*([\u4e00-\u9fff]{2,12}?)(?=(?:从|出发|住|坐|做|搭|乘|玩|旅游|旅行|行程|[，,。；;\s]|$))",
            r"(?:\d+|[一二三四五六七])\s*天(?:的)?([\u4e00-\u9fff]{2,10})(?:行程|旅行|旅游)",
            r"(?:安排|规划|制定|做)(?:一份|一个)?([\u4e00-\u9fff]{2,10})(?:行程|旅行|旅游)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = re.split(r"(?:玩|旅游|旅行|行程|[，,。；;\s]|\d+天)", match.group(1))[0]
                if value:
                    return value
        return None

    def _normalize_trip_create_plan(self, plan: AgentPlan, text: str) -> AgentPlan:
        """Trust explicit user facts over optional model-produced fields.

        The planning model may reasonably identify an intent, but it must not
        turn information the user already supplied into a new question.  Only
        destination, date and duration are actionable creation requirements;
        origin, accommodation and transport have safe defaults for this app.
        """
        normalized = plan.model_copy(deep=True)
        parsed = parse_trip_dates(text, self.settings.timezone)
        destination = self._extract_destination(text)
        if destination:
            normalized.requirements["destination"] = destination
        if parsed.start_date is not None:
            normalized.requirements["start_date"] = parsed.start_date.isoformat()
        if parsed.days is not None:
            normalized.requirements["days"] = parsed.days
        if "duration_days" in normalized.requirements and "days" not in normalized.requirements:
            normalized.requirements["days"] = normalized.requirements["duration_days"]
        normalized.requirements.pop("duration_days", None)

        aliases = {
            "destination": "destination", "目的地": "destination",
            "start_date": "start_date", "出行日期": "start_date", "日期": "start_date",
            "days": "days", "duration_days": "days", "游玩天数": "days", "天数": "days",
        }
        missing: list[str] = []
        for field in normalized.missing_fields:
            canonical = aliases.get(field)
            if canonical and canonical not in missing:
                missing.append(canonical)
        # A model may omit missing_fields even when it lacks the only fact that
        # cannot be safely inferred: a destination.
        if not normalized.requirements.get("destination") and "destination" not in missing:
            missing.append("destination")
        if not normalized.requirements.get("days") and "days" not in missing:
            missing.append("days")
        normalized.missing_fields = [
            field for field in missing if not normalized.requirements.get(field)
        ]
        normalized.tools = ["resolve_location", "places", "weather", "routes", "trip"]
        normalized.planned_steps = normalized.tools
        normalized.intent = "trip_create"
        return normalized

    def _fallback_plan(self, text: str, has_trip: bool) -> AgentPlan:
        memory = bool(re.search(r"记住|以后|今后", text))
        keep_outdoor = bool(re.search(
            r"(?:下雨|雨天|降雨).{0,10}(?:也要|仍要|还要|照样|无所谓|没关系|不影响).{0,8}(?:户外|室外)|"
            r"(?:户外|室外).{0,10}(?:也要|仍要|还要|照样|多一些|不减少)|"
            r"(?:多一些|更多).{0,6}(?:户外|室外)|(?:不怕雨|不介意下雨|坚持户外)",
            text,
        ))
        # Suitability questions are weather/trip-assessment requests even when
        # the user does not repeat the word “天气” (for example, “那适合旅游吗”).
        # Keep this deterministic fallback ahead of the optional planner model
        # so a selected trip always supplies the missing destination/date context.
        weather = bool(re.search(
            r"天气|气温|温度|下雨|降雨|晴|带伞|大风|冷不冷|热不热|明天|后天|"
            r"适合(?:旅游|出游|出行)|值得(?:去|旅游|出游)|能不能去|可以去",
            text,
        ))
        trip_create = bool(re.search(
            r"(?:去|到|前往).{1,16}(?:玩|旅游|旅行|行程)|"
            r"(?:安排|规划|制定|做).{0,18}(?:旅行|行程)|"
            r"(?:\d+|[一二三四五六七])\s*天(?:的)?.{1,12}(?:旅行|行程)|"
            r"(?:去|到|前往).{1,16}(?:出发|住|高铁|飞机|自驾)", text,
        ))
        update = bool(re.search(r"换|调整|改成|改为|删除|减少|增加|延长|缩短|放慢|紧凑", text))
        target_day = parse_target_day(text)
        destination = self._extract_destination(text)
        parsed_dates = parse_trip_dates(text, self.settings.timezone)
        if memory:
            intent, tools = "memory_update", ["memory"]
        elif trip_create:
            intent, tools = "trip_create", ["resolve_location", "places", "weather", "routes", "trip"]
        elif has_trip and (keep_outdoor or update):
            intent, tools = "trip_update", ["places", "routes", "trip"]
        elif has_trip and weather:
            intent, tools = "trip_weather_assessment", ["weather", "places", "routes", "trip"]
        elif has_trip:
            intent, tools = "trip_query", ["trip"]
        elif weather:
            intent, tools = "weather_query", ["resolve_location", "weather"]
        else:
            intent, tools = "clarification", []
        missing = ["destination"] if intent == "trip_create" and not destination else []
        requirements = {
            "destination": destination, "days": parse_day_count(text), "target_day": target_day,
            "start_date": parsed_dates.start_date.isoformat() if parsed_dates.start_date else None,
        }
        return AgentPlan(
            intent=intent, objective=text, tools=tools, target_day=target_day,
            requires_confirmation=intent == "trip_weather_assessment", missing_fields=missing,
            requirements={key: value for key, value in requirements.items() if value is not None},
            constraints=["天气、地点和路线事实必须来自工具", "已保存行程的修改需要版本校验"],
            planned_steps=tools, action_budget=8,
        )

    def _llm_plan(self, text: str, has_trip: bool, messages: list[dict[str, Any]], summary: dict[str, Any]) -> AgentPlan | None:
        if not self.llm:
            return None
        prompt = (
            "你是旅游出行助手的任务规划器。必须只输出一个 JSON 对象，不要 Markdown、解释或思考过程。"
            "该 JSON 必须符合下方 AgentPlan schema。"
            "识别目标、缺失字段、约束、所需工具和目标日期。"
            "天气事实必须调用 weather；旅行地点调用 places；活动确定后调用 routes。"
            "有当前行程时，天气问题是 trip_weather_assessment，修改请求是 trip_update。"
            "只有用户明确说记住或以后时才是 memory_update。不要把推断偏好写入长期记忆。"
            f"当前是否有行程：{has_trip}。摘要：{summary}。最近上下文：{messages[-10:]}。"
            f"AgentPlan JSON schema：{json.dumps(AgentPlan.model_json_schema(), ensure_ascii=False)}"
        )
        return invoke_json_model(self.llm, AgentPlan, [
            SystemMessage(content=prompt), HumanMessage(content=text),
        ])

    def _plan(self, state: UnifiedState) -> UnifiedState:
        fallback = self._fallback_plan(state["message"], bool(state.get("trip")))
        conversation_summary = state.get("summary", {}).get("conversation", {})
        pending_raw = conversation_summary.get("pending_plan") if isinstance(conversation_summary, dict) else None
        pending = AgentPlan.model_validate(pending_raw) if pending_raw else None
        if pending and pending.intent == "trip_create":
            pending = self._normalize_trip_create_plan(pending, state["message"])
        supplemental_dates = parse_trip_dates(state["message"], self.settings.timezone)
        destination = self._extract_destination(state["message"])
        if not destination and re.fullmatch(r"[\u4e00-\u9fff]{2,18}", state["message"].strip()):
            destination = state["message"].strip()
        # A reply such as “明天” or “9月3日，玩三天” belongs to the
        # clarification that preceded it.  Do this before normal intent
        # detection: those phrases can otherwise look like a fresh weather
        # query and leave the old missing start_date in place forever.
        supplements_pending = bool(
            pending
            and pending.intent == "trip_create"
            and (
                destination
                or supplemental_dates.start_date is not None
                or supplemental_dates.days is not None
            )
        )
        if pending and (fallback.intent == "clarification" or supplements_pending):
            plan = pending.model_copy(deep=True)
            if destination:
                plan.requirements["destination"] = destination
                plan.missing_fields = [item for item in plan.missing_fields if item != "destination"]
            if supplemental_dates.start_date is not None:
                plan.requirements["start_date"] = supplemental_dates.start_date.isoformat()
                plan.missing_fields = [item for item in plan.missing_fields if item != "start_date"]
            if supplemental_dates.days is not None:
                plan.requirements["days"] = supplemental_dates.days
                plan.missing_fields = [item for item in plan.missing_fields if item != "days"]
            plan.objective = f"{pending.objective}；用户补充：{state['message']}"
        else:
            plan = self._llm_plan(state["message"], bool(state.get("trip")), state.get("messages", []), state.get("summary", {})) or fallback
        if fallback.intent == "trip_create" and not supplements_pending:
            # For a clear creation request, use the explicit facts extracted
            # from the user's sentence.  The model may otherwise invent a
            # default duration such as one day.
            plan = self._normalize_trip_create_plan(fallback, state["message"])
        elif plan.intent == "trip_create" or fallback.intent == "trip_create":
            # A clear trip request should not be blocked by optional fields
            # invented by a model response.
            plan = self._normalize_trip_create_plan(plan if plan.intent == "trip_create" else fallback, state["message"])
        # A clear local edit/weather request must stay attached to the selected
        # trip.  Otherwise keep the model's richer interpretation instead of
        # replacing it with a keyword-derived intent every time.
        if state.get("trip") and fallback.intent == "trip_create":
            plan = fallback
        elif (
            state.get("trip")
            and fallback.intent in {"trip_update", "trip_weather_assessment"}
            and plan.intent in {"clarification", "trip_query", "weather_query"}
        ):
            plan = fallback
        if fallback.target_day:
            plan.target_day = fallback.target_day
        if fallback.intent == "trip_create" and fallback.requirements.get("destination"):
            plan.requirements["destination"] = fallback.requirements["destination"]
            plan.missing_fields = [item for item in plan.missing_fields if item != "destination"]
        labels = {
            "resolve_location": "确认具体地点", "weather": "查询天气", "places": "查找合适的地点",
            "routes": "计算活动之间的路程", "memory": "更新旅行偏好", "trip": "整理当前行程",
        }
        steps = [labels.get(name, name) for name in plan.planned_steps or plan.tools]
        emit({"type": "plan", "task_type": plan.intent, "title": "我会这样帮你", "steps": ["理解你的需求", *steps, "检查并整理结果"]})
        run_id = str(uuid4())
        self.profiles.create_agent_run(AgentRunRecord(
            run_id=run_id, conversation_id=state["conversation_id"], trip_id=state.get("trip_id"), intent=plan.intent,
        ))
        return {"plan": plan.model_dump(mode="json"), "run_id": run_id}

    def _fallback_action(self, state: UnifiedState) -> AgentAction:
        plan = AgentPlan.model_validate(state["plan"])
        observed = [AgentObservation.model_validate(item) for item in state.get("observations", [])]
        observed_tools = [item.action.tool for item in observed]
        if plan.missing_fields:
            return AgentAction(tool="finish", objective="向用户补充必要条件", arguments={"missing_fields": plan.missing_fields})
        if plan.intent == "trip_create":
            if "resolve_location" not in observed_tools:
                return AgentAction(tool="resolve_location", objective="确认目的地", arguments={"query": plan.requirements.get("destination", "")})
            return AgentAction(tool="trip_create", objective="生成并保存多日行程")
        mapping = {
            "weather_query": "weather", "trip_update": "trip_update", "trip_query": "trip_query",
            "trip_weather_assessment": "trip_weather_assessment", "memory_update": "memory",
        }
        tool = mapping.get(plan.intent, "finish")
        if tool in observed_tools or state.get("result"):
            return AgentAction(tool="finish", objective="目标已完成")
        return AgentAction(tool=tool, objective=plan.objective, arguments={"target_day": plan.target_day})

    def _llm_action(self, state: UnifiedState, fallback: AgentAction) -> AgentAction:
        if not self.llm or fallback.tool == "finish":
            return fallback
        plan = AgentPlan.model_validate(state["plan"])
        observations = [AgentObservation.model_validate(item) for item in state.get("observations", [])]
        compact_observations = [
            {
                "tool": item.action.tool, "success": item.result.success,
                "error_code": item.result.error_code,
            }
            for item in observations
        ]
        allowed = [fallback.tool, "finish"]
        prompt = (
            "你是受控旅游 Agent 的行动选择器。必须只输出一个 JSON 对象，不要 Markdown、解释或思考过程。"
            "该 JSON 必须符合下方 AgentAction schema。"
            f"任务计划：{plan.model_dump(mode='json')}。已观察结果：{compact_observations}。"
            f"本轮只允许选择：{allowed}。目标尚未取得最终结果时不能选择 finish。"
            "不要修改工具参数中的用户事实。"
            f"AgentAction JSON schema：{json.dumps(AgentAction.model_json_schema(), ensure_ascii=False)}"
        )
        action = invoke_json_model(self.llm, AgentAction, [
            SystemMessage(content=prompt), HumanMessage(content=state["message"]),
        ])
        if action is None:
            return fallback
        if action.tool not in allowed or action.tool == "finish" and not state.get("result"):
            return fallback
        if action.tool == fallback.tool:
            action.arguments = {**fallback.arguments, **action.arguments}
        return action

    def _decide(self, state: UnifiedState) -> UnifiedState:
        plan = AgentPlan.model_validate(state["plan"])
        if state.get("result"):
            return {"next_action": AgentAction(tool="finish", objective="目标已完成").model_dump(mode="json")}
        if state.get("action_count", 0) >= plan.action_budget:
            answer = "这次任务需要的步骤超过了安全执行上限。已停止继续调用工具，你可以缩小范围后再试。"
            self._stream(answer)
            return {"result": {"answer": answer, "sources": [], "error_code": "ACTION_BUDGET_EXCEEDED"}, "next_action": AgentAction(tool="finish", objective="达到执行上限").model_dump(mode="json"), "done": True}
        action = self._llm_action(state, self._fallback_action(state))
        if action.tool == "finish" and not state.get("result"):
            if plan.missing_fields:
                labels = {"destination": "目的地", "start_date": "出发日期", "days": "游玩天数"}
                missing = "、".join(labels.get(item, item) for item in plan.missing_fields)
                requirements = plan.requirements
                known: list[str] = []
                if requirements.get("start_date"):
                    known.append(f"出发日期是 {requirements['start_date']}")
                if requirements.get("destination"):
                    known.append(f"目的地是 {requirements['destination']}")
                if re.search(r"从[\u4e00-\u9fff]{2,12}(?:出发|去|到)", state["message"]):
                    origin = re.search(r"从([\u4e00-\u9fff]{2,12}?)(?:出发|去|到)", state["message"])
                    if origin:
                        known.append(f"从{origin.group(1)}出发")
                if "高铁" in state["message"]:
                    known.append("优先考虑高铁")
                if "民宿" in state["message"]:
                    known.append("住宿倾向民宿")
                known_text = "，".join(known)
                prefix = f"我先记下了：{known_text}。" if known_text else "我先记下了你的旅行需求。"
                answer = f"{prefix}为了把每天的安排做完整，还需要确认{missing}。补充后我就接着为你规划。"
                emit({"type": "clarification", "code": "MISSING_REQUIREMENTS", "message": answer})
            else:
                answer = "你可以直接问天气，或告诉我目的地、天数和偏好来创建旅行计划。"
            self._stream(answer)
            payload = {"answer": answer, "sources": []}
            if plan.missing_fields:
                payload["error_code"] = "MISSING_REQUIREMENTS"
            return {"result": payload, "next_action": action.model_dump(mode="json"), "done": True}
        emit({"type": "agent_action", "action": action.tool, "objective": action.objective, "sequence": state.get("action_count", 0) + 1})
        return {"next_action": action.model_dump(mode="json")}

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
        if re.search(r"公交|公共交通", text) and "transit" not in profile.transport_modes:
            profile.transport_modes.append("transit")
            saved.append("常用公共交通")
        if re.search(r"自驾|开车", text) and "driving" not in profile.transport_modes:
            profile.transport_modes.append("driving")
            saved.append("常用自驾")
        self.profiles.save_travel_profile(profile)
        answer = "已记住：" + "；".join(saved) if saved else "我没有识别出需要长期保存的旅行偏好，请用“记住我……”说明。"
        self._stream(answer)
        return {"answer": answer, "travel_profile": profile.model_dump(mode="json"), "sources": []}

    def _create_trip_from_text(self, text: str, profile: TravelProfile, destination: str, plan: AgentPlan) -> TripPlan:
        parsed = parse_trip_dates(text, self.settings.timezone)
        interests = list(dict.fromkeys(profile.interests + [item for item in ("自然风景", "博物馆", "亲子", "美食", "古镇", "购物") if item in text]))
        origin_match = re.search(r"从([\u4e00-\u9fff]{2,12}?)(?:去|到|前往|出发|[，,。；;\s]|$)", text)
        planned_start = plan.requirements.get("start_date")
        request = TripRequest(
            destination=destination, origin=origin_match.group(1) if origin_match else profile.home_city,
            start_date=parsed.start_date or planned_start,
            days=parsed.days or plan.requirements.get("days") or 1, pace=profile.pace,
            budget_level=profile.budget_level, interests=interests,
            transport_mode=profile.transport_modes[0] if profile.transport_modes else "transit",
            transport_preference="高铁" if "高铁" in text else "飞机" if "飞机" in text else None,
            accommodation_preference="民宿" if "民宿" in text else "酒店" if "酒店" in text else None,
            dietary_restrictions=profile.dietary_restrictions, special_needs=profile.accessibility_needs,
        )
        trip = TripPlan(trip_id=str(uuid4()), name=f"{destination}之旅", request=request)
        return self.profiles.create_trip(trip.trip_id, request, trip)

    def _answer_trip_day(self, trip: TripPlan, target_day: int) -> dict[str, Any]:
        if target_day > len(trip.days):
            answer = f"这份行程目前只有 {len(trip.days)} 天，还没有第 {target_day} 天的安排。"
            self._stream(answer)
            return {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": []}
        day = trip.days[target_day - 1]
        period_label = {"morning": "上午", "afternoon": "下午", "evening": "晚上"}
        lines = [f"第 {target_day} 天（{day.date}）｜天气：{day.weather_summary}"]
        for activity in day.activities:
            route = f"，从上一站过来约 {activity.route_from_previous.duration_s // 60} 分钟" if activity.route_from_previous and activity.route_from_previous.duration_s else ""
            time_range = f"{activity.start_time}–{activity.end_time} " if activity.start_time and activity.end_time else ""
            lines.append(f"- {time_range}{period_label[activity.period]}：{activity.poi.name}{route}。{activity.reason}")
        if not day.activities:
            lines.append("这一天还没有已验证的活动安排。")
        answer = "\n".join(lines)
        self._stream(answer)
        sources = [source.model_dump(mode="json") for activity in day.activities for source in activity.sources]
        return {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": sources}

    def _assess_trip_weather(self, trip: TripPlan, user_message: str = "") -> dict[str, Any]:
        emit({"type": "tool_start", "tool": "weather", "status": "running", "label": "正在查看行程日期的天气"})
        resolved = self.travel_agent.tools.resolve_location(trip.request.destination)
        if not resolved.success:
            if resolved.error_code == "LOCATION_AMBIGUOUS":
                answer = f"{resolved.user_message}请补充省份或城市后，我再评估行程天气。"
                self._stream(answer)
                return {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": [item.model_dump(mode="json") for item in resolved.sources], "error_code": resolved.error_code}
            raise AmapError(resolved.user_message or "地点解析失败", code=resolved.error_code or "LOCATION_NOT_FOUND")
        location = ResolvedLocation.model_validate(resolved.data)
        weather_result = self.travel_agent.tools.weather(location, forecast=True)
        if not weather_result.success:
            raise AmapError(weather_result.user_message or "天气查询失败", code=weather_result.error_code or "AMAP_WEATHER_FAILED")
        forecasts = [WeatherSnapshot.model_validate(item) for item in weather_result.data]
        by_date = {item.date: item for item in forecasts if item.date}
        proposed = trip.model_copy(deep=True)
        changes: list[str] = []
        # A rain forecast is not, by itself, permission to replace outdoor
        # activities. Respect an explicit preference such as “下雨天也要户外活动”
        # and only report practical precautions in that case.
        keep_outdoor = bool(re.search(
            r"(?:下雨|雨天|降雨).{0,10}(?:也要|仍要|还要|照样|无所谓|没关系|不影响).{0,8}(?:户外|室外)|"
            r"(?:户外|室外).{0,10}(?:也要|仍要|还要|照样|多一些|不减少)|"
            r"(?:多一些|更多).{0,6}(?:户外|室外)|"
            r"(?:不怕雨|不介意下雨|坚持户外)",
            user_message,
        ))
        sources = [item.model_dump(mode="json") for item in weather_result.sources]
        for day in proposed.days:
            forecast = by_date.get(day.date.isoformat())
            if not forecast:
                continue
            day.weather_summary = forecast.weather
            if keep_outdoor or not re.search(r"雨|雪|雷", forecast.weather):
                continue
            outdoor = next((item for item in day.activities if not item.indoor), None)
            if not outdoor:
                continue
            place_result = self.travel_agent.tools.places("博物馆", city=location.adcode)
            if not place_result.success or not place_result.data:
                continue
            replacement = POI.model_validate(place_result.data[0])
            old_name = outdoor.poi.name
            outdoor.poi = replacement
            outdoor.indoor = True
            outdoor.reason = f"根据{forecast.weather}改为室内活动"
            outdoor.sources = place_result.sources
            sources.extend(item.model_dump(mode="json") for item in place_result.sources)
            changes.append(f"{day.date}：{old_name} → {replacement.name}")
        if changes:
            emit({"type": "tool_start", "tool": "routes", "status": "running", "label": "正在更新受影响活动之间的路程"})
            route_count = 0
            for day in proposed.days:
                for index, activity in enumerate(day.activities):
                    activity.route_from_previous = None
                    if index == 0:
                        continue
                    previous = day.activities[index - 1]
                    route_result = self.travel_agent.tools.route(previous.poi, activity.poi, proposed.request.transport_mode)
                    if route_result.success:
                        activity.route_from_previous = RouteLeg.model_validate(route_result.data)
                        activity.sources.extend(route_result.sources)
                        sources.extend(item.model_dump(mode="json") for item in route_result.sources)
                        route_count += 1
                    else:
                        day.warnings.append(ConstraintWarning(type="route_unavailable", severity="warning", message=f"{previous.poi.name}到{activity.poi.name}暂未取得路线数据。"))
            emit({"type": "tool_result", "tool": "routes", "status": "complete", "label": f"已更新 {route_count} 段路线"})
        emit({"type": "tool_result", "tool": "weather", "status": "complete", "label": f"取得 {len(forecasts)} 天预报"})
        weather_summary = "、".join(f"{item.date} {item.weather}" for item in forecasts)
        rainy = any(re.search(r"雨|雪|雷", item.weather) for item in forecasts)
        suitability = (
            "从目前预报看可以出行，但有降水时段，户外活动建议安排在雨势较弱的时段，并准备雨具。"
            if rainy
            else "从目前预报看适合出行，户外活动可以按原计划进行。"
        )
        if changes:
            proposal = TripChangeProposal(
                proposal_id=str(uuid4()), trip_id=trip.trip_id, based_on_version=trip.version,
                title="天气影响调整建议", description="部分户外活动可能受降水影响，可以替换为室内活动。",
                changes=changes, proposed_plan=proposed,
            )
            self.profiles.save_change_proposal(proposal)
            emit({"type": "change_proposal", "proposal": proposal.model_dump(mode="json")})
            answer = f"{suitability}\n天气：{weather_summary}\n\n我整理了一份待确认的调整：\n\n" + "\n".join(f"- {item}" for item in changes)
        elif keep_outdoor:
            summaries = [f"{item.date}：{item.weather}" for item in forecasts]
            answer = (
                "我会按你的要求保留户外活动，不因为下雨自动替换行程。\n\n"
                + "天气：" + "、".join(summaries)
                + "\n\n建议：带雨具和防滑鞋，户外活动尽量避开雨势较强的时段，并预留交通缓冲。"
            )
        else:
            answer = f"{suitability}\n天气：{weather_summary}\n\n目前没有需要替换的活动，行程可以按原安排继续。"
        self._stream(answer)
        return {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": sources}

    def _stream(self, answer: str) -> None:
        for index in range(0, len(answer), 8):
            emit({"type": "token", "delta": answer[index:index + 8]})
            time.sleep(0.01)

    @staticmethod
    def _result_tool_result(tool: str, result: dict[str, Any]) -> ToolResult:
        sources = [SourceRecord.model_validate(item) for item in result.get("sources", [])]
        return ToolResult(
            tool=tool, success=not bool(result.get("error_code")), data=result,
            sources=sources, error_code=result.get("error_code"),
            user_message=result.get("answer") if result.get("error_code") else None,
        )

    def _act(self, state: UnifiedState) -> UnifiedState:
        action = AgentAction.model_validate(state["next_action"])
        plan = AgentPlan.model_validate(state["plan"])
        text = state["message"]
        trip = TripPlan.model_validate(state["trip"]) if state.get("trip") else None
        label = {
            "resolve_location": "正在确认具体地点", "weather": "正在查询天气", "memory": "正在保存旅行偏好",
            "trip_create": "正在生成多日行程", "trip_update": "正在修改指定行程", "trip_query": "正在读取行程",
            "trip_weather_assessment": "正在评估天气对行程的影响",
        }.get(action.tool, "正在执行任务")
        emit({"type": "tool_start", "tool": action.tool, "status": "running", "label": label})
        result: dict[str, Any] | None = None
        if action.tool == "resolve_location":
            tool_result = self.travel_agent.tools.resolve_location(str(action.arguments.get("query", "")))
            if not tool_result.success:
                choices = ["".join(filter(None, (item.get("province"), item.get("city"), item.get("district")))) for item in (tool_result.data or [])]
                answer = tool_result.user_message or "地点无法确认。"
                if choices:
                    answer += "可选地点：" + "、".join(choices) + "。"
                emit({
                    "type": "clarification", "code": tool_result.error_code or "LOCATION_NOT_FOUND",
                    "message": answer,
                    "choices": [{"label": item, "value": item} for item in choices],
                })
                self._stream(answer)
                result = {"answer": answer, "sources": [item.model_dump(mode="json") for item in tool_result.sources], "error_code": tool_result.error_code}
        elif action.tool == "memory":
            result = self._update_memory(text)
            tool_result = self._result_tool_result(action.tool, result)
        elif action.tool == "weather":
            result = self.weather_agent.run(state["conversation_id"], text, emit)
            tool_result = self._result_tool_result(action.tool, result)
            if result.get("error_code"):
                raise AmapError(result.get("answer", "高德天气查询失败"), code=result["error_code"])
        elif action.tool == "trip_create":
            destination = str(plan.requirements.get("destination") or self._extract_destination(text) or "")
            created = self._create_trip_from_text(text, TravelProfile.model_validate(state["profile"]), destination, plan)
            try:
                result = self.travel_agent.run(created.trip_id, text, emit, expected_version=created.version)
            except Exception:
                self.profiles.delete_trip(created.trip_id)
                raise
            tool_result = self._result_tool_result(action.tool, result)
        elif action.tool == "trip_weather_assessment" and trip:
            result = self._assess_trip_weather(trip, text)
            tool_result = self._result_tool_result(action.tool, result)
        elif action.tool == "trip_query" and trip:
            if plan.target_day:
                result = self._answer_trip_day(trip, plan.target_day)
            else:
                answer = f"你正在查看“{trip.name}”，共 {len(trip.days)} 天。可以继续问某一天，或告诉我需要修改哪个活动。"
                self._stream(answer)
                result = {"answer": answer, "trip": trip.model_dump(mode="json"), "sources": []}
            tool_result = self._result_tool_result(action.tool, result)
        elif action.tool == "trip_update" and trip:
            result = self.travel_agent.run(trip.trip_id, text, emit, expected_version=state.get("expected_version"))
            tool_result = self._result_tool_result(action.tool, result)
        else:
            tool_result = ToolResult(tool=action.tool, success=False, error_code="INVALID_AGENT_ACTION", user_message="当前操作缺少必要上下文。")
            result = {"answer": tool_result.user_message, "sources": [], "error_code": tool_result.error_code}
        observation = AgentObservation(action=action, result=tool_result)
        return {
            "observations": [*state.get("observations", []), observation.model_dump(mode="json")],
            "result": result or {}, "trip_id": (result or {}).get("trip", {}).get("trip_id") or state.get("trip_id"),
        }

    def _observe(self, state: UnifiedState) -> UnifiedState:
        latest = AgentObservation.model_validate(state.get("observations", [])[-1])
        count = state.get("action_count", 0) + 1
        self.profiles.add_tool_call(state["run_id"], latest.action.tool, latest.result.success, latest.result.model_dump(mode="json"))
        emit({
            "type": "tool_result", "tool": latest.action.tool,
            "status": "complete" if latest.result.success else "error",
            "label": "已取得可用结果" if latest.result.success else (latest.result.user_message or "工具执行失败"),
        })
        done = bool(state.get("result"))
        if count >= AgentPlan.model_validate(state["plan"]).action_budget and not done:
            answer = "这次任务已达到安全执行上限。你可以缩小范围后继续。"
            self._stream(answer)
            return {"action_count": count, "done": True, "result": {"answer": answer, "sources": [], "error_code": "ACTION_BUDGET_EXCEEDED"}}
        return {"action_count": count, "done": done}

    def _persist_context(self, state: UnifiedState) -> UnifiedState:
        result = state.get("result", {})
        plan = AgentPlan.model_validate(state["plan"])
        self.profiles.add_conversation_message(state["conversation_id"], "user", state["message"], plan.intent)
        self.profiles.add_conversation_message(state["conversation_id"], "assistant", result.get("answer", ""), plan.intent)
        conversation_summary = self.profiles.get_conversation_summary(state["conversation_id"])
        conversation_summary.recent_topics = (conversation_summary.recent_topics + [plan.objective])[-20:]
        needs_input_codes = {"LOCATION_AMBIGUOUS", "LOCATION_NOT_FOUND", "MISSING_REQUIREMENTS"}
        if plan.missing_fields or result.get("error_code") in needs_input_codes:
            conversation_summary.pending_plan = plan
            conversation_summary.unresolved_questions = list(plan.missing_fields) or [result.get("error_code")]
        else:
            conversation_summary.pending_plan = None
            conversation_summary.unresolved_questions = []
        self.profiles.save_conversation_summary(conversation_summary)
        trip_id = state.get("trip_id")
        if trip_id:
            self.profiles.add_trip_message(trip_id, "user", state["message"], plan.intent)
            self.profiles.add_trip_message(trip_id, "assistant", result.get("answer", ""), plan.intent)
            trip = self.profiles.get_trip(trip_id)
            if trip:
                summary = self.profiles.get_trip_summary(trip_id)
                summary.confirmed_requirements = [f"目的地：{trip.request.destination}", f"天数：{trip.request.days}", f"节奏：{trip.request.pace}"]
                summary.recent_changes = (summary.recent_changes + [state["message"]])[-20:]
                self.profiles.save_trip_summary(summary)
                result["trip"] = trip.model_dump(mode="json")
        status = (
            "needs_input" if plan.missing_fields or result.get("error_code") in needs_input_codes
            else "failed" if result.get("error_code") else "completed"
        )
        self.profiles.update_agent_run(state["run_id"], status=status, action_count=state.get("action_count", 0), error_code=result.get("error_code"))
        return {"result": result}

    def run(
        self, message: str, *, conversation_id: str, trip_id: str | None = None,
        expected_version: int | None = None, emit_callback: Any = None,
    ) -> dict[str, Any]:
        token = _emitter.set(emit_callback)
        try:
            result = self.graph.invoke(
                {
                    "message": message, "trip_id": trip_id, "conversation_id": conversation_id,
                    "expected_version": expected_version, "trip": None, "profile": {}, "messages": [],
                    "summary": {}, "plan": {}, "observations": [], "next_action": {}, "action_count": 0,
                    "done": False, "run_id": "", "result": {},
                },
                {"configurable": {"thread_id": f"assistant:{conversation_id}"}},
            )
            return result["result"]
        finally:
            _emitter.reset(token)
