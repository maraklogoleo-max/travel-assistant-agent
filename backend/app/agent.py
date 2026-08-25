import json
import re
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, TypedDict
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .amap import AmapClient, AmapError
from .config import Settings
from .db import ProfileRepository
from .llm_json import invoke_json_model
from .models import ResolvedLocation, SourceRecord, UserProfile, WeatherPlan, WeatherSnapshot


EventEmitter = Callable[[dict[str, Any]], None]
_event_emitter: ContextVar[EventEmitter | None] = ContextVar("weather_event_emitter", default=None)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    profile: dict[str, Any]
    plan: dict[str, Any]
    locations: list[dict[str, Any]]
    last_locations: list[dict[str, Any]]
    weather: list[dict[str, Any]]
    clarification: str | None
    answer: str
    sources: list[dict[str, Any]]
    errors: list[str]
    error_codes: list[str]
    memory_saved: list[str]


def _emit(step: str, status: str, label: str) -> None:
    emitter = _event_emitter.get()
    if emitter:
        emitter({"type": "step", "step": step, "status": status, "label": label})


def _emit_token(delta: str) -> None:
    emitter = _event_emitter.get()
    if emitter:
        emitter({"type": "token", "delta": delta})


def _emit_plan(plan: WeatherPlan) -> None:
    task_type = plan.task_type
    titles = {
        "weather_query": "天气查询计划",
        "travel_assessment": "旅游适配评估计划",
        "comparison": "多地点对比计划",
        "memory_update": "偏好记忆计划",
    }
    steps = ["理解你的问题", "确认地点与日期"]
    if plan.locations:
        steps.append("确认具体地点")
    if plan.needs_weather:
        steps.append("查询最新天气")
    if task_type in {"travel_assessment", "comparison"}:
        steps.append("基于温度、降水和风力进行判断")
    steps.append("整理结果并给出建议")
    emitter = _event_emitter.get()
    if emitter:
        emitter({"type": "plan", "task_type": task_type, "title": titles[task_type], "steps": steps})


class WeatherAgent:
    def __init__(self, settings: Settings, profiles: ProfileRepository, amap: AmapClient | None = None) -> None:
        self.settings = settings
        self.profiles = profiles
        self.amap = amap or AmapClient(
            settings.amap_api_key, timeout_seconds=settings.amap_timeout_seconds
        )
        self.timezone = ZoneInfo(settings.timezone)
        self.llm: ChatOpenAI | None = None
        if settings.deepseek_api_key:
            self.llm = ChatOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                temperature=0,
                timeout=25,
                max_retries=1,
            )

        checkpoint_path = Path(settings.database_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.checkpointer.setup()
        self.graph = self._build_graph()

    def close(self) -> None:
        self.amap.close()
        self._checkpoint_connection.close()

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_profile", self._load_profile)
        builder.add_node("plan", self._plan)
        builder.add_node("resolve", self._resolve)
        builder.add_node("weather", self._weather)
        builder.add_node("update_memory", self._update_memory)
        builder.add_node("respond", self._respond)
        builder.add_edge(START, "load_profile")
        builder.add_edge("load_profile", "plan")
        builder.add_conditional_edges("plan", self._after_plan)
        builder.add_conditional_edges("resolve", self._after_resolve)
        builder.add_edge("weather", "update_memory")
        builder.add_edge("update_memory", "respond")
        builder.add_edge("respond", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _load_profile(self, state: AgentState) -> AgentState:
        _emit("memory", "running", "读取你的偏好")
        profile = self.profiles.get_profile()
        _emit("memory", "complete", "已读取偏好")
        return {"profile": profile.model_dump()}

    def _explicit_memory(self, text: str) -> tuple[dict[str, str], list[str]]:
        update: dict[str, str] = {}
        locations: list[str] = []
        default_match = re.search(
            r"(?:以后|今后)?(?:默认|常用)(?:查|城市(?:是|为)?|地点(?:是|为)?)?\s*([\u4e00-\u9fff]{2,8})",
            text,
        )
        if default_match:
            value = re.sub(r"(?:的)?天气.*$", "", default_match.group(1)).strip()
            if value:
                update["default_location"] = value
                locations.append(value)
        preference_match = re.search(r"(?:记住|请记得)我(.{1,24})", text)
        if preference_match and "default_location" not in update:
            preference = preference_match.group(1).strip("，。！？,.!? ")
            if preference:
                update["advice_preference"] = preference
        return update, locations

    def _fallback_plan(self, text: str, state: AgentState) -> WeatherPlan:
        memory_update, locations = self._explicit_memory(text)
        unsupported = None
        limits = {
            r"下周|一周|七天|7天|未来(?:四|五|六|七|[4-9])天": "高德基础天气只提供当天起约三天预报，暂时无法查询更长期天气。",
            r"逐小时|每小时": "当前数据源不提供逐小时预报。",
            r"历史天气|去年|往年": "当前数据源不提供历史天气。",
            r"空气质量|AQI|雾霾": "当前数据源不提供空气质量信息。",
            r"灾害预警|气象预警|台风预警": "当前数据源不提供官方灾害预警。",
        }
        for pattern, reason in limits.items():
            if re.search(pattern, text, re.IGNORECASE):
                unsupported = reason
                break

        normalized = re.sub(r"[？?！!。]", "", text)
        if "default_location" not in memory_update:
            pattern = r"(?:^|[和与、,，])([\u4e00-\u9fff]{2,8}?)(?=今天|明天|后天|天气|哪里|[和与、,，])"
            for match in re.finditer(pattern, normalized):
                candidate = match.group(1).strip()
                candidate = re.sub(r"^(?:请问|帮我查|查询|看看|那)", "", candidate)
                if candidate and candidate not in {"今天", "明天", "后天", "适合出游", "适合"}:
                    locations.append(candidate)

        needs_weather = bool(
            re.search(r"天气|气温|温度|下雨|带伞|穿什么|出游|出行|旅游|适合.*(游|玩|出行)|值得|冷不冷|热不热|明天|后天", text)
        )
        pure_memory = bool(memory_update) and not needs_weather
        if pure_memory:
            needs_weather = False

        if not locations:
            previous = [ResolvedLocation.model_validate(item) for item in state.get("last_locations", [])]
            if previous and re.search(r"^(?:那|那么|再|它)|明天|后天|旅游|适合.*(游|玩|出行)|值得", text):
                locations = [item.query for item in previous]
            else:
                profile = UserProfile.model_validate(state["profile"])
                if profile.default_location and needs_weather:
                    locations = [profile.default_location.query]

        today = datetime.now(self.timezone).date()
        target_dates = []
        time_scope = "current"
        if "后天" in text:
            time_scope = "forecast"
            target_dates = [today + timedelta(days=2)]
        elif "明天" in text:
            time_scope = "forecast"
            target_dates = [today + timedelta(days=1)]
        elif re.search(r"未来|三天|预报", text):
            time_scope = "forecast"
            target_dates = [today + timedelta(days=index) for index in range(3)]
        elif "今天" in text:
            time_scope = "today"
            target_dates = [today]
        elif re.search(r"旅游|适合.*(游|玩|出行)|值得", text):
            time_scope = "today"
            target_dates = [today]

        advice_topics: list[str] = []
        for keyword in ("出游", "出行", "旅游", "穿什么", "带伞"):
            if keyword in text:
                advice_topics.append(keyword)
        locations = list(dict.fromkeys(locations))[:5]
        return WeatherPlan(
            task_type=("travel_assessment" if re.search(r"旅游|适合.*(游|玩|出行)|值得", text)
                       else "comparison" if len(locations) > 1 or "比较" in text or "哪里" in text
                       else "memory_update" if pure_memory else "weather_query"),
            locations=locations,
            time_scope=time_scope,
            target_dates=target_dates,
            compare=len(locations) > 1 or "比较" in text or "哪里" in text,
            advice_topics=advice_topics,
            needs_weather=needs_weather,
            unsupported_reason=unsupported,
            memory_update=memory_update,
        )

    def _llm_plan(self, text: str, state: AgentState) -> WeatherPlan | None:
        if self.llm is None:
            return None
        profile = UserProfile.model_validate(state["profile"])
        today = datetime.now(self.timezone).date().isoformat()
        previous = [ResolvedLocation.model_validate(item).query for item in state.get("last_locations", [])]
        prompt = (
            "你是中文天气助手的规划器。必须只输出一个 JSON 对象，不要 Markdown、解释或思考过程。"
            "该 JSON 必须符合下方 WeatherPlan schema。今天是"
            f"{today}（Asia/Shanghai）。识别最多5个地点、实时或未来三天范围、比较和建议主题。"
            "用户明确说‘记住/以后默认’时才填写 memory_update。"
            "基础数据不支持超过三天、逐小时、历史、AQI和灾害预警，遇到这些请求填写 unsupported_reason。"
            f"默认地点：{profile.default_location.model_dump() if profile.default_location else None}；"
            f"当前对话上次地点：{previous}。缺少地点时可使用上次地点，其次使用默认地点。"
            f"WeatherPlan JSON schema：{json.dumps(WeatherPlan.model_json_schema(), ensure_ascii=False)}"
        )
        return invoke_json_model(self.llm, WeatherPlan, [
            SystemMessage(content=prompt), HumanMessage(content=text),
        ])

    def _plan(self, state: AgentState) -> AgentState:
        _emit("plan", "running", "理解问题并制定计划")
        text = str(state["messages"][-1].content)
        fallback = self._fallback_plan(text, state)
        plan = self._llm_plan(text, state) or fallback
        # Follow-up questions often omit the place. Never let a valid model
        # plan erase the location already established in this conversation.
        previous_locations = [
            ResolvedLocation.model_validate(item).query
            for item in state.get("last_locations", [])
        ]
        contextual_followup = bool(
            re.search(r"旅游|适合.*(游|玩|出行)|值得|^(?:那|那么|再|它)", text)
        )
        if not plan.locations and (fallback.locations or (contextual_followup and previous_locations)):
            plan.locations = fallback.locations or previous_locations
        if contextual_followup and plan.locations:
            plan.needs_weather = True
            plan.advice_topics = list(dict.fromkeys(plan.advice_topics + fallback.advice_topics + ["旅游"]))
        explicit_update, explicit_locations = self._explicit_memory(text)
        if explicit_update:
            plan.memory_update.update(explicit_update)
            for location in explicit_locations:
                if location not in plan.locations:
                    plan.locations.insert(0, location)
        plan.locations = plan.locations[:5]
        if plan.task_type == "weather_query" and re.search(r"旅游|适合.*(游|玩|出行)|值得", text):
            plan.task_type = "travel_assessment"
        if plan.task_type == "weather_query" and (plan.compare or len(plan.locations) > 1):
            plan.task_type = "comparison"
        if plan.task_type == "weather_query" and not plan.needs_weather and plan.memory_update:
            plan.task_type = "memory_update"
        _emit_plan(plan)
        _emit("plan", "complete", "查询计划已生成")
        return {"plan": plan.model_dump(mode="json")}

    def _after_plan(self, state: AgentState) -> str:
        plan = WeatherPlan.model_validate(state["plan"])
        if plan.unsupported_reason:
            return "respond"
        if plan.locations:
            return "resolve"
        if plan.needs_weather:
            return "respond"
        return "update_memory"

    def _known_location(self, query: str, state: AgentState) -> ResolvedLocation | None:
        profile = UserProfile.model_validate(state["profile"])
        known = ([profile.default_location] if profile.default_location else []) + profile.favorite_locations
        known += [ResolvedLocation.model_validate(item) for item in state.get("last_locations", [])]
        for item in known:
            if item and query in {item.query, item.name, item.city, item.district}:
                return item
        return None

    def _resolve(self, state: AgentState) -> AgentState:
        _emit("location", "running", "解析地点")
        plan = WeatherPlan.model_validate(state["plan"])
        resolved: list[ResolvedLocation] = []
        for query in plan.locations:
            known = self._known_location(query, state)
            if known:
                resolved.append(known)
                continue
            try:
                candidates = self.amap.resolve_location(query)
            except AmapError as exc:
                _emit("location", "error", "地点解析失败")
                return {"errors": [str(exc)], "error_codes": [exc.code], "clarification": str(exc), "locations": []}
            if not candidates:
                message = f"没有找到“{query}”对应的行政区域，请补充省份或城市名称。"
                _emit("location", "needs_input", "需要更具体的地点")
                return {"clarification": message, "locations": []}
            exact = [
                item for item in candidates
                if query in {item.name, item.city.removesuffix("市"), item.district.removesuffix("区")}
            ]
            if len(candidates) > 1 and len(exact) != 1:
                options = "、".join(item.display_name for item in candidates[:4])
                message = f"“{query}”可能指多个地点：{options}。请告诉我具体省市。"
                _emit("location", "needs_input", "地点存在歧义")
                return {"clarification": message, "locations": []}
            resolved.append(exact[0] if len(exact) == 1 else candidates[0])
        _emit("location", "complete", f"已确认 {len(resolved)} 个地点")
        dumped = [item.model_dump() for item in resolved]
        return {"locations": dumped, "last_locations": dumped, "clarification": None}

    def _after_resolve(self, state: AgentState) -> str:
        if state.get("clarification"):
            return "respond"
        plan = WeatherPlan.model_validate(state["plan"])
        return "weather" if plan.needs_weather else "update_memory"

    def _weather(self, state: AgentState) -> AgentState:
        _emit("weather", "running", "查询高德实时天气")
        plan = WeatherPlan.model_validate(state["plan"])
        locations = [ResolvedLocation.model_validate(item) for item in state.get("locations", [])]
        tasks: list[tuple[str, ResolvedLocation]] = []
        for location in locations:
            if plan.time_scope in {"current", "today"}:
                tasks.append(("current", location))
            if plan.time_scope in {"forecast", "today"}:
                tasks.append(("forecast", location))
        snapshots: list[WeatherSnapshot] = []
        errors: list[str] = []
        error_codes: list[str] = []
        with ThreadPoolExecutor(max_workers=min(10, max(1, len(tasks)))) as executor:
            futures = {
                executor.submit(
                    self.amap.get_current_weather if kind == "current" else self.amap.get_forecast_weather,
                    location,
                ): (kind, location)
                for kind, location in tasks
            }
            for future in as_completed(futures):
                _, location = futures[future]
                try:
                    snapshots.extend(future.result())
                except AmapError as exc:
                    errors.append(f"{location.display_name}：{exc}")
                    error_codes.append(exc.code)

        target_dates = plan.target_dates
        if plan.time_scope == "today" and not target_dates:
            target_dates = [datetime.now(self.timezone).date()]
        if target_dates:
            wanted = {value.isoformat() for value in target_dates}
            snapshots = [item for item in snapshots if item.kind == "current" or item.date in wanted]
        snapshots.sort(key=lambda item: (item.location.adcode, item.date or "", item.kind))
        sources: list[SourceRecord] = []
        seen_sources: set[tuple[str, str, str]] = set()
        for item in snapshots:
            kind = "实时" if item.kind == "current" else "预报"
            key = (item.location.adcode, item.reporttime, kind)
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(
                    SourceRecord(location=item.location.display_name, reporttime=item.reporttime, kind=kind)
                )
        if snapshots:
            _emit("weather", "complete", "已取得高德天气数据")
        else:
            _emit("weather", "error", "天气服务暂时不可用")
        return {
            "weather": [item.model_dump() for item in snapshots],
            "last_locations": [item.model_dump() for item in locations],
            "sources": [item.model_dump() for item in sources],
            "errors": errors,
            "error_codes": error_codes,
        }

    def _update_memory(self, state: AgentState) -> AgentState:
        plan = WeatherPlan.model_validate(state["plan"])
        if not plan.memory_update:
            return {"memory_saved": []}
        _emit("save_memory", "running", "保存你明确要求记住的内容")
        profile = self.profiles.get_profile()
        saved: list[str] = []
        default_query = plan.memory_update.get("default_location")
        if default_query:
            resolved = [ResolvedLocation.model_validate(item) for item in state.get("locations", [])]
            match = next((item for item in resolved if item.query == default_query), None)
            if match:
                profile.default_location = match
                saved.append(f"默认地点：{match.display_name}")
        preference = plan.memory_update.get("advice_preference")
        if preference and preference not in profile.advice_preferences:
            profile.advice_preferences.append(preference)
            saved.append(f"建议偏好：{preference}")
        if saved:
            self.profiles.save_profile(profile)
            _emit("save_memory", "complete", "偏好已保存")
        else:
            _emit("save_memory", "complete", "没有新增记忆")
        return {"profile": profile.model_dump(), "memory_saved": saved}

    @staticmethod
    def _score(snapshot: WeatherSnapshot) -> int:
        score = 100
        if re.search(r"雨|雪|雷|沙|雾", snapshot.weather):
            score -= 35
        values = [snapshot.temperature, snapshot.day_temperature, snapshot.night_temperature]
        temperatures = [int(value) for value in values if value and re.fullmatch(r"-?\d+", value)]
        if temperatures:
            average = sum(temperatures) / len(temperatures)
            if average < 5 or average > 34:
                score -= 22
            elif average < 12 or average > 30:
                score -= 10
        if snapshot.wind_power and re.search(r"[6-9]|1\d", snapshot.wind_power):
            score -= 15
        return score

    def _fact_lines(self, snapshots: list[WeatherSnapshot], compare: bool) -> list[str]:
        lines: list[str] = []
        for item in snapshots:
            name = item.location.display_name
            if item.kind == "current":
                detail = f"{item.weather}，{item.temperature or '未知'}℃"
                if item.humidity:
                    detail += f"，湿度 {item.humidity}%"
                if item.wind_direction or item.wind_power:
                    detail += f"，{item.wind_direction or ''}风 {item.wind_power or ''}级".replace("风 风", "风")
                lines.append(f"- **{name}（实时）**：{detail}。")
            else:
                temperature = f"{item.night_temperature or '?'}～{item.day_temperature or '?'}℃"
                lines.append(f"- **{name}（{item.date}）**：{item.weather}，{temperature}，{item.wind_direction or '风向未知'}风 {item.wind_power or '?'}级。")
        if compare and snapshots:
            best = max(snapshots, key=self._score)
            lines.append(f"\n综合降水、温度和风力，**{best.location.display_name}** 当前更适合安排户外活动。")
        return lines

    def _respond(self, state: AgentState) -> AgentState:
        _emit("answer", "running", "整理天气和建议")
        plan = WeatherPlan.model_validate(state["plan"])
        if plan.unsupported_reason:
            answer = f"抱歉，{plan.unsupported_reason}目前可以查询实时天气和未来约三天的预报。"
        elif state.get("clarification"):
            answer = state["clarification"] or "请补充具体地点。"
        else:
            snapshots = [WeatherSnapshot.model_validate(item) for item in state.get("weather", [])]
            saved = state.get("memory_saved", [])
            if snapshots:
                facts = self._fact_lines(snapshots, plan.compare)
                answer = "我刚帮你看了天气，整理成下面这份容易安排行程的信息：\n\n" + "\n".join(facts)
                rainy = any(re.search(r"雨|雪|雷", item.weather) for item in snapshots)
                cold = any(
                    value and re.fullmatch(r"-?\d+", value) and int(value) < 12
                    for item in snapshots
                    for value in (item.temperature, item.day_temperature, item.night_temperature)
                )
                hot = any(
                    value and re.fullmatch(r"-?\d+", value) and int(value) > 32
                    for item in snapshots
                    for value in (item.temperature, item.day_temperature, item.night_temperature)
                )
                wind = any(
                    item.wind_power and re.search(r"[6-9]|1\d", item.wind_power)
                    for item in snapshots
                )
                current = next((item for item in snapshots if item.kind == "current"), None)
                forecast = next((item for item in snapshots if item.kind == "forecast"), None)
                if rainy and wind:
                    suitability = "有降水且风力较明显，建议减少登高、临水等户外安排，并准备室内备选。"
                elif rainy:
                    suitability = "可以出行，但最好避开降雨较强的时段，并准备一个室内备选。"
                elif cold or hot or wind:
                    suitability = "可以出行，不过温度或风力会影响体感，行程别排得太满。"
                else:
                    suitability = "天气比较适合出行，可以正常安排步行和户外活动。"
                clothing = "建议分层穿衣，早晚注意保暖。" if cold else "建议轻便透气穿着，并准备一件薄外套。" if hot else "建议按舒适层次穿衣，早晚带一件薄外套。"
                rain_tip = "带伞或轻便雨衣，并给景区交通预留缓冲时间。" if rainy else "目前没有明显降雨提示。"
                wind_tip = "风力较大时避免临水、登高等暴露性活动。" if wind else "风力整体温和，适合安排常规步行和观景。"
                answer += "\n\n出行建议：\n" + suitability + f" {clothing} {rain_tip} {wind_tip}"
                if forecast and current:
                    answer += f"\n- 今天趋势：白天约 {forecast.day_temperature or '未知'}℃，夜间约 {forecast.night_temperature or '未知'}℃，天气为{forecast.weather}。"
                answer += "\n\n天气信息来自高德。愿你出门时刚好遇见舒服的天气。"
            elif saved:
                answer = "好的，已记住：" + "；".join(saved) + "。之后的新对话也会使用这些偏好。"
            elif state.get("errors"):
                answer = "暂时没能取得高德天气数据：" + "；".join(state["errors"]) + "。请稍后重试。"
            elif plan.needs_weather and not plan.locations:
                answer = "你想查询哪个城市或区县？例如可以说“杭州明天天气怎么样”。"
            else:
                answer = "我可以查询实时天气和未来约三天预报，也能比较多个城市。请告诉我地点和时间。"
        for index in range(0, len(answer), 8):
            _emit_token(answer[index:index + 8])
            time.sleep(0.018)
        _emit("answer", "complete", "回答已生成")
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    def run(self, thread_id: str, message: str, emit: EventEmitter | None = None) -> dict[str, Any]:
        token = _event_emitter.set(emit)
        try:
            result = self.graph.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "plan": {},
                    "locations": [],
                    "weather": [],
                    "clarification": None,
                    "answer": "",
                    "sources": [],
                    "errors": [],
                    "error_codes": [],
                    "memory_saved": [],
                    "last_locations": self.profiles.get_last_locations(thread_id),
                },
                {"configurable": {"thread_id": thread_id}},
            )
            self.profiles.save_last_locations(thread_id, result.get("last_locations", []))
            return {
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "profile": result.get("profile", {}),
                "error_code": (result.get("error_codes") or [None])[0],
            }
        finally:
            _event_emitter.reset(token)
