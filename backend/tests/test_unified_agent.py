from datetime import date
from pathlib import Path

from app.config import Settings
from app.db import ProfileRepository, TripVersionConflict
from app.models import (
    POI,
    Activity,
    AgentPlan,
    DayPlan,
    ResolvedLocation,
    RouteLeg,
    TripPlan,
    TripRequest,
    WeatherSnapshot,
)
from app.travel_agent import TravelAgent
from app.unified_agent import TravelAssistantAgent


class FakeWeatherAgent:
    def run(self, conversation_id, message, emit):
        emit({"type": "tool_start", "tool": "weather", "status": "running", "label": "查询天气"})
        emit({"type": "tool_result", "tool": "weather", "status": "complete", "label": "天气已返回"})
        return {"answer": "杭州明天多云。", "sources": [{"provider": "高德开放平台", "location": "杭州", "reporttime": "2026-08-25 18:00:00", "kind": "预报"}]}


class FakeTravelAmap:
    def close(self):
        pass

    def resolve_location(self, query):
        return [ResolvedLocation(query=query, name=query, city=query, adcode="513225", longitude=103.8, latitude=33.2)]

    def search_places(self, keywords, **kwargs):
        name = "九寨沟博物馆" if keywords == "博物馆" else f"{keywords}体验地"
        return [
            POI(
                id=f"{keywords}-{index}", name=f"{name}{index}", type=keywords,
                city="513225", location=f"103.8{index},33.2{index}",
                longitude=103.8 + index / 100, latitude=33.2 + index / 100,
            )
            for index in range(1, 4)
        ]

    def get_forecast_weather(self, location):
        return [WeatherSnapshot(location=location, kind="forecast", reporttime="2026-08-25 18:00:00", date="2026-08-25", weather="中雨", day_temperature="20", night_temperature="12")]

    def plan_route(self, origin, destination, mode="transit"):
        return RouteLeg(origin=origin.name, destination=destination.name, mode=mode, distance_m=1200, duration_s=900, summary="模拟路线")


def build_agent(tmp_path: Path):
    settings = Settings(database_path=tmp_path / "unified.db", deepseek_api_key="", amap_api_key="test")
    profiles = ProfileRepository(settings.database_path)
    profiles.initialize()
    travel = TravelAgent(settings, profiles, amap=FakeTravelAmap())
    assistant = TravelAssistantAgent(settings, profiles, FakeWeatherAgent(), travel)
    profiles.create_conversation("conversation-1")
    return profiles, travel, assistant


def test_plain_weather_uses_unified_agent_without_creating_trip(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    events = []
    result = assistant.run("杭州明天天气怎么样", conversation_id="conversation-1", emit_callback=events.append)
    assert result["answer"] == "杭州明天多云。"
    assert profiles.list_trips() == []
    assert any(item["type"] == "plan" and item["task_type"] == "weather_query" for item in events)
    assert any(item["type"] == "agent_action" and item["action"] == "weather" for item in events)
    assert len(profiles.get_conversation_messages("conversation-1")) == 2
    travel.close()


def test_explicit_three_day_request_replaces_active_context_and_day_followup(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    old_trip = TripPlan(
        trip_id="old-one-day", name="旧的一日行程",
        request=TripRequest(destination="九寨沟", days=1), status="ready",
    )
    profiles.create_trip(old_trip.trip_id, old_trip.request, old_trip)
    create_events = []

    created = assistant.run(
        "帮我安排3天的九寨沟行程",
        conversation_id="conversation-1", trip_id=old_trip.trip_id,
        expected_version=1, emit_callback=create_events.append,
    )

    assert created["trip"]["trip_id"] != old_trip.trip_id
    assert created["trip"]["request"]["days"] == 3
    assert len(created["trip"]["days"]) == 3
    assert any(item["type"] == "plan" and item["task_type"] == "trip_create" for item in create_events)
    actions = [item["action"] for item in create_events if item["type"] == "agent_action"]
    assert actions[:2] == ["resolve_location", "trip_create"]
    new_trip_id = created["trip"]["trip_id"]
    new_version = created["trip"]["version"]

    followup_events = []
    followup = assistant.run(
        "第二天呢", conversation_id="conversation-1", trip_id=new_trip_id,
        expected_version=new_version, emit_callback=followup_events.append,
    )

    assert "第 2 天" in followup["answer"]
    assert followup["trip"]["version"] == new_version
    assert "一共 3 天" not in followup["answer"]
    assert any(item["type"] == "plan" and item["task_type"] == "trip_query" for item in followup_events)
    assert profiles.get_trip(new_trip_id).version == new_version
    travel.close()


def test_ambiguous_destination_stops_before_trip_creation(tmp_path: Path) -> None:
    class AmbiguousAmap(FakeTravelAmap):
        def resolve_location(self, query):
            if "北京" in query:
                return [ResolvedLocation(query=query, name="朝阳区", province="北京市", city="北京市", district="朝阳区", adcode="110105")]
            return [
                ResolvedLocation(query=query, name="朝阳区", province="北京市", city="北京市", district="朝阳区", adcode="110105"),
                ResolvedLocation(query=query, name="朝阳市", province="辽宁省", city="朝阳市", adcode="211300"),
            ]

    settings = Settings(database_path=tmp_path / "ambiguous.db", deepseek_api_key="", amap_api_key="test")
    profiles = ProfileRepository(settings.database_path)
    profiles.initialize()
    profiles.create_conversation("ambiguous-conversation")
    travel = TravelAgent(settings, profiles, amap=AmbiguousAmap())
    assistant = TravelAssistantAgent(settings, profiles, FakeWeatherAgent(), travel)
    events: list[dict] = []

    result = assistant.run(
        "帮我安排三天朝阳行程", conversation_id="ambiguous-conversation", emit_callback=events.append,
    )

    assert result["error_code"] == "LOCATION_AMBIGUOUS"
    assert profiles.list_trips() == []
    assert any(item["type"] == "clarification" and item["code"] == "LOCATION_AMBIGUOUS" for item in events)

    continued = assistant.run(
        "北京市朝阳区", conversation_id="ambiguous-conversation", emit_callback=lambda _: None,
    )
    assert continued["trip"]["request"]["days"] == 3
    assert continued["trip"]["request"]["destination"] == "北京市朝阳区"
    travel.close()


def test_pending_trip_plan_accepts_a_later_date_reply(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    summary = profiles.get_conversation_summary("conversation-1")
    summary.pending_plan = AgentPlan(
        intent="trip_create", objective="安排九寨沟三天行程", tools=["resolve_location", "trip"],
        requirements={"destination": "九寨沟", "days": 3}, missing_fields=["start_date"],
    )
    profiles.save_conversation_summary(summary)

    result = assistant.run(
        "9月3号出发", conversation_id="conversation-1", emit_callback=lambda _: None,
    )

    assert result["trip"]["request"]["destination"] == "九寨沟"
    assert result["trip"]["request"]["days"] == 3
    assert result["trip"]["request"]["start_date"].endswith("-09-03")
    assert profiles.get_conversation_summary("conversation-1").pending_plan is None
    travel.close()


def test_explicit_trip_details_are_not_asked_for_again(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    summary = profiles.get_conversation_summary("conversation-1")
    # This mirrors an older model response stored by the running application:
    # Chinese labels plus optional fields incorrectly marked as missing.
    summary.pending_plan = AgentPlan(
        intent="trip_create", objective="安排九寨沟行程", tools=["places", "routes"],
        requirements={"destination": "九寨沟", "duration_days": 3},
        missing_fields=["出行日期", "出发地点", "住宿偏好", "交通方式"],
    )
    profiles.save_conversation_summary(summary)

    result = assistant.run(
        "8月29号去九寨沟，从无锡出发，住民宿，坐高铁",
        conversation_id="conversation-1", emit_callback=lambda _: None,
    )

    request = result["trip"]["request"]
    assert request["destination"] == "九寨沟"
    assert request["origin"] == "无锡"
    assert request["days"] == 3
    assert request["start_date"].endswith("-08-29")
    assert profiles.get_conversation_summary("conversation-1").pending_plan is None
    travel.close()


def test_trip_creation_asks_for_missing_duration_instead_of_defaulting_to_one_day(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    result = assistant.run(
        "8月29号去九寨沟，从无锡出发，住民宿，坐高铁",
        conversation_id="conversation-1", emit_callback=lambda _: None,
    )

    assert result.get("error_code") == "MISSING_REQUIREMENTS"
    assert "游玩天数" in result["answer"]
    assert "高铁" in result["answer"]
    assert profiles.list_trips() == []
    travel.close()


def test_trip_weather_creates_confirmable_proposal_and_context(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    scenic = POI(id="scenic", name="九寨沟风景区", type="风景名胜", location="103.8,33.2", longitude=103.8, latitude=33.2)
    day = DayPlan(date=date(2026, 8, 25), activities=[Activity(id="a1", date=date(2026, 8, 25), period="morning", poi=scenic, indoor=False)])
    trip = TripPlan(trip_id="trip-1", name="九寨沟之旅", request=TripRequest(destination="九寨沟", days=1), days=[day], status="ready")
    profiles.create_trip(trip.trip_id, trip.request, trip)
    events = []

    result = assistant.run("这天下雨会影响行程吗", conversation_id="conversation-1", trip_id=trip.trip_id, expected_version=1, emit_callback=events.append)

    assert result["trip"]["version"] == 1
    pending = profiles.get_pending_proposals(trip.trip_id)
    assert len(pending) == 1
    assert "九寨沟风景区 → 九寨沟博物馆" in pending[0].changes[0]
    assert profiles.get_trip(trip.trip_id).version == 1
    assert len(profiles.get_trip_messages(trip.trip_id)) == 2
    assert profiles.get_trip_summary(trip.trip_id).confirmed_requirements
    assert any(item["type"] == "change_proposal" for item in events)
    assert any(item.get("tool") == "routes" for item in events)

    applied = profiles.save_trip(pending[0].proposed_plan, expected_version=pending[0].based_on_version)
    assert applied.version == 2
    try:
        profiles.save_trip(pending[0].proposed_plan, expected_version=1)
        raise AssertionError("expected a version conflict")
    except TripVersionConflict as exc:
        assert exc.current_version == 2
    travel.close()


def test_suitability_question_uses_selected_trip_weather_context(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    scenic = POI(id="scenic", name="九寨沟风景区", type="风景名胜", location="103.8,33.2", longitude=103.8, latitude=33.2)
    day = DayPlan(date=date(2026, 8, 25), activities=[Activity(id="a1", date=date(2026, 8, 25), period="morning", poi=scenic, indoor=False)])
    trip = TripPlan(trip_id="trip-suitability", name="九寨沟之旅", request=TripRequest(destination="九寨沟", days=1), days=[day], status="ready")
    profiles.create_trip(trip.trip_id, trip.request, trip)
    events = []

    result = assistant.run(
        "那适合旅游吗", conversation_id="conversation-1", trip_id=trip.trip_id,
        expected_version=1, emit_callback=events.append,
    )

    assert "可以出行" in result["answer"]
    assert "中雨" in result["answer"]
    assert any(item["type"] == "plan" and item["task_type"] == "trip_weather_assessment" for item in events)
    assert any(item["type"] == "agent_action" and item["action"] == "trip_weather_assessment" for item in events)
    travel.close()


def test_explicit_outdoor_preference_overrides_rain_replacement(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    scenic = POI(id="scenic", name="九寨沟风景区", type="风景名胜", location="103.8,33.2", longitude=103.8, latitude=33.2)
    day = DayPlan(date=date(2026, 8, 25), weather_summary="中雨", activities=[Activity(id="a1", date=date(2026, 8, 25), period="morning", poi=scenic, indoor=True)])
    trip = TripPlan(trip_id="trip-keep-outdoor", name="九寨沟之旅", request=TripRequest(destination="九寨沟", days=1), days=[day], status="ready")
    profiles.create_trip(trip.trip_id, trip.request, trip)

    events = []
    result = assistant.run(
        "下雨天也要户外活动", conversation_id="conversation-1", trip_id=trip.trip_id,
        expected_version=1, emit_callback=events.append,
    )

    assert "保留户外活动" in result["answer"]
    assert result["trip"]["days"][0]["activities"][0]["indoor"] is False
    assert any(item["type"] == "plan" and item["task_type"] == "trip_update" for item in events)
    assert not any(item.get("step") == "replan" for item in events)
    assert profiles.get_pending_proposals(trip.trip_id) == []
    assert profiles.get_trip(trip.trip_id).version == 2
    travel.close()


def test_rainy_indoor_to_outdoor_request_updates_the_trip(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    museum = POI(id="museum", name="室内博物馆", type="博物馆", location="103.8,33.2", longitude=103.8, latitude=33.2)
    day = DayPlan(date=date(2026, 8, 25), weather_summary="中雨", activities=[Activity(id="a1", date=date(2026, 8, 25), period="morning", poi=museum, indoor=True)])
    trip = TripPlan(trip_id="trip-indoor-outdoor", name="九寨沟之旅", request=TripRequest(destination="九寨沟", days=1), days=[day], status="ready")
    profiles.create_trip(trip.trip_id, trip.request, trip)
    events = []

    result = assistant.run(
        "下雨时把室内活动换成户外活动", conversation_id="conversation-1", trip_id=trip.trip_id,
        expected_version=1, emit_callback=events.append,
    )

    assert result["trip"]["days"][0]["activities"][0]["indoor"] is False
    assert "室内活动换为" in result["answer"]
    assert any(item["type"] == "plan" and item["task_type"] == "trip_update" for item in events)
    assert not any(item.get("step") == "replan" for item in events)
    travel.close()


def test_delete_trip_cascades_messages_summary_and_proposals(tmp_path: Path) -> None:
    profiles, travel, assistant = build_agent(tmp_path)
    scenic = POI(id="scenic", name="九寨沟风景区", type="风景名胜", location="103.8,33.2", longitude=103.8, latitude=33.2)
    day = DayPlan(date=date(2026, 8, 25), activities=[Activity(id="a1", date=date(2026, 8, 25), period="morning", poi=scenic)])
    trip = TripPlan(trip_id="trip-delete", name="待删除", request=TripRequest(destination="九寨沟"), days=[day])
    profiles.create_trip(trip.trip_id, trip.request, trip)
    assistant.run("下雨会影响吗", conversation_id="conversation-1", trip_id=trip.trip_id, emit_callback=lambda _: None)
    assert profiles.get_trip_messages(trip.trip_id)
    assert profiles.get_pending_proposals(trip.trip_id)

    profiles.delete_trip(trip.trip_id)

    assert profiles.get_trip(trip.trip_id) is None
    assert profiles.get_trip_messages(trip.trip_id) == []
    assert profiles.get_pending_proposals(trip.trip_id) == []
    travel.close()


def test_conversation_memory_compacts_old_messages_into_summary(tmp_path: Path) -> None:
    profiles, travel, _ = build_agent(tmp_path)
    for index in range(45):
        profiles.add_conversation_message("conversation-1", "user", f"第 {index} 条旅行条件", "旅行条件")

    assert len(profiles.get_conversation_messages("conversation-1")) == 40
    assert profiles.get_conversation_summary("conversation-1").recent_topics
    travel.close()
