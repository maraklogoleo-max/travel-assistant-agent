from pathlib import Path

from app.agent import WeatherAgent
from app.config import Settings
from app.db import ProfileRepository
from app.models import ResolvedLocation, WeatherSnapshot


class FakeAmap:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def close(self) -> None:
        pass

    def resolve_location(self, query: str):
        self.calls.append(f"resolve:{query}")
        codes = {"杭州": "330100", "北京": "110000", "上海": "310000", "成都": "510100", "九寨沟": "513225"}
        return [ResolvedLocation(query=query, name=f"{query}市", city=f"{query}市", adcode=codes[query])]

    def get_current_weather(self, location: ResolvedLocation):
        self.calls.append(f"current:{location.query}")
        return [WeatherSnapshot(
            location=location, kind="current", reporttime="2026-08-25 20:00:00",
            weather="晴", temperature="28", humidity="50", wind_direction="东", wind_power="3",
        )]

    def get_forecast_weather(self, location: ResolvedLocation):
        self.calls.append(f"forecast:{location.query}")
        return [WeatherSnapshot(
            location=location, kind="forecast", reporttime="2026-08-25 18:00:00",
            date="2026-08-26", weather="多云", day_temperature="30", night_temperature="22",
            wind_direction="东", wind_power="3",
        )]


def make_agent(tmp_path: Path) -> tuple[WeatherAgent, FakeAmap, ProfileRepository]:
    settings = Settings(database_path=tmp_path / "weather.db", deepseek_api_key="", amap_api_key="test")
    profiles = ProfileRepository(settings.database_path)
    profiles.initialize()
    fake = FakeAmap()
    return WeatherAgent(settings, profiles, amap=fake), fake, profiles


def test_weather_query_must_call_amap(tmp_path: Path) -> None:
    agent, amap, profiles = make_agent(tmp_path)
    profiles.create_conversation("thread-1")
    result = agent.run("thread-1", "杭州今天天气怎么样")
    assert "天气信息来自高德" in result["answer"]
    assert "resolve:杭州" in amap.calls
    assert "current:杭州" in amap.calls
    assert result["sources"][0]["reporttime"] == "2026-08-25 20:00:00"
    agent.close()


def test_explicit_default_location_persists_across_threads(tmp_path: Path) -> None:
    agent, amap, profiles = make_agent(tmp_path)
    profiles.create_conversation("thread-a")
    first = agent.run("thread-a", "以后默认查成都")
    assert "已记住" in first["answer"]
    assert profiles.get_profile().default_location.query == "成都"
    profiles.create_conversation("thread-b")
    second = agent.run("thread-b", "明天呢")
    assert "forecast:成都" in amap.calls
    assert "2026-08-26" in second["answer"]
    agent.close()


def test_missing_location_requests_clarification(tmp_path: Path) -> None:
    agent, _, profiles = make_agent(tmp_path)
    profiles.create_conversation("thread-1")
    result = agent.run("thread-1", "天气怎么样")
    assert "哪个城市" in result["answer"]
    agent.close()


def test_travel_followup_reuses_previous_location(tmp_path: Path) -> None:
    agent, amap, profiles = make_agent(tmp_path)
    profiles.create_conversation("thread-1")
    agent.run("thread-1", "九寨沟今天天气怎么样")
    result = agent.run("thread-1", "适合旅游吗")
    assert "九寨沟" in result["answer"]
    assert "出行建议" in result["answer"]
    assert amap.calls.count("current:九寨沟") == 2
    agent.close()


def test_agent_emits_structured_plan_for_travel_task(tmp_path: Path) -> None:
    agent, _, profiles = make_agent(tmp_path)
    profiles.create_conversation("thread-1")
    events: list[dict] = []
    agent.run("thread-1", "杭州适合旅游吗", emit=events.append)
    plan = next(event for event in events if event["type"] == "plan")
    assert plan["task_type"] == "travel_assessment"
    assert "查询最新天气" in plan["steps"]
    agent.close()


def test_unsupported_data_is_not_fabricated(tmp_path: Path) -> None:
    agent, amap, profiles = make_agent(tmp_path)
    profiles.create_conversation("thread-1")
    result = agent.run("thread-1", "杭州下周七天天气")
    assert "无法查询更长期天气" in result["answer"]
    assert amap.calls == []
    agent.close()
