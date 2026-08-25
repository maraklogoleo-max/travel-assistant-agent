from datetime import date, timedelta
from pathlib import Path

from app.config import Settings
from app.db import ProfileRepository
from app.models import Activity, DayPlan, POI, ResolvedLocation, RouteLeg, TripPlan, TripRequest, WeatherSnapshot
from app.travel_agent import TravelAgent


class FakeTravelAmap:
    def close(self) -> None:
        pass

    def resolve_location(self, query: str):
        return [ResolvedLocation(query=query, name=query, city=query, adcode="513225", longitude=103.8, latitude=33.2)]

    def search_places(self, keywords: str, **kwargs):
        return [POI(id=f"{keywords}-1", name=f"{keywords}体验馆", type=keywords, city="513225", location="103.8,33.2", longitude=103.8, latitude=33.2)]

    def get_forecast_weather(self, location):
        return [WeatherSnapshot(location=location, kind="forecast", reporttime="2026-08-25 18:00:00", date="2026-08-25", weather="多云", day_temperature="24", night_temperature="12")]

    def plan_route(self, origin, destination, mode="transit"):
        return RouteLeg(origin=origin.name, destination=destination.name, mode=mode, distance_m=1200, duration_s=900, summary="模拟路线")


def test_travel_agent_creates_and_updates_itinerary(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "travel.db", deepseek_api_key="", amap_api_key="test")
    profiles = ProfileRepository(settings.database_path)
    profiles.initialize()
    agent = TravelAgent(settings, profiles, amap=FakeTravelAmap())
    request = TripRequest(destination="九寨沟", days=3, pace="relaxed", interests=["自然风景"])
    trip = TripPlan(trip_id="trip-1", name="九寨沟之旅", request=request)
    profiles.create_trip("trip-1", request, trip)
    events: list[dict] = []
    result = agent.run("trip-1", "去九寨沟玩三天，喜欢自然风景，节奏轻松", events.append)
    assert result["trip"]["status"] == "ready"
    assert len(result["trip"]["days"]) == 3
    assert any(event["type"] == "tool_start" and event["tool"] == "search_places" for event in events)
    assert any(event["type"] == "itinerary_patch" for event in events)
    assert {item["kind"] for item in result["sources"]} >= {"地点", "预报"}
    assert result["trip"]["days"][0]["activities"][0]["sources"][0]["resource_id"]
    assert "费用方面" not in result["answer"]
    assert "出发前请留意" not in result["answer"]
    assert "符合你的兴趣、行程节奏和相邻区域安排" not in result["answer"]
    assert "如果想修改" not in result["answer"]
    assert "好呀，这趟 九寨沟 3 天行程已经为你排好" in result["answer"]
    assert "第一天先把脚步放慢" in result["answer"]
    assert profiles.get_trip("trip-1").version == 2
    agent.close()


def test_local_update_preserves_trip_length_and_replaces_only_target_day(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "local-update.db", deepseek_api_key="", amap_api_key="test")
    profiles = ProfileRepository(settings.database_path)
    profiles.initialize()
    agent = TravelAgent(settings, profiles, amap=FakeTravelAmap())
    request = TripRequest(destination="九寨沟", days=3, pace="relaxed", interests=["自然风景"])
    days = [
        DayPlan(
            date=date(2026, 8, 25) + timedelta(days=index),
            activities=[Activity(id=f"day-{index}", date=date(2026, 8, 25) + timedelta(days=index), period="morning", poi=POI(id=f"scenic-{index}", name=f"原景点{index}", type="风景名胜", location="103.8,33.2", longitude=103.8, latitude=33.2))],
        )
        for index in range(3)
    ]
    trip = TripPlan(trip_id="trip-local-update", name="九寨沟之旅", request=request, days=days)
    profiles.create_trip(trip.trip_id, request, trip)

    updated = agent.run(trip.trip_id, "把第二天换成亲子景点", expected_version=trip.version)

    assert updated["trip"]["request"]["days"] == 3
    assert len(updated["trip"]["days"]) == 3
    assert updated["trip"]["days"][1]["activities"][0]["poi"]["name"] == "亲子乐园体验馆"
    assert updated["trip"]["days"][0]["activities"][0]["poi"]["name"] != "亲子乐园体验馆"
    assert profiles.get_trip(trip.trip_id).version == 2
    agent.close()


def test_non_food_trip_does_not_schedule_fast_food_as_an_activity(tmp_path: Path) -> None:
    class ScenicWithFoodAmap(FakeTravelAmap):
        def search_places(self, keywords: str, **kwargs):
            return [
                POI(id="mcd", name="麦当劳", type="餐饮|快餐", city="513225", location="103.8,33.2", longitude=103.8, latitude=33.2),
                POI(id=f"{keywords}-scenic", name="九寨沟自然风景区", type="风景名胜", city="513225", location="103.81,33.21", longitude=103.81, latitude=33.21),
            ]

    settings = Settings(database_path=tmp_path / "no-food.db", deepseek_api_key="", amap_api_key="test")
    profiles = ProfileRepository(settings.database_path)
    profiles.initialize()
    agent = TravelAgent(settings, profiles, amap=ScenicWithFoodAmap())
    trip = TripPlan(
        trip_id="trip-no-food", name="九寨沟之旅",
        request=TripRequest(destination="九寨沟", days=1, interests=["自然风景"]),
    )
    profiles.create_trip(trip.trip_id, trip.request, trip)

    result = agent.run(trip.trip_id, "安排自然风景", expected_version=1)

    names = [item["poi"]["name"] for item in result["trip"]["days"][0]["activities"]]
    assert "麦当劳" not in names
    agent.close()
