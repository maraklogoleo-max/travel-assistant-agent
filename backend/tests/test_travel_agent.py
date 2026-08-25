from pathlib import Path

from app.config import Settings
from app.db import ProfileRepository
from app.models import POI, ResolvedLocation, RouteLeg, TripPlan, TripRequest, WeatherSnapshot
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
    assert "费用方面" not in result["answer"]
    assert "出发前请留意" not in result["answer"]
    assert profiles.get_trip("trip-1").version == 2
    agent.close()
