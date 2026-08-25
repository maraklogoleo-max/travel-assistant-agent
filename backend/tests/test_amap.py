import httpx
import pytest

from app.amap import AmapClient, AmapError
from app.models import POI, ResolvedLocation


def test_resolve_and_weather_normalize_missing_array_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/geocode/geo"):
            return httpx.Response(
                200,
                json={
                    "status": "1",
                    "geocodes": [{
                        "province": "浙江省", "city": "杭州市", "district": [],
                        "adcode": "330100",
                    }],
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "1",
                "lives": [{
                    "weather": "晴", "temperature": "28", "humidity": "50",
                    "winddirection": [], "windpower": "3", "reporttime": "2026-08-25 20:00:00",
                }],
            },
        )

    client = AmapClient("test-key", transport=httpx.MockTransport(handler), sleep=lambda _: None)
    locations = client.resolve_location("杭州")
    assert locations[0].adcode == "330100"
    weather = client.get_current_weather(locations[0])
    assert weather[0].weather == "晴"
    assert weather[0].wind_direction is None


def test_retry_then_success_and_cache() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(500, text="unavailable")
        return httpx.Response(200, json={"status": "1", "lives": [{
            "weather": "多云", "temperature": "26", "reporttime": "2026-08-25 20:00:00"
        }]})

    client = AmapClient("test-key", transport=httpx.MockTransport(handler), sleep=lambda _: None)
    location = ResolvedLocation(query="杭州", name="杭州市", city="杭州市", adcode="330100")
    assert client.get_current_weather(location)[0].temperature == "26"
    assert client.get_current_weather(location)[0].temperature == "26"
    assert calls == 3


def test_amap_failure_is_explicit() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"status": "0", "info": "INVALID_USER_KEY", "infocode": "10001"})
    )
    client = AmapClient("bad-key", transport=transport, sleep=lambda _: None)
    with pytest.raises(AmapError, match="INVALID_USER_KEY"):
        client.resolve_location("杭州")


def test_search_places_and_route_normalize_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/place/text"):
            return httpx.Response(200, json={"status": "1", "pois": [{
                "id": "B1", "name": "西湖", "type": "风景名胜", "address": "杭州",
                "location": "120.15,30.24", "cityname": "杭州市",
            }]})
        return httpx.Response(200, json={"status": "1", "route": {"paths": [{"distance": "1200", "duration": "900"}]}})

    client = AmapClient("test-key", transport=httpx.MockTransport(handler), sleep=lambda _: None)
    pois = client.search_places("景点", city="330100")
    assert pois[0].name == "西湖"
    route = client.plan_route(pois[0], POI(id="B2", name="断桥", location="120.16,30.25"), "walking")
    assert route.duration_s == 900
