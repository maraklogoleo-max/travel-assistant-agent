import time
from collections.abc import Callable
from threading import Lock
from typing import Any

import httpx

from .models import POI, ResolvedLocation, RouteLeg, WeatherSnapshot


class AmapError(RuntimeError):
    def __init__(self, message: str, *, code: str = "AMAP_ERROR") -> None:
        super().__init__(message)
        self.code = code


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _coordinates(value: Any) -> tuple[float | None, float | None]:
    raw = _text(value)
    try:
        longitude, latitude = raw.split(",", 1)
        return float(longitude), float(latitude)
    except (ValueError, AttributeError):
        return None, None


class TTLCache:
    def __init__(self) -> None:
        self._values: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._values.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= time.monotonic():
                self._values.pop(key, None)
                return None
            return value

    def put(self, key: str, value: Any, ttl_seconds: int) -> Any:
        with self._lock:
            self._values[key] = (time.monotonic() + ttl_seconds, value)
        return value


class AmapClient:
    base_url = "https://restapi.amap.com"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.sleep = sleep
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": "WeatherHelperAgent/0.1"},
        )
        self.cache = TTLCache()

    def close(self) -> None:
        self.client.close()

    def _request(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        if not self.api_key:
            raise AmapError("尚未配置高德 Web 服务 API Key。", code="AMAP_NOT_CONFIGURED")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.get(
                    path,
                    params={**params, "key": self.api_key, "output": "JSON"},
                )
                response.raise_for_status()
                payload = response.json()
                if str(payload.get("status")) != "1":
                    info = _text(payload.get("info")) or "高德服务返回失败"
                    code = _text(payload.get("infocode")) or "AMAP_ERROR"
                    raise AmapError(info, code=code)
                return payload
            except (httpx.HTTPError, ValueError, AmapError) as exc:
                last_error = exc
                if isinstance(exc, AmapError) and exc.code not in {
                    "10020", "10021", "10044", "10045", "AMAP_ERROR"
                }:
                    break
                if attempt < 2:
                    self.sleep((0.25, 0.75)[attempt])
        if isinstance(last_error, AmapError):
            raise last_error
        raise AmapError("连接高德天气服务失败，请稍后重试。", code="AMAP_UNAVAILABLE") from last_error

    def resolve_location(self, query: str) -> list[ResolvedLocation]:
        cache_key = f"geo:{query.strip()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        payload = self._request("/v3/geocode/geo", {"address": query.strip()})
        results: list[ResolvedLocation] = []
        seen: set[str] = set()
        for item in payload.get("geocodes") or []:
            if not isinstance(item, dict):
                continue
            adcode = _text(item.get("adcode"))
            if not adcode or adcode in seen:
                continue
            seen.add(adcode)
            province = _text(item.get("province"))
            city = _text(item.get("city"))
            district = _text(item.get("district"))
            longitude, latitude = _coordinates(item.get("location"))
            name = district or city or province or query.strip()
            results.append(
                ResolvedLocation(
                    query=query.strip(), name=name, province=province,
                    city=city, district=district, adcode=adcode,
                    longitude=longitude, latitude=latitude,
                )
            )
        return self.cache.put(cache_key, results[:5], 24 * 60 * 60)

    def get_current_weather(self, location: ResolvedLocation) -> list[WeatherSnapshot]:
        cache_key = f"current:{location.adcode}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        payload = self._request(
            "/v3/weather/weatherInfo", {"city": location.adcode, "extensions": "base"}
        )
        snapshots: list[WeatherSnapshot] = []
        for item in payload.get("lives") or []:
            if not isinstance(item, dict):
                continue
            snapshots.append(
                WeatherSnapshot(
                    location=location,
                    kind="current",
                    reporttime=_text(item.get("reporttime")) or "未知",
                    weather=_text(item.get("weather")) or "未知",
                    temperature=_text(item.get("temperature")) or None,
                    humidity=_text(item.get("humidity")) or None,
                    wind_direction=_text(item.get("winddirection")) or None,
                    wind_power=_text(item.get("windpower")) or None,
                )
            )
        return self.cache.put(cache_key, snapshots, 10 * 60)

    def get_forecast_weather(self, location: ResolvedLocation) -> list[WeatherSnapshot]:
        cache_key = f"forecast:{location.adcode}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        payload = self._request(
            "/v3/weather/weatherInfo", {"city": location.adcode, "extensions": "all"}
        )
        snapshots: list[WeatherSnapshot] = []
        for forecast in payload.get("forecasts") or []:
            if not isinstance(forecast, dict):
                continue
            reporttime = _text(forecast.get("reporttime")) or "未知"
            for cast in (forecast.get("casts") or [])[:3]:
                if not isinstance(cast, dict):
                    continue
                day_weather = _text(cast.get("dayweather")) or "未知"
                night_weather = _text(cast.get("nightweather")) or "未知"
                weather = day_weather if day_weather == night_weather else f"{day_weather}转{night_weather}"
                day_wind = _text(cast.get("daywind"))
                night_wind = _text(cast.get("nightwind"))
                day_power = _text(cast.get("daypower"))
                night_power = _text(cast.get("nightpower"))
                snapshots.append(
                    WeatherSnapshot(
                        location=location,
                        kind="forecast",
                        reporttime=reporttime,
                        date=_text(cast.get("date")) or None,
                        weather=weather,
                        day_temperature=_text(cast.get("daytemp")) or None,
                        night_temperature=_text(cast.get("nighttemp")) or None,
                        wind_direction=day_wind if day_wind == night_wind else f"{day_wind}转{night_wind}".strip("转"),
                        wind_power=day_power if day_power == night_power else f"{day_power}转{night_power}".strip("转"),
                    )
                )
        return self.cache.put(cache_key, snapshots, 60 * 60)

    def search_places(
        self, keywords: str, *, city: str | None = None, location: str | None = None,
        types: str | None = None, radius: int = 5000,
    ) -> list[POI]:
        cache_key = f"poi:{keywords}:{city}:{location}:{types}:{radius}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        params = {"keywords": keywords, "offset": "20", "page": "1", "extensions": "all"}
        if city:
            params["city"] = city
        if location:
            params["location"] = location
            params["radius"] = str(radius)
            path = "/v3/place/around"
        else:
            path = "/v3/place/text"
        if types:
            params["types"] = types
        payload = self._request(path, params)
        results: list[POI] = []
        for item in payload.get("pois") or []:
            if not isinstance(item, dict):
                continue
            longitude, latitude = _coordinates(item.get("location"))
            results.append(POI(
                id=_text(item.get("id")) or f"poi-{len(results)}",
                name=_text(item.get("name")) or "未命名地点",
                type=_text(item.get("type")), address=_text(item.get("address")),
                city=_text(item.get("cityname")), location=_text(item.get("location")),
                longitude=longitude, latitude=latitude, tel=_text(item.get("tel")) or None,
                distance=_text(item.get("distance")) or None, raw=item,
            ))
        return self.cache.put(cache_key, results[:20], 60 * 60)

    def plan_route(
        self, origin: POI | ResolvedLocation, destination: POI | ResolvedLocation,
        mode: str = "transit",
    ) -> RouteLeg:
        def point(value: POI | ResolvedLocation) -> str:
            if getattr(value, "location", ""):
                return value.location
            longitude = getattr(value, "longitude", None)
            latitude = getattr(value, "latitude", None)
            if longitude is None or latitude is None:
                raise AmapError("地点缺少坐标，无法规划路线。", code="AMAP_MISSING_COORDINATES")
            return f"{longitude},{latitude}"

        path_by_mode = {
            "driving": "/v5/direction/driving",
            "walking": "/v5/direction/walking",
            "transit": "/v5/direction/transit/integrated",
        }
        path = path_by_mode.get(mode, path_by_mode["transit"])
        payload = self._request(path, {
            "origin": point(origin), "destination": point(destination),
            "strategy": "0", "city": getattr(destination, "city", ""),
        })
        route = payload.get("route") or {}
        candidates = route.get("paths") or route.get("transits") or []
        first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        distance = _text(first.get("distance") or first.get("walking_distance"))
        cost = first.get("cost") if isinstance(first.get("cost"), dict) else {}
        duration = _text(first.get("duration") or cost.get("duration"))
        return RouteLeg(
            origin=getattr(origin, "name", getattr(origin, "query", "起点")),
            destination=getattr(destination, "name", getattr(destination, "query", "终点")),
            mode=mode if mode in {"walking", "transit", "driving"} else "transit",
            distance_m=int(distance) if distance.isdigit() else None,
            duration_s=int(duration) if duration.isdigit() else None,
            summary=f"{mode}路线由高德返回",
        )
