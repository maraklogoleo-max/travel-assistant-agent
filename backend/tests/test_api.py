from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_profile_and_conversation_lifecycle(tmp_path) -> None:
    settings = Settings(database_path=tmp_path / "api.db", deepseek_api_key="", amap_api_key="")
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert health["amap_configured"] is False
        created = client.post("/api/conversations")
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        assert client.get("/api/profile").json()["temperature_unit"] == "celsius"
        assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204


def test_trip_project_and_versions_lifecycle(tmp_path) -> None:
    settings = Settings(database_path=tmp_path / "trip-api.db", deepseek_api_key="", amap_api_key="")
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/trips", json={"destination": "杭州", "days": 2})
        assert created.status_code == 201
        trip_id = created.json()["trip_id"]
        assert client.get("/api/trips").json()[0]["trip_id"] == trip_id
        patched = client.patch(f"/api/trips/{trip_id}", json={"name": "杭州周末"})
        assert patched.status_code == 200
        assert patched.json()["name"] == "杭州周末"
        assert client.get(f"/api/trips/{trip_id}/versions").status_code == 200
        assert client.delete(f"/api/trips/{trip_id}").status_code == 204
