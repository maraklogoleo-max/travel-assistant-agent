import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent import WeatherAgent
from .config import Settings, get_settings
from .amap import AmapError
from .db import ProfileRepository, TripVersionConflict
from .models import MessageRequest, ProfilePatch, TravelProfile, TripCreateRequest, TripMessage, TripPlan, UnifiedMessageRequest, UserProfile
from .travel_agent import TravelAgent
from .unified_agent import TravelAssistantAgent


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    profiles = ProfileRepository(app_settings.database_path, app_settings.timezone)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        profiles.initialize()
        app.state.profiles = profiles
        app.state.agent = WeatherAgent(app_settings, profiles)
        app.state.travel_agent = TravelAgent(app_settings, profiles)
        app.state.assistant_agent = TravelAssistantAgent(app_settings, profiles, app.state.agent, app.state.travel_agent)
        yield
        app.state.agent.close()
        app.state.travel_agent.close()

    app = FastAPI(
        title="旅游出行小帮手 API",
        version="0.1.0",
        description="高德 POI/路线/天气事实 + LangGraph 旅行规划 + DeepSeek V4 Flash",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[app_settings.frontend_origin, "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "deepseek_configured": bool(app_settings.deepseek_api_key),
            "amap_configured": bool(app_settings.amap_api_key),
            "model": app_settings.deepseek_model,
        }

    @app.post("/api/conversations", status_code=status.HTTP_201_CREATED)
    async def create_conversation() -> dict:
        conversation_id = str(uuid4())
        profiles.create_conversation(conversation_id)
        return {"id": conversation_id}

    @app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_conversation(conversation_id: str) -> Response:
        if not profiles.conversation_exists(conversation_id):
            raise HTTPException(status_code=404, detail="对话不存在")
        profiles.delete_conversation(conversation_id)
        await asyncio.to_thread(app.state.agent.delete_thread, conversation_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/trips", response_model=TripPlan, status_code=status.HTTP_201_CREATED)
    async def create_trip(request: TripCreateRequest) -> TripPlan:
        trip_id = str(uuid4())
        plan = TripPlan(
            trip_id=trip_id,
            name=request.name or f"{request.destination}之旅",
            request=request.model_dump(exclude={"name"}),
            days=[],
            status="draft",
        )
        profiles.create_trip(trip_id, plan.request, plan)
        return plan

    @app.get("/api/trips", response_model=list[TripPlan])
    async def list_trips() -> list[TripPlan]:
        return profiles.list_trips()

    @app.get("/api/trips/{trip_id}", response_model=TripPlan)
    async def get_trip(trip_id: str) -> TripPlan:
        plan = profiles.get_trip(trip_id)
        if not plan:
            raise HTTPException(status_code=404, detail="旅行项目不存在")
        return plan

    @app.get("/api/trips/{trip_id}/messages", response_model=list[TripMessage])
    async def get_trip_messages(trip_id: str, limit: int = 40) -> list[TripMessage]:
        if not profiles.get_trip(trip_id):
            raise HTTPException(status_code=404, detail="旅行项目不存在")
        return profiles.get_trip_messages(trip_id, limit)

    @app.get("/api/trips/{trip_id}/proposals")
    async def get_trip_proposals(trip_id: str) -> list[dict]:
        if not profiles.get_trip(trip_id):
            raise HTTPException(status_code=404, detail="旅行项目不存在")
        return [item.model_dump(mode="json") for item in profiles.get_pending_proposals(trip_id)]

    @app.patch("/api/trips/{trip_id}", response_model=TripPlan)
    async def patch_trip(trip_id: str, patch: dict[str, object]) -> TripPlan:
        plan = profiles.get_trip(trip_id)
        if not plan:
            raise HTTPException(status_code=404, detail="旅行项目不存在")
        if isinstance(patch.get("name"), str):
            plan.name = str(patch["name"])
        if isinstance(patch.get("plan"), dict):
            plan = TripPlan.model_validate(patch["plan"])
        return profiles.save_trip(plan)

    @app.delete("/api/trips/{trip_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_trip(trip_id: str) -> Response:
        if not profiles.get_trip(trip_id):
            raise HTTPException(status_code=404, detail="旅行项目不存在")
        profiles.delete_trip(trip_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/trips/{trip_id}/versions", response_model=list[TripPlan])
    async def trip_versions(trip_id: str) -> list[TripPlan]:
        if not profiles.get_trip(trip_id):
            raise HTTPException(status_code=404, detail="旅行项目不存在")
        return profiles.list_trip_versions(trip_id)

    @app.post("/api/trips/{trip_id}/versions/{version}/restore", response_model=TripPlan)
    async def restore_trip(trip_id: str, version: int) -> TripPlan:
        restored = profiles.restore_trip(trip_id, version)
        if not restored:
            raise HTTPException(status_code=404, detail="行程版本不存在")
        return restored

    @app.post("/api/trips/{trip_id}/proposals/{proposal_id}/apply", response_model=TripPlan)
    async def apply_proposal(trip_id: str, proposal_id: str) -> TripPlan:
        proposal = profiles.get_change_proposal(proposal_id)
        if not proposal or proposal.trip_id != trip_id:
            raise HTTPException(status_code=404, detail="调整建议不存在")
        if proposal.status != "pending":
            raise HTTPException(status_code=409, detail="调整建议已经处理")
        try:
            updated = profiles.save_trip(proposal.proposed_plan, expected_version=proposal.based_on_version)
        except TripVersionConflict as exc:
            raise HTTPException(status_code=409, detail={"code": "TRIP_VERSION_CONFLICT", "current_version": exc.current_version}) from exc
        profiles.update_proposal_status(proposal_id, "applied")
        return updated

    @app.post("/api/trips/{trip_id}/proposals/{proposal_id}/dismiss", status_code=status.HTTP_204_NO_CONTENT)
    async def dismiss_proposal(trip_id: str, proposal_id: str) -> Response:
        proposal = profiles.get_change_proposal(proposal_id)
        if not proposal or proposal.trip_id != trip_id:
            raise HTTPException(status_code=404, detail="调整建议不存在")
        profiles.update_proposal_status(proposal_id, "dismissed")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/assistant/messages")
    async def unified_message(request: UnifiedMessageRequest) -> StreamingResponse:
        conversation_id = request.conversation_id or str(uuid4())
        if not profiles.conversation_exists(conversation_id):
            profiles.create_conversation(conversation_id)

        async def stream() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict | None] = asyncio.Queue()

            def emit(event: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event)

            async def execute() -> None:
                try:
                    result = await asyncio.to_thread(
                        app.state.assistant_agent.run, request.message.strip(),
                        conversation_id=conversation_id, trip_id=request.trip_id,
                        expected_version=request.expected_version, emit_callback=emit,
                    )
                    await queue.put({"type": "final", "conversation_id": conversation_id, **result})
                except TripVersionConflict as exc:
                    await queue.put({"type": "error", "code": "TRIP_VERSION_CONFLICT", "stage": "persist", "message": "行程已在其他操作中更新，已为你保留最新版本。", "retryable": True, "current_version": exc.current_version})
                except AmapError as exc:
                    await queue.put({"type": "error", "code": exc.code, "stage": "tool", "message": str(exc), "retryable": True})
                except ValueError as exc:
                    await queue.put({"type": "error", "code": "TRIP_NOT_FOUND", "stage": "context", "message": str(exc), "retryable": False})
                except Exception:
                    await queue.put({"type": "error", "code": "AGENT_ERROR", "stage": "agent", "message": "旅行助手暂时无法完成请求，请稍后重试。", "retryable": True})
                finally:
                    await queue.put(None)

            task = asyncio.create_task(execute())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield _sse(item)
            finally:
                await task

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/trips/{trip_id}/messages")
    async def send_trip_message(trip_id: str, request: MessageRequest) -> StreamingResponse:
        if not profiles.get_trip(trip_id):
            raise HTTPException(status_code=404, detail="旅行项目不存在")

        async def stream() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict | None] = asyncio.Queue()

            def emit(event: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event)

            async def execute() -> None:
                try:
                    result = await asyncio.to_thread(app.state.travel_agent.run, trip_id, request.message.strip(), emit)
                    await queue.put({"type": "final", **result})
                except Exception:
                    await queue.put({"type": "error", "code": "TRIP_AGENT_ERROR", "message": "旅行助手暂时无法完成规划，请稍后重试。", "retryable": True})
                finally:
                    await queue.put(None)

            task = asyncio.create_task(execute())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield _sse(item)
            finally:
                await task

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/conversations/{conversation_id}/messages")
    async def send_message(conversation_id: str, request: MessageRequest) -> StreamingResponse:
        if not profiles.conversation_exists(conversation_id):
            raise HTTPException(status_code=404, detail="对话不存在")

        async def stream() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict | None] = asyncio.Queue()

            def emit(event: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event)

            async def execute() -> None:
                try:
                    result = await asyncio.to_thread(
                        app.state.agent.run, conversation_id, request.message.strip(), emit
                    )
                    await queue.put({"type": "final", **result})
                except Exception:
                    await queue.put(
                        {
                            "type": "error",
                            "code": "AGENT_ERROR",
                            "message": "天气助手暂时无法完成请求，请稍后重试。",
                            "retryable": True,
                        }
                    )
                finally:
                    await queue.put(None)

            task = asyncio.create_task(execute())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield _sse(item)
            finally:
                await task

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/profile", response_model=UserProfile)
    async def get_profile() -> UserProfile:
        return profiles.get_profile()

    @app.patch("/api/profile", response_model=UserProfile)
    async def patch_profile(patch: ProfilePatch) -> UserProfile:
        return profiles.patch_profile(patch)

    @app.delete("/api/profile", status_code=status.HTTP_204_NO_CONTENT)
    async def clear_profile() -> Response:
        profiles.clear_profile()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/travel-profile", response_model=TravelProfile)
    async def get_travel_profile() -> TravelProfile:
        return profiles.get_travel_profile()

    @app.patch("/api/travel-profile", response_model=TravelProfile)
    async def patch_travel_profile(patch: dict[str, object]) -> TravelProfile:
        profile = profiles.get_travel_profile()
        allowed = {
            "home_city", "pace", "budget_level", "interests", "dietary_restrictions",
            "transport_modes", "accessibility_needs",
        }
        for field, value in patch.items():
            if field in allowed:
                setattr(profile, field, value)
        return profiles.save_travel_profile(TravelProfile.model_validate(profile))

    @app.delete("/api/travel-profile", status_code=status.HTTP_204_NO_CONTENT)
    async def clear_travel_profile() -> Response:
        profiles.clear_travel_profile()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
