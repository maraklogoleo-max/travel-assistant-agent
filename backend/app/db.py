import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

from .models import (
    ProfilePatch, TravelProfile, TripChangeProposal, TripMessage, TripPlan,
    TripRequest, TripSummary, UserProfile,
)


class TripVersionConflict(RuntimeError):
    def __init__(self, current_version: int) -> None:
        super().__init__("行程已被其他操作更新")
        self.current_version = current_version


class ProfileRepository:
    def __init__(self, path: Path, timezone: str = "Asia/Shanghai") -> None:
        self.path = path
        self.timezone = ZoneInfo(timezone)
        self._lock = Lock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    default_location TEXT,
                    favorite_locations TEXT NOT NULL DEFAULT '[]',
                    temperature_unit TEXT NOT NULL DEFAULT 'celsius'
                        CHECK (temperature_unit IN ('celsius', 'fahrenheit')),
                    advice_preferences TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_context (
                    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                    last_locations TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trips (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trip_versions (
                    trip_id TEXT NOT NULL REFERENCES trips(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (trip_id, version)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trip_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trip_id TEXT NOT NULL REFERENCES trips(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    event_summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trip_summaries (
                    trip_id TEXT PRIMARY KEY REFERENCES trips(id) ON DELETE CASCADE,
                    summary_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trip_change_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    trip_id TEXT NOT NULL REFERENCES trips(id) ON DELETE CASCADE,
                    based_on_version INTEGER NOT NULL,
                    proposal_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS travel_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_trip_messages_trip_id ON trip_messages(trip_id, id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_trip_proposals_trip_id ON trip_change_proposals(trip_id, status)")
            connection.execute("PRAGMA optimize")

    def _now(self) -> str:
        return datetime.now(self.timezone).isoformat(timespec="seconds")

    def get_profile(self, user_id: str = "local-user") -> UserProfile:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return UserProfile(user_id=user_id)
        return UserProfile(
            user_id=row["user_id"],
            default_location=json.loads(row["default_location"]) if row["default_location"] else None,
            favorite_locations=json.loads(row["favorite_locations"]),
            temperature_unit=row["temperature_unit"],
            advice_preferences=json.loads(row["advice_preferences"]),
            updated_at=row["updated_at"],
        )

    def save_profile(self, profile: UserProfile) -> UserProfile:
        profile.updated_at = self._now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles (
                    user_id, default_location, favorite_locations,
                    temperature_unit, advice_preferences, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    default_location=excluded.default_location,
                    favorite_locations=excluded.favorite_locations,
                    temperature_unit=excluded.temperature_unit,
                    advice_preferences=excluded.advice_preferences,
                    updated_at=excluded.updated_at
                """,
                (
                    profile.user_id,
                    json.dumps(profile.default_location.model_dump(), ensure_ascii=False)
                    if profile.default_location else None,
                    json.dumps([item.model_dump() for item in profile.favorite_locations], ensure_ascii=False),
                    profile.temperature_unit,
                    json.dumps(profile.advice_preferences, ensure_ascii=False),
                    profile.updated_at,
                ),
            )
        return profile

    def patch_profile(self, patch: ProfilePatch, user_id: str = "local-user") -> UserProfile:
        profile = self.get_profile(user_id)
        for field, value in patch.model_dump(exclude_unset=True).items():
            setattr(profile, field, value)
        return self.save_profile(UserProfile.model_validate(profile))

    def clear_profile(self, user_id: str = "local-user") -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))

    def create_conversation(self, conversation_id: str) -> str:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
                (conversation_id, self._now()),
            )
            connection.execute(
                "INSERT INTO conversation_context (conversation_id, last_locations, updated_at) VALUES (?, '[]', ?)",
                (conversation_id, self._now()),
            )
        return conversation_id

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def get_last_locations(self, conversation_id: str) -> list[dict]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT last_locations FROM conversation_context WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return json.loads(row["last_locations"]) if row else []

    def save_last_locations(self, conversation_id: str, locations: list[dict]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_context (conversation_id, last_locations, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    last_locations=excluded.last_locations,
                    updated_at=excluded.updated_at
                """,
                (conversation_id, json.dumps(locations, ensure_ascii=False), self._now()),
            )

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._lock, self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone() is not None

    def create_trip(self, trip_id: str, request: TripRequest, plan: TripPlan) -> TripPlan:
        now = self._now()
        plan.updated_at = now
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO trips (id, name, request_json, plan_json, status, version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (trip_id, plan.name, json.dumps(request.model_dump(mode="json"), ensure_ascii=False),
                 json.dumps(plan.model_dump(mode="json"), ensure_ascii=False), plan.status, plan.version, now, now),
            )
            connection.execute(
                "INSERT INTO trip_versions (trip_id, version, plan_json, created_at) VALUES (?, ?, ?, ?)",
                (trip_id, plan.version, json.dumps(plan.model_dump(mode="json"), ensure_ascii=False), now),
            )
        return plan

    def list_trips(self) -> list[TripPlan]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT plan_json FROM trips ORDER BY updated_at DESC").fetchall()
        return [TripPlan.model_validate(json.loads(row["plan_json"])) for row in rows]

    def get_trip(self, trip_id: str) -> TripPlan | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT plan_json FROM trips WHERE id = ?", (trip_id,)).fetchone()
        return TripPlan.model_validate(json.loads(row["plan_json"])) if row else None

    def save_trip(self, plan: TripPlan, *, name: str | None = None, expected_version: int | None = None) -> TripPlan:
        now = self._now()
        if name:
            plan.name = name
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT version FROM trips WHERE id = ?", (plan.trip_id,)).fetchone()
            if row is None:
                raise ValueError("旅行项目不存在")
            current_version = int(row["version"])
            if expected_version is not None and current_version != expected_version:
                raise TripVersionConflict(current_version)
            plan.version = current_version + 1
            plan.updated_at = now
            connection.execute(
                "UPDATE trips SET name = ?, request_json = ?, plan_json = ?, status = ?, version = ?, updated_at = ? WHERE id = ?",
                (plan.name, json.dumps(plan.request.model_dump(mode="json"), ensure_ascii=False),
                 json.dumps(plan.model_dump(mode="json"), ensure_ascii=False), plan.status, plan.version, now, plan.trip_id),
            )
            connection.execute(
                "INSERT INTO trip_versions (trip_id, version, plan_json, created_at) VALUES (?, ?, ?, ?)",
                (plan.trip_id, plan.version, json.dumps(plan.model_dump(mode="json"), ensure_ascii=False), now),
            )
        return plan

    def delete_trip(self, trip_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM trips WHERE id = ?", (trip_id,))

    def list_trip_versions(self, trip_id: str) -> list[TripPlan]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT plan_json FROM trip_versions WHERE trip_id = ? ORDER BY version DESC", (trip_id,)).fetchall()
        return [TripPlan.model_validate(json.loads(row["plan_json"])) for row in rows]

    def restore_trip(self, trip_id: str, version: int) -> TripPlan | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT plan_json FROM trip_versions WHERE trip_id = ? AND version = ?", (trip_id, version)).fetchone()
        if not row:
            return None
        plan = TripPlan.model_validate(json.loads(row["plan_json"]))
        return self.save_trip(plan)

    def get_travel_profile(self, user_id: str = "local-user") -> TravelProfile:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT profile_json FROM travel_profiles WHERE user_id = ?", (user_id,)).fetchone()
        return TravelProfile.model_validate(json.loads(row["profile_json"])) if row else TravelProfile(user_id=user_id)

    def save_travel_profile(self, profile: TravelProfile) -> TravelProfile:
        profile.updated_at = self._now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO travel_profiles (user_id, profile_json, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=excluded.updated_at",
                (profile.user_id, json.dumps(profile.model_dump(mode="json"), ensure_ascii=False), profile.updated_at),
            )
        return profile

    def clear_travel_profile(self, user_id: str = "local-user") -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM travel_profiles WHERE user_id = ?", (user_id,))

    def add_trip_message(self, trip_id: str, role: str, content: str, event_summary: str = "") -> TripMessage:
        now = self._now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO trip_messages (trip_id, role, content, event_summary, created_at) VALUES (?, ?, ?, ?, ?)",
                (trip_id, role, content, event_summary, now),
            )
            rows = connection.execute(
                "SELECT id FROM trip_messages WHERE trip_id = ? ORDER BY id DESC LIMIT -1 OFFSET 40", (trip_id,)
            ).fetchall()
            if rows:
                connection.executemany("DELETE FROM trip_messages WHERE id = ?", [(row["id"],) for row in rows])
        return TripMessage(id=cursor.lastrowid, trip_id=trip_id, role=role, content=content, event_summary=event_summary, created_at=now)

    def get_trip_messages(self, trip_id: str, limit: int = 40) -> list[TripMessage]:
        limit = max(1, min(40, limit))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM (SELECT * FROM trip_messages WHERE trip_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                (trip_id, limit),
            ).fetchall()
        return [TripMessage(**dict(row)) for row in rows]

    def get_trip_summary(self, trip_id: str) -> TripSummary:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT summary_json FROM trip_summaries WHERE trip_id = ?", (trip_id,)).fetchone()
        return TripSummary.model_validate(json.loads(row["summary_json"])) if row else TripSummary(trip_id=trip_id)

    def save_trip_summary(self, summary: TripSummary) -> TripSummary:
        summary.updated_at = self._now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO trip_summaries (trip_id, summary_json, updated_at) VALUES (?, ?, ?) ON CONFLICT(trip_id) DO UPDATE SET summary_json=excluded.summary_json, updated_at=excluded.updated_at",
                (summary.trip_id, json.dumps(summary.model_dump(mode="json"), ensure_ascii=False), summary.updated_at),
            )
        return summary

    def save_change_proposal(self, proposal: TripChangeProposal) -> TripChangeProposal:
        proposal.created_at = self._now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO trip_change_proposals (proposal_id, trip_id, based_on_version, proposal_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (proposal.proposal_id, proposal.trip_id, proposal.based_on_version, json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False), proposal.status, proposal.created_at),
            )
        return proposal

    def get_change_proposal(self, proposal_id: str) -> TripChangeProposal | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT proposal_json, status FROM trip_change_proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
        if not row:
            return None
        proposal = TripChangeProposal.model_validate(json.loads(row["proposal_json"]))
        proposal.status = row["status"]
        return proposal

    def get_pending_proposals(self, trip_id: str) -> list[TripChangeProposal]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT proposal_json FROM trip_change_proposals WHERE trip_id = ? AND status = 'pending' ORDER BY created_at DESC", (trip_id,)).fetchall()
        return [TripChangeProposal.model_validate(json.loads(row["proposal_json"])) for row in rows]

    def update_proposal_status(self, proposal_id: str, status: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE trip_change_proposals SET status = ? WHERE proposal_id = ?", (status, proposal_id))
