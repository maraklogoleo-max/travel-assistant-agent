from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class WeatherPlan(BaseModel):
    task_type: Literal["weather_query", "travel_assessment", "comparison", "memory_update"] = "weather_query"
    locations: list[str] = Field(default_factory=list, max_length=5)
    time_scope: Literal["current", "today", "forecast"] = "current"
    target_dates: list[date] = Field(default_factory=list, max_length=3)
    compare: bool = False
    advice_topics: list[str] = Field(default_factory=list, max_length=5)
    needs_weather: bool = True
    unsupported_reason: str | None = None
    memory_update: dict[str, str] = Field(default_factory=dict)

    @field_validator("locations")
    @classmethod
    def normalize_locations(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result[:5]


class ResolvedLocation(BaseModel):
    query: str
    name: str
    province: str = ""
    city: str = ""
    district: str = ""
    adcode: str
    longitude: float | None = None
    latitude: float | None = None

    @property
    def display_name(self) -> str:
        parts = [self.province, self.city, self.district]
        compact = " · ".join(part for index, part in enumerate(parts) if part and part not in parts[:index])
        return compact or self.name


class WeatherSnapshot(BaseModel):
    location: ResolvedLocation
    kind: Literal["current", "forecast"]
    reporttime: str
    date: str | None = None
    weather: str
    temperature: str | None = None
    day_temperature: str | None = None
    night_temperature: str | None = None
    humidity: str | None = None
    wind_direction: str | None = None
    wind_power: str | None = None


class SourceRecord(BaseModel):
    provider: str = "高德开放平台"
    location: str
    reporttime: str
    kind: Literal["实时", "预报", "地点", "路线"]
    resource_id: str | None = None
    query_time: str | None = None
    cached: bool = False
    detail: str = ""


class UserProfile(BaseModel):
    user_id: str = "local-user"
    default_location: ResolvedLocation | None = None
    favorite_locations: list[ResolvedLocation] = Field(default_factory=list)
    temperature_unit: Literal["celsius", "fahrenheit"] = "celsius"
    advice_preferences: list[str] = Field(default_factory=list)
    updated_at: str | None = None


class ProfilePatch(BaseModel):
    default_location: ResolvedLocation | None = None
    favorite_locations: list[ResolvedLocation] | None = None
    temperature_unit: Literal["celsius", "fahrenheit"] | None = None
    advice_preferences: list[str] | None = None


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class TravelProfile(BaseModel):
    user_id: str = "local-user"
    home_city: str | None = None
    pace: Literal["relaxed", "balanced", "packed"] = "balanced"
    budget_level: Literal["economy", "moderate", "premium"] = "moderate"
    interests: list[str] = Field(default_factory=list, max_length=10)
    dietary_restrictions: list[str] = Field(default_factory=list, max_length=10)
    transport_modes: list[Literal["walking", "transit", "driving"]] = Field(default_factory=list, max_length=3)
    accessibility_needs: list[str] = Field(default_factory=list, max_length=10)
    updated_at: str | None = None


class TripRequest(BaseModel):
    destination: str
    origin: str | None = None
    start_date: date | None = None
    days: int = Field(default=1, ge=1, le=7)
    travelers: int = Field(default=1, ge=1, le=20)
    budget_level: Literal["economy", "moderate", "premium"] = "moderate"
    pace: Literal["relaxed", "balanced", "packed"] = "balanced"
    interests: list[str] = Field(default_factory=list, max_length=10)
    transport_mode: Literal["walking", "transit", "driving"] = "transit"
    transport_preference: str | None = None
    accommodation_preference: str | None = None
    dietary_restrictions: list[str] = Field(default_factory=list, max_length=10)
    special_needs: list[str] = Field(default_factory=list, max_length=10)


class POI(BaseModel):
    id: str
    name: str
    type: str = ""
    address: str = ""
    city: str = ""
    location: str = ""
    longitude: float | None = None
    latitude: float | None = None
    tel: str | None = None
    distance: str | None = None
    source: str = "高德开放平台"
    raw: dict[str, Any] = Field(default_factory=dict)


class RouteLeg(BaseModel):
    origin: str
    destination: str
    mode: Literal["walking", "transit", "driving"]
    distance_m: int | None = None
    duration_s: int | None = None
    summary: str = ""
    source: str = "高德路径规划"
    query_time: str | None = None


class Activity(BaseModel):
    id: str
    date: date
    period: Literal["morning", "afternoon", "evening"]
    poi: POI
    duration_minutes: int = Field(default=90, ge=30, le=360)
    start_time: str | None = None
    end_time: str | None = None
    indoor: bool = False
    reason: str = ""
    route_from_previous: RouteLeg | None = None
    sources: list[SourceRecord] = Field(default_factory=list)
    data_confidence: Literal["verified", "partial", "estimated"] = "verified"


class ConstraintWarning(BaseModel):
    type: str
    severity: Literal["info", "warning", "error"] = "warning"
    message: str
    suggestion: str = ""


class DayPlan(BaseModel):
    date: date
    weather_summary: str = "天气待临近日期刷新"
    activities: list[Activity] = Field(default_factory=list, max_length=6)
    warnings: list[ConstraintWarning] = Field(default_factory=list)
    route_summary: str = ""
    sources: list[SourceRecord] = Field(default_factory=list)


class TripPlan(BaseModel):
    trip_id: str
    name: str
    request: TripRequest
    days: list[DayPlan] = Field(default_factory=list, max_length=7)
    budget_estimate: str = "仅供参考的粗略估算"
    warnings: list[ConstraintWarning] = Field(default_factory=list)
    version: int = 1
    status: Literal["draft", "ready", "needs_input"] = "draft"
    updated_at: str | None = None


class TripCreateRequest(TripRequest):
    name: str | None = None


class TripPatch(BaseModel):
    name: str | None = None
    plan: TripPlan | None = None


class AgentPlan(BaseModel):
    intent: Literal[
        "weather_query", "trip_create", "trip_update", "trip_weather_assessment",
        "trip_query", "memory_update", "clarification",
    ]
    objective: str
    tools: list[Literal["resolve_location", "weather", "places", "routes", "memory", "trip"]] = Field(default_factory=list, max_length=6)
    target_day: int | None = Field(default=None, ge=1, le=7)
    requires_confirmation: bool = False
    missing_fields: list[str] = Field(default_factory=list, max_length=5)
    requirements: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list, max_length=12)
    planned_steps: list[str] = Field(default_factory=list, max_length=12)
    action_budget: int = Field(default=8, ge=1, le=12)


class AgentAction(BaseModel):
    tool: Literal[
        "resolve_location", "weather", "places", "routes", "memory",
        "trip", "trip_create", "trip_update", "trip_query",
        "trip_weather_assessment", "finish",
    ]
    objective: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    success: bool
    data: Any = None
    sources: list[SourceRecord] = Field(default_factory=list)
    error_code: str | None = None
    user_message: str | None = None
    retryable: bool = False


class AgentObservation(BaseModel):
    action: AgentAction
    result: ToolResult


class ValidationResult(BaseModel):
    valid: bool = True
    warnings: list[ConstraintWarning] = Field(default_factory=list)
    retryable: bool = False
    affected_days: list[int] = Field(default_factory=list)


class UnifiedMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = None
    trip_id: str | None = None
    expected_version: int | None = Field(default=None, ge=1)


class TripMessage(BaseModel):
    id: int | None = None
    trip_id: str
    role: Literal["user", "assistant"]
    content: str
    event_summary: str = ""
    created_at: str | None = None


class ConversationMessage(BaseModel):
    id: int | None = None
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    event_summary: str = ""
    created_at: str | None = None


class ConversationSummary(BaseModel):
    conversation_id: str
    confirmed_context: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recent_topics: list[str] = Field(default_factory=list)
    pending_plan: AgentPlan | None = None
    updated_at: str | None = None


class TripSummary(BaseModel):
    trip_id: str
    confirmed_requirements: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recent_changes: list[str] = Field(default_factory=list)
    updated_at: str | None = None


class TripChangeProposal(BaseModel):
    proposal_id: str
    trip_id: str
    based_on_version: int
    kind: Literal["weather", "activity", "pace"] = "weather"
    title: str
    description: str
    changes: list[str] = Field(default_factory=list)
    proposed_plan: TripPlan
    status: Literal["pending", "applied", "dismissed"] = "pending"
    created_at: str | None = None


class AgentError(BaseModel):
    code: str
    stage: str
    message: str
    retryable: bool = False


class AgentRunRecord(BaseModel):
    run_id: str
    conversation_id: str
    trip_id: str | None = None
    intent: str
    status: Literal["running", "completed", "needs_input", "failed"] = "running"
    action_count: int = 0
    error_code: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
