export type StreamEvent =
  | { type: 'step'; step: string; status: string; label: string }
  | { type: 'plan'; task_type: string; title: string; steps: string[] }
  | { type: 'tool_start' | 'tool_result'; tool: string; status: string; label: string; count?: number }
  | { type: 'itinerary_patch'; days: DayPlan[]; warnings: Warning[] }
  | { type: 'change_proposal'; proposal: TripChangeProposal }
  | { type: 'token'; delta: string }
  | { type: 'final'; answer: string; sources: SourceRecord[]; profile?: UserProfile; travel_profile?: TravelProfile; trip?: TripPlan; conversation_id?: string }
  | { type: 'error'; code: string; stage?: string; message: string; retryable: boolean; current_version?: number };

export type SourceRecord = {
  provider: string;
  location: string;
  reporttime: string;
  kind: '实时' | '预报';
};

export type ResolvedLocation = {
  query: string;
  name: string;
  province: string;
  city: string;
  district: string;
  adcode: string;
};

export type UserProfile = {
  user_id: string;
  default_location: ResolvedLocation | null;
  favorite_locations: ResolvedLocation[];
  temperature_unit: 'celsius' | 'fahrenheit';
  advice_preferences: string[];
  updated_at: string | null;
};

export type POI = { id: string; name: string; type: string; address: string; location: string; longitude?: number | null; latitude?: number | null };
export type RouteLeg = { origin: string; destination: string; mode: 'walking' | 'transit' | 'driving'; distance_m?: number | null; duration_s?: number | null; summary: string };
export type Activity = { id: string; date: string; period: 'morning' | 'afternoon' | 'evening'; poi: POI; duration_minutes: number; indoor: boolean; reason: string; route_from_previous?: RouteLeg | null };
export type DayPlan = { date: string; weather_summary: string; activities: Activity[]; warnings: Warning[]; route_summary: string };
export type Warning = { type: string; severity: 'info' | 'warning' | 'error'; message: string; suggestion: string };
export type TripPlan = { trip_id: string; name: string; request: { destination: string; days: number; pace: string; interests: string[] }; days: DayPlan[]; budget_estimate: string; warnings: Warning[]; version: number; status: string; updated_at?: string | null };
export type TripMessage = { id?: number; trip_id: string; role: 'user' | 'assistant'; content: string; event_summary: string; created_at?: string | null };
export type TripChangeProposal = { proposal_id: string; trip_id: string; based_on_version: number; kind: string; title: string; description: string; changes: string[]; proposed_plan: TripPlan; status: 'pending' | 'applied' | 'dismissed'; created_at?: string | null };
export type TravelProfile = { user_id: string; home_city?: string | null; pace: 'relaxed' | 'balanced' | 'packed'; budget_level: 'economy' | 'moderate' | 'premium'; interests: string[]; dietary_restrictions: string[]; transport_modes: ('walking' | 'transit' | 'driving')[]; accessibility_needs: string[]; updated_at?: string | null };

export function consumeSse(buffer: string): { events: StreamEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, '\n');
  const frames = normalized.split('\n\n');
  const rest = frames.pop() ?? '';
  const events: StreamEvent[] = [];
  for (const frame of frames) {
    const data = frame
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n');
    if (!data) continue;
    try {
      events.push(JSON.parse(data) as StreamEvent);
    } catch {
      // Ignore malformed frames and keep the stream usable.
    }
  }
  return { events, rest };
}
