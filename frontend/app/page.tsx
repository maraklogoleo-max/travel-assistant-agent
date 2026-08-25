'use client';

import { FormEvent, useCallback, useEffect, useRef, useState } from 'react';
import { consumeSse, DayPlan, SourceRecord, StreamEvent, TravelProfile, TripChangeProposal, TripMessage, TripPlan, UserProfile, Warning } from '../lib/sse';

// The backend binds to IPv4 for local development. Using 127.0.0.1 avoids
// browsers resolving localhost to ::1 and silently losing the conversation API.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';
const suggestions = ['杭州明天天气怎么样？', '去九寨沟玩三天，喜欢自然风景，节奏轻松', '记住我喜欢轻松节奏和自然风景'];

type Step = { step: string; status: string; label: string };
type Message = { id: string; role: 'user' | 'assistant'; text: string; steps?: Step[]; plan?: { task_type: string; title: string; steps: string[] }; tools?: { tool: string; status: string; label: string }[]; sources?: SourceRecord[]; itinerary?: { days: DayPlan[]; warnings: Warning[] }; proposal?: TripChangeProposal; errorCode?: string; pending?: boolean };
type Health = { status: string; deepseek_configured: boolean; amap_configured: boolean; model: string };
type MapStatus = 'loading' | 'ready' | 'missing' | 'error';
type AMapApi = { Map: new (element: HTMLDivElement, options: { zoom: number; center: (number | null | undefined)[] }) => { setFitView: () => void; destroy?: () => void }; Marker: new (options: { map: unknown; position: (number | null | undefined)[]; label: { content: string; direction: string } }) => unknown };

function AmapMap({ days }: { days: DayPlan[] }) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<MapStatus>(process.env.NEXT_PUBLIC_AMAP_JS_KEY ? 'loading' : 'missing');
  useEffect(() => {
    const key = process.env.NEXT_PUBLIC_AMAP_JS_KEY;
    const canvas = canvasRef.current;
    if (!key || !canvas) { setStatus('missing'); return; }
    const win = window as typeof window & { AMap?: AMapApi; _AMapSecurityConfig?: { securityJsCode?: string } };
    win._AMapSecurityConfig = { securityJsCode: process.env.NEXT_PUBLIC_AMAP_SECURITY_CODE };
    let map: { setFitView: () => void; destroy?: () => void } | undefined;
    const init = () => {
      if (!win.AMap) { setStatus('error'); return; }
      const points = days.flatMap((day) => day.activities).filter((item) => item.poi.longitude && item.poi.latitude);
      map?.destroy?.();
      map = new win.AMap.Map(canvas, { zoom: 10, center: points.length ? [points[0].poi.longitude, points[0].poi.latitude] : [116.397, 39.916] });
      points.forEach((item, index) => new win.AMap!.Marker({ map, position: [item.poi.longitude, item.poi.latitude], label: { content: `${index + 1}. ${item.poi.name}`, direction: 'top' } }));
      if (points.length > 1) map.setFitView();
      setStatus('ready');
    };
    const existing = document.querySelector<HTMLScriptElement>('script[data-amap-js]');
    const onError = () => setStatus('error');
    if (existing) {
      existing.addEventListener('load', init); existing.addEventListener('error', onError);
      if (win.AMap) init();
      return () => { existing.removeEventListener('load', init); existing.removeEventListener('error', onError); map?.destroy?.(); };
    }
    const script = document.createElement('script');
    script.dataset.amapJs = 'true'; script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(key)}`; script.onload = init; script.onerror = onError; document.head.appendChild(script);
    return () => map?.destroy?.();
  }, [days]);
  const labels = { loading: '正在加载高德地图…', ready: '高德 JS API 已加载', missing: '未读取到高德 JS API Key，请重启前端加载环境变量', error: '地图加载失败，请检查 JS Key、域名白名单和安全密钥' };
  return <div className="map-card"><strong>高德地图</strong><span className={`map-status ${status}`}>{labels[status]}</span><div className="amap-canvas" ref={canvasRef} /></div>;
}

function locationLabel(location: UserProfile['default_location']) {
  if (!location) return '';
  return [location.province, location.city, location.district].filter((item, index, all) => item && all.indexOf(item) === index).join(' · ') || location.name;
}

function friendlyError(code: string, message: string) {
  const titles: Record<string, string> = { TRIP_VERSION_CONFLICT: '行程版本已更新', TRIP_NOT_FOUND: '行程不存在', AGENT_ERROR: '模型暂时不可用', INVALID_USER_KEY: '高德密钥无效', AMAP_ERROR: '高德服务失败', LOCATION_AMBIGUOUS: '地点需要确认' };
  return `${titles[code] ?? '执行失败'}：${message}`;
}

export default function Home() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState('');
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [travelProfile, setTravelProfile] = useState<TravelProfile | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [trip, setTrip] = useState<TripPlan | null>(null);
  const [trips, setTrips] = useState<TripPlan[]>([]);
  const [proposal, setProposal] = useState<TripChangeProposal | null>(null);
  const [itineraryOpen, setItineraryOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const createConversation = useCallback(async () => {
    const response = await fetch(`${API_BASE}/api/conversations`, { method: 'POST' });
    if (!response.ok) throw new Error('无法创建对话');
    const data = (await response.json()) as { id: string }; setConversationId(data.id); return data.id;
  }, []);
  const loadProfile = useCallback(async () => { const response = await fetch(`${API_BASE}/api/profile`); if (response.ok) setProfile((await response.json()) as UserProfile); }, []);
  const loadTravelProfile = useCallback(async () => { const response = await fetch(`${API_BASE}/api/travel-profile`); if (response.ok) setTravelProfile((await response.json()) as TravelProfile); }, []);
  const loadTrips = useCallback(async () => { const response = await fetch(`${API_BASE}/api/trips`); if (response.ok) setTrips((await response.json()) as TripPlan[]); }, []);
  const initialize = useCallback(async () => { await Promise.all([createConversation(), loadProfile(), loadTravelProfile(), loadTrips(), fetch(`${API_BASE}/api/health`).then(async (response) => { if (response.ok) setHealth((await response.json()) as Health); })]); }, [createConversation, loadProfile, loadTravelProfile, loadTrips]);
  useEffect(() => { void initialize().catch(() => setHealth(null)); }, [initialize]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const reconcileTrip = (latest: TripPlan) => {
    setTrip(latest);
    setTrips((current) => current.some((item) => item.trip_id === latest.trip_id) ? current.map((item) => item.trip_id === latest.trip_id ? latest : item) : [latest, ...current]);
  };
  const updateAssistant = (id: string, updater: (message: Message) => Message) => setMessages((current) => current.map((message) => message.id === id ? updater(message) : message));

  const openTrip = useCallback(async (tripId: string) => {
    if (busy) return;
    try {
      const [tripResponse, messagesResponse, proposalsResponse] = await Promise.all([fetch(`${API_BASE}/api/trips/${tripId}`), fetch(`${API_BASE}/api/trips/${tripId}/messages?limit=40`), fetch(`${API_BASE}/api/trips/${tripId}/proposals`)]);
      if (!tripResponse.ok) throw new Error('行程不存在或已删除');
      const latest = (await tripResponse.json()) as TripPlan;
      setTrip(latest); setTrips((current) => current.map((item) => item.trip_id === latest.trip_id ? latest : item));
      const history = messagesResponse.ok ? (await messagesResponse.json()) as TripMessage[] : [];
      setMessages(history.map((item, index) => ({ id: `${item.id ?? index}-${item.created_at ?? ''}`, role: item.role, text: item.content })));
      const pending = proposalsResponse.ok ? (await proposalsResponse.json()) as TripChangeProposal[] : [];
      setProposal(pending[0] ?? null); setSidebarOpen(false);
    } catch (error) { setMessages([{ id: crypto.randomUUID(), role: 'assistant', text: error instanceof Error ? error.message : '无法打开行程', errorCode: 'TRIP_NOT_FOUND' }]); }
  }, [busy]);

  const applyStreamEvent = (event: StreamEvent, assistantId: string) => {
    if (event.type === 'plan') updateAssistant(assistantId, (message) => ({ ...message, plan: event }));
    if (event.type === 'agent_action') updateAssistant(assistantId, (message) => ({ ...message, tools: [...(message.tools ?? []), { tool: event.action, status: 'planned', label: event.objective }] }));
    if (event.type === 'clarification') updateAssistant(assistantId, (message) => ({ ...message, text: event.message, errorCode: event.code, pending: false }));
    if (event.type === 'warning') updateAssistant(assistantId, (message) => ({ ...message, steps: [...(message.steps ?? []), { step: event.code ?? `warning-${message.steps?.length ?? 0}`, status: 'warning', label: event.message }] }));
    if (event.type === 'token') updateAssistant(assistantId, (message) => ({ ...message, text: message.text + event.delta }));
    if (event.type === 'tool_start' || event.type === 'tool_result') updateAssistant(assistantId, (message) => ({ ...message, tools: [...(message.tools ?? []), { tool: event.tool, status: event.status, label: event.label }] }));
    if (event.type === 'itinerary_patch') updateAssistant(assistantId, (message) => ({ ...message, itinerary: { days: event.days, warnings: event.warnings } }));
    if (event.type === 'change_proposal') { setProposal(event.proposal); updateAssistant(assistantId, (message) => ({ ...message, proposal: event.proposal })); }
    if (event.type === 'step') updateAssistant(assistantId, (message) => { const steps = [...(message.steps ?? [])]; const index = steps.findIndex((step) => step.step === event.step); const value = { step: event.step, status: event.status, label: event.label }; if (index >= 0) steps[index] = value; else steps.push(value); return { ...message, steps }; });
    if (event.type === 'final') { updateAssistant(assistantId, (message) => ({ ...message, text: event.answer, sources: event.sources, pending: false })); if (event.profile) setProfile(event.profile); if (event.travel_profile) setTravelProfile(event.travel_profile); if (event.trip) reconcileTrip(event.trip); if (event.conversation_id) setConversationId(event.conversation_id); }
    if (event.type === 'error') { updateAssistant(assistantId, (message) => ({ ...message, text: friendlyError(event.code, event.message), errorCode: event.code, pending: false })); if (event.code === 'TRIP_VERSION_CONFLICT' && trip) void openTrip(trip.trip_id); }
  };

  const send = async (text: string) => {
    const cleaned = text.trim(); if (!cleaned || busy) return;
    setBusy(true); setQuestion(''); const assistantId = crypto.randomUUID();
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: 'user', text: cleaned }, { id: assistantId, role: 'assistant', text: '', steps: [], pending: true }]);
    try {
      const response = await fetch(`${API_BASE}/api/assistant/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: cleaned, conversation_id: conversationId, trip_id: trip?.trip_id, expected_version: trip?.version }) });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
      while (true) { const { value, done } = await reader.read(); buffer += decoder.decode(value, { stream: !done }); const parsed = consumeSse(buffer); buffer = parsed.rest; parsed.events.forEach((event) => applyStreamEvent(event, assistantId)); if (done) break; }
      await Promise.all([loadProfile(), loadTravelProfile(), loadTrips()]);
    } catch (error) { updateAssistant(assistantId, (message) => ({ ...message, text: `暂时无法连接旅行助手：${error instanceof Error ? error.message : '未知错误'}。请稍后重试。`, errorCode: 'NETWORK_ERROR', pending: false })); }
    finally { setBusy(false); }
  };

  const newConversation = async () => { if (busy) return; setTrip(null); setProposal(null); setMessages([]); setItineraryOpen(false); await createConversation().catch(() => setConversationId(null)); };
  const deleteTrip = async (selected: TripPlan) => {
    if (!window.confirm(`确定永久删除“${selected.name}”吗？\n相关版本、对话和待确认建议也会一起删除，此操作不可撤销。`)) return;
    const response = await fetch(`${API_BASE}/api/trips/${selected.trip_id}`, { method: 'DELETE' });
    if (!response.ok) { window.alert('删除失败，请刷新后重试。'); return; }
    setTrips((current) => current.filter((item) => item.trip_id !== selected.trip_id)); if (trip?.trip_id === selected.trip_id) await newConversation();
  };
  const applyProposal = async () => { if (!trip || !proposal) return; const response = await fetch(`${API_BASE}/api/trips/${trip.trip_id}/proposals/${proposal.proposal_id}/apply`, { method: 'POST' }); if (response.status === 409) { setProposal(null); await openTrip(trip.trip_id); return; } if (!response.ok) { window.alert('调整建议应用失败，请重试。'); return; } reconcileTrip((await response.json()) as TripPlan); setProposal(null); };
  const dismissProposal = async () => { if (!trip || !proposal) return; const response = await fetch(`${API_BASE}/api/trips/${trip.trip_id}/proposals/${proposal.proposal_id}/dismiss`, { method: 'POST' }); if (response.ok) setProposal(null); };
  const clearMemory = async () => { const responses = await Promise.all([fetch(`${API_BASE}/api/profile`, { method: 'DELETE' }), fetch(`${API_BASE}/api/travel-profile`, { method: 'DELETE' })]); if (responses.every((item) => item.ok)) await Promise.all([loadProfile(), loadTravelProfile()]); };
  const onSubmit = (event: FormEvent) => { event.preventDefault(); void send(question); };
  const configured = health?.amap_configured && health?.deepseek_configured;

  return <main className="app-shell">
    <header className="topbar"><button className="mobile-menu" onClick={() => setSidebarOpen((value) => !value)} aria-label="打开对话菜单">☰</button><div className="brand-mark" aria-hidden="true">✦</div><div><p className="eyebrow">TRAVEL ASSISTANT</p><h1>旅游出行小帮手</h1></div>{trip && <button className="mobile-itinerary-toggle" onClick={() => setItineraryOpen((value) => !value)}>行程</button>}<span className={`status-pill ${configured ? '' : 'warning'}`} title={configured ? '服务配置完整' : '需要配置 API Key'}><i /> {configured ? '高德实时数据' : '等待服务配置'}</span></header>
    <section className="workspace">
      <aside className={`history-panel ${sidebarOpen ? 'open' : ''}`}><button className="new-chat" onClick={() => void newConversation()}>＋ 新对话 / 新行程</button><p className="panel-label">当前内容</p><button className="history-item active">{trip?.name ?? messages[0]?.text.slice(0, 18) ?? '新的旅行咨询'}</button>{trips.length > 0 && <><p className="panel-label">我的行程</p><div className="trip-list">{trips.map((item) => <div className="trip-row" key={item.trip_id}><button className={`history-item ${trip?.trip_id === item.trip_id ? 'active' : ''}`} onClick={() => void openTrip(item.trip_id)}>{item.name}<small>{item.days.length || item.request.days} 天 · 第 {item.version} 版</small></button><button className="trip-delete" onClick={() => void deleteTrip(item)} aria-label={`删除${item.name}`} title="永久删除">×</button></div>)}</div></>}<div className="scope-card"><strong>我能帮你</strong><span>查询天气和出行建议</span><span>查找景点、餐饮和路线</span><span>安排并修改多日行程</span><span>记住你主动保存的偏好</span></div></aside>
      <section className={`chat-panel ${messages.length ? 'has-messages' : ''}`}>
        {messages.length === 0 ? <div className="welcome-card"><div className="sun-orbit" aria-hidden="true"><span>🧭</span></div><p className="eyebrow">从天气到每天的安排</p><h2>一起把旅程安排妥当。</h2><p className="welcome-copy">你可以先问天气，也可以直接告诉我目的地、天数和喜好。行程建好后，继续说“第二天呢”或“换个室内景点”就可以了。</p><div className="suggestions">{suggestions.map((item) => <button key={item} onClick={() => void send(item)}>{item}</button>)}</div></div> : <div className="message-list" aria-live="polite">{messages.map((message) => <article key={message.id} className={`message ${message.role} ${message.errorCode ? 'has-error' : ''}`}><div className="avatar" aria-hidden="true">{message.role === 'user' ? '你' : '✦'}</div><div className="message-content">
          {message.plan && <div className="plan-card" aria-label="处理步骤"><div className="plan-heading"><span>✦</span><strong>{message.plan.title}</strong><em>处理中</em></div><div className="plan-steps">{message.plan.steps.map((item, index) => <div className="plan-step" key={`${item}-${index}`}><b>{index + 1}</b><span>{item}</span></div>)}</div></div>}
          {message.steps && message.steps.length > 0 && <div className="steps">{message.steps.map((step) => <div className={`step ${step.status}`} key={step.step}><span>{step.status === 'complete' ? '✓' : step.status === 'error' ? '!' : step.status === 'needs_input' ? '?' : ''}</span>{step.label}</div>)}</div>}
          {message.tools && message.tools.length > 0 && <div className="tool-trace">{message.tools.slice(-6).map((item, index) => <span key={`${item.tool}-${index}`} className={item.status}>{item.status === 'complete' ? '✓ ' : item.status === 'error' ? '! ' : '↻ '}{item.label}</span>)}</div>}
          {message.text ? <p>{message.text}</p> : <div className="typing"><i /><i /><i /></div>}
          {message.proposal && <div className="inline-proposal"><strong>有一份待确认的调整建议</strong><span>行程尚未被修改，请在右侧查看差异并确认。</span></div>}
          {message.sources && message.sources.length > 0 && <div className="sources">{message.sources.map((source, index) => <span key={`${source.location}-${source.resource_id ?? source.reporttime}-${index}`}>高德 · {source.location} · {source.kind}{source.detail ? ` · ${source.detail}` : ''} · {source.reporttime}</span>)}</div>}
          {message.errorCode && <button className="retry-button" onClick={() => setQuestion(messages.findLast((item) => item.role === 'user')?.text ?? '')}>重新填写并发送</button>}
        </div></article>)}<div ref={bottomRef} /></div>}
        <div className="composer-wrap"><form className="composer" onSubmit={onSubmit}><label className="sr-only" htmlFor="assistant-question">询问旅行助手</label><input id="assistant-question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={trip ? `继续修改或询问“${trip.name}”…` : '问天气，或告诉我目的地、天数和偏好…'} disabled={busy} autoComplete="off" /><button type="submit" disabled={busy || !question.trim()} aria-label="发送">↑</button></form><p className="disclaimer">天气、景点和路线事实来自高德开放平台；行程调整会明确提示是否需要确认</p></div>
      </section>
      <aside className={`memory-panel itinerary-panel ${itineraryOpen ? 'open' : ''}`}>{trip ? <><div className="memory-heading"><span>{trip.name}</span><small>第 {trip.version} 版</small></div>{proposal && <div className="proposal-card"><strong>{proposal.title}</strong><p>{proposal.description}</p><div className="proposal-diff">{proposal.changes.map((change) => <span key={change}>{change}</span>)}</div><small>基于第 {proposal.based_on_version} 版，确认前不会修改</small><div className="proposal-actions"><button onClick={() => void applyProposal()}>确认调整</button><button className="secondary" onClick={() => void dismissProposal()}>暂不调整</button></div></div>}
        {trip.days.length === 0 ? <div className="memory-empty"><div aria-hidden="true">✦</div><h3>等待生成行程</h3><p>在聊天中告诉我目的地、天数和偏好。</p></div> : <div className="itinerary-list">{trip.days.map((day) => <div className="day-card" key={day.date}><div className="day-heading"><strong>{day.date}</strong><span>{day.weather_summary}</span>{day.sources?.[0]?.reporttime && <small>高德发布 {day.sources[0].reporttime}</small>}</div>{day.activities.map((activity) => <div className="activity-card" key={activity.id}><b>{activity.start_time && activity.end_time ? `${activity.start_time}–${activity.end_time}` : activity.period === 'morning' ? '上午' : activity.period === 'afternoon' ? '下午' : '晚上'}</b><strong>{activity.poi.name}</strong><small>{activity.indoor ? '室内活动' : '户外活动'} · {activity.duration_minutes} 分钟 · 高德地点</small>{activity.route_from_previous?.duration_s && <small>↳ 高德路线约 {Math.round(activity.route_from_previous.duration_s / 60)} 分钟</small>}</div>)}</div>)}</div>}
        {trip.warnings.length > 0 && <div className="warning-list">{trip.warnings.slice(0, 3).map((warning, index) => <div key={`${warning.type}-${index}`}>⚠ {warning.message}</div>)}</div>}<AmapMap days={trip.days} /><div className="trip-actions"><button onClick={() => { setQuestion('第二天的天气会影响行程吗？'); setItineraryOpen(false); }}>刷新天气</button><button onClick={() => { setQuestion('只调整第二天，减少户外活动'); setItineraryOpen(false); }}>修改当天</button><button onClick={() => { setQuestion('放慢节奏，减少每天活动'); setItineraryOpen(false); }}>放慢节奏</button><button className="danger" onClick={() => void deleteTrip(trip)}>删除行程</button></div>
      </> : <><div className="memory-heading"><span>长期记忆</span>{(profile?.default_location || (profile?.advice_preferences.length ?? 0) > 0 || travelProfile?.home_city || (travelProfile?.interests.length ?? 0) > 0) && <button onClick={() => void clearMemory()}>清空</button>}</div>{profile?.default_location || (profile?.advice_preferences.length ?? 0) > 0 || travelProfile?.home_city || (travelProfile?.interests.length ?? 0) > 0 ? <div className="memory-list">{travelProfile?.home_city && <div className="memory-card"><small>常住地</small><strong>{travelProfile.home_city}</strong></div>}{travelProfile && <div className="memory-card"><small>旅行节奏</small><strong>{travelProfile.pace === 'relaxed' ? '轻松' : travelProfile.pace === 'packed' ? '紧凑' : '均衡'}</strong></div>}{travelProfile?.interests.map((item) => <div className="memory-card" key={`travel-${item}`}><small>旅行兴趣</small><strong>{item}</strong></div>)}{profile?.default_location && <div className="memory-card"><small>天气默认地点</small><strong>{locationLabel(profile.default_location)}</strong><span>普通天气追问可继续沿用</span></div>}{profile?.advice_preferences.map((item) => <div className="memory-card" key={item}><small>天气建议偏好</small><strong>{item}</strong></div>)}</div> : <div className="memory-empty"><div aria-hidden="true">⌁</div><h3>还没有长期记忆</h3><p>试试说“记住我喜欢轻松节奏”，只有明确要求才会跨行程保存。</p></div>}<div className="privacy-note"><span>✓</span><p><strong>三层记忆</strong><br />最近对话、行程版本，以及你明确要求保存的偏好</p></div>{!configured && <div className="setup-note"><strong>服务配置不完整</strong><p>请确认后端密钥已配置，并使用重启脚本让前端重新读取地图环境变量。</p></div>}</>}</aside>
    </section>
  </main>;
}
