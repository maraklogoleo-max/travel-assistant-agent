import { describe, expect, it } from 'vitest';
import { consumeSse } from './sse';

describe('consumeSse', () => {
  it('parses step and final events split across network chunks', () => {
    const first = consumeSse('data: {"type":"step","step":"weather","status":"running","label":"查询天');
    expect(first.events).toHaveLength(0);
    const second = consumeSse(first.rest + '气"}\n\ndata: {"type":"final","answer":"晴","sources":[]}\n\n');
    expect(second.events).toHaveLength(2);
    expect(second.events[0].type).toBe('step');
    expect(second.events[1]).toMatchObject({ type: 'final', answer: '晴' });
  });

  it('ignores malformed events without breaking later frames', () => {
    const result = consumeSse('data: not-json\n\ndata: {"type":"error","code":"x","message":"失败","retryable":true}\n\n');
    expect(result.events).toEqual([
      { type: 'error', code: 'x', message: '失败', retryable: true },
    ]);
  });

  it('parses incremental answer tokens', () => {
    const result = consumeSse('data: {"type":"token","delta":"天气很好"}\n\n');
    expect(result.events).toEqual([{ type: 'token', delta: '天气很好' }]);
  });

  it('parses travel plan and itinerary patches', () => {
    const result = consumeSse('data: {"type":"plan","task_type":"trip_planning","title":"目的地行程规划","steps":["搜索景点"]}\n\ndata: {"type":"itinerary_patch","days":[],"warnings":[]}\n\n');
    expect(result.events[0]).toMatchObject({ type: 'plan', task_type: 'trip_planning' });
    expect(result.events[1]).toMatchObject({ type: 'itinerary_patch', days: [] });
  });

  it('parses a confirmable change proposal and latest trip in final', () => {
    const result = consumeSse(
      'data: {"type":"change_proposal","proposal":{"proposal_id":"p1","trip_id":"t1","based_on_version":2,"kind":"weather","title":"雨天调整","description":"替换户外活动","changes":[],"proposed_plan":{"trip_id":"t1"},"status":"pending"}}\n\n' +
      'data: {"type":"final","answer":"待确认","sources":[],"trip":{"trip_id":"t1","version":2},"conversation_id":"c1"}\n\n',
    );
    expect(result.events[0]).toMatchObject({ type: 'change_proposal', proposal: { proposal_id: 'p1' } });
    expect(result.events[1]).toMatchObject({ type: 'final', trip: { version: 2 }, conversation_id: 'c1' });
  });

  it('parses autonomous actions, clarification choices and provenance', () => {
    const result = consumeSse(
      'data: {"type":"agent_action","action":"resolve_location","objective":"确认目的地","sequence":1}\n\n' +
      'data: {"type":"clarification","code":"LOCATION_AMBIGUOUS","message":"请选择地点","choices":[{"label":"北京市朝阳区","value":"北京市朝阳区"}]}\n\n' +
      'data: {"type":"final","answer":"请选择地点","sources":[{"provider":"高德开放平台","location":"北京市朝阳区","kind":"地点","reporttime":"2026-08-25","resource_id":"110105"}]}\n\n',
    );
    expect(result.events[0]).toMatchObject({ type: 'agent_action', action: 'resolve_location', sequence: 1 });
    expect(result.events[1]).toMatchObject({ type: 'clarification', code: 'LOCATION_AMBIGUOUS' });
    expect(result.events[2]).toMatchObject({ type: 'final', sources: [{ kind: '地点', resource_id: '110105' }] });
  });
});
