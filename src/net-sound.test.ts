import { describe, expect, it } from 'vitest';
import { portWorld, stemWorld } from './agents.ts';
import { AudioEngine } from './audio/engine.ts';
import { defaultParams } from './params.ts';
import {
  SOUND_PULL,
  attackSound,
  beginStroke,
  holdSound,
  moveStroke,
  pickSound,
  pickWire,
  releaseSound,
  strokePull,
} from './net-sound.ts';
import { spawnDesigned } from './net-edit.ts';
import { Sim } from './sim.ts';
import type { WorkletInMessage } from './audio/types.ts';

function scene() {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.snapRadius = 0;
  params.rewriteDuration = 0;
  const sim = new Sim(800, 600);
  const a = spawnDesigned(sim, 'con', 200, 300, 0, params)!;
  const b = spawnDesigned(sim, 'con', 400, 300, Math.PI, params)!;
  sim.wire(a.id, 'p', b.id, 'p', params);
  const wire = [...sim.graph.wires.values()][0]!;
  return { sim, params, a, b, wire };
}

function armed(sim: Sim): { engine: AudioEngine; posted: WorkletInMessage[] } {
  const engine = new AudioEngine();
  engine.armWithoutAudio();
  engine.frame(sim.graph, sim.agents);
  const posted: WorkletInMessage[] = [];
  engine.onPost = (m) => posted.push(m);
  return { engine, posted };
}

describe('pickSound', () => {
  it('prefers an open port, then a body, then a wire', () => {
    const { sim, a, b, wire } = scene();
    const left = portWorld(a, 'l', sim.w, sim.h);
    const hitPort = pickSound(sim, left.x, left.y, 1);
    expect(hitPort).toMatchObject({ kind: 'port', id: a.id, slot: 'l' });

    const hitBody = pickSound(sim, a.x, a.y, 1);
    expect(hitBody).toEqual({ kind: 'body', id: a.id });

    const sa = stemWorld(a, 'p', sim.w, sim.h);
    const sb = stemWorld(b, 'p', sim.w, sim.h);
    const mx = (sa.x + sb.x) * 0.5;
    const my = (sa.y + sb.y) * 0.5;
    const hitWire = pickSound(sim, mx, my, 1);
    expect(hitWire?.kind).toBe('wire');
    if (hitWire?.kind === 'wire') {
      expect(hitWire.id).toBe(wire.id);
      expect(hitWire.at).toBeGreaterThan(0.3);
      expect(hitWire.at).toBeLessThan(0.7);
    }
  });

  it('ignores empty space', () => {
    const { sim } = scene();
    expect(pickSound(sim, 40, 40, 1)).toBeNull();
    expect(pickWire(sim, 40, 40, 1)).toBeNull();
  });
});

describe('sound gestures', () => {
  it('taps a body into a strike and a wire into a narrow pluck', () => {
    const { sim, a, b, wire } = scene();
    const { engine, posted } = armed(sim);

    const body = beginStroke({ kind: 'body', id: a.id }, a.x, a.y, 0, 0.5);
    attackSound(sim, engine, body);
    expect(posted.some((m) => m.type === 'strike' && m.agentId === a.id)).toBe(true);

    posted.length = 0;
    const sa = stemWorld(a, 'p', sim.w, sim.h);
    const sb = stemWorld(b, 'p', sim.w, sim.h);
    const mx = (sa.x + sb.x) * 0.5;
    const my = (sa.y + sb.y) * 0.5;
    const wireHit = pickSound(sim, mx, my, 1);
    expect(wireHit?.kind).toBe('wire');
    const tap = beginStroke(wireHit!, mx, my, 0, 0.5);
    attackSound(sim, engine, tap);
    expect(posted.filter((m) => m.type === 'pluck')).toHaveLength(0);
    releaseSound(sim, engine, tap);
    const plucks = posted.filter((m) => m.type === 'pluck');
    expect(plucks).toHaveLength(1);
    if (plucks[0]?.type === 'pluck') {
      expect(plucks[0].wireId).toBe(wire.id);
      expect(plucks[0].width).toBeLessThan(1);
    }
  });

  it('pulls a wire into a guitar pluck and a port into a breath', () => {
    const { sim, a, b, wire } = scene();
    const { engine, posted } = armed(sim);
    const sa = stemWorld(a, 'p', sim.w, sim.h);
    const sb = stemWorld(b, 'p', sim.w, sim.h);
    const mx = (sa.x + sb.x) * 0.5;
    const my = (sa.y + sb.y) * 0.5;
    const hit = pickSound(sim, mx, my, 1)!;
    const stroke = beginStroke(hit, mx, my, 0, 0.5);
    moveStroke(stroke, mx, my + SOUND_PULL + 4, 80, 0.5);
    expect(strokePull(stroke)).toBeGreaterThan(SOUND_PULL);
    releaseSound(sim, engine, stroke);
    const pluck = posted.find((m) => m.type === 'pluck');
    expect(pluck?.type).toBe('pluck');
    if (pluck?.type === 'pluck') {
      expect(pluck.wireId).toBe(wire.id);
      expect(pluck.width).toBe(1);
      expect(pluck.gain).toBeGreaterThan(0.4);
    }

    posted.length = 0;
    const left = portWorld(a, 'l', sim.w, sim.h);
    const port = beginStroke(
      { kind: 'port', id: a.id, slot: 'l', x: left.x, y: left.y },
      left.x,
      left.y,
      0,
      0.5,
    );
    attackSound(sim, engine, port);
    expect(posted.some((m) => m.type === 'junction' && m.agentId === a.id)).toBe(true);
  });

  it('scrapes a wired body as a contact pair', () => {
    const { sim, a, b } = scene();
    const engine = new AudioEngine();
    const stroke = beginStroke({ kind: 'body', id: a.id }, a.x, a.y, 0, 0.5);
    moveStroke(stroke, a.x + 6, a.y, 40, 0.7);
    holdSound(sim, engine, stroke, 40);
    expect(engine.pointerContact).toEqual({
      agentA: a.id,
      agentB: b.id,
      load: expect.any(Number),
      slide: expect.any(Number),
    });
    releaseSound(sim, engine, stroke);
    expect(engine.pointerContact).toBeNull();
  });
});
