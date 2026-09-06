import { portWorld, stemWorld, type PortSlot } from './agents.ts';
import type { AudioEngine } from './audio/engine.ts';
import { bowSpeed, contactPeak, contactSeconds, getSampleRate } from './audio/presets.ts';
import { strikeSharpness } from './audio/voice.ts';
import { closestPointOnSegment } from './geom.ts';
import { pickAgent, pickPort } from './interact.ts';
import type { Sim } from './sim.ts';

const WIRE_REACH = 10;
/** World travel below this is a tap, above it a pull. */
export const SOUND_PULL = 10;
const BOW_PERIOD = 80;
const TAP_SPEED = 80;

export type SoundHit =
  | { kind: 'port'; id: number; slot: PortSlot; x: number; y: number }
  | { kind: 'body'; id: number }
  | { kind: 'wire'; id: number; at: number; x: number; y: number };

export interface SoundStroke {
  hit: SoundHit;
  x0: number;
  y0: number;
  x: number;
  y: number;
  t0: number;
  lastT: number;
  vx: number;
  vy: number;
  pressure: number;
  lastBow: number;
}

/** Mouse reports 0 or 0.5; treat a missing reading as a medium press. */
export function pointerPressure(n: number): number {
  if (!(n > 0)) return 0.5;
  return n > 1 ? 1 : n;
}

export function pickWire(
  sim: Sim,
  x: number,
  y: number,
  zoom: number,
): { id: number; at: number; x: number; y: number; dist: number } | null {
  const reach = WIRE_REACH / Math.max(0.2, zoom);
  const reach2 = reach * reach;
  let best: { id: number; at: number; x: number; y: number; dist: number } | null = null;
  let bestD = reach2;
  for (const w of sim.graph.wires.values()) {
    const A = sim.agents.get(w.a.id);
    const B = sim.agents.get(w.b.id);
    if (!A || !B) continue;
    const a = stemWorld(A, w.a.slot, sim.w, sim.h);
    const b = stemWorld(B, w.b.slot, sim.w, sim.h);
    const p = closestPointOnSegment(x, y, a.x, a.y, b.x, b.y);
    const d = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (d < bestD) {
      bestD = d;
      best = { id: w.id, at: p.t, x: p.x, y: p.y, dist: Math.sqrt(d) };
    }
  }
  return best;
}

/** Open port, then body, then wire. */
export function pickSound(sim: Sim, x: number, y: number, zoom: number): SoundHit | null {
  const port = pickPort(sim, x, y, zoom);
  if (port) {
    const a = sim.agents.get(port.id);
    if (a) {
      const p = portWorld(a, port.slot, sim.w, sim.h);
      return { kind: 'port', id: port.id, slot: port.slot, x: p.x, y: p.y };
    }
  }
  const body = pickAgent(sim, x, y);
  if (body) return { kind: 'body', id: body.id };
  const wire = pickWire(sim, x, y, zoom);
  if (wire) return { kind: 'wire', id: wire.id, at: wire.at, x: wire.x, y: wire.y };
  return null;
}

export function beginStroke(hit: SoundHit, x: number, y: number, now: number, pressure: number): SoundStroke {
  return {
    hit,
    x0: x,
    y0: y,
    x,
    y,
    t0: now,
    lastT: now,
    vx: 0,
    vy: 0,
    pressure: pointerPressure(pressure),
    lastBow: now - BOW_PERIOD,
  };
}

export function moveStroke(s: SoundStroke, x: number, y: number, now: number, pressure: number): void {
  const dt = Math.max(1e-3, (now - s.lastT) / 1000);
  s.vx = (x - s.x) / dt;
  s.vy = (y - s.y) / dt;
  s.x = x;
  s.y = y;
  s.lastT = now;
  s.pressure = pointerPressure(pressure);
}

export function strokePull(s: SoundStroke): number {
  return Math.hypot(s.x - s.x0, s.y - s.y0);
}

export function strokeSpeed(s: SoundStroke): number {
  return Math.hypot(s.vx, s.vy);
}

function wiredNeighbor(sim: Sim, id: number): number | null {
  for (const w of sim.graph.wires.values()) {
    if (w.a.id === id) return w.b.id;
    if (w.b.id === id) return w.a.id;
  }
  return null;
}

function thumpBody(sim: Sim, audio: AudioEngine, id: number, x: number, y: number, speed: number): void {
  const a = sim.agents.get(id);
  if (!a) return;
  const dx = x - a.x;
  const dy = y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const vN = Math.max(TAP_SPEED, speed);
  const tau = contactSeconds(a.mass, vN);
  audio.playStrike(
    id,
    contactPeak(a.mass, vN, tau),
    Math.max(2, Math.round(tau * getSampleRate())),
    strikeSharpness(a, dx / len, dy / len),
  );
}

/** Attack on press: bodies thump, ports puff. Wires wait for release. */
export function attackSound(sim: Sim, audio: AudioEngine, s: SoundStroke): void {
  if (s.hit.kind === 'body') thumpBody(sim, audio, s.hit.id, s.x, s.y, strokeSpeed(s));
  if (s.hit.kind === 'port') {
    const gain = 0.28 + 0.5 * Math.min(1, strokeSpeed(s) / 600) + 0.25 * (s.pressure - 0.5);
    audio.playJunction(s.hit.id, Math.max(0.12, gain));
  }
}

/** Continuous excitation while the pointer is down. */
export function holdSound(sim: Sim, audio: AudioEngine, s: SoundStroke, now: number): void {
  const pull = strokePull(s);
  const speed = strokeSpeed(s);
  const load = Math.max(0.15, Math.min(1, 0.35 + 0.5 * s.pressure + Math.min(1, pull / 40) * 0.3));

  if (s.hit.kind === 'body') {
    const other = wiredNeighbor(sim, s.hit.id);
    if (other !== null && speed > 20) {
      audio.pointerContact = { agentA: s.hit.id, agentB: other, load, slide: bowSpeed(speed) };
    } else {
      audio.pointerContact = null;
    }
    return;
  }

  audio.pointerContact = null;
  if (now - s.lastBow < BOW_PERIOD) return;

  if (s.hit.kind === 'port') {
    s.lastBow = now;
    audio.playJunction(s.hit.id, 0.08 * load);
    return;
  }

  if (s.hit.kind !== 'wire') return;
  if (wireAcross(sim, s) > SOUND_PULL) return;
  if (wireAlong(sim, s) > 12 && speed > 40) {
    s.lastBow = now;
    audio.playPluck(s.hit.id, Math.min(0.45, 0.08 + speed / 1800), s.hit.at, 0.35);
  }
}

/** Release: pulled wires pluck; a short wire tap mallets. */
export function releaseSound(sim: Sim, audio: AudioEngine, s: SoundStroke): void {
  audio.pointerContact = null;
  const pull = strokePull(s);
  if (s.hit.kind === 'body') {
    if (pull >= SOUND_PULL) thumpBody(sim, audio, s.hit.id, s.x, s.y, Math.max(strokeSpeed(s), pull * 8));
    return;
  }
  if (s.hit.kind === 'port') {
    if (pull >= SOUND_PULL) audio.playJunction(s.hit.id, Math.min(1.2, 0.2 + pull / 50));
    return;
  }
  if (s.hit.kind !== 'wire') return;
  if (pull >= SOUND_PULL) {
    audio.playPluck(s.hit.id, Math.min(1.6, 0.25 + pull / 36), wireAtFromPull(sim, s), 1);
    return;
  }
  audio.playPluck(s.hit.id, Math.min(1.1, 0.28 + strokeSpeed(s) / 900), s.hit.at, 0.45);
}

function wireEnds(sim: Sim, id: number): { ax: number; ay: number; bx: number; by: number } | null {
  const w = sim.graph.wires.get(id);
  if (!w) return null;
  const A = sim.agents.get(w.a.id);
  const B = sim.agents.get(w.b.id);
  if (!A || !B) return null;
  const a = stemWorld(A, w.a.slot, sim.w, sim.h);
  const b = stemWorld(B, w.b.slot, sim.w, sim.h);
  return { ax: a.x, ay: a.y, bx: b.x, by: b.y };
}

function wireAlong(sim: Sim, s: SoundStroke): number {
  if (s.hit.kind !== 'wire') return 0;
  const e = wireEnds(sim, s.hit.id);
  if (!e) return 0;
  const tx = e.bx - e.ax;
  const ty = e.by - e.ay;
  const len = Math.hypot(tx, ty) || 1;
  return Math.abs(((s.x - s.x0) * tx + (s.y - s.y0) * ty) / len);
}

function wireAcross(sim: Sim, s: SoundStroke): number {
  if (s.hit.kind !== 'wire') return 0;
  const e = wireEnds(sim, s.hit.id);
  if (!e) return 0;
  const tx = e.bx - e.ax;
  const ty = e.by - e.ay;
  const len = Math.hypot(tx, ty) || 1;
  return Math.abs(((s.x - s.x0) * -ty + (s.y - s.y0) * tx) / len);
}

function wireAtFromPull(sim: Sim, s: SoundStroke): number {
  if (s.hit.kind !== 'wire') return 0.5;
  const e = wireEnds(sim, s.hit.id);
  if (!e) return s.hit.at;
  return closestPointOnSegment(s.x0, s.y0, e.ax, e.ay, e.bx, e.by).t;
}
