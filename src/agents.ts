import type { Params } from './params.ts';
import { rotate, wrap, wrapDeltaVec, type Vec2 } from './wrap.ts';

export type AgentKind = 'era' | 'dup' | 'con';
export type PortSlot = 'p' | 'l' | 'r';

export interface PortRef {
  id: number;
  slot: PortSlot;
}

export interface Agent {
  id: number;
  kind: AgentKind;
  x: number;
  y: number;
  vx: number;
  vy: number;
  heading: number;
  omega: number;
  mass: number;
  alpha: number;
  scale: number;
  locked: boolean;
  stun: number;
  /** Pose at the start of the current integrate step (for XPBD velocity writeback). */
  prevX: number;
  prevY: number;
  prevHeading: number;
  integVx: number;
  integVy: number;
  integOmega: number;
}

export function portKey(p: PortRef): string {
  return `${p.id}:${p.slot}`;
}

export function slotsFor(kind: AgentKind): PortSlot[] {
  return kind === 'era' ? ['p'] : ['p', 'l', 'r'];
}

export function massFor(kind: AgentKind, params: Params): number {
  return kind === 'era' ? params.eraMass : params.nodeMass;
}

export const ERA_RADIUS = 8;

export function agentSize(kind: AgentKind): number {
  return kind === 'era' ? 9 : 16;
}

/** Local-space triangle matching the drawn glyph (Con / Dup). */
export function triangleLocal(scale: number): Vec2[] {
  const s = 16 * scale;
  return [
    { x: s * 1.05, y: 0 },
    { x: -s * 0.55, y: -s * 0.82 },
    { x: -s * 0.55, y: s * 0.82 },
  ];
}

export function triangleWorld(agent: Agent, ox: number, oy: number): Vec2[] {
  return triangleLocal(agent.scale).map((p) => {
    const r = rotate(p.x, p.y, agent.heading);
    return { x: ox + r.x, y: oy + r.y };
  });
}

export function boundRadius(agent: Agent): number {
  if (agent.kind === 'era') return (ERA_RADIUS + 1.2) * agent.scale;
  return agentSize(agent.kind) * 1.12 * agent.scale;
}

export function collideRadius(agent: Agent): number {
  return boundRadius(agent);
}

export function momentOfInertia(agent: Agent): number {
  const m = Math.max(0.08, agent.mass);
  if (agent.kind === 'era') {
    const r = ERA_RADIUS * agent.scale;
    return 0.5 * m * r * r;
  }
  let s = 0;
  for (const p of triangleLocal(agent.scale)) s += p.x * p.x + p.y * p.y;
  return (m * s) / 6;
}

export const PORT_EXTRUDE = 8;
const HANDLE_SCALE = 3;

/** Where a port stem meets the body. */
export function stemRoot(kind: AgentKind, slot: PortSlot): Vec2 {
  if (kind === 'era') {
    return slot === 'p' ? { x: 8, y: 0 } : { x: 0, y: 0 };
  }
  const s = agentSize(kind);
  if (slot === 'p') return { x: s * 1.05, y: 0 };
  const baseX = -s * 0.55;
  const halfBase = s * 0.82;
  const legY = halfBase * 0.7;
  return { x: baseX, y: slot === 'l' ? -legY : legY };
}

/** Port tip (snap / wire endpoint). Principal from the apex; aux legs go backward, parallel. */
export function portLocal(kind: AgentKind, slot: PortSlot): Vec2 {
  const root = stemRoot(kind, slot);
  if (kind === 'era' || slot === 'p') {
    return { x: root.x + PORT_EXTRUDE, y: root.y };
  }
  return { x: root.x - PORT_EXTRUDE, y: root.y };
}

export function createAgent(
  id: number,
  kind: AgentKind,
  x: number,
  y: number,
  heading: number,
  params: Params,
): Agent {
  return {
    id,
    kind,
    x,
    y,
    vx: 0,
    vy: 0,
    heading,
    omega: 0,
    mass: massFor(kind, params),
    alpha: 1,
    scale: 1,
    locked: false,
    stun: 0,
    prevX: x,
    prevY: y,
    prevHeading: heading,
    integVx: 0,
    integVy: 0,
    integOmega: 0,
  };
}

export function portAxis(agent: Agent, slot: PortSlot): Vec2 {
  const tip = portLocal(agent.kind, slot);
  const root = stemRoot(agent.kind, slot);
  const r = rotate(
    (tip.x - root.x) * agent.scale,
    (tip.y - root.y) * agent.scale,
    agent.heading,
  );
  const len = Math.hypot(r.x, r.y) || 1;
  return { x: r.x / len, y: r.y / len };
}

export function inSnapArc(
  agent: Agent,
  slot: PortSlot,
  tx: number,
  ty: number,
  w: number,
  h: number,
  radius: number,
  halfArc: number,
): boolean {
  const p = portWorld(agent, slot, w, h);
  const d = wrapDeltaVec(p.x, p.y, tx, ty, w, h);
  const dist = Math.hypot(d.x, d.y);
  if (dist > radius || dist < 1e-6) return false;
  const axis = portAxis(agent, slot);
  const cos = (d.x * axis.x + d.y * axis.y) / dist;
  return cos >= Math.cos(Math.min(halfArc, Math.PI * 0.49));
}

export function portOffset(agent: Agent, slot: PortSlot): Vec2 {
  const loc = portLocal(agent.kind, slot);
  return rotate(loc.x * agent.scale, loc.y * agent.scale, agent.heading);
}

export function portWorld(agent: Agent, slot: PortSlot, w: number, h: number): Vec2 {
  const o = portOffset(agent, slot);
  return { x: wrap(agent.x + o.x, w), y: wrap(agent.y + o.y, h) };
}

export function stemOffset(agent: Agent, slot: PortSlot): Vec2 {
  const loc = stemRoot(agent.kind, slot);
  return rotate(loc.x * agent.scale, loc.y * agent.scale, agent.heading);
}

export function stemWorld(agent: Agent, slot: PortSlot, w: number, h: number): Vec2 {
  const o = stemOffset(agent, slot);
  return { x: wrap(agent.x + o.x, w), y: wrap(agent.y + o.y, h) };
}

/** Control point along the port axis, HANDLE_SCALE times the visible stem. */
export function handleWorld(agent: Agent, slot: PortSlot, w: number, h: number, _restLen?: number): Vec2 {
  const root = stemWorld(agent, slot, w, h);
  const tip = portWorld(agent, slot, w, h);
  const d = wrapDeltaVec(root.x, root.y, tip.x, tip.y, w, h);
  const seg = Math.hypot(d.x, d.y) || 1;
  const handle = HANDLE_SCALE * seg;
  return {
    x: wrap(root.x + (d.x / seg) * handle, w),
    y: wrap(root.y + (d.y / seg) * handle, h),
  };
}

export function wireCubic(
  A: Agent,
  aSlot: PortSlot,
  B: Agent,
  bSlot: PortSlot,
  w: number,
  h: number,
  restLen?: number,
): { p0: Vec2; p1: Vec2; p2: Vec2; p3: Vec2 } {
  const p0 = stemWorld(A, aSlot, w, h);
  const hA = handleWorld(A, aSlot, w, h, restLen);
  const hB = handleWorld(B, bSlot, w, h, restLen);
  const rootB = stemWorld(B, bSlot, w, h);
  const d1 = wrapDeltaVec(p0.x, p0.y, hA.x, hA.y, w, h);
  const d2 = wrapDeltaVec(p0.x, p0.y, hB.x, hB.y, w, h);
  const d3 = wrapDeltaVec(p0.x, p0.y, rootB.x, rootB.y, w, h);
  return {
    p0,
    p1: { x: p0.x + d1.x, y: p0.y + d1.y },
    p2: { x: p0.x + d2.x, y: p0.y + d2.y },
    p3: { x: p0.x + d3.x, y: p0.y + d3.y },
  };
}
