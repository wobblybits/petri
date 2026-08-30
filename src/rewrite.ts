import { createAgent, portWorld, type Agent, type AgentKind, type PortRef } from './agents.ts';
import type { Graph } from './graph.ts';
import type { Params } from './params.ts';
import { angleDelta, clamp, easeInOut, lerp, wrap, wrapDeltaVec, wrapMid } from './wrap.ts';

export type Rule = 'era-era' | 'erase' | 'annihilate-con' | 'annihilate-dup' | 'commute';

export interface NetWire {
  a: PortRef;
  b: PortRef;
}

export interface NetAgent {
  id: number;
  kind: AgentKind;
}

export interface NetSnapshot {
  agents: NetAgent[];
  wires: NetWire[];
}

export interface Spawned {
  id: number;
  kind: AgentKind;
  role: 'era-l' | 'era-r' | 'con-u' | 'con-v' | 'dup-x' | 'dup-y';
}

export interface ApplyResult {
  net: NetSnapshot;
  nextId: number;
  spawned: Spawned[];
}

export interface Ghost {
  kind: AgentKind;
  x: number;
  y: number;
  heading: number;
  alpha: number;
  scale: number;
}

export interface Rewrite {
  rule: Rule;
  t: number;
  duration: number;
  a: number;
  b: number;
  eraId: number;
  binaryId: number;
  conId: number;
  dupId: number;
  ax: number;
  ay: number;
  ah: number;
  bx: number;
  by: number;
  bh: number;
  midX: number;
  midY: number;
  leftoverAL: PortRef | null;
  leftoverAR: PortRef | null;
  leftoverBL: PortRef | null;
  leftoverBR: PortRef | null;
  ghosts: Ghost[];
  targets: Ghost[];
}

function wireOf(wires: NetWire[], port: PortRef): NetWire | undefined {
  return wires.find(
    (w) =>
      (w.a.id === port.id && w.a.slot === port.slot) ||
      (w.b.id === port.id && w.b.slot === port.slot),
  );
}

export function otherEndOf(wire: NetWire, port: PortRef): PortRef {
  if (wire.a.id === port.id && wire.a.slot === port.slot) return wire.b;
  return wire.a;
}

export function leftoverOf(
  wires: NetWire[],
  agentId: number,
  slot: PortRef['slot'],
  dying: Set<number>,
): PortRef | null {
  const w = wireOf(wires, { id: agentId, slot });
  if (!w) return null;
  const o = otherEndOf(w, { id: agentId, slot });
  if (dying.has(o.id)) return null;
  return o;
}

function connected(wires: NetWire[], a: PortRef, b: PortRef): boolean {
  return wires.some(
    (w) =>
      (w.a.id === a.id && w.a.slot === a.slot && w.b.id === b.id && w.b.slot === b.slot) ||
      (w.b.id === a.id && w.b.slot === a.slot && w.a.id === b.id && w.a.slot === b.slot),
  );
}

export function portsConnected(net: NetSnapshot, a: PortRef, b: PortRef): boolean {
  return connected(net.wires, a, b);
}

function pushWire(wires: NetWire[], a: PortRef | null, b: PortRef | null): void {
  if (!a || !b) return;
  if (a.id === b.id && a.slot === b.slot) return;
  wires.push({ a, b });
}

function stripAgents(net: NetSnapshot, dying: Set<number>): NetSnapshot {
  return {
    agents: net.agents.filter((a) => !dying.has(a.id)),
    wires: net.wires.filter((w) => !dying.has(w.a.id) && !dying.has(w.b.id)),
  };
}

export function detectRule(kindA: AgentKind, kindB: AgentKind): Rule {
  if (kindA === 'era' && kindB === 'era') return 'era-era';
  if (kindA === 'era' || kindB === 'era') return 'erase';
  if (kindA === 'con' && kindB === 'con') return 'annihilate-con';
  if (kindA === 'dup' && kindB === 'dup') return 'annihilate-dup';
  return 'commute';
}


/**
 * Reconnect what is left when two agents die.
 *
 * A rule says which of the dying agents' ports get identified with each other.
 * Following one wire out from each is not enough: a port can lead straight back
 * into the dying pair, and then on again. The identity function is exactly that
 * shape — `λx.x` is a Con with its own two aux ports wired together — so an
 * application of it used to lose both of its connections and silently drop the
 * result on the floor.
 *
 * Union-find over ports instead: fuse every wire touching a dying agent and
 * every identification the rule makes, then emit one wire per class that still
 * has two surviving ends. A class with none was a closed loop and correctly
 * disappears; a class with one had a free end and stays free.
 */
function fuse(
  net: NetSnapshot,
  dying: Set<number>,
  identify: [PortRef, PortRef][],
): NetWire[] {
  const parent = new Map<string, string>();
  const key = (p: PortRef): string => `${p.id}.${p.slot}`;
  const ports = new Map<string, PortRef>();
  const find = (x: string): string => {
    let r = x;
    while ((parent.get(r) ?? r) !== r) r = parent.get(r) ?? r;
    let cur = x;
    while ((parent.get(cur) ?? cur) !== r) {
      const nxt = parent.get(cur) ?? cur;
      parent.set(cur, r);
      cur = nxt;
    }
    return r;
  };
  const add = (p: PortRef): string => {
    const k = key(p);
    if (!parent.has(k)) {
      parent.set(k, k);
      ports.set(k, p);
    }
    return k;
  };
  const union = (x: PortRef, y: PortRef): void => {
    const rx = find(add(x));
    const ry = find(add(y));
    if (rx !== ry) parent.set(rx, ry);
  };

  for (const w of net.wires) {
    if (dying.has(w.a.id) || dying.has(w.b.id)) union(w.a, w.b);
  }
  for (const [x, y] of identify) union(x, y);

  const classes = new Map<string, PortRef[]>();
  for (const [k, p] of ports) {
    if (dying.has(p.id)) continue;
    const r = find(k);
    const list = classes.get(r) ?? [];
    list.push(p);
    classes.set(r, list);
  }
  const out: NetWire[] = [];
  for (const ends of classes.values()) {
    if (ends.length === 2) out.push({ a: ends[0], b: ends[1] });
  }
  return out;
}

export function applyRewrite(
  net: NetSnapshot,
  rule: Rule,
  a: number,
  b: number,
  nextId: number,
): ApplyResult {
  const dying = new Set([a, b]);
  const kinds = new Map(net.agents.map((ag) => [ag.id, ag.kind]));
  const kindA = kinds.get(a);
  const kindB = kinds.get(b);
  if (!kindA || !kindB) return { net, nextId, spawned: [] };

  if (rule === 'era-era') {
    return { net: stripAgents(net, dying), nextId, spawned: [] };
  }

  if (rule === 'annihilate-dup') {
    // Straight through: l to l, r to r.
    const joined = fuse(net, dying, [
      [{ id: a, slot: 'l' }, { id: b, slot: 'l' }],
      [{ id: a, slot: 'r' }, { id: b, slot: 'r' }],
    ]);
    const next = stripAgents(net, dying);
    next.wires.push(...joined);
    return { net: next, nextId, spawned: [] };
  }

  if (rule === 'annihilate-con') {
    // Crossed: l to the other's r. This is beta reduction when the pair is an
    // application meeting an abstraction.
    const joined = fuse(net, dying, [
      [{ id: a, slot: 'l' }, { id: b, slot: 'r' }],
      [{ id: a, slot: 'r' }, { id: b, slot: 'l' }],
    ]);
    const next = stripAgents(net, dying);
    next.wires.push(...joined);
    return { net: next, nextId, spawned: [] };
  }

  if (rule === 'erase') {
    const eraId = kindA === 'era' ? a : b;
    const binId = eraId === a ? b : a;
    const eL: Spawned = { id: nextId++, kind: 'era', role: 'era-l' };
    const eR: Spawned = { id: nextId++, kind: 'era', role: 'era-r' };
    const joined = fuse(net, dying, [
      [{ id: binId, slot: 'l' }, { id: eL.id, slot: 'p' }],
      [{ id: binId, slot: 'r' }, { id: eR.id, slot: 'p' }],
    ]);
    const next = stripAgents(net, dying);
    next.agents.push({ id: eL.id, kind: 'era' }, { id: eR.id, kind: 'era' });
    next.wires.push(...joined);
    return { net: next, nextId, spawned: [eL, eR] };
  }

  const conId = kindA === 'con' ? a : b;
  const dupId = kindA === 'dup' ? a : b;
  const Cu: Spawned = { id: nextId++, kind: 'con', role: 'con-u' };
  const Cv: Spawned = { id: nextId++, kind: 'con', role: 'con-v' };
  const Dx: Spawned = { id: nextId++, kind: 'dup', role: 'dup-x' };
  const Dy: Spawned = { id: nextId++, kind: 'dup', role: 'dup-y' };
  const joined = fuse(net, dying, [
    [{ id: dupId, slot: 'l' }, { id: Cu.id, slot: 'p' }],
    [{ id: dupId, slot: 'r' }, { id: Cv.id, slot: 'p' }],
    [{ id: conId, slot: 'l' }, { id: Dx.id, slot: 'p' }],
    [{ id: conId, slot: 'r' }, { id: Dy.id, slot: 'p' }],
  ]);
  const next = stripAgents(net, dying);
  next.agents.push(
    { id: Cu.id, kind: 'con' },
    { id: Cv.id, kind: 'con' },
    { id: Dx.id, kind: 'dup' },
    { id: Dy.id, kind: 'dup' },
  );
  next.wires.push(...joined);
  pushWire(next.wires, { id: Cu.id, slot: 'l' }, { id: Dx.id, slot: 'l' });
  pushWire(next.wires, { id: Cu.id, slot: 'r' }, { id: Dy.id, slot: 'l' });
  pushWire(next.wires, { id: Cv.id, slot: 'l' }, { id: Dx.id, slot: 'r' });
  pushWire(next.wires, { id: Cv.id, slot: 'r' }, { id: Dy.id, slot: 'r' });
  return { net: next, nextId, spawned: [Cu, Cv, Dx, Dy] };
}

export function snapshotOf(
  agents: Map<number, Agent>,
  graph: Graph,
): NetSnapshot {
  return {
    agents: [...agents.values()].map((a) => ({ id: a.id, kind: a.kind })),
    wires: [...graph.wires.values()].map((w) => ({ a: w.a, b: w.b })),
  };
}

function headingTo(
  fromX: number,
  fromY: number,
  to: PortRef | null,
  agents: Map<number, Agent>,
  w: number,
  h: number,
  fallback: number,
): number {
  if (!to) return fallback;
  const ag = agents.get(to.id);
  if (!ag) return fallback;
  const p = portWorld(ag, to.slot, w, h);
  const d = wrapDeltaVec(fromX, fromY, p.x, p.y, w, h);
  if (d.x * d.x + d.y * d.y < 1e-6) return fallback;
  return Math.atan2(d.y, d.x);
}

export function beginRewrite(
  agentA: Agent,
  agentB: Agent,
  graph: Graph,
  agents: Map<number, Agent>,
  w: number,
  h: number,
  duration: number,
): Rewrite {
  const rule = detectRule(agentA.kind, agentB.kind);
  const dying = new Set([agentA.id, agentB.id]);
  const eraId = agentA.kind === 'era' ? agentA.id : agentB.kind === 'era' ? agentB.id : -1;
  const binaryId =
    eraId === agentA.id ? agentB.id : eraId === agentB.id ? agentA.id : -1;
  const conId = agentA.kind === 'con' ? agentA.id : agentB.kind === 'con' ? agentB.id : -1;
  const dupId = agentA.kind === 'dup' ? agentA.id : agentB.kind === 'dup' ? agentB.id : -1;
  const mid = wrapMid(agentA.x, agentA.y, agentB.x, agentB.y, w, h);
  agentA.locked = true;
  agentB.locked = true;
  const rw: Rewrite = {
    rule,
    t: 0,
    duration,
    a: agentA.id,
    b: agentB.id,
    eraId,
    binaryId,
    conId,
    dupId,
    ax: agentA.x,
    ay: agentA.y,
    ah: agentA.heading,
    bx: agentB.x,
    by: agentB.y,
    bh: agentB.heading,
    midX: mid.x,
    midY: mid.y,
    leftoverAL: graph.leftover({ id: agentA.id, slot: 'l' }, dying),
    leftoverAR: graph.leftover({ id: agentA.id, slot: 'r' }, dying),
    leftoverBL: graph.leftover({ id: agentB.id, slot: 'l' }, dying),
    leftoverBR: graph.leftover({ id: agentB.id, slot: 'r' }, dying),
    ghosts: [],
    targets: [],
  };
  rw.targets = snapshotTargets(rw, agents, w, h);
  return rw;
}

export function advanceRewrite(
  rw: Rewrite,
  agents: Map<number, Agent>,
  w: number,
  h: number,
  dt: number,
): boolean {
  rw.t += dt / Math.max(0.05, rw.duration);
  const t = clamp(rw.t, 0, 1);
  const e = easeInOut(t);
  const A = agents.get(rw.a);
  const B = agents.get(rw.b);
  if (!A || !B) return t >= 1;

  const toB = wrapDeltaVec(rw.ax, rw.ay, rw.bx, rw.by, w, h);
  const toA = wrapDeltaVec(rw.bx, rw.by, rw.ax, rw.ay, w, h);

  if (rw.rule === 'era-era' || rw.rule === 'annihilate-con' || rw.rule === 'annihilate-dup') {
    A.x = wrap(rw.ax + toB.x * e * 0.5, w);
    A.y = wrap(rw.ay + toB.y * e * 0.5, h);
    B.x = wrap(rw.bx + toA.x * e * 0.5, w);
    B.y = wrap(rw.by + toA.y * e * 0.5, h);
    A.scale = B.scale = lerp(1, 0.15, e);
    A.alpha = B.alpha = 1 - e;
    A.heading = rw.ah + angleDelta(rw.ah, Math.atan2(toB.y, toB.x)) * e;
    B.heading = rw.bh + angleDelta(rw.bh, Math.atan2(toA.y, toA.x)) * e;
    rw.ghosts = [];
  } else if (rw.rule === 'erase') {
    const era = A.kind === 'era' ? A : B;
    const bin = era === A ? B : A;
    const es = era === A ? { x: rw.ax, y: rw.ay } : { x: rw.bx, y: rw.by };
    const bs = era === A ? { x: rw.bx, y: rw.by } : { x: rw.ax, y: rw.ay };
    const d = wrapDeltaVec(es.x, es.y, bs.x, bs.y, w, h);
    era.x = wrap(es.x + d.x * e, w);
    era.y = wrap(es.y + d.y * e, h);
    era.scale = lerp(1, 0.4, e);
    era.alpha = 1 - e;
    bin.scale = lerp(1, 0.2, e);
    bin.alpha = 1 - e * 0.85;
    const appear = clamp((t - 0.35) / 0.65, 0, 1);
    const ae = easeInOut(appear);
    rw.ghosts = rw.targets.map((g) => ({
      kind: g.kind,
      x: wrap(rw.midX + wrapDeltaVec(rw.midX, rw.midY, g.x, g.y, w, h).x * ae, w),
      y: wrap(rw.midY + wrapDeltaVec(rw.midX, rw.midY, g.x, g.y, w, h).y * ae, h),
      heading: g.heading,
      alpha: appear,
      scale: lerp(0.3, 1, ae),
    }));
  } else {
    A.x = wrap(rw.ax + toB.x * e, w);
    A.y = wrap(rw.ay + toB.y * e, h);
    B.x = wrap(rw.bx + toA.x * e, w);
    B.y = wrap(rw.by + toA.y * e, h);
    A.alpha = B.alpha = 1 - e;
    A.scale = B.scale = lerp(1, 0.35, e);
    const appear = clamp((t - 0.35) / 0.65, 0, 1);
    const ae = easeInOut(appear);
    rw.ghosts = rw.targets.map((g) => ({
      kind: g.kind,
      x: wrap(rw.midX + wrapDeltaVec(rw.midX, rw.midY, g.x, g.y, w, h).x * ae, w),
      y: wrap(rw.midY + wrapDeltaVec(rw.midX, rw.midY, g.x, g.y, w, h).y * ae, h),
      heading: g.heading,
      alpha: appear,
      scale: lerp(0.25, 1, ae),
    }));
  }
  return t >= 1;
}

function leftoverForBinary(rw: Rewrite, slot: 'l' | 'r'): PortRef | null {
  if (rw.binaryId === rw.a) return slot === 'l' ? rw.leftoverAL : rw.leftoverAR;
  return slot === 'l' ? rw.leftoverBL : rw.leftoverBR;
}

function leftoverPair(rw: Rewrite, who: 'dup' | 'con', slot: 'l' | 'r'): PortRef | null {
  const id = who === 'dup' ? rw.dupId : rw.conId;
  if (id === rw.a) return slot === 'l' ? rw.leftoverAL : rw.leftoverAR;
  return slot === 'l' ? rw.leftoverBL : rw.leftoverBR;
}

function pushOut(
  midX: number,
  midY: number,
  x: number,
  y: number,
  w: number,
  h: number,
  extra: number,
): { x: number; y: number } {
  const d = wrapDeltaVec(midX, midY, x, y, w, h);
  const m = Math.hypot(d.x, d.y);
  if (m < 1e-4) return { x, y };
  const s = (m + extra) / m;
  return { x: wrap(midX + d.x * s, w), y: wrap(midY + d.y * s, h) };
}

function snapshotTargets(
  rw: Rewrite,
  agents: Map<number, Agent>,
  w: number,
  h: number,
): Ghost[] {
  const outward = (x: number, y: number) => {
    const d = wrapDeltaVec(rw.midX, rw.midY, x, y, w, h);
    if (d.x * d.x + d.y * d.y < 1e-6) return 0;
    return Math.atan2(d.y, d.x);
  };
  if (rw.rule === 'erase') {
    const bin = agents.get(rw.binaryId);
    if (!bin) return [];
    const lp = portWorld(bin, 'l', w, h);
    const rp = portWorld(bin, 'r', w, h);
    return [
      {
        kind: 'era',
        x: lp.x,
        y: lp.y,
        heading: headingTo(lp.x, lp.y, leftoverForBinary(rw, 'l'), agents, w, h, outward(lp.x, lp.y)),
        alpha: 1,
        scale: 1,
      },
      {
        kind: 'era',
        x: rp.x,
        y: rp.y,
        heading: headingTo(rp.x, rp.y, leftoverForBinary(rw, 'r'), agents, w, h, outward(rp.x, rp.y)),
        alpha: 1,
        scale: 1,
      },
    ];
  }
  if (rw.rule !== 'commute') return [];
  const con = agents.get(rw.conId);
  const dup = agents.get(rw.dupId);
  if (!con || !dup) return [];
  const ul = portWorld(dup, 'l', w, h);
  const ur = portWorld(dup, 'r', w, h);
  const cl = portWorld(con, 'l', w, h);
  const cr = portWorld(con, 'r', w, h);
  const u0 = pushOut(rw.midX, rw.midY, ul.x, ul.y, w, h, 22);
  const v0 = pushOut(rw.midX, rw.midY, ur.x, ur.y, w, h, 22);
  const x0 = pushOut(rw.midX, rw.midY, cl.x, cl.y, w, h, 22);
  const y0 = pushOut(rw.midX, rw.midY, cr.x, cr.y, w, h, 22);
  return [
    { kind: 'con', x: u0.x, y: u0.y, heading: headingTo(u0.x, u0.y, leftoverPair(rw, 'dup', 'l'), agents, w, h, outward(u0.x, u0.y)), alpha: 1, scale: 1 },
    { kind: 'con', x: v0.x, y: v0.y, heading: headingTo(v0.x, v0.y, leftoverPair(rw, 'dup', 'r'), agents, w, h, outward(v0.x, v0.y)), alpha: 1, scale: 1 },
    { kind: 'dup', x: x0.x, y: x0.y, heading: headingTo(x0.x, x0.y, leftoverPair(rw, 'con', 'l'), agents, w, h, outward(x0.x, x0.y)), alpha: 1, scale: 1 },
    { kind: 'dup', x: y0.x, y: y0.y, heading: headingTo(y0.x, y0.y, leftoverPair(rw, 'con', 'r'), agents, w, h, outward(y0.x, y0.y)), alpha: 1, scale: 1 },
  ];
}

export function commitRewrite(
  rw: Rewrite,
  agents: Map<number, Agent>,
  graph: Graph,
  params: Params,
  nextId: number,
  time: number,
  w: number,
  h: number,
): number {
  const result = applyRewrite(snapshotOf(agents, graph), rw.rule, rw.a, rw.b, nextId);
  const poses = spawnPoses(rw);
  graph.detachAgent(rw.a);
  graph.detachAgent(rw.b);
  agents.delete(rw.a);
  agents.delete(rw.b);
  for (const s of result.spawned) {
    const pose = poses[s.role];
    const ag = createAgent(s.id, s.kind, pose.x, pose.y, pose.heading, params);
    ag.vx = 0;
    ag.vy = 0;
    ag.stun = 0.45;
    agents.set(s.id, ag);
  }
  for (const wire of result.net.wires) {
    if (!agents.has(wire.a.id) || !agents.has(wire.b.id)) continue;
    if (graph.isFree(wire.a) && graph.isFree(wire.b)) {
      graph.connect(agents, wire.a, wire.b, w, h, params, time);
    }
  }
  return result.nextId;
}

function spawnPoses(rw: Rewrite): Record<Spawned['role'], { x: number; y: number; heading: number }> {
  const dummy = {
    'era-l': { x: rw.midX, y: rw.midY, heading: 0 },
    'era-r': { x: rw.midX, y: rw.midY, heading: 0 },
    'con-u': { x: rw.midX, y: rw.midY, heading: 0 },
    'con-v': { x: rw.midX, y: rw.midY, heading: 0 },
    'dup-x': { x: rw.midX, y: rw.midY, heading: 0 },
    'dup-y': { x: rw.midX, y: rw.midY, heading: 0 },
  };
  const source = rw.ghosts.length ? rw.ghosts : rw.targets;
  if (source.length === 2 && rw.rule === 'erase') {
    dummy['era-l'] = { x: source[0].x, y: source[0].y, heading: source[0].heading };
    dummy['era-r'] = { x: source[1].x, y: source[1].y, heading: source[1].heading };
    return dummy;
  }
  if (source.length === 4 && rw.rule === 'commute') {
    dummy['con-u'] = { x: source[0].x, y: source[0].y, heading: source[0].heading };
    dummy['con-v'] = { x: source[1].x, y: source[1].y, heading: source[1].heading };
    dummy['dup-x'] = { x: source[2].x, y: source[2].y, heading: source[2].heading };
    dummy['dup-y'] = { x: source[3].x, y: source[3].y, heading: source[3].heading };
  }
  return dummy;
}
