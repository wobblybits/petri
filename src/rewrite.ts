import { CHEM_LEN, CRITIC_LEN, EMIT, E_OUT, PLASTIC_BASE, PLASTIC_LEN, refreshReadsField, STATE_DIMS, TASTE, T_OUT, createAgent, portWorld, slotsFor, stemFromPose, stemWorld, type Agent, type AgentKind, type PortRef, type PortSlot } from './agents.ts';
import type { AgentStore } from './agent-store.ts';
import { DEBT_CAP_MAX, EXTRA_CAP } from './energy.ts';
import { otherEnd, type Graph } from './graph.ts';
import type { Params } from './params.ts';
import {
  angleDelta,
  clamp,
  easeInOut,
  lerp,
  rotate,
  wrap,
  wrapDeltaVec,
  wrapDeltaVecInto,
  wrapMid,
} from './wrap.ts';

export type Rule = 'era-era' | 'erase' | 'annihilate-con' | 'annihilate-dup' | 'commute';

export interface NetWire {
  a: PortRef;
  b: PortRef;
  /** Live graph id, when this snapshot came from a Graph. */
  id?: number;
  /**
   * Graph ids that fused into this reconnection. Commit rebinds the keeper
   * instead of minting a new wire, so leftover delay lines keep ringing.
   */
  sources?: number[];
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
  /** The principal wire being consumed. Needed at commit to release it. */
  wireId: number;
  /** Drift the pair shared when it began, so a collapse is not a dead stop. */
  vx: number;
  vy: number;
  /** Set once the bodies have met, so the contact only sounds on the onset. */
  struck: boolean;
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

/** Ghosts start spreading from the mid at this rewrite time. */
export const GHOST_APPEAR_START = 0.35;

export function rewriteAppear(t: number): number {
  return easeInOut(clamp((t - GHOST_APPEAR_START) / (1 - GHOST_APPEAR_START), 0, 1));
}

export const COMMUTE_ROLES = ['con-u', 'con-v', 'dup-x', 'dup-y'] as const;
export type CommuteRole = (typeof COMMUTE_ROLES)[number];

/** Lafont square among commute children. Drawn as ghost wires during appear. */
export const COMMUTE_K22: { a: CommuteRole; aSlot: PortSlot; b: CommuteRole; bSlot: PortSlot }[] = [
  { a: 'con-u', aSlot: 'l', b: 'dup-x', bSlot: 'l' },
  { a: 'con-u', aSlot: 'r', b: 'dup-y', bSlot: 'l' },
  { a: 'con-v', aSlot: 'l', b: 'dup-x', bSlot: 'r' },
  { a: 'con-v', aSlot: 'r', b: 'dup-y', bSlot: 'r' },
];

export function commuteGhost(rw: Rewrite, role: CommuteRole): Ghost | null {
  const src = rw.ghosts.length ? rw.ghosts : rw.targets;
  if (src.length !== 4) return null;
  return src[COMMUTE_ROLES.indexOf(role)] ?? null;
}

export function stemFromGhost(g: Ghost, slot: PortSlot, w: number, h: number): { x: number; y: number } {
  return stemFromPose(g.kind, g.x, g.y, g.heading, g.scale, slot, w, h);
}

/** The bodies have met by here; the wire is spent. */
export const PULL_END = 0.55;
/** Nothing shrinks or fades before this — the collapse is its own beat. */
export const COLLAPSE_START = 0.65;
/** Minimum half-extents of the Dup–Con frame commute children sit on, so the K₂,₂ chords do not stack. */
export const COMMUTE_ALONG_MIN = 48;
export const COMMUTE_ACROSS_MIN = 40;

/** Accelerating ease: a wire releasing tension arrives faster than it set off. */
function easeIn(t: number): number {
  return t * t;
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
 * Union-find over ports, because a port can lead straight back into the dying
 * pair and on again (`λx.x` is a Con with its own aux ports wired together).
 * Fuse every wire touching a dying agent and every identification the rule
 * makes, then emit one wire per class that still has two surviving ends. A
 * class with none was a closed loop and disappears; a class with one had a
 * free end and stays free.
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
  const sources = new Map<string, number[]>();
  for (const w of net.wires) {
    if (w.id === undefined) continue;
    if (!dying.has(w.a.id) && !dying.has(w.b.id)) continue;
    const r = find(key(w.a));
    const list = sources.get(r);
    if (list) list.push(w.id);
    else sources.set(r, [w.id]);
  }

  const out: NetWire[] = [];
  for (const [r, ends] of classes) {
    if (ends.length !== 2) continue;
    const src = sources.get(r);
    if (src && src.length) out.push({ a: ends[0], b: ends[1], sources: src });
    else out.push({ a: ends[0], b: ends[1] });
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
    // Crossed: l to the other's r. Beta reduction when the pair is an
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
    wires: [...graph.wires.values()].map((w) => ({ a: w.a, b: w.b, id: w.id })),
  };
}

/**
 * The part of the net a rewrite of `a` against `b` can see: the pair
 * themselves, and every wire touching either of them. `applyRewrite` reads
 * `net.agents` only for the kinds of `a` and `b`, and `fuse` and
 * `stripAgents` drop every wire not incident to a dying agent, so nothing
 * else is needed. `snapshotOf` stays for `lambda.ts` and the tests.
 */
export function localSnapshotOf(
  agents: Map<number, Agent>,
  graph: Graph,
  a: number,
  b: number,
): NetSnapshot {
  const outAgents: NetAgent[] = [];
  const outWires: NetWire[] = [];
  // The pair share a wire; without this it is in the list twice.
  const seen = new Set<number>();
  for (let i = 0; i < 2; i++) {
    const id = i === 0 ? a : b;
    const ag = agents.get(id);
    if (!ag) continue;
    outAgents.push({ id: ag.id, kind: ag.kind });
    for (const slot of slotsFor(ag.kind)) {
      const wire = graph.wireAtSlot(id, slot);
      if (!wire || seen.has(wire.id)) continue;
      seen.add(wire.id);
      outWires.push({ a: wire.a, b: wire.b, id: wire.id });
    }
  }
  // Ascending id: `fuse` unions in list order, and the order decides the
  // union-find roots and through them which end of a fused wire becomes `a`.
  outWires.sort((x, y) => (x.id ?? 0) - (y.id ?? 0));
  return { agents: outAgents, wires: outWires };
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
  wireId = -1,
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
    wireId,
    // The pair keeps the drift the two bodies had in common; only the closing
    // half of their motion is the rewrite's business.
    vx: (agentA.vx + agentB.vx) * 0.5,
    vy: (agentA.vy + agentB.vy) * 0.5,
    struck: false,
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
  rw.targets = snapshotTargets(rw, agents, graph, w, h);
  return rw;
}

/** Scratch for the deltas below; read and finished with inside one call. */
const advToB = { x: 0, y: 0 };
const advToA = { x: 0, y: 0 };
const advSpread = { x: 0, y: 0 };

/**
 * Grow the ghost trail off `rw.targets`, reusing the objects already there.
 * Callers test `rw.ghosts.length` to decide between ghosts and targets, so
 * the length has to track the target count exactly.
 */
function spreadGhosts(
  rw: Rewrite,
  w: number,
  h: number,
  appear: number,
  minScale: number,
): void {
  const targets = rw.targets;
  const ghosts = rw.ghosts;
  if (ghosts.length !== targets.length) ghosts.length = targets.length;
  for (let i = 0; i < targets.length; i++) {
    const g = targets[i];
    const d = wrapDeltaVecInto(rw.midX, rw.midY, g.x, g.y, w, h, advSpread);
    let out = ghosts[i];
    if (out === undefined) {
      out = { kind: g.kind, x: 0, y: 0, heading: 0, alpha: 0, scale: 0 };
      ghosts[i] = out;
    }
    out.kind = g.kind;
    out.x = wrap(rw.midX + d.x * appear, w);
    out.y = wrap(rw.midY + d.y * appear, h);
    out.heading = g.heading;
    out.alpha = appear;
    out.scale = lerp(minScale, 1, appear);
  }
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

  const toB = wrapDeltaVecInto(rw.ax, rw.ay, rw.bx, rw.by, w, h, advToB);
  const toA = wrapDeltaVecInto(rw.bx, rw.by, rw.ax, rw.ay, w, h, advToA);

  if (rw.rule === 'era-era' || rw.rule === 'annihilate-con' || rw.rule === 'annihilate-dup') {
    // Three beats: the wire hauls them together, they touch, and only then
    // does the pair collapse.
    const pull = easeIn(clamp(t / PULL_END, 0, 1));
    const collapse = easeInOut(clamp((t - COLLAPSE_START) / (1 - COLLAPSE_START), 0, 1));
    const dx = rw.vx * rw.t * rw.duration;
    const dy = rw.vy * rw.t * rw.duration;
    A.x = wrap(rw.ax + dx + toB.x * pull * 0.5, w);
    A.y = wrap(rw.ay + dy + toB.y * pull * 0.5, h);
    B.x = wrap(rw.bx + dx + toA.x * pull * 0.5, w);
    B.y = wrap(rw.by + dy + toA.y * pull * 0.5, h);
    A.scale = B.scale = lerp(1, 0.1, collapse);
    A.alpha = B.alpha = 1 - collapse;
    A.heading = rw.ah + angleDelta(rw.ah, Math.atan2(toB.y, toB.x)) * pull;
    B.heading = rw.bh + angleDelta(rw.bh, Math.atan2(toA.y, toA.x)) * pull;
    // Callers read the length to mean "no ghosts".
    rw.ghosts.length = 0;
  } else if (rw.rule === 'erase') {
    const era = A.kind === 'era' ? A : B;
    const bin = era === A ? B : A;
    const esx = era === A ? rw.ax : rw.bx;
    const esy = era === A ? rw.ay : rw.by;
    const bsx = era === A ? rw.bx : rw.ax;
    const bsy = era === A ? rw.by : rw.ay;
    era.x = wrap(esx + (bsx - esx) * e, w);
    era.y = wrap(esy + (bsy - esy) * e, h);
    era.scale = lerp(1, 0.4, e);
    era.alpha = 1 - e;
    bin.scale = lerp(1, 0.2, e);
    bin.alpha = 1 - e * 0.85;
    spreadGhosts(rw, w, h, rewriteAppear(t), 0.3);
  } else {
    A.x = wrap(rw.ax + toB.x * e, w);
    A.y = wrap(rw.ay + toB.y * e, h);
    B.x = wrap(rw.bx + toA.x * e, w);
    B.y = wrap(rw.by + toA.y * e, h);
    A.alpha = B.alpha = 1 - e;
    A.scale = B.scale = lerp(1, 0.35, e);
    spreadGhosts(rw, w, h, rewriteAppear(t), 0.25);
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

function roleForAux(rw: Rewrite, agentId: number, slot: PortRef['slot']): CommuteRole | null {
  if (slot !== 'l' && slot !== 'r') return null;
  if (agentId === rw.dupId) return slot === 'l' ? 'con-u' : 'con-v';
  if (agentId === rw.conId) return slot === 'l' ? 'dup-x' : 'dup-y';
  return null;
}

function annihilateMateLeftover(rw: Rewrite, agentId: number, slot: PortRef['slot']): PortRef | null {
  if (slot !== 'l' && slot !== 'r') return null;
  const fromA = agentId === rw.a;
  if (rw.rule === 'annihilate-dup') {
    if (fromA) return slot === 'l' ? rw.leftoverBL : rw.leftoverBR;
    return slot === 'l' ? rw.leftoverAL : rw.leftoverAR;
  }
  if (rw.rule === 'annihilate-con') {
    if (fromA) return slot === 'l' ? rw.leftoverBR : rw.leftoverBL;
    return slot === 'l' ? rw.leftoverAR : rw.leftoverAL;
  }
  return null;
}

/**
 * Where a leftover or fused rope should be drawn and reeled during a rewrite:
 * dying stems lerp onto the inheriting ghost (commute) or the identified
 * leftover mate (annihilate). Null means use the live agent stems.
 */
export function rewriteHandoffStems(
  rw: Rewrite,
  wire: { id: number; a: PortRef; b: PortRef },
  agents: Map<number, Agent>,
  w: number,
  h: number,
): { ax: number; ay: number; bx: number; by: number } | null {
  if (wire.id === rw.wireId) return null;
  const dying = (id: number) => id === rw.a || id === rw.b;
  if (!dying(wire.a.id) && !dying(wire.b.id)) return null;
  const ae = rewriteAppear(rw.t);
  const liveStem = (port: PortRef) => {
    const ag = agents.get(port.id);
    if (!ag) return { x: 0, y: 0 };
    return stemWorld(ag, port.slot, w, h);
  };
  const resolve = (port: PortRef): { x: number; y: number } => {
    const live = liveStem(port);
    if (!dying(port.id)) return live;
    if (rw.rule === 'commute') {
      const role = roleForAux(rw, port.id, port.slot);
      const g = role ? commuteGhost(rw, role) : null;
      if (!g) return live;
      const gs = stemFromGhost(g, 'p', w, h);
      return { x: lerp(live.x, gs.x, ae), y: lerp(live.y, gs.y, ae) };
    }
    if (rw.rule === 'annihilate-con' || rw.rule === 'annihilate-dup') {
      const mate = annihilateMateLeftover(rw, port.id, port.slot);
      if (!mate || dying(mate.id)) return live;
      const ms = liveStem(mate);
      return { x: lerp(live.x, ms.x, ae), y: lerp(live.y, ms.y, ae) };
    }
    return live;
  };
  const A = resolve(wire.a);
  const B = resolve(wire.b);
  return { ax: A.x, ay: A.y, bx: B.x, by: B.y };
}

/** Aux wired into the other dying agent becomes a principal among the children. */
function fusedPartnerRole(
  graph: Graph,
  rw: Rewrite,
  parentId: number,
  slot: 'l' | 'r',
): CommuteRole | null {
  const w = graph.wireAt({ id: parentId, slot });
  if (!w) return null;
  const o = otherEnd(w, { id: parentId, slot });
  if (o.id !== rw.a && o.id !== rw.b) return null;
  return roleForAux(rw, o.id, o.slot);
}

function acrossSign(agent: Agent, slot: 'l' | 'r', vx: number, vy: number): number {
  const localY = slot === 'l' ? -1 : 1;
  const world = rotate(0, localY, agent.heading);
  const s = world.x * vx + world.y * vy;
  if (Math.abs(s) < 1e-6) return localY;
  return s > 0 ? 1 : -1;
}

function snapshotTargets(
  rw: Rewrite,
  agents: Map<number, Agent>,
  graph: Graph,
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

  const axis = wrapDeltaVec(dup.x, dup.y, con.x, con.y, w, h);
  let ux = axis.x;
  let uy = axis.y;
  const axisLen = Math.hypot(ux, uy);
  if (axisLen < 1e-4) {
    ux = Math.cos(dup.heading);
    uy = Math.sin(dup.heading);
  } else {
    ux /= axisLen;
    uy /= axisLen;
  }
  const vx = -uy;
  const vy = ux;

  const alongOf = (x: number, y: number) => {
    const d = wrapDeltaVec(rw.midX, rw.midY, x, y, w, h);
    return d.x * ux + d.y * uy;
  };
  const acrossOf = (x: number, y: number) => {
    const d = wrapDeltaVec(rw.midX, rw.midY, x, y, w, h);
    return d.x * vx + d.y * vy;
  };
  const ul = portWorld(dup, 'l', w, h);
  const ur = portWorld(dup, 'r', w, h);
  const cl = portWorld(con, 'l', w, h);
  const cr = portWorld(con, 'r', w, h);
  // Span the ports along the axis so leftover eras stay outside the quad;
  // floor the across width so the frame is not a pancake.
  const alongHalf = Math.max(
    COMMUTE_ALONG_MIN,
    Math.abs(alongOf(ul.x, ul.y)),
    Math.abs(alongOf(ur.x, ur.y)),
    Math.abs(alongOf(cl.x, cl.y)),
    Math.abs(alongOf(cr.x, cr.y)),
  );
  const acrossHalf = Math.max(
    COMMUTE_ACROSS_MIN,
    Math.abs(acrossOf(ul.x, ul.y)),
    Math.abs(acrossOf(ur.x, ur.y)),
    Math.abs(acrossOf(cl.x, cl.y)),
    Math.abs(acrossOf(cr.x, cr.y)),
  );

  const at = (sAlong: number, sAcross: number) => ({
    x: wrap(rw.midX + ux * sAlong * alongHalf + vx * sAcross * acrossHalf, w),
    y: wrap(rw.midY + uy * sAlong * alongHalf + vy * sAcross * acrossHalf, h),
  });
  const uAcross = acrossSign(dup, 'l', vx, vy);
  const vAcross = acrossSign(dup, 'r', vx, vy);
  const xAcross = acrossSign(con, 'l', vx, vy);
  const yAcross = acrossSign(con, 'r', vx, vy);
  const poses: Record<CommuteRole, { x: number; y: number }> = {
    'con-u': at(-1, uAcross),
    'con-v': at(-1, vAcross),
    'dup-x': at(1, xAcross),
    'dup-y': at(1, yAcross),
  };
  const leftovers: Record<CommuteRole, PortRef | null> = {
    'con-u': leftoverPair(rw, 'dup', 'l'),
    'con-v': leftoverPair(rw, 'dup', 'r'),
    'dup-x': leftoverPair(rw, 'con', 'l'),
    'dup-y': leftoverPair(rw, 'con', 'r'),
  };
  const fused: Record<CommuteRole, CommuteRole | null> = {
    'con-u': fusedPartnerRole(graph, rw, rw.dupId, 'l'),
    'con-v': fusedPartnerRole(graph, rw, rw.dupId, 'r'),
    'dup-x': fusedPartnerRole(graph, rw, rw.conId, 'l'),
    'dup-y': fusedPartnerRole(graph, rw, rw.conId, 'r'),
  };
  const headingFor = (role: CommuteRole, sAlong: number, sAcross: number) => {
    const p = poses[role];
    const outAlong = Math.atan2(uy * sAlong, ux * sAlong);
    const outAcross = Math.atan2(vy * sAcross, vx * sAcross);
    const lo = leftovers[role];
    if (lo) return headingTo(p.x, p.y, lo, agents, w, h, outAlong);
    // Fused principals face out of the rectangle along v̂ so aux ports sit
    // on the interior.
    if (fused[role]) return outAcross;
    return outAlong;
  };
  return [
    {
      kind: 'con',
      x: poses['con-u'].x,
      y: poses['con-u'].y,
      heading: headingFor('con-u', -1, uAcross),
      alpha: 1,
      scale: 1,
    },
    {
      kind: 'con',
      x: poses['con-v'].x,
      y: poses['con-v'].y,
      heading: headingFor('con-v', -1, vAcross),
      alpha: 1,
      scale: 1,
    },
    {
      kind: 'dup',
      x: poses['dup-x'].x,
      y: poses['dup-x'].y,
      heading: headingFor('dup-x', 1, xAcross),
      alpha: 1,
      scale: 1,
    },
    {
      kind: 'dup',
      x: poses['dup-y'].x,
      y: poses['dup-y'].y,
      heading: headingFor('dup-y', 1, yAcross),
      alpha: 1,
      scale: 1,
    },
  ];
}

/**
 * The heritable scalars a Con+Dup commute recombines into its children:
 * physiology (tank depth, debt, rescue, demand conduction), properties of a
 * body rather than decisions it makes. Behaviour is inherited through the
 * genome's output heads instead. Serves "inheritance" in `docs/concepts.md`.
 */
export const TRAIT_KEYS = [
  'adenylate',
  'requestDecay',
  'energyCap',
  'debtCap',
  'rescueTo',
  'assort',
] as const;
export type TraitKey = (typeof TRAIT_KEYS)[number];

/**
 * How far one generation can nudge a single chemistry weight. Scaled to the
 * genome's size so that how far a child lands from its parents overall (the
 * root-sum-square of the per-gene steps) stays constant when the layout
 * changes; 0.06 at sixteen genes is the reference.
 */
export const CHEM_MUTATE = 0.06 * Math.sqrt(16 / CHEM_LEN);
/** Ceiling on a taste weight, positive or negative. */
export const CHEM_TASTE_MAX = 4;
/** Ceiling on an emit slope: one unit of voice is the base's whole budget. */
export const CHEM_SLOPE_MAX = 1;

/**
 * Bounds a bred trait is clamped to, and how far one generation's mutation
 * can nudge it. Matches the corresponding slider's range in params.ts, except
 * `energyCap`, which is a multiple of `EXTRA_CAP`, the tank a fresh Con or
 * Dup starts at.
 */
export const TRAIT_RANGE: Record<TraitKey, { min: number; max: number; mutate: number }> = {
  /* Working capital: how much work a body can have outstanding before it waits on its own regeneration. */
  adenylate: { min: 0.2, max: 6, mutate: 0.15 },
  requestDecay: { min: 0.5, max: 0.98, mutate: 0.03 },
  energyCap: { min: EXTRA_CAP * 0.5, max: EXTRA_CAP * 2, mutate: EXTRA_CAP * 0.1 },
  /* Strictly negative: at zero the rescue latch never fires. The clamp is the sign lock. */
  debtCap: { min: -2.5, max: DEBT_CAP_MAX, mutate: 0.12 },
  rescueTo: { min: 0, max: 1, mutate: 0.06 },
  /*
   * 0 is a lineage that blends every gene of every child, 1 one that copies
   * each from a single parent. Seeded in the middle; see `assortChance`.
   */
  assort: { min: 0, max: 1, mutate: 0.05 },
};

/**
 * How far outside the interval between two parents a blended child may land.
 *
 * A plain `lerp(a, b, random())` halves the variance of the population every
 * time it runs, and variance is what selection grips. At 0.5 the child is
 * drawn from `[a - 0.5(b-a), b + 0.5(b-a)]` (BLX-alpha), which holds
 * offspring variance roughly equal to parental variance. Every consumer
 * clamps afterwards (`TRAIT_RANGE`, the matrix bounds), so the widened draw
 * cannot put a value out of range.
 */
export const BLEND_WIDEN = 0.5;

/** A blend that does not collapse the population's spread. See `BLEND_WIDEN`. */
function blend(a: number, b: number): number {
  return lerp(a, b, -BLEND_WIDEN + Math.random() * (1 + 2 * BLEND_WIDEN));
}

function nudgeTrait(value: number, range: { min: number; max: number; mutate: number }): number {
  const mutated = value + (Math.random() * 2 - 1) * range.mutate;
  return Math.min(range.max, Math.max(range.min, mutated));
}

/**
 * How likely a single gene is copied whole from one parent, rather than
 * blended between the two: the parents' mean `assort`, offset by half a unit
 * up for a Dup child and down for a Con child, clamped. At the seeded 0.5 a
 * Con child blends everything and a Dup child copies everything; the flip
 * or the weight is drawn per gene, not once per child.
 */
function assortChance(child: Agent, conParent: Agent, dupParent: Agent): number {
  const mean = (conParent.assort + dupParent.assort) * 0.5;
  const p = child.kind === 'dup' ? mean + 0.5 : mean - 0.5;
  return p <= 0 ? 0 : p >= 1 ? 1 : p;
}

function inheritTraits(child: Agent, conParent: Agent, dupParent: Agent, kappa: number): void {
  const chance = assortChance(child, conParent, dupParent);
  // Depth is one past the deeper parent. Lineage follows the Con: arbitrary,
  // but deterministic, so lineage counts are a measurement.
  child.born = Math.max(conParent.born, dupParent.born) + 1;
  child.lineage = conParent.lineage;
  for (const key of TRAIT_KEYS) {
    const range = TRAIT_RANGE[key];
    const combined =
      Math.random() < chance
        ? Math.random() < 0.5
          ? conParent[key]
          : dupParent[key]
        : blend(conParent[key], dupParent[key]);
    child[key] = nudgeTrait(combined, range);
  }
  inheritChem(child, conParent, dupParent, chance, kappa);
}

/** Copy one parent onto the child, then the same mutation nudge commute uses. */
function inheritFromClone(child: Agent, parent: Agent, kappa: number): void {
  child.born = parent.born + 1;
  child.lineage = parent.lineage;
  for (const key of TRAIT_KEYS) {
    child[key] = nudgeTrait(parent[key], TRAIT_RANGE[key]);
  }
  // One parent, so blending and assorting are the same thing.
  inheritChem(child, parent, parent, 1, kappa);
}

/**
 * The genome recombines the same way the scalar traits do, per gene. Emit is
 * clamped non-negative and to a unit sum: a body has one voice to spend
 * across four channels, otherwise "louder" is strictly better and every
 * lineage converges on it. Taste is signed; a negative weight is avoidance.
 */
function inheritChem(child: Agent, con: Agent, dup: Agent, chance: number, kappa: number): void {
  const c = child.chem;
  /*
   * A parent hands on what it was born with plus what it learned, scaled by
   * `kappa` (`params.inheritLearned`): the child starts with an empty
   * plastic slate and the parents' experience arrives written into its
   * genome. At 0 learning is somatic and dies with the body. Consumes the
   * same random numbers in the same order regardless of `kappa`, which is
   * what keeps a pond with learning off hashing as it does.
   */
  const conP = con.store.plasticAll;
  const dupP = dup.store.plasticAll;
  const conO = con.slot * PLASTIC_LEN;
  const dupO = dup.slot * PLASTIC_LEN;
  const learnedHi = PLASTIC_BASE + PLASTIC_LEN;
  for (let k = 0; k < CHEM_LEN; k++) {
    let a = con.chem[k];
    let b = dup.chem[k];
    if (kappa !== 0 && k >= PLASTIC_BASE && k < learnedHi) {
      a += kappa * conP[conO + k - PLASTIC_BASE];
      b += kappa * dupP[dupO + k - PLASTIC_BASE];
    }
    const combined = Math.random() < chance ? (Math.random() < 0.5 ? a : b) : blend(a, b);
    c[k] = combined + (Math.random() * 2 - 1) * CHEM_MUTATE;
  }
  // The critic is averaged, not assorted: a prediction about the
  // neighbourhood the child is born into, and averaging consumes no random numbers.
  const childC = child.store.criticAll;
  const conC = con.store.criticAll;
  const dupC = dup.store.criticAll;
  const childO = child.slot * CRITIC_LEN;
  const conCo = con.slot * CRITIC_LEN;
  const dupCo = dup.slot * CRITIC_LEN;
  for (let k = 0; k < CRITIC_LEN; k++) {
    childC[childO + k] = kappa * 0.5 * (conC[conCo + k] + dupC[dupCo + k]);
  }
  /*
   * One unit of voice across all four channels, the ground included. Nothing
   * reads the ground's slot (farming is the expression head), but dropping it
   * would renormalise every genome in the library; see `effEmit`.
   */
  let sum = 0;
  for (let k0 = 0; k0 < 4; k0++) {
    const k = EMIT + k0;
    if (c[k] < 0) c[k] = 0;
    sum += c[k];
  }
  // A body with nothing left to say is mute, not amplified from noise.
  if (sum > 1e-6) for (let k0 = 0; k0 < 4; k0++) c[EMIT + k0] /= sum;
  for (let k = TASTE; k < TASTE + 4; k++) {
    c[k] = Math.min(CHEM_TASTE_MAX, Math.max(-CHEM_TASTE_MAX, c[k]));
  }
  /*
   * Slopes and matrices are bounded but signed. Every weight must be bounded:
   * `phi` keeps `h` inside (-1, 1), so a bounded weight against a bounded
   * activation is a bounded product; an unclamped matrix lets one runaway
   * entry produce a number the field cannot hold. `E` gets the tighter
   * bound because one unit of voice is a body's whole budget.
   */
  for (let k = E_OUT; k < E_OUT + 4 * STATE_DIMS; k++) {
    c[k] = Math.min(CHEM_SLOPE_MAX, Math.max(-CHEM_SLOPE_MAX, c[k]));
  }
  /*
   * Everything from `T` to the end of the genome is dimensionless
   * (`HEAD_SCALE` turns a head gene into a force), so one bound covers it.
   */
  for (let k = T_OUT; k < CHEM_LEN; k++) {
    c[k] = Math.min(CHEM_TASTE_MAX, Math.max(-CHEM_TASTE_MAX, c[k]));
  }
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
  store: AgentStore,
  breed = true,
): number {
  // Local, not the whole pond: only `result.spawned` and the fused joins in
  // `result.net.wires` are read.
  const result = applyRewrite(localSnapshotOf(agents, graph, rw.a, rw.b), rw.rule, rw.a, rw.b, nextId);
  const poses = spawnPoses(rw);
  // Read before the parents are deleted below.
  const conParent = rw.rule === 'commute' ? agents.get(rw.conId) : undefined;
  const dupParent = rw.rule === 'commute' ? agents.get(rw.dupId) : undefined;
  const eraParent = rw.rule === 'erase' ? agents.get(rw.eraId) : undefined;
  for (const s of result.spawned) {
    const pose = poses[s.role];
    const ag = createAgent(s.id, s.kind, pose.x, pose.y, pose.heading, params, store);
    ag.vx = 0;
    ag.vy = 0;
    ag.stun = 0.45;
    if (breed) {
      if (conParent && dupParent) inheritTraits(ag, conParent, dupParent, params.inheritLearned);
      else if (eraParent) inheritFromClone(ag, eraParent, params.inheritLearned);
      // Inheritance rewrote `chem`, so the cached sense gate is stale.
      refreshReadsField(ag);
    }
    agents.set(s.id, ag);
  }
  inheritLeftoverWires(graph, result.net.wires, agents, w, h, params, time);
  graph.detachAgent(rw.a);
  graph.detachAgent(rw.b);
  agents.delete(rw.a);
  agents.delete(rw.b);
  store.release(rw.a);
  store.release(rw.b);
  for (const wire of result.net.wires) {
    if (!agents.has(wire.a.id) || !agents.has(wire.b.id)) continue;
    if (graph.isFree(wire.a) && graph.isFree(wire.b)) {
      graph.connect(agents, wire.a, wire.b, w, h, params, time, { chord: true, silent: true });
    }
  }
  return result.nextId;
}

/**
 * Rebind the surviving graph wire onto the fused ports so its id, rest, and
 * rope stay. The principal that was consumed is not in `desired` and still
 * detaches.
 */
function inheritLeftoverWires(
  graph: Graph,
  desired: NetWire[],
  agents: Map<number, Agent>,
  w: number,
  h: number,
  params: Params,
  time: number,
): void {
  const claimed = new Set<number>();
  for (const spec of desired) {
    const src = spec.sources;
    if (!src || src.length === 0) continue;
    let keeper = -1;
    for (const id of src) {
      if (!graph.wires.has(id) || claimed.has(id)) continue;
      if (keeper < 0 || id < keeper) keeper = id;
    }
    if (keeper < 0) continue;
    if (!graph.rebind(keeper, spec.a, spec.b)) continue;
    graph.restitchChord(keeper, agents, w, h, params, time);
    claimed.add(keeper);
    for (const id of src) {
      if (id !== keeper) graph.detach(id);
    }
  }
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
