import { extraCapFor } from './energy.ts';
import type { Params } from './params.ts';
import { rotate, wrap, wrapAngle, wrapDeltaVec, angleDelta, type Vec2 } from './wrap.ts';

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
  /** Ornstein-Uhlenbeck self-propulsion magnitude along the heading. */
  drive: number;
  /** Trail sampled for steering. */
  trail: number;
  /** Pose at the start of the current integrate step (for XPBD velocity writeback). */
  prevX: number;
  prevY: number;
  prevHeading: number;
  integVx: number;
  integVy: number;
  integOmega: number;
  /**
   * Energy on top of existence, in [−1, 1]. Positive is stock it can spend or
   * pass on, negative is debt it must settle before it can do either, and −1
   * is death. Upkeep decrements this directly.
   */
  extra: number;
  /** Request gradient toward a hungry redex. 0 = quiet. */
  request: number;
  /**
   * Set when the body falls into debt, cleared when it is back on its feet.
   *
   * Hunger measured against break-even stops the moment the debt is settled,
   * which left a rescued body pinned at exactly 0 — alive, one frame of upkeep
   * from dying again, and permanently unable to afford the share a rewrite
   * costs. The latch is what lets the ask outlive the debt: while it is set,
   * the body keeps asking up to `rescueTo`, so a rescue tops it back up to
   * something it can act with instead of parking it on the line.
   */
  recovering: boolean;
  /**
   * Heritable traits. Seeded from the matching global slider when a body is
   * created outside a rewrite, so a fresh soup starts homogeneous just as it
   * did before these existed. A Con+Dup commute instead blends both parents'
   * values into each child (see `inheritTraits` in rewrite.ts), which is the
   * only place a population's traits can actually drift.
   */
  /** How much of this body's own demand survives one more hop outward. */
  requestDecay: number;
  /** The most this body can hold, in place of the flat per-kind cap. */
  energyCap: number;
  /** How much of a kick this body's own pumps hand off instead of keeping. */
  transportThrust: number;
  /** How hard this body recoils, per unit of energy it pumps to a neighbour. */
  transportRecoil: number;
  /**
   * Memoized cosine and sine of `heading`, with the heading they were taken
   * at. Every port position in the sim goes through `stemOffsetInto`, which
   * needs both; at pond scale that was ~60,000 sin and 60,000 cos a frame in
   * the wall-mask pass alone, and as much again in the length refresh and the
   * rope shape pass — all of them recomputing the same handful of headings,
   * because the work is indexed per wire-endpoint and the heading is per body.
   *
   * Memoizing on the agent rather than in a frame-keyed side table means the
   * guard is exact and self-invalidating: heading moves, the memo misses. NaN
   * starts it cold and keeps it cold if a heading ever goes bad.
   */
  csHeading: number;
  csCos: number;
  csSin: number;
}

/** Slot as a small integer: principal 0, left 1, right 2. */
export function slotIndex(slot: PortSlot): number {
  return slot === 'p' ? 0 : slot === 'l' ? 1 : 2;
}

/**
 * A port's identity as a number.
 *
 * This was a template string, and it is looked up constantly — every free-port
 * test in steering, deposit, flocking and snapping goes through it, some sixty
 * thousand times a frame on a grown pond. That was sixty thousand strings a
 * frame built only to be hashed and thrown away.
 */
export function portKeyAt(id: number, slot: PortSlot): number {
  return id * 3 + slotIndex(slot);
}

export function portKey(p: PortRef): number {
  return portKeyAt(p.id, p.slot);
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

/** Conservative bound that encloses the glyph. Broad phase and LOD size. */
export function boundRadius(agent: Agent): number {
  if (agent.kind === 'era') return (ERA_RADIUS + 1.2) * agent.scale;
  return agentSize(agent.kind) * 1.12 * agent.scale;
}

/**
 * Radius of the disc with the same area as the Con/Dup triangle, as a
 * multiple of `agentSize`. The glyph is 1.312 s^2 for s = 16 * scale, so the
 * equal-area radius is s * sqrt(1.312 / PI).
 */
export const TRI_DISC_RATIO = Math.sqrt(1.312 / Math.PI);

/**
 * Contact radius for the tiers that collide discs instead of SAT polygons.
 *
 * `boundRadius` is the circumscribed bound, which for a triangle is ~1.7x too
 * fat to use as a contact radius: a net that settles at ~22 px under SAT is
 * held ~36 px apart by bound discs, so it visibly inflates the moment the
 * camera crosses the LOD line. Equal area is the closest single radius to
 * where SAT actually settles, which is what keeps the tiers agreeing.
 */
export function discRadius(agent: Agent): number {
  if (agent.kind === 'era') return ERA_RADIUS * agent.scale;
  return agentSize(agent.kind) * TRI_DISC_RATIO * agent.scale;
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
  return stemRootInto(kind, slot, { x: 0, y: 0 });
}

/** `stemRoot` without the allocation. The single source of the geometry. */
export function stemRootInto(kind: AgentKind, slot: PortSlot, out: Vec2): Vec2 {
  if (kind === 'era') {
    out.x = slot === 'p' ? 8 : 0;
    out.y = 0;
    return out;
  }
  const s = agentSize(kind);
  if (slot === 'p') {
    out.x = s * 1.05;
    out.y = 0;
    return out;
  }
  const halfBase = s * 0.82;
  const legY = halfBase * 0.7;
  out.x = -s * 0.55;
  out.y = slot === 'l' ? -legY : legY;
  return out;
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
    drive: params.stepSpeed,
    trail: 0,
    prevX: x,
    prevY: y,
    prevHeading: heading,
    integVx: 0,
    integVy: 0,
    integOmega: 0,
    csHeading: NaN,
    csCos: 1,
    csSin: 0,
    extra: 0,
    request: 0,
    recovering: false,
    requestDecay: params.requestDecay,
    energyCap: extraCapFor(kind),
    transportThrust: params.transportThrust,
    transportRecoil: params.transportRecoil,
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

/** Body heading that aims `slot` along the world angle `target`. */
export function headingFacingPort(agent: Agent, slot: PortSlot, target: number): number {
  const axis = portAxis(agent, slot);
  const portAng = Math.atan2(axis.y, axis.x);
  return wrapAngle(agent.heading + angleDelta(portAng, target));
}

/** Body heading so `slot` on `agent` points from `from` toward `toward`. */
export function headingAlongWire(
  agent: Agent,
  slot: PortSlot,
  from: Vec2,
  toward: Vec2,
  w: number,
  h: number,
): number {
  const d = wrapDeltaVec(from.x, from.y, toward.x, toward.y, w, h);
  return headingFacingPort(agent, slot, Math.atan2(d.y, d.x));
}

/**
 * Like `headingAlongWire`, but when two headings satisfy the port aim,
 * pick the one closest to `prefer` (keeps tow chains from flipping 180°).
 */
export function headingAlongTow(
  agent: Agent,
  slot: PortSlot,
  from: Vec2,
  toward: Vec2,
  prefer: number,
  w: number,
  h: number,
): number {
  const d = wrapDeltaVec(from.x, from.y, toward.x, toward.y, w, h);
  const axis = Math.atan2(d.y, d.x);
  const a = headingFacingPort(agent, slot, axis);
  const b = wrapAngle(a + Math.PI);
  return Math.abs(angleDelta(prefer, a)) <= Math.abs(angleDelta(prefer, b)) ? a : b;
}

/**
 * Center-to-center distance along the meridian when stems are `stemRest` apart
 * and the aux agent leads the wired principal.
 */
export function meridianCenterGap(
  lead: Agent,
  leadSlot: PortSlot,
  follow: Agent,
  followSlot: PortSlot,
  stemRest: number,
  w: number,
  h: number,
): number {
  const heading = lead.heading;
  const cos = Math.cos(heading);
  const sin = Math.sin(heading);
  const snap = {
    lx: lead.x,
    ly: lead.y,
    lh: lead.heading,
    fx: follow.x,
    fy: follow.y,
    fh: follow.heading,
  };
  follow.x = 0;
  follow.y = 0;
  follow.heading = heading;
  lead.heading = heading;
  let lo = 0;
  let hi = Math.max(48, stemRest + 80);
  for (let i = 0; i < 28; i++) {
    const g = (lo + hi) * 0.5;
    lead.x = cos * g;
    lead.y = sin * g;
    const sa = stemWorld(lead, leadSlot, w, h);
    const sb = stemWorld(follow, followSlot, w, h);
    const span = Math.hypot(sb.x - sa.x, sb.y - sa.y);
    if (span > stemRest) hi = g;
    else lo = g;
  }
  const gap = (lo + hi) * 0.5;
  lead.x = snap.lx;
  lead.y = snap.ly;
  lead.heading = snap.lh;
  follow.x = snap.fx;
  follow.y = snap.fy;
  follow.heading = snap.fh;
  return gap;
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
  return stemOffsetAt(agent.heading, agent, slot);
}


/**
 * `stemOffset` writing into `out`.
 *
 * The allocating form costs three objects a call — a stem root, a rotation,
 * and the result — and the FAR pack calls it twice per wire, which on a pond
 * of 14000 wires is most of a hundred thousand short-lived objects a frame.
 */
/**
 * Cosine and sine of the agent's heading, computed at most once per heading.
 *
 * Exact, not approximate: `Math.cos` is deterministic for a given input, so a
 * hit returns the identical bits the call would have. The guard is a float
 * compare against the heading the memo was taken at.
 */
export function poseSinCos(agent: Agent): void {
  if (agent.csHeading !== agent.heading) {
    agent.csHeading = agent.heading;
    agent.csCos = Math.cos(agent.heading);
    agent.csSin = Math.sin(agent.heading);
  }
}

/*
 * Flattened on purpose. This is the single hottest geometric routine in the
 * sim — every port position in every pass comes through it, twice per wire —
 * and it used to reach `stemRootInto`, which reaches `agentSize`, through a
 * scratch object, then two calls to `wrap`. Measured at pond scale that chain
 * cost 71ns a call with the trigonometry already memoized away, which is call
 * overhead rather than arithmetic: about 4.2ms a frame in the wall-mask pass
 * alone. The bodies of `stemRootInto` and `agentSize` are inlined here; the
 * originals stay for everyone else. Same operations in the same order, so the
 * result is bit-for-bit what the chain produced.
 */
export function stemOffsetInto(agent: Agent, slot: PortSlot, out: Vec2): Vec2 {
  const kind = agent.kind;
  let rx: number;
  let ry: number;
  if (kind === 'era') {
    rx = slot === 'p' ? 8 : 0;
    ry = 0;
  } else {
    const sz = 16;
    if (slot === 'p') {
      rx = sz * 1.05;
      ry = 0;
    } else {
      const legY = sz * 0.82 * 0.7;
      rx = -sz * 0.55;
      ry = slot === 'l' ? -legY : legY;
    }
  }
  const scale = agent.scale;
  const lx = rx * scale;
  const ly = ry * scale;
  if (agent.csHeading !== agent.heading) {
    agent.csHeading = agent.heading;
    agent.csCos = Math.cos(agent.heading);
    agent.csSin = Math.sin(agent.heading);
  }
  const c = agent.csCos;
  const sn = agent.csSin;
  out.x = lx * c - ly * sn;
  out.y = lx * sn + ly * c;
  return out;
}


/** Stem root offset from body center at a given heading. */
export function stemOffsetAt(heading: number, agent: Agent, slot: PortSlot): Vec2 {
  const loc = stemRoot(agent.kind, slot);
  return rotate(loc.x * agent.scale, loc.y * agent.scale, heading);
}

export function stemWorld(agent: Agent, slot: PortSlot, w: number, h: number): Vec2 {
  const o = stemOffset(agent, slot);
  return { x: wrap(agent.x + o.x, w), y: wrap(agent.y + o.y, h) };
}

export function stemFromPose(
  kind: AgentKind,
  x: number,
  y: number,
  heading: number,
  scale: number,
  slot: PortSlot,
  w: number,
  h: number,
): Vec2 {
  const loc = stemRoot(kind, slot);
  const r = rotate(loc.x * scale, loc.y * scale, heading);
  return { x: wrap(x + r.x, w), y: wrap(y + r.y, h) };
}

/** Write the stem into `out` so a hot loop does not allocate a result vector. */
const stemWorldScratch: Vec2 = { x: 0, y: 0 };

export function stemWorldInto(
  agent: Agent,
  slot: PortSlot,
  w: number,
  h: number,
  out: { x: number; y: number },
): { x: number; y: number } {
  // Through the into-form: the allocating `stemOffset` costs three objects a
  // call, and this is called twice per wire by both the length refresh and the
  // rope shape pass. On a pond of 14000 wires that was six figures of garbage
  // a frame from the function whose entire point is not to make any.
  const o = stemOffsetInto(agent, slot, stemWorldScratch);
  // `wrap` is the identity — the world stopped being toroidal — and the two
  // calls did not always vanish in the JIT. Kept in the signature so the
  // shape is obvious if wrapping ever comes back.
  void w;
  void h;
  out.x = agent.x + o.x;
  out.y = agent.y + o.y;
  return out;
}

/**
 * Control point along the port axis. Scaled to the wire's length when it is
 * known: a fixed handle longer than a third of the span makes the two handles
 * cross, and the cubic then doubles back on itself — which reads as a shorter
 * wire than the straight line between the ports.
 */
export function handleWorld(
  agent: Agent,
  slot: PortSlot,
  w: number,
  h: number,
  restLen?: number,
  maxHandle?: number,
): Vec2 {
  const root = stemWorld(agent, slot, w, h);
  const tip = portWorld(agent, slot, w, h);
  const d = wrapDeltaVec(root.x, root.y, tip.x, tip.y, w, h);
  const seg = Math.hypot(d.x, d.y) || 1;
  let handle =
    restLen === undefined
      ? HANDLE_SCALE * seg
      : Math.max(seg * 0.75, Math.min(restLen * 0.32, HANDLE_SCALE * seg));
  if (maxHandle !== undefined) handle = Math.min(handle, Math.max(1, maxHandle));
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
  const rootB = stemWorld(B, bSlot, w, h);
  const spanVec = wrapDeltaVec(p0.x, p0.y, rootB.x, rootB.y, w, h);
  const span = Math.hypot(spanVec.x, spanVec.y);
  // A handle longer than a third of the span crosses its partner and the cubic
  // loops off-screen — which is exactly what a collision that shoves two ports
  // together used to draw.
  const cap = Math.max(4, span * 0.33);
  const hA = handleWorld(A, aSlot, w, h, restLen, cap);
  const hB = handleWorld(B, bSlot, w, h, restLen, cap);
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
