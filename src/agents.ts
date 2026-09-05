import { extraCapFor } from './energy.ts';
import { CH } from './fields.ts';
import type { Params } from './params.ts';
import { rotate, wrap, wrapAngle, wrapDeltaVec, angleDelta, type Vec2 } from './wrap.ts';
import { AgentStore, KIND_CODE, CODE_KIND } from './agent-store.ts';

export type AgentKind = 'era' | 'dup' | 'con';
export type PortSlot = 'p' | 'l' | 'r';

export interface PortRef {
  id: number;
  slot: PortSlot;
}

/*
 * A flyweight over AgentStore: every field is a getter/setter pair reading
 * and writing one slot of a shared set of typed arrays, not data the
 * instance itself holds. See agent-store.ts for why (the AoS-vs-SoA
 * rewrite) and for the field-by-field storage layout.
 *
 * The accessors live on `Agent.prototype` — ordinary `get`/`set` class
 * members, shared by every instance — not installed per instance. That
 * used to be the other way around (`Object.defineProperties(this, ...)`
 * in the constructor), specifically so `{ ...agent }` would copy real
 * data instead of nothing: prototype accessors are not *own* enumerable
 * properties, so a spread only ever sees the plain instance fields
 * (`store`, `slot`). What that design didn't account for is V8: an object
 * whose *own* shape includes accessor properties (not just its
 * prototype's) gets pushed into dictionary-mode property storage, which
 * measured 8x slower than a plain field for `slot` — a field that isn't
 * even an accessor — sitting right next to them. Prototype accessors keep
 * every instance on one fast, shared hidden class instead; `store`/`slot`
 * read at plain-object speed, and the accessors themselves came out ~3x
 * faster too. See `cloneAgent` below for how spread got replaced.
 */
export class Agent {
  readonly store: AgentStore;
  readonly slot: number;
  private _chem: Float32Array | null = null;
  private _chemGen = -1;

  constructor(store: AgentStore, slot: number) {
    this.store = store;
    this.slot = slot;
  }

  get id(): number {
    return this.store.id[this.slot];
  }
  set id(v: number) {
    this.store.id[this.slot] = v;
  }

  get kind(): AgentKind {
    return CODE_KIND[this.store.kindCode[this.slot]];
  }
  set kind(v: AgentKind) {
    this.store.kindCode[this.slot] = KIND_CODE[v];
  }

  get x(): number {
    return this.store.x[this.slot];
  }
  set x(v: number) {
    this.store.x[this.slot] = v;
  }

  get y(): number {
    return this.store.y[this.slot];
  }
  set y(v: number) {
    this.store.y[this.slot] = v;
  }

  get vx(): number {
    return this.store.vx[this.slot];
  }
  set vx(v: number) {
    this.store.vx[this.slot] = v;
  }

  get vy(): number {
    return this.store.vy[this.slot];
  }
  set vy(v: number) {
    this.store.vy[this.slot] = v;
  }

  get heading(): number {
    return this.store.heading[this.slot];
  }
  set heading(v: number) {
    this.store.heading[this.slot] = v;
  }

  get omega(): number {
    return this.store.omega[this.slot];
  }
  set omega(v: number) {
    this.store.omega[this.slot] = v;
  }

  get mass(): number {
    return this.store.mass[this.slot];
  }
  set mass(v: number) {
    this.store.mass[this.slot] = v;
  }

  get alpha(): number {
    return this.store.alpha[this.slot];
  }
  set alpha(v: number) {
    this.store.alpha[this.slot] = v;
  }

  get scale(): number {
    return this.store.scale[this.slot];
  }
  set scale(v: number) {
    this.store.scale[this.slot] = v;
  }

  get locked(): boolean {
    return this.store.locked[this.slot] !== 0;
  }
  set locked(v: boolean) {
    this.store.locked[this.slot] = v ? 1 : 0;
  }

  /** Designer pin: physics must not move this body. Rewrites still may. */
  get pinned(): boolean {
    return this.store.pinned[this.slot] !== 0;
  }
  set pinned(v: boolean) {
    this.store.pinned[this.slot] = v ? 1 : 0;
  }

  get stun(): number {
    return this.store.stun[this.slot];
  }
  set stun(v: number) {
    this.store.stun[this.slot] = v;
  }

  /** Ornstein-Uhlenbeck self-propulsion magnitude along the heading. */
  get drive(): number {
    return this.store.drive[this.slot];
  }
  set drive(v: number) {
    this.store.drive[this.slot] = v;
  }

  /** Trail sampled for steering. */
  get trail(): number {
    return this.store.trail[this.slot];
  }
  set trail(v: number) {
    this.store.trail[this.slot] = v;
  }

  /** Pose at the start of the current integrate step (for XPBD velocity writeback). */
  get prevX(): number {
    return this.store.prevX[this.slot];
  }
  set prevX(v: number) {
    this.store.prevX[this.slot] = v;
  }

  get prevY(): number {
    return this.store.prevY[this.slot];
  }
  set prevY(v: number) {
    this.store.prevY[this.slot] = v;
  }

  get prevHeading(): number {
    return this.store.prevHeading[this.slot];
  }
  set prevHeading(v: number) {
    this.store.prevHeading[this.slot] = v;
  }

  get integVx(): number {
    return this.store.integVx[this.slot];
  }
  set integVx(v: number) {
    this.store.integVx[this.slot] = v;
  }

  get integVy(): number {
    return this.store.integVy[this.slot];
  }
  set integVy(v: number) {
    this.store.integVy[this.slot] = v;
  }

  get integOmega(): number {
    return this.store.integOmega[this.slot];
  }
  set integOmega(v: number) {
    this.store.integOmega[this.slot] = v;
  }

  /**
   * Energy on top of existence, in [−1, 1]. Positive is stock it can spend or
   * pass on, negative is debt it must settle before it can do either, and −1
   * is death. Upkeep decrements this directly.
   */
  get extra(): number {
    return this.store.extra[this.slot];
  }
  set extra(v: number) {
    this.store.extra[this.slot] = v;
  }

  /** Request gradient toward a hungry redex. 0 = quiet. */
  get request(): number {
    return this.store.request[this.slot];
  }
  set request(v: number) {
    this.store.request[this.slot] = v;
  }

  /**
   * How much this body cares about matching its neighbours' heading and
   * velocity, and how hard it pushes off them when they crowd. Heritable, so a
   * lineage can become a shoal or a scatter — flocking was one number for the
   * whole pond, which meant every net moved with the same temperament.
   *
   * A pair uses the mean of the two, not each body's own: the force is equal
   * and opposite with mass weighting, and per-body gains would break the
   * momentum conservation the settled-net tests check for. The mean still lets
   * a high-align lineage shoal and a low-align one ignore its neighbours, and
   * makes a mixed pair negotiate rather than one of them win.
   */
  get flockAlign(): number {
    return this.store.flockAlign[this.slot];
  }
  set flockAlign(v: number) {
    this.store.flockAlign[this.slot] = v;
  }

  get flockSep(): number {
    return this.store.flockSep[this.slot];
  }
  set flockSep(v: number) {
    this.store.flockSep[this.slot] = v;
  }

  /**
   * Chemistry: what this body says, and what it listens for.
   *
   * Eight floats in one array — `emit` in 0..3, `taste` in 4..7 — because both
   * sides of the scent field were already linear maps with the coefficients
   * hardcoded by kind. A principal port laid into the channel for its kind;
   * `mixScent` was a fixed dot product chosen by a switch. Making them per-body
   * turns those constants into genes without changing the shape of anything.
   *
   * Seeded from kind so a fresh pond behaves exactly as it did, and inherited
   * with mutation at a commute like the other heritable traits. Once they
   * drift the channels stop meaning con/dup/era/aux and become four registers
   * whose meaning is whatever a lineage has settled on — which is the point:
   * a net can evolve onto a channel pair nobody else answers.
   *
   * Aux ports still lay into channel 3 regardless, so that channel keeps its
   * kind-independent "a free port is here" sense.
   *
   * Sixteen floats, not eight: each of emit and taste has a slope against the
   * body's inner state as well as a base, so what it says and what it listens
   * for can depend on how its neighbourhood is doing. See `chemState`.
   *
   * A live view into AgentStore.chemAll, cached and re-sliced only when the
   * store's `generation` changes (a growth reallocation). Always
   * index-written (`agent.chem[k] = ...`), never reassigned.
   */
  get chem(): Float32Array {
    const store = this.store;
    if (this._chemGen !== store.generation) {
      this._chem = store.chemAll.subarray(this.slot * CHEM_LEN, this.slot * CHEM_LEN + CHEM_LEN);
      this._chemGen = store.generation;
    }
    return this._chem!;
  }

  /**
   * Set when the body falls into debt, cleared when it is back on its feet.
   *
   * Hunger measured against break-even stops the moment the debt is settled,
   * which left a rescued body pinned at exactly 0 — alive, one frame of upkeep
   * from dying again, and permanently unable to afford the share a rewrite
   * costs. The latch is what lets the ask outlive the debt: while it is set,
   * the body keeps asking up to its own `rescueTo` fill, so a rescue tops it back up to
   * something it can act with instead of parking it on the line.
   */
  get recovering(): boolean {
    return this.store.recovering[this.slot] !== 0;
  }
  set recovering(v: boolean) {
    this.store.recovering[this.slot] = v ? 1 : 0;
  }

  /**
   * Heritable traits. Seeded from the matching global slider when a body is
   * created outside a rewrite, so a fresh soup starts homogeneous just as it
   * did before these existed. A Con+Dup commute instead recombines both
   * parents' values into each child — blended for a Con child, assorted
   * whole from one parent per trait for a Dup child (see `inheritTraits` in
   * rewrite.ts) — which is the only place a population's traits can drift.
   */
  /** How much of this body's own demand survives one more hop outward. */
  get requestDecay(): number {
    return this.store.requestDecay[this.slot];
  }
  set requestDecay(v: number) {
    this.store.requestDecay[this.slot] = v;
  }

  /** The most this body can hold, in place of the flat per-kind cap. */
  get energyCap(): number {
    return this.store.energyCap[this.slot];
  }
  set energyCap(v: number) {
    this.store.energyCap[this.slot] = v;
  }

  /**
   * Extra at which this body dies. Always negative — a debt depth, never a
   * second positive cap. Seeded from the slider, then inherited.
   */
  get debtCap(): number {
    return this.store.debtCap[this.slot];
  }
  set debtCap(v: number) {
    this.store.debtCap[this.slot] = v;
  }

  /**
   * How far up this body's own tank a rescue fills, 0 at `debtCap` to 1 at
   * `energyCap`. The absolute target is `debtCap + rescueTo * (energyCap - debtCap)`.
   */
  get rescueTo(): number {
    return this.store.rescueTo[this.slot];
  }
  set rescueTo(v: number) {
    this.store.rescueTo[this.slot] = v;
  }

  /** How much of a kick this body's own pumps hand off instead of keeping. */
  get transportThrust(): number {
    return this.store.transportThrust[this.slot];
  }
  set transportThrust(v: number) {
    this.store.transportThrust[this.slot] = v;
  }

  /** How hard this body recoils, per unit of energy it pumps to a neighbour. */
  get transportRecoil(): number {
    return this.store.transportRecoil[this.slot];
  }
  set transportRecoil(v: number) {
    this.store.transportRecoil[this.slot] = v;
  }

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
  get csHeading(): number {
    return this.store.csHeading[this.slot];
  }
  set csHeading(v: number) {
    this.store.csHeading[this.slot] = v;
  }

  get csCos(): number {
    return this.store.csCos[this.slot];
  }
  set csCos(v: number) {
    this.store.csCos[this.slot] = v;
  }

  get csSin(): number {
    return this.store.csSin[this.slot];
  }
  set csSin(v: number) {
    this.store.csSin[this.slot] = v;
  }
}

/**
 * An independent copy: same field values, own private single-agent store.
 *
 * Prototype accessors read `this.store`/`this.slot`, not a value captured
 * at construction, so a shallow `{ ...agent }` spread would copy those
 * live — aliasing straight back into the original's slot instead of
 * producing a real snapshot. This builds a genuinely separate `Agent`.
 * Exists mainly for native/solver.test.ts and native/solver-extra.test.ts,
 * which snapshot an agent before handing it to the JS reference path, so
 * the WASM-vs-JS comparison has something the solver hasn't already moved.
 */
export function cloneAgent(a: Agent): Agent {
  const store = new AgentStore(1);
  const clone = new Agent(store, store.allocate(a.id));
  clone.kind = a.kind;
  clone.x = a.x;
  clone.y = a.y;
  clone.vx = a.vx;
  clone.vy = a.vy;
  clone.heading = a.heading;
  clone.omega = a.omega;
  clone.mass = a.mass;
  clone.alpha = a.alpha;
  clone.scale = a.scale;
  clone.locked = a.locked;
  clone.pinned = a.pinned;
  clone.stun = a.stun;
  clone.drive = a.drive;
  clone.trail = a.trail;
  clone.prevX = a.prevX;
  clone.prevY = a.prevY;
  clone.prevHeading = a.prevHeading;
  clone.integVx = a.integVx;
  clone.integVy = a.integVy;
  clone.integOmega = a.integOmega;
  clone.extra = a.extra;
  clone.request = a.request;
  clone.flockAlign = a.flockAlign;
  clone.flockSep = a.flockSep;
  clone.chem.set(a.chem);
  clone.recovering = a.recovering;
  clone.requestDecay = a.requestDecay;
  clone.energyCap = a.energyCap;
  clone.debtCap = a.debtCap;
  clone.rescueTo = a.rescueTo;
  clone.transportThrust = a.transportThrust;
  clone.transportRecoil = a.transportRecoil;
  clone.csHeading = a.csHeading;
  clone.csCos = a.csCos;
  clone.csSin = a.csSin;
  return clone;
}

/** True when physics must not integrate this body (rewrite lock or designer pin). */
export function poseHeld(a: Agent): boolean {
  return a.locked || a.pinned;
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

/** `chem` layout: emit, taste, then each one's slope against inner state. */
export const EMIT = 0;
export const TASTE = 4;
export const EMIT_SLOPE = 8;
export const TASTE_SLOPE = 12;
export const CHEM_LEN = 16;

/**
 * The inner state a body's chemistry is modulated by, normalised to [0, 1].
 *
 * `request` rather than `extra`, and the difference matters: request is
 * already aggregated across the net by `spreadRequests`, which walks the wire
 * adjacency and decays per hop. A body's request therefore reflects its
 * *neighbourhood's* need, not its own hunger. Emitting it turns a gradient
 * that only travels along wires into one that travels through space, so a
 * starving net can call to a forager that is not attached to it — which is
 * something the sim has no way to do at all otherwise.
 *
 * Hunger drives the cost instead; see `params.emitCost`.
 */
export function chemState(a: { request: number }): number {
  const r = a.request;
  return r <= 0 ? 0 : r >= 1 ? 1 : r;
}

/**
 * A flocking gain as the force sees it: never negative.
 *
 * The genes are allowed below zero so mutation has no reflecting barrier at
 * the off position, but neither force survives a negative gain. Alignment
 * pushes a body toward its neighbour's velocity, so a negative gain pushes it
 * away and relative velocity grows without bound — negative damping.
 * Separation only acts while a pair is closer than it wants to be, so a
 * negative gain pulls them together with no equilibrium to stop at. Both blow
 * up. Below zero simply means as off as off gets.
 */
export function flockGain(v: number): number {
  return v > 0 ? v : 0;
}

/**
 * Emit weight for one channel at this body's current state. Never negative.
 *
 * Silent on `CH.energy`, unconditionally and here rather than at the three
 * places that lay a deposit — the JS scatter, the packed vector the solver
 * reads, and the GPU's. That channel holds the ground itself now, and a body
 * that could emit into it would be minting food out of nothing at five units
 * a port a frame. One choke point, because a fourth deposit path would
 * otherwise be a very quiet way to break the economy.
 */
export function effEmit(a: Agent, c: number): number {
  if (c === CH.energy) return 0;
  const v = a.chem[EMIT + c] + chemState(a) * a.chem[EMIT_SLOPE + c];
  return v > 0 ? v : 0;
}

/** Taste weight for one channel at this body's current state. May be negative. */
export function effTaste(a: Agent, c: number): number {
  return a.chem[TASTE + c] + chemState(a) * a.chem[TASTE_SLOPE + c];
}

/**
 * The hardcoded weights, written out as a genome.
 *
 * Emit is one-hot on the channel that kind's principal used to lay into. Taste
 * is the row `mixScent` used to select with a switch — Con seeks Dup, Dup seeks
 * Con, the pairing that makes a redex, and Era seeks both strongly. Everyone is
 * mildly drawn to the aux channel, which is "unwired tissue here" rather than a
 * kind.
 *
 * The attract sliders seed a new body and are not read again, which is how
 * every other heritable trait already works: the slider sets where a fresh
 * population starts, and breeding takes it from there.
 */
export function seedChem(kind: AgentKind, params: Params): Float32Array {
  // Slopes start at zero, so a seeded body says the same thing however its
  // net is doing and the whole modulation is inert until breeding moves it.
  const c = new Float32Array(CHEM_LEN);
  const S = params.attractStrong;
  const M = params.attractMedium;
  /*
   * Drawn to food when hungry, and blind to it when fed. On the slope against
   * `request`, not on the base — the one place in this seed where a slope
   * starts anywhere but zero, and it earns the exception.
   *
   * A constant attraction looks safe, on the argument that a flat field steers
   * nothing: where the ground is untouched both sensors read the same and
   * there is nothing to turn on. That argument is wrong, and measurably so. A
   * body harvests from the cell it is standing in, so within a frame or two it
   * has eaten a dip underneath itself — and then it smells the dip. It is
   * chasing a gradient of its own making, which is the same self-trail
   * artifact the sensor geometry is tuned to reject, arriving by a different
   * door. Seeded flat at 0.9 it cost eight tests: nets dispersed instead of
   * settling, and a lone body wound itself in circles on its own grazing.
   *
   * Gating on need fixes it at the root rather than by turning the gain down.
   * A fed body has nothing to gain from food and ignores it, so it keeps the
   * behaviour it always had; a hungry one — and `request` is the
   * neighbourhood's hunger, already spread along the wires — turns toward the
   * ground. Which is what foraging is, and what none of this was able to
   * express at any genome before the ground was something you could smell.
   */
  c[TASTE_SLOPE + CH.energy] = params.attractFood;
  if (kind === 'con') {
    c[EMIT] = 1;
    c[TASTE + 1] = M;
    c[TASTE + 3] = M;
  } else if (kind === 'dup') {
    c[EMIT + 1] = 1;
    c[TASTE] = M;
    c[TASTE + 3] = M;
  } else {
    /*
     * An Era says nothing at seed.
     *
     * It used to spend its whole unit of voice on channel 2, which no kind
     * has ever had a taste for — it was shouting into a band with no
     * receivers, and that is exactly why channel 2 was free for the ground to
     * move into. Emitting nothing is what it already amounted to; this just
     * stops pretending otherwise, and stops the unit-sum budget being spent
     * on a channel that cannot carry it.
     *
     * Not permanent. An Era is the body that produces energy rather than
     * spending it, so it is the obvious thing to give a voice back to once
     * emitting *is* producing — but that is an economy change and this is a
     * storage one. Breeding will hand its descendants a voice long before
     * then: one erase past the seed, mutation and the renormalisation give a
     * child a full unit spread across the three channels that carry.
     */
    c[TASTE] = S;
    c[TASTE + 1] = S;
    c[TASTE + 3] = M;
  }
  return c;
}

/**
 * `store` defaults to a fresh, private, single-agent `AgentStore` when
 * omitted — every one of this project's ~386 test-side `createAgent` calls
 * (and the few production call sites that don't yet thread a shared store,
 * e.g. render.ts's ghosts) keeps working exactly as it did when `Agent` was
 * a plain object, just with one small typed-array table backing it instead
 * of none. `Sim` and `commitRewrite` pass their own shared store explicitly.
 */
export function createAgent(
  id: number,
  kind: AgentKind,
  x: number,
  y: number,
  heading: number,
  params: Params,
  store: AgentStore = new AgentStore(1),
): Agent {
  const slot = store.allocate(id);
  const agent = new Agent(store, slot);
  agent.kind = kind;
  agent.x = x;
  agent.y = y;
  agent.vx = 0;
  agent.vy = 0;
  agent.heading = heading;
  agent.omega = 0;
  agent.mass = massFor(kind, params);
  agent.alpha = 1;
  agent.scale = 1;
  agent.locked = false;
  agent.pinned = false;
  agent.stun = 0;
  agent.drive = params.stepSpeed;
  agent.trail = 0;
  agent.prevX = x;
  agent.prevY = y;
  agent.prevHeading = heading;
  agent.integVx = 0;
  agent.integVy = 0;
  agent.integOmega = 0;
  agent.csHeading = NaN;
  agent.csCos = 1;
  agent.csSin = 0;
  agent.chem.set(seedChem(kind, params));
  agent.extra = 0;
  agent.request = 0;
  agent.flockAlign = params.flockAlign;
  agent.flockSep = params.flockSep;
  agent.recovering = false;
  agent.requestDecay = params.requestDecay;
  agent.energyCap = extraCapFor(kind);
  agent.debtCap = params.debtCap;
  agent.rescueTo = params.rescueTo;
  agent.transportThrust = params.transportThrust;
  agent.transportRecoil = params.transportRecoil;
  return agent;
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
