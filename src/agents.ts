import { EXTRA_CAP, extraCapFor } from './energy.ts';
import { REACT_B, REACT_C, REACT_D, REACT_SPECIES } from './agent-store.ts';
import {
  CHEM_LEN,
  EMIT,
  E_OUT,
  F_BASE,
  G_BASE,
  GW_BASE,
  GX_BASE,
  HEAD_SCALE,
  TX_A,
  SW_BASE,
  TX_B,
  TX_BASE,
  W_C,
  W_D,
  TX_C,
  TX_D,
  KS_BASE,
  L_BASE,
  B_STATE,
  F_OUT,
  IN_DEMAND,
  IN_FULL,
  IN_DIMS,
  IN_SENSE,
  P_BASE,
  STATE_DIMS,
  TASTE,
  T_OUT,
  W_IN,
} from './chem-layout.ts';
import { CH } from './fields.ts';
import type { Params } from './params.ts';
import { rotate, wrap, wrapAngle, wrapDeltaVec, angleDelta, type Vec2 } from './wrap.ts';
import { AgentStore, KIND_CODE, CODE_KIND } from './agent-store.ts';
import { KIND_ERA } from './native/solver.ts';

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
  private _h: Float64Array | null = null;
  private _hGen = -1;
  private _sense: Float64Array | null = null;
  private _senseGen = -1;

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
   * Thirty-two floats, not eight: each of emit and taste has a base per
   * channel and a slope against every dimension of the body's inner state, so
   * what it says and what it listens for can depend on how its neighbourhood
   * is doing, on how full it is itself, and on how much it likes where it is
   * standing — separately, and with a sign. See `STATE_DIMS`.
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
   * How many rewrites deep this body is from a founder, and which founder.
   *
   * Read by nothing the sim does. They are here so that "did this change
   * help?" is a question with an answer. Every heritable trait in this project
   * drifts as well as adapts, and with `CHEM_MUTATE` across a genome this size
   * the drift is not small — a genome that has moved away from its seed, which
   * is all `chem-evolution` can currently show, is equally consistent with
   * selection and with a random walk. Telling those apart needs to know how
   * deep a line is and which lines are still alive.
   */
  get born(): number {
    return this.store.born[this.slot];
  }
  set born(v: number) {
    this.store.born[this.slot] = v;
  }

  get lineage(): number {
    return this.store.lineage[this.slot];
  }
  set lineage(v: number) {
    this.store.lineage[this.slot] = v;
  }

  /**
   * This body's recurrent state, a live view into `AgentStore.hAll`.
   *
   * Re-sliced only when the store reallocates, like `chem`. Always
   * index-written, never reassigned.
   */
  get h(): Float64Array {
    const store = this.store;
    if (this._hGen !== store.generation) {
      this._h = store.hAll.subarray(this.slot * STATE_DIMS, this.slot * STATE_DIMS + STATE_DIMS);
      this._hGen = store.generation;
    }
    return this._h!;
  }

  /**
   * The four channel readings at this body's position, scaled as the genome
   * reads them: signals by `1 / SENSE_SCALE`, the ground by `1 / cellCap`.
   *
   * Scaled at the write, on both paths, rather than raw here and scaled in
   * place by the state pass — which left this holding scaled values on the
   * CPU path and raw ones on the GPU path, depending on which pass had run
   * last. On the CPU path it is only refreshed on frames where this body's
   * `Wx` actually reads the field; stale otherwise, and not fed into the
   * state when stale.
   */
  get sense(): Float64Array {
    const store = this.store;
    if (this._senseGen !== store.generation) {
      this._sense = store.senseAll.subarray(this.slot * 4, this.slot * 4 + 4);
      this._senseGen = store.generation;
    }
    return this._sense!;
  }

  /** This frame's locomotion head: cruise speed and turn gain. */
  get cruise(): number {
    return this.store.cruise[this.slot];
  }

  get turn(): number {
    return this.store.turn[this.slot];
  }

  /** Fraction of this body's ports that are attached. See `BOUND`. */
  get bound(): number {
    return this.store.bound[this.slot];
  }
  set bound(v: number) {
    this.store.bound[this.slot] = v;
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

  /**
   * How particulate this lineage's inheritance is, 0 to 1.
   *
   * Not read by the body itself — it is read when the body *breeds*, to decide
   * per gene whether a child copies one parent whole or blends the two. See
   * `assortChance`, which offsets it by the child's kind so that the seeded
   * 0.5 reproduces the old absolute Con-blends/Dup-assorts rule.
   */
  get assort(): number {
    return this.store.assort[this.slot];
  }
  set assort(v: number) {
    this.store.assort[this.slot] = v;
  }

  /** How much of a kick this body's own pumps hand off instead of keeping. */
  get transportThrust(): number {
    return this.store.transportThrust[this.slot];
  }
  set transportThrust(v: number) {
    this.store.transportThrust[this.slot] = v;
  }

  /**
   * How hard this body pulls fuel into its reactor, as a multiple of
   * `metabolicSupply`. The doc's per-agent `J`, and heritable.
   */
  get intake(): number {
    return this.store.intake[this.slot];
  }
  set intake(v: number) {
    this.store.intake[this.slot] = v;
  }

  /** How hard this body recoils, per unit of energy it pumps to a neighbour. */
  get transportQuantum(): number {
    return this.store.transportQuantum[this.slot];
  }
  set transportQuantum(v: number) {
    this.store.transportQuantum[this.slot] = v;
  }
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
  clone.born = a.born;
  clone.lineage = a.lineage;
  clone.bound = a.bound;
  // A clone that does not clone these is a body with someone else's genome and
  // a blank mind. Latent — only native-parity tests call this today — but the
  // next caller (a designer ghost, a rollback) would get it silently.
  clone.h.set(a.h);
  clone.sense.set(a.sense);
  /*
   * The reactor and the gut, for the same reason and with a sharper edge.
   * An empty vat is not a neutral one: the wave is `2C/(K+C) - 1`, so a body
   * holding no catalyst reads -1 rather than 0, and -1 is a real stroke and a
   * real change to its drag. Dropping these gave a clone a different drag law
   * from the body it was cloned from, which is exactly how the native-parity
   * tests caught it — 0.70px of divergence in a scene budgeted for 0.002.
   */
  const ro = a.slot * REACT_SPECIES;
  const co = clone.slot * REACT_SPECIES;
  for (let k = 0; k < REACT_SPECIES; k++) clone.store.react[co + k] = a.store.react[ro + k];
  clone.store.gut[clone.slot] = a.store.gut[a.slot];
  clone.store.intake[clone.slot] = a.store.intake[a.slot];
  clone.store.starve[clone.slot] = a.store.starve[a.slot];
  clone.store.gaitWave[clone.slot] = a.store.gaitWave[a.slot];
  clone.store.gaitAnchor[clone.slot] = a.store.gaitAnchor[a.slot];
  clone.store.anchor[clone.slot] = a.store.anchor[a.slot];
  refreshReadsField(clone);
  clone.stun = a.stun;
  clone.drive = a.drive;
  clone.trail = a.trail;
  clone.prevX = a.prevX;
  clone.prevY = a.prevY;
  clone.prevHeading = a.prevHeading;
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
  clone.assort = a.assort;
  clone.transportThrust = a.transportThrust;
  clone.transportRecoil = a.transportRecoil;
  clone.transportQuantum = a.transportQuantum;
  clone.csHeading = a.csHeading;
  clone.csCos = a.csCos;
  clone.csSin = a.csSin;
  return clone;
}

/** True when physics must not integrate this body (rewrite lock or designer pin). */
export function poseHeld(a: Agent): boolean {
  return poseHeldAt(a.store, a.slot);
}

/**
 * `poseHeld` from a store and a slot, for packs that walk every body.
 *
 * The flyweight's `locked` and `pinned` getters each reach through `store`
 * and `slot` to arrive at the array this reads directly. One definition, two
 * ways in: the packs hold the store already, and everything else holds a body.
 */
export function poseHeldAt(store: AgentStore, slot: number): boolean {
  return store.locked[slot] !== 0 || store.pinned[slot] !== 0;
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

/*
 * Frozen and shared, because `slotsFor` builds a fresh array on every call and
 * a `for...of` over it builds a fresh iterator. Once a body a frame that is
 * nothing; once a body in the GPU deposit pack, at five thousand bodies, it is
 * ten thousand short-lived objects for a value with two possible answers.
 * Callers that only iterate should reach for these; `slotsFor` stays for the
 * ones that want a list of their own.
 */
export const ERA_SLOTS: readonly PortSlot[] = Object.freeze(['p'] as PortSlot[]);
export const NODE_SLOTS: readonly PortSlot[] = Object.freeze(['p', 'l', 'r'] as PortSlot[]);

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

/** Area and perimeter of the unit Con/Dup triangle (`triangleLocal` at s=1). */
const TRI_AREA = 1.312;
const TRI_PERIM = 1.64 + 2 * Math.hypot(1.6, 0.82);

/**
 * Radius of the disc that stands in for the Con/Dup triangle, as a multiple
 * of `agentSize`.
 *
 * Matched on the *contact* region, not on the glyph. What a contact radius
 * has to reproduce is how far two bodies come to rest, and that is the
 * Minkowski sum A + (-B), whose boundary is exactly where two centres touch.
 * Averaged over relative heading its area is
 *
 *     2 * TRI_AREA + TRI_PERIM^2 / (2 PI)
 *
 * (the mixed-area term of a rotational average), and the disc pair that
 * excludes the same area has radius sqrt(that / 4 PI) = 0.7457 s.
 *
 * The glyph-area radius, sqrt(TRI_AREA / PI) = 0.6462 s, was the same
 * arithmetic applied to the wrong region: it matches the area one body
 * covers, which is not the area a pair of them keeps clear. It ran 13% small
 * — two Con/Dup rested 20.7 px apart where SAT rests them 23.6 px apart on
 * average — and the net stepped outward the moment the camera crossed the
 * LOD line and the pair started running SAT instead.
 */
export const TRI_DISC_RATIO = Math.sqrt(
  (2 * TRI_AREA + (TRI_PERIM * TRI_PERIM) / (2 * Math.PI)) / (4 * Math.PI),
);

/**
 * Contact radius for the tiers that collide discs instead of SAT polygons.
 *
 * `boundRadius` is the circumscribed bound, which for a triangle is ~1.5x too
 * fat to use as a contact radius. This is the disc that keeps as much room
 * clear as the triangle does, so a pair rests where SAT would rest and the
 * net does not change size when the camera crosses the LOD line. A disc
 * cannot have a heading, so it cannot reproduce the 15.3-33.6 px spread two
 * headings give SAT; it sits at the mean of that, 23.9 px against SAT's 23.6,
 * so the tiers agree on the average pair and not on any particular one.
 *
 * Callers must add `SKIN` twice, as SAT does. It is not folded in here
 * because the same radius sizes broad-phase cells and the world bound, and
 * neither wants a contact skin.
 */
export function discRadius(agent: Agent): number {
  if (agent.kind === 'era') return ERA_RADIUS * agent.scale;
  return agentSize(agent.kind) * TRI_DISC_RATIO * agent.scale;
}

/**
 * Sum of squared vertex radii of the unit Con/Dup triangle (`triangleLocal`
 * at `s = 1`): `1.05^2 + 2 * (0.55^2 + 0.82^2)`. The inertia below used to
 * build the three vertices and sum them, which allocated an array and three
 * points per call — and `packPose` asks for the inertia of every body every
 * frame, so that was two hundred thousand short-lived objects a frame at a
 * fifty-thousand-body pond, in a function whose answer is a constant times
 * `scale^2`.
 */
const TRI_VERTEX_R2 = 1.05 * 1.05 + 2 * (0.55 * 0.55 + 0.82 * 0.82);

export function momentOfInertia(agent: Agent): number {
  // Straight out of the store rather than back through the string: the code
  // is what is stored, and `agent.kind` exists to turn it into a name.
  return momentOfInertiaAt(agent.store.kindCode[agent.slot], agent.mass, agent.scale);
}

/**
 * `momentOfInertia` from a kind code rather than a body.
 *
 * The store already holds the kind as the same small integer the solver
 * wants, so a pack that goes through the flyweight turns it into a string to
 * compare against `'era'` and then back into an integer to write out. One
 * definition, and the packs no longer round-trip through the string.
 */
export function momentOfInertiaAt(kindCode: number, mass: number, scale: number): number {
  const m = Math.max(0.08, mass);
  if (kindCode === KIND_ERA) {
    const r = ERA_RADIUS * scale;
    return 0.5 * m * r * r;
  }
  const s = 16 * scale;
  return (m * TRI_VERTEX_R2 * s * s) / 6;
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

export {
  B_STATE,
  CHEM_LEN,
  GAIT_ANCHOR_MAX,
  G_BASE,
  G_OUT,
  CHEM_SPECIES,
  CRITIC_LEN,
  EMIT,
  E_OUT,
  F_BASE,
  F_OUT,
  HEAD_SCALE,
  L_BASE,
  L_OUT,
  IN_BOUND,
  IN_DEMAND,
  IN_DIMS,
  IN_FULL,
  IN_SENSE,
  LEARN_CRITIC,
  LEARN_PREV_V,
  LEARN_STRIDE,
  LEARN_TRACE,
  PLASTIC_BASE,
  PLASTIC_LEN,
  P_BASE,
  P_OUT,
  KS_BASE,
  SENSE_SCALE,
  STATE_DIMS,
  TASTE,
  T_OUT,
  W_IN,
  W_NET,
  W_SELF,
} from './chem-layout.ts';



/**
 * A bare body carrying just what `effEmit` and `effTaste` read, for tests that
 * want to poke a genome without building a `Sim`.
 *
 * It exists because the alternative kept going wrong. Tests were writing
 * `{ chem, request } as unknown as Agent`, and every time the shape changed
 * that cast turned what should have been a compile error into `undefined`
 * arithmetic, surfacing as a *zero* emit weight in some other file. Twice.
 * Going through here means the next change breaks the build at one function.
 */
export function bareBody(chem: Float32Array, over: { h?: number[] } = {}): Agent {
  return {
    chem,
    h: Float64Array.from(over.h ?? new Array(STATE_DIMS).fill(0)),
    sense: new Float64Array(4),
  } as unknown as Agent;
}



/**
 * Recompute the cached "does this genome look at the field" flag.
 *
 * Must be called after anything writes `chem`. That is birth — `createAgent`
 * and `inheritChem` — plus `cloneAgent` and whatever tests poke directly. A
 * stale `false` here is a body that has evolved sense weights and cannot see,
 * which would look exactly like the weights not working.
 */
export function refreshReadsField(a: Agent): void {
  const ch = a.chem;
  let reads = 0;
  for (let d = 0; d < STATE_DIMS && !reads; d++) {
    const wi = W_IN + d * IN_DIMS + IN_SENSE;
    for (let c = 0; c < 4; c++) {
      if (ch[wi + c] !== 0) {
        reads = 1;
        break;
      }
    }
  }
  a.store.readsField[a.slot] = reads;
  // This is the one choke point every chem write already has to pass through,
  // so it is also where the GPU's copy of the genome learns it is stale.
  a.store.markChem(a.slot);
}

/** One row of an output head: `base + row . h`. */
export function head(a: Agent, matrix: number, base: number, row: number): number {
  const ch = a.chem;
  const h = a.h;
  const o = matrix + row * STATE_DIMS;
  let v = ch[base + row];
  for (let d = 0; d < STATE_DIMS; d++) v += ch[o + d] * h[d];
  return v;
}

/**
 * The realised emit vector: what this genome actually says, into `out`.
 *
 * `relu(E.h + e0)` per channel and then **normalised to one unit across all
 * four**, which is the budget the whole honesty argument rests on and which
 * was, until this function existed, enforced nowhere.
 *
 * It had been applied to `e0` at birth by `inheritChem` and never again. `E`
 * is free to add up to `CHEM_SLOPE_MAX` per state dimension on top, so a
 * genome satisfying every invariant inheritance guarantees — bases
 * non-negative and summing to one, every `E` entry inside its bound —
 * realised a total of 17 against a documented budget of 1. "Louder is
 * strictly better" was reachable, which is the exact thing the budget exists
 * to prevent, and the trade-off that justified deleting `emitCost` did not
 * hold: measured, the ground channel and both signal channels rose *together*
 * off a single state dimension.
 *
 * Normalising the realised vector rather than tightening the slope bound is
 * the fix that restores the documented invariant instead of merely shrinking
 * the violation. It has to see all four channels at once, which is why this
 * is a vector and not four scalar calls.
 *
 * A body that has mutated its way to silence stays silent rather than being
 * amplified back out of noise — the same rule `inheritChem` uses on the bases.
 */
export function emitVector(chem: Float32Array, g: number, h: Float64Array, ho: number, out: Float64Array, oo: number): void {
  // Unrolled for the same reason the state update is: four channels by four
  // dimensions is a compile-time shape, and the loop around sixteen
  // multiply-adds costs more than the arithmetic.
  const h0 = h[ho];
  const h1 = h[ho + 1];
  const h2 = h[ho + 2];
  const h3 = h[ho + 3];
  let sum = 0;
  {
    const o = g + E_OUT + 0;
    const v =
      chem[g + EMIT + 0] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    const w = v > 0 ? v : 0;
    out[oo + 0] = w;
    sum += w;
  }
  {
    const o = g + E_OUT + 4;
    const v =
      chem[g + EMIT + 1] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    const w = v > 0 ? v : 0;
    out[oo + 1] = w;
    sum += w;
  }
  {
    const o = g + E_OUT + 8;
    const v =
      chem[g + EMIT + 2] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    const w = v > 0 ? v : 0;
    out[oo + 2] = w;
    sum += w;
  }
  {
    const o = g + E_OUT + 12;
    const v =
      chem[g + EMIT + 3] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    const w = v > 0 ? v : 0;
    out[oo + 3] = w;
    sum += w;
  }

  // Reciprocal once rather than four divides; this runs per body per frame.
  if (sum > 1e-6) {
    const inv = 1 / sum;
    for (let c = 0; c < 4; c++) out[oo + c] *= inv;
  }
}

/** The taste vector. Signed, and not normalised — a taste weight is compared
 *  against other taste weights rather than spent, so there is no budget. */
export function tasteVector(chem: Float32Array, g: number, h: Float64Array, ho: number, out: Float64Array, oo: number): void {
  const h0 = h[ho];
  const h1 = h[ho + 1];
  const h2 = h[ho + 2];
  const h3 = h[ho + 3];
  {
    const o = g + T_OUT + 0;
    const v =
      chem[g + TASTE + 0] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    out[oo + 0] = v;
  }
  {
    const o = g + T_OUT + 4;
    const v =
      chem[g + TASTE + 1] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    out[oo + 1] = v;
  }
  {
    const o = g + T_OUT + 8;
    const v =
      chem[g + TASTE + 2] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    out[oo + 2] = v;
  }
  {
    const o = g + T_OUT + 12;
    const v =
      chem[g + TASTE + 3] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    out[oo + 3] = v;
  }
}

const SCRATCH4 = new Float64Array(4);

/**
 * Emit weight for one *signal* channel. Never negative.
 *
 * Zero on `CH.energy`, and the reason is a choke point and nothing else. The
 * ground is a thing a body can put into the field — by dying, by spilling, by
 * paying the reactor's routed food back at `upkeepExcrete` — but it must never
 * reach the field through the scent deposit path, because that path multiplies
 * by `params.deposit`, which is five. A body emitting a whole unit would put
 * five units of food a frame into the world out of nothing.
 *
 * So the emit head's ground slot is read by nothing. It stays a quarter of
 * `emitVector`'s simplex because dropping it would renormalise every genome
 * in the library, and `seedChem` still writes an Era's unit of voice there
 * because a seed says what a kind is for; what is left of the question is
 * whether that simplex should be three wide, which is deferred for the same
 * reason.
 */
export function effEmit(a: Agent, c: number): number {
  if (c === CH.energy) return 0;
  emitVector(a.chem, 0, a.h, 0, SCRATCH4, 0);
  return SCRATCH4[c];
}

/** Taste weight for one channel. May be negative — that is avoidance. */
export function effTaste(a: Agent, c: number): number {
  tasteVector(a.chem, 0, a.h, 0, SCRATCH4, 0);
  return SCRATCH4[c];
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
 * How hard each kind grips at its point in the stroke: an Era is an oar and a
 * Con or a Dup is a foot.
 *
 * A base, not a matrix entry, and the matrix stays zero — so a fresh body
 * grips by a constant and a lineage is free to make it depend on `h`, or to
 * swap the two roles outright, by drifting `G`. Same shape as `e0`, and the
 * same argument: a behaviour should start as the constant it would otherwise
 * have been hardcoded to, and become a phenotype by evolving.
 *
 * The split is what the travel is made of. A wire swinging its rest length
 * moves both its bodies and not their centre; what is left over is the
 * velocity that correction induces, decaying at each body's own rate. Equal
 * rates, nothing left. So the two ends of a wire have to grip differently,
 * and an Era supplies that difference structurally: one port, so always a
 * leaf, with one uncancelled stroke where an interior body has three that
 * partly fight; light; and a producer holding a larger store, so it already
 * sits at a fullness its neighbour does not, which `grip` turns into a second
 * difference pointing the same way.
 *
 * Under `drag` on purpose. Past it the rate clamps at zero, both ends clamp
 * *together*, and the clamp destroys the asymmetry this exists to make.
 *
 * One structural condition is not seeded here and cannot be: an Era is a limb
 * only where it lands on an *auxiliary* port. A redex needs principals at
 * both ends (`Sim.collectReadyRedexes`) and an Era has nothing but a
 * principal — so on a Con's `l` or `r` it is an appendage, and on a Con's `p`
 * it is an erase waiting to happen.
 */
const GAIT_ANCHOR_ERA = 0.02;
const GAIT_ANCHOR_NODE = 0.25;

/**
 * What each kind broadcasts down its principal wire, at the seed: **one kind,
 * one species**, as genes a lineage drifts from rather than a rule it is held
 * to.
 *
 * An Era broadcasts the food it ate, a Con the catalyst C, a Dup the inhibitor
 * D. That is `docs/scratch.txt` §4.1 exactly, and the one species each is the
 * whole of the rule — there is no kind with a second channel and no case to
 * remember.
 *
 * **An Era sends fuel, not primer.** It used to send both, which was §6.1's
 * reading, and the argument against it is `docs/metabolism-spec.md` §2.3's own:
 * the gut and the primer are two timescales, the gut measuring minutes and the
 * primer seconds. Broadcasting the primer ships the fast currency, so it
 * arrives at one neighbour, drives that one body far up its own operating
 * curve, and makes it the instability the whole net is then slaved to.
 * Broadcasting the gut ships the slow one: the neighbour digests it and makes
 * its own primer, at its own rate, from its own `intake`. Computed over the
 * library and 38 grown nets, dropping the primer broadcast gives half again as
 * many independently oscillating modes and a quarter lower leading growth rate
 * — more of the net moving, and nothing running away.
 *
 * Note which way it runs. A Con's or a Dup's principal faces *out* of the net
 * toward a redex, so its signal travels toward whatever it is about to rewrite
 * with; an Era's principal is its only attachment, so its food travels inward.
 * Neither is a rule — both are where a fresh body starts.
 */
const SEED_SEND = 1;

/** The one species each kind speaks. `npm run pond -- spectrum` reads it. */
export const SEED_SPECIES: Record<AgentKind, number> = {
  era: TX_A,
  con: TX_C,
  dup: TX_D,
};

function seedGait(c: Float32Array, kind: AgentKind, params: Params): void {
  c[G_BASE] = kind === 'era' ? GAIT_ANCHOR_ERA : GAIT_ANCHOR_NODE;
  c[TX_BASE + SEED_SPECIES[kind]] = SEED_SEND;
  /*
   * The broadcast gate, **each species at the same fraction of its own natural
   * level**. It used to seed flat, one number across four pools whose scales
   * differ by orders — the catalyst runs about 1, the inhibitor `k3/d` times
   * that, the primer `(k3+d)/k2` — so a gate that pulsed one of them was
   * either inert or shut for the others. That is why it shipped at zero and
   * gated nothing.
   *
   * At the same fraction of each, it means one thing: *a body stays quiet
   * below this much of its own resting level and speaks above it*. The
   * broadcast was already a pulse, because mass action sends most at a peak
   * and least at a trough; this makes it a pulse with a **floor**, so a body
   * at its trough says nothing at all rather than a little.
   */
  const gate = params.metabolicGate;
  const reset = params.metabolicReset;
  const decay = params.metabolicDecay;
  const cat = params.metabolicCat;
  c[GX_BASE + TX_A] = gate * EXTRA_CAP;
  c[GX_BASE + TX_B] = gate * (cat > 0 ? (reset + decay) / cat : 0);
  c[GX_BASE + TX_C] = gate;
  c[GX_BASE + TX_D] = gate * (decay > 0 ? reset / decay : 0);
  /*
   * The two actuators' mixtures, seeded to exactly what they were welded to:
   * the stroke reads the catalyst, the grip reads the inhibitor 53 degrees
   * behind it. A fresh pond is bit for bit what it was, and what is new is
   * that a lineage can move either. See `SW_BASE`.
   */
  c[SW_BASE + W_C] = 1;
  c[GW_BASE + W_D] = 1;
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
  /*
   * The one place a matrix is seeded away from zero, and it is a two-hop
   * pathway rather than a weight: `h[0]` is wired to carry `DEMAND`, and taste
   * for the ground is wired to read `h[0]`. So a fresh body is drawn to food
   * exactly when its neighbourhood is short of energy, and is blind to it
   * otherwise — which is the behaviour, and it now has to be *built* out of
   * the same parts a lineage would use rather than hardcoded as a slope.
   *
   * Blind when fed matters for a reason worth keeping: a body harvests the
   * cell it stands in, so within a frame or two it has eaten a dip underneath
   * itself and would otherwise chase the dip. `phi` compresses [0,1] to
   * [0,0.5], so the gain is doubled to land where the old flat slope did.
   */
  c[W_IN + 0 * IN_DIMS + IN_DEMAND] = 1;
  c[T_OUT + CH.energy * STATE_DIMS + 0] = params.attractFood * 2;
  // Output-head bases: the sliders' values, so a fresh body's flocking and
  // pumping are the constants they used to be *at half a tank*.
  c[F_BASE] = params.flockAlign / HEAD_SCALE.align;
  c[F_BASE + 1] = params.flockSep / HEAD_SCALE.sep;
  c[P_BASE] = params.transportThrust / HEAD_SCALE.thrust;
  c[P_BASE + 1] = params.transportRecoil / HEAD_SCALE.recoil;
  c[L_BASE] = params.stepSpeed / HEAD_SCALE.cruise;
  c[L_BASE + 1] = params.turnRate / HEAD_SCALE.turn;
  /*
   * The second two-hop pathway, and the reason is written on `F_OUT`: "no body
   * could shoal while fed and scatter while starving, which is the obvious
   * thing for a forager to do and was not expressible at any genome".
   *
   * `h[1]` is wired to carry the body's own tank, **centred**: the weight is 2
   * and the bias −1, so the pre-activation runs [−1, 1] across an empty tank
   * to a full one and `phi` lands it on [−0.5, +0.5]. Centred so the swing is
   * symmetric about the slider's value — a body at half a tank behaves exactly
   * as it used to, a full one shoals and settles, an empty one scatters and
   * runs. Off-centre the base would be a floor and "scatter when starving"
   * could not be said at all.
   *
   * Two heads off the one wire, which is what a state dimension is for. The
   * bias is inside the learned block, so a lineage can move its own set-point
   * — where half a tank stops feeling like enough — without touching either
   * head.
   */
  c[W_IN + 1 * IN_DIMS + IN_FULL] = 2;
  c[B_STATE + 1] = -1;
  /*
   * `phi` gives ±0.5 at the ends, so a gain of one base swings each head over
   * half its value either way: an empty body shoals at half and runs at half
   * again as fast, a full one the other way round.
   *
   * Twice that was tried first and is too much. A fresh spawn arrives at about
   * four fifths of a tank, so at a full swing it barely cruises at all — 9.5
   * against a slider of 38 — and a pond of well-fed bodies that will not move
   * stops meeting, stops latching, and grows nets of two. The swing has to be
   * something a body does, not something that switches it off.
   */
  c[F_OUT + 0 * STATE_DIMS + 1] = c[F_BASE];
  seedGait(c, kind, params);
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
    /*
     * An Era's one unit of voice goes into the ground.
     *
     * It said nothing at all for a while, because its seeded channel became
     * the ground and nothing emits onto the ground through the scent path.
     * Farming is what an Era is *for* under this economy: one port, cannot
     * commute, pays no rent. Its whole job is to be somewhere useful, and the
     * useful thing to do with stock is put it where the logistic term can
     * multiply it — seeding a scarred cell restarts growth that a depleted
     * cell can never restart on its own.
     *
     * The *emit* slot, which nothing reads any more: farming is `excrete_2` on
     * the expression head now, and `seedProduction` is where an Era is told to
     * be a ground-maker. Left here because it is a quarter of a simplex and
     * because it still says what an Era is for, which is what a seed is; see
     * `effEmit` for why the slot itself has not gone.
     */
    c[EMIT + CH.energy] = 1;
    c[TASTE] = S;
    c[TASTE + 1] = S;
    c[TASTE + 3] = M;
  }
  /*
   * The affinity genes, at one natural unit each: a fresh body's uptake reads
   * `params.uptakeKs` on every species, which is exactly what it read before
   * this gene existed. `X` and its base stay at zero, so expression is flat
   * across all eight rows and every reaction runs at whatever constant it ran
   * at. See `chem-layout.ts`.
   */
  c[KS_BASE] = 1;
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
  /*
   * A nudge off the metabolic steady state, so a fresh body's pathway starts
   * somewhere rather than sitting exactly on its own fixed point.
   *
   * A hash of the id and not a multiple of it. `id * goldenAngle` was the
   * first try and it is not a scatter at all: consecutive ids land a constant
   * angle apart, so a chain latched in id order was already a wave of a
   * wavelength nothing chose. Mixed rather than drawn from `Math.random`, so
   * where a body starts is a property of the body and not of how many bodies
   * happened to be made before it.
   */
  let mix = Math.imul(id, 2654435761) >>> 0;
  mix ^= mix >>> 15;
  mix = Math.imul(mix, 2246822519) >>> 0;
  mix ^= mix >>> 13;
  mix = Math.imul(mix, 3266489917) >>> 0;
  mix ^= mix >>> 16;
  const traits = seedTraits(kind, params);
  store.intake[slot] = traits.intake;
  /*
   * A scattered start for the reactor, so two fresh bodies are not at the
   * same point of the same cycle. Only C is scattered: A settles to the fuel
   * it is given within a second whatever it starts at, and B and D follow C
   * round the loop. Seeded off the id hash rather than `Math.random`, so
   * where a body starts is a property of the body and not of how many were
   * made before it.
   */
  const r = slot * REACT_SPECIES;
  store.react[r + REACT_B] = 1;
  store.react[r + REACT_C] = 0.2 + (mix / 4294967296) * 2;
  store.react[r + REACT_D] = 2;
  // A body made outside a rewrite is a founder: generation zero of its own
  // line. `autoSpawn` makes a great many of these, which is the point of
  // being able to count them.
  agent.born = 0;
  agent.lineage = id;
  agent.bound = 0;
  refreshReadsField(agent);
  agent.stun = 0;
  agent.drive = params.stepSpeed;
  agent.trail = 0;
  agent.prevX = x;
  agent.prevY = y;
  agent.prevHeading = heading;
  agent.csHeading = NaN;
  agent.csCos = 1;
  agent.csSin = 0;
  agent.chem.set(seedChem(kind, params));
  agent.extra = traits.extra;
  agent.request = 0;
  // Phenotype, seeded so frame zero is right; `updateState` rewrites it from
  // `F` and `f0` every frame after that. The locomotion head gets the same
  // treatment: without it a body's first frame had a cruise of zero and a turn
  // gain of zero, so a fresh soup stood still for one tick and a newborn could
  // not steer until the state pass had run once.
  agent.flockAlign = params.flockAlign;
  agent.flockSep = params.flockSep;
  store.cruise[slot] = params.stepSpeed;
  store.turn[slot] = params.turnRate;
  agent.recovering = false;
  agent.requestDecay = traits.requestDecay;
  agent.energyCap = traits.energyCap;
  agent.debtCap = traits.debtCap;
  agent.rescueTo = traits.rescueTo;
  agent.assort = traits.assort;
  agent.transportThrust = params.transportThrust;
  agent.transportRecoil = params.transportRecoil;
  agent.transportQuantum = params.transportQuantum;
  return agent;
}

/**
 * The heritable scalars a fresh body of this kind carries.
 *
 * One table, read by the two places that need it: `createAgent`, and the net
 * migration when it meets a blob written before a scalar existed. It used to
 * live only in `createAgent`, so the migration had nothing to seed from and
 * a missing scalar arrived as whatever the decoder happened to read — see
 * `storedScalars` in `pond/net-blob.ts` for what that cost.
 *
 * `extra` is here because the blob carries it, and it is zero: a planted body
 * is given the tank the blob recorded, and a body whose blob did not record
 * one starts empty rather than fed.
 */
export interface SeededTraits {
  extra: number;
  requestDecay: number;
  energyCap: number;
  debtCap: number;
  rescueTo: number;
  assort: number;
  intake: number;
}

export function seedTraits(kind: AgentKind, params: Params): SeededTraits {
  return {
    extra: 0,
    requestDecay: params.requestDecay,
    energyCap: extraCapFor(kind, params.eraCapRatio),
    debtCap: params.debtCap,
    rescueTo: params.rescueTo,
    assort: params.assortBias,
    intake: params.intake,
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

const portWorldScratch: Vec2 = { x: 0, y: 0 };

/**
 * `portWorld` writing into `out`, and flattened for the same reason
 * `stemOffsetInto` is.
 *
 * The allocating chain is four objects a call: `stemRoot` makes one,
 * `portLocal` copies it into a second, `rotate` returns a third, and
 * `portWorld` builds the result. The GPU deposit pack walks every port of
 * every body, so on a pond of five thousand that is sixty thousand
 * short-lived objects a frame — measured at 5.95 ms, which was the largest
 * single piece of the field phase and none of it arithmetic.
 *
 * `wrap` is the identity — the world stopped being toroidal — so it is gone
 * here rather than sitting in the hot path hoping the JIT removes it. `w` and
 * `h` stay in the signature so the shape is obvious if wrapping comes back,
 * which is the convention `stemWorldInto` already set.
 */
export function portWorldInto(
  agent: Agent,
  slot: PortSlot,
  w: number,
  h: number,
  out: Vec2,
): Vec2 {
  const root = stemRootInto(agent.kind, slot, portWorldScratch);
  const ex = agent.kind === 'era' || slot === 'p' ? PORT_EXTRUDE : -PORT_EXTRUDE;
  const scale = agent.scale;
  const lx = (root.x + ex) * scale;
  const ly = root.y * scale;
  const heading = agent.heading;
  const c = Math.cos(heading);
  const sn = Math.sin(heading);
  void w;
  void h;
  out.x = agent.x + lx * c - ly * sn;
  out.y = agent.y + lx * sn + ly * c;
  return out;
}

/**
 * Bring the store's memoised cosine and sine of a body's heading up to date.
 *
 * The caller then reads `csCos[slot]` and `csSin[slot]`. Passing the three
 * arrays rather than a store keeps this off the flyweight, since every caller
 * is a loop over every body that has already hoisted what it needs.
 *
 * `stemOffsetInto` has always kept this memo, through the accessors. The point
 * of sharing it is that a heading turns into a sine and a cosine once a frame
 * however many passes want it: the latch pass computes them for every free
 * port, and the GPU probe pack wants them again a few phases later for the
 * same bodies at the same headings.
 *
 * A body whose heading has moved since the memo was written simply misses and
 * recomputes, so the answer is never stale — that is what `csHeading` is for.
 */
export function syncHeadingCosSin(
  csHeading: Float64Array,
  csCos: Float64Array,
  csSin: Float64Array,
  slot: number,
  heading: number,
): void {
  if (csHeading[slot] === heading) return;
  csHeading[slot] = heading;
  csCos[slot] = Math.cos(heading);
  csSin[slot] = Math.sin(heading);
}

/** A port's world position and the outward axis the snap arc is measured from. */
export interface PortFrame {
  x: number;
  y: number;
  ax: number;
  ay: number;
}

/**
 * `portWorldInto`'s position and the snap arc's outward axis at once, from a
 * heading the caller has already turned into a sine and a cosine.
 *
 * Both used to take a heading and call `Math.cos`/`Math.sin` on it: the
 * position once per free port, and the arc test (a deleted `inSnapArcAt`,
 * now inline in `Graph.snap`) twice per candidate pair, always on a body's
 * one heading. At fifty thousand bodies that came to about a hundred and
 * eighty thousand sine-cosine pairs a frame for fifty thousand distinct
 * angles, plus a flyweight property load each time to fetch the angle again.
 * The caller now turns each body's heading once and every port of that body
 * reads it.
 *
 * The axis is a unit vector because sine and cosine are, so the arc test
 * divides by the distance alone.
 */
export function portFrameInto(
  kind: AgentKind,
  slot: PortSlot,
  x: number,
  y: number,
  scale: number,
  cos: number,
  sin: number,
  out: PortFrame,
): void {
  const root = stemRootInto(kind, slot, portWorldScratch);
  // One condition, used twice: which way the stem points out of the body.
  const outward = kind === 'era' || slot === 'p';
  const ex = outward ? PORT_EXTRUDE : -PORT_EXTRUDE;
  const lx = (root.x + ex) * scale;
  const ly = root.y * scale;
  out.x = x + lx * cos - ly * sin;
  out.y = y + lx * sin + ly * cos;
  const sign = outward ? 1 : -1;
  out.ax = sign * cos;
  out.ay = sign * sin;
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
