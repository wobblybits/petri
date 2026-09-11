import { extraCapFor } from './energy.ts';
import {
  CHEM_LEN,
  CHEM_SPECIES,
  EMIT,
  SEED_PRODUCTION,
  SEED_UPTAKE,
  E_OUT,
  F_BASE,
  G_BASE,
  HEAD_SCALE,
  KS_BASE,
  L_BASE,
  IN_DEMAND,
  IN_DIMS,
  IN_SENSE,
  P_BASE,
  ROW_COUNT,
  ROW_EXCRETE,
  ROW_UPTAKE,
  STATE_DIMS,
  TASTE,
  T_OUT,
  W_IN,
  X_BASE,
  X_OUT,
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
 * and writing one slot of a shared set of typed arrays. The accessors live
 * on `Agent.prototype`, never per instance: own accessor properties push V8
 * into dictionary-mode storage. Consequently `{ ...agent }` copies nothing
 * but `store` and `slot`; use `cloneAgent`.
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
   * Energy on top of existence: positive is stock, negative is debt, and
   * `debtCap` is death. Upkeep decrements this directly.
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
   * velocity, and how hard it pushes off them when they crowd. A pair uses
   * the mean of the two: the force must stay equal and opposite, or momentum
   * is not conserved.
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
   * The genome, laid out by `chem-layout.ts`: what this body says, what it
   * listens for, and the heads that read its inner state. Seeded from kind,
   * inherited with mutation at a commute. Once the channels drift they stop
   * meaning con/dup/era/aux; aux ports still lay into channel 3 regardless.
   * A live view into `AgentStore.chemAll`, re-sliced only when the store's
   * `generation` changes. Always index-written, never reassigned.
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
   * Read by nothing the sim does; exists to be measured.
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

  /** This body's recurrent state, a live view into `AgentStore.hAll`, cached like `chem`. */
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
   * Scaled at the write on both field paths. On the CPU path it is only
   * refreshed on frames where this body's `Wx` reads the field.
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
   * Set when the body falls into debt, cleared once it is back at its
   * `rescueTo` fill. The latch lets the ask outlive the debt, so a rescue
   * tops a body up to something it can act with instead of parking it at 0.
   */
  get recovering(): boolean {
    return this.store.recovering[this.slot] !== 0;
  }
  set recovering(v: boolean) {
    this.store.recovering[this.slot] = v ? 1 : 0;
  }

  /**
   * Heritable traits. Seeded from the matching global slider when a body is
   * created outside a rewrite; a Con+Dup commute recombines both parents'
   * values into each child (`inheritTraits` in rewrite.ts), which is the
   * only place a population's traits can drift.
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
   * How particulate this lineage's inheritance is, 0 to 1: read when the body
   * breeds, per gene, to decide whether a child copies one parent whole or
   * blends the two. See `assortChance`.
   */
  get assort(): number {
    return this.store.assort[this.slot];
  }
  set assort(v: number) {
    this.store.assort[this.slot] = v;
  }

  /** The charged part of this body's adenylate pool. See `Sim.advanceGait`. */
  get atp(): number {
    return this.store.atp[this.slot];
  }
  set atp(v: number) {
    this.store.atp[this.slot] = v;
  }

  /**
   * How much adenylate this body carries: its working capital, and the
   * ceiling on how much work it can have outstanding at once. Heritable.
   */
  get adenylate(): number {
    return this.store.adenylate[this.slot];
  }
  set adenylate(v: number) {
    this.store.adenylate[this.slot] = v;
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
   * at. The guard is exact and self-invalidating: heading moves, the memo
   * misses. NaN starts it cold and keeps it cold if a heading ever goes bad.
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
 * A `{ ...agent }` spread would alias the original's slot instead.
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
  // Without these a clone has someone else's genome and a blank mind.
  clone.h.set(a.h);
  clone.sense.set(a.sense);
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

/** `poseHeld` from a store and a slot, for packs that walk every body. */
export function poseHeldAt(store: AgentStore, slot: number): boolean {
  return store.locked[slot] !== 0 || store.pinned[slot] !== 0;
}

/** Slot as a small integer: principal 0, left 1, right 2. */
export function slotIndex(slot: PortSlot): number {
  return slot === 'p' ? 0 : slot === 'l' ? 1 : 2;
}

/** A port's identity as a number: `id * 3 + slotIndex`. */
export function portKeyAt(id: number, slot: PortSlot): number {
  return id * 3 + slotIndex(slot);
}

export function portKey(p: PortRef): number {
  return portKeyAt(p.id, p.slot);
}

/*
 * Frozen and shared, for callers that only iterate; `slotsFor` stays for
 * the ones that want a list of their own.
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

/**
 * Radius of the disc with the same area as the Con/Dup triangle, as a
 * multiple of `agentSize`: the glyph is 1.312 s^2 for s = 16 * scale.
 */
export const TRI_DISC_RATIO = Math.sqrt(1.312 / Math.PI);

/**
 * Contact radius for the tiers that collide discs instead of SAT polygons.
 * Equal area, not the circumscribed `boundRadius`, so a net does not inflate
 * when the camera crosses the LOD line.
 */
export function discRadius(agent: Agent): number {
  if (agent.kind === 'era') return ERA_RADIUS * agent.scale;
  return agentSize(agent.kind) * TRI_DISC_RATIO * agent.scale;
}

/**
 * Sum of squared vertex radii of the unit Con/Dup triangle (`triangleLocal`
 * at `s = 1`): `1.05^2 + 2 * (0.55^2 + 0.82^2)`.
 */
const TRI_VERTEX_R2 = 1.05 * 1.05 + 2 * (0.55 * 0.55 + 0.82 * 0.82);

export function momentOfInertia(agent: Agent): number {
  return momentOfInertiaAt(agent.store.kindCode[agent.slot], agent.mass, agent.scale);
}

/** `momentOfInertia` from a kind code rather than a body, for the packs. */
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
  ROW_COUNT,
  ROW_EXCRETE,
  ROW_UPTAKE,
  uptakeKsOf,
  X_BASE,
  X_OUT,
  SENSE_SCALE,
  STATE_DIMS,
  TASTE,
  T_OUT,
  W_IN,
  W_NET,
  W_SELF,
} from './chem-layout.ts';



/**
 * A bare body carrying just what `effEmit` and `effTaste` read, for tests
 * that want to poke a genome without building a `Sim`. The one place the
 * cast lives, so a shape change breaks the build here rather than
 * surfacing as `undefined` arithmetic elsewhere.
 */
export function bareBody(chem: Float32Array, over: { h?: number[] } = {}): Agent {
  return {
    chem,
    h: Float64Array.from(over.h ?? new Array(STATE_DIMS).fill(0)),
    sense: new Float64Array(4),
  } as unknown as Agent;
}



/**
 * Recompute the cached "does this genome look at the field" flag. Must be
 * called after anything writes `chem`: a stale `false` is a body that has
 * evolved sense weights and cannot see.
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
  // The one choke point every chem write passes through, so the GPU's copy learns it is stale here.
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
 * The realised emit vector: `relu(E.h + e0)` per channel, then normalised to
 * one unit across all four, into `out`. The unit sum is the budget that
 * keeps "louder is strictly better" unreachable, and it must be enforced on
 * the realised vector, which is why this is a vector and not four scalar
 * calls. A body that has mutated its way to silence stays silent.
 */
export function emitVector(chem: Float32Array, g: number, h: Float64Array, ho: number, out: Float64Array, oo: number): void {
  // Unrolled: four channels by four dimensions is a compile-time shape.
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

  if (sum > 1e-6) {
    const inv = 1 / sum;
    for (let c = 0; c < 4; c++) out[oo + c] *= inv;
  }
}

/**
 * What a kind's metabolism makes, and what it can eat, at the seed. A Con
 * makes `conP` and `aux`, a Dup `dupP` and `aux`, an Era ground; every
 * uptake row lands on an eighth, the flat fallback's own value, so a fresh
 * body eats like a generalist and speaks like its kind. `SEED_UPTAKE` and
 * `SEED_PRODUCTION` are chosen so the production half is `ERA_GROUND_SHARE`
 * of the budget. Inherited and mutated, not learned: `X` sits outside the
 * plastic span.
 */
function seedProduction(c: Float32Array, kind: AgentKind): void {
  const x = X_BASE + ROW_EXCRETE;
  if (kind === 'con') {
    c[x + CH.conP] = SEED_PRODUCTION / 2;
    c[x + CH.aux] = SEED_PRODUCTION / 2;
  } else if (kind === 'dup') {
    c[x + CH.dupP] = SEED_PRODUCTION / 2;
    c[x + CH.aux] = SEED_PRODUCTION / 2;
  } else {
    c[x + CH.energy] = SEED_PRODUCTION;
  }
  const u = X_BASE + ROW_UPTAKE;
  for (let i = 0; i < CHEM_SPECIES; i++) c[u + i] = SEED_UPTAKE;
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
 * Emit weight for one signal channel. Never negative. Always zero on
 * `CH.energy`: the ground reaches the field only through its excretion row,
 * never through the scent deposit path, which multiplies by `params.deposit`
 * and would mint food. The emit head's ground slot stays a quarter of the
 * simplex because dropping it would renormalise every genome in the library.
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
 * A flocking gain as the force sees it: never negative. The genes may go
 * below zero so mutation has no reflecting barrier at off, but a negative
 * alignment gain is negative damping and a negative separation gain has no
 * equilibrium; both blow up.
 */
export function flockGain(v: number): number {
  return v > 0 ? v : 0;
}



/**
 * How hard each kind grips at its point in the stroke: an Era is an oar and
 * a Con or a Dup is a foot. A base, not a matrix entry, so a fresh body grips
 * by a constant and a lineage may make it depend on `h` by drifting `G`. The
 * two ends of a wire must grip differently or the swing moves nothing, and
 * an Era supplies that difference. Both under `drag`: past it the rate
 * clamps at zero on both ends together and the asymmetry is gone.
 */
const GAIT_ANCHOR_ERA = 0.02;
const GAIT_ANCHOR_NODE = 0.25;

function seedGait(c: Float32Array, kind: AgentKind): void {
  c[G_BASE] = kind === 'era' ? GAIT_ANCHOR_ERA : GAIT_ANCHOR_NODE;
}

/**
 * A kind's seed genome. Emit is one-hot on the kind's channel. Taste: Con
 * seeks Dup, Dup seeks Con, the pairing that makes a redex, and Era seeks
 * both strongly; everyone is mildly drawn to the aux channel, "unwired
 * tissue here". The attract sliders seed a new body and are not read again.
 * Slopes start at zero, so modulation is inert until breeding moves it.
 */
export function seedChem(kind: AgentKind, params: Params): Float32Array {
  const c = new Float32Array(CHEM_LEN);
  const S = params.attractStrong;
  const M = params.attractMedium;
  /*
   * The one place a matrix is seeded away from zero: a two-hop pathway,
   * `h[0]` carries `DEMAND` and taste for the ground reads `h[0]`, so a
   * fresh body is drawn to food exactly when its neighbourhood is short of
   * energy and blind to it otherwise. Blind when fed is load-bearing: a body
   * harvests the cell it stands in and would otherwise chase the dip it just
   * ate. `phi` compresses [0,1] to [0,0.5], so the gain is doubled.
   */
  c[W_IN + 0 * IN_DIMS + IN_DEMAND] = 1;
  c[T_OUT + CH.energy * STATE_DIMS + 0] = params.attractFood * 2;
  // Output-head bases: the sliders' values.
  c[F_BASE] = params.flockAlign / HEAD_SCALE.align;
  c[F_BASE + 1] = params.flockSep / HEAD_SCALE.sep;
  c[P_BASE] = params.transportRecoil / HEAD_SCALE.recoil;
  c[L_BASE] = params.stepSpeed / HEAD_SCALE.cruise;
  c[L_BASE + 1] = params.turnRate / HEAD_SCALE.turn;
  seedGait(c, kind);
  seedProduction(c, kind);
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
     * An Era's one unit of voice goes into the ground slot, which `effEmit`
     * never reads: `seedProduction` is where an Era is told to make ground.
     * Kept because it is a quarter of the simplex and says what an Era is for.
     */
    c[EMIT + CH.energy] = 1;
    c[TASTE] = S;
    c[TASTE + 1] = S;
    c[TASTE + 3] = M;
  }
  // The affinity genes, at one natural unit each: a fresh body's uptake
  // reads `params.uptakeKs` on every species. See `chem-layout.ts`.
  for (let k = 0; k < CHEM_SPECIES; k++) c[KS_BASE + k] = 1;
  return c;
}

/**
 * Expression: how a body divides one unit of chemical effort across the
 * eight rows of the reaction table. Relu, then normalised to a unit sum
 * across the whole table, like `emitVector`: the simplex is the trade-off,
 * a body cannot both shout and eat without giving something up. Seeded flat
 * at zero, which comes out as an even eighth each.
 */
export function expressVector(
  chem: Float32Array,
  g: number,
  h: Float64Array,
  ho: number,
  out: Float64Array,
  oo: number,
): void {
  const h0 = h[ho];
  const h1 = h[ho + 1];
  const h2 = h[ho + 2];
  const h3 = h[ho + 3];
  let sum = 0;
  for (let r = 0; r < ROW_COUNT; r++) {
    const o = g + X_OUT + r * STATE_DIMS;
    const v =
      chem[g + X_BASE + r] +
      chem[o] * h0 + chem[o + 1] * h1 + chem[o + 2] * h2 + chem[o + 3] * h3;
    const w = v > 0 ? v : 0;
    out[oo + r] = w;
    sum += w;
  }
  if (sum > 0) {
    const inv = 1 / sum;
    for (let r = 0; r < ROW_COUNT; r++) out[oo + r] *= inv;
  } else {
    // Nothing expressed. Flat rather than zero: a zero-seeded `X` must not
    // make a body inert from birth.
    const even = 1 / ROW_COUNT;
    for (let r = 0; r < ROW_COUNT; r++) out[oo + r] = even;
  }
}



/**
 * A founder body, seeded from the sliders. `store` defaults to a fresh,
 * private, single-agent `AgentStore`; `Sim` and `commitRewrite` pass their
 * own shared store explicitly.
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
   * A nudge off the metabolic steady state, so a fresh body's pathway does
   * not sit on its own fixed point. A hash of the id, not a multiple of it
   * (consecutive ids would land a constant phase apart) and not
   * `Math.random` (a body's start is a property of the body).
   */
  let mix = Math.imul(id, 2654435761) >>> 0;
  mix ^= mix >>> 15;
  mix = Math.imul(mix, 2246822519) >>> 0;
  mix ^= mix >>> 13;
  mix = Math.imul(mix, 3266489917) >>> 0;
  mix ^= mix >>> 16;
  const pool = params.adenylate;
  store.adenylate[slot] = pool;
  // Part charged, never all: a full pool has no ADP for the autocatalytic step.
  store.atp[slot] = pool * (0.25 + (mix / 4294967296) * 0.5);
  store.sub[slot] = 0.5;
  // A body made outside a rewrite is a founder: generation zero of its own line.
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
  agent.extra = 0;
  agent.request = 0;
  // Phenotype, seeded so frame zero is right; `updateState` rewrites it every
  // frame after that. Without the seed a newborn stands still for one tick.
  agent.flockAlign = params.flockAlign;
  agent.flockSep = params.flockSep;
  store.cruise[slot] = params.stepSpeed;
  store.turn[slot] = params.turnRate;
  agent.recovering = false;
  agent.requestDecay = params.requestDecay;
  agent.energyCap = extraCapFor(kind, params.eraCapRatio);
  agent.debtCap = params.debtCap;
  agent.rescueTo = params.rescueTo;
  agent.assort = params.assortBias;
  agent.transportRecoil = params.transportRecoil;
  agent.transportQuantum = params.transportQuantum;
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

const portWorldScratch: Vec2 = { x: 0, y: 0 };

/**
 * `portWorld` writing into `out`, flattened like `stemOffsetInto`. `wrap` is
 * the identity and is omitted; `w` and `h` stay in the signature so the
 * shape is obvious if wrapping comes back.
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
 * Bring the store's memoised cosine and sine of a body's heading up to date;
 * the caller then reads `csCos[slot]` and `csSin[slot]`. The same memo
 * `stemOffsetInto` keeps through the accessors. A heading that has moved
 * misses and recomputes, so the answer is never stale.
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
 * heading the caller has already turned into a sine and a cosine. The axis
 * is a unit vector, so the arc test divides by the distance alone.
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
 * `stemOffset` writing into `out`. The hottest geometric routine in the sim,
 * so `stemRootInto` and `agentSize` are inlined here in the same order, and
 * the result is bit-for-bit what that chain produces.
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
  const o = stemOffsetInto(agent, slot, stemWorldScratch);
  // `wrap` is the identity and is omitted; `w` and `h` stay in the signature.
  void w;
  void h;
  out.x = agent.x + o.x;
  out.y = agent.y + o.y;
  return out;
}

/**
 * Control point along the port axis, scaled to the wire's length when known:
 * a handle longer than a third of the span crosses its partner and the cubic
 * doubles back on itself.
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
  // A handle longer than a third of the span crosses its partner; see `handleWorld`.
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
