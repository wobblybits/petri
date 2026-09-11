import type { Agent, AgentKind } from './agents.ts';
import { CH, CHANNELS, FIELD_CELL, type Fields } from './fields.ts';
import { CHEM_LEN, CHEM_SPECIES, ERA_GROUND_SHARE, ROW_COUNT, ROW_EXCRETE, uptakeKsOf } from './chem-layout.ts';
import { CODE_KIND, type AgentStore } from './agent-store.ts';
import { KIND_ERA } from './native/solver.ts';
import type { Rule } from './rewrite.ts';

/**
 * `extra` is the one energy scalar a body carries: positive is stock,
 * negative is debt, and a whole unit of debt is death. Upkeep decrements it
 * directly; there is no second accumulator.
 */

/** What one end of a rewrite pays; a Con–Dup commute costs one from each side. */
export const REWRITE_SHARE = 1;

/**
 * The most a Con or Dup can hold; an Era holds `ERA_CAP_RATIO` times it.
 * Strictly more than `REWRITE_SHARE`, so a full body keeps headroom to
 * commute while upkeep drains it. Transport obeys the cap too.
 */
export const EXTRA_CAP = 1.25;

/**
 * How much more an Era holds than a Con or a Dup. An Era never spends, so
 * it is a battery: a net's reserve parks on its boundary. `BODY_VALUE` is
 * flat across kinds, so this buys storage, not a mint.
 */
export const ERA_CAP_RATIO = 2;

/**
 * Seed cap for a body born outside a rewrite. The live ceiling is the
 * heritable `energyCap`, recombined from both parents across a commute.
 */
export function extraCapFor(kind: AgentKind, eraRatio = ERA_CAP_RATIO): number {
  return kind === 'era' ? EXTRA_CAP * eraRatio : EXTRA_CAP;
}
/**
 * Seed death floor: a whole unit of debt. The live floor is the heritable
 * `debtCap`, which must stay strictly negative or the rescue latch never fires.
 */
export const EXTRA_FLOOR = -1;
/** Closest to zero a heritable `debtCap` may sit. Strictly negative. */
export const DEBT_CAP_MAX = -0.05;
export const EXTRA_FULL_EPS = 1e-6;

/**
 * One deferred conserved add: a world position, then a weight per species.
 * The GPU owns the field, so host adds cross as a list that `sim.ts` packs
 * into the shader's `Deposit` records with `conserve` set.
 */
export const PENDING_STRIDE = 2 + CHANNELS;

/**
 * What a body's existence is worth: a full tank. More than the
 * `REWRITE_SHARE` that built it, so commute-then-annihilate mints
 * `2 * (EXTRA_CAP − REWRITE_SHARE)` and a net that keeps rewriting feeds
 * itself. Set to `REWRITE_SHARE` for a strictly conserved pond.
 */
export const BODY_VALUE = EXTRA_CAP;

/**
 * Energy released by one death: the body's own worth plus what it held.
 * `value` is `params.bodyValue`; at `REWRITE_SHARE` the pond is conserved.
 */
export function deathYield(a: { extra: number }, value = BODY_VALUE): number {
  return Math.max(0, value + a.extra);
}

/** Energy locked up in one body's existence. Every kind costs the same. */
export const AGENT_VALUE: Record<AgentKind, number> = {
  era: BODY_VALUE,
  con: BODY_VALUE,
  dup: BODY_VALUE,
};

/**
 * How much of a neighbour's need reaches you, per hop: the fallback for a
 * body with no `requestDecay` of its own, and the seed for a fresh body's
 * trait. The field is `max(own need, best neighbour's field x decay)`, so
 * need spreads as a decaying scent and two comparable needs meet at a flat
 * watershed nothing crosses. Against `REQUEST_FLOOR` it sets how far a
 * shortage is audible: a whole unit carries 20 hops at 0.8 and 43 at 0.9.
 * Never 1: at 1 every body in a net holds the same need and transport stops.
 */
export const REQUEST_DECAY = 0.9;

/**
 * Field values below this are not carried further. Bounds the relaxation;
 * not a minimum transfer size, since a stalled redex is often one frame of
 * upkeep short.
 */
export const REQUEST_FLOOR = 0.01;

/** Smallest transfer worth doing. Guards against denormal churn, nothing more. */
const FLOW_EPS = 1e-9;

/** Field value at which the request ring is drawn at full strength. */
export const REQUEST_FULL = 1;

/**
 * Era's upkeep as a fraction of everyone else's, negative because an Era
 * produces rather than pays: a net's leaves are its income. Set to 0 for
 * Eras that are simply free.
 */
export const ERA_UPKEEP_RATIO = -0.2;

/** Upkeep per second for one body. `eraRatio` is `params.eraUpkeepRatio`; at 1 an Era pays rent. */
export function upkeepRateFor(kind: AgentKind, rate: number, eraRatio = ERA_UPKEEP_RATIO): number {
  return kind === 'era' ? rate * eraRatio : rate;
}

/**
 * The same, keyed on what a body expresses rather than on its glyph: the
 * discount interpolates on the ground share of the body's excretion row
 * against `ERA_GROUND_SHARE`. Exact at both ends — a seeded Era lands on
 * `eraRatio` bit for bit and a seeded Con on 1, which
 * `1 + (eraRatio - 1) * t` does not manage in floating point. Only
 * meaningful when the expression vectors are live (`UpkeepOptions.expressed`).
 */
export function upkeepRateOf(express: Float64Array, slot: number, rate: number, eraRatio: number): number {
  const share = express[slot * ROW_COUNT + ROW_EXCRETE + CH.energy] / ERA_GROUND_SHARE;
  if (share >= 1) return rate * eraRatio;
  if (!(share > 0)) return rate;
  return rate * ((1 - share) + share * eraRatio);
}

export function agentValue(kind: AgentKind): number {
  return AGENT_VALUE[kind];
}


/** Holding as much as it can. Nothing more can be harvested or pumped in. */
export function atCap(a: { extra: number; energyCap: number }): boolean {
  return a.extra >= a.energyCap - EXTRA_FULL_EPS;
}

/**
 * What this end owes toward a rewrite: the share, or everything it can hold,
 * whichever is smaller. A heritable `energyCap` can sit below `REWRITE_SHARE`,
 * and a body must never owe a debt it is physically unable to settle.
 */
export function rewriteShareOf(a: { energyCap: number }): number {
  return Math.min(REWRITE_SHARE, a.energyCap);
}

/** Has this end's stake been met? `paid` is what the escrow already holds for it. */
export function stakeMet(a: { energyCap: number }, paid: number): boolean {
  return paid >= rewriteShareOf(a) - EXTRA_FULL_EPS;
}

/** Able to pay its side of a rewrite outright. Not the same as being full. */
export function canPayShare(a: { extra: number; energyCap: number }): boolean {
  return stakeMet(a, a.extra);
}

/**
 * Move up to `want` out of a body and report what actually moved. Bounded by
 * `spareEnergy`: a body in debt contributes nothing, and contributing never
 * puts one into debt.
 */
export function payToward(a: SlotBody, want: number): number {
  if (!(want > 0)) return 0;
  const spare = spareEnergy(a);
  const give = want < spare ? want : spare;
  if (give <= 0) return 0;
  a.extra -= give;
  return give;
}

export function spendExtra(a: { extra: number }): void {
  a.extra = Math.max(0, a.extra - REWRITE_SHARE);
}

/** Bodies destroyed minus bodies created. */
export function bodyDelta(rule: Rule): number {
  switch (rule) {
    case 'era-era':
    case 'annihilate-con':
    case 'annihilate-dup':
      return 2;
    case 'erase':
      return 0;
    case 'commute':
      return -2;
  }
}

/**
 * What the pair pays up front: one share per end. Charged at the share
 * rather than `BODY_VALUE` so a commute does not need both ends exactly at cap.
 */
export function rewriteCost(rule: Rule): number {
  const d = bodyDelta(rule);
  return d < 0 ? -d * REWRITE_SHARE : 0;
}

/** Existence released when the rewrite commits, before the dying bodies' own stock. */
export function rewriteYield(rule: Rule, value = BODY_VALUE): number {
  const d = bodyDelta(rule);
  return d > 0 ? d * value : 0;
}

/*
 * Cell key packed into one integer: `CELL_KEY_OFFSET` shifts i/j positive
 * and `CELL_KEY_WIDTH` is `2 * CELL_KEY_OFFSET`, so rows never overlap. The
 * product stays under 2^52, inside float64's safe-integer range.
 */
const CELL_KEY_OFFSET = 1 << 25;
const CELL_KEY_WIDTH = CELL_KEY_OFFSET * 2;

function cellKey(i: number, j: number): number {
  return (i + CELL_KEY_OFFSET) * CELL_KEY_WIDTH + (j + CELL_KEY_OFFSET);
}

/**
 * Sparse world-space energy. Unvisited cells hold `ambient`; once touched,
 * whatever remains is stored explicitly (including 0). No decay or diffusion
 * on the sparse path. `inexhaustible` keeps every in-bounds cell at
 * `ambient` and makes `take` a read.
 */
export class EnergyGrid {
  cellSize: number;
  ambient: number;
  inexhaustible = false;
  private readonly cells = new Map<number, number>();

  /*
   * Where the energy lives. Bound to a field, a cell of this grid is a block
   * of field cells on `CH.energy` and this class is a coarse view onto them;
   * unbound it keeps the sparse map. The view stays coarse because
   * `harvestSlots` groups bodies by cell, and that grouping is the economy's
   * only crowding pressure.
   */
  private fields: Fields | null = null;

  /*
   * The world bound, a disk tracked to `home`. Outside it there is no ground:
   * nothing can be harvested and nothing deposited. The same disk is the
   * scent mask and the hard wall. Stored cells outside the bound are kept,
   * not pruned; dropping them would destroy energy.
   */
  private boundX = 0;
  private boundY = 0;
  private boundHalf = Infinity;
  /*
   * The lattice this grid is cut on, shared with the scent field: the field's
   * own cell-snapped origin, so an energy cell is an exact block of scent
   * cells provided `energyCell` stays a whole multiple of `FIELD_CELL`. Set
   * once, when the world is pinned; moving it would re-key every stored cell.
   */
  private originX = 0;
  private originY = 0;

  constructor(cellSize: number, ambient: number) {
    this.cellSize = Math.max(1, cellSize);
    this.ambient = Math.max(0, ambient);
  }

  /** Move storage onto `fields`, channel `CH.energy`. */
  bind(fields: Fields): void {
    this.fields = fields;
    this.cells.clear();
  }

  /*
   * Queued adds, for when the field is on the GPU and `fields.data` is a
   * stale copy: handed to the shader's `scatter` at the end of the frame.
   * Grown, never shrunk, and refilled from zero each frame.
   */
  private pending = new Float64Array(0);
  private nPending = 0;
  private deferring = false;

  /** Queue `addAt` rather than writing it. Set once, when the field moves. */
  deferAdds(on: boolean): void {
    this.deferring = on;
    this.nPending = 0;
  }

  get pendingAdds(): number {
    return this.nPending;
  }

  /** `pendingAdds` records of `PENDING_STRIDE`: x, y, then one weight per species. */
  get pendingData(): Float64Array {
    return this.pending;
  }

  clearPending(): void {
    this.nPending = 0;
  }

  /** Field cells per energy cell, along one axis. */
  get span(): number {
    return Math.max(1, Math.round(this.cellSize / FIELD_CELL));
  }

  /**
   * What one field cell holds when the ground is full, so that a whole
   * energy cell still holds `ambient`.
   */
  get cellCap(): number {
    const s = this.span;
    return this.ambient / (s * s);
  }

  /**
   * Lay down full ground across the disk. Field-backed only. Deferred like
   * `addAt`; `Sim.gpuFieldStep` picks the request up and dispatches `fill`.
   */
  seedGround(): void {
    if (this.deferring) {
      this.pendingSeed = this.cellCap;
      return;
    }
    this.fields?.fillDisk(CH.energy, this.cellCap);
  }

  /** The value a deferred `seedGround` asked for, or null. */
  pendingSeed: number | null = null;

  /**
   * The field cells under energy cell `(i, j)`, clipped to the grid; null
   * when the block falls outside it. Public because the GPU harvest is
   * handed this rect rather than deriving it.
   */
  blockRect(i: number, j: number): { fi: number; fj: number; wi: number; wj: number } | null {
    const b = this.rectScratch;
    if (!this.blockRectAt(i, j, b)) return null;
    return { fi: b[0], fj: b[1], wi: b[2], wj: b[3] };
  }

  private readonly rectScratch = new Int32Array(4);

  /** `blockRect` into a caller's four ints — `fi, fj, wi, wj` — without allocating. */
  blockRectAt(i: number, j: number, out: Int32Array): boolean {
    const f = this.fields;
    if (!f) return false;
    const s = this.span;
    let fi = i * s;
    let fj = j * s;
    let wi = s;
    let wj = s;
    if (fi < 0) {
      wi += fi;
      fi = 0;
    }
    if (fj < 0) {
      wj += fj;
      fj = 0;
    }
    if (fi + wi > f.cols) wi = f.cols - fi;
    if (fj + wj > f.rows) wj = f.rows - fj;
    if (wi <= 0 || wj <= 0) return false;
    out[0] = fi;
    out[1] = fj;
    out[2] = wi;
    out[3] = wj;
    return true;
  }

  /** Energy cells along one side of the field-backed grid; 0 when unbound. */
  get cellsPerSide(): number {
    const f = this.fields;
    return f ? Math.ceil(f.cols / this.span) : 0;
  }

  /**
   * Dense index of the energy cell under a point — `j * side + i` — or -1
   * off the field. `side` is `cellsPerSide`. What `HarvestPlan` bins on.
   */
  denseIndexAt(x: number, y: number, side: number): number {
    if (side <= 0) return -1;
    const i = Math.floor((x - this.originX) / this.cellSize);
    const j = Math.floor((y - this.originY) / this.cellSize);
    if (i < 0 || j < 0 || i >= side || j >= side) return -1;
    return j * side + i;
  }

  get lattice(): { x: number; y: number } {
    return { x: this.originX, y: this.originY };
  }

  /**
   * Centre and radius of the live disk, plus the shared lattice origin: pass
   * the scent field's own snapped `originX`/`originY` so the two line up.
   */
  setBounds(cx: number, cy: number, radius: number, originX = cx - radius, originY = cy - radius): void {
    this.boundX = cx;
    this.boundY = cy;
    this.boundHalf = radius;
    this.originX = originX;
    this.originY = originY;
  }

  /** Disk of radius `boundHalf` around the pinned centre. */
  inBounds(x: number, y: number): boolean {
    const dx = x - this.boundX;
    const dy = y - this.boundY;
    return dx * dx + dy * dy <= this.boundHalf * this.boundHalf;
  }

  clear(): void {
    this.cells.clear();
    // The deferred queue too, or this frame's deaths land on the next pond's ground.
    this.nPending = 0;
    this.pendingSeed = null;
    const f = this.fields;
    if (!f) return;
    const d = f.data;
    for (let k = CH.energy; k < d.length; k += CHANNELS) d[k] = 0;
  }

  configure(cellSize: number, ambient: number): void {
    this.cellSize = Math.max(1, cellSize);
    this.ambient = Math.max(0, ambient);
  }

  /*
   * Anchored to world coordinates, so a body at a given place always reads
   * the same energy cell however the field's window has scrolled.
   */
  index(x: number, y: number): { i: number; j: number; key: number } {
    const i = Math.floor((x - this.originX) / this.cellSize);
    const j = Math.floor((y - this.originY) / this.cellSize);
    return { i, j, key: cellKey(i, j) };
  }

  getCell(i: number, j: number): number {
    if (this.inexhaustible) return this.ambient;
    const f = this.fields;
    if (f) {
      const b = this.blockRect(i, j);
      if (!b) return 0;
      const d = f.data;
      let sum = 0;
      for (let y = 0; y < b.wj; y++) {
        let k = ((b.fj + y) * f.cols + b.fi) * CHANNELS + CH.energy;
        for (let x = 0; x < b.wi; x++, k += CHANNELS) sum += d[k];
      }
      return sum;
    }
    const key = cellKey(i, j);
    return this.cells.has(key) ? (this.cells.get(key) ?? 0) : this.ambient;
  }

  getAt(x: number, y: number): number {
    if (!this.inBounds(x, y)) return 0;
    const { i, j } = this.index(x, y);
    return this.getCell(i, j);
  }

  /**
   * Mean standing stock per cell across a key's block — the `S` a Monod rate
   * reads. The mean, not the total, so a rate constant means the same at any
   * block size and matches the shader's twin. Walks the same rect `take` walks.
   */
  density(key: number): number {
    return this.densityOf(key, CH.energy);
  }

  /** The same for any species. See `density`. */
  densityOf(key: number, ch: number): number {
    if (this.inexhaustible) return ch === CH.energy ? this.ambient : 0;
    const f = this.fields;
    if (f) {
      const i = Math.floor(key / CELL_KEY_WIDTH) - CELL_KEY_OFFSET;
      const j = key - (i + CELL_KEY_OFFSET) * CELL_KEY_WIDTH - CELL_KEY_OFFSET;
      const b = this.rectScratch;
      if (!this.blockRectAt(i, j, b)) return 0;
      const fi = b[0];
      const fj = b[1];
      const wi = b[2];
      const wj = b[3];
      const cells = wi * wj;
      if (cells <= 0) return 0;
      const d = f.data;
      let sum = 0;
      for (let y = 0; y < wj; y++) {
        let k = ((fj + y) * f.cols + fi) * CHANNELS + ch;
        for (let x = 0; x < wi; x++, k += CHANNELS) sum += d[k];
      }
      return sum / cells;
    }
    return ch === CH.energy && this.cells.has(key)
      ? (this.cells.get(key) ?? 0)
      : ch === CH.energy
        ? this.ambient
        : 0;
  }

  /** Take up to `n` of the ground from a cell, and return how much was taken. */
  take(key: number, n: number): number {
    return this.takeFrom(key, CH.energy, n);
  }

  /** The same for any species. */
  takeFrom(key: number, ch: number, n: number): number {
    let want = Math.max(0, n);
    if (this.inexhaustible) return Math.min(this.ambient, want);
    const f = this.fields;
    if (f) {
      // Cell by cell across the block rather than proportionally: grazing
      // leaves an uneven floor, which is what gives diffusion a gradient.
      // Key decoded and block clipped in place; this runs once a body a frame.
      const i = Math.floor(key / CELL_KEY_WIDTH) - CELL_KEY_OFFSET;
      const j = key - (i + CELL_KEY_OFFSET) * CELL_KEY_WIDTH - CELL_KEY_OFFSET;
      const b = this.rectScratch;
      if (!this.blockRectAt(i, j, b)) return 0;
      const fi = b[0];
      const fj = b[1];
      const wi = b[2];
      const wj = b[3];
      const d = f.data;
      let got = 0;
      for (let y = 0; y < wj && want > FLOW_EPS; y++) {
        let k = ((fj + y) * f.cols + fi) * CHANNELS + ch;
        for (let x = 0; x < wi && want > FLOW_EPS; x++, k += CHANNELS) {
          const have = d[k];
          if (have <= 0) continue;
          const g = have < want ? have : want;
          d[k] = have - g;
          got += g;
          want -= g;
        }
      }
      return got;
    }
    // The map-backed grid models only the ground.
    if (ch !== CH.energy) return 0;
    const have = this.cells.has(key) ? (this.cells.get(key) ?? 0) : this.ambient;
    const got = Math.min(have, want);
    this.cells.set(key, have - got);
    return got;
  }

  /**
   * A conserved add across every species at once. `addAt` is this with a
   * vector that is zero everywhere but `CH.energy`. Excretion needs the
   * vector, and the deposit must be the conserving one on all four channels
   * rather than the density the scent path lays down. See `Fields.addAt`.
   */
  addSpeciesAt(x: number, y: number, w: ArrayLike<number>): void {
    let any = false;
    for (let c = 0; c < CHANNELS; c++) {
      if (w[c] !== 0) {
        any = true;
        break;
      }
    }
    if (!any) return;
    if (!this.inBounds(x, y)) return;
    if (this.deferring) {
      const o = this.reservePending();
      this.pending[o] = x;
      this.pending[o + 1] = y;
      for (let c = 0; c < CHANNELS; c++) this.pending[o + 2 + c] = w[c];
      this.nPending++;
      return;
    }
    const f = this.fields;
    if (f) {
      for (let c = 0; c < CHANNELS; c++) if (w[c] !== 0) f.addAt(c, x, y, w[c]);
      return;
    }
    // The map-backed grid has only the ground.
    if (w[CH.energy] !== 0) this.addAt(x, y, w[CH.energy]);
  }

  /** Room for one pending record, growing the buffer if it is full. */
  private reservePending(): number {
    const o = this.nPending * PENDING_STRIDE;
    if (o + PENDING_STRIDE > this.pending.length) {
      const next = new Float64Array(Math.max(768, this.pending.length * 2));
      next.set(this.pending);
      this.pending = next;
    }
    return o;
  }

  addAt(x: number, y: number, amount: number): void {
    if (amount === 0) return;
    if (!this.inBounds(x, y)) return;
    if (this.deferring) {
      const o = this.reservePending();
      this.pending[o] = x;
      this.pending[o + 1] = y;
      for (let c = 0; c < CHANNELS; c++) this.pending[o + 2 + c] = c === CH.energy ? amount : 0;
      this.nPending++;
      return;
    }
    const f = this.fields;
    if (f) {
      // Into the one field cell it happened in, not spread across the block:
      // a corpse is a point event, and diffusion makes the plume.
      f.addAt(CH.energy, x, y, amount);
      return;
    }
    const { key } = this.index(x, y);
    this.cells.set(key, (this.cells.has(key) ? (this.cells.get(key) ?? 0) : this.ambient) + amount);
  }

  setCell(i: number, j: number, amount: number): void {
    const f = this.fields;
    if (f) {
      const b = this.blockRect(i, j);
      if (!b) return;
      // Spread evenly: the block is one cell as far as this API is concerned.
      const each = amount / (b.wi * b.wj);
      const d = f.data;
      for (let y = 0; y < b.wj; y++) {
        let k = ((b.fj + y) * f.cols + b.fi) * CHANNELS + CH.energy;
        for (let x = 0; x < b.wi; x++, k += CHANNELS) d[k] = each;
      }
      return;
    }
    this.cells.set(cellKey(i, j), amount);
  }

  /** Every unit of energy on the ground. */
  storedTotal(): number {
    const f = this.fields;
    if (f) {
      const d = f.data;
      let sum = 0;
      for (let k = CH.energy; k < d.length; k += CHANNELS) sum += d[k];
      return sum;
    }
    let s = 0;
    for (const v of this.cells.values()) s += v;
    return s;
  }

  /**
   * Touched cells only. Sparse path only, so it yields nothing when bound to
   * a field; kept as the one cover for the sparse path.
   */
  forEachStored(fn: (i: number, j: number, e: number) => void): void {
    for (const [key, e] of this.cells) {
      const i = Math.floor(key / CELL_KEY_WIDTH) - CELL_KEY_OFFSET;
      const j = (key - (i + CELL_KEY_OFFSET) * CELL_KEY_WIDTH) - CELL_KEY_OFFSET;
      fn(i, j, e);
    }
  }
}

export type SlotBody = Pick<
  Agent,
  | 'id'
  | 'kind'
  | 'x'
  | 'y'
  | 'extra'
  | 'locked'
  | 'request'
  | 'recovering'
  | 'requestDecay'
  | 'energyCap'
  | 'debtCap'
  | 'rescueTo'
  | 'transportQuantum'
>;

/**
 * Who eats from which block, and in what order. Shared by the CPU and GPU
 * harvests so they cannot compute it two ways. Both orderings are
 * load-bearing: bodies within a block eat in id order, and `EnergyGrid.take`
 * empties cells one at a time. Reused frame to frame; `build` overwrites.
 */
export class HarvestPlan {
  /** Six per block: fi, fj, wi, wj, first, count. */
  blocks = new Int32Array(0);
  /** Body slot per entry, in the order its block feeds them. */
  slots = new Int32Array(0);
  /**
   * The agent id in that slot when the plan was built. The GPU path credits a
   * plan a frame after building it, and the slot may have been reassigned.
   */
  ids = new Int32Array(0);
  /** `HARVEST_STRIDE` floats an entry: room, then per-species uptake rate and affinity. */
  rooms = new Float64Array(0);
  /** The energy-cell key per block, which the CPU runner needs and the GPU does not. */
  keys = new Float64Array(0);
  nBlocks = 0;
  nEntries = 0;
  /**
   * Whether the plan that was built is the metered one. Recorded here rather
   * than re-derived from `Params`, which a slider may edit between the fill
   * and the drain a frame later; the two must agree.
   */
  metered = false;
  /*
   * Binning as intrusive lists over a dense cell table: `head[c]` is the most
   * recent body to land in cell `c` and `next[slot]` the one before it. A
   * cell's chain is unwound when its block is written out and `head` cleared
   * as it goes, so the table is all -1 between builds without a fill.
   */
  private head = new Int32Array(0);
  private next = new Int32Array(0);
  private touched = new Int32Array(0);
  private readonly rect = new Int32Array(4);
  /** Room in each body's gut, by slot, read once per body per build. */
  private room = new Float64Array(0);

  /**
   * `kinetics` with `cap > 0` means the metered mouthful; absent, or at `cap`
   * 0, the single-species take-what-fits path the pond runs by default.
   */
  build(agents: Iterable<Agent>, store: AgentStore, grid: EnergyGrid, kinetics?: UptakeKinetics): void {
    /*
     * Whether uptake is metered, not whether a kinetics object was handed
     * over: `sim.ts` passes one every frame with `cap` from `uptakeVmax`,
     * which ships at zero. Must stay the same condition `runHarvestPlan`
     * uses, so the fill and the drain agree about whether the rates mean anything.
     */
    const meter = kinetics !== undefined && kinetics.cap > 0;
    this.metered = meter;
    const LOCKED = store.locked;
    const CHEM = store.chemAll;
    const KIND = store.kindCode;
    const EXTRA = store.extra;
    const CAP = store.energyCap;
    const X = store.x;
    const Y = store.y;
    const ID = store.id;
    this.nBlocks = 0;
    this.nEntries = 0;
    const side = grid.cellsPerSide;
    if (side <= 0) return;
    const cells = side * side;
    if (this.head.length < cells) {
      this.head = new Int32Array(cells).fill(-1);
      this.touched = new Int32Array(cells);
    }
    if (this.next.length < store.capacity) {
      this.next = new Int32Array(store.capacity);
      this.room = new Float64Array(store.capacity);
    }
    const head = this.head;
    const room = this.room;
    const next = this.next;
    const touched = this.touched;
    let nTouched = 0;
    let entries = 0;
    for (const a of agents) {
      const s = a.slot;
      if (LOCKED[s]) continue;
      // Metered, a mouthful is bounded by room in the gut: a body swallows
      // before it converts anything.
      if (meter) {
        const r = gutRoomOf(store, s, kinetics.gutSize);
        if (r <= EXTRA_FULL_EPS) continue;
        room[s] = r;
      } else if (EXTRA[s] >= CAP[s] - EXTRA_FULL_EPS) continue;
      const x = X[s];
      const y = Y[s];
      // Off the map is barren, not merely empty: no ambient either.
      if (!grid.inBounds(x, y)) continue;
      const c = grid.denseIndexAt(x, y, side);
      if (c < 0) continue;
      if (head[c] < 0) touched[nTouched++] = c;
      next[s] = head[c];
      head[c] = s;
      entries++;
    }
    if (nTouched === 0) return;

    if (this.blocks.length < nTouched * 6) this.blocks = new Int32Array(nTouched * 12);
    if (this.keys.length < nTouched) this.keys = new Float64Array(nTouched * 2);
    if (this.slots.length < entries) {
      this.slots = new Int32Array(entries * 2);
      this.ids = new Int32Array(entries * 2);
      this.rooms = new Float64Array(entries * 2 * HARVEST_STRIDE);
    }
    const B = this.blocks;
    const slots = this.slots;
    const ids = this.ids;
    const rooms = this.rooms;
    const rect = this.rect;
    for (let t = 0; t < nTouched; t++) {
      const c = touched[t];
      const i = c % side;
      const j = (c - i) / side;
      const first = this.nEntries;
      let k = first;
      /*
       * Bounded: a slot pushed onto the same cell twice gets `next[s] === s`
       * and the walk never reaches -1, which happens only if two live bodies
       * claim one store slot. `entries` bounds every honest chain. Failing
       * here names the broken invariant; spinning inside `step` says nothing.
       */
      const limit = first + entries;
      for (let s = head[c]; s >= 0; s = next[s]) {
        if (k > limit) {
          throw new Error(
            `HarvestPlan: cell ${c} chains past ${entries} entries — ` +
              'two bodies share a store slot',
          );
        }
        slots[k++] = s;
      }
      head[c] = -1;
      // A block clipped away entirely feeds nobody; its slots were written
      // past `nEntries` and are overwritten by the next block.
      if (!grid.blockRectAt(i, j, rect)) continue;
      // The chain is newest-first; reversing it is nearly the id sort, and
      // the insertion pass makes it exact.
      for (let lo = first, hi = k - 1; lo < hi; lo++, hi--) {
        const tmp = slots[lo];
        slots[lo] = slots[hi];
        slots[hi] = tmp;
      }
      for (let p = first + 1; p < k; p++) {
        const v = slots[p];
        const vid = ID[v];
        let q = p - 1;
        while (q >= first && ID[slots[q]] > vid) {
          slots[q + 1] = slots[q];
          q--;
        }
        slots[q + 1] = v;
      }
      const bo = this.nBlocks * 6;
      B[bo] = rect[0];
      B[bo + 1] = rect[1];
      B[bo + 2] = rect[2];
      B[bo + 3] = rect[3];
      B[bo + 4] = first;
      B[bo + 5] = k - first;
      this.keys[this.nBlocks] = cellKey(i, j);
      this.nBlocks++;
      for (let e = first; e < k; e++) {
        const sl = slots[e];
        ids[e] = ID[sl];
        const ro = e * HARVEST_STRIDE;
        if (meter) {
          /*
           * Per body: `total` is one mouthful a frame and `ks` the body's own
           * affinity gene per species. What the body can do with a species is
           * `Sim.runDigestion`'s question, after the swallowing.
           */
          const g = sl * CHEM_LEN;
          const yield_ = KIND[sl] === KIND_ERA ? kinetics.yEra : kinetics.yDirect;
          rooms[ro + HARVEST_ROOM] = room[sl];
          rooms[ro + HARVEST_TOTAL] = kinetics.cap * yield_;
          for (let c = 0; c < CHANNELS; c++) {
            rooms[ro + HARVEST_KS + c] = uptakeKsOf(CHEM, g, c, kinetics.ks);
          }
        } else {
          rooms[ro + HARVEST_ROOM] = CAP[sl] - EXTRA[sl];
        }
      }
      this.nEntries = k;
    }
  }
}

/**
 * How fast a body may draw from a cell: Monod, `v = vmax * S / (Ks + S)`.
 * `cap` is `params.uptakeVmax * dt`; zero means the unmetered path — take
 * what fits, instantly — and returns `Infinity`, which `Math.min` leaves
 * bit-identical. Exported as the reference law `field-kernel.test.ts`
 * checks against; `runHarvestPlan` and `field.wgsl` inline the same term
 * because they cap by share and mean the opposite thing by a rate of zero.
 */
export function uptakeRate(density: number, cap: number, ks: number): number {
  if (!(cap > 0)) return Infinity;
  if (density <= 0) return 0;
  return (cap * density) / (ks + density);
}

/**
 * `uptakeVmax * dt` and `uptakeKs`, as the harvest wants them. `cap` is the
 * whole mouthful, not a per-species rate: uptake always samples all four
 * species by what is standing in the cell, and the shares sum to `cap`.
 * What is swallowed lands in the gut as itself; whether a body can use it
 * is `Sim.runDigestion`'s question, which is what lets waste exist.
 */
export interface UptakeKinetics {
  cap: number;
  ks: number;
  /**
   * Yield on what a body takes up directly, and on what an Era does. Applied
   * to the rate, so at `yDirect` 0 a body draws nothing and the ground is
   * untouched: obligate trophic dependency. `yEra` above 1 makes an Era's
   * income come from the ground under it rather than a mint keyed on its
   * glyph. A fresh spawn has about `EXTRA_CAP / upkeep` seconds of tank,
   * which bounds how far `yDirect` can drop before the soup dies.
   */
  yDirect: number;
  yEra: number;
  /**
   * Hill coefficient on uptake: `v = vmax * S^n / (ks^n + S^n)`. Above 1 the
   * response is convex at low density, which is what makes committing to one
   * species pay; at 1, plain Monod, and everyone is a generalist.
   */
  hillN: number;
  /**
   * `params.gutSize`: how much a body may hold undigested, as a multiple of
   * its own `energyCap`. Bounds a mouthful. See `gutRoomOf`.
   */
  gutSize: number;
}

/**
 * One entry's row in the harvest's flow buffer: gut room, the frame's whole
 * budget `total`, then four affinities. Per body rather than uniforms because
 * the budget is scaled by the kind's trophic yield and `ks` comes off a gene.
 * Widened rather than given its own buffer because `field.wgsl` has no
 * bindings to spare. `got` overwrites `ks` on the way back out: the
 * affinities are consumed before anything is written, and `room` and `total`
 * sit below it. The shader's copies are literals; `field-kernel.test.ts`
 * pins them to these.
 */
export const HARVEST_ROOM = 0;
export const HARVEST_TOTAL = 1;
export const HARVEST_KS = 2;
export const HARVEST_GOT = HARVEST_KS;
export const HARVEST_STRIDE = HARVEST_KS + CHANNELS;

/**
 * Run a plan against the CPU field, crediting as it goes. The shader's
 * `harvest` is a line-for-line port of this loop; `field-kernel.test.ts`
 * holds the mirror. Which path runs is the plan's own `metered` verdict;
 * `uptake` only supplies the Hill coefficient here.
 */
export function runHarvestPlan(
  plan: HarvestPlan,
  store: AgentStore,
  grid: EnergyGrid,
  uptake?: UptakeKinetics,
): void {
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  const B = plan.blocks;
  // The fill's verdict, not a second reading of the kinetics: see `metered`.
  const metered = plan.metered;
  const hill = uptake !== undefined && uptake.hillN > 0 ? uptake.hillN : 1;
  const R = plan.rooms;
  for (let b = 0; b < plan.nBlocks; b++) {
    const first = B[b * 6 + 4];
    const count = B[b * 6 + 5];
    const key = plan.keys[b];
    if (!metered) {
      // The single-species path: take what fits, from the ground, to saturation.
      for (let e = 0; e < count; e++) {
        const s = plan.slots[first + e];
        const cap = CAP[s];
        const room = cap - EXTRA[s];
        if (room <= EXTRA_FULL_EPS) continue;
        const got = grid.take(key, room);
        if (got <= 0) break;
        EXTRA[s] = Math.min(cap, EXTRA[s] + got);
      }
      continue;
    }
    /*
     * The four uptake rows, drawn as one mouthful. Densities are read once
     * for the block, before anybody eats, so visit order cannot change what
     * any body may draw. Each species may have at most its share of `total`,
     * `total * density / stock`, so a body cannot decline the rest of the
     * mixture: scent-rich ground is poor ground, and a body eating its own
     * scent back displaces its income rather than adding to it. `left` is
     * gut room read off the plan, so both field paths bound a mouthful by
     * the same number. Species are visited in index order, which must be
     * identical here and in the shader; `Sim.runDigestion` uses `DIGEST_ORDER`.
     */
    let stock = 0;
    for (let c = 0; c < CHANNELS; c++) {
      const d = grid.densityOf(key, c);
      SPECIES_DENSITY[c] = d;
      if (d > 0) stock += d;
    }
    // Nothing in the water: `stock` is about to be a denominator.
    if (stock <= 0) continue;
    const GUT = store.gut;
    for (let e = 0; e < count; e++) {
      const s = plan.slots[first + e];
      const ro = (first + e) * HARVEST_STRIDE;
      let left = R[ro + HARVEST_ROOM];
      if (left <= EXTRA_FULL_EPS) continue;
      const total = R[ro + HARVEST_TOTAL];
      if (!(total > 0)) continue;
      const go = s * CHEM_SPECIES;
      // `FLOW_EPS` between species, as the shader has it; the two sides must
      // stop on the same crumb.
      for (let c = 0; c < CHANNELS && left > FLOW_EPS; c++) {
        const density = SPECIES_DENSITY[c];
        if (!(density > 0)) continue;
        /*
         * Monod inline rather than through `uptakeRate`: there `cap <= 0`
         * means unmetered, here a `total` of zero is a kind with no trophic
         * yield. `total` stands in for `vmax` on every species. Hill at `n`
         * is plain Monod at 1 and skips the two `pow` calls.
         */
        const ks = R[ro + HARVEST_KS + c];
        const sN = hill === 1 ? density : Math.pow(density, hill);
        const kN = hill === 1 ? ks : Math.pow(ks, hill);
        const rate = (total * sN) / (kN + sN);
        // The proportional sample: this species' share of one budget.
        const share = total * (density / stock);
        let want = rate < share ? rate : share;
        if (want > left) want = left;
        if (!(want > 0)) continue;
        const got = grid.takeFrom(key, c, want);
        if (got <= 0) continue;
        // Into the gut as itself, not into the tank as money; a species the
        // body cannot convert takes up room until `Sim.runExcretion` puts it back.
        GUT[go + c] += got;
        left -= got;
      }
    }
  }
}

/**
 * Room left in one body's gut: `gutSize` multiples of its heritable
 * `energyCap`, so one gene moves tank and gut together. Never negative.
 */
export function gutRoomOf(store: AgentStore, slot: number, gutSize: number): number {
  const gutCap = store.energyCap[slot] * gutSize;
  const room = gutCap - store.gutTotal(slot);
  return room > 0 ? room : 0;
}

/** Block densities, one per species, reused across blocks. */
const SPECIES_DENSITY = new Float64Array(CHANNELS);

/**
 * The harvest the pond runs: bin, then drain, on the store's arrays. At
 * `cap` 0 it is `harvestSlots` on a store, bit for bit; `harvestSlots`
 * exists so `energy.test.ts` can pin that path on plain literals.
 */
export function harvestSlotsFast(
  agents: Iterable<Agent>,
  store: AgentStore,
  grid: EnergyGrid,
  plan: HarvestPlan = new HarvestPlan(),
  uptake?: UptakeKinetics,
): void {
  plan.build(agents, store, grid, uptake);
  runHarvestPlan(plan, store, grid, uptake);
}

export interface UpkeepOptions {
  /**
   * Fraction of ordinary upkeep put back into the field rather than
   * destroyed, `params.upkeepExcrete`. 0: rent vanishes. 1: a body creates
   * and destroys nothing. Leaves as the body's own excretion mix through
   * `payOut`, which is why the store path also wants `expressed`. Not
   * `params.excreteRate`, which is the reaction table's dial.
   */
  rentBack?: number;
  /** `params.eraUpkeepRatio`; see `upkeepRateFor` and `upkeepRateOf`. */
  eraRatio?: number;
  /**
   * Whether `refreshExpression` ran this frame, so the vectors mean something.
   * Passed rather than derived so the default pays no pass over the roster.
   */
  expressed?: boolean;
}

/**
 * The upkeep the pond runs, on the store's arrays. Bills by `upkeepRateOf`
 * when the vectors are live (`opts.expressed`), by glyph otherwise, and pays
 * `rentBack` out through `payOut` as the body's own excretion mix.
 */
export function tickUpkeepFast(
  agents: Iterable<Agent>,
  store: AgentStore,
  dt: number,
  rate: number,
  grid?: EnergyGrid,
  opts: UpkeepOptions = {},
): number[] {
  if (!(dt > 0)) return [];
  const back = opts.rentBack ?? 0;
  const eraRatio = opts.eraRatio ?? ERA_UPKEEP_RATIO;
  const expressed = opts.expressed ?? false;
  const LOCKED = store.locked;
  const KIND_CODE = store.kindCode;
  const EXPRESS = store.expressAll;
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  const FLOOR = store.debtCap;
  const X = store.x;
  const Y = store.y;
  const ID = store.id;
  const dead: number[] = [];
  for (const a of agents) {
    const s = a.slot;
    if (LOCKED[s]) continue;
    const r = expressed ? upkeepRateOf(EXPRESS, s, rate, eraRatio) : upkeepRateFor(CODE_KIND[KIND_CODE[s]], rate, eraRatio);
    if (r === 0) continue;
    const was = EXTRA[s];
    const next = was - r * dt;
    const cap = CAP[s];
    const floor = FLOOR[s];
    if (next > cap) {
      // A full producer spills onto the ground rather than into nothing.
      if (grid) grid.addAt(X[s], Y[s], next - cap);
      EXTRA[s] = cap;
    } else {
      EXTRA[s] = Math.max(floor, next);
    }
    // The ground gets what left the tank, not what was billed: a body in debt excretes nothing.
    if (grid && back > 0) {
      const paid = Math.max(0, was) - Math.max(0, EXTRA[s]);
      if (paid > 0) payOut(grid, EXPRESS, s, X[s], Y[s], paid * back);
    }
    if (was > floor && EXTRA[s] <= floor) dead.push(ID[s]);
  }
  return dead;
}

/** One body's payout mix, reused. */
const PAYOUT = new Float64Array(CHANNELS);

/**
 * Put `amount` that left this body's tank back into the field as its
 * excretion mix. The one road out for every tank-to-dish spender — rent,
 * the row cost, the gait's pathway — so they all leave as the same mix, and
 * a body fouls the cell it is standing in. Ground when the body expresses
 * no excretion row at all: rent still has to land somewhere.
 */
export function payOut(grid: EnergyGrid, express: Float64Array, slot: number, x: number, y: number, amount: number): void {
  const eo = slot * ROW_COUNT + ROW_EXCRETE;
  let sum = 0;
  for (let c = 0; c < CHANNELS; c++) sum += express[eo + c];
  if (!(sum > 0)) {
    grid.addAt(x, y, amount);
    return;
  }
  const k = amount / sum;
  for (let c = 0; c < CHANNELS; c++) PAYOUT[c] = express[eo + c] * k;
  grid.addSpeciesAt(x, y, PAYOUT);
}

/** Debt as a need: zero at break-even, `-debtCap` at the point of death. */
export function hungerNeed(a: SlotBody): number {
  return a.extra < 0 ? -a.extra : 0;
}

/**
 * Where on this body's own tank a rescue aims, from `debtCap` at 0 to
 * `energyCap` at 1. Always between the two.
 */
export function rescueTarget(a: {
  debtCap: number;
  energyCap: number;
  rescueTo: number;
}): number {
  const lo = a.debtCap;
  const hi = a.energyCap;
  const span = hi - lo;
  if (!(span > 0)) return lo;
  const t = a.rescueTo;
  const pct = t <= 0 ? 0 : t >= 1 ? 1 : t;
  return lo + pct * span;
}

/**
 * How much a body asks for once it has been in debt, until it is standing
 * again: what it is short of `rescueTarget`, with `recovering` latched on
 * the way past zero in both directions. Without the latch the ask goes
 * silent at exactly 0 and the same body is rescued every frame. Only bodies
 * that went under ask this way, so a merely poor net does not level out.
 */
export function rescueNeed(a: SlotBody): number {
  const target = rescueTarget(a);
  if (a.extra < 0) a.recovering = true;
  else if (a.extra >= target - EXTRA_FULL_EPS) a.recovering = false;
  const hunger = hungerNeed(a);
  if (!a.recovering) return hunger;
  const gap = target - a.extra;
  return gap > hunger ? gap : hunger;
}

/**
 * How much more this end of a stalled redex needs before the pair can fire,
 * against the share it has to pay, not the cap. `banked` is what the escrow
 * already holds against this end. Reads `extra` rather than `spareEnergy`:
 * a body in debt asks for enough to clear it and pay.
 */
export function redexNeed(a: SlotBody, banked = 0): number {
  const gap = rewriteShareOf(a) - banked - a.extra;
  return gap > 0 ? gap : 0;
}

/**
 * What this body can pass on: its stock above break-even. A body in debt
 * has none, so energy arriving settles the debt first.
 */
export function spareEnergy(a: SlotBody): number {
  return a.extra > 0 ? a.extra : 0;
}

/**
 * Flat neighbour lists for the wire graph, in the caller's index space:
 * `off[i]..off[i+1]` are body `i`'s neighbours in `nei`. Built by counting
 * sort, so neighbours arrive in wire order. Self-wires and ends outside the
 * index are dropped; duplicate wires between one pair are kept and both conduct.
 */
export class WireAdjacency {
  off = new Int32Array(1);
  nei = new Int32Array(0);
  private cursor = new Int32Array(0);
  private work = new Int32Array(0);

  /**
   * Scratch for a relaxation over the graph. Sized generously: a body can be
   * re-queued each time a bigger need reaches it.
   */
  queue(n: number): Int32Array {
    const want = Math.max(64, n * 8);
    if (this.work.length < want) this.work = new Int32Array(want);
    return this.work;
  }

  /**
   * Builds the CSR from an id-keyed index. Only the tests call this; the
   * frame reads `Sim.refreshWakeGraph`'s CSR. `index` maps agent id to its
   * slot in the caller's dense list. `wires` is a factory, not an iterable:
   * the counting sort walks the set twice, and a one-shot iterator would
   * leave the second pass empty with every neighbour reading as body 0.
   */
  build(
    n: number,
    index: Map<number, number>,
    wires: () => Iterable<{ a: { id: number }; b: { id: number } }>,
  ): void {
    if (this.off.length < n + 1) this.off = new Int32Array(Math.max(16, (n + 1) * 2));
    if (this.cursor.length < n) this.cursor = new Int32Array(Math.max(16, n * 2));
    this.off.fill(0, 0, n + 1);
    let total = 0;
    for (const w of wires()) {
      const ia = index.get(w.a.id);
      const ib = index.get(w.b.id);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      this.off[ia + 1]++;
      this.off[ib + 1]++;
      total += 2;
    }
    for (let i = 0; i < n; i++) this.off[i + 1] += this.off[i];
    if (this.nei.length < total) this.nei = new Int32Array(Math.max(16, total * 2));
    for (let i = 0; i < n; i++) this.cursor[i] = this.off[i];
    for (const w of wires()) {
      const ia = index.get(w.a.id);
      const ib = index.get(w.b.id);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      this.nei[this.cursor[ia]++] = ib;
      this.nei[this.cursor[ib]++] = ia;
    }
  }
}

export function resetRequests(agents: Iterable<SlotBody>): void {
  for (const a of agents) a.request = 0;
}

/** Raise a body's own need. The field never lowers what is already there. */
export function seedRequest(agent: SlotBody, amount: number): void {
  if (amount > agent.request) agent.request = amount;
}

/**
 * Relax the need field to its fixpoint, inside this frame: the reach-0 path
 * and the pond's default. Queue-driven, `O(n + wires)`, and identical to
 * what iterating `spreadRequests` converges to. What it cannot do is take
 * time: the field has no history, so nothing propagates.
 */
export function relaxRequestsFast(
  list: Agent[],
  store: AgentStore,
  adj: WireAdjacency,
  decay?: number,
): void {
  const { off, nei } = adj;
  const q = adj.queue(list.length);
  const REQUEST = store.request;
  const REQUEST_DECAY = store.requestDecay;
  let head = 0;
  let tail = 0;
  for (let i = 0; i < list.length; i++) {
    if (REQUEST[list[i].slot] > REQUEST_FLOOR) q[tail++] = i;
  }
  while (head < tail) {
    const at = q[head++];
    const sAt = list[at].slot;
    const keep = Math.min(0.99, Math.max(0, decay ?? REQUEST_DECAY[sAt]));
    const next = REQUEST[sAt] * keep;
    if (next <= REQUEST_FLOOR) continue;
    for (let k = off[at]; k < off[at + 1]; k++) {
      const ni = nei[k];
      const other = list[ni];
      if (!other) continue;
      const sNi = other.slot;
      if (REQUEST[sNi] >= next) continue;
      REQUEST[sNi] = next;
      if (tail >= q.length) return;
      q[tail++] = ni;
    }
  }
}

/** Store-based twin of `spreadRequests` — see `harvestSlotsFast`'s note. */
export function spreadRequestsFast(
  list: Agent[],
  store: AgentStore,
  adj: WireAdjacency,
  prev: Float64Array,
  decay?: number,
): void {
  const { off, nei } = adj;
  const REQUEST = store.request;
  const REQUEST_DECAY = store.requestDecay;
  for (let i = 0; i < list.length; i++) {
    let best = 0;
    for (let k = off[i]; k < off[i + 1]; k++) {
      const ni = nei[k];
      const other = list[ni];
      if (!other) continue;
      const keep = Math.min(0.99, Math.max(0, decay ?? REQUEST_DECAY[other.slot]));
      const v = prev[ni] * keep;
      if (v > best) best = v;
    }
    if (best <= REQUEST_FLOOR) continue;
    const s = list[i].slot;
    if (best > REQUEST[s]) REQUEST[s] = best;
  }
}

/**
 * How one hop of flow is sized. A body gives to its neediest neighbour, only
 * one needier than itself, capped by what the recipient is short of; two
 * comparable needs meet at a flat spot and nothing crosses, which is
 * intended. `quantum` 0 is the continuous law. Above zero a transfer is one
 * whole packet or nothing, and only a body holding at least a packet can
 * send one, which is what puts `transportRecoil`'s impulse above the pond's
 * noise. `grid` is required once `quantum` is on: a whole packet is sent
 * whether or not the far end has room, and the remainder must land somewhere.
 */
export interface FlowOptions {
  /**
   * Override every body's own quantum. Omit — as the simulation does — and
   * each sender uses its own heritable `transportQuantum`, so a chain of
   * bodies with different quanta is a row of coupled relaxation oscillators.
   */
  quantum?: number;
  /** Where a full receiver's overflow is deposited. Required with `quantum`. */
  grid?: EnergyGrid;
}

/**
 * The packet the two flow laws share: send, spill what will not fit, report.
 * Recoil is billed on the whole packet, since the impulse is the sender's.
 */
function deliver(
  give: number,
  toCap: number,
  toExtra: number,
  x: number,
  y: number,
  grid: EnergyGrid | undefined,
): number {
  const room = toCap - toExtra;
  if (give <= room) return toExtra + give;
  const spill = give - Math.max(0, room);
  if (spill > 0) {
    if (!grid) {
      throw new Error('flowCharges: a packet overflowed with no grid to take it; the spill would be minted away');
    }
    grid.addAt(x, y, spill);
  }
  return room > 0 ? toCap : toExtra;
}

/*
 * Scratch for `flowChargesFast`: grown, never shrunk, and shared — one pond
 * per `Sim`, and this pass is not reentrant.
 */
let flowDonors = new Int32Array(0);
let flowSlot = new Int32Array(0);
let flowTaken = new Uint8Array(0);

/** Store-based twin of `flowCharges` — see `harvestSlotsFast`'s note. */
export function flowChargesFast(
  list: Agent[],
  store: AgentStore,
  adj: WireAdjacency,
  onMoved?: (from: Agent, to: Agent, amount: number) => void,
  opts: FlowOptions = {},
): number {
  const n = list.length;
  const forced = opts.quantum;
  const { off, nei } = adj;
  const LOCKED = store.locked;
  const REQUEST = store.request;
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  const ID = store.id;
  const X = store.x;
  const Y = store.y;
  const QUANT = store.transportQuantum;
  if (flowDonors.length < n) {
    flowDonors = new Int32Array(n);
    flowSlot = new Int32Array(n);
    flowTaken = new Uint8Array(n);
  }
  const donors = flowDonors;
  // Slots resolved once, up front; the sort comparator reads them.
  const slotOf = flowSlot;
  const taken = flowTaken;
  taken.fill(0, 0, n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const s = list[i].slot;
    slotOf[i] = s;
    if (LOCKED[s]) continue;
    // `spareEnergy(a) > FLOW_EPS`, inlined.
    const q = forced !== undefined ? forced : QUANT[s];
    if (EXTRA[s] > FLOW_EPS && EXTRA[s] >= (q > 0 ? q : FLOW_EPS)) donors[count++] = i;
  }
  // Neediest donor first, so a body that is itself being fed passes on what it
  // does not need in the same frame rather than sitting on it.
  donors
    .subarray(0, count)
    .sort((p, q) => REQUEST[slotOf[q]] - REQUEST[slotOf[p]] || ID[slotOf[p]] - ID[slotOf[q]]);
  let moved = 0;
  for (let d = 0; d < count; d++) {
    const di = donors[d];
    const ds = slotOf[di];
    let bestIdx = -1;
    let bestSlot = -1;
    let bestR = REQUEST[ds];
    for (let k = off[di]; k < off[di + 1]; k++) {
      const ni = nei[k];
      if (taken[ni]) continue;
      const other = list[ni];
      if (!other) continue;
      const os = slotOf[ni];
      if (LOCKED[os]) continue;
      if (REQUEST[os] > bestR) {
        bestIdx = ni;
        bestSlot = os;
        bestR = REQUEST[os];
      }
    }
    if (bestIdx < 0) continue;
    let give: number;
    const dq = forced !== undefined ? forced : QUANT[ds];
    if (dq > 0) {
      // See `flowCharges`: whole packet or nothing, overflow to the ground.
      give = dq;
    } else {
      const spareD = EXTRA[ds] > 0 ? EXTRA[ds] : 0;
      give = Math.min(spareD, REQUEST[bestSlot], CAP[bestSlot] - EXTRA[bestSlot]);
      if (give <= FLOW_EPS) continue;
    }
    EXTRA[ds] -= give;
    EXTRA[bestSlot] = deliver(give, CAP[bestSlot], EXTRA[bestSlot], X[bestSlot], Y[bestSlot], opts.grid);
    taken[bestIdx] = 1;
    moved += give;
    onMoved?.(list[di], list[bestIdx], give);
  }
  return moved;
}

/**
 * Put a rewrite's released energy back into the net, neediest first. Each
 * body takes only what it has room for; the remainder drops to the ground
 * at the rewrite's midpoint.
 */
export function settlePool(
  pool: number,
  recipients: Iterable<SlotBody>,
  grid: EnergyGrid,
  x: number,
  y: number,
): void {
  const list = [...recipients].sort((a, b) => b.request - a.request || a.id - b.id);
  let left = pool;
  for (const a of list) {
    if (left <= 0) break;
    const room = a.energyCap - a.extra;
    if (room <= EXTRA_FULL_EPS) continue;
    const give = Math.min(left, room);
    a.extra += give;
    left -= give;
  }
  if (left > 0) grid.addAt(x, y, left);
}
