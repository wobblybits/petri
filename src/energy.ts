import type { Agent, AgentKind } from './agents.ts';
import { CH, CHANNELS, FIELD_CELL, type Fields } from './fields.ts';
import type { AgentStore } from './agent-store.ts';
import { KIND_ERA } from './native/solver.ts';
import type { Rule } from './rewrite.ts';

/**
 * `extra` is the one energy scalar a body carries: positive is stock it can
 * spend or pass on, negative is debt, and a whole unit of debt is death.
 * Upkeep decrements it directly — there is no second accumulator, so "how much
 * does this body need" is just how far below zero it has fallen.
 */

/**
 * What one end of a rewrite pays. A Con–Dup commute turns two bodies into
 * four, so it costs two units, one from each side.
 */
export const REWRITE_SHARE = 1;

/**
 * The most a body can hold — deliberately more than a rewrite share.
 *
 * These were the same number, and that made a commute a knife edge: a pair
 * could only afford one at the exact instant both ends were at the cap, and
 * since upkeep drains continuously they sat permanently a fraction below it.
 * The quarter-unit of headroom is what a body has spare *after* it can pay,
 * so at the default upkeep a full body stays able to commute for ten seconds
 * rather than for one frame.
 *
 * Transport obeys the cap too: a pump can never push a body past full, which
 * is most of why pumping is slow. This is the figure for a Con or a Dup; an
 * Era holds `ERA_CAP_RATIO` times it.
 */
export const EXTRA_CAP = 1.25;

/**
 * How much more an Era holds than a Con or a Dup.
 *
 * An Era is the one body that cannot spend: it has a single port, it never
 * commutes, and it pays no rent — it produces (`ERA_UPKEEP_RATIO`). Everything
 * it earns is therefore for somebody else, and the one thing it can usefully
 * be is a battery. At the same cap as everyone else it filled and then sat
 * there wasting its income, so a net's leaves were its smallest reserve
 * instead of its largest.
 *
 * Doubling the tank does not double what an Era is worth dead — `BODY_VALUE`
 * is flat across kinds, and a rewrite still costs and yields the same — so
 * this buys storage, not a mint. What it does change is the shape of a net's
 * reserve: energy parked on the boundary rather than spread thin through the
 * middle, and a deeper buffer between a lean patch and a wave of starvation.
 */
export const ERA_CAP_RATIO = 2;

/**
 * What a fresh body of this kind can hold, before any breeding drifts it.
 *
 * A body's actual ceiling lives on it as `energyCap` — heritable, and
 * recombined from both parents across a Con+Dup commute's children rather
 * than reset to this — so this is only the seed `createAgent` gives a body
 * born outside a rewrite.
 */
export function extraCapFor(kind: AgentKind): number {
  return kind === 'era' ? EXTRA_CAP * ERA_CAP_RATIO : EXTRA_CAP;
}
/**
 * Default death floor: a whole unit of debt. A body's own `debtCap` takes
 * over once it is alive — this is only the seed, and the sign lock: a debt
 * cap is never allowed to reach break-even, or the rescue latch never fires.
 */
export const EXTRA_FLOOR = -1;
/** Closest to zero a heritable `debtCap` may sit. Strictly negative. */
export const DEBT_CAP_MAX = -0.05;
const EXTRA_FULL_EPS = 1e-6;

/**
 * What a body's existence is worth: a full tank.
 *
 * Note this is more than the `REWRITE_SHARE` that built it, so a body is worth
 * more dead than it cost to make and the cycle commute-then-annihilate mints
 * `2 * (EXTRA_CAP − REWRITE_SHARE)` = 0.5. That is the metabolism, not an
 * accounting slip: upkeep drains continuously, so a net that keeps rewriting
 * feeds itself and a net that sits still starves. Set this to `REWRITE_SHARE`
 * for a strictly conserved pond.
 */
export const BODY_VALUE = EXTRA_CAP;

/** Energy released by one death: the body's own worth plus what it held. */
export function deathYield(a: { extra: number }): number {
  return Math.max(0, BODY_VALUE + a.extra);
}

/** Energy locked up in one body's existence. Every kind costs the same. */
export const AGENT_VALUE: Record<AgentKind, number> = {
  era: BODY_VALUE,
  con: BODY_VALUE,
  dup: BODY_VALUE,
};

/**
 * How much of a neighbour's need reaches you, per hop. The fallback for a
 * body with no `requestDecay` of its own, and what the `requestDecay` slider
 * seeds a fresh body's trait to; `spreadRequests` otherwise reads each
 * body's own value.
 *
 * The field is `max(own need, best neighbour's field x decay)`, so need spreads
 * as a decaying scent rather than a hop count. Two consequences matter. Need
 * competes by magnitude, so a starving body outpulls a redex that is nearly
 * paid for even from further away. And where two comparable needs face each
 * other across a net the fields meet at equal value, the local gradient there
 * is flat, and nothing crosses — the watershed sits wherever the needs happen
 * to balance instead of at a fixed hop count.
 *
 * The value sets how far a shortage is audible, against `REQUEST_FLOOR`: a
 * whole unit of need carries 20 hops at 0.8 and 43 at 0.9. It also sets how
 * much moves, because a transfer is capped by the field at the receiving end —
 * so raising it makes distant demand both visible and worth answering.
 *
 * Not 1. At 1 the field stops being a gradient: every body in a connected net
 * ends up holding the same largest need, no neighbour is ever strictly needier
 * than another, and transport stops entirely.
 */
export const REQUEST_DECAY = 0.9;

/**
 * Field values below this are not worth carrying further. This bounds the
 * relaxation; it is deliberately *not* a minimum transfer size. A stalled
 * redex is often a fraction of a percent short — one frame of upkeep — and
 * refusing to move that much is refusing to let it ever fire.
 */
export const REQUEST_FLOOR = 0.01;

/** Smallest transfer worth doing. Guards against denormal churn, nothing more. */
const FLOW_EPS = 1e-9;

/**
 * Field value at which the request ring is drawn at full strength. One whole
 * extra of unmet need is as loud as the display needs to get.
 */
export const REQUEST_FULL = 1;

/**
 * Era's upkeep as a fraction of everyone else's, negative because an Era pays
 * energy in rather than out.
 *
 * An Era is a sink for structure, not a machine that runs: it has one port, it
 * cannot commute, and the only thing it does is end a wire. Charging it rent
 * makes the cheapest body in the net the one most likely to starve, which
 * inverts what the glyph means. Producing instead makes a net's leaves its
 * income, so growing a boundary is worth something and a net that erases
 * itself down to Eras is quietly refuelling.
 *
 * Small on purpose: at the default upkeep an Era yields one extra every 200 s
 * against a Con or Dup spending one every 40 s, so a third-Era soup produces
 * about a tenth of what it burns. Set to 0 for Eras that are simply free.
 */
export const ERA_UPKEEP_RATIO = -0.2;

/** Upkeep per second for one body, given the global rate. */
export function upkeepRateFor(kind: AgentKind, rate: number): number {
  return kind === 'era' ? rate * ERA_UPKEEP_RATIO : rate;
}

export function agentValue(kind: AgentKind): number {
  return AGENT_VALUE[kind];
}


/** Holding as much as it can. Nothing more can be harvested or pumped in. */
export function atCap(a: { extra: number; energyCap: number }): boolean {
  return a.extra >= a.energyCap - EXTRA_FULL_EPS;
}

/**
 * Able to pay its side of a rewrite. Not the same as being full.
 *
 * A heritable `energyCap` can sit below `REWRITE_SHARE` — breeding walks it
 * down to `EXTRA_CAP * 0.5`. Asking for a whole share then makes a full tank
 * still unable to commute, so leftover principal pairs sit idle after a few
 * copies. A body pays the share, or everything it can hold, whichever is
 * smaller; `spendExtra` already floors at zero.
 */
export function canPayShare(a: { extra: number; energyCap: number }): boolean {
  const share = Math.min(REWRITE_SHARE, a.energyCap);
  return a.extra >= share - EXTRA_FULL_EPS;
}

export function extrasOf(
  a: { extra: number; energyCap: number },
  b: { extra: number; energyCap: number },
): number {
  return (canPayShare(a) ? 1 : 0) + (canPayShare(b) ? 1 : 0);
}

export function spendExtra(a: { extra: number }): void {
  a.extra = Math.max(0, a.extra - REWRITE_SHARE);
}

/**
 * Bodies destroyed minus bodies created. Annihilation and era–era take two
 * away; erase swaps two for two; a commute builds four from two.
 */
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
 * What the pair pays up front, in shares — one per end, so a commute costs
 * two. Charged at the share rather than at `BODY_VALUE` on purpose: pricing a
 * commute at the pair's entire full tank would mean it could only ever fire
 * with both ends exactly at the cap, which is the knife edge the headroom
 * exists to remove.
 */
export function rewriteCost(rule: Rule): number {
  const d = bodyDelta(rule);
  return d < 0 ? -d * REWRITE_SHARE : 0;
}

/** Existence released when the rewrite commits, before the dying bodies' own stock. */
export function rewriteYield(rule: Rule): number {
  const d = bodyDelta(rule);
  return d > 0 ? d * BODY_VALUE : 0;
}

/*
 * Packed into one integer rather than a `"${i},${j}"` string. This runs once
 * per agent per frame in `harvestSlots` plus every render and every rewrite
 * settlement, and a template-literal string allocates and then costs a
 * string hash on every `Map` lookup; a number does neither. `CELL_KEY_OFFSET`
 * shifts i/j positive before packing (a `Map` key needs a total order, not a
 * sign), and `CELL_KEY_WIDTH` is `2 * CELL_KEY_OFFSET`, so the shifted value
 * fills it exactly with no overlap between rows. The product tops out under
 * 2^52, comfortably inside float64's 2^53 safe-integer range, and the offset
 * covers cell indices out past ±33 million — thousands of pond-widths in
 * either direction — so nothing this sim does can wrap it.
 */
const CELL_KEY_OFFSET = 1 << 25;
const CELL_KEY_WIDTH = CELL_KEY_OFFSET * 2;

function cellKey(i: number, j: number): number {
  return (i + CELL_KEY_OFFSET) * CELL_KEY_WIDTH + (j + CELL_KEY_OFFSET);
}

function decodeKey(key: number): { i: number; j: number } {
  const i = Math.floor(key / CELL_KEY_WIDTH) - CELL_KEY_OFFSET;
  const j = key - (i + CELL_KEY_OFFSET) * CELL_KEY_WIDTH - CELL_KEY_OFFSET;
  return { i, j };
}

/**
 * Sparse world-space energy. Unvisited cells hold `ambient`; once a cell is
 * touched, whatever remains is stored explicitly (including 0).
 * No decay or diffusion — occupancy is the only transport.
 *
 * `inexhaustible` keeps every in-bounds cell at `ambient` and makes `take`
 * a read. The designer uses that so Play is never starved of extra.
 */
export class EnergyGrid {
  cellSize: number;
  ambient: number;
  inexhaustible = false;
  private readonly cells = new Map<number, number>();

  /*
   * Where the energy actually lives.
   *
   * Bound to a field, a cell of this grid is a block of that field's cells on
   * channel `CH.energy`, and this class becomes a coarse view onto them
   * rather than a store of its own. Unbound it keeps the sparse map, which is
   * what every test that builds an `EnergyGrid` by hand still gets.
   *
   * The point of moving is that the ground stops being inert. On the field it
   * diffuses, so a grazed patch refills from its neighbours instead of being
   * gone forever, and `grow` can put a carrying capacity on it — which is the
   * whole difference between a resource and a seam of ore.
   *
   * The view stays coarse on purpose. `harvestSlots` groups bodies by cell so
   * that everyone standing in the same place competes for the same stock, and
   * that grouping is the economy's only crowding pressure. A field cell is
   * ten units across and a body is bigger than that, so indexing straight to
   * one would have quietly deleted the competition.
   */
  private fields: Fields | null = null;

  /*
   * The world bound, a disk tracked to `home`. Outside it there is no ground:
   * nothing can be harvested and nothing deposited, so a body that drifts past
   * the edge starves on whatever it was carrying. The same disk is the scent
   * mask and the hard wall, so "off the map" means one thing rather than two.
   *
   * Stored cells outside the bound are kept rather than pruned. The bound
   * used to move with home, and a cell that falls outside today can fall back
   * inside tomorrow with its contents intact; dropping them would quietly
   * destroy energy and the economy is supposed to conserve it.
   */
  private boundX = 0;
  private boundY = 0;
  private boundHalf = Infinity;
  /*
   * The lattice this grid is cut on, shared with the scent field.
   *
   * It used to index from the world origin while the field indexed from its
   * own, so the two grids were offset by whatever fraction of a cell the pond
   * happened to sit at — two rasters of the same world that never lined up.
   * Taking the field's own (already cell-snapped) origin, rather than
   * recomputing one from the raw pin point, makes an energy cell an exact
   * block of scent cells, provided `energyCell` stays a whole multiple of
   * `FIELD_CELL`. Deriving it independently from `cx - half` looked the same
   * but wasn't: the field snaps its origin to a whole cell and this didn't,
   * so the two lattices agreed only when the pin point happened to land on
   * one already.
   *
   * Set once, when the world is pinned. Moving it later would re-key every
   * stored cell, which is why `setBounds` is not called per frame any more.
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

  /** Field cells per energy cell, along one axis. */
  get span(): number {
    return Math.max(1, Math.round(this.cellSize / FIELD_CELL));
  }

  /**
   * What one *field* cell holds when the ground is full, so that a whole
   * energy cell still holds `ambient`.
   *
   * This is the rescaling that keeps the economy where it was. An energy cell
   * is 40 units and a field cell is 10, so sixteen field cells stand where
   * one grid cell used to, and each holds a sixteenth as much. A body walking
   * onto full ground finds the same meal it always did.
   */
  get cellCap(): number {
    const s = this.span;
    return this.ambient / (s * s);
  }

  /** Lay down full ground across the disk. Field-backed only. */
  seedGround(): void {
    this.fields?.fillDisk(CH.energy, this.cellCap);
  }

  /**
   * The field cells under energy cell `(i, j)`: `[fi, fi + span)` squared,
   * clipped to the grid. Empty when the block falls outside it.
   */
  private block(i: number, j: number): { fi: number; fj: number; wi: number; wj: number } | null {
    const f = this.fields;
    if (!f) return null;
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
    return wi > 0 && wj > 0 ? { fi, fj, wi, wj } : null;
  }

  get lattice(): { x: number; y: number } {
    return { x: this.originX, y: this.originY };
  }

  /**
   * Centre and radius of the live disk, in world units, plus the shared
   * lattice origin this grid's cells are cut against — pass the scent field's
   * own `originX`/`originY` (already snapped to a whole field cell) so the two
   * line up. Defaults to the unsnapped `cx - radius`/`cy - radius` for callers
   * that only care about the bound, not alignment with the field.
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
   * Anchored at the world origin, and the cell size is a whole multiple of the
   * scent field's, so an energy cell is an exact block of scent cells rather
   * than a grid at some unrelated offset and pitch. The field slides with home
   * and this does not, which is fine precisely because both are anchored to
   * world coordinates: a body at a given place always reads the same energy
   * cell, however the field's window has scrolled.
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
      const b = this.block(i, j);
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

  /** Take up to `n` from a cell and return how much was taken. */
  take(key: number, n: number): number {
    let want = Math.max(0, n);
    if (this.inexhaustible) return Math.min(this.ambient, want);
    const f = this.fields;
    if (f) {
      // Cell by cell across the block rather than proportionally: grazing
      // leaves an uneven floor, and an uneven floor is what diffusion then
      // has a gradient to work against. Taking a flat share off every cell
      // would keep the block uniform and there would be nothing to flow.
      const { i, j } = decodeKey(key);
      const b = this.block(i, j);
      if (!b) return 0;
      const d = f.data;
      let got = 0;
      for (let y = 0; y < b.wj && want > FLOW_EPS; y++) {
        let k = ((b.fj + y) * f.cols + b.fi) * CHANNELS + CH.energy;
        for (let x = 0; x < b.wi && want > FLOW_EPS; x++, k += CHANNELS) {
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
    const have = this.cells.has(key) ? (this.cells.get(key) ?? 0) : this.ambient;
    const got = Math.min(have, want);
    this.cells.set(key, have - got);
    return got;
  }

  addAt(x: number, y: number, amount: number): void {
    if (amount === 0) return;
    if (!this.inBounds(x, y)) return;
    const f = this.fields;
    if (f) {
      // Into the one field cell it happened in, not spread across the block.
      // A corpse is a point event, and letting it start as a point is what
      // gives diffusion a plume to make out of it.
      f.addAt(CH.energy, x, y, amount);
      return;
    }
    const { key } = this.index(x, y);
    this.cells.set(key, (this.cells.has(key) ? (this.cells.get(key) ?? 0) : this.ambient) + amount);
  }

  setCell(i: number, j: number, amount: number): void {
    const f = this.fields;
    if (f) {
      const b = this.block(i, j);
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
   * Touched cells only — the implicit ambient field is not stored.
   *
   * Sparse path only, and so dead in production: bound to a field there is no
   * map to walk and this yields nothing, whatever the ground actually holds.
   * The one caller left is a test on a hand-built grid. Kept because that test
   * is the only remaining cover for the sparse path, which is still what an
   * `EnergyGrid` built by hand gives you; anything wanting the real ground
   * wants `getCell` over a range, the way `drawEnergyGrid` does it.
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
>;

/**
 * Unlocked agents below cap take from their cell, in id order, up to what
 * they still have room for. Ambient 0.1 therefore takes ten cells to fill.
 */
export function harvestSlots(agents: Iterable<SlotBody>, grid: EnergyGrid): void {
  const hungry = new Map<number, SlotBody[]>();
  for (const a of agents) {
    if (a.locked || atCap(a)) continue;
    // Off the map is barren, not merely empty: no ambient either.
    if (!grid.inBounds(a.x, a.y)) continue;
    const { key } = grid.index(a.x, a.y);
    let list = hungry.get(key);
    if (!list) {
      list = [];
      hungry.set(key, list);
    }
    list.push(a);
  }
  for (const [key, list] of hungry) {
    list.sort((a, b) => a.id - b.id);
    for (const a of list) {
      const cap = a.energyCap;
      const room = cap - a.extra;
      if (room <= EXTRA_FULL_EPS) continue;
      const got = grid.take(key, room);
      if (got <= 0) break;
      a.extra = Math.min(cap, a.extra + got);
    }
  }
}

/**
 * Store-based twin of `harvestSlots`, for sim.ts's per-frame hot path.
 * Identical behavior, reading and writing `AgentStore`'s arrays directly by
 * slot instead of through `Agent`'s accessors.
 *
 * Not just `harvestSlots` sped up in place: `energy.test.ts` builds
 * `SlotBody`-shaped plain object literals directly (not real `Agent`
 * instances) to exercise this logic in isolation, so `harvestSlots` keeps
 * its `Iterable<SlotBody>` signature for that and any other caller that
 * doesn't have a store to hand. `sim.ts` does, every frame, for every
 * agent, which is what makes the accessor overhead worth cutting here.
 */
export function harvestSlotsFast(agents: Iterable<Agent>, store: AgentStore, grid: EnergyGrid): void {
  const LOCKED = store.locked;
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  const X = store.x;
  const Y = store.y;
  const ID = store.id;
  const hungry = new Map<number, number[]>();
  for (const a of agents) {
    const s = a.slot;
    if (LOCKED[s]) continue;
    if (EXTRA[s] >= CAP[s] - EXTRA_FULL_EPS) continue;
    const x = X[s];
    const y = Y[s];
    // Off the map is barren, not merely empty: no ambient either.
    if (!grid.inBounds(x, y)) continue;
    const { key } = grid.index(x, y);
    let list = hungry.get(key);
    if (!list) {
      list = [];
      hungry.set(key, list);
    }
    list.push(s);
  }
  for (const [key, list] of hungry) {
    list.sort((a, b) => ID[a] - ID[b]);
    for (const s of list) {
      const cap = CAP[s];
      const room = cap - EXTRA[s];
      if (room <= EXTRA_FULL_EPS) continue;
      const got = grid.take(key, room);
      if (got <= 0) break;
      EXTRA[s] = Math.min(cap, EXTRA[s] + got);
    }
  }
}

/**
 * Bill every body continuously. `rate` is energy per second, per kind via
 * `upkeepRateFor`, so a full body has `1/rate` seconds of life in hand and
 * slides smoothly into debt after that.
 *
 * A negative rate pays the body instead, capped at a full extra like every
 * other source. Returns the ids that reached their own `debtCap` — death
 * rather than a detachment.
 */
export function tickUpkeep(
  agents: Iterable<SlotBody>,
  dt: number,
  rate: number,
  grid?: EnergyGrid,
): number[] {
  if (!(dt > 0)) return [];
  const dead: number[] = [];
  for (const a of agents) {
    if (a.locked) continue;
    const r = upkeepRateFor(a.kind, rate);
    if (r === 0) continue;
    const was = a.extra;
    const next = a.extra - r * dt;
    if (next > a.energyCap) {
      /*
       * A full producer spills onto the ground rather than into nothing.
       *
       * An Era's upkeep is negative — it makes energy instead of spending it —
       * and it has a tank like anything else. Clamping at the cap quietly
       * destroyed whatever it made past full, so a net's Eras stopped being
       * worth anything the moment they topped up, and the conservation the
       * rest of the economy is careful about had a hole in it.
       *
       * This is the only path that can overfill a body: harvest and transport
       * are both bounded by the room the receiver actually has, and a rewrite's
       * leftovers already go to the grid. If another producer ever appears it
       * should come through here too.
       */
      if (grid) grid.addAt(a.x, a.y, next - a.energyCap);
      a.extra = a.energyCap;
    } else {
      a.extra = Math.max(a.debtCap, next);
    }
    if (was > a.debtCap && a.extra <= a.debtCap) dead.push(a.id);
  }
  return dead;
}

/** Store-based twin of `tickUpkeep` — see `harvestSlotsFast`'s note. */
export function tickUpkeepFast(
  agents: Iterable<Agent>,
  store: AgentStore,
  dt: number,
  rate: number,
  grid?: EnergyGrid,
): number[] {
  if (!(dt > 0)) return [];
  const LOCKED = store.locked;
  const KIND_CODE = store.kindCode;
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
    const r = KIND_CODE[s] === KIND_ERA ? rate * ERA_UPKEEP_RATIO : rate;
    if (r === 0) continue;
    const was = EXTRA[s];
    const next = was - r * dt;
    const cap = CAP[s];
    const floor = FLOOR[s];
    if (next > cap) {
      // See tickUpkeep's own comment: a full producer spills onto the
      // ground rather than into nothing.
      if (grid) grid.addAt(X[s], Y[s], next - cap);
      EXTRA[s] = cap;
    } else {
      EXTRA[s] = Math.max(floor, next);
    }
    if (was > floor && EXTRA[s] <= floor) dead.push(ID[s]);
  }
  return dead;
}

/**
 * How much energy this body is in debt by, and therefore how badly it wants
 * some. Zero at break-even; at the point of death this is `-debtCap`.
 */
export function hungerNeed(a: SlotBody): number {
  return a.extra < 0 ? -a.extra : 0;
}

/**
 * Where on this body's own tank a rescue aims, from `debtCap` at 0 to
 * `energyCap` at 1. Always between the two, so a small tank cannot ask past
 * full and a deep-debt body cannot set a target above its ceiling.
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
 * How much a body asks for once it has been in debt, and until it is standing
 * again. Returns what it is short of `rescueTarget`, and latches `recovering`
 * on the way past zero in both directions.
 *
 * `hungerNeed` alone is an ambulance that stops at the kerb: it goes silent
 * the instant `extra` reaches 0, so a rescued body sat at exactly break-even
 * with a full neighbour beside it and no way to ask for more. It could not
 * afford a rewrite (that costs a whole share), and one frame of upkeep put it
 * back in debt — so the same body was rescued over and over while the surplus
 * two hops away stayed untouched. The latch keeps the ask alive across the
 * zero crossing, which is what turns a rescue into a refill.
 *
 * Only bodies that actually went under ask this way. A body that is merely
 * poor stays quiet, so a well-fed net does not turn into a diffusion pond
 * where every stock levels out and nobody can concentrate enough to act.
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
 * How much more this end of a stalled redex needs before the pair can fire.
 *
 * Measured against the share it has to pay, not against the storage cap — it
 * is asking for enough to commute, not for a full tank. Per end rather than
 * per pair: an end holding 0.8 is short 0.2 no matter what its partner holds.
 */
export function redexNeed(a: SlotBody): number {
  const gap = REWRITE_SHARE - a.extra;
  return gap > 0 ? gap : 0;
}

/**
 * What this body can pass on: its stock above break-even, and nothing more.
 *
 * A body in debt has none, so energy arriving at one settles that debt first
 * and only what is left over travels further. That falls out of `extra` being
 * a single signed scalar rather than needing a rule of its own.
 */
export function spareEnergy(a: SlotBody): number {
  return a.extra > 0 ? a.extra : 0;
}

/**
 * Flat neighbour lists for the wire graph, in the caller's index space.
 *
 * A Map of arrays cost one array per body per frame — some ten thousand
 * short-lived objects on a grown pond — for a structure that is rebuilt from
 * scratch every time anyway. This fills two reusable typed arrays instead:
 * `off[i]..off[i+1]` are body `i`'s neighbours in `nei`. Built by counting
 * sort, so neighbours arrive in wire order exactly as appending would give.
 *
 * Self-wires and ends outside the index are dropped; a duplicate wire between
 * the same pair is not, which matches the graph — two ports can join the same
 * two bodies and both should conduct.
 */
export class WireAdjacency {
  off = new Int32Array(1);
  nei = new Int32Array(0);
  private cursor = new Int32Array(0);
  private work = new Int32Array(0);

  /**
   * Scratch for a relaxation over the graph. Sized generously: a body can be
   * re-queued each time a bigger need reaches it, and the decay bounds how
   * often that can happen.
   */
  queue(n: number): Int32Array {
    const want = Math.max(64, n * 8);
    if (this.work.length < want) this.work = new Int32Array(want);
    return this.work;
  }

  /**
   * `index` maps agent id to its slot in the caller's dense list.
   *
   * `wires` is a factory, not an iterable, because the counting sort walks the
   * set twice and the obvious thing to hand in — `graph.wires.values()` — is a
   * one-shot iterator. Passing it directly leaves the second pass empty and
   * every neighbour reading as body 0, which is wrong quietly rather than
   * loudly: the field still spreads, just through a graph nobody built.
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

/** Store-based twin of `resetRequests` — see `harvestSlotsFast`'s note. */
export function resetRequestsFast(agents: Iterable<Agent>, store: AgentStore): void {
  const REQUEST = store.request;
  for (const a of agents) REQUEST[a.slot] = 0;
}

/** Raise a body's own need. The field never lowers what is already there. */
export function seedRequest(agent: SlotBody, amount: number): void {
  if (amount > agent.request) agent.request = amount;
}

/**
 * Relax the need field over the wire graph until it stops improving.
 *
 * Every body ends up holding the largest need it can see, attenuated per hop
 * by whoever is relaying it — so the field is a potential whose gradient
 * points at whoever is neediest, weighted by how badly and discounted by how
 * far. A body can be improved more than once (a bigger need further away can
 * beat a small one next door), so this is a relaxation rather than one BFS
 * sweep; the decay bounds it, since a value below `REQUEST_FLOOR` stops
 * travelling.
 *
 * `decay`, when passed, overrides every body's own `requestDecay` trait —
 * useful for a test that wants one uniform rate. Left out, each body relays
 * at its own rate, which is what lets `requestDecay` actually be heritable:
 * a body that conducts demand efficiently ends up embedded in longer chains
 * than one that muffles it.
 *
 * Locked bodies still conduct, so a rewrite in progress does not cut the net
 * in two.
 */
export function spreadRequests(list: SlotBody[], adj: WireAdjacency, decay?: number): void {
  const { off, nei } = adj;
  // Indices throughout. Queueing the bodies themselves would need a lookup
  // back to their slot, and a Map keyed on the objects is the allocation this
  // whole structure exists to avoid.
  const q = adj.queue(list.length);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < list.length; i++) {
    if (list[i].request > REQUEST_FLOOR) q[tail++] = i;
  }
  while (head < tail) {
    const at = q[head++];
    const body = list[at];
    // Clamped below 1 for the reason on REQUEST_DECAY: an undecayed field is
    // flat, and a flat field moves nothing.
    const keep = Math.min(0.99, Math.max(0, decay ?? body.requestDecay));
    const next = body.request * keep;
    if (next <= REQUEST_FLOOR) continue;
    for (let k = off[at]; k < off[at + 1]; k++) {
      const ni = nei[k];
      const n = list[ni];
      if (!n || n.request >= next) continue;
      n.request = next;
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

/**
 * One hop of flow down the field, per frame. Returns the energy moved.
 *
 * A body gives to its neediest neighbour, and only to one that is needier
 * than itself, so energy climbs the gradient one wire at a time and a surplus
 * at one end of a net reaches a shortage at the other end over several frames.
 * It gives what it can spare against its own need, capped by what the
 * recipient is actually short of — so nothing is over-delivered and a donor
 * about to be billed itself keeps enough to pay.
 *
 * Where two comparable needs face each other, the fields meet at equal value
 * in the middle, `n.request > d.request` is false on both sides, and nothing
 * crosses. That flat spot is the intended behaviour, not a stall.
 *
 * `onMoved` is called for each transfer, in the order they happen. The sim
 * uses it to recoil the two bodies against each other; nothing about the
 * energy accounting depends on it.
 */
export function flowCharges(
  list: SlotBody[],
  adj: WireAdjacency,
  onMoved?: (from: SlotBody, to: SlotBody, amount: number) => void,
): number {
  const { off, nei } = adj;
  const donors: number[] = [];
  for (let i = 0; i < list.length; i++) {
    const a = list[i];
    if (a.locked) continue;
    if (spareEnergy(a) > FLOW_EPS) donors.push(i);
  }
  // Neediest donor first, so a body that is itself being fed passes on what it
  // does not need in the same frame rather than sitting on it.
  donors.sort((p, q) => list[q].request - list[p].request || list[p].id - list[q].id);
  const taken = new Set<number>();
  let moved = 0;
  for (const di of donors) {
    const d = list[di];
    let best: SlotBody | null = null;
    let bestSlot = -1;
    let bestR = d.request;
    for (let k = off[di]; k < off[di + 1]; k++) {
      const ni = nei[k];
      if (taken.has(ni)) continue;
      const n = list[ni];
      if (!n || n.locked) continue;
      if (n.request > bestR) {
        best = n;
        bestSlot = ni;
        bestR = n.request;
      }
    }
    if (!best) continue;
    // Capped by the recipient's *field* value, not its own need. The field is
    // how much unmet need is visible from there, so a conduit that needs
    // nothing itself still accepts the attenuated demand behind it and the
    // relay works. Capping by local need instead would strand every shortage
    // more than one wire from a donor; capping by nothing at all would send
    // the whole surplus, flip which of the two is the needy one, and leave a
    // wired pair swapping the same unit back and forth every frame.
    //
    // Because the field decays per hop, a distant shortage is fed in smaller
    // increments than a near one. That is the intended shape: demand you can
    // barely see moves less energy than demand next door.
    const give = Math.min(spareEnergy(d), best.request, best.energyCap - best.extra);
    if (give <= FLOW_EPS) continue;
    d.extra -= give;
    best.extra += give;
    taken.add(bestSlot);
    moved += give;
    onMoved?.(d, best, give);
  }
  return moved;
}

/** Store-based twin of `flowCharges` — see `harvestSlotsFast`'s note. */
export function flowChargesFast(
  list: Agent[],
  store: AgentStore,
  adj: WireAdjacency,
  onMoved?: (from: Agent, to: Agent, amount: number) => void,
): number {
  const { off, nei } = adj;
  const LOCKED = store.locked;
  const REQUEST = store.request;
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  const ID = store.id;
  const donors: number[] = [];
  for (let i = 0; i < list.length; i++) {
    const s = list[i].slot;
    if (LOCKED[s]) continue;
    // spareEnergy(a) > FLOW_EPS, inlined: spareEnergy is `extra > 0 ? extra
    // : 0`, and FLOW_EPS > 0, so the comparison is equivalent to extra
    // itself exceeding FLOW_EPS.
    if (EXTRA[s] > FLOW_EPS) donors.push(i);
  }
  // Neediest donor first, so a body that is itself being fed passes on what it
  // does not need in the same frame rather than sitting on it.
  donors.sort(
    (p, q) => REQUEST[list[q].slot] - REQUEST[list[p].slot] || ID[list[p].slot] - ID[list[q].slot],
  );
  const taken = new Set<number>();
  let moved = 0;
  for (const di of donors) {
    const ds = list[di].slot;
    let bestIdx = -1;
    let bestSlot = -1;
    let bestR = REQUEST[ds];
    for (let k = off[di]; k < off[di + 1]; k++) {
      const ni = nei[k];
      if (taken.has(ni)) continue;
      const other = list[ni];
      if (!other) continue;
      const os = other.slot;
      if (LOCKED[os]) continue;
      if (REQUEST[os] > bestR) {
        bestIdx = ni;
        bestSlot = os;
        bestR = REQUEST[os];
      }
    }
    if (bestIdx < 0) continue;
    const spareD = EXTRA[ds] > 0 ? EXTRA[ds] : 0;
    const give = Math.min(spareD, REQUEST[bestSlot], CAP[bestSlot] - EXTRA[bestSlot]);
    if (give <= FLOW_EPS) continue;
    EXTRA[ds] -= give;
    EXTRA[bestSlot] += give;
    taken.add(bestIdx);
    moved += give;
    onMoved?.(list[di], list[bestIdx], give);
  }
  return moved;
}

/**
 * Put a rewrite's released energy back into the net, neediest first.
 *
 * The leftovers and the newborns are where the energy comes out, so it enters
 * the graph there and the ordinary gradient carries it onward over the next
 * few frames. Each body can only take what it has room for — that per-body
 * cap is the bandwidth limit — and an annihilation in a well-fed net releases
 * more than the survivors can hold, so the remainder drops to the ground at
 * the rewrite's midpoint for whoever forages over it later.
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
