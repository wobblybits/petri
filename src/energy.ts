import type { Agent, AgentKind } from './agents.ts';
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
 * is most of why pumping is slow.
 */
export const EXTRA_CAP = 1.25;
/** A whole unit of debt. The body dies. */
export const EXTRA_FLOOR = -1;
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
 * How much of a neighbour's need reaches you, per hop.
 *
 * The field is `max(own need, best neighbour's field x DECAY)`, so need spreads
 * as a decaying scent rather than a hop count. Two consequences matter. Need
 * competes by magnitude, so a starving body outpulls a redex that is nearly
 * paid for even from further away. And where two comparable needs face each
 * other across a net the fields meet at equal value, the local gradient there
 * is flat, and nothing crosses — the watershed sits wherever the needs happen
 * to balance instead of at a fixed hop count.
 */
export const REQUEST_DECAY = 0.8;

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
export function atCap(a: { extra: number }): boolean {
  return a.extra >= EXTRA_CAP - EXTRA_FULL_EPS;
}

/** Able to pay its side of a rewrite. Not the same as being full. */
export function canPayShare(a: { extra: number }): boolean {
  return a.extra >= REWRITE_SHARE - EXTRA_FULL_EPS;
}

/** New latches (snap / drag-wire) only. Rewrite leftovers ignore this. */
export function canLatch(_a: { extra: number }): boolean {
  return true;
}

/** Metabolic debt. At −1 the body dies. */
export function isStarving(a: { extra: number }): boolean {
  return a.extra < 0;
}

export function extrasOf(a: { extra: number }, b: { extra: number }): number {
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

function cellKey(i: number, j: number): string {
  return `${i},${j}`;
}

/**
 * Sparse world-space energy. Unvisited cells hold `ambient`; once a cell is
 * touched, whatever remains is stored explicitly (including 0).
 * No decay or diffusion — occupancy is the only transport.
 */
export class EnergyGrid {
  cellSize: number;
  ambient: number;
  private readonly cells = new Map<string, number>();

  constructor(cellSize: number, ambient: number) {
    this.cellSize = Math.max(1, cellSize);
    this.ambient = Math.max(0, ambient);
  }

  clear(): void {
    this.cells.clear();
  }

  configure(cellSize: number, ambient: number): void {
    this.cellSize = Math.max(1, cellSize);
    this.ambient = Math.max(0, ambient);
  }

  index(x: number, y: number): { i: number; j: number; key: string } {
    const i = Math.floor(x / this.cellSize);
    const j = Math.floor(y / this.cellSize);
    return { i, j, key: cellKey(i, j) };
  }

  getCell(i: number, j: number): number {
    const key = cellKey(i, j);
    return this.cells.has(key) ? (this.cells.get(key) ?? 0) : this.ambient;
  }

  getAt(x: number, y: number): number {
    const { i, j } = this.index(x, y);
    return this.getCell(i, j);
  }

  /** Take up to `n` from a cell and return how much was taken. */
  take(key: string, n: number): number {
    const want = Math.max(0, n);
    const have = this.cells.has(key) ? (this.cells.get(key) ?? 0) : this.ambient;
    const got = Math.min(have, want);
    this.cells.set(key, have - got);
    return got;
  }

  addAt(x: number, y: number, amount: number): void {
    if (amount === 0) return;
    const { key } = this.index(x, y);
    this.cells.set(key, (this.cells.has(key) ? (this.cells.get(key) ?? 0) : this.ambient) + amount);
  }

  setCell(i: number, j: number, amount: number): void {
    this.cells.set(cellKey(i, j), amount);
  }

  /** Sum of explicitly stored cells (not implicit ambient). */
  storedTotal(): number {
    let s = 0;
    for (const v of this.cells.values()) s += v;
    return s;
  }

  /** Touched cells only — the implicit ambient field is not stored. */
  forEachStored(fn: (i: number, j: number, e: number) => void): void {
    for (const [key, e] of this.cells) {
      const c = key.indexOf(',');
      fn(+key.slice(0, c), +key.slice(c + 1), e);
    }
  }
}

export type SlotBody = Pick<Agent, 'id' | 'kind' | 'x' | 'y' | 'extra' | 'locked' | 'request'>;

/**
 * Unlocked agents below cap take from their cell, in id order, up to what
 * they still have room for. Ambient 0.1 therefore takes ten cells to fill.
 */
export function harvestSlots(agents: Iterable<SlotBody>, grid: EnergyGrid): void {
  const hungry = new Map<string, SlotBody[]>();
  for (const a of agents) {
    if (a.locked || atCap(a)) continue;
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
      const room = EXTRA_CAP - a.extra;
      if (room <= EXTRA_FULL_EPS) continue;
      const got = grid.take(key, room);
      if (got <= 0) break;
      a.extra = Math.min(EXTRA_CAP, a.extra + got);
    }
  }
}

/**
 * Bill every body continuously. `rate` is energy per second, per kind via
 * `upkeepRateFor`, so a full body has `1/rate` seconds of life in hand and
 * slides smoothly into debt after that.
 *
 * A negative rate pays the body instead, capped at a full extra like every
 * other source. Returns the ids that reached the floor — a whole unit of
 * debt, which the caller treats as death rather than a detachment.
 */
export function tickUpkeep(agents: Iterable<SlotBody>, dt: number, rate: number): number[] {
  if (!(dt > 0)) return [];
  const dead: number[] = [];
  for (const a of agents) {
    if (a.locked) continue;
    const r = upkeepRateFor(a.kind, rate);
    if (r === 0) continue;
    const was = a.extra;
    a.extra = Math.min(EXTRA_CAP, Math.max(EXTRA_FLOOR, a.extra - r * dt));
    if (was > EXTRA_FLOOR && a.extra <= EXTRA_FLOOR) dead.push(a.id);
  }
  return dead;
}

/**
 * How much energy this body is in debt by, and therefore how badly it wants
 * some. Zero for anything at or above break-even; one at the point of death.
 */
export function hungerNeed(a: SlotBody): number {
  return a.extra < 0 ? -a.extra : 0;
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

/** Raise a body's own need. The field never lowers what is already there. */
export function seedRequest(agent: SlotBody, amount: number): void {
  if (amount > agent.request) agent.request = amount;
}

/**
 * Relax the need field over the wire graph until it stops improving.
 *
 * Every body ends up holding the largest need it can see, attenuated by
 * `REQUEST_DECAY` per hop — so the field is a potential whose gradient points
 * at whoever is neediest, weighted by how badly and discounted by how far.
 * A body can be improved more than once (a bigger need further away can beat
 * a small one next door), so this is a relaxation rather than one BFS sweep;
 * the decay bounds it, since a value below `REQUEST_FLOOR` stops travelling.
 *
 * Locked bodies still conduct, so a rewrite in progress does not cut the net
 * in two.
 */
export function spreadRequests(list: SlotBody[], adj: WireAdjacency): void {
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
    const next = list[at].request * REQUEST_DECAY;
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
    const give = Math.min(spareEnergy(d), best.request, EXTRA_CAP - best.extra);
    if (give <= FLOW_EPS) continue;
    d.extra -= give;
    best.extra += give;
    taken.add(bestSlot);
    moved += give;
    onMoved?.(d, best, give);
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
    const room = EXTRA_CAP - a.extra;
    if (room <= EXTRA_FULL_EPS) continue;
    const give = Math.min(left, room);
    a.extra += give;
    left -= give;
  }
  if (left > 0) grid.addAt(x, y, left);
}
