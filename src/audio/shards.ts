import type { NetTopology } from './types.ts';

/**
 * Synthesis shards.
 *
 * One waveguide cannot scale past its own voice table: 64 wires, 44 awake
 * bodies. A pair is not worth a dedicated thread, but a pond of pairs will
 * silence each other if they all share one table. Soup slots pack small
 * islands until a table is full, then overflow onto the next. Dedicated
 * slots are sticky homes for machines of `PROMOTE_N` or more.
 *
 * A wire's two agents always share a slot, because they share a component;
 * cross-slot coupling is one-way injects (strikes), never a sample-rate spring.
 */
export const SOUP_COUNT = 3;
/** First soup slot, and the fallback for an unknown id. */
export const SOUP_SLOT = 0;
export const MAX_DEDICATED = 3;
export const SHARD_COUNT = SOUP_COUNT + MAX_DEDICATED;

export function isSoupSlot(slot: number): boolean {
  return slot >= 0 && slot < SOUP_COUNT;
}

export function isDedicatedSlot(slot: number): boolean {
  return slot >= SOUP_COUNT && slot < SHARD_COUNT;
}

/**
 * Bodies in a component before it earns a dedicated worker. A latch pair is
 * 2; a commute that has run a few times is well above this.
 */
export const PROMOTE_N = 8;
/**
 * Stay dedicated down to this so a rewrite that momentarily splits a body
 * off does not bounce the rest back into the soup and kill its delay lines.
 */
export const DEMOTE_N = 5;

/**
 * Agent count at which a soup synth is treated as full.
 *
 * Matches the awake-body ceiling: isolated knockers are one agent each, and
 * packing past this is the same steal the single soup already did. Wires are
 * usually fewer than agents in a pair soup, so the 64-wire table is not the
 * tighter bound.
 */
export const SOUP_PACK_N = 44;

const EMPTY_TOPO: NetTopology = { wires: [], agents: [], tissues: [] };

export function emptyTopology(height?: number): NetTopology {
  return height === undefined ? EMPTY_TOPO : { wires: [], agents: [], tissues: [], height };
}

/**
 * Sticky map from agent id to synth slot.
 *
 * Dedicated incumbents keep their slot while they stay at least `DEMOTE_N`
 * large. Free dedicated slots go to the largest remaining components of
 * `PROMOTE_N` or more. Small islands pack onto soup slots, sticky, filling
 * one table to `SOUP_PACK_N` before opening the next. A slot is never shared
 * across components: after a split the larger piece keeps it and the smaller
 * re-homes. After a merge the plurality slot wins and the other is freed.
 */
export class ShardAssigner {
  private readonly home = new Map<number, number>();

  /** Current slot for an agent; first soup if never seen. */
  slotOf(id: number): number {
    return this.home.get(id) ?? SOUP_SLOT;
  }

  /** Copy of the live assignment. Engine reads this after `assign`. */
  snapshot(): Map<number, number> {
    return new Map(this.home);
  }

  /**
   * Recompute homes from this frame's component roots.
   *
   * `roots` is `graph.componentIds`: agent id → root id. Isolated agents
   * are components of size 1 and pack as soup islands.
   */
  assign(roots: Map<number, number>): Map<number, number> {
    const groups = new Map<number, number[]>();
    for (const [id, root] of roots) {
      const list = groups.get(root);
      if (list) list.push(id);
      else groups.set(root, [id]);
    }

    type Group = { ids: number[]; size: number; incumbent: number };
    const all: Group[] = [];
    for (const ids of groups.values()) {
      const votes = new Map<number, number>();
      for (const id of ids) {
        const h = this.home.get(id);
        if (h === undefined) continue;
        votes.set(h, (votes.get(h) ?? 0) + 1);
      }
      let incumbent = -1;
      let best = 0;
      for (const [slot, n] of votes) {
        if (n > best || (n === best && (incumbent < 0 || slot < incumbent))) {
          best = n;
          incumbent = slot;
        }
      }
      all.push({ ids, size: ids.length, incumbent });
    }

    const takenDedicated = new Set<number>();
    const placed = new Set<Group>();
    const pickDedicated = (): number => {
      for (let s = SOUP_COUNT; s < SHARD_COUNT; s++) if (!takenDedicated.has(s)) return s;
      return -1;
    };

    const incumbents = all
      .filter((g) => isDedicatedSlot(g.incumbent) && g.size >= DEMOTE_N)
      .sort((a, b) => b.size - a.size || a.incumbent - b.incumbent);
    for (const g of incumbents) {
      if (takenDedicated.size >= MAX_DEDICATED) break;
      let slot = g.incumbent;
      if (takenDedicated.has(slot)) slot = pickDedicated();
      if (slot < 0) continue;
      takenDedicated.add(slot);
      placed.add(g);
      for (const id of g.ids) this.home.set(id, slot);
    }

    const newcomers = all
      .filter((g) => !placed.has(g) && g.size >= PROMOTE_N)
      .sort((a, b) => b.size - a.size);
    for (const g of newcomers) {
      const slot = pickDedicated();
      if (slot < 0) break;
      takenDedicated.add(slot);
      placed.add(g);
      for (const id of g.ids) this.home.set(id, slot);
    }

    const load = new Array<number>(SOUP_COUNT).fill(0);
    const soupKeep = all
      .filter((g) => !placed.has(g) && isSoupSlot(g.incumbent))
      .sort((a, b) => b.size - a.size || a.incumbent - b.incumbent);
    for (const g of soupKeep) {
      placed.add(g);
      load[g.incumbent] += g.size;
      for (const id of g.ids) this.home.set(id, g.incumbent);
    }

    const pickSoup = (need: number): number => {
      for (let s = 0; s < SOUP_COUNT; s++) {
        if (load[s] + need <= SOUP_PACK_N) return s;
      }
      let best = 0;
      for (let s = 1; s < SOUP_COUNT; s++) if (load[s] < load[best]) best = s;
      return best;
    };

    const rest = all.filter((g) => !placed.has(g)).sort((a, b) => b.size - a.size);
    for (const g of rest) {
      const slot = pickSoup(g.size);
      load[slot] += g.size;
      for (const id of g.ids) this.home.set(id, slot);
    }

    for (const id of [...this.home.keys()]) {
      if (!roots.has(id)) this.home.delete(id);
    }
    return this.snapshot();
  }
}

/** Split a whole-pond topology into per-slot subsets. Missing slots are empty. */
export function splitTopology(
  topo: NetTopology,
  shardOf: Map<number, number>,
  slotCount = SHARD_COUNT,
): NetTopology[] {
  const parts: NetTopology[] = [];
  for (let s = 0; s < slotCount; s++) {
    parts.push({ wires: [], agents: [], tissues: [], height: topo.height });
  }
  for (const a of topo.agents) {
    const s = shardOf.get(a.id) ?? SOUP_SLOT;
    parts[s < slotCount ? s : SOUP_SLOT].agents.push(a);
  }
  for (const w of topo.wires) {
    const s = shardOf.get(w.agentA) ?? SOUP_SLOT;
    parts[s < slotCount ? s : SOUP_SLOT].wires.push(w);
  }
  if (topo.tissues) {
    for (const t of topo.tissues) {
      let s = SOUP_SLOT;
      for (const a of topo.agents) {
        if (a.tissueId === t.id) {
          s = shardOf.get(a.id) ?? SOUP_SLOT;
          break;
        }
      }
      const part = parts[s < slotCount ? s : SOUP_SLOT];
      if (!part.tissues) part.tissues = [];
      part.tissues.push(t);
    }
  }
  return parts;
}

/** Pairs whose bodies share a slot. Cross-slot springs are dropped. */
export function partitionPairs<T extends { agentA: number; agentB: number }>(
  items: T[],
  shardOf: Map<number, number>,
  slotCount = SHARD_COUNT,
): T[][] {
  const parts: T[][] = [];
  for (let s = 0; s < slotCount; s++) parts.push([]);
  for (const it of items) {
    const sa = shardOf.get(it.agentA) ?? SOUP_SLOT;
    const sb = shardOf.get(it.agentB) ?? SOUP_SLOT;
    if (sa !== sb) continue;
    parts[sa < slotCount ? sa : SOUP_SLOT].push(it);
  }
  return parts;
}

/** Wire scrapes follow the string's slot. Unknown wires stay in the soup. */
export function partitionWirePairs<T extends { wireA: number }>(
  items: T[],
  wireSlot: Map<number, number>,
  slotCount = SHARD_COUNT,
): T[][] {
  const parts: T[][] = [];
  for (let s = 0; s < slotCount; s++) parts.push([]);
  for (const it of items) {
    const s = wireSlot.get(it.wireA) ?? SOUP_SLOT;
    parts[s < slotCount ? s : SOUP_SLOT].push(it);
  }
  return parts;
}

export function wireSlots(topo: NetTopology, shardOf: Map<number, number>): Map<number, number> {
  const out = new Map<number, number>();
  for (const w of topo.wires) out.set(w.id, shardOf.get(w.agentA) ?? SOUP_SLOT);
  return out;
}
