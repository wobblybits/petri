import { describe, expect, it } from 'vitest';
import {
  DEMOTE_N,
  MAX_DEDICATED,
  PROMOTE_N,
  SOUP_PACK_N,
  SOUP_SLOT,
  ShardAssigner,
  isDedicatedSlot,
  isSoupSlot,
  partitionPairs,
  splitTopology,
  wireSlots,
} from './shards.ts';
import type { NetTopology } from './types.ts';

function rootsOf(groups: number[][]): Map<number, number> {
  const roots = new Map<number, number>();
  for (const g of groups) {
    const root = g[0];
    for (const id of g) roots.set(id, root);
  }
  return roots;
}

function slots(a: ShardAssigner, ids: number[]): number[] {
  return ids.map((id) => a.slotOf(id));
}

describe('ShardAssigner', () => {
  it('keeps small islands in the soup', () => {
    const a = new ShardAssigner();
    a.assign(rootsOf([[1, 2], [3], [4, 5, 6]]));
    expect(slots(a, [1, 2, 3, 4, 5, 6]).every(isSoupSlot)).toBe(true);
    expect(new Set(slots(a, [1, 2, 3, 4, 5, 6])).size).toBe(1);
  });

  it('promotes a self-contained net once it is large enough', () => {
    const a = new ShardAssigner();
    const big = Array.from({ length: PROMOTE_N }, (_, i) => i + 1);
    a.assign(rootsOf([big, [100, 101]]));
    const home = a.slotOf(1);
    expect(isDedicatedSlot(home)).toBe(true);
    for (const id of big) expect(a.slotOf(id)).toBe(home);
    expect(a.slotOf(100)).toBe(SOUP_SLOT);
  });

  it('keeps the same slot while the net grows', () => {
    const a = new ShardAssigner();
    const first = Array.from({ length: PROMOTE_N }, (_, i) => i + 1);
    a.assign(rootsOf([first]));
    const home = a.slotOf(1);
    const grown = first.concat([first.length + 1, first.length + 2]);
    a.assign(rootsOf([grown]));
    for (const id of grown) expect(a.slotOf(id)).toBe(home);
  });

  it('does not bounce a dedicated net that shrinks to DEMOTE_N', () => {
    const a = new ShardAssigner();
    const big = Array.from({ length: PROMOTE_N + 2 }, (_, i) => i + 1);
    a.assign(rootsOf([big]));
    const home = a.slotOf(1);
    const kept = big.slice(0, DEMOTE_N);
    const leftover = big.slice(DEMOTE_N);
    a.assign(rootsOf([kept, leftover.slice(0, 2), leftover.slice(2)]));
    for (const id of kept) expect(a.slotOf(id)).toBe(home);
    for (const id of leftover) expect(isSoupSlot(a.slotOf(id))).toBe(true);
  });

  it('returns a net to the soup once it is no longer a machine', () => {
    const a = new ShardAssigner();
    const big = Array.from({ length: PROMOTE_N }, (_, i) => i + 1);
    a.assign(rootsOf([big]));
    expect(isDedicatedSlot(a.slotOf(1))).toBe(true);
    a.assign(rootsOf([[1, 2], [3, 4], big.slice(4)]));
    expect(isSoupSlot(a.slotOf(1))).toBe(true);
  });

  it('merges two dedicated nets onto one slot', () => {
    const a = new ShardAssigner();
    const left = Array.from({ length: PROMOTE_N }, (_, i) => i + 1);
    const right = Array.from({ length: PROMOTE_N }, (_, i) => i + 100);
    a.assign(rootsOf([left, right]));
    const sL = a.slotOf(left[0]);
    const sR = a.slotOf(right[0]);
    expect(sL).not.toBe(sR);
    expect(isDedicatedSlot(sL)).toBe(true);
    expect(isDedicatedSlot(sR)).toBe(true);

    a.assign(rootsOf([left.concat(right)]));
    const merged = a.slotOf(left[0]);
    expect(isDedicatedSlot(merged)).toBe(true);
    for (const id of left.concat(right)) expect(a.slotOf(id)).toBe(merged);
    // One dedicated slot must have been freed — a third net can take it.
    const third = Array.from({ length: PROMOTE_N }, (_, i) => i + 200);
    a.assign(rootsOf([left.concat(right), third]));
    expect(isDedicatedSlot(a.slotOf(third[0]))).toBe(true);
    expect(a.slotOf(third[0])).not.toBe(merged);
  });

  it('gives a split-off machine its own slot when one is free', () => {
    const a = new ShardAssigner();
    const all = Array.from({ length: PROMOTE_N * 2 + 2 }, (_, i) => i + 1);
    a.assign(rootsOf([all]));
    const home = a.slotOf(1);
    const keep = all.slice(0, PROMOTE_N + 2);
    const other = all.slice(PROMOTE_N + 2);
    expect(other.length).toBeGreaterThanOrEqual(PROMOTE_N);
    a.assign(rootsOf([keep, other]));
    expect(a.slotOf(keep[0])).toBe(home);
    expect(isDedicatedSlot(a.slotOf(other[0]))).toBe(true);
    expect(a.slotOf(other[0])).not.toBe(home);
  });

  it('does not steal a living dedicated slot for a slightly larger newcomer', () => {
    const a = new ShardAssigner();
    const nets: number[][] = [];
    for (let n = 0; n < MAX_DEDICATED; n++) {
      nets.push(Array.from({ length: PROMOTE_N }, (_, i) => n * 100 + i + 1));
    }
    a.assign(rootsOf(nets));
    const homes = nets.map((g) => a.slotOf(g[0]));
    expect(new Set(homes).size).toBe(MAX_DEDICATED);

    const huge = Array.from({ length: PROMOTE_N + 6 }, (_, i) => 1000 + i);
    a.assign(rootsOf([...nets, huge]));
    for (let n = 0; n < MAX_DEDICATED; n++) {
      expect(a.slotOf(nets[n][0])).toBe(homes[n]);
    }
    expect(isSoupSlot(a.slotOf(huge[0]))).toBe(true);
  });

  it('forgets agents that left the graph', () => {
    const a = new ShardAssigner();
    const big = Array.from({ length: PROMOTE_N }, (_, i) => i + 1);
    a.assign(rootsOf([big]));
    a.assign(rootsOf([big.slice(0, PROMOTE_N - 1)]));
    expect(a.snapshot().has(PROMOTE_N)).toBe(false);
  });

  it('overflows small islands onto the next soup slot once one table is full', () => {
    const a = new ShardAssigner();
    const pairs: number[][] = [];
    const nPairs = Math.floor(SOUP_PACK_N / 2) + 3;
    for (let i = 0; i < nPairs; i++) pairs.push([i * 2 + 1, i * 2 + 2]);
    a.assign(rootsOf(pairs));
    const first = a.slotOf(1);
    const overflow = a.slotOf(pairs[nPairs - 1][0]);
    expect(isSoupSlot(first)).toBe(true);
    expect(isSoupSlot(overflow)).toBe(true);
    expect(overflow).not.toBe(first);
    expect(a.slotOf(1)).toBe(a.slotOf(2));
  });

  it('does not bounce a soup island when an earlier slot has room again', () => {
    const a = new ShardAssigner();
    const pairs: number[][] = [];
    const nPairs = Math.floor(SOUP_PACK_N / 2) + 2;
    for (let i = 0; i < nPairs; i++) pairs.push([i * 2 + 1, i * 2 + 2]);
    a.assign(rootsOf(pairs));
    const overflow = pairs[nPairs - 1];
    const home = a.slotOf(overflow[0]);
    expect(home).not.toBe(SOUP_SLOT);
    a.assign(rootsOf(pairs.slice(4)));
    expect(a.slotOf(overflow[0])).toBe(home);
    expect(a.slotOf(overflow[1])).toBe(home);
  });

  it('merges two soup islands onto the plurality slot', () => {
    const a = new ShardAssigner();
    const pairs: number[][] = [];
    const nPairs = Math.floor(SOUP_PACK_N / 2) + 1;
    for (let i = 0; i < nPairs; i++) pairs.push([i * 2 + 1, i * 2 + 2]);
    a.assign(rootsOf(pairs));
    const stay = [...pairs[0], ...pairs[1]];
    const overflow = pairs[nPairs - 1];
    expect(a.slotOf(stay[0])).toBe(SOUP_SLOT);
    expect(isSoupSlot(a.slotOf(overflow[0]))).toBe(true);
    expect(a.slotOf(overflow[0])).not.toBe(SOUP_SLOT);
    a.assign(rootsOf([[...stay, ...overflow]]));
    expect(a.slotOf(overflow[0])).toBe(SOUP_SLOT);
    expect(a.slotOf(stay[0])).toBe(SOUP_SLOT);
  });
});

describe('splitTopology', () => {
  function topo(agentIds: number[], wires: [number, number, number][]): NetTopology {
    return {
      height: 0.4,
      agents: agentIds.map((id) => ({ id, kind: 0 as const, openPorts: 0, impedance: 1 })),
      wires: wires.map(([id, a, b]) => ({
        id,
        length: 40,
        loss: 0.99,
        bend: 0,
        agentA: a,
        agentB: b,
      })),
    };
  }

  it('puts a dedicated net on its own subset and leaves soup behind', () => {
    const assigner = new ShardAssigner();
    const big = Array.from({ length: PROMOTE_N }, (_, i) => i + 1);
    const shardOf = assigner.assign(rootsOf([big, [100, 101]]));
    const home = assigner.slotOf(1);
    const t = topo(
      [...big, 100, 101],
      [
        [1, 1, 2],
        [2, 100, 101],
      ],
    );
    const parts = splitTopology(t, shardOf);
    expect(parts[home].agents.map((a) => a.id).sort((x, y) => x - y)).toEqual(big);
    expect(parts[home].wires.map((w) => w.id)).toEqual([1]);
    expect(parts[SOUP_SLOT].agents.map((a) => a.id).sort((x, y) => x - y)).toEqual([100, 101]);
    expect(parts[SOUP_SLOT].wires.map((w) => w.id)).toEqual([2]);
    expect(parts[home].height).toBe(0.4);
  });

  it('keeps a collapsed mesh on the same shard as its bodies', () => {
    const assigner = new ShardAssigner();
    const big = Array.from({ length: PROMOTE_N }, (_, i) => i + 1);
    const shardOf = assigner.assign(rootsOf([big, [100, 101]]));
    const home = assigner.slotOf(1);
    const t: NetTopology = {
      height: 0.4,
      agents: [
        ...big.map((id) => ({
          id,
          kind: 1 as const,
          openPorts: 0,
          impedance: 1,
          tissue: id > 2,
          tissueId: 1,
        })),
        { id: 100, kind: 0 as const, openPorts: 0, impedance: 1 },
        { id: 101, kind: 0 as const, openPorts: 0, impedance: 1 },
      ],
      wires: [{ id: 9, length: 40, loss: 0.99, bend: 0, agentA: 1, agentB: 2 }],
      tissues: [{ id: 1, n: PROMOTE_N, delay: 48 }],
    };
    const parts = splitTopology(t, shardOf);
    expect(parts[home].tissues?.map((x) => x.id)).toEqual([1]);
    expect(parts[SOUP_SLOT].tissues ?? []).toEqual([]);
  });

  it('drops cross-slot contacts so Hertzian stays inside one synth', () => {
    const shardOf = new Map<number, number>([
      [1, 1],
      [2, 1],
      [3, 0],
      [4, 0],
    ]);
    const parts = partitionPairs(
      [
        { agentA: 1, agentB: 2, load: 1 },
        { agentA: 1, agentB: 3, load: 1 },
        { agentA: 3, agentB: 4, load: 1 },
      ],
      shardOf,
    );
    expect(parts[1]).toEqual([{ agentA: 1, agentB: 2, load: 1 }]);
    expect(parts[0]).toEqual([{ agentA: 3, agentB: 4, load: 1 }]);
  });

  it('maps a wire to the slot of its bodies', () => {
    const shardOf = new Map([
      [1, 2],
      [2, 2],
    ]);
    const t = topo([1, 2], [[9, 1, 2]]);
    expect(wireSlots(t, shardOf).get(9)).toBe(2);
  });
});
