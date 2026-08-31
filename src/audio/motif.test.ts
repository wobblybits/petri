import { describe, expect, it } from 'vitest';
import { CLOCK_MAX, CLOCK_MIN, MESH_MIN, clockRoots, median, meshRoots, skinAgents } from './motif.ts';
import { buildTopology } from './topology.ts';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { slotsFor } from '../agents.ts';
import { Sim } from '../sim.ts';
import { LOD_NEAR } from './lod.ts';
import type { PanView } from './types.ts';

const farView = (x = 400, y = 300): PanView => ({
  x,
  y,
  zoom: 0.05,
  viewW: 800,
  viewH: 600,
});

const closeView = (x: number, y: number): PanView => ({
  x,
  y,
  zoom: 4,
  viewW: 800,
  viewH: 600,
});

function chain(n: number, kind: 'dup' | 'con' | 'era' = 'dup'): Sim {
  const sim = new Sim(1600, 400);
  const params = defaultParams();
  params.spawnInterval = 0;
  params.maxAgents = n + 4;
  const made = [];
  for (let i = 0; i < n; i++) {
    made.push(sim.spawn(kind, 40 + i * 36, 180, 0, params, true)!);
  }
  for (let i = 0; i < n - 1; i++) {
    sim.wire(made[i].id, i === 0 ? 'p' : 'l', made[i + 1].id, 'p', params);
  }
  return sim;
}

describe('meshRoots', () => {
  it('a pair is not a mesh', () => {
    const roots = new Map([
      [1, 1],
      [2, 1],
    ]);
    expect(meshRoots(roots, () => 1).size).toBe(0);
  });

  it('eight Dup/Con bodies are a mesh', () => {
    const roots = new Map<number, number>();
    for (let i = 1; i <= MESH_MIN; i++) roots.set(i, 1);
    expect([...meshRoots(roots, () => 1)]).toEqual([1]);
  });

  it('an Era in the mix is not a mesh', () => {
    const roots = new Map<number, number>();
    for (let i = 1; i <= MESH_MIN; i++) roots.set(i, 1);
    const kindOf = (id: number) => (id === 3 ? 0 : 1);
    expect(meshRoots(roots, kindOf).size).toBe(0);
  });
});

describe('clockRoots', () => {
  it('the oscillator signature is a closed mixed machine', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    const roots = sim.graph.componentIds(sim.agents);
    const clocks = clockRoots(
      roots,
      (id) => {
        const a = sim.agents.get(id)!;
        return a.kind === 'era' ? 0 : a.kind === 'dup' ? 1 : 2;
      },
      (id) => {
        const a = sim.agents.get(id)!;
        let n = 0;
        for (const slot of slotsFor(a.kind)) {
          if (sim.graph.isFree({ id, slot })) n++;
        }
        return n;
      },
    );
    expect(clocks.size).toBe(1);
  });

  it('a pair and an open forager are not clocks', () => {
    const roots = new Map([
      [1, 1],
      [2, 1],
      [3, 3],
    ]);
    expect(clockRoots(roots, () => 0, () => 0).size).toBe(0);
    const big = new Map<number, number>();
    for (let i = 1; i <= CLOCK_MIN; i++) big.set(i, 1);
    expect(clockRoots(big, () => 0, () => 1).size).toBe(0);
    expect(CLOCK_MAX).toBeGreaterThan(CLOCK_MIN);
  });
});

describe('skinAgents', () => {
  it('promotes one hop of neighbours, not a flood', () => {
    const roots = new Map<number, number>();
    const wires = [];
    for (let i = 1; i <= 8; i++) roots.set(i, 1);
    for (let i = 1; i < 8; i++) wires.push({ agentA: i, agentB: i + 1 });
    const skin = skinAgents(new Set([1]), roots, [1], wires);
    expect([...skin].sort((a, b) => a - b)).toEqual([1, 2]);
  });

  it('bodies outside a mesh stay skin', () => {
    const roots = new Map([
      [1, 1],
      [2, 1],
      [9, 9],
    ]);
    for (let i = 1; i <= 8; i++) roots.set(i, 1);
    const skin = skinAgents(new Set([1]), roots, [], []);
    expect(skin.has(9)).toBe(true);
    expect(skin.has(1)).toBe(false);
  });
});

describe('median', () => {
  it('is the middle of a sorted list', () => {
    expect(median([])).toBe(0);
    expect(median([4])).toBe(4);
    expect(median([1, 3, 9])).toBe(3);
    expect(median([2, 8])).toBe(5);
  });
});

describe('mesh collapse', () => {
  it('without a camera, a large Dup chain keeps every wire', () => {
    const sim = chain(16);
    const topo = buildTopology(sim.graph, sim.agents, null);
    expect(topo.wires).toHaveLength(15);
    expect(topo.tissues).toEqual([]);
    expect(topo.agents.every((a) => !a.tissue)).toBe(true);
  });

  it('zoomed out, the interior becomes one tissue resonator', () => {
    const sim = chain(16);
    const topo = buildTopology(sim.graph, sim.agents, farView());
    expect(topo.tissues).toHaveLength(1);
    expect(topo.tissues![0].n).toBe(16);
    expect(topo.wires).toHaveLength(0);
    expect(topo.agents.every((a) => a.tissue)).toBe(true);
  });

  it('a pair stays explicit even when far away', () => {
    const sim = chain(2);
    const topo = buildTopology(sim.graph, sim.agents, farView());
    expect(topo.tissues).toEqual([]);
    expect(topo.wires).toHaveLength(1);
  });

  it('an Era in the component blocks collapse', () => {
    const sim = chain(8, 'dup');
    const params = defaultParams();
    params.spawnInterval = 0;
    const last = [...sim.agents.values()].at(-1)!;
    const era = sim.spawn('era', last.x + 36, last.y, 0, params, true)!;
    sim.wire(last.id, 'l', era.id, 'p', params);
    const topo = buildTopology(sim.graph, sim.agents, farView());
    expect(topo.tissues).toEqual([]);
    expect(topo.wires.length).toBeGreaterThanOrEqual(8);
  });

  it('the cell under the cursor stays a string, one hop, not the whole net', () => {
    const sim = chain(16);
    const first = [...sim.agents.values()][0];
    const topo = buildTopology(sim.graph, sim.agents, closeView(first.x, first.y));
    expect(topo.tissues).toHaveLength(1);
    expect(topo.wires.length).toBeGreaterThan(0);
    expect(topo.wires.length).toBeLessThan(8);
    const skin = topo.agents.filter((a) => !a.tissue);
    const tissue = topo.agents.filter((a) => a.tissue);
    expect(skin.length).toBeGreaterThan(0);
    expect(tissue.length).toBeGreaterThan(skin.length);
  });

  it('a gesture force-skins a far cell so its wire is not omitted', () => {
    const sim = chain(16);
    const agents = [...sim.agents.values()];
    const a = agents[7];
    const b = agents[8];
    const far = buildTopology(sim.graph, sim.agents, farView());
    expect(far.wires).toHaveLength(0);
    const kept = buildTopology(sim.graph, sim.agents, farView(), null, new Set([a.id, b.id]));
    expect(kept.wires.length).toBeGreaterThan(0);
    expect(kept.agents.find((x) => x.id === a.id)?.tissue).toBe(false);
    expect(kept.agents.find((x) => x.id === b.id)?.tissue).toBe(false);
  });
});

describe('clock voicing', () => {
  it('looking at one oscillator cell voices the whole machine', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    const era = [...sim.agents.values()].find((a) => a.kind === 'era')!;
    const topo = buildTopology(sim.graph, sim.agents, closeView(era.x, era.y));
    expect(topo.agents.every((a) => a.lod === LOD_NEAR)).toBe(true);
    expect(topo.wires.every((w) => w.lod === LOD_NEAR)).toBe(true);
  });
});
