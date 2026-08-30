import { describe, expect, it } from 'vitest';
import {
  AGENT_BAND,
  HYSTERESIS,
  LOD_FAR,
  LOD_MID,
  LOD_NEAR,
  LodSelector,
  WIRE_BAND,
  COST_US,
  VOICE_BUDGET_US,
  agentKey,
  apparentPx,
  assign,
  assignedCostUs,
  onScreen,
  tierFor,
  wireKey,
  type LodCandidate,
} from './lod.ts';
import { buildTopology } from './topology.ts';
import { defaultParams } from '../params.ts';
import { Sim } from '../sim.ts';
import type { PanView } from './types.ts';

const view = (zoom: number, x = 400, y = 300): PanView => ({
  x,
  y,
  zoom,
  viewW: 800,
  viewH: 600,
});

describe('apparent size', () => {
  it('scales with zoom, and stands in for world size without a camera', () => {
    expect(apparentPx(100, view(1))).toBe(100);
    expect(apparentPx(100, view(4))).toBe(400);
    expect(apparentPx(100, view(0.25))).toBe(25);
    expect(apparentPx(100, null)).toBe(100);
  });

  it('counts a wire straddling the edge as on screen', () => {
    // Centre 40 px past the right edge, but the wire is 200 px long, so half
    // of it is still in view.
    expect(onScreen(400 + 440, 300, 200, view(1))).toBe(true);
    // Same offset, but a small body has nothing left inside.
    expect(onScreen(400 + 440, 300, 8, view(1))).toBe(false);
  });

  it('treats a missing camera as everything visible', () => {
    expect(onScreen(1e6, 1e6, 1, null)).toBe(true);
  });
});

describe('tier thresholds', () => {
  it('splits on apparent size', () => {
    expect(tierFor(WIRE_BAND.near, true, WIRE_BAND)).toBe(LOD_NEAR);
    expect(tierFor(WIRE_BAND.near - 1, true, WIRE_BAND)).toBe(LOD_MID);
    expect(tierFor(WIRE_BAND.mid, true, WIRE_BAND)).toBe(LOD_MID);
    expect(tierFor(WIRE_BAND.mid - 1, true, WIRE_BAND)).toBe(LOD_FAR);
  });

  it('sends anything off screen to the ensemble whatever its size', () => {
    expect(tierFor(10_000, false, WIRE_BAND)).toBe(LOD_FAR);
  });
});

describe('hysteresis', () => {
  it('holds a wire sitting exactly on a threshold', () => {
    const lod = new LodSelector();
    const key = wireKey(1);
    // Starts NEAR at the threshold.
    expect(lod.tier(key, WIRE_BAND.near, true, WIRE_BAND)).toBe(LOD_NEAR);
    // Jitter around the line must not move it either way.
    for (let i = 0; i < 20; i++) {
      const px = WIRE_BAND.near + (i % 2 === 0 ? -1 : 1);
      expect(lod.tier(key, px, true, WIRE_BAND)).toBe(LOD_NEAR);
    }
  });

  it('demotes only past the lower edge of the band, and promotes past the upper', () => {
    const lod = new LodSelector();
    const key = wireKey(1);
    lod.tier(key, WIRE_BAND.near, true, WIRE_BAND);
    // Just inside the band: still NEAR.
    expect(lod.tier(key, WIRE_BAND.near * (1 - HYSTERESIS) + 1, true, WIRE_BAND)).toBe(LOD_NEAR);
    // Past it: MID.
    expect(lod.tier(key, WIRE_BAND.near * (1 - HYSTERESIS) - 1, true, WIRE_BAND)).toBe(LOD_MID);
    // Coming back needs the upper edge, not the threshold itself.
    expect(lod.tier(key, WIRE_BAND.near + 1, true, WIRE_BAND)).toBe(LOD_MID);
    expect(lod.tier(key, WIRE_BAND.near * (1 + HYSTERESIS) + 1, true, WIRE_BAND)).toBe(LOD_NEAR);
  });

  it('drops off screen immediately — no hysteresis holds a wire you cannot see', () => {
    const lod = new LodSelector();
    const key = wireKey(1);
    lod.tier(key, 400, true, WIRE_BAND);
    expect(lod.tier(key, 400, false, WIRE_BAND)).toBe(LOD_FAR);
  });

  it('forgets ids that stopped being tiered', () => {
    const lod = new LodSelector();
    lod.tier(wireKey(1), 100, true, WIRE_BAND);
    lod.tier(wireKey(2), 100, true, WIRE_BAND);
    lod.sweep();
    expect(lod.size).toBe(2);
    lod.tier(wireKey(1), 100, true, WIRE_BAND);
    lod.sweep();
    expect(lod.size).toBe(1);
  });

  it('keeps wire and agent id spaces apart', () => {
    expect(wireKey(3)).not.toBe(agentKey(3));
  });
});

describe('tiers through buildTopology', () => {
  function soup(n: number): { sim: Sim; params: ReturnType<typeof defaultParams> } {
    const sim = new Sim(1600, 1200);
    const params = defaultParams();
    params.spawnInterval = 0;
    params.maxAgents = n;
    const kinds = ['era', 'dup', 'con'] as const;
    const made = [];
    for (let i = 0; i < n; i++) {
      made.push(sim.spawn(kinds[i % 3], 60 + (i % 12) * 90, 60 + Math.floor(i / 12) * 90, 0, params, true)!);
    }
    for (let i = 0; i < made.length - 1; i += 2) {
      const a = made[i];
      const b = made[i + 1];
      sim.wire(a.id, 'p', b.id, 'p', params);
    }
    return { sim, params };
  }

  /** `n` agents packed into a fixed patch of world, so density rises with n. */
  function packed(n: number, w: number, h: number): { sim: Sim } {
    const sim = new Sim(1600, 1200);
    const params = defaultParams();
    params.spawnInterval = 0;
    params.maxAgents = n;
    const kinds = ['era', 'dup', 'con'] as const;
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    const made = [];
    for (let i = 0; i < n; i++) {
      const x = 400 - w / 2 + ((i % cols) + 0.5) * (w / cols);
      const y = 300 - h / 2 + (Math.floor(i / cols) + 0.5) * (h / rows);
      made.push(sim.spawn(kinds[i % 3], x, y, 0, params, true)!);
    }
    for (let i = 0; i + 1 < made.length; i += 2) {
      sim.wire(made[i].id, 'p', made[i + 1].id, 'p', params);
    }
    return { sim };
  }

  it('zoomed in, the wires on screen are NEAR', () => {
    const { sim } = soup(24);
    const topo = buildTopology(sim.graph, sim.agents, view(4, 100, 100));
    const near = topo.wires.filter((w) => w.lod === LOD_NEAR);
    expect(near.length).toBeGreaterThan(0);
  });

  it('zoomed far out, nothing is NEAR and the tail is all FAR', () => {
    const { sim } = soup(24);
    const topo = buildTopology(sim.graph, sim.agents, view(0.05));
    expect(topo.wires.every((w) => w.lod === LOD_FAR)).toBe(true);
    expect(topo.agents.every((a) => a.lod === LOD_FAR)).toBe(true);
  });

  it('panning away from a cluster demotes it without changing the net', () => {
    const { sim } = soup(24);
    const here = buildTopology(sim.graph, sim.agents, view(2, 200, 200));
    const away = buildTopology(sim.graph, sim.agents, view(2, 90_000, 90_000));
    expect(here.wires.length).toBe(away.wires.length);
    expect(here.wires.some((w) => w.lod !== LOD_FAR)).toBe(true);
    expect(away.wires.every((w) => w.lod === LOD_FAR)).toBe(true);
  });

  it('stops resolving wires once density shrinks them past the threshold', () => {
    // Same patch of world, same camera, more and more agents packed in. Wires
    // get shorter as the net gets denser, and once they fall under the NEAR
    // threshold they stop being resolvable individually.
    //
    // Note what this does NOT show: that the NEAR set is bounded. Wires may
    // overlap, so a tangle of long wires can be entirely NEAR at once. Size
    // alone is a stable, perceptually sensible criterion, not a cost bound —
    // that takes the ranked budget in assign().
    const v = view(1);
    const nearOf = (n: number) => {
      const { sim } = packed(n, 700, 500);
      const topo = buildTopology(sim.graph, sim.agents, v);
      return topo.wires.filter((w) => w.lod === LOD_NEAR).length;
    };
    expect(nearOf(12)).toBeGreaterThan(0);
    expect(nearOf(400)).toBe(0);
  });

  it('agents use their own band, so a body is not judged as if it were a wire', () => {
    const { sim } = soup(8);
    const topo = buildTopology(sim.graph, sim.agents, view(1, 200, 200));
    // A body is ~24 world px across, under WIRE_BAND.near but over AGENT_BAND.near.
    expect(AGENT_BAND.near).toBeLessThan(WIRE_BAND.near);
    expect(topo.agents.some((a) => a.lod === LOD_NEAR)).toBe(true);
  });
});

describe('ranked budget', () => {
  const wireCand = (id: number, px: number): LodCandidate => ({
    key: wireKey(id),
    px,
    visible: true,
    band: WIRE_BAND,
    nearUs: COST_US.nearWire,
    midUs: COST_US.midWire,
  });

  it('never spends more than the budget', () => {
    const cands = [];
    for (let i = 0; i < 400; i++) cands.push(wireCand(i, 200));
    const tiers = assign(cands, null, 500);
    expect(assignedCostUs(cands, tiers)).toBeLessThanOrEqual(500);
  });

  it('spends it on the biggest things first', () => {
    const cands = [wireCand(1, 60), wireCand(2, 500), wireCand(3, 300)];
    // Room for exactly one NEAR wire.
    const tiers = assign(cands, null, COST_US.nearWire + COST_US.midWire * 2);
    expect(tiers.get(wireKey(2))).toBe(LOD_NEAR);
    expect(tiers.get(wireKey(3))).toBe(LOD_MID);
    expect(tiers.get(wireKey(1))).toBe(LOD_MID);
  });

  it('demotes rather than mutes when the budget runs out', () => {
    const cands = [];
    for (let i = 0; i < 200; i++) cands.push(wireCand(i, 200));
    const tiers = assign(cands, null, 40);
    // Everything still has a tier — nothing is dropped from the mix.
    expect(tiers.size).toBe(200);
    for (const c of cands) expect(tiers.get(c.key)).toBeDefined();
  });

  it('does not let a tangle of long wires blow the budget', () => {
    // The case fixed thresholds get wrong: 200 overlapping full-size wires,
    // all of them individually deserving NEAR.
    const cands = [];
    for (let i = 0; i < 200; i++) cands.push(wireCand(i, 400));
    const naive = cands.length * COST_US.nearWire;
    const tiers = assign(cands, null);
    expect(naive).toBeGreaterThan(VOICE_BUDGET_US * 2);
    expect(assignedCostUs(cands, tiers)).toBeLessThanOrEqual(VOICE_BUDGET_US);
  });

  it('leaves a small net entirely at full detail', () => {
    const cands = [];
    for (let i = 0; i < 12; i++) cands.push(wireCand(i, 400));
    const tiers = assign(cands, null);
    for (const c of cands) expect(tiers.get(c.key)).toBe(LOD_NEAR);
  });

  it('keeps objects below the threshold out of the competition entirely', () => {
    const tiny = [];
    for (let i = 0; i < 500; i++) tiny.push(wireCand(i, WIRE_BAND.mid - 1));
    const big = wireCand(999, 400);
    const tiers = assign([...tiny, big], null);
    // The tail neither takes budget nor blocks the one wire that earned it.
    expect(tiers.get(wireKey(999))).toBe(LOD_NEAR);
    expect(assignedCostUs(tiny, tiers)).toBe(0);
  });

  it('holds a budget-forced demotion instead of flipping back', () => {
    const sel = new LodSelector();
    const cands = [wireCand(1, 400), wireCand(2, 400)];
    const tight = assign(cands, sel, COST_US.nearWire + COST_US.midWire);
    const demoted = cands.find((c) => tight.get(c.key) === LOD_MID)!;
    sel.sweep();
    // Budget opens up, but the demoted wire needs to re-earn the promotion
    // rather than snapping back the very next frame.
    expect(sel.peek(demoted.key)).toBe(LOD_MID);
  });
});
