import { describe, expect, it } from 'vitest';
import { channelTotal, fixedParams, pondMatter } from './test-params.ts';
import {
  agentValue,
  atCap,
  BODY_VALUE,
  bodyDelta,
  canPayShare,
  deathYield,
  EnergyGrid,
  ERA_CAP_RATIO,
  EXTRA_CAP,
  EXTRA_FLOOR,
  extraCapFor,
  flowChargesFast,
  harvestSlotsFast,
  uptakeRate,
  hungerNeed,
  redexNeed,
  rewriteShareOf,
  stakeMet,
  rescueNeed,
  rescueTarget,
  REQUEST_DECAY,
  resetRequests,
  REWRITE_SHARE,
  rewriteCost,
  rewriteYield,
  seedRequest,
  settlePool,
  spareEnergy,
  spendExtra,
  relaxRequestsFast,
  tickUpkeepFast,
  WireAdjacency,
} from './energy.ts';
import { Sim } from './sim.ts';
import type { Agent, AgentKind } from './agents.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { CH } from './fields.ts';

/** Ecology off: nothing spawns, grows, bills, snaps or rewrites on its own. */
function quietParams(): Params {
  const p = defaultParams();
  p.spawnInterval = 0;
  p.ambientEnergy = 0;
  p.energyRegrow = 0;
  p.upkeep = 0;
  p.snapRadius = 0;
  p.rewriteDuration = 0;
  return p;
}

/** Where the pond is pinned; a body's `x, y` below is an offset from it. */
const CX = 400;
const CY = 300;

/**
 * A pinned pond with its ground on the field at `ambient` a cell, ten units
 * a cell, so a body's cell is the one under its position. `body` spawns a
 * real agent and sets the fields the energy passes read; `net` builds the
 * flat adjacency over the wires the test laid.
 */
function pond(ambient = 0) {
  const params = quietParams();
  params.ambientEnergy = ambient;
  params.energyCell = 10;
  const sim = new Sim(800, 600, 128);
  sim.pinWorld(CX, CY, params);
  const store = sim.agentStore;
  const grid = sim.energy;
  /** Defaults to `con`: Era has its own upkeep rate, so kind matters here. */
  const body = (
    x: number,
    y: number,
    extra = 0,
    request = 0,
    locked = false,
    kind: AgentKind = 'con',
    energyCap = extraCapFor(kind),
  ): Agent => {
    const a = sim.spawn(kind, CX + x, CY + y, 0, params, true)!;
    a.extra = extra;
    a.request = request;
    a.locked = locked;
    a.recovering = false;
    a.energyCap = energyCap;
    a.requestDecay = REQUEST_DECAY;
    a.debtCap = EXTRA_FLOOR;
    a.rescueTo = 0.9;
    a.transportQuantum = 0;
    return a;
  };
  /** Wire consecutive bodies `r` to `l`. */
  const chain = (list: Agent[]): void => {
    for (let i = 0; i + 1 < list.length; i++) sim.wire(list[i].id, 'r', list[i + 1].id, 'l', params);
  };
  const net = (list: Agent[]) => {
    const index = new Map<number, number>();
    list.forEach((b, i) => index.set(b.id, i));
    const adj = new WireAdjacency();
    adj.build(list.length, index, () => [...sim.graph.wires.values()]);
    return { list, adj };
  };
  /** The need field at its fixpoint: `Sim` at the shipped `requestReach` 0. */
  const settle = (list: Agent[], adj: WireAdjacency, decay?: number): void => {
    relaxRequestsFast(list, store, adj, decay);
  };
  const flow = (list: Agent[], adj: WireAdjacency, opts?: Parameters<typeof flowChargesFast>[4]): number =>
    flowChargesFast(list, store, adj, undefined, opts);
  const upkeep = (list: Agent[], dt: number, rate: number): number[] => tickUpkeepFast(list, store, dt, rate);
  const harvest = (list: Agent[]): void => harvestSlotsFast(list, store, grid);
  return { sim, params, store, grid, body, chain, net, settle, flow, upkeep, harvest };
}

describe('wire adjacency', () => {
  /** Reference: the Map of arrays this replaced. */
  function reference(n: number, index: Map<number, number>, wires: { a: { id: number }; b: { id: number } }[]) {
    const out: number[][] = [];
    for (let i = 0; i < n; i++) out.push([]);
    for (const w of wires) {
      const ia = index.get(w.a.id);
      const ib = index.get(w.b.id);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      out[ia].push(ib);
      out[ib].push(ia);
    }
    return out;
  }

  it('matches a per-body neighbour list, order included', () => {
    const ids = [7, 3, 11, 2, 9];
    const index = new Map(ids.map((id, i) => [id, i]));
    const wires = [
      { a: { id: 7 }, b: { id: 3 } },
      { a: { id: 11 }, b: { id: 7 } },
      { a: { id: 2 }, b: { id: 9 } },
      { a: { id: 3 }, b: { id: 11 } },
      // Dropped: a self-wire, and an end that is not in the index.
      { a: { id: 9 }, b: { id: 9 } },
      { a: { id: 9 }, b: { id: 404 } },
    ];
    const adj = new WireAdjacency();
    adj.build(ids.length, index, () => wires);
    const want = reference(ids.length, index, wires);
    for (let i = 0; i < ids.length; i++) {
      const got = Array.from(adj.nei.subarray(adj.off[i], adj.off[i + 1]));
      expect(got, `body ${i}`).toEqual(want[i]);
    }
  });

  it('survives a source that can only be walked once', () => {
    // The counting sort walks the wires twice. Handed `map.values()` directly
    // the second pass would see nothing and every neighbour would read as body
    // zero — wrong, but quietly: the field still spreads, through a graph
    // nobody built. Taking a factory is what makes that unrepresentable.
    const index = new Map([
      [1, 0],
      [2, 1],
      [3, 2],
    ]);
    const wires = new Map([
      [10, { a: { id: 1 }, b: { id: 2 } }],
      [11, { a: { id: 2 }, b: { id: 3 } }],
    ]);
    const adj = new WireAdjacency();
    adj.build(3, index, () => wires.values());
    expect(Array.from(adj.nei.subarray(adj.off[0], adj.off[1]))).toEqual([1]);
    expect(Array.from(adj.nei.subarray(adj.off[1], adj.off[2]))).toEqual([0, 2]);
    expect(Array.from(adj.nei.subarray(adj.off[2], adj.off[3]))).toEqual([1]);
  });

  it('is reusable across builds of different sizes', () => {
    const adj = new WireAdjacency();
    adj.build(3, new Map([[1, 0], [2, 1], [3, 2]]), () => [{ a: { id: 1 }, b: { id: 3 } }]);
    expect(Array.from(adj.nei.subarray(adj.off[0], adj.off[1]))).toEqual([2]);
    adj.build(2, new Map([[5, 0], [6, 1]]), () => [{ a: { id: 5 }, b: { id: 6 } }]);
    expect(Array.from(adj.nei.subarray(adj.off[0], adj.off[1]))).toEqual([1]);
    expect(Array.from(adj.nei.subarray(adj.off[1], adj.off[2]))).toEqual([0]);
  });
});

describe('rewrite energy', () => {
  it('values every kind of body the same, at a full tank', () => {
    expect(agentValue('era')).toBe(BODY_VALUE);
    expect(agentValue('con')).toBe(BODY_VALUE);
    expect(agentValue('dup')).toBe(BODY_VALUE);
  });

  it('returns a body\'s worth plus its stock when it dies', () => {
    const { body } = pond();
    expect(deathYield(body(0, 0, EXTRA_CAP)), 'full').toBeCloseTo(2.5, 6);
    expect(deathYield(body(0, 0, 0)), 'break-even').toBeCloseTo(BODY_VALUE, 6);
    expect(deathYield(body(0, 0, EXTRA_FLOOR)), 'starved').toBeCloseTo(0.25, 6);
  });

  it('counts bodies in and out, and prices the two directions differently', () => {
    expect(bodyDelta('era-era')).toBe(2);
    expect(bodyDelta('annihilate-con')).toBe(2);
    expect(bodyDelta('annihilate-dup')).toBe(2);
    expect(bodyDelta('erase'), 'two in, two out').toBe(0);
    expect(bodyDelta('commute'), 'four out of two').toBe(-2);

    // A commute is charged in shares, one per end, so it stays affordable.
    expect(rewriteCost('commute')).toBe(2 * REWRITE_SHARE);
    expect(rewriteCost('era-era')).toBe(0);
    // A death returns a whole body's worth, which is more than built it.
    expect(rewriteYield('annihilate-con')).toBe(2 * BODY_VALUE);
    expect(rewriteYield('commute')).toBe(0);
    expect(rewriteYield('erase'), 'swaps bodies, releases nothing').toBe(0);
    expect(
      rewriteYield('annihilate-con') - rewriteCost('commute'),
      'and the gap is what a rewrite cycle mints',
    ).toBeCloseTo(0.5, 6);
  });
});

describe('EnergyGrid', () => {
  it('reads ambient until a cell is touched, then keeps the remainder', () => {
    const g = new EnergyGrid(10, 0.1);
    expect(g.getAt(3, 3)).toBeCloseTo(0.1);
    expect(g.take(g.index(3, 3).key, 1)).toBeCloseTo(0.1);
    expect(g.getAt(3, 3)).toBe(0);
    expect(g.getAt(15, 3)).toBeCloseTo(0.1);
    g.addAt(3, 3, 2);
    expect(g.getAt(3, 3)).toBeCloseTo(2);
  });

  it('inexhaustible cells stay at ambient after harvest', () => {
    const g = new EnergyGrid(10, 8);
    g.inexhaustible = true;
    expect(g.take(g.index(3, 3).key, 5)).toBe(5);
    expect(g.getAt(3, 3)).toBe(8);
    expect(g.take(g.index(3, 3).key, 100)).toBe(8);
    expect(g.getAt(3, 3)).toBe(8);
  });
});

describe('harvest slots', () => {
  it('accumulates 0.1 ambient instead of filling in one visit', () => {
    const { grid, body, harvest } = pond(0.1);
    const a = body(2, 2);
    harvest([a]);
    expect(a.extra).toBeCloseTo(0.1);
    expect(canPayShare(a)).toBe(false);
    expect(grid.getAt(a.x, a.y)).toBe(0);
  });

  it('fills a slot from a cell of 1', () => {
    const { grid, body, harvest } = pond(1);
    const a = body(2, 2);
    harvest([a]);
    expect(a.extra).toBeCloseTo(1);
    expect(grid.getAt(a.x, a.y)).toBe(0);
  });

  it('gives a shared cell of 1 to the lower id only', () => {
    const { grid, body, harvest } = pond(1);
    const a = body(2, 2);
    const b = body(3, 2);
    harvest([a, b]);
    expect(a.extra).toBeCloseTo(1);
    expect(b.extra).toBe(0);
    expect(grid.getAt(a.x, a.y)).toBe(0);
  });

  /*
   * Monod uptake. `docs/history/energy-chemistry-plan.md` §4.
   *
   * Two claims worth a test each: at `cap` 0 nothing whatsoever changes, and
   * above it the id-order artifact — older bodies systematically eating first
   * in a contested cell, a fitness gradient on age nobody chose — stops being
   * a thing.
   */
  it('is Monod above zero and unmetered at zero', () => {
    expect(uptakeRate(1, 0, 0.25)).toBe(Infinity);
    expect(uptakeRate(0, 2, 0.25)).toBe(0);
    // Half-saturation is exactly that: at S = Ks the rate is vmax/2.
    expect(uptakeRate(0.25, 2, 0.25)).toBeCloseTo(1, 12);
    // Saturating, and monotone in S but never past vmax.
    expect(uptakeRate(1000, 2, 0.25)).toBeGreaterThan(1.99);
    expect(uptakeRate(1000, 2, 0.25)).toBeLessThan(2);
    expect(uptakeRate(0.5, 2, 0.25)).toBeGreaterThan(uptakeRate(0.25, 2, 0.25));
  });

  /*
   * The metered path — the share ceiling, the contested cell, the rich cell
   * at the rate and the poor one under it — is pinned in `chemistry.test.ts`.
   * These pin the unmetered take-what-fits path the pond runs by default.
   */

  it('does not fill a slot that is already at the cap', () => {
    const { grid, body, harvest } = pond(1);
    const a = body(2, 2, EXTRA_CAP);
    expect(atCap(a)).toBe(true);
    harvest([a]);
    expect(a.extra).toBe(EXTRA_CAP);
    expect(grid.getAt(a.x, a.y), 'the cell is untouched').toBe(1);
  });

  it('keeps harvesting a body that can pay a share but is not yet full', () => {
    // The two used to be the same test. Being able to commute is not being
    // full, and a body that stopped topping up at the share would have no
    // headroom against the next few seconds of upkeep.
    const { body, harvest } = pond(1);
    const a = body(2, 2, REWRITE_SHARE);
    expect(canPayShare(a)).toBe(true);
    expect(atCap(a)).toBe(false);
    harvest([a]);
    expect(a.extra).toBeCloseTo(EXTRA_CAP, 6);
  });

  it('skips locked agents', () => {
    const { grid, body, harvest } = pond(1);
    const a = body(2, 2, 0, 0, true);
    harvest([a]);
    expect(a.extra).toBe(0);
    expect(grid.getAt(a.x, a.y)).toBe(1);
  });

  it('fills every agent on an inexhaustible cell', () => {
    const { grid, body, harvest } = pond(8);
    grid.inexhaustible = true;
    const a = body(2, 2);
    const b = body(3, 2);
    harvest([a, b]);
    expect(a.extra).toBeCloseTo(EXTRA_CAP, 6);
    expect(b.extra).toBeCloseTo(EXTRA_CAP, 6);
    expect(grid.getAt(a.x, a.y)).toBe(8);
  });
});

describe('upkeep', () => {
  it("drains extra continuously and reports death at the body's own debt cap", () => {
    const { body, upkeep } = pond();
    const a = body(0, 0, 1);
    expect(upkeep([a], 0.5, 1)).toEqual([]);
    expect(a.extra, 'half a second at one a second').toBeCloseTo(0.5, 6);
    expect(upkeep([a], 1, 1)).toEqual([]);
    expect(a.extra, 'and straight on into debt').toBeCloseTo(-0.5, 6);
    expect(upkeep([a], 1, 1), 'reaching the floor is a death').toEqual([a.id]);
    expect(a.extra).toBe(EXTRA_FLOOR);
    expect(upkeep([a], 1, 1), 'reported once, not every frame after').toEqual([]);
    expect(a.extra, 'and never falls past it').toBe(EXTRA_FLOOR);
  });

  it('kills a shallow-debt body before a deep-debt one', () => {
    const { body, upkeep } = pond();
    const shallow = body(0, 0, 0);
    shallow.debtCap = -0.2;
    const deep = body(0, 0, 0);
    deep.debtCap = -2;
    expect(upkeep([shallow, deep], 0.3, 1)).toEqual([shallow.id]);
    expect(shallow.extra).toBeCloseTo(-0.2, 6);
    expect(deep.extra).toBeCloseTo(-0.3, 6);
    expect(deep.extra, 'still alive, still in debt').toBeGreaterThan(deep.debtCap);
  });

  it('skips locked agents', () => {
    const { body, upkeep } = pond();
    const a = body(0, 0, 1, 0, true);
    expect(upkeep([a], 10, 1)).toEqual([]);
    expect(a.extra).toBe(1);
  });
});

describe('request gradient', () => {
  it('attenuates need by distance instead of counting hops', () => {
    const { body, chain, net, settle } = pond();
    const a = [body(0, 0), body(10, 0), body(20, 0)];
    chain(a);
    const { list, adj } = net(a);
    seedRequest(a[0], 1);
    settle(list, adj);
    expect(a[0].request).toBe(1);
    expect(a[1].request).toBeCloseTo(REQUEST_DECAY, 6);
    expect(a[2].request).toBeCloseTo(REQUEST_DECAY ** 2, 6);
  });

  it('lets a big distant need outrank a small near one', () => {
    // Chain 1-2-3-4. A large need at 1 must beat a small need at 4 for the
    // body at 3, even though 4 is adjacent — that is what magnitude buys.
    const { body, chain, net, settle } = pond();
    const a = [body(0, 0), body(10, 0), body(20, 0), body(30, 0)];
    chain(a);
    const { list, adj } = net(a);
    seedRequest(a[0], 1);
    seedRequest(a[3], 0.2);
    settle(list, adj);
    // 1 reaches 3 at 0.8^2 = 0.64; 4 only offers 0.2 there.
    expect(a[2].request).toBeCloseTo(REQUEST_DECAY ** 2, 6);
    expect(a[2].request).toBeGreaterThan(a[3].request);
  });

  it('walks energy up the gradient, one hop a frame', () => {
    const { body, chain, net, flow } = pond();
    const a = [body(0, 0, 0, 1), body(10, 0, 0, REQUEST_DECAY), body(20, 0, 1, REQUEST_DECAY ** 2)];
    chain(a);
    const { list, adj } = net(a);
    // 3 holds the surplus and 1 is the one that needs it, two hops away. Each
    // frame moves one wire's worth, throttled by the field at the receiving
    // end — 0.8 here, since that is how much of 1's need is visible from 2.
    expect(flow(list, adj)).toBeCloseTo(REQUEST_DECAY, 6);
    expect(a[2].extra).toBeCloseTo(1 - REQUEST_DECAY, 6);
    expect(a[1].extra).toBeCloseTo(REQUEST_DECAY, 6);
    flow(list, adj);
    expect(a[0].extra, 'reaches the body that needs it').toBeGreaterThan(0.5);
  });

  it('delivers only as much as the recipient can hold', () => {
    const { body, chain, net, flow } = pond();
    const a = [body(0, 0, 0.7, 0.3), body(10, 0, 1, 0.24)];
    chain(a);
    const { list, adj } = net(a);
    expect(flow(list, adj)).toBeCloseTo(0.3, 6);
    expect(a[0].extra).toBeCloseTo(1, 6);
    expect(a[1].extra, 'donor keeps the rest').toBeCloseTo(0.7, 6);
  });

  it('relays through a body that needs nothing itself', () => {
    // The middle body is not hungry and is not a redex. If it could only
    // accept what it needs, every shortage more than one wire from a surplus
    // would be unreachable.
    const { body, chain, net, flow } = pond();
    const a = [body(0, 0, 0, 1), body(10, 0, 0, REQUEST_DECAY), body(20, 0, 1, REQUEST_DECAY ** 2)];
    chain(a);
    const { list, adj } = net(a);
    flow(list, adj);
    expect(a[1].extra, 'held by the conduit').toBeCloseTo(REQUEST_DECAY, 6);
  });

  it('leaves a flat spot where two equal needs meet', () => {
    // Needs of the same size at both ends of a 3-chain give the middle body
    // the same field from either side, so nothing crosses it.
    const { body, chain, net, settle, flow } = pond();
    const a = [body(0, 0, 0), body(10, 0, 1), body(20, 0, 0)];
    chain(a);
    const { list, adj } = net(a);
    seedRequest(a[0], 1);
    seedRequest(a[2], 1);
    settle(list, adj);
    expect(a[0].request).toBeCloseTo(a[2].request, 6);
    // The middle body is the one holding energy, and both neighbours pull on
    // it equally hard, so it gives to exactly one of them rather than tearing.
    const moved = flow(list, adj);
    expect(moved).toBeGreaterThan(0);
    const fed = [a[0].extra, a[2].extra];
    expect(fed.filter((e) => e > 0.5).length, 'one of the two, not both').toBe(1);
  });

  it('does not pass energy to a body that is no needier than the donor', () => {
    const { body, chain, net, flow } = pond();
    const a = [body(0, 0, 1, 1), body(10, 0, 0, 1)];
    chain(a);
    const { list, adj } = net(a);
    expect(flow(list, adj)).toBe(0);
    expect(a[0].extra).toBe(1);
    expect(canPayShare(a[0]), 'the donor could have paid a share').toBe(true);
    expect(canPayShare(a[1]), 'the other could not, and got nothing to change that').toBe(false);
  });

  it('reads hunger straight off the debt', () => {
    const { body } = pond();
    expect(hungerNeed(body(0, 0, 1)), 'stocked').toBe(0);
    expect(hungerNeed(body(0, 0, 0)), 'break-even').toBe(0);
    expect(hungerNeed(body(0, 0, -0.1))).toBeCloseTo(0.1, 6);
    expect(hungerNeed(body(0, 0, -1)), 'at the point of death').toBe(1);
  });

  it('has nothing to pass on while it is in debt', () => {
    const { body } = pond();
    expect(spareEnergy(body(0, 0, 0.4))).toBeCloseTo(0.4, 6);
    expect(spareEnergy(body(0, 0, 0))).toBe(0);
    expect(spareEnergy(body(0, 0, -0.4))).toBe(0);
  });

  it('settles a body\'s own debt before anything travels further', () => {
    // 1 is deep in debt, 2 is shallowly in debt, 3 has stock. What reaches 2
    // pays 2 off first; only what is left over can reach 1.
    const { body, chain, net, settle, flow } = pond();
    const a = [body(0, 0, -0.9), body(10, 0, -0.2), body(20, 0, 1)];
    chain(a);
    const { list, adj } = net(a);
    for (let i = 0; i < 2; i++) {
      resetRequests(a);
      for (const b of a) seedRequest(b, hungerNeed(b));
      settle(list, adj);
      flow(list, adj);
    }
    expect(a[1].extra, 'out of debt first').toBeGreaterThanOrEqual(0);
    const total = a.reduce((t, b) => t + b.extra, 0);
    expect(total, 'and nothing minted').toBeCloseTo(-0.1, 6);
  });

  it('carries a shortage further at a slower decay', () => {
    const chainAt = (decay: number) => {
      const { body, chain, net, settle } = pond();
      const a = Array.from({ length: 6 }, (_, i) => body(i * 10, 0));
      a[0].request = 1;
      chain(a);
      const { list, adj } = net(a);
      settle(list, adj, decay);
      return a[5].request;
    };
    expect(chainAt(0.8), 'five hops at 0.8').toBeCloseTo(0.8 ** 5, 6);
    expect(chainAt(0.9), 'and further at 0.9').toBeCloseTo(0.9 ** 5, 6);
    expect(chainAt(0.9)).toBeGreaterThan(chainAt(0.8));
  });

  it('lets each body relay demand at its own rate instead of one global decay', () => {
    const { body, chain, net, settle } = pond();
    const a = Array.from({ length: 6 }, (_, i) => body(i * 10, 0));
    a[0].request = 1;
    for (const b of a) b.requestDecay = 0.9;
    a[2].requestDecay = 0.05;
    chain(a);
    const { list, adj } = net(a);
    settle(list, adj); // no override: each body's own field governs
    expect(a[1].request, 'upstream of the lossy relay is unaffected').toBeCloseTo(0.9, 6);
    expect(a[3].request, 'the lossy relay chokes what crosses it').toBeCloseTo(0.9 * 0.9 * 0.05, 6);
    expect(a[5].request).toBeLessThan(0.9 ** 5 * 0.1);
  });

  it('never lets the field go flat', () => {
    // At decay 1 every body holds the same need, no neighbour is strictly
    // needier than its donor, and nothing moves at all.
    const { body, chain, net, settle, flow } = pond();
    const a = [body(0, 0, 0, 1), body(10, 0, 1)];
    chain(a);
    const { list, adj } = net(a);
    settle(list, adj, 1);
    expect(a[1].request).toBeLessThan(a[0].request);
    expect(flow(list, adj), 'so the surplus still crosses').toBeGreaterThan(0);
  });

  it('keeps a rescued body asking until it can act again', () => {
    // Fill fraction 1 on a tank whose cap is a rewrite share: rescue aims at
    // extra=1, which is what used to be the global absolute target.
    const { body } = pond();
    const a = body(0, 0, -0.4);
    a.energyCap = 1;
    a.debtCap = -1;
    a.rescueTo = 1;
    expect(rescueNeed(a), 'in debt: still short of a whole share').toBeCloseTo(1.4, 6);
    expect(a.recovering).toBe(true);
    a.extra = 0;
    expect(rescueNeed(a), 'out of debt but not yet standing').toBeCloseTo(1, 6);
    a.extra = 0.6;
    expect(rescueNeed(a)).toBeCloseTo(0.4, 6);
    a.extra = 1;
    expect(rescueNeed(a), 'discharged').toBe(0);
    expect(a.recovering).toBe(false);
    a.extra = 0.2;
    expect(rescueNeed(a), 'and it does not start asking again on its own').toBe(0);
  });

  it('leaves a body that has never been in debt quiet', () => {
    const { body } = pond();
    const poor = body(0, 0, 0.05);
    expect(rescueNeed(poor)).toBe(0);
    expect(poor.recovering).toBe(false);
  });

  it('rescues only to break-even at fill 0, as it used to', () => {
    const { body } = pond();
    const a = body(0, 0, -0.4);
    a.rescueTo = 0;
    expect(rescueNeed(a)).toBeCloseTo(0.4, 6);
    a.extra = 0;
    expect(rescueNeed(a)).toBe(0);
  });

  it("aims a rescue between the body's own floor and cap", () => {
    const { body } = pond();
    const a = body(0, 0);
    a.debtCap = -0.5;
    a.energyCap = 2;
    a.rescueTo = 0.5;
    expect(rescueTarget(a)).toBeCloseTo(0.75, 6);
    a.rescueTo = 0;
    expect(rescueTarget(a)).toBe(a.debtCap);
    a.rescueTo = 1;
    expect(rescueTarget(a)).toBe(a.energyCap);
    a.rescueTo = 1.4;
    expect(rescueTarget(a), 'cannot overshoot the cap').toBe(a.energyCap);
    a.rescueTo = -0.2;
    expect(rescueTarget(a), 'cannot undershoot the floor').toBe(a.debtCap);
  });

  it('measures a stalled redex end by what it is short of a full extra', () => {
    const { body } = pond();
    expect(redexNeed(body(0, 0, 0.8))).toBeCloseTo(0.2, 6);
    expect(redexNeed(body(0, 0, 1))).toBe(0);
  });

  it('clears requests', () => {
    const { body } = pond();
    const a = body(0, 0, 0, 4);
    resetRequests([a]);
    expect(a.request).toBe(0);
  });
});

describe('settle pool', () => {
  it('gives the neediest survivor first refusal', () => {
    // Released energy enters the net at the body the gradient would have sent
    // it to. `easy` has the lower id, so id order would have fed it first and
    // left the hungry one short.
    const { body } = pond();
    const grid = new EnergyGrid(10, 0);
    const easy = body(0, 0, 1, 0.1);
    const hungry = body(0, 0, -0.5, 0.9);
    settlePool(1, [easy, hungry], grid, 5, 5);
    expect(hungry.extra, 'all of it went to the needy one').toBeCloseTo(0.5, 6);
    expect(easy.extra, 'which had room but no claim on it').toBe(1);
    expect(grid.getAt(5, 5)).toBe(0);
  });

  it('drops what the survivors cannot hold onto the ground', () => {
    // Two bodies, 2.0 of room between them, against a two-body annihilation.
    // The per-body cap is the bandwidth limit; the rest lands where it died.
    const { body } = pond();
    const grid = new EnergyGrid(10, 0);
    const a = body(0, 0, -0.5, 0.9);
    const b = body(0, 0, 1, 0.1);
    settlePool(2 * BODY_VALUE, [a, b], grid, 5, 5);
    expect(a.extra).toBeCloseTo(EXTRA_CAP, 6);
    expect(b.extra).toBeCloseTo(EXTRA_CAP, 6);
    expect(grid.getAt(5, 5), 'overflow').toBeCloseTo(2 * BODY_VALUE - 2, 6);
  });

  it('fills empty leftover slots before dumping to the grid', () => {
    const { body } = pond();
    const grid = new EnergyGrid(10, 0);
    const a = body(0, 0);
    const b = body(0, 0, EXTRA_CAP);
    settlePool(2, [a, b], grid, 5, 5);
    expect(a.extra, 'the empty slot takes what it can hold').toBeCloseTo(EXTRA_CAP, 6);
    expect(b.extra, 'the full one takes nothing').toBe(EXTRA_CAP);
    expect(grid.getAt(5, 5), 'and the remainder goes to the ground')
      .toBeCloseTo(2 - EXTRA_CAP, 6);
  });
});

describe('sim energy', () => {
  it('blocks commute when the pair has no extras', () => {
    const sim = new Sim(800, 600);
    const params = fixedParams();
    params.spawnInterval = 0;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0.12;
    params.wireShrink = 0.08;
    const c = sim.spawn('con', 380, 300, 0, params, true)!;
    const d = sim.spawn('dup', 420, 300, Math.PI, params, true)!;
    sim.wire(c.id, 'p', d.id, 'p', params);
    for (let f = 0; f < 240; f++) sim.step(1 / 60, params);
    expect(sim.agents.size).toBe(2);
    expect(sim.rewrites.length).toBe(0);
  });

  it('carries demand a hop a frame when the reach is set, and everywhere when it is not', () => {
    /*
     * What `requestReach` actually changes. The field is the same potential
     * either way and settles in the same place; the difference is whether
     * getting there takes time. At 0 a shortage is known across the whole net
     * on the frame it appears, which is why nothing in the pond can carry a
     * wave — there is never anything left to travel.
     *
     * Every body but the last is full, so nobody else is making a claim and
     * the only thing in the field is the one at the far end.
     */
    const build = (reach: number) => {
      const sim = new Sim(1200, 600);
      const params = fixedParams();
      params.requestReach = reach;
      params.spawnInterval = 0;
      params.ambientEnergy = 0;
      params.upkeep = 0;
      params.rewriteDuration = 0;
      params.snapRadius = 0;
      params.transportRecoil = 0;
      sim.energy.configure(params.energyCell, 0);
      const n = 8;
      const ids: number[] = [];
      for (let i = 0; i < n; i++) ids.push(sim.spawn('con', 100 + i * 45, 300, 0, params, true)!.id);
      for (let i = 0; i + 1 < n; i++) sim.wire(ids[i], 'r', ids[i + 1], 'l', params);
      for (const id of ids) sim.agents.get(id)!.extra = EXTRA_CAP;
      sim.agents.get(ids[n - 1])!.extra = -0.5;
      return { sim, params, ids };
    };
    const heardAt = (o: ReturnType<typeof build>, frames: number): number => {
      for (let f = 0; f < frames; f++) o.sim.step(1 / 60, o.params);
      return o.sim.agents.get(o.ids[0])!.request;
    };
    expect(heardAt(build(0), 1), 'no reach: seven wires away on the first frame').toBeGreaterThan(0);
    expect(heardAt(build(1), 1), 'reach 1: still silent after one').toBe(0);
    expect(heardAt(build(1), 3), 'and after three').toBe(0);
    expect(heardAt(build(1), 12), 'audible once it has had the frames to travel').toBeGreaterThan(0);
  });

  it('drains a reservoir across an empty corridor to a dying end', () => {
    // The reported symptom: one end of a net full, the other starving, and no
    // sign of the surplus ever crossing. It did cross — but only the few
    // hundredths that put the patient back on exactly zero, after which nobody
    // was in debt, nobody was asking, and 3.75 units sat parked nine hops away
    // for the rest of the run.
    const sim = new Sim(1200, 600);
    const params = fixedParams();
    // Continuous transport: this is about a corridor relaying a shortfall to
    // the body that has it, and it asserts the delivery lands *on* the rescue
    // target. A packet crosses whole whatever the gradient asked for, so it
    // overshoots to the cap on purpose — which `quantised transport` covers.
    params.transportQuantum = 0;
    params.spawnInterval = 0;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    params.snapRadius = 0;
    params.transportRecoil = 0;
    sim.energy.configure(params.energyCell, 0);
    const n = 12;
    const ids: number[] = [];
    for (let i = 0; i < n; i++) ids.push(sim.spawn('con', 100 + i * 45, 300, 0, params, true)!.id);
    for (let i = 0; i + 1 < n; i++) sim.wire(ids[i], 'r', ids[i + 1], 'l', params);
    for (let i = 0; i < n; i++) sim.agents.get(ids[i])!.extra = i >= n - 3 ? EXTRA_CAP : 0;
    const dying = sim.agents.get(ids[0])!;
    dying.extra = -0.5;
    const reservoir = () =>
      ids.slice(n - 3).reduce((t, id) => t + (sim.agents.get(id)?.extra ?? 0), 0);
    const held = reservoir();
    for (let f = 0; f < 120; f++) sim.step(1 / 60, params);
    expect(dying.extra, 'fed up toward its rescue fill, not parked on the line').toBeCloseTo(
      rescueTarget(dying),
      2,
    );
    expect(held - reservoir(), 'and it came from nine hops away').toBeGreaterThan(1);
  });

  it('lets a commute fire when both agents hold an extra', () => {
    const sim = new Sim(800, 600);
    const params = fixedParams();
    params.spawnInterval = 0;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0.12;
    params.wireShrink = 0.08;
    const c = sim.spawn('con', 380, 300, 0, params, true)!;
    const d = sim.spawn('dup', 420, 300, Math.PI, params, true)!;
    c.extra = 1;
    d.extra = 1;
    sim.wire(c.id, 'p', d.id, 'p', params);
    for (let f = 0; f < 240; f++) sim.step(1 / 60, params);
    expect(sim.agents.size).toBe(4);
  });

  it('lets a commute fire when both tanks are full but smaller than a share', () => {
    const sim = new Sim(800, 600);
    const params = fixedParams();
    params.spawnInterval = 0;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0.12;
    params.wireShrink = 0.08;
    const cap = EXTRA_CAP * 0.6;
    const c = sim.spawn('con', 380, 300, 0, params, true)!;
    const d = sim.spawn('dup', 420, 300, Math.PI, params, true)!;
    c.energyCap = cap;
    d.energyCap = cap;
    c.extra = cap;
    d.extra = cap;
    sim.wire(c.id, 'p', d.id, 'p', params);
    for (let f = 0; f < 240; f++) sim.step(1 / 60, params);
    expect(sim.agents.size).toBe(4);
  });

  it('returns annihilation energy to the grid when the net vanishes', () => {
    const sim = new Sim(800, 600);
    const params = fixedParams();
    params.spawnInterval = 0;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0.12;
    params.wireShrink = 0.08;
    const a = sim.spawn('era', 380, 300, 0, params, true)!;
    const b = sim.spawn('era', 420, 300, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    for (let f = 0; f < 240; f++) sim.step(1 / 60, params);
    expect(sim.agents.size).toBe(0);
    expect(sim.totalFree()).toBe(0);
    expect(sim.energy.storedTotal()).toBeGreaterThanOrEqual(2);
  });

  it('walks a neighbour extra onto a hungry commute pair', () => {
    const sim = new Sim(800, 600);
    const params = fixedParams();
    params.spawnInterval = 0;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0.12;
    params.wireShrink = 0.08;
    const c = sim.spawn('con', 380, 300, 0, params, true)!;
    const d = sim.spawn('dup', 420, 300, Math.PI, params, true)!;
    const e = sim.spawn('era', 340, 300, Math.PI, params, true)!;
    e.extra = 1;
    sim.wire(c.id, 'p', d.id, 'p', params);
    sim.wire(c.id, 'l', e.id, 'p', params);
    for (let f = 0; f < 240; f++) sim.step(1 / 60, params);
    expect(e.extra).toBe(0);
    // It used to end up in one end's tank, where it sat: a pair needs both
    // shares and one full end could not spend a thing. It is now staked on the
    // redex itself, which is the same walk with somewhere to put the energy at
    // the end of it. Still short of the two shares a commute costs, so the
    // pair has not fired — it is saving, not idle.
    expect(sim.escrowTotal(), 'the neighbour extra reached the redex').toBeCloseTo(1, 6);
    expect(c.extra + d.extra, 'and is committed rather than banked').toBeCloseTo(0, 6);
    expect(sim.agents.size).toBe(3);
  });

  /** Two wired Cons on barren ground. Era is excluded: it never starves. */
  function pair(aExtra: number, bExtra: number, upkeep = 1) {
    const sim = new Sim(320, 200);
    const params = fixedParams();
    params.spawnInterval = 0;
    params.ambientEnergy = 0;
    params.upkeep = upkeep;
    params.rewriteDuration = 0;
    params.snapRadius = 0;
    const a = sim.spawn('con', 100, 100, 0, params, true)!;
    const b = sim.spawn('con', 180, 100, Math.PI, params, true)!;
    a.extra = aExtra;
    b.extra = bExtra;
    sim.wire(a.id, 'l', b.id, 'r', params);
    return { sim, params, a, b };
  }

  it('disconnects an agent after a full upkeep tick to −1', () => {
    // Neither body has anything to give, so nothing can save either of them.
    const { sim, params, a } = pair(0, 0);
    for (let i = 0; i < 60; i++) sim.step(1 / 60, params);
    expect(a.extra).toBe(-1);
    expect(sim.graph.wires.size).toBe(0);
  });

  it('feeds a body from its neighbour, so a wired one outlives a lone one', () => {
    // The point of the field: a body sliding into debt is fed by whoever in
    // the net has stock. Measured against the same body with nobody to ask.
    const wired = pair(0, 1, 0.2);
    for (let i = 0; i < 60; i++) wired.sim.step(1 / 60, wired.params, null);

    const alone = pair(0, 1, 0.2);
    alone.sim.graph.detachAgent(alone.b.id);
    for (let i = 0; i < 60; i++) alone.sim.step(1 / 60, alone.params, null);

    expect(wired.a.extra, 'better off wired').toBeGreaterThan(alone.a.extra);
    expect(wired.b.extra, "out of its neighbour's pocket").toBeLessThan(alone.b.extra);
    expect(wired.sim.graph.wires.size, 'and still wired').toBe(1);
  });

  it('charges an Era instead of billing it, and never starves one', () => {
    const { body, upkeep } = pond();
    const era = body(0, 0, 0, 0, false, 'era');
    const con = body(0, 0, 0, 0, false, 'con');
    // Long enough that a Con has been billed several times over.
    for (let i = 0; i < 600; i++) upkeep([era, con], 1, 0.025);
    expect(con.extra, 'a Con burns down to the floor').toBe(EXTRA_FLOOR);
    expect(era.extra, 'an Era fills instead, to its own deeper cap').toBe(extraCapFor('era'));
    expect(upkeep([era], 1000, 0.025), 'and is never reported starved').toEqual([]);
  });

  it('gives an Era a deeper tank than a Con or a Dup', () => {
    expect(extraCapFor('era')).toBeCloseTo(EXTRA_CAP * ERA_CAP_RATIO, 6);
    expect(extraCapFor('con')).toBe(EXTRA_CAP);
    expect(extraCapFor('dup')).toBe(EXTRA_CAP);
    const { body } = pond();
    const era = body(0, 0, EXTRA_CAP, 0, false, 'era');
    expect(atCap(era), 'a Con-sized tankful is only half an Era').toBe(false);
    era.extra = extraCapFor('era');
    expect(atCap(era)).toBe(true);
  });

  it('caps a body by its own energyCap, not by a flat per-kind number', () => {
    // energyCap is heritable now — two Cons can carry different tanks — so the
    // cap has to come from the body, and extraCapFor is only ever the seed a
    // fresh one starts at.
    const { grid, body, harvest } = pond();
    const roomy = body(0, 0, EXTRA_CAP, 0, false, 'con', EXTRA_CAP * 2);
    const cramped = body(0, 0, EXTRA_CAP * 0.5, 0, false, 'con', EXTRA_CAP * 0.5);
    expect(atCap(roomy), 'a normal-sized tankful is not full for the bigger tank').toBe(false);
    expect(atCap(cramped), 'but the smaller tank is already topped out at the same level').toBe(
      true,
    );
    const cell = grid.index(roomy.x, roomy.y);
    grid.setCell(cell.i, cell.j, 10);
    harvest([roomy, cramped]);
    expect(roomy.extra).toBeCloseTo(roomy.energyCap, 6);
    expect(cramped.extra, 'no room left to harvest into').toBeCloseTo(EXTRA_CAP * 0.5, 6);
  });

  it('lets a full tank pay a rewrite even when breeding shrank it below a share', () => {
    const { body } = pond();
    const cramped = body(0, 0, EXTRA_CAP * 0.5, 0, false, 'con', EXTRA_CAP * 0.5);
    expect(cramped.energyCap).toBeLessThan(REWRITE_SHARE);
    expect(canPayShare(cramped)).toBe(true);
    const roomy = body(0, 0, EXTRA_CAP * 0.5, 0, false, 'con', EXTRA_CAP);
    expect(canPayShare(roomy), 'a default tank still needs a whole share').toBe(false);
    expect(rewriteShareOf(cramped), 'and owes only what it can hold').toBeCloseTo(
      EXTRA_CAP * 0.5,
      6,
    );
    expect(stakeMet(cramped, cramped.extra), 'so a full small tank meets its stake').toBe(true);
  });

  it('fills an Era past a full Con from the ground and along a wire', () => {
    const { sim, params, grid, body, net, settle, flow, harvest } = pond();
    const era = body(0, 0, 0, 0, false, 'era');
    const cell = grid.index(era.x, era.y);
    grid.setCell(cell.i, cell.j, 4);
    harvest([era]);
    expect(era.extra, 'forages up to its own cap').toBeCloseTo(extraCapFor('era'), 6);

    const drained = body(0, 0, 0.5, 0, false, 'era');
    const donor = body(10, 0, EXTRA_CAP, 0, false, 'con');
    sim.wire(drained.id, 'p', donor.id, 'l', params);
    const { list, adj } = net([drained, donor]);
    seedRequest(drained, 1);
    settle(list, adj);
    flow(list, adj);
    expect(drained.extra, 'and a pump can push it past a Con-sized full').toBeGreaterThan(
      EXTRA_CAP,
    );
  });

  it('caps an Era at full however long it produces for', () => {
    const cap = extraCapFor('era');
    const { body, upkeep } = pond();
    const era = body(0, 0, 0, 0, false, 'era');
    for (let i = 0; i < 5000; i++) upkeep([era], 1, 0.025);
    expect(era.extra, 'no banking past the cap').toBe(cap);
    // And the next share has to be earned at the same rate as the first.
    spendExtra(era);
    expect(era.extra, 'a share out of a full tank leaves the headroom')
      .toBeCloseTo(cap - REWRITE_SHARE, 6);
    upkeep([era], 1, 0.025);
    expect(era.extra).toBeCloseTo(cap - REWRITE_SHARE + 0.005, 6);
  });

  it('drops wires at −1 and still snaps in debt', () => {
    const sim = new Sim(320, 200);
    const params = fixedParams();
    params.spawnInterval = 0;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    params.snapRadius = 40;
    params.rewriteDuration = 20;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 200, 100, Math.PI, params, true)!;
    a.extra = 1;
    b.extra = 1;
    sim.wire(a.id, 'p', b.id, 'p', params);
    a.extra = -1;
    sim.graph.detachAgent(a.id);
    expect(sim.graph.wires.size).toBe(0);

    const c = sim.spawn('era', 100, 100, 0, params, true)!;
    const d = sim.spawn('era', 100, 103, 0, params, true)!;
    c.extra = -0.5;
    d.extra = -0.5;
    sim.latchPass(params);
    expect(sim.graph.wires.size).toBe(1);
  });
});

describe('EnergyGrid.forEachStored', () => {
  it('visits only cells that have been written', () => {
    const grid = new EnergyGrid(48, 0.1);
    grid.addAt(10, 10, 1);
    grid.setCell(-3, 4, 0);
    const seen: string[] = [];
    grid.forEachStored((i, j, e) => seen.push(`${i},${j}:${e}`));
    expect(seen).toHaveLength(2);
    expect(seen).toContain('0,0:1.1');
    expect(seen).toContain('-3,4:0');
  });
});

describe('conservation', () => {
  /*
   * `docs/history/energy-chemistry-plan.md` §5, as one assertion:
   *
   *   > The dish is driven — feed in, kill out, patterned.
   *   > The bodies are conservative — no reaction a body runs creates or
   *   > destroys matter.
   *
   * The second is the invariant that makes selection honest, and it is only
   * reachable with the three phase-2 dials at their conservative settings.
   * Every one of them ships at the value that reproduces today's pond, so
   * this is a test of what the dials *can* do, not of what the pond does.
   */
  function conservativeParams(): Params {
    const p = fixedParams();
    p.soupCount = 120;
    // The drive, off. `grow` is a source and `decay` a sink; both are the
    // dish, and the dish is allowed to be driven. This test is about bodies.
    p.energyRegrow = 0;
    p.decay = 0;
    // Immigrants are matter created out of nothing, which is deliberate — it
    // is what stops a pond going permanently extinct — and it is not a body
    // reaction.
    p.spawnInterval = 0;
    /*
     * And the mint. `params.deposit` puts five times a body's voice into three
     * channels out of nothing — a third source alongside `energyRegrow` and
     * immigration, and as deliberate as either. It never had to be turned off
     * here while the total counted `CH.energy` alone; now that rent leaves
     * through the excretion rows and the total counts every channel, a pond
     * that mints reads as one that creates.
     */
    p.deposit = 0;
    /*
     * And diffusion, for the reason `chemistry.test.ts` gives: the dish wall
     * absorbs the three signal species and reflects only `CH.energy`, which is
     * the dish doing its job rather than a body failing to conserve. It did
     * not matter while a body could only put ground on the ground; rent now
     * leaves as `conP` and `aux` too, and those the rim eats.
     */
    p.diffuse = 0;
    // The three dials.
    p.upkeepExcrete = 1;
    p.bodyValue = REWRITE_SHARE;
    p.eraUpkeepRatio = 1;
    return p;
  }

  it('diffusion moves the substance without destroying it', () => {
    /*
     * The dish wall absorbs a signal and reflects the substance.
     *
     * It used to absorb both, which cost the ground 10.6% over 900 frames at
     * 128 cells a side with nothing else running — a sink nobody asked for, on
     * the one channel `decayRate` is zeroed for so that nothing could destroy
     * it. It hid in the production dish because the loss goes as perimeter
     * over area.
     */
    const p = fixedParams();
    p.soupCount = 0;
    p.spawnInterval = 0;
    p.energyRegrow = 0;
    p.decay = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const before = sim.energy.storedTotal();
    expect(before, 'the dish should start with ground in it').toBeGreaterThan(0);
    for (let i = 0; i < 600; i++) sim.step(1 / 60, p);
    expect(sim.energy.storedTotal()).toBeCloseTo(before, 6);
  });

  it('a signal still leaks into the rim, which is what a rim is for', () => {
    // The other half of the same rule. Without it the dish would fill with
    // everything anything had ever said.
    const sim = new Sim(1600, 1200, 128);
    const p = fixedParams();
    p.soupCount = 0;
    p.spawnInterval = 0;
    p.energyRegrow = 0;
    p.decay = 0;
    loadPreset(sim, 'soup', p);
    sim.fields.fillDisk(CH.conP, 1);
    const before = channelTotal(sim, CH.conP);
    expect(before).toBeGreaterThan(0);
    for (let i = 0; i < 600; i++) sim.step(1 / 60, p);
    expect(channelTotal(sim, CH.conP)).toBeLessThan(before * 0.99);
  });

  it('bodies neither create nor destroy, across a pond that rewrites', () => {
    const p = conservativeParams();
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const before = pondMatter(sim, p.bodyValue);
    let worst = 0;
    for (let i = 0; i < 900; i++) {
      sim.step(1 / 60, p);
      worst = Math.max(worst, Math.abs(pondMatter(sim, p.bodyValue) - before));
    }
    // Something has to have happened, or this asserts about a still pond.
    expect(sim.tally.annihilations + sim.tally.commutes, 'no rewrites ran').toBeGreaterThan(4);
    /*
     * Not exact: a pair that annihilates while in debt releases less than it
     * holds, and that debt dies with it rather than being minted away — see
     * `tickRewrites`, where the pool is clamped at zero. Everything else
     * balances to a part in a hundred thousand, over nine hundred frames of a
     * pond that latched, commuted, erased and annihilated throughout.
     *
     * 3e-4 relative, and it was 1e-4 while rent could only land on one
     * channel. Rent now leaves through the excretion rows, so what was one
     * small `Float32` add a body a frame is four, and the quantisation is
     * coarser for it. Measured on this pond when the bound moved: 2.46e-4
     * over 900 frames against 1.72e-4 over 300 — sub-linear in the run, which
     * is accumulation and not a leak, and at `upkeep = 0` exactly zero. The
     * bound sits just above that measurement rather than an order past it,
     * so a real leak, which grows with the run, still trips it. The
     * chemistry suite's 'still conserves' carries the same argument.
     */
    expect(worst / before, `drifted ${worst} of ${before}`).toBeLessThan(3e-4);
  });

  it('conserves across a whole pond on the shipping path', () => {
    /*
     * Packets crossing wires and spilling onto the ground for nine hundred
     * frames of a pond that latches, commutes and annihilates, and the books
     * still balance.
     */
    const p = conservativeParams();
    p.transportQuantum = 0.5;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const before = pondMatter(sim, p.bodyValue);
    let worst = 0;
    for (let i = 0; i < 900; i++) {
      sim.step(1 / 60, p);
      worst = Math.max(worst, Math.abs(pondMatter(sim, p.bodyValue) - before));
    }
    expect(sim.tally.annihilations + sim.tally.commutes, 'no rewrites ran').toBeGreaterThan(4);
    expect(worst / before, `drifted ${worst} of ${before}`).toBeLessThan(3e-4);
  });

  it('destroys the rent when upkeepExcrete is off, which is today', () => {
    // The dial's other end, so the test above cannot pass by the invariant
    // being vacuous. Rent vanishing is what the pond does now.
    const p = conservativeParams();
    p.upkeepExcrete = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const before = pondMatter(sim, p.bodyValue);
    for (let i = 0; i < 900; i++) sim.step(1 / 60, p);
    expect(pondMatter(sim, p.bodyValue)).toBeLessThan(before * 0.999);
  });
});

describe('quantised transport', () => {
  /*
   * Transport as packets rather than a trickle.
   *
   * The continuous law gives whatever the gradient asks for, which at steady
   * state is about 4e-4 a frame — so `transportRecoil` at 100 lands an impulse
   * of 0.04 against settled speeds of 20 to 60 px/s, and two sweeps found the
   * whole momentum machinery invisible because of it. A packet is what puts
   * the impulse above the pond's own noise. These are about the arithmetic of
   * the packet; whether it makes anything swim is a measurement, not a test.
   */
  const packet = 0.5;

  it('will not let a body send what it does not yet hold', () => {
    // The accumulate-and-fire half. Under the continuous law this donor gives
    // its 0.4 away immediately; under a packet it has to wait until it has one.
    const { body, chain, net, flow } = pond();
    const a = [body(0, 0, 0, 1), body(10, 0, 0.4, 0)];
    chain(a);
    const { list, adj } = net(a);
    const grid = new EnergyGrid(10, 0);
    expect(flow(list, adj, { quantum: packet, grid })).toBe(0);
    expect(a[1].extra, 'the donor keeps it all').toBeCloseTo(0.4, 9);
    expect(a[0].extra).toBe(0);
    expect(grid.storedTotal(), 'and nothing leaked to the ground').toBeCloseTo(0, 9);
    // The same pond under the law it replaces, so this cannot pass vacuously.
    expect(flow(list, adj)).toBeCloseTo(0.4, 9);
  });

  it('sends a whole packet, however little the receiver asked for', () => {
    // Demand still decides *whether* to send — the receiver must be strictly
    // needier — but it no longer decides how much. That cap is what kept every
    // transfer down to the size of the gradient.
    const { body, chain, net, flow } = pond();
    const a = [body(0, 0, 0, 0.2), body(10, 0, 1, 0)];
    chain(a);
    const { list, adj } = net(a);
    const grid = new EnergyGrid(10, 0);
    const moved = flow(list, adj, { quantum: packet, grid });
    expect(moved, 'a packet, not the 0.2 the gradient asked for').toBeCloseTo(packet, 9);
    expect(a[0].extra).toBeCloseTo(0.5, 9);
    expect(a[1].extra).toBeCloseTo(0.5, 9);
  });

  it('spills what will not fit onto the ground under the receiver', () => {
    // A whole packet crosses whether or not the far end has room, so the
    // remainder has to land somewhere. The same place a rewrite's leftovers go.
    const { body, chain, net, flow } = pond();
    const a = [body(0, 0, 1, 1), body(10, 0, 1, 0)];
    chain(a);
    const { list, adj } = net(a);
    const grid = new EnergyGrid(10, 0);
    const cap = a[0].energyCap;
    const before = a[0].extra + a[1].extra;
    flow(list, adj, { quantum: packet, grid });
    expect(a[0].extra, 'receiver fills to the brim').toBeCloseTo(cap, 9);
    expect(a[1].extra, 'donor is a whole packet lighter').toBeCloseTo(0.5, 9);
    expect(grid.storedTotal()).toBeCloseTo(packet - (cap - 1), 9);
    const after = a[0].extra + a[1].extra + grid.storedTotal();
    expect(after, 'nothing minted, nothing lost').toBeCloseTo(before, 9);
  });

  it('refuses to lose an overflow it has nowhere to put', () => {
    // Silently dropping the spill would put a hole in the one invariant this
    // file exists to protect, and it would only show up as a slow leak. It
    // throws where the loss would happen rather than up front, so a run whose
    // receivers all have room is not made to carry a grid it never needs.
    const roomy = pond();
    const ra = [roomy.body(0, 0, 0, 1), roomy.body(10, 0, 1, 0)];
    roomy.chain(ra);
    const r = roomy.net(ra);
    expect(() => roomy.flow(r.list, r.adj, { quantum: packet })).not.toThrow();

    // The receiver has 0.25 of room and a whole packet is coming.
    const full = pond();
    const fa = [full.body(0, 0, 1, 1), full.body(10, 0, 1, 0)];
    full.chain(fa);
    const f = full.net(fa);
    expect(() => full.flow(f.list, f.adj, { quantum: packet })).toThrow(/grid/);
  });

  it('is the continuous law exactly at a forced quantum of zero', () => {
    const one = pond();
    const oa = [one.body(0, 0, 0, 0.7), one.body(10, 0, 1, 0)];
    one.chain(oa);
    const o = one.net(oa);
    const two = pond();
    const ta = [two.body(0, 0, 0, 0.7), two.body(10, 0, 1, 0)];
    two.chain(ta);
    const t = two.net(ta);
    const withOpts = one.flow(o.list, o.adj, { quantum: 0 });
    expect(withOpts).toBe(two.flow(t.list, t.adj));
    expect(oa[0].extra).toBe(ta[0].extra);
  });
});
