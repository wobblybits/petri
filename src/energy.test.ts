import { describe, expect, it } from 'vitest';
import { fixedParams } from './test-params.ts';
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
  flowCharges,
  harvestSlots,
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
  type SlotBody,
  spareEnergy,
  spendExtra,
  snapshotRequests,
  spreadRequests,
  tickUpkeep,
  WireAdjacency,
} from './energy.ts';
import { Sim } from './sim.ts';
import { type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { CH, CHANNELS } from './fields.ts';

/** Sum of one channel over the whole field — the harness's own helper. */
function channelTotal(sim: Sim, ch: number): number {
  const d = sim.fields.data;
  let s = 0;
  for (let k = ch; k < d.length; k += CHANNELS) s += d[k];
  return s;
}

/** Defaults to `con`: Era has its own upkeep rate, so kind matters here. */
function body(
  id: number,
  x: number,
  y: number,
  extra = 0,
  request = 0,
  locked = false,
  kind: SlotBody['kind'] = 'con',
  energyCap = extraCapFor(kind),
  requestDecay = REQUEST_DECAY,
  debtCap = EXTRA_FLOOR,
  rescueTo = 0.9,
): SlotBody {
  return {
    id, kind, x, y, extra, request, locked, recovering: false,
    energyCap, requestDecay, debtCap, rescueTo, transportQuantum: 0,
  };
}


/**
 * Dense list plus the flat adjacency the energy passes take. They work in
 * index space now: a Map of neighbour arrays cost an array per body per frame.
 */
function net(agents: Map<number, SlotBody>, wires: { a: { id: number }; b: { id: number } }[]) {
  const list = [...agents.values()];
  const index = new Map<number, number>();
  list.forEach((b, i) => index.set(b.id, i));
  const adj = new WireAdjacency();
  adj.build(list.length, index, () => wires);
  return { list, adj };
}

/**
 * Run the need field to its fixpoint.
 *
 * `spreadRequests` carries demand one hop a frame now, so a test that wants
 * the settled potential has to let it settle. The fixpoint is the same one
 * the old in-frame relaxation solved, and a path can be no longer than the
 * roster, so this many steps always reaches it.
 */
function settle(list: Parameters<typeof spreadRequests>[0], adj: Parameters<typeof spreadRequests>[1], decay?: number): void {
  const prev: number[] = [];
  for (let i = 0; i <= list.length; i++) {
    snapshotRequests(list, prev);
    spreadRequests(list, adj, prev, decay);
  }
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
    expect(deathYield(body(1, 0, 0, EXTRA_CAP)), 'full').toBeCloseTo(2.5, 6);
    expect(deathYield(body(2, 0, 0, 0)), 'break-even').toBeCloseTo(BODY_VALUE, 6);
    expect(deathYield(body(3, 0, 0, EXTRA_FLOOR)), 'starved').toBeCloseTo(0.25, 6);
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
    const grid = new EnergyGrid(10, 0.1);
    const a = body(1, 2, 2);
    harvestSlots([a], grid);
    expect(a.extra).toBeCloseTo(0.1);
    expect(canPayShare(a)).toBe(false);
    expect(grid.getAt(2, 2)).toBe(0);
  });

  it('fills a slot from a cell of 1', () => {
    const grid = new EnergyGrid(10, 1);
    const a = body(1, 2, 2);
    harvestSlots([a], grid);
    expect(a.extra).toBeCloseTo(1);
    expect(grid.getAt(2, 2)).toBe(0);
  });

  it('gives a shared cell of 1 to the lower id only', () => {
    const grid = new EnergyGrid(10, 1);
    const a = body(1, 2, 2);
    const b = body(2, 3, 2);
    harvestSlots([a, b], grid);
    expect(a.extra).toBeCloseTo(1);
    expect(b.extra).toBe(0);
    expect(grid.getAt(2, 2)).toBe(0);
  });

  /*
   * Monod uptake. `docs/energy-chemistry-plan.md` §4.
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

  it('takes what fits, exactly as before, at uptakeVmax zero', () => {
    // Bit-identical, not close: the whole discipline of the plan is that a
    // dial at its neutral value leaves the pond it was added to alone.
    const plain = new EnergyGrid(10, 1);
    const a1 = body(1, 2, 2);
    harvestSlots([a1], plain);

    const metered = new EnergyGrid(10, 1);
    const a2 = body(1, 2, 2);
    harvestSlots([a2], metered, { cap: 0, ks: 0.25, table: false, yDirect: 1, yEra: 1, hillN: 1, coSubstrate: 0 });

    expect(a2.extra).toBe(a1.extra);
    expect(metered.getAt(2, 2)).toBe(plain.getAt(2, 2));
  });

  it('shares a contested cell instead of handing it to the lowest id', () => {
    // The sibling test above — 'gives a shared cell of 1 to the lower id
    // only' — is the artifact this removes. Same cell, same two bodies.
    const grid = new EnergyGrid(10, 1);
    const a = body(1, 2, 2);
    const b = body(2, 3, 2);
    // A rate well under the cell's stock, so neither can drain it in a frame.
    harvestSlots([a, b], grid, { cap: 0.2, ks: 0.25, table: false, yDirect: 1, yEra: 1, hillN: 1, coSubstrate: 0 });
    expect(a.extra).toBeGreaterThan(0);
    expect(b.extra).toBeGreaterThan(0);
    // Drawn concurrently, so they get the same rate; order stops mattering
    // except at exhaustion.
    expect(b.extra).toBeCloseTo(a.extra, 9);
    expect(a.extra + b.extra).toBeLessThanOrEqual(1 + 1e-9);
  });

  it('caps a rich cell at the rate and a poor one below it', () => {
    const rich = new EnergyGrid(10, 4);
    const r = body(1, 2, 2);
    harvestSlots([r], rich, { cap: 0.2, ks: 0.25, table: false, yDirect: 1, yEra: 1, hillN: 1, coSubstrate: 0 });
    // Saturated: near vmax*dt, and nowhere near the tank's room.
    expect(r.extra).toBeGreaterThan(0.18);
    expect(r.extra).toBeLessThanOrEqual(0.2 + 1e-9);

    const poor = new EnergyGrid(10, 0.05);
    const p = body(1, 2, 2);
    harvestSlots([p], poor, { cap: 0.2, ks: 0.25, table: false, yDirect: 1, yEra: 1, hillN: 1, coSubstrate: 0 });
    // Below half-saturation, so the rate is well under vmax — and this is the
    // half of Monod that makes a low-Ks scavenger a viable different strategy
    // rather than a strictly worse grazer.
    expect(p.extra).toBeLessThan(r.extra);
    expect(p.extra).toBeCloseTo(uptakeRate(0.05, 0.2, 0.25), 9);
  });

  it('does not fill a slot that is already at the cap', () => {
    const grid = new EnergyGrid(10, 1);
    const a = body(1, 2, 2, EXTRA_CAP);
    expect(atCap(a)).toBe(true);
    harvestSlots([a], grid);
    expect(a.extra).toBe(EXTRA_CAP);
    expect(grid.getAt(2, 2), 'the cell is untouched').toBe(1);
  });

  it('keeps harvesting a body that can pay a share but is not yet full', () => {
    // The two used to be the same test. Being able to commute is not being
    // full, and a body that stopped topping up at the share would have no
    // headroom against the next few seconds of upkeep.
    const grid = new EnergyGrid(10, 1);
    const a = body(1, 2, 2, REWRITE_SHARE);
    expect(canPayShare(a)).toBe(true);
    expect(atCap(a)).toBe(false);
    harvestSlots([a], grid);
    expect(a.extra).toBeCloseTo(EXTRA_CAP, 6);
  });

  it('skips locked agents', () => {
    const grid = new EnergyGrid(10, 1);
    const a = body(1, 2, 2, 0, 0, true);
    harvestSlots([a], grid);
    expect(a.extra).toBe(0);
    expect(grid.getAt(2, 2)).toBe(1);
  });

  it('fills every agent on an inexhaustible cell', () => {
    const grid = new EnergyGrid(10, 8);
    grid.inexhaustible = true;
    const a = body(1, 2, 2);
    const b = body(2, 3, 2);
    harvestSlots([a, b], grid);
    expect(a.extra).toBeCloseTo(EXTRA_CAP, 6);
    expect(b.extra).toBeCloseTo(EXTRA_CAP, 6);
    expect(grid.getAt(2, 2)).toBe(8);
  });
});

describe('upkeep', () => {
  it("drains extra continuously and reports death at the body's own debt cap", () => {
    const a = body(1, 0, 0, 1);
    expect(tickUpkeep([a], 0.5, 1)).toEqual([]);
    expect(a.extra, 'half a second at one a second').toBeCloseTo(0.5, 6);
    expect(tickUpkeep([a], 1, 1)).toEqual([]);
    expect(a.extra, 'and straight on into debt').toBeCloseTo(-0.5, 6);
    expect(tickUpkeep([a], 1, 1), 'reaching the floor is a death').toEqual([1]);
    expect(a.extra).toBe(EXTRA_FLOOR);
    expect(tickUpkeep([a], 1, 1), 'reported once, not every frame after').toEqual([]);
    expect(a.extra, 'and never falls past it').toBe(EXTRA_FLOOR);
  });

  it('kills a shallow-debt body before a deep-debt one', () => {
    const shallow = body(1, 0, 0, 0);
    shallow.debtCap = -0.2;
    const deep = body(2, 0, 0, 0);
    deep.debtCap = -2;
    expect(tickUpkeep([shallow, deep], 0.3, 1)).toEqual([1]);
    expect(shallow.extra).toBeCloseTo(-0.2, 6);
    expect(deep.extra).toBeCloseTo(-0.3, 6);
    expect(deep.extra, 'still alive, still in debt').toBeGreaterThan(deep.debtCap);
  });

  it('skips locked agents', () => {
    const a = body(1, 0, 0, 1, 0, true);
    expect(tickUpkeep([a], 10, 1)).toEqual([]);
    expect(a.extra).toBe(1);
  });
});

describe('request gradient', () => {
  it('attenuates need by distance instead of counting hops', () => {
    const agents = new Map([
      [1, body(1, 0, 0)],
      [2, body(2, 10, 0)],
      [3, body(3, 20, 0)],
    ]);
    const { list, adj } = net(agents, [
      { a: { id: 1 }, b: { id: 2 } },
      { a: { id: 2 }, b: { id: 3 } },
    ]);
    seedRequest(agents.get(1)!, 1);
    settle(list, adj);
    expect(agents.get(1)!.request).toBe(1);
    expect(agents.get(2)!.request).toBeCloseTo(REQUEST_DECAY, 6);
    expect(agents.get(3)!.request).toBeCloseTo(REQUEST_DECAY ** 2, 6);
  });

  it('lets a big distant need outrank a small near one', () => {
    // Chain 1-2-3-4. A large need at 1 must beat a small need at 4 for the
    // body at 3, even though 4 is adjacent — that is what magnitude buys.
    const agents = new Map([
      [1, body(1, 0, 0)],
      [2, body(2, 10, 0)],
      [3, body(3, 20, 0)],
      [4, body(4, 30, 0)],
    ]);
    const { list, adj } = net(agents, [
      { a: { id: 1 }, b: { id: 2 } },
      { a: { id: 2 }, b: { id: 3 } },
      { a: { id: 3 }, b: { id: 4 } },
    ]);
    seedRequest(agents.get(1)!, 1);
    seedRequest(agents.get(4)!, 0.2);
    settle(list, adj);
    // 1 reaches 3 at 0.8^2 = 0.64; 4 only offers 0.2 there.
    expect(agents.get(3)!.request).toBeCloseTo(REQUEST_DECAY ** 2, 6);
    expect(agents.get(3)!.request).toBeGreaterThan(agents.get(4)!.request);
  });

  it('walks energy up the gradient, one hop a frame', () => {
    const agents = new Map([
      [1, body(1, 0, 0, 0, 1)],
      [2, body(2, 10, 0, 0, REQUEST_DECAY)],
      [3, body(3, 20, 0, 1, REQUEST_DECAY ** 2)],
    ]);
    const { list, adj } = net(agents, [
      { a: { id: 1 }, b: { id: 2 } },
      { a: { id: 2 }, b: { id: 3 } },
    ]);
    // 3 holds the surplus and 1 is the one that needs it, two hops away. Each
    // frame moves one wire's worth, throttled by the field at the receiving
    // end — 0.8 here, since that is how much of 1's need is visible from 2.
    expect(flowCharges(list, adj)).toBeCloseTo(REQUEST_DECAY, 6);
    expect(agents.get(3)!.extra).toBeCloseTo(1 - REQUEST_DECAY, 6);
    expect(agents.get(2)!.extra).toBeCloseTo(REQUEST_DECAY, 6);
    flowCharges(list, adj);
    expect(agents.get(1)!.extra, 'reaches the body that needs it').toBeGreaterThan(0.5);
  });

  it('delivers only as much as the recipient can hold', () => {
    const agents = new Map([
      [1, body(1, 0, 0, 0.7, 0.3)],
      [2, body(2, 10, 0, 1, 0.24)],
    ]);
    const { list, adj } = net(agents, [{ a: { id: 1 }, b: { id: 2 } }]);
    expect(flowCharges(list, adj)).toBeCloseTo(0.3, 6);
    expect(agents.get(1)!.extra).toBeCloseTo(1, 6);
    expect(agents.get(2)!.extra, 'donor keeps the rest').toBeCloseTo(0.7, 6);
  });

  it('relays through a body that needs nothing itself', () => {
    // The middle body is not hungry and is not a redex. If it could only
    // accept what it needs, every shortage more than one wire from a surplus
    // would be unreachable.
    const agents = new Map([
      [1, body(1, 0, 0, 0, 1)],
      [2, body(2, 10, 0, 0, REQUEST_DECAY)],
      [3, body(3, 20, 0, 1, REQUEST_DECAY ** 2)],
    ]);
    const { list, adj } = net(agents, [
      { a: { id: 1 }, b: { id: 2 } },
      { a: { id: 2 }, b: { id: 3 } },
    ]);
    flowCharges(list, adj);
    expect(agents.get(2)!.extra, 'held by the conduit').toBeCloseTo(REQUEST_DECAY, 6);
  });

  it('leaves a flat spot where two equal needs meet', () => {
    // Needs of the same size at both ends of a 3-chain give the middle body
    // the same field from either side, so nothing crosses it.
    const agents = new Map([
      [1, body(1, 0, 0, 0)],
      [2, body(2, 10, 0, 1)],
      [3, body(3, 20, 0, 0)],
    ]);
    const { list, adj } = net(agents, [
      { a: { id: 1 }, b: { id: 2 } },
      { a: { id: 2 }, b: { id: 3 } },
    ]);
    seedRequest(agents.get(1)!, 1);
    seedRequest(agents.get(3)!, 1);
    settle(list, adj);
    expect(agents.get(1)!.request).toBeCloseTo(agents.get(3)!.request, 6);
    // The middle body is the one holding energy, and both neighbours pull on
    // it equally hard, so it gives to exactly one of them rather than tearing.
    const moved = flowCharges(list, adj);
    expect(moved).toBeGreaterThan(0);
    const fed = [agents.get(1)!.extra, agents.get(3)!.extra];
    expect(fed.filter((e) => e > 0.5).length, 'one of the two, not both').toBe(1);
  });

  it('does not pass energy to a body that is no needier than the donor', () => {
    const agents = new Map([
      [1, body(1, 0, 0, 1, 1)],
      [2, body(2, 10, 0, 0, 1)],
    ]);
    const { list, adj } = net(agents, [{ a: { id: 1 }, b: { id: 2 } }]);
    expect(flowCharges(list, adj)).toBe(0);
    expect(agents.get(1)!.extra).toBe(1);
    expect(canPayShare(agents.get(1)!), 'the donor could have paid a share').toBe(true);
    expect(canPayShare(agents.get(2)!), 'the other could not, and got nothing to change that').toBe(
      false,
    );
  });

  it('reads hunger straight off the debt', () => {
    expect(hungerNeed(body(1, 0, 0, 1)), 'stocked').toBe(0);
    expect(hungerNeed(body(2, 0, 0, 0)), 'break-even').toBe(0);
    expect(hungerNeed(body(3, 0, 0, -0.1))).toBeCloseTo(0.1, 6);
    expect(hungerNeed(body(4, 0, 0, -1)), 'at the point of death').toBe(1);
  });

  it('has nothing to pass on while it is in debt', () => {
    expect(spareEnergy(body(1, 0, 0, 0.4))).toBeCloseTo(0.4, 6);
    expect(spareEnergy(body(2, 0, 0, 0))).toBe(0);
    expect(spareEnergy(body(3, 0, 0, -0.4))).toBe(0);
  });

  it('settles a body\'s own debt before anything travels further', () => {
    // 1 is deep in debt, 2 is shallowly in debt, 3 has stock. What reaches 2
    // pays 2 off first; only what is left over can reach 1.
    const agents = new Map([
      [1, body(1, 0, 0, -0.9)],
      [2, body(2, 10, 0, -0.2)],
      [3, body(3, 20, 0, 1)],
    ]);
    const { list, adj } = net(agents, [
      { a: { id: 1 }, b: { id: 2 } },
      { a: { id: 2 }, b: { id: 3 } },
    ]);
    for (let i = 0; i < 2; i++) {
      resetRequests(agents.values());
      for (const a of agents.values()) seedRequest(a, hungerNeed(a));
      settle(list, adj);
      flowCharges(list, adj);
    }
    expect(agents.get(2)!.extra, 'out of debt first').toBeGreaterThanOrEqual(0);
    const total = [...agents.values()].reduce((t, a) => t + a.extra, 0);
    expect(total, 'and nothing minted').toBeCloseTo(-0.1, 6);
  });

  it('carries a shortage further at a slower decay', () => {
    const chain = (decay: number) => {
      const agents = new Map(
        Array.from({ length: 6 }, (_, i) => [i + 1, body(i + 1, i * 10, 0)] as const),
      );
      agents.get(1)!.request = 1;
      const wires = Array.from({ length: 5 }, (_, i) => ({
        a: { id: i + 1 },
        b: { id: i + 2 },
      }));
      const { list, adj } = net(agents, wires);
      settle(list, adj, decay);
      return agents.get(6)!.request;
    };
    expect(chain(0.8), 'five hops at 0.8').toBeCloseTo(0.8 ** 5, 6);
    expect(chain(0.9), 'and further at 0.9').toBeCloseTo(0.9 ** 5, 6);
    expect(chain(0.9)).toBeGreaterThan(chain(0.8));
  });

  it('lets each body relay demand at its own rate instead of one global decay', () => {
    const agents = new Map(
      Array.from({ length: 6 }, (_, i) => [i + 1, body(i + 1, i * 10, 0)] as const),
    );
    agents.get(1)!.request = 1;
    for (const a of agents.values()) a.requestDecay = 0.9;
    agents.get(3)!.requestDecay = 0.05;
    const wires = Array.from({ length: 5 }, (_, i) => ({
      a: { id: i + 1 },
      b: { id: i + 2 },
    }));
    const { list, adj } = net(agents, wires);
    settle(list, adj); // no override: each body's own field governs
    expect(agents.get(2)!.request, 'upstream of the lossy relay is unaffected').toBeCloseTo(
      0.9,
      6,
    );
    expect(agents.get(4)!.request, 'the lossy relay chokes what crosses it').toBeCloseTo(
      0.9 * 0.9 * 0.05,
      6,
    );
    expect(agents.get(6)!.request).toBeLessThan(0.9 ** 5 * 0.1);
  });

  it('never lets the field go flat', () => {
    // At decay 1 every body holds the same need, no neighbour is strictly
    // needier than its donor, and nothing moves at all.
    const agents = new Map([
      [1, body(1, 0, 0, 0, 1)],
      [2, body(2, 10, 0, 1)],
    ]);
    const { list, adj } = net(agents, [{ a: { id: 1 }, b: { id: 2 } }]);
    settle(list, adj, 1);
    expect(agents.get(2)!.request).toBeLessThan(agents.get(1)!.request);
    expect(flowCharges(list, adj), 'so the surplus still crosses').toBeGreaterThan(0);
  });

  it('keeps a rescued body asking until it can act again', () => {
    // Fill fraction 1 on a tank whose cap is a rewrite share: rescue aims at
    // extra=1, which is what used to be the global absolute target.
    const a = body(1, 0, 0, -0.4);
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
    const poor = body(1, 0, 0, 0.05);
    expect(rescueNeed(poor)).toBe(0);
    expect(poor.recovering).toBe(false);
  });

  it('rescues only to break-even at fill 0, as it used to', () => {
    const a = body(1, 0, 0, -0.4);
    a.rescueTo = 0;
    expect(rescueNeed(a)).toBeCloseTo(0.4, 6);
    a.extra = 0;
    expect(rescueNeed(a)).toBe(0);
  });

  it("aims a rescue between the body's own floor and cap", () => {
    const a = body(1, 0, 0);
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
    expect(redexNeed(body(1, 0, 0, 0.8))).toBeCloseTo(0.2, 6);
    expect(redexNeed(body(2, 0, 0, 1))).toBe(0);
  });

  it('clears requests', () => {
    const a = body(1, 0, 0, 0, 4);
    resetRequests([a]);
    expect(a.request).toBe(0);
  });
});

describe('settle pool', () => {
  it('gives the neediest survivor first refusal', () => {
    // Released energy enters the net at the body the gradient would have sent
    // it to. `easy` has the lower id, so id order would have fed it first and
    // left the hungry one short.
    const grid = new EnergyGrid(10, 0);
    const easy = body(1, 0, 0, 1, 0.1);
    const hungry = body(2, 0, 0, -0.5, 0.9);
    settlePool(1, [easy, hungry], grid, 5, 5);
    expect(hungry.extra, 'all of it went to the needy one').toBeCloseTo(0.5, 6);
    expect(easy.extra, 'which had room but no claim on it').toBe(1);
    expect(grid.getAt(5, 5)).toBe(0);
  });

  it('drops what the survivors cannot hold onto the ground', () => {
    // Two bodies, 2.0 of room between them, against a two-body annihilation.
    // The per-body cap is the bandwidth limit; the rest lands where it died.
    const grid = new EnergyGrid(10, 0);
    const a = body(1, 0, 0, -0.5, 0.9);
    const b = body(2, 0, 0, 1, 0.1);
    settlePool(2 * BODY_VALUE, [a, b], grid, 5, 5);
    expect(a.extra).toBeCloseTo(EXTRA_CAP, 6);
    expect(b.extra).toBeCloseTo(EXTRA_CAP, 6);
    expect(grid.getAt(5, 5), 'overflow').toBeCloseTo(2 * BODY_VALUE - 2, 6);
  });

  it('fills empty leftover slots before dumping to the grid', () => {
    const grid = new EnergyGrid(10, 0);
    const a = body(1, 0, 0);
    const b = body(2, 0, 0, EXTRA_CAP);
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
    const era = body(1, 0, 0, 0, 0, false, 'era');
    const con = body(2, 0, 0, 0, 0, false, 'con');
    // Long enough that a Con has been billed several times over.
    for (let i = 0; i < 600; i++) tickUpkeep([era, con], 1, 0.025);
    expect(con.extra, 'a Con burns down to the floor').toBe(EXTRA_FLOOR);
    expect(era.extra, 'an Era fills instead, to its own deeper cap').toBe(extraCapFor('era'));
    expect(tickUpkeep([era], 1000, 0.025), 'and is never reported starved').toEqual([]);
  });

  it('gives an Era a deeper tank than a Con or a Dup', () => {
    expect(extraCapFor('era')).toBeCloseTo(EXTRA_CAP * ERA_CAP_RATIO, 6);
    expect(extraCapFor('con')).toBe(EXTRA_CAP);
    expect(extraCapFor('dup')).toBe(EXTRA_CAP);
    const era = body(1, 0, 0, EXTRA_CAP, 0, false, 'era');
    expect(atCap(era), 'a Con-sized tankful is only half an Era').toBe(false);
    era.extra = extraCapFor('era');
    expect(atCap(era)).toBe(true);
  });

  it('caps a body by its own energyCap, not by a flat per-kind number', () => {
    // energyCap is heritable now — two Cons can carry different tanks — so the
    // cap has to come from the body, and extraCapFor is only ever the seed a
    // fresh one starts at.
    const roomy = body(1, 0, 0, EXTRA_CAP, 0, false, 'con', EXTRA_CAP * 2);
    const cramped = body(2, 0, 0, EXTRA_CAP * 0.5, 0, false, 'con', EXTRA_CAP * 0.5);
    expect(atCap(roomy), 'a normal-sized tankful is not full for the bigger tank').toBe(false);
    expect(atCap(cramped), 'but the smaller tank is already topped out at the same level').toBe(
      true,
    );
    const grid = new EnergyGrid(10, 0);
    grid.setCell(0, 0, 10);
    harvestSlots([roomy, cramped], grid);
    expect(roomy.extra).toBeCloseTo(roomy.energyCap, 6);
    expect(cramped.extra, 'no room left to harvest into').toBeCloseTo(EXTRA_CAP * 0.5, 6);
  });

  it('lets a full tank pay a rewrite even when breeding shrank it below a share', () => {
    const cramped = body(1, 0, 0, EXTRA_CAP * 0.5, 0, false, 'con', EXTRA_CAP * 0.5);
    expect(cramped.energyCap).toBeLessThan(REWRITE_SHARE);
    expect(canPayShare(cramped)).toBe(true);
    const roomy = body(2, 0, 0, EXTRA_CAP * 0.5, 0, false, 'con', EXTRA_CAP);
    expect(canPayShare(roomy), 'a default tank still needs a whole share').toBe(false);
    expect(rewriteShareOf(cramped), 'and owes only what it can hold').toBeCloseTo(
      EXTRA_CAP * 0.5,
      6,
    );
    expect(stakeMet(cramped, cramped.extra), 'so a full small tank meets its stake').toBe(true);
  });

  it('fills an Era past a full Con from the ground and along a wire', () => {
    const grid = new EnergyGrid(10, 0);
    const era = body(1, 0, 0, 0, 0, false, 'era');
    grid.setCell(0, 0, 4);
    harvestSlots([era], grid);
    expect(era.extra, 'forages up to its own cap').toBeCloseTo(extraCapFor('era'), 6);

    const drained = body(2, 0, 0, 0.5, 0, false, 'era');
    const donor = body(3, 10, 0, EXTRA_CAP, 0, false, 'con');
    const agents = new Map([
      [drained.id, drained],
      [donor.id, donor],
    ]);
    const { list, adj } = net(agents, [{ a: { id: 2 }, b: { id: 3 } }]);
    seedRequest(drained, 1);
    settle(list, adj);
    flowCharges(list, adj);
    expect(drained.extra, 'and a pump can push it past a Con-sized full').toBeGreaterThan(
      EXTRA_CAP,
    );
  });

  it('caps an Era at full however long it produces for', () => {
    const cap = extraCapFor('era');
    const era = body(1, 0, 0, 0, 0, false, 'era');
    for (let i = 0; i < 5000; i++) tickUpkeep([era], 1, 0.025);
    expect(era.extra, 'no banking past the cap').toBe(cap);
    // And the next share has to be earned at the same rate as the first.
    spendExtra(era);
    expect(era.extra, 'a share out of a full tank leaves the headroom')
      .toBeCloseTo(cap - REWRITE_SHARE, 6);
    tickUpkeep([era], 1, 0.025);
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
   * `docs/energy-chemistry-plan.md` §5, as one assertion:
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
    // The three dials.
    p.upkeepExcrete = 1;
    p.bodyValue = REWRITE_SHARE;
    p.eraUpkeepRatio = 1;
    return p;
  }

  /**
   * Everything the pond is made of, counting a body's debt against it.
   *
   * `totalFree` deliberately floors at zero — it answers "how much can be
   * spent" — and that is the wrong question here. A body one unit into debt
   * holds `bodyValue - 1` of real matter, and `deathYield` releases exactly
   * that when it dies. Counting its stock as zero instead would make every
   * starvation look like matter vanishing, when what vanished was a debt.
   */
  const pondTotal = (sim: Sim, bodyValue: number): number => {
    let inBodies = 0;
    for (const a of sim.agents.values()) inBodies += bodyValue + a.extra;
    /*
     * Plus what is in flight. A rewrite charges its pair `rewriteCost` when it
     * *begins* and pays the pool out when it *commits*, and in between the
     * shares are held by the `Rewrite` itself — not by a body, not by the
     * escrow map, and so not by any of the three totals above. Two seconds of
     * rewrite duration at two shares apiece is a swing of a few units that
     * closes itself every time.
     */
    let inFlight = 0;
    // Per rule: only a commute costs shares up front. An erase or an
    // annihilation is free to start and pays out on commit.
    for (const rw of sim.rewrites) inFlight += rewriteCost(rw.rule);
    return inBodies + inFlight + sim.escrowTotal() + sim.energy.storedTotal();
  };

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
    const before = pondTotal(sim, p.bodyValue);
    let worst = 0;
    for (let i = 0; i < 900; i++) {
      sim.step(1 / 60, p);
      worst = Math.max(worst, Math.abs(pondTotal(sim, p.bodyValue) - before));
    }
    // Something has to have happened, or this asserts about a still pond.
    expect(sim.tally.annihilations + sim.tally.commutes, 'no rewrites ran').toBeGreaterThan(4);
    /*
     * Not exact: a pair that annihilates while in debt releases less than it
     * holds, and that debt dies with it rather than being minted away — see
     * `tickRewrites`, where the pool is clamped at zero. Everything else
     * balances to a part in a hundred thousand, over nine hundred frames of a
     * pond that latched, commuted, erased and annihilated throughout.
     */
    expect(worst / before, `drifted ${worst} of ${before}`).toBeLessThan(1e-4);
  });

  it('conserves across a whole pond on the shipping path', () => {
    /*
     * The one that matters. `flowChargesFast` is what runs, it is a hand-kept
     * twin of the reference above, and nothing else in the suite compares the
     * two — so this drives the real thing for nine hundred frames of a pond
     * that latches, commutes and annihilates, with packets crossing wires and
     * spilling onto the ground throughout, and checks the books still balance.
     */
    const p = conservativeParams();
    p.transportQuantum = 0.5;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const before = pondTotal(sim, p.bodyValue);
    let worst = 0;
    for (let i = 0; i < 900; i++) {
      sim.step(1 / 60, p);
      worst = Math.max(worst, Math.abs(pondTotal(sim, p.bodyValue) - before));
    }
    expect(sim.tally.annihilations + sim.tally.commutes, 'no rewrites ran').toBeGreaterThan(4);
    expect(worst / before, `drifted ${worst} of ${before}`).toBeLessThan(1e-4);
  });

  it('destroys the rent when upkeepExcrete is off, which is today', () => {
    // The dial's other end, so the test above cannot pass by the invariant
    // being vacuous. Rent vanishing is what the pond does now.
    const p = conservativeParams();
    p.upkeepExcrete = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const before = pondTotal(sim, p.bodyValue);
    for (let i = 0; i < 900; i++) sim.step(1 / 60, p);
    expect(pondTotal(sim, p.bodyValue)).toBeLessThan(before * 0.999);
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
    const agents = new Map([
      [1, body(1, 0, 0, 0, 1)],
      [2, body(2, 10, 0, 0.4, 0)],
    ]);
    const { list, adj } = net(agents, [{ a: { id: 1 }, b: { id: 2 } }]);
    const grid = new EnergyGrid(10, 0);
    expect(flowCharges(list, adj, undefined, { quantum: packet, grid })).toBe(0);
    expect(agents.get(2)!.extra, 'the donor keeps it all').toBeCloseTo(0.4, 9);
    expect(agents.get(1)!.extra).toBe(0);
    expect(grid.storedTotal(), 'and nothing leaked to the ground').toBeCloseTo(0, 9);
    // The same pond under the law it replaces, so this cannot pass vacuously.
    expect(flowCharges(list, adj)).toBeCloseTo(0.4, 9);
  });

  it('sends a whole packet, however little the receiver asked for', () => {
    // Demand still decides *whether* to send — the receiver must be strictly
    // needier — but it no longer decides how much. That cap is what kept every
    // transfer down to the size of the gradient.
    const agents = new Map([
      [1, body(1, 0, 0, 0, 0.2)],
      [2, body(2, 10, 0, 1, 0)],
    ]);
    const { list, adj } = net(agents, [{ a: { id: 1 }, b: { id: 2 } }]);
    const grid = new EnergyGrid(10, 0);
    const moved = flowCharges(list, adj, undefined, { quantum: packet, grid });
    expect(moved, 'a packet, not the 0.2 the gradient asked for').toBeCloseTo(packet, 9);
    expect(agents.get(1)!.extra).toBeCloseTo(0.5, 9);
    expect(agents.get(2)!.extra).toBeCloseTo(0.5, 9);
  });

  it('spills what will not fit onto the ground under the receiver', () => {
    // A whole packet crosses whether or not the far end has room, so the
    // remainder has to land somewhere. The same place a rewrite's leftovers go.
    const agents = new Map([
      [1, body(1, 0, 0, 1, 1)],
      [2, body(2, 10, 0, 1, 0)],
    ]);
    const { list, adj } = net(agents, [{ a: { id: 1 }, b: { id: 2 } }]);
    const grid = new EnergyGrid(10, 0);
    const cap = agents.get(1)!.energyCap;
    const before = agents.get(1)!.extra + agents.get(2)!.extra;
    flowCharges(list, adj, undefined, { quantum: packet, grid });
    expect(agents.get(1)!.extra, 'receiver fills to the brim').toBeCloseTo(cap, 9);
    expect(agents.get(2)!.extra, 'donor is a whole packet lighter').toBeCloseTo(0.5, 9);
    expect(grid.storedTotal()).toBeCloseTo(packet - (cap - 1), 9);
    const after = agents.get(1)!.extra + agents.get(2)!.extra + grid.storedTotal();
    expect(after, 'nothing minted, nothing lost').toBeCloseTo(before, 9);
  });

  it('refuses to lose an overflow it has nowhere to put', () => {
    // Silently dropping the spill would put a hole in the one invariant this
    // file exists to protect, and it would only show up as a slow leak. It
    // throws where the loss would happen rather than up front, so a run whose
    // receivers all have room is not made to carry a grid it never needs.
    const roomy = new Map([[1, body(1, 0, 0, 0, 1)], [2, body(2, 10, 0, 1, 0)]]);
    const a = net(roomy, [{ a: { id: 1 }, b: { id: 2 } }]);
    expect(() => flowCharges(a.list, a.adj, undefined, { quantum: packet })).not.toThrow();

    // The receiver has 0.25 of room and a whole packet is coming.
    const full = new Map([[1, body(1, 0, 0, 1, 1)], [2, body(2, 10, 0, 1, 0)]]);
    const b = net(full, [{ a: { id: 1 }, b: { id: 2 } }]);
    expect(() => flowCharges(b.list, b.adj, undefined, { quantum: packet })).toThrow(/grid/);
  });

  it('is the old law exactly at zero', () => {
    const agents = new Map([[1, body(1, 0, 0, 0, 0.7)], [2, body(2, 10, 0, 1, 0)]]);
    const { list, adj } = net(agents, [{ a: { id: 1 }, b: { id: 2 } }]);
    const plain = new Map([[1, body(1, 0, 0, 0, 0.7)], [2, body(2, 10, 0, 1, 0)]]);
    const two = net(plain, [{ a: { id: 1 }, b: { id: 2 } }]);
    const withOpts = flowCharges(list, adj, undefined, { quantum: 0 });
    expect(withOpts).toBe(flowCharges(two.list, two.adj));
    expect(agents.get(1)!.extra).toBe(plain.get(1)!.extra);
  });
});
