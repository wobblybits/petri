import { describe, expect, it } from 'vitest';
import { CH } from './fields.ts';
import { EMIT, seedChem } from './agents.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';


describe('port occupancy', () => {
  it('tracks how much of a body is attached, and updates when that changes', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.rewriteDuration = 0;
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('con', 2000, 2000, 0, params, true)!;
    const b = sim.spawn('era', 2060, 2000, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    expect(a.bound, 'nothing attached').toBe(0);
    expect(b.bound).toBe(0);

    // A Con has three ports, so one wire is a third of it. An Era has one, so
    // the same wire is the whole of it.
    sim.wire(a.id, 'p', b.id, 'p', params);
    sim.step(1 / 60, params);
    expect(a.bound).toBeCloseTo(1 / 3, 9);
    expect(b.bound).toBe(1);

    // And it comes back when the socket reopens, which is the case the whole
    // dimension exists for.
    for (const w of [...sim.graph.wires.keys()]) sim.graph.detach(w);
    sim.step(1 / 60, params);
    expect(a.bound).toBe(0);
    expect(b.bound).toBe(0);
  });
});

describe('farming', () => {
  /*
   * Farming is the ground's excretion row, at `excreteRate`.
   *
   * It was a pass of its own, `farmRate` times the emit head's ground slot,
   * until §3's table finally claimed it: `seedProduction` puts an Era's whole
   * production half on `excrete_2`, so a farmer is simply a body whose
   * metabolism makes ground. Mass action rather than a flat rate: a frame's
   * excretion is under the stock at any `excreteRate * dt * ROW_COUNT` below
   * one, and `runExcretion`'s scale `k` is the clamp for a long frame or a
   * rate above that.
   */
  /** A lone body on ground it has already stripped, so growth has nothing to work on. */
  function scarred(params: ReturnType<typeof defaultParams>) {
    const sim = new Sim(10000, 10000);
    const a = sim.spawn('era', 5000, 5000, 0, params, true)!;
    sim.step(1 / 60, params);
    const f = sim.fields;
    for (let i = CH.energy; i < f.data.length; i += 4) f.data[i] = 0;
    return { sim, a };
  }

  /*
   * A farmer eats its own crop.
   *
   * It deposits into the cell it is standing in, and unless it is at capacity
   * it harvests that same cell on the next frame — so the ground under a lone
   * farmer holds about one frame's deposit however long it runs, and the stock
   * cycles rather than accumulating. Not a bug and not worth preventing: it is
   * what makes farming a thing you do *for somewhere else*, whether by
   * diffusion carrying it off or by another body arriving to eat it. The
   * assertions below are written against that, which is why they look small.
   */
  it('moves stock onto the ground one for one, and no further', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    params.energyRegrow = 0;
    params.energyDiffuse = 0;
    params.excreteRate = 0.05;
    /*
     * The gait's pathway buys substrate out of the same tank, and this asks
     * whether what leaves it arrives. `upkeepExcrete` is what routes that
     * spend onto the dish rather than destroying it, so with it on both
     * spenders are conservative and the total is the claim being made.
     */
    params.upkeepExcrete = 1;
    const { sim, a } = scarred(params);
    a.extra = 1;
    const before = a.extra + sim.energy.storedTotal();
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
    expect(sim.energy.storedTotal(), 'nothing reached the ground').toBeGreaterThan(0);
    expect(a.extra, 'the body did not pay for it').toBeLessThan(1);
    // One for one at the transfer: with growth off nothing is created, so
    // whatever cycles between tank and ground still totals what it started at.
    expect(a.extra + sim.energy.storedTotal()).toBeCloseTo(before, 4);
  });

  it('will not farm itself into debt', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    params.energyRegrow = 0;
    params.energyDiffuse = 0;
    params.excreteRate = 5;
    const { sim, a } = scarred(params);
    a.extra = 0.2;
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
    expect(a.extra, 'farming should stop at break-even').toBeGreaterThanOrEqual(0);
  });

  it('is what makes a dead cell recoverable at all', () => {
    // The whole argument for farming. Growth is proportional to what is in a
    // cell, so a cell at zero is stuck there — seeding it is the only thing
    // that restarts it, and then the ground pays back more than went in.
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    params.energyDiffuse = 0;
    params.energyRegrow = 0.3;

    const run = (farm: number): number => {
      params.excreteRate = farm;
      const { sim, a } = scarred(params);
      a.extra = 1;
      for (let f = 0; f < 60 * 20; f++) sim.step(1 / 60, params);
      return sim.energy.storedTotal();
    };
    const barren = run(0);
    const seeded = run(0.02);
    expect(barren, 'a scar with nothing in it cannot regrow').toBeCloseTo(0, 6);
    // Small because the farmer keeps eating what it plants; the point is that
    // it is not zero, and cannot be without something to restart the logistic.
    expect(seeded, 'seeding it should have restarted growth').toBeGreaterThan(0.005);
  });

  it('feeds a neighbour that is not standing on the crop', () => {
    // Which is the useful case, and the one the self-harvest loop points at:
    // a farmer is worth having next to you, not worth being.
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    params.energyRegrow = 0;
    params.energyDiffuse = 0.4;
    params.excreteRate = 0.05;
    const sim = new Sim(10000, 10000);
    const farmer = sim.spawn('era', 5000, 5000, 0, params, true)!;
    const eater = sim.spawn('con', 5000 + 60, 5000, 0, params, true)!;
    sim.step(1 / 60, params);
    const f = sim.fields;
    for (let i = CH.energy; i < f.data.length; i += 4) f.data[i] = 0;
    farmer.extra = farmer.energyCap;
    eater.extra = 0;
    for (let n = 0; n < 60 * 10; n++) sim.step(1 / 60, params);
    expect(eater.extra, 'nothing crossed from the farmer to its neighbour').toBeGreaterThan(0);
    expect(farmer.extra, 'the farmer should have paid for it').toBeLessThan(farmer.energyCap);
  });

  it('still seeds an Era\'s voice onto a slot nothing reads', () => {
    // A pin on a seed value with no consumer. The emit head's ground slot
    // stopped being farming when farming became the ground's excretion row;
    // it stays a quarter of the simplex because dropping it would renormalise
    // every genome in the library (see `effEmit`), and `seedChem` still writes
    // an Era's unit there because a seed says what a kind is for. The honesty
    // mechanism this test used to name lives on `X` now — production trades
    // against uptake — and is pinned in `chemistry.test.ts`.
    const params = defaultParams();
    const era = seedChem('era', params);
    let sum = 0;
    for (let k = 0; k < 4; k++) sum += era[EMIT + k];
    expect(era[EMIT + CH.energy], 'an Era should spend itself on the ground').toBeCloseTo(1, 6);
    expect(sum, 'and have nothing left over to shout with').toBeCloseTo(1, 6);
  });
});
