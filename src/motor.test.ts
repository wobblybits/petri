import { describe, expect, it } from 'vitest';
import { CH } from './fields.ts';
import { EMIT, TASTE, seedChem } from './agents.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * A net's wires as a fuel line.
 *
 * Swimming used to be free, which meant thrust was a property of a body and
 * not of the net feeding it: a starving swimmer swam exactly as hard as a full
 * one. The transport machinery had nothing to be *for* beyond keeping redexes
 * alive, and a net could not have a motor because there was nothing a motor
 * would cost.
 *
 * `swimCost` is the price, and `forageAsk` is what makes the bill directional
 * — a body asks in proportion to how much it likes what it can smell, so the
 * energy goes to whichever of a net's swimmers is standing somewhere worth
 * standing.
 */

function lone(params: ReturnType<typeof defaultParams>, speed: number): Sim {
  const sim = new Sim(4000, 4000);
  const a = sim.spawn('con', 2000, 2000, 0, params, true)!;
  a.extra = 1;
  a.vx = speed;
  return sim;
}

describe('swimming costs energy', () => {
  it('bills a moving body and leaves a still one alone', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.ambientEnergy = 0;
    params.rewriteDuration = 0;
    params.stepSpeed = 0;
    params.swimNoise = 0;
    params.drag = 0;
    params.swimCost = 0.001;

    const moving = lone(params, 60);
    const still = lone(params, 0);
    for (let f = 0; f < 60; f++) {
      moving.step(1 / 60, params);
      still.step(1 / 60, params);
    }
    const m = [...moving.agents.values()][0];
    const s = [...still.agents.values()][0];
    expect(m.extra, 'a moving body paid nothing').toBeLessThan(1 - 0.01);
    expect(s.extra, 'a still body was billed for standing').toBeCloseTo(1, 3);
  });

  it('charges nothing at all when the price is off', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.ambientEnergy = 0;
    params.rewriteDuration = 0;
    params.stepSpeed = 0;
    params.swimNoise = 0;
    params.drag = 0;
    params.swimCost = 0;
    const sim = lone(params, 60);
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
    expect([...sim.agents.values()][0].extra).toBeCloseTo(1, 6);
  });

  it('does not bill a wired-in body for the net dragging it about', () => {
    // Cargo, not an engine: a body whose principal is attached is moved by the
    // constraint rather than by its own port, and charging it for the net's
    // motion would make being carried expensive.
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.ambientEnergy = 0;
    params.rewriteDuration = 0;
    params.stepSpeed = 0;
    params.swimNoise = 0;
    params.drag = 0;
    params.swimCost = 0.001;
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('era', 2000, 2000, 0, params, true)!;
    const b = sim.spawn('era', 2060, 2000, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    a.extra = 1;
    b.extra = 1;
    a.vx = 60;
    b.vx = 60;
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
    for (const e of sim.agents.values()) {
      expect(e.extra, 'cargo was billed for the ride').toBeCloseTo(1, 3);
    }
  });
});

describe('appetite', () => {
  it('asks for energy in proportion to what it likes about where it is', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    params.forageAsk = 0.5;

    // Two bodies alike but for their taste: one is drawn to the ground it is
    // standing on, one is indifferent to everything.
    const sim = new Sim(10000, 10000);
    const keen = sim.spawn('con', 5000, 5000, 0, params, true)!;
    const dull = sim.spawn('con', 5400, 5000, 0, params, true)!;
    for (let k = 0; k < 4; k++) {
      keen.chem[TASTE + k] = 0;
      dull.chem[TASTE + k] = 0;
    }
    keen.chem[TASTE + CH.energy] = 4;
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params);

    expect(keen.request, 'a body that likes where it is should be asking').toBeGreaterThan(0);
    expect(dull.request, 'a body that wants nothing should be quiet').toBe(0);
  });

  it('stays quiet about somewhere it actively dislikes', () => {
    // Taste is signed, so a trail can be negative. That is a reason to leave,
    // not a reason to be fed.
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    params.forageAsk = 0.5;
    const sim = new Sim(10000, 10000);
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    for (let k = 0; k < 4; k++) a.chem[TASTE + k] = 0;
    a.chem[TASTE + CH.energy] = -4;
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params);
    expect(a.trail, 'the setup should have given it something to dislike').toBeLessThan(0);
    expect(a.request).toBe(0);
  });

  it('is off when the dial is', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    params.forageAsk = 0;
    const sim = new Sim(10000, 10000);
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    for (let k = 0; k < 4; k++) a.chem[TASTE + k] = 0;
    a.chem[TASTE + CH.energy] = 4;
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params);
    expect(a.request).toBe(0);
  });
});

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
    params.farmRate = 0.05;
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
    params.farmRate = 5;
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
      params.farmRate = farm;
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
    params.farmRate = 0.05;
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

  it('costs a body its voice, which is what keeps it honest', () => {
    // The ground is in the same unit-sum budget as the three things a body can
    // say, so feeding the dish and being heard are the same budget. That
    // trade-off is the honesty mechanism — no separate cost term.
    const params = defaultParams();
    const era = seedChem('era', params);
    let sum = 0;
    for (let k = 0; k < 4; k++) sum += era[EMIT + k];
    expect(era[EMIT + CH.energy], 'an Era should spend itself on the ground').toBeCloseTo(1, 6);
    expect(sum, 'and have nothing left over to shout with').toBeCloseTo(1, 6);
  });
});
