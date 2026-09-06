import { describe, expect, it } from 'vitest';
import { CH } from './fields.ts';
import { TASTE } from './agents.ts';
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
