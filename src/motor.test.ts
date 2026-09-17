import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { fixedParams } from './test-params.ts';
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
 * `swimCost` is the price. It had a companion — `forageAsk`, which made the
 * bill directional by letting a body ask in proportion to how much it liked
 * what it could smell — and that is gone: it shipped at 0, so no pond ever
 * asked. What is left is the price itself, and the three claims on the need
 * field are again the two shortfalls, `rescueNeed` and `redexNeed`.
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
    const params = fixedParams();
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
    const params = fixedParams();
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
    const params = fixedParams();
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
describe('port occupancy', () => {
  it('tracks how much of a body is attached, and updates when that changes', () => {
    const params = fixedParams();
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

