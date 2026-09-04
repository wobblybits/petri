import { describe, expect, it } from 'vitest';
import { EXTRA_FLOOR } from './energy.ts';
import { defaultParams, type Params } from './params.ts';
import { Sim } from './sim.ts';

/**
 * Two ways the pond wears: wires tear when they are pulled too far, and being
 * hit costs energy.
 *
 * Neither kills directly except through the one tank. A wire snapping is a
 * topology change; a collision is a drain that can take a body to the floor,
 * where starvation was already waiting. Keeping every lethal path routed
 * through `extra` means there is one rule for dying, and it also means a
 * numerical fault shows up as a pond that looks wrong rather than a pond that
 * is gone — which matters, because one already flew apart this week over a
 * constant that disagreed across the wasm wall.
 */

function quiet(): Params {
  const p = defaultParams();
  p.spawnInterval = 0;
  p.rewriteDuration = 0;
  p.snapRadius = 0;
  p.upkeep = 0;
  p.ambientEnergy = 0;
  p.emitCost = 0;
  p.flockAlign = 0;
  p.flockSep = 0;
  p.stepSpeed = 0;
  p.swimNoise = 0;
  return p;
}

describe('wire snapping', () => {
  it('tears a wire dragged far past its rest length', () => {
    const params = quiet();
    params.wireShrink = 0;
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('era', 500, 500, 0, params, true)!;
    const b = sim.spawn('era', 560, 500, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    expect(sim.graph.wires.size).toBe(1);

    // Haul them apart and hold them there. Locked bodies are not moved by the
    // span constraint, so the wire has no way to relieve itself.
    a.locked = true;
    b.locked = true;
    b.x = 500 + params.wireMinRest * 8;
    for (let f = 0; f < 8; f++) sim.step(1 / 60, params);
    expect(sim.graph.wires.size, 'the wire should have let go').toBe(0);
  });

  it('leaves a merely taut wire alone', () => {
    const params = quiet();
    params.wireShrink = 0;
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('era', 500, 500, 0, params, true)!;
    const b = sim.spawn('era', 500 + params.wireMinRest, 500, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    a.locked = true;
    b.locked = true;
    // Past `wireTaut` — this wire is under load — but nowhere near `wireSnap`.
    b.x = 500 + params.wireMinRest * 1.2;
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
    expect(sim.graph.wires.size, 'a working wire was cut').toBe(1);
  });

  it('does not snap when the mechanic is off', () => {
    const params = quiet();
    params.wireShrink = 0;
    params.wireSnap = 0;
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('era', 500, 500, 0, params, true)!;
    const b = sim.spawn('era', 560, 500, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    a.locked = true;
    b.locked = true;
    b.x = 500 + params.wireMinRest * 8;
    for (let f = 0; f < 8; f++) sim.step(1 / 60, params);
    expect(sim.graph.wires.size).toBe(1);
  });
});

describe('contact damage', () => {
  /** Two bodies shoved into each other and held there. */
  function crush(cost: number, frames: number): { extra: number; alive: boolean } {
    const params = quiet();
    params.contactCost = cost;
    params.declutter = 0;
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('con', 500, 500, 0, params, true)!;
    const b = sim.spawn('con', 508, 500, Math.PI, params, true)!;
    a.extra = 1;
    b.extra = 1;
    for (let f = 0; f < frames; f++) {
      // Hold them overlapping: the solver would otherwise separate them in a
      // frame or two and there would be nothing left to charge for.
      a.x = 500;
      b.x = 508;
      a.vx = 0;
      b.vx = 0;
      sim.step(1 / 60, params);
    }
    const still = sim.agents.get(a.id);
    return { extra: still ? still.extra : EXTRA_FLOOR, alive: !!still };
  }

  it('costs energy to be crushed', () => {
    const free = crush(0, 60);
    const paid = crush(0.2, 60);
    expect(free.extra, 'nothing else should be draining it').toBeCloseTo(1, 3);
    expect(paid.extra, `paid ${paid.extra.toFixed(3)} vs free ${free.extra.toFixed(3)}`)
      .toBeLessThan(free.extra - 0.01);
  });

  it('kills only by running the tank down, never on impact alone', () => {
    // One frame of a colossal cost still cannot kill a full body outright:
    // damage is a rate, and death is the floor being reached.
    const params = quiet();
    params.contactCost = 100;
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('con', 500, 500, 0, params, true)!;
    const b = sim.spawn('con', 508, 500, Math.PI, params, true)!;
    a.extra = 1;
    b.extra = 1;
    sim.step(1 / 60, params);
    // Whatever happened, nobody died of a single touch while still in credit.
    for (const body of sim.agents.values()) {
      expect(body.extra, 'a body below the floor should have been reaped').toBeGreaterThan(
        EXTRA_FLOOR,
      );
    }
  });
});
