import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * Split out of sim.test.ts's 'transport recoil' block: the only test in
 * there that runs a long soak (1800 frames) rather than a handful of steps,
 * so it moves to the serial `bench` project instead of slowing down `suite`.
 */
describe('transport recoil', () => {
  it('does not let a driven chain wind itself up', () => {
    // A standing gradient — one end held full, the other held hungry — is a
    // momentum pump at any thrust above 0, and this runs at the default. What
    // bounds it is drag, not symmetry: the chain has to reach a cruising speed
    // and stay there rather than gain energy frame after frame.
    //
    // Seeded: steering kicks each body with coloured noise off Math.random, so
    // an unseeded run put the ratio either side of the bound at random.
    const realRandom = Math.random;
    let rs = 20260902 >>> 0;
    Math.random = () => {
      rs = (rs * 1664525 + 1013904223) >>> 0;
      return rs / 4294967296;
    };
    try {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.rewriteDuration = 0;
    params.snapRadius = 0;
    params.upkeep = 0;
    params.ambientEnergy = 0;
    params.transportRecoil = 12;
    const sim = new Sim(900, 600);
    sim.energy.configure(params.energyCell, 0);
    const ids: number[] = [];
    for (let i = 0; i < 8; i++) {
      const a = sim.spawn(i % 2 === 0 ? 'con' : 'dup', 300 + i * 45, 300, 0, params, true)!;
      a.extra = 0;
      ids.push(a.id);
    }
    for (let i = 0; i + 1 < 8; i++) sim.wire(ids[i], 'r', ids[i + 1], 'l', params);
    const head = sim.agents.get(ids[7])!;
    const tail = sim.agents.get(ids[0])!;
    let early = 0;
    for (let f = 0; f < 1800; f++) {
      head.extra = 1.25;
      tail.extra = -0.9;
      sim.step(1 / 60, params);
      if (f === 600) early = sim.kineticEnergy();
    }
    expect(sim.kineticEnergy(), `KE ${early.toFixed(0)} -> ${sim.kineticEnergy().toFixed(0)}`)
      .toBeLessThan(early * 1.5);
    } finally {
      Math.random = realRandom;
    }
  });
});
