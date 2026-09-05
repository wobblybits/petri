import { describe, expect, it } from 'vitest';
import { CH, CHANNELS } from './fields.ts';
import { TASTE, seedChem, type Agent } from './agents.ts';
import { defaultParams } from './params.ts';
import { Sim, mixScent } from './sim.ts';
import { nativeSolver } from './native/solver.ts';

/**
 * Smelling the ground.
 *
 * Energy shares the field with the three signal channels but not their units.
 * A signal channel is accumulated deposits — five units a port a frame against
 * a decay of a percent — and runs to peaks around ten. A cell of ground at
 * full capacity holds `ambientEnergy / 16`, about a sixteenth of one unit. Two
 * orders of magnitude apart, in one dot product, against a taste weight capped
 * at 4.
 *
 * So without a scale the ground is not merely quiet, it is *unreachable*: no
 * genome inside the legal range could weight it enough to change a decision,
 * and a lineage that wanted to care about food could not evolve into it. The
 * scale is `1 / cellCap`, which asks the readable question — how full is the
 * ground here, 0 to 1 — and keeps asking it when the sliders move.
 */

const AT = { x: 5000, y: 5000 };

/** A body that cares about exactly one channel, and nothing else. */
function taster(sim: Sim, params: ReturnType<typeof defaultParams>, ch: number, w: number): Agent {
  const a = sim.spawn('con', AT.x, AT.y, Math.PI / 2, params, true)!;
  for (let k = 0; k < 4; k++) a.chem[TASTE + k] = 0;
  a.chem[TASTE + ch] = w;
  return a;
}

/** Empty the ground everywhere left of `x`, leaving the rest at capacity. */
function scarLeftOf(sim: Sim, x: number): void {
  const f = sim.fields;
  const cs = f.cellSize;
  for (let j = 0; j < f.rows; j++) {
    for (let i = 0; i < f.cols; i++) {
      if (f.originX + (i + 0.5) * cs < x) f.data[(j * f.cols + i) * CHANNELS + CH.energy] = 0;
    }
  }
}

describe('the ground as something to smell', () => {
  it('reads a full cell as 1, whatever the capacity is set to', () => {
    const params = defaultParams();
    const a = { chem: seedChem('con', params), request: 0 } as unknown as Agent;
    a.chem[TASTE + CH.energy] = 1;
    for (let k = 0; k < 4; k++) if (k !== CH.energy) a.chem[TASTE + k] = 0;

    // Two worlds whose ground is stocked very differently read the same, which
    // is the point: the gene means one thing across every setting.
    const rich = mixScent(a, 0, 0, 0.5, 0, 1 / 0.5);
    const lean = mixScent(a, 0, 0, 0.0625, 0, 1 / 0.0625);
    expect(rich).toBeCloseTo(1, 9);
    expect(lean).toBeCloseTo(1, 9);
    // And half-grazed ground reads as half.
    expect(mixScent(a, 0, 0, 0.03125, 0, 1 / 0.0625)).toBeCloseTo(0.5, 9);
  });

  it('is silent about a world with no ground in it', () => {
    const params = defaultParams();
    params.ambientEnergy = 0;
    params.spawnInterval = 0;
    const sim = new Sim(10000, 10000);
    const a = taster(sim, params, CH.energy, 4);
    sim.step(1 / 60, params);
    // `cellCap` is 0, so the scale is 0 rather than an infinity — a body
    // cannot smell what is not there, and asking must not divide by it.
    expect(Number.isFinite(a.x)).toBe(true);
    expect(Number.isFinite(a.heading)).toBe(true);
  });

  it('swims out of a scar toward ground that still has something in it', async () => {
    await nativeSolver.init();
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    // No noise and no wandering, so what is left is the gradient.
    params.swimNoise = 0;
    params.energyRegrow = 0;
    params.energyDiffuse = 0;

    const run = (w: number): number => {
      const sim = new Sim(10000, 10000);
      const a = taster(sim, params, CH.energy, w);
      // One frame to pin the world and lay the ground down, then cut the scar
      // so its edge runs through the body. The sensors sit `sensorDist` out at
      // `sensorAngle` either side of a heading of +y, which is only about 11
      // units apart in x — put the edge further off than that and both sensors
      // read the same empty ground, there is no gradient to turn on, and the
      // body swims straight up forever. Which it did, the first time.
      sim.step(1 / 60, params);
      scarLeftOf(sim, AT.x);
      const x0 = sim.agents.get(a.id)!.x;
      for (let f = 0; f < 240; f++) sim.step(1 / 60, params);
      return sim.agents.get(a.id)!.x - x0;
    };

    // Facing across the gradient at the scar's edge: a body that likes food
    // should end up to the right of where it started, and one that is
    // indifferent should not care which way it went.
    const drawn = run(4);
    const blind = run(0);
    expect(drawn, `drawn moved ${drawn.toFixed(1)}, blind ${blind.toFixed(1)}`).toBeGreaterThan(
      blind,
    );
    expect(drawn, 'it should have made real progress toward the food').toBeGreaterThan(20);
  });

  it('lets avoidance work too, since taste is signed', () => {
    const params = defaultParams();
    const a = { chem: seedChem('con', params), request: 0 } as unknown as Agent;
    for (let k = 0; k < 4; k++) a.chem[TASTE + k] = 0;
    a.chem[TASTE + CH.energy] = -2;
    // Nothing in the pipeline clamps a negative weight on the ground away. A
    // body that flees full ground is a strange thing to be, but it is the same
    // machinery that lets one flee a crowd, and the genome is allowed to try.
    expect(mixScent(a, 0, 0, 0.0625, 0, 1 / 0.0625)).toBeCloseTo(-2, 9);
  });
});
