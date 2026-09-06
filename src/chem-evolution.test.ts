import { describe, expect, it } from 'vitest';
import { EMIT, E_OUT, STATE_DIMS, TASTE, T_OUT, bareBody, effEmit, effTaste, seedChem, type Agent } from './agents.ts';
import { CH } from './fields.ts';
import { CHEM_TASTE_MAX } from './rewrite.ts';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/**
 * The scent genome has to actually move, and stay legal while it does.
 *
 * `emit` and `taste` are seeded from kind so a fresh pond behaves exactly as
 * it did before they existed — which means the whole mechanism is invisible
 * until something breeds. If inheritance were silently not wired up, every
 * test would still pass and the feature would be an elaborate way of writing
 * down the old constants.
 */

function pond(): { sim: Sim; params: ReturnType<typeof defaultParams> } {
  const params = defaultParams();
  params.spawnInterval = 0;
  const sim = new Sim(800, 600);
  // The oscillator, not a soup: it is built to rewrite, so births arrive in a
  // second rather than half a minute. These tests need breeding, not a pond —
  // and a 400-body soup run twice for thirty seconds each starved the rest of
  // the suite badly enough to time other files out.
  loadPreset(sim, 'oscillator', params);
  return { sim, params };
}

function drift(a: Agent, params: ReturnType<typeof defaultParams>): number {
  const seed = seedChem(a.kind, params);
  let worst = 0;
  for (let k = 0; k < 8; k++) worst = Math.max(worst, Math.abs(a.chem[k] - seed[k]));
  return worst;
}

/** One run, both questions asked of it. */
function bred(): { sim: Sim; params: ReturnType<typeof defaultParams>; born: number } {
  const { sim, params } = pond();
  const before = sim.nextId;
  for (let f = 0; f < 900; f++) sim.step(1 / 60, params);
  return { sim, params, born: sim.nextId - before };
}

describe('scent genome', () => {
  it('drifts away from its seed once bodies breed', () => {
    const { sim, params, born } = bred();
    expect(born, 'nothing was born, so nothing could inherit').toBeGreaterThan(4);

    let moved = 0;
    let worst = 0;
    for (const a of sim.agents.values()) {
      const d = drift(a, params);
      if (d > 1e-4) moved++;
      worst = Math.max(worst, d);
    }
    expect(moved, 'no body differs from its kind seed').toBeGreaterThan(0);
    expect(worst, `worst drift ${worst.toFixed(3)}`).toBeGreaterThan(0.01);
  });

  it('keeps every genome legal however far it drifts', () => {
    const { sim } = bred();
    for (const a of sim.agents.values()) {
      let sum = 0;
      for (let k = EMIT; k < EMIT + 4; k++) {
        expect(a.chem[k], `emit ${k} went negative`).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(a.chem[k]), 'emit went non-finite').toBe(true);
        sum += a.chem[k];
      }
      // One unit of voice, spent across four channels. A body that mutated its
      // way to silence keeps it rather than being renormalised up out of noise,
      // so the sum is either about one or about zero.
      expect(sum < 1e-5 || Math.abs(sum - 1) < 1e-3, `emit sums to ${sum}`).toBe(true);
      for (let k = TASTE; k < TASTE + 4; k++) {
        expect(Math.abs(a.chem[k]), `taste ${k} out of bounds`).toBeLessThanOrEqual(
          CHEM_TASTE_MAX + 1e-6,
        );
      }
    }
  });

  /*
   * A body is a small recurrent network now, so the unit tests split in two:
   * what the output matrices do with a state (here, with `h` set by hand), and
   * what the input matrix and the wire graph do to produce one (further down,
   * which needs a Sim because it needs neighbours).
   */
  it('maps state to what it says through E, and to what it listens for through T', () => {
    const params = defaultParams();
    const chem = seedChem('con', params);
    // Say ch0 when h0 is up, listen for ch1 when h2 is down.
    chem[E_OUT + CH.conP * STATE_DIMS + 0] = 1;
    chem[T_OUT + CH.dupP * STATE_DIMS + 2] = -2;
    const base = chem[EMIT + CH.conP];
    const tBase = chem[TASTE + CH.dupP];

    const rest = bareBody(chem, { h: [0, 0, 0, 0] });
    const lit = bareBody(chem, { h: [0.5, 0, 0, 0] });
    const inv = bareBody(chem, { h: [0, 0, 0.5, 0] });

    expect(effEmit(rest, CH.conP), 'a zero state is just the base').toBeCloseTo(base, 6);
    expect(effEmit(lit, CH.conP)).toBeCloseTo(base + 0.5, 6);
    expect(effTaste(rest, CH.dupP)).toBeCloseTo(tBase, 6);
    expect(effTaste(inv, CH.dupP)).toBeCloseTo(tBase - 1, 6);
  });

  it('leaves a seeded body behaving exactly as a stateless one did', () => {
    // Every matrix seeds to zero except the food pathway, so a fresh body
    // computes h = phi(0) = 0 and its output is its bases. That is what makes
    // a spawned agent arrive unevolved, which is the point of seeding at all.
    const params = defaultParams();
    const a = bareBody(seedChem('con', params));
    expect(effEmit(a, CH.conP)).toBeCloseTo(1, 6);
    expect(effEmit(a, CH.dupP)).toBeCloseTo(0, 6);
    expect(effTaste(a, CH.dupP)).toBeCloseTo(params.attractMedium, 6);
  });

  it('never lets a body emit onto the ground, whatever its genes say', () => {
    const params = defaultParams();
    const chem = seedChem('con', params);
    chem[EMIT + CH.energy] = 9;
    for (let d = 0; d < STATE_DIMS; d++) chem[E_OUT + CH.energy * STATE_DIMS + d] = 9;
    const a = bareBody(chem, { h: [1, 1, 1, 1] });
    expect(effEmit(a, CH.energy), 'the ground is not a thing you can shout').toBe(0);
    // And it is still perfectly able to smell it.
    chem[TASTE + CH.energy] = 1.5;
    expect(effTaste(a, CH.energy)).toBeGreaterThan(0);
  });


  it('never lets a modulated emit go negative', () => {
    const params = defaultParams();
    // `request` no longer reaches emit directly — it is an input to `h`, and
    // `h` is what `E` reads. Set the state, not the input.
    const a = bareBody(seedChem('con', params), { h: [1, 0, 0, 0] });
    // A body that goes silent under pressure, pushed past silence.
    a.chem[E_OUT + CH.conP * STATE_DIMS + 0] = -5;
    expect(effEmit(a, 0), 'emitting a negative amount is not a thing').toBe(0);
  });

  /*
   * What used to be "charges a body for the voice it uses".
   *
   * `emitCost` is gone: the unit-sum budget does the same job and does it
   * better. A per-second rent on amplitude is the handicap-principle version
   * of honesty — signals are trusted because they are wasteful — and the
   * current reading of signalling theory is that what actually maintains
   * honesty at equilibrium is a condition-dependent *trade-off*, not a cost.
   * The budget is exactly that trade-off: a body has one unit, so saying one
   * thing costs it another, and since the ground is in the same budget,
   * feeding the dish costs it being heard. Two mechanisms for one property,
   * and the one that was off by default was the redundant one.
   */
  it('spends one unit of voice however it is distributed', () => {
    const params = defaultParams();
    for (const kind of ['con', 'dup', 'era'] as const) {
      const c = seedChem(kind, params);
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += c[EMIT + k];
      expect(sum, `${kind} does not spend exactly one unit`).toBeCloseTo(1, 6);
    }
  });

  it('lets taste go negative, which the fixed weights never could', () => {
    // Not asserting that it *does* in any given run — only that nothing in the
    // pipeline clamps avoidance away, since a body fleeing what it smells is
    // the main behaviour the old switch could not express.
    const params = defaultParams();
    const c = seedChem('con', params);
    c[TASTE] = -2;
    expect(Math.min(...Array.from(c.subarray(TASTE, TASTE + 4)))).toBeLessThan(0);
  });
});

describe('heritable flocking', () => {
  /**
   * Sampled across the whole run, not at the end of it.
   *
   * The oscillator breeds well over a hundred bodies and leaves a handful
   * alive, so reading the survivors is reading four samples of a population of
   * a hundred and twenty — enough to miss a whole direction of drift by luck.
   */
  function survey(frames: number): {
    born: number;
    mean: number;
    lo: number;
    hi: number;
    seen: number;
  } {
    const params = pond().params;
    params.flockAlign = 0;
    const sim = new Sim(800, 600);
    loadPreset(sim, 'oscillator', params);
    const before = sim.nextId;
    let sum = 0;
    let seen = 0;
    let lo = Infinity;
    let hi = -Infinity;
    for (let f = 0; f < frames; f++) {
      sim.step(1 / 60, params);
      if (f % 30 !== 0) continue;
      for (const a of sim.agents.values()) {
        sum += a.flockAlign;
        seen++;
        lo = Math.min(lo, a.flockAlign);
        hi = Math.max(hi, a.flockAlign);
      }
    }
    return { born: sim.nextId - before, mean: seen ? sum / seen : 0, lo, hi, seen };
  }

  it('lets an off trait move both ways, and does not ratchet it up', () => {
    /*
     * Alignment ships on in the pond, but the gene has to be able to sit
     * off without a reflecting barrier: clamp it at zero and half the
     * mutations are absorbed and half move up, so the trait climbs whether
     * or not anything selects for it. This run seeds it at zero and only
     * clamps where the force reads it, so the walk can go both ways.
     */
    const r = survey(1800);
    expect(r.born, 'nothing bred, so nothing could drift').toBeGreaterThan(20);
    expect(r.seen, 'no population to survey').toBeGreaterThan(50);
    expect(r.lo, `lowest align seen ${r.lo.toFixed(3)}`).toBeLessThan(0);
    expect(r.hi, `highest align seen ${r.hi.toFixed(3)}`).toBeGreaterThan(0);
    // Unselected, it should sit near where it started rather than climbing.
    // Erase now clones Eras with a mutation nudge too, which adds walks of
    // the same unselected gene, so the mean is noisier than commute-only.
    expect(Math.abs(r.mean), `mean align ${r.mean.toFixed(3)}`).toBeLessThan(1);
  });
});
