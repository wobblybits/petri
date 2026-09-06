import { describe, expect, it } from 'vitest';
import { BOUND, EMIT, EMIT_SLOPE, FULL, HERE, NEED, STATE_DIMS, TASTE, TASTE_SLOPE, bareBody, effEmit, effTaste, seedChem, type Agent } from './agents.ts';
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

  it('modulates what a body says by how its neighbourhood is doing', () => {
    const params = defaultParams();
    const quiet = bareBody(seedChem('con', params), { request: 0, extra: 0, trail: 0 });
    const needy = bareBody(seedChem('con', params), { request: 1, extra: 0, trail: 0 });
    // Seeded, the slope is zero and state changes nothing at all.
    expect(effEmit(needy, 0)).toBe(effEmit(quiet, 0));
    expect(effTaste(needy, 1)).toBe(effTaste(quiet, 1));

    // Give it something to say under pressure: quiet when fed, loud on the
    // aux channel when its net is hungry — and listening harder for the
    // ground while it is at it, which is the one thing a hungry body most
    // wants to find.
    needy.chem[EMIT_SLOPE + CH.aux * STATE_DIMS + NEED] = 0.8;
    needy.chem[TASTE_SLOPE + CH.energy * STATE_DIMS + NEED] = 2;
    expect(effEmit(needy, CH.aux), 'a needy body should be saying more').toBeGreaterThan(
      effEmit(quiet, CH.aux),
    );
    expect(effTaste(needy, CH.energy), 'and listening harder').toBeGreaterThan(
      effTaste(quiet, CH.energy),
    );

    // Half as needy, half the shift: the response is linear in state.
    const half = bareBody(Float32Array.from(needy.chem), { request: 0.5, extra: 0, trail: 0 });
    expect(effEmit(half, CH.aux)).toBeCloseTo(
      (effEmit(quiet, CH.aux) + effEmit(needy, CH.aux)) / 2,
      6,
    );
  });

  /*
   * The ground is on channel 2, so nothing emits into it — ever, at any
   * genome, however far a lineage drifts. A body that could would be minting
   * food from nothing at five units a free port a frame, which is the whole
   * economy gone, so this is worth a test of its own rather than trusting
   * three deposit paths to keep agreeing about it.
   */
  it('never lets a body emit onto the ground, whatever its genes say', () => {
    const params = defaultParams();
    const a = bareBody(seedChem('con', params), { request: 1, extra: 0, trail: 0 });
    a.chem[EMIT + CH.energy] = 9;
    a.chem[EMIT_SLOPE + CH.energy * STATE_DIMS + NEED] = 9;
    expect(effEmit(a, CH.energy), 'the ground is not a thing you can shout').toBe(0);
    // And it is still perfectly able to smell it.
    a.chem[TASTE + CH.energy] = 1.5;
    expect(effTaste(a, CH.energy)).toBeGreaterThan(0);
  });

  /*
   * The three state dimensions have to be about different things, or the
   * widening bought nothing.
   *
   * `NEED` is the neighbourhood's, spread over the wires. `FULL` is this
   * body's own tank. A rule with two clauses — "shout when my net is hungry
   * *but* I am full", the shape a body that has something to give would want —
   * needs both, and needs them separable. With one input it was not merely
   * hard to express, it was outside the language.
   */
  it('separates what my net needs from what I have', () => {
    const params = defaultParams();
    const mk = (request: number, extra: number) =>
      bareBody(seedChem('con', params), { request, extra });

    const donor = mk(1, 1); // net starving, I am full
    const beggar = mk(1, 0); // net starving, I am empty too
    const idle = mk(0, 1); // net fine, I am full

    for (const a of [donor, beggar, idle]) {
      a.chem[EMIT + CH.conP] = 0;
      a.chem[EMIT_SLOPE + CH.conP * STATE_DIMS + NEED] = 0.5;
      a.chem[EMIT_SLOPE + CH.conP * STATE_DIMS + FULL] = 0.5;
    }
    // Both clauses true, so this one is loudest — and the two bodies that
    // satisfy exactly one of them are equally quiet, which is what proves the
    // dimensions are not two names for the same reading.
    expect(effEmit(donor, CH.conP)).toBeCloseTo(1, 6);
    expect(effEmit(beggar, CH.conP)).toBeCloseTo(0.5, 6);
    expect(effEmit(idle, CH.conP)).toBeCloseTo(0.5, 6);
  });

  it('lets a body condition on where it is, with a sign', () => {
    const params = defaultParams();
    const mk = (trail: number) =>
      bareBody(seedChem('con', params), { trail });
    const a = mk(4);
    const b = mk(-4);
    for (const x of [a, b]) {
      x.chem[TASTE + CH.aux] = 0;
      x.chem[TASTE_SLOPE + CH.aux * STATE_DIMS + HERE] = 1;
    }
    // Squashed, so a strong like and a strong dislike land either side of zero
    // and neither can run away with the weight.
    expect(effTaste(a, CH.aux)).toBeGreaterThan(0.5);
    expect(effTaste(b, CH.aux)).toBeLessThan(-0.5);
    expect(Math.abs(effTaste(a, CH.aux))).toBeLessThan(1);
  });

  /*
   * `BOUND` is what lets one channel mean two things in a lifetime.
   *
   * A port's seeded scent is doing latching work only while that port is open.
   * Once it is matched the meaning has done its job and the channel is free to
   * carry whatever the lineage has drifted onto; when a neighbour dies and the
   * socket reopens, latching is wanted again — and at exactly the moment it
   * becomes useful, because an open port is how two nets fuse. A body that
   * cannot read its own occupancy has to mean one thing forever, and every
   * signal it might evolve competes with mate-finding.
   */
  it('lets a body say one thing with sockets open and another once wired in', () => {
    const params = defaultParams();
    const loose = bareBody(seedChem('con', params), { bound: 0 });
    const wiredIn = bareBody(seedChem('con', params), { bound: 1 });
    for (const a of [loose, wiredIn]) {
      a.chem[EMIT + CH.conP] = 1;
      a.chem[EMIT + CH.aux] = 0;
      // Trade the latching channel away for the aux one as the ports fill.
      a.chem[EMIT_SLOPE + CH.conP * STATE_DIMS + BOUND] = -1;
      a.chem[EMIT_SLOPE + CH.aux * STATE_DIMS + BOUND] = 1;
    }
    expect(effEmit(loose, CH.conP), 'an open body should still be latching').toBeCloseTo(1, 6);
    expect(effEmit(loose, CH.aux)).toBeCloseTo(0, 6);
    expect(effEmit(wiredIn, CH.conP), 'a matched body is done latching').toBeCloseTo(0, 6);
    expect(effEmit(wiredIn, CH.aux), 'and free to say something else').toBeCloseTo(1, 6);
  });

  it('never lets a modulated emit go negative', () => {
    const params = defaultParams();
    const a = bareBody(seedChem('con', params), { request: 1, extra: 0, trail: 0 });
    // A body that goes silent under pressure, pushed past silence.
    a.chem[EMIT_SLOPE + CH.conP * STATE_DIMS + NEED] = -5;
    expect(effEmit(a, 0), 'emitting a negative amount is not a thing').toBe(0);
  });

  it('charges a body for the voice it uses', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.ambientEnergy = 0;
    params.rewriteDuration = 0;
    const run = (cost: number): number => {
      params.emitCost = cost;
      const sim = new Sim(800, 600);
      const a = sim.spawn('con', 400, 300, 0, params, true)!;
      a.extra = 1;
      for (let f = 0; f < 120; f++) sim.step(1 / 60, params);
      return sim.agents.get(a.id)!.extra;
    };
    const free = run(0);
    const paid = run(0.02);
    expect(free, 'nothing else should be draining it').toBeCloseTo(1, 3);
    expect(paid, `paid ${paid.toFixed(4)} vs free ${free.toFixed(4)}`).toBeLessThan(free - 0.01);
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
