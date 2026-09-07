import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * `Sim.rewriteMix` is the instrument for the question this simulation is
 * about: is a latch something a body did, or something that happened to it?
 *
 * The three rules do very different things to a population — a commute makes
 * four bodies from two, an annihilation takes two away — so the mix decides
 * whether a pond grows. And the mix is set by which kinds meet, which is
 * either steering or crowding. `chance` is the mix crowding alone would give,
 * so `selectivity` going positive is the pond starting to choose.
 */
describe('rewriteMix', () => {
  function pond(era: number, dup: number, con: number): Sim {
    const params = defaultParams();
    params.spawnInterval = 0;
    const sim = new Sim(4000, 4000);
    let x = 100;
    const put = (kind: 'era' | 'dup' | 'con', n: number) => {
      for (let i = 0; i < n; i++) {
        // Spread far apart: this test is about the census, not about physics.
        sim.spawn(kind, 100 + (x % 3000), 100 + Math.floor(x / 3000) * 120, 0, params, true);
        x += 140;
      }
    };
    put('era', era);
    put('dup', dup);
    put('con', con);
    return sim;
  }

  it('reports no share until something has rewritten', () => {
    const m = pond(3, 3, 3).rewriteMix();
    expect(m.rewrites).toBe(0);
    expect(m.share).toBeNull();
    expect(m.selectivity).toBeNull();
  });

  it('prices chance as the odds two bodies drawn at random would commute', () => {
    // Only a Dup meeting a Con commutes, so for equal thirds that is the two
    // ordered pairs dup-con and con-dup: 2 * (1/3) * (1/3).
    const even = pond(30, 30, 30).rewriteMix();
    expect(even.chance).toBeCloseTo(2 / 9, 10);

    // No Cons, so nothing can commute however the pairs fall.
    expect(pond(30, 30, 0).rewriteMix().chance).toBe(0);
    // No Eras: half Dup, half Con, so half of all random pairs commute.
    expect(pond(0, 40, 40).rewriteMix().chance).toBeCloseTo(0.5, 10);
    // A dish that is almost all Era can barely commute at all.
    expect(pond(98, 1, 1).rewriteMix().chance).toBeCloseTo(2 * 0.01 * 0.01, 10);
  });

  it('measures selectivity as the gap between what happened and chance', () => {
    const sim = pond(30, 30, 30);
    const t = sim.tally;
    // Exactly the chance mix: nine rewrites, two of them commutes.
    t.commutes = 2;
    t.erases = 5;
    t.annihilations = 2;
    const at = sim.rewriteMix();
    expect(at.rewrites).toBe(9);
    expect(at.share).toBeCloseTo(2 / 9, 10);
    expect(at.selectivity).toBeCloseTo(0, 10);

    // Twice as many commutes as chance would give: the pond is choosing.
    t.commutes = 4;
    t.erases = 3;
    t.annihilations = 2;
    const above = sim.rewriteMix();
    expect(above.share).toBeCloseTo(4 / 9, 10);
    expect(above.selectivity!).toBeGreaterThan(0);

    // Fewer: crowding, or a net working through the pairs it already made.
    t.commutes = 0;
    t.erases = 7;
    t.annihilations = 2;
    expect(sim.rewriteMix().selectivity!).toBeLessThan(0);
  });

  it('tracks the census, not the seed mix', () => {
    const sim = pond(0, 40, 40);
    expect(sim.rewriteMix().chance).toBeCloseTo(0.5, 10);
    // Kill every Con and the odds of a commute go to nothing.
    for (const a of [...sim.agents.values()]) if (a.kind === 'con') sim.kill(a.id);
    expect(sim.rewriteMix().chance).toBe(0);
  });
});
