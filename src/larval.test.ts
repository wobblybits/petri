import { describe, expect, it } from 'vitest';
import { LarvalWindow } from './larval.ts';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/*
 * The larval window decides whether obligate trophic dependency is a
 * structuring pressure or a way to kill the soup, so a wrong number here
 * would be read as an answer to a question the plan is holding open. Both
 * halves are pinned: the histogram, whose quantiles have arithmetic answers,
 * and the wiring, where the risk is that nothing ever calls it.
 */

describe('the larval histogram', () => {
  it('has nothing to say before anything has happened', () => {
    const w = new LarvalWindow();
    expect(w.read()).toEqual({ latched: 0, p50: null, p90: null, died: 0, loneliness: null });
  });

  it('puts a quantile in the right bin across four decades', () => {
    const w = new LarvalWindow();
    for (const s of [0.05, 0.5, 5, 50, 500]) w.latchedAfter(s);
    const r = w.read();
    // Five samples: the median is the third, 5 s. Log bins are 9% wide here.
    expect(r.p50).toBeGreaterThan(4.5);
    expect(r.p50).toBeLessThan(5.5);
    expect(r.p90).toBeGreaterThan(450);
    expect(r.latched).toBe(5);
  });

  it('clamps rather than losing what falls off either end', () => {
    const w = new LarvalWindow();
    w.latchedAfter(0);          // latched on the frame it arrived
    w.latchedAfter(1e9);        // longer than the top of the range
    expect(w.read().latched).toBe(2);
    // Neither is lost: one lands in the bottom bin, one in the top, whose
    // centre reads about 870 s -- ten times the tank life, which is as much
    // as the question needs.
    expect(w.read().p50).toBeLessThan(0.02);
    expect(w.read().p90).toBeGreaterThan(500);
  });

  it('measures loneliness over arrivals, and time over the latched only', () => {
    const w = new LarvalWindow();
    w.latchedAfter(2);
    for (let i = 0; i < 3; i++) w.diedAlone();
    const r = w.read();
    expect(r.loneliness).toBeCloseTo(0.75, 12);
    // A body that never latched has no time-to-latch. Giving it one — the
    // run's length, say — would make the median a function of how long the
    // run happened to be.
    expect(r.p50).toBeGreaterThan(1.8);
    expect(r.p50).toBeLessThan(2.2);
  });
});

describe('the window, wired into a pond', () => {
  it('fills in as a soup latches, and every time is a real one', () => {
    const params = defaultParams();
    params.soupCount = 200;
    params.spawnInterval = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    // Nothing has latched at t = 0, and the founders all arrived at 0.
    expect(sim.larval.read().latched).toBe(0);

    for (let i = 0; i < 900; i++) sim.step(1 / 60, params);
    const r = sim.larval.read();
    expect(r.latched).toBeGreaterThan(0);
    // Fifteen simulated seconds have passed, so no body can have taken longer,
    // and none can have taken less than nothing.
    expect(r.p50).toBeGreaterThan(0);
    expect(r.p50).toBeLessThan(15);
    expect(r.p90).toBeLessThan(15);
  });

  it('counts a body once, however often it re-latches', () => {
    const params = defaultParams();
    params.soupCount = 60;
    params.spawnInterval = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    for (let i = 0; i < 600; i++) sim.step(1 / 60, params);
    // Never more latched bodies than there have ever been bodies. A body that
    // detaches and re-latches would break this, and the first latch is the
    // only one the tank question is about.
    const r = sim.larval.read();
    expect(r.latched + r.died).toBeLessThanOrEqual(60 + sim.tally.born + sim.tally.spawned);
  });

  it('reports the tank life the window has to be read against', async () => {
    const { measureDiversity } = await import('./pond/measure.ts');
    const params = defaultParams();
    params.soupCount = 40;
    params.spawnInterval = 0;
    params.upkeep = 0.015;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    for (let i = 0; i < 300; i++) sim.step(1 / 60, params);
    const d = measureDiversity(sim, params);
    // 1.25 of tank at 0.015 a second: the 83 seconds the plan quotes.
    expect(d.tankLife).toBeCloseTo(83.33, 1);
    // And null rather than Infinity when upkeep is off, because a body that
    // pays nothing has no window, which is not the same as an endless one.
    params.upkeep = 0;
    expect(measureDiversity(sim, params).tankLife).toBeNull();
    expect(measureDiversity(sim).tankLife).toBeNull();
  });
});
