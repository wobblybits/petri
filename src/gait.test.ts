import { describe, expect, it } from 'vitest';
import { REWRITE_SHARE } from './energy.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';
import { pondMatter } from './test-params.ts';

/*
 * The gait: an ADP-activated pathway in every body, and the one actuator it
 * drives.
 *
 * These exist because the pathway ships switched off — see `metabolicRate`,
 * which is at zero until a sweep says what it should be — and a mechanism at
 * its neutral value is a mechanism nothing else in the suite touches. Every
 * other test either predates the pathway or names it in a control and turns
 * it off. So this file is the only place it runs, and each of these turns it
 * on by hand rather than inheriting a default that is one sweep away from
 * moving.
 *
 * All four are change detectors on the mechanism and none is a measure of
 * behaviour: whether the pond walks, and how far, is a `npm run pond`
 * question and not one the suite can answer.
 */

/** A dish with nothing in it but the pathway: no dish drive, no immigrants. */
function gaitParams(): Params {
  const p = defaultParams();
  p.soupCount = 0;
  p.spawnInterval = 0;
  p.energyRegrow = 0;
  p.decay = 0;
  p.diffuse = 0;
  p.deposit = 0;
  p.snapRadius = 0;
  p.stepSpeed = 0;
  p.swimNoise = 0;
  p.turnRate = 0;
  p.flockAlign = 0;
  p.flockSep = 0;
  p.declutter = 0;
  p.learnRate = 0;
  // The pathway, at the value the slider ships as its middle rather than the
  // zero the params ship. Everything below is about what it does when it is
  // running, so it has to be running.
  p.metabolicRate = 6;
  return p;
}

describe('the gait ships off', () => {
  it('is exactly one at the shipped default, on every wire', () => {
    /*
     * The claim `metabolicRate` makes about itself, pinned. At zero
     * `advanceGait` returns with the wave and the anchor untouched, so
     * `strokeOf` is exactly 1 — not nearly, exactly, because it is a
     * multiplier on a rest length and a wire that starts satisfied has to
     * stay that way to the last bit.
     */
    const p = gaitParams();
    p.metabolicRate = defaultParams().metabolicRate;
    expect(p.metabolicRate, 'the pathway ships at its neutral value').toBe(0);
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn('con', 1960, 2000, 0, p, true)!;
    const b = sim.spawn('con', 2040, 2000, 0, p, true)!;
    const wire = sim.graph.connect(sim.agents, { id: a.id, slot: 'l' }, { id: b.id, slot: 'r' }, sim.w, sim.h, p, sim.time)!;
    for (let f = 0; f < 60; f++) sim.step(1 / 60, p);
    const store = sim.agentStore;
    expect(store.gaitWave[a.slot]).toBe(0);
    expect(store.gaitWave[b.slot]).toBe(0);
    expect(store.anchor[a.slot]).toBe(0);
    expect(store.anchor[b.slot]).toBe(0);
    expect(sim.graph.strokeOf(wire.a, wire.b, sim.agents, store.gaitWave, p)).toBe(1);
  });
});

describe('the pathway', () => {
  it('oscillates once it is switched on, rather than settling', () => {
    /*
     * What makes it a clock the body owns rather than a charge it walks to.
     * The pathway either relaxes to a steady adenylate charge or runs round a
     * limit cycle, and only the second is a gait — so the wave has to cross
     * zero, and cross it back, more than the once any transient gets for free.
     *
     * The band that does is narrow and is `metabolicRegen`'s own note to
     * explain; this is the assertion that says which side of it the shipped
     * value is on. It was on the wrong side twice: at `regen` 1 the wave sat
     * at exactly −1 from the first second and the gait had never once moved,
     * and at 4 it sat just as still at nearly full charge.
     *
     * Long enough to catch a period. At a rate of 6 the cycle runs about 7 s,
     * so ten seconds is one turn and a bit — enough for two crossings and not
     * enough to be reading a transient as a cycle.
     */
    const p = gaitParams();
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn('con', 2000, 2000, 0, p, true)!;
    const wave = sim.agentStore.gaitWave;
    let crossings = 0;
    let lo = Infinity;
    let hi = -Infinity;
    let was = wave[a.slot];
    for (let f = 0; f < 1200; f++) {
      // Fed, so this is the pathway's own dynamics and not a body starving:
      // `buys its substrate out of the tank` is the test about the tank.
      a.extra = a.energyCap;
      sim.step(1 / 60, p);
      const now = wave[a.slot];
      if ((was <= 0 && now > 0) || (was >= 0 && now < 0)) crossings++;
      if (f > 600) {
        lo = Math.min(lo, now);
        hi = Math.max(hi, now);
      }
      was = now;
    }
    expect(crossings, 'the pathway settled instead of oscillating').toBeGreaterThan(1);
    // And swings, rather than shivering about a steady charge. The full range
    // is 2; a body that has settled reads a few thousandths.
    expect(hi - lo, 'the wave barely moved').toBeGreaterThan(1);
  });

  it('buys its substrate out of the tank and puts the price on the dish', () => {
    /*
     * The join to the economy, and the rule that makes it one: the pathway
     * spends, and what it spends leaves through the same road rent does. A
     * body that metabolises hard fertilises the cell it is standing in, so
     * the pond's total does not move.
     *
     * With every other source and sink off, so the only thing running is the
     * pathway. `upkeepExcrete` is the road; at zero the spend would simply be
     * destroyed, which is a different test and not this one.
     */
    const p = gaitParams();
    p.upkeepExcrete = 1;
    p.bodyValue = REWRITE_SHARE;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn('con', sim.w * 0.5, sim.h * 0.5, 0, p, true)!;
    a.pinned = true;
    a.extra = a.energyCap;
    const before = pondMatter(sim, p.bodyValue);
    let worst = 0;
    for (let f = 0; f < 300; f++) {
      sim.step(1 / 60, p);
      worst = Math.max(worst, Math.abs(pondMatter(sim, p.bodyValue) - before));
    }
    // Something has to have been spent, or this is a test about a still body.
    expect(a.extra, 'the pathway bought nothing').toBeLessThan(a.energyCap);
    // Float32 cells summed across the whole field, which is the floor
    // `chemistry.test.ts` gives its own conservation assertions.
    expect(worst).toBeLessThan(before * 1e-5);
  });
});

describe('the stroke', () => {
  it('is the mean of a wire’s two ends, floored well above nothing', () => {
    /*
     * A wire is one muscle rather than two arguing, so a phase difference
     * across it shortens the stroke instead of tearing it in half. Written on
     * the wave directly, because the point is the arithmetic and not the
     * pathway that produced it.
     */
    const p = gaitParams();
    p.gaitSwell = 0.3;
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn('con', 1960, 2000, 0, p, true)!;
    const b = sim.spawn('con', 2040, 2000, 0, p, true)!;
    const wire = sim.graph.connect(sim.agents, { id: a.id, slot: 'l' }, { id: b.id, slot: 'r' }, sim.w, sim.h, p, sim.time)!;
    const wave = sim.agentStore.gaitWave;
    const strokeOf = (): number => sim.graph.strokeOf(wire.a, wire.b, sim.agents, wave, p);

    wave[a.slot] = 1;
    wave[b.slot] = 1;
    expect(strokeOf()).toBeCloseTo(1.3, 12);
    // Opposed ends cancel: the muscle between them is doing nothing.
    wave[a.slot] = 1;
    wave[b.slot] = -1;
    expect(strokeOf()).toBeCloseTo(1, 12);
    // And a wire hauling its ends into contact is a rewrite, so the floor
    // holds however hard both ends pull.
    p.gaitSwell = 4;
    wave[a.slot] = -1;
    wave[b.slot] = -1;
    expect(strokeOf()).toBe(0.4);
  });

  it('leaves a fresh wire satisfied at the length it latched at', () => {
    /*
     * The span a wire is observed to have is the *contracted* one when its
     * bodies are mid-stroke. `connect` divides the stroke back out of the
     * length it seeds the shrink ramp with, so that the first `syncRest`
     * hands the solver the span the wire is actually at.
     *
     * Without that, `rest` stepped by the whole stroke on the frame after
     * every latch — a near-rigid span constraint asked to resolve a fifth of
     * a wire's length in one substep, which reads as a kick, and which is the
     * thing `connect`'s own comment is careful about for the shrink ramp.
     */
    const p = gaitParams();
    p.gaitSwell = 0.3;
    // The breath rides on top of the stroke and would blur the comparison by
    // a few per cent; it is `wireBreathe`'s own tests' business, not this one.
    p.wireBreathe = 0;
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn('con', 1960, 2000, 0, p, true)!;
    const b = sim.spawn('con', 2040, 2000, 0, p, true)!;
    // Let the two pathways run apart, so the wire is latched by bodies that
    // are genuinely out of phase rather than both sitting at the seed.
    for (let f = 0; f < 40; f++) sim.step(1 / 60, p);

    const wire = sim.graph.connect(sim.agents, { id: a.id, slot: 'l' }, { id: b.id, slot: 'r' }, sim.w, sim.h, p, sim.time)!;
    const span = wire.rest;
    const stroke = sim.graph.strokeOf(wire.a, wire.b, sim.agents, sim.agentStore.gaitWave, p);
    // Or this asserts nothing: at a stroke of 1 every arrangement agrees.
    expect(Math.abs(stroke - 1), 'the two ends were in phase, so there was no step to make').toBeGreaterThan(0.01);
    sim.graph.syncRest(sim.time, p, sim.agents, sim.agentStore.gaitWave, undefined);
    expect(wire.rest).toBeCloseTo(span, 6);
    // Which is a different number from the one the double-application gave.
    expect(Math.abs(wire.rest - span * stroke)).toBeGreaterThan(span * 0.005);
  });
});
