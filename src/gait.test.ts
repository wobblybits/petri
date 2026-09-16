import { describe, expect, it } from 'vitest';
import { REACT_A, REACT_B, REACT_C, REACT_D, REACT_SPECIES } from './agent-store.ts';
import { REWRITE_SHARE } from './energy.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';
import { pondMatter } from './test-params.ts';

/*
 * The gait: a four-chemical reactor in every body, what it broadcasts down a
 * wire, and the one actuator it drives.
 *
 * The reactor is `docs/scratch.txt` §3 — A primary fuel, B active primer, C
 * saturated catalyst, D reset inhibitor — and `Sim.advanceGait` has the
 * reactions. It runs in the shipping pond, so this file is the only place in
 * the suite that leaves it on: `fixedParams` pins it off for everything else,
 * the way it already pins learning, because a second spender makes every
 * assertion about a tank probabilistic.
 *
 * These are change detectors on the mechanism and none is a measure of
 * behaviour. Whether the pond walks, and how far, is a `npm run pond`
 * question and not one the suite can answer.
 */

/** A dish with nothing in it but the reactor: no dish drive, no immigrants. */
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
  return p;
}

function speciesOf(sim: Sim, slot: number): { A: number; B: number; C: number; D: number } {
  const o = slot * REACT_SPECIES;
  const r = sim.agentStore.react;
  return { A: r[o + REACT_A], B: r[o + REACT_B], C: r[o + REACT_C], D: r[o + REACT_D] };
}

describe('the gait ships on', () => {
  it('runs at the shipped default, and is exactly one at zero', () => {
    const shipped = defaultParams();
    expect(shipped.metabolicRate, 'the reactor ships off again').toBeGreaterThan(0);
    expect(shipped.metabolicDiffuse, 'the broadcast ships off').toBeGreaterThan(0);

    /*
     * The neutral behaviour is still worth pinning, because it is what every
     * test that is not about the gait runs at: at zero `advanceGait` returns
     * with the wave and the anchor untouched, so `strokeOf` is exactly 1 —
     * not nearly, exactly, because it is a multiplier on a rest length and a
     * wire that starts satisfied has to stay that way to the last bit.
     */
    const p = gaitParams();
    p.metabolicRate = 0;
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn('con', 1960, 2000, 0, p, true)!;
    const b = sim.spawn('con', 2040, 2000, 0, p, true)!;
    const wire = sim.graph.connect(sim.agents, { id: a.id, slot: 'l' }, { id: b.id, slot: 'r' }, sim.w, sim.h, p, sim.time)!;
    for (let f = 0; f < 60; f++) sim.step(1 / 60, p);
    const store = sim.agentStore;
    expect(store.gaitWave[a.slot]).toBe(0);
    expect(store.anchor[a.slot]).toBe(0);
    expect(sim.graph.strokeOf(wire.a, wire.b, sim.agents, store.gaitWave, p)).toBe(1);
  });
});

describe('the reactor', () => {
  /** One fed, pinned body, watched for long enough to catch several cycles. */
  function lone(p: Params, fullness = 1, frames = 1500) {
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn('con', 2000, 2000, 0, p, true)!;
    a.pinned = true;
    const store = sim.agentStore;
    let lo = Infinity;
    let hi = -Infinity;
    let crossings = 0;
    let was = 0;
    let first = -1;
    let last = -1;
    for (let f = 0; f < frames; f++) {
      a.extra = a.energyCap * fullness;
      sim.step(1 / 60, p);
      const w = store.gaitWave[a.slot];
      if (f > frames * 0.4) {
        lo = Math.min(lo, w);
        hi = Math.max(hi, w);
        const now = w > 0 ? 1 : 0;
        if (was === 0 && now === 1) {
          crossings++;
          if (first < 0) first = f;
          last = f;
        }
        was = now;
      }
    }
    return { sim, a, swing: hi - lo, lo, hi, crossings,
             period: crossings > 1 ? (last - first) / (crossings - 1) / 60 : null };
  }

  it('runs a limit cycle in B, C and D while A sits steady', () => {
    /*
     * The shape of the doc's reactor, and the thing its own python never
     * checks. A is fed from outside and only feeds B, so its row decouples
     * and it settles to `J / (k1 + decay)` whatever it started at; the
     * oscillation is the loop B -> C -> D -| B. If A moved with the cycle
     * this would be a different system than the one the constants were
     * chosen for.
     */
    const p = gaitParams();
    const r = lone(p);
    expect(r.crossings, 'the reactor settled instead of oscillating').toBeGreaterThan(2);
    expect(r.swing, 'the wave barely moved').toBeGreaterThan(1);

    const before = speciesOf(r.sim, r.a.slot);
    for (let f = 0; f < 120; f++) {
      r.a.extra = r.a.energyCap;
      r.sim.step(1 / 60, p);
    }
    const after = speciesOf(r.sim, r.a.slot);
    // A within a whisker of where it was, across two seconds that carried a
    // whole cycle of the other three.
    expect(Math.abs(after.A - before.A), 'the fuel moved with the cycle').toBeLessThan(0.02 * before.A + 1e-6);
    expect(before.A, 'the fuel never arrived').toBeGreaterThan(0.1);
    // And it sits where the algebra says: J / (k1 + decay).
    const want = (p.metabolicSupply * 1 * 1) / (p.metabolicFuel + p.metabolicDecay);
    expect(after.A).toBeCloseTo(want, 1);
    expect(after.B + after.C + after.D, 'the loop is empty').toBeGreaterThan(0.5);
  });

  it('has a fuel window with both edges', () => {
    /*
     * The gate, and nothing had to be added to get it. A starving body sits
     * below the Hopf boundary and is still; a fed one runs the cycle; and the
     * period shortens as the fuel rises, so a full body strokes faster than a
     * lean one. `metabolicFuel` carries the stability analysis this comes
     * from and `metabolicSupply` places the tank on it.
     */
    const p = gaitParams();
    const starved = lone(p, 0.12);
    const lean = lone(p, 0.5);
    const fed = lone(p, 1);
    expect(starved.swing, 'a starving body undulated').toBeLessThan(0.2);
    expect(lean.swing, 'a half-full body was still').toBeGreaterThan(1);
    expect(fed.swing, 'a fed body was still').toBeGreaterThan(1);
    expect(fed.period!, 'a fed body did not stroke faster than a lean one').toBeLessThan(lean.period!);
  });

  it('buys its fuel out of the tank and puts the price on the dish', () => {
    /*
     * The join to the economy, and the rule that makes it one: the reactor
     * spends, and what it spends leaves through the same road rent does. A
     * body that metabolises hard fertilises the cell it is standing in, so
     * the pond's total does not move.
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
    expect(a.extra, 'the reactor bought nothing').toBeLessThan(a.energyCap);
    // Float32 cells summed across the whole field, which is the floor
    // `chemistry.test.ts` gives its own conservation assertions.
    expect(worst).toBeLessThan(before * 1e-5);
  });
});

describe('the broadcast', () => {
  /*
   * The doc's §4.1 transmission presets, as genes: a Con broadcasts the
   * catalyst C, a Dup broadcasts the inhibitor D, and a body broadcasts out
   * of its principal port only. The wire carries each species as one signed
   * number both ends read.
   */
  function pair(kindA: 'con' | 'dup', slotA: 'p' | 'l', kindB: 'con' | 'dup', slotB: 'p' | 'l', p: Params) {
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn(kindA, 1960, 2000, 0, p, true)!;
    const b = sim.spawn(kindB, 2040, 2000, 0, p, true)!;
    a.pinned = true;
    b.pinned = true;
    const wire = sim.graph.connect(sim.agents, { id: a.id, slot: slotA }, { id: b.id, slot: slotB }, sim.w, sim.h, p, sim.time)!;
    return { sim, a, b, wire };
  }

  function totals(sim: Sim, a: { extra: number; energyCap: number }, b: { extra: number; energyCap: number }, wire: { fluxC: number; fluxD: number }, p: Params) {
    let C = 0;
    let D = 0;
    for (let f = 0; f < 900; f++) {
      a.extra = a.energyCap;
      b.extra = b.energyCap;
      sim.step(1 / 60, p);
      C += wire.fluxC;
      D += wire.fluxD;
    }
    return { C, D };
  }

  it('sends the catalyst from a Con and the inhibitor from a Dup', () => {
    const p = gaitParams();
    {
      const { sim, a, b, wire } = pair('con', 'p', 'con', 'l', p);
      const t = totals(sim, a, b, wire, p);
      expect(t.C, 'the Con sent no catalyst').toBeGreaterThan(0);
      expect(t.D, 'the Con sent inhibitor').toBe(0);
    }
    {
      const { sim, a, b, wire } = pair('dup', 'p', 'con', 'l', p);
      const t = totals(sim, a, b, wire, p);
      expect(t.D, 'the Dup sent no inhibitor').toBeGreaterThan(0);
      expect(t.C, 'the Dup sent catalyst').toBe(0);
    }
  });

  it('broadcasts out of the principal port, and only that way', () => {
    const p = gaitParams();
    // The principal on `a` sends toward `b`, so the flux is positive.
    {
      const { sim, a, b, wire } = pair('con', 'p', 'con', 'l', p);
      const t = totals(sim, a, b, wire, p);
      expect(t.C).toBeGreaterThan(0);
    }
    // Mirrored: the principal on `b` sends toward `a`, so it is negative.
    {
      const { sim, a, b, wire } = pair('con', 'l', 'con', 'p', p);
      const t = totals(sim, a, b, wire, p);
      expect(t.C).toBeLessThan(0);
    }
  });

  it('moves catalyst between the two ends of one wire, and mints none', () => {
    /*
     * The wire is the bucket: one signed number per species, read by both
     * ends, so what leaves one body is exactly what arrives at the other.
     * The reactor is slowed to nothing so the broadcast is the only thing
     * moving C, and the sender's catalyst is set by hand so it is speaking.
     */
    const p = gaitParams();
    p.metabolicRate = 1e-15;
    const { sim, a, b, wire } = pair('con', 'p', 'con', 'l', p);
    const store = sim.agentStore;
    const lone = sim.spawn('con', 2000, 2400, 0, p, true)!;
    lone.pinned = true;
    // Two frames so the heads have been read off the genome at least once.
    sim.step(1 / 60, p);
    sim.step(1 / 60, p);
    const oa = a.slot * REACT_SPECIES + REACT_C;
    const ob = b.slot * REACT_SPECIES + REACT_C;
    const ol = lone.slot * REACT_SPECIES + REACT_C;
    store.react[oa] = 6;
    store.react[ob] = 0;
    store.react[ol] = 6;
    sim.step(1 / 60, p);
    const moved = wire.fluxC;
    expect(moved).toBeGreaterThan(0);
    expect(store.react[ob]).toBeCloseTo(moved, 12);
    expect(store.react[oa] + store.react[ob]).toBeCloseTo(6, 9);
    expect(store.react[ol], 'an unwired body was touched').toBeCloseTo(6, 9);

    // And nothing at all with the broadcast dial at zero.
    p.metabolicDiffuse = 0;
    store.react[oa] = 6;
    store.react[ob] = 0;
    sim.step(1 / 60, p);
    expect(store.react[ob]).toBeCloseTo(0, 9);
  });
});

describe('the stroke', () => {
  it('is the mean of a wire’s two ends, floored well above nothing', () => {
    const p = gaitParams();
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const a = sim.spawn('con', 1960, 2000, 0, p, true)!;
    const b = sim.spawn('con', 2040, 2000, 0, p, true)!;
    const wire = sim.graph.connect(sim.agents, { id: a.id, slot: 'l' }, { id: b.id, slot: 'r' }, sim.w, sim.h, p, sim.time)!;
    const wave = sim.agentStore.gaitWave;
    wave[a.slot] = 1;
    wave[b.slot] = -1;
    expect(sim.graph.strokeOf(wire.a, wire.b, sim.agents, wave, p)).toBeCloseTo(1, 12);
    wave[a.slot] = 1;
    wave[b.slot] = 1;
    expect(sim.graph.strokeOf(wire.a, wire.b, sim.agents, wave, p)).toBeCloseTo(1 + p.gaitSwell, 12);
    // Floored: a wire hauling its ends into contact is a rewrite, not a gait.
    const deep = { ...p, gaitSwell: 4 };
    wave[a.slot] = -1;
    wave[b.slot] = -1;
    expect(sim.graph.strokeOf(wire.a, wire.b, sim.agents, wave, deep)).toBe(0.4);
  });
});

describe('a chain', () => {
  /*
   * What the broadcast is for. Six pinned Cons wired principal to auxiliary,
   * so every body's one mouth faces the next body's ear, started at the
   * scattered catalyst levels `createAgent` seeds and fed throughout.
   *
   * Uncoupled they free-run at whatever phases they were seeded with.
   * Coupled they lock with a lag per wire, and a fixed phase difference per
   * wire along a chain is a travelling wave — which along a body is
   * peristalsis. The bounds are loose because this is a change detector and
   * not a measure; the numbers it was written against are in
   * `metabolicDiffuse`.
   */
  function lockOf(coupling: number, n = 6) {
    const p = gaitParams();
    p.metabolicDiffuse = coupling;
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const bodies: ReturnType<Sim['spawn']>[] = [];
    for (let i = 0; i < n; i++) {
      const a = sim.spawn('con', 1200 + i * 60, 2000, 0, p, true)!;
      a.pinned = true;
      bodies.push(a);
    }
    for (let i = 0; i + 1 < n; i++) {
      sim.graph.connect(sim.agents, { id: bodies[i]!.id, slot: 'p' }, { id: bodies[i + 1]!.id, slot: 'l' }, sim.w, sim.h, p, sim.time);
    }
    const store = sim.agentStore;
    const feed = () => {
      for (const a of sim.agents.values()) a.extra = a.energyCap;
    };
    for (let f = 0; f < 900; f++) {
      feed();
      sim.step(1 / 60, p);
    }
    const cross: number[][] = bodies.map(() => []);
    const was = bodies.map((b) => store.gaitWave[b!.slot]);
    for (let f = 0; f < 900; f++) {
      feed();
      sim.step(1 / 60, p);
      for (let i = 0; i < n; i++) {
        const w = store.gaitWave[bodies[i]!.slot];
        if (was[i] <= 0 && w > 0) cross[i].push(f);
        was[i] = w;
      }
    }
    const first = cross[0];
    expect(first.length, 'the head of the chain never cycled').toBeGreaterThan(1);
    const period = (first[first.length - 1] - first[0]) / (first.length - 1);
    const lags: number[] = [];
    for (let i = 1; i < n; i++) {
      const d: number[] = [];
      for (const t of cross[i]) {
        let best: number | null = null;
        for (const u of cross[i - 1]) {
          const g = t - u;
          if (g >= 0 && (best === null || g < best)) best = g;
        }
        if (best !== null && best < period) d.push(best);
      }
      d.sort((x, y) => x - y);
      if (d.length) lags.push(d[d.length >> 1]);
    }
    /*
     * Interior wires only. The last body's principal is free, so it only ever
     * receives: the catalyst it is fed runs to D, D quenches its primer, and
     * its own loop stalls. That is the doc's stoichiometry and it is a real
     * thing about a chain's loose end, but it is not what the lock is about.
     */
    const inner = lags.slice(0, n - 3);
    expect(inner.length, 'no interior wire had a comparable pair of crossings').toBeGreaterThan(1);
    const lag = inner.reduce((x, y) => x + y, 0) / inner.length;
    const spread = Math.max(...inner) - Math.min(...inner);
    let swing = { lo: Infinity, hi: -Infinity };
    for (let f = 0; f < 300; f++) {
      feed();
      sim.step(1 / 60, p);
      const w = store.gaitWave[bodies[0]!.slot];
      swing = { lo: Math.min(swing.lo, w), hi: Math.max(swing.hi, w) };
    }
    return { lag, period, inner, spread, headSwing: swing.hi - swing.lo };
  }

  it('free-runs uncoupled and locks with a lag when it is coupled', () => {
    const free = lockOf(0);
    const locked = lockOf(defaultParams().metabolicDiffuse);

    /*
     * A lock is a *consistent* phase difference, not a small one. Uncoupled,
     * the offsets between neighbours are whatever the seeded catalyst levels
     * left and they never converge, so they are scattered: measured, the
     * interior wires read 18, 0 and 0 frames against a period of 101. Coupled
     * at the shipped rate they read 10, 9 and 9 against 99. So the spread is
     * what says whether the chain locked, and the lag says what it locked at.
     */
    expect(free.spread, 'uncoupled bodies held a consistent offset anyway').toBeGreaterThan(8);
    expect(locked.spread, 'the chain did not lock').toBeLessThan(4);

    // And what it locked at is a wave: a tenth of a cycle a wire, which is
    // neither synchrony nor a free run.
    expect(locked.lag, 'no lag, so no wave').toBeGreaterThan(2);
    expect(locked.lag, 'the chain synchronised instead of travelling').toBeLessThan(locked.period * 0.25);
    expect(locked.headSwing, 'the coupled chain stopped oscillating').toBeGreaterThan(1);
  });

  it('flatlines when the broadcast is turned up past the reactor', () => {
    /*
     * The upper edge, and it is the reactor's. Broadcasting spends the very
     * catalyst the sender's own loop runs on, so a chain shouted through
     * loudly enough stops oscillating altogether rather than locking harder.
     * That is why `metabolicDiffuse` ships nearer the bottom of its band than
     * the top, and it is the reason there is a band at all.
     */
    const p = gaitParams();
    p.metabolicDiffuse = 8;
    const sim = new Sim(4000, 4000);
    loadPreset(sim, 'soup', p);
    const bodies: ReturnType<Sim['spawn']>[] = [];
    for (let i = 0; i < 6; i++) {
      const a = sim.spawn('con', 1200 + i * 60, 2000, 0, p, true)!;
      a.pinned = true;
      bodies.push(a);
    }
    for (let i = 0; i + 1 < 6; i++) {
      sim.graph.connect(sim.agents, { id: bodies[i]!.id, slot: 'p' }, { id: bodies[i + 1]!.id, slot: 'l' }, sim.w, sim.h, p, sim.time);
    }
    const store = sim.agentStore;
    const feed = () => {
      for (const a of sim.agents.values()) a.extra = a.energyCap;
    };
    for (let f = 0; f < 1800; f++) {
      feed();
      sim.step(1 / 60, p);
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (let f = 0; f < 300; f++) {
      feed();
      sim.step(1 / 60, p);
      const w = store.gaitWave[bodies[0]!.slot];
      lo = Math.min(lo, w);
      hi = Math.max(hi, w);
    }
    // Against about 1.4 at the shipped rate, on the same chain.
    expect(hi - lo, 'the chain survived being shouted through').toBeLessThan(0.5);
  });
});
