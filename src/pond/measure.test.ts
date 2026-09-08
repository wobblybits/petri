import { describe, expect, it } from 'vitest';
import { CHEM_LEN, TASTE } from '../chem-layout.ts';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { measureDiversity, shannon } from './measure.ts';
import { effects, type TrialRow } from './analyze.ts';

/*
 * These measure the measures. Every one of them is a summary statistic that
 * nothing in the simulation reads, which means a wrong one is invisible: it
 * produces a plausible number, the number goes in a table, and the table is
 * used to decide what to build next. So the tests are about the cases where a
 * statistic has a *known* answer, and about the two ways each one can be
 * degenerate — nothing to compare, and everything identical.
 */

describe('shannon', () => {
  it('is zero for one group and log n for n equal ones', () => {
    expect(shannon([5]).h).toBe(0);
    expect(shannon([5]).effective).toBe(1);
    const four = shannon([3, 3, 3, 3]);
    expect(four.h).toBeCloseTo(Math.log(4), 12);
    // The readable unit: four equal groups are effectively four groups.
    expect(four.effective).toBeCloseTo(4, 10);
  });

  it('reads a dominated distribution as far fewer groups than there are', () => {
    // A hundred lines of which one holds almost everything is not a hundred
    // lines, and a count says it is. This is the whole reason for the measure.
    const counts = [970, ...Array(29).fill(1)];
    expect(counts.length).toBe(30);
    const s = shannon(counts);
    expect(s.effective).toBeLessThan(2);
  });

  it('ignores empty groups and an empty population', () => {
    expect(shannon([]).effective).toBe(0);
    expect(shannon([0, 0]).effective).toBe(0);
    expect(shannon([4, 0, 4]).effective).toBeCloseTo(2, 10);
  });
});

/** A pond with `n` founders, run far enough that genomes have drifted apart. */
function pond(n: number, frames: number): Sim {
  const params = defaultParams();
  params.soupCount = n;
  params.spawnInterval = 0;
  const sim = new Sim(1600, 1200, 128);
  loadPreset(sim, 'soup', params);
  for (let i = 0; i < frames; i++) sim.step(1 / 60, params);
  return sim;
}

describe('diversity', () => {
  it('reports nothing to compare rather than zero, on a fresh soup', () => {
    /*
     * A fresh soup's genomes are identical per kind — every matrix seeds to
     * zero — so there is no variance anywhere and F_ST is undefined, not zero.
     * Zero would read as "no differentiation", which is a finding; null is the
     * truth, which is that the question has no answer yet.
     */
    const sim = pond(60, 0);
    const d = measureDiversity(sim);
    expect(d.bodies).toBe(60);
    /*
     * Null, not 1. Every body is its own component and its own founder line,
     * so within-group variance is identically zero and `1 - vW/vT` is 1 by
     * construction — a fact about the grouping and not about the data. It read
     * 1 for the first sample of every run, which is the value that would mean
     * the thing this measure exists to watch for.
     */
    expect(d.netFst).toBeNull();
    expect(d.lineFst).toBeNull();
    expect(d.varianceDrifted).toBe(0);
    // Every founder is its own line, and none has any wires yet.
    expect(d.lines).toBe(60);
    expect(d.linesEffective).toBeCloseTo(60, 6);
    expect(d.nets).toBe(0);
    expect(d.commutesPerLatch).toBeNull();
  });

  it('reports nothing to compare when every group is a singleton', () => {
    // The degenerate case on its own, because it is the one that produced a
    // confident wrong answer rather than an obviously broken one.
    const sim = pond(40, 0);
    const store = sim.agentStore;
    // Give the genomes real variance, so the only thing making F_ST
    // undefined is that no group has two members to vary within.
    let k = 0;
    for (const a of sim.agents.values()) {
      store.chemAll[a.slot * CHEM_LEN + TASTE + 4] = k++ * 0.1;
    }
    const d = measureDiversity(sim);
    expect(d.varianceDrifted).toBeGreaterThan(0);
    expect(d.nets).toBe(0);
    expect(d.netFst).toBeNull();
    expect(d.lineFst).toBeNull();
  });

  it('separates richness from evenness', () => {
    const sim = pond(80, 600);
    const d = measureDiversity(sim);
    // A pond keeps most of its founders this early, so richness is high and
    // near-even; what the pair is for is telling those apart later.
    expect(d.lines).toBeGreaterThan(1);
    expect(d.linesEffective).toBeGreaterThan(1);
    expect(d.linesEffective).toBeLessThanOrEqual(d.lines + 1e-9);
    expect(d.lineDominance).toBeGreaterThan(0);
    expect(d.lineDominance).toBeLessThanOrEqual(1);
  });

  it('reads a pond with no structure as undifferentiated', () => {
    /*
     * The calibration that matters. Genomes assigned at random to nets carry
     * no between-net signal, so F_ST has to come out near zero — if it does
     * not, the statistic is measuring group *sizes* rather than group content
     * and every reading of it would be an artifact.
     */
    const sim = pond(120, 900);
    const d = measureDiversity(sim);
    if (d.netFst === null) return; // too few nets in this pond; nothing to check
    const store = sim.agentStore;
    const slots = [...sim.agents.values()].map((a) => a.slot);
    // Shuffle the drifted span between bodies, keeping the population's
    // per-locus distribution exactly and destroying any association with net.
    const span = CHEM_LEN - (TASTE + 4);
    const buf = new Float32Array(slots.length * span);
    for (let i = 0; i < slots.length; i++) {
      for (let k = 0; k < span; k++) buf[i * span + k] = store.chemAll[slots[i] * CHEM_LEN + TASTE + 4 + k];
    }
    for (let k = 0; k < span; k++) {
      for (let i = slots.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const a = buf[i * span + k];
        buf[i * span + k] = buf[j * span + k];
        buf[j * span + k] = a;
      }
    }
    for (let i = 0; i < slots.length; i++) {
      for (let k = 0; k < span; k++) store.chemAll[slots[i] * CHEM_LEN + TASTE + 4 + k] = buf[i * span + k];
    }
    const shuffled = measureDiversity(sim);
    expect(shuffled.netFst).not.toBeNull();
    // Not zero — with small groups the sampling noise alone lifts it — but far
    // below what a structured pond reads, and this is the floor to compare to.
    expect(shuffled.netFst!).toBeLessThan(0.5);
  });

  it('reads perfectly separated groups as fully differentiated', () => {
    // The other calibration: give every net one constant genome of its own and
    // F_ST must saturate. Between-group variance is then all of it.
    const sim = pond(120, 900);
    const before = measureDiversity(sim);
    if (before.nets < 2) return;
    const store = sim.agentStore;
    const comps = sim.graph.componentIds(sim.agents, sim.rosterVersion);
    const tag = new Map<number, number>();
    for (const a of sim.agents.values()) {
      const root = comps.get(a.id)!;
      if (!tag.has(root)) tag.set(root, tag.size);
      const v = tag.get(root)!;
      for (let k = TASTE + 4; k < CHEM_LEN; k++) store.chemAll[a.slot * CHEM_LEN + k] = v;
    }
    const after = measureDiversity(sim);
    expect(after.netFst).not.toBeNull();
    expect(after.netFst!).toBeGreaterThan(0.99);
  });
});

describe('effect ranking', () => {
  const trial = (point: Record<string, number>, seed: number, y: number): TrialRow => ({
    runId: seed,
    seed,
    point,
    values: { bodies: y },
  });

  it('scores an axis that explains everything near one and noise near zero', () => {
    const clean = [
      trial({ a: 0 }, 1, 10), trial({ a: 0 }, 2, 10), trial({ a: 1 }, 1, 20), trial({ a: 1 }, 2, 20),
    ];
    const [e] = effects(clean, ['bodies']);
    expect(e.eta2).toBeCloseTo(1, 10);
    expect(e.noise).toBe(0);
    expect(e.low).toEqual({ level: 0, mean: 10 });
    expect(e.high).toEqual({ level: 1, mean: 20 });

    // The same spread, but all of it between seeds rather than between levels.
    const noisy = [
      trial({ a: 0 }, 1, 10), trial({ a: 0 }, 2, 20), trial({ a: 1 }, 1, 20), trial({ a: 1 }, 2, 10),
    ];
    expect(effects(noisy, ['bodies'])[0].eta2).toBeCloseTo(0, 10);
  });

  it('drops nulls rather than counting them as zero', () => {
    // `netFst` is null for a pond with one net. Scoring that as no
    // differentiation would invent a finding out of an absence.
    const rows: TrialRow[] = [
      { runId: 1, seed: 1, point: { a: 0 }, values: { netFst: null } },
      { runId: 2, seed: 2, point: { a: 0 }, values: { netFst: 0.2 } },
      { runId: 3, seed: 1, point: { a: 1 }, values: { netFst: 0.8 } },
      { runId: 4, seed: 2, point: { a: 1 }, values: { netFst: 0.8 } },
    ];
    const [e] = effects(rows, ['netFst']);
    expect(e.trials).toBe(3);
    expect(e.low.mean).toBeCloseTo(0.2, 10);
    expect(e.high.mean).toBeCloseTo(0.8, 10);
  });

  it('says nothing about an axis that never varied', () => {
    const flat = [trial({ a: 1 }, 1, 10), trial({ a: 1 }, 2, 12)];
    expect(effects(flat, ['bodies'])).toEqual([]);
  });
});
