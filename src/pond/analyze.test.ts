import { describe, expect, it } from 'vitest';
import { defaultParams } from '../params.ts';
import { PondDb } from './db.ts';
import {
  METRICS,
  effectTable,
  effects,
  fold,
  metricKey,
  metricLabel,
  parseMetric,
  readEffect,
  seedBudget,
  sweepTrials,
  type Effect,
  type Series,
} from './analyze.ts';

/*
 * The fold is part of what a number means, and the case these are written
 * from is a real one: a patchy pond forages at 1.6-1.9x chance ground in its
 * first minute and is back at parity by the close, and `analyze` read the
 * close and said "no foraging". So the tests are about the folds having their
 * known answers on a hand-drawn timeline, about a metric reaching into the
 * JSON beside the columns, and about the effect table saying `thin` for the
 * one-seed case whatever eta-squared it happens to compute.
 */

const S: Series = { t: [0, 10, 20, 30], v: [1, 1.9, 1.2, 0.9] };

describe('fold', () => {
  it('has the obvious answers on a hump', () => {
    expect(fold(S, 'last')).toBe(0.9);
    expect(fold(S, 'peak')).toBe(1.9);
    expect(fold(S, 'trough')).toBe(0.9);
    // Warm-up of zero: the mean of the four.
    expect(fold(S, 'mean', 0)).toBeCloseTo(1.25, 12);
    // Nothing at or past sixty seconds, so `mean` falls back to everything.
    expect(fold(S, 'mean')).toBeCloseTo(1.25, 12);
    // Change per second over the last interval.
    expect(fold(S, 'window')).toBeCloseTo(-0.03, 12);
  });

  it('measures slope per simulated minute, past the warm-up', () => {
    const rising: Series = { t: [0, 60, 120, 180], v: [100, 110, 120, 130] };
    // One a second is sixty a minute, whichever samples are used.
    expect(fold(rising, 'slope', 0)).toBeCloseTo(10, 6);
    expect(fold(rising, 'slope')).toBeCloseTo(10, 6);
    // A bend before the warm-up is ignored once there are enough samples past it.
    const bent: Series = { t: [0, 60, 120, 180], v: [0, 110, 120, 130] };
    expect(fold(bent, 'slope')).toBeCloseTo(10, 6);
  });

  it('is null where there is nothing to stand on', () => {
    expect(fold({ t: [], v: [] }, 'last')).toBeNull();
    expect(fold({ t: [0], v: [5] }, 'window')).toBeNull();
    expect(fold({ t: [0], v: [5] }, 'slope')).toBeNull();
    // A single sample at t=0 is still the peak and the trough.
    expect(fold({ t: [0], v: [5] }, 'peak')).toBe(5);
  });
});

describe('metric keys', () => {
  it('parses a fold and falls back to the default', () => {
    expect(parseMetric('net_fst')).toEqual({ metric: 'net_fst', summary: 'last' });
    expect(parseMetric('net_fst@slope')).toEqual({ metric: 'net_fst', summary: 'slope' });
    expect(parseMetric('forage_ratio')).toEqual({ metric: 'forage_ratio', summary: 'peak' });
    expect(() => parseMetric('nope')).toThrow(/unknown metric/);
    expect(() => parseMetric('bodies@median')).toThrow(/unknown summary/);
  });

  it('keys and labels the shortest way', () => {
    expect(metricKey('forage_ratio', 'peak')).toBe('forage_ratio');
    expect(metricKey('forage_ratio', 'last')).toBe('forage_ratio@last');
    expect(metricLabel('forage_ratio@last')).toBe('forage@last');
    expect(metricLabel('commutes')).toBe('commutes/s');
  });

  it('every metric is a column, a json path or a derived rule', () => {
    for (const [k, spec] of Object.entries(METRICS)) {
      const src = spec.source as Record<string, unknown>;
      expect('column' in src || 'json' in src || 'derived' in src, k).toBe(true);
    }
  });
});

/** A `sample` row with everything the insert needs, then overrides. */
function sample(t: number, over: Record<string, unknown>): Record<string, unknown> {
  return {
    t, bodies: 100, wires: 0, lines: 1, bornMean: 0, bornMax: 0,
    spawned: 0, born: 0, died: 0, commutes: 0, erases: 0, annihilations: 0,
    latches: 0, snaps: 0, free: 0, ground: 0, escrow: 0, meanExtra: 0,
    canPay: 0, ppWires: 0, conDupWires: 0, commuteShare: null,
    commuteChance: 0, commuteEdge: null, matrixDrift: 0,
    ...over,
  };
}

describe('sweepTrials', () => {
  it('folds each metric its own way, and reaches into the json', () => {
    const db = new PondDb(':memory:');
    try {
      const params = { ...defaultParams(), senseScale: 0.5 };
      const run = db.startRun({
        seed: 1, seconds: 30, dt: 1 / 60, world: { w: 100, h: 100 }, fieldCells: 64,
        preset: 'soup', soupCount: 100, parentRun: null, params, commit: null, note: null,
        sweep: 'demo', point: { x: 1 },
      });
      const forage = [1, 1.9, 1.2, 0.9];
      const commutes = [0, 10, 30, 40];
      const latches = [0, 100, 120, 125];
      const bodies = [100, 110, 120, 130];
      const demand = [0, 0.7, 0.3, 0.05];
      const sigP90 = [0, 4, 4.4, 4.2];
      [0, 10, 20, 30].forEach((t, i) => {
        db.addSample(
          run,
          sample(t, {
            bodies: bodies[i],
            commutes: commutes[i],
            latches: latches[i],
            diversity: {
              forageRatio: forage[i],
              commutesPerLatch: latches[i] > 0 ? commutes[i] / latches[i] : null,
              demandMean: demand[i],
              signalP90: sigP90[i],
              loci: { wDemandH0: 1, wSelf00: 0.1 * i, tFoodH0: 1.8 },
            },
          }),
        );
      });
      const [row] = sweepTrials(db, 'demo', {
        metrics: [
          'forage_ratio', 'forage_ratio@last', 'commutes', 'commutes@last', 'commutes_per_latch',
          'commutes_per_latch@last', 'demand_mean', 'sense_read_p90', 'bodies@slope', 'locus_self_00', 'locus_self_00@slope',
        ],
        warmup: 0,
      });
      expect(row.point).toEqual({ x: 1 });
      expect(row.values.forage_ratio).toBe(1.9);
      expect(row.values['forage_ratio@last']).toBe(0.9);
      // A rate over the last ten seconds, and the total.
      expect(row.values.commutes).toBeCloseTo(1, 12);
      expect(row.values['commutes@last']).toBe(40);
      // Ten commutes against five latches in the window; the cumulative column otherwise.
      expect(row.values.commutes_per_latch).toBeCloseTo(2, 12);
      expect(row.values['commutes_per_latch@last']).toBeCloseTo(40 / 125, 12);
      expect(row.values.demand_mean).toBeCloseTo(0.7, 12);
      // p90 at the close times this run's own sense scale.
      expect(row.values.sense_read_p90).toBeCloseTo(4.2 * 0.5, 12);
      // One body a second is sixty a minute.
      expect(row.values['bodies@slope']).toBeCloseTo(60, 6);
      expect(row.values.locus_self_00).toBeCloseTo(0.3, 6);
      expect(row.values['locus_self_00@slope']).toBeCloseTo(0.6, 6);
      // The run's params come along, so a report can say what was held.
      expect(row.params?.senseScale).toBe(0.5);
    } finally {
      db.close();
    }
  });

  it('keeps nulls null and drops runs with no samples', () => {
    const db = new PondDb(':memory:');
    try {
      const mk = (point: Record<string, number>) =>
        db.startRun({
          seed: 1, seconds: 30, dt: 1 / 60, world: { w: 100, h: 100 }, fieldCells: 64,
          preset: 'soup', soupCount: 100, parentRun: null, params: defaultParams(), commit: null, note: null,
          sweep: 'demo', point,
        });
      const a = mk({ x: 0 });
      db.addSample(a, sample(10, { diversity: { netFst: null, forageRatio: null } }));
      mk({ x: 1 }); // no samples: a run that died before its first sample
      const rows = sweepTrials(db, 'demo', { metrics: ['net_fst', 'forage_ratio', 'demand_mean'] });
      expect(rows).toHaveLength(1);
      expect(rows[0].values.net_fst).toBeNull();
      expect(rows[0].values.forage_ratio).toBeNull();
      // A json path that is not there is null, not zero.
      expect(rows[0].values.demand_mean).toBeNull();
    } finally {
      db.close();
    }
  });
});

describe('reading an effect', () => {
  const trial = (x: number, seed: number, v: number) => ({ runId: 0, seed, point: { x }, values: { m: v }, params: {} });

  it('says thin with one seed a level, whatever eta-squared says', () => {
    const [e] = effects([trial(0, 1, 1), trial(1, 1, 100)], ['m']);
    expect(e.eta2).toBeCloseTo(1, 12);
    expect(readEffect(e)).toMatch(/^thin/);
    expect(effectTable([e])).toContain('thin');
  });

  it('resolves a clean separation at three seeds and reads its direction along the axis', () => {
    const [e] = effects([trial(0, 1, 1), trial(0, 2, 1.1), trial(0, 3, 0.9), trial(1, 1, 5), trial(1, 2, 5.2), trial(1, 3, 4.8)], ['m']);
    expect(readEffect(e)).toBe('resolved');
    expect(e.first.level).toBe(0);
    expect(e.last.level).toBe(1);
    expect(e.last.mean).toBeGreaterThan(e.first.mean);
  });

  it('calls a noisy gap unresolved and says how many seeds it would want', () => {
    const [e] = effects([trial(0, 1, 1), trial(0, 2, 9), trial(0, 3, 5), trial(1, 1, 2), trial(1, 2, 10), trial(1, 3, 6)], ['m']);
    expect(e.eta2).toBeLessThan(0.3);
    expect(readEffect(e)).toMatch(/^unresolved; ~\d+\/level/);
  });

  it('budgets seeds from the spread and the gap', () => {
    // Two standard errors clear at a gap equal to the spread: eight a level.
    expect(seedBudget(1, 1)).toBe(8);
    expect(seedBudget(0.5, 1)).toBe(2);
    expect(seedBudget(0.728, 0.3)).toBe(48);
    expect(seedBudget(1, 0)).toBeNull();
    expect(seedBudget(0, 1)).toBe(1);
  });

  it('effectTable names levels when asked', () => {
    const e: Effect = {
      axis: 'arm', metric: 'm', eta2: 0.9, noise: 0.1,
      low: { level: 0, mean: 1 }, high: { level: 1, mean: 2 },
      first: { level: 0, mean: 1 }, last: { level: 1, mean: 2 }, levels: 2, trials: 6,
    };
    const text = effectTable([e], 5, { levelName: (_axis, level) => (level === 0 ? 'minted' : 'conserved') });
    expect(text).toContain('minted: 1');
    expect(text).toContain('conserved: 2');
  });
});
