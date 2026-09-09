import { describe, expect, it } from 'vitest';
import { latinHypercube, parseAxis } from './sample.ts';
import {
  conditionalEffects,
  correlate,
  kmeans,
  matrix,
  pca,
  pls,
  standardise,
} from './explore.ts';

/*
 * An analysis tool that is subtly wrong does not fail — it produces a
 * plausible number and someone believes it. So every one of these is a case
 * with an answer known in advance, and the important ones are the *negative*
 * controls: a method that finds structure in noise is worse than no method.
 */

/** A seeded stream, so a "random" test is a reproducible one. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
const gauss = (r: () => number): number =>
  Math.sqrt(-2 * Math.log(Math.max(1e-12, r()))) * Math.cos(2 * Math.PI * r());

describe('sampling', () => {
  it('takes ranges from the sliders and log-scales the wide ones', () => {
    expect(parseAxis('diffuse')).toEqual({ key: 'diffuse', min: 0, max: 1, log: false });
    // A step of one or more means a count: the sim floors it, so the draw is
    // snapped and the stored parameter is the one that actually ran.
    expect(parseAxis('groundPatches').step).toBe(1);
    expect(parseAxis('groundPatches=0..64').step).toBe(1);
    expect(parseAxis('excreteRate').step).toBeUndefined();
    expect(latinHypercube([parseAxis('groundPatches=0..64')], 8, 5).every(
      (p) => Number.isInteger(p.groundPatches),
    )).toBe(true);
    // Spans 200x with a floor above zero, so log.
    expect(parseAxis('uptakeKs').log).toBe(true);
    // A floor at zero is a switch, not a scale.
    expect(parseAxis('deposit').log).toBe(false);
    expect(parseAxis('excreteRate=0.001..0.5')).toEqual({
      key: 'excreteRate', min: 0.001, max: 0.5, log: true,
    });
    expect(() => parseAxis('soupCount')).toThrow(/no declared range/);
    expect(() => parseAxis('diffuse=1..0')).toThrow(/backwards/);
  });

  it('covers every range without gaps, which independent draws do not', () => {
    /*
     * The point of a Latin hypercube. Twenty independent uniforms leave whole
     * stretches of a range untouched by chance, and a transition hiding in one
     * of those stretches is invisible; stratifying guarantees one draw per
     * twentieth of every axis.
     */
    const axes = [parseAxis('diffuse'), parseAxis('decay')];
    const pts = latinHypercube(axes, 20, 7);
    expect(pts).toHaveLength(20);
    for (const axis of axes) {
      const bins = new Set<number>();
      for (const p of pts) {
        const u = (p[axis.key as string] - axis.min) / (axis.max - axis.min);
        expect(u).toBeGreaterThanOrEqual(0);
        expect(u).toBeLessThanOrEqual(1);
        bins.add(Math.min(19, Math.floor(u * 20)));
      }
      expect(bins.size, `${String(axis.key)} left bins empty`).toBe(20);
    }
    // Reproducible, like a grid.
    expect(latinHypercube(axes, 20, 7)).toEqual(pts);
    expect(latinHypercube(axes, 20, 8)).not.toEqual(pts);
  });

  it('spreads a log axis evenly in the exponent, not the value', () => {
    const axis = parseAxis('excreteRate=0.001..1');
    const pts = latinHypercube([axis], 40, 3).map((p) => p.excreteRate);
    // The lowest of the three decades, 0.001 to 0.01. Uniform sampling would
    // put about one per cent of its draws there; log-uniform puts a third.
    const below = pts.filter((v) => v < 0.01).length;
    expect(below).toBeGreaterThan(8);
    expect(below).toBeLessThan(20);
  });
});

describe('standardise', () => {
  it('centres, scales, and drops what never varied', () => {
    const m = matrix(
      [
        [1, 5, 2],
        [3, 5, 4],
        [5, 5, 6],
      ],
      ['moves', 'constant', 'alsoMoves'],
    );
    const z = standardise(m);
    expect(z.names).toEqual(['moves', 'alsoMoves']);
    let sum = 0;
    let sq = 0;
    for (let i = 0; i < z.rows; i++) {
      sum += z.data[i * z.cols];
      sq += z.data[i * z.cols] ** 2;
    }
    expect(sum).toBeCloseTo(0, 12);
    expect(sq / z.rows).toBeCloseTo(1, 12);
  });
});

describe('pca', () => {
  it('finds a known direction in a plane', () => {
    // Everything on the line y = x, so the first component is (1,1)/sqrt(2)
    // and it accounts for all of the variance.
    const rows: number[][] = [];
    for (let i = -10; i <= 10; i++) rows.push([i, i]);
    const c = pca(standardise(matrix(rows, ['a', 'b'])), 2);
    expect(Math.abs(c[0].loading[0])).toBeCloseTo(Math.SQRT1_2, 6);
    expect(Math.abs(c[0].loading[1])).toBeCloseTo(Math.SQRT1_2, 6);
    expect(c[0].loading[0] * c[0].loading[1]).toBeGreaterThan(0);
    expect(c[0].explained).toBeCloseTo(1, 6);
  });

  it('separates two independent groups of correlated columns', () => {
    /*
     * One group gets *three* columns and the other two, on purpose. Scaling
     * the factors instead would not work: `standardise` puts every column at
     * unit variance, so a difference in factor amplitude is erased before PCA
     * sees it, and two equally strong factors leave the leading eigenvalues
     * equal — any rotation within their plane is then an equally valid answer
     * and the groups come back mixed. That is a property of the method rather
     * than a bug, and a real caveat for the pipeline: equally strong outcome
     * factors will not be separated by this pass, which is part of why the PLS
     * and clustering passes exist beside it.
     */
    const r = lcg(11);
    const rows: number[][] = [];
    for (let i = 0; i < 400; i++) {
      const f1 = gauss(r);
      const f2 = gauss(r);
      rows.push([
        f1 + 0.1 * gauss(r), f1 + 0.1 * gauss(r), f1 + 0.1 * gauss(r),
        f2 + 0.1 * gauss(r), f2 + 0.1 * gauss(r),
      ]);
    }
    const c = pca(standardise(matrix(rows, ['a1', 'a2', 'a3', 'b1', 'b2'])), 5);
    expect(c[0].explained + c[1].explained).toBeGreaterThan(0.9);
    // Each of the first two components belongs to one group and not the other.
    for (const comp of c.slice(0, 2)) {
      const groupA = (Math.abs(comp.loading[0]) + Math.abs(comp.loading[1]) + Math.abs(comp.loading[2])) / 3;
      const groupB = (Math.abs(comp.loading[3]) + Math.abs(comp.loading[4])) / 2;
      expect(Math.max(groupA, groupB) / Math.min(groupA, groupB)).toBeGreaterThan(5);
    }
  });

  it('finds nothing in independent noise, which is the important case', () => {
    // Uncorrelated columns have no preferred direction; a method that reports
    // one here would invent structure in every library it was pointed at.
    const r = lcg(5);
    const rows: number[][] = [];
    for (let i = 0; i < 600; i++) rows.push([gauss(r), gauss(r), gauss(r), gauss(r), gauss(r)]);
    const c = pca(standardise(matrix(rows, ['a', 'b', 'c', 'd', 'e'])), 5);
    // Five equal directions would be 0.2 each; nothing should stand far out.
    expect(c[0].explained).toBeLessThan(0.33);
  });
});

describe('pls', () => {
  it('picks out which parameter drives which outcome', () => {
    /*
     * Two parameters and two outcomes: `p0` drives `y0`, `p1` drives `y1`, and
     * `p2` drives nothing. This is the question the pipeline exists to answer,
     * so it is asked here with the answer known.
     */
    const r = lcg(23);
    const px: number[][] = [];
    const py: number[][] = [];
    for (let i = 0; i < 300; i++) {
      const p0 = gauss(r);
      const p1 = gauss(r);
      const p2 = gauss(r);
      px.push([p0, p1, p2]);
      py.push([2 * p0 + 0.3 * gauss(r), -1.5 * p1 + 0.3 * gauss(r)]);
    }
    const X = standardise(matrix(px, ['drivesY0', 'drivesY1', 'inert']));
    const Y = standardise(matrix(py, ['y0', 'y1']));
    const comps = pls(X, Y, 2);
    expect(comps.length).toBeGreaterThanOrEqual(2);
    for (const c of comps.slice(0, 2)) {
      // Whichever outcome this component is about, the matching parameter
      // carries it and the inert one does not.
      const which = Math.abs(c.yLoading[0]) > Math.abs(c.yLoading[1]) ? 0 : 1;
      expect(Math.abs(c.xLoading[which])).toBeGreaterThan(Math.abs(c.xLoading[2]) * 3);
      expect(Math.abs(c.correlation)).toBeGreaterThan(0.7);
    }
  });
});

describe('kmeans', () => {
  it('recovers two well-separated groups', () => {
    const r = lcg(31);
    const rows: number[][] = [];
    for (let i = 0; i < 100; i++) rows.push([gauss(r) * 0.2 - 4, gauss(r) * 0.2]);
    for (let i = 0; i < 100; i++) rows.push([gauss(r) * 0.2 + 4, gauss(r) * 0.2]);
    const k = kmeans(matrix(rows, ['x', 'y']), 2, 9);
    const firstHalf = new Set(k.label.slice(0, 100));
    const secondHalf = new Set(k.label.slice(100));
    expect(firstHalf.size).toBe(1);
    expect(secondHalf.size).toBe(1);
    expect([...firstHalf][0]).not.toBe([...secondHalf][0]);
  });
});

describe('conditional effects', () => {
  it('finds an effect that exists in half the space and not the other half', () => {
    /*
     * The failure mode no linear method can express, and the one this pond
     * keeps producing: `driver` moves `y` only where `gate` is high. Its
     * *marginal* correlation is about half its conditional one, which reads as
     * a weak main effect and says nothing about the conditionality — which is
     * the wrong conclusion, not merely an imprecise one.
     */
    const r = lcg(41);
    const px: number[][] = [];
    const py: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const driver = gauss(r);
      const gate = gauss(r);
      const inert = gauss(r);
      px.push([driver, gate, inert]);
      py.push([(gate > 0 ? 3 * driver : 0) + 0.3 * gauss(r)]);
    }
    const X = standardise(matrix(px, ['driver', 'gate', 'inert']));
    const Y = standardise(matrix(py, ['y']));

    const marginal = Math.abs(correlate(
      Array.from({ length: X.rows }, (_, i) => X.data[i * X.cols]),
      Array.from({ length: Y.rows }, (_, i) => Y.data[i * Y.cols]),
    ));

    /*
     * An interaction is symmetric: "driver matters only where gate is high"
     * and "gate matters only where driver is large" are the same fact seen
     * from either end, and the screen reports both. What must be true is that
     * the strongest swing is between that pair and does not involve the inert
     * parameter.
     */
    const top = conditionalEffects(X, Y)[0];
    expect([top.driver, top.condition].sort()).toEqual(['driver', 'gate']);
    expect(top.swing).toBeGreaterThan(0.6);
    /*
     * Read from the `gate | driver` end the two halves come out equal and
     * opposite rather than present and absent — `y` tracks `gate` upward where
     * `driver` is large and downward where it is small. So the swing is the
     * statistic that carries the interaction, and a difference of magnitudes
     * is not: it is near zero in that orientation while the effect is as real
     * as ever.
     */
    // And the `driver | gate` orientation, where it is present-versus-absent.
    const oriented = conditionalEffects(X, Y).find((e) => e.driver === 'driver' && e.condition === 'gate')!;
    expect(Math.abs(oriented.high) - Math.abs(oriented.low)).toBeGreaterThan(0.5);
    // The marginal view is a clear understatement of where the effect is real,
    // and carries no hint that it is conditional at all — which is the wrong
    // conclusion rather than merely an imprecise one.
    expect(marginal).toBeLessThan(Math.abs(oriented.high) - 0.15);
  });

  it('reports no swing when an effect is the same everywhere', () => {
    // The negative control. A plain main effect must not be dressed up as an
    // interaction, or every screen returns a page of them.
    const r = lcg(43);
    const px: number[][] = [];
    const py: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const driver = gauss(r);
      const other = gauss(r);
      px.push([driver, other]);
      py.push([2 * driver + 0.3 * gauss(r)]);
    }
    const found = conditionalEffects(standardise(matrix(px, ['driver', 'other'])), standardise(matrix(py, ['y'])));
    expect(found[0].swing).toBeLessThan(0.25);
  });
});
