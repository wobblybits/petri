import { describe, expect, it } from 'vitest';
import { defaultParams, type Params } from '../params.ts';
import { PondDb } from './db.ts';
import { exploreLibrary, groupConfounds, renderExplore } from './explore-report.ts';
import { matrix, standardise } from './explore.ts';

/*
 * The report is run over a library whose structure was planted by hand, so
 * every claim below is one that is true of the data by construction. A
 * descriptive tool has no ground truth in the wild — this is the only place
 * it ever will.
 */

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

interface Planted {
  /** `excreteRate`, the driver. */
  excrete: number;
  /** `groundPatches`, the condition. */
  patches: number;
  finished?: boolean;
  sweep?: string | null;
}

/**
 * A library where `excreteRate` drives `bodies` — but only in a patchy dish.
 * On a uniform dish it does nothing. That is the interaction, and only
 * `conditionalEffects` can see it.
 */
function library(rows: Planted[], tweak?: (p: Params, i: number) => void): PondDb {
  const db = new PondDb(':memory:');
  const r = lcg(7);
  rows.forEach((p, i) => {
    const params = defaultParams();
    params.excreteRate = p.excrete;
    params.groundPatches = p.patches;
    tweak?.(params, i);
    const id = db.startRun({
      seed: 1 + (i % 3),
      seconds: 120,
      dt: 1 / 60,
      world: { w: 1600, h: 1200 },
      fieldCells: 256,
      preset: 'soup',
      soupCount: 400,
      parentRun: null,
      params,
      commit: null,
      note: null,
      sweep: p.sweep ?? null,
      point: null,
    });
    const patchy = p.patches > 0;
    for (let k = 1; k <= 3; k++) {
      const t = k * 40;
      const bodies = 400 + (patchy ? 600 * p.excrete : 0) + 20 * r();
      db.addSample(id, {
        t,
        bodies,
        wires: bodies * 1.4,
        lines: 40,
        bornMean: 3 + 4 * p.excrete,
        bornMax: 12,
        spawned: 0,
        born: 100 * k,
        died: 50 * k,
        commutes: 200 * k,
        erases: 10,
        annihilations: 5,
        latches: 300 * k,
        snaps: 2,
        free: 900,
        ground: 500 - 100 * p.patches + 5 * r(),
        escrow: 3, gut: 0,
        meanExtra: 2 + r(),
        canPay: 0.8,
        ppWires: 100,
        conDupWires: 200,
        commuteShare: 0.4,
        commuteChance: 0.3,
        commuteEdge: 0.1,
        matrixDrift: 0.02 * k,
        diversity: {
          linesEffective: 12 + 3 * r(),
          netsEffective: 30 + 5 * r(),
          netFst: 0.2 + 0.1 * r(),
          lineFst: 0.15,
          varianceDrifted: 0.4,
          signalTotal: 10 + 20 * p.excrete,
          forageRatio: 1 + 0.5 * p.patches,
          fullMean: 0.5,
          demandMean: 0.3,
          signalP90: 1,
        },
      });
    }
    if (p.finished !== false) db.finishRun(id, 7200, 1000, {});
  });
  return db;
}

const plan = (n: number): Planted[] => {
  const r = lcg(3);
  return Array.from({ length: n }, (_, i) => ({
    excrete: Math.round(r() * 1000) / 1000,
    patches: i % 2 === 0 ? 0 : 6,
  }));
};

describe('exploring a library', () => {
  it('says nothing at all when the library is too thin', () => {
    const db = library(plan(4));
    try {
      const r = exploreLibrary(db);
      expect(r.components).toHaveLength(0);
      expect(r.notes.join(' ')).toMatch(/nothing to say/);
      // And it renders rather than throwing, because that is what the CLI does.
      expect(renderExplore(r)).toMatch(/4 runs/);
    } finally {
      db.close();
    }
  });

  it('reads only finished runs, and only the sweep asked for', () => {
    const rows = plan(40);
    rows.forEach((p, i) => {
      p.sweep = i < 20 ? 'a' : 'b';
      if (i === 0) p.finished = false;
    });
    const db = library(rows);
    try {
      expect(exploreLibrary(db).runs).toBe(39);
      expect(exploreLibrary(db, { sweep: 'a' }).runs).toBe(19);
      expect(exploreLibrary(db, { sweep: 'b' }).runs).toBe(20);
    } finally {
      db.close();
    }
  });

  it('finds the planted driver, and the condition that gates it', () => {
    const db = library(plan(60));
    try {
      const r = exploreLibrary(db, { clusters: 2 });
      expect(r.runs).toBe(60);
      // Only two parameters were ever moved; everything else is constant and
      // must be dropped, or the loadings are diluted by 78 columns of nothing.
      expect(r.paramNames).toEqual(['excreteRate', 'groundPatches']);

      // PLS: the parameters reach the outcomes at all.
      expect(r.cross[0].correlation).toBeGreaterThan(0.5);
      const named = r.cross.flatMap((c) => c.params.map((p) => p.name));
      expect(named).toContain('excreteRate');

      // The interaction, which is the only thing here PCA cannot see.
      const gated = r.conditionals.find(
        (c) => c.driver === 'excreteRate' && c.condition === 'groundPatches' && c.outcome === 'bodies',
      );
      expect(gated).toBeDefined();
      // Drives bodies where there are patches, does nothing where there are not.
      expect(Math.abs(gated!.high)).toBeGreaterThan(0.8);
      expect(Math.abs(gated!.low)).toBeLessThan(0.3);
    } finally {
      db.close();
    }
  });

  it('splits the two dishes into regimes and names what made them', () => {
    const db = library(plan(60));
    try {
      const r = exploreLibrary(db, { clusters: 2 });
      expect(r.regimes).toHaveLength(2);
      // `ground` was planted 600 apart between the dishes, so the split is on
      // it, and `groundPatches` is what the report should hand back as cause.
      for (const g of r.regimes) {
        expect(g.outcomes.map((o) => o.name)).toContain('ground');
        expect(g.params.map((p) => p.name)).toContain('groundPatches');
      }
      const [a, b] = r.regimes;
      const gr = (x: typeof a): number => x.outcomes.find((o) => o.name === 'ground')!.value;
      expect(Math.sign(gr(a))).toBe(-Math.sign(gr(b)));
    } finally {
      db.close();
    }
  });

  it('folds a set that never moved apart into one named column', () => {
    // c is b reversed, d is independent: three columns become two.
    const rows = Array.from({ length: 20 }, (_, i) => {
      const b = i % 7;
      return [i, b, -3 * b, (i * 13) % 5];
    });
    const g = groupConfounds(standardise(matrix(rows, ['a', 'b', 'c', 'd'])));
    expect(g.matrix.names).toEqual(['a', 'b~-c', 'd']);
    expect(g.groups).toEqual([['b', '-c']]);
    // The surviving column is the first member's, untouched.
    expect(g.matrix.data[1 * g.matrix.cols + 1]).toBeCloseTo(
      standardise(matrix(rows, ['a', 'b', 'c', 'd'])).data[1 * 4 + 1],
    );
  });

  it('leaves a matrix with nothing confounded exactly as it was', () => {
    const m = standardise(matrix(
      Array.from({ length: 20 }, (_, i) => [i, (i * 7) % 5, (i * 3) % 4]),
      ['a', 'b', 'c'],
    ));
    expect(groupConfounds(m).matrix).toBe(m);
    expect(groupConfounds(m).groups).toEqual([]);
  });

  it('names a parameter that only ever changed between sweeps', () => {
    const rows = plan(40);
    rows.forEach((p, i) => {
      p.sweep = i < 20 ? 'a' : 'b';
    });
    // A fuse set on the second sweep and never reached. It is a label for
    // which sweep a run came from, and everything else that differed between
    // them -- the seed stream, the code, the machine -- loads onto it.
    const db = library(rows, (p, i) => {
      p.maxAgents = i < 20 ? 100000 : 8000;
    });
    try {
      const r = exploreLibrary(db);
      expect(r.notes.join(' ')).toMatch(/labels for which sweep .* maxAgents/);
      // And --drop takes it out entirely.
      expect(exploreLibrary(db, { drop: ['maxAgents'] }).paramNames).not.toContain('maxAgents');
    } finally {
      db.close();
    }
  });

  it('says nothing about sweep labels when there is only one sweep', () => {
    const rows = plan(40);
    for (const p of rows) p.sweep = 'only';
    const db = library(rows, (p, i) => {
      p.maxAgents = i < 20 ? 100000 : 8000;
    });
    try {
      // It varies *within* the sweep here, so it is a parameter like any other.
      expect(exploreLibrary(db).notes.join(' ')).not.toMatch(/labels for which sweep/);
    } finally {
      db.close();
    }
  });

  it('gives each regime as a configuration, not only as z-scores', () => {
    const db = library(plan(60));
    try {
      const r = exploreLibrary(db, { clusters: 2 });
      for (const g of r.regimes) {
        const patches = g.recipe.find((e) => e.name === 'groundPatches');
        expect(patches).toBeDefined();
        // The two dishes were planted at 0 and 6, and the recipe is in those
        // units -- a number to put behind `--set`, not a distance from a mean.
        expect([0, 6]).toContain(patches!.value);
      }
      const values = r.regimes.map((g) => g.recipe.find((e) => e.name === 'groundPatches')!.value);
      expect(new Set(values).size).toBe(2);
      expect(renderExplore(r)).toMatch(/--set groundPatches=/);
    } finally {
      db.close();
    }
  });

  it('warns when two parameters never moved apart', () => {
    const rows = plan(40);
    // Tie the condition to the driver — the shape a sweep makes when a script
    // sets both at every point. Nothing downstream can tell them apart.
    for (const p of rows) p.patches = 8 * p.excrete;
    const db = library(rows);
    try {
      const r = exploreLibrary(db);
      expect(r.notes.join(' ')).toMatch(/moved as one dial/);
      // And they are one column now, not two.
      expect(r.paramNames).toEqual(['excreteRate~groundPatches']);
    } finally {
      db.close();
    }
  });
});
