import { defaultParams, SLIDERS } from '../params.ts';
import { libraryTrials, metricLabel, type TrialRow } from './analyze.ts';
import type { PondDb } from './db.ts';
import {
  conditionalEffects,
  kmeans,
  matrix,
  pca,
  pls,
  standardise,
  varies,
  type Conditional,
  type Matrix,
} from './explore.ts';

/*
 * Four passes over whatever the library holds.
 *
 *   joint PCA   what moves together at all, parameters and outcomes in one
 *               column space, which is the pass that catches the unexpected
 *   PLS         the same question asked across the blocks, so a direction has
 *               to link parameters TO outcomes to score
 *   k-means     clustering the outcomes, then reading back what made them:
 *               the regimes, not the dials
 *   conditional where one dial changes what another means — the only pass
 *               that sees interactions, and interactions are where this
 *               simulation keeps hiding things
 *
 * None of it is causal. It is a map of a library that was not designed as an
 * experiment, and its job is to say which experiment to run.
 */

export interface Loading {
  name: string;
  value: number;
}

export interface ExploreReport {
  runs: number;
  paramNames: string[];
  outcomeNames: string[];
  components: { explained: number; top: Loading[] }[];
  cross: { correlation: number; params: Loading[]; outcomes: Loading[] }[];
  regimes: {
    size: number;
    outcomes: Loading[];
    params: Loading[];
    runs: number[];
    /** The regime's centre in the units you would type, for the dials that define it. */
    recipe: { name: string; value: number }[];
  }[];
  conditionals: Conditional[];
  notes: string[];
}

export interface ExploreOptions {
  sweep?: string | string[] | null;
  clusters?: number;
  components?: number;
  minSwing?: number;
  /** Metric keys, each optionally `@summary`, as `analyze` takes them. */
  metrics?: string[];
  warmup?: number;
  /** Parameter names to leave out of the matrix entirely. */
  drop?: string[];
}

/*
 * `analyze`'s defaults, minus the ones that are each other. A joint PCA over
 * duplicated columns finds the duplication and reports it as structure, which
 * is true and useless: `nets` and `nets_effective`, `lines` and
 * `lines_effective`, would take a whole component to say so.
 */
export const EXPLORE_METRICS: readonly string[] = [
  'bodies',
  'wires',
  'lines_effective',
  'nets_effective',
  'net_fst',
  'line_fst',
  'var_drifted',
  'matrix_drift',
  'born_mean',
  'commutes_per_latch@window',
  'forage_ratio@peak',
  'ground',
  'mean_extra',
  'full_mean',
  'signal_total',
];

const LOADING_FLOOR = 0.15;
/** Above this, two parameters are one parameter. */
const CONFOUND = 0.95;
const REGIME_FLOOR = 0.3;

function top(names: string[], values: number[], limit: number, floor: number): Loading[] {
  return names
    .map((name, j) => ({ name, value: values[j] }))
    .filter((e) => Math.abs(e.value) > floor)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, limit);
}

function columnMean(m: Matrix, rows: number[], j: number): number {
  let s = 0;
  for (const i of rows) s += m.data[i * m.cols + j];
  return s / rows.length;
}

export function exploreLibrary(db: PondDb, opts: ExploreOptions = {}): ExploreReport {
  const metrics = opts.metrics ?? [...EXPLORE_METRICS];
  const trials = libraryTrials(db, { sweep: opts.sweep ?? null, metrics, warmup: opts.warmup });
  const notes: string[] = [];

  /*
   * A run is usable when every metric folded to a number. Dropping the rest
   * whole is the honest move: a matrix with holes patched to the column mean
   * has correlations that were put there by the patching.
   */
  const has = (t: TrialRow, k: string): boolean => t.values[k] !== null && t.values[k] !== undefined;
  /*
   * A library accumulates columns. `measureDiversity` gained `fullMean` after
   * most of these runs were made, and requiring every metric of every run
   * would let the newest column silently veto the entire history — which it
   * did, the first time this was pointed at a real database. So a metric that
   * most of the library lacks is dropped instead of the library, out loud.
   */
  const missing = metrics.map((k) => ({ k, n: trials.filter((t) => !has(t, k)).length }));
  const gone = missing.filter((e) => e.n > trials.length / 2);
  const live = metrics.filter((k) => !gone.some((e) => e.k === k));
  if (gone.length > 0) {
    notes.push(
      `dropped as absent from most of the library: ${gone.map((e) => `${e.k} (missing in ${e.n} of ${trials.length})`).join(', ')}`,
    );
  }
  const keep = trials.filter((t) => live.every((k) => has(t, k)));
  if (keep.length < trials.length) {
    // Say which metric cost the runs, so it can be dropped with `--metric`.
    const cost = missing.filter((e) => e.n > 0 && live.includes(e.k)).sort((a, b) => b.n - a.n);
    notes.push(
      `${trials.length - keep.length} of ${trials.length} runs dropped for missing metrics: ` +
        cost.map((e) => `${e.k} (${e.n})`).join(', '),
    );
    /*
     * And say whether they were a random third. A null is usually a pond that
     * did not do the thing — `commutes_per_latch` is null when nothing
     * latched in the window — so the runs a metric costs are exactly the ones
     * where some parameter pushed the pond into silence. Dropping them
     * quietly restricts the sample to ponds that worked, and every loading
     * below is then conditional on that.
     *
     * Measured once, on `phys1`: one metric cost 36 per cent of the runs, and
     * the first principal component came back with its signs reversed.
     */
    const lost = trials.filter((t) => !live.every((k) => has(t, k)));
    for (const b of biases(keep, lost)) {
      notes.push(
        `the dropped runs were not a random sample: ${b.name} averages ${fmt(b.lostMean)} in them ` +
          `against ${fmt(b.keptMean)} in the rest (${b.d.toFixed(1)} sd apart), ` +
          'so everything below is conditional on that',
      );
    }
  }

  const empty: ExploreReport = {
    runs: keep.length, paramNames: [], outcomeNames: [], components: [], cross: [], regimes: [],
    conditionals: [], notes,
  };
  if (keep.length < 8) {
    notes.push(`${keep.length} usable runs; nothing to say until there are more like 40`);
    return empty;
  }

  const paramKeys = [...Object.keys(defaultParams()), 'runSeconds', 'runBodies', 'runFieldCells']
    .filter((k) => !(opts.drop ?? []).includes(k));
  const P = standardise(matrix(keep.map((t) => paramKeys.map((k) => t.params?.[k] ?? 0)), paramKeys));
  const Y = standardise(matrix(keep.map((t) => live.map((k) => t.values[k] as number)), live.map(metricLabel)));
  if (P.cols === 0) {
    notes.push('no parameter varies across these runs');
    return { ...empty, outcomeNames: Y.names };
  }
  if (Y.cols === 0) {
    notes.push('no outcome varies across these runs');
    return { ...empty, paramNames: P.names };
  }

  /*
   * Fold the confounded sets away before anything reads the matrix, so a
   * finding is never printed once per member of a set that moved as one.
   */
  const collapsed = groupConfounds(P);
  const X = collapsed.matrix;
  for (const g of collapsed.groups) {
    notes.push(`moved as one dial, and nothing here can tell them apart: ${g.join(' ~ ')}`);
  }

  const nComp = Math.max(1, Math.min(opts.components ?? 3, X.cols, Y.cols));
  const joint = standardise(
    matrix(
      keep.map((t, i) => [
        ...Array.from({ length: X.cols }, (_, j) => X.data[i * X.cols + j]),
        ...live.map((k) => t.values[k] as number),
      ]),
      [...X.names, ...Y.names],
    ),
  );
  const components = pca(joint, Math.min(opts.components ?? 3, joint.cols)).map((c) => ({
    explained: c.explained,
    top: top(joint.names, c.loading, 8, LOADING_FLOOR),
  }));
  const cross = pls(X, Y, nComp).map((c) => ({
    correlation: c.correlation,
    params: top(X.names, c.xLoading, 6, LOADING_FLOOR),
    outcomes: top(Y.names, c.yLoading, 6, LOADING_FLOOR),
  }));

  // Cluster what the ponds *became*, then look back at what made them.
  const k = Math.max(2, Math.min(opts.clusters ?? 3, Math.floor(Y.rows / 4)));
  const label = kmeans(Y, k, 17).label;
  const regimes = [];
  for (let c = 0; c < k; c++) {
    const rows = label.map((l, i) => (l === c ? i : -1)).filter((i) => i >= 0);
    if (rows.length === 0) continue;
    /*
     * A regime described only in standard deviations is a description. The
     * median of each defining dial, in its own units, is a configuration --
     * something to put behind `--set` and run at several seeds, which is the
     * only way any of this becomes a finding rather than a picture.
     *
     * Median rather than mean: these are the parameters of a cluster found in
     * outcome space, so their distribution has no reason to be symmetric, and
     * one extreme draw should not move the recipe.
     */
    const params = top(X.names, X.names.map((_, j) => columnMean(X, rows, j)), 6, REGIME_FLOOR);
    const recipe = params.map((e) => {
      const vs = rows.map((i) => keep[i].params?.[e.name] ?? 0).sort((a, b) => a - b);
      return { name: e.name, value: vs[vs.length >> 1] };
    });
    regimes.push({
      size: rows.length,
      outcomes: top(Y.names, Y.names.map((_, j) => columnMean(Y, rows, j)), 6, REGIME_FLOOR),
      params,
      runs: rows.map((i) => keep[i].runId),
      recipe,
    });
  }
  regimes.sort((a, b) => b.size - a.size);

  const minSwing = opts.minSwing ?? 0.3;
  const conditionals = conditionalEffects(X, Y).filter((e) => e.swing >= minSwing).slice(0, 12);

  /*
   * The floor below which a loading is a coin. With p columns and n runs a
   * correlation of about 1/sqrt(n) arises from nothing at all, and a library
   * this thin will hand you a confident-looking component made of noise.
   */
  const noise = 1 / Math.sqrt(keep.length);
  if (noise > LOADING_FLOOR) {
    notes.push(`${keep.length} runs: loadings under ${noise.toFixed(2)} are sampling noise, which is most of what is printed below`);
  }
  if (X.cols > keep.length / 4) {
    notes.push(`${X.cols} parameters varying across ${keep.length} runs — too few runs to separate them; expect confounded loadings`);
  }
  /*
   * A parameter that never varied *within* a sweep, only between sweeps, is
   * not a parameter here — it is a label for which sweep a run came from, and
   * every other thing that differed between those runs loads onto it. It
   * happened immediately: `survey2` capped `maxAgents` at 8000 as a fuse
   * against a runaway config, the fuse never blew (peak population 3788), and
   * pooled with `survey1` the constant showed up carrying interactions.
   */
  const bySweep = new Map<string, TrialRow[]>();
  for (const t of keep) {
    const k = t.sweep ?? '';
    const list = bySweep.get(k);
    if (list) list.push(t);
    else bySweep.set(k, [t]);
  }
  if (bySweep.size > 1) {
    const labels = P.names.filter((n) => {
      if (n.startsWith('run')) return false;
      for (const rows of bySweep.values()) {
        const first = rows[0].params?.[n];
        if (rows.some((t) => t.params?.[n] !== first)) return false;
      }
      return true;
    });
    if (labels.length > 0) {
      notes.push(
        `constant within every sweep and different between them, so these are labels for which sweep a run came from, not parameters — anything else that differed between those runs loads onto them: ${labels.join(', ')} (drop with --drop)`,
      );
    }
  }
  const noRange = P.names.filter((n) => !n.startsWith('run') && !SLIDERS.some((s) => s.key === n));
  if (noRange.length > 0) notes.push(`varied but has no declared slider range, so sampling cannot reach them: ${noRange.join(', ')}`);
  return { runs: keep.length, paramNames: X.names, outcomeNames: Y.names, components, cross, regimes, conditionals, notes };
}

/**
 * Parameters on which the dropped runs differ from the kept ones by half a
 * standard deviation or more — the shape of a sample that is no longer random.
 */
function biases(
  kept: TrialRow[],
  lost: TrialRow[],
): { name: string; keptMean: number; lostMean: number; d: number }[] {
  if (kept.length < 4 || lost.length < 4) return [];
  const out = [];
  const mean = (xs: number[]): number => xs.reduce((p, q) => p + q, 0) / xs.length;
  for (const name of Object.keys(kept[0].params ?? {})) {
    const a = kept.map((t) => t.params?.[name] ?? 0);
    const b = lost.map((t) => t.params?.[name] ?? 0);
    const all = [...a, ...b];
    const m = mean(all);
    const sd = Math.sqrt(all.reduce((p, q) => p + (q - m) * (q - m), 0) / all.length);
    if (!varies(m, sd)) continue;
    const d = Math.abs(mean(a) - mean(b)) / sd;
    if (d >= 0.5) out.push({ name, keptMean: mean(a), lostMean: mean(b), d });
  }
  return out.sort((x, y) => y.d - x.d).slice(0, 4);
}

/**
 * Collapse parameters that never moved independently into one column.
 *
 * A sweep that sets three dials together at every point makes them one dial
 * as far as any of this is concerned, and leaving them separate does not just
 * fail to say so — it prints every finding three times and splits the loading
 * three ways. The group keeps the first member's column and is named for all
 * of them, with a `-` on the ones that ran backwards, so nothing downstream
 * can attribute an effect to a single member of a set that never varied
 * apart.
 */
export function groupConfounds(P: Matrix): { matrix: Matrix; groups: string[][] } {
  const r = (a: number, b: number): number => {
    let s = 0;
    for (let i = 0; i < P.rows; i++) s += P.data[i * P.cols + a] * P.data[i * P.cols + b];
    return s / P.rows;
  };
  const of = new Array<number>(P.cols).fill(-1);
  const heads: number[] = [];
  const names: string[][] = [];
  for (let j = 0; j < P.cols; j++) {
    if (of[j] >= 0) continue;
    of[j] = heads.length;
    names.push([P.names[j]]);
    for (let k = j + 1; k < P.cols; k++) {
      if (of[k] >= 0) continue;
      const c = r(j, k);
      if (Math.abs(c) > CONFOUND) {
        of[k] = heads.length;
        names[heads.length].push(c < 0 ? `-${P.names[k]}` : P.names[k]);
      }
    }
    heads.push(j);
  }
  if (heads.length === P.cols) return { matrix: P, groups: [] };
  const data = new Float64Array(P.rows * heads.length);
  for (let i = 0; i < P.rows; i++) {
    for (let h = 0; h < heads.length; h++) data[i * heads.length + h] = P.data[i * P.cols + heads[h]];
  }
  return {
    matrix: { rows: P.rows, cols: heads.length, data, names: names.map((n) => n.join('~')) },
    groups: names.filter((n) => n.length > 1),
  };
}

/** Enough digits to matter and no more: these are dial settings, not data. */
const round = (v: number): string => {
  if (v === 0) return '0';
  const d = Math.max(0, 3 - Math.floor(Math.log10(Math.abs(v))) - 1);
  return Number(v.toFixed(Math.min(d, 6))).toString();
};

const fmt = (v: number): string => round(v);

const sig = (v: number): string => (v >= 0 ? '+' : '-') + Math.abs(v).toFixed(2);
const list = (l: Loading[], none: string): string =>
  l.length === 0 ? none : l.map((e) => `${e.name}${sig(e.value)}`).join('  ');

export function renderExplore(r: ExploreReport): string {
  const out: string[] = [];
  out.push(`${r.runs} runs   ${r.paramNames.length} parameters varied   ${r.outcomeNames.length} outcomes`);
  for (const n of r.notes) out.push(`  ! ${n}`);
  if (r.paramNames.length === 0 || r.outcomeNames.length === 0) return `${out.join('\n')}\n`;
  out.push(`\nvaried: ${r.paramNames.join(' ')}`);

  out.push('\njoint PCA — parameters and outcomes in one space; what moves together');
  r.components.forEach((c, i) => {
    out.push(`  PC${i + 1}  ${(c.explained * 100).toFixed(0)}% of variance`);
    out.push(`        ${list(c.top, '(nothing above the floor)')}`);
  });

  out.push('\nPLS — the directions that link parameters to outcomes');
  r.cross.forEach((c, i) => {
    out.push(`  C${i + 1}  block correlation ${c.correlation.toFixed(2)}`);
    out.push(`        params:   ${list(c.params, '(none)')}`);
    out.push(`        outcomes: ${list(c.outcomes, '(none)')}`);
  });

  out.push('\nregimes — the kinds of pond this makes, in units of sd from the library mean');
  r.regimes.forEach((g, i) => {
    out.push(`  regime ${i + 1}  ${g.size} runs   e.g. ${g.runs.slice(0, 6).join(' ')}`);
    out.push(`        is:   ${list(g.outcomes, '(the ordinary pond)')}`);
    out.push(`        from: ${list(g.params, '(no parameter distinguishes it — it is seed noise)')}`);
    if (g.recipe.length > 0) {
      out.push(`        run:  ${g.recipe.map((e) => `--set ${e.name}=${round(e.value)}`).join(' ')}`);
    }
  });

  out.push('\nconditional effects — where one dial changes what another means');
  if (r.conditionals.length === 0) out.push('  none above the threshold');
  for (const c of r.conditionals) {
    out.push(
      `  ${c.driver} -> ${c.outcome}:  r=${c.low.toFixed(2)} at low ${c.condition}, ` +
        `r=${c.high.toFixed(2)} at high   (swing ${c.swing.toFixed(2)}, n ${c.nLow}/${c.nHigh})`,
    );
  }
  return `${out.join('\n')}\n`;
}
