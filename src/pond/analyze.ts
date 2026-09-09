import type { PondDb } from './db.ts';

/*
 * The reading half of a sweep.
 *
 * A sweep produces a table of numbers, and a table of numbers is not
 * knowledge. The question a dial is being asked is always the same shape —
 * *does moving this change that, by more than the seeds disagree among
 * themselves* — and answering it by eye across a grid is how a project ends up
 * with tuning folklore instead of findings.
 *
 * So the summary here is two things and not one. The per-point table says what
 * happened. The **effect table** ranks each axis by how much of the variation
 * in a metric it explains, against the seed-to-seed noise it has to beat.
 *
 * The statistic is eta-squared: the share of a metric's total variance that
 * lies between an axis's levels rather than within them. It is bounded in
 * [0, 1], needs no distributional assumption, and has exactly the reading you
 * want — 0.02 means the dial is doing nothing you could see over three seeds,
 * 0.6 means it is most of what is happening. The same decomposition
 * `measure.ts` uses for `netFst`, applied to a parameter instead of a net.
 *
 * Two caveats the table now prints rather than leaving to the reader. With
 * one axis moving and everything else held, eta-squared over a small grid
 * describes *this* sweep and estimates nothing — it ranks; it does not test.
 * And it is a ratio of variances, so with one seed a level it is whatever the
 * two draws happened to be: a 25-second smoke run once reported 0.91 for a
 * dial whose three-seed version read 0.21. A row with fewer than three seeds
 * a level is marked `thin` whatever its eta-squared says, and a row under
 * 0.3 is marked `unresolved` with the seed count the observed spread would
 * need. See `docs/experiments.md` §1.D and §5.
 *
 * ## Summaries
 *
 * A run is a timeline and a trial is one number per metric, so every metric
 * has to be folded, and the fold is part of what the number means. The last
 * sample is right for a standing measure and wrong for anything that happens
 * and then stops: foraging on a patchy dish peaks at 1.6-1.9x chance ground
 * in the first minute and is back at parity by the close, so `last` says "no
 * foraging" from a run that plainly foraged. Cumulative counters — commutes,
 * latches, deaths — are only readable as a rate over a window, because the
 * opening latch storm drags every cumulative ratio down for minutes.
 *
 * So each metric carries a default fold (`peak` for foraging, `window` for
 * counters, `last` for most things) and any fold can be asked for by name:
 * `net_fst@slope`, `bodies@trough`. `mean` and `slope` skip the warm-up.
 */

export type Summary = 'last' | 'peak' | 'trough' | 'mean' | 'slope' | 'window';
export const SUMMARIES: readonly Summary[] = ['last', 'peak', 'trough', 'mean', 'slope', 'window'];

/**
 * Simulated seconds `mean` and `slope` ignore at the start of a run.
 *
 * A preset drops its whole population in as founders and the first half
 * minute is a latch storm; the README's rule is that under a couple of
 * minutes is warm-up. Sixty seconds is the compromise that leaves a
 * three-minute foraging run with two minutes of signal.
 */
export const DEFAULT_WARMUP = 60;

export interface MetricSpec {
  /** Column heading. */
  label: string;
  /** Where the number lives in a `sample` row. */
  source:
    | { column: string }
    | { json: string }
    | { derived: 'commutes_per_latch' | 'sense_read_p90' };
  /** How the timeline folds to one number, unless the caller says otherwise. */
  summary: Summary;
}

const col = (column: string, label: string, summary: Summary = 'last'): MetricSpec => ({
  label,
  source: { column },
  summary,
});
const js = (path: string, label: string, summary: Summary = 'last'): MetricSpec => ({
  label,
  source: { json: path },
  summary,
});

/**
 * Everything `analyze` can rank.
 *
 * Columns are the `sample` table's; `json` paths reach into the whole
 * `Sample` stored beside them, so a measure added to `measureDiversity` is
 * rankable without a migration. `docs/experiments.md` §3 is the catalogue —
 * what each says, its seed-to-seed noise, and its caveats.
 */
export const METRICS: Record<string, MetricSpec> = {
  bodies: col('bodies', 'bodies'),
  wires: col('wires', 'wires'),
  lines: col('lines', 'lines'),
  lines_effective: col('lines_effective', 'linesEff'),
  line_dominance: col('line_dominance', 'lineDom'),
  nets: col('nets', 'nets'),
  nets_effective: col('nets_effective', 'netsEff'),
  net_dominance: col('net_dominance', 'netDom'),
  net_fst: col('net_fst', 'netFst'),
  line_fst: col('line_fst', 'lineFst'),
  var_drifted: col('var_drifted', 'varDrift'),
  matrix_drift: col('matrix_drift', 'drift'),
  born_mean: col('born_mean', 'depth'),
  born_max: col('born_max', 'depthMax'),
  // Cumulative since the run began. Read as a rate over the last sample
  // interval, or ask for `@last` to get the total.
  commutes: col('commutes', 'commutes/s', 'window'),
  latches: col('latches', 'latches/s', 'window'),
  died: col('died', 'died/s', 'window'),
  born: col('born', 'born/s', 'window'),
  spawned: col('spawned', 'spawned/s', 'window'),
  commutes_per_latch: { label: 'com/latch', source: { derived: 'commutes_per_latch' }, summary: 'window' },
  commute_edge: col('commute_edge', 'comEdge'),
  can_pay: col('can_pay', 'canPay'),
  free: col('free', 'free'),
  ground: col('ground', 'ground'),
  mean_extra: col('mean_extra', 'extra'),
  pp_wires: col('pp_wires', 'ppWires'),
  con_dup_wires: col('con_dup_wires', 'conDup'),
  signal_total: col('signal_total', 'signal'),
  // A transient: bodies find the patches while hungry and stop when fed.
  forage_ratio: col('forage_ratio', 'forage', 'peak'),
  // Engagement gauges. See `measureDiversity`.
  demand_mean: js('$.diversity.demandMean', 'demand', 'peak'),
  full_mean: js('$.diversity.fullMean', 'full'),
  signal_p90: js('$.diversity.signalP90', 'sigP90'),
  sense_read_p90: { label: 'senseRead', source: { derived: 'sense_read_p90' }, summary: 'last' },
  // Three named genes: the seeded foraging pathway and the hunger memory.
  locus_demand_h0: js('$.diversity.loci.wDemandH0', 'Wx[0][DEM]'),
  locus_self_00: js('$.diversity.loci.wSelf00', 'Wh[0][0]'),
  locus_food_h0: js('$.diversity.loci.tFoodH0', 'T[food][0]'),
  // The larval window. Read `latch_p50` against `tank_life`, not on its own.
  latch_p50: js('$.diversity.latchP50', 'latchP50'),
  latch_p90: js('$.diversity.latchP90', 'latchP90'),
  loneliness: js('$.diversity.loneliness', 'alone'),
  tank_life: js('$.diversity.tankLife', 'tankLife'),
};

/** `net_fst@slope` -> the metric and the fold it asks for. */
export function parseMetric(key: string): { metric: string; summary: Summary } {
  const at = key.indexOf('@');
  const metric = at < 0 ? key : key.slice(0, at);
  const spec = METRICS[metric];
  if (!spec) throw new Error(`pond: unknown metric ${JSON.stringify(metric)}`);
  if (at < 0) return { metric, summary: spec.summary };
  const summary = key.slice(at + 1) as Summary;
  if (!SUMMARIES.includes(summary)) {
    throw new Error(`pond: unknown summary ${JSON.stringify(summary)} in ${key}; one of ${SUMMARIES.join(', ')}`);
  }
  return { metric, summary };
}

/** The key for a metric at a fold, shortest form: the bare name at its default. */
export function metricKey(metric: string, summary?: Summary): string {
  const spec = METRICS[metric];
  if (!spec) throw new Error(`pond: unknown metric ${JSON.stringify(metric)}`);
  return summary === undefined || summary === spec.summary ? metric : `${metric}@${summary}`;
}

/**
 * Column heading for a key: the metric's label, plus the fold when it is not
 * the default. A key that is not a registered metric is its own label, so a
 * table over hand-built rows still prints.
 */
export function metricLabel(key: string): string {
  const at = key.indexOf('@');
  const metric = at < 0 ? key : key.slice(0, at);
  const spec = METRICS[metric];
  if (!spec) return key;
  const { summary } = parseMetric(key);
  return summary === spec.summary ? spec.label : `${spec.label}@${summary}`;
}

/** One metric's timeline over a run, nulls already dropped. */
export interface Series {
  t: number[];
  v: number[];
}

/** Fold a timeline to one number. Null when the fold has nothing to stand on. */
export function fold(series: Series, summary: Summary, warmup = DEFAULT_WARMUP): number | null {
  const n = series.t.length;
  if (n === 0) return null;
  // Samples after the opening, or all of them if the run is that short.
  const after = (from: number): number[] => {
    const idx: number[] = [];
    for (let i = 0; i < n; i++) if (series.t[i] >= from) idx.push(i);
    return idx.length > 0 ? idx : Array.from({ length: n }, (_, i) => i);
  };
  switch (summary) {
    case 'last':
      return series.v[n - 1];
    case 'peak': {
      let m = -Infinity;
      for (const i of after(Number.MIN_VALUE)) if (series.v[i] > m) m = series.v[i];
      return Number.isFinite(m) ? m : null;
    }
    case 'trough': {
      let m = Infinity;
      for (const i of after(Number.MIN_VALUE)) if (series.v[i] < m) m = series.v[i];
      return Number.isFinite(m) ? m : null;
    }
    case 'mean': {
      const idx = after(warmup);
      let s = 0;
      for (const i of idx) s += series.v[i];
      return s / idx.length;
    }
    case 'slope': {
      const idx = after(warmup);
      if (idx.length < 2) return null;
      let st = 0;
      let sv = 0;
      for (const i of idx) {
        st += series.t[i];
        sv += series.v[i];
      }
      const mt = st / idx.length;
      const mv = sv / idx.length;
      let num = 0;
      let den = 0;
      for (const i of idx) {
        num += (series.t[i] - mt) * (series.v[i] - mv);
        den += (series.t[i] - mt) * (series.t[i] - mt);
      }
      // Per simulated minute, which is the unit a ten-minute run reads in.
      return den > 0 ? (num / den) * 60 : null;
    }
    case 'window': {
      if (n < 2) return null;
      const dt = series.t[n - 1] - series.t[n - 2];
      return dt > 0 ? (series.v[n - 1] - series.v[n - 2]) / dt : null;
    }
  }
}

export interface TrialRow {
  runId: number;
  seed: number;
  point: Record<string, number>;
  /** Keyed by the metric key as asked for: `bodies`, `forage_ratio@last`. */
  values: Record<string, number | null>;
  /** The run's whole `Params`, for saying what was held. Absent on a row built by hand. */
  params?: Record<string, number>;
}

export interface TrialOptions {
  /** Metric keys, each optionally `@summary`. Default: every metric at its default fold. */
  metrics?: string[];
  warmup?: number;
}

function jsonPath(obj: unknown, path: string): unknown {
  const parts = path.split('.');
  let cur: unknown = obj;
  for (const p of parts) {
    if (p === '$') continue;
    if (cur === null || typeof cur !== 'object') return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

/**
 * Every run in a sweep, folded to one row of numbers.
 *
 * Reads each run's whole timeline once and folds it per metric. The fold is
 * part of the answer — see the header — so the same run can appear with
 * `forage_ratio` (its peak) and `forage_ratio@last` (its close) side by side.
 */
export function sweepTrials(db: PondDb, sweep: string, opts: TrialOptions = {}): TrialRow[] {
  const runs = db.db
    .prepare('SELECT id, seed, point, params FROM run WHERE sweep = ? ORDER BY id')
    .all(sweep) as unknown as RunRow[];
  return foldRuns(db, runs, opts);
}

/**
 * Every finished run in the library, folded the same way.
 *
 * The whole point of one database rather than thirty-eight is that a question
 * asked today can be answered with machine time spent last week, so this does
 * not care which sweep a run belonged to — or whether it belonged to one.
 */
export function libraryTrials(db: PondDb, opts: TrialOptions & { sweep?: string | null } = {}): TrialRow[] {
  const where = opts.sweep ? 'sweep = ? AND ' : '';
  const runs = db.db
    .prepare(`SELECT id, seed, point, params, seconds, soup_count, field_cells FROM run
              WHERE ${where}finished_at IS NOT NULL ORDER BY id`)
    .all(...(opts.sweep ? [opts.sweep] : [])) as unknown as RunRow[];
  return foldRuns(db, runs, opts);
}

interface RunRow {
  id: number;
  seed: number;
  point: string | null;
  params: string;
  /*
   * Present only from `libraryTrials`. They are the *setup*, not parameters,
   * and leaving them out is the confound the exploration pipeline exists to
   * avoid: runs in a library differ in duration, founder count and dish
   * resolution, and a regression blind to that credits whichever parameter
   * moved alongside them.
   */
  seconds?: number;
  soup_count?: number;
  field_cells?: number;
}

function foldRuns(db: PondDb, runs: RunRow[], opts: TrialOptions): TrialRow[] {
  const keys = opts.metrics ?? Object.keys(METRICS);
  const warmup = opts.warmup ?? DEFAULT_WARMUP;
  const asked = keys.map((k) => ({ key: k, ...parseMetric(k) }));
  const samplesOf = db.db.prepare('SELECT * FROM sample WHERE run_id = ? ORDER BY t');
  const out: TrialRow[] = [];
  for (const r of runs) {
    const rows = samplesOf.all(r.id) as Record<string, unknown>[];
    if (rows.length === 0) continue;
    const params = JSON.parse(r.params) as Record<string, number>;
    let parsed: (unknown | null)[] | null = null;
    const json = (i: number): unknown => {
      if (!parsed) parsed = rows.map((row) => (typeof row.json === 'string' ? JSON.parse(row.json) : null));
      return parsed[i];
    };
    const series = (get: (i: number) => unknown): Series => {
      const s: Series = { t: [], v: [] };
      for (let i = 0; i < rows.length; i++) {
        const v = get(i);
        if (typeof v === 'number' && Number.isFinite(v)) {
          s.t.push(Number(rows[i].t));
          s.v.push(v);
        } else if (typeof v === 'bigint') {
          s.t.push(Number(rows[i].t));
          s.v.push(Number(v));
        }
      }
      return s;
    };
    const column = (name: string): Series =>
      series((i) => {
        const v = rows[i][name];
        return v === null || v === undefined ? null : Number(v);
      });
    const values: Record<string, number | null> = {};
    for (const a of asked) {
      const spec = METRICS[a.metric];
      const src = spec.source;
      if ('column' in src) {
        values[a.key] = fold(column(src.column), a.summary, warmup);
      } else if ('json' in src) {
        values[a.key] = fold(series((i) => jsonPath(json(i), src.json)), a.summary, warmup);
      } else if (src.derived === 'commutes_per_latch') {
        /*
         * A ratio of two windows, not the window of a ratio. The cumulative
         * ratio is a column and any other fold reads it; the windowed form
         * is what the number is for — the opening latch storm drags the
         * cumulative below 1 for minutes whatever the pond then does.
         */
        if (a.summary === 'window') {
          const c = fold(column('commutes'), 'window', warmup);
          const l = fold(column('latches'), 'window', warmup);
          values[a.key] = c !== null && l !== null && l > 0 ? c / l : null;
        } else {
          values[a.key] = fold(column('commutes_per_latch'), a.summary, warmup);
        }
      } else {
        // What one unit of signal reads as on the way into `x`, at this run's
        // sense scale: the number that says whether sense genes are in range.
        const p90 = fold(series((i) => jsonPath(json(i), '$.diversity.signalP90')), a.summary, warmup);
        const scale = params.senseScale;
        values[a.key] = p90 !== null && typeof scale === 'number' ? p90 * scale : null;
      }
    }
    if (r.seconds !== undefined) {
      params.runSeconds = Number(r.seconds);
      params.runBodies = Number(r.soup_count);
      params.runFieldCells = Number(r.field_cells);
    }
    out.push({
      runId: Number(r.id),
      seed: Number(r.seed),
      point: r.point ? (JSON.parse(r.point) as Record<string, number>) : {},
      values,
      params,
    });
  }
  return out;
}

function stats(xs: number[]): { n: number; mean: number; sd: number } {
  const n = xs.length;
  if (n === 0) return { n: 0, mean: NaN, sd: NaN };
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const varr = xs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
  return { n, mean, sd: Math.sqrt(varr) };
}

export interface Effect {
  axis: string;
  /** The metric key, as asked for. */
  metric: string;
  /** Share of the metric's variance explained by this axis, 0 to 1. */
  eta2: number;
  /** Seed-to-seed spread within a level, averaged — the noise it has to beat. */
  noise: number;
  /** Level with the lowest mean, and its value; and the highest. */
  low: { level: number; mean: number };
  high: { level: number; mean: number };
  /** Mean at the smallest level value, and at the largest — the direction along the axis. */
  first: { level: number; mean: number };
  last: { level: number; mean: number };
  levels: number;
  trials: number;
}

/**
 * Eta-squared per axis per metric, plus which way the metric moves.
 *
 * Nulls are dropped rather than zeroed: `netFst` is null for a pond with one
 * net, and calling that zero differentiation would be an answer where there
 * is none.
 */
export function effects(trials: TrialRow[], metrics?: string[]): Effect[] {
  const keys = metrics ?? (trials.length > 0 ? Object.keys(trials[0].values) : []);
  const axes = new Set<string>();
  for (const t of trials) for (const k of Object.keys(t.point)) axes.add(k);
  const out: Effect[] = [];
  for (const axis of axes) {
    for (const metric of keys) {
      const usable = trials.filter(
        (t) => t.point[axis] !== undefined && t.values[metric] !== null && Number.isFinite(t.values[metric]),
      );
      if (usable.length < 2) continue;
      const byLevel = new Map<number, number[]>();
      for (const t of usable) {
        const level = t.point[axis];
        const list = byLevel.get(level);
        if (list) list.push(t.values[metric] as number);
        else byLevel.set(level, [t.values[metric] as number]);
      }
      if (byLevel.size < 2) continue;
      const all = usable.map((t) => t.values[metric] as number);
      const grand = stats(all);
      if (!(grand.sd > 0)) continue;
      let ssBetween = 0;
      let noise = 0;
      let low = { level: 0, mean: Infinity };
      let high = { level: 0, mean: -Infinity };
      let first = { level: Infinity, mean: 0 };
      let last = { level: -Infinity, mean: 0 };
      for (const [level, xs] of byLevel) {
        const s = stats(xs);
        ssBetween += xs.length * (s.mean - grand.mean) * (s.mean - grand.mean);
        noise += s.sd;
        if (s.mean < low.mean) low = { level, mean: s.mean };
        if (s.mean > high.mean) high = { level, mean: s.mean };
        if (level < first.level) first = { level, mean: s.mean };
        if (level > last.level) last = { level, mean: s.mean };
      }
      const ssTotal = all.reduce((a, b) => a + (b - grand.mean) * (b - grand.mean), 0);
      out.push({
        axis,
        metric,
        eta2: ssTotal > 0 ? ssBetween / ssTotal : 0,
        noise: noise / byLevel.size,
        low,
        high,
        first,
        last,
        levels: byLevel.size,
        trials: usable.length,
      });
    }
  }
  return out.sort((a, b) => b.eta2 - a.eta2);
}

/** Eta-squared at or above which a row is read as resolved. The README's rule. */
export const RESOLVED_ETA2 = 0.3;
/** Seeds a level below which a row is thin whatever its eta-squared. */
export const THIN_SEEDS = 3;

/**
 * Seeds per level to put a gap of `delta` two standard errors clear of a
 * within-level spread of `noise`, comparing two levels: `8 * (noise/delta)^2`.
 * Null when there is no gap to resolve.
 */
export function seedBudget(noise: number, delta: number): number | null {
  if (!(delta > 0) || !(noise >= 0)) return null;
  if (noise === 0) return 1;
  return Math.ceil((8 * noise * noise) / (delta * delta));
}

/** How to read a row: resolved, thin, or unresolved with a seed estimate. */
export function readEffect(e: Effect): string {
  const perLevel = e.trials / e.levels;
  if (perLevel < THIN_SEEDS) return `thin (${perLevel.toFixed(1)}/level)`;
  if (e.eta2 >= RESOLVED_ETA2) return 'resolved';
  const need = seedBudget(e.noise, Math.abs(e.high.mean - e.low.mean));
  return need === null ? 'unresolved' : `unresolved; ~${need > 999 ? '999+' : need}/level`;
}

export interface TableOptions {
  /** Print an axis's level as a name — arms, say — instead of its number. */
  levelName?: (axis: string, level: number) => string;
}

/** Mean and spread of every metric at every grid point. */
export function pointTable(trials: TrialRow[], metrics?: string[], opts: TableOptions = {}): string {
  const keys = metrics ?? (trials.length > 0 ? Object.keys(trials[0].values) : []);
  const groups = new Map<string, TrialRow[]>();
  for (const t of trials) {
    const key = JSON.stringify(t.point);
    const g = groups.get(key);
    if (g) g.push(t);
    else groups.set(key, [t]);
  }
  const head = ['point', 'n', ...keys.map(metricLabel)];
  const rows: string[][] = [];
  for (const [key, g] of groups) {
    const cells = keys.map((m) => {
      const xs = g.map((t) => t.values[m]).filter((v): v is number => v !== null && Number.isFinite(v));
      if (xs.length === 0) return '-';
      const s = stats(xs);
      return xs.length > 1 ? `${fmt(s.mean)}±${fmt(s.sd)}` : fmt(s.mean);
    });
    rows.push([shortPoint(key, opts.levelName), String(g.length), ...cells]);
  }
  return table(head, rows);
}

/** `{"a":1,"b":2}` as `a=1 b=2`, which is what fits in a column. */
function shortPoint(key: string, levelName?: TableOptions['levelName']): string {
  const p = JSON.parse(key) as Record<string, number>;
  const parts = Object.entries(p).map(([k, v]) => `${k}=${levelName ? levelName(k, v) : v}`);
  return parts.length === 0 ? '(base)' : parts.join(' ');
}

function fmt(n: number): string {
  if (!Number.isFinite(n)) return '-';
  if (Number.isInteger(n)) return String(n);
  const a = Math.abs(n);
  return a >= 100 ? n.toFixed(0) : a >= 1 ? n.toFixed(2) : n.toFixed(4);
}

export function effectTable(list: Effect[], limit = 25, opts: TableOptions = {}): string {
  const name = (axis: string, level: number) => (opts.levelName ? opts.levelName(axis, level) : fmt(level));
  const head = ['axis', 'metric', 'eta2', 'noise', 'low', '->', 'high', 'n', 'read'];
  const rows = list
    .slice(0, limit)
    .map((e) => [
      e.axis,
      metricLabel(e.metric),
      fmt(e.eta2),
      fmt(e.noise),
      `${name(e.axis, e.low.level)}: ${fmt(e.low.mean)}`,
      '->',
      `${name(e.axis, e.high.level)}: ${fmt(e.high.mean)}`,
      String(e.trials),
      readEffect(e),
    ]);
  return table(head, rows);
}

export function table(head: string[], rows: string[][]): string {
  const widths = head.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? '').length)));
  const line = (r: string[]) => r.map((c, i) => (c ?? '').padEnd(widths[i])).join('  ').trimEnd();
  return [line(head), line(widths.map((w) => '-'.repeat(w))), ...rows.map(line)].join('\n');
}
