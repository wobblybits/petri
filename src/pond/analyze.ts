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
 * Caveat worth keeping in view: with one axis moving and everything else held,
 * eta-squared over a small grid is a description of *this* sweep and not an
 * estimate of anything. It ranks; it does not test.
 */

/** Metrics read off the final sample of each run. Column name -> label. */
export const METRICS: Record<string, string> = {
  bodies: 'bodies',
  wires: 'wires',
  lines: 'lines',
  lines_effective: 'linesEff',
  line_dominance: 'lineDom',
  nets: 'nets',
  nets_effective: 'netsEff',
  net_fst: 'netFst',
  line_fst: 'lineFst',
  var_drifted: 'varDrift',
  matrix_drift: 'drift',
  born_mean: 'depth',
  commutes: 'commutes',
  commutes_per_latch: 'com/latch',
  commutes_per_latch_window: 'com/latch*',
  commute_edge: 'comEdge',
  can_pay: 'canPay',
  free: 'free',
  ground: 'ground',
};

export interface TrialRow {
  runId: number;
  seed: number;
  point: Record<string, number>;
  values: Record<string, number | null>;
}

/**
 * The last sample of every run in a sweep.
 *
 * The *last*, not the mean over the timeline: these are trajectories, and a
 * pond spends its opening seconds resolving the preset rather than doing
 * anything a dial is responsible for. Anything that needs the shape of the
 * curve should read `sample` directly — it is all there.
 */
export function sweepTrials(db: PondDb, sweep: string): TrialRow[] {
  // Not a column: derived below from the last two samples. See its note.
  const derived = new Set(['commutes_per_latch_window']);
  const cols = Object.keys(METRICS).filter((c) => !derived.has(c));
  const runs = db.db
    .prepare('SELECT id, seed, point FROM run WHERE sweep = ? ORDER BY id')
    .all(sweep) as { id: number; seed: number; point: string | null }[];
  const last = db.db.prepare(
    `SELECT ${cols.join(', ')} FROM sample WHERE run_id = ? ORDER BY t DESC LIMIT 1`,
  );
  /*
   * The last two samples, for the counters that are cumulative.
   *
   * `commutes`, `latches` and the rest of `tally` count from the start of the
   * run, and a soup's opening is a latch storm that drags the ratio below 1
   * for minutes whatever the pond then does — 6,775 latches against 1,095
   * commutes in the first thirty seconds of the first sweep run through this
   * harness. Differencing the last window is what makes the number read as
   * what the pond is doing *now*, which is the only version of it worth
   * ranking a dial by.
   */
  const window = db.db.prepare(
    'SELECT commutes, latches FROM sample WHERE run_id = ? ORDER BY t DESC LIMIT 2',
  );
  const out: TrialRow[] = [];
  for (const r of runs) {
    const row = last.get(r.id) as Record<string, unknown> | undefined;
    if (!row) continue;
    const values: Record<string, number | null> = {};
    for (const c of cols) {
      const v = row[c];
      values[c] = v === null || v === undefined ? null : Number(v);
    }
    const pair = window.all(r.id) as { commutes: number; latches: number }[];
    if (pair.length === 2) {
      const dLatch = Number(pair[0].latches) - Number(pair[1].latches);
      values.commutes_per_latch_window =
        dLatch > 0 ? (Number(pair[0].commutes) - Number(pair[1].commutes)) / dLatch : null;
    } else {
      values.commutes_per_latch_window = null;
    }
    out.push({
      runId: Number(r.id),
      seed: Number(r.seed),
      point: r.point ? (JSON.parse(r.point) as Record<string, number>) : {},
      values,
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
  metric: string;
  /** Share of the metric's variance explained by this axis, 0 to 1. */
  eta2: number;
  /** Seed-to-seed spread within a level, averaged — the noise it has to beat. */
  noise: number;
  /** Level with the lowest mean, and its value; and the highest. */
  low: { level: number; mean: number };
  high: { level: number; mean: number };
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
export function effects(trials: TrialRow[], metrics = Object.keys(METRICS)): Effect[] {
  const axes = new Set<string>();
  for (const t of trials) for (const k of Object.keys(t.point)) axes.add(k);
  const out: Effect[] = [];
  for (const axis of axes) {
    for (const metric of metrics) {
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
      for (const [level, xs] of byLevel) {
        const s = stats(xs);
        ssBetween += xs.length * (s.mean - grand.mean) * (s.mean - grand.mean);
        noise += s.sd;
        if (s.mean < low.mean) low = { level, mean: s.mean };
        if (s.mean > high.mean) high = { level, mean: s.mean };
      }
      const ssTotal = all.reduce((a, b) => a + (b - grand.mean) * (b - grand.mean), 0);
      out.push({
        axis,
        metric,
        eta2: ssTotal > 0 ? ssBetween / ssTotal : 0,
        noise: noise / byLevel.size,
        low,
        high,
        levels: byLevel.size,
        trials: usable.length,
      });
    }
  }
  return out.sort((a, b) => b.eta2 - a.eta2);
}

/** Mean and spread of every metric at every grid point. */
export function pointTable(trials: TrialRow[], metrics = Object.keys(METRICS)): string {
  const groups = new Map<string, TrialRow[]>();
  for (const t of trials) {
    const key = JSON.stringify(t.point);
    const g = groups.get(key);
    if (g) g.push(t);
    else groups.set(key, [t]);
  }
  const head = ['point', 'n', ...metrics.map((m) => METRICS[m] ?? m)];
  const rows: string[][] = [];
  for (const [key, g] of groups) {
    const cells = metrics.map((m) => {
      const xs = g.map((t) => t.values[m]).filter((v): v is number => v !== null && Number.isFinite(v));
      if (xs.length === 0) return '-';
      const s = stats(xs);
      return xs.length > 1 ? `${fmt(s.mean)}±${fmt(s.sd)}` : fmt(s.mean);
    });
    rows.push([shortPoint(key), String(g.length), ...cells]);
  }
  return table(head, rows);
}

/** `{"a":1,"b":2}` as `a=1 b=2`, which is what fits in a column. */
function shortPoint(key: string): string {
  const p = JSON.parse(key) as Record<string, number>;
  const parts = Object.entries(p).map(([k, v]) => `${k}=${v}`);
  return parts.length === 0 ? '(base)' : parts.join(' ');
}

function fmt(n: number): string {
  if (!Number.isFinite(n)) return '-';
  if (Number.isInteger(n)) return String(n);
  const a = Math.abs(n);
  return a >= 100 ? n.toFixed(0) : a >= 1 ? n.toFixed(2) : n.toFixed(4);
}

export function effectTable(list: Effect[], limit = 25): string {
  const head = ['axis', 'metric', 'eta2', 'noise', 'low', '->', 'high', 'n'];
  const rows = list
    .slice(0, limit)
    .map((e) => [
      e.axis,
      METRICS[e.metric] ?? e.metric,
      fmt(e.eta2),
      fmt(e.noise),
      `${fmt(e.low.level)}: ${fmt(e.low.mean)}`,
      '->',
      `${fmt(e.high.level)}: ${fmt(e.high.mean)}`,
      String(e.trials),
    ]);
  return table(head, rows);
}

function table(head: string[], rows: string[][]): string {
  const widths = head.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? '').length)));
  const line = (r: string[]) => r.map((c, i) => (c ?? '').padEnd(widths[i])).join('  ').trimEnd();
  return [line(head), line(widths.map((w) => '-'.repeat(w))), ...rows.map(line)].join('\n');
}
