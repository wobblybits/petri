import { Sim } from '../sim.ts';
import { defaultParams, type Params } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { CH } from '../fields.ts';
import { EMIT, TASTE, CHEM_LEN } from '../chem-layout.ts';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

/*
 * A harness for studying the pond rather than asserting about it.
 *
 * The test suite answers "does this still work"; nothing in the repo answered
 * "what does this dial do to evolution", and the sliders that decide it —
 * `declutter`, `flockAlign`, `spawnInterval`, `swimCost`, the six that ship
 * at zero — were tuned by eye against how the pond looks. The one number that
 * says whether selection is happening at all, `census().bornMean`, has been
 * read a handful of times by hand.
 *
 * So: a trial is a seeded pond run for a fixed number of simulated seconds,
 * sampled on a schedule into a timeline; a sweep is a grid of parameter
 * values crossed with seeds, each run as a trial; the output is rows you can
 * plot, not a pass/fail. Everything runs on the CPU path (`Sim.step`), which
 * is what Node can run — the GPU path is the same simulation minus the
 * genome-on-GPU frame of latency, and a regime tuned here transfers.
 *
 * Run through vitest's `experiments` project so `?raw` imports and the wasm
 * blob resolve: `npm run experiment -- breeding`. Output lands in
 * `experiments/out/`, one JSON per run, plus a summary table on stdout.
 */

/** The same LCG the test suite seeds with, so a trial is a reproducible thing. */
export function seededRandom(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export interface TrialSpec {
  /** Simulated seconds to run. */
  seconds: number;
  /** Frame step; 1/60 is what the app runs at. */
  dt?: number;
  seed: number;
  /** Overrides on `defaultParams()`. */
  params?: Partial<Params>;
  /** Bodies in the opening soup. */
  soupCount: number;
  /** Simulated seconds between samples. */
  sampleEvery?: number;
  /** World size handed to `Sim`; the disk is the field's, this is only the spawn box. */
  world?: { w: number; h: number };
}

export interface Sample {
  t: number;
  bodies: number;
  wires: number;
  /** Distinct surviving founders — the number that separates selection from drift. */
  lines: number;
  bornMean: number;
  bornMax: number;
  /** Cumulative event counts since the trial began. */
  spawned: number;
  born: number;
  died: number;
  commutes: number;
  erases: number;
  annihilations: number;
  latches: number;
  snaps: number;
  /** Energy: stock in bodies, on the ground, in escrow. */
  free: number;
  ground: number;
  escrow: number;
  meanExtra: number;
  /** Fraction of bodies that could pay a rewrite share right now. */
  canPay: number;
  /** Principal-to-principal wires, and how many of those are Con-Dup (a commute waiting to happen). */
  ppWires: number;
  conDupWires: number;
  /** Population mean and spread of the emit bases and taste bases, per channel. */
  emitMean: number[];
  emitSd: number[];
  tasteMean: number[];
  tasteSd: number[];
  /** Mean absolute size of the non-seeded matrix weights: how far the genome has walked. */
  matrixDrift: number;
  /** Heritable scalars, from `census()`. */
  trait: Record<string, { mean: number; sd: number }>;
}

export interface Trial {
  spec: TrialSpec;
  samples: Sample[];
  /** Wall time the trial took, ms. */
  wallMs: number;
}

/** Everything the harness reads off a pond at a sample point. */
export function sampleSim(sim: Sim, t: number): Sample {
  const c = sim.census();
  const n = c.bodies;
  const store = sim.agentStore;
  const CHEM = store.chemAll;
  const emitSum = [0, 0, 0, 0];
  const emitSq = [0, 0, 0, 0];
  const tasteSum = [0, 0, 0, 0];
  const tasteSq = [0, 0, 0, 0];
  let driftSum = 0;
  let extraSum = 0;
  let canPay = 0;
  for (const a of sim.agents.values()) {
    const g = a.slot * CHEM_LEN;
    for (let k = 0; k < 4; k++) {
      const e = CHEM[g + EMIT + k];
      emitSum[k] += e;
      emitSq[k] += e * e;
      const tv = CHEM[g + TASTE + k];
      tasteSum[k] += tv;
      tasteSq[k] += tv * tv;
    }
    // Everything past the taste bases is matrices and heads, dimensionless
    // and (bar the two seeded entries) zero at birth.
    let d = 0;
    for (let k = TASTE + 4; k < CHEM_LEN; k++) d += Math.abs(CHEM[g + k]);
    driftSum += d / (CHEM_LEN - TASTE - 4);
    extraSum += a.extra;
    if (a.extra >= Math.min(1, a.energyCap) - 1e-6) canPay++;
  }
  let pp = 0;
  let conDup = 0;
  for (const w of sim.graph.wires.values()) {
    if (w.a.slot !== 'p' || w.b.slot !== 'p') continue;
    pp++;
    const A = sim.agents.get(w.a.id);
    const B = sim.agents.get(w.b.id);
    if (!A || !B) continue;
    if ((A.kind === 'con' && B.kind === 'dup') || (A.kind === 'dup' && B.kind === 'con')) conDup++;
  }
  const mean = (s: number[]) => s.map((v) => (n > 0 ? v / n : 0));
  const sd = (s: number[], sq: number[]) =>
    s.map((v, k) => (n > 0 ? Math.sqrt(Math.max(0, sq[k] / n - (v / n) * (v / n))) : 0));
  const tally = sim.tally;
  return {
    t,
    bodies: n,
    wires: sim.graph.wires.size,
    lines: c.lines,
    bornMean: c.bornMean,
    bornMax: c.bornMax,
    spawned: tally.spawned,
    born: tally.born,
    died: tally.died,
    commutes: tally.commutes,
    erases: tally.erases,
    annihilations: tally.annihilations,
    latches: tally.latches,
    snaps: tally.snaps,
    free: sim.totalFree(),
    ground: sim.energy.storedTotal(),
    escrow: sim.escrowTotal(),
    meanExtra: n > 0 ? extraSum / n : 0,
    canPay: n > 0 ? canPay / n : 0,
    ppWires: pp,
    conDupWires: conDup,
    emitMean: mean(emitSum),
    emitSd: sd(emitSum, emitSq),
    tasteMean: mean(tasteSum),
    tasteSd: sd(tasteSum, tasteSq),
    matrixDrift: n > 0 ? driftSum / n : 0,
    trait: c.trait,
  };
}

/** Run one seeded pond and sample it on a schedule. */
export function runTrial(spec: TrialSpec): Trial {
  const realRandom = Math.random;
  Math.random = seededRandom(spec.seed);
  const t0 = performance.now();
  try {
    const params: Params = { ...defaultParams(), ...(spec.params ?? {}) };
    params.soupCount = spec.soupCount;
    const world = spec.world ?? { w: 1600, h: 1200 };
    const sim = new Sim(world.w, world.h);
    loadPreset(sim, 'soup', params);
    const dt = spec.dt ?? 1 / 60;
    const every = spec.sampleEvery ?? 5;
    const samples: Sample[] = [];
    let nextSample = 0;
    let t = 0;
    while (t < spec.seconds - 1e-9) {
      if (t >= nextSample - 1e-9) {
        samples.push(sampleSim(sim, t));
        nextSample += every;
      }
      sim.step(dt, params);
      t += dt;
    }
    samples.push(sampleSim(sim, t));
    return { spec, samples, wallMs: performance.now() - t0 };
  } finally {
    Math.random = realRandom;
  }
}

export interface SweepSpec {
  name: string;
  /** Parameter axes; every combination is run. */
  grid: Partial<Record<keyof Params, number[]>>;
  seeds: number[];
  base: Omit<TrialSpec, 'seed' | 'params'> & { params?: Partial<Params> };
}

export interface SweepRow {
  point: Partial<Record<keyof Params, number>>;
  seed: number;
  trial: Trial;
}

function* product(
  axes: [keyof Params, number[]][],
  at = 0,
  acc: Partial<Record<keyof Params, number>> = {},
): Generator<Partial<Record<keyof Params, number>>> {
  if (at === axes.length) {
    yield { ...acc };
    return;
  }
  const [key, values] = axes[at];
  for (const v of values) yield* product(axes, at + 1, { ...acc, [key]: v });
}

/** Every grid point crossed with every seed. */
export function runSweep(spec: SweepSpec, onRow?: (row: SweepRow, done: number, total: number) => void): SweepRow[] {
  const axes = Object.entries(spec.grid) as [keyof Params, number[]][];
  const points = [...product(axes)];
  const total = points.length * spec.seeds.length;
  const rows: SweepRow[] = [];
  for (const point of points) {
    for (const seed of spec.seeds) {
      const trial = runTrial({
        ...spec.base,
        seed,
        params: { ...(spec.base.params ?? {}), ...point },
      });
      const row = { point, seed, trial };
      rows.push(row);
      onRow?.(row, rows.length, total);
    }
  }
  return rows;
}

/** The last sample of each row, keyed by grid point, averaged over seeds. */
export function summarize(rows: SweepRow[], keys: (keyof Sample)[]): string {
  const groups = new Map<string, SweepRow[]>();
  for (const r of rows) {
    const k = JSON.stringify(r.point);
    const g = groups.get(k) ?? [];
    g.push(r);
    groups.set(k, g);
  }
  const head = ['point', 'seeds', ...keys.map(String)];
  const lines = [head.join('\t')];
  for (const [k, g] of groups) {
    const cells = keys.map((key) => {
      let sum = 0;
      for (const r of g) {
        const v = r.trial.samples[r.trial.samples.length - 1][key];
        sum += typeof v === 'number' ? v : 0;
      }
      return (sum / g.length).toFixed(3);
    });
    lines.push([k, String(g.length), ...cells].join('\t'));
  }
  return lines.join('\n');
}

/** Write a sweep to `experiments/out/<name>-<stamp>.json`. Returns the path. */
export function writeSweep(name: string, rows: SweepRow[], root = 'experiments/out'): string {
  mkdirSync(root, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const path = join(root, `${name}-${stamp}.json`);
  writeFileSync(path, JSON.stringify({ name, at: stamp, rows }, null, 1));
  return path;
}

/** Sum of a channel over the field, for a quick "is anything being said". */
export function channelTotal(sim: Sim, ch: number): number {
  const d = sim.fields.data;
  let s = 0;
  for (let k = ch; k < d.length; k += 4) s += d[k];
  return s;
}

export { CH };
