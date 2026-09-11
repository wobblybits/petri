import type { Params } from '../params.ts';
import { PondDb } from './db.ts';
import { gitCommit } from './provenance.ts';
import { paramsWith, runPond, type PondRunSpec } from './run.ts';

/*
 * A parameter sweep whose results accumulate in the pond library: a trial is
 * a `run` row with its whole `Params`, its commit, its timeline and its nets;
 * a sweep is a name and a grid point recorded on each. `analyze.ts` is the
 * reading half.
 */

export interface SweepSpec {
  name: string;
  /** Parameter axes. Every combination of every axis is run, once per seed. */
  grid: Record<string, number[]>;
  /** Points sampled from continuous ranges instead of crossed as a grid. Replaces `grid` when present; see `sample.ts`. */
  points?: Record<string, number>[];
  seeds: number[];
  /** Applied to every point, before the grid overrides it. */
  base: Record<string, number>;
  seconds: number;
  dt: number;
  soupCount: number;
  fieldCells: number;
  world: { w: number; h: number };
  sampleEvery: number;
  gpu: 'auto' | 'on' | 'off';
  /** Store nets from each trial, or only the timeline. Off by default: the genomes are hundreds of megabytes. */
  keepNets: boolean;
  note: string | null;
}

export interface SweepPoint {
  point: Record<string, number>;
  seed: number;
  runId: number;
  wallMs: number;
}

/** Every combination of every axis, in a stable order. */
export function gridPoints(grid: Record<string, number[]>): Record<string, number>[] {
  const axes = Object.entries(grid);
  let out: Record<string, number>[] = [{}];
  for (const [key, values] of axes) {
    const next: Record<string, number>[] = [];
    for (const acc of out) for (const v of values) next.push({ ...acc, [key]: v });
    out = next;
  }
  return out;
}

export interface SweepHooks {
  onTrial?: (row: SweepPoint, done: number, total: number) => void;
}

/** Run every point of a sweep into `db`. */
export async function runSweep(
  db: PondDb,
  spec: SweepSpec,
  hooks: SweepHooks = {},
): Promise<SweepPoint[]> {
  const points = spec.points ?? gridPoints(spec.grid);
  const total = points.length * spec.seeds.length;
  const rows: SweepPoint[] = [];
  const commit = gitCommit();
  for (const point of points) {
    for (const seed of spec.seeds) {
      const overrides = new Map<string, number>(Object.entries({ ...spec.base, ...point }));
      const params: Params = paramsWith(overrides);
      const runSpec: PondRunSpec = {
        seconds: spec.seconds,
        dt: spec.dt,
        seed,
        world: spec.world,
        fieldCells: spec.fieldCells,
        params,
        soupCount: spec.soupCount,
        seeds: [],
        sampleEvery: spec.sampleEvery,
        harvestEvery: 0,
        minBodies: 2,
        // A sweep that stores nets stores only the biggest handful per trial.
        limit: spec.keepNets ? 8 : 0,
        gpu: spec.gpu,
      };
      const runId = db.startRun({
        seed,
        seconds: spec.seconds,
        dt: spec.dt,
        world: spec.world,
        fieldCells: spec.fieldCells,
        preset: 'soup',
        soupCount: spec.soupCount,
        parentRun: null,
        params,
        commit,
        note: spec.note,
        sweep: spec.name,
        point,
      });
      const result = await runPond(db, runId, runSpec);
      db.finishRun(runId, result.frames, result.wallMs, result.census);
      const row = { point, seed, runId, wallMs: result.wallMs };
      rows.push(row);
      hooks.onTrial?.(row, rows.length, total);
    }
  }
  return rows;
}
