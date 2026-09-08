import { execFileSync } from 'node:child_process';
import type { Params } from '../params.ts';
import { PondDb } from './db.ts';
import type { GroundSpec } from './ground.ts';
import { paramsWith, runPond, type PondRunSpec } from './run.ts';

/*
 * A parameter sweep whose results accumulate.
 *
 * `src/experiments/` already sweeps: a grid crossed with seeds, a table on
 * stdout, one JSON file per invocation. What it cannot do is remember. Every
 * sweep is a fresh file, nothing relates one to the next, and the question
 * "what have we learned about `declutter`" is answered by opening the files
 * you happen to still have and reading them by eye.
 *
 * So this writes into the pond library instead. A trial is a `run` row with
 * its whole `Params`, its commit, its timeline and its nets; a sweep is a
 * name and a grid point recorded on each of them. Which makes the analysis a
 * query over everything ever run rather than over this afternoon — and makes
 * a sweep resumable, comparable across commits, and joinable against the nets
 * it produced.
 *
 * `analyze.ts` is the reading half.
 */

export interface SweepSpec {
  name: string;
  /** Parameter axes. Every combination of every axis is run, once per seed. */
  grid: Record<string, number[]>;
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
  /** How the ground is arranged; see `ground.ts`. */
  ground: GroundSpec;
  /**
   * Store nets from each trial, or only the timeline.
   *
   * Off by default. A sweep is a hundred ponds and their genomes are hundreds
   * of megabytes; the timeline is what a sweep is for, and a point worth
   * keeping bodies from can be re-run on its own.
   */
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

function gitCommit(): string | null {
  try {
    return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
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
  const points = gridPoints(spec.grid);
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
        // A sweep that stores nets stores only the biggest handful per trial;
        // the timeline is the deliverable and the genomes are a by-product.
        limit: spec.keepNets ? 8 : 0,
        gpu: spec.gpu,
        ground: spec.ground,
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
