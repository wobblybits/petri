import { CHANNELS } from '../fields.ts';
import { defaultParams, type Params } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { sampleSim, seededRandom, type Sample } from '../experiments/harness.ts';
import { fieldGpu } from '../gpu/field-gpu.ts';
import { genomeGpu } from '../gpu/genome-gpu.ts';
import { nativeSolver } from '../native/solver.ts';
import { captureNets, plantNet, type CapturedNet } from './capture.ts';
import { layGround } from './ground.ts';
import { measureDiversity, type Diversity } from './measure.ts';
import { openWebGpu } from './webgpu-node.ts';
import { encodeNet } from './net-blob.ts';
import type { NetData } from './net-blob.ts';
import type { PondDb } from './db.ts';

/*
 * The headless pond: the frame that ships, unmodified (`stepAsync`, the wasm
 * solver, and the field and genome shaders where Dawn gives Node a device;
 * see `webgpu-node.ts`). The measurement is `sampleSim` from the experiment
 * harness, so a sweep's JSON and a run's timeline are the same numbers.
 */

/** A stored net to plant, and where it came from. */
/** A harness sample plus the structural measures. `db.addSample` reads `diversity` if it is there. */
export type PondSample = Sample & { diversity: Diversity };

export interface SeedNet {
  netId: number;
  data: NetData;
}

export interface PondRunSpec {
  seconds: number;
  dt: number;
  seed: number;
  world: { w: number; h: number };
  fieldCells: number;
  params: Params;
  /** Founders dropped in as a soup at t = 0, on top of anything planted. */
  soupCount: number;
  seeds: SeedNet[];
  /** Simulated seconds between timeline samples. */
  sampleEvery: number;
  /**
   * Simulated seconds between net harvests, or 0 for the close of the run
   * only. A harvest writes every qualifying component's whole genome, so this
   * is the setting that decides whether the file is megabytes or gigabytes.
   */
  harvestEvery: number;
  /** Components smaller than this are not stored. */
  minBodies: number;
  /** Store only the largest this-many components per harvest. */
  limit: number | null;
  /** How the ground is arranged at the start, at the same total mass. See `ground.ts`. */
  /**
   * Whether to put the field and genome on the GPU. `auto` uses a device if
   * Dawn can open one; `on` fails the run without one; `off` never asks.
   */
  gpu: 'auto' | 'on' | 'off';
}

export interface PondRunResult {
  runId: number;
  frames: number;
  wallMs: number;
  samples: PondSample[];
  harvests: { t: number; nets: number; bytes: number }[];
  census: ReturnType<Sim['census']> & { tally: Sim['tally']; channelTotals: number[] };
  /** Which of the three paths actually engaged. */
  paths: { wasm: boolean; fieldGpu: boolean; genomeGpu: boolean; adapter: string | null };
}

export interface RunHooks {
  onSample?: (s: PondSample) => void;
  onHarvest?: (t: number, nets: CapturedNet[]) => void;
}

/** Where to put `k` planted nets in a dish of radius `r`: a sunflower spiral, deterministic. */
function spiral(cx: number, cy: number, r: number, k: number): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  if (k === 1) return [{ x: cx, y: cy }];
  for (let i = 0; i < k; i++) {
    const rad = r * Math.sqrt((i + 0.5) / k);
    const ang = i * 2.399963;
    out.push({ x: cx + Math.cos(ang) * rad, y: cy + Math.sin(ang) * rad });
  }
  return out;
}

/**
 * Run a pond and write it to the library. `Math.random` is replaced for the
 * duration and restored in a `finally`, so a throw part-way through does not
 * leave the process on a rigged stream.
 */
export async function runPond(
  db: PondDb,
  runId: number,
  spec: PondRunSpec,
  hooks: RunHooks = {},
): Promise<PondRunResult> {
  // The seeded stream spans every `await` below; two ponds run concurrently
  // in one process would get neither's stream.
  const realRandom = Math.random;
  Math.random = seededRandom(spec.seed);
  const t0 = performance.now();
  try {
    const params = spec.params;
    params.soupCount = spec.soupCount;
    const sim = new Sim(spec.world.w, spec.world.h, spec.fieldCells);
    // Always through the preset: it is what pins the world bound.
    loadPreset(sim, 'soup', params);

    let adapter: string | null = null;
    let onField = false;
    if (spec.gpu !== 'off') {
      const open = await openWebGpu();
      adapter = open.adapter ?? null;
      // `openFieldGpu` is one way and has to happen before the first frame:
      // it flips the ground's writes to deferred and reseeds it, and a queued
      // seed is only picked up by `gpuFieldStep`.
      onField = open.ok && (await sim.openFieldGpu());
      if (!onField && spec.gpu === 'on') {
        throw new Error(`pond: --gpu on, but no device: ${open.error ?? fieldGpu.lastError}`);
      }
    }
    // After the device is open: a non-uniform ground crosses as deferred
    // conserved adds, and `deferAdds` is switched on by `openFieldGpu`.
    layGround(sim, params.groundPatches);

    if (spec.seeds.length > 0) {
      const cx = sim.w * 0.5;
      const cy = sim.h * 0.5;
      const at = spiral(cx, cy, Math.max(0, sim.worldR - 120), spec.seeds.length);
      for (let i = 0; i < spec.seeds.length; i++) {
        // Negative founder lines, one per planted net, name the row it
        // arrived from. See `PlantOptions.lineage`.
        const lineage = -(i + 1);
        const ids = plantNet(sim, params, spec.seeds[i].data, at[i].x, at[i].y, {
          lineage,
          heading: Math.random() * Math.PI * 2,
        });
        if (ids.length === 0) {
          throw new Error(
            `pond: net ${spec.seeds[i].netId} would not fit (${spec.seeds[i].data.bodies.length} bodies; maxAgents ${params.maxAgents})`,
          );
        }
        db.addPlant(runId, lineage, spec.seeds[i].netId, ids.length);
      }
    }

    const samples: PondSample[] = [];
    const take = (at: number): PondSample => ({ ...sampleSim(sim, at), diversity: measureDiversity(sim, params) });
    const harvests: PondRunResult['harvests'] = [];
    const harvest = async (t: number, frame: number): Promise<void> => {
      // On the GPU path the host's copy of the learning is fresh only for
      // rewrite parents; fetch it before storing `plastic` and `critic`.
      if (!(await sim.syncLearningToHost())) {
        throw new Error('pond: could not read the learning back off the device');
      }
      const nets = captureNets(sim, {
        minBodies: spec.minBodies,
        limit: spec.limit ?? undefined,
      });
      let bytes = 0;
      db.transaction(() => {
        for (const net of nets) {
          const blob = encodeNet(net.data);
          bytes += blob.byteLength;
          db.addNet(runId, t, frame, net, blob);
        }
      });
      harvests.push({ t, nets: nets.length, bytes });
      hooks.onHarvest?.(t, nets);
    };

    // Driven by frame count, with `t` derived from it: accumulating `t += dt`
    // drifts, and `t` is a primary key in `sample`.
    const total = Math.max(0, Math.round(spec.seconds / spec.dt));
    const everySample = Math.max(1, Math.round(spec.sampleEvery / spec.dt));
    const everyHarvest = spec.harvestEvery > 0 ? Math.max(1, Math.round(spec.harvestEvery / spec.dt)) : 0;
    for (let frame = 0; frame < total; frame++) {
      const t = frame * spec.dt;
      if (frame % everySample === 0) {
        const s = take(t);
        samples.push(s);
        db.addSample(runId, s as unknown as Record<string, unknown>);
        hooks.onSample?.(s);
      }
      // Never at frame zero: a harvest of the opening soup is a harvest of the preset.
      if (everyHarvest > 0 && frame > 0 && frame % everyHarvest === 0) await harvest(t, frame);
      // On the GPU path `fields.data` is a stale copy and `wantFieldReadback`
      // brings it back at the end of the frame that asks, so the frame before
      // a sample asks; `sampleSim` reads it for `ground`.
      const next = frame + 1;
      sim.wantFieldReadback = next === total || next % everySample === 0;
      await sim.stepAsync(spec.dt, params);
    }
    const end = total * spec.dt;
    const last = take(end);
    samples.push(last);
    db.addSample(runId, last as unknown as Record<string, unknown>);
    hooks.onSample?.(last);
    await harvest(end, total);

    // What is left in the water at the close, per channel; the field itself is not stored.
    const field = sim.fields.data;
    const channelTotals = new Array<number>(CHANNELS).fill(0);
    for (let k = 0; k < field.length; k += CHANNELS) {
      for (let c = 0; c < CHANNELS; c++) channelTotals[c] += field[k + c];
    }
    const census = { ...sim.census(), tally: { ...sim.tally }, channelTotals };
    return {
      runId,
      frames: total,
      wallMs: performance.now() - t0,
      samples,
      harvests,
      census,
      paths: {
        wasm: nativeSolver.ready,
        fieldGpu: onField,
        genomeGpu: onField && genomeGpu.ready,
        adapter,
      },
    };
  } finally {
    Math.random = realRandom;
  }
}

/** `defaultParams()` with `key=value` overrides applied and checked. */
export function paramsWith(overrides: Map<string, number>): Params {
  const params = defaultParams();
  const known = new Set(Object.keys(params));
  for (const [key, value] of overrides) {
    if (!known.has(key)) {
      throw new Error(`pond: unknown parameter ${JSON.stringify(key)}`);
    }
    (params as unknown as Record<string, number>)[key] = value;
  }
  return params;
}
