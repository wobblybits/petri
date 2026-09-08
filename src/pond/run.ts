import { CHANNELS } from '../fields.ts';
import { defaultParams, type Params } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { sampleSim, seededRandom, type Sample } from '../experiments/harness.ts';
import { fieldGpu } from '../gpu/field-gpu.ts';
import { genomeGpu } from '../gpu/genome-gpu.ts';
import { nativeSolver } from '../native/solver.ts';
import { captureNets, plantNet, type CapturedNet } from './capture.ts';
import { layGround, type GroundSpec } from './ground.ts';
import { measureDiversity, type Diversity } from './measure.ts';
import { openWebGpu } from './webgpu-node.ts';
import { encodeNet } from './net-blob.ts';
import type { NetData } from './net-blob.ts';
import type { PondDb } from './db.ts';

/*
 * The headless pond: the same simulation the page runs, with nobody watching.
 *
 * `stepAsync` with the wasm solver on, and the field and genome shaders too
 * where Dawn gives Node a device — that is, the frame that ships, unmodified.
 * Nothing in `src/gpu/` knows this is not a browser; see `webgpu-node.ts`.
 *
 * The measurement is `sampleSim` from the experiment harness, not a second
 * one written here. A sweep's JSON and a run's timeline should be the same
 * numbers or one of them is lying.
 */

/** A stored net to plant, and where it came from. */
/**
 * A harness sample plus the structural measures.
 *
 * Kept as a wrapper rather than folded into `Sample` so the experiment
 * harness's own shape — which its JSON output and its two sweeps already
 * depend on — does not move. `db.addSample` reads `diversity` if it is there.
 */
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
  /**
   * How the ground is arranged at the start, at the same total mass.
   *
   * See `ground.ts`: the uniform dish every preset lays down is what §0 of the
   * chemistry plan calls a puddle, and comparing it against a patchy one at
   * *equal mass* is the only way to ask about structure rather than about how
   * much food there is.
   */
  ground: GroundSpec;
  /**
   * Whether to put the field and genome on the GPU.
   *
   * `auto` uses a device if Dawn can open one and runs on the CPU otherwise;
   * `on` fails the run rather than quietly costing five times the wall clock;
   * `off` never asks.
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

/**
 * Where to put `k` planted nets in a dish of radius `r`.
 *
 * A sunflower spiral: even coverage for any count without a grid's corners,
 * and the same points every time so a run seeded from the same nets starts
 * from the same arrangement.
 */
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
 * Run a pond and write it to the library.
 *
 * `Math.random` is replaced for the duration and restored in a `finally`, the
 * way the experiment harness does it — a run is a seeded, reproducible thing,
 * and a throw part-way through must not leave the process on a rigged stream.
 */
export async function runPond(
  db: PondDb,
  runId: number,
  spec: PondRunSpec,
  hooks: RunHooks = {},
): Promise<PondRunResult> {
  /*
   * The seeded stream spans every `await` below. Nothing else in this process
   * runs between frames — the CLI does one run and exits — so a trial stays
   * the reproducible thing the experiment harness makes it. A caller that ran
   * two ponds concurrently would get neither.
   */
  const realRandom = Math.random;
  Math.random = seededRandom(spec.seed);
  const t0 = performance.now();
  try {
    const params = spec.params;
    params.soupCount = spec.soupCount;
    const sim = new Sim(spec.world.w, spec.world.h, spec.fieldCells);
    // Always through the preset, even for a soup of nothing: it is what pins
    // the world bound, and a pond without one has no dish.
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
    /*
     * After the device is open, not before. A non-uniform ground crosses as
     * deferred conserved adds, and `deferAdds` is switched on by
     * `openFieldGpu` — laid down earlier it would land in the host's mirror,
     * which the shader does not read, and the GPU pond would start barren.
     */
    layGround(sim, spec.ground);

    if (spec.seeds.length > 0) {
      const cx = sim.w * 0.5;
      const cy = sim.h * 0.5;
      const at = spiral(cx, cy, Math.max(0, sim.worldR - 120), spec.seeds.length);
      for (let i = 0; i < spec.seeds.length; i++) {
        // Negative founder lines, one per planted net: ids are positive, so a
        // negative line says "arrived from the database" and names the row it
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
    const take = (at: number): PondSample => ({ ...sampleSim(sim, at), diversity: measureDiversity(sim) });
    const harvests: PondRunResult['harvests'] = [];
    const harvest = async (t: number, frame: number): Promise<void> => {
      /*
       * The learning lives on the device on the GPU path, and the host's copy
       * is fresh only for rewrite parents. `plastic` and `critic` are two of
       * the things a stored net is *for*, so fetch them before reading them —
       * otherwise the database fills with the zeros the host happened to hold,
       * which is worse than an empty column because it looks like an answer.
       */
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

    /*
     * Driven by frame count, with `t` derived from it.
     *
     * Accumulating `t += dt` drifts — at a sixtieth of a second, a minute of
     * pond lands on 59.9999999999979 — and `t` is a primary key in `sample`
     * and the value `--from-run` matches a harvest on. A schedule you can
     * write down is worth the multiply.
     */
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
      // Never at frame zero: a harvest of the opening soup is a harvest of
      // the preset, and the close of the run stores everything anyway.
      if (everyHarvest > 0 && frame > 0 && frame % everyHarvest === 0) await harvest(t, frame);
      /*
       * On the GPU path `fields.data` is a stale copy — the field lives in
       * device memory and only `wantFieldReadback` brings it back, at the end
       * of the frame that asks. `sampleSim` reads it for `ground`, through
       * `energy.storedTotal()`, so the frame *before* a sample is the one that
       * has to ask. Without this every `ground` in the timeline is whatever
       * the field held when it moved to the device, which is a plausible
       * number and a false one.
       *
       * It is a 16 MB copy, paid once per sample rather than once per frame:
       * at the default schedule, one frame in six hundred.
       */
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

    // What is left in the water at the close, per channel — the one summary
    // of the field worth carrying, since the field itself is not stored.
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
      // Read off the singletons the sim uses rather than through new
      // accessors on `Sim`: the headless runner should not be the reason the
      // shipping class grows a getter.
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
