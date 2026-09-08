import { CH } from '../fields.ts';
import type { Sim } from '../sim.ts';

/*
 * How the ground is laid out at the start of a run.
 *
 * `loadPreset` seeds it one way: `cellCap` in every cell of the disk, which is
 * uniform, stationary, and the thing `docs/energy-chemistry-plan.md` §0 calls
 * a puddle. Measured, a 500-body pond dents an *ungrowing* dish by five per
 * cent in a simulated minute and never bares a cell — so there is nothing to
 * contest and nowhere worth going, and every sweep so far has been run in that
 * world.
 *
 * The point of this module is that **the same total mass** can be arranged
 * differently. Comparing a uniform dish against a patchy one at equal mass is
 * a comparison about *structure*; comparing it against a thinner one is a
 * comparison about how much food there is, which is a different and much less
 * interesting question. Everything here conserves the total the uniform seed
 * would have laid down.
 *
 * Seeding goes through the deferred conserved add, which is the one crossing a
 * non-uniform pattern has: `fillDisk` writes the host's mirror, and on the GPU
 * path the shader does not read it. So this must run *after* `openFieldGpu`.
 *
 * One consequence to expect in a timeline rather than to debug: on the GPU
 * path a patchy run's **first sample reads zero ground**. The adds are queued
 * here and do not reach the device until the first `gpuFieldStep` dispatches
 * them, so `t = 0` catches the dish after the clear and before the refill. By
 * the next sample the mass is there — measured, 12,752 against a uniform
 * dish's 12,769, the difference being `uniformMass`'s approximation at the rim.
 */

export type GroundMode = 'uniform' | 'patches' | 'none';

export interface GroundSpec {
  mode: GroundMode;
  /** Patches to spread the mass over, when `mode` is `patches`. */
  patches: number;
}

export function parseGround(spec: string): GroundSpec {
  if (spec === 'uniform' || spec === 'none') return { mode: spec, patches: 0 };
  const m = /^patches(?::(\d+))?$/.exec(spec);
  if (!m) throw new Error(`pond: --ground wants uniform, none, or patches[:n], got ${JSON.stringify(spec)}`);
  const patches = m[1] ? Number(m[1]) : 24;
  if (!(patches > 0)) throw new Error('pond: --ground patches wants a positive count');
  return { mode: 'patches', patches };
}

/**
 * The mass the uniform seed lays down, computed rather than measured.
 *
 * `storedTotal()` reads `fields.data`, which is a stale mirror on the GPU
 * path — measuring it there would give a patchy pond a different total from a
 * uniform one and quietly turn a structure comparison into a quantity one.
 * The area of the disk over the area of a cell is the count `fillDisk` writes
 * to, near enough at the rim for a seeding.
 */
export function uniformMass(sim: Sim): number {
  const f = sim.fields;
  const cell = f.worldW / f.cols;
  const cells = (Math.PI * sim.worldR * sim.worldR) / (cell * cell);
  return sim.energy.cellCap * cells;
}

/**
 * Rearrange the ground without changing how much of it there is.
 *
 * `uniform` leaves what the preset laid down. `none` takes it away, which is
 * the barren control. `patches` clears the dish and puts the same mass back in
 * `spec.patches` places on a sunflower spiral — even coverage without a grid's
 * corners, and the same arrangement every time, so a seed is still a
 * reproducible thing.
 */
export function layGround(sim: Sim, spec: GroundSpec): void {
  if (spec.mode === 'uniform') return;
  const mass = uniformMass(sim);
  /*
   * Clearing has to reach the device. `pendingSeed` is what `seedGround` uses
   * and it dispatches the shader's `fill` entry point, so setting it to zero
   * is the one way to empty the field on both paths with one line. On the CPU
   * path nothing reads it, hence the direct fill as well.
   */
  sim.energy.pendingSeed = 0;
  sim.fields.fillDisk(CH.energy, 0);
  if (spec.mode === 'none') return;

  const n = spec.patches;
  const each = mass / n;
  const cx = sim.w * 0.5;
  const cy = sim.h * 0.5;
  const r = Math.max(0, sim.worldR - 120);
  for (let i = 0; i < n; i++) {
    const rad = n === 1 ? 0 : r * Math.sqrt((i + 0.5) / n);
    const ang = i * 2.399963;
    // Through `addAt`, so it is deferred and packed on the GPU path and lands
    // as a conserved quantity rather than a density on either.
    sim.energy.addAt(cx + Math.cos(ang) * rad, cy + Math.sin(ang) * rad, each);
  }
}
