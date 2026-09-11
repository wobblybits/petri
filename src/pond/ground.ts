import { CH } from '../fields.ts';
import type { Sim } from '../sim.ts';

/*
 * How the ground is laid out at the start of a run: the same total mass the
 * uniform seed would lay down, arranged differently, so a comparison is
 * about structure and not about how much food there is.
 *
 * Seeding goes through the deferred conserved add, the one crossing a
 * non-uniform pattern has on the GPU path, so this must run after
 * `openFieldGpu`. On that path a patchy run's first sample reads zero
 * ground: the adds reach the device on the first `gpuFieldStep`.
 */

/** `--ground <spec>` as the CLI spells it, resolved to `params.groundPatches`. Sugar over the parameter. */
export function parseGround(spec: string): number {
  if (spec === 'uniform') return 0;
  const m = /^patches(?::(\d+))?$/.exec(spec);
  if (!m) throw new Error(`pond: --ground wants uniform or patches[:n], got ${JSON.stringify(spec)}`);
  const patches = m[1] ? Number(m[1]) : 24;
  if (!(patches > 0)) throw new Error('pond: --ground patches wants a positive count');
  return patches;
}

/**
 * The mass the uniform seed lays down, computed rather than measured:
 * `storedTotal()` reads `fields.data`, a stale mirror on the GPU path.
 */
export function uniformMass(sim: Sim): number {
  const f = sim.fields;
  const cell = f.worldW / f.cols;
  const cells = (Math.PI * sim.worldR * sim.worldR) / (cell * cell);
  return sim.energy.cellCap * cells;
}

/**
 * Rearrange the ground without changing how much of it there is. Zero leaves
 * what the preset laid down; above it, the dish is cleared and the same mass
 * goes back in that many places on a deterministic sunflower spiral.
 */
export function layGround(sim: Sim, patches: number): void {
  if (!(patches > 0)) return;
  const mass = uniformMass(sim);
  // Clearing has to reach the device: `pendingSeed` dispatches the shader's
  // `fill`. On the CPU path nothing reads it, hence the direct fill as well.
  sim.energy.pendingSeed = 0;
  sim.fields.fillDisk(CH.energy, 0);

  const n = patches;
  const each = mass / n;
  const cx = sim.w * 0.5;
  const cy = sim.h * 0.5;
  const r = Math.max(0, sim.worldR - 120);
  for (let i = 0; i < n; i++) {
    const rad = n === 1 ? 0 : r * Math.sqrt((i + 0.5) / n);
    const ang = i * 2.399963;
    // Through `addAt`, so it is deferred on the GPU path and conserved on either.
    sim.energy.addAt(cx + Math.cos(ang) * rad, cy + Math.sin(ang) * rad, each);
  }
}
