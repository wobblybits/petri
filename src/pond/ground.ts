/*
 * How the ground is laid out at the start of a run.
 *
 * `loadPreset` seeds it one way: `cellCap` in every cell of the disk, which is
 * uniform, stationary, and a puddle. Measured, a 500-body pond dents an *ungrowing* dish by five per
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
 * Where that arranging lives: `Energy.seedGround`, reading `Energy.patches`,
 * which `configure` carries over from `params.groundPatches` every frame. It
 * used to live here, called by the headless runner and by nobody else, so the
 * page has never in its life seen a patch. What is left in this file is the
 * CLI's spelling of the parameter.
 *
 * One consequence to expect in a timeline rather than to debug: on the GPU
 * path a patchy run's **first sample reads zero ground**. The adds are queued
 * and do not reach the device until the first `gpuFieldStep` dispatches them,
 * so `t = 0` catches the dish after the clear and before the refill. By the
 * next sample the mass is there — measured, 12,752 against a uniform dish's
 * 12,769, the difference being the disk-area approximation at the rim.
 */

/**
 * `--ground <spec>` as the CLI spells it, resolved to `params.groundPatches`.
 *
 * Sugar over the parameter, not a second source of truth: a sweep sets the
 * parameter directly through `--axis groundPatches=0,4,16` and never comes
 * through here.
 */
export function parseGround(spec: string): number {
  if (spec === 'uniform') return 0;
  const m = /^patches(?::(\d+))?$/.exec(spec);
  if (!m) throw new Error(`pond: --ground wants uniform or patches[:n], got ${JSON.stringify(spec)}`);
  const patches = m[1] ? Number(m[1]) : 24;
  if (!(patches > 0)) throw new Error('pond: --ground patches wants a positive count');
  return patches;
}
