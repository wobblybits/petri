import { describe, expect, it } from 'vitest';
import { CH, CHANNELS } from '../fields.ts';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { parseGround } from './ground.ts';

/*
 * The property that makes a ground comparison mean anything: **the same total
 * mass, arranged differently.** A patchy dish against a thinner one is a
 * comparison about how much food there is, which is not the question §0 poses.
 */

/**
 * The layout is a parameter now, not a function call: `Energy.configure`
 * carries `groundPatches` over every frame and lays the dish again when it
 * changes, which is what makes the slider live and what makes the page, the
 * runner and the designer agree. These tests drive it the way the pond does.
 */
const lay = (sim: Sim, patches: number): void => {
  sim.energy.patches = patches;
  sim.energy.seedGround();
};

function pond(patches = 0): { sim: Sim; params: ReturnType<typeof defaultParams> } {
  const params = defaultParams();
  params.groundPatches = patches;
  params.soupCount = 0;
  params.spawnInterval = 0;
  params.energyRegrow = 0;
  params.decay = 0;
  params.diffuse = 0;
  const sim = new Sim(1600, 1200, 128);
  loadPreset(sim, 'soup', params);
  return { sim, params };
}

const groundTotal = (sim: Sim): number => {
  const d = sim.fields.data;
  let s = 0;
  for (let k = CH.energy; k < d.length; k += CHANNELS) s += d[k];
  return s;
};

describe('ground layout', () => {
  it('parses the spec, and refuses one it cannot', () => {
    expect(parseGround('uniform')).toBe(0);
    expect(parseGround('patches')).toBe(24);
    expect(parseGround('patches:7')).toBe(7);
    expect(() => parseGround('patches:0')).toThrow(/positive/);
    expect(() => parseGround('lumpy')).toThrow(/uniform or patches/);
  });

  it('leaves a uniform dish alone', () => {
    const { sim } = pond();
    const before = groundTotal(sim);
    expect(before).toBeGreaterThan(0);
    lay(sim, 0);
    expect(groundTotal(sim)).toBe(before);
  });

  it('puts the same mass into patches that it took out of the dish', () => {
    /*
     * The whole point. `Energy.uniformMass` is computed from the disk's area rather
     * than measured off `fields.data`, because that array is a stale mirror on
     * the GPU path — measuring there would hand a patchy pond a different
     * total from a uniform one and quietly turn a structure comparison into a
     * quantity one.
     */
    const { sim } = pond();
    const before = groundTotal(sim);
    const analytic = sim.energy.uniformMass;
    // The analytic count is the disk's area over a cell's; near enough at the
    // rim that the two agree to a few per cent, which is what makes it usable.
    expect(Math.abs(analytic - before) / before).toBeLessThan(0.05);

    lay(sim, 12);
    expect(groundTotal(sim)).toBeCloseTo(analytic, 3);

    // And laying it again lands the same mass, not twice it: the seed drops
    // its own queue first, so the two callers that both ask on the first
    // frame — `configure` noticing the change, then `pinWorld` — agree.
    lay(sim, 12);
    expect(groundTotal(sim)).toBeCloseTo(analytic, 3);
  });

  it('makes the dish uneven, which uniform never is', () => {
    const occupied = (sim: Sim): number[] => {
      const d = sim.fields.data;
      const out: number[] = [];
      for (let k = CH.energy; k < d.length; k += CHANNELS) if (d[k] > 0) out.push(d[k]);
      return out;
    };
    const flat = pond();
    const flatCells = occupied(flat.sim).length;

    const patchy = pond();
    lay(patchy.sim, 12);
    const patchyCells = occupied(patchy.sim).length;

    /*
     * Same mass over a quarter of the cells, so what is standing there is four
     * times as rich and three quarters of the dish is bare. That is the
     * structure a uniform dish cannot express and the reason nothing on one
     * has anywhere to go. A quarter and not a tenth on purpose: patches are
     * blobs with an extent a body can stand on and graze down, not points.
     */
    expect(patchyCells / flatCells).toBeGreaterThan(0.2);
    expect(patchyCells / flatCells).toBeLessThan(0.32);
  });

  it('feeds a body standing on a patch and starves one that is not', () => {
    const params = defaultParams();
    params.soupCount = 0;
    params.spawnInterval = 0;
    params.energyRegrow = 0;
    params.decay = 0;
    params.diffuse = 0;
    params.upkeep = 0;
    params.groundPatches = 1; // one patch, at the centre by construction
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    const on = sim.spawn('con', sim.w * 0.5, sim.h * 0.5, 0, params, true)!;
    const off = sim.spawn('con', sim.w * 0.5 + 900, sim.h * 0.5, 0, params, true)!;
    on.pinned = true;
    off.pinned = true;
    on.extra = 0;
    off.extra = 0;
    for (let i = 0; i < 30; i++) sim.step(1 / 60, params);
    expect(on.extra).toBeGreaterThan(0);
    expect(off.extra).toBe(0);
  });
});
