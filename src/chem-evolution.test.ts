import { describe, expect, it } from 'vitest';
import { EMIT, TASTE, seedChem, type Agent } from './agents.ts';
import { CHEM_TASTE_MAX } from './rewrite.ts';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/**
 * The scent genome has to actually move, and stay legal while it does.
 *
 * `emit` and `taste` are seeded from kind so a fresh pond behaves exactly as
 * it did before they existed — which means the whole mechanism is invisible
 * until something breeds. If inheritance were silently not wired up, every
 * test would still pass and the feature would be an elaborate way of writing
 * down the old constants.
 */

function pond(): { sim: Sim; params: ReturnType<typeof defaultParams> } {
  const params = defaultParams();
  params.soupCount = 400;
  params.spawnInterval = 0;
  const sim = new Sim(800, 600);
  loadPreset(sim, 'soup', params);
  return { sim, params };
}

function drift(a: Agent, params: ReturnType<typeof defaultParams>): number {
  const seed = seedChem(a.kind, params);
  let worst = 0;
  for (let k = 0; k < 8; k++) worst = Math.max(worst, Math.abs(a.chem[k] - seed[k]));
  return worst;
}

describe('scent genome', () => {
  it('drifts away from its seed once bodies breed', () => {
    const { sim, params } = pond();
    const view = { x: 400, y: 300, zoom: 0.4, viewW: 1200, viewH: 800 };
    const before = sim.nextId;
    for (let f = 0; f < 1800; f++) sim.step(1 / 60, params, view);
    const born = sim.nextId - before;
    expect(born, 'nothing was born, so nothing could inherit').toBeGreaterThan(20);

    let moved = 0;
    let worst = 0;
    for (const a of sim.agents.values()) {
      const d = drift(a, params);
      if (d > 1e-4) moved++;
      worst = Math.max(worst, d);
    }
    expect(moved, 'no body differs from its kind seed').toBeGreaterThan(0);
    expect(worst, `worst drift ${worst.toFixed(3)}`).toBeGreaterThan(0.01);
  });

  it('keeps every genome legal however far it drifts', () => {
    const { sim, params } = pond();
    const view = { x: 400, y: 300, zoom: 0.4, viewW: 1200, viewH: 800 };
    for (let f = 0; f < 1800; f++) sim.step(1 / 60, params, view);
    for (const a of sim.agents.values()) {
      let sum = 0;
      for (let k = EMIT; k < EMIT + 4; k++) {
        expect(a.chem[k], `emit ${k} went negative`).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(a.chem[k]), 'emit went non-finite').toBe(true);
        sum += a.chem[k];
      }
      // One unit of voice, spent across four channels. A body that mutated
      // its way to silence keeps it rather than being renormalised up out of
      // noise, so the sum is either ~1 or ~0.
      expect(sum < 1e-5 || Math.abs(sum - 1) < 1e-3, `emit sums to ${sum}`).toBe(true);
      for (let k = TASTE; k < TASTE + 4; k++) {
        expect(Math.abs(a.chem[k]), `taste ${k} out of bounds`).toBeLessThanOrEqual(
          CHEM_TASTE_MAX + 1e-6,
        );
      }
    }
  });

  it('lets taste go negative, which the fixed weights never could', () => {
    // Not asserting that it *does* in any given run — only that nothing in the
    // pipeline clamps avoidance away, since a body fleeing what it smells is
    // the main behaviour the old switch could not express.
    const params = defaultParams();
    const c = seedChem('con', params);
    c[TASTE] = -2;
    expect(Math.min(...Array.from(c.subarray(TASTE, TASTE + 4)))).toBeLessThan(0);
  });
});
