import { describe, expect, it } from 'vitest';
import { nativeSolver } from './native/solver.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * End-to-end frame cost across agent count, JS-only (native's MAX_BODIES is
 * 32768, well under the 100k tier here, so nativeForces stays off
 * throughout for a fair, uniform comparison).
 *
 * This is the real-app counterpart to the throwaway AoS-vs-SoA microbenchmark
 * that motivated the agent-store rewrite: that one isolated the integration
 * loop and showed AoS collapsing non-linearly past ~100k agents while flat
 * SoA stayed linear. This drives the actual Sim — every pass a frame runs,
 * not just integration — to see whether that cliff is actually gone here.
 * Density is held roughly constant across tiers (world area scales with
 * agent count) so a slowdown reflects agent count, not a denser pond.
 *
 * Stops at 100k on purpose. 1M agents spawned as plain, individually-tracked
 * Sim bodies (not the bulk/instanced population this project's scaling story
 * is actually about) pushed a single Node process past a few GB of heap and
 * multiple minutes per frame in both the current code and the pre-rewrite
 * baseline — a real cost, but one that swamps the per-frame comparison this
 * test is for and says nothing more than "a plain-object-per-agent design
 * doesn't hold a million of them," which was never in question.
 */
describe('agent-count scaling', () => {
  const DENSITY = 16_000; // px^2 per agent, matching a 60-agent soup in a 1200x800 world

  const cases: { n: number; frames: number }[] = [
    { n: 10_000, frames: 30 },
    { n: 100_000, frames: 8 },
  ];

  for (const { n, frames } of cases) {
    it(`runs ${n.toLocaleString()} agents at a steady frame cost`, async () => {
      expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
      Sim.nativeForces = false;
      try {
        const side = Math.sqrt(n * DENSITY);
        const sim = new Sim(side, side);
        const params = defaultParams();
        params.spawnInterval = 0;
        params.upkeep = 0;

        const t0 = performance.now();
        for (let i = 0; i < n; i++) {
          const kind = i % 3 === 0 ? 'era' : i % 3 === 1 ? 'con' : 'dup';
          sim.spawn(
            kind,
            Math.random() * side,
            Math.random() * side,
            Math.random() * Math.PI * 2,
            params,
            true,
          );
        }
        const spawnMs = performance.now() - t0;
        expect(sim.agents.size).toBe(n);

        // Zoomed out enough that the whole pond is FAR tier — the LOD tier
        // orders of magnitude more agents actually run in.
        const view = { x: side / 2, y: side / 2, zoom: 1200 / side, viewW: 1200, viewH: 800 };
        sim.setViewExtent((view.viewW / view.zoom) * 1.5, (view.viewH / view.zoom) * 1.5);

        for (let f = 0; f < 2; f++) sim.step(1 / 60, params, view); // warm up
        const ts: number[] = [];
        for (let f = 0; f < frames; f++) {
          const t1 = performance.now();
          sim.step(1 / 60, params, view);
          ts.push(performance.now() - t1);
        }
        ts.sort((a, b) => a - b);
        const median = ts[Math.floor(ts.length / 2)];
        const mean = ts.reduce((a, b) => a + b, 0) / ts.length;
        console.log(
          `n=${n.toLocaleString().padStart(9)}  spawn=${spawnMs.toFixed(0).padStart(6)}ms  ` +
            `frame median=${median.toFixed(2).padStart(8)}ms  mean=${mean.toFixed(2).padStart(8)}ms  ` +
            `per-1k-agents=${((median / n) * 1000).toFixed(4)}ms`,
        );
      } finally {
        Sim.nativeForces = true;
      }
    }, 280_000);
  }
});
