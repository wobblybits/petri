import { describe, expect, it } from 'vitest';
import type { Agent } from '../agents.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { nativeSolver } from './solver.ts';

/**
 * What the flocking pass costs, timed directly.
 *
 * It used to be a 6-hop breadth-first search from every body every frame;
 * caching the pair list per topology turned that into a replay. Measured
 * against the same bench on the previous build:
 *
 *              9600      20000
 *   searching  2.16ms    4.72ms
 *   replaying  1.10ms    2.36ms
 *   rebuilding 1.95ms    4.39ms
 *
 * So the cache halves a pass that is a few percent of the frame — worth
 * having, not transformative. Both scale linearly in the pond; the earlier
 * claim that searching was super-linear was an artifact of how it was
 * measured, see below.
 *
 * Timed by calling the solver entry point straight, NOT by differencing whole
 * frames with flocking on and off. Differencing reads a ~1.5ms pass out of two
 * ~130ms frames, which is well inside the machine's run-to-run drift: it
 * reported 1.3ms and 8.7ms for the same build ten minutes apart, and made a
 * linear pass look like it grew 4.6x. Calling the pass in a loop measures the
 * pass.
 *
 * Both halves are timed because they trade off: the replay is what almost
 * every frame does, and the rebuild is what a frame pays when a wire changes.
 * A rebuild costs about what the old search did, so the cache is a win as long
 * as topology holds still for more than a frame or two — which it does, at
 * roughly one rebuild per sixty frames in a settled pond.
 */

function bigNets(sim: Sim, params: Params, netSizes: number[]): void {
  let ox = 400;
  for (const size of netSizes) {
    const ids: number[] = [];
    const cols = Math.max(2, Math.round(Math.sqrt(size)));
    for (let i = 0; i < size; i++) {
      const c = i % cols;
      const r = (i / cols) | 0;
      const kind = i % 3 === 0 ? 'era' : i % 3 === 1 ? 'con' : 'dup';
      const a: Agent | null = sim.spawn(kind, ox + c * 52, 400 + r * 52, (i * 0.7) % 6.28, params);
      ids.push(a ? a.id : -1);
    }
    for (let i = 0; i < size; i++) {
      const c = i % cols;
      const r = (i / cols) | 0;
      if (c + 1 < cols && i + 1 < size && ids[i] > 0 && ids[i + 1] > 0) {
        sim.wire(ids[i], 'r', ids[i + 1], 'l', params);
      }
      if (i + cols < size && r % 2 === 0 && ids[i] > 0 && ids[i + cols] > 0) {
        sim.wire(ids[i], 'p', ids[i + cols], 'p', params);
      }
    }
    ox += cols * 52 + 400;
  }
}

function median(a: number[]): number {
  return a.slice().sort((x, y) => x - y)[a.length >> 1];
}

/** A distinct owner per measurement, so priming is explicit. */
let owner = 900_000;

describe('flock pass cost', () => {
  it('replays far cheaper than it rebuilds, and stays flat with size', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const rows: { label: string; n: number; replay: number; rebuild: number; pairs: number }[] = [];

    for (const [label, nets] of [
      ['9600', [1600, 1600, 1600, 1600, 1600, 1600]],
      ['20000', [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]],
    ] as [string, number[]][]) {
      const params = defaultParams();
      params.maxAgents = 100_000;
      params.spawnInterval = 0;
      params.rewriteDuration = 0;
      params.snapRadius = 0;
      const sim = new Sim(60_000, 60_000);
      bigNets(sim, params, nets);
      const view = { x: 6000, y: 3000, zoom: 0.05, viewW: 1600, viewH: 900 };
      // Warm up, and leave the solver arrays packed with this sim's pose.
      for (let f = 0; f < 20; f++) sim.step(1 / 60, params, view);

      const n = sim.agents.size;
      const align = params.flockAlign;
      const sep = params.flockSep;
      const desired = Math.max(18, params.wireMinRest * 0.9);
      const call = (id: number, ver: number): void => {
        nativeSolver.flock(n, align, sep, 1 / 60, params.turnRate, desired, 6, id, ver, 1);
      };

      // Prime: this call finds no cache of its own and builds one.
      const me = owner++;
      call(me, 1);
      const pairs = nativeSolver.flockPairs();

      const replays: number[] = [];
      for (let k = 0; k < 40; k++) {
        const t0 = performance.now();
        call(me, 1);
        replays.push(performance.now() - t0);
      }
      // A fresh version every time, so every call is a full search and rebuild.
      const rebuilds: number[] = [];
      for (let k = 0; k < 12; k++) {
        const t0 = performance.now();
        call(me, 100 + k);
        rebuilds.push(performance.now() - t0);
      }
      rows.push({ label, n, replay: median(replays), rebuild: median(rebuilds), pairs });
    }

    console.log(
      `\nflock pass, timed directly\n` +
        rows
          .map(
            (r) =>
              `  ${r.label} bodies  ${r.pairs} cached pairs\n` +
              `    replay   ${r.replay.toFixed(2)}ms\n` +
              `    rebuild  ${r.rebuild.toFixed(2)}ms  (${(r.rebuild / r.replay).toFixed(0)}x a replay)`,
          )
          .join('\n') +
        '\n',
    );

    const [small, large] = rows;
    // A replay is a walk over the pair list, so it should track the pair count
    // and nothing else. 2.1x the bodies is 2.1x the pairs is ~2.2x the time.
    expect(
      large.replay,
      `20k replay ${large.replay.toFixed(2)}ms vs 9600 ${small.replay.toFixed(2)}ms — ` +
        'growing faster than the pair list, so something is being rebuilt per call',
    ).toBeLessThan(small.replay * 3.5);
    // The cache only earns its keep if replaying beats searching. Rebuilding
    // is searching plus the recording, so this ratio is the margin.
    expect(
      large.rebuild / large.replay,
      `rebuild ${large.rebuild.toFixed(2)}ms is barely above replay ` +
        `${large.replay.toFixed(2)}ms — the cache is not saving the search`,
    ).toBeGreaterThan(1.4);
  }, 900_000);
});
