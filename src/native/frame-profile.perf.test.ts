import { describe, expect, it } from 'vitest';
import type { Agent } from '../agents.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { nativeSolver } from './solver.ts';

/**
 * Where a 20k frame actually goes, phase by phase.
 *
 * A map rather than a pass/fail bench. Two rounds of guessing which phase was
 * expensive were both wrong, and the obvious way to check — turn a pass off
 * and difference whole frames — cannot resolve anything smaller than the
 * machine's own drift, which is several milliseconds on a frame this size. So
 * `Sim.profile` charges each phase as the frame runs and this prints the
 * ledger.
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

describe('20k frame profile', () => {
  it('maps the per-frame JS work', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = defaultParams();
    params.maxAgents = 100_000;
    params.spawnInterval = 0;
    params.rewriteDuration = 0;
    params.snapRadius = 0;
    const sim = new Sim(60_000, 60_000);
    bigNets(sim, params, [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]);
    const view = { x: 6000, y: 3000, zoom: 0.05, viewW: 1600, viewH: 900 };
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params, view);

    const n = sim.agents.size;
    const g = sim.graph;

    /*
     * The whole frame, phase by phase, from the profiler on Sim. Everything
     * here is charged to exactly one phase, so unlike timing callable pieces
     * from outside it does add up.
     */
    const prof = new Map<string, number>();
    Sim.profile = prof;
    const FRAMES = 30;
    const t0 = performance.now();
    try {
      for (let f = 0; f < FRAMES; f++) sim.step(1 / 60, params, view);
    } finally {
      Sim.profile = null;
    }
    const whole = (performance.now() - t0) / FRAMES;
    const rows = [...prof.entries()]
      .map(([name, ms]) => [name, ms / FRAMES] as [string, number])
      .sort((a, b) => b[1] - a[1]);

    const named = rows.reduce((acc, r) => acc + r[1], 0);
    console.log(
      `\n20k frame profile  (${n} bodies, ${g.wires.size} wires, ` +
        `${FRAMES} frames)\n` +
        rows
          .map(
            ([name, ms]) =>
              `  ${name.padEnd(22)} ${ms.toFixed(2).padStart(7)}ms` +
              `  ${((ms / whole) * 100).toFixed(1).padStart(5)}%`,
          )
          .join('\n') +
        `\n  ${'-'.repeat(22)} ${'-'.repeat(7)}\n` +
        `  ${'accounted'.padEnd(22)} ${named.toFixed(2).padStart(7)}ms` +
        `  ${((named / whole) * 100).toFixed(1).padStart(5)}%\n` +
        `  ${'whole frame'.padEnd(22)} ${whole.toFixed(2).padStart(7)}ms\n`,
    );
    // The profiler charges every phase, so it should account for nearly all of
    // the frame. A large gap means a phase was added without a marker.
    expect(named / whole, `only ${((named / whole) * 100).toFixed(0)}% accounted`)
      .toBeGreaterThan(0.9);
    expect(whole).toBeGreaterThan(0);
  }, 900_000);
});
