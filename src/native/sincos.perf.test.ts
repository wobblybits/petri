import { describe, expect, it } from 'vitest';
import type { Agent } from '../agents.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { nativeSolver } from './solver.ts';

/**
 * What memoizing cos/sin of the heading on the agent is worth.
 *
 * Every port position goes through `stemOffsetInto`, which needs both. The
 * work is indexed per wire-endpoint but the heading is per body, so a body
 * with three wires had its heading's sine taken six times a frame by one pass
 * and six more by the next.
 *
 * A/B'd inside one process by clearing the memo before each cold call, so both
 * arms run the identical code over the identical data back to back. Comparing
 * absolute phase times across runs does not work here: the same build measured
 * 96ms and 129ms a frame twenty minutes apart on this machine.
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

describe('heading sin/cos memo', () => {
  it('saves real time in the passes that walk wire endpoints', async () => {
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

    const g = sim.graph;
    const detailed = (sim as unknown as { wireDetailed: (w: never) => boolean }).wireDetailed;
    const frozen = (sim as unknown as { rewriteFrozen: Set<number> }).rewriteFrozen;
    const bodies = [...sim.agents.values()];
    const chill = (): void => {
      for (const a of bodies) a.csHeading = NaN;
    };

    // Alternating blocks, so neither arm gets the warmer cache or the cooler
    // machine. Cold clears the memo first, and that clear is charged to it —
    // it is a walk of 20k bodies writing one field, which is cheap next to the
    // 60k transcendentals it is standing in for.
    const warm: number[] = [];
    const cold: number[] = [];
    for (let block = 0; block < 10; block++) {
      const isCold = block % 2 === 1;
      const ts: number[] = [];
      for (let k = 0; k < 8; k++) {
        if (isCold) chill();
        const t0 = performance.now();
        g.refreshLengths(sim.agents, sim.w, sim.h, frozen, detailed as never);
        ts.push(performance.now() - t0);
      }
      (isCold ? cold : warm).push(median(ts));
    }
    const w = median(warm);
    const c = median(cold);
    console.log(
      `\nrefreshLengths at ${sim.agents.size} bodies / ${g.wires.size} wires\n` +
        `  memo warm  ${w.toFixed(2)}ms\n` +
        `  memo cold  ${c.toFixed(2)}ms\n` +
        `  saved      ${(c - w).toFixed(2)}ms  (${(((c - w) / c) * 100).toFixed(0)}%)\n`,
    );
    expect(w, `warm ${w.toFixed(2)}ms is not faster than cold ${c.toFixed(2)}ms`).toBeLessThan(c);
  }, 900_000);
});
