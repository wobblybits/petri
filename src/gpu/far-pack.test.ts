import { describe, expect, it } from 'vitest';
import type { Agent } from '../agents.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';

/**
 * `packFar` must not disturb the shared wire cache.
 *
 * It builds a *filtered* wire list — only wires whose endpoints are both in
 * the pack — and it used to build it in `this.wirePack`, which is the array
 * `wireListResolved` caches and stamps valid against the graph and roster
 * versions, with `wireEndA`/`wireEndB` resolved to line up with it entry by
 * entry. Overwriting it left the cache holding a different list under a stamp
 * that still claimed to be current, so every later reader paired wire k with
 * some other wire's endpoints. Port torques then reel unrelated bodies
 * together and wires appear to grow without bound.
 *
 * It only ever bit on the GPU path, because that is `packFar`'s only caller
 * and the path had never run: the wasm solve always won the frame ahead of it.
 * The first time it did run — at the zoom where the LOD puts every body on the
 * FAR tier, which is the only place the GPU is allowed to take a frame — the
 * pond came apart.
 */

type Inner = {
  packFar(p: Params): unknown;
  wireListResolved(): { id: number }[];
  wireEndA: (Agent | undefined)[];
  wireEndB: (Agent | undefined)[];
};

function net(): { sim: Sim; params: Params } {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.rewriteDuration = 0;
  params.snapRadius = 0;
  const sim = new Sim(1200, 800);
  const cols = 4;
  const rows = 4;
  const ids: number[] = [];
  const step = params.wireMinRest + 16;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      const kind = k % 3 === 0 ? 'era' : k % 3 === 1 ? 'con' : 'dup';
      ids.push(sim.spawn(kind, 300 + i * step, 300 + j * step, 0, params, true)!.id);
    }
  }
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      if (i + 1 < cols) sim.wire(ids[k], 'r', ids[k + 1], 'l', params);
      if (j + 1 < rows && (i + j) % 2 === 0) sim.wire(ids[k], 'p', ids[k + cols], 'p', params);
    }
  }
  return { sim, params };
}

/** Every cached endpoint must be the agent the wire at that slot names. */
function cacheAgrees(sim: Sim): boolean {
  const inner = sim as unknown as Inner;
  const list = inner.wireListResolved();
  const a = inner.wireEndA;
  const b = inner.wireEndB;
  if (a.length !== list.length || b.length !== list.length) return false;
  const wires = sim.graph.wires;
  for (let i = 0; i < list.length; i++) {
    const w = wires.get(list[i].id);
    if (!w) return false;
    if (a[i]?.id !== w.a.id || b[i]?.id !== w.b.id) return false;
  }
  return true;
}

describe('packFar and the wire cache', () => {
  it('leaves the resolved wire list intact', () => {
    const { sim, params } = net();
    sim.step(1 / 60, params);
    expect(cacheAgrees(sim), 'cache was wrong before packFar even ran').toBe(true);

    (sim as unknown as Inner).packFar(params);

    expect(
      cacheAgrees(sim),
      'packFar left the wire cache pointing at the wrong endpoints',
    ).toBe(true);
  });

  it('still leaves it intact on a later frame, when the stamp says reuse', () => {
    // The damaging case is the frame *after*: the versions have not moved, so
    // nothing rebuilds and the mismatched pairing is simply used.
    const { sim, params } = net();
    sim.step(1 / 60, params);
    (sim as unknown as Inner).packFar(params);
    sim.step(1 / 60, params);
    expect(cacheAgrees(sim)).toBe(true);
  });
});
