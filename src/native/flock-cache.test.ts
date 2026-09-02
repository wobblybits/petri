import { describe, expect, it } from 'vitest';
import type { Agent } from '../agents.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { stateHash } from '../state-hash.ts';
import { nativeSolver } from './solver.ts';

/**
 * The flocking neighbourhood cache.
 *
 * Flocking is a 6-hop breadth-first search from every body, and it used to run
 * every frame for a neighbourhood that only changes when a wire does. The pair
 * list is now built once per topology and replayed, keyed on `graph.version`
 * and `rosterVersion` together — wires alone are not enough, because `agents`
 * is a Map iterated in insertion order and these are indices into that order.
 *
 * The saving is about half the pass (see flock.perf.test.ts). What makes it
 * safe to take is that it is bit-identical, which is what the first test here
 * checks and what the pair list is ordered to preserve.
 *
 * Three things have to hold, and only the first is obvious:
 *
 *   1. replaying gives the same answer as searching, to the bit;
 *   2. a topology change invalidates, including one that adds no wires;
 *   3. two Sims sharing the singleton solver do not replay each other's list.
 *
 * (3) is the one that would have shipped. The wasm module is process-wide and
 * the cached pairs are indices into whichever Sim packed last, so without an
 * ownership check the second Sim in a process silently flocks against the
 * first one's graph.
 */

function brick(sim: Sim, params: Params, cols: number, rows: number): Agent[] {
  const out: Agent[] = [];
  const step = params.wireMinRest + 16;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      const kind = k % 3 === 0 ? 'era' : k % 3 === 1 ? 'con' : 'dup';
      const a = sim.spawn(kind, 300 + i * step, 300 + j * step, (k * 0.7) % 6.28, params);
      if (a) out.push(a);
    }
  }
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      if (j + 1 < rows) sim.wire(out[k].id, 'p', out[k + cols].id, 'p', params);
      if (i + 1 < cols && (i + j) % 2 === 0) sim.wire(out[k].id, 'r', out[k + 1].id, 'l', params);
    }
  }
  return out;
}

function settled(): Params {
  const p = defaultParams();
  p.maxAgents = 5000;
  p.spawnInterval = 0;
  p.rewriteDuration = 0;
  p.snapRadius = 0;
  return p;
}

const HOPS = 6;
const VIEW = { x: 600, y: 600, zoom: 0.3, viewW: 1200, viewH: 800 };

/** `forceRebuild` bumps the roster key every frame, so nothing is ever reused. */
function run(frames: number, forceRebuild: boolean): string {
  const params = settled();
  const sim = new Sim(4000, 4000);
  brick(sim, params, 8, 8);
  for (let f = 0; f < frames; f++) {
    if (forceRebuild) sim.rosterVersion++;
    sim.step(1 / 60, params, VIEW);
  }
  return stateHash(sim);
}

describe('flock neighbourhood cache', () => {
  it('replays a cached pair list bit-for-bit', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const cached = run(120, false);
    const searched = run(120, true);
    expect(cached, `cached ${cached} vs searched ${searched}`).toBe(searched);
  });

  it('actually caches — the pair list is populated and reused', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = settled();
    const sim = new Sim(4000, 4000);
    brick(sim, params, 8, 8);
    sim.step(1 / 60, params, VIEW);
    expect(nativeSolver.flockPairs(), 'no pair list built').toBeGreaterThan(0);
    const n = sim.agents.size;
    expect(
      nativeSolver.flockCacheHolds(
        // simId identifies the owner and is deliberately not public; nothing
        // outside the solver binding has any business passing it.
        (sim as unknown as { simId: number }).simId,
        sim.graph.version,
        sim.rosterVersion,
        n,
        HOPS,
      ),
      'cache not claimed after a step',
    ).toBe(true);
  });

  it('invalidates when the roster turns over without touching a wire', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = settled();
    const sim = new Sim(4000, 4000);
    brick(sim, params, 6, 6);
    // A free body: killing it detaches nothing, so the wire version holds.
    const free = sim.spawn('dup', 2000, 2000, 0, params, true)!;
    sim.step(1 / 60, params, VIEW);
    const wireVersion = sim.graph.version;
    const n = sim.agents.size;
    const id = (sim as unknown as { simId: number }).simId;
    expect(nativeSolver.flockCacheHolds(id, wireVersion, sim.rosterVersion, n, HOPS)).toBe(true);

    // The precise hazard: kill an unwired body and spawn another. No wire
    // changes and the count comes back to where it was, so neither the graph
    // version nor `n` notices — but every index after the dead one shifted.
    sim.kill(free.id);
    sim.spawn('dup', 2100, 2100, 0, params, true);
    expect(sim.graph.version, 'wire version moved after all').toBe(wireVersion);
    expect(sim.agents.size, 'body count moved after all').toBe(n);
    expect(
      nativeSolver.flockCacheHolds(id, wireVersion, sim.rosterVersion, n, HOPS),
      'cache survived a roster turnover that renumbered every index',
    ).toBe(false);
  });

  it("does not let one sim replay another sim's neighbourhoods", async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    // Same body count, different topology, so the two pair lists differ in
    // length and the count alone says whose list is loaded.
    const pA = settled();
    const a = new Sim(4000, 4000);
    brick(a, pA, 6, 6);
    const pB = settled();
    const b = new Sim(4000, 4000);
    brick(b, pB, 4, 9);
    expect(a.agents.size).toBe(b.agents.size);

    a.step(1 / 60, pA, VIEW);
    const pairsA = nativeSolver.flockPairs();
    b.step(1 / 60, pB, VIEW);
    const pairsB = nativeSolver.flockPairs();
    expect(pairsA).toBeGreaterThan(0);
    expect(pairsB).toBeGreaterThan(0);
    expect(pairsB, 'the two shapes need different pair counts to tell apart')
      .not.toBe(pairsA);

    // Back to A. Its key no longer holds, so it must rebuild — if ownership
    // were untracked it would replay B's list and report B's count.
    const idA = (a as unknown as { simId: number }).simId;
    expect(
      nativeSolver.flockCacheHolds(idA, a.graph.version, a.rosterVersion, a.agents.size, HOPS),
      'A still claims a cache that B overwrote',
    ).toBe(false);
    a.step(1 / 60, pA, VIEW);
    expect(nativeSolver.flockPairs(), 'A replayed B\'s pair list').toBe(pairsA);
  });
});
