import { describe, expect, it } from 'vitest';
import { Sim } from './sim.ts';
import { defaultParams, type Params } from './params.ts';
import type { Agent, PortRef } from './agents.ts';
import type { Wire } from './graph.ts';
import { slotsFor } from './agents.ts';

/**
 * The latch crossing test is indexed: `snap` bins every wire's box and a chord
 * only walks the boxes it reaches. The index may change how many wires get
 * examined and nothing else, so the guard here is differential -- indexed
 * against the scan over the whole map, on a pond messy enough to have real
 * crossings in it.
 *
 * `latchCrosses` scans everything when no index is live, which is what makes
 * the comparison possible: the same method is the reference and the subject
 * depending only on whether an index has been built.
 */
function messyPond(): { sim: Sim; params: Params } {
  const params = defaultParams();
  params.stepSpeed = 0;
  const sim = new Sim(400, 300);
  let seed = 987654321;
  const rnd = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const kinds = ['era', 'dup', 'con'] as const;
  for (let i = 0; i < 120; i++) {
    sim.spawn(kinds[i % 3], rnd() * 400, rnd() * 300, rnd() * 6.28, params, true);
  }
  // Let it latch into something with wires lying across each other.
  for (let i = 0; i < 40; i++) sim.step(1 / 60, params);
  return { sim, params };
}

function freePorts(sim: Sim): PortRef[] {
  const out: PortRef[] = [];
  for (const agent of sim.agents.values()) {
    for (const slot of slotsFor(agent.kind)) {
      const ref: PortRef = { id: agent.id, slot };
      if (sim.graph.isFree(ref)) out.push(ref);
    }
  }
  return out;
}

/** The index is private; a differential test is exactly when reaching in is right. */
interface Indexable {
  buildLatchIndex(
    wires: Wire[],
    endA: readonly (Agent | undefined)[],
    endB: readonly (Agent | undefined)[],
    w: number,
    h: number,
  ): void;
  latchIndexed: boolean;
  latchWires: Wire[];
}

/**
 * The resolved wire list `Sim.latchPass` hands the index, reached the same
 * way. Building the index is the subject here, and it needs what the frame
 * gives it; going through `latchPass` instead would also latch, which would
 * move the pond out from under the comparison.
 */
interface Resolvable {
  wireListResolved(): Wire[];
  wirePack: Wire[];
  wireEndA: readonly (Agent | undefined)[];
  wireEndB: readonly (Agent | undefined)[];
}

describe('latch crossing index', () => {
  it('blocks exactly the chords the full scan blocks', () => {
    const { sim } = messyPond();
    expect(sim.graph.wires.size).toBeGreaterThan(10);
    const ports = freePorts(sim);
    expect(ports.length).toBeGreaterThan(10);

    const pairs: [PortRef, PortRef][] = [];
    for (let i = 0; i < ports.length; i++) {
      for (let j = i + 1; j < ports.length; j++) {
        if (ports[i].id !== ports[j].id) pairs.push([ports[i], ports[j]]);
      }
    }
    expect(pairs.length).toBeGreaterThan(100);

    const inner = sim.graph as unknown as Indexable;
    const ask = (): boolean[] =>
      pairs.map(([a, b]) => sim.graph.latchCrosses(sim.agents, a, b, sim.w, sim.h));

    inner.latchIndexed = false;
    const plain = ask();
    const outer = sim as unknown as Resolvable;
    outer.wireListResolved();
    inner.buildLatchIndex(outer.wirePack, outer.wireEndA, outer.wireEndB, sim.w, sim.h);
    // An index that came out empty agrees with nothing and disagrees with
    // everything, which reads as hundreds of wrong answers rather than as the
    // one thing that went wrong. Say it plainly instead.
    expect(inner.latchWires.length, 'the index came out empty').toBe(sim.graph.wires.size);
    const indexed = ask();
    inner.latchIndexed = false;

    // Both saying "nothing crosses" everywhere would agree and prove nothing.
    const blocked = plain.filter(Boolean).length;
    expect(blocked, 'no chord was blocked, so the comparison is vacuous')
      .toBeGreaterThan(0);
    const disagreements = indexed.filter((v, i) => v !== plain[i]).length;
    expect(disagreements, `${disagreements} of ${pairs.length} pairs disagreed`).toBe(0);
  });

  it('still scans every wire when nothing has been indexed', () => {
    // A wall between two bodies, the case sim.test.ts pins end to end. Called
    // from outside `snap` there is no index, so this is the fallback path.
    const params = defaultParams();
    params.stepSpeed = 0;
    const sim = new Sim(400, 240);
    const wallA = sim.spawn('era', 200, 40, Math.PI / 2, params, true)!;
    const wallB = sim.spawn('era', 200, 200, -Math.PI / 2, params, true)!;
    sim.wire(wallA.id, 'p', wallB.id, 'p', params);
    const left = sim.spawn('era', 140, 120, 0, params, true)!;
    const right = sim.spawn('era', 260, 120, Math.PI, params, true)!;
    expect(
      sim.graph.latchCrosses(
        sim.agents,
        { id: left.id, slot: 'p' },
        { id: right.id, slot: 'p' },
        sim.w,
        sim.h,
      ),
    ).toBe(true);
  });
});
