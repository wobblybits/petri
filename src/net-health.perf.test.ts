import { afterEach, describe, expect, it } from 'vitest';
import { boundRadius, stemRoot, stemWorld, type Agent } from './agents.ts';
import { segmentsIntersect } from './geom.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/**
 * The minute-of-soup soak tests from net-health.test.ts. Split out because
 * they run 3600 frames across 3 seeds each — correctness checks, not frame
 * budgets, but too slow to share the parallel `suite` project with everything
 * else. See vite.config.ts for why `bench` runs its files serially.
 */

const realRandom = Math.random;
afterEach(() => {
  Math.random = realRandom;
});

function seed(n: number): void {
  let s = n >>> 0;
  Math.random = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function run(sim: Sim, params: Params, frames: number): void {
  for (let i = 0; i < frames; i++) sim.step(1 / 60, params);
}

/**
 * Signed distance of a wire's far stem from the near agent's centreline,
 * measured toward that port's own side and scaled by the port's own offset.
 * Positive means the wire stays on its own side; negative means it has crossed.
 */
function auxSideOffset(
  sim: Sim,
  agent: Agent,
  slot: 'l' | 'r',
  other: Agent,
  otherSlot: 'p' | 'l' | 'r',
): number {
  const root = stemRoot(agent.kind, slot);
  const side = root.y < 0 ? -1 : 1;
  const mx = -side * Math.sin(agent.heading);
  const my = side * Math.cos(agent.heading);
  const far = stemWorld(other, otherSlot, sim.w, sim.h);
  return ((far.x - agent.x) * mx + (far.y - agent.y) * my) / Math.abs(root.y);
}

describe('aux wires keep to their own side', () => {
  it('keeps most aux wires uncrossed across a minute of soup', () => {
    // Parallel aux axes mean nothing geometrically forbids a crossing, so this
    // is a rate, not an invariant. Aiming each aux port slightly to its own side
    // roughly halves it; holding personal space between nets costs a little of
    // that back (18.4% to 21.7%) in exchange for nets not resting on each other.
    // Fixing reconnection so rewrites stop dropping wires roughly doubled how
    // many wires a soup carries (21.5 to 40.7 on average), and denser nets
    // cross a little more often again.
    let obs = 0;
    let crossed = 0;
    for (const sd of [999, 12345, 5150]) {
      seed(sd);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      params.upkeep = 0;
      // Pinned, not inherited: the bound below is calibrated against this
      // density, so tuning the product's default soup must not quietly change
      // what this test means (or triple how long it takes).
      params.soupCount = 28;
      loadPreset(sim, 'soup', params);
      for (let f = 1; f <= 3600; f++) {
        sim.step(1 / 60, params);
        if (f % 30) continue;
        for (const wire of sim.graph.wires.values()) {
          for (const [near, nearSlot, far, farSlot] of [
            [wire.a, wire.a.slot, wire.b, wire.b.slot],
            [wire.b, wire.b.slot, wire.a, wire.a.slot],
          ] as const) {
            if (nearSlot === 'p') continue;
            const A = sim.agents.get(near.id);
            const B = sim.agents.get(far.id);
            if (!A || !B) continue;
            obs++;
            if (auxSideOffset(sim, A, nearSlot, B, farSlot) < 0) crossed++;
          }
        }
      }
    }
    const rate = crossed / Math.max(1, obs);
    expect(rate, `${(rate * 100).toFixed(1)}% of aux ends crossed (${crossed}/${obs})`).toBeLessThan(
      0.30,
    );
  });
});

describe('crowding and tangling', () => {
  it('keeps other nets out of a saturated agent\'s space', () => {
    // Flocking separation is scoped to one net, so nothing but this constraint
    // pushes separate nets apart.
    let saturated = 0;
    let invaded = 0;
    for (const sd of [999, 12345, 5150]) {
      seed(sd);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      params.upkeep = 0;
      // Pinned, not inherited: the bound below is calibrated against this
      // density, so tuning the product's default soup must not quietly change
      // what this test means (or triple how long it takes).
      params.soupCount = 28;
      loadPreset(sim, 'soup', params);
      for (let f = 1; f <= 3600; f++) {
        sim.step(1 / 60, params);
        if (f % 60) continue;
        const comp = sim.graph.componentIds(sim.agents);
        const list = [...sim.agents.values()];
        for (const a of list) {
          if (!sim.graph.portsFilled(a)) continue;
          saturated++;
          for (const b of list) {
            if (b.id === a.id || comp.get(a.id) === comp.get(b.id)) continue;
            if (Math.hypot(b.x - a.x, b.y - a.y) < params.wireMinRest - 0.5) {
              invaded++;
              break;
            }
          }
        }
      }
    }
    const rate = invaded / Math.max(1, saturated);
    // 36.6% before this existed; nothing had ever separated two different nets.
    expect(rate, `${(rate * 100).toFixed(1)}% of saturated agents crowded by another net`)
      .toBeLessThan(0.1);
  });

  it('keeps ropes out of bodies they are not attached to', () => {
    let samples = 0;
    let inside = 0;
    for (const sd of [999, 12345, 5150]) {
      seed(sd);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      params.upkeep = 0;
      // Pinned, not inherited: the bound below is calibrated against this
      // density, so tuning the product's default soup must not quietly change
      // what this test means (or triple how long it takes).
      params.soupCount = 28;
      loadPreset(sim, 'soup', params);
      for (let f = 1; f <= 3600; f++) {
        sim.step(1 / 60, params);
        if (f % 60) continue;
        for (const wire of sim.graph.wires.values()) {
          for (const agent of sim.agents.values()) {
            if (agent.id === wire.a.id || agent.id === wire.b.id) continue;
            const r = boundRadius(agent);
            for (const node of wire.nodes) {
              samples++;
              if (Math.hypot(node.x - agent.x, node.y - agent.y) < r) inside++;
            }
          }
        }
      }
    }
    const rate = inside / Math.max(1, samples);
    // 0.32% of rope-node/body pairs overlapped before wire clearance existed.
    expect(rate, `${(rate * 100).toFixed(2)}% of rope nodes inside a foreign body`)
      .toBeLessThan(0.003);
  });

  it('lets independent wires cross rather than spending the frame uncrossing them', () => {
    let crossings = 0;
    let samples = 0;
    for (const sd of [999, 12345, 5150]) {
      seed(sd);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      params.upkeep = 0;
      // Pinned, not inherited: the bound below is calibrated against this
      // density, so tuning the product's default soup must not quietly change
      // what this test means (or triple how long it takes).
      params.soupCount = 28;
      loadPreset(sim, 'soup', params);
      for (let f = 1; f <= 3600; f++) {
        sim.step(1 / 60, params);
        if (f % 30) continue;
        samples++;
        const chords = [...sim.graph.wires.values()].flatMap((w) => {
          const A = sim.agents.get(w.a.id);
          const B = sim.agents.get(w.b.id);
          if (!A || !B) return [];
          const sa = stemWorld(A, w.a.slot, sim.w, sim.h);
          const sb = stemWorld(B, w.b.slot, sim.w, sim.h);
          return [{ a: w.a.id, b: w.b.id, sa, sb }];
        });
        for (let i = 0; i < chords.length; i++) {
          for (let j = i + 1; j < chords.length; j++) {
            const S = chords[i];
            const T = chords[j];
            if (S.a === T.a || S.a === T.b || S.b === T.a || S.b === T.b) continue;
            if (segmentsIntersect(S.sa.x, S.sa.y, S.sb.x, S.sb.y, T.sa.x, T.sa.y, T.sb.x, T.sb.y)) {
              crossings++;
            }
          }
        }
      }
    }
    const rate = crossings / Math.max(1, samples);
    // The anti-tangle half. There used to be a lower bound here as well,
    // standing in for "the sim does not spend the frame uncrossing things",
    // but it read a soup that has since thinned out: 27 short chords spread
    // across a wrapping world cross zero times for reasons that have nothing
    // to do with the uncrossing force. That property is asserted directly by
    // the test below instead.
    expect(rate, `${rate.toFixed(2)} wire crossings per frame`).toBeLessThan(8);
  });
});

describe('the net actually rewrites', () => {
  it('fires rewrites in a minute of soup', () => {
    const counts: number[] = [];
    for (const s of [999, 12345, 5150]) {
      seed(s);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      params.upkeep = 0;
      // Pinned, not inherited: the bound below is calibrated against this
      // density, so tuning the product's default soup must not quietly change
      // what this test means (or triple how long it takes).
      params.soupCount = 28;
      loadPreset(sim, 'soup', params);
      let started = 0;
      const inner = (sim as unknown as { startRewrites(p: Params): void }).startRewrites.bind(sim);
      (sim as unknown as { startRewrites(p: Params): void }).startRewrites = (p: Params) => {
        const before = sim.rewrites.length;
        inner(p);
        started += Math.max(0, sim.rewrites.length - before);
      };
      run(sim, params, 3600);
      counts.push(started);
    }
    const total = counts.reduce((a, b) => a + b, 0);
    expect(total, `rewrites per seed over 60 s: ${counts.join(', ')}`).toBeGreaterThanOrEqual(9);
  });
});
