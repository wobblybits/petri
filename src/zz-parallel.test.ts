import { describe, it, expect } from 'vitest';
import { Sim } from './sim.ts';
import { defaultParams } from './params.ts';
import { AudioEngine } from './audio/engine.ts';
import { WaveguideNet } from './audio/waveguide.ts';
import { COST_US } from './audio/lod.ts';
import type { WorkletInMessage } from './audio/types.ts';

function seed(n: number): void {
  let s = n >>> 0;
  Math.random = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

/**
 * How the *awake* cost of a busy net distributes across connected components.
 *
 * This is the number component-parallelism lives or dies on. Static component
 * size is the wrong proxy: a big settled component that is silent costs
 * nothing, and a worker owning it would idle.
 */
describe('parallel headroom', () => {
  it('measures awake cost per component over a running soup', () => {
    const L: string[] = [];
    for (const [label, agents, world] of [
      ['80 agents', 80, 900],
      ['200 agents', 200, 1400],
    ] as const) {
      seed(31337);
      const sim = new Sim(world, world * 0.7);
      const params = defaultParams();
      params.maxAgents = agents;
      const engine = new AudioEngine();
      engine.armWithoutAudio();
      const posted: WorkletInMessage[] = [];
      engine.onPost = (m) => posted.push(m);
      engine.contacts = sim.contacts as never;
      sim.graph.onLatch = (ev) => engine.push(ev, sim.graph, sim.agents);
      for (let i = 0; i < agents; i++) {
        sim.spawn(i % 3 === 0 ? 'era' : i % 3 === 1 ? 'dup' : 'con',
          30 + (i * 71) % (world - 60), 30 + (i * 53) % (world * 0.7 - 60), i, params, true);
      }
      const net = new WaveguideNet();
      const view = { x: world / 2, y: world * 0.35, zoom: 1, viewW: world, viewH: world * 0.7 };

      const serialFrac: number[] = [];
      const bound4: number[] = [];
      const bound8: number[] = [];
      let totalSamples = 0;
      let sumTotal = 0;
      let sumBiggest = 0;

      for (let f = 0; f < 60 * 45; f++) {
        sim.step(1 / 60, params);
        posted.length = 0;
        engine.frame(sim.graph, sim.agents, 1 / 60, view);
        for (const m of posted) net.handle(m as never);
        for (let i = 0; i < 128; i++) net.tick();
        if (f % 30 !== 0 || f < 300) continue;

        const comps = sim.graph.componentIds(sim.agents);
        const cost = new Map<number, number>();
        let total = 0;
        for (const w of net.wires) {
          if (!w.active || w.quiet) continue;
          const wire = sim.graph.wires.get(w.wireId);
          const root = wire ? comps.get(wire.a.id) : undefined;
          const key = root ?? -1;
          cost.set(key, (cost.get(key) ?? 0) + COST_US.nearWire);
          total += COST_US.nearWire;
        }
        for (const a of net.agents) {
          if (!a.active || a.quiet) continue;
          const root = comps.get(a.id);
          const key = root ?? -1;
          cost.set(key, (cost.get(key) ?? 0) + COST_US.nearAgent);
          total += COST_US.nearAgent;
        }
        if (total <= 0) continue;
        const costs = [...cost.values()].sort((x, y) => y - x);
        const biggest = costs[0];
        serialFrac.push(biggest / total);
        bound4.push(total / Math.max(biggest, total / 4));
        bound8.push(total / Math.max(biggest, total / 8));
        sumTotal += total;
        sumBiggest += biggest;
        totalSamples++;
      }

      const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
      const pct = (xs: number[], p: number) => {
        const s = [...xs].sort((a, b) => a - b);
        return s[Math.floor(s.length * p)] ?? 0;
      };
      L.push(`${label}: ${totalSamples} samples over 45 s`);
      L.push(`  largest component's share of awake cost: mean ${(mean(serialFrac) * 100).toFixed(0)}%` +
        `, median ${(pct(serialFrac, 0.5) * 100).toFixed(0)}%, p90 ${(pct(serialFrac, 0.9) * 100).toFixed(0)}%`);
      L.push(`  cost-weighted share (what actually matters): ${((sumBiggest / sumTotal) * 100).toFixed(0)}%`);
      L.push(`  speedup bound, 4 workers: mean ${mean(bound4).toFixed(2)}x, worst ${Math.min(...bound4).toFixed(2)}x`);
      L.push(`  speedup bound, 8 workers: mean ${mean(bound8).toFixed(2)}x, worst ${Math.min(...bound8).toFixed(2)}x`);
    }
    expect.fail('\n' + L.join('\n'));
  });
}, 600000);
