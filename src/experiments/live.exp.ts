import { describe, it } from 'vitest';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { seededRandom } from './harness.ts';

/*
 * How lively is the pond, and how much do formed nets still deform?
 *
 * Not a bench with the motors off — the shipped soup at the shipped defaults,
 * which is the thing anyone actually looks at.
 */
describe('liveliness', () => {
  it('measures agitation and net deformation in a shipped soup', () => {
    const realRandom = Math.random;
    Math.random = seededRandom(11);
    try {
      const params = { ...defaultParams(), soupCount: 300 };
      const sim = new Sim(1600, 1200, 512);
      loadPreset(sim, 'soup', params);
      const dt = 1 / 60;
      for (let f = 0; f < 45 * 60; f++) sim.step(dt, params);

      // Settle-state snapshot, then watch how much things actually move.
      const before = new Map<number, { x: number; y: number }>();
      for (const a of sim.agents.values()) before.set(a.id, { x: a.x, y: a.y });
      let ke = 0;
      let peak = 0;
      for (let f = 0; f < 10 * 60; f++) {
        sim.step(dt, params);
        ke += sim.kineticEnergy();
        peak = Math.max(peak, sim.peakSpeed());
      }
      // Per-wire length change over the window: how much formed structure flexes.
      let flex = 0;
      let nWire = 0;
      for (const w of sim.graph.wires.values()) {
        const A = sim.agents.get(w.a.id);
        const B = sim.agents.get(w.b.id);
        const a0 = A ? before.get(A.id) : undefined;
        const b0 = B ? before.get(B.id) : undefined;
        if (!A || !B || !a0 || !b0) continue;
        flex += Math.abs(Math.hypot(B.x - A.x, B.y - A.y) - Math.hypot(b0.x - a0.x, b0.y - a0.y));
        nWire++;
      }
      let moved = 0;
      let n = 0;
      for (const a of sim.agents.values()) {
        const p = before.get(a.id);
        if (!p) continue;
        moved += Math.hypot(a.x - p.x, a.y - p.y);
        n++;
      }
      const c = sim.census();
      console.log(
        `LIVE bodies=${sim.agents.size} wires=${sim.graph.wires.size} ` +
          `meanKE=${(ke / 600).toFixed(1)} peak=${peak.toFixed(1)} ` +
          `drift=${(moved / Math.max(1, n)).toFixed(2)} flex=${(flex / Math.max(1, nWire)).toFixed(3)} ` +
          `commutes=${c.bornMean.toFixed(2)} latches=${sim.tally.latches}`,
      );
    } finally {
      Math.random = realRandom;
    }
  });
});
