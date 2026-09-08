import { describe, it } from 'vitest';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { seededRandom } from './harness.ts';

/*
 * How many free ports does a net actually have, and how does that scale?
 *
 * The question decides whether a boundary can be a motor. Drag in this sim is
 * charged per body, so a net's total drag goes as its size; if thrust came from
 * free ports and free ports went as a net's *boundary*, terminal speed would go
 * as boundary over volume and every large organism would be nearly immobile.
 * If instead free ports go as size, a boundary motor scales and the objection
 * does not bite.
 *
 * Counting, not arguing. A Con or Dup has three ports and an Era one, a wire
 * eats two, so free = sum(ports) - 2 * wires. Reported per connected component,
 * bucketed by component size, because "a large net" is the case in question and
 * a pond average would hide it.
 *
 *     npm run experiment -- ports
 */

function componentsOf(sim: Sim): Map<number, { bodies: number; ports: number; wires: number }> {
  const parent = new Map<number, number>();
  const find = (x: number): number => {
    let r = x;
    while (parent.get(r) !== r) r = parent.get(r)!;
    while (parent.get(x) !== r) {
      const next = parent.get(x)!;
      parent.set(x, r);
      x = next;
    }
    return r;
  };
  for (const a of sim.agents.values()) parent.set(a.id, a.id);
  for (const w of sim.graph.wires.values()) {
    if (!parent.has(w.a.id) || !parent.has(w.b.id)) continue;
    const ra = find(w.a.id);
    const rb = find(w.b.id);
    if (ra !== rb) parent.set(ra, rb);
  }
  const out = new Map<number, { bodies: number; ports: number; wires: number }>();
  for (const a of sim.agents.values()) {
    const r = find(a.id);
    const c = out.get(r) ?? { bodies: 0, ports: 0, wires: 0 };
    c.bodies++;
    c.ports += a.kind === 'era' ? 1 : 3;
    out.set(r, c);
  }
  for (const w of sim.graph.wires.values()) {
    if (!parent.has(w.a.id)) continue;
    const c = out.get(find(w.a.id));
    if (c) c.wires++;
  }
  return out;
}

describe('experiment: free ports against net size', () => {
  it('counts them in a grown pond', () => {
    const realRandom = Math.random;
    Math.random = seededRandom(7);
    try {
      const params = { ...defaultParams(), soupCount: 400 };
      const sim = new Sim(1600, 1200, 512);
      loadPreset(sim, 'soup', params);
      const seconds = Number(process.env.EXP_SECONDS ?? 90);
      for (let t = 0; t < seconds * 60; t++) sim.step(1 / 60, params);

      const comps = [...componentsOf(sim).values()];
      const buckets: { lo: number; hi: number; label: string }[] = [
        { lo: 1, hi: 1, label: '1' },
        { lo: 2, hi: 3, label: '2-3' },
        { lo: 4, hi: 7, label: '4-7' },
        { lo: 8, hi: 15, label: '8-15' },
        { lo: 16, hi: 31, label: '16-31' },
        { lo: 32, hi: 63, label: '32-63' },
        { lo: 64, hi: 1e9, label: '64+' },
      ];
      console.log(
        `\npond: ${sim.agents.size} bodies, ${sim.graph.wires.size} wires, ` +
          `${comps.length} components, after ${seconds}s\n`,
      );
      console.log(
        `${'size'.padStart(6)} ${'nets'.padStart(6)} ${'mean bodies'.padStart(12)} ` +
          `${'mean free'.padStart(10)} ${'free/body'.padStart(10)} ${'wires/body'.padStart(11)}`,
      );
      for (const b of buckets) {
        const g = comps.filter((c) => c.bodies >= b.lo && c.bodies <= b.hi);
        if (g.length === 0) continue;
        const bodies = g.reduce((s, c) => s + c.bodies, 0) / g.length;
        const free = g.reduce((s, c) => s + (c.ports - 2 * c.wires), 0) / g.length;
        const wires = g.reduce((s, c) => s + c.wires, 0) / g.length;
        console.log(
          `${b.label.padStart(6)} ${String(g.length).padStart(6)} ${bodies.toFixed(1).padStart(12)} ` +
            `${free.toFixed(1).padStart(10)} ${(free / bodies).toFixed(3).padStart(10)} ` +
            `${(wires / bodies).toFixed(3).padStart(11)}`,
        );
      }
      const big = comps.sort((a, b) => b.bodies - a.bodies).slice(0, 5);
      console.log('\nlargest nets:');
      for (const c of big) {
        console.log(
          `  ${String(c.bodies).padStart(4)} bodies, ${String(c.wires).padStart(4)} wires, ` +
            `${String(c.ports - 2 * c.wires).padStart(4)} free ports ` +
            `(${((c.ports - 2 * c.wires) / c.bodies).toFixed(3)} a body)`,
        );
      }
    } finally {
      Math.random = realRandom;
    }
  });
});
