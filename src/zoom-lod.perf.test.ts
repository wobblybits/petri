import { describe, expect, it } from 'vitest';
import { nativeSolver } from './native/solver.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * Zooming out must never cost more than zooming in.
 *
 * It used to. A wire-stroke constant (`wiresDrawable`) gated body physics, so
 * every agent went from SAT to packed-FAR in one wheel notch at zoom 0.407 —
 * nowhere near either edge of the agent band — and the frame cost, the resting
 * spacing and the peak velocity all stepped with it. Cost is now monotone in
 * zoom because the tier is decided by apparent size alone.
 */
describe('physics LOD across zoom', () => {
  it('never costs more as the camera pulls back', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const side = 16;
    const rows: { zoom: number; p50: number; detailed: number }[] = [];
    for (const zoom of [1, 0.6, 0.5, 0.42, 0.3, 0.2, 0.1, 0.05]) {
      const sim = new Sim(2000, 2000);
      const params = defaultParams();
      params.spawnInterval = 0;
      params.upkeep = 0;
      const ids: number[][] = [];
      for (let j = 0; j < side; j++) {
        const row: number[] = [];
        for (let i = 0; i < side; i++) {
          const k = (i + j) % 3 === 0 ? 'era' : (i + j) % 3 === 1 ? 'con' : 'dup';
          const a = sim.spawn(k, 200 + i * 46, 200 + j * 46, (i * 0.7 + j) % 6.28, params);
          row.push(a ? a.id : -1);
        }
        ids.push(row);
      }
      for (let j = 0; j < side; j++)
        for (let i = 0; i + 1 < side; i++)
          if (ids[j][i] > 0 && ids[j][i + 1] > 0) sim.wire(ids[j][i], 'r', ids[j][i + 1], 'l', params);
      for (let j = 0; j + 1 < side; j++)
        for (let i = 0; i < side; i += 2)
          if (ids[j][i] > 0 && ids[j + 1][i] > 0) sim.wire(ids[j][i], 'p', ids[j + 1][i], 'p', params);
      const view = { x: 550, y: 550, zoom, viewW: 1200, viewH: 800 };
      sim.setFieldCover((view.viewW / zoom) * 1.7, (view.viewH / zoom) * 1.7);
      for (let f = 0; f < 20; f++) sim.step(1 / 60, params, view);
      const ts: number[] = [];
      let det = 0;
      for (let f = 0; f < 90; f++) {
        const t0 = performance.now();
        sim.step(1 / 60, params, view);
        ts.push(performance.now() - t0);
        for (const a of sim.agents.values()) if (sim.isPhysicsDetailed(a.id)) det++;
      }
      ts.sort((a, b) => a - b);
      rows.push({ zoom, p50: ts[45], detailed: det / 90 });
    }
    const label = rows
      .map((r) => `z${r.zoom} ${r.p50.toFixed(2)}ms/${r.detailed.toFixed(0)}det`)
      .join('  ');
    // Closest in is the most expensive frame, and it has to fit 60 fps.
    expect(rows[0].p50, label).toBeLessThan(1000 / 60);
    // Monotone, with room for measurement noise. A 20% band is far tighter
    // than the 3x step the stroke-width gate used to put at 0.407.
    for (let i = 1; i < rows.length; i++) {
      expect(rows[i].p50, `${label} (z${rows[i].zoom} vs z${rows[i - 1].zoom})`)
        .toBeLessThan(rows[i - 1].p50 * 1.2);
    }
    // And the fully pulled-back frame is genuinely cheaper than the close one.
    expect(rows[rows.length - 1].p50, label).toBeLessThan(rows[0].p50 * 0.75);
  }, 300_000);
});
