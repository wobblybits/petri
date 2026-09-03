import { describe, expect, it } from 'vitest';
import { defaultParams } from '../params.ts';
import { Sim } from '../sim.ts';
import { nativeSolver } from './solver.ts';

/**
 * What the scent field actually covers, which is not the world.
 *
 * `Fields` is a sliding window pinned to the centre of mass, and main.ts sizes
 * it at 1.7x the *camera viewport*. The column count is fixed at 160, so
 * zooming out does not widen the grid — it makes every cell bigger. Scent
 * outside the window is discarded as the window scrolls, and a resize of more
 * than 18% clears the field outright.
 *
 * Two consequences worth having written down:
 *
 *   - the field costs the same at 200 bodies and at 20,000, which is why it is
 *     ~1.3ms of a 130ms frame and why moving it to the GPU as it stands would
 *     be chasing 1% with a dispatch overhead of 24-71us per pass;
 *   - the chemistry is camera-dependent. Zoom changes cell size, so it changes
 *     what an agent smells.
 */
describe('field extent', () => {
  it('reports what the scent window actually covers', async () => {
    expect(await nativeSolver.init()).toBe(true);
    const params = defaultParams();
    params.maxAgents = 100_000;
    params.spawnInterval = 0;
    const sim = new Sim(60_000, 60_000);
    let ox = 400;
    for (let k = 0; k < 8; k++) {
      const cols = 50;
      const ids: number[] = [];
      for (let i = 0; i < 2500; i++) {
        const c = i % cols;
        const r = (i / cols) | 0;
        const kind = i % 3 === 0 ? 'era' : i % 3 === 1 ? 'con' : 'dup';
        const a = sim.spawn(kind, ox + c * 52, 400 + r * 52, 0, params);
        ids.push(a ? a.id : -1);
      }
      for (let i = 0; i < 2500; i++) {
        const c = i % cols;
        if (c + 1 < cols && i + 1 < 2500) sim.wire(ids[i], 'r', ids[i + 1], 'l', params);
      }
      ox += cols * 52 + 400;
    }
    const view = { x: 6000, y: 3000, zoom: 0.05, viewW: 1600, viewH: 900 };
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params, view);

    const f = sim.fields;
    const cellW = f.worldW / f.cols;
    const cellH = f.worldH / f.rows;
    let inside = 0;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const a of sim.agents.values()) {
      if (a.x < minX) minX = a.x;
      if (a.x > maxX) maxX = a.x;
      if (a.y < minY) minY = a.y;
      if (a.y > maxY) maxY = a.y;
      const gx = ((a.x - f.originX) / f.worldW) * f.cols;
      const gy = ((a.y - f.originY) / f.worldH) * f.rows;
      if (gx >= 0 && gx < f.cols && gy >= 0 && gy < f.rows) inside++;
    }
    const n = sim.agents.size;
    let nonzero = 0;
    for (let k = 0; k < f.data.length; k++) if (f.data[k] !== 0) nonzero++;
    console.log(
      `\nscent window\n` +
        `  grid            ${f.cols} x ${f.rows} = ${f.cols * f.rows} cells\n` +
        `  covers          ${f.worldW.toFixed(0)} x ${f.worldH.toFixed(0)} world units\n` +
        `  cell size       ${cellW.toFixed(0)} x ${cellH.toFixed(0)} world units\n` +
        `  agent diameter  ~21 world units (${(cellW / 21).toFixed(0)} agents per cell edge)\n` +
        `  pond spans      ${(maxX - minX).toFixed(0)} x ${(maxY - minY).toFixed(0)} world units\n` +
        `  bodies inside   ${inside} / ${n}  (${((inside / n) * 100).toFixed(0)}%)\n` +
        `  bodies per cell ${(inside / (f.cols * f.rows)).toFixed(1)}\n` +
        `  nonzero scent   ${nonzero} / ${f.data.length} slots\n`,
    );
    expect(n).toBeGreaterThan(0);
    // Not a target, a tripwire: if the window ever stops dwarfing the pond, or
    // cells stop being many agents wide, the tradeoffs above have changed.
    expect(inside, 'bodies fell outside the scent window').toBe(n);
  }, 900_000);
});
