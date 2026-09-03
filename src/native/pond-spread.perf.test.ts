import { describe, expect, it } from 'vitest';
import { FIELD_EXTENT } from '../fields.ts';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { nativeSolver } from './solver.ts';

/**
 * A soup has to settle, not fly apart.
 *
 * It did fly apart: measured in the browser, a running pond reached a spread
 * of 187,768 world units with a median body speed of 22,239 px/s. Two bugs,
 * both in the field:
 *
 *   - the deposit normalisation constant was 20 in the solver and 10 in the
 *     twin, so the native path laid four times the scent the reference did;
 *   - the resident scent buffer was only synchronised over the live box, so
 *     everything outside it still held whatever the previous Sim left there.
 *
 * The second only bites a process holding several Sims, which is the test
 * suite rather than the app — the determinism harness caught it as a run
 * failing to reproduce itself. The first bit everywhere.
 *
 * This is the shape of the guard: spread rises while the soup expands out of
 * its spawn box, then turns over and holds as the world bound catches it, and
 * speeds settle to tens of px/s rather than thousands.
 */
describe('pond spread', () => {
  it('settles instead of flying apart', async () => {
    expect(await nativeSolver.init()).toBe(true);
    const params = defaultParams();
    const sim = new Sim(800, 600);
    loadPreset(sim, 'soup', params);
    const view = { x: 400, y: 300, zoom: 0.4, viewW: 1200, viewH: 800 };
    const marks: string[] = [];
    for (let f = 1; f <= 1800; f++) {
      sim.step(1 / 60, params, view);
      if (f % 300 === 0) {
        let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
        const vs: number[] = [];
        for (const a of sim.agents.values()) {
          mnx = Math.min(mnx, a.x); mxx = Math.max(mxx, a.x);
          mny = Math.min(mny, a.y); mxy = Math.max(mxy, a.y);
          vs.push(Math.hypot(a.vx, a.vy));
        }
        vs.sort((p, q) => p - q);
        marks.push(
          `  ${String(f / 60).padStart(2)}s  n=${String(sim.agents.size).padStart(4)}` +
            `  wires=${String(sim.graph.wires.size).padStart(4)}` +
            `  spread ${(mxx - mnx).toFixed(0).padStart(7)}x${(mxy - mny).toFixed(0).padStart(6)}` +
            `  medV ${vs[vs.length >> 1].toFixed(1).padStart(8)}`,
        );
      }
    }
    console.log(`\nsoup over 30s\n${marks.join('\n')}\n`);
    expect(marks.length).toBe(6);
    let mnx = Infinity, mxx = -Infinity;
    const vs: number[] = [];
    for (const a of sim.agents.values()) {
      mnx = Math.min(mnx, a.x); mxx = Math.max(mxx, a.x);
      vs.push(Math.hypot(a.vx, a.vy));
    }
    vs.sort((p, q) => p - q);
    const medV = vs[vs.length >> 1];
    // Generous by an order of magnitude either way: this is a tripwire for a
    // pond coming apart, not a pin on where it settles.
    expect(medV, `median speed ${medV.toFixed(0)} px/s`).toBeLessThan(400);
    expect(mxx - mnx, `spread ${(mxx - mnx).toFixed(0)}`).toBeLessThan(FIELD_EXTENT * 3);
  }, 900_000);
});
