import { describe, expect, it } from 'vitest';
import { stemWorld } from './agents.ts';
import { nativeSolver } from './native/solver.ts';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

const far = { x: 400, y: 300, zoom: 0.05, viewW: 800, viewH: 600 };
const close = { x: 400, y: 300, zoom: 2, viewW: 800, viewH: 600 };

function bbox(sim: Sim) {
  const agents = [...sim.agents.values()];
  const xs = agents.map((a) => a.x);
  const ys = agents.map((a) => a.y);
  return { w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
}

/**
 * `frozen` holds the topology still so the two runs differ only in solver
 * tier. Without it the oscillator commutes, the far and close runs end up as
 * different nets — six bodies against eight — and comparing their geometry
 * compares two different machines. Commuting at far zoom has its own test.
 */
function run(view: typeof far, frames = 180, frozen = false) {
  const sim = new Sim(800, 600);
  const params = defaultParams();
  params.spawnInterval = 0;
  params.upkeep = 0;
  params.wireShrink = 0.9;
  if (frozen) params.rewriteDuration = 0;
  loadPreset(sim, 'oscillator', params);
  sim.setViewExtent((view.viewW / view.zoom) * 1.7, (view.viewH / view.zoom) * 1.7);
  let peakSpeed = 0;
  let peakLastLen = 0;
  for (let f = 0; f < frames; f++) {
    sim.step(1 / 60, params, view);
    for (const a of sim.agents.values()) {
      const s = Math.hypot(a.vx, a.vy);
      if (Number.isFinite(s)) peakSpeed = Math.max(peakSpeed, s);
    }
    for (const w of sim.graph.wires.values()) {
      if (Number.isFinite(w.lastLen)) peakLastLen = Math.max(peakLastLen, w.lastLen);
    }
  }
  return { sim, peakSpeed, peakLastLen };
}

function auxBodyGap(sim: Sim): number | null {
  for (const w of sim.graph.wires.values()) {
    if (w.a.slot === 'p' || w.b.slot === 'p') continue;
    const A = sim.agents.get(w.a.id)!;
    const B = sim.agents.get(w.b.id)!;
    if (A.kind === 'era' || B.kind === 'era') continue;
    return Math.hypot(B.x - A.x, B.y - A.y);
  }
  return null;
}

describe('oscillator far zoom', () => {
  it('settles the same net to the same shape as a close-up run', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const closeSim = run(close, 180, true);
    const farRun = run(far, 180, true);
    const closeBox = bbox(closeSim.sim);
    const farBox = bbox(farRun.sim);
    // Nothing is stretched past breaking. Deliberately not `|stem - rest| < 8`:
    // rest is a moving setpoint that shrinks from the latch length toward
    // wireMinRest, and the span is a compliant spring, so the geometry lags it
    // at every zoom — a close-up run shows the same 13-25 px gaps on a loaded
    // member as a far one. What is worth asserting is that no wire is being
    // hauled, which is what a tier mismatch actually looks like.
    for (const w of farRun.sim.graph.wires.values()) {
      if (w.rest < 8 || w.collapse > 0) continue;
      const A = farRun.sim.agents.get(w.a.id)!;
      const B = farRun.sim.agents.get(w.b.id)!;
      const sa = stemWorld(A, w.a.slot, farRun.sim.w, farRun.sim.h);
      const sb = stemWorld(B, w.b.slot, farRun.sim.w, farRun.sim.h);
      const stem = Math.hypot(sb.x - sa.x, sb.y - sa.y);
      const cap = Math.max(w.rest * 2, w.rest + 40);
      expect(stem, `stem ${stem.toFixed(1)} vs rest ${w.rest.toFixed(1)}`).toBeLessThan(cap);
    }
    // Not crumpled: the machine is still at least a wire across. The two runs
    // diverge in detail — this is a chaotic oscillator and the two tiers
    // commute at different frames — so the extents are not compared directly.
    // The invariant that does have to hold between tiers is the settled pair
    // spacing below, which is local and does not depend on trajectory.
    const farSpan = Math.max(farBox.w, farBox.h);
    const label = `far ${farBox.w.toFixed(1)}x${farBox.h.toFixed(1)} close ${closeBox.w.toFixed(1)}x${closeBox.h.toFixed(1)}`;
    expect(farSpan, label).toBeGreaterThan(defaultParams().wireMinRest);
    const closeAux = auxBodyGap(closeSim.sim);
    const farAux = auxBodyGap(farRun.sim);
    expect(closeAux).not.toBeNull();
    expect(farAux).not.toBeNull();
    // The cheap tier used to hold this pair apart at the SAT bound (36 px)
    // where SAT settles it near 22, so a zoom-out visibly inflated the net.
    expect(
      Math.abs(farAux! - closeAux!),
      `far aux gap ${farAux!.toFixed(1)} vs close ${closeAux!.toFixed(1)}`,
    ).toBeLessThan(10);
    expect(farRun.peakSpeed, `peak speed ${farRun.peakSpeed.toFixed(1)}`).toBeLessThan(4000);
    expect(farRun.peakLastLen, `peak lastLen ${farRun.peakLastLen.toFixed(1)}`).toBeLessThan(2000);
  });

  it('native packed FAR at zoom 0.05 stays finite for 8s', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const farRun = run(far, 480);
    expect(farRun.peakSpeed, `peak speed ${farRun.peakSpeed.toFixed(1)}`).toBeLessThan(4000);
    expect(farRun.peakLastLen, `peak lastLen ${farRun.peakLastLen.toFixed(1)}`).toBeLessThan(2000);
    const box = bbox(farRun.sim);
    expect(Math.max(box.w, box.h), `bbox ${box.w.toFixed(1)}x${box.h.toFixed(1)}`).toBeGreaterThan(20);
    for (const a of farRun.sim.agents.values()) {
      expect(Number.isFinite(a.x + a.y + a.vx + a.vy)).toBe(true);
    }
  });

  it('still commutes when fully zoomed out', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = defaultParams();
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.wireShrink = 0.08;
    params.rewriteDuration = 0.12;
    const sim = new Sim(800, 600);
    loadPreset(sim, 'oscillator', params);
    sim.setViewExtent((far.viewW / far.zoom) * 1.7, (far.viewH / far.zoom) * 1.7);
    const start = sim.agents.size;
    let rewrote = false;
    for (let f = 0; f < 120; f++) {
      sim.step(1 / 60, params, far);
      if (sim.rewrites.length > 0 || sim.agents.size !== start) {
        rewrote = true;
        break;
      }
    }
    expect(rewrote).toBe(true);
  });
});
