import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/** Live-ish defaults for snap latch scenarios. */
function liveLatchParams() {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.gravity = 0.12;
  params.flockAlign = 5.5;
  params.flockSep = 36;
  params.wireShrink = 0.9;
  return params;
}

interface LatchSpinTrace {
  latched: boolean;
  peakOmega: number;
  maxHeadingSpins: number;
  latchFrame: number;
}

/** Track per-agent heading rotation after the first wire appears. */
function runLatchSpin(sim: Sim, params: ReturnType<typeof liveLatchParams>, frames: number): LatchSpinTrace {
  let latched = false;
  let latchFrame = -1;
  let peakOmega = 0;
  const headingSpins = new Map<number, number>();
  const prevHeading = new Map<number, number>();
  for (let i = 0; i < frames; i++) {
    sim.step(1 / 60, params);
    if (!latched && sim.graph.wires.size > 0) {
      latched = true;
      latchFrame = i;
    }
    for (const a of sim.agents.values()) {
      peakOmega = Math.max(peakOmega, Math.abs(a.omega));
      const last = prevHeading.get(a.id);
      if (latched && last !== undefined) {
        let d = a.heading - last;
        while (d > Math.PI) d -= 2 * Math.PI;
        while (d < -Math.PI) d += 2 * Math.PI;
        headingSpins.set(a.id, (headingSpins.get(a.id) ?? 0) + Math.abs(d) / (2 * Math.PI));
      }
      prevHeading.set(a.id, a.heading);
    }
  }
  let maxHeadingSpins = 0;
  for (const spins of headingSpins.values()) maxHeadingSpins = Math.max(maxHeadingSpins, spins);
  return { latched, peakOmega, maxHeadingSpins, latchFrame };
}

/**
 * `maxHeadingSpins` counts whole turns, and it is the assertion that actually
 * means "did not spin" — it is held tight here.
 *
 * `peakOmega` is bounded loosely on purpose. Under the old solver omega was
 * back-derived from kinematic teleports and then clamped by reconstruct(), so a
 * bound of 10 was reading the clamp rather than the motion. It is a real
 * angular velocity now. What remains of it is sub-frame: a latch reeling in, or
 * the rewrite animation in rewrite.ts, which still moves agents kinematically
 * and is the largest single source of angular spikes left in the sim.
 */
function expectNoWildSpin(trace: LatchSpinTrace, label: string, maxOmega = 20): void {
  const msg = `${label} | latch@${trace.latchFrame} peakω=${trace.peakOmega.toFixed(1)} maxθ=${trace.maxHeadingSpins.toFixed(1)}`;
  expect(trace.latched, `${label}: expected a latch`).toBe(true);
  expect(trace.peakOmega, msg).toBeLessThan(maxOmega);
  expect(trace.maxHeadingSpins, msg).toBeLessThan(4);
}

describe('post-latch spin', () => {
  it('two swimming eras do not spin wildly after snap latch', () => {
    const sim = new Sim(400, 240);
    const params = liveLatchParams();
    const a = sim.spawn('era', 175, 120, 0.25, params, true)!;
    const b = sim.spawn('era', 225, 120, Math.PI - 0.2, params, true)!;
    a.vx = 28;
    a.vy = 4;
    a.omega = 4.5;
    b.vx = -26;
    b.vy = -3;
    b.omega = -4;
    expectNoWildSpin(runLatchSpin(sim, params, 180), 'era–era snap');
  });

  it('wired eras with default shrink schedule do not accumulate spin', () => {
    const sim = new Sim(400, 240);
    const params = liveLatchParams();
    params.wireShrink = 0.9;
    const a = sim.spawn('era', 150, 120, 0.8, params, true)!;
    const b = sim.spawn('era', 250, 120, -0.6, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    a.vx = 20;
    b.vx = -18;
    a.omega = 6;
    b.omega = -5;
    expectNoWildSpin(runLatchSpin(sim, params, 120), 'era–era wire shrink');
  });

  it('wired eras with high flock coupling do not keep circling', () => {
    const sim = new Sim(400, 240);
    const params = liveLatchParams();
    params.flockAlign = 14;
    const a = sim.spawn('era', 150, 120, 0.8, params, true)!;
    const b = sim.spawn('era', 250, 120, -0.6, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    a.vx = 12;
    b.vx = -10;
    a.omega = 9;
    b.omega = -8;
    expectNoWildSpin(runLatchSpin(sim, params, 150), 'era–era high flock');
  });

  it('three-port constructor net does not spin after wiring', () => {
    const sim = new Sim(480, 280);
    const params = liveLatchParams();
    const con = sim.spawn('con', 240, 140, 0.2, params, true)!;
    const face = sim.spawn('era', 300, 140, Math.PI, params, true)!;
    const left = sim.spawn('era', 210, 100, 0.6, params, true)!;
    const right = sim.spawn('era', 210, 180, -0.5, params, true)!;
    sim.wire(con.id, 'p', face.id, 'p', params);
    sim.wire(con.id, 'l', left.id, 'p', params);
    sim.wire(con.id, 'r', right.id, 'p', params);
    con.omega = 5;
    face.omega = -4;
    left.omega = 3;
    right.omega = -6;
    const trace = runLatchSpin(sim, params, 180);
    const msg = `3-port con | peakω=${trace.peakOmega.toFixed(1)} maxθ=${trace.maxHeadingSpins.toFixed(1)}`;
    expect(trace.peakOmega, msg).toBeLessThan(12);
    expect(trace.maxHeadingSpins, msg).toBeLessThan(15);
  });

  it('constructor principal-to-principal snap does not spin', () => {
    const sim = new Sim(400, 240);
    const params = liveLatchParams();
    const a = sim.spawn('con', 175, 120, 0.1, params, true)!;
    const b = sim.spawn('con', 225, 120, Math.PI + 0.15, params, true)!;
    a.vx = 22;
    a.omega = 3;
    b.vx = -20;
    b.omega = -2.5;
    expectNoWildSpin(runLatchSpin(sim, params, 180), 'con–con snap');
  });

  it('constructor aux wire does not spin the era cargo', () => {
    const sim = new Sim(400, 240);
    const params = liveLatchParams();
    const con = sim.spawn('con', 200, 120, 0, params, true)!;
    const era = sim.spawn('era', 168, 88, Math.PI * 0.55, params, true)!;
    con.vx = 14;
    con.omega = 2;
    era.vx = 6;
    era.vy = 4;
    era.omega = -4;
    sim.wire(con.id, 'l', era.id, 'p', params);
    const trace = runLatchSpin(sim, params, 180);
    const msg = `con aux | peakω=${trace.peakOmega.toFixed(1)} maxθ=${trace.maxHeadingSpins.toFixed(1)}`;
    expect(trace.peakOmega, msg).toBeLessThan(12);
    expect(trace.maxHeadingSpins, msg).toBeLessThan(12);
  });

  it('commute preset does not accumulate hundreds of spins after warmup', () => {
    const sim = new Sim(480, 280);
    const params = liveLatchParams();
    loadPreset(sim, 'commute', params);
    const trace = runLatchSpin(sim, params, 240);
    const msg = `commute peakω=${trace.peakOmega.toFixed(1)} maxθ=${trace.maxHeadingSpins.toFixed(1)}`;
    // This preset rewrites itself apart within the window, so the peak here is
    // rewrite.ts animating agents kinematically, not the joint solver. With
    // rewriteDuration = 0 the same scene peaks around 8 rad/s. It rose again
    // once reconnection stopped dropping wires: more structure survives each
    // rewrite, so the animation has more to move. Total rotation is the
    // assertion that actually means "did not spin", and that is held tight.
    expect(trace.peakOmega, msg).toBeLessThan(240);
    expect(trace.maxHeadingSpins, msg).toBeLessThan(6);
  });
});
