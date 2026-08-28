import { describe, expect, it } from 'vitest';
import { Sim } from '../sim.ts';
import { defaultParams } from '../params.ts';
import { AudioEngine } from './engine.ts';
import { WaveguideNet } from './waveguide.ts';
import type { WorkletInMessage } from './types.ts';

/**
 * End-to-end: run the real simulation, let the real engine plan messages, and
 * render them through the real waveguide. This is the path the browser takes,
 * minus Web Audio itself.
 */

describe('sim to sound', () => {
  it('a running soup makes sound, stays in range, and never runs away', () => {
    const sim = new Sim(360, 260);
    const params = defaultParams();
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);
    sim.graph.onLatch = (ev) => engine.push(ev, sim.graph, sim.agents);

    for (let i = 0; i < 18; i++) {
      sim.spawn(i % 3 === 0 ? 'era' : i % 3 === 1 ? 'dup' : 'con', 40 + (i * 37) % 300, 40 + (i * 61) % 200, i, params, true);
    }

    const net = new WaveguideNet();
    const dt = 1 / 60;
    let peak = 0;
    let sum = 0;
    let n = 0;
    let heard = 0;
    for (let i = 0; i < 60 * 12; i++) {
      sim.step(dt, params);
      posted.length = 0;
      engine.frame(sim.graph, sim.agents, dt);
      for (const m of posted) {
        if (m.type !== 'topology' && m.type !== 'gain') heard++;
        net.handle(m as never);
      }
      for (let s = 0; s < 800; s++) {
        const v = net.tick();
        const a = v < 0 ? -v : v;
        if (a > peak) peak = a;
        sum += a;
        n++;
        expect(Number.isFinite(v)).toBe(true);
      }
    }

    expect(heard).toBeGreaterThan(0);
    // Audible, but the output stage keeps it inside the rails.
    expect(peak).toBeGreaterThan(0.02);
    expect(peak).toBeLessThanOrEqual(1);
    // A dense net must not sit pinned at the limiter.
    expect(sum / n).toBeLessThan(0.35);
  });

  it('goes silent after the sim stops feeding it', () => {
    const sim = new Sim(360, 260);
    const params = defaultParams();
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);
    sim.graph.onLatch = (ev) => engine.push(ev, sim.graph, sim.agents);
    for (let i = 0; i < 10; i++) {
      sim.spawn(i % 2 ? 'era' : 'dup', 60 + i * 25, 90 + (i % 3) * 30, i, params, true);
    }

    const net = new WaveguideNet();
    for (let i = 0; i < 60 * 6; i++) {
      sim.step(1 / 60, params);
      posted.length = 0;
      engine.frame(sim.graph, sim.agents, 1 / 60);
      for (const m of posted) net.handle(m as never);
      for (let s = 0; s < 800; s++) net.tick();
    }

    // Stop the sim entirely. No new events, no topology updates.
    for (let s = 0; s < 48000 * 20; s++) net.tick();
    expect(net.energy()).toBe(0);
    let tail = 0;
    for (let s = 0; s < 48000; s++) {
      const v = Math.abs(net.tick());
      if (v > tail) tail = v;
    }
    // No drone, no self-oscillation, nothing left ringing.
    expect(tail).toBeLessThan(1e-10);
  });

  it('caps how many excitations a single busy frame can fire', () => {
    const { posted } = (() => {
      const sim = new Sim(200, 200);
      const params = defaultParams();
      const engine = new AudioEngine();
      engine.armWithoutAudio();
      const posted: WorkletInMessage[] = [];
      engine.onPost = (m) => posted.push(m);
      const agents = [];
      for (let i = 0; i < 12; i++) {
        agents.push(sim.spawn('era', 100 + i * 0.4, 100, i, params, true)!);
      }
      for (let i = 0; i < 40; i++) {
        engine.push(
          { type: 'collision', agentA: agents[i % 12].id, agentB: agents[(i + 1) % 12].id,
            kindA: 'era', kindB: 'era', impact: 30 + i, overlap: 2, nx: 1, ny: 0,
            headingA: i, headingB: -i, spin: 0, effMass: 1.4, vN: 30 + i, vT: 4 },
          sim.graph, sim.agents,
        );
      }
      posted.length = 0;
      engine.frame(sim.graph, sim.agents, 1 / 60);
      return { posted };
    })();
    const strikes = posted.filter((m) => m.type === 'strike').length;
    expect(strikes).toBeGreaterThan(0);
    // 5 events per frame, 2 strike messages each.
    expect(strikes).toBeLessThanOrEqual(10);
  });
});
