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
    params.spawnInterval = 0;
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
    params.spawnInterval = 0;
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

  it('a visible knock still sounds when the frame is full of latches', () => {
    const sim = new Sim(200, 200);
    const params = defaultParams();
    params.spawnInterval = 0;
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);
    const agents = [];
    for (let i = 0; i < 12; i++) {
      agents.push(sim.spawn('era', 20 + i * 12, 40, 0, params, true)!);
    }
    for (let i = 0; i < 5; i++) {
      const a = agents[i * 2];
      const b = agents[i * 2 + 1];
      sim.wire(a.id, 'p', b.id, 'p', params);
      const wire = [...sim.graph.wires.values()].at(-1)!;
      engine.push(
        {
          type: 'latch',
          wireId: wire.id,
          agentA: a.id,
          agentB: b.id,
          slotA: 'p',
          slotB: 'p',
          kindA: 'era',
          kindB: 'era',
          rest: wire.rest,
          latchLen: wire.latchLen,
        },
        sim.graph,
        sim.agents,
      );
    }
    const hitA = agents[10];
    const hitB = agents[11];
    engine.push(
      {
        type: 'collision',
        agentA: hitA.id,
        agentB: hitB.id,
        kindA: 'era',
        kindB: 'era',
        impact: 12,
        overlap: 2,
        nx: 1,
        ny: 0,
        headingA: 0,
        headingB: Math.PI,
        spin: 0,
        effMass: 1.4,
        vN: 18,
        vT: 0,
      },
      sim.graph,
      sim.agents,
    );
    posted.length = 0;
    engine.frame(sim.graph, sim.agents, 1 / 60);
    const strikes = posted.filter((m) => m.type === 'strike');
    expect(strikes.some((m) => m.type === 'strike' && m.agentId === hitA.id)).toBe(true);
    expect(strikes.some((m) => m.type === 'strike' && m.agentId === hitB.id)).toBe(true);
  });

  it('distinct knocks in one frame all sound', () => {
    const sim = new Sim(400, 200);
    const params = defaultParams();
    params.spawnInterval = 0;
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);
    const agents = [];
    for (let i = 0; i < 16; i++) {
      agents.push(sim.spawn('era', 20 + i * 20, 80, 0, params, true)!);
    }
    for (let i = 0; i < 8; i++) {
      engine.push(
        {
          type: 'collision',
          agentA: agents[i * 2].id,
          agentB: agents[i * 2 + 1].id,
          kindA: 'era',
          kindB: 'era',
          impact: 20,
          overlap: 2,
          nx: 1,
          ny: 0,
          headingA: 0,
          headingB: Math.PI,
          spin: 0,
          effMass: 1.4,
          vN: 16,
          vT: 0,
        },
        sim.graph,
        sim.agents,
      );
    }
    posted.length = 0;
    engine.frame(sim.graph, sim.agents, 1 / 60);
    const strikes = posted.filter((m) => m.type === 'strike');
    expect(strikes).toHaveLength(16);
  });

  it('a pile-up of knocks is still finite', () => {
    const { posted } = (() => {
      const sim = new Sim(200, 200);
      const params = defaultParams();
      params.spawnInterval = 0;
      const engine = new AudioEngine();
      engine.armWithoutAudio();
      const posted: WorkletInMessage[] = [];
      engine.onPost = (m) => posted.push(m);
      const agents = [];
      for (let i = 0; i < 12; i++) {
        agents.push(sim.spawn('era', 100 + i * 0.4, 100, i, params, true)!);
      }
      for (let i = 0; i < 80; i++) {
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
    // Safety valve, not the five-event structural budget. Two strikes per knock.
    expect(strikes).toBeLessThanOrEqual(48);
  });
});

describe('sustained level', () => {
  it('a full soup does not fade out as it runs', () => {
    // Seeded: the soup auto-spawns and rewrites off Math.random, so an
    // unseeded run moved the measured RMS around the threshold and this
    // failed perhaps one time in two for no reason anyone could act on.
    const realRandom = Math.random;
    let rs = 20260831 >>> 0;
    Math.random = () => {
      rs = (rs * 1664525 + 1013904223) >>> 0;
      return rs / 4294967296;
    };
    try {
    // Regression for a real cutout: with enough agents the awake count sits at
    // the voice cap, and the old steal threshold ratcheted away until it was
    // silencing every body. The net went ~20x quieter a few seconds in and
    // never recovered. The cascade test above did not catch it, because it only
    // asked whether voices come *down* — never whether any survive.
    const sim = new Sim(700, 500);
    const params = defaultParams();
    params.spawnInterval = 0;
    // Upkeep off. This is a test of the voice manager — the fault it guards is
    // voices ratcheting themselves to silence — and the relative bound below
    // assumes a stable cast. With upkeep on, the soup starves down over the
    // nine seconds and gets quieter for reasons that have nothing to do with
    // the audio path.
    params.upkeep = 0;
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);
    engine.contacts = sim.contacts as never;
    sim.graph.onLatch = (ev) => engine.push(ev, sim.graph, sim.agents);

    // A full default soup: maxAgents is 80, and the fault needed that many.
    for (let i = 0; i < 78; i++) {
      sim.spawn(i % 3 === 0 ? 'era' : i % 3 === 1 ? 'dup' : 'con',
        40 + (i * 71) % 620, 40 + (i * 53) % 420, i, params, true);
    }

    const net = new WaveguideNet();
    const view = { x: 350, y: 250, zoom: 1, viewW: 700, viewH: 500 };
    const secRms: number[] = [];
    for (let sec = 0; sec < 9; sec++) {
      let sum = 0;
      let n = 0;
      for (let f = 0; f < 60; f++) {
        sim.step(1 / 60, params);
        posted.length = 0;
        engine.frame(sim.graph, sim.agents, 1 / 60, view);
        for (const m of posted) net.handle(m as never);
        for (let i = 0; i < 800; i++) {
          net.tick(false);
          const v = net.outDryL + net.outWetL;
          expect(Number.isFinite(v)).toBe(true);
          sum += v * v;
          n++;
        }
      }
      secRms.push(Math.sqrt(sum / n));
    }

    // Compare the settled stretch against the opening, ignoring the first
    // second while the net is still latching itself together.
    const early = Math.max(secRms[1], secRms[2], secRms[3]);
    const late = Math.max(secRms[6], secRms[7], secRms[8]);
    // A floor that means "sounding", not "as loud as it was on the day this
    // was written". The absolute number used to sit within 6% of the measured
    // level, so any change to the mix — the wire-friction bed coming out, for
    // one — failed it without the guarded fault having recurred.
    expect(early, `early RMS ${early.toFixed(4)}`).toBeGreaterThan(0.002);
    expect(late, `late RMS ${late.toFixed(4)}`).toBeGreaterThan(0.002);
    // The actual regression: it may settle as the net stops rewiring, but the
    // fault was a 20x collapse it never came back from.
    expect(late, `${early.toFixed(4)} -> ${late.toFixed(4)}`).toBeGreaterThan(early * 0.15);
    } finally {
      Math.random = realRandom;
    }
  });
});
