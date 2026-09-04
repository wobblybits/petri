import { describe, it } from 'vitest';
import { WaveguideNet, MAX_AGENTS, MAX_STUBS, MAX_AIR, MAX_WIRES } from './waveguide.ts';
import type { AgentTopo, NetTopology, WireTopo } from './types.ts';
import { planAirMessage } from './dispatch.ts';
import { AudioEngine } from './engine.ts';
import { Sim } from '../sim.ts';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { slotsFor } from '../agents.ts';

function agentSpec(id: number, stubs: boolean): AgentTopo {
  return {
    id,
    kind: (id % 3) as 0 | 1 | 2,
    openPorts: stubs ? 2 : 0,
    stubs: stubs
      ? [
          { slot: 0, length: 10, z: 1 },
          { slot: 1, length: 8, z: 0.62 },
        ]
      : undefined,
    impedance: 1,
    pan: 0,
    dist: 0.5,
    modeHz: [180, 380, 620],
    modeT60: [0.25, 0.14, 0.08],
    modeGain: [0.7, 0.4, 0.22],
    coupling: 1,
  };
}

function soupTopo(nAgents: number, nWires: number): NetTopology {
  const agents: AgentTopo[] = [];
  for (let i = 0; i < nAgents; i++) agents.push(agentSpec(i + 1, true));
  const wires: WireTopo[] = [];
  for (let i = 0; i < nWires; i++) {
    wires.push({
      id: i + 1,
      length: 80 + (i % 40),
      loss: 0.999,
      bend: 0.05,
      agentA: (i % nAgents) + 1,
      agentB: ((i + 1) % nAgents) + 1,
      zA: 1,
      zB: 1,
      damp: 0.5,
    });
  }
  return { wires, agents, height: 0.45 };
}

/**
 * 750 quanta is two seconds of audio. It used to be 80, which is a tenth of a
 * second — short enough that JIT warmup dominated and the numbers were fiction:
 * light loads read up to 4x their real cost and applyTopology read 10x. Anyone
 * optimising against those would have optimised the wrong thing.
 */
function usPerQuantum(net: WaveguideNet, quanta = 750, shed = false): number {
  const n = 128;
  // Warm up long enough for the tick loop to be compiled and settled.
  for (let i = 0; i < n * 60; i++) net.tick(shed);
  const t0 = performance.now();
  for (let q = 0; q < quanta; q++) {
    for (let i = 0; i < n; i++) net.tick(shed);
  }
  return ((performance.now() - t0) * 1000) / quanta;
}

function usPerQuantumShed(net: WaveguideNet, quanta = 750): number {
  return usPerQuantum(net, quanta, true);
}

function log(label: string, us: number): void {
  const budget = (128 / 48000) * 1e6;
  console.log(`${label}: ${us.toFixed(0)} us / 128-sample quantum (budget ${budget.toFixed(0)} us, ${(us / budget * 100).toFixed(0)}%)`);
}

describe('audio load probe', () => {
  it('prints tick cost at soup scale', () => {
    const budget = (128 / 48000) * 1e6;

    {
      const net = new WaveguideNet();
      net.handle({ type: 'topology', topo: soupTopo(2, 1) });
      net.handle({ type: 'latch', topo: soupTopo(2, 1), wireId: 1, gain: 1 });
      log('2 agents, 1 wire, ringing', usPerQuantum(net));
    }

    {
      const net = new WaveguideNet();
      net.handle({ type: 'topology', topo: soupTopo(80, 20) });
      log('80 agents silent, 20 wires quiet', usPerQuantum(net));
    }

    {
      const net = new WaveguideNet();
      const topo = soupTopo(80, 20);
      net.handle({ type: 'topology', topo });
      net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
      for (let i = 2; i <= 8; i++) net.injectPluck(i, 0.8);
      log('80 agents, 8 wires ringing', usPerQuantum(net));
    }

    {
      const net = new WaveguideNet();
      const topo = soupTopo(80, 20);
      net.handle({ type: 'topology', topo });
      const items = [];
      for (let i = 0; i < MAX_AIR; i++) {
        items.push({ agentA: i + 1, agentB: i + 2, length: 32, gain: 0.1, damp: 0.5 });
      }
      net.handle({ type: 'air', items });
      net.handle({ type: 'strike', agentId: 1, peak: 1.2, dur: 40, sharp: 0.5 });
      for (let i = 0; i < 256; i++) net.tick();
      let awake = 0;
      for (const ag of net.agents) if (ag.active && !ag.quiet) awake++;
      let airLive = 0;
      for (const ar of net.airs) if (ar.active && !ar.quiet) airLive++;
      console.log(`80 agents, 48 air, one strike: ${awake} bodies awake, ${airLive} air paths live`);
      log('80 agents, 48 air, one strike', usPerQuantum(net));
    }

    {
      const net = new WaveguideNet();
      const topo = soupTopo(200, 64);
      net.handle({ type: 'topology', topo });
      const items = [];
      for (let i = 0; i < MAX_AIR; i++) {
        items.push({ agentA: i + 1, agentB: i + 2, length: 32, gain: 0.1, damp: 0.5 });
      }
      net.handle({ type: 'air', items });
      for (let i = 1; i <= 16; i++) net.injectPluck(i, 0.6);
      for (let i = 1; i <= 30; i++) net.handle({ type: 'strike', agentId: i, peak: 0.8, dur: 30, sharp: 0.4 });
      for (let i = 0; i < 256; i++) net.tick();
      let awake = 0;
      for (const ag of net.agents) if (ag.active && !ag.quiet) awake++;
      console.log(`200 busy: ${awake} bodies awake`);
      log('200 agents, 64 wires, 48 air, busy', usPerQuantum(net, 40));
      log('200 busy, air shed', usPerQuantumShed(net, 40));
    }

    {
      const net = new WaveguideNet();
      const topo = soupTopo(80, 20);
      // Warm up first: 40 cold calls read ~10x the settled cost.
      for (let i = 0; i < 40; i++) net.handle({ type: 'topology', topo });
      const t0 = performance.now();
      for (let i = 0; i < 200; i++) net.handle({ type: 'topology', topo });
      const us = ((performance.now() - t0) * 1000) / 200;
      console.log(`applyTopology 80 agents: ${us.toFixed(0)} us (budget ${budget.toFixed(0)} us)`);
    }

    {
      const net = new WaveguideNet();
      const topo = soupTopo(200, 64);
      for (let i = 0; i < 40; i++) net.handle({ type: 'topology', topo });
      const t0 = performance.now();
      for (let i = 0; i < 200; i++) net.handle({ type: 'topology', topo });
      const us = ((performance.now() - t0) * 1000) / 200;
      console.log(`applyTopology 200 agents: ${us.toFixed(0)} us (budget ${budget.toFixed(0)} us)`);
    }

    {
      const buf = new Float32Array(4096);
      const t0 = performance.now();
      for (let i = 0; i < 2000; i++) {
        buf.fill(0);
        buf.fill(0);
      }
      const us = ((performance.now() - t0) * 1000) / 2000;
      console.log(`two Float32Array(4096).fill(0): ${us.toFixed(1)} us`);
    }

    {
      const sim = new Sim(800, 600);
      const params = defaultParams();
      params.maxAgents = 80;
      for (let i = 0; i < 80; i++) {
        sim.spawn(i % 3 === 0 ? 'era' : i % 3 === 1 ? 'dup' : 'con', 40 + (i % 10) * 50, 40 + Math.floor(i / 10) * 50, 0, params, true);
      }
      const t0 = performance.now();
      for (let i = 0; i < 200; i++) planAirMessage(sim.agents);
      console.log(`planAirMessage 80 agents: ${(((performance.now() - t0) * 1000) / 200).toFixed(0)} us`);
    }

    {
      const sim = new Sim(800, 600);
      const params = defaultParams();
      params.maxAgents = 200;
      for (let i = 0; i < 200; i++) {
        sim.spawn(i % 3 === 0 ? 'era' : i % 3 === 1 ? 'dup' : 'con', 20 + (i % 20) * 30, 20 + Math.floor(i / 20) * 30, 0, params, true);
      }
      const t0 = performance.now();
      for (let i = 0; i < 80; i++) planAirMessage(sim.agents);
      console.log(`planAirMessage 200 agents: ${(((performance.now() - t0) * 1000) / 80).toFixed(0)} us`);

      const engine = new AudioEngine();
      engine.armWithoutAudio();
      const t1 = performance.now();
      for (let i = 0; i < 80; i++) engine.frame(sim.graph, sim.agents, 1 / 60, { x: 400, y: 300, zoom: 1, viewW: 800, viewH: 600 });
      console.log(`AudioEngine.frame 200 agents: ${(((performance.now() - t1) * 1000) / 80).toFixed(0)} us`);
    }

    {
      const sim = new Sim(1600, 800);
      const params = defaultParams();
      params.spawnInterval = 0;
      params.maxAgents = 200;
      const kinds = ['era', 'dup', 'con'] as const;
      const spawned = [];
      for (let i = 0; i < 200; i++) {
        spawned.push(
          sim.spawn(kinds[i % 3], 40 + (i % 20) * 40, 40 + Math.floor(i / 20) * 36, 0, params, true)!,
        );
      }
      for (let i = 0; i < spawned.length - 1; i++) {
        const a = spawned[i];
        const b = spawned[i + 1];
        const sa = slotsFor(a.kind).find((s) => sim.graph.isFree({ id: a.id, slot: s }));
        const sb = slotsFor(b.kind).find((s) => sim.graph.isFree({ id: b.id, slot: s }));
        if (sa && sb) sim.wire(a.id, sa, b.id, sb, params);
      }
      for (let i = 0; i < 20; i++) sim.step(1 / 60, params);
      const t0 = performance.now();
      const n = 40;
      for (let i = 0; i < n; i++) sim.step(1 / 60, params);
      console.log(`Sim.step 200 wired: ${(((performance.now() - t0) * 1000) / n).toFixed(0)} us`);
    }

    {
      const sim = new Sim(1200, 800);
      const params = defaultParams();
      params.maxAgents = 200;
      params.soupCount = 200;
      params.spawnInterval = 1;
      loadPreset(sim, 'soup', params);
      for (let i = 0; i < 90; i++) sim.step(1 / 60, params);
      const t0 = performance.now();
      const n = 40;
      for (let i = 0; i < n; i++) sim.step(1 / 60, params);
      console.log(`Sim.step 200 soup: ${(((performance.now() - t0) * 1000) / n).toFixed(0)} us`);
    }

    console.log(`caps: agents=${MAX_AGENTS} wires=${MAX_WIRES} stubs=${MAX_STUBS} air=${MAX_AIR}`);
  });
});
