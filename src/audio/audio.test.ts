import { describe, expect, it } from 'vitest';
import { Sim } from '../sim.ts';
import { defaultParams } from '../params.ts';
import workletSrc from './worklet/net-processor.ts?raw';
import {
  bendLoss,
  collisionGain,
  delaySamplesForPath,
  delaySamplesForHz,
  latchGain,
  contactPeak,
  contactSeconds,
  MAX_DELAY,
  MAX_WIRES,
  portImpedance,
  tautBrighten,
  tautness,
  bowSpeed,
  setWaveSpeed,
  waveSpeed,
  wirePitchHz,
  airGain,
  airDelaySamples,
  AIR_MIN_PX,
  AIR_CUTOFF_PX,
  MAX_AIR_PATHS,
  MAX_AIR_DELAY,
  MAX_STUBS,
  MAX_STUB_DELAY,
  distanceDry,
  distanceWet,
  distanceDamp,
  distanceRoom,
} from './presets.ts';
import { planCollisionMessages, planLatchMessages, planRewriteMessages, planAirMessage } from './dispatch.ts';
import { bodyTone } from './body.ts';
import { buildTopology } from './topology.ts';
import { AudioEngine, audio } from './engine.ts';
import {
  distDry,
  distWet,
  distRoom,
  distDamp,
  MAX_DELAY as WG_MAX_DELAY,
  MAX_WIRES as WG_MAX_WIRES,
  MAX_AIR as WG_MAX_AIR,
  MAX_AIR_DELAY as WG_MAX_AIR_DELAY,
  MAX_STUBS as WG_MAX_STUBS,
  MAX_STUB_DELAY as WG_MAX_STUB_DELAY,
  distDry as WG_distDry,
  distWet as WG_distWet,
  distDamp as WG_distDamp,
  distRoom as WG_distRoom,
  WaveguideNet,
} from './waveguide.ts';
import type { LatchEvent, WorkletInMessage } from './types.ts';
import { albedo, blendVoices, sharpness, strikeSharpness, voiceFromAgent } from './voice.ts';

/** Two unwired bodies that can touch each other. */
function sampleContactTopo() {
  const one = sampleAgentTopo(1);
  const two = sampleAgentTopo(2);
  return { wires: [], agents: [one.agents[0], two.agents[0]] };
}

/** One agent, no wires: the case that used to be silent. */
function sampleAgentTopo(id: number) {
  return {
    wires: [],
    agents: [
      {
        id,
        kind: 0 as const,
        openPorts: 3,
        impedance: 1,
        pan: 0,
        modeHz: [283, 591, 972],
        modeT60: [0.25, 0.12, 0.08],
        modeGain: [1, 0.42, 0.26],
        coupling: 1,
      },
    ],
  };
}

function latchEv(wire: { id: number; rest: number; latchLen: number }, a: number, b: number): LatchEvent {
  return {
    type: 'latch',
    wireId: wire.id,
    agentA: a,
    agentB: b,
    slotA: 'p',
    slotB: 'p',
    kindA: 'era',
    kindB: 'dup',
    rest: wire.rest,
    latchLen: wire.latchLen,
  };
}

describe('worklet bundle', () => {
  it('processor is a thin wrapper around waveguide.ts', () => {
    expect(workletSrc).toContain('registerProcessor');
    expect(workletSrc).toContain("from '../waveguide.ts'");
  });
});

describe('audio dispatch', () => {
  it('latch plan emits a single atomic latch message with topology', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 30, 0, params, true)!;
    const b = sim.spawn('dup', 90, 30, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    const msgs = planLatchMessages(latchEv(wire, a.id, b.id), sim.graph, sim.agents);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].type).toBe('latch');
  });

  it('collision plan sends a contact force to each body', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 30, 0, params, true)!;
    const b = sim.spawn('con', 90, 30, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const msgs = planCollisionMessages(
      {
        type: 'collision',
        agentA: a.id,
        agentB: b.id,
        kindA: 'era',
        kindB: 'con',
        impact: 40,
        overlap: 3,
        nx: 1,
        ny: 0,
        headingA: 0,
        headingB: Math.PI,
        spin: 0,
        effMass: 1.4,
        vN: 40,
        vT: 6,
      },
    );
    expect(msgs.every((m) => m.type === 'strike')).toBe(true);
    expect(msgs).toHaveLength(2);
  });

  it('maps rope path onto travel-time delay', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 30, 0, params, true)!;
    const b = sim.spawn('dup', 90, 30, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    wire.rest = 36;
    wire.ropeLen = 36;
    wire.lastLen = 36;
    const topo = buildTopology(sim.graph, sim.agents);
    const voice = blendVoices(voiceFromAgent(a, 'p'), voiceFromAgent(b, 'p'));
    expect(topo.wires[0].length).toBeCloseTo(delaySamplesForPath(36, 0, voice.disp), 5);
    expect(topo.wires[0].zA).toBeGreaterThan(0);
    expect(topo.wires[0].zB).toBeGreaterThan(0);
    expect(topo.wires[0].pan).toBeGreaterThanOrEqual(-1);
    expect(topo.wires[0].pan).toBeLessThanOrEqual(1);
    const dup = topo.agents.find((ag) => ag.id === b.id)!;
    expect(dup.stubs?.length).toBe(2);
    const era = topo.agents.find((ag) => ag.id === a.id)!;
    expect(era.stubs?.length ?? 0).toBe(0);
  });

  it('a taut rope is brighter than a slack one', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 30, 0, params, true)!;
    const b = sim.spawn('dup', 90, 30, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    wire.rest = 40;
    wire.ropeLen = 40;
    wire.lastLen = 40;
    const slack = buildTopology(sim.graph, sim.agents);
    wire.lastLen = 52;
    const taut = buildTopology(sim.graph, sim.agents);
    expect(taut.wires[0].damp ?? 0).toBeGreaterThan(slack.wires[0].damp ?? 0);
  });

  it('a draped rope retunes from its live path, not the rest cubic', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 30, 0, params, true)!;
    const b = sim.spawn('dup', 90, 30, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    wire.rest = 40;
    wire.ropeLen = 40;
    wire.lastLen = 40;
    const rest = buildTopology(sim.graph, sim.agents);
    wire.ropeLen = 52;
    wire.lastLen = 52;
    const draped = buildTopology(sim.graph, sim.agents);
    expect(draped.wires[0].length).toBeGreaterThan(rest.wires[0].length);
  });

  it('a yank does not fully cancel the pitch drop from extra length', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 30, 0, params, true)!;
    const b = sim.spawn('dup', 90, 30, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    wire.rest = 40;
    wire.ropeLen = 40;
    wire.lastLen = 40;
    const slack = buildTopology(sim.graph, sim.agents);
    wire.lastLen = 52;
    const yank = buildTopology(sim.graph, sim.agents);
    expect(yank.wires[0].length).toBeGreaterThan(slack.wires[0].length * 0.95);
    expect(yank.wires[0].damp ?? 0).toBeGreaterThan(slack.wires[0].damp ?? 0);
  });

  it('spreads stereo when the camera zooms in', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 30, 0, params, true)!;
    const b = sim.spawn('dup', 90, 30, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const view = { x: 60, y: 30, zoom: 1, viewW: 400, viewH: 300 };
    const wide = buildTopology(sim.graph, sim.agents, view);
    const close = buildTopology(sim.graph, sim.agents, { ...view, zoom: 4 });
    const agentA = (topo: { agents: { id: number; pan?: number; dist?: number }[] }) =>
      topo.agents.find((ag) => ag.id === a.id)!;
    expect(Math.abs(agentA(close).pan ?? 0)).toBeGreaterThan(Math.abs(agentA(wide).pan ?? 0));
    expect(agentA(close).dist ?? 1).toBeLessThan(agentA(wide).dist ?? 0);
  });

  it('zooming out at the centre is a real dolly, not a tiny height nudge', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    sim.spawn('era', 100, 80, 0, params, true);
    const view = { x: 100, y: 80, zoom: 1, viewW: 800, viewH: 600 };
    const close = buildTopology(sim.graph, sim.agents, { ...view, zoom: 6 });
    const wide = buildTopology(sim.graph, sim.agents, { ...view, zoom: 0.18 });
    expect(wide.height ?? 0).toBeGreaterThan((close.height ?? 1) * 8);
    expect(wide.agents[0].dist ?? 0).toBeGreaterThan((close.agents[0].dist ?? 1) * 8);
  });

  it('rewrite begin and commit each emit a rewrite worklet message', () => {
    const begin = planRewriteMessages({
      type: 'rewrite',
      phase: 'begin',
      rule: 'commute',
      agentA: 1,
      agentB: 2,
      kindA: 'con',
      kindB: 'dup',
      wireId: 9,
      leftovers: [],
    });
    const commit = planRewriteMessages({
      type: 'rewrite',
      phase: 'commit',
      rule: 'annihilate-con',
      agentA: 3,
      agentB: 4,
      kindA: 'con',
      kindB: 'con',
      wireId: 0,
      leftovers: [5, 6],
    });
    expect(begin).toHaveLength(1);
    expect(begin[0].type).toBe('rewrite');
    if (begin[0].type === 'rewrite') expect(begin[0].phase).toBe(0);
    expect(commit[0].type).toBe('rewrite');
    if (commit[0].type === 'rewrite') {
      expect(commit[0].phase).toBe(1);
      expect(commit[0].leftovers).toEqual([5, 6]);
    }
  });
});

describe('AudioEngine event flow', () => {
  it('drops events before armed', () => {
    const engine = new AudioEngine();
    const sim = new Sim(100, 100);
    const params = defaultParams();
    const a = sim.spawn('era', 10, 10, 0, params, true)!;
    const b = sim.spawn('era', 50, 10, 0, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    const posted: unknown[] = [];
    engine.onPost = (m) => posted.push(m);

    engine.push(latchEv(wire, a.id, b.id), sim.graph, sim.agents);
    engine.frame(sim.graph, sim.agents);
    expect(posted).toHaveLength(0);

    engine.armWithoutAudio();
    engine.push(latchEv(wire, a.id, b.id), sim.graph, sim.agents);
    // Events queue and drain on the next frame, so a burst in one step cannot
    // fire an unbounded number of excitations.
    expect(posted.some((m) => (m as { type: string }).type === 'latch')).toBe(false);
    engine.frame(sim.graph, sim.agents);
    expect(posted.some((m) => (m as { type: string }).type === 'latch')).toBe(true);
  });

  it('posts a tune when the pitch actually changes, not a topology rebuild', async () => {
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 30, 0, params, true)!;
    const b = sim.spawn('dup', 90, 30, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);

    await engine.frame(sim.graph, sim.agents);
    const first = posted.filter((m) => m.type === 'topology');
    expect(first).toHaveLength(1);

    const wire = [...sim.graph.wires.values()][0];
    wire.rest = 24;
    wire.ropeLen = 24;
    wire.lastLen = 24;
    posted.length = 0;
    await engine.frame(sim.graph, sim.agents);
    expect(posted.filter((m) => m.type === 'topology')).toHaveLength(0);
    const tuned = posted.filter((m) => m.type === 'tune');
    expect(tuned).toHaveLength(1);
    const voice = blendVoices(voiceFromAgent(a, 'p'), voiceFromAgent(b, 'p'));
    expect(tuned[0].type).toBe('tune');
    if (tuned[0].type === 'tune') {
      expect(tuned[0].wires[0].length).toBeCloseTo(
        delaySamplesForPath(24, 0, voice.disp),
        5,
      );
    }

    // Nothing changed: no redundant post, no structured clone across the thread.
    posted.length = 0;
    await engine.frame(sim.graph, sim.agents);
    expect(posted.filter((m) => m.type === 'topology')).toHaveLength(0);
    expect(posted.filter((m) => m.type === 'tune')).toHaveLength(0);
  });

  it('posts a listen update when the camera pans, not a topology rebuild', () => {
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const sim = new Sim(200, 160);
    const params = defaultParams();
    sim.spawn('era', 100, 80, 0, params, true);
    const posted: { type: string }[] = [];
    engine.onPost = (m) => posted.push(m);
    const view = { x: 100, y: 80, zoom: 1, viewW: 400, viewH: 300 };
    engine.frame(sim.graph, sim.agents, 1 / 60, view);
    expect(posted.filter((m) => m.type === 'topology')).toHaveLength(1);
    posted.length = 0;
    // 40 px is enough to move stereo pan across a key step, but not dist.
    engine.frame(sim.graph, sim.agents, 1 / 60, { ...view, x: 140 });
    expect(posted.filter((m) => m.type === 'topology')).toHaveLength(0);
    expect(posted.filter((m) => m.type === 'listen')).toHaveLength(1);
  });

  it('ten sequential latches each produce audible output', () => {
    const net = new WaveguideNet();
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const peaks: number[] = [];

    for (let i = 0; i < 10; i++) {
      const a = sim.spawn('era', 20 + i * 3, 20, 0, params, true)!;
      const b = sim.spawn('dup', 80 + i * 3, 20, Math.PI, params, true)!;
      sim.wire(a.id, 'p', b.id, 'p', params);
      const wires = [...sim.graph.wires.values()];
      const wire = wires[wires.length - 1];
      const msgs = planLatchMessages(latchEv(wire, a.id, b.id), sim.graph, sim.agents);
      for (const m of msgs) net.handle(m);
      let peak = 0;
      for (let s = 0; s < 200; s++) peak = Math.max(peak, Math.abs(net.tick()));
      peaks.push(peak);
      for (let s = 0; s < 4000; s++) net.tick();
    }

    expect(peaks.every((p) => p > 0.01)).toBe(true);
  });
});

describe('audio presets', () => {
  it('a bad rope length does not produce a NaN delay', () => {
    expect(Number.isFinite(delaySamplesForPath(Number.NaN))).toBe(true);
    expect(Number.isFinite(delaySamplesForPath(Number.POSITIVE_INFINITY))).toBe(true);
    expect(delaySamplesForPath(40)).toBeCloseTo(delaySamplesForHz(400), 5);
  });

  it('wave speed scales every rope together', () => {
    const a = wirePitchHz(40);
    expect(wirePitchHz(60)).toBeLessThan(wirePitchHz(12));
    expect(wirePitchHz(40, 0.4)).toBeGreaterThan(a);
    expect(delaySamplesForPath(80)).toBeGreaterThan(delaySamplesForPath(20));
    setWaveSpeed(2);
    try {
      expect(wirePitchHz(40)).toBeCloseTo(a * 2, 5);
    } finally {
      setWaveSpeed(1);
    }
    expect(wirePitchHz(40)).toBeCloseTo(a, 5);
  });

  it('scales latch and collision gains', () => {
    expect(latchGain(40)).toBeGreaterThan(latchGain(10));
    expect(wirePitchHz(40)).toBeCloseTo(400, 5);
    expect(collisionGain(100, 4)).toBeGreaterThan(collisionGain(20, 4));
    expect(portImpedance('con', 'p')).toBeGreaterThan(portImpedance('dup', 'p'));
    expect(bendLoss(3, 50, 40)).toBeGreaterThan(bendLoss(0, 40, 40));
  });

  it('treats a yank as taut and a sag as slack', () => {
    expect(tautness(40, 40, 40)).toBe(0);
    expect(tautness(50, 40, 50)).toBe(0);
    expect(tautness(52, 40, 40)).toBeGreaterThan(0.2);
    expect(waveSpeed(0.5)).toBeGreaterThan(waveSpeed(0));
    expect(tautBrighten(0.4, 0.4)).toBeGreaterThan(0.4);
  });

  it('maps slide speed to bow velocity, not a sounding pitch', () => {
    const slow = bowSpeed(20);
    const fast = bowSpeed(90);
    expect(slow).toBeGreaterThan(0);
    expect(fast).toBeGreaterThan(slow);
    expect(bowSpeed(400)).toBe(0.05);
    expect(bowSpeed(-400)).toBe(-0.05);
    expect(bowSpeed(-40)).toBeCloseTo(-bowSpeed(40));
  });

});

describe('sim latch integration', () => {
  it('fires onLatch for each new wire', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    const events: LatchEvent[] = [];
    sim.graph.onLatch = (ev) => events.push(ev);
    const a = sim.spawn('era', 40, 40, 0, params, true)!;
    const b = sim.spawn('dup', 100, 40, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    expect(events).toHaveLength(1);
  });

  it('does not retrigger collision audio while two bodies rest in contact', () => {
    audio.armWithoutAudio();
    const posted: { type: string }[] = [];
    audio.onPost = (m) => posted.push(m);

    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.spawnInterval = 0;
    sim.spawn('era', 100, 80, 0, params, true);
    sim.spawn('era', 101, 80, Math.PI, params, true);

    for (let i = 0; i < 45; i++) {
      sim.step(1 / 60, params);
      audio.frame(sim.graph, sim.agents, 1 / 60);
    }

    const strikes = posted.filter((m) => m.type === 'strike').length;
    audio.onPost = null;
    expect(strikes).toBeGreaterThan(0);
    expect(strikes).toBeLessThan(16);
  });

  it('indexes a traveling-wave snapshot by wire id', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: {
      wires: [{ id: 7, length: 80, loss: 0.999, bend: 0.05, agentA: 1, agentB: 2, zA: 1, zB: 1 }],
      agents: [
        { id: 1, kind: 0, openPorts: 0, impedance: 1 },
        { id: 2, kind: 2, openPorts: 0, impedance: 1 },
      ],
    }, wireId: 7, gain: 1 });
    const engine = new AudioEngine();
    engine.acceptWaves(new Float32Array(net.fillWaveSnapshot()));
    const waves = engine.waves;
    expect(waves).not.toBeNull();
    expect(waves!.index.has(7)).toBe(true);
    expect(waves!.index.has(99)).toBe(false);
    const o = waves!.index.get(7)!;
    expect(waves!.packed[o + 1]).toBeGreaterThan(0);
  });

  it('an empty snapshot drops the last wave index', () => {
    const engine = new AudioEngine();
    engine.acceptWaves(new Float32Array([1, 4, 7, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]));
    expect(engine.waves).not.toBeNull();
    engine.acceptWaves(new Float32Array([0, 32]));
    expect(engine.waves).toBeNull();
  });
});

describe('distance curves', () => {
  it('the worklet copies stay in step with the main-thread ones', () => {
    // waveguide.ts is inlined into the worklet as one import-free file, so it
    // carries its own copy of these four curves. Nothing stops the two drifting
    // apart, and the symptom would be depth that reads one way in the topology
    // builder and another in the audio — so pin them numerically rather than
    // trusting a comment to keep them honest.
    for (let d = 0; d <= 6; d += 0.05) {
      expect(distDry(d)).toBeCloseTo(distanceDry(d), 12);
      expect(distWet(d)).toBeCloseTo(distanceWet(d), 12);
      expect(distRoom(d)).toBeCloseTo(distanceRoom(d), 12);
      expect(distDamp(d)).toBeCloseTo(distanceDamp(d), 12);
    }
    // Negative and non-finite inputs must not produce NaN gains either.
    for (const bad of [-1, -0.001]) {
      expect(Number.isFinite(distDry(bad))).toBe(true);
      expect(Number.isFinite(distWet(bad))).toBe(true);
      expect(Number.isFinite(distRoom(bad))).toBe(true);
      expect(Number.isFinite(distDamp(bad))).toBe(true);
    }
  });

  it('further away is quieter, darker, and proportionally wetter', () => {
    // The three cues that actually carry distance. The third is the one that
    // level alone cannot buy: turning a source down scales direct and reverb
    // together, leaving D/R — the cue the ear reads — unchanged.
    const near = 0.2;
    const far = 3;
    expect(distanceDry(far)).toBeLessThan(distanceDry(near));
    expect(distanceDamp(far)).toBeLessThan(distanceDamp(near));
    const drNear = distanceDry(near) / distanceWet(near);
    const drFar = distanceDry(far) / distanceWet(far);
    expect(drFar).toBeLessThan(drNear);
  });
});

describe('worklet constants', () => {
  it('the inlined copy stays in step with the main-thread copy', () => {
    // waveguide.ts cannot import anything — the worklet plugin inlines it as a
    // single file — so these constants exist twice and can silently diverge.
    expect(WG_MAX_WIRES).toBe(MAX_WIRES);
    expect(WG_MAX_DELAY).toBe(MAX_DELAY);
    expect(WG_MAX_AIR).toBe(MAX_AIR_PATHS);
    expect(WG_MAX_AIR_DELAY).toBe(MAX_AIR_DELAY);
    expect(WG_MAX_STUBS).toBe(MAX_STUBS);
    expect(WG_MAX_STUB_DELAY).toBe(MAX_STUB_DELAY);
    expect(WG_distDry(1)).toBe(distanceDry(1));
    expect(WG_distWet(2)).toBe(distanceWet(2));
    expect(WG_distDamp(1.5)).toBe(distanceDamp(1.5));
    expect(WG_distRoom(3)).toBe(distanceRoom(3));
  });
});

describe('air paths', () => {
  it('gain falls with distance', () => {
    expect(airGain(40)).toBeGreaterThan(airGain(120));
    expect(airGain(120)).toBeGreaterThan(airGain(220));
    expect(airDelaySamples(80)).toBeGreaterThan(airDelaySamples(40));
  });

  it('plans a path for nearby bodies and skips touching or distant ones', () => {
    const sim = new Sim(400, 200);
    const params = defaultParams();
    const nearA = sim.spawn('era', 40, 40, 0, params, true)!;
    const nearB = sim.spawn('dup', 40 + 50, 40, 0, params, true)!;
    const tooClose = sim.spawn('con', 40 + 8, 40, 0, params, true)!;
    const far = sim.spawn('era', 40 + AIR_CUTOFF_PX + 40, 40, 0, params, true)!;
    const msg = planAirMessage(sim.agents);
    expect(msg.type).toBe('air');
    const key = (a: number, b: number) => [a, b].sort((x, y) => x - y).join('-');
    const pairs = msg.items.map((it) => key(it.agentA, it.agentB));
    expect(pairs).toContain(key(nearA.id, nearB.id));
    expect(pairs).not.toContain(key(nearA.id, tooClose.id));
    expect(pairs).not.toContain(key(nearA.id, far.id));
    expect(AIR_MIN_PX).toBeGreaterThan(0);
  });
});

describe('listener distance', () => {
  it('direct sound falls faster than the reverb send, so D/R drops', () => {
    expect(distanceDry(0)).toBeGreaterThan(distanceDry(1));
    expect(distanceDry(1)).toBeGreaterThan(distanceDry(2));
    expect(distanceWet(0) / distanceDry(0)).toBeLessThan(distanceWet(2) / distanceDry(2));
    expect(distanceDamp(0)).toBeGreaterThan(distanceDamp(2));
  });

  it('the hall falls as the listener leaves the room', () => {
    expect(distanceRoom(0)).toBeGreaterThan(distanceRoom(2));
    expect(distanceRoom(2)).toBeGreaterThan(distanceRoom(5));
  });
});

describe('shape drives the impulse', () => {
  it('a sharper body feature makes a narrower excitation', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const era = sim.spawn('era', 30, 30, 0, params, true)!;
    const con = sim.spawn('con', 90, 30, Math.PI, params, true)!;
    // The triangle apex is the sharpest feature on any body here.
    expect(sharpness('con', 'p')).toBeGreaterThan(sharpness('con', 'l'));
    expect(sharpness('con', 'p')).toBeGreaterThan(sharpness('era', 'p'));
    // Sharper feature, narrower contact.
    expect(voiceFromAgent(con, 'p').width).toBeLessThan(voiceFromAgent(con, 'l').width);
    void era;
  });

  it('what you see is what you hear: the dark body is the darker voice', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const dup = sim.spawn('dup', 30, 30, 0, params, true)!;
    const con = sim.spawn('con', 90, 30, 0, params, true)!;
    // Dup is drawn #111213, Con #f4f4f4 — same triangle, opposite albedo.
    expect(albedo('dup')).toBeLessThan(albedo('con'));
    expect(voiceFromAgent(dup, 'p').damp).toBeLessThan(voiceFromAgent(con, 'p').damp);
  });

  it('a triangle struck on a vertex is sharper than one struck on a face', () => {
    // Heading 0 puts the apex at +x, so a normal along +x is a vertex hit and
    // one along -x lands flat on the base.
    const apex = strikeSharpness({ kind: 'con', heading: 0 }, 1, 0);
    const face = strikeSharpness({ kind: 'con', heading: 0 }, -1, 0);
    expect(apex).toBeGreaterThan(face);
    // A circle has no orientation, so it answers the same from every side.
    const a = strikeSharpness({ kind: 'era', heading: 0 }, 1, 0);
    const b = strikeSharpness({ kind: 'era', heading: 1.1 }, -0.3, 0.95);
    expect(a).toBe(b);
  });
});

describe('contact physics', () => {
  it('a harder impact is a shorter contact, not just a louder one', () => {
    // Hertzian contact time goes as v^(-1/5). This is the reason a hard knock
    // sounds brighter rather than merely bigger, so it is worth pinning down.
    const soft = contactSeconds(1.4, 5);
    const hard = contactSeconds(1.4, 300);
    expect(hard).toBeLessThan(soft);
    expect(contactPeak(1.4, 300, hard)).toBeGreaterThan(contactPeak(1.4, 5, soft));
  });

  it('a heavier pair stays in contact longer', () => {
    expect(contactSeconds(4, 80)).toBeGreaterThan(contactSeconds(1.4, 80));
  });

  it('the contact force reaches the wires as a pulse, not a spike', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleAgentTopo(1) });
    net.handle({ type: 'strike', agentId: 1, peak: 1.2, dur: 96, sharp: 0.5 });
    // A 96-sample contact must still be pushing force well after sample 1.
    let early = 0;
    for (let i = 0; i < 8; i++) early = Math.max(early, Math.abs(net.tick()));
    let mid = 0;
    for (let i = 0; i < 60; i++) mid = Math.max(mid, Math.abs(net.tick()));
    expect(mid).toBeGreaterThan(early);
  });

  it('an unwired body rings when knocked and then goes silent', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleAgentTopo(1) });
    expect(net.activeWireCount()).toBe(0);
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 60, sharp: 0.8 });
    let pk = 0;
    for (let i = 0; i < 24000; i++) pk = Math.max(pk, Math.abs(net.tick()));
    expect(pk).toBeGreaterThan(0.005);
    for (let i = 0; i < 48000 * 8; i++) net.tick();
    let tail = 0;
    for (let i = 0; i < 4800; i++) tail = Math.max(tail, Math.abs(net.tick()));
    expect(tail).toBeLessThan(1e-10);
  });

  it('body pitch tracks visible size', () => {
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const era = sim.spawn('era', 30, 30, 0, params, true)!;
    const con = sim.spawn('con', 90, 30, 0, params, true)!;
    // The circle is drawn about half the triangle's size, so it pings higher.
    expect(bodyTone(era).freq[0]).toBeGreaterThan(bodyTone(con).freq[0] * 1.5);
    // Higher modes always die before lower ones.
    const t = bodyTone(con);
    expect(t.decay[2]).toBeLessThan(t.decay[0]);
  });

  it('rubbing is louder the harder and faster it slides', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleContactTopo() });
    const level = (load: number, vT: number) => {
      net.handle({
        type: 'contact',
        items: [{ agentA: 1, agentB: 2, load, slide: bowSpeed(vT) }],
      });
      let pk = 0;
      for (let i = 0; i < 12000; i++) pk = Math.max(pk, Math.abs(net.tick()));
      return pk;
    };
    const gentle = level(0.1, 4);
    const hard = level(1, 120);
    expect(hard).toBeGreaterThan(gentle);
  });

  it('rubbing stops dead when the sim stops reporting contact', () => {
    // Friction is the one continuous excitation in the whole system, so this is
    // the guarantee that keeps it from becoming a drone: an empty list is how
    // the sim says nothing is touching, and silence has to follow immediately.
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleContactTopo() });
    net.handle({
      type: 'contact',
      items: [{ agentA: 1, agentB: 2, load: 1, slide: bowSpeed(120) }],
    });
    let loud = 0;
    for (let i = 0; i < 24000; i++) loud = Math.max(loud, Math.abs(net.tick()));
    expect(loud).toBeGreaterThan(1e-4);

    net.handle({ type: 'contact', items: [] });
    for (let i = 0; i < 48000 * 4; i++) net.tick();
    let stopped = 0;
    for (let i = 0; i < 4800; i++) stopped = Math.max(stopped, Math.abs(net.tick()));
    expect(stopped).toBeLessThan(1e-10);
  });
});

describe('contact radiation', () => {
  it('two identical bodies rubbing do not cancel to silence', () => {
    // Regression: the contact force is equal and opposite by Newton's third
    // law, so radiating it directly made a same-kind pair sum to exactly zero
    // at the pickup — the same failure mode as summing both ends of a wire.
    // Surface roughness is drawn per body, so the two are correlated but never
    // identical, and the pair stays audible.
    const net = new WaveguideNet();
    const one = sampleAgentTopo(1);
    const two = sampleAgentTopo(2);
    net.handle({ type: 'topology', topo: { wires: [], agents: [one.agents[0], two.agents[0]] } });
    net.handle({
      type: 'contact',
      items: [{ agentA: 1, agentB: 2, load: 1, slide: bowSpeed(120) }],
    });
    let pk = 0;
    for (let i = 0; i < 12000; i++) pk = Math.max(pk, Math.abs(net.tick()));
    expect(pk).toBeGreaterThan(0.01);
  });
});

describe('friction entrainment', () => {
  it('with one dominant resonator, rubbing locks to it', () => {
    // Guard on the friction model itself, in a deliberately controlled setup:
    // one long wire, no stubs, nothing else to compete. Under those conditions
    // stick-slip must entrain to the wire's round trip — that is Helmholtz
    // motion, and it is what separates bowing from scraping.
    //
    // This exists because it has already been broken once: raising the contact
    // roughness far enough drags the limit cycle off the string and onto the
    // body modes, and the whole thing turns to metallic hash. Nothing caught
    // that but a listener.
    const net = new WaveguideNet();
    const L = 150;
    const agent = (id: number) => ({
      id,
      kind: 0 as const,
      openPorts: 0,
      impedance: 1,
      pan: 0,
      modeHz: [283, 591, 972],
      modeT60: [0.25, 0.12, 0.08],
      modeGain: [1, 0.42, 0.26],
      coupling: 1,
    });
    net.handle({
      type: 'topology',
      topo: {
        wires: [
          {
            id: 1, length: L, loss: 0.9992, bend: 0.04, agentA: 1, agentB: 2,
            zA: 1, zB: 1, damp: 0.6, disp: 0, pan: 0, exAt: 0.16, exWidth: 1,
          },
        ],
        agents: [agent(1), agent(2)],
      },
    });
    net.handle({
      type: 'contact',
      items: [{ agentA: 1, agentB: 2, load: 1, slide: bowSpeed(120) }],
    });
    for (let i = 0; i < 20000; i++) net.tick();

    const x: number[] = [];
    for (let i = 0; i < 8192; i++) x.push(net.tick());
    let mean = 0;
    for (const v of x) mean += v;
    mean /= x.length;
    let energy = 0;
    for (const v of x) energy += (v - mean) * (v - mean);
    let best = 0;
    let bestLag = 0;
    // Floor well above zero: normalized autocorrelation is trivially ~1 at tiny
    // lags for any smooth signal, which is its own way to fool yourself.
    for (let lag = 30; lag < 900; lag++) {
      let s = 0;
      for (let i = 0; i + lag < x.length; i++) s += (x[i] - mean) * (x[i + lag] - mean);
      if (s / energy > best) {
        best = s / energy;
        bestLag = lag;
      }
    }
    expect(best).toBeGreaterThan(0.4);
    // The wire's round trip is 2L. Body modes sit near lag 160; landing there
    // instead is the failure this test is looking for.
    expect(bestLag).toBeGreaterThan(2 * L * 0.9);
    expect(bestLag).toBeLessThan(2 * L * 1.1);
  });
});

describe('output headroom', () => {
  it('the two buses cannot sum past full scale at the master', () => {
    // The worklet soft-clips each bus to 1, and the engine sums them through
    // dry and wet gains, so those gains have to sum to at most 1 or a dense
    // moment clips at the destination. They were 0.85 + 0.7.
    const engine = new AudioEngine();
    const gains = engine.busGains();
    expect(gains.dry).toBeGreaterThan(0);
    expect(gains.wet).toBeGreaterThan(0);
    expect(gains.dry + gains.wet).toBeLessThanOrEqual(1.0000001);
  });

  it('a soft-clipped bus never leaves the worklet above full scale', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleAgentTopo(1) });
    // Drive it far harder than anything the sim produces.
    for (let i = 0; i < 12; i++) {
      net.handle({ type: 'strike', agentId: 1, peak: 8, dur: 40, sharp: 1 });
      for (let s = 0; s < 200; s++) {
        net.tick(false);
        expect(Math.abs(net.outDryL)).toBeLessThanOrEqual(1);
        expect(Math.abs(net.outDryR)).toBeLessThanOrEqual(1);
        expect(Math.abs(net.outWetL)).toBeLessThanOrEqual(1);
        expect(Math.abs(net.outWetR)).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe('stereo placement', () => {
  function panned(pan: number): { l: number; r: number } {
    const net = new WaveguideNet();
    const agent = (id: number) => ({
      id,
      kind: 0 as const,
      openPorts: 0,
      impedance: 1,
      pan,
      modeHz: [283, 591, 972],
      modeT60: [0.25, 0.12, 0.08],
      modeGain: [1, 0.42, 0.26],
      coupling: 1,
    });
    net.handle({
      type: 'topology',
      topo: {
        wires: [
          {
            id: 1, length: 150, loss: 0.9992, bend: 0.04, agentA: 1, agentB: 2,
            zA: 1, zB: 1, damp: 0.6, disp: 0, pan, exAt: 0.16, exWidth: 1,
          },
        ],
        agents: [agent(1), agent(2)],
      },
    });
    net.injectPluck(1, 0.9);
    let l = 0;
    let r = 0;
    for (let i = 0; i < 4096; i++) {
      net.tick();
      l += net.outDryL * net.outDryL;
      r += net.outDryR * net.outDryR;
    }
    return { l: Math.sqrt(l / 4096), r: Math.sqrt(r / 4096) };
  }

  it('sends a hard-left source left and a hard-right source right', () => {
    const left = panned(-1);
    const right = panned(1);
    expect(left.l).toBeGreaterThan(0);
    expect(left.r).toBeLessThan(left.l * 0.05);
    expect(right.r).toBeGreaterThan(0);
    expect(right.l).toBeLessThan(right.r * 0.05);
  });

  it('keeps a centred source equal in both channels', () => {
    const mid = panned(0);
    expect(mid.l).toBeGreaterThan(0);
    expect(Math.abs(mid.l - mid.r)).toBeLessThan(mid.l * 1e-6);
  });

  it('holds power constant across the sweep, not amplitude', () => {
    // Equal power: L^2 + R^2 is flat, so a source does not dip in level as it
    // crosses the middle. The gains are cached per source now, so this is
    // really asking whether the cache is refreshed when pan is set.
    const power = [-1, -0.5, 0, 0.5, 1].map((p) => {
      const { l, r } = panned(p);
      return l * l + r * r;
    });
    const first = power[0];
    for (const p of power) expect(Math.abs(p - first)).toBeLessThan(first * 0.02);
  });
});

describe('AudioEngine play helpers', () => {
  it('posts a pluck with at and width', () => {
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const sim = new Sim(200, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 30, 80, 0, params, true)!;
    const b = sim.spawn('era', 120, 80, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    engine.frame(sim.graph, sim.agents);
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);
    const wire = [...sim.graph.wires.values()][0]!;
    engine.playPluck(wire.id, 0.8, 0.3, 0.4);
    const pluck = posted.find((m) => m.type === 'pluck');
    expect(pluck).toEqual({ type: 'pluck', wireId: wire.id, gain: 0.8, at: 0.3, width: 0.4 });
  });
});
