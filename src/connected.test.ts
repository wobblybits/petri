import { describe, expect, it } from 'vitest';
import type { Agent, AgentKind, PortSlot } from './agents.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/** Live defaults, but topology cannot change under us. */
function frozenLive(): Params {
  const params = defaultParams();
  params.snapRadius = 0;
  params.snapWell = 0;
  // No rewrites at all, not slow ones. A thirty-second rewrite still begins
  // hauling its pair together on its first frame, and the Eras wired to the
  // pair are towed along at a steady 30 px/s — which this file then read as
  // a net that could not settle. The topology is meant to hold still here.
  params.rewriteDuration = 0;
  params.spawnInterval = 0;
  return params;
}

/** Constraint + collision only: no motors, flock, gravity, or scent. */
function frozenPassive(): Params {
  const params = frozenLive();
  params.stepSpeed = 0;
  params.faceAttract = 0;
  params.flockAlign = 0;
  params.flockSep = 0;
  params.deposit = 0;
  params.sense = 0;
  // The transport pump is a motor too: a pair pumping energy recoils along
  // the wire, which is the swimming stroke. A passive net has it off.
  params.transportSpeed = 0;
  return params;
}

interface Spike {
  id: number;
  kind: AgentKind;
  speed: number;
  jerk: number;
  omega: number;
}

interface Trace {
  peakSpeed: number;
  peakJerk: number;
  peakOmega: number;
  keStart: number;
  keEnd: number;
  keMax: number;
  jerkFrames: number;
  worst: Spike;
}

function agentSpeed(a: Agent): number {
  return Math.hypot(a.vx, a.vy);
}

function measure(sim: Sim, params: Params, warmup: number, window: number): Trace {
  const prev = new Map<number, { vx: number; vy: number }>();
  const empty: Spike = { id: -1, kind: 'era', speed: 0, jerk: 0, omega: 0 };
  const trace: Trace = {
    peakSpeed: 0,
    peakJerk: 0,
    peakOmega: 0,
    keStart: 0,
    keEnd: 0,
    keMax: 0,
    jerkFrames: 0,
    worst: { ...empty },
  };

  const sample = (recordKe: boolean, first: boolean): number => {
    const ke = sim.kineticEnergy();
    if (recordKe) {
      if (first) trace.keStart = ke;
      trace.keMax = Math.max(trace.keMax, ke);
      trace.keEnd = ke;
    }
    let frameJerk = 0;
    for (const a of sim.agents.values()) {
      const speed = agentSpeed(a);
      const last = prev.get(a.id);
      const jerk = last ? Math.hypot(a.vx - last.vx, a.vy - last.vy) : 0;
      if (jerk > trace.peakJerk || (jerk === trace.peakJerk && speed > trace.worst.speed)) {
        trace.peakJerk = jerk;
        trace.worst = { id: a.id, kind: a.kind, speed, jerk, omega: a.omega };
      }
      trace.peakSpeed = Math.max(trace.peakSpeed, speed);
      trace.peakOmega = Math.max(trace.peakOmega, Math.abs(a.omega));
      frameJerk = Math.max(frameJerk, jerk);
      prev.set(a.id, { vx: a.vx, vy: a.vy });
    }
    return frameJerk;
  };

  for (let i = 0; i < warmup; i++) {
    sim.step(1 / 60, params);
    sample(false, false);
  }
  trace.peakSpeed = 0;
  trace.peakJerk = 0;
  trace.peakOmega = 0;
  trace.keStart = 0;
  trace.keMax = 0;
  trace.keEnd = 0;
  trace.jerkFrames = 0;
  trace.worst = { ...empty };
  prev.clear();

  for (let i = 0; i < window; i++) {
    sim.step(1 / 60, params);
    const jerk = sample(true, i === 0);
    if (jerk > 8) trace.jerkFrames += 1;
  }
  return trace;
}

function label(t: Trace): string {
  const w = t.worst;
  return [
    `peakSpeed=${t.peakSpeed.toFixed(2)}`,
    `peakJerk=${t.peakJerk.toFixed(2)}`,
    `peakOmega=${t.peakOmega.toFixed(2)}`,
    `ke ${t.keStart.toFixed(1)}→${t.keEnd.toFixed(1)} (max ${t.keMax.toFixed(1)})`,
    `jerkFrames=${t.jerkFrames}`,
    `worst=${w.kind}#${w.id} v=${w.speed.toFixed(2)} dv=${w.jerk.toFixed(2)} ω=${w.omega.toFixed(2)}`,
  ].join(' | ');
}

function expectCalm(
  t: Trace,
  maxSpeed: number,
  maxJerk: number,
  maxKe?: number,
): void {
  const msg = label(t);
  expect(t.peakSpeed, msg).toBeLessThan(maxSpeed);
  expect(t.peakJerk, msg).toBeLessThan(maxJerk);
  const keCap = maxKe ?? Math.max(t.keStart * 1.25 + 12, 40);
  expect(t.keEnd, msg).toBeLessThan(keCap);
  expect(t.jerkFrames, msg).toBeLessThan(8);
}

function plug(sim: Sim, agent: Agent, slot: PortSlot, params: Params, dist = 36): Agent {
  const hx = Math.cos(agent.heading);
  const hy = Math.sin(agent.heading);
  let x = agent.x;
  let y = agent.y;
  let heading = agent.heading + Math.PI;
  if (slot === 'p') {
    x += hx * dist;
    y += hy * dist;
  } else {
    const side = slot === 'l' ? 1 : -1;
    x -= hx * dist * 0.7;
    y -= hy * dist * 0.7;
    x += -hy * side * dist * 0.55;
    y += hx * side * dist * 0.55;
    heading = Math.atan2(agent.y - y, agent.x - x);
  }
  const era = sim.spawn('era', x, y, heading, params, true)!;
  sim.wire(agent.id, slot, era.id, 'p', params);
  return era;
}

describe('connected nets after latch (passive constraints)', () => {
  const warmup = 180;
  const window = 90;

  it('two eras, principal-to-principal', () => {
    const sim = new Sim(400, 240);
    const params = frozenPassive();
    sim.spawn('era', 140, 120, 0, params, true);
    sim.spawn('era', 260, 120, Math.PI, params, true);
    const [a, b] = [...sim.agents.values()];
    sim.wire(a.id, 'p', b.id, 'p', params);
    expect(sim.graph.wires.size).toBe(1);
    expectCalm(measure(sim, params, warmup, window), 14, 6);
  });

  it('constructor with only its principal wired', () => {
    const sim = new Sim(400, 240);
    const params = frozenPassive();
    const con = sim.spawn('con', 180, 120, 0, params, true)!;
    plug(sim, con, 'p', params, 48);
    expect(sim.graph.wires.size).toBe(1);
    expectCalm(measure(sim, params, warmup, window), 14, 6);
  });

  it('constructor with only an aux wired', () => {
    const sim = new Sim(400, 240);
    const params = frozenPassive();
    const con = sim.spawn('con', 200, 120, 0, params, true)!;
    plug(sim, con, 'l', params, 48);
    expect(sim.graph.wires.size).toBe(1);
    expectCalm(measure(sim, params, warmup, window), 14, 6);
  });

  it('constructor with all three ports wired', () => {
    const sim = new Sim(480, 280);
    const params = frozenPassive();
    const con = sim.spawn('con', 240, 140, 0, params, true)!;
    plug(sim, con, 'p', params, 52);
    plug(sim, con, 'l', params, 48);
    plug(sim, con, 'r', params, 48);
    expect(sim.graph.wires.size).toBe(3);
    expectCalm(measure(sim, params, warmup, window), 16, 8);
  });

  it('duplicator with all three ports wired', () => {
    const sim = new Sim(480, 280);
    const params = frozenPassive();
    const dup = sim.spawn('dup', 240, 140, 0, params, true)!;
    plug(sim, dup, 'p', params, 52);
    plug(sim, dup, 'l', params, 48);
    plug(sim, dup, 'r', params, 48);
    expect(sim.graph.wires.size).toBe(3);
    expectCalm(measure(sim, params, warmup, window), 16, 8);
  });

  it('line of era-con-con-era', () => {
    const sim = new Sim(640, 240);
    const params = frozenPassive();
    const e1 = sim.spawn('era', 80, 120, 0, params, true)!;
    const c1 = sim.spawn('con', 200, 120, Math.PI, params, true)!;
    const c2 = sim.spawn('con', 320, 120, 0, params, true)!;
    const e2 = sim.spawn('era', 440, 120, Math.PI, params, true)!;
    sim.wire(e1.id, 'p', c1.id, 'p', params);
    sim.wire(c1.id, 'l', c2.id, 'l', params);
    sim.wire(c2.id, 'p', e2.id, 'p', params);
    expect(sim.graph.wires.size).toBe(3);
    expectCalm(measure(sim, params, warmup, window), 22, 8);
  });

  it('commute net (con-dup principals, aux eras)', () => {
    const sim = new Sim(480, 280);
    const params = frozenPassive();
    loadPreset(sim, 'commute', params);
    expect(sim.graph.wires.size).toBe(5);
    expectCalm(measure(sim, params, warmup, window), 18, 8);
  });

  it('annihilate-con net', () => {
    const sim = new Sim(480, 280);
    const params = frozenPassive();
    loadPreset(sim, 'annihilate-con', params);
    expect(sim.graph.wires.size).toBe(5);
    expectCalm(measure(sim, params, warmup, window), 18, 8);
  });

  it('oscillator net', () => {
    const sim = new Sim(480, 280);
    const params = frozenPassive();
    loadPreset(sim, 'oscillator', params);
    expect(sim.graph.wires.size).toBe(4);
    expectCalm(measure(sim, params, warmup, window), 24, 18, 400);
  });
});

describe('connected nets after latch (live motors / flock / gravity)', () => {
  const warmup = 180;
  const window = 90;

  it('two eras, both principals latched', () => {
    const sim = new Sim(400, 240);
    const params = frozenLive();
    sim.spawn('era', 140, 120, 0, params, true);
    sim.spawn('era', 260, 120, Math.PI, params, true);
    const [a, b] = [...sim.agents.values()];
    sim.wire(a.id, 'p', b.id, 'p', params);
    expectCalm(measure(sim, params, warmup, window), 16, 8);
  });

  it('constructor with all ports latched still does not spike', () => {
    const sim = new Sim(480, 280);
    const params = frozenLive();
    const con = sim.spawn('con', 240, 140, 0, params, true)!;
    plug(sim, con, 'p', params, 52);
    plug(sim, con, 'l', params, 48);
    plug(sim, con, 'r', params, 48);
    expectCalm(measure(sim, params, warmup, window), 50, 8, 1600);
  });

  it('commute net does not keep jerking', () => {
    const sim = new Sim(480, 280);
    const params = frozenLive();
    loadPreset(sim, 'commute', params);
    expectCalm(measure(sim, params, warmup, window), 22, 10);
  });

  it('a free-principal constructor towing an aux era still cruises without exploding', () => {
    const sim = new Sim(400, 240);
    const params = frozenLive();
    const con = sim.spawn('con', 180, 120, 0, params, true)!;
    plug(sim, con, 'l', params, 48);
    const t = measure(sim, params, 90, 60);
    expect(Math.hypot(con.vx, con.vy), label(t)).toBeGreaterThan(5);
    expect(t.peakJerk, label(t)).toBeLessThan(40);
    expect(t.peakSpeed, label(t)).toBeLessThan(80);
  });
});
