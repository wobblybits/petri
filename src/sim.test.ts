import { describe, expect, it } from 'vitest';
import { boundRadius, stemWorld } from './agents.ts';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { queryHit, SLOP } from './collide.ts';
import { closestPointOnSegment, transverseProfile, WIRE_RADIUS } from './geom.ts';
import { mixScent, scentSlowFactor, scentTurnBoost, Sim } from './sim.ts';
import { nativeSolver } from './native/solver.ts';
import { CH } from './fields.ts';
import { angleDelta } from './wrap.ts';

function fastParams() {
  const params = defaultParams();
  params.wireShrink = 0.08;
  params.wireMinRest = 40;
  params.rewriteDuration = 0.12;
  params.springK = 90;
  params.gravity = 0;
  params.homing = 0;
  params.spawnInterval = 0;
  return params;
}

function step(sim: Sim, params: ReturnType<typeof fastParams>, n: number): void {
  for (let i = 0; i < n; i++) sim.step(1 / 60, params);
}

describe('scent steering', () => {
  it('scentSlowFactor and scentTurnBoost trade linear speed for rotation', () => {
    expect(scentSlowFactor(0)).toBe(1);
    expect(scentTurnBoost(0)).toBe(1);
    const trail = 56;
    expect(scentSlowFactor(trail)).toBeLessThan(0.4);
    expect(scentTurnBoost(trail)).toBeGreaterThan(3);
  });

  it('does not mix an agent’s own principal channel', () => {
    const params = defaultParams();
    expect(mixScent('dup', 1, 50, 3, params)).toBe(mixScent('dup', 1, 0, 3, params));
    expect(mixScent('con', 50, 2, 3, params)).toBe(mixScent('con', 0, 2, 3, params));
    expect(mixScent('era', 1, 2, 3, params)).toBeGreaterThan(mixScent('era', 0, 0, 3, params));
  });

  it('goes straight when the two sensors are symmetric', () => {
    for (const kind of ['era', 'con', 'dup'] as const) {
      const sim = new Sim(240, 160);
      const params = defaultParams();
      params.faceAttract = 0;
      params.snapWell = 0;
      params.snapRadius = 0;
      params.gravity = 0;
      params.homing = 0;
      const heading = 0.4;
      const agent = sim.spawn(kind, 120, 80, heading, params, true)!;
      step(sim, params, 45);
      expect(agent.heading).toBeCloseTo(heading, 5);
      const hx = Math.cos(heading);
      const hy = Math.sin(heading);
      const along = agent.vx * hx + agent.vy * hy;
      expect(agent.vx).toBeCloseTo(along * hx, 5);
      expect(agent.vy).toBeCloseTo(along * hy, 5);
    }
  });

  it('does not self-propel when the principal port is latched', () => {
    const sim = new Sim(320, 200);
    const params = defaultParams();
    params.gravity = 0;
    params.homing = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.snapRadius = 0;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    params.stepSpeed = 50;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 200, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 30);
    expect(Math.hypot(a.vx, a.vy)).toBeLessThan(18);
    expect(Math.hypot(b.vx, b.vy)).toBeLessThan(18);
  });

  it('pushes overlapping free agents apart', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('con', 120, 80, 0, params, true)!;
    const b = sim.spawn('con', 121, 80, Math.PI, params, true)!;
    step(sim, params, 12);
    expect(queryHit(a, b, sim.w, sim.h)?.overlap ?? 0).toBeLessThanOrEqual(SLOP + 0.08);
  });

  it('pushes overlapping wired agents apart', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    params.rewriteDuration = 20;
    const a = sim.spawn('era', 120, 80, 0, params, true)!;
    const b = sim.spawn('era', 121, 80, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 24);
    expect(queryHit(a, b, sim.w, sim.h)?.overlap ?? 0).toBeLessThan(2);
  });

  it('collision changes linear velocity', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    params.gravity = 0;
    params.homing = 0;
    const a = sim.spawn('con', 100, 80, 0, params, true)!;
    const b = sim.spawn('con', 118, 80, Math.PI, params, true)!;
    a.vx = 60;
    sim.step(1 / 60, params);
    expect(a.vx).toBeLessThan(60);
    expect(b.vx).toBeGreaterThan(0);
  });

  it('a rope drapes around a visitor instead of cutting through it', () => {
    const sim = new Sim(320, 200);
    const params = defaultParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    params.gravity = 0;
    params.homing = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.declutter = 0;
    params.rewriteDuration = 20;
    params.spawnInterval = 0;
    const a = sim.spawn('era', 60, 100, 0, params, true)!;
    const b = sim.spawn('era', 260, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    expect(wire.nodes.length).toBeGreaterThan(2);
    const mid = wire.nodes[Math.floor(wire.nodes.length / 2)];
    const visitor = sim.spawn('con', mid.x, mid.y, 0, params, true)!;
    const vx0 = visitor.x;
    const vy0 = visitor.y;
    step(sim, params, 30);
    const pts = [
      stemWorld(a, 'p', sim.w, sim.h),
      ...wire.nodes,
      stemWorld(b, 'p', sim.w, sim.h),
    ];
    let dist = Infinity;
    for (let i = 0; i < pts.length - 1; i++) {
      const q = closestPointOnSegment(
        visitor.x,
        visitor.y,
        pts[i].x,
        pts[i].y,
        pts[i + 1].x,
        pts[i + 1].y,
      );
      dist = Math.min(dist, Math.hypot(q.x - visitor.x, q.y - visitor.y));
    }
    const keep = boundRadius(visitor) + WIRE_RADIUS;
    expect(dist).toBeGreaterThan(keep - 4);
    expect(Math.hypot(visitor.x - vx0, visitor.y - vy0)).toBeLessThan(3);
  });
});

describe('transverse profile', () => {
  it('pins the ends and reports the bow height', () => {
    const p = transverseProfile([
      { x: 0, y: 0 },
      { x: 50, y: 20 },
      { x: 100, y: 0 },
    ]);
    expect(p.samples[0]).toBe(0);
    expect(p.samples[p.samples.length - 1]).toBe(0);
    expect(p.peak).toBeGreaterThan(10);
  });
});

describe('hop distances', () => {
  it('reuses the hop table while the graph is unchanged', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.spawnInterval = 0;
    const a = sim.spawn('era', 40, 80, 0, params, true)!;
    const b = sim.spawn('dup', 120, 80, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const first = sim.graph.hopDistances(sim.agents);
    expect(sim.graph.hopDistances(sim.agents)).toBe(first);
    const c = sim.spawn('era', 200, 80, Math.PI, params, true)!;
    expect(sim.graph.hopDistances(sim.agents)).not.toBe(first);
    sim.wire(b.id, 'l', c.id, 'p', params);
    const after = sim.graph.hopDistances(sim.agents);
    expect(after).not.toBe(first);
    expect(after.get(a.id)?.get(c.id)).toBe(2);
  });
});

describe('simulation presets', () => {
  it('steps a soup without throwing', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'soup', params);
    expect(sim.agents.size).toBeGreaterThan(0);
    step(sim, params, 30);
    expect(sim.agents.size).toBeGreaterThan(0);
  });

  it('auto-spawns a free agent about every ten seconds', () => {
    const sim = new Sim(400, 240);
    const params = defaultParams();
    params.spawnInterval = 10;
    params.snapRadius = 0;
    params.maxAgents = 80;
    sim.spawn('era', 200, 120, 0, params, true);
    const n0 = sim.agents.size;
    step(sim, params, 599);
    expect(sim.agents.size).toBe(n0);
    step(sim, params, 2);
    expect(sim.agents.size).toBe(n0 + 1);
  });

  it('filled ports do not deposit scent', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.decay = 0;
    params.diffuse = 0;
    params.deposit = 4;
    params.wireShrink = 10;
    params.rewriteDuration = 10;
    const a = sim.spawn('era', 80, 80, 0, params, true)!;
    const b = sim.spawn('era', 160, 80, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    sim.fields.clear();
    step(sim, params, 8);
    expect(sim.fields.peak()).toBe(0);
  });

  it('rewrites a principal meeting even when aux ports are still free', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    const c = sim.spawn('con', 200, 160, 0, params, true)!;
    const d = sim.spawn('dup', 280, 160, Math.PI, params, true)!;
    sim.wire(c.id, 'p', d.id, 'p', params);
    expect(sim.graph.portsFilled(c)).toBe(false);
    step(sim, params, 50);
    // Commute: the original pair is gone, four agents remain.
    expect(sim.rewrites.length + sim.agents.size).toBeGreaterThan(2);
    expect(sim.agents.size).not.toBe(2);
  });

  it('snap joins facing ports and only once', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 28;
    params.snapArc = 0.45;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    const a = sim.spawn('era', 90, 80, 0, params, true)!;
    const b = sim.spawn('era', 130, 80, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    const between = [...sim.graph.wires.values()].filter(
      (w) =>
        (w.a.id === a.id && w.b.id === b.id) || (w.a.id === b.id && w.b.id === a.id),
    );
    expect(between.length).toBe(1);
  });

  it('does not snap ports that are close but not facing or touching', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 90;
    params.snapArc = 0.3;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    sim.spawn('con', 120, 80, 0, params, true);
    sim.spawn('con', 128, 80, 0, params, true);
    sim.step(1 / 60, params);
    expect(sim.graph.wires.size).toBe(0);
  });

  it('latches when free port tips touch even outside the snap arc', () => {
    const sim = new Sim(320, 200);
    const params = defaultParams();
    params.snapRadius = 40;
    params.snapArc = 0.12;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    // Same heading: not facing. Tips nearly coincident.
    const a = sim.spawn('era', 100, 100, 0, params, true)!;
    const b = sim.spawn('era', 100, 103, 0, params, true)!;
    sim.graph.snap(sim.agents, sim.w, sim.h, params, sim.time);
    expect(sim.graph.wires.size).toBe(1);
    expect(sim.graph.isFree({ id: a.id, slot: 'p' })).toBe(false);
    expect(sim.graph.isFree({ id: b.id, slot: 'p' })).toBe(false);
  });

  it('γ–δ commutation copies into two cons and two dups before further reduction', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    loadPreset(sim, 'commute', params);
    expect(sim.agents.size).toBe(6);
    let sawCopy = false;
    for (let i = 0; i < 120; i++) {
      sim.step(1 / 60, params);
      const kinds = [...sim.agents.values()].map((a) => a.kind);
      if (
        kinds.filter((k) => k === 'con').length === 2 &&
        kinds.filter((k) => k === 'dup').length === 2 &&
        kinds.filter((k) => k === 'era').length === 4
      ) {
        sawCopy = true;
        break;
      }
    }
    expect(sawCopy).toBe(true);
  });

  it('γ–γ annihilation consumes both constructors', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    loadPreset(sim, 'annihilate-con', params);
    step(sim, params, 120);
    expect([...sim.agents.values()].every((a) => a.kind !== 'con')).toBe(true);
  });

  it('δ–δ annihilation consumes both duplicators', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    loadPreset(sim, 'annihilate-dup', params);
    step(sim, params, 120);
    const kinds = [...sim.agents.values()].map((a) => a.kind);
    expect(kinds.every((k) => k === 'era')).toBe(true);
  });

  it('oscillator keeps a net after the first rewrite', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    loadPreset(sim, 'oscillator', params);
    expect(sim.agents.size).toBe(4);
    step(sim, params, 90);
    expect(sim.agents.size).toBeGreaterThan(0);
  });

  it('tracks the mass-weighted center of all shapes', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 0, 0, 0, params, true)!;
    const b = sim.spawn('era', 100, 0, 0, params, true)!;
    const com = sim.centerOfMass()!;
    const m = a.mass + b.mass;
    expect(com.x).toBeCloseTo((a.mass * 0 + b.mass * 100) / m);
    expect(com.y).toBeCloseTo(0);
  });

  it('weights flocking by graph hops and ignores disconnected agents', () => {
    const sim = new Sim(400, 200);
    const params = defaultParams();
    params.rewriteDuration = 20;
    const a = sim.spawn('era', 40, 100, 0, params, true)!;
    const b = sim.spawn('con', 120, 100, 0, params, true)!;
    const c = sim.spawn('era', 200, 100, 0, params, true)!;
    const d = sim.spawn('era', 300, 100, 0, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    sim.wire(b.id, 'l', c.id, 'p', params);
    const hops = sim.graph.hopDistances(sim.agents);
    expect(hops.get(a.id)?.get(b.id)).toBe(1);
    expect(hops.get(a.id)?.get(c.id)).toBe(2);
    expect(hops.get(a.id)?.has(d.id)).toBe(false);
    expect(hops.get(d.id)?.get(d.id)).toBe(0);
  });
});

describe('conservative mechanics', () => {
  function passiveParams() {
    const params = defaultParams();
    params.gravity = 0;
    params.homing = 0;
    params.stepSpeed = 0;
    params.turnRate = 0;
    params.snapRadius = 0;
    params.snapWell = 0;
    params.faceAttract = 0;
    params.drag = 0;
    params.angDrag = 0;
    params.deposit = 0;
    params.rewriteDuration = 20;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.spawnInterval = 0;
    return params;
  }

  it('conserves linear and angular momentum in a collision', () => {
    const sim = new Sim(400, 200);
    const params = passiveParams();
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 140, 104, Math.PI, params, true)!;
    a.vx = 50;
    a.omega = 2;
    const before = sim.momentum();
    const e0 = sim.kineticEnergy();
    step(sim, params, 40);
    const after = sim.momentum();
    expect(after.px).toBeCloseTo(before.px, 2);
    expect(after.py).toBeCloseTo(before.py, 2);
    expect(after.L).toBeCloseTo(before.L, 1);
    expect(sim.kineticEnergy()).toBeLessThan(e0 * 1.35);
    expect(sim.kineticEnergy()).toBeGreaterThan(e0 * 0.45);
    expect(b.id).toBeGreaterThan(0);
  });

  it('gravity does not move the center of mass', () => {
    const sim = new Sim(240, 160);
    const params = passiveParams();
    params.gravity = 0.4;
    sim.spawn('era', 40, 40, 0, params, true);
    sim.spawn('con', 180, 120, 1, params, true);
    const com0 = sim.centerOfMass()!;
    step(sim, params, 30);
    const com1 = sim.centerOfMass()!;
    expect(com1.x).toBeCloseTo(com0.x, 3);
    expect(com1.y).toBeCloseTo(com0.y, 3);
  });

  it('homing does not crumple a net toward its own centre', () => {
    const sim = new Sim(480, 240);
    const params = passiveParams();
    params.homing = 2;
    params.homeComp = 2;
    params.wireShrink = 30;
    const nodes = [80, 160, 240, 320].map((x) => sim.spawn('con', x, 120, 0, params, true)!);
    for (let i = 0; i < 3; i++) sim.wire(nodes[i].id, 'r', nodes[i + 1].id, 'l', params);
    const width0 = nodes[3].x - nodes[0].x;
    step(sim, params, 90);
    expect(nodes[3].x - nodes[0].x).toBeGreaterThan(width0 * 0.85);
  });

  it('a scentless stray walks toward the nearest free port, not the origin', () => {
    const sim = new Sim(480, 240);
    const params = passiveParams();
    params.homing = 2;
    params.drag = 0.4;
    const stray = sim.spawn('era', 80, 200, 0, params, true)!;
    const host = sim.spawn('con', 360, 80, 0, params, true)!;
    host.locked = true;
    step(sim, params, 50);
    const d1 = Math.hypot(stray.x - host.x, stray.y - host.y);
    expect(d1).toBeLessThan(Math.hypot(80 - 360, 200 - 80) - 12);
    expect(stray.x).toBeGreaterThan(80);
    expect(host.x).toBeLessThan(360 + 8);
  });

  it('does not latch through an intervening wire', () => {
    const sim = new Sim(400, 240);
    const params = passiveParams();
    params.snapRadius = 90;
    params.snapArc = 0.6;
    const wallA = sim.spawn('era', 200, 40, Math.PI / 2, params, true)!;
    const wallB = sim.spawn('era', 200, 200, -Math.PI / 2, params, true)!;
    sim.wire(wallA.id, 'p', wallB.id, 'p', params);
    const left = sim.spawn('era', 140, 120, 0, params, true)!;
    const right = sim.spawn('era', 260, 120, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    const between = [...sim.graph.wires.values()].filter(
      (w) =>
        (w.a.id === left.id && w.b.id === right.id) ||
        (w.a.id === right.id && w.b.id === left.id),
    );
    expect(between).toHaveLength(0);
    expect(sim.graph.wires.size).toBe(1);
  });

  it('allows wire chains to cross without pushing each other', () => {
    const sim = new Sim(400, 240);
    const params = passiveParams();
    params.springK = 40;
    const h1 = sim.spawn('era', 60, 120, 0, params, true)!;
    const h2 = sim.spawn('era', 340, 120, Math.PI, params, true)!;
    const v1 = sim.spawn('era', 200, 20, Math.PI / 2, params, true)!;
    const v2 = sim.spawn('era', 200, 220, -Math.PI / 2, params, true)!;
    sim.wire(h1.id, 'p', h2.id, 'p', params);
    sim.wire(v1.id, 'p', v2.id, 'p', params);
    step(sim, params, 20);
    const wires = [...sim.graph.wires.values()];
    expect(wires).toHaveLength(2);
    const mid = (w: (typeof wires)[0]) => w.nodes[Math.floor(w.nodes.length / 2)];
    const m0 = mid(wires[0]);
    const m1 = mid(wires[1]);
    expect(Math.hypot(m0.x - m1.x, m0.y - m1.y)).toBeLessThan(18);
  });

  it('keeps scent from diffusing through a wire wall', () => {
    const sim = new Sim(320, 200);
    const params = passiveParams();
    params.diffuse = 0.55;
    params.decay = 0;
    params.deposit = 0;
    params.spawnInterval = 0;
    params.wireShrink = 30;
    const top = sim.spawn('era', 160, 30, Math.PI / 2, params, true)!;
    const bot = sim.spawn('era', 160, 170, -Math.PI / 2, params, true)!;
    sim.wire(top.id, 'p', bot.id, 'p', params);
    sim.setFieldCover(320, 200);
    sim.fields.cover(160, 100, 320, 200);
    for (let y = 40; y <= 90; y += 2) {
      for (let x = 20; x <= 100; x += 2) sim.fields.deposit(CH.conP, x, y, 20);
    }
    expect(sim.fields.sample(CH.conP, 60, 60)).toBeGreaterThan(5);
    expect(sim.fields.sample(CH.conP, 260, 60)).toBeLessThan(0.2);
    for (let i = 0; i < 60; i++) {
      sim.step(1 / 60, params);
    }
    expect(sim.fields.sample(CH.conP, 60, 60)).toBeGreaterThan(0.8);
    expect(sim.fields.sample(CH.conP, 260, 60)).toBeLessThan(1.5);
  });
});

function quietParams() {
  const params = defaultParams();
  params.gravity = 0;
  params.homing = 0;
  params.flockAlign = 0;
  params.flockSep = 0;
  params.snapRadius = 0;
  params.snapWell = 0;
  params.faceAttract = 0;
  params.deposit = 0;
  params.diffuse = 0;
  params.decay = 0;
  params.rewriteDuration = 20;
  params.wireShrink = 20;
  params.spawnInterval = 0;
  // These tests target deterministic mechanics — the scent-to-cruise mapping,
  // locomotion gating — not self-propulsion dynamics. Noise swamps single-run
  // comparisons and persistence delays them past their measurement window, so
  // both are turned off to isolate what is actually under test.
  params.swimNoise = 0;
  params.swimTau = 0.05;
  return params;
}

describe('isolated motion rules', () => {
  it('a free principal still cruises along its heading', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    step(sim, params, 40);
    expect(a.vx).toBeGreaterThan(12);
    expect(Math.abs(a.vy)).toBeLessThan(0.8);
  });

  it('wired eras pull together during the shrink phase', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.wireShrink = 0.45;
    const a = sim.spawn('era', 120, 120, 0, params, true)!;
    const b = sim.spawn('era', 260, 120, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const d0 = Math.hypot(b.x - a.x, b.y - a.y);
    step(sim, params, 20);
    const d1 = Math.hypot(b.x - a.x, b.y - a.y);
    expect(d1).toBeLessThan(d0 - 8);
  });

  it('a settled latch does not keep injecting kinetic energy', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 0;
    params.drag = 0.8;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 200, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 90);
    const e0 = sim.kineticEnergy();
    const v0 = Math.hypot(a.vx, a.vy) + Math.hypot(b.vx, b.vy);
    step(sim, params, 60);
    expect(sim.kineticEnergy()).toBeLessThan(e0 + 8);
    expect(Math.hypot(a.vx, a.vy) + Math.hypot(b.vx, b.vy)).toBeLessThan(v0 + 4);
  });

  it('does not bounce a constructor off its own wires', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.drag = 0.8;
    const con = sim.spawn('con', 200, 120, 0, params, true)!;
    const left = sim.spawn('era', 140, 90, Math.PI, params, true)!;
    const right = sim.spawn('era', 140, 150, Math.PI, params, true)!;
    const face = sim.spawn('era', 280, 120, Math.PI, params, true)!;
    sim.wire(con.id, 'l', left.id, 'p', params);
    sim.wire(con.id, 'r', right.id, 'p', params);
    sim.wire(con.id, 'p', face.id, 'p', params);
    step(sim, params, 90);
    expect(Math.hypot(con.vx, con.vy)).toBeLessThan(28);
    expect(Math.abs(con.omega)).toBeLessThan(8);
  });

  it('wires do not spontaneously spin a settled pair', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 0;
    params.drag = 1.2;
    params.angDrag = 4;
    const a = sim.spawn('era', 100, 100, 0.2, params, true)!;
    const b = sim.spawn('era', 220, 100, Math.PI - 0.2, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 120);
    const h0 = a.heading;
    const o0 = Math.abs(a.omega) + Math.abs(b.omega);
    step(sim, params, 60);
    expect(Math.abs(a.omega) + Math.abs(b.omega)).toBeLessThan(o0 + 0.4);
    expect(Math.abs(a.heading - h0)).toBeLessThan(0.35);
  });

  it('principal–principal wires tend to 180° heading alignment', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.flockAlign = 14;
    params.angDrag = 0.6;
    params.turnRate = 0;
    const a = sim.spawn('era', 120, 120, 0.35, params, true)!;
    const b = sim.spawn('era', 240, 120, 0.5, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 160);
    expect(Math.abs(Math.abs(angleDelta(a.heading, b.heading)) - Math.PI)).toBeLessThan(0.5);
  });

  it('principal–aux wires tend to 0° heading alignment', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.flockAlign = 14;
    params.angDrag = 0.6;
    params.turnRate = 0;
    const con = sim.spawn('con', 200, 120, 0.2, params, true)!;
    const era = sim.spawn('era', 140, 120, Math.PI * 0.7, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    step(sim, params, 160);
    expect(Math.abs(angleDelta(con.heading, era.heading))).toBeLessThan(0.5);
  });

  it('aux–aux wires tend to 180° heading alignment', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.flockAlign = 16;
    params.angDrag = 0.4;
    params.turnRate = 0;
    const a = sim.spawn('con', 160, 120, 0.15, params, true)!;
    const b = sim.spawn('con', 250, 120, Math.PI + 0.45, params, true)!;
    sim.wire(a.id, 'l', b.id, 'r', params);
    step(sim, params, 220);
    // Both aux ports aim off their neighbour by params.auxSpread * 0.35 rad
    // so wires keep to their own side, which costs exact antiparallelism.
    const auxSplay = params.auxSpread * 0.35;
    expect(Math.abs(Math.abs(angleDelta(a.heading, b.heading)) - Math.PI)).toBeLessThan(
      0.7 + 2 * auxSplay,
    );
  });

  it('stronger scent slows free cruise', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.deposit = 0;
    const clear = sim.spawn('era', 80, 60, 0, params, true)!;
    step(sim, params, 40);
    const vClear = Math.hypot(clear.vx, clear.vy);
    sim.clear();
    const thick = sim.spawn('era', 80, 60, 0, params, true)!;
    for (let i = 0; i < 40; i++) {
      for (let dy = -12; dy <= 12; dy += 4) {
        sim.fields.deposit(CH.conP, 80 + i * 3, 60 + dy, 18);
      }
    }
    step(sim, params, 40);
    const vThick = Math.hypot(thick.vx, thick.vy);
    expect(vClear).toBeGreaterThan(12);
    expect(vThick).toBeLessThan(vClear * 0.75);
  });

  it('stronger scent increases turn agility while slowing cruise', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.deposit = 0;
    const bias = (s: Sim) => {
      for (let i = 0; i < 24; i++) {
        s.fields.deposit(CH.conP, 80 + 22, 60 + 10, 28);
      }
    };
    const clear = sim.spawn('era', 80, 60, 0, params, true)!;
    bias(sim);
    step(sim, params, 20);
    const omegaClear = Math.abs(clear.omega);
    sim.clear();
    const thick = sim.spawn('era', 80, 60, 0, params, true)!;
    for (let i = 0; i < 40; i++) {
      for (let dy = -12; dy <= 12; dy += 4) {
        sim.fields.deposit(CH.conP, 80 + i * 3, 60 + dy, 18);
      }
    }
    bias(sim);
    step(sim, params, 20);
    const vThick = Math.hypot(thick.vx, thick.vy);
    const omegaThick = Math.abs(thick.omega);
    expect(vThick).toBeLessThan(Math.hypot(clear.vx, clear.vy) * 0.85);
    expect(omegaThick).toBeGreaterThan(omegaClear * 1.15);
  });

  it('an aux-only wire does not disable constructor locomotion', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.springK = 6;
    params.flockAlign = 5;
    const con = sim.spawn('con', 180, 120, 0, params, true)!;
    const era = sim.spawn('era', 140, 120, Math.PI, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    expect(sim.graph.isFree({ id: con.id, slot: 'p' })).toBe(true);
    expect(sim.graph.isFree({ id: era.id, slot: 'p' })).toBe(false);
    step(sim, params, 35);
    expect(con.vx).toBeGreaterThan(10);
    expect(Math.abs(con.omega)).toBeLessThan(5);
  });

  it('towing does not penalize cruise speed with component mass', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.flockAlign = 5;
    const solo = sim.spawn('con', 100, 120, 0, params, true)!;
    step(sim, params, 40);
    const vSolo = solo.vx;
    sim.clear();
    const con = sim.spawn('con', 180, 120, 0, params, true)!;
    const era = sim.spawn('era', 140, 120, Math.PI, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    step(sim, params, 40);
    expect(con.vx).toBeGreaterThan(vSolo * 0.5);
    expect(Math.abs(con.omega)).toBeLessThan(5);
  });

  it('does not apply flock separation to disconnected agents', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.flockSep = 90;
    params.flockAlign = 8;
    params.stepSpeed = 0;
    const a = sim.spawn('era', 100, 100, 0, params, true)!;
    const b = sim.spawn('era', 145, 100, Math.PI, params, true)!;
    const d0 = Math.hypot(b.x - a.x, b.y - a.y);
    step(sim, params, 25);
    const d1 = Math.hypot(b.x - a.x, b.y - a.y);
    expect(Math.abs(d1 - d0)).toBeLessThan(3);
  });

  it('reports a blocking wire on a would-be latch chord', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    const wallA = sim.spawn('era', 200, 40, Math.PI / 2, params, true)!;
    const wallB = sim.spawn('era', 200, 200, -Math.PI / 2, params, true)!;
    sim.wire(wallA.id, 'p', wallB.id, 'p', params);
    const left = sim.spawn('era', 140, 120, 0, params, true)!;
    const right = sim.spawn('era', 260, 120, Math.PI, params, true)!;
    expect(
      sim.graph.latchCrosses(sim.agents, { id: left.id, slot: 'p' }, { id: right.id, slot: 'p' }, sim.w, sim.h),
    ).toBe(true);
  });

  it('does not treat a clear gap as a crossing latch', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    const wallA = sim.spawn('era', 80, 40, Math.PI / 2, params, true)!;
    const wallB = sim.spawn('era', 80, 200, -Math.PI / 2, params, true)!;
    sim.wire(wallA.id, 'p', wallB.id, 'p', params);
    const left = sim.spawn('era', 200, 120, 0, params, true)!;
    const right = sim.spawn('era', 280, 120, Math.PI, params, true)!;
    expect(
      sim.graph.latchCrosses(sim.agents, { id: left.id, slot: 'p' }, { id: right.id, slot: 'p' }, sim.w, sim.h),
    ).toBe(false);
  });
});

describe('physics lod', () => {
  const zoomedOut = { x: 120, y: 80, zoom: 0.18, viewW: 800, viewH: 600 };
  const closeUp = { x: 120, y: 80, zoom: 2, viewW: 400, viewH: 300 };

  it('still separates overlapping agents when they are FAR', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('era', 120, 80, 0, params, true)!;
    const b = sim.spawn('era', 121, 80, Math.PI, params, true)!;
    for (let i = 0; i < 12; i++) sim.step(1 / 60, params, zoomedOut);
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(12);
  });

  it('does not emit Hertzian contacts for FAR pairs', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    sim.spawn('era', 120, 80, 0, params, true);
    sim.spawn('era', 121, 80, Math.PI, params, true);
    sim.step(1 / 60, params, zoomedOut);
    expect(sim.contacts.size).toBe(0);
  });

  it('keeps Hertzian contacts when the same pair is NEAR', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    sim.spawn('era', 120, 80, 0, params, true);
    sim.spawn('era', 121, 80, Math.PI, params, true);
    sim.step(1 / 60, params, closeUp);
    expect(sim.contacts.size).toBeGreaterThan(0);
  });

  it('stepAsync without a GPU device still separates FAR overlap', async () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('era', 120, 80, 0, params, true)!;
    const b = sim.spawn('era', 121, 80, Math.PI, params, true)!;
    for (let i = 0; i < 12; i++) await sim.stepAsync(1 / 60, params, zoomedOut);
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(12);
  });

  it('does not change the no-view path: overlapping triangles still sit on SAT', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('con', 120, 80, 0, params, true)!;
    const b = sim.spawn('con', 121, 80, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    expect(sim.contacts.size).toBeGreaterThan(0);
    for (let i = 0; i < 11; i++) sim.step(1 / 60, params);
    expect(queryHit(a, b, sim.w, sim.h)?.overlap ?? 0).toBeLessThanOrEqual(SLOP + 0.08);
  });

  it('does not promote a whole chain because one end is on screen', () => {
    const sim = new Sim(2000, 200);
    const params = defaultParams();
    params.snapRadius = 0;
    params.spawnInterval = 0;
    params.stepSpeed = 0;
    const ids: number[] = [];
    for (let i = 0; i < 16; i++) {
      ids.push(sim.spawn('era', 80 + i * 70, 100, 0, params, true)!.id);
      if (i > 0) sim.wire(ids[i - 1], 'p', ids[i], 'p', params);
    }
    sim.step(1 / 60, params, { x: 80, y: 100, zoom: 2, viewW: 400, viewH: 300 });
    expect(sim.isPhysicsDetailed(ids[0])).toBe(true);
    expect(sim.isPhysicsDetailed(ids[15])).toBe(false);
  });

  it('still rewrites a FAR era–era pair', () => {
    const sim = new Sim(240, 160);
    const params = fastParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('era', 100, 80, 0, params, true)!;
    const b = sim.spawn('era', 140, 80, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const far = { x: 120, y: 80, zoom: 0.18, viewW: 800, viewH: 600 };
    for (let i = 0; i < 80; i++) sim.step(1 / 60, params, far);
    expect(sim.rewrites.length + (2 - sim.agents.size)).toBeGreaterThan(0);
    expect(sim.agents.size).toBeLessThan(2);
  });
});

describe('native mixed solve', () => {
  it('SAT still separates overlapping triangles with no view', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('con', 120, 80, 0, params, true)!;
    const b = sim.spawn('con', 121, 80, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    expect(sim.contacts.size).toBeGreaterThan(0);
    for (let i = 0; i < 11; i++) sim.step(1 / 60, params);
    expect(queryHit(a, b, sim.w, sim.h)?.overlap ?? 0).toBeLessThanOrEqual(SLOP + 0.08);
  });

  it('keeps Hertzian contacts when the same pair is NEAR', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    sim.spawn('era', 120, 80, 0, params, true);
    sim.spawn('era', 121, 80, Math.PI, params, true);
    sim.step(1 / 60, params, { x: 120, y: 80, zoom: 2, viewW: 400, viewH: 300 });
    expect(sim.contacts.size).toBeGreaterThan(0);
  });

  it('holds a wired pair near rest without injecting energy', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 0;
    params.drag = 0.8;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 200, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const rest = [...sim.graph.wires.values()][0].rest;
    for (let i = 0; i < 90; i++) sim.step(1 / 60, params);
    const sa = stemWorld(a, 'p', sim.w, sim.h);
    const sb = stemWorld(b, 'p', sim.w, sim.h);
    expect(Math.abs(Math.hypot(sb.x - sa.x, sb.y - sa.y) - rest)).toBeLessThan(8);
    const e0 = sim.kineticEnergy();
    for (let i = 0; i < 60; i++) sim.step(1 / 60, params);
    expect(sim.kineticEnergy()).toBeLessThan(e0 + 8);
  });
});
