import { afterEach, describe, expect, it } from 'vitest';
import { AudioEngine } from './audio/engine.ts';
import type { WorkletInMessage } from './audio/types.ts';
import { closestOnSegments, WIRE_RADIUS } from './geom.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';
import type { Wire } from './graph.ts';
import { stemWorld } from './agents.ts';

const realRandom = Math.random;
afterEach(() => {
  Math.random = realRandom;
});

function quietParams() {
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
  params.faceAttract = 0;
  params.snapWell = 0;
  params.portStiff = 0;
  return params;
}

function ropeClosest(
  sim: Sim,
  P: Wire,
  Q: Wire,
): number {
  const PA = sim.agents.get(P.a.id)!;
  const PB = sim.agents.get(P.b.id)!;
  const QA = sim.agents.get(Q.a.id)!;
  const QB = sim.agents.get(Q.b.id)!;
  const pA = stemWorld(PA, P.a.slot, sim.w, sim.h);
  const pB = stemWorld(PB, P.b.slot, sim.w, sim.h);
  const qA = stemWorld(QA, Q.a.slot, sim.w, sim.h);
  const qB = stemWorld(QB, Q.b.slot, sim.w, sim.h);
  const p = [pA, ...P.nodes, pB];
  const q = [qA, ...Q.nodes, qB];
  let min = Infinity;
  for (let i = 0; i < p.length - 1; i++) {
    for (let j = 0; j < q.length - 1; j++) {
      const c = closestOnSegments(
        p[i].x,
        p[i].y,
        p[i + 1].x,
        p[i + 1].y,
        q[j].x,
        q[j].y,
        q[j + 1].x,
        q[j + 1].y,
      );
      const d = Math.hypot(c.bx - c.ax, c.by - c.ay);
      if (d < min) min = d;
    }
  }
  return min;
}

function nodeSnapshot(wire: Wire): { x: number; y: number }[] {
  return wire.nodes.map((n) => ({ x: n.x, y: n.y }));
}

function maxNodeShift(before: { x: number; y: number }[], wire: Wire): number {
  let max = 0;
  for (let i = 0; i < wire.nodes.length; i++) {
    const d = Math.hypot(wire.nodes[i].x - before[i].x, wire.nodes[i].y - before[i].y);
    if (d > max) max = d;
  }
  return max;
}

describe('wire-wire pairs do not shove', () => {
  it('crossing ropes stay crossed instead of pushing apart', () => {
    const sim = new Sim(320, 240);
    const params = quietParams();
    const a = sim.spawn('era', 40, 120, 0, params, true)!;
    const b = sim.spawn('era', 280, 120, Math.PI, params, true)!;
    const c = sim.spawn('era', 160, 20, Math.PI / 2, params, true)!;
    const d = sim.spawn('era', 160, 220, -Math.PI / 2, params, true)!;
    a.locked = true;
    b.locked = true;
    c.locked = true;
    d.locked = true;
    sim.wire(a.id, 'p', b.id, 'p', params);
    sim.wire(c.id, 'p', d.id, 'p', params);
    const wires = [...sim.graph.wires.values()];
    const before = wires.map(nodeSnapshot);
    const startDist = ropeClosest(sim, wires[0], wires[1]);
    expect(startDist).toBeLessThan(WIRE_RADIUS * 3);
    for (let i = 0; i < 20; i++) sim.step(1 / 60, params);
    expect(ropeClosest(sim, wires[0], wires[1])).toBeLessThan(WIRE_RADIUS * 3);
    for (let i = 0; i < wires.length; i++) {
      expect(maxNodeShift(before[i], wires[i])).toBeLessThan(12);
    }
  });
});

describe('wire-wire slip-slide audio', () => {
  it('posts a symmetric bow on both wire ids while they scrape, then an empty list', () => {
    const sim = new Sim(320, 240);
    const params = quietParams();
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);
    sim.graph.onLatch = (ev) => engine.push(ev, sim.graph, sim.agents);
    engine.contacts = sim.contacts;
    engine.wireContacts = sim.wireContacts;

    const a = sim.spawn('era', 40, 120, 0, params, true)!;
    const b = sim.spawn('era', 280, 120, Math.PI, params, true)!;
    const c = sim.spawn('era', 160, 20, Math.PI / 2, params, true)!;
    const d = sim.spawn('era', 160, 220, -Math.PI / 2, params, true)!;
    a.locked = true;
    b.locked = true;
    sim.wire(a.id, 'p', b.id, 'p', params);
    sim.wire(c.id, 'p', d.id, 'p', params);
    const wires = [...sim.graph.wires.values()];
    c.vy = 220;
    d.vy = 220;

    let scrape: Extract<WorkletInMessage, { type: 'wireContact' }> | undefined;
    for (let i = 0; i < 40; i++) {
      sim.step(1 / 60, params);
      posted.length = 0;
      engine.frame(sim.graph, sim.agents, 1 / 60);
      for (const m of posted) {
        if (m.type === 'wireContact' && m.items.length > 0) scrape = m;
      }
    }
    expect(scrape).toBeDefined();
    const item = scrape!.items[0];
    expect([item.wireA, item.wireB].sort()).toEqual([wires[0].id, wires[1].id].sort());
    expect(item.load).toBeGreaterThan(0);
    expect(Math.abs(item.slide)).toBeGreaterThan(0);
    expect(item.atA).toBeGreaterThan(0);
    expect(item.atA).toBeLessThan(1);
    expect(item.atB).toBeGreaterThan(0);
    expect(item.atB).toBeLessThan(1);

    c.x = 16;
    c.y = 20;
    d.x = 16;
    d.y = 220;
    c.vx = 0;
    c.vy = 0;
    d.vx = 0;
    d.vy = 0;
    let lifted = false;
    for (let i = 0; i < 20; i++) {
      sim.step(1 / 60, params);
      posted.length = 0;
      engine.frame(sim.graph, sim.agents, 1 / 60);
      for (const m of posted) {
        if (m.type === 'wireContact' && m.items.length === 0) lifted = true;
      }
    }
    expect(lifted).toBe(true);
  });
});

describe('wire-body slip-slide audio', () => {
  it('a visitor rubbing a rope posts a bow on that wire, then an empty list', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    const engine = new AudioEngine();
    engine.armWithoutAudio();
    const posted: WorkletInMessage[] = [];
    engine.onPost = (m) => posted.push(m);
    sim.graph.onLatch = (ev) => engine.push(ev, sim.graph, sim.agents);
    engine.contacts = sim.contacts;
    engine.wireContacts = sim.wireContacts;

    const a = sim.spawn('era', 40, 100, 0, params, true)!;
    const b = sim.spawn('era', 280, 100, Math.PI, params, true)!;
    a.locked = true;
    b.locked = true;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    const mid = wire.nodes[Math.floor(wire.nodes.length / 2)];
    const visitor = sim.spawn('con', mid.x, mid.y, 0, params, true)!;
    visitor.vx = 260;
    visitor.omega = 14;

    let scrape: Extract<WorkletInMessage, { type: 'wireContact' }> | undefined;
    let bestSlide = 0;
    for (let i = 0; i < 40; i++) {
      sim.step(1 / 60, params);
      posted.length = 0;
      engine.frame(sim.graph, sim.agents, 1 / 60);
      for (const m of posted) {
        if (m.type !== 'wireContact') continue;
        for (const it of m.items) {
          if (it.wireB !== 0) continue;
          const s = Math.abs(it.slide);
          if (s >= bestSlide) {
            bestSlide = s;
            scrape = m;
          }
        }
      }
    }
    expect(scrape).toBeDefined();
    const item = scrape!.items.find((it) => it.wireB === 0)!;
    expect(item.wireA).toBe(wire.id);
    expect(item.wireB).toBe(0);
    expect(item.load).toBeGreaterThan(0);
    expect(Math.abs(item.slide)).toBeGreaterThan(0);
    expect(item.atA).toBeGreaterThan(0);
    expect(item.atA).toBeLessThan(1);

    visitor.x = 16;
    visitor.y = 20;
    visitor.vx = 0;
    visitor.vy = 0;
    visitor.omega = 0;
    sim.step(1 / 60, params);
    expect([...sim.wireContacts.values()].some((c) => c.wireB === 0)).toBe(false);
  });
});
