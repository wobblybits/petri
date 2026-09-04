import { afterEach, describe, expect, it } from 'vitest';
import { createAgent, stemWorld, wireCubic, type Agent } from './agents.ts';
import { bezierLength, bezierPoint } from './curve.ts';
import { closestPointOnSegment, wireBowBudget } from './geom.ts';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { wireStrokePoints } from './render.ts';
import { Sim } from './sim.ts';
import type { Wire } from './graph.ts';

const realRandom = Math.random;
afterEach(() => {
  Math.random = realRandom;
});

function seed(n: number): void {
  let s = n >>> 0;
  Math.random = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function quietParams() {
  const params = defaultParams();
  params.snapRadius = 0;
  params.stepSpeed = 0;
  params.flockAlign = 0;
  params.flockSep = 0;
  params.declutter = 0;
  params.rewriteDuration = 20;
  params.spawnInterval = 0;
  params.faceAttract = 0;
  params.snapWell = 0;
  return params;
}

function pathLength(pts: { x: number; y: number }[]): number {
  let len = 0;
  for (let i = 1; i < pts.length; i++) {
    len += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  }
  return len;
}

function distToSegment(
  p: { x: number; y: number },
  a: { x: number; y: number },
  b: { x: number; y: number },
): number {
  const q = closestPointOnSegment(p.x, p.y, a.x, a.y, b.x, b.y);
  return Math.hypot(p.x - q.x, p.y - q.y);
}

function maxSegmentDeviation(
  pts: { x: number; y: number }[],
  a: { x: number; y: number },
  b: { x: number; y: number },
): number {
  let max = 0;
  for (const p of pts) max = Math.max(max, distToSegment(p, a, b));
  return max;
}

function strokeOf(sim: Sim, wire: Wire): { x: number; y: number }[] {
  const A = sim.agents.get(wire.a.id)!;
  const B = sim.agents.get(wire.b.id)!;
  return wireStrokePoints(A, B, wire, sim.w, sim.h);
}

function stems(sim: Sim, A: Agent, B: Agent, wire: Wire) {
  return {
    a: stemWorld(A, wire.a.slot, sim.w, sim.h),
    b: stemWorld(B, wire.b.slot, sim.w, sim.h),
  };
}

/** A body-sized bow is fine. A loop that leaves the two anchors' neighbourhood is not. */
function bowBudget(span: number, rest: number): number {
  return wireBowBudget(span, rest);
}

function strokeReport(sim: Sim, wire: Wire): {
  span: number;
  rest: number;
  lastLen: number;
  dev: number;
  len: number;
} {
  const A = sim.agents.get(wire.a.id)!;
  const B = sim.agents.get(wire.b.id)!;
  const s = stems(sim, A, B, wire);
  const span = Math.hypot(s.b.x - s.a.x, s.b.y - s.a.y);
  const pts = strokeOf(sim, wire);
  return {
    span,
    rest: wire.rest,
    lastLen: wire.lastLen,
    dev: maxSegmentDeviation(pts, s.a, s.b),
    len: pathLength(pts),
  };
}

function expectStrokeSane(sim: Sim, wire: Wire, label: string): void {
  const r = strokeReport(sim, wire);
  // The renderer clamps to exactly this budget, so a bowed stroke sits *on*
  // the bound rather than under it and the comparison is a float tie. One ulp
  // of slack, not a loosened bound.
  const budget = bowBudget(r.span, r.rest);
  expect(
    r.dev,
    `${label}: stroke ${r.dev.toFixed(1)} px off a ${r.span.toFixed(1)} px chord (lastLen ${r.lastLen.toFixed(1)}, stroke ${r.len.toFixed(1)})`,
  ).toBeLessThanOrEqual(budget * (1 + 1e-9));
}

function wiredPair(sim: Sim, params: ReturnType<typeof quietParams>) {
  const a = sim.spawn('era', 80, 100, 0, params, true)!;
  const b = sim.spawn('era', 220, 100, Math.PI, params, true)!;
  sim.wire(a.id, 'p', b.id, 'p', params);
  const wire = [...sim.graph.wires.values()][0];
  return { a, b, wire };
}

describe('rest cubic does not loop off the chord', () => {
  const params = defaultParams();
  const w = 400;
  const h = 240;
  const headings = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];

  it('stays near the stem chord when ports are close and rest is long', () => {
    for (const hA of headings) {
      for (const hB of headings) {
        const A = createAgent(1, 'era', 200, 120, hA, params);
        const B = createAgent(2, 'era', 208, 120, hB, params);
        const c = wireCubic(A, 'p', B, 'p', w, h, 200);
        const sA = stemWorld(A, 'p', w, h);
        const sB = stemWorld(B, 'p', w, h);
        const span = Math.hypot(sB.x - sA.x, sB.y - sA.y);
        const samples: { x: number; y: number }[] = [];
        for (let i = 0; i <= 32; i++) samples.push(bezierPoint(c.p0, c.p1, c.p2, c.p3, i / 32));
        const curveDev = maxSegmentDeviation(samples, sA, sB);
        const handleDev = Math.max(distToSegment(c.p1, sA, sB), distToSegment(c.p2, sA, sB));
        const len = bezierLength(c.p0, c.p1, c.p2, c.p3);
        expect(
          curveDev,
          `headings ${hA.toFixed(2)}, ${hB.toFixed(2)}: curve ${curveDev.toFixed(1)} px on span ${span.toFixed(1)}`,
        ).toBeLessThan(bowBudget(span, 200));
        expect(
          handleDev,
          `headings ${hA.toFixed(2)}, ${hB.toFixed(2)}: handle ${handleDev.toFixed(1)} px off a ${span.toFixed(1)} px chord`,
        ).toBeLessThan(bowBudget(span, 200));
        expect(len).toBeLessThan(span * 4 + 80);
      }
    }
  });
});

describe('drawn ropes stay near their anchors during collisions', () => {
  it('a visitor draping the rope does not throw the stroke off-screen', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    const a = sim.spawn('era', 60, 100, 0, params, true)!;
    const b = sim.spawn('era', 260, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    const mid = wire.nodes[Math.floor(wire.nodes.length / 2)];
    sim.spawn('con', mid.x, mid.y, 0, params, true);
    for (let i = 0; i < 45; i++) {
      sim.step(1 / 60, params);
      expectStrokeSane(sim, wire, `drape frame ${i}`);
    }
  });

  it('a fast body punching through a rope does not throw the stroke off-screen', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    const { wire } = wiredPair(sim, params);
    const visitor = sim.spawn('con', 160, 20, 0, params, true)!;
    visitor.vy = 900;
    for (let i = 0; i < 50; i++) {
      sim.step(1 / 60, params);
      expectStrokeSane(sim, wire, `punch frame ${i}`);
    }
  });

  it('crushing the two anchors together does not loop the rope off-screen', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    const { a, b, wire } = wiredPair(sim, params);
    a.locked = true;
    b.vx = -700;
    for (let i = 0; i < 40; i++) {
      sim.step(1 / 60, params);
      expectStrokeSane(sim, wire, `crush frame ${i}`);
    }
  });

  it('spinning ports do not throw Catmull handles off-screen', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    const { a, b, wire } = wiredPair(sim, params);
    a.omega = 28;
    b.omega = -28;
    for (let i = 0; i < 40; i++) {
      sim.step(1 / 60, params);
      expectStrokeSane(sim, wire, `spin frame ${i}`);
    }
  });

  it('two ropes shoved through each other do not throw a stroke off-screen', () => {
    const sim = new Sim(320, 240);
    const params = quietParams();
    const a = sim.spawn('era', 40, 80, 0, params, true)!;
    const b = sim.spawn('era', 280, 80, Math.PI, params, true)!;
    const c = sim.spawn('era', 160, 20, Math.PI / 2, params, true)!;
    const d = sim.spawn('era', 160, 220, -Math.PI / 2, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    sim.wire(c.id, 'p', d.id, 'p', params);
    c.vy = 600;
    d.vy = -600;
    const wires = [...sim.graph.wires.values()];
    for (let i = 0; i < 40; i++) {
      sim.step(1 / 60, params);
      for (const wire of wires) expectStrokeSane(sim, wire, `cross frame ${i} wire ${wire.id}`);
    }
  });
});

describe('runaway rope nodes', () => {
  it('the renderer does not follow a node that has been thrown far off the chord', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    const { wire } = wiredPair(sim, params);
    const mid = wire.nodes[Math.floor(wire.nodes.length / 2)];
    mid.x = 160;
    mid.y = 1000;
    const r = strokeReport(sim, wire);
    expectStrokeSane(sim, wire, 'runaway node');
    expect(r.len).toBeLessThan(r.span + 2 * bowBudget(r.span, r.rest) + 80);
  });

  it('a runaway node is gone by the end of the next frame', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    const { wire } = wiredPair(sim, params);
    const mid = wire.nodes[Math.floor(wire.nodes.length / 2)];
    mid.x = 160;
    mid.y = 1000;
    mid.prevX = mid.x;
    mid.prevY = mid.y;
    sim.step(1 / 60, params);
    expectStrokeSane(sim, wire, 'after one frame');
  });
});

describe('a live soup does not grow off-screen wires', () => {
  it('keeps every stroke near its own chord for a few seconds', () => {
    seed(999);
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    // Pinned: this is a geometry check at a particular density, not a test of
    // whatever the product's default soup happens to be.
    params.soupCount = 28;
    loadPreset(sim, 'soup', params);
    for (let i = 0; i < 180; i++) {
      sim.step(1 / 60, params);
      for (const wire of sim.graph.wires.values()) {
        const A = sim.agents.get(wire.a.id);
        const B = sim.agents.get(wire.b.id);
        if (!A || !B) continue;
        expectStrokeSane(sim, wire, `soup frame ${i} wire ${wire.id}`);
      }
    }
  });
});
