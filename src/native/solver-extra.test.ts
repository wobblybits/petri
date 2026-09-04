import { afterEach, describe, expect, it } from 'vitest';
import { createAgent, momentOfInertia, boundRadius, discRadius, cloneAgent, type Agent } from '../agents.ts';
import { solveContact, solveWire, solveWireSpan, type ChainNode } from '../chain.ts';
import { queryHit, SLOP } from '../collide.ts';
import { FAR, FAR_STRIDE } from '../gpu/far-kernel.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { nativeSolver as sharedSolver } from './solver.ts';
import { wrapAngle } from '../wrap.ts';
import {
  HIT,
  KIND_CON,
  KIND_DUP,
  KIND_ERA,
  NativeSolver,
  ND,
  NODE_STRIDE,
  WF_FULL,
  WIRE_NEAR_STRIDE,
  WN,
} from './solver.ts';

function cloneNodes(nodes: ChainNode[]): ChainNode[] {
  return nodes.map((n) => ({ ...n }));
}

function makeNode(x: number, y: number): ChainNode {
  return { x, y, vx: 0, vy: 0, prevX: x, prevY: y, integVx: 0, integVy: 0 };
}

function slotCode(slot: 'p' | 'l' | 'r'): number {
  return slot === 'p' ? 0 : slot === 'l' ? 1 : 2;
}

function packAgent(native: NativeSolver, i: number, a: Agent, detailed: boolean): void {
  const bodies = native.bodies!;
  const o = i * FAR_STRIDE;
  bodies[o + FAR.x] = a.x;
  bodies[o + FAR.y] = a.y;
  bodies[o + FAR.vx] = a.vx;
  bodies[o + FAR.vy] = a.vy;
  bodies[o + FAR.heading] = a.heading;
  bodies[o + FAR.omega] = a.omega;
  bodies[o + FAR.invMass] = a.locked ? 0 : 1 / Math.max(0.08, a.mass);
  bodies[o + FAR.radius] = boundRadius(a);
  bodies[o + FAR.locked] = a.locked ? 1 : 0;
  bodies[o + FAR.prevX] = a.prevX;
  bodies[o + FAR.prevY] = a.prevY;
  bodies[o + FAR.prevHeading] = a.prevHeading;
  native.invInertia![i] = a.locked ? 0 : 1 / Math.max(1e-4, momentOfInertia(a));
  native.scale![i] = a.scale;
  native.kind![i] = a.kind === 'era' ? KIND_ERA : a.kind === 'dup' ? KIND_DUP : KIND_CON;
  native.detailed![i] = detailed ? 1 : 0;
}

function unpackAgent(native: NativeSolver, i: number, a: Agent): void {
  const bodies = native.bodies!;
  const o = i * FAR_STRIDE;
  a.x = bodies[o + FAR.x];
  a.y = bodies[o + FAR.y];
  a.vx = bodies[o + FAR.vx];
  a.vy = bodies[o + FAR.vy];
  a.heading = bodies[o + FAR.heading];
  a.omega = bodies[o + FAR.omega];
}

function packWire(
  native: NativeSolver,
  wi: number,
  spec: {
    a: number;
    b: number;
    rest: number;
    rope: number;
    aSlot: number;
    bSlot: number;
    scale?: number;
    slack?: number;
    node0?: number;
    nNodes?: number;
    flags?: number;
  },
): void {
  const W = native.wiresNear!;
  const o = wi * WIRE_NEAR_STRIDE;
  W[o + WN.a] = spec.a;
  W[o + WN.b] = spec.b;
  W[o + WN.rest] = spec.rest;
  W[o + WN.rope] = spec.rope;
  W[o + WN.scale] = spec.scale ?? 1;
  W[o + WN.slack] = spec.slack ?? 1;
  W[o + WN.aSlot] = spec.aSlot;
  W[o + WN.bSlot] = spec.bSlot;
  W[o + WN.node0] = spec.node0 ?? 0;
  W[o + WN.nNodes] = spec.nNodes ?? 0;
  W[o + WN.flags] = spec.flags ?? 0;
}

function packNodes(native: NativeSolver, node0: number, nodes: ChainNode[]): void {
  const N = native.nodes!;
  for (let i = 0; i < nodes.length; i++) {
    const o = (node0 + i) * NODE_STRIDE;
    N[o + ND.x] = nodes[i].x;
    N[o + ND.y] = nodes[i].y;
  }
}

function expectPose(got: Agent, want: Agent, tol: number): void {
  expect(Math.abs(got.x - want.x)).toBeLessThan(tol);
  expect(Math.abs(got.y - want.y)).toBeLessThan(tol);
  expect(Math.abs(wrapAngle(got.heading - want.heading))).toBeLessThan(tol);
}

function expectFinite(a: Agent): void {
  expect(Number.isFinite(a.x) && Number.isFinite(a.y) && Number.isFinite(a.heading)).toBe(true);
}

/** Translation-only FAR disc pair, matching native/solver.c near_contacts disc branch. */
function solveDiscPair(A: Agent, B: Agent, h: number): void {
  const dx = B.x - A.x;
  const dy = B.y - A.y;
  const dist = Math.hypot(dx, dy);
  const keep = discRadius(A) + discRadius(B);
  if (dist >= keep || dist < 1e-6) return;
  const depth = keep - dist - SLOP;
  if (depth <= 0) return;
  const wA = A.locked ? 0 : 1 / Math.max(0.08, A.mass);
  const wB = B.locked ? 0 : 1 / Math.max(0.08, B.mass);
  const denom = wA + wB + 4.0e-6 / Math.max(1e-12, h * h);
  if (denom < 1e-12) return;
  const s = depth / denom / dist;
  if (!A.locked && wA > 0) {
    A.x -= dx * s * wA;
    A.y -= dy * s * wA;
  }
  if (!B.locked && wB > 0) {
    B.x += dx * s * wB;
    B.y += dy * s * wB;
  }
}

function expectHitClose(
  H: Float32Array,
  hit: NonNullable<ReturnType<typeof queryHit>>,
  aIdx: number,
  bIdx: number,
  tol: number,
): void {
  const ia = H[HIT.a];
  const ib = H[HIT.b];
  const flip = ia === bIdx && ib === aIdx;
  expect(flip || (ia === aIdx && ib === bIdx)).toBe(true);
  const s = flip ? -1 : 1;
  expect(Math.abs(H[HIT.overlap] - hit.overlap)).toBeLessThan(tol);
  expect(Math.abs(H[HIT.nx] - s * hit.nx)).toBeLessThan(tol);
  expect(Math.abs(H[HIT.ny] - s * hit.ny)).toBeLessThan(tol);
}

describe('native WASM solver extras', () => {
  it('matches four sequential JS spans on disjoint era-era wires in one nearWires pass', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const h = 1 / 60 / 8;
    const specs = [
      { ax: 0, ay: 0, bx: 80, by: 10, ah: 0.4, bh: -0.3, rest: 40 },
      { ax: 0, ay: 400, bx: 90, by: 390, ah: -0.2, bh: 1.1, rest: 55 },
      { ax: 400, ay: 0, bx: 430, by: 20, ah: 0.8, bh: -1.0, rest: 32 },
      { ax: 400, ay: 400, bx: 480, by: 415, ah: 2.0, bh: 0.1, rest: 70 },
    ];
    const agents: Agent[] = [];
    const js: Agent[] = [];
    for (let i = 0; i < specs.length; i++) {
      const s = specs[i];
      const a = createAgent(i * 2 + 1, 'era', s.ax, s.ay, s.ah, params);
      const b = createAgent(i * 2 + 2, 'era', s.bx, s.by, s.bh, params);
      agents.push(a, b);
      js.push(cloneAgent(a), cloneAgent(b));
    }
    const stiff = { scale: 1, slack: 1 };
    for (let i = 0; i < specs.length; i++) {
      solveWireSpan(js[i * 2], 'p', js[i * 2 + 1], 'p', specs[i].rest, stiff, h);
    }
    for (let i = 0; i < agents.length; i++) packAgent(native, i, agents[i], true);
    for (let i = 0; i < specs.length; i++) {
      packWire(native, i, {
        a: i * 2,
        b: i * 2 + 1,
        rest: specs[i].rest,
        rope: specs[i].rest,
        aSlot: 0,
        bSlot: 0,
      });
    }
    native.nearWires(8, 4, h);
    for (let i = 0; i < agents.length; i++) {
      unpackAgent(native, i, agents[i]);
      expectPose(agents[i], js[i], 2e-4);
    }
  });

  it('matches JS Gauss-Seidel order on a 3-wire cons chain and stays rest-ish', async () => {
    // Shared-body Gauss-Seidel: WASM nearWires walks pack order A-B, B-C, C-D,
    // the same as sequential JS solveWireSpan. They must match; 2e-3 leaves
    // float/double room without hiding a Jacobi/SIMD reorder of the spans.
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const h = 1 / 60 / 8;
    const rest = 40;
    const A = createAgent(1, 'con', 0, 0, 0.35, params);
    const B = createAgent(2, 'con', 70, 6, -0.2, params);
    const C = createAgent(3, 'con', 145, -10, 0.9, params);
    const D = createAgent(4, 'con', 220, 8, -0.55, params);
    const aJs = cloneAgent(A);
    const bJs = cloneAgent(B);
    const cJs = cloneAgent(C);
    const dJs = cloneAgent(D);
    const stiff = { scale: 1, slack: 1 };
    solveWireSpan(aJs, 'p', bJs, 'l', rest, stiff, h);
    solveWireSpan(bJs, 'r', cJs, 'l', rest, stiff, h);
    solveWireSpan(cJs, 'r', dJs, 'p', rest, stiff, h);

    packAgent(native, 0, A, true);
    packAgent(native, 1, B, true);
    packAgent(native, 2, C, true);
    packAgent(native, 3, D, true);
    packWire(native, 0, {
      a: 0,
      b: 1,
      rest,
      rope: rest,
      aSlot: slotCode('p'),
      bSlot: slotCode('l'),
    });
    packWire(native, 1, {
      a: 1,
      b: 2,
      rest,
      rope: rest,
      aSlot: slotCode('r'),
      bSlot: slotCode('l'),
    });
    packWire(native, 2, {
      a: 2,
      b: 3,
      rest,
      rope: rest,
      aSlot: slotCode('r'),
      bSlot: slotCode('p'),
    });
    native.nearWires(4, 3, h);
    unpackAgent(native, 0, A);
    unpackAgent(native, 1, B);
    unpackAgent(native, 2, C);
    unpackAgent(native, 3, D);
    expectPose(A, aJs, 2e-3);
    expectPose(B, bJs, 2e-3);
    expectPose(C, cJs, 2e-3);
    expectPose(D, dJs, 2e-3);
    for (const body of [A, B, C, D]) expectFinite(body);
    const lens = [Math.hypot(B.x - A.x, B.y - A.y), Math.hypot(C.x - B.x, C.y - B.y), Math.hypot(D.x - C.x, D.y - C.y)];
    for (const len of lens) {
      expect(Number.isFinite(len)).toBe(true);
      expect(len).toBeGreaterThan(rest * 0.25);
      expect(len).toBeLessThan(rest * 3);
    }
  });

  it.each([0, Math.PI / 2, 2.2])(
    'SAT: overlapping dup/con triangles at heading %s match queryHit + solveContact',
    async (heading) => {
      const native = new NativeSolver();
      expect(await native.init(), native.lastError).toBe(true);
      const params = defaultParams();
      const a = createAgent(1, 'dup', 0, 0, heading, params);
      const b = createAgent(2, 'con', 2, 0, heading + Math.PI, params);
      packAgent(native, 0, a, true);
      packAgent(native, 1, b, true);
      a.heading = native.bodies![FAR.heading];
      b.heading = native.bodies![FAR_STRIDE + FAR.heading];
      const hit = queryHit(a, b, 240, 160);
      expect(hit).not.toBeNull();
      const aTs = cloneAgent(a);
      const bTs = cloneAgent(b);
      const h = 1 / 60 / 8;
      solveContact(aTs, bTs, hit!, SLOP, h);

      const np = native.nearContacts(2, 0, h);
      unpackAgent(native, 0, a);
      unpackAgent(native, 1, b);
      expect(np).toBeGreaterThan(0);
      expect(native.hitCount()).toBeGreaterThan(0);
      expectHitClose(native.hits!, hit!, 0, 1, 3e-3);
      expectPose(a, aTs, 2e-3);
      expectPose(b, bTs, 2e-3);
    },
  );

  it('SAT miss: far-apart cons are not moved and hitCount is 0', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const a = createAgent(1, 'con', 0, 0, 0.4, params);
    const b = createAgent(2, 'con', 200, 0, -0.3, params);
    const ax = a.x;
    const ay = a.y;
    const ah = a.heading;
    const bx = b.x;
    const by = b.y;
    const bh = b.heading;
    packAgent(native, 0, a, true);
    packAgent(native, 1, b, true);
    native.nearContacts(2, 0, 1 / 60 / 8);
    unpackAgent(native, 0, a);
    unpackAgent(native, 1, b);
    expect(native.hitCount()).toBe(0);
    expect(a.x).toBeCloseTo(ax, 5);
    expect(a.y).toBeCloseTo(ay, 5);
    expect(a.heading).toBeCloseTo(ah, 5);
    expect(b.x).toBeCloseTo(bx, 5);
    expect(b.y).toBeCloseTo(by, 5);
    expect(b.heading).toBeCloseTo(bh, 5);
  });

  it('nearContacts SATs detailed cons and disc-pushes FAR eras in one mixed scene', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const h = 1 / 60 / 8;
    const c0 = createAgent(1, 'con', 0, 0, 0, params);
    const c1 = createAgent(2, 'con', 2, 0, Math.PI, params);
    const e0 = createAgent(3, 'era', 300, 80, 0.2, params);
    const e1 = createAgent(4, 'era', 302, 80, 1.1, params);
    packAgent(native, 0, c0, true);
    packAgent(native, 1, c1, true);
    packAgent(native, 2, e0, false);
    packAgent(native, 3, e1, false);

    const hit = queryHit(c0, c1, 240, 160);
    expect(hit).not.toBeNull();
    const c0Ts = cloneAgent(c0);
    const c1Ts = cloneAgent(c1);
    solveContact(c0Ts, c1Ts, hit!, SLOP, h);
    const e0Ts = cloneAgent(e0);
    const e1Ts = cloneAgent(e1);
    solveDiscPair(e0Ts, e1Ts, h);

    native.nearContacts(4, 0, h);
    unpackAgent(native, 0, c0);
    unpackAgent(native, 1, c1);
    unpackAgent(native, 2, e0);
    unpackAgent(native, 3, e1);

    expect(native.hitCount()).toBe(1);
    expectHitClose(native.hits!, hit!, 0, 1, 3e-3);
    expectPose(c0, c0Ts, 2e-3);
    expectPose(c1, c1Ts, 2e-3);
    expectPose(e0, e0Ts, 2e-3);
    expectPose(e1, e1Ts, 2e-3);
    expect(Math.hypot(e1.x - e0.x, e1.y - e0.y)).toBeGreaterThan(2);
  });

  it('stepNear 8 substeps separates two overlapping cons beyond bound radii', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const a = createAgent(1, 'con', 0, 0, 0, params);
    const b = createAgent(2, 'con', 2, 0, Math.PI, params);
    packAgent(native, 0, a, true);
    packAgent(native, 1, b, true);
    expect(native.stepNear(2, 0, 1 / 60, 8, 1, -1, 160, 0, 0)).toBe(true);
    unpackAgent(native, 0, a);
    unpackAgent(native, 1, b);
    expectFinite(a);
    expectFinite(b);
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(boundRadius(a) + boundRadius(b));
  });

  it('matches JS solveWire on two disjoint 5-node ropes in one nearWires pass', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const h = 1 / 60 / 8;
    const stiff = { scale: 1, slack: 1 };
    const a0 = createAgent(1, 'era', 60, 100, 0, params);
    const b0 = createAgent(2, 'era', 300, 100, Math.PI, params);
    const n0: ChainNode[] = [
      makeNode(100, 100),
      makeNode(140, 128),
      makeNode(180, 90),
      makeNode(220, 118),
      makeNode(260, 100),
    ];
    const a1 = createAgent(3, 'era', 60, 500, 0.5, params);
    const b1 = createAgent(4, 'era', 320, 510, -0.8, params);
    const n1: ChainNode[] = [
      makeNode(110, 505),
      makeNode(155, 470),
      makeNode(200, 530),
      makeNode(245, 490),
      makeNode(280, 508),
    ];
    const a0Ts = cloneAgent(a0);
    const b0Ts = cloneAgent(b0);
    const n0Ts = cloneNodes(n0);
    const a1Ts = cloneAgent(a1);
    const b1Ts = cloneAgent(b1);
    const n1Ts = cloneNodes(n1);
    solveWire(a0Ts, 'p', b0Ts, 'p', n0Ts, 150, 150, [], stiff, h);
    solveWire(a1Ts, 'p', b1Ts, 'p', n1Ts, 160, 160, [], stiff, h);

    packAgent(native, 0, a0, true);
    packAgent(native, 1, b0, true);
    packAgent(native, 2, a1, true);
    packAgent(native, 3, b1, true);
    packWire(native, 0, {
      a: 0,
      b: 1,
      rest: 150,
      rope: 150,
      aSlot: 0,
      bSlot: 0,
      node0: 0,
      nNodes: 5,
      flags: WF_FULL,
    });
    packWire(native, 1, {
      a: 2,
      b: 3,
      rest: 160,
      rope: 160,
      aSlot: 0,
      bSlot: 0,
      node0: 5,
      nNodes: 5,
      flags: WF_FULL,
    });
    packNodes(native, 0, n0);
    packNodes(native, 5, n1);
    native.nearWires(4, 2, h);
    unpackAgent(native, 0, a0);
    unpackAgent(native, 1, b0);
    unpackAgent(native, 2, a1);
    unpackAgent(native, 3, b1);
    expectPose(a0, a0Ts, 2e-4);
    expectPose(b0, b0Ts, 2e-4);
    expectPose(a1, a1Ts, 2e-4);
    expectPose(b1, b1Ts, 2e-4);
    const N = native.nodes!;
    for (let i = 0; i < 5; i++) {
      const o0 = i * NODE_STRIDE;
      const o1 = (5 + i) * NODE_STRIDE;
      expect(Math.abs(N[o0 + ND.x] - n0Ts[i].x)).toBeLessThan(2e-4);
      expect(Math.abs(N[o0 + ND.y] - n0Ts[i].y)).toBeLessThan(2e-4);
      expect(Math.abs(N[o1 + ND.x] - n1Ts[i].x)).toBeLessThan(2e-4);
      expect(Math.abs(N[o1 + ND.y] - n1Ts[i].y)).toBeLessThan(2e-4);
    }
  });

  it('matches JS solveWire on four disjoint 5-node ropes (SIMD batch)', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const h = 1 / 60 / 8;
    const stiff = { scale: 1, slack: 1 };
    const specs = [
      { ax: 40, ay: 80, bx: 280, by: 80, ha: 0, hb: Math.PI, rest: 140 },
      { ax: 40, ay: 240, bx: 300, by: 250, ha: 0.4, hb: -1.1, rest: 155 },
      { ax: 40, ay: 400, bx: 260, by: 410, ha: -0.3, hb: 2.2, rest: 130 },
      { ax: 40, ay: 560, bx: 310, by: 540, ha: 1.2, hb: -2.5, rest: 165 },
    ];
    const agents: Agent[] = [];
    const nodes: ChainNode[][] = [];
    const jsAgents: Agent[] = [];
    const jsNodes: ChainNode[][] = [];
    specs.forEach((s, i) => {
      const a = createAgent(i * 2 + 1, 'era', s.ax, s.ay, s.ha, params);
      const b = createAgent(i * 2 + 2, 'era', s.bx, s.by, s.hb, params);
      const nd: ChainNode[] = [];
      for (let k = 0; k < 5; k++) {
        const t = (k + 1) / 6;
        nd.push(
          makeNode(s.ax + (s.bx - s.ax) * t, s.ay + (s.by - s.ay) * t + ((k % 2) * 2 - 1) * 12),
        );
      }
      agents.push(a, b);
      nodes.push(nd);
      const aTs = cloneAgent(a);
      const bTs = cloneAgent(b);
      const nTs = cloneNodes(nd);
      solveWire(aTs, 'p', bTs, 'p', nTs, s.rest, s.rest, [], stiff, h);
      jsAgents.push(aTs, bTs);
      jsNodes.push(nTs);
    });
    agents.forEach((ag, i) => packAgent(native, i, ag, true));
    specs.forEach((s, i) => {
      packWire(native, i, {
        a: i * 2,
        b: i * 2 + 1,
        rest: s.rest,
        rope: s.rest,
        aSlot: 0,
        bSlot: 0,
        node0: i * 5,
        nNodes: 5,
        flags: WF_FULL,
      });
      packNodes(native, i * 5, nodes[i]);
    });
    native.nearWires(8, 4, h);
    const N = native.nodes!;
    for (let i = 0; i < 4; i++) {
      unpackAgent(native, i * 2, agents[i * 2]);
      unpackAgent(native, i * 2 + 1, agents[i * 2 + 1]);
      expectPose(agents[i * 2], jsAgents[i * 2], 2e-4);
      expectPose(agents[i * 2 + 1], jsAgents[i * 2 + 1], 2e-4);
      for (let k = 0; k < 5; k++) {
        const o = (i * 5 + k) * NODE_STRIDE;
        expect(Math.abs(N[o + ND.x] - jsNodes[i][k].x)).toBeLessThan(2e-4);
        expect(Math.abs(N[o + ND.y] - jsNodes[i][k].y)).toBeLessThan(2e-4);
      }
    }
  });
});

// `Sim.nativeForces` is a static, and vitest reuses a worker process across
// test files. A test that throws between flipping it and flipping it back
// leaks the JS path into whatever file runs next in that worker, which shows
// up as unrelated tests failing in some runs and not others.
afterEach(() => {
  Sim.nativeForces = true;
});

describe('force passes: WASM against the JS reference', () => {
  /** A wired net with every port kind in play and nothing else running. */
  function net(): { sim: Sim; params: Params } {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.snapRadius = 0;
    params.stepSpeed = 0;
    params.sense = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.declutter = 0;
    params.uncross = 0;
    params.rewriteDuration = 0;
    params.upkeep = 0;
    const sim = new Sim(600, 400);
    const con = sim.spawn('con', 280, 200, 0.3, params, true)!;
    const dup = sim.spawn('dup', 340, 210, Math.PI * 0.8, params, true)!;
    const e1 = sim.spawn('era', 240, 250, 1.1, params, true)!;
    const e2 = sim.spawn('era', 380, 160, -0.6, params, true)!;
    const e3 = sim.spawn('era', 250, 150, 2.4, params, true)!;
    sim.wire(con.id, 'p', dup.id, 'p', params);
    sim.wire(con.id, 'l', e1.id, 'p', params);
    sim.wire(con.id, 'r', e3.id, 'p', params);
    sim.wire(dup.id, 'l', e2.id, 'p', params);
    for (const a of sim.agents.values()) a.omega = (a.id % 3) - 1;
    return { sim, params };
  }

  function spins(sim: Sim): number[] {
    return [...sim.agents.values()].sort((a, b) => a.id - b.id).map((a) => a.omega);
  }

  it('port torques match to a few parts in 10^4', async () => {
    expect(await sharedSolver.init(), sharedSolver.lastError).toBe(true);
    const dt = 1 / 60;

    Sim.nativeForces = false;
    const js = net();
    js.sim.step(dt, js.params);
    const wantSpin = spins(js.sim);

    Sim.nativeForces = true;
    const wasm = net();
    wasm.sim.step(dt, wasm.params);
    const gotSpin = spins(wasm.sim);

    expect(gotSpin.length).toBe(wantSpin.length);
    for (let i = 0; i < wantSpin.length; i++) {
      expect(
        Math.abs(gotSpin[i] - wantSpin[i]),
        `agent ${i}: wasm ${gotSpin[i].toFixed(6)} vs js ${wantSpin[i].toFixed(6)}`,
      ).toBeLessThan(2e-3);
    }
    // And it is doing something: the torques actually moved the spins.
    expect(wantSpin.some((v, i) => Math.abs(v - ((i % 3) - 1)) > 1e-6)).toBe(true);
  });

  it('holds parity over many frames rather than only the first', async () => {
    expect(await sharedSolver.init(), sharedSolver.lastError).toBe(true);
    Sim.nativeForces = false;
    const js = net();
    Sim.nativeForces = true;
    const wasm = net();
    for (let f = 0; f < 120; f++) {
      Sim.nativeForces = false;
      js.sim.step(1 / 60, js.params);
      Sim.nativeForces = true;
      wasm.sim.step(1 / 60, wasm.params);
    }
    const a = [...js.sim.agents.values()].sort((p, q) => p.id - q.id);
    const b = [...wasm.sim.agents.values()].sort((p, q) => p.id - q.id);
    for (let i = 0; i < a.length; i++) {
      const drift = Math.hypot(b[i].x - a[i].x, b[i].y - a[i].y);
      expect(drift, `agent ${a[i].id} drifted ${drift.toFixed(4)} px over 2 s`).toBeLessThan(1);
    }
  });
});

describe('every ported force pass against its JS reference', () => {
  /** Two crowded nets so declutter has separate components to push apart. */
  function crowd(): { sim: Sim; params: Params } {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.snapRadius = 0;
    params.stepSpeed = 0;
    params.sense = 0;
    params.rewriteDuration = 0;
    params.upkeep = 0;
    params.uncross = 0;
    // The passes under test, all on.
    params.declutter = 1.4;
    params.flockAlign = 0.6;
    params.flockSep = 0.6;
    params.portStiff = 1;
    const sim = new Sim(600, 400);
    const mk = (ox: number, oy: number) => {
      const c = sim.spawn('con', ox, oy, 0.2, params, true)!;
      const d = sim.spawn('dup', ox + 44, oy + 6, Math.PI * 0.9, params, true)!;
      const e = sim.spawn('era', ox - 30, oy + 28, 1.0, params, true)!;
      const f = sim.spawn('era', ox + 74, oy - 24, -0.7, params, true)!;
      sim.wire(c.id, 'p', d.id, 'p', params);
      sim.wire(c.id, 'l', e.id, 'p', params);
      sim.wire(d.id, 'r', f.id, 'p', params);
    };
    // Close enough that the two nets crowd each other.
    mk(250, 190);
    mk(300, 215);
    return { sim, params };
  }

  it('a full frame of forces agrees whichever side runs them', async () => {
    expect(await sharedSolver.init(), sharedSolver.lastError).toBe(true);
    Sim.nativeForces = false;
    const js = crowd();
    Sim.nativeForces = true;
    const wasm = crowd();
    for (let f = 0; f < 120; f++) {
      Sim.nativeForces = false;
      js.sim.step(1 / 60, js.params);
      Sim.nativeForces = true;
      wasm.sim.step(1 / 60, wasm.params);
    }
    const a = [...js.sim.agents.values()].sort((p, q) => p.id - q.id);
    const b = [...wasm.sim.agents.values()].sort((p, q) => p.id - q.id);
    expect(b.length).toBe(a.length);
    let worst = 0;
    for (let i = 0; i < a.length; i++) {
      worst = Math.max(worst, Math.hypot(b[i].x - a[i].x, b[i].y - a[i].y));
    }
    expect(worst, `worst drift ${worst.toFixed(4)} px over 2 s`).toBeLessThan(2);
    // The scene has to have actually moved, or agreement is meaningless.
    const spread = Math.hypot(a[0].x - a[4].x, a[0].y - a[4].y);
    expect(spread, 'declutter pushed the two nets apart').toBeGreaterThan(20);
  });
});

describe('steering: WASM against the JS reference', () => {
  /**
   * Foraging, face attraction, the snap well and locomotion all live.
   *
   * `swimNoise` is 0 on purpose. The Ornstein-Uhlenbeck kick draws from
   * Math.random, and the two paths walk the bodies in different orders — the
   * JS pass in spatial-grid order, the packed one in id order — so the two
   * runs would consume the stream differently and the comparison would be
   * measuring the noise rather than the steering.
   */
  function forager(): { sim: Sim; params: Params } {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.rewriteDuration = 0;
    params.upkeep = 0;
    params.uncross = 0;
    params.swimNoise = 0;
    params.wireShrink = 0.9;
    params.sense = 220;
    params.stepSpeed = 30;
    params.faceAttract = 40;
    params.snapWell = 60;
    params.snapRadius = 40;
    const sim = new Sim(600, 400);
    sim.setViewExtent(600, 400);
    for (let i = 0; i < 14; i++) {
      const kind = i % 3 === 0 ? 'era' : i % 3 === 1 ? 'con' : 'dup';
      sim.spawn(kind, 120 + (i * 61) % 360, 110 + (i * 97) % 200, i * 0.83, params, true);
    }
    // A wired pair, so the principal-wire bias term is exercised too.
    const ids = [...sim.agents.keys()];
    sim.wire(ids[1], 'p', ids[2], 'p', params);
    // Lay down scent so the sensors have a gradient to climb.
    for (let f = 0; f < 30; f++) sim.step(1 / 60, params);
    return { sim, params };
  }

  it('agrees on where a foraging soup ends up', async () => {
    expect(await sharedSolver.init(), sharedSolver.lastError).toBe(true);
    Sim.nativeForces = false;
    const js = forager();
    Sim.nativeForces = true;
    const wasm = forager();
    const start = [...js.sim.agents.values()]
      .sort((p, q) => p.id - q.id)
      .map((x) => ({ x: x.x, y: x.y }));
    for (let f = 0; f < 90; f++) {
      Sim.nativeForces = false;
      js.sim.step(1 / 60, js.params);
      Sim.nativeForces = true;
      wasm.sim.step(1 / 60, wasm.params);
    }
    const a = [...js.sim.agents.values()].sort((p, q) => p.id - q.id);
    const b = [...wasm.sim.agents.values()].sort((p, q) => p.id - q.id);
    expect(b.length).toBe(a.length);
    let worst = 0;
    let travelled = 0;
    let moved = 0;
    for (let i = 0; i < a.length; i++) {
      worst = Math.max(worst, Math.hypot(b[i].x - a[i].x, b[i].y - a[i].y));
      travelled = Math.max(travelled, Math.hypot(a[i].x - start[i].x, a[i].y - start[i].y));
      moved = Math.max(moved, Math.hypot(a[i].vx, a[i].vy));
    }
    expect(moved, 'the soup is actually swimming').toBeGreaterThan(1);
    /*
     * As a fraction of how far the soup actually went, not in pixels.
     *
     * These are two sims on two code paths run forward independently, so any
     * disagreement compounds: a foraging soup is chaotic, and a steering
     * decision that lands either side of the dead zone in f32 and f64 sends
     * the two copies off on different trajectories from then on. An absolute
     * pixel budget therefore measures how fast the sim separates rather than
     * whether the ports agree, and it tightens on its own every time the field
     * gets coarser — which is what it did when a cell went from a couple of
     * world units to twenty.
     */
    expect(
      worst / travelled,
      `worst drift ${worst.toFixed(2)}px against ${travelled.toFixed(1)}px travelled`,
    ).toBeLessThan(0.12);
  });

  it('agrees on trail strength, which feeds turn authority', async () => {
    expect(await sharedSolver.init(), sharedSolver.lastError).toBe(true);
    Sim.nativeForces = false;
    const js = forager();
    Sim.nativeForces = true;
    const wasm = forager();
    const t = (s: Sim) => [...s.agents.values()].sort((p, q) => p.id - q.id).map((a) => a.trail);
    const wantTrail = t(js.sim);
    const gotTrail = t(wasm.sim);
    expect(Math.max(...wantTrail), 'there is scent to smell').toBeGreaterThan(0);
    for (let i = 0; i < wantTrail.length; i++) {
      expect(Math.abs(gotTrail[i] - wantTrail[i]), `agent ${i} trail`).toBeLessThan(1e-3);
    }
  });
});

describe('scent writing: WASM against the JS reference', () => {
  /** A wired net that lays scent from its free ports. */
  function pond(): { sim: Sim; params: Params } {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.rewriteDuration = 0;
    params.upkeep = 0;
    params.uncross = 0;
    const sim = new Sim(600, 400);
    sim.setViewExtent(600, 400);
    for (let i = 0; i < 12; i++) {
      const kind = i % 3 === 0 ? 'era' : i % 3 === 1 ? 'con' : 'dup';
      sim.spawn(kind, 120 + (i * 61) % 360, 110 + (i * 97) % 200, i * 0.83, params, true);
    }
    const ids = [...sim.agents.keys()];
    sim.wire(ids[1], 'p', ids[2], 'p', params);
    sim.wire(ids[4], 'l', ids[5], 'r', params);
    return { sim, params };
  }

  it('lays the same scent as the JS twin', async () => {
    expect(await sharedSolver.init(), sharedSolver.lastError).toBe(true);
    // The two passes are run on one settled sim from the same field, rather
    // than comparing two sims after a shared history. Anything that steps the
    // whole frame moves the bodies differently on each path, and then this is
    // measuring how fast a chaotic sim separates instead of whether the port
    // is right — the fields read 47% apart that way with nothing wrong.
    const { sim, params } = pond();
    Sim.nativeForces = false;
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params);

    const inner = sim as never as {
      deposit(p: Params): void;
      scentWriteNative(p: Params): boolean;
    };
    const startData = Float32Array.from(sim.fields.data);

    inner.deposit(params);
    const jsData = Float32Array.from(sim.fields.data);

    sim.fields.data.set(startData);
    Sim.nativeForces = true;
    expect(inner.scentWriteNative(params), 'the native path has to have run').toBe(true);

    const cells = sim.fields.cols * sim.fields.rows;
    let peak = 0;
    let worst = 0;
    for (let i = 0; i < cells * 4; i++) {
      peak = Math.max(peak, Math.abs(jsData[i]));
      worst = Math.max(worst, Math.abs(jsData[i] - sim.fields.data[i]));
    }
    expect(peak, 'there is scent to compare').toBeGreaterThan(0.01);
    expect(worst / peak, `worst scent cell off by ${((worst / peak) * 100).toFixed(3)}% of peak`)
      .toBeLessThan(0.002);
  });
});
