import { describe, expect, it } from 'vitest';
import { createAgent, momentOfInertia, boundRadius, type Agent } from '../agents.ts';
import { solveContact, solveWire, solveWireSpan, contactMechanics, type ChainNode } from '../chain.ts';
import { queryHit, SLOP } from '../collide.ts';
import { Fields } from '../fields.ts';
import { FAR, FAR_STRIDE, FAR_SUBSTEPS, FAR_WIRE_STRIDE, packFarWire, stepFarKernel } from '../gpu/far-kernel.ts';
import { defaultParams } from '../params.ts';
import { wrapAngle } from '../wrap.ts';
import {
  HIT,
  KIND_CON,
  KIND_ERA,
  NativeSolver,
  ND,
  NODE_STRIDE,
  WF_FULL,
  WN,
} from './solver.ts';

function particle(
  data: Float32Array,
  i: number,
  x: number,
  y: number,
  radius: number,
  mass = 1,
): void {
  const o = i * FAR_STRIDE;
  data[o + FAR.x] = x;
  data[o + FAR.y] = y;
  data[o + FAR.vx] = 0;
  data[o + FAR.vy] = 0;
  data[o + FAR.heading] = 0;
  data[o + FAR.omega] = 0;
  data[o + FAR.invMass] = 1 / mass;
  data[o + FAR.radius] = radius;
  data[o + FAR.locked] = 0;
}

function cloneAgent(a: Agent): Agent {
  return { ...a };
}

function cloneNodes(nodes: ChainNode[]): ChainNode[] {
  return nodes.map((n) => ({ ...n }));
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
  native.kind![i] = a.kind === 'era' ? KIND_ERA : KIND_CON;
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

function packWires(...rows: number[][]): Float32Array {
  const wires = new Float32Array(rows.length * FAR_WIRE_STRIDE);
  for (let k = 0; k < rows.length; k++) {
    const [a, b, rest, oax = 0, oay = 0, obx = 0, oby = 0] = rows[k];
    packFarWire(wires, k, a, b, rest, oax, oay, obx, oby);
  }
  return wires;
}

describe('native WASM solver', () => {
  it('loads and matches the TS FAR kernel on a two-body overlap', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const ts = new Float32Array(2 * FAR_STRIDE);
    const wa = new Float32Array(2 * FAR_STRIDE);
    particle(ts, 0, 0, 0, 10);
    particle(ts, 1, 2, 0, 10);
    wa.set(ts);
    stepFarKernel(ts, 2, new Float32Array(FAR_WIRE_STRIDE), 0, 1 / 60, FAR_SUBSTEPS);
    expect(native.stepFar(wa, 2, new Float32Array(FAR_WIRE_STRIDE), 0, 1 / 60, FAR_SUBSTEPS)).toBe(true);
    expect(Math.abs(wa[FAR.x] - ts[FAR.x])).toBeLessThan(0.05);
    expect(Math.abs(wa[FAR_STRIDE + FAR.x] - ts[FAR_STRIDE + FAR.x])).toBeLessThan(0.05);
  });

  it('does not disc-push a wired pair whose rest sits inside the discs', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 15);
    particle(data, 1, 20, 0, 15);
    const wires = packWires([0, 1, 20]);
    expect(native.stepFar(data, 2, wires, 1, 1 / 60, FAR_SUBSTEPS)).toBe(true);
    const dist = Math.abs(data[FAR.x] - data[FAR_STRIDE + FAR.x]);
    expect(dist).toBeLessThan(24);
    expect(dist).toBeGreaterThan(16);
  });

  it('holds a chord near rest length', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 4);
    particle(data, 1, 80, 0, 4);
    const wires = packWires([0, 1, 40]);
    native.stepFar(data, 2, wires, 1, 1 / 60, FAR_SUBSTEPS);
    const dist = Math.hypot(
      data[FAR_STRIDE + FAR.x] - data[FAR.x],
      data[FAR_STRIDE + FAR.y] - data[FAR.y],
    );
    expect(dist).toBeLessThan(50);
    expect(dist).toBeGreaterThan(30);
  });

  it('matches TS FAR on two stem-offset chords of one pair', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const ts = new Float32Array(2 * FAR_STRIDE);
    const wa = new Float32Array(2 * FAR_STRIDE);
    particle(ts, 0, 0, 0, 18);
    particle(ts, 1, 80, 0, 18);
    wa.set(ts);
    const wires = packWires(
      [0, 1, 46, 17, 0, -17, 0],
      [0, 1, Math.hypot(80, 40), 0, 20, 0, -20],
    );
    stepFarKernel(ts, 2, wires, 2, 1 / 60, FAR_SUBSTEPS);
    expect(native.stepFar(wa, 2, wires, 2, 1 / 60, FAR_SUBSTEPS)).toBe(true);
    for (const field of [FAR.x, FAR.y, FAR.vx, FAR.vy] as const) {
      expect(Math.abs(wa[field] - ts[field])).toBeLessThan(0.05);
      expect(Math.abs(wa[FAR_STRIDE + field] - ts[FAR_STRIDE + field])).toBeLessThan(0.05);
    }
  });

  it('does not move a locked body', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 2, 0, 10);
    data[FAR.locked] = 1;
    data[FAR.invMass] = 0;
    native.stepFar(data, 2, new Float32Array(FAR_WIRE_STRIDE), 0, 1 / 60, FAR_SUBSTEPS);
    expect(data[FAR.x]).toBe(0);
    expect(data[FAR_STRIDE + FAR.x]).toBeGreaterThan(2);
  });

  it('matches JS scent diffusion on a small stamp', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    /*
     * A small grid on purpose. The solver's diffuse walks every cell and needs
     * the field copied both ways to do it, so it declines the world grid — a
     * million cells is 16 MB across and back for arithmetic the TS twin does
     * over the live box alone. What is under test is the stencil agreeing, and
     * that is the same stencil at any size.
     */
    const js = new Fields(128);
    js.deposit(0, 40, 40, 8);
    const wa = new Fields(128);
    wa.data.set(js.data);
    js.diffuse(0.28);
    expect(native.scentDiffuse(wa, 0.28), 'the solver declined this grid').toBe(true);
    let maxDiff = 0;
    for (let i = 0; i < js.data.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(js.data[i] - wa.data[i]));
    }
    expect(maxDiff).toBeLessThan(1e-5);
  });

  it('matches one JS XPBD wire iteration', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const a = createAgent(1, 'era', 60, 100, 0, params);
    const b = createAgent(2, 'era', 300, 100, Math.PI, params);
    const nodes: ChainNode[] = [
      { x: 120, y: 100, vx: 0, vy: 0, prevX: 120, prevY: 100, integVx: 0, integVy: 0 },
      { x: 180, y: 130, vx: 0, vy: 0, prevX: 180, prevY: 130, integVx: 0, integVy: 0 },
      { x: 240, y: 100, vx: 0, vy: 0, prevX: 240, prevY: 100, integVx: 0, integVy: 0 },
    ];
    const aTs = cloneAgent(a);
    const bTs = cloneAgent(b);
    const nTs = cloneNodes(nodes);
    const stiff = { scale: 1, slack: 1 };
    const h = 1 / 60 / 8;
    solveWire(aTs, 'p', bTs, 'p', nTs, 120, 120, [], stiff, h);

    packAgent(native, 0, a, true);
    packAgent(native, 1, b, true);
    const W = native.wiresNear!;
    W[WN.a] = 0;
    W[WN.b] = 1;
    W[WN.rest] = 120;
    W[WN.rope] = 120;
    W[WN.scale] = 1;
    W[WN.slack] = 1;
    W[WN.aSlot] = 0;
    W[WN.bSlot] = 0;
    W[WN.node0] = 0;
    W[WN.nNodes] = 3;
    W[WN.flags] = WF_FULL;
    const N = native.nodes!;
    for (let i = 0; i < 3; i++) {
      const o = i * NODE_STRIDE;
      N[o + ND.x] = nodes[i].x;
      N[o + ND.y] = nodes[i].y;
    }
    native.nearWires(2, 1, h);
    unpackAgent(native, 0, a);
    unpackAgent(native, 1, b);
    expect(Math.abs(a.x - aTs.x)).toBeLessThan(2e-4);
    expect(Math.abs(b.x - bTs.x)).toBeLessThan(2e-4);
    expect(Math.abs(wrapAngle(a.heading - aTs.heading))).toBeLessThan(2e-4);
    expect(Math.abs(wrapAngle(b.heading - bTs.heading))).toBeLessThan(2e-4);
    for (let i = 0; i < 3; i++) {
      const o = i * NODE_STRIDE;
      expect(Math.abs(N[o + ND.x] - nTs[i].x)).toBeLessThan(2e-4);
      expect(Math.abs(N[o + ND.y] - nTs[i].y)).toBeLessThan(2e-4);
    }
  });

  it('matches one JS span iteration with port torque', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const a = createAgent(1, 'con', 0, 0, 0.4, params);
    const b = createAgent(2, 'con', 80, 10, -0.3, params);
    const aTs = cloneAgent(a);
    const bTs = cloneAgent(b);
    const h = 1 / 60 / 8;
    solveWireSpan(aTs, 'l', bTs, 'r', 40, { scale: 1, slack: 1 }, h);

    packAgent(native, 0, a, true);
    packAgent(native, 1, b, true);
    const W = native.wiresNear!;
    W[WN.a] = 0;
    W[WN.b] = 1;
    W[WN.rest] = 40;
    W[WN.rope] = 40;
    W[WN.scale] = 1;
    W[WN.slack] = 1;
    W[WN.aSlot] = 1;
    W[WN.bSlot] = 2;
    W[WN.node0] = 0;
    W[WN.nNodes] = 0;
    W[WN.flags] = 0;
    native.nearWires(2, 1, h);
    unpackAgent(native, 0, a);
    unpackAgent(native, 1, b);
    expect(Math.abs(a.x - aTs.x)).toBeLessThan(2e-4);
    expect(Math.abs(a.y - aTs.y)).toBeLessThan(2e-4);
    expect(Math.abs(b.x - bTs.x)).toBeLessThan(2e-4);
    expect(Math.abs(a.heading - aTs.heading)).toBeLessThan(2e-4);
    expect(Math.abs(b.heading - bTs.heading)).toBeLessThan(2e-4);
  });

  it('disc-solves FAR-FAR and leaves detailed bodies to SAT', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const data = native.bodies!;
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 2, 0, 10);
    particle(data, 2, 100, 0, 10);
    particle(data, 3, 102, 0, 10);
    native.detailed![0] = 1;
    native.detailed![1] = 0;
    native.detailed![2] = 0;
    native.detailed![3] = 0;
    native.invInertia!.fill(0, 0, 4);
    native.scale!.fill(1, 0, 4);
    native.kind!.fill(KIND_ERA, 0, 4);
    const h = 1 / 60 / 8;
    const np = native.nearDisc(4, h);
    expect(np).toBeGreaterThan(0);
    // Detailed 0 overlapping FAR 1: C must not push 0.
    expect(data[FAR.x]).toBe(0);
    // FAR 2-3 overlap should separate.
    expect(data[2 * FAR_STRIDE + FAR.x]).toBeLessThan(100);
    expect(data[3 * FAR_STRIDE + FAR.x]).toBeGreaterThan(102);
  });

  it('matches one JS SAT contact on overlapping triangles', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const a = createAgent(1, 'con', 0, 0, 0, params);
    const b = createAgent(2, 'con', 2, 0, Math.PI, params);
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
    const H = native.hits!;
    expect(Math.abs(H[HIT.overlap] - hit!.overlap)).toBeLessThan(2e-3);
    expect(Math.abs(H[HIT.nx] - hit!.nx)).toBeLessThan(2e-3);
    expect(Math.abs(H[HIT.ny] - hit!.ny)).toBeLessThan(2e-3);
    expect(Math.abs(H[HIT.px] - hit!.px)).toBeLessThan(0.05);
    expect(Math.abs(H[HIT.py] - hit!.py)).toBeLessThan(0.05);
    expect(Math.abs(a.x - aTs.x)).toBeLessThan(1e-3);
    expect(Math.abs(a.y - aTs.y)).toBeLessThan(1e-3);
    expect(Math.abs(b.x - bTs.x)).toBeLessThan(1e-3);
    expect(Math.abs(wrapAngle(a.heading - aTs.heading))).toBeLessThan(1e-3);
    expect(Math.abs(wrapAngle(b.heading - bTs.heading))).toBeLessThan(1e-3);
  });

  it('snapshots Hertzian kinematics at the SAT hit, not after the impulse', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const a = createAgent(1, 'era', 0, 0, 0, params);
    const b = createAgent(2, 'era', 4, 0, Math.PI, params);
    a.vx = 40;
    b.vx = -25;
    a.omega = 1.2;
    b.omega = -0.8;
    packAgent(native, 0, a, true);
    packAgent(native, 1, b, true);
    const hit = queryHit(a, b, 240, 160);
    expect(hit).not.toBeNull();
    const m = contactMechanics(a, b, hit!);
    native.nearContacts(2, 0, 1 / 60 / 8);
    expect(native.hitCount()).toBeGreaterThan(0);
    const H = native.hits!;
    expect(Math.abs(H[HIT.vN] - m.vN)).toBeLessThan(2e-3);
    expect(Math.abs(H[HIT.vT] - m.vT)).toBeLessThan(2e-3);
    expect(Math.abs(H[HIT.effMass] - m.effMass)).toBeLessThan(2e-3);
  });

  it('SAT-separates overlapping eras in one full NEAR step', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const params = defaultParams();
    const a = createAgent(1, 'era', 0, 0, 0, params);
    const b = createAgent(2, 'era', 2, 0, Math.PI, params);
    packAgent(native, 0, a, true);
    packAgent(native, 1, b, true);
    expect(native.stepNear(2, 0, 1 / 60, 8, 1, -1, 160, 0, 0)).toBe(true);
    unpackAgent(native, 0, a);
    unpackAgent(native, 1, b);
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(12);
  });
});
