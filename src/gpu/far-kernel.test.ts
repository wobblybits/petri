import { describe, expect, it } from 'vitest';
import {
  FAR,
  FAR_STRIDE,
  FAR_SUBSTEPS,
  FAR_WIRE_STRIDE,
  packFarWire,
  stepFarKernel,
} from './far-kernel.ts';

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

function wiresOf(...rows: number[][]): Float32Array {
  const wires = new Float32Array(rows.length * FAR_WIRE_STRIDE);
  for (let k = 0; k < rows.length; k++) {
    const [a, b, rest, oax = 0, oay = 0, obx = 0, oby = 0] = rows[k];
    packFarWire(wires, k, a, b, rest, oax, oay, obx, oby);
  }
  return wires;
}

describe('FAR kernel', () => {
  it('pushes overlapping discs apart', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 2, 0, 10);
    stepFarKernel(data, 2, new Float32Array(FAR_WIRE_STRIDE), 0, 1 / 60, FAR_SUBSTEPS);
    expect(Math.abs(data[FAR.x] - data[FAR_STRIDE + FAR.x])).toBeGreaterThan(12);
  });

  it('does not disc-push a wired pair whose rest sits inside the discs', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 15);
    particle(data, 1, 20, 0, 15);
    stepFarKernel(data, 2, wiresOf([0, 1, 20]), 1, 1 / 60, FAR_SUBSTEPS);
    const dist = Math.abs(data[FAR.x] - data[FAR_STRIDE + FAR.x]);
    expect(dist).toBeLessThan(24);
    expect(dist).toBeGreaterThan(16);
  });

  it('unsticks a coincident wired pair', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 0, 0, 10);
    stepFarKernel(data, 2, wiresOf([0, 1, 20]), 1, 1 / 60, FAR_SUBSTEPS);
    const dist = Math.hypot(
      data[FAR_STRIDE + FAR.x] - data[FAR.x],
      data[FAR_STRIDE + FAR.y] - data[FAR.y],
    );
    expect(dist).toBeGreaterThan(8);
    expect(Number.isFinite(data[FAR.vx])).toBe(true);
    expect(Math.hypot(data[FAR.vx], data[FAR.vy])).toBeLessThan(2000);
  });

  it('holds a chord near its rest length', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 4);
    particle(data, 1, 80, 0, 4);
    stepFarKernel(data, 2, wiresOf([0, 1, 40]), 1, 1 / 60, FAR_SUBSTEPS);
    const dist = Math.hypot(
      data[FAR_STRIDE + FAR.x] - data[FAR.x],
      data[FAR_STRIDE + FAR.y] - data[FAR.y],
    );
    expect(dist).toBeLessThan(50);
    expect(dist).toBeGreaterThan(30);
  });

  it('holds two stem spans on the same pair at different rests', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 18);
    particle(data, 1, 80, 0, 18);
    const wires = wiresOf(
      [0, 1, 46, 17, 0, -17, 0],
      [0, 1, Math.hypot(80, 40), 0, 20, 0, -20],
    );
    stepFarKernel(data, 2, wires, 2, 1 / 60, FAR_SUBSTEPS);
    const ax = data[FAR.x];
    const ay = data[FAR.y];
    const bx = data[FAR_STRIDE + FAR.x];
    const by = data[FAR_STRIDE + FAR.y];
    const principal = Math.hypot(bx - 17 - (ax + 17), by - ay);
    const aux = Math.hypot(bx - (ax), by - 20 - (ay + 20));
    expect(Math.abs(principal - 46)).toBeLessThan(2);
    expect(Math.abs(aux - Math.hypot(80, 40))).toBeLessThan(2);
    expect(Math.hypot(data[FAR.vx], data[FAR.vy])).toBeLessThan(200);
    expect(Math.hypot(data[FAR_STRIDE + FAR.vx], data[FAR_STRIDE + FAR.vy])).toBeLessThan(200);
  });

  it('does not move a locked body', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 2, 0, 10);
    data[FAR.locked] = 1;
    data[FAR.invMass] = 0;
    stepFarKernel(data, 2, new Float32Array(FAR_WIRE_STRIDE), 0, 1 / 60, FAR_SUBSTEPS);
    expect(data[FAR.x]).toBe(0);
    expect(data[FAR_STRIDE + FAR.x]).toBeGreaterThan(2);
  });
});
