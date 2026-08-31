import { describe, expect, it } from 'vitest';
import {
  FAR,
  FAR_STRIDE,
  FAR_SUBSTEPS,
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

describe('FAR kernel', () => {
  it('pushes overlapping discs apart', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 2, 0, 10);
    stepFarKernel(data, 2, new Float32Array(4), 0, 1 / 60, FAR_SUBSTEPS);
    expect(Math.abs(data[FAR.x] - data[FAR_STRIDE + FAR.x])).toBeGreaterThan(12);
  });

  it('holds a chord near its rest length', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 4);
    particle(data, 1, 80, 0, 4);
    const wires = new Float32Array([0, 1, 40, 0]);
    stepFarKernel(data, 2, wires, 1, 1 / 60, FAR_SUBSTEPS);
    const dist = Math.hypot(
      data[FAR_STRIDE + FAR.x] - data[FAR.x],
      data[FAR_STRIDE + FAR.y] - data[FAR.y],
    );
    expect(dist).toBeLessThan(50);
    expect(dist).toBeGreaterThan(30);
  });

  it('does not move a locked body', () => {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 2, 0, 10);
    data[FAR.locked] = 1;
    data[FAR.invMass] = 0;
    stepFarKernel(data, 2, new Float32Array(4), 0, 1 / 60, FAR_SUBSTEPS);
    expect(data[FAR.x]).toBe(0);
    expect(data[FAR_STRIDE + FAR.x]).toBeGreaterThan(2);
  });
});
