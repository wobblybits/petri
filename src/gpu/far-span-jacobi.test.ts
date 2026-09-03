import { describe, expect, it } from 'vitest';
import {
  FAR,
  FAR_SPAN_COMP,
  FAR_STRIDE,
  FAR_SUBSTEPS,
  FAR_WIRE_STRIDE,
  FW,
  farApply,
  farDisc,
  farFinalize,
  farIntegrate,
  farSpan,
} from './far-kernel.ts';

/**
 * The span constraint as the GPU actually solves it.
 *
 * `far-kernel.ts` is the reference and solves the span constraints in
 * sequence, so the second wire on a body already sees what the first one did.
 * A compute shader cannot do that: every body is solved in parallel against
 * the same start pose and the per-wire corrections are summed, which is Jacobi
 * where the reference is Gauss-Seidel.
 *
 * That difference is the obvious suspect for a solver that blows up, and it
 * was the first thing I blamed when `solveFarGpu` ran for the first time and
 * sent wires to infinite length. It is not the cause — measured below, summing
 * the corrections settles a stretched mesh onto its rest length just as the
 * sequential version does, at this compliance and substep count. The real
 * fault was in `packFar`, which borrowed a scratch array that had since become
 * a cache. Under-relaxing the shader would have been a plausible-looking fix
 * for a bug that was somewhere else entirely.
 *
 * There is no WebGPU under Node, so far.wgsl cannot be executed here. Its
 * arithmetic can be: `spanJacobi` mirrors the shader line for line, and these
 * pin it against the reference so a future divergence is caught on a machine
 * that cannot run it.
 */

/** Mirrors the `span` entry point in far.wgsl. */
function spanJacobi(
  data: Float32Array,
  n: number,
  h: number,
  wires: Float32Array,
  nWires: number,
  delta: Float32Array,
  underRelax: boolean,
): void {
  const invH2 = 1 / Math.max(1e-12, h * h);
  delta.fill(0, 0, n * 2);
  for (let i = 0; i < n; i++) {
    const oi = i * FAR_STRIDE;
    if (data[oi + FAR.locked] >= 0.5 || data[oi + FAR.invMass] <= 0) continue;
    let px = 0;
    let py = 0;
    let touching = 0;
    for (let w = 0; w < nWires; w++) {
      const o = w * FAR_WIRE_STRIDE;
      const ia = wires[o + FW.a] | 0;
      const ib = wires[o + FW.b] | 0;
      let j: number;
      let oix: number;
      let oiy: number;
      let ojx: number;
      let ojy: number;
      if (ia === i) {
        j = ib;
        oix = wires[o + FW.oax];
        oiy = wires[o + FW.oay];
        ojx = wires[o + FW.obx];
        ojy = wires[o + FW.oby];
      } else if (ib === i) {
        j = ia;
        oix = wires[o + FW.obx];
        oiy = wires[o + FW.oby];
        ojx = wires[o + FW.oax];
        ojy = wires[o + FW.oay];
      } else {
        continue;
      }
      const oj = j * FAR_STRIDE;
      let dx = data[oj + FAR.x] + ojx - (data[oi + FAR.x] + oix);
      let dy = data[oj + FAR.y] + ojy - (data[oi + FAR.y] + oiy);
      let dist = Math.hypot(dx, dy);
      if (!Number.isFinite(dist)) continue;
      if (dist < 1e-6) {
        dx = i < j ? 1 : -1;
        dy = 0;
        dist = 1;
      }
      const nx = dx / dist;
      const ny = dy / dist;
      const C = dist - wires[o + FW.rest];
      const soft = wires[o + FW.soft] > 0 ? wires[o + FW.soft] : 1;
      const alpha = FAR_SPAN_COMP * soft * invH2;
      const denom = data[oi + FAR.invMass] + data[oj + FAR.invMass] + alpha;
      if (denom < 1e-12) continue;
      const lam = -C / denom;
      px += -nx * (lam * data[oi + FAR.invMass]);
      py += -ny * (lam * data[oi + FAR.invMass]);
      touching++;
    }
    if (underRelax && touching > 1) {
      px /= touching;
      py /= touching;
    }
    delta[i * 2] = px;
    delta[i * 2 + 1] = py;
  }
}

/** A body with three wires is the case that diverges; build a small mesh. */
function mesh(): { data: Float32Array; wires: Float32Array; n: number; nWires: number } {
  const cols = 4;
  const rows = 4;
  const n = cols * rows;
  const rest = 40;
  const data = new Float32Array(n * FAR_STRIDE);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const o = (j * cols + i) * FAR_STRIDE;
      // Stretched off rest, so the constraints actually have work to do. A
      // mesh started exactly at rest length has C = 0 on every wire and
      // exercises nothing at all.
      data[o + FAR.x] = 100 + i * rest * 1.35;
      data[o + FAR.y] = 100 + j * rest * 1.35;
      data[o + FAR.invMass] = 1;
      data[o + FAR.radius] = 10.34;
    }
  }
  const list: number[] = [];
  const push = (a: number, b: number): void => {
    list.push(a, b, rest, 1, 0, 0, 0, 0);
  };
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      if (i + 1 < cols) push(k, k + 1);
      if (j + 1 < rows) push(k, k + cols);
    }
  }
  return { data, wires: new Float32Array(list), n, nWires: list.length / FAR_WIRE_STRIDE };
}

/** The GPU dispatch order from far-gpu.ts, with the span swapped for the mirror. */
function runGpuOrder(frames: number, underRelax: boolean): number {
  const { data, wires, n, nWires } = mesh();
  const delta = new Float32Array(n * 2);
  const h = 1 / 60 / FAR_SUBSTEPS;
  for (let f = 0; f < frames; f++) {
    for (let s = 0; s < FAR_SUBSTEPS; s++) {
      farIntegrate(data, n, h);
      farDisc(data, n, h, delta, wires, nWires);
      farApply(data, n, delta);
      spanJacobi(data, n, h, wires, nWires, delta, underRelax);
      farApply(data, n, delta);
      farFinalize(data, n, h);
    }
  }
  return worstWire(data, wires, nWires);
}

function runReference(frames: number): number {
  const { data, wires, n, nWires } = mesh();
  const delta = new Float32Array(n * 2);
  const h = 1 / 60 / FAR_SUBSTEPS;
  for (let f = 0; f < frames; f++) {
    for (let s = 0; s < FAR_SUBSTEPS; s++) {
      farIntegrate(data, n, h);
      farDisc(data, n, h, delta, wires, nWires);
      farApply(data, n, delta);
      farSpan(data, n, h, wires, nWires, delta);
      farFinalize(data, n, h);
    }
  }
  return worstWire(data, wires, nWires);
}

function worstWire(data: Float32Array, wires: Float32Array, nWires: number): number {
  let worst = 0;
  for (let w = 0; w < nWires; w++) {
    const o = w * FAR_WIRE_STRIDE;
    const a = (wires[o + FW.a] | 0) * FAR_STRIDE;
    const b = (wires[o + FW.b] | 0) * FAR_STRIDE;
    const d = Math.hypot(data[b + FAR.x] - data[a + FAR.x], data[b + FAR.y] - data[a + FAR.y]);
    if (!Number.isFinite(d)) return Infinity;
    worst = Math.max(worst, d);
  }
  return worst;
}

describe('GPU span solve', () => {
  it('settles a stretched mesh onto its rest length', () => {
    // Summed corrections, exactly as the shader does it. Started 35% long.
    const worst = runGpuOrder(60, false);
    expect(Number.isFinite(worst), 'wires went non-finite').toBe(true);
    expect(worst, `worst wire ${worst.toFixed(2)} against a rest of 40`).toBeLessThan(45);
  });

  it('lands where the sequential reference lands', () => {
    // The guard that matters: parallel and sequential have to agree, or the
    // GPU is running different physics from every other path in the sim.
    const gpu = runGpuOrder(60, false);
    const cpu = runReference(60);
    expect(Math.abs(gpu - cpu), `gpu ${gpu.toFixed(3)} vs cpu ${cpu.toFixed(3)}`)
      .toBeLessThan(1);
  });

  it('would notice if summing the corrections ever did diverge', () => {
    // Averaging is the standard fix if it ever does; pinned so the comparison
    // above is known to be measuring something.
    const summed = runGpuOrder(60, false);
    const averaged = runGpuOrder(60, true);
    expect(Number.isFinite(summed) && Number.isFinite(averaged)).toBe(true);
    expect(Math.abs(summed - averaged), 'the two relaxations disagree wildly')
      .toBeLessThan(6);
  });
});
