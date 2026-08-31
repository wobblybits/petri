/**
 * CPU twin of far.wgsl. Same Jacobi disc + chord span, so the shader can be
 * tested without a GPU. Translation-only: FAR bodies do not pick up contact
 * torque.
 */
export const FAR_SLOP = 0.35;
export const FAR_CONTACT_COMP = 4.0e-6;
export const FAR_SPAN_COMP = 3.0e-6;
export const FAR_STRIDE = 12;
export const FAR_SUBSTEPS = 8;

export const FAR = {
  x: 0,
  y: 1,
  vx: 2,
  vy: 3,
  heading: 4,
  omega: 5,
  invMass: 6,
  radius: 7,
  locked: 8,
  prevX: 9,
  prevY: 10,
  prevHeading: 11,
} as const;

const PI = Math.PI;
const TAU = Math.PI * 2;

function wrapAngle(a: number): number {
  if (a >= -PI && a < PI) return a;
  let r = a % TAU;
  if (r >= PI) r -= TAU;
  if (r < -PI) r += TAU;
  return r;
}

export function farIntegrate(data: Float32Array, n: number, h: number): void {
  for (let i = 0; i < n; i++) {
    const o = i * FAR_STRIDE;
    data[o + FAR.prevX] = data[o + FAR.x];
    data[o + FAR.prevY] = data[o + FAR.y];
    data[o + FAR.prevHeading] = data[o + FAR.heading];
    if (data[o + FAR.locked] >= 0.5) continue;
    data[o + FAR.x] += data[o + FAR.vx] * h;
    data[o + FAR.y] += data[o + FAR.vy] * h;
    data[o + FAR.heading] = wrapAngle(data[o + FAR.heading] + data[o + FAR.omega] * h);
  }
}

export function farDisc(data: Float32Array, n: number, h: number, delta: Float32Array): void {
  const alpha = FAR_CONTACT_COMP / Math.max(1e-12, h * h);
  delta.fill(0, 0, n * 2);
  for (let i = 0; i < n; i++) {
    const oi = i * FAR_STRIDE;
    if (data[oi + FAR.locked] >= 0.5 || data[oi + FAR.invMass] <= 0) continue;
    const ix = data[oi + FAR.x];
    const iy = data[oi + FAR.y];
    const wA = data[oi + FAR.invMass];
    const ri = data[oi + FAR.radius];
    let px = 0;
    let py = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const oj = j * FAR_STRIDE;
      const dx = data[oj + FAR.x] - ix;
      const dy = data[oj + FAR.y] - iy;
      const dist = Math.hypot(dx, dy);
      const keep = ri + data[oj + FAR.radius];
      if (dist >= keep || dist < 1e-6) continue;
      const depth = keep - dist - FAR_SLOP;
      if (depth <= 0) continue;
      const denom = wA + data[oj + FAR.invMass] + alpha;
      if (denom < 1e-12) continue;
      const lam = depth / denom;
      const s = (lam * wA) / dist;
      px -= dx * s;
      py -= dy * s;
    }
    delta[i * 2] = px;
    delta[i * 2 + 1] = py;
  }
}

export function farSpan(
  data: Float32Array,
  n: number,
  h: number,
  wires: Float32Array,
  nWires: number,
  delta: Float32Array,
): void {
  const alpha = FAR_SPAN_COMP / Math.max(1e-12, h * h);
  delta.fill(0, 0, n * 2);
  for (let i = 0; i < n; i++) {
    const oi = i * FAR_STRIDE;
    if (data[oi + FAR.locked] >= 0.5 || data[oi + FAR.invMass] <= 0) continue;
    const ix = data[oi + FAR.x];
    const iy = data[oi + FAR.y];
    const wA = data[oi + FAR.invMass];
    let px = 0;
    let py = 0;
    for (let w = 0; w < nWires; w++) {
      const a = wires[w * 4] | 0;
      const b = wires[w * 4 + 1] | 0;
      const rest = wires[w * 4 + 2];
      let j = -1;
      if (a === i) j = b;
      else if (b === i) j = a;
      else continue;
      const oj = j * FAR_STRIDE;
      const dx = data[oj + FAR.x] - ix;
      const dy = data[oj + FAR.y] - iy;
      const dist = Math.hypot(dx, dy);
      if (dist < 1e-9) continue;
      const C = dist - rest;
      const denom = wA + data[oj + FAR.invMass] + alpha;
      if (denom < 1e-12) continue;
      const lam = -C / denom;
      const s = (lam * wA) / dist;
      px -= dx * s;
      py -= dy * s;
    }
    delta[i * 2] = px;
    delta[i * 2 + 1] = py;
  }
}

export function farApply(data: Float32Array, n: number, delta: Float32Array): void {
  for (let i = 0; i < n; i++) {
    const o = i * FAR_STRIDE;
    if (data[o + FAR.locked] >= 0.5) continue;
    data[o + FAR.x] += delta[i * 2];
    data[o + FAR.y] += delta[i * 2 + 1];
  }
}

export function farFinalize(data: Float32Array, n: number, h: number): void {
  const invH = 1 / h;
  for (let i = 0; i < n; i++) {
    const o = i * FAR_STRIDE;
    if (data[o + FAR.locked] >= 0.5) {
      data[o + FAR.vx] = 0;
      data[o + FAR.vy] = 0;
      data[o + FAR.omega] = 0;
      continue;
    }
    data[o + FAR.vx] = (data[o + FAR.x] - data[o + FAR.prevX]) * invH;
    data[o + FAR.vy] = (data[o + FAR.y] - data[o + FAR.prevY]) * invH;
    data[o + FAR.omega] = wrapAngle(data[o + FAR.heading] - data[o + FAR.prevHeading]) * invH;
  }
}

/** One frame of FAR physics. `wires` is `[a, b, rest, pad] * nWires`. */
export function stepFarKernel(
  data: Float32Array,
  n: number,
  wires: Float32Array,
  nWires: number,
  dt: number,
  substeps = FAR_SUBSTEPS,
): void {
  if (n <= 0 || dt <= 0) return;
  const h = dt / substeps;
  const delta = new Float32Array(n * 2);
  for (let s = 0; s < substeps; s++) {
    farIntegrate(data, n, h);
    farDisc(data, n, h, delta);
    farApply(data, n, delta);
    if (nWires > 0) {
      farSpan(data, n, h, wires, nWires, delta);
      farApply(data, n, delta);
    }
    farFinalize(data, n, h);
  }
}
