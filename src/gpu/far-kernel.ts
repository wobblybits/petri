/**
 * CPU twin of far.wgsl disc, with Gauss–Seidel stem-span (the shader stays
 * Jacobi: one thread per body). Translation-only: FAR bodies do not pick up
 * contact torque.
 *
 * The tier has to agree with NEAR, or crossing the LOD line moves the net.
 * Two things buy that: the packed radius is the glyph-area disc rather than
 * the SAT bound (see `discRadius`), and span compliance is the same per-wire
 * softness NEAR uses rather than a fixed constant. Wired pairs still skip
 * disc — span owns that gap and fighting it costs a substep for nothing.
 * A coincident pair is the exception: span has no normal at dist 0, so
 * skipping it too would freeze a pile. The unstick axis is unit-length so
 * the impulse is not 1/1e-6.
 */
export const FAR_SLOP = 0.35;
export const FAR_CONTACT_COMP = 4.0e-6;
export const FAR_SPAN_COMP = 3.0e-6;
export const FAR_STRIDE = 12;
export const FAR_WIRE_STRIDE = 8;
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

export const FW = {
  a: 0,
  b: 1,
  rest: 2,
  /** Per-wire softness: params.springK scale times the birth-slack ramp. */
  soft: 3,
  oax: 4,
  oay: 5,
  obx: 6,
  oby: 7,
} as const;

const PI = Math.PI;
const TAU = Math.PI * 2;
const EMPTY_WIRES = new Float32Array(0);
let discNei = new Int32Array(0);
let deltaScratch = new Float32Array(0);

function wrapAngle(a: number): number {
  if (a >= -PI && a < PI) return a;
  if (!Number.isFinite(a)) return 0;
  let r = a % TAU;
  if (r >= PI) r -= TAU;
  if (r < -PI) r += TAU;
  return r;
}

export function packFarWire(
  wires: Float32Array,
  k: number,
  a: number,
  b: number,
  rest: number,
  oax = 0,
  oay = 0,
  obx = 0,
  oby = 0,
  soft = 1,
): void {
  const o = k * FAR_WIRE_STRIDE;
  wires[o + FW.a] = a;
  wires[o + FW.b] = b;
  wires[o + FW.rest] = rest;
  wires[o + FW.soft] = soft > 0 && Number.isFinite(soft) ? soft : 1;
  wires[o + FW.oax] = oax;
  wires[o + FW.oay] = oay;
  wires[o + FW.obx] = obx;
  wires[o + FW.oby] = oby;
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

/**
 * Wired-neighbour table, three slots per body — an agent has three ports, so
 * that is the whole of it. Built once per pass: scanning the wire list inside
 * the body loop made the cheap tier O(bodies x wires).
 */
function fillDiscNeighbours(n: number, wires: Float32Array, nWires: number): Int32Array {
  if (discNei.length < n * 3) discNei = new Int32Array(Math.max(48, n * 6));
  discNei.fill(-1, 0, n * 3);
  for (let w = 0; w < nWires; w++) {
    const o = w * FAR_WIRE_STRIDE;
    const a = wires[o + FW.a] | 0;
    const b = wires[o + FW.b] | 0;
    if (a < 0 || b < 0 || a >= n || b >= n || a === b) continue;
    for (let k = 0; k < 3; k++) {
      if (discNei[a * 3 + k] < 0) {
        discNei[a * 3 + k] = b;
        break;
      }
    }
    for (let k = 0; k < 3; k++) {
      if (discNei[b * 3 + k] < 0) {
        discNei[b * 3 + k] = a;
        break;
      }
    }
  }
  return discNei;
}

export function farDisc(
  data: Float32Array,
  n: number,
  h: number,
  delta: Float32Array,
  wires: Float32Array = EMPTY_WIRES,
  nWires = 0,
): void {
  const alpha = FAR_CONTACT_COMP / Math.max(1e-12, h * h);
  delta.fill(0, 0, n * 2);
  const nei = fillDiscNeighbours(n, wires, nWires);
  for (let i = 0; i < n; i++) {
    const oi = i * FAR_STRIDE;
    if (data[oi + FAR.locked] >= 0.5 || data[oi + FAR.invMass] <= 0) continue;
    const ix = data[oi + FAR.x];
    const iy = data[oi + FAR.y];
    const wA = data[oi + FAR.invMass];
    const ri = data[oi + FAR.radius];
    // Span already owns the gap on a wire; colliding the pair as well just
    // fights the chord.
    const n0 = nei[i * 3];
    const n1 = nei[i * 3 + 1];
    const n2 = nei[i * 3 + 2];
    let px = 0;
    let py = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const oj = j * FAR_STRIDE;
      let dx = data[oj + FAR.x] - ix;
      let dy = data[oj + FAR.y] - iy;
      let dist = Math.hypot(dx, dy);
      const keep = ri + data[oj + FAR.radius];
      if (dist >= keep) continue;
      const wired = j === n0 || j === n1 || j === n2;
      // Coincident: span has no normal, so a sanitize pile would stay a pile
      // if we skipped these the way we skip a healthy wired gap. Index order
      // so both bodies do not pick the same world axis and translate together.
      if (dist < 1e-6) {
        dx = i < j ? 1 : -1;
        dy = 0;
        dist = 1;
      } else if (wired) {
        continue;
      }
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
  _delta: Float32Array,
): void {
  const invH2 = 1 / Math.max(1e-12, h * h);
  for (let w = 0; w < nWires; w++) {
    const o = w * FAR_WIRE_STRIDE;
    const soft = wires[o + FW.soft] > 0 ? wires[o + FW.soft] : 1;
    const alpha = FAR_SPAN_COMP * soft * invH2;
    const i = wires[o + FW.a] | 0;
    const j = wires[o + FW.b] | 0;
    const rest = wires[o + FW.rest];
    if (!Number.isFinite(rest) || rest < 0) continue;
    if (i < 0 || j < 0 || i >= n || j >= n || i === j) continue;
    const oi = i * FAR_STRIDE;
    const oj = j * FAR_STRIDE;
    const oax = wires[o + FW.oax];
    const oay = wires[o + FW.oay];
    const obx = wires[o + FW.obx];
    const oby = wires[o + FW.oby];
    let dx = data[oj + FAR.x] + obx - (data[oi + FAR.x] + oax);
    let dy = data[oj + FAR.y] + oby - (data[oi + FAR.y] + oay);
    let dist = Math.hypot(dx, dy);
    if (!Number.isFinite(dist)) continue;
    if (dist < 1e-6) {
      dx = i < j ? 1 : -1;
      dy = 0;
      dist = 1;
    }
    const C = dist - rest;
    const wA = data[oi + FAR.invMass];
    const wB = data[oj + FAR.invMass];
    const denom = wA + wB + alpha;
    if (denom < 1e-12) continue;
    const lam = -C / denom;
    const s = lam / dist;
    if (data[oi + FAR.locked] < 0.5 && wA > 0) {
      data[oi + FAR.x] -= dx * s * wA;
      data[oi + FAR.y] -= dy * s * wA;
    }
    if (data[oj + FAR.locked] < 0.5 && wB > 0) {
      data[oj + FAR.x] += dx * s * wB;
      data[oj + FAR.y] += dy * s * wB;
    }
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

/** One frame of FAR physics. `wires` is `[a, b, rest, pad, oax, oay, obx, oby] * nWires`. */
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
  if (deltaScratch.length < n * 2) deltaScratch = new Float32Array(Math.max(32, n * 4));
  const delta = deltaScratch;
  for (let s = 0; s < substeps; s++) {
    farIntegrate(data, n, h);
    farDisc(data, n, h, delta, wires, nWires);
    farApply(data, n, delta);
    if (nWires > 0) farSpan(data, n, h, wires, nWires, delta);
    farFinalize(data, n, h);
  }
}
