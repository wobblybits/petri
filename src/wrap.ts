export interface Vec2 {
  x: number;
  y: number;
}

/** Positions live on an infinite plane; kept as a no-op for call-site compatibility. */
export function wrap(x: number, _size?: number): number {
  return x;
}

export function wrapDelta(a: number, b: number, _size?: number): number {
  return b - a;
}

export function wrapDeltaVec(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  _w?: number,
  _h?: number,
): Vec2 {
  return { x: bx - ax, y: by - ay };
}

/**
 * `wrapDeltaVec` writing into a caller's vector instead of minting one.
 *
 * The same trade `stemWorldInto` and `bezierPointInto` already make. A delta
 * inside a per-frame loop over everything is not somewhere to be allocating:
 * `advanceRewrite` alone was churning thousands of these a frame.
 */
export function wrapDeltaVecInto(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  _w: number,
  _h: number,
  out: Vec2,
): Vec2 {
  out.x = bx - ax;
  out.y = by - ay;
  return out;
}

export function wrapDist(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  w?: number,
  h?: number,
): number {
  const d = wrapDeltaVec(ax, ay, bx, by, w ?? 0, h ?? 0);
  return Math.hypot(d.x, d.y);
}

export function wrapMid(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  _w?: number,
  _h?: number,
): Vec2 {
  return { x: (ax + bx) * 0.5, y: (ay + by) * 0.5 };
}

const TAU = Math.PI * 2;

/**
 * Normalize an angle to [-pi, pi).
 *
 * This was atan2(sin a, cos a) — three transcendentals to do arithmetic, and
 * 3.8% of a step at soup scale. Angles here are nearly always already in
 * range (a heading plus one substep of rotation), so the common case is a
 * pair of comparisons and no work at all.
 */
export function wrapAngle(a: number): number {
  if (a >= -Math.PI && a < Math.PI) return a;
  if (!Number.isFinite(a)) return 0;
  let r = a % TAU;
  if (r >= Math.PI) r -= TAU;
  else if (r < -Math.PI) r += TAU;
  return r;
}

export function angleDelta(from: number, to: number): number {
  return wrapAngle(to - from);
}

/** Smallest turn that makes two headings parallel (same or opposite). */
export function nematicDelta(from: number, to: number): number {
  let e = angleDelta(from, to);
  if (e > Math.PI * 0.5) e -= Math.PI;
  if (e < -Math.PI * 0.5) e += Math.PI;
  return e;
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function easeInOut(t: number): number {
  return t * t * (3 - 2 * t);
}

export function rotate(x: number, y: number, heading: number): Vec2 {
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  return { x: x * c - y * s, y: x * s + y * c };
}
