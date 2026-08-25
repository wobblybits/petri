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

export function wrapAngle(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
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
