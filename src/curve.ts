import type { Vec2 } from './wrap.ts';

export interface Cubic {
  p0: Vec2;
  p1: Vec2;
  p2: Vec2;
  p3: Vec2;
}

export function bezierPointInto(
  p0: Vec2,
  p1: Vec2,
  p2: Vec2,
  p3: Vec2,
  t: number,
  out: Vec2,
): Vec2 {
  const u = 1 - t;
  const uu = u * u;
  const tt = t * t;
  out.x = uu * u * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + tt * t * p3.x;
  out.y = uu * u * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + tt * t * p3.y;
  return out;
}

export function bezierPoint(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: number): Vec2 {
  return bezierPointInto(p0, p1, p2, p3, t, { x: 0, y: 0 });
}

export function bezierLength(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, steps = 24): number {
  let len = 0;
  let prev = p0;
  for (let i = 1; i <= steps; i++) {
    const p = bezierPoint(p0, p1, p2, p3, i / steps);
    len += Math.hypot(p.x - prev.x, p.y - prev.y);
    prev = p;
  }
  return len;
}
