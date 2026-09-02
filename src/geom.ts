import type { Vec2 } from './wrap.ts';

export const WIRE_RADIUS = 2.5;
/** World px of string displacement at waveguide |sample| = 1 after AGC. */
export const WAVE_DISP_PX = 16;

export function orient(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
): number {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

/** Strict proper intersection; shared endpoints and colinear overlaps are false. */
export function segmentsIntersect(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
  dx: number,
  dy: number,
): boolean {
  const o1 = orient(ax, ay, bx, by, cx, cy);
  const o2 = orient(ax, ay, bx, by, dx, dy);
  const o3 = orient(cx, cy, dx, dy, ax, ay);
  const o4 = orient(cx, cy, dx, dy, bx, by);
  return o1 * o2 < 0 && o3 * o4 < 0;
}

export function closestOnSegments(
  p1x: number,
  p1y: number,
  p2x: number,
  p2y: number,
  q1x: number,
  q1y: number,
  q2x: number,
  q2y: number,
): { ax: number; ay: number; bx: number; by: number; t: number; u: number } {
  const dpx = p2x - p1x;
  const dpy = p2y - p1y;
  const dqx = q2x - q1x;
  const dqy = q2y - q1y;
  const rx = p1x - q1x;
  const ry = p1y - q1y;
  const a = dpx * dpx + dpy * dpy;
  const e = dqx * dqx + dqy * dqy;
  const f = dqx * rx + dqy * ry;
  let t: number;
  let u: number;

  if (a <= 1e-12 && e <= 1e-12) {
    t = 0;
    u = 0;
  } else if (a <= 1e-12) {
    t = 0;
    u = Math.max(0, Math.min(1, f / e));
  } else {
    const c = dpx * rx + dpy * ry;
    if (e <= 1e-12) {
      u = 0;
      t = Math.max(0, Math.min(1, -c / a));
    } else {
      const b = dpx * dqx + dpy * dqy;
      const denom = a * e - b * b;
      if (denom !== 0) t = Math.max(0, Math.min(1, (b * f - c * e) / denom));
      else t = 0;
      u = (b * t + f) / e;
      if (u < 0) {
        u = 0;
        t = Math.max(0, Math.min(1, -c / a));
      } else if (u > 1) {
        u = 1;
        t = Math.max(0, Math.min(1, (b - c) / a));
      }
    }
  }

  return {
    ax: p1x + dpx * t,
    ay: p1y + dpy * t,
    bx: q1x + dqx * u,
    by: q1y + dqy * u,
    t,
    u,
  };
}

/** Parameter of the closest point on AB to P, in [0, 1]. */
export function closestTOnSegment(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const denom = abx * abx + aby * aby;
  if (denom < 1e-12) return 0;
  return Math.max(0, Math.min(1, ((px - ax) * abx + (py - ay) * aby) / denom));
}

export function closestPointOnSegment(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): { x: number; y: number; t: number } {
  const t = closestTOnSegment(px, py, ax, ay, bx, by);
  return { x: ax + (bx - ax) * t, y: ay + (by - ay) * t, t };
}

export function segmentsInterfere(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
  dx: number,
  dy: number,
  minDist: number,
): boolean {
  if (segmentsIntersect(ax, ay, bx, by, cx, cy, dx, dy)) return true;
  const c = closestOnSegments(ax, ay, bx, by, cx, cy, dx, dy);
  const dist = Math.hypot(c.bx - c.ax, c.by - c.ay);
  return dist < minDist;
}

export function polylineInterfere(
  a: Vec2[],
  b: Vec2[],
  minDist: number,
  skipA: (i: number) => boolean = () => false,
  skipB: (i: number) => boolean = () => false,
): boolean {
  for (let i = 0; i < a.length - 1; i++) {
    if (skipA(i)) continue;
    for (let j = 0; j < b.length - 1; j++) {
      if (skipB(j)) continue;
      if (
        segmentsInterfere(
          a[i].x,
          a[i].y,
          a[i + 1].x,
          a[i + 1].y,
          b[j].x,
          b[j].y,
          b[j + 1].x,
          b[j + 1].y,
          minDist,
        )
      ) {
        return true;
      }
    }
  }
  return false;
}


/**
 * How far a live rope may sit off its stem–stem chord. A body-sized bow is
 * fine; a loop that leaves the two anchors' neighbourhood is not.
 */
export function wireBowBudget(span: number, rest: number): number {
  return Math.max(rest, span * 0.4) + 48;
}

/**
 * Pull interior points onto a tube around the chord through the ends. The
 * endpoints stay put — those are the stems.
 */
export function clampPolylineToChord(
  pts: { x: number; y: number }[],
  maxDev: number,
): void {
  if (pts.length < 3 || !(maxDev > 0) || !Number.isFinite(maxDev)) return;
  const a = pts[0];
  const b = pts[pts.length - 1];
  for (let i = 1; i < pts.length - 1; i++) {
    const q = closestPointOnSegment(pts[i].x, pts[i].y, a.x, a.y, b.x, b.y);
    const dx = pts[i].x - q.x;
    const dy = pts[i].y - q.y;
    const d = Math.hypot(dx, dy);
    if (d <= maxDev) continue;
    const k = maxDev / d;
    pts[i].x = q.x + dx * k;
    pts[i].y = q.y + dy * k;
  }
}

