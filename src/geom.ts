import type { Vec2 } from './wrap.ts';

export const WIRE_RADIUS = 2.5;
/** World px of string displacement at waveguide |sample| = 1 after AGC. */
export const WAVE_DISP_PX = 16;
/** Spatial samples in a wire-pluck profile. */
export const PLUCK_BINS = 16;

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
 * Signed perpendicular offsets of `n` arc-length samples from the chord
 * through the polyline's ends. Ends are pinned to 0 — a string.
 */
export function transverseProfile(
  pts: Vec2[],
  n = PLUCK_BINS,
): { samples: number[]; peak: number; at: number } {
  const samples = new Array<number>(Math.max(2, n)).fill(0);
  const bins = samples.length;
  if (pts.length < 2) return { samples, peak: 0, at: 0.5 };
  const a = pts[0];
  const b = pts[pts.length - 1];
  const cx = b.x - a.x;
  const cy = b.y - a.y;
  const cLen = Math.hypot(cx, cy);
  const nx = cLen > 1e-6 ? -cy / cLen : 0;
  const ny = cLen > 1e-6 ? cx / cLen : -1;

  const seg: number[] = [];
  let total = 0;
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    seg.push(d);
    total += d;
  }
  if (total < 1e-6) return { samples, peak: 0, at: 0.5 };

  let peak = 0;
  let at = 0.5;
  for (let i = 0; i < bins; i++) {
    if (i === 0 || i === bins - 1) {
      samples[i] = 0;
      continue;
    }
    const target = (i / (bins - 1)) * total;
    let acc = 0;
    let s = 0;
    while (s < seg.length - 1 && acc + seg[s] < target) {
      acc += seg[s];
      s++;
    }
    const span = seg[s];
    const u = span < 1e-9 ? 0 : (target - acc) / span;
    const px = pts[s].x + (pts[s + 1].x - pts[s].x) * u;
    const py = pts[s].y + (pts[s + 1].y - pts[s].y) * u;
    const off = (px - a.x) * nx + (py - a.y) * ny;
    samples[i] = off;
    if (Math.abs(off) > peak) {
      peak = Math.abs(off);
      at = i / (bins - 1);
    }
  }
  return { samples, peak, at };
}

/** Axis-aligned bounds of a rope polyline, stems included. */
export function ropeAabb(
  a: Vec2,
  nodes: Vec2[],
  b: Vec2,
): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = a.x < b.x ? a.x : b.x;
  let maxX = a.x > b.x ? a.x : b.x;
  let minY = a.y < b.y ? a.y : b.y;
  let maxY = a.y > b.y ? a.y : b.y;
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (n.x < minX) minX = n.x;
    else if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y;
    else if (n.y > maxY) maxY = n.y;
  }
  return { minX, minY, maxX, maxY };
}
