import { ERA_RADIUS, boundRadius, triangleWorld, type Agent } from './agents.ts';
import { wrapDeltaVec, type Vec2 } from './wrap.ts';

const SKIN = 0.85;
export const SLOP = 0.35;

export interface Hit {
  nx: number;
  ny: number;
  overlap: number;
  px: number;
  py: number;
}

type Shape =
  | { tag: 'circle'; x: number; y: number; r: number; points: Vec2[] }
  | { tag: 'poly'; x: number; y: number; r: number; points: Vec2[] };

function project(points: Vec2[], nx: number, ny: number): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const p of points) {
    const d = p.x * nx + p.y * ny;
    if (d < min) min = d;
    if (d > max) max = d;
  }
  return { min: min - SKIN, max: max + SKIN };
}

function projectCircle(x: number, y: number, r: number, nx: number, ny: number): { min: number; max: number } {
  const m = x * nx + y * ny;
  const rad = r + SKIN;
  return { min: m - rad, max: m + rad };
}

function projShape(s: Shape, nx: number, ny: number): { min: number; max: number } {
  if (s.tag === 'circle') return projectCircle(s.x, s.y, s.r, nx, ny);
  return project(s.points, nx, ny);
}

function overlapOnAxis(
  a: { min: number; max: number },
  b: { min: number; max: number },
): number | null {
  const left = b.max - a.min;
  const right = a.max - b.min;
  if (left <= 0 || right <= 0) return null;
  return Math.min(left, right);
}

function support(points: Vec2[], nx: number, ny: number): Vec2 {
  let best = points[0];
  let bestD = best.x * nx + best.y * ny;
  for (let i = 1; i < points.length; i++) {
    const d = points[i].x * nx + points[i].y * ny;
    if (d > bestD) {
      bestD = d;
      best = points[i];
    }
  }
  return best;
}

function polyAxes(verts: Vec2[]): Vec2[] {
  const axes: Vec2[] = [];
  for (let i = 0; i < verts.length; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % verts.length];
    const ex = b.x - a.x;
    const ey = b.y - a.y;
    const len = Math.hypot(ex, ey) || 1;
    axes.push({ x: -ey / len, y: ex / len });
  }
  return axes;
}

function closestPointOnSeg(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): Vec2 {
  const abx = bx - ax;
  const aby = by - ay;
  const t = Math.max(
    0,
    Math.min(1, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby || 1)),
  );
  return { x: ax + abx * t, y: ay + aby * t };
}

function closestOnPoly(px: number, py: number, verts: Vec2[]): Vec2 {
  let best = verts[0];
  let bestD = Infinity;
  for (let i = 0; i < verts.length; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % verts.length];
    const q = closestPointOnSeg(px, py, a.x, a.y, b.x, b.y);
    const d = (q.x - px) * (q.x - px) + (q.y - py) * (q.y - py);
    if (d < bestD) {
      bestD = d;
      best = q;
    }
  }
  return best;
}

function shapeAt(agent: Agent, x: number, y: number): Shape {
  if (agent.kind === 'era') {
    const r = ERA_RADIUS * agent.scale;
    return { tag: 'circle', x, y, r, points: [{ x, y }] };
  }
  return { tag: 'poly', x, y, r: boundRadius(agent), points: triangleWorld(agent, x, y) };
}

export function queryHit(A: Agent, B: Agent, w: number, h: number): Hit | null {
  const d = wrapDeltaVec(A.x, A.y, B.x, B.y, w, h);
  if (Math.hypot(d.x, d.y) > boundRadius(A) + boundRadius(B) + SKIN * 2 + 2) return null;

  const sa = shapeAt(A, A.x, A.y);
  const sb = shapeAt(B, A.x + d.x, A.y + d.y);
  const toBx = sb.x - sa.x;
  const toBy = sb.y - sa.y;

  if (sa.tag === 'circle' && sb.tag === 'circle') {
    const dist = Math.hypot(toBx, toBy);
    const minDist = sa.r + sb.r + SKIN * 2;
    if (dist >= minDist) return null;
    if (dist < 1e-6) {
      return { nx: 1, ny: 0, overlap: minDist, px: sa.x + sa.r, py: sa.y };
    }
    const nx = toBx / dist;
    const ny = toBy / dist;
    return {
      nx,
      ny,
      overlap: minDist - dist,
      px: sa.x + nx * sa.r,
      py: sa.y + ny * sa.r,
    };
  }

  const axes: Vec2[] = [];
  if (sa.tag === 'poly') axes.push(...polyAxes(sa.points));
  if (sb.tag === 'poly') axes.push(...polyAxes(sb.points));
  if (sa.tag === 'circle' && sb.tag === 'poly') {
    const q = closestOnPoly(sa.x, sa.y, sb.points);
    const dx = q.x - sa.x;
    const dy = q.y - sa.y;
    const len = Math.hypot(dx, dy);
    if (len > 1e-6) axes.push({ x: dx / len, y: dy / len });
  } else if (sb.tag === 'circle' && sa.tag === 'poly') {
    const q = closestOnPoly(sb.x, sb.y, sa.points);
    const dx = q.x - sb.x;
    const dy = q.y - sb.y;
    const len = Math.hypot(dx, dy);
    if (len > 1e-6) axes.push({ x: dx / len, y: dy / len });
  }

  let nx = 1;
  let ny = 0;
  let bestO = Infinity;
  for (const ax of axes) {
    const o = overlapOnAxis(projShape(sa, ax.x, ax.y), projShape(sb, ax.x, ax.y));
    if (o === null) return null;
    let x = ax.x;
    let y = ax.y;
    const len = Math.hypot(x, y) || 1;
    x /= len;
    y /= len;
    if (x * toBx + y * toBy < 0) {
      x = -x;
      y = -y;
    }
    if (o < bestO) {
      bestO = o;
      nx = x;
      ny = y;
    }
  }
  if (!isFinite(bestO) || bestO <= 0) return null;

  const pA =
    sa.tag === 'circle' ? { x: sa.x + nx * sa.r, y: sa.y + ny * sa.r } : support(sa.points, nx, ny);
  const pB =
    sb.tag === 'circle'
      ? { x: sb.x - nx * sb.r, y: sb.y - ny * sb.r }
      : support(sb.points, -nx, -ny);
  return {
    nx,
    ny,
    overlap: bestO,
    px: (pA.x + pB.x) * 0.5,
    py: (pA.y + pB.y) * 0.5,
  };
}

