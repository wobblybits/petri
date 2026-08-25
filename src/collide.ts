import {
  ERA_RADIUS,
  boundRadius,
  momentOfInertia,
  stemWorld,
  triangleWorld,
  type Agent,
} from './agents.ts';
import { CHAIN_MASS, unwrapPoints, type ChainNode } from './chain.ts';
import { closestOnSegments, WIRE_RADIUS } from './geom.ts';
import type { Graph, Wire } from './graph.ts';
import { wrap, wrapDeltaVec, type Vec2 } from './wrap.ts';

const SKIN = 0.85;
const RESTITUTION = 0;
const FRICTION = 0;
export const SLOP = 0.35;
const CORRECT = 0.55;
export { WIRE_RADIUS };

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

export function resolveHit(A: Agent, B: Agent, hit: Hit, w: number, h: number): void {
  const invMA = 1 / Math.max(0.08, A.mass);
  const invMB = 1 / Math.max(0.08, B.mass);
  const invIA = 1 / Math.max(1e-4, momentOfInertia(A));
  const invIB = 1 / Math.max(1e-4, momentOfInertia(B));
  const { nx, ny } = hit;

  const rAx = hit.px - A.x;
  const rAy = hit.py - A.y;
  const d = wrapDeltaVec(A.x, A.y, B.x, B.y, w, h);
  const Bx = A.x + d.x;
  const By = A.y + d.y;
  const rBx = hit.px - Bx;
  const rBy = hit.py - By;

  const vAx = A.vx - A.omega * rAy;
  const vAy = A.vy + A.omega * rAx;
  const vBx = B.vx - B.omega * rBy;
  const vBy = B.vy + B.omega * rBx;
  const rvx = vBx - vAx;
  const rvy = vBy - vAy;
  const velN = rvx * nx + rvy * ny;

  const rnA = rAx * ny - rAy * nx;
  const rnB = rBx * ny - rBy * nx;
  const denomN = invMA + invMB + rnA * rnA * invIA + rnB * rnB * invIB;

  if (velN < 0 && denomN > 1e-8) {
    const jn = (-(1 + RESTITUTION) * velN) / denomN;
    A.vx -= jn * invMA * nx;
    A.vy -= jn * invMA * ny;
    A.omega -= rnA * jn * invIA;
    B.vx += jn * invMB * nx;
    B.vy += jn * invMB * ny;
    B.omega += rnB * jn * invIB;

    if (FRICTION > 0) {
      const tx = -ny;
      const ty = nx;
      const vAx2 = A.vx - A.omega * rAy;
      const vAy2 = A.vy + A.omega * rAx;
      const vBx2 = B.vx - B.omega * rBy;
      const vBy2 = B.vy + B.omega * rBx;
      const velT = (vBx2 - vAx2) * tx + (vBy2 - vAy2) * ty;
      const rtA = rAx * ty - rAy * tx;
      const rtB = rBx * ty - rBy * tx;
      const denomT = invMA + invMB + rtA * rtA * invIA + rtB * rtB * invIB;
      if (denomT > 1e-8) {
        let jt = -velT / denomT;
        const maxF = FRICTION * jn;
        if (jt > maxF) jt = maxF;
        else if (jt < -maxF) jt = -maxF;
        A.vx -= jt * invMA * tx;
        A.vy -= jt * invMA * ty;
        A.omega -= rtA * jt * invIA;
        B.vx += jt * invMB * tx;
        B.vy += jt * invMB * ty;
        B.omega += rtB * jt * invIB;
      }
    }
  }

  const corr = (Math.max(0, hit.overlap - SLOP) * CORRECT) / (invMA + invMB);
  if (corr > 0) {
    A.x -= nx * corr * invMA;
    A.y -= ny * corr * invMA;
    B.x += nx * corr * invMB;
    B.y += ny * corr * invMB;
    A.x = wrap(A.x, w);
    A.y = wrap(A.y, h);
    B.x = wrap(B.x, w);
    B.y = wrap(B.y, h);
  }
}

function circleProbe(x: number, y: number, r: number, mass: number): Agent {
  return {
    id: -2,
    kind: 'era',
    x,
    y,
    vx: 0,
    vy: 0,
    heading: 0,
    omega: 0,
    mass,
    alpha: 1,
    scale: r / ERA_RADIUS,
    locked: false,
    stun: 0,
    prevX: x,
    prevY: y,
    prevHeading: 0,
    integVx: 0,
    integVy: 0,
    integOmega: 0,
  };
}

function skipWireOwner(endA: number, endB: number, agent: Agent): boolean {
  return agent.id === endA || agent.id === endB;
}

type EndRef =
  | { tag: 'node'; node: ChainNode }
  | { tag: 'agent'; agent: Agent; rx: number; ry: number }
  | { tag: 'none' };

function wirePolyline(
  wire: Wire,
  agents: Map<number, Agent>,
  w: number,
  h: number,
): { A: Agent; B: Agent; pts: Vec2[] } | null {
  const A = agents.get(wire.a.id);
  const B = agents.get(wire.b.id);
  if (!A || !B) return null;
  const stemA = stemWorld(A, wire.a.slot, w, h);
  const stemB = stemWorld(B, wire.b.slot, w, h);
  return { A, B, pts: unwrapPoints([stemA, ...wire.nodes, stemB], w, h) };
}

function endAt(
  wire: Wire,
  A: Agent,
  B: Agent,
  pts: Vec2[],
  index: number,
  w: number,
  h: number,
): EndRef {
  if (index === 0) {
    const stem = stemWorld(A, wire.a.slot, w, h);
    return { tag: 'agent', agent: A, rx: stem.x - A.x, ry: stem.y - A.y };
  }
  if (index === pts.length - 1) {
    const stem = stemWorld(B, wire.b.slot, w, h);
    return { tag: 'agent', agent: B, rx: stem.x - B.x, ry: stem.y - B.y };
  }
  const node = wire.nodes[index - 1];
  if (!node) return { tag: 'none' };
  return { tag: 'node', node };
}

function endLinearInv(e: EndRef): number {
  if (e.tag === 'node') return 1 / CHAIN_MASS;
  if (e.tag === 'agent') {
    if (e.agent.locked) return 0;
    return 1 / Math.max(0.08, e.agent.mass);
  }
  return 0;
}

function endEffInv(e: EndRef, nx: number, ny: number): number {
  if (e.tag === 'node') return 1 / CHAIN_MASS;
  if (e.tag === 'agent') {
    if (e.agent.locked) return 0;
    const invM = 1 / Math.max(0.08, e.agent.mass);
    const invI = 1 / Math.max(1e-4, momentOfInertia(e.agent));
    const rn = e.rx * ny - e.ry * nx;
    return invM + rn * rn * invI;
  }
  return 0;
}

function endVel(e: EndRef): Vec2 {
  if (e.tag === 'node') return { x: e.node.vx, y: e.node.vy };
  if (e.tag === 'agent') {
    return {
      x: e.agent.vx - e.agent.omega * e.ry,
      y: e.agent.vy + e.agent.omega * e.rx,
    };
  }
  return { x: 0, y: 0 };
}

function applyEndImpulse(e: EndRef, jx: number, jy: number): void {
  if (e.tag === 'node') {
    const inv = 1 / CHAIN_MASS;
    e.node.vx += jx * inv;
    e.node.vy += jy * inv;
  } else if (e.tag === 'agent' && !e.agent.locked) {
    const invM = 1 / Math.max(0.08, e.agent.mass);
    const invI = 1 / Math.max(1e-4, momentOfInertia(e.agent));
    e.agent.vx += jx * invM;
    e.agent.vy += jy * invM;
    e.agent.omega += (e.rx * jy - e.ry * jx) * invI;
  }
}

function applyEndShift(e: EndRef, dx: number, dy: number): void {
  if (e.tag === 'node') {
    e.node.x += dx;
    e.node.y += dy;
  } else if (e.tag === 'agent' && !e.agent.locked) {
    e.agent.x += dx;
    e.agent.y += dy;
  }
}

function segInv(e0: EndRef, e1: EndRef, t: number, nx: number, ny: number): number {
  const s = 1 - t;
  return s * s * endEffInv(e0, nx, ny) + t * t * endEffInv(e1, nx, ny);
}

function segLinInv(e0: EndRef, e1: EndRef, t: number): number {
  const s = 1 - t;
  return s * s * endLinearInv(e0) + t * t * endLinearInv(e1);
}

function segVel(e0: EndRef, e1: EndRef, t: number): Vec2 {
  const a = endVel(e0);
  const b = endVel(e1);
  return { x: a.x * (1 - t) + b.x * t, y: a.y * (1 - t) + b.y * t };
}

function applySegImpulse(e0: EndRef, e1: EndRef, t: number, jx: number, jy: number): void {
  applyEndImpulse(e0, jx * (1 - t), jy * (1 - t));
  applyEndImpulse(e1, jx * t, jy * t);
}

function applySegShift(e0: EndRef, e1: EndRef, t: number, dx: number, dy: number): void {
  applyEndShift(e0, dx * (1 - t), dy * (1 - t));
  applyEndShift(e1, dx * t, dy * t);
}

function resolveCapsules(
  eA0: EndRef,
  eA1: EndRef,
  t: number,
  eB0: EndRef,
  eB1: EndRef,
  u: number,
  nx: number,
  ny: number,
  overlap: number,
): void {
  const invA = segInv(eA0, eA1, t, nx, ny);
  const invB = segInv(eB0, eB1, u, nx, ny);
  const denom = invA + invB;
  if (denom < 1e-10) return;

  const vA = segVel(eA0, eA1, t);
  const vB = segVel(eB0, eB1, u);
  const velN = (vB.x - vA.x) * nx + (vB.y - vA.y) * ny;
  if (velN < 0) {
    const jn = (-(1 + RESTITUTION) * velN) / denom;
    applySegImpulse(eA0, eA1, t, -jn * nx, -jn * ny);
    applySegImpulse(eB0, eB1, u, jn * nx, jn * ny);
  }

  const linA = segLinInv(eA0, eA1, t);
  const linB = segLinInv(eB0, eB1, u);
  const lin = linA + linB;
  if (lin < 1e-10) return;
  const corr = (Math.max(0, overlap - SLOP) * CORRECT) / lin;
  if (corr > 0) {
    applySegShift(eA0, eA1, t, -nx * corr * linA, -ny * corr * linA);
    applySegShift(eB0, eB1, u, nx * corr * linB, ny * corr * linB);
  }
}

function skipIncidentStem(wire: Wire, seg: number, nSeg: number, agentId: number): boolean {
  if (seg === 0 && wire.a.id === agentId) return true;
  if (seg === nSeg - 1 && wire.b.id === agentId) return true;
  return false;
}

export function collideWireAgents(
  graph: Graph,
  agents: Map<number, Agent>,
  w: number,
  h: number,
): void {
  for (let pass = 0; pass < 3; pass++) {
    for (const wire of graph.wires.values()) {
      const poly = wirePolyline(wire, agents, w, h);
      if (!poly) continue;
      const { A, B, pts } = poly;
      const n = wire.nodes.length;

      for (let i = 0; i < n; i++) {
        const node = wire.nodes[i];
        for (const agent of agents.values()) {
          if (agent.locked) continue;
          if (skipWireOwner(A.id, B.id, agent)) continue;
          const probe = circleProbe(node.x, node.y, WIRE_RADIUS, CHAIN_MASS);
          probe.vx = node.vx;
          probe.vy = node.vy;
          const hit = queryHit(probe, agent, w, h);
          if (!hit) continue;
          resolveHit(probe, agent, hit, w, h);
          node.x = wrap(probe.x, w);
          node.y = wrap(probe.y, h);
          node.vx = probe.vx;
          node.vy = probe.vy;
        }
      }

      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[i];
        const p1 = pts[i + 1];
        const mx = (p0.x + p1.x) * 0.5;
        const my = (p0.y + p1.y) * 0.5;
        for (const agent of agents.values()) {
          if (agent.locked) continue;
          if (skipWireOwner(A.id, B.id, agent)) continue;
          const probe = circleProbe(wrap(mx, w), wrap(my, h), WIRE_RADIUS, CHAIN_MASS);
          const hit = queryHit(probe, agent, w, h);
          if (!hit) continue;
          const ox = probe.x;
          const oy = probe.y;
          resolveHit(probe, agent, hit, w, h);
          const dx = probe.x - ox;
          const dy = probe.y - oy;
          if (i > 0 && i <= n) {
            wire.nodes[i - 1].x = wrap(wire.nodes[i - 1].x + dx * 0.5, w);
            wire.nodes[i - 1].y = wrap(wire.nodes[i - 1].y + dy * 0.5, h);
          }
          if (i < n) {
            wire.nodes[i].x = wrap(wire.nodes[i].x + dx * 0.5, w);
            wire.nodes[i].y = wrap(wire.nodes[i].y + dy * 0.5, h);
          }
        }
      }
    }
  }
}

export function collideWires(
  graph: Graph,
  agents: Map<number, Agent>,
  w: number,
  h: number,
): void {
  const wires = [...graph.wires.values()];
  const minDist = WIRE_RADIUS * 2;

  for (let pass = 0; pass < 4; pass++) {
    const polys = wires.map((wire) => wirePolyline(wire, agents, w, h));
    for (let a = 0; a < wires.length; a++) {
      const pa = polys[a];
      if (!pa) continue;
      const nA = pa.pts.length - 1;
      for (let b = a; b < wires.length; b++) {
        const pb = polys[b];
        if (!pb) continue;
        const nB = pb.pts.length - 1;
        const same = a === b;
        for (let i = 0; i < nA; i++) {
          for (let j = 0; j < nB; j++) {
            if (same && Math.abs(i - j) <= 2) continue;
            if (same && i >= j) continue;
            if (!same) {
              const shared = [pa.A.id, pa.B.id].filter((id) => id === pb.A.id || id === pb.B.id);
              if (
                shared.some(
                  (id) =>
                    skipIncidentStem(wires[a], i, nA, id) && skipIncidentStem(wires[b], j, nB, id),
                )
              ) {
                continue;
              }
            }
            const c = closestOnSegments(
              pa.pts[i].x,
              pa.pts[i].y,
              pa.pts[i + 1].x,
              pa.pts[i + 1].y,
              pb.pts[j].x,
              pb.pts[j].y,
              pb.pts[j + 1].x,
              pb.pts[j + 1].y,
            );
            const dx = c.bx - c.ax;
            const dy = c.by - c.ay;
            let dist = Math.hypot(dx, dy);
            if (dist >= minDist) continue;
            let nx: number;
            let ny: number;
            if (dist < 1e-8) {
              const sx = pa.pts[i + 1].x - pa.pts[i].x;
              const sy = pa.pts[i + 1].y - pa.pts[i].y;
              const sl = Math.hypot(sx, sy) || 1;
              nx = -sy / sl;
              ny = sx / sl;
              dist = 0;
            } else {
              nx = dx / dist;
              ny = dy / dist;
            }
            const eA0 = endAt(wires[a], pa.A, pa.B, pa.pts, i, w, h);
            const eA1 = endAt(wires[a], pa.A, pa.B, pa.pts, i + 1, w, h);
            const eB0 = endAt(wires[b], pb.A, pb.B, pb.pts, j, w, h);
            const eB1 = endAt(wires[b], pb.A, pb.B, pb.pts, j + 1, w, h);
            resolveCapsules(eA0, eA1, c.t, eB0, eB1, c.u, nx, ny, minDist - dist);
          }
        }
      }
    }
  }
}

