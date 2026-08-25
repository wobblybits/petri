import { momentOfInertia, PORT_EXTRUDE, portLocal, stemRoot, type Agent, type PortSlot } from './agents.ts';
import { bezierPoint, type Cubic } from './curve.ts';
import type { Params } from './params.ts';
import { clamp, rotate, wrap, wrapAngle, wrapDeltaVec, type Vec2 } from './wrap.ts';

export const TARGET_LINK = 6;
export const MIN_LINKS = 4;
export const MAX_LINKS = 24;
export const MIN_SEG = 3.5;
export const CHAIN_MASS = 0.08;

export interface ChainNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  prevX: number;
  prevY: number;
  integVx: number;
  integVy: number;
}

export function desiredLinks(rest: number): number {
  return clamp(Math.round(rest / TARGET_LINK), MIN_LINKS, MAX_LINKS);
}

export function sampleChain(c: Cubic, nLinks: number, w: number, h: number): ChainNode[] {
  const links = clamp(nLinks, MIN_LINKS, MAX_LINKS);
  const nodes: ChainNode[] = [];
  for (let i = 1; i < links; i++) {
    const p = bezierPoint(c.p0, c.p1, c.p2, c.p3, i / links);
    const x = wrap(p.x, w);
    const y = wrap(p.y, h);
    nodes.push({ x, y, vx: 0, vy: 0, prevX: x, prevY: y, integVx: 0, integVy: 0 });
  }
  return nodes;
}

export function reduceChain(nodes: ChainNode[], rest: number): void {
  while (nodes.length + 1 > MIN_LINKS && rest / (nodes.length + 1) < MIN_SEG) {
    nodes.splice(Math.floor(nodes.length / 2), 1);
  }
}

function smoothstep(t: number): number {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

/** Stiff at the hull, eases along the original port, then softer on the free wire. */
export function taperStiffness(s: number, portLen: number, k: number): number {
  const kBase = k * 2.4;
  const kPort = k * 0.85;
  const kWire = k * 0.12;
  const L = Math.max(1e-6, portLen);
  if (s <= L) return kBase + (kPort - kBase) * smoothstep(s / L);
  const t = smoothstep((s - L) / (L * 3.5 + 18));
  return kPort + (kWire - kPort) * t;
}

function portLength(agent: Agent): number {
  return PORT_EXTRUDE * agent.scale;
}

function portOutward(agent: Agent, slot: PortSlot, heading: number): Vec2 {
  const tip = portLocal(agent.kind, slot);
  const root = stemRoot(agent.kind, slot);
  return rotate(
    (tip.x - root.x) * agent.scale,
    (tip.y - root.y) * agent.scale,
    heading,
  );
}

export function unwrapPoints(pts: Vec2[], w: number, h: number): Vec2[] {
  if (pts.length === 0) return [];
  const out: Vec2[] = [{ x: pts[0].x, y: pts[0].y }];
  for (let i = 1; i < pts.length; i++) {
    const d = wrapDeltaVec(out[i - 1].x, out[i - 1].y, pts[i].x, pts[i].y, w, h);
    out.push({ x: out[i - 1].x + d.x, y: out[i - 1].y + d.y });
  }
  return out;
}

export function polylineLength(pts: Vec2[], w: number, h: number): number {
  let len = 0;
  for (let i = 1; i < pts.length; i++) {
    const d = wrapDeltaVec(pts[i - 1].x, pts[i - 1].y, pts[i].x, pts[i].y, w, h);
    len += Math.hypot(d.x, d.y);
  }
  return len;
}

export function catmullSegment(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2): Cubic {
  return {
    p0: p1,
    p1: { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 },
    p2: { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 },
    p3: p2,
  };
}

interface Pose {
  x: number;
  y: number;
  heading: number;
}

function attachOffset(agent: Agent, slot: PortSlot, heading: number): Vec2 {
  const loc = stemRoot(agent.kind, slot);
  return rotate(loc.x * agent.scale, loc.y * agent.scale, heading);
}

function endWeight(agent: Agent, r: Vec2, nx: number, ny: number, rot: number): number {
  if (agent.locked) return 0;
  const invM = 1 / Math.max(0.08, agent.mass);
  const invI = (1 / Math.max(1e-4, momentOfInertia(agent))) * rot;
  const rxn = r.x * ny - r.y * nx;
  return invM + invI * rxn * rxn;
}

function applyEnd(
  pose: Pose,
  agent: Agent,
  slot: PortSlot,
  nx: number,
  ny: number,
  dlambda: number,
  rot: number,
): void {
  if (agent.locked || Math.abs(dlambda) < 1e-12) return;
  const invM = 1 / Math.max(0.08, agent.mass);
  const invI = (1 / Math.max(1e-4, momentOfInertia(agent))) * rot;
  const r = attachOffset(agent, slot, pose.heading);
  pose.x += invM * dlambda * nx;
  pose.y += invM * dlambda * ny;
  pose.heading += invI * (r.x * ny - r.y * nx) * dlambda;
}

function stemOf(pose: Pose, agent: Agent, slot: PortSlot): Vec2 {
  const r = attachOffset(agent, slot, pose.heading);
  return { x: pose.x + r.x, y: pose.y + r.y };
}

function commitPose(agent: Agent, pose: Pose): void {
  if (agent.locked) return;
  agent.x = pose.x;
  agent.y = pose.y;
  agent.heading = wrapAngle(pose.heading);
}

function pinPortRay(
  pts: Vec2[],
  n: number,
  pose: Pose,
  agent: Agent,
  slot: PortSlot,
  portLen: number,
  linkRest: number,
  fromA: boolean,
  k: number,
  invMNode: number,
  dt2: number,
  lambda: Float64Array,
  lambdaOff: number,
): void {
  const origin = fromA ? pts[0] : pts[n + 1];
  const axis = portOutward(agent, slot, pose.heading);
  const alen = Math.hypot(axis.x, axis.y) || 1;
  const ux = axis.x / alen;
  const uy = axis.y / alen;
  for (let i = 0; i < n; i++) {
    const s = fromA ? (i + 1) * linkRest : (n - i) * linkRest;
    if (s > portLen * 1.08) continue;
    const p = pts[i + 1];
    const tx = origin.x + ux * s;
    const ty = origin.y + uy * s;
    const kPin = taperStiffness(s, portLen, k);
    const at = 1 / Math.max(16, kPin * 160) / dt2;
    const Cx = p.x - tx;
    const Cy = p.y - ty;
    const denom = invMNode + at;
    if (denom < 1e-10) continue;
    const li = lambdaOff + i;
    const dlamX = (-Cx - at * lambda[li]) / denom;
    const dlamY = (-Cy - at * lambda[li]) / denom;
    lambda[li] += 0.5 * (dlamX + dlamY);
    p.x += invMNode * dlamX;
    p.y += invMNode * dlamY;
  }
}

function xpbdBend(
  a: Vec2,
  b: Vec2,
  c: Vec2,
  wA: number,
  wB: number,
  wC: number,
  alphaTilde: number,
  lambda: number,
): number {
  const Cx = a.x - 2 * b.x + c.x;
  const Cy = a.y - 2 * b.y + c.y;
  const denom = wA + 4 * wB + wC + alphaTilde;
  if (denom < 1e-10) return lambda;
  const dlamX = (-Cx - alphaTilde * lambda) / denom;
  const dlamY = (-Cy - alphaTilde * lambda) / denom;
  a.x += wA * dlamX;
  a.y += wA * dlamY;
  b.x -= 2 * wB * dlamX;
  b.y -= 2 * wB * dlamY;
  c.x += wC * dlamX;
  c.y += wC * dlamY;
  return lambda + 0.5 * (dlamX + dlamY);
}

/**
 * XPBD rubber-band: stiff stretch, soft compression, bending that prefers a
 * straight geodesic, and force+torque at the ports so agents turn to face the band.
 */
export function solveChain(
  A: Agent,
  aSlot: PortSlot,
  B: Agent,
  bSlot: PortSlot,
  nodes: ChainNode[],
  rest: number,
  params: Params,
  dt: number,
  w: number,
  h: number,
  rotA = 1,
  rotB = 1,
): number {
  const n = nodes.length;
  const nPts = n + 2;
  const nLinks = n + 1;
  const linkRest = rest / nLinks;
  const invMNode = 1 / CHAIN_MASS;
  const k = Math.max(0.5, params.springK);
  const portA = portLength(A);
  const portB = portLength(B);
  const sub = 2;
  const sdt = dt / sub;
  const iters = 10;

  const dCom = wrapDeltaVec(A.x, A.y, B.x, B.y, w, h);
  const poseA: Pose = { x: A.x, y: A.y, heading: A.heading };
  const poseB: Pose = { x: A.x + dCom.x, y: A.y + dCom.y, heading: B.heading };

  const pts: Vec2[] = new Array(nPts);
  const prev: Vec2[] = new Array(nPts);

  for (let s = 0; s < sub; s++) {
    pts[0] = stemOf(poseA, A, aSlot);
    const raw: Vec2[] = [pts[0], ...nodes, stemOf(poseB, B, bSlot)];
    const un = unwrapPoints(raw, w, h);
    for (let i = 0; i < nPts; i++) {
      pts[i] = { x: un[i].x, y: un[i].y };
      prev[i] = { x: un[i].x, y: un[i].y };
    }
    for (let i = 0; i < n; i++) {
      pts[i + 1].x += nodes[i].vx * sdt;
      pts[i + 1].y += nodes[i].vy * sdt;
    }

    const lamDist = new Float64Array(nLinks);
    const lamBend = new Float64Array(Math.max(0, nPts - 2));
    const lamPin = new Float64Array(n * 2);
    const dt2 = sdt * sdt;

    for (let it = 0; it < iters; it++) {
      pts[0] = stemOf(poseA, A, aSlot);
      pts[nPts - 1] = stemOf(poseB, B, bSlot);

      for (let i = 0; i < nLinks; i++) {
        const p = pts[i];
        const q = pts[i + 1];
        const dx = q.x - p.x;
        const dy = q.y - p.y;
        const dist = Math.hypot(dx, dy) || 1e-6;
        const nx = dx / dist;
        const ny = dy / dist;
        const C = dist - linkRest;
        const sA = (i + 0.5) * linkRest;
        const sB = rest - sA;
        const kLink = Math.max(taperStiffness(sA, portA, k), taperStiffness(sB, portB, k));
        const alphaStretch = 1 / Math.max(6, kLink * 36);
        const alpha = C >= 0 ? alphaStretch : alphaStretch * 14;
        const at = alpha / dt2;
        const rA = attachOffset(A, aSlot, poseA.heading);
        const rB = attachOffset(B, bSlot, poseB.heading);
        const wP = i === 0 ? endWeight(A, rA, nx, ny, rotA) : invMNode;
        const wQ = i === nLinks - 1 ? endWeight(B, rB, nx, ny, rotB) : invMNode;
        const denom = wP + wQ + at;
        if (denom < 1e-10) continue;
        const dlam = (-C - at * lamDist[i]) / denom;
        lamDist[i] += dlam;
        if (i === 0) applyEnd(poseA, A, aSlot, -nx, -ny, dlam, rotA);
        else {
          p.x -= invMNode * dlam * nx;
          p.y -= invMNode * dlam * ny;
        }
        if (i === nLinks - 1) applyEnd(poseB, B, bSlot, nx, ny, dlam, rotB);
        else {
          q.x += invMNode * dlam * nx;
          q.y += invMNode * dlam * ny;
        }
      }

      pinPortRay(
        pts,
        n,
        poseA,
        A,
        aSlot,
        portA,
        linkRest,
        true,
        k,
        invMNode,
        dt2,
        lamPin,
        0,
      );
      pinPortRay(
        pts,
        n,
        poseB,
        B,
        bSlot,
        portB,
        linkRest,
        false,
        k,
        invMNode,
        dt2,
        lamPin,
        n,
      );

      for (let i = 1; i < nPts - 1; i++) {
        const sA = i * linkRest;
        const sB = rest - sA;
        const kBend = Math.max(taperStiffness(sA, portA, k), taperStiffness(sB, portB, k));
        const wA = i - 1 === 0 ? 0 : invMNode;
        const wC = i + 1 === nPts - 1 ? 0 : invMNode;
        lamBend[i - 1] = xpbdBend(
          pts[i - 1],
          pts[i],
          pts[i + 1],
          wA,
          invMNode,
          wC,
          1 / Math.max(4, kBend * 28) / dt2,
          lamBend[i - 1],
        );
      }
    }

    const keep = Math.exp(-Math.max(0, params.springDamp) * sdt);
    for (let i = 0; i < n; i++) {
      const node = nodes[i];
      const p = pts[i + 1];
      node.vx = ((p.x - prev[i + 1].x) / sdt) * keep;
      node.vy = ((p.y - prev[i + 1].y) / sdt) * keep;
      node.x = wrap(p.x, w);
      node.y = wrap(p.y, h);
    }
  }

  commitPose(A, poseA);
  commitPose(B, poseB);

  const sA = stemOf(poseA, A, aSlot);
  const sB = stemOf(poseB, B, bSlot);
  return polylineLength([sA, ...nodes, sB], w, h);
}

export function chordDeviation(pts: Vec2[], w: number, h: number): number {
  if (pts.length < 3) return 0;
  const un = unwrapPoints(pts, w, h);
  const a = un[0];
  const b = un[un.length - 1];
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  let max = 0;
  for (let i = 1; i < un.length - 1; i++) {
    const t = ((un[i].x - a.x) * dx + (un[i].y - a.y) * dy) / (len * len);
    const px = a.x + dx * t;
    const py = a.y + dy * t;
    max = Math.max(max, Math.hypot(un[i].x - px, un[i].y - py));
  }
  return max;
}
