import { momentOfInertia, portLocal, poseHeld, stemRoot, type Agent, type PortSlot } from './agents.ts';
import { bezierPoint, type Cubic } from './curve.ts';
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
}

/**
 * XPBD compliance, in (length² / force) units. Smaller is stiffer. These are the
 * only stiffness numbers in the joint solver; everything else follows from them.
 *
 *   span — the joint: holds two wired ports at the wire's rest length
 *   link — near-rigid against stretch, weak against compression: a rope, not a
 *          rod. Slack has to hang harmlessly, otherwise surplus rope pushes its
 *          own anchors around and the length feedback loop goes unstable.
 *   bend — soft, so the rope curves smoothly rather than kinking
 *   contact — stiff but not rigid. Resolving a deep overlap in a single
 *          substep turns into an enormous derived angular velocity, because
 *          velocity here is a position delta divided by h.
 *   shape — pulls the rope toward the curve that leaves both ports along their
 *          axes. Without it a slack rope is neutrally stable: nothing decides
 *          which of its many slack shapes it should take, so it wanders, and
 *          every wander is amplified into velocity by 1/h.
 *
 * Port-axis alignment is deliberately not here — it is an actuator, not a
 * material property. See Sim.portTorques.
 */
export const COMPLIANCE: Record<"span" | "link" | "bend" | "contact" | "shape", number> = {
  span: 3.0e-6,
  link: 2.0e-6,
  bend: 1.5e-4,
  shape: 3.0e-4,
  contact: 4.0e-6,
} as const;

/** Compression is this much softer than stretch on a rope link. */
const SLACK_RATIO = 2;

export interface WireStiffness {
  /** Global inverse-stiffness scale, from params.springK. */
  scale: number;
  /** Extra compliance while a latch is young — a fresh joint reaches, an old one holds. */
  slack: number;
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
    nodes.push({ x, y, vx: 0, vy: 0, prevX: x, prevY: y });
  }
  return nodes;
}

export function reduceChain(nodes: ChainNode[], rest: number): void {
  while (nodes.length + 1 > MIN_LINKS && rest / (nodes.length + 1) < MIN_SEG) {
    nodes.splice(Math.floor(nodes.length / 2), 1);
  }
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

export function catmullSegment(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2): Cubic {
  const h1x = (p2.x - p0.x) / 6;
  const h1y = (p2.y - p0.y) / 6;
  const h2x = (p3.x - p1.x) / 6;
  const h2y = (p3.y - p1.y) / 6;
  const max1 = Math.hypot(p2.x - p1.x, p2.y - p1.y) * 0.45;
  const max2 = max1;
  const len1 = Math.hypot(h1x, h1y);
  const len2 = Math.hypot(h2x, h2y);
  const s1 = len1 > max1 && len1 > 1e-6 ? max1 / len1 : 1;
  const s2 = len2 > max2 && len2 > 1e-6 ? max2 / len2 : 1;
  return {
    p0: p1,
    p1: { x: p1.x + h1x * s1, y: p1.y + h1y * s1 },
    p2: { x: p2.x - h2x * s2, y: p2.y - h2y * s2 },
    p3: p2,
  };
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

// --------------------------------------------------------------- rigid bodies

const invMass = (agent: Agent): number => (poseHeld(agent) ? 0 : 1 / Math.max(0.08, agent.mass));
const invInertia = (agent: Agent): number =>
  poseHeld(agent) ? 0 : 1 / Math.max(1e-4, momentOfInertia(agent));

/** Offset from body centre to a port's stem root, at the body's current heading. */
export function attachOffset(agent: Agent, slot: PortSlot): Vec2 {
  const loc = stemRoot(agent.kind, slot);
  return rotate(loc.x * agent.scale, loc.y * agent.scale, agent.heading);
}

export function stemPoint(agent: Agent, slot: PortSlot): Vec2 {
  const r = attachOffset(agent, slot);
  return { x: agent.x + r.x, y: agent.y + r.y };
}

/** Unit vector the port points along, in world space. */
export function portAxisWorld(agent: Agent, slot: PortSlot): Vec2 {
  const tip = portLocal(agent.kind, slot);
  const root = stemRoot(agent.kind, slot);
  const r = rotate(
    (tip.x - root.x) * agent.scale,
    (tip.y - root.y) * agent.scale,
    agent.heading,
  );
  const len = Math.hypot(r.x, r.y) || 1;
  return { x: r.x / len, y: r.y / len };
}

/**
 * Generalized inverse mass of a body at attachment `r` along direction `n`:
 * `1/m + (r × n)² / I`. This is what makes a rope hanging off an off-centre
 * port torque the body instead of only dragging it.
 */
function genInvMass(agent: Agent, r: Vec2, nx: number, ny: number): number {
  const rxn = r.x * ny - r.y * nx;
  return invMass(agent) + invInertia(agent) * rxn * rxn;
}

/** Apply a positional impulse `lambda` along `n` at attachment `r`. */
function applyImpulse(agent: Agent, r: Vec2, nx: number, ny: number, lambda: number): void {
  if (poseHeld(agent) || lambda === 0) return;
  const im = invMass(agent);
  agent.x += im * lambda * nx;
  agent.y += im * lambda * ny;
  agent.heading = wrapAngle(
    agent.heading + invInertia(agent) * (r.x * ny - r.y * nx) * lambda,
  );
}

// --------------------------------------------------------------- constraints

/**
 * Anchors the end of a rope at a port stem. One-way: the stem moves the node,
 * never the reverse.
 *
 * Bodies drive the rope; the rope never drives the bodies. Spacing belongs to
 * the span joint and orientation to the port torques, so letting rope tension
 * also push its own anchors adds nothing but a feedback path — and it is the
 * path that made every earlier version blow up, because a rope's slack shape is
 * the least constrained thing in the system.
 */
function solveBodyNodeLink(
  agent: Agent,
  slot: PortSlot,
  node: ChainNode,
  rest: number,
  alphaTilde: number,
): void {
  const r = attachOffset(agent, slot);
  const dx = node.x - (agent.x + r.x);
  const dy = node.y - (agent.y + r.y);
  const dist = Math.hypot(dx, dy);
  if (dist < 1e-9) return;
  const C = dist - rest;
  const wNode = 1 / CHAIN_MASS;
  const denom = wNode + (C < 0 ? alphaTilde * SLACK_RATIO : alphaTilde);
  if (denom < 1e-12) return;
  const lambda = -C / denom;
  node.x += (wNode * lambda * dx) / dist;
  node.y += (wNode * lambda * dy) / dist;
}

/**
 * The joint proper: holds the two port stems `rest` apart. The rope alone
 * cannot do this — it only constrains arc length, so slack lets the bodies
 * drift together and the rope buckle back through its own port.
 */
function solveSpan(
  A: Agent,
  aSlot: PortSlot,
  B: Agent,
  bSlot: PortSlot,
  rest: number,
  alphaTilde: number,
): void {
  const rA = attachOffset(A, aSlot);
  const rB = attachOffset(B, bSlot);
  const dx = B.x + rB.x - (A.x + rA.x);
  const dy = B.y + rB.y - (A.y + rA.y);
  const dist = Math.hypot(dx, dy);
  if (dist < 1e-9 || !Number.isFinite(dist) || !Number.isFinite(rest)) return;
  const nx = dx / dist;
  const ny = dy / dist;
  const C = dist - rest;
  const denom = genInvMass(A, rA, nx, ny) + genInvMass(B, rB, nx, ny) + alphaTilde;
  if (denom < 1e-12) return;
  const lambda = -C / denom;
  applyImpulse(A, rA, -nx, -ny, lambda);
  applyImpulse(B, rB, nx, ny, lambda);
}

/** Soft pull of a rope node toward its place on the wire's rest curve. */
function solveShape(node: ChainNode, target: Vec2, alphaTilde: number): void {
  const dx = node.x - target.x;
  const dy = node.y - target.y;
  const C = Math.hypot(dx, dy);
  if (C < 1e-9) return;
  const w = 1 / CHAIN_MASS;
  const lambda = -C / (w + alphaTilde);
  node.x += (w * lambda * dx) / C;
  node.y += (w * lambda * dy) / C;
}

/** Distance constraint between two rope nodes. */
function solveNodeLink(a: ChainNode, b: ChainNode, rest: number, alphaTilde: number): void {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dist = Math.hypot(dx, dy);
  if (dist < 1e-9) return;
  const nx = dx / dist;
  const ny = dy / dist;
  const C = dist - rest;
  const w = 1 / CHAIN_MASS;
  const lambda = -C / (2 * w + (C < 0 ? alphaTilde * SLACK_RATIO : alphaTilde));
  a.x -= w * lambda * nx;
  a.y -= w * lambda * ny;
  b.x += w * lambda * nx;
  b.y += w * lambda * ny;
}

/** Straightening (Laplacian) constraint on a rope triple. Soft — this is the rope's give. */
function solveBend(a: Vec2, b: ChainNode, c: Vec2, wA: number, wC: number, alphaTilde: number): void {
  const Cx = a.x - 2 * b.x + c.x;
  const Cy = a.y - 2 * b.y + c.y;
  const wB = 1 / CHAIN_MASS;
  const denom = wA + 4 * wB + wC + alphaTilde;
  if (denom < 1e-12) return;
  b.x += 2 * wB * (Cx / denom);
  b.y += 2 * wB * (Cy / denom);
}

/**
 * One XPBD iteration over a single wire: rope links, bending, and the port-axis
 * preference at each end. Mutates agents and nodes in place so that wires
 * sharing an agent see each other's corrections within the same substep.
 */
export function solveWire(
  A: Agent,
  aSlot: PortSlot,
  B: Agent,
  bSlot: PortSlot,
  nodes: ChainNode[],
  rest: number,
  ropeLen: number,
  shape: Vec2[],
  stiff: WireStiffness,
  h: number,
): void {
  const n = nodes.length;
  if (n === 0) return;
  const invH2 = 1 / Math.max(1e-12, h * h);
  const soft = stiff.scale * stiff.slack;
  const aLink = COMPLIANCE.link * soft * invH2;
  const aBend = COMPLIANCE.bend * soft * invH2;
  const aShape = COMPLIANCE.shape * soft * invH2;
  const aSpan = COMPLIANCE.span * soft * invH2;

  if (!Number.isFinite(rest) || rest < 0) return;
  solveSpan(A, aSlot, B, bSlot, rest, aSpan);
  const linkRest = ropeLen / (n + 1);

  solveBodyNodeLink(A, aSlot, nodes[0], linkRest, aLink);
  for (let i = 0; i < n - 1; i++) solveNodeLink(nodes[i], nodes[i + 1], linkRest, aLink);
  solveBodyNodeLink(B, bSlot, nodes[n - 1], linkRest, aLink);

  const sA = stemPoint(A, aSlot);
  const sB = stemPoint(B, bSlot);
  for (let i = 0; i < n; i++) {
    const prev = i === 0 ? sA : nodes[i - 1];
    const next = i === n - 1 ? sB : nodes[i + 1];
    const wPrev = i === 0 ? 0 : 1 / CHAIN_MASS;
    const wNext = i === n - 1 ? 0 : 1 / CHAIN_MASS;
    solveBend(prev, nodes[i], next, wPrev, wNext, aBend);
  }

  if (shape.length === n) {
    for (let i = 0; i < n; i++) solveShape(nodes[i], shape[i], aShape);
  }
}

/** Joint only — no rope nodes. FAR wires keep their rest length this way. */
export function solveWireSpan(
  A: Agent,
  aSlot: PortSlot,
  B: Agent,
  bSlot: PortSlot,
  rest: number,
  stiff: WireStiffness,
  h: number,
): void {
  if (!Number.isFinite(rest) || rest < 0) return;
  const invH2 = 1 / Math.max(1e-12, h * h);
  solveSpan(A, aSlot, B, bSlot, rest, COMPLIANCE.span * stiff.scale * stiff.slack * invH2);
}

/**
 * The mechanics the audio needs from a contact, taken from the same quantities
 * the solver uses rather than re-guessed from positions.
 *
 * `effMass` is the generalized effective mass at the contact point along the
 * normal — it already includes the lever arm, so an off-centre hit on a
 * triangle correctly transfers less linear momentum and more spin than a
 * square one.
 */
export interface ContactMechanics {
  effMass: number;
  /** Closing speed along the normal. Positive means approaching. */
  vN: number;
  /** Signed sliding velocity along the tangent (A relative to B). */
  vT: number;
  /** Momentum that has to be turned around: effMass * vN. */
  impulse: number;
}

/** Velocity of the material point of `agent` currently at world (px, py). */
function pointVelocity(agent: Agent, px: number, py: number): Vec2 {
  const rx = px - agent.x;
  const ry = py - agent.y;
  return { x: agent.vx - agent.omega * ry, y: agent.vy + agent.omega * rx };
}

export function contactMechanics(
  A: Agent,
  B: Agent,
  hit: { nx: number; ny: number; px: number; py: number },
): ContactMechanics {
  const rA = { x: hit.px - A.x, y: hit.py - A.y };
  const rB = { x: hit.px - B.x, y: hit.py - B.y };
  const wA = genInvMass(A, rA, hit.nx, hit.ny);
  const wB = genInvMass(B, rB, hit.nx, hit.ny);
  const effMass = 1 / Math.max(1e-9, wA + wB);

  const pa = pointVelocity(A, hit.px, hit.py);
  const pb = pointVelocity(B, hit.px, hit.py);
  const rvx = pa.x - pb.x;
  const rvy = pa.y - pb.y;
  // Normal points from A toward B, so approaching means a negative projection.
  const vN = -(rvx * hit.nx + rvy * hit.ny);
  const vT = rvx * -hit.ny + rvy * hit.nx;
  return { effMass, vN, vT, impulse: effMass * Math.abs(vN) };
}

/** XPBD non-penetration between two bodies, from an existing contact manifold. */
export function solveContact(
  A: Agent,
  B: Agent,
  hit: { nx: number; ny: number; overlap: number; px: number; py: number },
  slop: number,
  h: number,
): void {
  const depth = hit.overlap - slop;
  if (depth <= 0) return;
  const rA = { x: hit.px - A.x, y: hit.py - A.y };
  const rB = { x: hit.px - B.x, y: hit.py - B.y };
  const wA = genInvMass(A, rA, hit.nx, hit.ny);
  const wB = genInvMass(B, rB, hit.nx, hit.ny);
  const denom = wA + wB + COMPLIANCE.contact / Math.max(1e-12, h * h);
  if (denom < 1e-12) return;
  const lambda = depth / denom;
  applyImpulse(A, rA, -hit.nx, -hit.ny, lambda);
  applyImpulse(B, rB, hit.nx, hit.ny, lambda);
}

/** Signed angle from a port's axis to the direction its wire actually leaves in. */
export function portExitAngle(agent: Agent, slot: PortSlot, target: Vec2): number {
  const u = portAxisWorld(agent, slot);
  const s = stemPoint(agent, slot);
  const dx = target.x - s.x;
  const dy = target.y - s.y;
  if (Math.hypot(dx, dy) < 1e-6) return 0;
  return Math.atan2(u.x * dy - u.y * dx, u.x * dx + u.y * dy);
}
