import {
  boundRadius,
  ERA_RADIUS,
  portLocal,
  triangleLocal,
  type Agent,
  type AgentKind,
  type PortSlot,
} from '../agents.ts';

/**
 * Excitation and termination character derived from an agent's actual body,
 * not from a lookup keyed on its kind.
 *
 * The agent is the mallet and the wire is the resonator, so geometry here sets
 * *how* a wire is struck, never what pitch it sounds. A sharp vertex is a
 * concentrated contact and makes a narrow bright impulse; a flat face spreads
 * the contact and makes a wide soft one; a heavy body is a more rigid
 * termination and lets the wire ring longer.
 *
 * Worth knowing: Dup and Con are the same triangle with the same ports. Driven
 * by geometry alone they are the same instrument. What separates them is the
 * one thing that *is* different on screen — Dup is drawn dark and Con light —
 * so albedo drives brightness, and the picture and the sound agree.
 */
export interface Voice {
  /** Seconds to -60 dB. */
  t60: number;
  /** One-pole damping per traversal: lower is darker. */
  damp: number;
  /** Allpass dispersion. Stretches partials sharp, which reads as bar or bell. */
  disp: number;
  /** Excitation footprint along the wire: <1 is a mallet, 1 is a full pluck. */
  width: number;
  /** Excitation position along the wire. */
  at: number;
  level: number;
}

/** How light the body is drawn, 0..1. Matches the fills in render.ts. */
export function albedo(kind: AgentKind): number {
  if (kind === 'era') return 0.95; // #f3f3f3
  if (kind === 'con') return 0.96; // #f4f4f4
  return 0.07; // #111213
}

/** Interior angle at the polygon vertex a port sits on, in radians. */
export function featureAngle(kind: AgentKind, slot: PortSlot): number {
  if (kind === 'era') return Math.PI;
  const tri = triangleLocal(1);
  // Ports sit at apex (index 0) and the two base corners.
  const i = slot === 'p' ? 0 : slot === 'l' ? 1 : 2;
  const v = tri[i];
  const a = tri[(i + 1) % 3];
  const b = tri[(i + 2) % 3];
  const ax = a.x - v.x;
  const ay = a.y - v.y;
  const bx = b.x - v.x;
  const by = b.y - v.y;
  const dot = ax * bx + ay * by;
  const mag = Math.hypot(ax, ay) * Math.hypot(bx, by);
  return Math.acos(Math.max(-1, Math.min(1, dot / Math.max(1e-6, mag))));
}

/**
 * Radius of curvature at the body feature a port sits on, in world pixels.
 *
 * This, not the corner angle, is what sets how concentrated a contact is: a
 * small circle is a *tight* contact even though it has no corner at all, while
 * a wide-open corner on a big body is comparatively blunt. Hertzian contact
 * width goes as the square root of this, which is what `width` uses.
 */
export function featureRadius(agent: Agent, slot: PortSlot): number {
  if (agent.kind === 'era') return ERA_RADIUS * agent.scale;
  // A polygon corner is a singularity; blunt it by the half-angle so the apex
  // reads sharper than the base corners rather than both being infinite.
  return 1.2 * Math.tan(featureAngle(agent.kind, slot) * 0.5) * agent.scale;
}

/** 0 = flat face, 1 = needle point. Handy for reasoning about the geometry. */
export function sharpness(kind: AgentKind, slot: PortSlot): number {
  return Math.max(0, Math.min(1, 1 - featureAngle(kind, slot) / Math.PI));
}

/** How far the port tip reaches past the body, as a multiple of its radius. */
export function reach(agent: Agent, slot: PortSlot): number {
  const tip = portLocal(agent.kind, slot);
  return Math.hypot(tip.x, tip.y) / Math.max(1e-6, boundRadius(agent) / agent.scale);
}

/**
 * The voice one agent brings to a wire it holds by `slot`.
 *
 * `at` is deliberately kept near an end: the terminations do not invert, so the
 * modes are cosines with antinodes at the ends and a midpoint strike would
 * cancel the fundamental outright.
 */
export function voiceFromAgent(agent: Agent, slot: PortSlot): Voice {
  const size = boundRadius(agent);
  const light = albedo(agent.kind);

  // Contact width goes as the square root of the curvature radius, scaled by
  // how big the striking body is. A triangle vertex is a click; the circle is
  // rounder and softer; a big body spreads the contact further still.
  const r = featureRadius(agent, slot);
  const width = Math.max(0.06, Math.min(1, 0.06 + 0.55 * Math.sqrt(r / 10) * (size / 14)));
  // A long stem lands further along the wire, but never near the midpoint,
  // which would cancel the fundamental outright.
  const at = Math.max(0.07, Math.min(0.26, 0.07 + (reach(agent, slot) - 1) * 0.42));
  // What you see is what you hear: the dark body is the dark voice.
  const damp = 0.34 + light * 0.42;
  // A heavier body is a stiffer termination, so the wire holds its energy.
  const t60 = Math.max(0.5, Math.min(9, 0.9 + agent.mass * 1.7));
  // Bigger bodies ring with more inharmonicity, like a plate rather than a bar.
  const disp = Math.min(0.3, 0.02 + (size / 18 - 1) * 0.18);
  const level = Math.min(1.2, 0.6 + size / 34);

  return { t60, damp, disp: Math.max(0, disp), width, at, level };
}

/** A wire belongs to both agents that hold it, so its voice is the blend. */
export function blendVoices(a: Voice, b: Voice): Voice {
  return {
    t60: Math.sqrt(a.t60 * b.t60),
    damp: (a.damp + b.damp) * 0.5,
    disp: (a.disp + b.disp) * 0.5,
    width: (a.width + b.width) * 0.5,
    at: (a.at + b.at) * 0.5,
    level: (a.level + b.level) * 0.5,
  };
}

/**
 * How concentrated a contact is when this body is hit from world direction
 * (nx, ny). A circle answers the same from every side; a triangle struck on a
 * vertex is a click and struck on a face is a thud, so the same two agents
 * colliding twice do not sound the same twice.
 */
export function strikeSharpness(
  agent: { kind: AgentKind; heading: number },
  nx: number,
  ny: number,
): number {
  if (agent.kind === 'era') return 0.5;
  // Contact normal in the body's own frame.
  const c = Math.cos(-agent.heading);
  const s = Math.sin(-agent.heading);
  const lx = nx * c - ny * s;
  const ly = nx * s + ny * c;
  const theta = Math.atan2(ly, lx);
  let best = -1;
  for (const v of triangleLocal(1)) {
    const d = Math.abs(wrapPi(theta - Math.atan2(v.y, v.x)));
    // 1 when the normal points straight at a vertex, 0 when it points at a face.
    best = Math.max(best, Math.max(0, 1 - d / (Math.PI * 0.5)));
  }
  return Math.max(0, Math.min(1, best));
}

function wrapPi(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}
