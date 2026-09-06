import { portWorld, stemWorld, type Agent, type AgentKind, type PortSlot } from './agents.ts';
import type { Params } from './params.ts';
import type { Sim } from './sim.ts';

/** One click of the demo splat tool. */
export const SPLATTER_COUNT = 50;
/** Disk radius, world units — denser than soup, still a handful not a pile. */
export const SPLATTER_RADIUS = 240;

function soupKind(): AgentKind {
  const roll = Math.random();
  return roll < 0.34 ? 'era' : roll < 0.67 ? 'dup' : 'con';
}

export type PresetName = 'soup' | 'commute' | 'annihilate-con' | 'annihilate-dup' | 'oscillator';

/**
 * `n` points in a disk, on a jittered hexagonal lattice in a shuffled order.
 *
 * Uniform random placement put a tenth of a ten-thousand-body soup inside
 * another body's radius, and the disc contact resolved the pile at speeds in
 * the thousands for the first two seconds — the "flung apart" that showed up
 * in every early census and had nothing to do with the solver. A lattice at
 * the spacing the area affords, jittered by a fifth of it, never overlaps
 * until the dish is genuinely fuller than its bodies can be.
 *
 * Shuffled, because ids are handed out in this order and a spatially sorted
 * roster would make every id-ordered tie-break — harvest, latch — a
 * left-to-right sweep across the dish. Falls back to random points for any
 * shortfall at the rim.
 */
function latticeInDisk(cx: number, cy: number, r: number, n: number): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  if (n <= 0) return pts;
  // Hex cells of side `s` tile at (sqrt(3)/2) s^2 each; aim for 15% more
  // cells than bodies so the rim's partial cells still leave enough.
  const s = Math.sqrt((Math.PI * r * r) / (0.8660254 * n * 1.15));
  const rowH = s * 0.8660254;
  const jitter = s * 0.2;
  const inside = r * r;
  for (let row = 0, y = -r; y <= r; row++, y += rowH) {
    const x0 = row % 2 === 0 ? 0 : s * 0.5;
    for (let x = -r + x0; x <= r; x += s) {
      const px = x + (Math.random() * 2 - 1) * jitter;
      const py = y + (Math.random() * 2 - 1) * jitter;
      if (px * px + py * py > inside) continue;
      pts.push({ x: cx + px, y: cy + py });
    }
  }
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = pts[i];
    pts[i] = pts[j];
    pts[j] = t;
  }
  while (pts.length < n) {
    const theta = Math.random() * Math.PI * 2;
    const rr = r * Math.sqrt(Math.random());
    pts.push({ x: cx + Math.cos(theta) * rr, y: cy + Math.sin(theta) * rr });
  }
  pts.length = n;
  return pts;
}

function plugAux(sim: Sim, agent: Agent, slot: PortSlot, params: Params): void {
  const root = stemWorld(agent, slot, sim.w, sim.h);
  const tip = portWorld(agent, slot, sim.w, sim.h);
  const dx = tip.x - root.x;
  const dy = tip.y - root.y;
  const era = sim.spawn(
    'era',
    tip.x + dx * 1.2,
    tip.y + dy * 1.2,
    Math.atan2(-dy, -dx),
    params,
    true,
  );
  if (era) sim.wire(agent.id, slot, era.id, 'p', params);
}

export function loadPreset(sim: Sim, name: PresetName, params: Params): void {
  sim.clear();
  const cx = sim.w * 0.5;
  const cy = sim.h * 0.5;
  if (name === 'soup') {
    sim.pinWorld(cx, cy, params);
    const n = Math.min(params.soupCount, params.maxAgents);
    const radius = Math.max(0, sim.worldR - 24);
    const pts = latticeInDisk(cx, cy, radius, n);
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      sim.spawn(soupKind(), p.x, p.y, Math.random() * Math.PI * 2, params, true);
    }
    return;
  }
  if (name === 'commute') {
    const c = sim.spawn('con', cx - 34, cy, 0, params, true);
    const d = sim.spawn('dup', cx + 34, cy, Math.PI, params, true);
    if (!c || !d) return;
    sim.wire(c.id, 'p', d.id, 'p', params);
    c.extra = 1;
    d.extra = 1;
    plugAux(sim, c, 'l', params);
    plugAux(sim, c, 'r', params);
    plugAux(sim, d, 'l', params);
    plugAux(sim, d, 'r', params);
    return;
  }
  if (name === 'annihilate-con' || name === 'annihilate-dup') {
    const kind = name === 'annihilate-con' ? 'con' : 'dup';
    const a = sim.spawn(kind, cx - 34, cy, 0, params, true);
    const b = sim.spawn(kind, cx + 34, cy, Math.PI, params, true);
    if (!a || !b) return;
    sim.wire(a.id, 'p', b.id, 'p', params);
    plugAux(sim, a, 'l', params);
    plugAux(sim, a, 'r', params);
    plugAux(sim, b, 'l', params);
    plugAux(sim, b, 'r', params);
    return;
  }

  const dup = sim.spawn('dup', cx - 40, cy, 0, params, true);
  const con = sim.spawn('con', cx + 40, cy, Math.PI, params, true);
  const era1 = sim.spawn('era', cx - 70, cy - 28, Math.PI, params, true);
  const era2 = sim.spawn('era', cx + 70, cy - 28, 0, params, true);
  if (!dup || !con || !era1 || !era2) return;
  sim.wire(dup.id, 'p', con.id, 'p', params);
  sim.wire(dup.id, 'l', era1.id, 'p', params);
  sim.wire(con.id, 'r', era2.id, 'p', params);
  sim.wire(dup.id, 'r', con.id, 'l', params);
  dup.extra = 1;
  con.extra = 1;
}

/**
 * Drop up to `n` free agents in a disk around `(x, y)`. Skips points outside
 * the world bound and stops at `maxAgents`. Returns how many actually landed.
 */
export function splatter(
  sim: Sim,
  x: number,
  y: number,
  params: Params,
  n = SPLATTER_COUNT,
): number {
  const want = Math.min(Math.max(0, n), Math.max(0, params.maxAgents - sim.agents.size));
  let spawned = 0;
  let attempts = 0;
  const attemptCap = Math.max(want * 8, want);
  while (spawned < want && attempts < attemptCap) {
    attempts++;
    const theta = Math.random() * Math.PI * 2;
    const r = SPLATTER_RADIUS * Math.sqrt(Math.random());
    const px = x + Math.cos(theta) * r;
    const py = y + Math.sin(theta) * r;
    if (!sim.energy.inBounds(px, py)) continue;
    if (!sim.spawn(soupKind(), px, py, Math.random() * Math.PI * 2, params, true)) continue;
    spawned++;
  }
  return spawned;
}
