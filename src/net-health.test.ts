import { describe, expect, it } from 'vitest';
import { portAxis, stemRoot, stemWorld, type Agent } from './agents.ts';
import { queryHit } from './collide.ts';
import type { Wire } from './graph.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/**
 * Outcome metrics for the joint solver. These assert what a settled net should
 * *look like*, not which forces produced it — so they survive changes to the
 * solver internals. Thresholds describe the target, not today's behaviour.
 */

/** Topology is frozen and the scent field is off; motors and flocking still run. */
function liveParams(): Params {
  const p = defaultParams();
  p.snapRadius = 0;
  p.snapWell = 0;
  p.rewriteDuration = 0;
  p.spawnInterval = 0;
  p.upkeep = 0;
  return p;
}

/** Constraints and contacts only — no motors, flocking or scent. */
function passiveParams(): Params {
  const p = liveParams();
  p.stepSpeed = 0;
  p.faceAttract = 0;
  p.flockAlign = 0;
  p.flockSep = 0;
  p.deposit = 0;
  p.sense = 0;
  p.wireBreathe = 0;
  return p;
}

function run(sim: Sim, params: Params, frames: number): void {
  for (let i = 0; i < frames; i++) sim.step(1 / 60, params);
}

// ---------------------------------------------------------------- metrics

function spans(sim: Sim): { wire: Wire; span: number; err: number }[] {
  const out: { wire: Wire; span: number; err: number }[] = [];
  for (const wire of sim.graph.wires.values()) {
    const A = sim.agents.get(wire.a.id);
    const B = sim.agents.get(wire.b.id);
    if (!A || !B) continue;
    const sa = stemWorld(A, wire.a.slot, sim.w, sim.h);
    const sb = stemWorld(B, wire.b.slot, sim.w, sim.h);
    const span = Math.hypot(sb.x - sa.x, sb.y - sa.y);
    out.push({ wire, span, err: span - wire.rest });
  }
  return out;
}

function maxSpanError(sim: Sim): number {
  return spans(sim).reduce((m, s) => Math.max(m, Math.abs(s.err)), 0);
}

/** Coefficient of variation of stem-to-stem span across all wires. */
function edgeCv(sim: Sim): number {
  const d = spans(sim).map((s) => s.span);
  if (d.length < 2) return 0;
  const mean = d.reduce((a, b) => a + b, 0) / d.length;
  const varr = d.reduce((a, b) => a + (b - mean) ** 2, 0) / d.length;
  return Math.sqrt(varr) / Math.max(1e-6, mean);
}

function diameter(sim: Sim): number {
  const list = [...sim.agents.values()];
  let d = 0;
  for (let i = 0; i < list.length; i++) {
    for (let j = i + 1; j < list.length; j++) {
      d = Math.max(d, Math.hypot(list[j].x - list[i].x, list[j].y - list[i].y));
    }
  }
  return d;
}

function penetration(sim: Sim): number {
  const list = [...sim.agents.values()];
  let worst = 0;
  for (let i = 0; i < list.length; i++) {
    for (let j = i + 1; j < list.length; j++) {
      const hit = queryHit(list[i], list[j], sim.w, sim.h);
      if (hit) worst = Math.max(worst, hit.overlap);
    }
  }
  return worst;
}

function maxSpin(sim: Sim): number {
  let m = 0;
  for (const a of sim.agents.values()) m = Math.max(m, Math.abs(a.omega));
  return m;
}

/**
 * Angle between a port's axis and the direction the wire actually leaves in.
 * This is what makes a drawing read as a Lafont figure: the wire departs along
 * the port, then curves freely. Measured against the wire's first rope node.
 */
/**
 * Worst departure from the angle the port torque is actually aiming for.
 *
 * Not the bare port axis: an aux port is deliberately aimed `auxSpread * 0.35`
 * off its own axis, toward its own side of the body, so that the uncrossed
 * pose is the stable one. Measuring against the axis therefore reads a
 * correctly splayed net as 34 degrees of error at the default spread, which
 * is the setpoint, not a miss.
 */
function portExitError(sim: Sim, params: Params): number {
  const splay = params.auxSpread * 0.35;
  let worst = 0;
  for (const wire of sim.graph.wires.values()) {
    const A = sim.agents.get(wire.a.id);
    const B = sim.agents.get(wire.b.id);
    if (!A || !B) continue;
    const ends: [Agent, 'p' | 'l' | 'r', { x: number; y: number }][] = [
      [A, wire.a.slot, wire.nodes[0] ?? stemWorld(B, wire.b.slot, sim.w, sim.h)],
      [B, wire.b.slot, wire.nodes[wire.nodes.length - 1] ?? stemWorld(A, wire.a.slot, sim.w, sim.h)],
    ];
    for (const [agent, slot, target] of ends) {
      const s = stemWorld(agent, slot, sim.w, sim.h);
      const dx = target.x - s.x;
      const dy = target.y - s.y;
      if (Math.hypot(dx, dy) < 1e-6) continue;
      const ax = portAxis(agent, slot);
      const cos = (ax.x * dx + ax.y * dy) / Math.hypot(dx, dy);
      const off = Math.acos(Math.max(-1, Math.min(1, cos)));
      const want = slot === 'p' ? 0 : splay;
      worst = Math.max(worst, Math.abs(off - want));
    }
  }
  return worst;
}

/** A six-agent net: two junction agents facing each other, four aux leaves. */
function commuteNet(): { sim: Sim; params: Params } {
  const sim = new Sim(480, 280);
  const params = passiveParams();
  loadPreset(sim, 'commute', params);
  return { sim, params };
}

// ---------------------------------------------------------------- tests

describe('settled net geometry', () => {
  it('holds every wire near its rest length', () => {
    const { sim, params } = commuteNet();
    run(sim, params, 600);
    const err = maxSpanError(sim);
    const detail = spans(sim)
      .map((s) => `${s.span.toFixed(1)}/${s.wire.rest.toFixed(1)}`)
      .join(' ');
    expect(err, `max |span-rest| = ${err.toFixed(2)} (span/rest: ${detail})`).toBeLessThan(4);
  });

  it('keeps edge lengths uniform', () => {
    const { sim, params } = commuteNet();
    run(sim, params, 600);
    const cv = edgeCv(sim);
    expect(cv, `edge length CV = ${cv.toFixed(3)}`).toBeLessThan(0.15);
  });

  it('stays compact instead of dispersing', () => {
    const { sim, params } = commuteNet();
    run(sim, params, 900);
    const d = diameter(sim);
    const cap = params.wireMinRest * 8;
    expect(d, `net diameter = ${d.toFixed(0)} px, cap ${cap}`).toBeLessThan(cap);
  });

  it('leaves no bodies interpenetrating', () => {
    const { sim, params } = commuteNet();
    run(sim, params, 600);
    const pen = penetration(sim);
    expect(pen, `worst penetration = ${pen.toFixed(2)} px`).toBeLessThan(1.5);
  });

  it('lets a wire leave a port at the angle the torque is aiming for', () => {
    const sim = new Sim(480, 280);
    const params = passiveParams();
    const con = sim.spawn('con', 240, 140, 0, params, true)!;
    const era = sim.spawn('era', 200, 190, 0, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    run(sim, params, 600);
    const err = (portExitError(sim, params) * 180) / Math.PI;
    expect(err, `worst port exit error = ${err.toFixed(0)}°`).toBeLessThan(10);
  });

  it('finds a compromise when a junction cannot satisfy every port', () => {
    // A Con's two aux axes are antiparallel to its principal, so a fully wired
    // one cannot aim all three ports at once. The body settles where the port
    // torques cancel; ~45° of residual exit error is the geometric floor for
    // this glyph, not a solver failure. Splaying the aux axes would remove it.
    const { sim, params } = commuteNet();
    run(sim, params, 600);
    const err = (portExitError(sim, params) * 180) / Math.PI;
    expect(err, `worst port exit error = ${err.toFixed(0)}°`).toBeLessThan(55);
  });
});

describe('settled net dynamics', () => {
  it('does not gain energy once it has settled', () => {
    const { sim, params } = commuteNet();
    run(sim, params, 300);
    const early = sim.kineticEnergy();
    run(sim, params, 600);
    const late = sim.kineticEnergy();
    expect(late, `KE ${early.toFixed(1)} → ${late.toFixed(1)}`).toBeLessThan(early + 1);
  });

  it('does not windmill', () => {
    const { sim, params } = commuteNet();
    run(sim, params, 600);
    const spin = maxSpin(sim);
    expect(spin, `max |omega| = ${spin.toFixed(2)} rad/s`).toBeLessThan(1.5);
  });

  it('stays calm with motors, flocking and gravity live', () => {
    const sim = new Sim(480, 280);
    const params = liveParams();
    loadPreset(sim, 'commute', params);
    run(sim, params, 900);
    const err = maxSpanError(sim);
    const d = diameter(sim);
    expect(err, `max |span-rest| = ${err.toFixed(2)}`).toBeLessThan(10);
    expect(d, `net diameter = ${d.toFixed(0)} px`).toBeLessThan(params.wireMinRest * 10);
  });

  it('recovers when a net is dropped in badly tangled', () => {
    const sim = new Sim(480, 280);
    const params = passiveParams();
    loadPreset(sim, 'oscillator', params);
    run(sim, params, 900);
    const err = maxSpanError(sim);
    const pen = penetration(sim);
    expect(err, `max |span-rest| = ${err.toFixed(2)}`).toBeLessThan(6);
    expect(pen, `worst penetration = ${pen.toFixed(2)} px`).toBeLessThan(1.5);
  });
});

/**
 * Signed distance of a wire's far stem from the near agent's centreline,
 * measured toward that port's own side and scaled by the port's own offset.
 * Positive means the wire stays on its own side; negative means it has crossed.
 */
function auxSideOffset(
  sim: Sim,
  agent: Agent,
  slot: 'l' | 'r',
  other: Agent,
  otherSlot: 'p' | 'l' | 'r',
): number {
  const root = stemRoot(agent.kind, slot);
  const side = root.y < 0 ? -1 : 1;
  const mx = -side * Math.sin(agent.heading);
  const my = side * Math.cos(agent.heading);
  const far = stemWorld(other, otherSlot, sim.w, sim.h);
  return ((far.x - agent.x) * mx + (far.y - agent.y) * my) / Math.abs(root.y);
}

describe('aux wires keep to their own side', () => {
  it('splays a constructor\'s two aux neighbours to opposite sides', () => {
    const sim = new Sim(480, 280);
    const params = passiveParams();
    const con = sim.spawn('con', 240, 140, 0, params, true)!;
    const left = sim.spawn('era', 200, 128, 0, params, true)!;
    const right = sim.spawn('era', 200, 152, 0, params, true)!;
    sim.wire(con.id, 'l', left.id, 'p', params);
    sim.wire(con.id, 'r', right.id, 'p', params);
    run(sim, params, 600);
    const l = auxSideOffset(sim, con, 'l', left, 'p');
    const r = auxSideOffset(sim, con, 'r', right, 'p');
    expect(l, `left neighbour side offset = ${l.toFixed(2)} port-widths`).toBeGreaterThan(0.5);
    expect(r, `right neighbour side offset = ${r.toFixed(2)} port-widths`).toBeGreaterThan(0.5);
  });
});

describe('crowding and tangling', () => {
  it('pushes a stranger out of a saturated agent\'s space', () => {
    const sim = new Sim(480, 280);
    const params = passiveParams();
    const con = sim.spawn('con', 240, 140, 0, params, true)!;
    const p1 = sim.spawn('era', 290, 140, Math.PI, params, true)!;
    const l1 = sim.spawn('era', 200, 120, 0, params, true)!;
    const r1 = sim.spawn('era', 200, 160, 0, params, true)!;
    sim.wire(con.id, 'p', p1.id, 'p', params);
    sim.wire(con.id, 'l', l1.id, 'p', params);
    sim.wire(con.id, 'r', r1.id, 'p', params);
    expect(sim.graph.portsFilled(con)).toBe(true);
    // A loose agent parked right on top of it.
    const stranger = sim.spawn('era', 252, 146, 0, params, true)!;
    run(sim, params, 600);
    const gap = Math.hypot(stranger.x - con.x, stranger.y - con.y);
    // Contact alone would settle around the two bound radii, ~27 px.
    expect(gap, `stranger sits ${gap.toFixed(1)} px away`).toBeGreaterThan(34);
  });
});

describe('topology', () => {
  it('lets two agents be joined by more than one port', () => {
    const sim = new Sim(480, 280);
    const params = defaultParams();
    params.spawnInterval = 0;
    params.rewriteDuration = 0;
    // Back to back, so both pairs of aux ports face each other.
    const a = sim.spawn('con', 220, 140, Math.PI, params, true)!;
    const b = sim.spawn('con', 262, 140, 0, params, true)!;
    run(sim, params, 300);
    const between = [...sim.graph.wires.values()].filter(
      (w) =>
        (w.a.id === a.id && w.b.id === b.id) || (w.a.id === b.id && w.b.id === a.id),
    );
    expect(between.length, `wires between the pair: ${between.length}`).toBe(2);
    const slots = between
      .flatMap((w) => [`${w.a.id}.${w.a.slot}`, `${w.b.id}.${w.b.slot}`])
      .sort();
    expect(new Set(slots).size, `each end on its own port: ${slots.join(' ')}`).toBe(4);
  });
});
