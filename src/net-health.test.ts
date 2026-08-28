import { afterEach, describe, expect, it } from 'vitest';
import { boundRadius, portAxis, stemRoot, stemWorld, type Agent } from './agents.ts';
import { queryHit } from './collide.ts';
import { segmentsIntersect } from './geom.ts';
import type { Wire } from './graph.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/**
 * Outcome metrics for the joint solver. These assert what a settled net should
 * *look like*, not which forces produced it — so they survive changes to the
 * solver internals. Thresholds describe the target, not today's behaviour.
 */

const realRandom = Math.random;
afterEach(() => {
  Math.random = realRandom;
});

function seed(n: number): void {
  let s = n >>> 0;
  Math.random = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Topology is frozen and the scent field is off; motors and flocking still run. */
function liveParams(): Params {
  const p = defaultParams();
  p.snapRadius = 0;
  p.snapWell = 0;
  p.rewriteDuration = 0;
  p.spawnInterval = 0;
  p.gravity = 0.12;
  return p;
}

/** Constraints and contacts only — no motors, flocking, gravity or scent. */
function passiveParams(): Params {
  const p = liveParams();
  p.stepSpeed = 0;
  p.faceAttract = 0;
  p.gravity = 0;
  p.homing = 0;
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
function portExitError(sim: Sim): number {
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
      worst = Math.max(worst, Math.acos(Math.max(-1, Math.min(1, cos))));
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

  it('lets a wire leave along its port axis when the ports allow it', () => {
    const sim = new Sim(480, 280);
    const params = passiveParams();
    const con = sim.spawn('con', 240, 140, 0, params, true)!;
    const era = sim.spawn('era', 200, 190, 0, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    run(sim, params, 600);
    const err = (portExitError(sim) * 180) / Math.PI;
    expect(err, `worst port exit error = ${err.toFixed(0)}°`).toBeLessThan(20);
  });

  it('finds a compromise when a junction cannot satisfy every port', () => {
    // A Con's two aux axes are antiparallel to its principal, so a fully wired
    // one cannot aim all three ports at once. The body settles where the port
    // torques cancel; ~45° of residual exit error is the geometric floor for
    // this glyph, not a solver failure. Splaying the aux axes would remove it.
    const { sim, params } = commuteNet();
    run(sim, params, 600);
    const err = (portExitError(sim) * 180) / Math.PI;
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

  it('keeps most aux wires uncrossed across a minute of soup', () => {
    // Parallel aux axes mean nothing geometrically forbids a crossing, so this
    // is a rate, not an invariant. Aiming each aux port slightly to its own side
    // roughly halves it; holding personal space between nets costs a little of
    // that back (18.4% to 21.7%) in exchange for nets not resting on each other.
    let obs = 0;
    let crossed = 0;
    for (const sd of [999, 12345, 5150]) {
      seed(sd);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      loadPreset(sim, 'soup', params);
      for (let f = 1; f <= 3600; f++) {
        sim.step(1 / 60, params);
        if (f % 30) continue;
        for (const wire of sim.graph.wires.values()) {
          for (const [near, nearSlot, far, farSlot] of [
            [wire.a, wire.a.slot, wire.b, wire.b.slot],
            [wire.b, wire.b.slot, wire.a, wire.a.slot],
          ] as const) {
            if (nearSlot === 'p') continue;
            const A = sim.agents.get(near.id);
            const B = sim.agents.get(far.id);
            if (!A || !B) continue;
            obs++;
            if (auxSideOffset(sim, A, nearSlot, B, farSlot) < 0) crossed++;
          }
        }
      }
    }
    const rate = crossed / Math.max(1, obs);
    expect(rate, `${(rate * 100).toFixed(1)}% of aux ends crossed (${crossed}/${obs})`).toBeLessThan(
      0.26,
    );
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

  it('keeps other nets out of a saturated agent\'s space', () => {
    // Flocking separation is scoped to one net, so nothing but this constraint
    // pushes separate nets apart — while homing pulls them together.
    let saturated = 0;
    let invaded = 0;
    for (const sd of [999, 12345, 5150]) {
      seed(sd);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      loadPreset(sim, 'soup', params);
      for (let f = 1; f <= 3600; f++) {
        sim.step(1 / 60, params);
        if (f % 60) continue;
        const comp = sim.graph.componentIds(sim.agents);
        const list = [...sim.agents.values()];
        for (const a of list) {
          if (!sim.graph.portsFilled(a)) continue;
          saturated++;
          for (const b of list) {
            if (b.id === a.id || comp.get(a.id) === comp.get(b.id)) continue;
            if (Math.hypot(b.x - a.x, b.y - a.y) < params.wireMinRest - 0.5) {
              invaded++;
              break;
            }
          }
        }
      }
    }
    const rate = invaded / Math.max(1, saturated);
    // 36.6% before this existed; nothing had ever separated two different nets.
    expect(rate, `${(rate * 100).toFixed(1)}% of saturated agents crowded by another net`)
      .toBeLessThan(0.06);
  });

  it('keeps ropes out of bodies they are not attached to', () => {
    let samples = 0;
    let inside = 0;
    for (const sd of [999, 12345, 5150]) {
      seed(sd);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      loadPreset(sim, 'soup', params);
      for (let f = 1; f <= 3600; f++) {
        sim.step(1 / 60, params);
        if (f % 60) continue;
        for (const wire of sim.graph.wires.values()) {
          for (const agent of sim.agents.values()) {
            if (agent.id === wire.a.id || agent.id === wire.b.id) continue;
            const r = boundRadius(agent);
            for (const node of wire.nodes) {
              samples++;
              if (Math.hypot(node.x - agent.x, node.y - agent.y) < r) inside++;
            }
          }
        }
      }
    }
    const rate = inside / Math.max(1, samples);
    // 0.32% of rope-node/body pairs overlapped before wire clearance existed.
    expect(rate, `${(rate * 100).toFixed(2)}% of rope nodes inside a foreign body`)
      .toBeLessThan(0.003);
  });

  it('leaves few wire crossings in a minute of soup', () => {
    let crossings = 0;
    let samples = 0;
    for (const sd of [999, 12345, 5150]) {
      seed(sd);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      loadPreset(sim, 'soup', params);
      for (let f = 1; f <= 3600; f++) {
        sim.step(1 / 60, params);
        if (f % 30) continue;
        samples++;
        const chords = [...sim.graph.wires.values()].flatMap((w) => {
          const A = sim.agents.get(w.a.id);
          const B = sim.agents.get(w.b.id);
          if (!A || !B) return [];
          const sa = stemWorld(A, w.a.slot, sim.w, sim.h);
          const sb = stemWorld(B, w.b.slot, sim.w, sim.h);
          return [{ a: w.a.id, b: w.b.id, sa, sb }];
        });
        for (let i = 0; i < chords.length; i++) {
          for (let j = i + 1; j < chords.length; j++) {
            const S = chords[i];
            const T = chords[j];
            if (S.a === T.a || S.a === T.b || S.b === T.a || S.b === T.b) continue;
            if (segmentsIntersect(S.sa.x, S.sa.y, S.sb.x, S.sb.y, T.sa.x, T.sa.y, T.sb.x, T.sb.y)) {
              crossings++;
            }
          }
        }
      }
    }
    const rate = crossings / Math.max(1, samples);
    // 2.33 per frame before saturated agents started holding their space.
    expect(rate, `${rate.toFixed(2)} wire crossings per frame`).toBeLessThan(2.1);
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

describe('the net actually rewrites', () => {
  it('fires rewrites in a minute of soup', () => {
    const counts: number[] = [];
    for (const s of [999, 12345, 5150]) {
      seed(s);
      const sim = new Sim(900, 600);
      const params = defaultParams();
      loadPreset(sim, 'soup', params);
      let started = 0;
      const inner = (sim as unknown as { startRewrites(p: Params): void }).startRewrites.bind(sim);
      (sim as unknown as { startRewrites(p: Params): void }).startRewrites = (p: Params) => {
        const before = sim.rewrites.length;
        inner(p);
        started += Math.max(0, sim.rewrites.length - before);
      };
      run(sim, params, 3600);
      counts.push(started);
    }
    const total = counts.reduce((a, b) => a + b, 0);
    expect(total, `rewrites per seed over 60 s: ${counts.join(', ')}`).toBeGreaterThanOrEqual(9);
  });
});
