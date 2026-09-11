import { describe, expect, it } from 'vitest';
import { stemRoot, stemWorld, type Agent } from './agents.ts';
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

/**
 * Angle between a port's axis and the direction the wire actually leaves in.
 * This is what makes a drawing read as a Lafont figure: the wire departs along
 * the port, then curves freely. Measured against the wire's first rope node.
 */
/** A six-agent net: two junction agents facing each other, four aux leaves. */
function commuteNet(): { sim: Sim; params: Params } {
  const sim = new Sim(480, 280);
  const params = passiveParams();
  loadPreset(sim, 'commute', params);
  return { sim, params };
}

// ---------------------------------------------------------------- tests

describe('settled net dynamics', () => {
  it('does not gain energy once it has settled', () => {
    const { sim, params } = commuteNet();
    run(sim, params, 300);
    const early = sim.kineticEnergy();
    run(sim, params, 600);
    const late = sim.kineticEnergy();
    expect(late, `KE ${early.toFixed(1)} → ${late.toFixed(1)}`).toBeLessThan(early + 1);
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
