import { describe, expect, it } from 'vitest';
import { stemWorld } from './agents.ts';
import { chordDeviation } from './chain.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';
import { angleDelta } from './wrap.ts';

function quietOrganizeParams() {
  const params = defaultParams();
  params.gravity = 0;
  params.homing = 0;
  params.flockAlign = 0;
  params.flockSep = 0;
  params.snapRadius = 0;
  params.stepSpeed = 0;
  params.faceAttract = 0;
  params.deposit = 0;
  params.diffuse = 0;
  params.decay = 0;
  params.spawnInterval = 0;
  params.rewriteDuration = 20;
  params.wireShrink = 20;
  return params;
}

function step(sim: Sim, params: ReturnType<typeof quietOrganizeParams>, n: number): void {
  for (let i = 0; i < n; i++) sim.step(1 / 60, params);
}

describe('wire organize', () => {
  it('pulls a misaligned era pair into a straight meridian', () => {
    const sim = new Sim(400, 240);
    const params = quietOrganizeParams();
    const a = sim.spawn('era', 110, 90, 0.55, params, true)!;
    const b = sim.spawn('era', 250, 150, -0.4, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 110);
    expect(Math.abs(Math.abs(angleDelta(a.heading, b.heading)) - Math.PI)).toBeLessThan(0.35);
    const sa = stemWorld(a, 'p', sim.w, sim.h);
    const sb = stemWorld(b, 'p', sim.w, sim.h);
    const axis = Math.atan2(sb.y - sa.y, sb.x - sa.x);
    expect(Math.abs(angleDelta(a.heading, axis))).toBeLessThan(0.4);
    expect(Math.abs(angleDelta(b.heading, axis + Math.PI))).toBeLessThan(0.4);
  });

  it('straightens a sagging wire chain toward the chord', () => {
    const sim = new Sim(400, 240);
    const params = quietOrganizeParams();
    const a = sim.spawn('era', 100, 120, 0, params, true)!;
    const b = sim.spawn('era', 280, 120, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    for (const node of wire.nodes) {
      node.y += node.x < 200 ? 35 : -28;
    }
    const pts0 = [
      stemWorld(a, 'p', sim.w, sim.h),
      ...wire.nodes,
      stemWorld(b, 'p', sim.w, sim.h),
    ];
    const dev0 = chordDeviation(pts0, sim.w, sim.h);
    step(sim, params, 90);
    const pts1 = [
      stemWorld(a, 'p', sim.w, sim.h),
      ...wire.nodes,
      stemWorld(b, 'p', sim.w, sim.h),
    ];
    const dev1 = chordDeviation(pts1, sim.w, sim.h);
    expect(dev1).toBeLessThan(dev0 * 0.55);
    expect(dev1).toBeLessThan(14);
  });

  // Removed: 'seats a three-port constructor toward its neighbor centroid'.
  // It asserted that organizeJunctions() existed — a positional nudge toward
  // the mean of a junction's neighbours. A junction's resting place is now
  // wherever its span joints and port torques balance, which for Lafont's
  // parallel aux axes is not the centroid. The outcomes that test was reaching
  // for (uniform edge length, compact nets, sane junction angles) are asserted
  // directly in net-health.test.ts.
});
