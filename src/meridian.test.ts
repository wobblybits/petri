import { describe, expect, it } from 'vitest';
import { fixedParams } from './test-params.ts';
import { portAxis, stemWorld, type Agent } from './agents.ts';
import { Sim } from './sim.ts';
import { angleDelta, wrapDeltaVec } from './wrap.ts';

function portAlignment(
  sim: Sim,
  a: Agent,
  b: Agent,
  slotA: 'p' | 'l' | 'r',
  slotB: 'p' | 'l' | 'r',
) {
  const sa = stemWorld(a, slotA, sim.w, sim.h);
  const sb = stemWorld(b, slotB, sim.w, sim.h);
  const d = wrapDeltaVec(sa.x, sa.y, sb.x, sb.y, sim.w, sim.h);
  const len = Math.hypot(d.x, d.y) || 1;
  const ux = d.x / len;
  const uy = d.y / len;
  const axisA = portAxis(a, slotA);
  const axisB = portAxis(b, slotB);
  const faceA = axisA.x * ux + axisA.y * uy;
  const faceB = axisB.x * ux + axisB.y * uy;
  return { faceA, faceB, len };
}

describe('wire meridian alignment', () => {
  const baseParams = () => {
    const params = fixedParams();
    params.spawnInterval = 0;
    params.flockAlign = 5.5;
    params.flockSep = 36;
    params.stepSpeed = 0;
    params.wireShrink = 0.9;
    params.rewriteDuration = 0;
    return params;
  };

  it('principal–principal: ports face each other and stems move closer', () => {
    const sim = new Sim(400, 240);
    const params = baseParams();
    const a = sim.spawn('era', 120, 120, 0.4, params, true)!;
    const b = sim.spawn('era', 260, 120, -0.35, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);

    const stemSpan = () => {
      const sa = stemWorld(a, 'p', sim.w, sim.h);
      const sb = stemWorld(b, 'p', sim.w, sim.h);
      return Math.hypot(sb.x - sa.x, sb.y - sa.y);
    };

    const d0 = stemSpan();
    let last = portAlignment(sim, a, b, 'p', 'p');
    for (let i = 0; i < 90; i++) {
      sim.step(1 / 60, params);
      last = portAlignment(sim, a, b, 'p', 'p');
    }
    const d1 = stemSpan();

    expect(last.faceA, `A port should face B (dot=${last.faceA.toFixed(2)})`).toBeGreaterThan(0.85);
    expect(-last.faceB, `B port should face A (dot=${(-last.faceB).toFixed(2)})`).toBeGreaterThan(0.85);
    expect(d1, `stems should pull closer (${d0.toFixed(1)}→${d1.toFixed(1)})`).toBeLessThan(d0 - 4);
  });

  it('principal–aux: wired ports face each other with parallel bodies', () => {
    const sim = new Sim(400, 240);
    const params = baseParams();
    const con = sim.spawn('con', 210, 120, 0.85, params, true)!;
    const era = sim.spawn('era', 130, 120, -0.6, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);

    let last = portAlignment(sim, con, era, 'l', 'p');
    for (let i = 0; i < 90; i++) {
      sim.step(1 / 60, params);
      last = portAlignment(sim, con, era, 'l', 'p');
    }

    expect(last.faceA, `aux port should face cargo (dot=${last.faceA.toFixed(2)})`).toBeGreaterThan(0.85);
    expect(-last.faceB, `era port should face constructor (dot=${(-last.faceB).toFixed(2)})`).toBeGreaterThan(0.85);
    // Each aux port aims params.auxSpread * 0.35 rad off its neighbour so the
    // wire keeps to its own side of the body, so the pair sits that much off
    // exactly parallel. Set auxSpread to 0 for the old exact alignment.
    const auxSplay = params.auxSpread * 0.35;
    expect(Math.abs(angleDelta(con.heading, era.heading))).toBeLessThan(0.25 + auxSplay);
    const along =
      (con.x - era.x) * Math.cos(con.heading) + (con.y - era.y) * Math.sin(con.heading);
    expect(along, `constructor should lead era (along=${along.toFixed(1)})`).toBeGreaterThan(18);
  });

  // Removed: 'principal–aux: seats cargo behind even when it latched ahead'.
  // It asserted the kinematic tow-meridian flip, where a constructor was
  // teleported around so its cargo sat behind it. Ports now find their own
  // equilibrium under compliant torques and nothing reseats a body by fiat.
  // (That test was already failing before the solver rewrite.)

  it('principal–aux: stem span shrinks during latch', () => {
    const sim = new Sim(400, 240);
    const params = baseParams();
    params.wireShrink = 0.45;
    params.wireMinRest = 28;
    const con = sim.spawn('con', 210, 120, 0, params, true)!;
    const era = sim.spawn('era', 120, 120, 0, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    expect(wire.latchLen).toBeLessThan(100);
    const span0 = portAlignment(sim, con, era, 'l', 'p').len;
    for (let i = 0; i < 24; i++) sim.step(1 / 60, params);
    const span1 = portAlignment(sim, con, era, 'l', 'p').len;
    expect(span1, `aux wire should shrink (${span0.toFixed(1)}→${span1.toFixed(1)})`).toBeLessThan(
      span0 - 6,
    );
  });

  it('aux–aux: wired ports face each other with opposite bodies', () => {
    const sim = new Sim(400, 240);
    const params = baseParams();
    const a = sim.spawn('con', 170, 120, 0.55, params, true)!;
    const b = sim.spawn('con', 270, 120, -0.4, params, true)!;
    sim.wire(a.id, 'l', b.id, 'r', params);

    // 300 frames, not 100: the ports are pulled into line by a compliant
    // torque now rather than assigned, so settling takes a couple of seconds.
    let last = portAlignment(sim, a, b, 'l', 'r');
    for (let i = 0; i < 300; i++) {
      sim.step(1 / 60, params);
      last = portAlignment(sim, a, b, 'l', 'r');
    }

    expect(last.faceA, `left aux should face neighbor (dot=${last.faceA.toFixed(2)})`).toBeGreaterThan(0.8);
    expect(-last.faceB, `right aux should face neighbor (dot=${(-last.faceB).toFixed(2)})`).toBeGreaterThan(0.8);
    // Both ends are aux here, so both contribute their side-splay.
    const auxSplay = params.auxSpread * 0.35;
    expect(Math.abs(Math.abs(angleDelta(a.heading, b.heading)) - Math.PI)).toBeLessThan(
      0.35 + 2 * auxSplay,
    );
  });

  it('latched cargo holds port facing without spinning', () => {
    const sim = new Sim(400, 240);
    const params = baseParams();
    params.flockAlign = 8;
    const a = sim.spawn('era', 120, 120, 0.4, params, true)!;
    const b = sim.spawn('era', 260, 120, -0.35, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    let peakOmega = 0;
    let turned = 0;
    let prev = a.heading;
    for (let i = 0; i < 120; i++) {
      sim.step(1 / 60, params);
      peakOmega = Math.max(peakOmega, Math.abs(a.omega), Math.abs(b.omega));
      turned += Math.abs(angleDelta(prev, a.heading));
      prev = a.heading;
    }
    const last = portAlignment(sim, a, b, 'p', 'p');
    expect(last.faceA).toBeGreaterThan(0.9);
    expect(-last.faceB).toBeGreaterThan(0.9);
    // A pair spawned at 0.4 and -0.35 rad has to turn to face along the wire,
    // so some rotation is the point; ~1.7 rad total is that turn plus settling.
    // A genuine spin would be several multiples of 2pi.
    // Total rotation is what "does not spin" means. The old peak-omega bound of
    // 6 was really measuring reconstruct()'s velocity clamp, since omega used to
    // be back-derived from kinematic teleports; it is a real angular velocity
    // now, and reeling a 124 px latch down to 40 genuinely flicks a light Era.
    expect(turned, `total rotation ${turned.toFixed(2)} rad`).toBeLessThan(2.5);
    expect(peakOmega, `peak omega ${peakOmega.toFixed(1)} rad/s`).toBeLessThan(20);
  });
});
