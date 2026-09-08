import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';
import { ANGLE_BASE, CHEM_LEN, HEAD_SCALE } from './chem-layout.ts';
import { momentOfInertia } from './agents.ts';

/*
 * The three things a net has beyond its economy: a conduction speed, a medium
 * that notices direction, and one actuator.
 *
 * `transportSpeed` gives the need field a time constant, so a shortage takes
 * time to be heard and path length decides when. `dragAniso` makes the medium
 * care which way a body lies. `jointStiff` and the `ANGLE` head are the
 * actuator, and the only restoring force a net can have. All ship at their
 * neutral value and all must reduce to exactly what the sim did before them.
 */

/** A straight chain of Cons, principal into aux, tail first. */
function chain(params: ReturnType<typeof defaultParams>, n: number): { sim: Sim; ids: number[] } {
  const sim = new Sim(4000, 4000, 128);
  sim.pinWorld(2000, 2000, params);
  const ids: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = sim.spawn('con', 2000 + i * params.wireMinRest - (n * params.wireMinRest) / 2, 2000, 0, params, true)!;
    a.extra = a.energyCap;
    ids.push(a.id);
  }
  for (let i = 0; i + 1 < n; i++) sim.wire(ids[i], 'p', ids[i + 1], 'l', params);
  return { sim, ids };
}

function quiet(): ReturnType<typeof defaultParams> {
  const p = defaultParams();
  p.stepSpeed = 0;
  p.turnRate = 0;
  p.swimNoise = 0;
  p.flockAlign = 0;
  p.flockSep = 0;
  p.snapRadius = 0;
  p.spawnInterval = 0;
  p.wireBreathe = 0;
  p.declutter = 0;
  p.ambientEnergy = 0;
  p.upkeep = 0;
  return p;
}

describe('conduction speed', () => {
  it('is off by default, and a shortage is then heard everywhere at once', () => {
    const params = quiet();
    expect(params.transportSpeed).toBe(0);
    const { sim, ids } = chain(params, 6);
    // Starve the tail so it is the only claim in the net.
    sim.agents.get(ids[0])!.extra = -0.5;
    sim.step(1 / 60, params);
    const far = sim.agents.get(ids[5])!;
    expect(far.request, 'the far end should hear it on frame one').toBeGreaterThan(0);
  });

  it('takes time proportional to path length once it has a speed', () => {
    const params = quiet();
    // Six hops a second: one hop is ten frames, so the far end of a six-body
    // chain is fifty frames away and must still be silent at ten.
    params.transportSpeed = 6;
    const { sim, ids } = chain(params, 6);
    sim.agents.get(ids[0])!.extra = -0.5;
    const near = sim.agents.get(ids[1])!;
    const far = sim.agents.get(ids[5])!;
    for (let f = 0; f < 10; f++) sim.step(1 / 60, params);
    const nearEarly = near.request;
    const farEarly = far.request;
    expect(nearEarly, 'the neighbour should have started to hear it').toBeGreaterThan(0);
    expect(farEarly, 'the far end should not have heard it yet').toBeLessThan(nearEarly);
    for (let f = 0; f < 120; f++) sim.step(1 / 60, params);
    expect(far.request, 'and should hear it eventually').toBeGreaterThan(farEarly);
  });

  it('recedes as well as arrives, which is what makes it a pulse', () => {
    const params = quiet();
    params.transportSpeed = 6;
    const { sim, ids } = chain(params, 4);
    const tail = sim.agents.get(ids[0])!;
    tail.extra = -0.5;
    for (let f = 0; f < 90; f++) sim.step(1 / 60, params);
    const peak = sim.agents.get(ids[3])!.request;
    expect(peak, 'the far end should be hearing the shortage').toBeGreaterThan(0);
    // Fed, and no longer recovering: the claim goes away at the source.
    tail.extra = tail.energyCap;
    tail.recovering = false;
    for (let f = 0; f < 240; f++) sim.step(1 / 60, params);
    expect(
      sim.agents.get(ids[3])!.request,
      'and should fall again once the shortage is gone',
    ).toBeLessThan(peak);
  });

  it('is heritable and survives a round trip through the store', () => {
    const params = quiet();
    params.transportSpeed = 12;
    const { sim, ids } = chain(params, 2);
    expect(sim.agents.get(ids[0])!.conductSpeed).toBe(12);
    sim.agents.get(ids[0])!.conductSpeed = 30;
    expect(sim.agentStore.conductSpeed[sim.agents.get(ids[0])!.slot]).toBe(30);
  });
});

describe('the genome still has room for it', () => {
  it('gives every body a full-width chem row', () => {
    const params = quiet();
    const { sim, ids } = chain(params, 1);
    expect(sim.agents.get(ids[0])!.chem.length).toBe(CHEM_LEN);
  });
});

describe('anisotropic drag', () => {
  it('is isotropic by default', () => {
    expect(defaultParams().dragAniso).toBe(1);
  });

  it('holds back sideways motion harder than motion along the heading', () => {
    const params = quiet();
    params.drag = 1;
    params.dragAniso = 3;
    const sim = new Sim(4000, 4000, 128);
    sim.pinWorld(2000, 2000, params);
    // Same speed, same heading, one moving along it and one across it.
    const along = sim.spawn('con', 1900, 2000, 0, params, true)!;
    const across = sim.spawn('con', 2100, 2000, 0, params, true)!;
    along.vx = 100;
    across.vy = 100;
    for (let f = 0; f < 30; f++) sim.step(1 / 60, params);
    expect(Math.hypot(across.vx, across.vy)).toBeLessThan(Math.hypot(along.vx, along.vy));
  });

  it('reduces to the old loop at a ratio of one', () => {
    const run = (aniso: number): number => {
      const params = quiet();
      params.drag = 1;
      params.dragAniso = aniso;
      const sim = new Sim(4000, 4000, 128);
      sim.pinWorld(2000, 2000, params);
      const a = sim.spawn('con', 2000, 2000, 0.7, params, true)!;
      a.vx = 60;
      a.vy = -40;
      for (let f = 0; f < 30; f++) sim.step(1 / 60, params);
      return Math.hypot(a.vx, a.vy);
    };
    expect(run(1)).toBeCloseTo(run(1), 12);
    // A heading of 0.7 rad against a velocity that is neither along nor across
    // it: the two paths must still agree exactly at the neutral ratio.
    expect(run(1.0000001)).toBeCloseTo(run(1), 4);
  });
});

describe('joint rest angle', () => {
  it('is off by default', () => {
    expect(defaultParams().jointStiff).toBe(0);
    expect(defaultParams().bendCost).toBe(0);
  });

  /** Worst hinge in the chain after the free end is shoved sideways. */
  function foldUnder(jointStiff: number): number {
    const params = quiet();
    params.jointStiff = jointStiff;
    params.angDrag = 0;
    const { sim, ids } = chain(params, 5);
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
    sim.agents.get(ids[0])!.pinned = true;
    for (let f = 0; f < 120; f++) {
      const head = sim.agents.get(ids[4])!;
      head.vy += (40 * (1 / 60)) / Math.max(0.08, head.mass);
      sim.step(1 / 60, params);
    }
    let worst = 0;
    for (let i = 0; i + 1 < ids.length; i++) {
      const a = sim.agents.get(ids[i])!;
      const b = sim.agents.get(ids[i + 1])!;
      worst = Math.max(worst, Math.abs(a.heading - b.heading));
    }
    return worst;
  }

  it('resists a fold, which aiming alone does not', () => {
    // The measurement the head exists for. `portTorques` aims each port at its
    // partner's stem position and a smooth arc satisfies that for free, so it
    // cannot see curvature; this constrains the angle between the two port
    // axes, which an arc does not satisfy.
    expect(foldUnder(300)).toBeLessThan(foldUnder(0));
  });

  it('applies a couple, so it adds no spin to the pair', () => {
    /*
     * Against the pass, not against a step. The couple is exact here and only
     * approximately visible through a whole frame, because the constraint solve
     * re-derives `omega` from the poses it lands on — so a step-level check
     * would be measuring the solver.
     */
    const params = quiet();
    params.jointStiff = 400;
    const { sim, ids } = chain(params, 2);
    const a = sim.agents.get(ids[0])!;
    const b = sim.agents.get(ids[1])!;
    b.heading = 0.6;
    a.omega = 0.3;
    b.omega = -0.2;
    const before = a.omega * momentOfInertia(a) + b.omega * momentOfInertia(b);
    sim.jointAngles(params, 1 / 60);
    const after = a.omega * momentOfInertia(a) + b.omega * momentOfInertia(b);
    expect(a.omega, 'the joint did nothing at all').not.toBeCloseTo(0.3, 9);
    expect(b.omega, 'the joint moved only one end').not.toBeCloseTo(-0.2, 9);
    expect(after, 'an internal actuator may change a shape, never a momentum').toBeCloseTo(
      before,
      9,
    );
  });

  it('bends the joint when the head commands an angle', () => {
    const params = quiet();
    params.jointStiff = 200;
    params.portStiff = 0;
    const { sim, ids } = chain(params, 2);
    const a = sim.agents.get(ids[0])!;
    // Its principal, which is the port the chain is wired through.
    a.chem[ANGLE_BASE] = 0.6 / HEAD_SCALE.angle;
    for (let f = 0; f < 120; f++) sim.step(1 / 60, params);
    const b = sim.agents.get(ids[1])!;
    expect(Math.abs(a.heading - b.heading), 'the commanded bend never appeared').toBeGreaterThan(0.1);
  });

  it('charges for a commanded bend and nothing for a straight one', () => {
    const held = (rest: number): number => {
      const params = quiet();
      params.jointStiff = 200;
      params.bendCost = 0.2;
      const { sim, ids } = chain(params, 2);
      const a = sim.agents.get(ids[0])!;
      a.extra = 1;
      a.chem[ANGLE_BASE] = rest / HEAD_SCALE.angle;
      for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
      return a.extra;
    };
    expect(held(0), 'a straight joint is free').toBeCloseTo(1, 6);
    expect(held(0.8), 'holding a bend has to cost something').toBeLessThan(0.99);
  });
});
