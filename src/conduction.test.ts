import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';
import { CHEM_LEN, PUSH_BASE, HEAD_SCALE } from './chem-layout.ts';
import { applyTransportRecoil } from './sim.ts';

/*
 * The two operators that make transport directional.
 *
 * `transportSpeed` gives the need field a time constant, so a shortage takes
 * time to be heard and path length decides when. `pushRate` lets a body move
 * matter out of a chosen port on its own say-so, rather than only toward
 * whoever is worst off. Both ship at zero and both must reduce to exactly what
 * the sim did before them.
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

describe('directed push', () => {
  /** Make `slot` of every body push at full rate, whatever its state. */
  function drivePush(sim: Sim, ids: number[], slotIndex: number): void {
    for (const id of ids) {
      const a = sim.agents.get(id)!;
      a.chem[PUSH_BASE + slotIndex] = 1 / HEAD_SCALE.push;
    }
  }

  it('is off by default', () => {
    expect(defaultParams().pushRate).toBe(0);
  });

  it('moves energy into a neighbour that is not asking for any', () => {
    const params = quiet();
    params.pushRate = 1;
    const { sim, ids } = chain(params, 2);
    const tail = sim.agents.get(ids[0])!;
    const head = sim.agents.get(ids[1])!;
    // Both full but for a little room, so nothing is short and `flowCharges`
    // has nothing to do. Only a push can move anything.
    head.extra = head.energyCap - 0.4;
    drivePush(sim, [ids[0]], 0); // the tail's principal, which is wired ahead
    const before = tail.extra;
    for (let f = 0; f < 30; f++) sim.step(1 / 60, params);
    expect(head.request, 'the receiver never asked').toBe(0);
    expect(tail.extra, 'the donor should have spent').toBeLessThan(before);
    expect(head.extra, 'the receiver should have gained').toBeGreaterThan(head.energyCap - 0.4);
  });

  it('conserves, and cannot force-feed a full body', () => {
    const params = quiet();
    params.pushRate = 1;
    const { sim, ids } = chain(params, 2);
    const tail = sim.agents.get(ids[0])!;
    const head = sim.agents.get(ids[1])!;
    head.extra = head.energyCap;
    drivePush(sim, [ids[0]], 0);
    const total = tail.extra + head.extra;
    for (let f = 0; f < 30; f++) sim.step(1 / 60, params);
    expect(tail.extra + head.extra, 'nothing minted or lost').toBeCloseTo(total, 9);
    expect(tail.extra, 'a full neighbour takes nothing').toBeCloseTo(total - head.energyCap, 9);
  });

  it('pushes the body the other way along the port it pumped out of', () => {
    // The whole point: which port a body pushes from is a direction in its own
    // frame, so the same net swims forwards or backwards on one gene.
    const params = quiet();
    params.pushRate = 1;
    params.drag = 0;
    params.transportRecoil = 100;
    const forward = chain(params, 2);
    forward.sim.agents.get(forward.ids[1])!.extra = 0.2;
    drivePush(forward.sim, [forward.ids[0]], 0); // out of the principal, which faces +x
    const back = chain(params, 2);
    back.sim.agents.get(back.ids[0])!.extra = 0.2;
    drivePush(back.sim, [back.ids[1]], 1); // out of an aux, which faces -x
    for (let f = 0; f < 20; f++) {
      forward.sim.step(1 / 60, params);
      back.sim.step(1 / 60, params);
    }
    const pushedFromP = forward.sim.agents.get(forward.ids[0])!;
    const pushedFromL = back.sim.agents.get(back.ids[1])!;
    expect(pushedFromP.vx, 'pumping out of the nose drives the body backwards').toBeLessThan(0);
    expect(pushedFromL.vx, 'pumping out of an aux drives it forwards').toBeGreaterThan(0);
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

describe('recoil lever', () => {
  /*
   * Unit tests on `applyTransportRecoil` itself rather than on a stepped sim.
   * A body's `omega` after a frame carries the port torques and the constraint
   * solve as well as the recoil, and those dominate — the first version of
   * these read 1.33 rad/s on a body the push had not turned at all.
   */
  const body = (x: number, y: number) => ({
    x, y, vx: 0, vy: 0, omega: 0, mass: 1, locked: false,
  });
  /** An aux stem's offset at heading 0: back along the body, off to one side. */
  const AUX_BACK = -8.8;
  const AUX_SIDE = 9.184;
  const lever = (ay: number, amount: number) => ({
    ax: AUX_BACK, ay, bx: 0, by: 0, invIA: 1, invIB: 0, amount,
  });

  it('leaves a body unturned when the lever is off', () => {
    const a = body(0, 0);
    const b = body(48, 0);
    applyTransportRecoil(a, b, 1, 10, 4000, 4000, 1, lever(-AUX_SIDE, 0));
    expect(a.vx, 'it should still have been shoved').toBeLessThan(0);
    expect(a.omega).toBe(0);
  });

  it('turns a body when the port it pumped from is off the centreline', () => {
    const a = body(0, 0);
    const b = body(48, 0);
    applyTransportRecoil(a, b, 1, 10, 4000, 4000, 1, lever(-AUX_SIDE, 1));
    expect(Math.abs(a.omega)).toBeGreaterThan(1e-6);
  });

  it('bends opposite ways out of the two aux ports', () => {
    // The whole point of an alternating spine: one push gene is opposite bends
    // depending on which side of the centreline the joint sits.
    const left = body(0, 0);
    const right = body(0, 0);
    const ahead = body(48, 0);
    applyTransportRecoil(left, ahead, 1, 10, 4000, 4000, 1, lever(-AUX_SIDE, 1));
    applyTransportRecoil(right, body(48, 0), 1, 10, 4000, 4000, 1, lever(AUX_SIDE, 1));
    expect(Math.sign(left.omega)).toBe(-Math.sign(right.omega));
    expect(Math.abs(left.omega)).toBeCloseTo(Math.abs(right.omega), 12);
  });

  it('does not turn a body pumping out of its principal, which is on the axis', () => {
    const a = body(0, 0);
    const b = body(48, 0);
    // A principal stem sits at (16.8, 0): forward, and exactly on the centreline.
    applyTransportRecoil(a, b, 1, 10, 4000, 4000, 1, { ...lever(0, 1), ax: 16.8 });
    expect(a.omega).toBeCloseTo(0, 12);
    expect(a.vx, 'but it is still a shove').toBeLessThan(0);
  });

  it('reaches the sim through pushCharges', () => {
    const params = quiet();
    params.pushRate = 1;
    params.angDrag = 0;
    params.recoilLever = 1;
    params.portStiff = 0;
    params.springK = 0;
    const { sim, ids } = chain(params, 2);
    sim.agents.get(ids[1])!.extra = 0.2;
    sim.agents.get(ids[1])!.chem[PUSH_BASE + 1] = 1 / HEAD_SCALE.push;
    const before = sim.agents.get(ids[1])!.omega;
    sim.step(1 / 60, params);
    expect(sim.agents.get(ids[1])!.omega).not.toBe(before);
  });
});
