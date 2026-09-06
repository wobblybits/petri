import { describe, expect, it } from 'vitest';
import { CH } from './fields.ts';
import { B_STATE, IN_BOUND, IN_DEMAND, IN_DIMS, IN_FULL, IN_SENSE, L_BASE, L_OUT, STATE_DIMS, W_IN, W_NET, W_SELF, refreshReadsField } from './agents.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * The recurrent half: `h <- phi(Wx.x + Wh.h + Wn.mean(h_j) + b)`.
 *
 * These need a Sim, because the whole point of `Wn` is neighbours and the whole
 * point of `Wx` is inputs the body does not carry itself. The output side —
 * what `E` and `T` do with a state — is unit-tested in `chem-evolution` with
 * `h` set by hand.
 *
 * `phi` is `x / (1 + |x|)`, so a saturating input lands at 0.5 rather than 1 and
 * the expectations below are written against that.
 */

function pond(): { sim: Sim; params: ReturnType<typeof defaultParams> } {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.rewriteDuration = 0;
  params.upkeep = 0;
  return { sim: new Sim(10000, 10000), params };
}

/**
 * Zero every weight, so a test only sees the one path it sets.
 *
 * Anything that writes `chem` outside birth has to refresh the cached
 * "does this genome read the field" flag — `updateState` trusts it and will
 * skip sampling for a body that has just been given sense weights, which looks
 * exactly like the weights not working.
 */
function blank(sim: Sim): void {
  for (const a of sim.agents.values()) {
    for (let k = W_IN; k < B_STATE + STATE_DIMS; k++) a.chem[k] = 0;
    refreshReadsField(a);
  }
}

/** Call after poking `Wx`'s sense columns by hand. See `blank`. */
function rewire(a: ReturnType<Sim['spawn']>): void {
  refreshReadsField(a!);
}


describe('the state network', () => {
  it('stays at zero when every weight is zero', () => {
    const { sim, params } = pond();
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    sim.step(1 / 60, params);
    blank(sim);
    for (let f = 0; f < 30; f++) sim.step(1 / 60, params);
    for (let d = 0; d < STATE_DIMS; d++) expect(a.h[d]).toBe(0);
  });

  it('carries an input into the state through Wx', () => {
    const { sim, params } = pond();
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    sim.step(1 / 60, params);
    blank(sim);
    a.chem[W_IN + 0 * IN_DIMS + IN_FULL] = 1;
    a.extra = a.energyCap; // FULL saturates at 1
    for (let f = 0; f < 10; f++) sim.step(1 / 60, params);
    // phi(1) = 0.5.
    expect(a.h[0]).toBeCloseTo(0.5, 2);
    expect(a.h[1], 'nothing else should have moved').toBe(0);
  });

  it('reads the ground on the same scale as a signal', () => {
    // Raw, a full cell of ground is `ambientEnergy / 16` against signal peaks
    // near ten. `updateState` divides by `cellCap` so a full cell reads 1,
    // which is the only way a weight bounded at 4 can reach it at all.
    const { sim, params } = pond();
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    // Full *before* the first frame. The first step is what pins the world and
    // lays the ground down, and a hungry body harvests on that same frame —
    // its `energyCap` is twenty cells' worth, so it strip-mines its own block
    // and then reads the hole. Topping it up afterwards is too late.
    a.extra = a.energyCap;
    sim.step(1 / 60, params);
    blank(sim);
    a.chem[W_IN + 0 * IN_DIMS + IN_SENSE + CH.energy] = 1;
    rewire(a);
    for (let f = 0; f < 10; f++) sim.step(1 / 60, params);
    expect(a.sense[CH.energy], 'full ground should read about 1').toBeCloseTo(1, 1);
    expect(a.h[0]).toBeGreaterThan(0.3);
  });

  it('remembers, so a state outlives the input that made it', () => {
    const { sim, params } = pond();
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    sim.step(1 / 60, params);
    blank(sim);
    a.chem[W_IN + 0 * IN_DIMS + IN_FULL] = 1;
    a.chem[W_SELF + 0 * STATE_DIMS + 0] = 0.9;
    a.extra = a.energyCap;
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params);
    const charged = a.h[0];
    expect(charged).toBeGreaterThan(0.4);

    // Take the input away. Without `Wh` this would be zero on the next frame;
    // with it the state decays instead, which is what memory is.
    a.chem[W_IN + 0 * IN_DIMS + IN_FULL] = 0;
    a.extra = 0;
    sim.step(1 / 60, params);
    expect(a.h[0], 'the state vanished the instant its input did').toBeGreaterThan(charged * 0.3);
  });

  it('carries state along a wire, one hop per frame', () => {
    const { sim, params } = pond();
    // A short chain: only the head has an input, and the rest have to hear it
    // from their neighbours.
    const ids = [0, 1, 2].map((i) => sim.spawn('con', 5000 + i * 60, 5000, 0, params, true)!);
    sim.step(1 / 60, params);
    sim.wire(ids[0].id, 'r', ids[1].id, 'l', params);
    sim.wire(ids[1].id, 'r', ids[2].id, 'l', params);
    sim.step(1 / 60, params);
    blank(sim);
    for (const a of ids) a.chem[W_NET + 0 * STATE_DIMS + 0] = 1;
    ids[0].chem[W_IN + 0 * IN_DIMS + IN_FULL] = 1;
    ids[0].extra = ids[0].energyCap;

    for (let f = 0; f < 30; f++) sim.step(1 / 60, params);
    // Monotonically weaker along the chain: each hop is another pass through
    // `phi` and another averaging with a quieter neighbour.
    expect(ids[0].h[0]).toBeGreaterThan(ids[1].h[0]);
    expect(ids[1].h[0]).toBeGreaterThan(ids[2].h[0]);
    expect(ids[2].h[0], 'nothing reached the far end').toBeGreaterThan(0);
  });

  it('averages neighbours rather than summing them', () => {
    // A sum would make the same genome behave differently by degree — a hub
    // saturating `phi` while a leaf barely moves, for a reason that is not
    // about position. `BOUND` carries degree, bounded and on purpose.
    const build = (leaves: number): number => {
      const { sim, params } = pond();
      const hub = sim.spawn('era', 5000, 5000, 0, params, true)!;
      const arms: ReturnType<Sim['spawn']>[] = [];
      for (let i = 0; i < leaves; i++) {
        arms.push(sim.spawn('con', 5000 + (i + 1) * 60, 5000, 0, params, true)!);
      }
      sim.step(1 / 60, params);
      // Star: every leaf onto the hub. A Con has l and r spare.
      for (let i = 0; i < leaves; i++) sim.wire(hub.id, 'p', arms[i]!.id, i === 0 ? 'l' : 'r', params);
      sim.step(1 / 60, params);
      blank(sim);
      hub.chem[W_NET + 0 * STATE_DIMS + 0] = 1;
      for (const a of arms) {
        a!.chem[W_IN + 0 * IN_DIMS + IN_FULL] = 1;
        a!.extra = a!.energyCap;
      }
      for (let f = 0; f < 30; f++) sim.step(1 / 60, params);
      return hub.h[0];
    };
    const one = build(1);
    const three = build(3);
    expect(one).toBeGreaterThan(0);
    // Every leaf carries the same state, so their mean is that state whatever
    // the degree. A sum would have made three leaves roughly triple.
    expect(three).toBeCloseTo(one, 1);
  });

  it('bounds the state however hard it is driven', () => {
    const { sim, params } = pond();
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    sim.step(1 / 60, params);
    blank(sim);
    for (let d = 0; d < STATE_DIMS; d++) {
      a.chem[B_STATE + d] = 50;
      a.chem[W_SELF + d * STATE_DIMS + d] = 50;
      a.chem[W_IN + d * IN_DIMS + IN_DEMAND] = 50;
      a.chem[W_IN + d * IN_DIMS + IN_BOUND] = 50;
    }
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
    for (let d = 0; d < STATE_DIMS; d++) {
      expect(Number.isFinite(a.h[d])).toBe(true);
      expect(Math.abs(a.h[d]), 'phi should have held it inside one').toBeLessThan(1);
    }
  });
});

describe('the locomotion head', () => {
  it('gives a seeded body exactly the sliders it always had', () => {
    const { sim, params } = pond();
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    sim.step(1 / 60, params);
    expect(a.cruise).toBeCloseTo(params.stepSpeed, 4);
    expect(a.turn).toBeCloseTo(params.turnRate, 4);
  });

  it('lets one body swim while another beside it does not', () => {
    // The thing a global `stepSpeed` could never express. Same pond, same
    // frame, two genomes.
    const { sim, params } = pond();
    params.swimNoise = 0;
    const fast = sim.spawn('con', 5000, 5000, 0, params, true)!;
    const still = sim.spawn('con', 5000, 5400, 0, params, true)!;
    sim.step(1 / 60, params);
    still.chem[L_BASE] = 0;
    const x0 = still.x;
    const f0 = fast.x;
    for (let f = 0; f < 120; f++) sim.step(1 / 60, params);
    expect(still.cruise, 'it should have stopped wanting to swim').toBeCloseTo(0, 6);
    expect(Math.abs(fast.x - f0), 'the other one should still be going').toBeGreaterThan(
      Math.abs(still.x - x0),
    );
  });

  it('drives locomotion from state, not just from the base', () => {
    const { sim, params } = pond();
    // Barren, or setting `extra` to zero does nothing: harvest runs before
    // `updateState`, so a hungry body on full ground is full again by the time
    // `FULL` is read.
    params.ambientEnergy = 0;
    const a = sim.spawn('con', 5000, 5000, 0, params, true)!;
    sim.step(1 / 60, params);
    blank(sim);
    a.chem[L_BASE] = 0;
    // Swim only when full: the rule a global speed cannot hold.
    a.chem[W_IN + 0 * IN_DIMS + IN_FULL] = 1;
    a.chem[L_OUT + 0 * STATE_DIMS + 0] = 2;
    a.extra = 0;
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params);
    const hungry = a.cruise;
    a.extra = a.energyCap;
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params);
    expect(hungry, 'an empty body should not have been moving').toBeCloseTo(0, 4);
    expect(a.cruise, 'a full one should').toBeGreaterThan(5);
  });
});
