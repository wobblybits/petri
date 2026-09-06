import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * Saving up for a commute.
 *
 * Affordability used to be an instantaneous gate: a ready pair was asked for
 * both shares in one frame, and a pair that met while poor lost the meeting
 * along with the need it had posted. The escrow makes the pot a property of
 * the wire instead of the moment, so energy arriving at a redex stays there.
 *
 * See `Sim.accrueRedexes`. The whole-pond total that has to balance is
 * bodies + ground + `escrowTotal`.
 */

function facing(): { sim: Sim; params: ReturnType<typeof defaultParams> } {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.ambientEnergy = 0;
  params.upkeep = 0;
  params.rewriteDuration = 0.12;
  params.wireShrink = 0.08;
  return { sim: new Sim(800, 600), params };
}

/** A Con and a Dup nose to nose: the one rule that costs anything to fire. */
function commutePair(sim: Sim, params: ReturnType<typeof defaultParams>) {
  const c = sim.spawn('con', 380, 300, 0, params, true)!;
  const d = sim.spawn('dup', 420, 300, Math.PI, params, true)!;
  sim.wire(c.id, 'p', d.id, 'p', params);
  return { c, d };
}

/** Everything the pond is holding, wherever it is holding it. */
function pondTotal(sim: Sim): number {
  let inBodies = 0;
  for (const a of sim.agents.values()) inBodies += a.extra;
  return inBodies + sim.energy.storedTotal() + sim.escrowTotal();
}

describe('a redex saves up', () => {
  it('commutes on a drip neither end could ever hold a share of', () => {
    // A tenth of a share at a time, taken away again every frame. Under the
    // old rule — both ends holding a full share in the same frame — this pair
    // could not have bred however long it ran.
    const { sim, params } = facing();
    commutePair(sim, params);
    let mostHeld = 0;
    for (let f = 0; f < 900; f++) {
      for (const a of sim.agents.values()) {
        if (a.extra < 0.1) a.extra = 0.1;
        if (a.extra > mostHeld) mostHeld = a.extra;
      }
      sim.step(1 / 60, params);
      if (sim.agents.size > 2) break;
    }
    expect(mostHeld, 'no end was ever close to affording its own share').toBeLessThan(0.5);
    expect(sim.agents.size, 'a commute builds four from two').toBe(4);
  });

  it('holds the stake on the wire, not on the bodies', () => {
    const { sim, params } = facing();
    const { c, d } = commutePair(sim, params);
    for (let f = 0; f < 200; f++) {
      c.extra = 0.05;
      d.extra = 0.05;
      sim.step(1 / 60, params);
      if (sim.escrowTotal() > 0) break;
    }
    expect(sim.escrowTotal(), 'nothing was ever banked').toBeGreaterThan(0);
    expect(c.extra + d.extra, 'and it left the tanks to get there').toBeLessThan(0.1);
  });

  it('gives the stake back when the redex comes apart', () => {
    const { sim, params } = facing();
    const { c, d } = commutePair(sim, params);
    for (let f = 0; f < 200; f++) {
      if (c.extra < 0.2) c.extra = 0.2;
      if (d.extra < 0.2) d.extra = 0.2;
      sim.step(1 / 60, params);
      if (sim.escrowTotal() > 0.2) break;
    }
    const staked = sim.escrowTotal();
    expect(staked, 'nothing was staked, so there is nothing to refund').toBeGreaterThan(0.2);
    const before = pondTotal(sim);

    for (const w of [...sim.graph.wires.keys()]) sim.graph.detach(w);
    sim.step(1 / 60, params);
    expect(sim.escrowTotal(), 'the pot outlived its wire').toBe(0);
    expect(pondTotal(sim), 'and the energy in it went nowhere').toBeCloseTo(before, 6);
    expect(c.extra + d.extra, 'it should have gone back to the two who put it up')
      .toBeGreaterThanOrEqual(staked - 1e-6);
  });

  it('does not burn the stake when a partner dies', () => {
    const { sim, params } = facing();
    const { c, d } = commutePair(sim, params);
    for (let f = 0; f < 200; f++) {
      if (c.extra < 0.2) c.extra = 0.2;
      if (d.extra < 0.2) d.extra = 0.2;
      sim.step(1 / 60, params);
      if (sim.escrowTotal() > 0.2) break;
    }
    expect(sim.escrowTotal()).toBeGreaterThan(0.2);
    // `kill` detaches every wire, so the pot loses the thing it hangs on.
    // A dead body's stake has to land on the ground rather than vanish.
    const before = pondTotal(sim) + 0; // deathYield is added on top by `kill`
    sim.kill(d.id);
    sim.step(1 / 60, params);
    expect(sim.escrowTotal()).toBe(0);
    expect(pondTotal(sim), 'the stake evaporated with the body').toBeGreaterThanOrEqual(
      before - 1e-6,
    );
  });

  it('hands everything back when rewrites are switched off', () => {
    const { sim, params } = facing();
    const { c, d } = commutePair(sim, params);
    for (let f = 0; f < 200; f++) {
      if (c.extra < 0.2) c.extra = 0.2;
      if (d.extra < 0.2) d.extra = 0.2;
      sim.step(1 / 60, params);
      if (sim.escrowTotal() > 0.2) break;
    }
    expect(sim.escrowTotal()).toBeGreaterThan(0.2);
    const before = pondTotal(sim);
    params.rewriteDuration = 0;
    sim.step(1 / 60, params);
    expect(sim.escrowTotal(), 'energy frozen in a pot nothing can spend').toBe(0);
    expect(pondTotal(sim)).toBeCloseTo(before, 6);
  });

  it('leaves a body climbing out of debt alone', () => {
    // `pulseRequests` runs first, so rescue energy lands on the frame before
    // accrual. Swallowing it would make a rescue impossible to complete for
    // any body unlucky enough to be half of a redex.
    const { sim, params } = facing();
    const { c, d } = commutePair(sim, params);
    for (let f = 0; f < 200; f++) {
      if (c.extra < 0.2) c.extra = 0.2;
      if (d.extra < 0.2) d.extra = 0.2;
      sim.step(1 / 60, params);
      if (sim.escrowTotal() > 0.2) break;
    }
    // Both ends, and equally: two bodies of identical need pass nothing to
    // each other (see `flowCharges`), so anything that moves here moved into
    // the pot and nowhere else.
    c.extra = -0.2;
    d.extra = -0.2;
    sim.step(1 / 60, params);
    expect(c.recovering && d.recovering, 'the setup should have put both in debt').toBe(true);
    const banked = sim.escrowTotal();
    c.extra = 0.3;
    d.extra = 0.3;
    sim.step(1 / 60, params);
    expect(sim.escrowTotal(), 'the pot took a rescue that was keeping a body alive')
      .toBeCloseTo(banked, 6);
    expect(c.extra + d.extra, 'and the rescue stayed where it landed').toBeCloseTo(0.6, 6);
  });
});
