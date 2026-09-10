import { describe, expect, it } from 'vitest';
import { G_BASE } from './agents.ts';
import { defaultParams, type Params } from './params.ts';
import { applyTransportRecoil, Sim } from './sim.ts';

/**
 * Locomotion, which lives in the drag law.
 *
 * Under one drag law for everyone, a net cannot move itself. The rope shares
 * its corrections by mass and a transport kick is equal and opposite, so
 * every internal force cancels at the centre of mass — which left
 * `transportThrust`, whose whole job was to *not* cancel, minting momentum
 * wherever energy flowed. `grip` is the other way out, and the one every
 * crawler uses: anchor one end, let the other slide, and an internal stroke
 * becomes travel with nothing invented.
 *
 * A speed-dependent term was built beside it and withdrawn. It swam — 1.63
 * px/s on the worm bench against grip's 2.31, and the two did not add — but
 * two mechanics doing one job is the thing `docs/concepts.md` exists to make
 * visible, and grip is the one whose stroke depends on the energy pattern a
 * net actually controls.
 *
 * These are about the drag law, not about the pond. The last two build the
 * configuration the mechanic is for — a pair pumping along a wire — and check
 * that the centre goes somewhere, and which way.
 */

/** A dish with nothing in it but the drag law: no swimming, flocking or spawning. */
function stillParams(): Params {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.stepSpeed = 0;
  params.turnRate = 0;
  params.swimNoise = 0;
  params.flockAlign = 0;
  params.flockSep = 0;
  params.declutter = 0;
  params.snapRadius = 0;
  params.snapWell = 0;
  params.faceAttract = 0;
  params.deposit = 0;
  params.upkeep = 0;
  params.ambientEnergy = 0;
  params.rewriteDuration = 0;
  params.angDrag = 0;
  // The gait is the other half of the drag law and these are about `grip`.
  // Its own tests turn it back on. Coupling too: it pulls two wired bodies to
  // a fixed phase difference, which is a third asymmetry on top of the three
  // the gait tests below take one at a time.
  params.gaitRate = 0;
  params.gaitCouple = 0;
  return params;
}

/** Two bodies a long way apart, one with a full tank and one with none. */
function pair(params: Params): { sim: Sim; full: ReturnType<Sim['spawn']> & object; empty: ReturnType<Sim['spawn']> & object } {
  const sim = new Sim(4000, 4000);
  const full = sim.spawn('con', 1500, 2000, 0, params, true)!;
  const empty = sim.spawn('con', 2500, 2000, 0, params, true)!;
  full.extra = full.energyCap;
  empty.extra = 0;
  full.vx = 100;
  empty.vx = 100;
  return { sim, full, empty };
}

function run(sim: Sim, params: Params, seconds: number): void {
  for (let f = 0; f < Math.round(seconds * 60); f++) sim.step(1 / 60, params);
}

describe('grip', () => {
  it('is off at zero: a full tank and an empty one drag alike, at the old rate', () => {
    const params = stillParams();
    params.grip = 0;
    const { sim, full, empty } = pair(params);
    run(sim, params, 1);
    // Not bit-identical: the two sit at different places and the dish's own
    // position-dependent forces do not associate. Equal to 1e-10 of 57 px/s.
    expect(full.vx, 'fullness changes nothing').toBeCloseTo(empty.vx, 6);
    // The whole law, unchanged: one second of exp(-drag t) off 100 px/s.
    expect(full.vx).toBeCloseTo(100 * Math.exp(-params.drag), 6);
  });

  it('drags a full body harder above zero and lighter below it', () => {
    const heavy = stillParams();
    heavy.grip = 2;
    const h = pair(heavy);
    run(h.sim, heavy, 1);
    expect(h.full.vx, 'a full tank grips').toBeLessThan(h.empty.vx);
    /*
     * Four places, not six. The table used to be indexed by fullness, so a
     * body at 0 or 1 landed exactly on a node and the read was the
     * arithmetic. It is indexed by total rate now — the gait's anchor is a
     * third term and there is nothing narrower to key on — so no particular
     * fullness lands on a node, and linear interpolation of a convex `exp`
     * leaves about 1e-5 relative. That is 1.4e-7 of the speed being measured
     * and nothing physical reads it; the claim here is which way the drag
     * goes, and that is unchanged.
     */
    expect(h.empty.vx, 'an empty one is untouched: grip scales with fullness').toBeCloseTo(
      100 * Math.exp(-heavy.drag),
      4,
    );
    expect(h.full.vx).toBeCloseTo(100 * Math.exp(-(heavy.drag + 2)), 4);

    // Chosen against this pond's own `drag` rather than a fixed -0.4, which
    // silently drove the rate negative once `drag` came down and the clamp,
    // not the arithmetic, decided the answer.
    const slick = stillParams();
    slick.grip = -0.5 * slick.drag;
    const s = pair(slick);
    run(s.sim, slick, 1);
    expect(s.full.vx, 'a full tank slides').toBeGreaterThan(s.empty.vx);
    expect(s.full.vx).toBeCloseTo(100 * Math.exp(-(slick.drag + slick.grip)), 4);
  });

  it('cancels drag but never inverts it', () => {
    // Negative drag is an energy source and the soup comes apart in seconds,
    // so the sum is clamped at the rate. At -2 against a drag of 0.55 a body
    // is frictionless from 27% of a tank up, and never faster than it started.
    const params = stillParams();
    params.grip = -2;
    const { sim, full, empty } = pair(params);
    run(sim, params, 4);
    expect(full.vx, 'coasts, and gains nothing').toBeCloseTo(100, 9);
    expect(empty.vx).toBeLessThan(100 * Math.exp(-params.drag * 4) + 1e-6);
  });

  it('interpolates rather than sorting the pond into buckets', () => {
    // The rate comes off a 256-step table, spanning `drag + grip` here. Two
    // bodies a hair apart in fullness have to stay a hair apart in drag, or
    // the table quietly becomes a speciation mechanism.
    const params = stillParams();
    params.grip = 2;
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('con', 1500, 2000, 0, params, true)!;
    const b = sim.spawn('con', 2500, 2000, 0, params, true)!;
    // Both inside one bucket — a step is 2.55/256 = 0.0100 of rate, and these
    // two are 0.004 apart — so a table read without interpolation hands them
    // the identical rate.
    a.extra = a.energyCap * 0.5;
    b.extra = b.energyCap * 0.502;
    a.vx = 100;
    b.vx = 100;
    run(sim, params, 1);
    expect(a.vx).not.toBe(b.vx);
    expect(a.vx).toBeGreaterThan(b.vx);
    // Each tracks its own fullness, to well under the difference between them.
    expect(a.vx).toBeCloseTo(100 * Math.exp(-(params.drag + 2 * 0.5)), 3);
    expect(b.vx).toBeCloseTo(100 * Math.exp(-(params.drag + 2 * 0.502)), 3);
  });

  it('turns an equal-and-opposite kick into travel, which one drag law cannot', () => {
    /*
     * The claim, at its smallest. A pump at `transportThrust` 0 conserves
     * momentum exactly, so the pair's centre is fixed under a uniform rate
     * whatever the kick. Give the two bodies different rates and the centre
     * moves: an impulse `p` into mass `m` at rate `k` carries it `p / (m k)`,
     * so the pair ends up `|p| (1/k_s - 1/k_r) / (m_s + m_r)` along the
     * sender's recoil, and the mass falls out of the direction.
     */
    const travel = (grip: number): number => {
      const params = stillParams();
      params.grip = grip;
      const sim = new Sim(4000, 4000);
      // Transport runs down the demand gradient, so the sender is the fuller.
      const sender = sim.spawn('con', 1960, 2000, 0, params, true)!;
      const receiver = sim.spawn('con', 2040, 2000, 0, params, true)!;
      sender.extra = sender.energyCap;
      receiver.extra = 0;
      const before = (sender.x + receiver.x) / 2;
      applyTransportRecoil(sender, receiver, 1, 6, sim.w, sim.h, 0);
      expect(sender.vx * sender.mass + receiver.vx * receiver.mass, 'no momentum minted').toBeCloseTo(0, 9);
      run(sim, params, 4);
      return (sender.x + receiver.x) / 2 - before;
    };

    expect(Math.abs(travel(0)), 'a uniform rate cannot move the centre').toBeLessThan(1e-6);
    // Positive: the full sender grips and the empty receiver slides, so the
    // pair walks the way the energy went — toward its own hungry end.
    expect(travel(2), 'positive grip walks with the flow').toBeGreaterThan(1);
    // Negative: the sender slides instead, and the pair walks back toward its
    // supply, which is the direction `transportThrust` already drifts it.
    expect(travel(-0.4), 'negative grip walks against it').toBeLessThan(-1);
  });

  it('carries a wired pair, because the rope moves bodies and not their centre', () => {
    /*
     * The configuration the mechanic is actually for. A wire pulls the two
     * back together after the kick, by a mass-weighted position correction —
     * which is exactly the operation that leaves a centre of mass where it
     * was. So the stroke survives being tied together, and a net can swim.
     */
    const displace = (grip: number): number => {
      const params = stillParams();
      params.grip = grip;
      const sim = new Sim(4000, 4000);
      const sender = sim.spawn('con', 1960, 2000, 0, params, true)!;
      const receiver = sim.spawn('con', 2040, 2000, 0, params, true)!;
      sim.graph.attach({ id: sender.id, slot: 'l' }, { id: receiver.id, slot: 'r' }, 80, 0);
      let before = 0;
      let after = 0;
      for (let f = 0; f < 300; f++) {
        // A pump every tenth of a second, always the same way down the wire,
        // as a net holding a standing gradient does.
        if (f % 6 === 0) {
          sender.extra = sender.energyCap;
          receiver.extra = 0;
          applyTransportRecoil(sender, receiver, 1, 6, sim.w, sim.h, 0);
        }
        if (f === 60) before = (sender.x + receiver.x) / 2;
        sim.step(1 / 60, params);
      }
      after = (sender.x + receiver.x) / 2;
      return after - before;
    };

    expect(Math.abs(displace(0)), 'tied together and pumping, and going nowhere').toBeLessThan(0.5);
    expect(displace(2), 'the same pump, now a stroke').toBeGreaterThan(2);
  });
});

describe('the gait', () => {
  /**
   * The stroke, with no pump at all.
   *
   * `grip` needs an impulse to work on: it turns a transport kick into travel
   * by letting the two ends coast different distances. The gait needs none.
   * Every body carries a phase; its cosine goes into the drag rate and into
   * an equal and opposite impulse along each of its wires.
   *
   * The pair's centre keeps `∮ F (1/k_a - 1/k_b) dt / M`, so the whole thing
   * is driven by the *difference* between a wire's two ends, and there are
   * exactly three places that difference can come from. These tests take them
   * one at a time, which is why `phase` is settable: scattering it at birth
   * is itself one of the three, and it would otherwise contaminate the other
   * two.
   */
  const walk = (
    over: Partial<Params> = {},
    opts: { kindB?: 'con' | 'era'; fullB?: number; phase?: number; flip?: boolean } = {},
  ): number => {
    const params = stillParams();
    params.gaitRate = 2;
    params.grip = 2;
    Object.assign(params, over);
    const sim = new Sim(4000, 4000);
    const a = sim.spawn('con', 1960, 2000, 0, params, true)!;
    const b = sim.spawn(opts.kindB ?? 'con', 2040, 2000, 0, params, true)!;
    if (opts.phase !== undefined) {
      a.gaitPhase = opts.phase;
      b.gaitPhase = opts.phase;
    }
    if (opts.flip) {
      a.chem[G_BASE + 1] = -a.chem[G_BASE + 1];
      b.chem[G_BASE + 1] = -b.chem[G_BASE + 1];
    }
    // An Era has only a principal, and two principals are a redex — so a
    // limb lands on an auxiliary port. See `Sim.collectReadyRedexes`.
    sim.graph.attach({ id: b.id, slot: opts.kindB === 'era' ? 'p' : 'r' }, { id: a.id, slot: 'l' }, 80, 0);
    const hold = () => {
      a.extra = a.energyCap;
      b.extra = b.energyCap * (opts.fullB ?? 0);
    };
    for (let f = 0; f < 300; f++) {
      hold();
      sim.step(1 / 60, params);
    }
    const before = (a.x + b.x) / 2;
    for (let f = 0; f < 60 * 20; f++) {
      hold();
      sim.step(1 / 60, params);
    }
    return (a.x + b.x) / 2 - before;
  };

  it('walks a wired pair on its own clock, with nothing pumping', () => {
    expect(Math.abs(walk({ gaitRate: 0 })), 'no clock, no stroke').toBeLessThan(0.05);
    expect(Math.abs(walk()), 'the clock alone carries the pair').toBeGreaterThan(0.5);
  });

  it('goes nowhere when a wire has two identical ends', () => {
    /*
     * The null the whole mechanism rests on. Same kind, same tank, same
     * phase: `k_a` and `k_b` are equal at every instant, the difference in
     * the integral is identically zero, and no amount of stroking moves the
     * centre. Everything below is a way of breaking exactly this.
     */
    expect(Math.abs(walk({}, { fullB: 1, phase: 0 })), 'two of the same').toBeLessThan(0.05);
    expect(Math.abs(walk({ grip: 0 }, { phase: 0 })), 'and with no grip to differ in').toBeLessThan(0.05);
  });

  it('takes its asymmetry from any of three places, and they are separable', () => {
    /*
     * Fullness is the economic one: `grip * fullness` differs, so the two
     * ends damp differently and the net walks only while it holds a
     * gradient. Phase is the accidental one — `createAgent` scatters it, so
     * two bodies are at different points of the same cosine. Kind is the
     * structural one, and the reason an Era is a limb: it strokes and does
     * not grip, where a Con grips and barely strokes.
     */
    const fullness = walk({}, { fullB: 0, phase: 0 });
    const phase = walk({}, { fullB: 1 });
    const kind = walk({}, { kindB: 'era', fullB: 1, phase: 0 });
    expect(Math.abs(fullness), 'a gradient across the wire').toBeGreaterThan(0.5);
    expect(Math.abs(phase), 'a phase difference across the wire').toBeGreaterThan(0.1);
    expect(Math.abs(kind), 'an Era on the end of it').toBeGreaterThan(0.5);
  });

  it('walks best on an Era, which is what makes a leaf a limb', () => {
    /*
     * The claim about body plan. Both pairs are fed identically and held at
     * one phase, so fullness and phase are both out of it and the only thing
     * left is what is on the end of the wire. An Era leaf should beat a
     * second Con by a wide margin — it strokes seven times as hard, grips a
     * tenth as much, and weighs half.
     */
    const era = Math.abs(walk({}, { kindB: 'era', fullB: 1, phase: 0 }));
    const con = Math.abs(walk({}, { kindB: 'con', fullB: 1, phase: 0 }));
    expect(con, 'two interior bodies have nothing to differ in').toBeLessThan(0.05);
    expect(era / Math.max(con, 1e-6), 'an Era leaf is the limb').toBeGreaterThan(10);
  });

  it('reverses when the stroke does, which is what selection has to steer by', () => {
    const fwd = walk({}, { kindB: 'era', fullB: 1, phase: 0 });
    const back = walk({}, { kindB: 'era', fullB: 1, phase: 0, flip: true });
    expect(Math.abs(back), 'still walking').toBeGreaterThan(0.5);
    expect(Math.sign(back), 'the other way').not.toBe(Math.sign(fwd));
  });
});
