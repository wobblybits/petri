import { describe, expect, it } from 'vitest';
import { boundRadius, portWorld, stemWorld, seedChem} from './agents.ts';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { queryHit, SLOP } from './collide.ts';
import { closestPointOnSegment, WIRE_RADIUS } from './geom.ts';
import { mixScent, scentSlowFactor, scentTurnBoost, Sim } from './sim.ts';
import { nativeSolver } from './native/solver.ts';
import { CH, FIELD_CELLS, FIELD_EXTENT } from './fields.ts';
import { EMIT, bareBody } from './agents.ts';
import { angleDelta } from './wrap.ts';
import { AGENT_BAND, WIRE_HAIRLINE_PX, WIRE_STROKE_PX, wiresDrawable } from './audio/lod.ts';

/** Zoom at which a wire stroke falls to a hairline and the rope is dropped. */
const hairlineZoom = WIRE_HAIRLINE_PX / WIRE_STROKE_PX;

function fastParams() {
  const params = defaultParams();
  params.wireShrink = 0.08;
  params.wireMinRest = 40;
  params.rewriteDuration = 0.12;
  params.springK = 90;
  params.spawnInterval = 0;
  params.upkeep = 0;
  return params;
}

function step(sim: Sim, params: ReturnType<typeof fastParams>, n: number): void {
  for (let i = 0; i < n; i++) sim.step(1 / 60, params);
}

describe('scent steering', () => {
  it('scentSlowFactor and scentTurnBoost trade linear speed for rotation', () => {
    expect(scentSlowFactor(0)).toBe(1);
    expect(scentTurnBoost(0)).toBe(1);
    const trail = 56;
    expect(scentSlowFactor(trail)).toBeLessThan(0.4);
    expect(scentTurnBoost(trail)).toBeGreaterThan(3);
  });

  it('does not mix an agent’s own principal channel', () => {
    const params = defaultParams();
    // Kind now only seeds the weights; a body senses through its own genome.
    // mixScent reads through the body, because the weights are modulated by
    // its inner state; a bare genome is not enough to ask the question with.
    const body = (kind: 'con' | 'dup' | 'era') =>
      bareBody(seedChem(kind, params)) as unknown as Parameters<typeof mixScent>[0];
    const dup = body('dup');
    const con = body('con');
    const era = body('era');
    // A Dup ignores dup-scent and a Con ignores con-scent: each seeks the kind
    // that completes a redex with it, not its own.
    expect(mixScent(dup, 1, 50, 0, 3)).toBe(mixScent(dup, 1, 0, 0, 3));
    expect(mixScent(con, 50, 2, 0, 3)).toBe(mixScent(con, 0, 2, 0, 3));
    expect(mixScent(era, 1, 2, 0, 3)).toBeGreaterThan(mixScent(era, 0, 0, 0, 3));
  });

  it('goes straight when there is nothing to smell', () => {
    // Locomotion symmetry, isolated from the field: with no scent at all both
    // sensors read zero, so anything that turns the body is a thrust that is
    // not along its heading. Resolution-independent by construction.
    for (const kind of ['era', 'con', 'dup'] as const) {
      const sim = new Sim(240, 160);
      const params = defaultParams();
      params.faceAttract = 0;
      params.snapWell = 0;
      params.snapRadius = 0;
      params.deposit = 0;
      // Barren, because the ground is a channel now and every kind is seeded
      // with a taste for it. A body standing on full ground eats a dip under
      // itself within a frame or two and then smells the dip — which is a real
      // behaviour and the wrong one to have inside a test whose whole premise
      // is that there is nothing to smell.
      params.ambientEnergy = 0;
      // The swimming kick is coloured noise off Math.random, and it moves the
      // agent, which moves where it lays scent, which is what the two sensors
      // read. Left in, this test asks whether an unseeded random walk happened
      // to stay symmetric to five places.
      params.swimNoise = 0;
      const heading = 0.4;
      const agent = sim.spawn(kind, 120, 80, heading, params, true)!;
      step(sim, params, 45);
      expect(agent.heading, kind).toBeCloseTo(heading, 5);
      const hx = Math.cos(heading);
      const hy = Math.sin(heading);
      const along = agent.vx * hx + agent.vy * hy;
      expect(agent.vx, kind).toBeCloseTo(along * hx, 5);
      expect(agent.vy, kind).toBeCloseTo(along * hy, 5);
    }
  });

  it('wanders on its own trail without winding up', () => {
    /*
     * A body laying scent and then smelling it does not hold a straight line,
     * and on a fixed grid it cannot: its deposit is splatted bilinearly onto
     * cells that are not symmetric about a heading of 0.4, so the two sensors
     * read slightly different values and it turns. This used to be asserted
     * away — the old version of this test ran at 1.5-unit cells, finer than
     * the app was ever rendered at, where the artifact vanished into the fifth
     * decimal place.
     *
     * What actually matters is that the wander is bounded rather than a spin:
     * measured over fifteen seconds a body drifts either way and comes back
     * (one goes +0.08, +1.78, -1.73, +0.67 radians) instead of accumulating.
     */
    for (const kind of ['era', 'con', 'dup'] as const) {
      const sim = new Sim(240, 160);
      const params = defaultParams();
      params.faceAttract = 0;
      params.snapWell = 0;
      params.snapRadius = 0;
      params.swimNoise = 0;
      const start = 0.4;
      const agent = sim.spawn(kind, 120, 80, start, params, true)!;
      let worst = 0;
      for (let f = 0; f < 900; f++) {
        sim.step(1 / 60, params);
        worst = Math.max(worst, Math.abs(agent.heading - start));
      }
      /*
       * Net drift, not total turning. A body correcting constantly racks up a
       * lot of absolute rotation — measured around 18 radians over fifteen
       * seconds — while going nowhere in particular, and that is the wander
       * this is meant to permit. Winding up is the failure: heading walking off
       * in one direction and never coming back.
       */
      // Measured around 3.5 radians — a body does turn right around and come
      // back. A body actually spinning racks up its turn rate times fifteen
      // seconds, which is well past this even at a lazy one radian a second.
      expect(
        worst,
        `${kind} strayed ${worst.toFixed(2)} rad from its heading in 15s`,
      ).toBeLessThan(2 * Math.PI);
    }
  });

  it('does not self-propel when the principal port is latched', () => {
    const sim = new Sim(320, 200);
    const params = defaultParams();
    params.flockAlign = 0;
    params.flockSep = 0;
    params.snapRadius = 0;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    params.stepSpeed = 50;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 200, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 30);
    expect(Math.hypot(a.vx, a.vy)).toBeLessThan(18);
    expect(Math.hypot(b.vx, b.vy)).toBeLessThan(18);
  });

  it('pushes overlapping free agents apart', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('con', 120, 80, 0, params, true)!;
    const b = sim.spawn('con', 121, 80, Math.PI, params, true)!;
    step(sim, params, 12);
    expect(queryHit(a, b, sim.w, sim.h)?.overlap ?? 0).toBeLessThanOrEqual(SLOP + 0.08);
  });

  it('pushes overlapping wired agents apart', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    params.rewriteDuration = 20;
    const a = sim.spawn('era', 120, 80, 0, params, true)!;
    const b = sim.spawn('era', 121, 80, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 24);
    expect(queryHit(a, b, sim.w, sim.h)?.overlap ?? 0).toBeLessThan(2);
  });

  it('collision changes linear velocity', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('con', 100, 80, 0, params, true)!;
    const b = sim.spawn('con', 118, 80, Math.PI, params, true)!;
    a.vx = 60;
    sim.step(1 / 60, params);
    expect(a.vx).toBeLessThan(60);
    expect(b.vx).toBeGreaterThan(0);
  });

  it('a rope drapes around a visitor instead of cutting through it', () => {
    const sim = new Sim(320, 200);
    const params = defaultParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.declutter = 0;
    params.rewriteDuration = 20;
    params.spawnInterval = 0;
    const a = sim.spawn('era', 60, 100, 0, params, true)!;
    const b = sim.spawn('era', 260, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    expect(wire.nodes.length).toBeGreaterThan(2);
    const mid = wire.nodes[Math.floor(wire.nodes.length / 2)];
    const visitor = sim.spawn('con', mid.x, mid.y, 0, params, true)!;
    const vx0 = visitor.x;
    const vy0 = visitor.y;
    step(sim, params, 30);
    const pts = [
      stemWorld(a, 'p', sim.w, sim.h),
      ...wire.nodes,
      stemWorld(b, 'p', sim.w, sim.h),
    ];
    let dist = Infinity;
    for (let i = 0; i < pts.length - 1; i++) {
      const q = closestPointOnSegment(
        visitor.x,
        visitor.y,
        pts[i].x,
        pts[i].y,
        pts[i + 1].x,
        pts[i + 1].y,
      );
      dist = Math.min(dist, Math.hypot(q.x - visitor.x, q.y - visitor.y));
    }
    const keep = boundRadius(visitor) + WIRE_RADIUS;
    expect(dist).toBeGreaterThan(keep - 4);
    expect(Math.hypot(visitor.x - vx0, visitor.y - vy0)).toBeLessThan(3);
  });
});

describe('simulation presets', () => {
  it('seeds a soup with the configured agent count', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    expect(params.spawnInterval).toBe(0.5);
    loadPreset(sim, 'soup', params);
    expect(sim.agents.size).toBe(params.soupCount);
  });

  it('steps a soup without throwing', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    params.soupCount = 28;
    loadPreset(sim, 'soup', params);
    expect(sim.agents.size).toBeGreaterThan(0);
    step(sim, params, 30);
    expect(sim.agents.size).toBeGreaterThan(0);
  });

  it('auto-spawns a free agent about every ten seconds', () => {
    const sim = new Sim(400, 240);
    const params = defaultParams();
    params.spawnInterval = 10;
    params.snapRadius = 0;
    params.maxAgents = 80;
    sim.spawn('era', 200, 120, 0, params, true);
    const n0 = sim.agents.size;
    step(sim, params, 599);
    expect(sim.agents.size).toBe(n0);
    step(sim, params, 2);
    expect(sim.agents.size).toBe(n0 + 1);
  });

  /*
   * A filled port stops advertising a socket. It does not stop the body
   * talking — those were one rule and are now two.
   *
   * This used to assert that a wired pair laid down nothing at all, and it
   * passed for a reason that stopped being the reason: an Era says nothing at
   * seed, so a mute pair emitting its voice looks exactly like a pair that
   * cannot emit. Both Eras are given a voice here so the two halves can be
   * told apart.
   */
  it('keeps a wired body audible, but stops its filled ports marking', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.decay = 0;
    params.diffuse = 0;
    params.deposit = 4;
    params.wireShrink = 10;
    params.rewriteDuration = 10;
    params.ambientEnergy = 0;
    const a = sim.spawn('era', 80, 80, 0, params, true)!;
    const b = sim.spawn('era', 160, 80, Math.PI, params, true)!;
    for (const e of [a, b]) {
      for (let k = 0; k < 4; k++) e.chem[EMIT + k] = 0;
      e.chem[EMIT + CH.conP] = 1;
    }
    // Two single-ported bodies wired to each other: between them there is not
    // one free port in the world.
    sim.wire(a.id, 'p', b.id, 'p', params);
    sim.fields.clear();
    step(sim, params, 8);
    expect(sim.fields.peak(CH.conP), 'a wired net went silent').toBeGreaterThan(0);
    expect(sim.fields.peak(CH.aux), 'a filled port advertised a socket').toBe(0);
  });

  it('starving agents still deposit from free ports', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.decay = 0;
    params.diffuse = 0;
    params.deposit = 4;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    params.spawnInterval = 0;
    params.rewriteDuration = 20;
    params.stepSpeed = 0;
    // A Con, because an Era says nothing at seed — its voice used to go on
    // the channel the ground now lives on, and nothing emits onto the ground.
    const a = sim.spawn('con', 80, 80, 0, params, true)!;
    a.extra = -0.5;
    sim.fields.clear();
    step(sim, params, 8);
    const p = portWorld(a, 'p', sim.w, sim.h);
    expect(sim.fields.sample(CH.conP, p.x, p.y)).toBeGreaterThan(0.5);
  });

  it('rewrites a principal meeting even when aux ports are still free', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    const c = sim.spawn('con', 200, 160, 0, params, true)!;
    const d = sim.spawn('dup', 280, 160, Math.PI, params, true)!;
    c.extra = 1;
    d.extra = 1;
    sim.wire(c.id, 'p', d.id, 'p', params);
    expect(sim.graph.portsFilled(c)).toBe(false);
    step(sim, params, 50);
    // Commute: the original pair is gone, four agents remain.
    expect(sim.rewrites.length + sim.agents.size).toBeGreaterThan(2);
    expect(sim.agents.size).not.toBe(2);
  });

  it('snap joins facing ports and only once', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 28;
    params.snapArc = 0.45;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    const a = sim.spawn('era', 90, 80, 0, params, true)!;
    const b = sim.spawn('era', 130, 80, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    const between = [...sim.graph.wires.values()].filter(
      (w) =>
        (w.a.id === a.id && w.b.id === b.id) || (w.a.id === b.id && w.b.id === a.id),
    );
    expect(between.length).toBe(1);
  });

  it('does not latch when snap reach is zero', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.rewriteDuration = 20;
    sim.spawn('era', 90, 80, 0, params, true);
    sim.spawn('era', 110, 80, Math.PI, params, true);
    sim.step(1 / 60, params);
    expect(sim.graph.wires.size).toBe(0);
  });

  it('starving agents still snap', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 28;
    params.snapArc = 0.45;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    const a = sim.spawn('era', 90, 80, 0, params, true)!;
    const b = sim.spawn('era', 130, 80, Math.PI, params, true)!;
    a.extra = -0.5;
    b.extra = -0.5;
    sim.step(1 / 60, params);
    expect(sim.graph.wires.size).toBe(1);
    expect(a.extra).toBeLessThan(0);
    expect(b.extra).toBeLessThan(0);
  });

  it('does not snap ports that are close but not facing or touching', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 90;
    params.snapArc = 0.3;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    sim.spawn('con', 120, 80, 0, params, true);
    sim.spawn('con', 128, 80, 0, params, true);
    sim.step(1 / 60, params);
    expect(sim.graph.wires.size).toBe(0);
  });

  it('latches when free port tips touch even outside the snap arc', () => {
    const sim = new Sim(320, 200);
    const params = defaultParams();
    params.snapRadius = 40;
    params.snapArc = 0.12;
    params.wireShrink = 20;
    params.rewriteDuration = 20;
    // Same heading: not facing. Tips nearly coincident.
    const a = sim.spawn('era', 100, 100, 0, params, true)!;
    const b = sim.spawn('era', 100, 103, 0, params, true)!;
    a.extra = 1;
    b.extra = 1;
    sim.latchPass(params);
    expect(sim.graph.wires.size).toBe(1);
    expect(sim.graph.isFree({ id: a.id, slot: 'p' })).toBe(false);
    expect(sim.graph.isFree({ id: b.id, slot: 'p' })).toBe(false);
  });

  it('γ–δ commutation copies into two cons and two dups before further reduction', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    loadPreset(sim, 'commute', params);
    expect(sim.agents.size).toBe(6);
    let sawCopy = false;
    for (let i = 0; i < 120; i++) {
      sim.step(1 / 60, params);
      const kinds = [...sim.agents.values()].map((a) => a.kind);
      if (
        kinds.filter((k) => k === 'con').length === 2 &&
        kinds.filter((k) => k === 'dup').length === 2 &&
        kinds.filter((k) => k === 'era').length === 4
      ) {
        sawCopy = true;
        break;
      }
    }
    expect(sawCopy).toBe(true);
  });

  it('γ–γ annihilation consumes both constructors', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    loadPreset(sim, 'annihilate-con', params);
    step(sim, params, 120);
    expect([...sim.agents.values()].every((a) => a.kind !== 'con')).toBe(true);
  });

  it('δ–δ annihilation consumes both duplicators', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    loadPreset(sim, 'annihilate-dup', params);
    step(sim, params, 120);
    const kinds = [...sim.agents.values()].map((a) => a.kind);
    expect(kinds.every((k) => k === 'era')).toBe(true);
  });

  it('oscillator keeps a net after the first rewrite', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    loadPreset(sim, 'oscillator', params);
    expect(sim.agents.size).toBe(4);
    step(sim, params, 90);
    expect(sim.agents.size).toBeGreaterThan(0);
  });

  it('tracks the mass-weighted center of all shapes', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    const a = sim.spawn('era', 0, 0, 0, params, true)!;
    const b = sim.spawn('era', 100, 0, 0, params, true)!;
    const com = sim.centerOfMass()!;
    const m = a.mass + b.mass;
    expect(com.x).toBeCloseTo((a.mass * 0 + b.mass * 100) / m);
    expect(com.y).toBeCloseTo(0);
  });

});

describe('conservative mechanics', () => {
  function passiveParams() {
    const params = defaultParams();
    params.stepSpeed = 0;
    params.turnRate = 0;
    params.snapRadius = 0;
    params.snapWell = 0;
    params.faceAttract = 0;
    params.drag = 0;
    params.angDrag = 0;
    params.deposit = 0;
    params.rewriteDuration = 20;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.spawnInterval = 0;
    return params;
  }

  it('conserves linear and angular momentum in a collision', () => {
    const sim = new Sim(400, 200);
    const params = passiveParams();
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 140, 104, Math.PI, params, true)!;
    a.vx = 50;
    a.omega = 2;
    const before = sim.momentum();
    const e0 = sim.kineticEnergy();
    step(sim, params, 40);
    const after = sim.momentum();
    expect(after.px).toBeCloseTo(before.px, 2);
    expect(after.py).toBeCloseTo(before.py, 2);
    expect(after.L).toBeCloseTo(before.L, 1);
    expect(sim.kineticEnergy()).toBeLessThan(e0 * 1.35);
    expect(sim.kineticEnergy()).toBeGreaterThan(e0 * 0.45);
    expect(b.id).toBeGreaterThan(0);
  });

  it('does not latch through an intervening wire', () => {
    const sim = new Sim(400, 240);
    const params = passiveParams();
    params.snapRadius = 90;
    params.snapArc = 0.6;
    const wallA = sim.spawn('era', 200, 40, Math.PI / 2, params, true)!;
    const wallB = sim.spawn('era', 200, 200, -Math.PI / 2, params, true)!;
    sim.wire(wallA.id, 'p', wallB.id, 'p', params);
    const left = sim.spawn('era', 140, 120, 0, params, true)!;
    const right = sim.spawn('era', 260, 120, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    const between = [...sim.graph.wires.values()].filter(
      (w) =>
        (w.a.id === left.id && w.b.id === right.id) ||
        (w.a.id === right.id && w.b.id === left.id),
    );
    expect(between).toHaveLength(0);
    expect(sim.graph.wires.size).toBe(1);
  });

  it('allows wire chains to cross without pushing each other', () => {
    const sim = new Sim(400, 240);
    const params = passiveParams();
    params.springK = 40;
    const h1 = sim.spawn('era', 60, 120, 0, params, true)!;
    const h2 = sim.spawn('era', 340, 120, Math.PI, params, true)!;
    const v1 = sim.spawn('era', 200, 20, Math.PI / 2, params, true)!;
    const v2 = sim.spawn('era', 200, 220, -Math.PI / 2, params, true)!;
    sim.wire(h1.id, 'p', h2.id, 'p', params);
    sim.wire(v1.id, 'p', v2.id, 'p', params);
    step(sim, params, 20);
    const wires = [...sim.graph.wires.values()];
    expect(wires).toHaveLength(2);
    const mid = (w: (typeof wires)[0]) => w.nodes[Math.floor(w.nodes.length / 2)];
    const m0 = mid(wires[0]);
    const m1 = mid(wires[1]);
    expect(Math.hypot(m0.x - m1.x, m0.y - m1.y)).toBeLessThan(18);
  });

});

function quietParams() {
  const params = defaultParams();
  params.flockAlign = 0;
  params.flockSep = 0;
  params.snapRadius = 0;
  params.snapWell = 0;
  params.faceAttract = 0;
  params.deposit = 0;
  params.diffuse = 0;
  params.decay = 0;
  params.rewriteDuration = 20;
  params.wireShrink = 20;
  params.spawnInterval = 0;
  // These tests target deterministic mechanics — the scent-to-cruise mapping,
  // locomotion gating — not self-propulsion dynamics. Noise swamps single-run
  // comparisons and persistence delays them past their measurement window, so
  // both are turned off to isolate what is actually under test.
  params.swimNoise = 0;
  params.swimTau = 0.05;
  return params;
}

describe('isolated motion rules', () => {
  it('a free principal still cruises along its heading', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    step(sim, params, 40);
    expect(a.vx).toBeGreaterThan(12);
    expect(Math.abs(a.vy)).toBeLessThan(0.8);
  });

  it('wired eras pull together during the shrink phase', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.wireShrink = 0.45;
    const a = sim.spawn('era', 120, 120, 0, params, true)!;
    const b = sim.spawn('era', 260, 120, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const d0 = Math.hypot(b.x - a.x, b.y - a.y);
    step(sim, params, 20);
    const d1 = Math.hypot(b.x - a.x, b.y - a.y);
    expect(d1).toBeLessThan(d0 - 8);
  });

  it('a settled latch does not keep injecting kinetic energy', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 0;
    params.drag = 0.8;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 200, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 90);
    const e0 = sim.kineticEnergy();
    const v0 = Math.hypot(a.vx, a.vy) + Math.hypot(b.vx, b.vy);
    step(sim, params, 60);
    expect(sim.kineticEnergy()).toBeLessThan(e0 + 8);
    expect(Math.hypot(a.vx, a.vy) + Math.hypot(b.vx, b.vy)).toBeLessThan(v0 + 4);
  });

  it('does not bounce a constructor off its own wires', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.drag = 0.8;
    const con = sim.spawn('con', 200, 120, 0, params, true)!;
    const left = sim.spawn('era', 140, 90, Math.PI, params, true)!;
    const right = sim.spawn('era', 140, 150, Math.PI, params, true)!;
    const face = sim.spawn('era', 280, 120, Math.PI, params, true)!;
    sim.wire(con.id, 'l', left.id, 'p', params);
    sim.wire(con.id, 'r', right.id, 'p', params);
    sim.wire(con.id, 'p', face.id, 'p', params);
    step(sim, params, 90);
    expect(Math.hypot(con.vx, con.vy)).toBeLessThan(28);
    expect(Math.abs(con.omega)).toBeLessThan(8);
  });

  it('wires do not spontaneously spin a settled pair', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 0;
    params.drag = 1.2;
    params.angDrag = 4;
    const a = sim.spawn('era', 100, 100, 0.2, params, true)!;
    const b = sim.spawn('era', 220, 100, Math.PI - 0.2, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 120);
    const h0 = a.heading;
    const o0 = Math.abs(a.omega) + Math.abs(b.omega);
    step(sim, params, 60);
    expect(Math.abs(a.omega) + Math.abs(b.omega)).toBeLessThan(o0 + 0.4);
    expect(Math.abs(a.heading - h0)).toBeLessThan(0.35);
  });

  it('principal–principal wires tend to 180° heading alignment', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.flockAlign = 14;
    params.angDrag = 0.6;
    params.turnRate = 0;
    const a = sim.spawn('era', 120, 120, 0.35, params, true)!;
    const b = sim.spawn('era', 240, 120, 0.5, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    step(sim, params, 160);
    expect(Math.abs(Math.abs(angleDelta(a.heading, b.heading)) - Math.PI)).toBeLessThan(0.5);
  });

  it('principal–aux wires tend to 0° heading alignment', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.flockAlign = 14;
    params.angDrag = 0.6;
    params.turnRate = 0;
    const con = sim.spawn('con', 200, 120, 0.2, params, true)!;
    const era = sim.spawn('era', 140, 120, Math.PI * 0.7, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    step(sim, params, 160);
    expect(Math.abs(angleDelta(con.heading, era.heading))).toBeLessThan(0.5);
  });

  it('aux–aux wires tend to 180° heading alignment', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    params.flockAlign = 16;
    params.angDrag = 0.4;
    params.turnRate = 0;
    const a = sim.spawn('con', 160, 120, 0.15, params, true)!;
    const b = sim.spawn('con', 250, 120, Math.PI + 0.45, params, true)!;
    sim.wire(a.id, 'l', b.id, 'r', params);
    step(sim, params, 220);
    // Both aux ports aim off their neighbour by params.auxSpread * 0.35 rad
    // so wires keep to their own side, which costs exact antiparallelism.
    const auxSplay = params.auxSpread * 0.35;
    expect(Math.abs(Math.abs(angleDelta(a.heading, b.heading)) - Math.PI)).toBeLessThan(
      0.7 + 2 * auxSplay,
    );
  });

  it('stronger scent slows free cruise', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.deposit = 0;
    const clear = sim.spawn('era', 80, 60, 0, params, true)!;
    step(sim, params, 40);
    const vClear = Math.hypot(clear.vx, clear.vy);
    sim.clear();
    const thick = sim.spawn('era', 80, 60, 0, params, true)!;
    for (let i = 0; i < 40; i++) {
      for (let dy = -12; dy <= 12; dy += 4) {
        sim.fields.deposit(CH.conP, 80 + i * 3, 60 + dy, 18);
      }
    }
    step(sim, params, 40);
    const vThick = Math.hypot(thick.vx, thick.vy);
    expect(vClear).toBeGreaterThan(12);
    expect(vThick).toBeLessThan(vClear * 0.75);
  });

  it('a starving agent still turns toward scent', () => {
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.sense = 800;
    params.ambientEnergy = 0;
    params.upkeep = 0;
    /*
     * A broad gradient, not a bright point.
     *
     * Steering compares two sensors 22 world units apart against a dead zone
     * of 5% of the signal's own magnitude. On the world-fixed field a cell is
     * 160 units, so those sensors straddle a seventh of one cell: a sharp
     * strong blob gives a large magnitude and so a large dead zone, while the
     * difference across the sensors stays small, and the turn is swallowed. A
     * gradient spread over several cells is what the field can represent and
     * what an agent can actually climb.
     */
    const cell = FIELD_EXTENT / FIELD_CELLS;
    const bias = (s: Sim) => {
      // Deposits are normalised to a density, so dividing by that scale asks
      // for a field *value* and keeps the test independent of resolution.
      const unit = 1.5 / s.fields.depositScale;
      for (let r = 0; r < cell * 3; r += cell * 0.25) {
        for (let ang = 0; ang < 6.283; ang += 0.3) {
          s.fields.deposit(
            CH.conP,
            // Within sensing range and off the heading axis. These tests run
            // with diffusion off, so the field is only what was splatted: put
            // the source further than a couple of cells and nothing of it
            // reaches the sensors at all. Dead ahead would also read the same
            // on both sensors, and the body would correctly not turn.
            80 + cell * 1.5 + Math.cos(ang) * r,
            60 + cell * 1.5 + Math.sin(ang) * r,
            unit / (1 + r / cell),
          );
        }
      }
    };

    const fed = new Sim(320, 200);
    const a = fed.spawn('era', 80, 60, 0, params, true)!;
    a.extra = 1;
    bias(fed);
    step(fed, params, 36);

    const starved = new Sim(320, 200);
    const b = starved.spawn('era', 80, 60, 0, params, true)!;
    b.extra = -0.5;
    bias(starved);
    step(starved, params, 36);

    expect(Math.abs(a.heading), `fed heading ${a.heading.toFixed(3)}`).toBeGreaterThan(0.08);
    expect(Math.abs(b.heading), `starved heading ${b.heading.toFixed(3)}`).toBeGreaterThan(0.08);
  });

  it('stronger scent increases turn agility while slowing cruise', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.deposit = 0;
    /*
     * The same broad gradient as the test above.
     *
     * Steering compares two sensors 22 world units apart against a dead zone
     * of 5% of the signal's own magnitude. On the world-fixed field a cell is
     * 160 units, so those sensors straddle a seventh of one cell: a sharp
     * strong blob gives a large magnitude and so a large dead zone, while the
     * difference across the sensors stays small, and the turn is swallowed. A
     * gradient spread over several cells is what the field can represent and
     * what an agent can actually climb.
     */
    const cell = FIELD_EXTENT / FIELD_CELLS;
    const bias = (s: Sim) => {
      // Deposits are normalised to a density, so dividing by that scale asks
      // for a field *value* and keeps the test independent of resolution.
      const unit = 1.5 / s.fields.depositScale;
      for (let r = 0; r < cell * 3; r += cell * 0.25) {
        for (let ang = 0; ang < 6.283; ang += 0.3) {
          s.fields.deposit(
            CH.conP,
            // Within sensing range and off the heading axis. These tests run
            // with diffusion off, so the field is only what was splatted: put
            // the source further than a couple of cells and nothing of it
            // reaches the sensors at all. Dead ahead would also read the same
            // on both sensors, and the body would correctly not turn.
            80 + cell * 1.5 + Math.cos(ang) * r,
            60 + cell * 1.5 + Math.sin(ang) * r,
            unit / (1 + r / cell),
          );
        }
      }
    };
    const clear = sim.spawn('era', 80, 60, 0, params, true)!;
    bias(sim);
    step(sim, params, 20);
    const omegaClear = Math.abs(clear.omega);
    sim.clear();
    const thick = sim.spawn('era', 80, 60, 0, params, true)!;
    // A thick uniform bed the body sits inside, under the same gradient.
    const bed = 26 / sim.fields.depositScale;
    for (let j = -4; j <= 4; j++) {
      for (let i = -4; i <= 4; i++) {
        sim.fields.deposit(CH.conP, 80 + i * cell, 60 + j * cell, bed);
      }
    }
    bias(sim);
    step(sim, params, 20);
    const vThick = Math.hypot(thick.vx, thick.vy);
    const omegaThick = Math.abs(thick.omega);
    expect(vThick).toBeLessThan(Math.hypot(clear.vx, clear.vy) * 0.85);
    expect(omegaThick).toBeGreaterThan(omegaClear * 1.15);
  });

  it('an aux-only wire does not disable constructor locomotion', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.springK = 6;
    params.flockAlign = 5;
    const con = sim.spawn('con', 180, 120, 0, params, true)!;
    const era = sim.spawn('era', 140, 120, Math.PI, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    expect(sim.graph.isFree({ id: con.id, slot: 'p' })).toBe(true);
    expect(sim.graph.isFree({ id: era.id, slot: 'p' })).toBe(false);
    step(sim, params, 35);
    expect(con.vx).toBeGreaterThan(10);
    expect(Math.abs(con.omega)).toBeLessThan(5);
  });

  it('towing does not penalize cruise speed with component mass', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 48;
    params.drag = 0.2;
    params.flockAlign = 5;
    const solo = sim.spawn('con', 100, 120, 0, params, true)!;
    step(sim, params, 40);
    const vSolo = solo.vx;
    sim.clear();
    const con = sim.spawn('con', 180, 120, 0, params, true)!;
    const era = sim.spawn('era', 140, 120, Math.PI, params, true)!;
    sim.wire(con.id, 'l', era.id, 'p', params);
    step(sim, params, 40);
    expect(con.vx).toBeGreaterThan(vSolo * 0.5);
    expect(Math.abs(con.omega)).toBeLessThan(5);
  });

  it('does not apply flock separation to disconnected agents', () => {
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.flockSep = 90;
    params.flockAlign = 8;
    params.stepSpeed = 0;
    const a = sim.spawn('era', 100, 100, 0, params, true)!;
    const b = sim.spawn('era', 145, 100, Math.PI, params, true)!;
    const d0 = Math.hypot(b.x - a.x, b.y - a.y);
    step(sim, params, 25);
    const d1 = Math.hypot(b.x - a.x, b.y - a.y);
    expect(Math.abs(d1 - d0)).toBeLessThan(3);
  });

  it('reports a blocking wire on a would-be latch chord', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    const wallA = sim.spawn('era', 200, 40, Math.PI / 2, params, true)!;
    const wallB = sim.spawn('era', 200, 200, -Math.PI / 2, params, true)!;
    sim.wire(wallA.id, 'p', wallB.id, 'p', params);
    const left = sim.spawn('era', 140, 120, 0, params, true)!;
    const right = sim.spawn('era', 260, 120, Math.PI, params, true)!;
    expect(
      sim.graph.latchCrosses(sim.agents, { id: left.id, slot: 'p' }, { id: right.id, slot: 'p' }, sim.w, sim.h),
    ).toBe(true);
  });

  it('does not treat a clear gap as a crossing latch', () => {
    const sim = new Sim(400, 240);
    const params = quietParams();
    params.stepSpeed = 0;
    const wallA = sim.spawn('era', 80, 40, Math.PI / 2, params, true)!;
    const wallB = sim.spawn('era', 80, 200, -Math.PI / 2, params, true)!;
    sim.wire(wallA.id, 'p', wallB.id, 'p', params);
    const left = sim.spawn('era', 200, 120, 0, params, true)!;
    const right = sim.spawn('era', 280, 120, Math.PI, params, true)!;
    expect(
      sim.graph.latchCrosses(sim.agents, { id: left.id, slot: 'p' }, { id: right.id, slot: 'p' }, sim.w, sim.h),
    ).toBe(false);
  });
});

describe('physics lod', () => {
  const zoomedOut = { x: 120, y: 80, zoom: 0.18, viewW: 800, viewH: 600 };
  const closeUp = { x: 120, y: 80, zoom: 2, viewW: 400, viewH: 300 };

  it('does not inflate a FAR wired pair whose stems sit inside the bound discs', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    params.spawnInterval = 0;
    params.wireShrink = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.declutter = 0;
    const a = sim.spawn('con', 100, 80, 0, params, true)!;
    const b = sim.spawn('con', 122, 80, 0, params, true)!;
    sim.wire(a.id, 'r', b.id, 'l', params);
    for (let i = 0; i < 24; i++) sim.step(1 / 60, params, zoomedOut);
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    expect(dist).toBeLessThan(boundRadius(a) + boundRadius(b) - 4);
    expect(dist).toBeGreaterThan(8);
  });

  it('still separates overlapping agents when they are FAR', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('era', 120, 80, 0, params, true)!;
    const b = sim.spawn('era', 121, 80, Math.PI, params, true)!;
    for (let i = 0; i < 12; i++) sim.step(1 / 60, params, zoomedOut);
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(12);
  });

  it('does not emit Hertzian contacts for FAR pairs', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    sim.spawn('era', 120, 80, 0, params, true);
    sim.spawn('era', 121, 80, Math.PI, params, true);
    sim.step(1 / 60, params, zoomedOut);
    expect(sim.contacts.size).toBe(0);
  });

  it('keeps Hertzian contacts when the same pair is NEAR', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    sim.spawn('era', 120, 80, 0, params, true);
    sim.spawn('era', 121, 80, Math.PI, params, true);
    sim.step(1 / 60, params, closeUp);
    expect(sim.contacts.size).toBeGreaterThan(0);
  });

  it('stepAsync without a GPU device still separates FAR overlap', async () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('era', 120, 80, 0, params, true)!;
    const b = sim.spawn('era', 121, 80, Math.PI, params, true)!;
    for (let i = 0; i < 12; i++) await sim.stepAsync(1 / 60, params, zoomedOut);
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(12);
  });

  it('does not change the no-view path: overlapping triangles still sit on SAT', () => {
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('con', 120, 80, 0, params, true)!;
    const b = sim.spawn('con', 121, 80, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    expect(sim.contacts.size).toBeGreaterThan(0);
    for (let i = 0; i < 11; i++) sim.step(1 / 60, params);
    expect(queryHit(a, b, sim.w, sim.h)?.overlap ?? 0).toBeLessThanOrEqual(SLOP + 0.08);
  });

  it('does not promote a whole chain because one end is on screen', () => {
    const sim = new Sim(2000, 200);
    const params = defaultParams();
    params.snapRadius = 0;
    params.spawnInterval = 0;
    params.stepSpeed = 0;
    const ids: number[] = [];
    for (let i = 0; i < 16; i++) {
      ids.push(sim.spawn('era', 80 + i * 70, 100, 0, params, true)!.id);
      if (i > 0) sim.wire(ids[i - 1], 'p', ids[i], 'p', params);
    }
    sim.step(1 / 60, params, { x: 80, y: 100, zoom: 2, viewW: 400, viewH: 300 });
    expect(sim.isPhysicsDetailed(ids[0])).toBe(true);
    expect(sim.isPhysicsDetailed(ids[15])).toBe(false);
  });

  it('still promotes a close-up rewrite onto SAT so leftover ropes reel with the pull', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    loadPreset(sim, 'commute', params);
    const close = { x: 240, y: 160, zoom: 2, viewW: 400, viewH: 300 };
    let sawRewrite = false;
    for (let i = 0; i < 80; i++) {
      sim.step(1 / 60, params, close);
      if (sim.rewrites.length > 0) {
        sawRewrite = true;
        const rw = sim.rewrites[0];
        expect(sim.isPhysicsDetailed(rw.a)).toBe(true);
        expect(sim.isPhysicsDetailed(rw.b)).toBe(true);
        break;
      }
    }
    expect(sawRewrite).toBe(true);
  });

  it('keeps a FAR rewrite on the packed path instead of promoting leftover ropes', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    loadPreset(sim, 'commute', params);
    const far = { x: 240, y: 160, zoom: 0.05, viewW: 800, viewH: 600 };
    let sawRewrite = false;
    for (let i = 0; i < 80; i++) {
      sim.step(1 / 60, params, far);
      if (sim.rewrites.length > 0) {
        sawRewrite = true;
        for (const a of sim.agents.values()) {
          expect(sim.isPhysicsDetailed(a.id), `agent ${a.id} detailed during FAR rewrite`).toBe(false);
        }
        for (const w of sim.graph.wires.values()) {
          expect(sim.wireSimulatesRope(w), `wire ${w.id} live during FAR rewrite`).toBe(false);
        }
        break;
      }
    }
    expect(sawRewrite).toBe(true);
  });

  it('FAR commute still copies into two cons and two dups', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    loadPreset(sim, 'commute', params);
    const far = { x: 240, y: 160, zoom: 0.05, viewW: 800, viewH: 600 };
    let sawCopy = false;
    for (let i = 0; i < 120; i++) {
      sim.step(1 / 60, params, far);
      const kinds = [...sim.agents.values()].map((a) => a.kind);
      if (
        kinds.filter((k) => k === 'con').length === 2 &&
        kinds.filter((k) => k === 'dup').length === 2 &&
        kinds.filter((k) => k === 'era').length === 4
      ) {
        sawCopy = true;
        break;
      }
    }
    expect(sawCopy).toBe(true);
    for (const w of sim.graph.wires.values()) {
      expect(Number.isFinite(w.rest)).toBe(true);
      expect(w.rest).toBeLessThan(3000);
      expect(Number.isFinite(w.lastLen)).toBe(true);
      expect(w.lastLen).toBeLessThan(8000);
    }
  });

  it('FAR annihilate-con still consumes both constructors', () => {
    const sim = new Sim(480, 320);
    const params = fastParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    loadPreset(sim, 'annihilate-con', params);
    const far = { x: 240, y: 160, zoom: 0.05, viewW: 800, viewH: 600 };
    for (let i = 0; i < 120; i++) sim.step(1 / 60, params, far);
    expect([...sim.agents.values()].every((a) => a.kind !== 'con')).toBe(true);
  });

  it('still rewrites a FAR era–era pair', () => {
    const sim = new Sim(240, 160);
    const params = fastParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('era', 100, 80, 0, params, true)!;
    const b = sim.spawn('era', 140, 80, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const far = { x: 120, y: 80, zoom: 0.18, viewW: 800, viewH: 600 };
    for (let i = 0; i < 80; i++) sim.step(1 / 60, params, far);
    expect(sim.rewrites.length + (2 - sim.agents.size)).toBeGreaterThan(0);
    expect(sim.agents.size).toBeLessThan(2);
  });

  it('drops live ropes at hairline zoom', () => {
    // A wire whose stroke is sub-pixel draws the same streak whether or not a
    // rope is under it, so the nodes go. Derived zoom: this test went stale
    // the first time the hairline threshold moved.
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    params.spawnInterval = 0;
    const a = sim.spawn('con', 120, 80, 0, params, true)!;
    const b = sim.spawn('con', 160, 80, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const hairline = { x: 140, y: 80, zoom: hairlineZoom * 0.9, viewW: 800, viewH: 600 };
    sim.step(1 / 60, params, hairline);
    expect(wiresDrawable(hairline.zoom)).toBe(false);
    const wire = [...sim.graph.wires.values()][0];
    expect(sim.wireSimulatesRope(wire)).toBe(false);
  });

  it('demotes bodies on apparent size, independently of the wire hairline', () => {
    const build = () => {
      const sim = new Sim(240, 160);
      const params = defaultParams();
      params.snapRadius = 0;
      params.stepSpeed = 0;
      params.spawnInterval = 0;
      const a = sim.spawn('con', 120, 80, 0, params, true)!;
      const b = sim.spawn('con', 160, 80, Math.PI, params, true)!;
      sim.wire(a.id, 'p', b.id, 'p', params);
      return { sim, params, a };
    };
    const { a: probe } = build();
    const bodyMidZoom = AGENT_BAND.mid / (boundRadius(probe) * 2);
    // Hairline is past the agent band now, so wires stay drawn after bodies
    // have already gone FAR. The two cuts used to be one number, which put a
    // SAT-to-packed cliff at whatever zoom the stroke happened to vanish.
    expect(hairlineZoom).toBeLessThan(bodyMidZoom);

    const { sim: closeSim, params: closeParams, a: closeA } = build();
    closeSim.step(1 / 60, closeParams, { x: 140, y: 80, zoom: bodyMidZoom * 1.3, viewW: 800, viewH: 600 });
    expect(closeSim.isPhysicsDetailed(closeA.id)).toBe(true);
    expect(wiresDrawable(bodyMidZoom * 1.3)).toBe(true);

    const between = (hairlineZoom + bodyMidZoom) * 0.5;
    const { sim: midSim, params: midParams, a: midA } = build();
    midSim.step(1 / 60, midParams, { x: 140, y: 80, zoom: between, viewW: 800, viewH: 600 });
    expect(wiresDrawable(between)).toBe(true);
    expect(midSim.isPhysicsDetailed(midA.id)).toBe(false);

    const { sim, params, a } = build();
    sim.step(1 / 60, params, { x: 140, y: 80, zoom: 0.05, viewW: 800, viewH: 600 });
    expect(sim.isPhysicsDetailed(a.id)).toBe(false);
    expect(wiresDrawable(0.05)).toBe(true);
  });
});

describe('native mixed solve', () => {
  it('SAT still separates overlapping triangles with no view', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    const a = sim.spawn('con', 120, 80, 0, params, true)!;
    const b = sim.spawn('con', 121, 80, Math.PI, params, true)!;
    sim.step(1 / 60, params);
    expect(sim.contacts.size).toBeGreaterThan(0);
    for (let i = 0; i < 11; i++) sim.step(1 / 60, params);
    expect(queryHit(a, b, sim.w, sim.h)?.overlap ?? 0).toBeLessThanOrEqual(SLOP + 0.08);
  });

  it('keeps Hertzian contacts when the same pair is NEAR', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const sim = new Sim(240, 160);
    const params = defaultParams();
    params.snapRadius = 0;
    params.faceAttract = 0;
    params.snapWell = 0;
    params.stepSpeed = 0;
    sim.spawn('era', 120, 80, 0, params, true);
    sim.spawn('era', 121, 80, Math.PI, params, true);
    sim.step(1 / 60, params, { x: 120, y: 80, zoom: 2, viewW: 400, viewH: 300 });
    expect(sim.contacts.size).toBeGreaterThan(0);
  });

  it('holds a wired pair near rest without injecting energy', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const sim = new Sim(320, 200);
    const params = quietParams();
    params.stepSpeed = 0;
    params.drag = 0.8;
    const a = sim.spawn('era', 80, 100, 0, params, true)!;
    const b = sim.spawn('era', 200, 100, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const rest = [...sim.graph.wires.values()][0].rest;
    for (let i = 0; i < 90; i++) sim.step(1 / 60, params);
    const sa = stemWorld(a, 'p', sim.w, sim.h);
    const sb = stemWorld(b, 'p', sim.w, sim.h);
    expect(Math.abs(Math.hypot(sb.x - sa.x, sb.y - sa.y) - rest)).toBeLessThan(8);
    const e0 = sim.kineticEnergy();
    for (let i = 0; i < 60; i++) sim.step(1 / 60, params);
    expect(sim.kineticEnergy()).toBeLessThan(e0 + 8);
  });
});

describe('far zoom', () => {
  it('keeps poses and rest lengths finite when the camera is fully zoomed out', () => {
    const sim = new Sim(800, 600);
    const params = defaultParams();
    params.upkeep = 0;
    params.spawnInterval = 0;
    params.soupCount = 28;
    loadPreset(sim, 'soup', params);
    const view = { x: 400, y: 300, zoom: 0.05, viewW: 800, viewH: 600 };
    sim.setViewExtent((view.viewW / view.zoom) * 1.7, (view.viewH / view.zoom) * 1.7);
    for (let i = 0; i < 90; i++) sim.step(1 / 60, params, view);
    for (const a of sim.agents.values()) {
      expect(Number.isFinite(a.x), `agent ${a.id} x`).toBe(true);
      expect(Number.isFinite(a.y), `agent ${a.id} y`).toBe(true);
      expect(Math.abs(a.x)).toBeLessThan(50_000);
      expect(Math.abs(a.y)).toBeLessThan(50_000);
    }
    for (const w of sim.graph.wires.values()) {
      expect(Number.isFinite(w.rest)).toBe(true);
      expect(w.rest).toBeLessThan(3000);
      expect(Number.isFinite(w.lastLen)).toBe(true);
      expect(w.lastLen).toBeLessThan(8000);
    }
  });
});
