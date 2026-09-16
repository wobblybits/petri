import { describe, expect, it } from 'vitest';
import { CRITIC_LEN, PLASTIC_BASE, PLASTIC_LEN } from './chem-layout.ts';
import { EXTRA_CAP } from './energy.ts';
import type { Params } from './params.ts';
import { fixedParams } from './test-params.ts';
import { loadPreset } from './presets.ts';
import { CHEM_TASTE_MAX } from './rewrite.ts';
import { Sim } from './sim.ts';
import { stateHash } from './state-hash.ts';
import { CHEM_LEN } from './agents.ts';

/**
 * Weights that change while a body is alive.
 *
 * The rule is three-factor Hebbian gated by a temporal-difference error from
 * the body's own critic. What these check is
 * not that the arithmetic is some particular arithmetic, but the four
 * properties the design actually rests on: that it is off when it is off,
 * that the teacher is the tank and nothing else, that nothing ever fades,
 * and that a parent's experience reaches its children only as far as the
 * slider says.
 */

const realRandom = Math.random;

function seed(n: number): void {
  let s = n >>> 0;
  Math.random = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** No ground, no rent, no latching: a body's tank is exactly what we set. */
function still(): Params {
  const p = fixedParams();
  p.spawnInterval = 0;
  p.snapRadius = 0;
  p.rewriteDuration = 0;
  p.upkeep = 0;
  p.ambientEnergy = 0;
  p.energyRegrow = 0;
  return p;
}

function learnedSpan(sim: Sim, slot: number): Float32Array {
  return sim.agentStore.plasticAll.subarray(slot * PLASTIC_LEN, (slot + 1) * PLASTIC_LEN);
}

function anyNonZero(a: ArrayLike<number>): boolean {
  for (let i = 0; i < a.length; i++) if (a[i] !== 0) return true;
  return false;
}

describe('plasticity: off', () => {
  it('leaves every learned weight at zero when the rate is zero', () => {
    const sim = new Sim(600, 400);
    const params = still();
    params.learnRate = 0;
    loadPreset(sim, 'soup', { ...params, soupCount: 40 });
    for (let f = 0; f < 90; f++) sim.step(1 / 60, params);
    expect(anyNonZero(sim.agentStore.plasticAll)).toBe(false);
    expect(anyNonZero(sim.agentStore.plasticOn)).toBe(false);
    expect(anyNonZero(sim.agentStore.criticAll)).toBe(false);
  });

  it('hashes identically however the inheritance slider is set', () => {
    // The consolidation path reads `plastic`, which is all zeros here, and
    // must therefore add nothing and — the easy thing to get wrong — draw no
    // random numbers of its own.
    const run = (kappa: number): string => {
      seed(99);
      try {
        const sim = new Sim(600, 400);
        const params = fixedParams();
        params.soupCount = 50;
        params.learnRate = 0;
        params.inheritLearned = kappa;
        loadPreset(sim, 'soup', params);
        for (let f = 0; f < 200; f++) sim.step(1 / 60, params);
        return stateHash(sim);
      } finally {
        Math.random = realRandom;
      }
    };
    expect(run(1)).toBe(run(0));
  });
});

describe('plasticity: the teacher is the tank', () => {
  it('teaches a hungry body and leaves a full one alone', () => {
    const sim = new Sim(600, 400);
    const params = still();
    params.learnRate = 0.01;
    const full = sim.spawn('con', 200, 200, 0, params, true)!;
    const hungry = sim.spawn('con', 400, 200, 0, params, true)!;
    for (let f = 0; f < 40; f++) {
      // Held there against anything the frame does to them, so the only
      // difference between the two bodies is how full each one is.
      full.extra = EXTRA_CAP;
      hungry.extra = 0;
      sim.step(1 / 60, params);
    }
    expect(sim.agentStore.plasticOn[hungry.slot], 'a hungry body should learn').toBe(1);
    expect(anyNonZero(learnedSpan(sim, hungry.slot))).toBe(true);
    // A body with nothing to want has no error to learn from: its cost is
    // zero, its critic predicts zero, and the difference between them is
    // what drives every weight here.
    expect(sim.agentStore.plasticOn[full.slot], 'a full body has nothing to learn from').toBe(0);
    expect(anyNonZero(learnedSpan(sim, full.slot))).toBe(false);
  });

  it('teaches a full body that is losing, which the level signal cannot', () => {
    /*
     * The level signal is `IN_FULL - 1`: zero when full, so a body at its cap
     * has no gradient however its fortunes are moving. Measured in a live
     * soup, 31.7% of bodies sit there. The rate signal is the change in
     * `IN_FULL` per second, and it reads for exactly those bodies.
     *
     * Two bodies held near the top of their tanks, one steady and one sliding
     * down. Under the rate teacher the sliding one has something to learn
     * from and the steady one does not — which is the dead zone closing.
     */
    const sim = new Sim(600, 400);
    const params = still();
    params.learnRate = 0.01;
    params.learnReward = 1;
    const steady = sim.spawn('con', 200, 200, 0, params, true)!;
    const sliding = sim.spawn('con', 400, 200, 0, params, true)!;
    for (let f = 0; f < 60; f++) {
      steady.extra = EXTRA_CAP;
      // From full to about a third, over the window. Never zero, so the
      // no-history sentinel is not what is being read.
      sliding.extra = EXTRA_CAP * (1 - f / 90);
      sim.step(1 / 60, params);
    }
    expect(anyNonZero(learnedSpan(sim, sliding.slot)), 'a body losing ground should learn').toBe(true);
    expect(anyNonZero(learnedSpan(sim, steady.slot)), 'a body holding steady at full should not').toBe(false);
  });

  it('gives the critic something to say', () => {
    const sim = new Sim(600, 400);
    const params = still();
    params.learnRate = 0.01;
    const a = sim.spawn('con', 300, 200, 0, params, true)!;
    for (let f = 0; f < 60; f++) {
      a.extra = 0;
      sim.step(1 / 60, params);
    }
    const critic = sim.agentStore.criticAll.subarray(a.slot * CRITIC_LEN, (a.slot + 1) * CRITIC_LEN);
    // The bias is the part that can move with `h` at zero, and a body that
    // is persistently empty should end up predicting exactly that.
    expect(critic[CRITIC_LEN - 1]).toBeLessThan(0);
  });
});

describe('plasticity: nothing fades', () => {
  it('keeps what a body learned after it is cut loose from its net', () => {
    const sim = new Sim(600, 400);
    const params = still();
    params.learnRate = 0.01;
    const a = sim.spawn('con', 300, 200, 0, params, true)!;
    const b = sim.spawn('dup', 340, 200, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    for (let f = 0; f < 60; f++) {
      a.extra = 0;
      b.extra = 0;
      sim.step(1 / 60, params);
    }
    const learned = Float32Array.from(learnedSpan(sim, a.slot));
    expect(anyNonZero(learned)).toBe(true);

    // Cut it loose and stop teaching it. What it worked out inside the net
    // is what it carries into the next one, and that carriage is the whole
    // reason for learning rather than only breeding.
    sim.graph.detachAgent(a.id);
    params.learnRate = 0;
    for (let f = 0; f < 120; f++) sim.step(1 / 60, params);
    expect(Array.from(learnedSpan(sim, a.slot))).toEqual(Array.from(learned));
  });

  it('holds the effective weight inside the range a gene may take', () => {
    const sim = new Sim(600, 400);
    const params = still();
    // Absurd, on purpose: the clamp is the only thing between a saturating
    // modulator and a weight the state pass cannot survive reading.
    params.learnRate = 5;
    const a = sim.spawn('con', 300, 200, 0, params, true)!;
    for (let f = 0; f < 200; f++) {
      a.extra = 0;
      sim.step(1 / 60, params);
    }
    const chem = sim.agentStore.chemAll;
    const plastic = sim.agentStore.plasticAll;
    let worst = 0;
    for (let k = 0; k < PLASTIC_LEN; k++) {
      const eff = chem[a.slot * CHEM_LEN + PLASTIC_BASE + k] + plastic[a.slot * PLASTIC_LEN + k];
      worst = Math.max(worst, Math.abs(eff));
    }
    expect(worst).toBeLessThanOrEqual(CHEM_TASTE_MAX + 1e-5);
    expect(Number.isFinite(worst)).toBe(true);
  });
});

describe('plasticity: inheritance', () => {
  /** A commute, with both parents holding a known amount of learning. */
  function commuteWith(kappa: number, planted: number): { chem: Float32Array; slots: number[] } {
    seed(7);
    try {
      const sim = new Sim(600, 400);
      const params = fixedParams();
      params.spawnInterval = 0;
      params.learnRate = 0;
      params.inheritLearned = kappa;
      loadPreset(sim, 'commute', params);
      const before = new Set(sim.agents.keys());
      for (const a of sim.agents.values()) {
        if (a.kind === 'era') continue;
        const o = a.slot * PLASTIC_LEN;
        for (let k = 0; k < PLASTIC_LEN; k++) sim.agentStore.plasticAll[o + k] = planted;
        sim.agentStore.plasticOn[a.slot] = 1;
      }
      for (let f = 0; f < 300 && sim.agents.size <= before.size; f++) sim.step(1 / 60, params);
      const slots: number[] = [];
      for (const [id, a] of sim.agents) if (!before.has(id) && a.kind !== 'era') slots.push(a.slot);
      return { chem: sim.agentStore.chemAll, slots };
    } finally {
      Math.random = realRandom;
    }
  }

  function meanLearned(chem: Float32Array, slots: number[]): number {
    let sum = 0;
    let n = 0;
    for (const slot of slots) {
      for (let k = 0; k < PLASTIC_LEN; k++) {
        sum += chem[slot * CHEM_LEN + PLASTIC_BASE + k];
        n++;
      }
    }
    return n > 0 ? sum / n : 0;
  }

  it('writes a parent’s learning into its children at full inheritance', () => {
    const on = commuteWith(1, 2);
    expect(on.slots.length, 'the commute did not fire').toBeGreaterThan(0);
    // Both parents held 2 across the learned span, so a child blended or
    // assorted from them lands near 2, mutation aside.
    expect(meanLearned(on.chem, on.slots)).toBeGreaterThan(1.5);
  });

  it('leaves the children with none of it at zero', () => {
    const off = commuteWith(0, 2);
    expect(off.slots.length, 'the commute did not fire').toBeGreaterThan(0);
    // The genome the parents were born with is all the children get, and its
    // state matrices seed at zero.
    expect(Math.abs(meanLearned(off.chem, off.slots))).toBeLessThan(0.1);
  });
});
