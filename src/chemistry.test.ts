import { describe, expect, it } from 'vitest';
import { bareBody, expressVector, seedChem, uptakeKsOf } from './agents.ts';
import { CHEM_LEN, CHEM_SPECIES, KS_BASE, ROW_COUNT, ROW_EXCRETE, ROW_UPTAKE, STATE_DIMS, X_BASE, X_OUT } from './chem-layout.ts';
import { REWRITE_SHARE, rewriteCost } from './energy.ts';
import { CH, CHANNELS } from './fields.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';

/*
 * The body reaction table. `docs/energy-chemistry-plan.md` §3.
 *
 * Two properties carry the whole design and both are testable directly. The
 * simplex is what makes expression a *budget* — a body cannot both shout and
 * eat without giving something up — and conservation is what makes the budget
 * honest, because a body that has nothing cannot spend anything however loudly
 * its genome would like to.
 */

/**
 * A pond with the dish switched off, so only body reactions move anything.
 *
 * `ambientEnergy = 0` matters more than it looks: with ground in the dish the
 * bodies harvest between steps, and a test that means to watch a tank empty
 * watches it refill instead. `diffuse = 0` matters for the same kind of
 * reason — the rim absorbs the three signal species, which is the dish doing
 * its job and not a body failing to conserve.
 */
function chemistryParams(): Params {
  const p = defaultParams();
  p.soupCount = 60;
  p.spawnInterval = 0;
  p.energyRegrow = 0;
  p.ambientEnergy = 0;
  p.decay = 0;
  p.diffuse = 0;
  p.upkeep = 0;
  // Bodies conservative, so a commute is a transfer and not a mint. See
  // `matter` and `energy.test.ts`'s conservation suite.
  p.bodyValue = REWRITE_SHARE;
  return p;
}

/**
 * Everything in the pond: what bodies are made of and hold, what is in
 * escrow or in flight, and every species in the field.
 *
 * A body's *existence* has to be in here, not just its stock. Deaths and
 * rewrites move `bodyValue` between the two, and a total that counted only
 * `extra` would read every commute as matter appearing — which is exactly what
 * it read, at four per cent over three simulated seconds, before this counted
 * bodies. Pair it with `bodyValue = REWRITE_SHARE`, which is what makes those
 * transfers conservative in the first place.
 */
function matter(sim: Sim, bodyValue: number): number {
  let held = 0;
  for (const a of sim.agents.values()) held += bodyValue + a.extra;
  let inFlight = 0;
  for (const rw of sim.rewrites) inFlight += rewriteCost(rw.rule);
  let field = 0;
  const d = sim.fields.data;
  for (let k = 0; k < d.length; k++) field += d[k];
  return held + inFlight + sim.escrowTotal() + field;
}

describe('expression', () => {
  const h = new Float64Array(STATE_DIMS);
  const out = new Float64Array(ROW_COUNT);

  it('divides one unit across the whole table', () => {
    const chem = seedChem('con', defaultParams());
    expressVector(chem, 0, h, 0, out, 0);
    let sum = 0;
    for (let r = 0; r < ROW_COUNT; r++) {
      expect(out[r], `row ${r} is negative`).toBeGreaterThanOrEqual(0);
      sum += out[r];
    }
    expect(sum).toBeCloseTo(1, 12);
  });

  it('is flat at the seed, so every reaction runs at its own constant', () => {
    // `X` and its base seed to zero, so relu leaves nothing to normalise. An
    // eighth each is the even division; zero would make a fresh body inert
    // from birth, which is not what shipping at the neutral value means.
    for (const kind of ['era', 'dup', 'con'] as const) {
      const chem = seedChem(kind, defaultParams());
      expressVector(chem, 0, h, 0, out, 0);
      for (let r = 0; r < ROW_COUNT; r++) expect(out[r]).toBeCloseTo(1 / ROW_COUNT, 12);
    }
  });

  it('reads the state, so a body can express differently when hungry', () => {
    const chem = seedChem('con', defaultParams());
    // One row wired to one state dimension: the smallest thing that makes
    // expression a phenotype rather than a constant.
    chem[X_OUT + ROW_UPTAKE * STATE_DIMS + 0] = 2;
    const calm = new Float64Array(STATE_DIMS);
    const roused = new Float64Array(STATE_DIMS);
    roused[0] = 0.9;
    const a = new Float64Array(ROW_COUNT);
    const b = new Float64Array(ROW_COUNT);
    expressVector(chem, 0, calm, 0, a, 0);
    expressVector(chem, 0, roused, 0, b, 0);
    expect(b[ROW_UPTAKE]).toBeGreaterThan(a[ROW_UPTAKE]);
    // And it came out of the budget, not out of nowhere.
    expect(b[ROW_EXCRETE]).toBeLessThan(a[ROW_EXCRETE]);
    let sum = 0;
    for (let r = 0; r < ROW_COUNT; r++) sum += b[r];
    expect(sum).toBeCloseTo(1, 12);
  });

  it('keeps a body that mutated its way to silence silent', () => {
    // The same rule `emitVector` follows: relu then normalise would otherwise
    // amplify a genome sitting at all-negative back out of noise.
    const chem = seedChem('con', defaultParams());
    for (let r = 0; r < ROW_COUNT; r++) chem[X_BASE + r] = -1;
    expressVector(chem, 0, h, 0, out, 0);
    // Nothing expressed at all is the flat fallback, not an arbitrary row.
    for (let r = 0; r < ROW_COUNT; r++) expect(out[r]).toBeCloseTo(1 / ROW_COUNT, 12);
  });

  it('gives each species its own affinity, floored above zero', () => {
    const chem = seedChem('dup', defaultParams());
    for (let c = 0; c < CHEM_SPECIES; c++) {
      // Seeded at one natural unit, so a fresh body uses the global.
      expect(uptakeKsOf(chem, 0, c, 0.25)).toBeCloseTo(0.25, 12);
    }
    const half = seedChem('dup', defaultParams());
    half[X_BASE + ROW_COUNT + 0] = 0.5;
    expect(uptakeKsOf(half, 0, 0, 0.25)).toBeCloseTo(0.125, 12);
    // A gene mutated to or past zero would be an infinitely good transporter,
    // and a division by zero downstream.
    const dead = seedChem('dup', defaultParams());
    dead[X_BASE + ROW_COUNT + 1] = -3;
    expect(uptakeKsOf(dead, 0, 1, 0.25)).toBeGreaterThan(0);
  });

  it('leaves the genome it reads alone', () => {
    const chem = seedChem('con', defaultParams());
    const before = [...chem];
    expressVector(chem, 0, h, 0, out, 0);
    expect([...chem]).toEqual(before);
    expect(bareBody(chem).chem.length).toBe(CHEM_LEN);
  });
});

describe('excretion', () => {
  it('is off at zero, and the voice is still minted', () => {
    const p = chemistryParams();
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 1;
    const before = matter(sim, p.bodyValue);
    for (let i = 0; i < 120; i++) sim.step(1 / 60, p);
    // The scent path multiplies by `params.deposit`, which is five, and takes
    // nothing out of any tank. Matter grows, and that is today's pond.
    expect(matter(sim, p.bodyValue)).toBeGreaterThan(before);
  });

  it('conserves what it moves, and stops the mint', () => {
    const p = chemistryParams();
    p.excreteRate = 0.4;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    // Bodies need something to excrete, and a barren pond gives them none.
    for (const a of sim.agents.values()) a.extra = 1;
    const before = matter(sim, p.bodyValue);
    for (let i = 0; i < 120; i++) sim.step(1 / 60, p);
    const after = matter(sim, p.bodyValue);
    // Exactly, not nearly: with the dish off there is nothing but the bodies'
    // own reactions moving anything, and they are conservative by construction.
    expect(after / before, `matter went ${before} -> ${after}`).toBeCloseTo(1, 6);
  });

  it('actually moves something', () => {
    const p = chemistryParams();
    p.excreteRate = 0.4;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 1;
    const held = () => [...sim.agents.values()].reduce((n, a) => n + a.extra, 0);
    const before = held();
    for (let i = 0; i < 120; i++) sim.step(1 / 60, p);
    expect(held()).toBeLessThan(before * 0.9);
    // Onto every species, including the two the scent path always firewalled
    // `CH.energy` away from and the one it never touched.
    const total = (ch: number) => {
      const d = sim.fields.data;
      let s = 0;
      for (let k = ch; k < d.length; k += CHANNELS) s += d[k];
      return s;
    };
    expect(total(CH.conP)).toBeGreaterThan(0);
    expect(total(CH.aux)).toBeGreaterThan(0);
  });

  it('cannot be afforded by a body with nothing', () => {
    /*
     * Absolute honesty, which the simplex alone cannot buy. The rate is mass
     * action on the tank, so a poor body physically cannot shout however its
     * genome divides the budget.
     */
    const p = chemistryParams();
    p.excreteRate = 0.4;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const bodies = [...sim.agents.values()];
    const rich = bodies[0];
    const poor = bodies[1];
    for (const a of bodies) a.extra = 0;
    rich.extra = 1;
    sim.step(1 / 60, p);
    const store = sim.agentStore;
    const out = (a: typeof rich) => {
      let s = 0;
      for (let c = 0; c < CHEM_SPECIES; c++) s += store.excreteAll[a.slot * CHEM_SPECIES + c];
      return s;
    };
    expect(out(rich)).toBeGreaterThan(0);
    expect(out(poor)).toBe(0);
    // And never into debt: what it spent is what it had, at most.
    expect(rich.extra).toBeGreaterThanOrEqual(0);
  });
});

describe('uptake', () => {
  /** Standing stock of one species across the whole field. */
  const total = (sim: Sim, ch: number): number => {
    const d = sim.fields.data;
    let s = 0;
    for (let k = ch; k < d.length; k += CHANNELS) s += d[k];
    return s;
  };

  it('eats only the ground until excretion stops the minting', () => {
    /*
     * The coupling that matters, and the reason `UptakeKinetics.table` is not
     * simply `uptakeVmax > 0`.
     *
     * Metering uptake on all four species while the scent path still mints is
     * a fountain: `params.deposit` puts five times a body's voice into three
     * channels out of nothing, and metered uptake then lets it eat that back.
     * Measured before this was coupled, it ran the pond at twice the rate cap
     * and filled every tank.
     */
    const p = chemistryParams();
    p.ambientEnergy = 0;
    p.uptakeVmax = 0.6;
    p.diffuse = 0.6;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 0;
    for (let i = 0; i < 120; i++) sim.step(1 / 60, p);
    // Scent was minted, so there is plenty to eat if anything is allowed to.
    expect(total(sim, CH.conP)).toBeGreaterThan(0);
    // And nothing did: with no ground in the dish, nobody ate at all.
    expect(sim.totalFree()).toBeCloseTo(0, 6);
  });

  it('closes the loop once the table is on', () => {
    // One body's excretion is another's food, which is the whole point of four
    // species rather than one substance and three decorations.
    const p = chemistryParams();
    p.excreteRate = 0.5;
    p.uptakeVmax = 1.5;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 1;
    for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
    // Something was excreted onto a signal species and something took it back.
    expect(total(sim, CH.conP)).toBeGreaterThan(0);
    const store = sim.agentStore;
    let excreted = 0;
    for (const a of sim.agents.values()) {
      for (let c = 0; c < CHEM_SPECIES; c++) excreted += store.excreteAll[a.slot * CHEM_SPECIES + c];
    }
    expect(excreted).toBeGreaterThan(0);
    // Held stock is not simply draining away: what left tanks is coming back.
    const held = [...sim.agents.values()].reduce((n, a) => n + a.extra, 0);
    expect(held).toBeGreaterThan(0);
  });

  it('conserves with the whole table running', () => {
    const p = chemistryParams();
    p.excreteRate = 0.5;
    p.uptakeVmax = 1.5;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 1;
    const before = matter(sim, p.bodyValue);
    for (let i = 0; i < 180; i++) sim.step(1 / 60, p);
    const after = matter(sim, p.bodyValue);
    expect(after / before, `matter went ${before} -> ${after}`).toBeCloseTo(1, 6);
  });

  it('gives a fast grazer and a scavenger different answers', () => {
    /*
     * The non-dominating pair, which is what §4 says the diversity comes from.
     * Same expression, same ground, different affinity: on rich ground the
     * high-affinity gene barely helps, and on poor ground it is most of the
     * difference. Neither body wins everywhere, which is the property.
     */
    const p = chemistryParams();
    p.excreteRate = 0;
    p.uptakeVmax = 1.5;
    p.uptakeKs = 0.5;

    const run = (ambient: number, ksGene: number): number => {
      const sim = new Sim(1600, 1200, 128);
      const q = { ...p, ambientEnergy: ambient };
      loadPreset(sim, 'soup', q);
      const a = [...sim.agents.values()][0];
      a.extra = 0;
      // `KS_BASE + CH.energy`, not `+ 0`: off the reaction table only the
      // ground's row is metered, so the affinity that matters is the ground's.
      a.chem[KS_BASE + CH.energy] = ksGene;
      for (let i = 0; i < 30; i++) sim.step(1 / 60, q);
      return a.extra;
    };

    const richGeneralist = run(2, 1);
    const richSpecialist = run(2, 0.05);
    const poorGeneralist = run(0.02, 1);
    const poorSpecialist = run(0.02, 0.05);
    // A better transporter is worth little where there is plenty...
    const richEdge = richSpecialist / Math.max(richGeneralist, 1e-9);
    // ...and a great deal where there is not.
    const poorEdge = poorSpecialist / Math.max(poorGeneralist, 1e-9);
    expect(poorEdge).toBeGreaterThan(richEdge * 1.5);
  });
});
