import { describe, expect, it } from 'vitest';
import { bareBody, expressVector, seedChem, uptakeKsOf } from './agents.ts';
import { CHEM_LEN, CHEM_SPECIES, KS_BASE, ROW_COUNT, ROW_EXCRETE, ROW_UPTAKE, STATE_DIMS, X_BASE, X_OUT } from './chem-layout.ts';
import { HarvestPlan, REWRITE_SHARE, harvestSlotsFast, rewriteCost } from './energy.ts';
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
  // The gut is a fourth place matter can be: swallowed, not yet digested, and
  // in neither the ground nor a tank. See `Sim.totalGut`.
  return held + inFlight + sim.escrowTotal() + sim.totalGut() + field;
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

  it('eats the mixture it is standing in, ground or not', () => {
    /*
     * What replaced the switch this used to test.
     *
     * Uptake was held to the ground alone until `excreteRate` stopped the
     * minting, because four independent rates meant four times the cap:
     * `params.deposit` puts five times a body's voice into three channels out
     * of nothing, and a body could then eat its own scent back for a profit.
     * Measured before that coupling, it ran the pond at twice the rate cap and
     * filled every tank.
     *
     * One budget shared out by what is in the water closes it without the
     * switch — the mint buys the minter nothing, it only dilutes what it is
     * standing in. So there is no arm in which uptake means a different
     * mechanism, and no dish so filthy that nothing in it is food. The
     * ceiling itself is the test below.
     */
    const p = chemistryParams();
    p.ambientEnergy = 0;
    p.uptakeVmax = 0.6;
    p.diffuse = 0.6;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 0;
    for (let i = 0; i < 120; i++) sim.step(1 / 60, p);
    // Scent was minted, and with no ground in the dish it is all there is.
    expect(total(sim, CH.conP)).toBeGreaterThan(0);
    // And it was eaten, which is what a body with no ports to spare and no
    // ground under it can still do.
    expect(sim.totalFree()).toBeGreaterThan(0);
  });

  it('caps each species at its share of one budget', () => {
    /*
     * A pure cell is tasty and a smelly one is polluted.
     *
     * Two cells with the same standing stock: one all ground, one a quarter
     * ground and three quarters scent. A body may take a *sample* of the
     * water, never the good part of it, so the ground it can draw out of the
     * filthy cell is a quarter of its budget however good its transporter for
     * ground is — the rest of the mouthful is spent on three species that need
     * catabolic machinery before they are worth anything. Same stock, four
     * times the work for the ground in it.
     */
    const p = chemistryParams();
    p.soupCount = 1;
    p.uptakeVmax = 0.6;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    /*
     * One frame first, for two reasons. The grid moves onto `fields` when the
     * world is pinned and until then answers out of a sparse map nothing here
     * writes to; and `refreshExpression` is what fills the uptake rows, so
     * before a frame has run every row is zero and a body cannot eat at all.
     */
    sim.step(1 / 60, p);
    const a = [...sim.agents.values()][0];
    const cell = sim.energy.index(a.x, a.y);
    const rect = sim.energy.blockRect(cell.i, cell.j)!;

    /** Lay `per[c]` of every species flat across the body's own block. */
    const paint = (per: number[]): void => {
      const d = sim.fields.data;
      d.fill(0);
      for (let y = 0; y < rect.wj; y++) {
        for (let x = 0; x < rect.wi; x++) {
          const k = ((rect.fj + y) * sim.fields.cols + (rect.fi + x)) * CHANNELS;
          for (let c = 0; c < CHANNELS; c++) d[k + c] = per[c];
        }
      }
    };
    const cap = p.uptakeVmax / 60;
    const kinetics = { cap, ks: p.uptakeKs, yDirect: 1, yEra: 1, hillN: 1, gutSize: p.gutSize };
    /** Ground drawn out of the block in one frame. */
    const ground = (per: number[]): number => {
      paint(per);
      a.extra = 0;
      // An empty gut as well as an empty tank: room in the gut is what bounds
      // a mouthful now, so a second run on a full gut would measure nothing.
      sim.agentStore.gut.fill(0, a.slot * CHEM_SPECIES, a.slot * CHEM_SPECIES + CHEM_SPECIES);
      const before = sim.energy.storedTotal();
      harvestSlotsFast([a], sim.agentStore, sim.energy, new HarvestPlan(), kinetics);
      return before - sim.energy.storedTotal();
    };

    const alone = (c: number, v: number): number[] => {
      const per = new Array(CHANNELS).fill(0);
      per[c] = v;
      return per;
    };
    const clean = ground(alone(CH.energy, 0.4));
    const filthy = ground(new Array(CHANNELS).fill(0.4 / CHANNELS));
    // The ceiling is the share, exactly: one part in four of the cell is
    // ground, so at most a quarter of one budget can be drawn as ground.
    expect(filthy).toBeLessThanOrEqual(cap / CHANNELS + 1e-12);
    // And it binds — the clean cell is not up against it, so this is the
    // mixture doing the work and not the tank or the Monod rate.
    expect(clean).toBeGreaterThan(cap / CHANNELS);
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

describe('trophic yield', () => {
  const fed = (tweak: (p: Params) => void): { free: number; ground: number } => {
    const p = chemistryParams();
    p.ambientEnergy = 1;
    p.uptakeVmax = 1.5;
    tweak(p);
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 0;
    for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
    return { free: sim.totalFree(), ground: sim.energy.storedTotal() };
  };

  it('feeds everyone at a yield of one, which is today', () => {
    const base = fed(() => {});
    expect(base.free).toBeGreaterThan(0);
  });

  it('starves a body that cannot feed itself, and leaves the ground alone', () => {
    /*
     * Obligate dependency, as one dial. Applied to the *rate*, so a body with
     * no yield does not draw and the ground it is standing on is untouched —
     * conservative without a second write per body per species, and it means
     * a starving body is not also a wasteful one.
     */
    const base = fed(() => {});
    const obligate = fed((p) => {
      p.yDirect = 0;
      p.yEra = 0;
    });
    expect(obligate.free).toBe(0);
    expect(obligate.ground).toBeGreaterThan(base.ground);
  });

  it('makes an Era the net’s mouth without a mint', () => {
    // §5's replacement for `ERA_UPKEEP_RATIO`: an Era's income comes from the
    // ground under it, not from a rule keyed on its glyph.
    const p = chemistryParams();
    p.ambientEnergy = 1;
    p.uptakeVmax = 1.5;
    p.yDirect = 0.2;
    p.yEra = 2;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const cx = sim.w * 0.5;
    const cy = sim.h * 0.5;
    const era = sim.spawn('era', cx - 60, cy, 0, p, true)!;
    const con = sim.spawn('con', cx + 60, cy, 0, p, true)!;
    era.pinned = true;
    con.pinned = true;
    era.extra = 0;
    con.extra = 0;
    for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
    expect(era.extra).toBeGreaterThan(con.extra * 2);
  });
});

describe('superadditivity', () => {
  /*
   * §3's condition, stated precisely: division of labour needs a trade-off,
   * and a linear budget is not one. With a linear constraint and concave
   * payoffs — Monod at `n = 1` is concave — the optimum is interior and
   * everyone becomes a generalist, because specialising beats splitting only
   * when `f(1) > 2 f(1/2)`, which concavity forbids. These are the two dials
   * that break the concavity, and both ship at the neutral value.
   */

  it('charges a generalist for every row it runs', () => {
    const p = chemistryParams();
    p.ambientEnergy = 0;
    p.uptakeVmax = 1.5;
    p.rowCost = 0.02;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const a = [...sim.agents.values()][0];
    a.extra = 1;
    // Flat expression is all eight rows, so a seeded body pays for all eight.
    // Charging it nothing would make "say nothing, act as a generalist" free
    // and strictly best at any cost, which is the opposite of the pressure.
    sim.step(1 / 60, p);
    expect(a.extra).toBeCloseTo(1 - 0.02 * (1 / 60) * ROW_COUNT, 6);
  });

  it('lets a specialist keep what breadth costs', () => {
    const p = chemistryParams();
    p.ambientEnergy = 0;
    p.uptakeVmax = 1.5;
    p.rowCost = 0.02;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const bodies = [...sim.agents.values()];
    const generalist = bodies[0];
    const specialist = bodies[1];
    generalist.extra = 1;
    specialist.extra = 1;
    // Relu makes "switched off" an exact question, and driving a row's
    // pre-activation below zero is how a lineage specialises.
    for (let r = 1; r < ROW_COUNT; r++) specialist.chem[X_BASE + r] = -1;
    specialist.chem[X_BASE] = 1;
    sim.step(1 / 60, p);
    expect(specialist.extra).toBeGreaterThan(generalist.extra);
    expect(1 - specialist.extra).toBeCloseTo((1 - generalist.extra) / ROW_COUNT, 6);
  });

  it('is off at zero, and does not destroy what it charges', () => {
    /*
     * With `excreteRate` on, because conservation is only a question once the
     * minting has stopped. Left minting, the field fills with scent nobody
     * paid for and the total climbs past twenty-eight thousand from a hundred
     * and twenty — which is today's pond working as designed, and nothing to
     * do with the row cost.
     */
    const p = chemistryParams();
    p.ambientEnergy = 0;
    p.uptakeVmax = 1.5;
    p.excreteRate = 0.5;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 1;
    const before = matter(sim, p.bodyValue);
    for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
    // Five places, not six: `Fields.data` is Float32 and `matter` sums a
    // million cells of it, so the rounding floor is around a part in ten
    // million of the total and not something the reactions can do better than.
    expect(matter(sim, p.bodyValue)).toBeCloseTo(before, 5);

    /*
     * And with the cost on, what leaves a tank arrives on the ground.
     *
     * Gentle enough that nobody reaches debt, and the test checks that they
     * did not. A body billed past empty runs a debt rather than moving matter
     * — correctly, since it has none to move — so a pond that starved would
     * show a shortfall here that is an artifact of the question rather than a
     * leak. `energy.test.ts` documents the same hole from the other side.
     */
    const q = { ...p, rowCost: 0.01, excreteRate: 0.1, upkeepExcrete: 1 };
    const sim2 = new Sim(1600, 1200, 128);
    loadPreset(sim2, 'soup', q);
    for (const a of sim2.agents.values()) a.extra = 1;
    const before2 = matter(sim2, q.bodyValue);
    for (let i = 0; i < 60; i++) sim2.step(1 / 60, q);
    const poorest = Math.min(...[...sim2.agents.values()].map((a) => a.extra));
    expect(poorest, 'somebody ran a debt; the total below is not the test').toBeGreaterThan(0);
    expect(sim2.totalFree()).toBeLessThan(60);
    expect(matter(sim2, q.bodyValue)).toBeCloseTo(before2, 5);
  });

  it('makes uptake convex at low density above n = 1', () => {
    // The other dial. At n = 1 Monod is concave everywhere; above it the
    // response is convex at low density, which is what makes committing pay.
    const monod = (s: number, n: number) => (1.5 * s ** n) / (0.5 ** n + s ** n);
    const half = monod(0.1, 1);
    const full = monod(0.2, 1);
    expect(full).toBeLessThan(2 * half);
    const halfH = monod(0.1, 3);
    const fullH = monod(0.2, 3);
    expect(fullH).toBeGreaterThan(2 * halfH);
  });
});

describe('the gut', () => {
  /*
   * §6c. The tank is one scalar, so before this everything a body swallowed
   * became `extra` the instant it crossed the membrane and there was no such
   * thing as an un-metabolised substance inside a body. These are the two
   * halves of there being one: a body holds what it cannot convert, and what
   * it is holding is what stops it eating more.
   */
  const oneBody = (tweak: (p: Params) => void) => {
    const p = chemistryParams();
    p.soupCount = 1;
    p.uptakeVmax = 6;
    tweak(p);
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    // A frame to pin the world and fill the expression rows; see the note in
    // 'caps each species at its share of one budget'.
    sim.step(1 / 60, p);
    const a = [...sim.agents.values()][0];
    const cell = sim.energy.index(a.x, a.y);
    const rect = sim.energy.blockRect(cell.i, cell.j)!;
    const paint = (per: number[]): void => {
      const d = sim.fields.data;
      d.fill(0);
      for (let y = 0; y < rect.wj; y++) {
        for (let x = 0; x < rect.wi; x++) {
          const k = ((rect.fj + y) * sim.fields.cols + (rect.fi + x)) * CHANNELS;
          for (let c = 0; c < CHANNELS; c++) d[k + c] = per[c];
        }
      }
    };
    const gut = (c: number): number => sim.agentStore.gut[a.slot * CHEM_SPECIES + c];
    const held = (): number => sim.agentStore.gutTotal(a.slot);
    return { sim, p, a, paint, gut, held };
  };

  const alone = (c: number, v: number): number[] => {
    const per = new Array(CHANNELS).fill(0);
    per[c] = v;
    return per;
  };

  it('holds what it swallowed and could not convert', () => {
    /*
     * A body standing on nothing but scent, with the ground as the
     * co-substrate. It cannot decline the mouthful — that is what a sample is
     * — and it cannot convert it either, because nothing it swallowed is
     * ground. So it ends the frame holding scent it has no use for, which is
     * the thing that could not previously be true of anything.
     */
    const { sim, p, a, paint, gut } = oneBody((q) => {
      q.catCoSubstrate = 1;
      q.excreteRate = 0;
    });
    paint(alone(CH.conP, 0.6));
    a.extra = 0.5;
    const before = a.extra;
    for (let i = 0; i < 20; i++) sim.step(1 / 60, p);
    // Swallowed: it is inside the body, not on the ground.
    expect(gut(CH.conP)).toBeGreaterThan(0);
    // And worth nothing to it, because it had no ground to convert it with.
    expect(a.extra).toBeLessThanOrEqual(before);
  });

  it('cannot eat past a full gut', () => {
    /*
     * Satiety, three mechanisms deep rather than a clamp: digestion is bounded
     * by room in the tank, so a full body cannot digest, so its gut fills, so
     * it cannot eat. Here the middle step is skipped and the gut simply filled,
     * which is the state a body that cannot digest what it holds arrives at.
     */
    const { sim, p, a, paint, held } = oneBody(() => {});
    paint(alone(CH.energy, 0.6));
    const store = sim.agentStore;
    const go = a.slot * CHEM_SPECIES;
    // Filled to its cap with something, and `gutSize` is a multiple of that.
    store.gut[go + CH.conP] = store.energyCap[a.slot] * p.gutSize;
    expect(held()).toBeGreaterThan(0);
    const before = sim.energy.storedTotal();
    harvestSlotsFast([a], store, sim.energy, new HarvestPlan(), {
      cap: p.uptakeVmax / 60,
      ks: p.uptakeKs,
      yDirect: 1,
      yEra: 1,
      hillN: 1,
      gutSize: p.gutSize,
    });
    // Ground under it, an empty tank, and not a unit taken: there is nowhere
    // to put a mouthful.
    expect(sim.energy.storedTotal()).toBeCloseTo(before, 12);
  });
});

describe('catabolism', () => {
  /*
   * §6b. Eating a signalling species raw is what phase 3 shipped and what the
   * sweeps liked least; `catCoSubstrate` makes the ground the co-substrate the
   * others are converted *with*. It buys access, never amplification —
   * conservation is unchanged — and the gradient runs continuously from zero
   * so that a body with a little capability is a little better off than one
   * with none.
   */
  const seeded = (tweak: (p: Params) => void) => {
    const p = chemistryParams();
    p.excreteRate = 0.015;
    p.uptakeVmax = 6;
    p.ambientEnergy = 0;
    tweak(p);
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    const cx = sim.w * 0.5;
    const cy = sim.h * 0.5;
    // One body, standing on a patch of one signalling species and nothing
    // else, with a tank to start from.
    const a = sim.spawn('con', cx, cy, 0, p, true)!;
    a.pinned = true;
    a.extra = 0.2;
    sim.fields.fillDisk(CH.conP, 4);
    return { sim, p, a };
  };

  it('lets a body eat a signalling species raw when the gate is off', () => {
    const { sim, p, a } = seeded(() => {});
    const before = a.extra;
    for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
    /*
     * Phase 3's behaviour, kept reachable: it fed, on a dish seeded with no
     * ground at all. Not asserted as "the ground stays empty" — excretion is
     * on, and a body excretes onto `CH.energy` along with the rest, so it lays
     * down a little ground of its own as it goes. What matters is that it came
     * out ahead while standing on a species it should not be able to eat raw.
     */
    expect(a.extra).toBeGreaterThan(before);
  });

  it('cannot live on scent alone once the ground is the co-substrate', () => {
    const { sim, p, a } = seeded((q) => {
      q.catCoSubstrate = 1;
    });
    const before = a.extra;
    for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
    // Plenty of species 0 under it and no ground to convert it with.
    expect(a.extra).toBeLessThan(before);
  });

  it('converts it where there is ground to convert it with', () => {
    const { sim, p } = seeded((q) => {
      q.catCoSubstrate = 1;
      q.ambientEnergy = 0.3;
    });
    const beforeScent = (() => {
      const d = sim.fields.data;
      let s = 0;
      for (let k = CH.conP; k < d.length; k += CHANNELS) s += d[k];
      return s;
    })();
    for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
    const afterScent = (() => {
      const d = sim.fields.data;
      let s = 0;
      for (let k = CH.conP; k < d.length; k += CHANNELS) s += d[k];
      return s;
    })();
    // The species was consumed, which it could not be without the ground.
    expect(afterScent).toBeLessThan(beforeScent);
  });

  it('leaves a slope to climb rather than a cliff', () => {
    /*
     * The property the whole design turns on. A hard requirement would make
     * machinery worthless until complete and leave selection nothing to
     * ascend; blending means every increment of ground availability is worth
     * something, monotonically.
     */
    const fed = (co: number, ambient: number): number => {
      const { sim, p, a } = seeded((q) => {
        q.catCoSubstrate = co;
        q.ambientEnergy = ambient;
      });
      const before = a.extra;
      for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
      return a.extra - before;
    };
    /*
     * Levels against the scent the dish is seeded with, which is 4. The gate
     * reads the *sample*, and a body standing in a cell that is 99 parts scent
     * to one part ground swallows 99 parts scent — so what matters here is the
     * ratio, and levels far under the scent all sit at the same floor rather
     * than on the slope. That is the mechanism, not a threshold: it is
     * continuous in the ratio the whole way.
     */
    const none = fed(1, 0);
    const some = fed(1, 1);
    const plenty = fed(1, 4);
    expect(some).toBeGreaterThan(none);
    expect(plenty).toBeGreaterThan(some);
  });

  it('still conserves', () => {
    const { sim, p } = seeded((q) => {
      q.catCoSubstrate = 1;
      q.ambientEnergy = 0.3;
    });
    const before = matter(sim, p.bodyValue);
    for (let i = 0; i < 120; i++) sim.step(1 / 60, p);
    /*
     * Relative, and bounded well above what the ground can represent.
     *
     * This asked for `toBeCloseTo(before, 5)` — half of 1e-5 absolute on a
     * total near 50,000, which is 1e-10 relative. `Fields.data` is a
     * `Float32Array`, so a cell holding ~12 units resolves to about 1e-6, and
     * a total spread over thousands of cells cannot be pinned tighter than
     * the square root of that count times a cell's quantum. The old bound
     * held only because the bodies kept landing on the same cells: turning on
     * `grip` moved them, the distribution changed, and the total shifted by
     * 3e-5 without a unit going anywhere.
     *
     * 1e-7 relative is a hundred and fifty times the drift this run actually
     * shows and still far tighter than any leak would be. A real one grows
     * with the run; this does not — see the packet path's own conservation in
     * `energy.test.ts`, held to 1e-4 over nine hundred frames of a pond that
     * latches, commutes and annihilates.
     */
    const after = matter(sim, p.bodyValue);
    expect(Math.abs(after - before) / before, `drifted ${after - before}`).toBeLessThan(1e-7);
  });
});
