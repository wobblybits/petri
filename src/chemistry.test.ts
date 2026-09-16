import { describe, expect, it } from 'vitest';
import { bareBody, seedChem } from './agents.ts';
import { CHEM_LEN, CHEM_SPECIES, KS_BASE, uptakeKsOf } from './chem-layout.ts';
import { HarvestPlan, REWRITE_SHARE, harvestSlotsFast, uptakeRate, type UptakeKinetics } from './energy.ts';
import { CH, CHANNELS } from './fields.ts';
import type { Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';
import { channelTotal, fixedParams, pondMatter } from './test-params.ts';

/*
 * The body reaction table.
 *
 * Two properties carry the whole design and both are testable directly. The
 * simplex is what makes expression a *budget* — a body cannot both shout and
 * eat without giving something up — and conservation is what makes the budget
 * honest, because a body that has nothing cannot spend anything however loudly
 * its genome would like to.
 */

/**
 * A pond with the dish switched off, so only body reactions move anything.
 * Not the conservation control `energy.test.ts` builds — the mint stays on,
 * because half of what is tested here is what the mint does.
 *
 * `ambientEnergy = 0` matters more than it looks: with ground in the dish the
 * bodies harvest between steps, and a test that means to watch a tank empty
 * watches it refill instead. `diffuse = 0` matters for the same kind of
 * reason — the rim absorbs the three signal species, which is the dish doing
 * its job and not a body failing to conserve.
 */
function chemistryParams(): Params {
  const p = fixedParams();
  p.soupCount = 60;
  p.spawnInterval = 0;
  p.energyRegrow = 0;
  p.ambientEnergy = 0;
  p.decay = 0;
  p.diffuse = 0;
  p.upkeep = 0;
  /*
   * The gait's pathway buys substrate out of the tank, and `upkeepExcrete` is
   * what decides whether that spend lands on the dish or is destroyed. It is
   * rent by another name and it is on by default, so a conservation test that
   * left this at zero was watching a pond with a spender it had not accounted
   * for. Nothing else here is affected: `upkeep` is zero, so there is no rent
   * for it to route.
   */
  p.upkeepExcrete = 1;
  // Bodies conservative, so a commute is a transfer and not a mint. See
  // `matter` and `energy.test.ts`'s conservation suite.
  p.bodyValue = REWRITE_SHARE;
  return p;
}

/**
 * One body in a pond with the dish off, standing on a block this test paints.
 *
 * One frame is stepped before anything is read, for two reasons. The grid
 * moves onto `fields` when the world is pinned and until then answers out of
 * a sparse map nothing here writes to; and `refreshExpression` is what fills
 * the uptake rows, so before a frame has run every row is zero and a body
 * cannot digest anything it swallows. `paint` lays `per[c]` of every species
 * flat across the body's own block, so `densityOf` reads exactly `per`.
 */
function oneBody(tweak: (p: Params) => void) {
  const p = chemistryParams();
  p.soupCount = 1;
  p.uptakeVmax = 6;
  tweak(p);
  const sim = new Sim(1600, 1200, 128);
  loadPreset(sim, 'soup', p);
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
  // One gut, one number: a body eats the ground and nothing else.
  const gut = (of = a): number => sim.agentStore.gut[of.slot];
  const held = (of = a): number => sim.agentStore.gutTotal(of.slot);
  const emptyGut = (of = a): void => {
    sim.agentStore.gut[of.slot] = 0;
  };
  return { sim, p, a, paint, gut, held, emptyGut };
}

/** `v` of one species and nothing else. */
function alone(c: number, v: number): number[] {
  const per = new Array(CHANNELS).fill(0);
  per[c] = v;
  return per;
}

/** The harvest's kinetics for one frame of `p`, the way `Sim` builds them. */
function kineticsOf(p: Params, dt = 1 / 60): UptakeKinetics {
  return { cap: p.uptakeVmax * dt, ks: p.uptakeKs, yDirect: p.yDirect, yEra: p.yEra, hillN: p.hillN, gutSize: p.gutSize };
}

describe('uptake affinity', () => {
  it('is one gene, seeded at the global and floored above zero', () => {
    /*
     * One species is eaten, so one affinity. It used to be four, one per
     * channel, back when a mouthful was a sample of the water.
     */
    const chem = seedChem('dup', fixedParams());
    // Seeded at one natural unit, so a fresh body uses the global.
    expect(uptakeKsOf(chem, 0, 0.25)).toBeCloseTo(0.25, 12);
    const half = seedChem('dup', fixedParams());
    half[KS_BASE] = 0.5;
    expect(uptakeKsOf(half, 0, 0.25)).toBeCloseTo(0.125, 12);
    // A gene mutated to or past zero would be an infinitely good transporter,
    // and a division by zero downstream.
    const dead = seedChem('dup', fixedParams());
    dead[KS_BASE] = -3;
    expect(uptakeKsOf(dead, 0, 0.25)).toBeGreaterThan(0);
  });

  it('is the whole of what a body owns about eating', () => {
    /*
     * `X` and `x0` were a head from `h` to eight reaction rows — four for what
     * a body produced and four for what it could digest. Nothing excretes and
     * there is one thing to digest, so both halves lost their subject and the
     * forty floats went. What a lineage owns of `(vmax, ks)` is the affinity;
     * `vmax` is `uptakeVmax`, a law of the world.
     */
    const chem = seedChem('con', fixedParams());
    expect(chem.length).toBe(CHEM_LEN);
    expect(bareBody(chem).chem.length).toBe(CHEM_LEN);
    // Seeded at exactly one, which is "use the global".
    expect(chem[KS_BASE]).toBe(1);
  });
});

describe('uptake', () => {
  it('eats the ground and leaves the signals alone', () => {
    /*
     * The rule that replaced "a mouthful is a sample of the water".
     *
     * A body used to swallow all four channels at once, so a scent was a meal
     * as well as a message: `params.deposit` minted food by shouting, a body
     * could eat its own signal back, and every conservation sum had to carry
     * four columns. Matter is the ground and signal is the other three, and
     * nothing crosses. One channel is eaten; the rest are left where they are.
     */
    const { sim, p, a, paint, gut, emptyGut } = oneBody((q) => {
      q.uptakeVmax = 0.6;
    });
    const kinetics = kineticsOf(p);
    paint(new Array(CHANNELS).fill(0.4));
    emptyGut();
    const before: number[] = [];
    for (let c = 0; c < CHANNELS; c++) before.push(channelTotal(sim, c));
    harvestSlotsFast([a], sim.agentStore, sim.energy, new HarvestPlan(), kinetics);
    expect(gut(), 'it ate the ground').toBeGreaterThan(0);
    for (let c = 0; c < CHANNELS; c++) {
      if (c === CH.energy) continue;
      expect(channelTotal(sim, c), `channel ${c} is where it was`).toBeCloseTo(before[c], 9);
    }
  });

  it('is not slowed by a cell full of signal', () => {
    /*
     * The inverse of the test this replaced, and the point of the change.
     *
     * A smelly cell used to be a poor one: four species competed for one gut
     * and each could take at most its share of the budget, so ground in a
     * filthy cell cost four times the mouthful to get at. A signal is not a
     * pollutant any more. Same ground, same draw, however much shouting is
     * going on in the same water.
     */
    const { sim, p, a, paint, gut, emptyGut } = oneBody((q) => {
      q.uptakeVmax = 0.6;
    });
    const kinetics = kineticsOf(p);
    const draw = (per: number[]): number => {
      paint(per);
      emptyGut();
      harvestSlotsFast([a], sim.agentStore, sim.energy, new HarvestPlan(), kinetics);
      return gut();
    };
    const clean = draw(alone(CH.energy, 0.4));
    const noisy = alone(CH.energy, 0.4);
    for (let c = 0; c < CHANNELS; c++) if (c !== CH.energy) noisy[c] = 3;
    expect(draw(noisy), 'the signal changed nothing').toBeCloseTo(clean, 9);
  });

  it('caps a rich cell at the rate and a poor one below it', () => {
    /*
     * The Monod half, on the path that ships. On rich ground the draw sits
     * just under the budget; below half-saturation it is well under — and
     * this is the half of Monod that makes a low-Ks scavenger a viable
     * different strategy rather than a strictly worse grazer.
     */
    const { sim, p, a, paint, gut, emptyGut } = oneBody((q) => {
      q.uptakeVmax = 12;
    });
    const kinetics = kineticsOf(p);
    const ks = p.uptakeKs; // the seed's affinity gene is 1
    const draw = (density: number): number => {
      paint(alone(CH.energy, density));
      emptyGut();
      harvestSlotsFast([a], sim.agentStore, sim.energy, new HarvestPlan(), kinetics);
      return gut();
    };
    // Eight places: the draw came out of Float32 cells. See the note above.
    const rich = draw(4);
    expect(rich).toBeCloseTo(uptakeRate(4, kinetics.cap, ks), 8);
    expect(rich).toBeLessThan(kinetics.cap);
    const poor = draw(0.05);
    expect(poor).toBeCloseTo(uptakeRate(0.05, kinetics.cap, ks), 8);
    expect(poor).toBeLessThan(rich);
  });

  it('shares a contested cell instead of handing it to the lowest id', () => {
    /*
     * The id-order artifact §4 exists to remove: densities are read once for
     * the block before anybody eats, so two bodies on one cell draw the same
     * mouthful, and order only matters at exhaustion.
     */
    const { sim, p, a, paint, gut, emptyGut } = oneBody((q) => {
      q.uptakeVmax = 12;
    });
    // On the same spot, and no frame in between: the block was painted around
    // where `a` stood at the fixture's one step, and a body that is not pinned
    // walks off it.
    const b = sim.spawn('con', a.x, a.y, 0, p, true)!;
    paint(alone(CH.energy, 1));
    emptyGut(a);
    emptyGut(b);
    const before = sim.energy.storedTotal();
    harvestSlotsFast([a, b], sim.agentStore, sim.energy, new HarvestPlan(), kineticsOf(p));
    expect(gut(a)).toBeGreaterThan(0);
    expect(gut(b)).toBeCloseTo(gut(a), 9);
    expect(gut(a) + gut(b)).toBeLessThanOrEqual(before + 1e-9);
  });

  it('keeps matter and signal apart', () => {
    /*
     * What replaced "one body's excretion is another's food".
     *
     * There were four species because a body ate all of them and laid all of
     * them down, so a signal was a substance and the loop closed through the
     * scent field. It does not any more: a body eats the ground, excretes
     * nothing, and shouts on three channels that no stomach can reach. The
     * loop that matters is ground -> gut -> tank -> ground, and the signal
     * field sits outside it.
     */
    const p = chemistryParams();
    p.uptakeVmax = 1.5;
    p.deposit = 5;
    // The rig is a bare dish by default; there has to be ground to eat.
    p.ambientEnergy = 0.5;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 1;
    const signalBefore = channelTotal(sim, CH.conP);
    const groundBefore = channelTotal(sim, CH.energy);
    for (let i = 0; i < 60; i++) sim.step(1 / 60, p);
    // Bodies shouted, so the signal field grew — and none of it is matter.
    expect(channelTotal(sim, CH.conP), 'the voice is minted').toBeGreaterThan(signalBefore);
    // They ate, out of the one channel that is matter, and nothing else moved.
    expect(channelTotal(sim, CH.energy), 'the ground was eaten').toBeLessThan(groundBefore);
    const held = [...sim.agents.values()].reduce((n, a) => n + a.extra, 0);
    expect(held, 'tanks are not simply draining').toBeGreaterThan(0);
  });

  it('conserves with the whole table running', () => {
    const p = chemistryParams();
    p.uptakeVmax = 1.5;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', p);
    for (const a of sim.agents.values()) a.extra = 1;
    const before = pondMatter(sim, p.bodyValue);
    for (let i = 0; i < 180; i++) sim.step(1 / 60, p);
    const after = pondMatter(sim, p.bodyValue);
    expect(after / before, `matter went ${before} -> ${after}`).toBeCloseTo(1, 6);
  });

  it('gives a fast grazer and a scavenger different answers', () => {
    /*
     * What a lineage buys by breeding a better transporter depends on the dish
     * it is standing in, which is what §4 says the diversity comes from. Same
     * body, same expression, same species, different affinity: where there is
     * plenty the high-affinity gene barely helps, and where there is not it is
     * most of the difference. So no one gene is the answer everywhere, and
     * there is something for selection to pull apart.
     *
     * One painted body and one harvest. Reading `a.extra` out of a soup was not
     * the same question: the preset hands back whichever body it spawned
     * first, and across the four arms that was a Con, a Dup, a Con and an Era
     * — three kinds and three trophic yields, with the affinity the question
     * is about somewhere underneath. The arms have to differ in the gene and
     * in nothing else.
     */
    const { sim, p, a, paint, gut, emptyGut } = oneBody((q) => {
      q.uptakeVmax = 1.5;
      q.uptakeKs = 0.5;
    });
    const kinetics = kineticsOf(p);

    const run = (density: number, ksGene: number): number => {
      // One gene, because a body eats one species.
      a.chem[KS_BASE] = ksGene;
      paint(alone(CH.energy, density));
      // Room in the gut is what bounds a mouthful, so it is emptied between
      // runs; the tank does not enter into it.
      emptyGut();
      harvestSlotsFast([a], sim.agentStore, sim.energy, new HarvestPlan(), kinetics);
      return gut();
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
    /*
     * The gait's pathway off. These read the ground a body is standing on, and
     * the pathway buys substrate out of the tank and returns the price to that
     * ground — so a well-fed body fertilises the cell under it and a starving
     * one does not, which is the opposite sign to the thing under test and
     * large enough to flip it. It is a different mechanism spending from the
     * same tank, and this is a uptake-yield question.
     */
    p.metabolicRate = 0;
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
    /*
     * §5's replacement for `ERA_UPKEEP_RATIO`: an Era's income comes from the
     * ground under it, not from a rule keyed on its glyph.
     *
     * Which is exactly what swapping the two yields over asks. If the glyph
     * were doing the work the Era would win either way; it is the dial, so the
     * answer swaps with it.
     *
     * That, and not a ratio. A mouthful ten times the size does not buy ten
     * times the income: `total` scales with the yield but gut room does not,
     * and gut room is the same for both, so satiety compresses a tenfold mouth
     * to well under a twofold tank. This asked for twice and got 1.7, which is
     * the bound reading a pond that had the gait's pathway draining both tanks
     * when it was written rather than anything about the yield.
     */
    const earned = (yDirect: number, yEra: number): { era: number; con: number } => {
      const p = chemistryParams();
      p.ambientEnergy = 1;
      p.uptakeVmax = 1.5;
      p.yDirect = yDirect;
      p.yEra = yEra;
      // A different mechanism spending from the same tank — see `fed`, which
      // takes it off for the same reason.
      p.metabolicRate = 0;
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
      return { era: era.extra, con: con.extra };
    };
    const mouth = earned(0.2, 2);
    expect(mouth.era, 'the Era eats better when the Era yield is the high one').toBeGreaterThan(mouth.con);
    const swapped = earned(2, 0.2);
    expect(swapped.con, 'and worse when it is not, which a glyph rule could not do').toBeGreaterThan(swapped.era);
  });
});

describe('uptake shape', () => {
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
  it('holds what it swallowed until digestion takes it', () => {
    /*
     * A body swallows before it converts, so there is such a thing as an
     * un-metabolised substance inside a body — which is what makes "a full
     * tank and an empty gut has no clock" expressible at all.
     *
     * It used to be demonstrated with scent a body could not convert, back
     * when a mouthful was a sample of all four channels. A body eats the
     * ground and nothing else now, so the holding is shown the only way left:
     * stop digestion and watch the gut fill while the tank does not.
     */
    const { sim, p, a, paint, gut } = oneBody((q) => {
      q.digestRate = 0;
      // The reactor spends, which would move the tank for the wrong reason.
      q.metabolicRate = 0;
    });
    paint(alone(CH.energy, 0.6));
    a.extra = 0.5;
    const before = a.extra;
    for (let i = 0; i < 20; i++) sim.step(1 / 60, p);
    // Swallowed: it is inside the body, not on the ground.
    expect(gut(), 'the gut holds it').toBeGreaterThan(0);
    // And not yet money: nothing digests it, so the tank is where it started.
    expect(a.extra, 'the tank did not move').toBeCloseTo(before, 12);
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
    harvestSlotsFast([a], store, sim.energy, new HarvestPlan(), kineticsOf(p));
    // Ground under it, an empty tank, and not a unit taken: there is nowhere
    // to put a mouthful.
    expect(sim.energy.storedTotal()).toBeCloseTo(before, 12);
  });
});
