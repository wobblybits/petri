import { describe, expect, it } from 'vitest';
import { CH, CHANNELS, FERTILISE_CH, FIELD_CELL, Fields } from './fields.ts';
import { EnergyGrid } from './energy.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * The ground, now that it lives on the field.
 *
 * It used to be a seam of ore: a sparse map with an implicit ambient, no
 * transport but occupancy, and no way back once a cell was taken. A pond ate
 * outward from wherever it started and left dead lattice behind it forever,
 * which is the wrong shape for a world that is supposed to sustain anything.
 *
 * On the field it diffuses, so a grazed patch draws on its neighbours, and it
 * regrows logistically, so the dish has a carrying capacity rather than a
 * refill timer. The property that makes that ecology rather than bookkeeping
 * is in `growsBackFromItsEdges`: growth is proportional to what is there, so
 * zero is a fixed point and a patch taken to the floor can only be recolonised
 * from outside.
 */

const CAP = 0.0625;
/** A quarter-size dish, same ten-unit cell; the growth law is per-cell. */
const DISH = 256;
const AT = { x: 1280, y: 1280 };

function dish(radius = 1200): Fields {
  const f = new Fields(DISH, DISH * FIELD_CELL);
  f.setWorldBound(AT.x, AT.y, radius);
  f.decayRate[CH.energy] = 0;
  f.fillDisk(CH.energy, CAP);
  return f;
}

function total(f: Fields): number {
  let sum = 0;
  for (let i = CH.energy; i < f.data.length; i += CHANNELS) sum += f.data[i];
  return sum;
}

/** Energy in the one field cell containing a world point. */
function at(f: Fields, x: number, y: number): number {
  return f.sample(CH.energy, x, y);
}

/** Take everything from a disc of radius `r` around a point. */
function graze(f: Fields, x: number, y: number, r: number): void {
  const cs = f.cellSize;
  for (let j = 0; j < f.rows; j++) {
    for (let i = 0; i < f.cols; i++) {
      const dx = f.originX + (i + 0.5) * cs - x;
      const dy = f.originY + (j + 0.5) * cs - y;
      if (dx * dx + dy * dy <= r * r) f.data[(j * f.cols + i) * CHANNELS + CH.energy] = 0;
    }
  }
}

describe('ground', () => {
  it('lays full ground across the disk and nothing outside it', () => {
    const f = dish();
    for (let j = 0; j < f.rows; j++) {
      for (let i = 0; i < f.cols; i++) {
        const e = f.data[(j * f.cols + i) * CHANNELS + CH.energy];
        expect(e).toBe(f.cellOut(i, j) ? 0 : CAP);
      }
    }
  });

  it('never grows past capacity, however long it runs', () => {
    const f = dish();
    graze(f, AT.x, AT.y, 400);
    for (let i = 0; i < 2000; i++) f.grow(CH.energy, 0.05, CAP);
    let worst = 0;
    for (let i = CH.energy; i < f.data.length; i += CHANNELS) {
      if (f.data[i] > worst) worst = f.data[i];
    }
    expect(worst, 'a cell grew past the carrying capacity').toBeLessThanOrEqual(CAP + 1e-9);
  });

  it('leaves a cell above capacity alone rather than pulling it down', () => {
    // A corpse drops more than a cell can hold. That is not a signal to be
    // decayed back — it is energy, and the economy conserves it.
    const f = dish();
    f.addAt(CH.energy, AT.x, AT.y, 3);
    const before = total(f);
    for (let i = 0; i < 500; i++) f.grow(CH.energy, 0.05, CAP);
    expect(total(f)).toBeCloseTo(before, 6);
  });

  /*
   * The property the whole change exists for.
   *
   * Growth is `r * E * (1 - E/cap)`, so it is proportional to what is already
   * in the cell and zero is a fixed point. Graze a patch to the floor and no
   * amount of waiting brings it back — the only thing that can is diffusion
   * from a neighbour that still has something, which arrives from the rim
   * inward at a finite speed. Overgrazing makes a scar.
   */
  it('grows back from its edges, and not at all without them', () => {
    const sealed = dish();
    graze(sealed, AT.x, AT.y, 400);
    const emptied = total(sealed);
    // No diffusion: growth alone, for a long time.
    for (let i = 0; i < 5000; i++) sealed.grow(CH.energy, 0.05, CAP);
    expect(at(sealed, AT.x, AT.y), 'a dead cell regrew with no neighbour to seed it').toBe(0);
    expect(total(sealed), 'the scar healed from nothing').toBeCloseTo(emptied, 6);

    // Same scar, now with the ground allowed to spread into it.
    const healing = dish();
    graze(healing, AT.x, AT.y, 400);
    healing.diffuseRate[CH.energy] = 0.05;
    for (let i = 0; i < 5000; i++) {
      healing.diffuse(0.6);
      healing.grow(CH.energy, 0.05, CAP);
    }
    expect(at(healing, AT.x, AT.y), 'the scar never closed').toBeGreaterThan(CAP * 0.5);
    expect(total(healing)).toBeGreaterThan(emptied);
  });

  it('keeps a deposit whole right up against the rim', () => {
    // A point inside the disk can straddle cells whose centres are outside it,
    // and those are dropped. Renormalising over the ones that will take it is
    // what stops a body that dies near the wall taking part of its worth with
    // it — silently, and worse the closer to the edge it got.
    const r = 1200;
    for (const d of [0, 0.5, 0.9, 0.99]) {
      const f = new Fields(DISH, DISH * FIELD_CELL);
      f.setWorldBound(AT.x, AT.y, r);
      const before = total(f);
      f.addAt(CH.energy, AT.x + r * d, AT.y, 5);
      expect(total(f) - before, `deposit at ${d * 100}% of the radius`).toBeCloseTo(5, 6);
    }
  });

  it('drops a deposit that lands outside the dish entirely', () => {
    const f = new Fields(DISH, DISH * FIELD_CELL);
    f.setWorldBound(AT.x, AT.y, 1200);
    f.addAt(CH.energy, AT.x + 1250, AT.y, 5);
    expect(total(f)).toBe(0);
  });

  it('heals inward, so the middle of a scar is the last to come back', () => {
    const f = dish();
    graze(f, AT.x, AT.y, 400);
    f.diffuseRate[CH.energy] = 0.05;
    for (let i = 0; i < 300; i++) {
      f.diffuse(0.6);
      f.grow(CH.energy, 0.05, CAP);
    }
    const middle = at(f, AT.x, AT.y);
    const edge = at(f, AT.x + 380, AT.y);
    expect(edge, 'the rim of the scar should be ahead of its middle').toBeGreaterThan(middle);
  });
});

describe('ground through the sim', () => {
  it('starts a pinned pond on full ground, at the same total as the old grid', () => {
    const params = defaultParams();
    params.soupCount = 12;
    params.spawnInterval = 0;
    const sim = new Sim(1600, 900);
    for (let i = 0; i < 12; i++) sim.spawn('con', 700 + i * 12, 450, 0, params, true);
    sim.step(1 / 60, params);

    // An energy cell is `span^2` field cells, each holding `ambient / span^2`,
    // so a cell still holds `ambient` and the dish still holds what it did.
    const g = sim.energy;
    expect(g.span).toBe(params.energyCell / 10);
    expect(g.cellCap * g.span * g.span).toBeCloseTo(params.ambientEnergy, 9);

    const cells = Math.PI * sim.worldR * sim.worldR / (params.energyCell * params.energyCell);
    expect(g.storedTotal()).toBeGreaterThan(cells * params.ambientEnergy * 0.98);
    expect(g.storedTotal()).toBeLessThan(cells * params.ambientEnergy * 1.02);
  });

  it('keeps the coarse cell, so bodies in one place still compete', () => {
    // The view is coarse on purpose: `harvestSlots` groups by cell, and that
    // grouping is the economy's only crowding pressure. A field cell is ten
    // units and a body is bigger than that, so indexing straight to one would
    // have deleted the competition without anything failing.
    const params = defaultParams();
    const sim = new Sim(1600, 900);
    sim.spawn('con', 800, 450, 0, params, true);
    sim.step(1 / 60, params);
    const g = sim.energy;
    const a = g.index(800, 450);
    const b = g.index(800 + params.energyCell * 0.4, 450);
    expect(a.key, 'two bodies a third of a cell apart fell in different cells').toBe(b.key);
  });

  it('does not mint: with regrowth off, the ground only ever goes down', () => {
    const params = defaultParams();
    params.energyRegrow = 0;
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.rewriteDuration = 0;
    const sim = new Sim(1600, 900);
    for (let i = 0; i < 8; i++) sim.spawn('con', 700 + i * 40, 450, i, params, true);
    sim.step(1 / 60, params);
    let last = sim.energy.storedTotal();
    for (let f = 0; f < 120; f++) {
      sim.step(1 / 60, params);
      const now = sim.energy.storedTotal();
      expect(now, 'the ground gained energy with regrowth off').toBeLessThanOrEqual(last + 1e-4);
      last = now;
    }
  });

  it('binds the grid to the field, so the ground is one of the channels', () => {
    const params = defaultParams();
    const sim = new Sim(1600, 900);
    sim.spawn('con', 800, 450, 0, params, true);
    sim.step(1 / 60, params);
    // Not a separate store any more: what the grid reports and what sits on
    // channel 2 are the same numbers.
    let field = 0;
    for (let i = CH.energy; i < sim.fields.data.length; i += CHANNELS) field += sim.fields.data[i];
    expect(sim.energy.storedTotal()).toBeCloseTo(field, 6);
  });

  it('leaves an unbound grid on its old sparse behaviour', () => {
    // Every test that builds an `EnergyGrid` by hand still gets the map.
    const g = new EnergyGrid(40, 1);
    const { key } = g.index(0, 0);
    expect(g.take(key, 0.4)).toBeCloseTo(0.4, 9);
    expect(g.getCell(0, 0)).toBeCloseTo(0.6, 9);
  });
});

/**
 * The fertiliser term: `r * (1 + fertilise * C)` on the growth rate.
 *
 * What it buys over `farmRate` is a reason for two lineages to need each other.
 * Farming moves stock from a tank onto the dish, and the body that invests is
 * the body that gets it back. This is catalysis — a lineage emitting on the
 * fertiliser channel creates nothing, it makes the ground recover faster
 * wherever it stands, bounded by the same carrying capacity. It cannot feed
 * itself that way; it can only make its patch worth more to somebody else.
 */
describe('fertiliser', () => {
  /** A dish grazed to a fraction of capacity, so growth has room to act. */
  function grazed(): Fields {
    const f = dish();
    for (let i = CH.energy; i < f.data.length; i += CHANNELS) {
      if (f.data[i] > 0) f.data[i] = CAP * 0.25;
    }
    return f;
  }

  /** Lay the catalyst across the whole dish at `c`. */
  function spread(f: Fields, c: number): void {
    const { lo, hi } = (f as unknown as { spans(): { lo: Int32Array; hi: Int32Array } }).spans();
    for (let j = 0; j < f.rows; j++) {
      for (let i = lo[j]; i <= hi[j]; i++) {
        f.data[(j * f.cols + i) * CHANNELS + FERTILISE_CH] = c;
      }
    }
  }

  it('grows the ground faster where the catalyst is', () => {
    const bare = grazed();
    const fed = grazed();
    spread(fed, 1);
    for (let i = 0; i < 200; i++) {
      bare.grow(CH.energy, 0.01, CAP, FERTILISE_CH, 3);
      fed.grow(CH.energy, 0.01, CAP, FERTILISE_CH, 3);
    }
    expect(total(fed), 'the catalyst did nothing').toBeGreaterThan(total(bare) * 1.05);
  });

  it('does nothing at all when the dial is off', () => {
    const a = grazed();
    const b = grazed();
    spread(a, 1);
    spread(b, 1);
    for (let i = 0; i < 50; i++) {
      a.grow(CH.energy, 0.01, CAP, FERTILISE_CH, 0);
      b.grow(CH.energy, 0.01, CAP);
    }
    expect(total(a)).toBeCloseTo(total(b), 9);
  });

  it('never grows past capacity, catalyst or not', () => {
    const f = grazed();
    spread(f, 4);
    for (let i = 0; i < 500; i++) f.grow(CH.energy, 0.05, CAP, FERTILISE_CH, 8);
    let worst = 0;
    for (let i = CH.energy; i < f.data.length; i += CHANNELS) {
      if (f.data[i] > worst) worst = f.data[i];
    }
    expect(worst, 'catalysis raised the ceiling').toBeLessThanOrEqual(CAP + 1e-9);
  });

  it('lets an inhibitor stall growth but never run it backwards', () => {
    // A negative gamma is a body that poisons the ground around it, which is a
    // thing a lineage should be able to become. Ground destroyed by being
    // smelled at would be a hole in the conservation the economy depends on,
    // so the effective rate clamps at zero rather than going negative.
    const f = grazed();
    spread(f, 1);
    const before = total(f);
    for (let i = 0; i < 200; i++) f.grow(CH.energy, 0.01, CAP, FERTILISE_CH, -5);
    expect(total(f), 'an inhibitor destroyed ground').toBeGreaterThanOrEqual(before - 1e-6);
  });

  it('refuses to catalyse a channel with itself', () => {
    const a = grazed();
    const b = grazed();
    for (let i = 0; i < 50; i++) {
      a.grow(CH.energy, 0.01, CAP, CH.energy, 5);
      b.grow(CH.energy, 0.01, CAP);
    }
    expect(total(a)).toBeCloseTo(total(b), 9);
  });
});
