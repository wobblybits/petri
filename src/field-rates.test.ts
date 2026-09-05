import { describe, expect, it } from 'vitest';
import { CH, CHANNELS, Fields } from './fields.ts';

/**
 * Per-channel diffuse and decay rates.
 *
 * The four channels used to be four of the same substance and are about to
 * stop being: energy wants to spread and stay, a signal wants to spread and
 * fade, and a reaction-diffusion pair only patterns at all when its two
 * species move at different speeds. The rates are the multiplier that lets one
 * global slider mean different things per channel.
 *
 * The important property is the boring one. All ones has to be *exactly* the
 * pass it replaces — not close to it — because everything that already works
 * runs at all ones, and a field that lands a few bits off moves a settled pond
 * for no reason anybody could later find.
 */

/** Somewhere inside the grid, away from the rim, in world units. */
const AT = { x: 5000, y: 5000 };

function seeded(): Fields {
  const f = new Fields();
  for (let ch = 0; ch < CHANNELS; ch++) f.deposit(ch, AT.x, AT.y, 10);
  return f;
}

/** Total of one channel over the whole grid. */
function total(f: Fields, ch: number): number {
  let sum = 0;
  for (let i = ch; i < f.data.length; i += CHANNELS) sum += f.data[i];
  return sum;
}

/** One frame's worth of field work, as `Sim.step` does it. */
function frame(f: Fields, diffuse = 0.6, decay = 0.01): void {
  f.diffuse(diffuse);
  f.diffuse(diffuse * 0.65);
  f.decay(decay);
}

describe('per-channel field rates', () => {
  it('starts at one on every channel, so a fresh field is the old field', () => {
    const f = new Fields();
    for (let ch = 0; ch < CHANNELS; ch++) {
      expect(f.diffuseRate[ch]).toBe(1);
      expect(f.decayRate[ch]).toBe(1);
    }
  });

  it('leaves every channel identical while the rates are equal', () => {
    const f = seeded();
    for (let i = 0; i < 30; i++) frame(f);
    // Same deposit, same rates, so the four channels cannot have diverged —
    // bit-for-bit, not within a tolerance.
    for (let ch = 1; ch < CHANNELS; ch++) {
      for (let i = 0; i < f.data.length; i += CHANNELS) {
        if (f.data[i] !== f.data[i + ch]) {
          throw new Error(`channel ${ch} diverged from 0 at cell ${i / CHANNELS}`);
        }
      }
    }
  });

  /*
   * Conservation, which is the whole reason the rates exist.
   *
   * Diffusion only moves a channel about, so one that does not decay keeps
   * what it was given — and it has to keep it *exactly*, because energy is
   * about to live on this grid and a leak that is small, silent and unbounded
   * in time is the one kind the economy cannot absorb. A tenth of a percent a
   * minute is a tenth of the pond an hour later.
   *
   * What is left is float32's own rounding, a few parts in ten million over a
   * minute of passes, in both directions. Nothing structural: the box widens
   * a cell ahead of the front on every pass, so the front never reaches an
   * edge that would clip it, and the only boundary the field has is the
   * Dirichlet disk — which absorbs on purpose, and which the deposit here
   * stays well clear of.
   */
  it('conserves a channel whose decay rate is zero', () => {
    const f = seeded();
    f.decayRate[CH.energy] = 0;
    const before = total(f, CH.energy);
    for (let i = 0; i < 60; i++) frame(f);
    const after = total(f, CH.energy);
    const decayed = total(f, CH.conP);

    expect(Math.abs(after - before) / before, 'a conserved channel drifted').toBeLessThan(1e-5);
    // While its neighbours, on the same field and the same passes, do not.
    expect(decayed).toBeLessThan(before * 0.6);
  });

  it('spreads a slow channel less far than a fast one', () => {
    const f = seeded();
    f.diffuseRate[CH.energy] = 0.25;
    for (let i = 0; i < 40; i++) frame(f);

    // Peak height is the readable proxy for how far a blob has spread: the
    // same mass over a wider disc is a lower one.
    let fastPeak = 0;
    let slowPeak = 0;
    for (let i = 0; i < f.data.length; i += CHANNELS) {
      if (f.data[i + CH.conP] > fastPeak) fastPeak = f.data[i + CH.conP];
      if (f.data[i + CH.energy] > slowPeak) slowPeak = f.data[i + CH.energy];
    }
    expect(slowPeak).toBeGreaterThan(fastPeak * 1.5);
  });

  it('treats a rate of zero as a channel that does not move at all', () => {
    const f = seeded();
    f.diffuseRate[CH.aux] = 0;
    f.decayRate[CH.aux] = 0;
    const before = total(f, CH.aux);
    // The deposit is bilinear across four cells; with no diffusion and no
    // decay those four cells are the whole channel, forever.
    const touched: number[] = [];
    for (let i = 0; i < f.data.length; i += CHANNELS) {
      if (f.data[i + CH.aux] !== 0) touched.push(i);
    }
    for (let i = 0; i < 60; i++) frame(f);
    expect(total(f, CH.aux)).toBeCloseTo(before, 5);
    for (let i = 0; i < f.data.length; i += CHANNELS) {
      const still = f.data[i + CH.aux] !== 0;
      expect(still, `cell ${i / CHANNELS} changed occupancy`).toBe(touched.includes(i));
    }
  });

  it('clamps a rate that would overshoot instead of going unstable', () => {
    const f = seeded();
    // A diffusion mix above 1 is a cell that overshoots its own neighbours,
    // which oscillates and then blows up. Rate * mix is clamped at 1, so the
    // worst a channel can do is become the average of its neighbours.
    f.diffuseRate[CH.dupP] = 50;
    f.decayRate[CH.dupP] = 50;
    for (let i = 0; i < 60; i++) frame(f);
    for (let i = 0; i < f.data.length; i += CHANNELS) {
      expect(Number.isFinite(f.data[i + CH.dupP])).toBe(true);
      expect(f.data[i + CH.dupP]).toBeGreaterThanOrEqual(0);
    }
  });

  it('holds the disk mask however the rates are set', () => {
    const f = seeded();
    f.decayRate[CH.energy] = 0;
    f.diffuseRate[CH.energy] = 1;
    // A tight disk around the deposit: everything outside it must stay zero,
    // including the channel that never decays.
    f.setWorldBound(AT.x, AT.y, 200);
    for (let i = 0; i < 60; i++) frame(f);
    for (let j = 0; j < f.rows; j++) {
      for (let i = 0; i < f.cols; i++) {
        if (!f.cellOut(i, j)) continue;
        const base = (j * f.cols + i) * CHANNELS;
        for (let ch = 0; ch < CHANNELS; ch++) {
          expect(f.data[base + ch], `ch ${ch} leaked outside the disk`).toBe(0);
        }
      }
    }
  });
});
