import { describe, expect, it } from 'vitest';
import { CH, CHANNELS, Fields } from './fields.ts';

/**
 * A field that computes, rather than four that only remember.
 *
 * Every channel is otherwise a hill around whoever is emitting, so what a body
 * smells is always *who is there*. A reaction between two of them puts maxima
 * where nobody is standing and fronts that travel on their own, which is the
 * difference between a field that carries information and one that holds any.
 *
 * These check the arithmetic and its guards, not the aesthetics. Whether a
 * given `feed`/`kill` lands somewhere interesting is a question for a screen —
 * the pair lives in a thin sliver of its own plane and the sim ships with the
 * reaction off.
 */

const AT = { x: 5000, y: 5000 };
const U = CH.conP;
const V = CH.dupP;

/**
 * The standard Gray-Scott initial condition: substrate at 1 everywhere, and a
 * small square where it has been knocked down to 0.5 with 0.25 of activator
 * put in its place.
 *
 * The scale matters and is easy to get wrong. Both species belong in [0, 1] —
 * `u` is normalised toward 1 by the feed term, and `uvv` is cubic, so an
 * activator seeded at 1 per cell consumes the entire substrate in a single
 * step, `u` clamps at zero, and the reaction then has nothing to run on and
 * simply decays. Which looks exactly like the kernel being wrong.
 */
function seeded(radius = 1200): Fields {
  const f = new Fields();
  f.setWorldBound(AT.x, AT.y, radius);
  f.fillDisk(U, 1);
  const cs = f.cellSize;
  for (let j = 0; j < f.rows; j++) {
    for (let i = 0; i < f.cols; i++) {
      const dx = f.originX + (i + 0.5) * cs - AT.x;
      const dy = f.originY + (j + 0.5) * cs - AT.y;
      if (Math.abs(dx) > 60 || Math.abs(dy) > 60) continue;
      const base = (j * f.cols + i) * CHANNELS;
      f.data[base + U] = 0.5;
      f.data[base + V] = 0.25;
    }
  }
  return f;
}

function total(f: Fields, ch: number): number {
  let s = 0;
  for (let i = ch; i < f.data.length; i += CHANNELS) s += f.data[i];
  return s;
}

describe('cross-channel reaction', () => {
  it('does nothing at all when both rates are off', () => {
    const f = seeded();
    const before = f.data.slice();
    f.react(U, V, 0, 0, 1);
    expect(f.data).toEqual(before);
  });

  it('refuses to react a channel with itself', () => {
    const f = seeded();
    const before = f.data.slice();
    f.react(U, U, 0.037, 0.06, 1);
    expect(f.data).toEqual(before);
  });

  it('converts substrate into activator where both are present', () => {
    const f = seeded();
    const u0 = f.sample(U, AT.x, AT.y);
    const v0 = f.sample(V, AT.x, AT.y);
    for (let i = 0; i < 40; i++) f.react(U, V, 0.037, 0.06, 0.5);
    // `u + 2v -> 3v` runs where the two overlap, so the substrate is eaten and
    // the activator grows. That is the whole autocatalysis.
    expect(f.sample(U, AT.x, AT.y), 'substrate was not consumed').toBeLessThan(u0);
    expect(f.sample(V, AT.x, AT.y), 'activator did not grow').toBeGreaterThan(v0);
  });

  it('leaves a cell alone where the activator is absent', () => {
    // Away from the seed there is no v, so uvv is zero and only the feed term
    // acts — the substrate relaxes toward 1 and nothing is created.
    const f = seeded();
    // Well clear of the seed but still inside the dish — outside it the
    // substrate is masked to zero and this would be testing the mask instead.
    const far = { x: AT.x + 800, y: AT.y };
    f.touchWorld(far.x, far.y);
    for (let i = 0; i < 40; i++) f.react(U, V, 0.037, 0.06, 0.5);
    expect(f.sample(V, far.x, far.y)).toBe(0);
    expect(f.sample(U, far.x, far.y)).toBeCloseTo(1, 6);
  });

  it('stays finite and non-negative over a long run', () => {
    const f = seeded();
    f.diffuseRate[U] = 1;
    f.diffuseRate[V] = 0.5;
    for (let i = 0; i < 400; i++) {
      f.diffuse(0.4);
      f.react(U, V, 0.037, 0.06, 0.6);
    }
    for (let i = 0; i < f.data.length; i += CHANNELS) {
      expect(Number.isFinite(f.data[i + U])).toBe(true);
      expect(Number.isFinite(f.data[i + V])).toBe(true);
      expect(f.data[i + U]).toBeGreaterThanOrEqual(0);
      expect(f.data[i + V]).toBeGreaterThanOrEqual(0);
    }
    // And it actually did something, rather than washing flat.
    expect(total(f, V)).toBeGreaterThan(0);
  });

  it('honours the disk, so a masked world stays masked', () => {
    const f = seeded(300);
    for (let i = 0; i < 40; i++) f.react(U, V, 0.037, 0.06, 0.5);
    for (let j = 0; j < f.rows; j++) {
      for (let i = 0; i < f.cols; i++) {
        if (!f.cellOut(i, j)) continue;
        const base = (j * f.cols + i) * CHANNELS;
        // The feed term would otherwise stock the whole grid with substrate,
        // including the parts of it that are not world.
        expect(f.data[base + U], 'the reaction fed cells outside the dish').toBe(0);
      }
    }
  });
});
