import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { portKeyAt, slotIndex, stemRoot, portLocal, stemOffset, createAgent } from './agents.ts';
import type { PortSlot } from './agents.ts';
import { defaultParams } from './params.ts';

/**
 * The glyph geometry, pinned.
 *
 * `stemRoot` is the single definition of where a port sits on a body, and it
 * is duplicated by hand in native/solver.c. `stemOffsetInto` was added to give
 * the packing loops an allocation-free path to the same numbers — the
 * allocating form costs three objects a call and the FAR pack makes two calls
 * per wire. Both are easy to drift; neither has an obvious symptom when it
 * does, because everything downstream just quietly bends.
 *
 * The literals below pin this side. `stem_local` in solver.c is pinned to them
 * at the bottom of this file, by reading its own numbers out of the C: a
 * suite that fixes one copy and lets the other move is only half a pin, and
 * the half it leaves loose is the one running in the pond most of the time.
 */
describe('stem geometry', () => {
  it('stemRoot is unchanged', () => {
    expect(stemRoot('era', 'p')).toEqual({ x: 8, y: 0 });
    expect(stemRoot('era', 'l')).toEqual({ x: 0, y: 0 });
    expect(stemRoot('era', 'r')).toEqual({ x: 0, y: 0 });
    for (const k of ['con', 'dup'] as const) {
      expect(stemRoot(k, 'p')).toEqual({ x: 16 * 1.05, y: 0 });
      expect(stemRoot(k, 'l')).toEqual({ x: -16 * 0.55, y: -(16 * 0.82 * 0.7) });
      expect(stemRoot(k, 'r')).toEqual({ x: -16 * 0.55, y: 16 * 0.82 * 0.7 });
    }
  });

  it('stemRoot results are independent objects', () => {
    const a = stemRoot('con', 'l');
    const b = stemRoot('con', 'r');
    expect(a).not.toBe(b);
    expect(a.y).toBe(-(16 * 0.82 * 0.7));
  });

  it('stemOffset matches a hand-rolled rotation', () => {
    const params = defaultParams();
    for (const k of ['era', 'con', 'dup'] as const) {
      for (const slot of ['p', 'l', 'r'] as const) {
        for (const h of [0, 0.7, -2.1, 3.0]) {
          const ag = createAgent(1, k, 0, 0, h, params);
          ag.scale = 0.83;
          const root = stemRoot(k, slot);
          const lx = root.x * ag.scale;
          const ly = root.y * ag.scale;
          const want = { x: lx * Math.cos(h) - ly * Math.sin(h), y: lx * Math.sin(h) + ly * Math.cos(h) };
          const got = stemOffset(ag, slot);
          expect(got.x, `${k}.${slot} h=${h}`).toBeCloseTo(want.x, 12);
          expect(got.y, `${k}.${slot} h=${h}`).toBeCloseTo(want.y, 12);
        }
      }
    }
  });

  it('portLocal still extrudes the right way', () => {
    expect(portLocal('era', 'p')).toEqual({ x: 16, y: 0 });
    expect(portLocal('con', 'p')).toEqual({ x: 16 * 1.05 + 8, y: 0 });
    expect(portLocal('con', 'l').x).toBe(-16 * 0.55 - 8);
  });
});

describe('port keys', () => {
  it('never gives two ports the same number', () => {
    // The key is id * 3 + slot, used as a Map key for every free-port test in
    // the sim. A collision would read as a port being occupied by a wire
    // belonging to someone else, silently.
    const slots: PortSlot[] = ['p', 'l', 'r'];
    const seen = new Map<number, string>();
    for (let id = 1; id <= 5000; id++) {
      for (const slot of slots) {
        const k = portKeyAt(id, slot);
        const where = `${id}.${slot}`;
        expect(seen.has(k), `${where} collides with ${seen.get(k)}`).toBe(false);
        seen.set(k, where);
      }
    }
    expect(seen.size).toBe(15000);
  });

  it('numbers the slots the way the solver does', () => {
    expect(slotIndex('p')).toBe(0);
    expect(slotIndex('l')).toBe(1);
    expect(slotIndex('r')).toBe(2);
  });
});

describe('the solver carries the same glyph geometry', () => {
  /*
   * `stem_local` in native/solver.c is `stemRootInto` transcribed by hand, and
   * it is the copy the pond actually runs whenever the native force passes are
   * on. Nothing compared the two. A drift here does not crash and does not
   * look like a bug: ports sit slightly off their glyphs, every wire leaves at
   * a slightly wrong angle, and the pond simply settles somewhere else.
   *
   * Read out of the C rather than restated, so this cannot drift the way the
   * thing it is checking did. If `stem_local` is rewritten into a shape these
   * patterns do not match, that is a failure too — it should be, because the
   * transcription is what is being trusted.
   */
  const csrc = readFileSync(new URL('../native/solver.c', import.meta.url), 'utf8');
  const body = /static void stem_local\([^)]*\)\s*\{([\s\S]*?)\n\}/.exec(csrc);

  /** The nth float literal in `stem_local`, in source order. */
  function literals(): number[] {
    expect(body, 'no stem_local in native/solver.c').not.toBeNull();
    // `8.f` and `-1.f` are the C spelling: a trailing dot with no digits.
    return [...body![1].matchAll(/(-?\d+(?:\.\d*)?)f\b/g)].map((m) => Number(m[1]));
  }

  it('uses the same numbers the glyph is drawn from', () => {
    // In the order `stem_local` writes them: the Era stem and its two zeroes,
    // the Con/Dup size, the principal's reach, a zero, the legs' setback, the
    // two leg signs, the half base and the leg fraction.
    expect(literals()).toEqual([8, 0, 0, 16, 1.05, 0, 0.55, -1, 1, 0.82, 0.7]);
  });

  it('puts every port where stemRoot does, at scale 1', () => {
    const [eraStem, , , size, reach, , back, , , halfBase, legFrac] = literals();
    /** `stem_local` re-run in JS from its own literals, at `sc` 1. */
    const cStem = (kind: 'era' | 'con' | 'dup', slot: 0 | 1 | 2): { x: number; y: number } => {
      if (kind === 'era') return { x: slot === 0 ? eraStem : 0, y: 0 };
      const s = size;
      if (slot === 0) return { x: s * reach, y: 0 };
      return { x: -s * back, y: (slot === 1 ? -1 : 1) * s * halfBase * legFrac };
    };
    for (const kind of ['era', 'con', 'dup'] as const) {
      for (const slot of ['p', 'l', 'r'] as const) {
        const mine = stemRoot(kind, slot);
        const theirs = cStem(kind, slotIndex(slot) as 0 | 1 | 2);
        expect(theirs.x, `solver.c's ${kind} ${slot} x`).toBeCloseTo(mine.x, 10);
        expect(theirs.y, `solver.c's ${kind} ${slot} y`).toBeCloseTo(mine.y, 10);
      }
    }
  });

  it('agrees which number means which port', () => {
    /*
     * `stem_local` branches on `slot == 0` and `slot == 1`, and `Sim.slotCode`
     * is what fills that field. If the two ever disagreed about which is the
     * principal, every body would wire from its legs.
     */
    expect(slotIndex('p')).toBe(0);
    expect(slotIndex('l')).toBe(1);
    expect(slotIndex('r')).toBe(2);
    expect(/slot == 0/.test(body![1]), "solver.c stopped keying the principal on 0").toBe(true);
    expect(/slot == 1 \? -1/.test(body![1]), "solver.c stopped keying the left leg on 1").toBe(true);
  });
});
