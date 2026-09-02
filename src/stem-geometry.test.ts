import { describe, expect, it } from 'vitest';
import { stemRoot, portLocal, stemOffset, createAgent } from './agents.ts';
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
