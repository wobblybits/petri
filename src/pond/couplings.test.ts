import { describe, expect, it } from 'vitest';
import { defaultParams } from '../params.ts';
import { COUPLINGS, checkCouplings } from './couplings.ts';

/*
 * The table is data about the model, and the two things that can go wrong
 * with it are a name that is not a parameter and a check that does not fire
 * for the case it was written from. The second case here is the one that
 * cost two afternoons: an axis swept with a coupled constant held.
 */
describe('couplings', () => {
  it('names only real parameters', () => {
    const known = new Set(Object.keys(defaultParams()));
    for (const c of COUPLINGS) {
      expect(known.has(c.axis), `${c.axis} is not a parameter`).toBe(true);
      expect(known.has(c.constant), `${c.constant} is not a parameter`).toBe(true);
      if (c.unless) expect(known.has(c.unless.key), `${c.unless.key} is not a parameter`).toBe(true);
    }
  });

  it('warns when uptakeVmax is swept with the gut dials held', () => {
    // The gut dials are coupled to the metering switch, both of them: nothing
    // fills a gut at `uptakeVmax` 0, so they are inert there and live above it.
    const warnings = checkCouplings(['uptakeVmax']).map((w) => w.constant);
    expect(warnings).toContain('digestRate');
    expect(warnings).toContain('gutSize');
    // `senseScale` used to be here, coupled to `excreteRate`; a signal is
    // always minted now, so there is no second regime to re-choose it for.
    expect(warnings).not.toContain('senseScale');
  });

  it('is quiet when the coupled constant moves with the axis, on the grid or per arm', () => {
    // `digestRate` and `gutSize` are here for the `uptakeVmax` axis in the
    // grid form; per arm they are moot, because `uptakeVmax` is not an axis.
    const moved = ['uptakeVmax', 'digestRate', 'gutSize'];
    expect(checkCouplings(['uptakeVmax', ...moved])).toEqual([]);
    expect(checkCouplings(['uptakeVmax'], moved)).toEqual([]);
  });

  it('is quiet about an axis nothing is coupled to', () => {
    expect(checkCouplings(['drag', 'turnRate'])).toEqual([]);
  });

  it('skips a coupling the held settings make moot, and only then', () => {
    // A layout comparison with growth off is the clean control the fix asks for.
    const off = checkCouplings(['groundPatches'], [], { ...defaultParams(), energyRegrow: 0, learnRate: 0 } as unknown as Record<string, number>);
    expect(off).toEqual([]);
    // With growth on and learning on, both couplings fire. Both are set
    // explicitly rather than taken from the defaults: `energyRegrow` ships at
    // 0 now — the drops are the income and regrowth was what flattened the
    // dish — so "growth on" is a thing this test has to say, not inherit.
    const on = checkCouplings(['groundPatches'], [], { ...defaultParams(), energyRegrow: 0.02, learnRate: 0.01 } as unknown as Record<string, number>);
    expect(on.map((w) => w.constant).sort()).toEqual(['energyRegrow', 'learnDiscount', 'learnTrace']);
  });
});
