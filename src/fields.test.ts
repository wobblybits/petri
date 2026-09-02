import { describe, expect, it } from 'vitest';
import { Fields } from './fields.ts';

describe('Fields.markSegment', () => {
  it('returns quickly on a span that would otherwise walk the plane', () => {
    const fields = new Fields(800, 600);
    const t0 = Date.now();
    fields.markSegment(0, 0, 1e12, 1e12);
    fields.markSegment(NaN, 0, 1, 1);
    fields.markSegment(0, 0, Infinity, Infinity);
    expect(Date.now() - t0).toBeLessThan(100);
  });
});
