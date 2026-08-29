import { describe, expect, it } from 'vitest';
import { createAgent } from './agents.ts';
import { queryHit } from './collide.ts';
import { closestPointOnSegment, segmentsIntersect, segmentsInterfere } from './geom.ts';
import { defaultParams } from './params.ts';

describe('shape collision', () => {
  it('detects overlapping triangles and ignores separated ones', () => {
    const params = defaultParams();
    const a = createAgent(1, 'con', 80, 80, 0, params);
    const b = createAgent(2, 'con', 82, 80, Math.PI, params);
    expect(queryHit(a, b, 240, 160)).not.toBeNull();
    b.x = 160;
    expect(queryHit(a, b, 240, 160)).toBeNull();
  });

  it('detects a circle overlapping a triangle', () => {
    const params = defaultParams();
    const era = createAgent(1, 'era', 80, 80, 0, params);
    const con = createAgent(2, 'con', 90, 80, Math.PI, params);
    expect(queryHit(era, con, 240, 160)).not.toBeNull();
  });
});

describe('segment geometry', () => {
  it('detects a proper crossing and ignores a near miss', () => {
    expect(segmentsIntersect(0, 0, 10, 10, 0, 10, 10, 0)).toBe(true);
    expect(segmentsIntersect(0, 0, 10, 0, 0, 5, 10, 5)).toBe(false);
    expect(segmentsInterfere(0, 0, 20, 0, 10, 1, 10, 8, 5)).toBe(true);
    expect(segmentsInterfere(0, 0, 20, 0, 10, 8, 10, 16, 5)).toBe(false);
  });

  it('closestPointOnSegment pins to the interval and reports t', () => {
    const mid = closestPointOnSegment(5, 4, 0, 0, 10, 0);
    expect(mid.t).toBeCloseTo(0.5);
    expect(mid.x).toBeCloseTo(5);
    expect(mid.y).toBe(0);
    const past = closestPointOnSegment(20, 3, 0, 0, 10, 0);
    expect(past.t).toBe(1);
    expect(past.x).toBe(10);
  });
});
