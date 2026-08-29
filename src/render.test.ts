import { describe, expect, it } from 'vitest';
import { waveDisplace } from './render.ts';

describe('waveDisplace', () => {
  it('is silent below the envelope floor', () => {
    expect(waveDisplace(0.5, 0.5, 1e-6, 1)).toBe(0);
  });

  it('pins the ends', () => {
    expect(waveDisplace(1, 1, 1, 0)).toBe(0);
  });

  it('a quiet pluck and a loud ring occupy similar pixels', () => {
    const quiet = Math.abs(waveDisplace(0.02, 0.02, 0.04, 1));
    const loud = Math.abs(waveDisplace(0.5, 0.5, 1, 1));
    expect(quiet).toBeGreaterThan(8);
    expect(loud).toBeGreaterThan(8);
    expect(loud / quiet).toBeLessThan(3);
  });
});
