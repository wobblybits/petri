import { describe, expect, it } from 'vitest';
import {
  agentFillRgb,
  buildFarInstances,
  FAR_INSTANCE_STRIDE,
  KIND_BW_RGB,
  kindChroma,
  kindFillRgb,
  waveDisplace,
} from './render.ts';
import { EXTRA_CAP, EXTRA_FLOOR } from './energy.ts';
import { WAVE_DISP_PX } from './geom.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

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

  it('caps a full-scale ring at three-quarters of WAVE_DISP_PX', () => {
    const peak = Math.abs(waveDisplace(2, 2, 1, 1));
    expect(peak).toBeCloseTo(WAVE_DISP_PX * 0.75 * 1.7, 5);
  });
});

describe('kind colors', () => {
  it('paints a full tank as red, blue, and yellow', () => {
    expect(kindFillRgb('dup', EXTRA_CAP)).toEqual([255, 0, 0]);
    expect(kindFillRgb('con', EXTRA_CAP)).toEqual([0, 0, 255]);
    expect(kindFillRgb('era', EXTRA_CAP)).toEqual([255, 255, 0]);
  });

  it('maps energy to saturation, gray at the floor', () => {
    const dead = kindFillRgb('con', EXTRA_FLOOR);
    expect(dead[0]).toBe(dead[1]);
    expect(dead[1]).toBe(dead[2]);
    expect(kindChroma('dup', EXTRA_FLOOR)).toBe(0);
    expect(kindChroma('dup', 0)).toBeGreaterThan(kindChroma('dup', EXTRA_FLOOR));
    expect(kindChroma('dup', EXTRA_CAP)).toBeGreaterThan(kindChroma('dup', 0));
    expect(kindChroma('con', EXTRA_CAP)).toBeGreaterThan(kindChroma('con', 0));
    expect(kindChroma('era', EXTRA_CAP)).toBeGreaterThan(kindChroma('era', 0));
  });

  it('grayscale fills match Canvas2D, not kind hues', () => {
    expect(agentFillRgb('dup', EXTRA_CAP, false)).toEqual(KIND_BW_RGB.dup);
    expect(agentFillRgb('con', EXTRA_CAP, false)).toEqual(KIND_BW_RGB.con);
    expect(agentFillRgb('era', EXTRA_CAP, false)).toEqual(KIND_BW_RGB.era);
    expect(agentFillRgb('dup', EXTRA_CAP, true)).toEqual([255, 0, 0]);
  });
});

describe('buildFarInstances', () => {
  it('packs kind hues when color is on and grayscale fills when off', () => {
    const sim = new Sim(800, 600);
    const params = defaultParams();
    const dup = sim.spawn('dup', 100, 100, 0, params, true)!;
    dup.extra = EXTRA_CAP;
    sim.isFarTier = () => true;
    const out = new Float32Array(FAR_INSTANCE_STRIDE);
    buildFarInstances(sim, out, true);
    expect([out[5], out[6], out[7]]).toEqual([1, 0, 0]);
    buildFarInstances(sim, out, false);
    expect(out[5]).toBeCloseTo(KIND_BW_RGB.dup[0] / 255);
    expect(out[6]).toBeCloseTo(KIND_BW_RGB.dup[1] / 255);
    expect(out[7]).toBeCloseTo(KIND_BW_RGB.dup[2] / 255);
  });
});
