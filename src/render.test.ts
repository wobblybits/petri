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
import { ERA_RADIUS, boundRadius, createAgent, triangleLocal } from './agents.ts';
import { EXTRA_CAP, EXTRA_FLOOR } from './energy.ts';
import shader from './gpu/agents.wgsl?raw';
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

/**
 * The FAR tier draws the population on the GPU and the NEAR tier draws it on
 * a canvas, and the two have to be the same pond seen from further away. A
 * WGSL literal has nothing keeping it honest against the geometry in
 * `agents.ts`, so this is that thing: the glyph constants were eyeballed once
 * and the whole population changed size across the LOD line.
 */
describe('the GPU glyph matches the canvas glyph', () => {
  const vec2 = (name: string): { x: number; y: number } => {
    const m = new RegExp(`const ${name} = vec2f\\(\\s*(-?[\\d.]+)\\s*,\\s*(-?[\\d.]+)\\s*\\)`).exec(shader);
    expect(m, `${name} in agents.wgsl`).not.toBeNull();
    return { x: Number(m![1]), y: Number(m![2]) };
  };
  const f32 = (name: string): number => {
    const m = new RegExp(`const ${name}(?::\\s*f32)? = (-?[\\d.]+)`).exec(shader);
    expect(m, `${name} in agents.wgsl`).not.toBeNull();
    return Number(m![1]);
  };

  it('puts the Con/Dup triangle where triangleLocal does', () => {
    const params = defaultParams();
    const con = createAgent(1, 'con', 0, 0, 0, params);
    // The shader's UV is the quad, and the quad is boundRadius across.
    const uv = boundRadius(con);
    const [tip, backA, backB] = triangleLocal(con.scale);
    expect(vec2('TRI_TIP').x * uv).toBeCloseTo(tip.x, 3);
    expect(vec2('TRI_TIP').y * uv).toBeCloseTo(tip.y, 3);
    // The shader names the backs by which side of +X they are on, which is
    // the opposite of triangleLocal's order; the glyph is symmetric, so
    // compare the shared x and the half-base.
    expect(vec2('TRI_BACK_L').x * uv).toBeCloseTo(backA.x, 3);
    expect(vec2('TRI_BACK_R').x * uv).toBeCloseTo(backB.x, 3);
    expect(Math.abs(vec2('TRI_BACK_L').y) * uv).toBeCloseTo(Math.abs(backA.y), 3);
    expect(Math.abs(vec2('TRI_BACK_R').y) * uv).toBeCloseTo(Math.abs(backB.y), 3);
  });

  it('draws an Era the size drawEra draws it', () => {
    const params = defaultParams();
    const era = createAgent(1, 'era', 0, 0, 0, params);
    expect(f32('ERA_UV') * boundRadius(era)).toBeCloseTo(ERA_RADIUS * era.scale, 3);
  });
});

describe('buildFarInstances', () => {
  it('reads fullness against the body\'s own caps, the way drawAgent does', () => {
    const sim = new Sim(800, 600);
    const params = defaultParams();
    const con = sim.spawn('con', 100, 100, 0, params, true)!;
    // A body whose tank is twice the default and whose debt runs deeper —
    // both inside TRAIT_RANGE, both heritable.
    con.extra = 0.5;
    con.debtCap = -2.5;
    con.energyCap = EXTRA_CAP * 2;
    sim.isFarTier = () => true;
    const out = new Float32Array(FAR_INSTANCE_STRIDE);
    buildFarInstances(sim, out, true);
    const own = kindFillRgb('con', con.extra, con.debtCap, con.energyCap);
    const globals = kindFillRgb('con', con.extra);
    // The point of the test: for this body the two readings differ, so a
    // packer that drops the caps cannot pass by accident.
    expect(own).not.toEqual(globals);
    // Through an f32, so per channel rather than deep equality.
    for (let k = 0; k < 3; k++) expect(out[5 + k] * 255).toBeCloseTo(own[k], 3);
  });

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
