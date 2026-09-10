import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';
import { describePlant, netUrlFromQuery, plantNetBytes } from './net-load.ts';
import { fixturePath, listFixtures } from './pond/net-file.ts';
import { captureNets } from './pond/capture.ts';
import { encodeNetAs, currentLayout } from './pond/net-blob.ts';
import { CHEM_SEGMENTS, PLASTIC_BASE } from './chem-layout.ts';

/*
 * The page's door for a stored net, exercised without a page: the bytes a
 * `fetch` or a dropped file would hand over, into a `Sim`, and the line the
 * panel would print.
 */

describe('planting bytes', () => {
  const names = listFixtures();

  it('puts a fixture file into a dish, whole, on its own founder line', () => {
    expect(names.length).toBeGreaterThan(0);
    const bytes = readFileSync(fixturePath(names[0]));
    const params = defaultParams();
    params.soupCount = 0;
    params.spawnInterval = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    const p = plantNetBytes(sim, params, bytes, sim.w * 0.5, sim.h * 0.5);
    expect(p.ids.length).toBe(p.header.bodies);
    expect(sim.agents.size).toBe(p.header.bodies);
    expect(sim.census().lines).toBe(1);
    expect(describePlant(names[0], p)).toContain(`${p.header.bodies} bodies`);
    for (let i = 0; i < 30; i++) sim.step(1 / 60, params);
  });

  it('accepts a Buffer at any offset, as a fetched or read file arrives', () => {
    const bytes = readFileSync(fixturePath(names[0]));
    const shifted = new Uint8Array(bytes.byteLength + 3);
    shifted.set(bytes, 3);
    const params = defaultParams();
    params.soupCount = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    expect(plantNetBytes(sim, params, shifted.subarray(3), sim.w * 0.5, sim.h * 0.5).ids.length).toBeGreaterThan(0);
  });

  it('says why when the blob cannot be read here', () => {
    const params = defaultParams();
    params.soupCount = 40;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    for (let i = 0; i < 240; i++) sim.step(1 / 60, params);
    const net = captureNets(sim)[0].data;
    const L = currentLayout();
    const foreign = encodeNetAs(net, { ...L, state: L.state + 1, critic: L.critic + 1 }, CHEM_SEGMENTS, PLASTIC_BASE);
    expect(() => plantNetBytes(sim, params, foreign, 0, 0)).toThrow(/recurrent state/);
  });

  it('reports a net that would not fit rather than planting part of it', () => {
    const bytes = readFileSync(fixturePath(names[0]));
    const params = defaultParams();
    params.soupCount = 0;
    params.maxAgents = 3;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    const p = plantNetBytes(sim, params, bytes, sim.w * 0.5, sim.h * 0.5);
    expect(p.ids).toEqual([]);
    expect(sim.agents.size).toBe(0);
    expect(describePlant('x', p)).toMatch(/would not fit/);
  });
});

describe('?net=', () => {
  it('names a file under nets/ unless given a path', () => {
    expect(netUrlFromQuery('')).toBeNull();
    expect(netUrlFromQuery('?soup=0')).toBeNull();
    expect(netUrlFromQuery('?net=mixed-308')).toBe('nets/mixed-308.petrinet');
    expect(netUrlFromQuery('?net=mixed-308.petrinet')).toBe('mixed-308.petrinet');
    expect(netUrlFromQuery('?net=/somewhere/else.petrinet')).toBe('/somewhere/else.petrinet');
    expect(netUrlFromQuery('?soup=0&net=deep-87')).toBe('nets/deep-87.petrinet');
  });
});
