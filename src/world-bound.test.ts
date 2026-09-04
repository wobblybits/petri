import { describe, expect, it } from 'vitest';
import { EnergyGrid } from './energy.ts';
import { FIELD_CELL, FIELD_CELLS, FIELD_EXTENT, FIELD_HALF, Fields, worldBoundRadius } from './fields.ts';
import { defaultParams } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';
import { nativeSolver } from './native/solver.ts';

/**
 * The world bound: a disk inscribed in the square field, shared by the scent
 * mask, the energy grid, and a hard rim the integrator collides against.
 *
 * Outside it there is no world. Nothing to smell, nothing to eat, and a body
 * that swims into the rim slides along it. That is the pressure that makes a
 * finite field honest: scent locked to 0 beyond the disk decays toward the
 * wall, so agents are not attracted outward, and the wall itself is what
 * stops a net leaving the map.
 */

const VIEW = { x: 600, y: 400, zoom: 0.5, viewW: 1200, viewH: 800 };

describe('world bound geometry', () => {
  it('sizes the grid to fit a WebGPU storage binding', () => {
    const bytesPerBuffer = FIELD_CELLS * FIELD_CELLS * 4 * 4;
    expect(bytesPerBuffer).toBeLessThanOrEqual(128 * 1024 * 1024);
    expect(FIELD_EXTENT).toBe(FIELD_CELLS * FIELD_CELL);
    expect(FIELD_HALF * 2).toBe(FIELD_EXTENT);
  });

  it('sizes a cell against the sensor baseline, not the wire', () => {
    const p = defaultParams();
    const span = 2 * p.sensorDist * Math.sin(p.sensorAngle);
    expect(span / FIELD_CELL, `sensors span ${(span / FIELD_CELL).toFixed(2)} cells`)
      .toBeGreaterThan(0.8);
  });

  it('holds a pond worth of bodies at the spacing one settles to', () => {
    const spacing = defaultParams().wireMinRest * 1.3;
    const capacity = (FIELD_EXTENT / spacing) ** 2;
    expect(capacity, `${capacity.toFixed(0)} bodies fit`).toBeGreaterThan(5000);
  });

  it('inscribes the live disk inside the snapped field window', () => {
    const f = new Fields();
    const cx = 601;
    const cy = 407;
    f.cover(cx, cy);
    const r = worldBoundRadius(cx, cy, f.originX, f.originY);
    const left = cx - f.originX;
    const right = f.originX + FIELD_EXTENT - cx;
    const top = cy - f.originY;
    const bottom = f.originY + FIELD_EXTENT - cy;
    const inscribed = Math.min(left, right, top, bottom);
    expect(r).toBeLessThan(inscribed);
    expect(r).toBeGreaterThan(inscribed - FIELD_CELL - 1e-6);
    expect(r + FIELD_CELL).toBeLessThanOrEqual(inscribed + 1e-6);
  });
});

describe('energy grid bound', () => {
  it('is barren outside, not merely empty', () => {
    const g = new EnergyGrid(48, 0.5);
    g.setBounds(0, 0, 1000);
    expect(g.getAt(0, 0), 'inside should hold ambient').toBe(0.5);
    expect(g.getAt(5000, 0), 'outside should hold nothing at all').toBe(0);
  });

  it('is a disk, so a square corner is outside', () => {
    const g = new EnergyGrid(48, 0.5);
    g.setBounds(0, 0, 1000);
    expect(g.getAt(900, 900), 'square corner is past the circle').toBe(0);
    expect(g.inBounds(0, 999)).toBe(true);
    expect(g.inBounds(800, 800)).toBe(false);
  });

  it('swallows a deposit made outside', () => {
    const g = new EnergyGrid(48, 0);
    g.setBounds(0, 0, 1000);
    g.addAt(5000, 0, 10);
    expect(g.getAt(5000, 0)).toBe(0);
    expect(g.storedTotal(), 'nothing should have been stored').toBe(0);
  });

  it('keeps what it stored when the bound moves away, and gives it back', () => {
    const g = new EnergyGrid(48, 0);
    g.setBounds(0, 0, 1000);
    g.addAt(500, 0, 7);
    expect(g.getAt(500, 0)).toBe(7);
    g.setBounds(-3000, 0, 1000);
    expect(g.getAt(500, 0), 'out of bounds now').toBe(0);
    g.setBounds(0, 0, 1000);
    expect(g.getAt(500, 0), 'came back intact').toBe(7);
  });

  it('is unbounded until a bound is set', () => {
    const g = new EnergyGrid(48, 0.25);
    expect(g.getAt(1e9, -1e9)).toBe(0.25);
  });
});

describe('scent Dirichlet disk', () => {
  it('drops deposits whose cells sit outside the live radius', () => {
    const f = new Fields(64);
    f.setWorldBound(5120, 5120, 400);
    f.deposit(0, 80, 80, 12);
    expect(f.sample(0, 80, 80)).toBe(0);
    f.deposit(0, 5120, 5120, 12);
    expect(f.sample(0, 5120, 5120)).toBeGreaterThan(0);
  });

  it('zeros cells past the rim so a gradient points inward', () => {
    const f = new Fields(64);
    f.setWorldBound(5120, 5120, 800);
    f.deposit(0, 5120, 5120, 40);
    for (let k = 0; k < 8; k++) f.diffuse(0.9);
    const cell = f.cellSize;
    const i = Math.floor((5120 - f.originX) / cell);
    const j = Math.floor((5120 - f.originY) / cell);
    expect(f.cellOut(0, 0)).toBe(true);
    const at = (ii: number, jj: number) => f.data[(jj * f.cols + ii) * 4];
    expect(at(0, 0)).toBe(0);
    expect(at(i, j)).toBeGreaterThan(0);
    const mid = f.sample(0, 5120 + 200, 5120);
    const rim = f.sample(0, 5120 + 700, 5120);
    expect(mid).toBeGreaterThan(rim);
  });
});

describe('hard rim', () => {
  type Params0 = ReturnType<typeof defaultParams>;

  function loner(sim: Sim, params: Params0, x: number, y: number): number {
    const a = sim.spawn('dup', x, y, 0, params, true)!;
    return a.id;
  }

  function settle(): Params0 {
    const p = defaultParams();
    p.maxAgents = 100;
    p.spawnInterval = 0;
    p.upkeep = 0;
    p.stepSpeed = 0;
    p.swimNoise = 0;
    p.ambientEnergy = 0;
    return p;
  }

  it('leaves a body inside the bound alone', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = settle();
    const sim = new Sim(1200, 800);
    sim.pinWorld(600, 400);
    for (let i = 0; i < 8; i++) loner(sim, params, 600 + i * 60, 400);
    const id = loner(sim, params, 600 + sim.worldR * 0.4, 400);
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params, VIEW);
    const a = sim.agents.get(id)!;
    expect(Math.abs(a.vx), `vx ${a.vx}`).toBeLessThan(5);
  });

  it('projects a body outside the bound back onto the rim', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = settle();
    const sim = new Sim(1200, 800);
    sim.pinWorld(600, 400);
    for (let i = 0; i < 8; i++) loner(sim, params, 600 + i * 60, 400);
    const startX = 600 + sim.worldR * 2;
    const id = loner(sim, params, startX, 400);
    sim.step(1 / 60, params, VIEW);
    const a = sim.agents.get(id)!;
    const dist = Math.hypot(a.x - 600, a.y - 400);
    expect(dist, `still at ${a.x}`).toBeLessThan(sim.worldR + 1);
    expect(a.x).toBeLessThan(startX);
  });

  it('kills outward radial velocity at the rim', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = settle();
    params.stepSpeed = 80;
    const sim = new Sim(1200, 800);
    sim.pinWorld(600, 400);
    for (let i = 0; i < 20; i++) loner(sim, params, 600 + (i % 5) * 40, 400 + ((i / 5) | 0) * 40);
    const r = sim.worldR - 12;
    const id = loner(sim, params, 600 + r, 400);
    const body = sim.agents.get(id)!;
    body.vx = 200;
    body.vy = 0;
    for (let f = 0; f < 30; f++) sim.step(1 / 60, params, VIEW);
    const a = sim.agents.get(id)!;
    const dist = Math.hypot(a.x - 600, a.y - 400);
    expect(dist).toBeLessThan(sim.worldR + 1);
    expect(a.vx, 'should not keep swimming out').toBeLessThan(20);
  });
});

describe('soup placement', () => {
  it('lands inside the live disk', () => {
    const params = defaultParams();
    params.soupCount = 80;
    params.maxAgents = 80;
    const sim = new Sim(1200, 800);
    loadPreset(sim, 'soup', params);
    expect(sim.worldR).toBeGreaterThan(1000);
    let outside = 0;
    let n = 0;
    for (const a of sim.agents.values()) {
      n++;
      const d = Math.hypot(a.x - sim.worldX, a.y - sim.worldY);
      if (d > sim.worldR) outside++;
    }
    expect(n).toBe(80);
    expect(outside).toBe(0);
  });
});
