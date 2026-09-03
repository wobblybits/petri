import { describe, expect, it } from 'vitest';
import { EnergyGrid } from './energy.ts';
import { FIELD_CELL, FIELD_CELLS, FIELD_EXTENT, FIELD_HALF } from './fields.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';
import { nativeSolver } from './native/solver.ts';

/**
 * The world bound: one square, shared by the scent field and the energy grid.
 *
 * Outside it there is no world. Nothing to smell, nothing to eat, and a weak
 * pull back toward home — so drifting out is survivable if you turn around and
 * fatal if you do not. That is the pressure that makes a finite field honest:
 * the alternative is agents wandering into a region with no gradient to bring
 * them back, which is what happens today with an unbounded grid and no pull.
 *
 * The pull is deliberately soft. A wall would be cheaper and would also stop a
 * net dead at the edge, which reads as a bug rather than a place.
 */

const VIEW = { x: 600, y: 400, zoom: 0.5, viewW: 1200, viewH: 800 };

describe('world bound geometry', () => {
  it('sizes the grid to fit a WebGPU storage binding', () => {
    // 4 channels x 4 bytes. The spec only guarantees 128 MiB per binding, and
    // a field buffer is one binding — this is what caps the grid, not memory.
    const bytesPerBuffer = FIELD_CELLS * FIELD_CELLS * 4 * 4;
    expect(bytesPerBuffer).toBeLessThanOrEqual(128 * 1024 * 1024);
    expect(FIELD_EXTENT).toBe(FIELD_CELLS * FIELD_CELL);
    expect(FIELD_HALF * 2).toBe(FIELD_EXTENT);
  });

  it('gives a cell one wire length, so a cell holds about one link', () => {
    expect(FIELD_CELL).toBe(defaultParams().wireMinRest);
  });

  it('covers a settled pond with room to drift', () => {
    // Measured: a settled pond is ~21,500 units across with a ~14,500 radius.
    expect(FIELD_EXTENT).toBeGreaterThan(21_500 * 1.5);
  });
});

describe('energy grid bound', () => {
  it('is barren outside, not merely empty', () => {
    const g = new EnergyGrid(48, 0.5);
    g.setBounds(0, 0, 1000);
    expect(g.getAt(0, 0), 'inside should hold ambient').toBe(0.5);
    expect(g.getAt(5000, 0), 'outside should hold nothing at all').toBe(0);
  });

  it('swallows a deposit made outside', () => {
    const g = new EnergyGrid(48, 0);
    g.setBounds(0, 0, 1000);
    g.addAt(5000, 0, 10);
    expect(g.getAt(5000, 0)).toBe(0);
    expect(g.storedTotal(), 'nothing should have been stored').toBe(0);
  });

  it('keeps what it stored when the bound moves away, and gives it back', () => {
    // The bound tracks home, so a cell can fall outside and later fall back in.
    // Dropping its contents would quietly destroy energy.
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

describe('edge pull', () => {
  function loner(sim: Sim, params: Params0, x: number, y: number): number {
    const a = sim.spawn('dup', x, y, 0, params, true)!;
    return a.id;
  }
  type Params0 = ReturnType<typeof defaultParams>;

  function settle(): Params0 {
    const p = defaultParams();
    p.maxAgents = 100;
    p.spawnInterval = 0;
    p.gravity = 0;
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
    // A cluster at the origin so home lands there, plus one body well inside.
    for (let i = 0; i < 8; i++) loner(sim, params, 600 + i * 60, 400);
    const id = loner(sim, params, 600 + FIELD_HALF * 0.5, 400);
    for (let f = 0; f < 60; f++) sim.step(1 / 60, params, VIEW);
    const a = sim.agents.get(id)!;
    // No pull means no systematic drift back toward the cluster.
    expect(Math.abs(a.vx), `vx ${a.vx}`).toBeLessThan(5);
  });

  it('walks a body outside the bound back toward home', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = settle();
    const sim = new Sim(1200, 800);
    for (let i = 0; i < 8; i++) loner(sim, params, 600 + i * 60, 400);
    const startX = 600 + FIELD_HALF * 2;
    const id = loner(sim, params, startX, 400);
    for (let f = 0; f < 120; f++) sim.step(1 / 60, params, VIEW);
    const a = sim.agents.get(id)!;
    expect(a.x, `moved from ${startX} to ${a.x}`).toBeLessThan(startX);
    expect(a.vx, 'should be heading home, i.e. negative x').toBeLessThan(0);
  });

  it('pulls harder the further out a body is', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const speeds: number[] = [];
    for (const over of [0.2, 1.0]) {
      const params = settle();
      const sim = new Sim(1200, 800);
      for (let i = 0; i < 8; i++) loner(sim, params, 600 + i * 60, 400);
      const id = loner(sim, params, 600 + FIELD_HALF * (1 + over), 400);
      for (let f = 0; f < 30; f++) sim.step(1 / 60, params, VIEW);
      speeds.push(-sim.agents.get(id)!.vx);
    }
    expect(speeds[0]).toBeGreaterThan(0);
    expect(speeds[1], `${speeds[1]} should exceed ${speeds[0]}`).toBeGreaterThan(speeds[0]);
  });

  it('does nothing at all when edgePull is 0', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = settle();
    params.edgePull = 0;
    const sim = new Sim(1200, 800);
    for (let i = 0; i < 8; i++) loner(sim, params, 600 + i * 60, 400);
    const startX = 600 + FIELD_HALF * 2;
    const id = loner(sim, params, startX, 400);
    for (let f = 0; f < 120; f++) sim.step(1 / 60, params, VIEW);
    const a = sim.agents.get(id)!;
    expect(Math.abs(a.x - startX), 'should have stayed put').toBeLessThan(50);
  });
});
