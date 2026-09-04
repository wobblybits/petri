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

  it('sizes a cell against the sensor baseline, not the wire', () => {
    /*
     * The cell size is set by steering, not by geometry. Two sensors sit
     * `sensorDist` apart and are compared against a dead zone of 5% of the
     * signal — a zone that is earning its keep, since it rejects the sampling
     * asymmetry a body reads off its own trail. So a real gradient is only
     * visible when the sensors straddle something near a whole cell.
     *
     * Measured at 160-unit cells, where they straddled a seventh of one, a
     * genuine gradient read 0.043 against a dead zone of 0.237 and nothing
     * turned at all.
     */
    const p = defaultParams();
    const span = 2 * p.sensorDist * Math.sin(p.sensorAngle);
    expect(span / FIELD_CELL, `sensors span ${(span / FIELD_CELL).toFixed(2)} cells`)
      .toBeGreaterThan(0.8);
  });

  it('holds a pond worth of bodies at the spacing one settles to', () => {
    /*
     * Extent is the expensive axis and fidelity is the cheap one — cost tracks
     * the number of cells, so a finer cell is free and a wider world is not.
     * The world is therefore sized to what it needs to hold rather than to any
     * measurement of how far a pond spreads: ponds fill whatever box they are
     * spawned into, so that measurement would only have been reading back the
     * spawn box.
     *
     * Declutter holds unwired bodies about two wire lengths apart.
     */
    const spacing = defaultParams().wireMinRest * 1.3;
    const capacity = (FIELD_EXTENT / spacing) ** 2;
    expect(capacity, `${capacity.toFixed(0)} bodies fit`).toBeGreaterThan(5000);
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

  /*
   * Confinement no longer runs inside step() — it's a loose failsafe driven
   * by its own background thread pool on its own cadence (Sim.runConfineLoop),
   * not a per-frame-exact force, and a threaded dispatch can't fit inside
   * step()'s synchronous frame. These tests want confinement's exact
   * behaviour on a specific frame, so they call the underlying synchronous
   * confineOnce directly instead of relying on it happening automatically.
   */
  function stepWithConfine(sim: Sim, params: Params0, frames: number): void {
    for (let f = 0; f < frames; f++) {
      sim.step(1 / 60, params, VIEW);
      sim.confineOnce(1 / 60, params.edgePull);
    }
  }

  it('leaves a body inside the bound alone', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const params = settle();
    const sim = new Sim(1200, 800);
    // A cluster at the origin so home lands there, plus one body well inside.
    for (let i = 0; i < 8; i++) loner(sim, params, 600 + i * 60, 400);
    const id = loner(sim, params, 600 + FIELD_HALF * 0.5, 400);
    stepWithConfine(sim, params, 60);
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
    stepWithConfine(sim, params, 120);
    const a = sim.agents.get(id)!;
    expect(a.x, `moved from ${startX} to ${a.x}`).toBeLessThan(startX);
    expect(a.vx, 'should be heading home, i.e. negative x').toBeLessThan(0);
  });

  it('pulls harder the further out a body is', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const speeds: number[] = [];
    for (const over of [1, 3]) {
      const params = settle();
      const sim = new Sim(1200, 800);
      // A heavy cluster, so `home` stays put. One stray body is enough to drag
      // the centre of mass outward, and the bound follows it — which is how the
      // first draft of this test placed a body 3,072 units out and had it land
      // inside a 2,560 bound, feeling no pull at all.
      for (let i = 0; i < 40; i++) {
        loner(sim, params, 600 + (i % 8) * 60, 400 + ((i / 8) | 0) * 60);
      }
      const id = loner(sim, params, 600 + FIELD_HALF * (1 + over), 400);
      stepWithConfine(sim, params, 30);
      speeds.push(-sim.agents.get(id)!.vx);
    }
    expect(speeds[0], `near-edge pull ${speeds[0]}`).toBeGreaterThan(0);
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
    stepWithConfine(sim, params, 120);
    const a = sim.agents.get(id)!;
    expect(Math.abs(a.x - startX), 'should have stayed put').toBeLessThan(50);
  });
});
