import { describe, expect, it } from 'vitest';
import {
  ropeIsLive,
  ropePathOf,
  ROPE_TAUT_HYSTERESIS,
  type RopePath,
  type Wire,
} from './graph.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

function fakeWire(over: Partial<Wire> = {}): Wire {
  return {
    id: 1,
    a: { id: 1, slot: 'p' },
    b: { id: 2, slot: 'p' },
    collapse: 0,
    pitchFloor: 20,
    latchLen: 40,
    lastLen: 40,
    rest: 40,
    ropeLen: 40,
    shape: [],
    born: 0,
    nodes: [],
    ropePath: 'full',
    ...over,
  };
}

function pathAt(age: number, lastLen: number, ropePath: RopePath = 'full', params = defaultParams()): RopePath {
  return ropePathOf(fakeWire({ born: 0, lastLen, rest: 40, ropePath }), age, params);
}

describe('ropePathOf', () => {
  it('keeps a young taut wire on the full rope', () => {
    expect(pathAt(0.5, 40)).toBe('full');
    expect(pathAt(1.9, 41)).toBe('full');
  });

  it('drops shape once taut past wireShapeAge', () => {
    expect(pathAt(2, 40)).toBe('no-shape');
    expect(pathAt(9.9, 42)).toBe('no-shape');
  });

  it('goes span-only once taut past wireSpanAge', () => {
    expect(pathAt(10, 40)).toBe('span');
    expect(pathAt(40, 40)).toBe('span');
  });

  it('keeps a slack leftover on the full rope no matter how old', () => {
    expect(pathAt(40, 80)).toBe('full');
    expect(pathAt(40, 80, 'span')).toBe('full');
  });

  it('treats 0 duration as never', () => {
    const params = defaultParams();
    params.wireShapeAge = 0;
    params.wireSpanAge = 0;
    expect(pathAt(100, 40, 'full', params)).toBe('full');
  });

  it('lets span-only win when both ages have elapsed', () => {
    const params = defaultParams();
    params.wireShapeAge = 10;
    params.wireSpanAge = 2;
    expect(pathAt(3, 40, 'full', params)).toBe('span');
  });

  it('hysteresis keeps a coarsened wire coarsened until it is clearly slack', () => {
    const taut = defaultParams().wireTaut;
    const band = taut + ROPE_TAUT_HYSTERESIS * 0.5;
    expect(pathAt(12, 40 * band, 'span')).toBe('span');
    expect(pathAt(12, 40 * (taut + ROPE_TAUT_HYSTERESIS + 0.01), 'span')).toBe('full');
  });

  it('ropeIsLive is false for span-only and for view-FAR', () => {
    expect(ropeIsLive(fakeWire({ ropePath: 'full' }))).toBe(true);
    expect(ropeIsLive(fakeWire({ ropePath: 'no-shape' }))).toBe(true);
    expect(ropeIsLive(fakeWire({ ropePath: 'span' }))).toBe(false);
    expect(ropeIsLive(fakeWire({ ropePath: 'full' }), () => false)).toBe(false);
  });
});

describe('aged taut wires in the sim', () => {
  function quiet() {
    const params = defaultParams();
    params.snapRadius = 0;
    params.stepSpeed = 0;
    params.spawnInterval = 0;
    params.rewriteDuration = 20;
    params.wireShrink = 0.2;
    params.gravity = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.deposit = 0;
    return params;
  }

  function wiredPair(params: ReturnType<typeof quiet>) {
    const sim = new Sim(400, 240);
    const a = sim.spawn('era', 120, 120, 0, params, true)!;
    const b = sim.spawn('era', 200, 120, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const wire = [...sim.graph.wires.values()][0];
    for (let i = 0; i < 60; i++) sim.step(1 / 60, params);
    return { sim, a, b, wire, params };
  }

  it('drops shape, then the rope, on a taut latch', () => {
    const { sim, wire, params } = wiredPair(quiet());
    expect(wire.ropePath).toBe('full');
    expect(sim.wireSimulatesRope(wire)).toBe(true);

    wire.born = sim.time - params.wireShapeAge - 0.2;
    wire.lastLen = wire.rest;
    sim.step(1 / 60, params);
    expect(wire.ropePath).toBe('no-shape');
    expect(wire.shape.length).toBe(0);
    expect(sim.wireSimulatesRope(wire)).toBe(true);

    wire.born = sim.time - params.wireSpanAge - 0.2;
    wire.lastLen = wire.rest;
    sim.step(1 / 60, params);
    expect(wire.ropePath).toBe('span');
    expect(sim.wireSimulatesRope(wire)).toBe(false);
    expect(wire.nodes.length).toBeGreaterThan(0);
  });

  it('resamples the rope when a span-only wire goes slack', () => {
    const { sim, b, wire, params } = wiredPair(quiet());
    wire.born = sim.time - params.wireSpanAge - 0.5;
    wire.lastLen = wire.rest;
    sim.step(1 / 60, params);
    expect(wire.ropePath).toBe('span');

    b.x += 90;
    sim.step(1 / 60, params);
    sim.step(1 / 60, params);
    expect(wire.ropePath).toBe('full');
    expect(sim.wireSimulatesRope(wire)).toBe(true);
    expect(wire.nodes.length).toBeGreaterThan(0);
  });

  it('ignores age when both duration knobs are 0', () => {
    const params = quiet();
    params.wireShapeAge = 0;
    params.wireSpanAge = 0;
    const { sim, wire } = wiredPair(params);
    wire.born = sim.time - 100;
    wire.lastLen = wire.rest;
    sim.step(1 / 60, params);
    expect(wire.ropePath).toBe('full');
    expect(sim.wireSimulatesRope(wire)).toBe(true);
  });
});
