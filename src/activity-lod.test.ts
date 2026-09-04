import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

describe('activity LOD (dual-rate islands)', () => {
  function quiet() {
    const params = defaultParams();
    params.snapRadius = 0;
    params.snapWell = 0;
    params.faceAttract = 0;
    params.stepSpeed = 0;
    params.swimNoise = 0;
    params.spawnInterval = 0;
    params.rewriteDuration = 0;
    params.wireShrink = 0;
    params.portStiff = 0;
    params.declutter = 0;
    params.uncross = 0;
    params.wireClear = 0;
    params.flockAlign = 0;
    params.flockSep = 0;
    params.deposit = 0;
    params.nearBudget = 800;
    return params;
  }

  function agedChain(params = quiet()) {
    const sim = new Sim(480, 240);
    const nodes = [80, 170, 260, 350].map((x) => sim.spawn('con', x, 120, 0, params, true)!);
    for (let i = 0; i < 3; i++) sim.wire(nodes[i].id, 'r', nodes[i + 1].id, 'l', params);
    for (let i = 0; i < 8; i++) sim.step(1 / 60, params);
    for (const w of sim.graph.wires.values()) {
      w.born = sim.time - params.wireSpanAge - 1;
      w.lastLen = w.rest;
    }
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      a.x = 80 + i * 90;
      a.y = 120;
      a.vx = 0;
      a.vy = 0;
      a.omega = 0;
    }
    for (let i = 0; i < 36; i++) {
      for (const w of sim.graph.wires.values()) w.lastLen = w.rest;
      sim.step(1 / 60, params);
    }
    return { sim, nodes, params };
  }

  it('a taut aged chain sleeps even with no view', () => {
    const { sim, nodes } = agedChain();
    for (const w of sim.graph.wires.values()) expect(w.ropePath).toBe('span');
    expect(sim.isPhysicsDetailed(nodes[1].id)).toBe(false);
    expect(sim.isPhysicsDetailed(nodes[2].id)).toBe(false);
  });

  it('nearBudget 0 keeps an aged taut chain on SAT', () => {
    const params = quiet();
    params.nearBudget = 0;
    const sim = new Sim(480, 240);
    const nodes = [80, 170, 260, 350].map((x) => sim.spawn('con', x, 120, 0, params, true)!);
    for (let i = 0; i < 3; i++) sim.wire(nodes[i].id, 'r', nodes[i + 1].id, 'l', params);
    for (let i = 0; i < 8; i++) sim.step(1 / 60, params);
    for (const w of sim.graph.wires.values()) {
      w.born = sim.time - params.wireSpanAge - 1;
      w.lastLen = w.rest;
    }
    sim.step(1 / 60, params);
    expect(sim.isPhysicsDetailed(nodes[1].id)).toBe(true);
  });

  it('loners stay NEAR', () => {
    const params = quiet();
    const sim = new Sim(240, 160);
    const a = sim.spawn('era', 40, 80, 0, params, true)!;
    sim.step(1 / 60, params);
    expect(sim.isPhysicsDetailed(a.id)).toBe(true);
  });

  it('a grab wakes a sleeping net', () => {
    const { sim, nodes, params } = agedChain();
    expect(sim.isPhysicsDetailed(nodes[1].id)).toBe(false);
    sim.grabbed = { id: nodes[0].id, x: nodes[0].x, y: nodes[0].y };
    sim.step(1 / 60, params);
    expect(sim.isPhysicsDetailed(nodes[0].id)).toBe(true);
    expect(sim.isPhysicsDetailed(nodes[1].id)).toBe(true);
  });

  it('a young latch stays NEAR', () => {
    const params = quiet();
    const sim = new Sim(400, 240);
    const a = sim.spawn('era', 120, 120, 0, params, true)!;
    const b = sim.spawn('era', 200, 120, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    sim.step(1 / 60, params);
    expect(sim.isPhysicsDetailed(a.id)).toBe(true);
    expect(sim.isPhysicsDetailed(b.id)).toBe(true);
  });
});
