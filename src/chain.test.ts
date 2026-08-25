import { describe, expect, it } from 'vitest';
import { createAgent, stemWorld } from './agents.ts';
import { chordDeviation, desiredLinks, solveChain, taperStiffness, type ChainNode } from './chain.ts';
import { defaultParams } from './params.ts';

describe('chain XPBD', () => {
  it('keeps a dense chain for long latches', () => {
    expect(desiredLinks(160)).toBeGreaterThanOrEqual(12);
    expect(desiredLinks(40)).toBeGreaterThanOrEqual(4);
  });

  it('is stiffest at the hull and softer past the original port', () => {
    const k = 55;
    const port = 8;
    const base = taperStiffness(0, port, k);
    const atPort = taperStiffness(port, port, k);
    const far = taperStiffness(port + 40, port, k);
    expect(base).toBeGreaterThan(atPort);
    expect(atPort).toBeGreaterThan(far);
  });
  it('straightens a zigzag toward the port-to-port chord', () => {
    const params = defaultParams();
    params.springK = 80;
    params.springDamp = 4;
    const a = createAgent(1, 'era', 80, 80, 0, params);
    const b = createAgent(2, 'era', 220, 80, Math.PI, params);
    a.locked = true;
    b.locked = true;
    const nodes: ChainNode[] = [
      { x: 120, y: 150, vx: 0, vy: 0, prevX: 120, prevY: 150, integVx: 0, integVy: 0 },
      { x: 170, y: 20, vx: 0, vy: 0, prevX: 170, prevY: 20, integVx: 0, integVy: 0 },
    ];
    expect(Math.abs(nodes[0].y - 80) + Math.abs(nodes[1].y - 80)).toBeGreaterThan(80);
    for (let i = 0; i < 24; i++) {
      solveChain(a, 'p', b, 'p', nodes, 140, params, 1 / 60, 400, 200);
    }
    expect(Math.abs(nodes[0].y - 80)).toBeLessThan(18);
    expect(Math.abs(nodes[1].y - 80)).toBeLessThan(18);
    const after = chordDeviation(
      [stemWorld(a, 'p', 400, 200), ...nodes, stemWorld(b, 'p', 400, 200)],
      400,
      200,
    );
    expect(after).toBeLessThan(20);
  });
});
