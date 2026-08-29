import { describe, expect, it } from 'vitest';
import { createAgent, stemWorld } from './agents.ts';
import {
  chordDeviation,
  catmullSegment,
  desiredLinks,
  portExitAngle,
  solveWire,
  stemPoint,
  type ChainNode,
  type WireStiffness,
} from './chain.ts';
import { defaultParams } from './params.ts';

function node(x: number, y: number): ChainNode {
  return { x, y, vx: 0, vy: 0, prevX: x, prevY: y, integVx: 0, integVy: 0 };
}

const firm: WireStiffness = { scale: 1, slack: 1 };

/** Run the solver the way the sim does: many substeps, one iteration each. */
function relax(fn: (h: number) => void, frames = 60, substeps = 8): void {
  const h = 1 / 60 / substeps;
  for (let f = 0; f < frames; f++) for (let s = 0; s < substeps; s++) fn(h);
}

describe('wire constraints', () => {
  it('keeps a dense chain for long latches', () => {
    expect(desiredLinks(160)).toBeGreaterThanOrEqual(12);
    expect(desiredLinks(40)).toBeGreaterThanOrEqual(4);
  });

  it('pulls a zigzag rope straight between two anchored ports', () => {
    const params = defaultParams();
    const a = createAgent(1, 'era', 80, 80, 0, params);
    const b = createAgent(2, 'era', 220, 80, Math.PI, params);
    a.locked = true;
    b.locked = true;
    const nodes = [node(120, 150), node(170, 20)];
    const before = chordDeviation(
      [stemWorld(a, 'p', 400, 200), ...nodes, stemWorld(b, 'p', 400, 200)],
      400,
      200,
    );
    relax((h) => solveWire(a, "p", b, "p", nodes, 140, 140, [], firm, h));
    const after = chordDeviation(
      [stemWorld(a, 'p', 400, 200), ...nodes, stemWorld(b, 'p', 400, 200)],
      400,
      200,
    );
    expect(after, `chord deviation ${before.toFixed(1)} → ${after.toFixed(1)}`).toBeLessThan(
      before * 0.4,
    );
    expect(after).toBeLessThan(20);
  });

  it('drives the stem-to-stem span toward the rest length', () => {
    const params = defaultParams();
    const a = createAgent(1, 'era', 60, 100, 0, params);
    const b = createAgent(2, 'era', 300, 100, Math.PI, params);
    const nodes = [node(120, 100), node(180, 100), node(240, 100)];
    relax((h) => solveWire(a, "p", b, "p", nodes, 120, 120, [], firm, h), 240);
    const sa = stemPoint(a, 'p');
    const sb = stemPoint(b, 'p');
    const span = Math.hypot(sb.x - sa.x, sb.y - sa.y);
    expect(Math.abs(span - 120), `span ${span.toFixed(1)} vs rest 120`).toBeLessThan(4);
  });

  it('measures the angle a wire leaves its port at', () => {
    const params = defaultParams();
    const con = createAgent(1, 'con', 200, 120, 0, params);
    // The aux-l axis points backwards (-x). A target dead behind is aligned...
    expect(Math.abs(portExitAngle(con, 'l', { x: 100, y: 120 }))).toBeLessThan(0.2);
    // ...and one straight ahead is a half turn out.
    expect(Math.abs(portExitAngle(con, 'l', { x: 300, y: 120 }))).toBeGreaterThan(2.9);
  });

  it('is softer at the port than along the link', () => {
    // A port constraint that outranked the link would collapse the rope onto
    // the axis; the link must win.
    const params = defaultParams();
    const a = createAgent(1, 'era', 100, 100, 0, params);
    const b = createAgent(2, 'era', 100, 220, Math.PI, params);
    a.locked = true;
    b.locked = true;
    const nodes = [node(140, 130), node(150, 160), node(140, 190)];
    relax((h) => solveWire(a, "p", b, "p", nodes, 120, 120, [], firm, h), 240);
    const seg = Math.hypot(nodes[1].x - nodes[0].x, nodes[1].y - nodes[0].y);
    expect(Math.abs(seg - 30), `link length ${seg.toFixed(1)} vs 30`).toBeLessThan(6);
  });
});

describe('catmull handles', () => {
  it('clamps handles to the local span so a far neighbour cannot loop a segment', () => {
    const c = catmullSegment(
      { x: -200, y: 0 },
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 400, y: 0 },
    );
    const h1 = Math.hypot(c.p1.x - c.p0.x, c.p1.y - c.p0.y);
    expect(h1).toBeLessThanOrEqual(10 * 0.45 + 1e-6);
  });
});
