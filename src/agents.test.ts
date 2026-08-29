import { describe, expect, it } from 'vitest';
import {
  createAgent,
  headingAlongWire,
  portAxis,
  stemWorld,
  wireCubic,
} from './agents.ts';
import { bezierLength } from './curve.ts';
import { defaultParams } from './params.ts';
import { angleDelta } from './wrap.ts';

describe('port meridian headings', () => {
  const params = defaultParams();
  const w = 400;
  const h = 240;

  it('principal–principal: bodies face opposite along the wire', () => {
    const a = createAgent(1, 'era', 100, 120, 0.3, params);
    const b = createAgent(2, 'era', 220, 120, -0.2, params);
    const sa = stemWorld(a, 'p', w, h);
    const sb = stemWorld(b, 'p', w, h);
    const hA = headingAlongWire(a, 'p', sa, sb, w, h);
    const hB = headingAlongWire(b, 'p', sb, sa, w, h);
    expect(Math.abs(Math.abs(angleDelta(hA, hB)) - Math.PI)).toBeLessThan(0.08);
  });

  it('principal–aux: bodies face the same way along the wire', () => {
    const con = createAgent(1, 'con', 200, 120, 0.15, params);
    const era = createAgent(2, 'era', 140, 120, 0.9, params);
    const sc = stemWorld(con, 'l', w, h);
    const se = stemWorld(era, 'p', w, h);
    const hCon = headingAlongWire(con, 'l', sc, se, w, h);
    const hEra = headingAlongWire(era, 'p', se, sc, w, h);
    expect(Math.abs(angleDelta(hCon, hEra))).toBeLessThan(0.12);
  });

  it('each port axis points toward its neighbor after alignment', () => {
    const con = createAgent(1, 'con', 200, 120, 0, params);
    const era = createAgent(2, 'era', 130, 120, Math.PI, params);
    const sc = stemWorld(con, 'p', w, h);
    const se = stemWorld(era, 'p', w, h);
    con.heading = headingAlongWire(con, 'p', sc, se, w, h);
    era.heading = headingAlongWire(era, 'p', se, sc, w, h);
    const axisC = portAxis(con, 'p');
    const axisE = portAxis(era, 'p');
    const d = { x: se.x - sc.x, y: se.y - sc.y };
    const len = Math.hypot(d.x, d.y) || 1;
    expect(axisC.x * (d.x / len) + axisC.y * (d.y / len)).toBeGreaterThan(0.98);
    expect(axisE.x * (-d.x / len) + axisE.y * (-d.y / len)).toBeGreaterThan(0.98);
  });
});

describe('wire cubics', () => {
  it('caps handles so close ports cannot loop off-screen', () => {
    const params = defaultParams();
    const a = createAgent(1, 'era', 100, 100, 0, params);
    const b = createAgent(2, 'era', 108, 100, Math.PI, params);
    const c = wireCubic(a, 'p', b, 'p', 400, 240, 200);
    const span = Math.hypot(c.p3.x - c.p0.x, c.p3.y - c.p0.y);
    const len = bezierLength(c.p0, c.p1, c.p2, c.p3);
    expect(len).toBeLessThan(span * 3.5);
  });
});
