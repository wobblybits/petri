import { describe, expect, it } from 'vitest';
import {
  createAgent,
  headingAlongTow,
  meridianCenterGap,
  stemWorld,
  type Agent,
} from './agents.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';
import type { Wire } from './graph.ts';

function quietParams() {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.gravity = 0;
  params.flockAlign = 0;
  params.flockSep = 0;
  params.stepSpeed = 0;
  params.faceAttract = 0;
  params.deposit = 0;
  params.diffuse = 0;
  params.decay = 0;
  params.rewriteDuration = 0;
  params.snapRadius = 0;
  return params;
}

function stemSpan(
  sim: Sim,
  a: Agent,
  slotA: 'p' | 'l' | 'r',
  b: Agent,
  slotB: 'p' | 'l' | 'r',
): number {
  const sa = stemWorld(a, slotA, sim.w, sim.h);
  const sb = stemWorld(b, slotB, sim.w, sim.h);
  return Math.hypot(sb.x - sa.x, sb.y - sa.y);
}

function step(sim: Sim, params: ReturnType<typeof quietParams>, n: number): void {
  for (let i = 0; i < n; i++) sim.step(1 / 60, params);
}

function spanError(sim: Sim, wire: Wire): number {
  const A = sim.agents.get(wire.a.id)!;
  const B = sim.agents.get(wire.b.id)!;
  const span = stemSpan(sim, A, wire.a.slot, B, wire.b.slot);
  return span - wire.rest;
}

describe('principal joint rest length', () => {
  describe('meridianCenterGap geometry', () => {
    it('yields center gap so stems end up stemRest apart', () => {
      const params = defaultParams();
      const w = 400;
      const h = 240;
      const stemRest = 40;
      const con = createAgent(1, 'con', 200, 120, 0, params);
      const era = createAgent(2, 'era', 100, 120, 0, params);
      const pa = stemWorld(con, 'l', w, h);
      const pb = stemWorld(era, 'p', w, h);
      const prefer = Math.atan2(pa.y - pb.y, pa.x - pb.x);
      const towHeading = headingAlongTow(con, 'l', pa, pb, prefer, w, h);
      con.heading = towHeading;
      era.heading = towHeading;
      const gap = meridianCenterGap(con, 'l', era, 'p', stemRest, w, h);
      era.x = con.x - Math.cos(towHeading) * gap;
      era.y = con.y - Math.sin(towHeading) * gap;
      const span = Math.hypot(
        stemWorld(era, 'p', w, h).x - stemWorld(con, 'l', w, h).x,
        stemWorld(era, 'p', w, h).y - stemWorld(con, 'l', w, h).y,
      );
      expect(Math.abs(span - stemRest), `span=${span.toFixed(2)} vs rest=${stemRest}`).toBeLessThan(
        0.05,
      );
    });

  });

  describe('principal–principal', () => {
    it('stem span matches wire.rest after settling', () => {
      const sim = new Sim(400, 240);
      const params = quietParams();
      params.wireShrink = 20;
      params.wireMinRest = 40;
      const a = sim.spawn('era', 120, 120, 0.4, params, true)!;
      const b = sim.spawn('era', 260, 120, -0.35, params, true)!;
      sim.wire(a.id, 'p', b.id, 'p', params);
      step(sim, params, 90);
      const wire = [...sim.graph.wires.values()][0];
      const err = spanError(sim, wire);
      expect(
        Math.abs(err),
        `stem span should match wire.rest (span err=${err.toFixed(2)}, rest=${wire.rest.toFixed(2)})`,
      ).toBeLessThan(0.5);
    });

    it('FAR packed path still holds stem rest, not center rest', () => {
      const sim = new Sim(400, 240);
      const params = quietParams();
      params.wireShrink = 20;
      params.wireMinRest = 40;
      const a = sim.spawn('era', 120, 120, 0, params, true)!;
      const b = sim.spawn('era', 260, 120, Math.PI, params, true)!;
      sim.wire(a.id, 'p', b.id, 'p', params);
      const far = { x: 190, y: 120, zoom: 0.05, viewW: 800, viewH: 600 };
      for (let i = 0; i < 90; i++) sim.step(1 / 60, params, far);
      const wire = [...sim.graph.wires.values()][0];
      const err = spanError(sim, wire);
      const cen = Math.hypot(b.x - a.x, b.y - a.y);
      expect(Math.abs(err), `stem err=${err.toFixed(2)}`).toBeLessThan(2);
      expect(cen, `centers should sit outside the stems (cen=${cen.toFixed(1)})`).toBeGreaterThan(
        wire.rest + 6,
      );
    });

    it('stem span follows wire.rest while the wire shrinks', () => {
      const sim = new Sim(400, 240);
      const params = quietParams();
      params.wireShrink = 0.45;
      params.wireMinRest = 28;
      const a = sim.spawn('era', 120, 120, 0, params, true)!;
      const b = sim.spawn('era', 260, 120, Math.PI, params, true)!;
      sim.wire(a.id, 'p', b.id, 'p', params);
      const wire = [...sim.graph.wires.values()][0];
      const latchLen = wire.latchLen;
      let maxErr = 0;
      for (let i = 0; i < 36; i++) {
        sim.step(1 / 60, params);
        maxErr = Math.max(maxErr, Math.abs(spanError(sim, wire)));
      }
      const span = stemSpan(sim, a, 'p', b, 'p');
      expect(span, 'wire should shrink from latch length').toBeLessThan(latchLen - 6);
      expect(
        maxErr,
        `stem span should track wire.rest during shrink (max |span−rest|=${maxErr.toFixed(2)})`,
      ).toBeLessThan(1.5);
    });
  });

  describe('principal–aux tow', () => {
    it('stem span matches wire.rest after settling', () => {
      const sim = new Sim(400, 240);
      const params = quietParams();
      params.wireShrink = 20;
      params.wireMinRest = 40;
      const con = sim.spawn('con', 210, 120, 0, params, true)!;
      const era = sim.spawn('era', 130, 120, 0, params, true)!;
      sim.wire(con.id, 'l', era.id, 'p', params);
      step(sim, params, 90);
      const wire = [...sim.graph.wires.values()][0];
      const err = spanError(sim, wire);
      expect(
        Math.abs(err),
        `stem span should match wire.rest (span err=${err.toFixed(2)}, rest=${wire.rest.toFixed(2)})`,
      ).toBeLessThan(0.5);
    });

    it('stem span follows wire.rest while the wire shrinks', () => {
      const sim = new Sim(400, 240);
      const params = quietParams();
      params.wireShrink = 0.45;
      params.wireMinRest = 28;
      const con = sim.spawn('con', 210, 120, 0, params, true)!;
      const era = sim.spawn('era', 120, 120, 0, params, true)!;
      sim.wire(con.id, 'l', era.id, 'p', params);
      const wire = [...sim.graph.wires.values()][0];
      const latchLen = wire.latchLen;
      let maxErr = 0;
      for (let i = 0; i < 36; i++) {
        sim.step(1 / 60, params);
        maxErr = Math.max(maxErr, Math.abs(spanError(sim, wire)));
      }
      const span = stemSpan(sim, con, 'l', era, 'p');
      expect(span, 'aux wire should shrink from latch length').toBeLessThan(latchLen - 6);
      expect(
        maxErr,
        `stem span should track wire.rest during shrink (max |span−rest|=${maxErr.toFixed(2)})`,
      ).toBeLessThan(1.5);
    });

    it('cargo does not run away from the tug while settling distance', () => {
      const sim = new Sim(400, 240);
      const params = quietParams();
      params.wireShrink = 0.9;
      params.wireMinRest = 40;
      const con = sim.spawn('con', 210, 120, 0, params, true)!;
      const era = sim.spawn('era', 130, 120, 0, params, true)!;
      sim.wire(con.id, 'l', era.id, 'p', params);
      for (let i = 0; i < 120; i++) {
        sim.step(1 / 60, params);
        const sep = Math.hypot(era.x - con.x, era.y - con.y);
        expect(sep, `cargo ran away at frame ${i + 1} (sep=${sep.toFixed(0)})`).toBeLessThan(200);
      }
    });
  });
});
