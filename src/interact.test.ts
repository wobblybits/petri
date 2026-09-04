import { describe, expect, it } from 'vitest';
import { portWorld } from './agents.ts';
import { Camera } from './camera.ts';
import { Interaction, PAINT_DEPOSIT, pickAgent, pickPort } from './interact.ts';
import { defaultParams } from './params.ts';
import { SPLATTER_COUNT, SPLATTER_RADIUS, splatter } from './presets.ts';
import { Sim } from './sim.ts';

function scene() {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.snapRadius = 0;
  const sim = new Sim(800, 600);
  const camera = new Camera();
  camera.setView(800, 600);
  camera.zoom = 1;
  camera.snap(400, 300);
  return { sim, camera, params, ui: new Interaction(sim, camera) };
}

describe('picking', () => {
  it('finds a body under the pointer and nothing in empty space', () => {
    const { sim, params } = scene();
    const con = sim.spawn('con', 400, 300, 0, params, true)!;
    expect(pickAgent(sim, 400, 300)?.id).toBe(con.id);
    expect(pickAgent(sim, 400, 900)).toBeNull();
  });

  it('finds a free port but ignores one already wired', () => {
    const { sim, camera, params } = scene();
    const a = sim.spawn('era', 380, 300, 0, params, true)!;
    const b = sim.spawn('era', 460, 300, Math.PI, params, true)!;
    a.extra = 1;
    b.extra = 1;
    const pa = portWorld(a, 'p', sim.w, sim.h);
    expect(pickPort(sim, pa.x, pa.y, camera.zoom)?.id).toBe(a.id);
    sim.wire(a.id, 'p', b.id, 'p', params);
    expect(pickPort(sim, pa.x, pa.y, camera.zoom)).toBeNull();
  });
});

describe('gestures', () => {
  it('drags a body without writing its position directly', () => {
    const { sim, params, ui } = scene();
    const con = sim.spawn('con', 400, 300, 0, params, true)!;
    ui.begin(400, 300, 400, 300);
    expect(ui.gesture.kind).toBe('drag');
    expect(sim.grabbed?.id).toBe(con.id);
    ui.move(500, 300, 500, 300);
    // The pointer target moved; the agent has not teleported to it.
    expect(sim.grabbed?.x).toBe(500);
    expect(con.x).toBe(400);
    // It only follows once the solver runs.
    for (let i = 0; i < 60; i++) sim.dragStep(params, 1 / 60);
    expect(con.x, `agent x = ${con.x.toFixed(1)}`).toBeGreaterThan(480);
    expect(Math.hypot(con.vx, con.vy), 'no stored-up velocity').toBeLessThan(1);
  });

  it('pans only once the pointer has really moved, and reports a click otherwise', () => {
    const { camera, ui } = scene();
    const x0 = camera.x;
    ui.begin(0, 0, 100, 100);
    expect(ui.gesture.kind).toBe('pan');
    ui.move(0, 0, 101, 100);
    expect(camera.x, 'a jittery press should not pan').toBe(x0);
    expect(ui.end(0, 0, () => {}), 'and should still count as a click').not.toBeNull();

    ui.begin(0, 0, 100, 100);
    ui.move(0, 0, 160, 100);
    expect(camera.x).toBeLessThan(x0);
    expect(ui.end(0, 0, () => {}), 'a real drag is not a click').toBeNull();
    expect(ui.freeCamera).toBe(true);
  });

  it('wires two free ports and refuses to wire an agent to itself', () => {
    const { sim, camera, params, ui } = scene();
    const con = sim.spawn('con', 400, 300, 0, params, true)!;
    const era = sim.spawn('era', 470, 300, Math.PI, params, true)!;
    con.extra = 1;
    era.extra = 1;
    const pc = portWorld(con, 'p', sim.w, sim.h);
    const pe = portWorld(era, 'p', sim.w, sim.h);

    // Same agent, two ports: nothing should connect.
    const pl = portWorld(con, 'l', sim.w, sim.h);
    ui.begin(pc.x, pc.y, 0, 0);
    ui.move(pl.x, pl.y, 0, 0);
    expect(ui.gesture.kind === 'wire' && ui.gesture.over).toBeNull();
    let made = 0;
    ui.end(pl.x, pl.y, () => made++);
    expect(made).toBe(0);

    ui.begin(pc.x, pc.y, 0, 0);
    ui.move(pe.x, pe.y, 0, 0);
    expect(ui.gesture.kind === 'wire' && ui.gesture.over?.id).toBe(era.id);
    ui.end(pe.x, pe.y, (a, b) => sim.wire(a.id, a.slot, b.id, b.slot, params));
    expect(sim.graph.wires.size).toBe(1);
    expect(camera.zoom).toBe(1);
  });

  it('releases the grab when a gesture is cancelled', () => {
    const { sim, params, ui } = scene();
    sim.spawn('con', 400, 300, 0, params, true);
    ui.begin(400, 300, 0, 0);
    expect(sim.grabbed).not.toBeNull();
    ui.cancel();
    expect(sim.grabbed).toBeNull();
    expect(ui.gesture.kind).toBe('none');
  });
});

describe('eraser', () => {
  it('kills a body on a plain click, and a click is not a spawn point', () => {
    const { sim, params, ui } = scene();
    const con = sim.spawn('con', 400, 300, 0, params, true)!;
    ui.eraserMode = true;
    ui.begin(400, 300, 400, 300);
    expect(sim.agents.has(con.id)).toBe(false);
    expect(ui.end(400, 300, () => {})).toBeNull();
  });

  it('sweeps the whole drag path, not just its endpoints', () => {
    const { sim, params, ui } = scene();
    // Sits on the line from begin to move, but far from either endpoint.
    const mid = sim.spawn('con', 450, 300, 0, params, true)!;
    ui.eraserMode = true;
    ui.begin(400, 300, 400, 300);
    expect(sim.agents.has(mid.id), 'too far from the press point alone').toBe(true);
    ui.move(500, 300, 500, 300);
    expect(sim.agents.has(mid.id), 'but the segment passed right over it').toBe(false);
  });

  it('erases instead of starting a wire, even over a free port', () => {
    const { sim, params, ui } = scene();
    const con = sim.spawn('con', 400, 300, 0, params, true)!;
    const pc = portWorld(con, 'p', sim.w, sim.h);
    ui.eraserMode = true;
    ui.begin(pc.x, pc.y, 0, 0);
    expect(ui.gesture.kind).toBe('erase');
    expect(sim.grabbed).toBeNull();
    expect(sim.agents.has(con.id)).toBe(false);
  });

  it('leaves normal picking alone once switched off again', () => {
    const { sim, params, ui } = scene();
    const con = sim.spawn('con', 400, 300, 0, params, true)!;
    ui.eraserMode = true;
    ui.eraserMode = false;
    ui.begin(400, 300, 400, 300);
    expect(ui.gesture.kind).toBe('drag');
    expect(sim.agents.has(con.id)).toBe(true);
  });
});

describe('paint', () => {
  it('deposits energy on a click and does not spawn', () => {
    const { sim, ui } = scene();
    const before = sim.energy.getAt(24, 24);
    ui.tool = 'paint';
    ui.begin(24, 24, 24, 24);
    expect(sim.energy.getAt(24, 24)).toBeCloseTo(before + PAINT_DEPOSIT);
    expect(ui.end(24, 24, () => {})).toBeNull();
  });

  it('covers cells along the drag, not just the endpoints', () => {
    const { sim, ui } = scene();
    const size = sim.energy.cellSize;
    ui.tool = 'paint';
    ui.begin(size * 0.5, size * 0.5, 0, 0);
    ui.move(size * 4.5, size * 0.5, 0, 0);
    expect(sim.energy.getAt(size * 2.5, size * 0.5)).toBeGreaterThan(sim.energy.ambient);
  });

  it('does not restack a cell during one stroke', () => {
    const { sim, ui } = scene();
    ui.tool = 'paint';
    ui.begin(24, 24, 0, 0);
    const once = sim.energy.getAt(24, 24);
    ui.move(25, 24, 0, 0);
    expect(sim.energy.getAt(24, 24)).toBe(once);
  });

  it('paints instead of wiring, even over a free port', () => {
    const { sim, params, ui } = scene();
    const con = sim.spawn('con', 400, 300, 0, params, true)!;
    const pc = portWorld(con, 'p', sim.w, sim.h);
    ui.tool = 'paint';
    ui.begin(pc.x, pc.y, 0, 0);
    expect(ui.gesture.kind).toBe('paint');
    expect(sim.agents.has(con.id)).toBe(true);
    expect(sim.grabbed).toBeNull();
  });
});

describe('splat', () => {
  it('fires onSplat once per press, not per move', () => {
    const { ui } = scene();
    const hits: Array<[number, number]> = [];
    ui.tool = 'splat';
    ui.onSplat = (x, y) => hits.push([x, y]);
    ui.begin(400, 300, 400, 300);
    ui.move(500, 300, 500, 300);
    ui.end(500, 300, () => {});
    expect(hits).toEqual([[400, 300]]);
    expect(ui.gesture.kind).toBe('none');
  });

  it('does not start a wire or a pan', () => {
    const { sim, params, ui } = scene();
    const con = sim.spawn('con', 400, 300, 0, params, true)!;
    const pc = portWorld(con, 'p', sim.w, sim.h);
    ui.tool = 'splat';
    ui.begin(pc.x, pc.y, 0, 0);
    expect(ui.gesture.kind).toBe('splat');
    expect(sim.grabbed).toBeNull();
  });
});

describe('splatter', () => {
  it('drops about fifty mixed agents in a disk', () => {
    const { sim, params } = scene();
    const n = splatter(sim, 400, 300, params);
    expect(n).toBe(SPLATTER_COUNT);
    expect(sim.agents.size).toBe(SPLATTER_COUNT);
    let era = 0;
    let dup = 0;
    let con = 0;
    for (const a of sim.agents.values()) {
      const d = Math.hypot(a.x - 400, a.y - 300);
      expect(d).toBeLessThanOrEqual(SPLATTER_RADIUS + 1e-6);
      if (a.kind === 'era') era++;
      else if (a.kind === 'dup') dup++;
      else con++;
    }
    expect(era).toBeGreaterThan(0);
    expect(dup).toBeGreaterThan(0);
    expect(con).toBeGreaterThan(0);
  });

  it('stops at maxAgents', () => {
    const { sim, params } = scene();
    params.maxAgents = 10;
    expect(splatter(sim, 400, 300, params, 50)).toBe(10);
    expect(splatter(sim, 400, 300, params, 50)).toBe(0);
  });
});
