import { describe, expect, it } from 'vitest';
import { portWorld, stemWorld } from './agents.ts';
import { Camera } from './camera.ts';
import { defaultParams } from './params.ts';
import {
  NetEditor,
  capture,
  copyFragment,
  pasteFragment,
  pointInPolygon,
  restore,
  rotateAgents,
  selectionCentroid,
  spawnDesigned,
  translateAgents,
} from './net-edit.ts';
import { Sim } from './sim.ts';

function scene() {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.snapRadius = 0;
  params.rewriteDuration = 0;
  const sim = new Sim(800, 600);
  const camera = new Camera();
  camera.setView(800, 600);
  camera.zoom = 1;
  camera.snap(400, 300);
  const editor = new NetEditor(sim, camera, params);
  return { sim, camera, params, editor };
}

describe('pointInPolygon', () => {
  it('contains the centre of a square and not a point outside', () => {
    const sq = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ];
    expect(pointInPolygon(5, 5, sq)).toBe(true);
    expect(pointInPolygon(20, 5, sq)).toBe(false);
  });
});

describe('place and aim', () => {
  it('places a body on click and dragging an existing body repositions it', () => {
    const { editor, sim } = scene();
    editor.tool = 'paint-con';
    editor.begin(400, 300, 400, 300);
    editor.end(400, 300);
    expect(sim.agents.size).toBe(1);
    const con = [...sim.agents.values()][0]!;
    editor.begin(con.x, con.y, con.x, con.y);
    expect(editor.gesture.kind).toBe('move');
    editor.move(430, 310, 430, 310);
    editor.end(430, 310);
    expect(sim.agents.size).toBe(1);
    expect(con.x).toBeCloseTo(430, 5);
    expect(con.y).toBeCloseTo(310, 5);
  });

  it('aims the placed body toward the drag instead of stamping a chain', () => {
    const { editor, sim } = scene();
    editor.tool = 'paint-con';
    editor.begin(200, 300, 200, 300);
    editor.move(200, 360, 200, 360);
    editor.end(200, 360);
    expect(sim.agents.size).toBe(1);
    expect(sim.graph.wires.size).toBe(0);
    const con = [...sim.agents.values()][0]!;
    expect(con.x).toBeCloseTo(200, 5);
    expect(con.y).toBeCloseTo(300, 5);
    expect(con.heading).toBeCloseTo(Math.PI / 2, 5);
  });

  it('does not rotate on a tiny drag that is still a click', () => {
    const { editor, sim } = scene();
    editor.tool = 'paint-era';
    editor.begin(200, 300, 200, 300);
    editor.move(201, 300, 201, 300);
    editor.end(201, 300);
    const era = [...sim.agents.values()][0]!;
    expect(era.heading).toBeCloseTo(0, 5);
  });

  it('wires the principal to another agent if the aim line lands on it', () => {
    const { editor, sim, params } = scene();
    const dest = spawnDesigned(sim, 'era', 380, 300, Math.PI, params)!;
    editor.tool = 'paint-era';
    editor.begin(200, 300, 200, 300);
    const placed = [...editor.selection][0]!;
    editor.move(dest.x, dest.y, dest.x, dest.y);
    editor.end(dest.x, dest.y);
    expect(sim.agents.size).toBe(2);
    expect(sim.graph.wires.size).toBe(1);
    expect(sim.graph.isFreeAt(placed, 'p')).toBe(false);
    expect(sim.graph.isFreeAt(dest.id, 'p')).toBe(false);
  });

  it('wires to a free port nub, not only the body', () => {
    const { editor, sim, params } = scene();
    const dest = spawnDesigned(sim, 'con', 400, 300, Math.PI, params)!;
    const pl = portWorld(dest, 'l', sim.w, sim.h);
    editor.tool = 'paint-con';
    editor.begin(200, 300, 200, 300);
    const placed = [...editor.selection][0]!;
    editor.move(pl.x, pl.y, pl.x, pl.y);
    editor.end(pl.x, pl.y);
    expect(sim.graph.wires.size).toBe(1);
    expect(sim.graph.isFreeAt(placed, 'p')).toBe(false);
    expect(sim.graph.isFreeAt(dest.id, 'l')).toBe(false);
    expect(sim.graph.isFreeAt(dest.id, 'p')).toBe(true);
  });

  it('does not wire to itself', () => {
    const { editor, sim } = scene();
    editor.tool = 'paint-con';
    editor.begin(200, 300, 200, 300);
    editor.move(205, 300, 205, 300);
    editor.end(205, 300);
    expect(sim.graph.wires.size).toBe(0);
  });

  it('skips a press that would overlap an existing body', () => {
    const { editor, sim, params } = scene();
    spawnDesigned(sim, 'con', 148, 300, 0, params);
    editor.tool = 'paint-dup';
    editor.begin(180, 300, 180, 300);
    editor.end(180, 300);
    expect(sim.agents.size).toBe(1);
    expect([...sim.agents.values()][0]!.kind).toBe('con');
  });
});

describe('lasso select', () => {
  it('selects bodies whose centres sit inside the loop', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'era', 200, 200, 0, params)!;
    const b = spawnDesigned(sim, 'era', 400, 400, 0, params)!;
    editor.tool = 'select';
    editor.begin(150, 150, 150, 150);
    editor.move(250, 150, 250, 150);
    editor.move(250, 250, 250, 250);
    editor.move(150, 250, 150, 250);
    editor.end(150, 250);
    expect([...editor.selection]).toEqual([a.id]);
    expect(editor.selection.has(b.id)).toBe(false);
  });

  it('a click on a body selects it, a click on empty space clears', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 400, 300, 0, params)!;
    editor.begin(400, 300, 400, 300);
    editor.end(400, 300);
    expect([...editor.selection]).toEqual([a.id]);
    editor.begin(10, 10, 10, 10);
    editor.end(10, 10);
    expect(editor.selection.size).toBe(0);
  });
});

describe('clipboard', () => {
  it('copies internal wires and pastes a translated duplicate', () => {
    const { sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 200, 200, 0.4, params)!;
    const b = spawnDesigned(sim, 'dup', 260, 200, -0.4, params)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const frag = copyFragment(sim, [a.id, b.id])!;
    expect(frag.agents.length).toBe(2);
    expect(frag.wires.length).toBe(1);
    const ids = pasteFragment(sim, params, frag, 400, 400);
    expect(ids.length).toBe(2);
    expect(sim.agents.size).toBe(4);
    expect(sim.graph.wires.size).toBe(2);
    const c = selectionCentroid(sim, ids)!;
    expect(c.x).toBeCloseTo(400, 5);
    expect(c.y).toBeCloseTo(400, 5);
  });

  it('cut removes the selection and paste puts it back', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'era', 200, 200, 0, params)!;
    editor.selection.add(a.id);
    expect(editor.cut()).toBe(true);
    expect(sim.agents.size).toBe(0);
    editor.paste(300, 300);
    expect(sim.agents.size).toBe(1);
    const pasted = sim.agents.get([...editor.selection][0]!)!;
    expect(pasted.x).toBeCloseTo(300, 5);
    expect(pasted.y).toBeCloseTo(300, 5);
  });
});

describe('rotate and translate', () => {
  it('rotates a pair as a rigid body around their centroid', () => {
    const { sim, params } = scene();
    const a = spawnDesigned(sim, 'era', 0, 0, 0, params)!;
    const b = spawnDesigned(sim, 'era', 10, 0, 0, params)!;
    rotateAgents(sim, [a.id, b.id], Math.PI / 2, 5, 0, params);
    expect(a.x).toBeCloseTo(5, 5);
    expect(a.y).toBeCloseTo(-5, 5);
    expect(b.x).toBeCloseTo(5, 5);
    expect(b.y).toBeCloseTo(5, 5);
    expect(a.heading).toBeCloseTo(Math.PI / 2, 5);
  });

  it('translates every selected body by the same delta', () => {
    const { sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 10, 20, 0, params)!;
    translateAgents(sim, [a.id], 5, -3, params);
    expect(a.x).toBeCloseTo(15, 5);
    expect(a.y).toBeCloseTo(17, 5);
  });
});

describe('port cycle', () => {
  it('walks a wire around a constructor, including empty slots', () => {
    const { editor, sim, params } = scene();
    const con = spawnDesigned(sim, 'con', 400, 300, 0, params)!;
    const era = spawnDesigned(sim, 'era', 460, 300, Math.PI, params)!;
    sim.wire(con.id, 'p', era.id, 'p', params);
    editor.selection.add(con.id);
    expect(editor.cycleSelection(1)).toBe(true);
    expect(sim.graph.wireAtSlot(con.id, 'p')).toBeUndefined();
    expect(sim.graph.wireAtSlot(con.id, 'l')?.id).toBeDefined();
    expect(sim.graph.isFreeAt(era.id, 'p')).toBe(false);
    editor.cycleSelection(1);
    expect(sim.graph.wireAtSlot(con.id, 'r')?.id).toBeDefined();
    editor.cycleSelection(1);
    expect(sim.graph.wireAtSlot(con.id, 'p')?.id).toBeDefined();
  });

  it('does nothing to an Era', () => {
    const { editor, sim, params } = scene();
    const era = spawnDesigned(sim, 'era', 400, 300, 0, params)!;
    editor.selection.add(era.id);
    expect(editor.cycleSelection(1)).toBe(false);
  });
});

describe('undo and redo', () => {
  it('restores pose and wiring after a paint stroke', () => {
    const { editor, sim } = scene();
    editor.tool = 'paint-con';
    editor.begin(200, 200, 200, 200);
    editor.end(200, 200);
    expect(sim.agents.size).toBe(1);
    expect(editor.undo()).toBe(true);
    expect(sim.agents.size).toBe(0);
    expect(editor.redo()).toBe(true);
    expect(sim.agents.size).toBe(1);
  });

  it('capture/restore round-trips ids, kinds, and wires', () => {
    const { sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 120, 80, 0.2, params)!;
    const b = spawnDesigned(sim, 'dup', 180, 90, -0.5, params)!;
    sim.wire(a.id, 'l', b.id, 'r', params);
    const snap = capture(sim);
    a.x = 0;
    restore(sim, snap, params);
    expect(sim.agents.size).toBe(2);
    expect(sim.agents.get(a.id)?.x).toBeCloseTo(120, 5);
    expect(sim.agents.get(a.id)?.kind).toBe('con');
    expect(sim.graph.wires.size).toBe(1);
    expect(sim.graph.isFreeAt(a.id, 'l')).toBe(false);
    expect(sim.graph.isFreeAt(b.id, 'r')).toBe(false);
  });
});

describe('position pin', () => {
  it('toggles pin on the selection and restores it through undo', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 200, 300, 0, params)!;
    editor.selection.add(a.id);
    expect(editor.selectionIsPinned()).toBe(false);
    expect(editor.togglePinSelection()).toBe(true);
    expect(a.pinned).toBe(true);
    expect(editor.selectionIsPinned()).toBe(true);
    expect(editor.togglePinSelection()).toBe(true);
    expect(a.pinned).toBe(false);
    expect(editor.undo()).toBe(true);
    expect(sim.agents.get(a.id)?.pinned).toBe(true);
  });

  it('capture/restore keeps the pin bit', () => {
    const { sim, params } = scene();
    const a = spawnDesigned(sim, 'era', 120, 80, 0.2, params)!;
    a.pinned = true;
    const snap = capture(sim);
    a.pinned = false;
    restore(sim, snap, params);
    expect(sim.agents.get(a.id)?.pinned).toBe(true);
  });

  it('keeps a pinned body still while settle pulls its neighbour', () => {
    const { sim, params, editor } = scene();
    params.flockAlign = 0;
    params.flockSep = 0;
    params.declutter = 0;
    params.springK = 90;
    const a = spawnDesigned(sim, 'con', 200, 300, 0, params)!;
    const b = spawnDesigned(sim, 'con', 360, 300, Math.PI, params)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    editor.selection.add(a.id);
    editor.togglePinSelection();
    const ax = a.x;
    const ay = a.y;
    for (let i = 0; i < 40; i++) sim.step(1 / 60, params);
    expect(a.x).toBeCloseTo(ax, 5);
    expect(a.y).toBeCloseTo(ay, 5);
    expect(Math.hypot(b.x - 360, b.y - 300)).toBeGreaterThan(2);
  });
});

describe('port drag is the default wire', () => {
  it('starts a wire from a free port while the rotate tool is selected', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'era', 200, 300, 0, params)!;
    const b = spawnDesigned(sim, 'era', 380, 300, Math.PI, params)!;
    const pa = portWorld(a, 'p', sim.w, sim.h);
    const pb = portWorld(b, 'p', sim.w, sim.h);
    editor.tool = 'rotate';
    editor.selection.add(a.id);
    editor.begin(pa.x, pa.y, pa.x, pa.y);
    expect(editor.gesture.kind).toBe('wire');
    editor.move(pb.x, pb.y, pb.x, pb.y);
    editor.end(pb.x, pb.y);
    expect(sim.graph.wires.size).toBe(1);
    expect(sim.graph.isFreeAt(a.id, 'p')).toBe(false);
  });

  it('starts a wire from a free port while a place tool is selected', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 200, 300, 0, params)!;
    const b = spawnDesigned(sim, 'dup', 380, 300, Math.PI, params)!;
    const pa = portWorld(a, 'p', sim.w, sim.h);
    const pb = portWorld(b, 'p', sim.w, sim.h);
    editor.tool = 'paint-era';
    editor.begin(pa.x, pa.y, pa.x, pa.y);
    expect(editor.gesture.kind).toBe('wire');
    expect(sim.agents.size).toBe(2);
    editor.end(pb.x, pb.y);
    expect(sim.agents.size).toBe(2);
    expect(sim.graph.wires.size).toBe(1);
  });

  it('still erases when the press is on a free port', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 200, 300, 0, params)!;
    const pa = portWorld(a, 'p', sim.w, sim.h);
    editor.tool = 'erase';
    editor.begin(pa.x, pa.y, pa.x, pa.y);
    editor.end(pa.x, pa.y);
    expect(sim.agents.has(a.id)).toBe(false);
    expect(sim.graph.wires.size).toBe(0);
  });
});

describe('body drag repositions on most tools', () => {
  it('moves an agent while a place tool is selected', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'era', 200, 300, 0, params)!;
    editor.tool = 'paint-con';
    editor.begin(200, 300, 200, 300);
    expect(editor.gesture.kind).toBe('move');
    editor.move(240, 320, 240, 320);
    editor.end(240, 320);
    expect(sim.agents.size).toBe(1);
    expect(a.x).toBeCloseTo(240, 5);
    expect(a.y).toBeCloseTo(320, 5);
  });

  it('moves an agent while the rotate tool is selected', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'dup', 200, 300, 0.4, params)!;
    const heading = a.heading;
    editor.tool = 'rotate';
    editor.begin(200, 300, 200, 300);
    expect(editor.gesture.kind).toBe('move');
    editor.move(250, 300, 250, 300);
    editor.end(250, 300);
    expect(a.x).toBeCloseTo(250, 5);
    expect(a.heading).toBeCloseTo(heading, 5);
  });

  it('still rotates the selection when the rotate tool drags empty space', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'era', 200, 300, 0, params)!;
    editor.tool = 'rotate';
    editor.selection.add(a.id);
    editor.begin(40, 40, 40, 40);
    expect(editor.gesture.kind).toBe('rotate');
    editor.move(40, 80, 40, 80);
    editor.end(40, 80);
    expect(a.heading).not.toBeCloseTo(0, 5);
  });
});

describe('touch tool', () => {
  it('plays a body without moving it or taking an undo checkpoint', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 200, 300, 0, params)!;
    const x = a.x;
    const y = a.y;
    editor.tool = 'touch';
    editor.begin(x, y, x, y);
    expect(editor.gesture.kind).toBe('sound');
    if (editor.gesture.kind === 'sound') expect(editor.gesture.stroke.hit).toEqual({ kind: 'body', id: a.id });
    editor.move(x + 12, y, x + 12, y);
    editor.end(x + 12, y);
    expect(a.x).toBeCloseTo(x, 5);
    expect(a.y).toBeCloseTo(y, 5);
    expect(editor.canUndo).toBe(false);
    expect(editor.gesture.kind).toBe('none');
  });

  it('pans empty space instead of lassoing', () => {
    const { editor } = scene();
    editor.tool = 'touch';
    const camX = editor.camera.x;
    editor.begin(40, 40, 40, 40);
    expect(editor.gesture.kind).toBe('pan');
    editor.move(80, 40, 80, 40);
    editor.end(80, 40);
    expect(editor.camera.x).toBeLessThan(camX);
  });
});

describe('cut wires', () => {
  function pair() {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 200, 300, 0, params)!;
    const b = spawnDesigned(sim, 'con', 400, 300, Math.PI, params)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const sa = stemWorld(a, 'p', sim.w, sim.h);
    const sb = stemWorld(b, 'p', sim.w, sim.h);
    const mx = (sa.x + sb.x) * 0.5;
    const my = (sa.y + sb.y) * 0.5;
    return { editor, sim, a, b, mx, my };
  }

  it('erases a wire without killing its bodies', () => {
    const { editor, sim, a, b, mx, my } = pair();
    editor.tool = 'erase';
    editor.begin(mx, my, mx, my);
    editor.end(mx, my);
    expect(sim.graph.wires.size).toBe(0);
    expect(sim.agents.has(a.id)).toBe(true);
    expect(sim.agents.has(b.id)).toBe(true);
  });

  it('cuts a wire with a click on the select tool', () => {
    const { editor, sim, a, b, mx, my } = pair();
    editor.tool = 'select';
    editor.begin(mx, my, mx, my);
    expect(editor.gesture.kind).toBe('lasso');
    editor.end(mx, my);
    expect(sim.graph.wires.size).toBe(0);
    expect(sim.agents.size).toBe(2);
    expect(sim.agents.has(a.id)).toBe(true);
    expect(sim.agents.has(b.id)).toBe(true);
  });
});

describe('move one body', () => {
  it('drags only the grabbed agent even when the whole net is selected', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'con', 200, 300, 0, params)!;
    const b = spawnDesigned(sim, 'con', 400, 300, Math.PI, params)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    editor.selection = new Set([a.id, b.id]);
    editor.begin(a.x, a.y, a.x, a.y);
    editor.move(a.x, a.y + 40, a.x, a.y + 40);
    editor.end(a.x, a.y + 40);
    expect(a.y).toBeCloseTo(340, 5);
    expect(b.x).toBeCloseTo(400, 5);
    expect(b.y).toBeCloseTo(300, 5);
    expect([...editor.selection]).toEqual([a.id]);
  });

  it('shift-drag moves the rest of the selection with it', () => {
    const { editor, sim, params } = scene();
    const a = spawnDesigned(sim, 'era', 200, 300, 0, params)!;
    const b = spawnDesigned(sim, 'era', 400, 300, 0, params)!;
    editor.selection = new Set([a.id, b.id]);
    editor.begin(a.x, a.y, a.x, a.y, { shift: true });
    editor.move(a.x, a.y + 30, a.x, a.y + 30);
    editor.end(a.x, a.y + 30);
    expect(a.y).toBeCloseTo(330, 5);
    expect(b.y).toBeCloseTo(330, 5);
    expect(editor.selection.has(a.id)).toBe(true);
    expect(editor.selection.has(b.id)).toBe(true);
  });
});
