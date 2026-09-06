import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { Camera } from './camera.ts';
import { formatNet, looksLikeNet, parseNet } from './net-text.ts';
import { NetEditor, restore } from './net-edit.ts';
import { Sim } from './sim.ts';

function scene() {
  const params = defaultParams();
  params.spawnInterval = 0;
  params.snapRadius = 0;
  params.rewriteDuration = 0;
  const sim = new Sim(800, 600);
  return { sim, params };
}

function kinds(sim: Sim): string[] {
  return [...sim.agents.values()]
    .sort((a, b) => a.id - b.id)
    .map((a) => a.kind);
}

function load(sim: Sim, src: string): void {
  const snap = parseNet(src);
  expect(snap).not.toBeNull();
  restore(sim, snap!, defaultParams());
}

describe('formatNet', () => {
  it('encodes an empty dish as an empty string', () => {
    const { sim } = scene();
    expect(formatNet(sim)).toBe('');
  });

  it('encodes a lone era as *', () => {
    const { sim, params } = scene();
    sim.spawn('era', 0, 0, 0, params, true);
    expect(formatNet(sim)).toBe('*');
  });

  it('encodes two eras wired principal-to-principal as a redex', () => {
    const { sim, params } = scene();
    const a = sim.spawn('era', 0, 0, 0, params, true)!;
    const b = sim.spawn('era', 40, 0, 0, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    expect(formatNet(sim)).toBe('& * ~ *');
  });

  it('folds eras on a constructor’s aux ports', () => {
    const { sim, params } = scene();
    const c = sim.spawn('con', 0, 0, 0, params, true)!;
    const l = sim.spawn('era', -40, 0, 0, params, true)!;
    const r = sim.spawn('era', 40, 0, 0, params, true)!;
    sim.wire(c.id, 'l', l.id, 'p', params);
    sim.wire(c.id, 'r', r.id, 'p', params);
    expect(formatNet(sim)).toBe('(* *)');
  });

  it('encodes a commute active pair', () => {
    const { sim, params } = scene();
    const c = sim.spawn('con', 0, 0, 0, params, true)!;
    const d = sim.spawn('dup', 40, 0, Math.PI, params, true)!;
    sim.wire(c.id, 'p', d.id, 'p', params);
    for (const agent of [c, d]) {
      const l = sim.spawn('era', 0, 0, 0, params, true)!;
      const r = sim.spawn('era', 0, 0, 0, params, true)!;
      sim.wire(agent.id, 'l', l.id, 'p', params);
      sim.wire(agent.id, 'r', r.id, 'p', params);
    }
    expect(formatNet(sim)).toBe('& (* *) ~ {* *}');
  });

  it('encodes a selection and treats cut wires as free ports', () => {
    const { sim, params } = scene();
    const c = sim.spawn('con', 0, 0, 0, params, true)!;
    const l = sim.spawn('era', -40, 0, 0, params, true)!;
    const r = sim.spawn('era', 40, 0, 0, params, true)!;
    sim.wire(c.id, 'l', l.id, 'p', params);
    sim.wire(c.id, 'r', r.id, 'p', params);
    expect(formatNet(sim, [c.id])).toBe('(a b)');
    expect(formatNet(sim, [c.id, l.id])).toBe('(* a)');
  });
});

describe('parseNet', () => {
  it('parses an empty string as an empty net', () => {
    expect(parseNet('')).toEqual({ nextId: 1, agents: [], wires: [] });
    expect(parseNet('   ')).toEqual({ nextId: 1, agents: [], wires: [] });
  });

  it('parses * as one free era', () => {
    const { sim } = scene();
    load(sim, '*');
    expect(kinds(sim)).toEqual(['era']);
    expect(sim.graph.wires.size).toBe(0);
    expect(sim.graph.isFreeAt([...sim.agents.keys()][0]!, 'p')).toBe(true);
  });

  it('parses a closed era redex', () => {
    const { sim } = scene();
    load(sim, '& * ~ *');
    expect(kinds(sim)).toEqual(['era', 'era']);
    expect(sim.graph.wires.size).toBe(1);
    for (const id of sim.agents.keys()) expect(sim.graph.isFreeAt(id, 'p')).toBe(false);
  });

  it('parses nested constructors and unmatched names as free ports', () => {
    const { sim } = scene();
    load(sim, '(a b)');
    expect(kinds(sim)).toEqual(['con']);
    const id = [...sim.agents.keys()][0]!;
    expect(sim.graph.isFreeAt(id, 'p')).toBe(true);
    expect(sim.graph.isFreeAt(id, 'l')).toBe(true);
    expect(sim.graph.isFreeAt(id, 'r')).toBe(true);
  });

  it('parses a labeled dup and &! as a redex', () => {
    const { sim } = scene();
    load(sim, '&! (a b) ~ {1 a b}');
    expect(kinds(sim)).toEqual(['con', 'dup']);
    expect(sim.graph.wires.size).toBe(3);
  });

  it('parses an @name = book definition from the HVM2 docs', () => {
    const src = `
      @main
        = R
        & ((R *) (x y))
        ~ ((y x) (z z))
    `;
    const { sim } = scene();
    load(sim, src);
    expect(sim.agents.size).toBe(7);
    expect([...sim.agents.values()].filter((a) => a.kind === 'era')).toHaveLength(1);
    expect([...sim.agents.values()].filter((a) => a.kind === 'con')).toHaveLength(6);
  });

  it('rejects numbers, refs, and junk', () => {
    expect(parseNet('@sum ~ (28 (0 a))')).toBeNull();
    expect(parseNet('(a, b)')).toBeNull();
    expect(parseNet('hello world')).toBeNull();
  });
});

describe('round-trip', () => {
  it('preserves commute topology without storing pose', () => {
    const src = '& (* *) ~ {* *}';
    expect(formatNet(restoreToSim(src))).toBe(src);
  });

  it('preserves a constructor with free aux names after a parse cycle', () => {
    const src = '(a b)';
    expect(formatNet(restoreToSim(src))).toBe(src);
  });

  it('leaves imported agents on a grid, not at their old coordinates', () => {
    const { sim, params } = scene();
    const a = sim.spawn('era', 120, 340, 1.2, params, true)!;
    const b = sim.spawn('era', 500, 80, -0.4, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    const text = formatNet(sim);
    const again = new Sim(800, 600);
    load(again, text);
    const xs = [...again.agents.values()].map((ag) => ag.x);
    expect(xs).toEqual([0, 52]);
    expect([...again.agents.values()].every((ag) => ag.heading === 0)).toBe(true);
  });
});

describe('looksLikeNet', () => {
  it('accepts HVM2 fragments and book defs, not prose', () => {
    expect(looksLikeNet('*')).toBe(true);
    expect(looksLikeNet('& * ~ *')).toBe(true);
    expect(looksLikeNet('@main = *')).toBe(true);
    expect(looksLikeNet('R & (R *) ~ *')).toBe(true);
    expect(looksLikeNet('copy this')).toBe(false);
    expect(looksLikeNet('')).toBe(false);
  });
});

describe('NetEditor import/export', () => {
  it('replaces the dish and records history', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.snapRadius = 0;
    const sim = new Sim(800, 600);
    const camera = new Camera();
    const editor = new NetEditor(sim, camera, params);
    sim.spawn('era', 0, 0, 0, params, true);
    expect(editor.importText('& (* *) ~ {* *}')).toBe(true);
    expect(sim.agents.size).toBe(6);
    expect(editor.exportText()).toBe('& (* *) ~ {* *}');
    editor.undo();
    expect(sim.agents.size).toBe(1);
    expect(editor.importText('not a net')).toBe(false);
    expect(sim.agents.size).toBe(1);
  });

  it('stamps a net without clearing the dish', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    params.snapRadius = 0;
    const sim = new Sim(800, 600);
    const camera = new Camera();
    const editor = new NetEditor(sim, camera, params);
    sim.spawn('era', 0, 0, 0, params, true);
    const ids = editor.insertText('(* *)', 200, 200);
    expect(ids.length).toBe(3);
    expect(sim.agents.size).toBe(4);
    expect(editor.exportText()).toContain('(* *)');
    editor.selection = new Set(ids);
    expect(editor.exportPiece()).toBe('(* *)');
  });
});

function restoreToSim(src: string): Sim {
  const sim = new Sim(800, 600);
  load(sim, src);
  return sim;
}
