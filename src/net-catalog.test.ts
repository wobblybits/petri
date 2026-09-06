import { describe, expect, it } from 'vitest';
import { CATALOG, GALLERY_TABS, catalogPieces } from './net-catalog.ts';
import { ap, church, compile, evalChurch, findRedex, lam, v, PLUS } from './lambda.ts';
import { formatWires, parseNet } from './net-text.ts';
import { applyRewrite, detectRule, type NetSnapshot } from './rewrite.ts';
import type { AgentKind } from './agents.ts';

function principalPairs(text: string): number {
  const snap = parseNet(text);
  expect(snap).not.toBeNull();
  return snap!.wires.filter((w) => w.a.slot === 'p' && w.b.slot === 'p').length;
}

function asNet(text: string): { net: NetSnapshot; nextId: number } {
  const snap = parseNet(text);
  expect(snap).not.toBeNull();
  return {
    net: {
      agents: snap!.agents.map((a) => ({ id: a.id, kind: a.kind })),
      wires: snap!.wires.map((w) => ({ a: w.a, b: w.b })),
    },
    nextId: snap!.nextId,
  };
}

function step(net: NetSnapshot, nextId: number): { net: NetSnapshot; nextId: number } | null {
  const redex = findRedex(net);
  if (!redex) return null;
  const kindOf = (id: number): AgentKind | undefined => net.agents.find((a) => a.id === id)?.kind;
  const ka = kindOf(redex.a);
  const kb = kindOf(redex.b);
  if (!ka || !kb) return null;
  const out = applyRewrite(net, detectRule(ka, kb), redex.a, redex.b, nextId);
  return { net: out.net, nextId: out.nextId };
}

function reduceSteps(text: string, max = 40): number {
  let { net, nextId } = asNet(text);
  let steps = 0;
  while (steps < max) {
    const out = step(net, nextId);
    if (!out) break;
    net = out.net;
    nextId = out.nextId;
    steps++;
  }
  return steps;
}

describe('gallery tabs', () => {
  it('keeps numerals inside Lambda rather than a separate tab', () => {
    expect(GALLERY_TABS.map((t) => t.id)).toEqual([
      'yours',
      'rules',
      'quines',
      'worms',
      'meshes',
      'lambda',
    ]);
    expect(CATALOG.lambda.some((p) => p.name === '0')).toBe(true);
    expect(CATALOG.lambda.some((p) => p.name === '2 + 2')).toBe(true);
    expect(catalogPieces('yours')).toEqual([]);
  });
});

describe('catalog nets', () => {
  it('every piece is a valid HVM2 net', () => {
    for (const [tab, pieces] of Object.entries(CATALOG)) {
      expect(pieces.length, tab).toBeGreaterThan(0);
      for (const piece of pieces) {
        const snap = parseNet(piece.text);
        expect(snap, `${tab}/${piece.name}`).not.toBeNull();
        expect(snap!.agents.length, `${tab}/${piece.name}`).toBeGreaterThan(0);
      }
    }
  });

  it('rules, quines, worms, and meshes each contain a principal pair', () => {
    for (const tab of ['rules', 'quines', 'worms', 'meshes'] as const) {
      for (const piece of CATALOG[tab]) {
        expect(principalPairs(piece.text), `${tab}/${piece.name}`).toBeGreaterThan(0);
      }
    }
  });

  it('encodes a compiled Church numeral as a parseable net', () => {
    const c = compile(church(2));
    const text = formatWires(c.net.agents, c.net.wires);
    const snap = parseNet(text);
    expect(snap).not.toBeNull();
    expect(snap!.agents).toHaveLength(c.net.agents.length);
    expect(snap!.wires).toHaveLength(c.net.wires.length);
  });
});

describe('worms rewrite along the chain', () => {
  it('cons worm is one open commute packet', () => {
    const piece = CATALOG.worms.find((p) => p.name === 'cons worm');
    expect(piece).toBeDefined();
    expect(piece!.text.replace(/\s+/g, ' ').trim()).toBe('& {* a} ~ (a b)');
    expect(principalPairs(piece!.text)).toBe(1);
    expect(reduceSteps(piece!.text)).toBeGreaterThan(0);
  });

  it('dup worm is the mirrored open packet', () => {
    const piece = CATALOG.worms.find((p) => p.name === 'dup worm');
    expect(piece).toBeDefined();
    expect(piece!.text.replace(/\s+/g, ' ').trim()).toBe('& {a b} ~ (b *)');
    expect(principalPairs(piece!.text)).toBe(1);
    expect(reduceSteps(piece!.text)).toBeGreaterThan(0);
  });
});

describe('lambda pieces', () => {
  it('applied arithmetic matches the compiler', () => {
    expect(evalChurch(ap(lam('x', v('x')), church(2))).value).toBe(2);
    expect(evalChurch(ap(PLUS, church(2), church(2))).value).toBe(4);
  });

  it('applied catalog terms start with a redex; combinators do not', () => {
    for (const name of ['I 2', 'SUCC 1', '2 + 2', '2 × 1', 'Ω']) {
      const piece = CATALOG.lambda.find((p) => p.name === name);
      expect(piece, name).toBeDefined();
      expect(principalPairs(piece!.text), name).toBeGreaterThan(0);
    }
    for (const name of ['I', 'K', '0', '1', '2', 'TRUE', 'FALSE']) {
      const piece = CATALOG.lambda.find((p) => p.name === name);
      expect(piece, name).toBeDefined();
      expect(principalPairs(piece!.text), name).toBe(0);
    }
  });
});
