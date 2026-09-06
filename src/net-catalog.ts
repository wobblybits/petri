import type { AgentKind, PortRef, PortSlot } from './agents.ts';
import { ap, app, church, compile, lam, v, MULT, PLUS, type Term } from './lambda.ts';
import { formatWires } from './net-text.ts';

export type GalleryTabId = 'yours' | 'rules' | 'quines' | 'worms' | 'meshes' | 'lambda';

export const GALLERY_TABS: { id: GalleryTabId; label: string }[] = [
  { id: 'yours', label: 'Yours' },
  { id: 'rules', label: 'Rules' },
  { id: 'quines', label: 'Quines' },
  { id: 'worms', label: 'Worms' },
  { id: 'meshes', label: 'Meshes' },
  { id: 'lambda', label: 'Lambda' },
];

export interface CatalogPiece {
  name: string;
  text: string;
}

class Sketch {
  agents: { id: number; kind: AgentKind }[] = [];
  wires: { a: PortRef; b: PortRef }[] = [];
  private next = 1;

  add(kind: AgentKind): number {
    const id = this.next++;
    this.agents.push({ id, kind });
    return id;
  }

  join(a: number, as: PortSlot, b: number, bs: PortSlot): void {
    this.wires.push({ a: { id: a, slot: as }, b: { id: b, slot: bs } });
  }

  text(): string {
    return formatWires(this.agents, this.wires);
  }
}

function lambda(term: Term): string {
  const c = compile(term);
  return formatWires(c.net.agents, c.net.wires);
}

function plugAux(s: Sketch, id: number): void {
  s.join(id, 'l', s.add('era'), 'p');
  s.join(id, 'r', s.add('era'), 'p');
}

/** Isolated dressed redexes — the lab commute / annihilate, tiled. */
function dressedField(cols: number, rows: number, a: AgentKind, b: AgentKind): string {
  const s = new Sketch();
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const left = s.add(a);
      const right = s.add(b);
      s.join(left, 'p', right, 'p');
      plugAux(s, left);
      plugAux(s, right);
    }
  }
  return s.text();
}

function addOscillator(s: Sketch): void {
  const dup = s.add('dup');
  const con = s.add('con');
  s.join(dup, 'p', con, 'p');
  s.join(dup, 'l', s.add('era'), 'p');
  s.join(con, 'r', s.add('era'), 'p');
  s.join(dup, 'r', con, 'l');
}

function addLafont(s: Sketch): void {
  const dup = s.add('dup');
  const con = s.add('con');
  s.join(dup, 'p', con, 'p');
  s.join(dup, 'l', con, 'l');
  s.join(dup, 'r', con, 'r');
}

function repeat(n: number, add: (s: Sketch) => void): string {
  const s = new Sketch();
  for (let i = 0; i < n; i++) add(s);
  return s.text();
}

/** Con/dup checkerboard, vertical principals facing, horizontal aux rails. */
function checkerboard(w: number, h: number): string {
  const s = new Sketch();
  const ids: number[][] = [];
  for (let y = 0; y < h; y++) {
    const row: number[] = [];
    for (let x = 0; x < w; x++) {
      row.push(s.add((x + y) % 2 === 0 ? 'con' : 'dup'));
    }
    ids.push(row);
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x + 1 < w) s.join(ids[y]![x]!, 'r', ids[y]![x + 1]!, 'l');
      if (y + 1 < h) s.join(ids[y]![x]!, 'p', ids[y + 1]![x]!, 'p');
    }
  }
  return s.text();
}

const I = lam('x', v('x'));
const K = lam('x', lam('y', v('x')));
const S = lam('x', lam('y', lam('z', ap(v('x'), v('z'), app(v('y'), v('z'))))));
const TRUE = lam('t', lam('f', v('t')));
const FALSE = lam('t', lam('f', v('f')));
const NOT = lam('b', lam('t', lam('f', ap(v('b'), v('f'), v('t')))));
const SUCC = lam('n', lam('f', lam('x', app(v('f'), ap(v('n'), v('f'), v('x'))))));
const Y = lam(
  'f',
  ap(lam('x', app(v('f'), app(v('x'), v('x')))), lam('x', app(v('f'), app(v('x'), v('x'))))),
);
const OMEGA = ap(lam('x', app(v('x'), v('x'))), lam('x', app(v('x'), v('x'))));

export const CATALOG: Record<Exclude<GalleryTabId, 'yours'>, CatalogPiece[]> = {
  rules: [
    { name: 'era–era', text: '& * ~ *' },
    { name: 'annihilate con', text: '& (* *) ~ (* *)' },
    { name: 'annihilate dup', text: '& {* *} ~ {* *}' },
    { name: 'commute', text: '& (* *) ~ {* *}' },
    { name: 'erase con', text: '& * ~ (* *)' },
    { name: 'erase dup', text: '& * ~ {* *}' },
    { name: 'oscillator', text: '& {* a} ~ (a *)' },
  ],
  quines: [
    { name: 'Lafont', text: '& {a b} ~ (a b)' },
    { name: 'Lafont crossed', text: '& {a b} ~ (b a)' },
    { name: 'oscillator', text: '& {* a} ~ (a *)' },
  ],
  worms: [
    { name: 'cons worm', text: '& {* a} ~ (a b)' },
    { name: 'dup worm', text: '& {a b} ~ (b *)' },
  ],
  meshes: [
    { name: 'commute field', text: dressedField(3, 2, 'con', 'dup') },
    { name: 'annihilate field', text: dressedField(3, 2, 'con', 'con') },
    { name: 'oscillator field', text: repeat(4, addOscillator) },
    { name: 'Lafont field', text: repeat(4, addLafont) },
    { name: 'checkerboard', text: checkerboard(4, 3) },
  ],
  lambda: [
    { name: 'I', text: lambda(I) },
    { name: 'K', text: lambda(K) },
    { name: 'S', text: lambda(S) },
    { name: 'TRUE', text: lambda(TRUE) },
    { name: 'FALSE', text: lambda(FALSE) },
    { name: 'NOT', text: lambda(NOT) },
    { name: '0', text: lambda(church(0)) },
    { name: '1', text: lambda(church(1)) },
    { name: '2', text: lambda(church(2)) },
    { name: '3', text: lambda(church(3)) },
    { name: '4', text: lambda(church(4)) },
    { name: 'SUCC', text: lambda(SUCC) },
    { name: 'PLUS', text: lambda(PLUS) },
    { name: 'MULT', text: lambda(MULT) },
    { name: 'I 2', text: lambda(ap(I, church(2))) },
    { name: 'SUCC 1', text: lambda(ap(SUCC, church(1))) },
    { name: '2 + 2', text: lambda(ap(PLUS, church(2), church(2))) },
    { name: '2 × 1', text: lambda(ap(MULT, church(2), church(1))) },
    { name: 'Y', text: lambda(Y) },
    { name: 'Ω', text: lambda(OMEGA) },
  ],
};

export function catalogPieces(tab: GalleryTabId): CatalogPiece[] {
  if (tab === 'yours') return [];
  return CATALOG[tab];
}

export function galleryHint(tab: GalleryTabId): string {
  switch (tab) {
    case 'yours':
      return 'Saves the selection, or the whole dish. Click a piece to stamp it.';
    case 'rules':
      return 'The six interactions, plus a two-step loop. Click to stamp, then Play.';
    case 'quines':
      return 'Graphs that cycle under rewrite. Click to stamp, then Play.';
    case 'worms':
      return 'A commute packet with one open thread. Click to stamp, then Play.';
    case 'meshes':
      return 'Many active pairs at once. Click to stamp, then Play.';
    case 'lambda':
      return 'Combinators and numerals sit still; applied terms rewrite. Not sound for every term.';
  }
}
