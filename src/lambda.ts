import {
  createAgent,
  portKey,
  portLocal,
  slotsFor,
  stemRoot,
  type Agent,
  type AgentKind,
  type PortRef,
  type PortSlot,
} from './agents.ts';
import { closestOnSegments, segmentsIntersect } from './geom.ts';
import type { Graph } from './graph.ts';
import type { Params } from './params.ts';
import {
  applyRewrite,
  detectRule,
  snapshotOf,
  type NetSnapshot,
  type NetWire,
} from './rewrite.ts';
import { wrapAngle } from './wrap.ts';

/**
 * Lambda terms compiled to interaction nets, using the three combinators this
 * simulation already runs: Con is γ, Dup is δ, Era is ε.
 *
 * Port convention, forced by the rules in rewrite.ts rather than chosen:
 *
 *   abstraction λx.b   Con: p = its value, l = the binder x, r = the body
 *   application (f a)  Con: p = the function, l = the result, r = the argument
 *
 * Beta reduction is then exactly `annihilate-con`. Note that rule reconnects
 * *crossed* — `l1↔r2, r1↔l2` — which is why an application keeps its result on
 * `l` and its argument on `r`. Straight through, an application would hand its
 * argument to the body and its result to the binder, which is nonsense.
 *
 * A variable used more than once becomes a Dup tree; an unused one gets an Era.
 * This is the standard encoding and it is *not* sound for every lambda term —
 * general duplication of higher-order terms needs bookkeeping this does not
 * have. It is sound for the arithmetic here, and `normalize` reports when a
 * term fails to reach a normal form rather than pretending it did.
 */

export type Term =
  | { tag: 'var'; name: string }
  | { tag: 'lam'; param: string; body: Term }
  | { tag: 'app'; fn: Term; arg: Term };

export const v = (name: string): Term => ({ tag: 'var', name });
export const lam = (param: string, body: Term): Term => ({ tag: 'lam', param, body });
export const app = (fn: Term, arg: Term): Term => ({ tag: 'app', fn, arg });

/** Left-associated application, so `ap(m, f, x)` is `((m f) x)`. */
export function ap(...terms: Term[]): Term {
  return terms.reduce((f, a) => app(f, a));
}

/** Church numeral: λf.λx. f applied n times to x. */
export function church(n: number): Term {
  let body: Term = v('x');
  for (let i = 0; i < n; i++) body = app(v('f'), body);
  return lam('f', lam('x', body));
}

export const PLUS = lam(
  'm',
  lam('n', lam('f', lam('x', ap(v('m'), v('f'), ap(v('n'), v('f'), v('x')))))),
);

export const MULT = lam('m', lam('n', lam('f', ap(v('m'), app(v('n'), v('f'))))));

// ------------------------------------------------------------------ compiling

/** One end of a link: a real port, or a named wire waiting to be resolved. */
type End = { port: PortRef } | { name: string };

interface Build {
  agents: { id: number; kind: AgentKind }[];
  links: [End, End][];
  /** Where each named wire is bound (the λ's binder port). */
  bind: Map<string, PortRef>;
  next: number;
}

const isPort = (e: End): e is { port: PortRef } => 'port' in e;

function node(b: Build, kind: AgentKind): number {
  const id = b.next++;
  b.agents.push({ id, kind });
  return id;
}

function port(id: number, slot: PortSlot): PortRef {
  return { id, slot };
}

export interface Compiled {
  net: NetSnapshot;
  /**
   * Aux port of an inert marker node wired to the term's value. The value port
   * itself cannot be the handle: rewrites consume the nodes it sits on, and a
   * *free* port is not propagated by the rules — `leftoverOf` returns null for
   * it and the connection is simply dropped. Anchoring to a marker's aux port
   * keeps a stable handle, and because the marker's principal stays free it can
   * never be half of a redex, so it never reduces.
   */
  root: PortRef;
  nextId: number;
}

/**
 * Compile a closed term. Free variables are an error rather than a silent
 * dangling wire, since a dangling wire would latch onto whatever is nearby.
 */
export function compile(term: Term, startId = 1): Compiled {
  const b: Build = { agents: [], links: [], bind: new Map(), next: startId };
  let fresh = 0;

  const go = (t: Term, env: Map<string, string>): End => {
    if (t.tag === 'var') {
      const wire = env.get(t.name);
      if (!wire) throw new Error(`unbound variable: ${t.name}`);
      return { name: wire };
    }
    if (t.tag === 'lam') {
      const id = node(b, 'con');
      const wire = `${t.param}#${fresh++}`;
      b.bind.set(wire, port(id, 'l'));
      const inner = new Map(env);
      inner.set(t.param, wire);
      b.links.push([{ port: port(id, 'r') }, go(t.body, inner)]);
      return { port: port(id, 'p') };
    }
    const id = node(b, 'con');
    b.links.push([{ port: port(id, 'p') }, go(t.fn, env)]);
    b.links.push([{ port: port(id, 'r') }, go(t.arg, env)]);
    return { port: port(id, 'l') };
  };

  const marker = node(b, 'con');
  const rootEnd = go(term, new Map());
  b.links.push([{ port: port(marker, 'l') }, rootEnd]);

  // Resolve named wires. Each name has one binder and zero or more uses: none
  // needs an Era, one is a plain wire, more than one needs a Dup tree.
  const uses = new Map<string, PortRef[]>();
  const wires: NetWire[] = [];
  for (const [x, y] of b.links) {
    if (isPort(x) && isPort(y)) {
      wires.push({ a: x.port, b: y.port });
      continue;
    }
    const named = isPort(x) ? y : x;
    const real = isPort(x) ? x : y;
    if (!isPort(real)) throw new Error('two named ends linked together');
    const list = uses.get((named as { name: string }).name) ?? [];
    list.push(real.port);
    uses.set((named as { name: string }).name, list);
  }

  for (const [name, binder] of b.bind) {
    const sites = uses.get(name) ?? [];
    if (sites.length === 0) {
      const e = node(b, 'era');
      wires.push({ a: binder, b: port(e, 'p') });
      continue;
    }
    wires.push({ a: binder, b: fanOut(b, sites, wires) });
  }

  return { net: { agents: b.agents, wires }, root: port(marker, 'l'), nextId: b.next };
}

/**
 * Wire one source to many uses through a tree of Dup nodes, and return the port
 * the source should attach to. Dup's principal faces the binder, so a copy
 * travelling down the tree meets each use head-on.
 */
function fanOut(b: Build, sites: PortRef[], wires: NetWire[]): PortRef {
  if (sites.length === 1) return sites[0];
  const id = node(b, 'dup');
  const half = Math.ceil(sites.length / 2);
  wires.push({ a: port(id, 'l'), b: fanOut(b, sites.slice(0, half), wires) });
  wires.push({ a: port(id, 'r'), b: fanOut(b, sites.slice(half), wires) });
  return port(id, 'p');
}

// ------------------------------------------------------------------ reducing

export function follow(net: NetSnapshot, p: PortRef): PortRef | null {
  for (const w of net.wires) {
    if (w.a.id === p.id && w.a.slot === p.slot) return w.b;
    if (w.b.id === p.id && w.b.slot === p.slot) return w.a;
  }
  return null;
}

/** A redex is two principal ports facing each other — nothing more. */
export function findRedex(net: NetSnapshot): { a: number; b: number } | null {
  for (const w of net.wires) {
    if (w.a.slot === 'p' && w.b.slot === 'p' && w.a.id !== w.b.id) {
      return { a: w.a.id, b: w.b.id };
    }
  }
  return null;
}

export interface Normalized {
  net: NetSnapshot;
  steps: number;
  /** False when the step budget ran out — the term may not have a normal form. */
  done: boolean;
}

export function normalize(start: Compiled, maxSteps = 20000): Normalized {
  let net = start.net;
  let nextId = start.nextId;
  let steps = 0;
  const kindOf = (n: NetSnapshot, id: number): AgentKind | undefined =>
    n.agents.find((a) => a.id === id)?.kind;
  for (; steps < maxSteps; steps++) {
    const redex = findRedex(net);
    if (!redex) return { net, steps, done: true };
    const ka = kindOf(net, redex.a);
    const kb = kindOf(net, redex.b);
    if (!ka || !kb) return { net, steps, done: true };
    const out = applyRewrite(net, detectRule(ka, kb), redex.a, redex.b, nextId);
    net = out.net;
    nextId = out.nextId;
  }
  return { net, steps, done: false };
}

// ------------------------------------------------------------------ decoding

/**
 * Read a normal form back as a Church numeral, or null if it is not one.
 *
 * Walks λf.λx. then counts applications down the spine, checking each one is
 * applied to f and that the spine ends at x.
 */
export function decodeChurch(net: NetSnapshot, root: PortRef): number | null {
  const kind = (id: number): AgentKind | undefined =>
    net.agents.find((a) => a.id === id)?.kind;

  const outer = follow(net, root);
  if (!outer || outer.slot !== 'p' || kind(outer.id) !== 'con') return null;
  const inner = follow(net, port(outer.id, 'r'));
  if (!inner || inner.slot !== 'p' || kind(inner.id) !== 'con') return null;

  const fBinder = port(outer.id, 'l');
  const xBinder = port(inner.id, 'l');

  // Which ports can legitimately supply f to an application. That is the binder
  // itself when f is used once, or the aux ports of the Dup tree fanning out of
  // it when f is used several times. An application reaches its function by
  // following its own principal, so this is the set that end must land in.
  const supply = new Set<string>();
  const visit = (from: PortRef): void => {
    const key = `${from.id}.${from.slot}`;
    if (supply.has(key)) return;
    supply.add(key);
    const onward = follow(net, from);
    if (onward && onward.slot === 'p' && kind(onward.id) === 'dup') {
      visit(port(onward.id, 'l'));
      visit(port(onward.id, 'r'));
    }
  };
  visit(fBinder);

  let n = 0;
  let cur = follow(net, port(inner.id, 'r'));
  const guard = net.agents.length + 4;
  while (n <= guard) {
    if (!cur) return null;
    // Reached the x binder: the spine is finished.
    if (cur.id === xBinder.id && cur.slot === xBinder.slot) return n;
    if (cur.slot !== 'l' || kind(cur.id) !== 'con') return null;
    // This application must be applying f.
    const fn = follow(net, port(cur.id, 'p'));
    if (!fn || !supply.has(`${fn.id}.${fn.slot}`)) return null;
    n++;
    cur = follow(net, port(cur.id, 'r'));
  }
  return null;
}

/** Compile, reduce, and read the result as a number. */
export function evalChurch(term: Term, maxSteps = 20000): { value: number | null; steps: number; done: boolean } {
  const built = compile(term);
  const out = normalize(built, maxSteps);
  return { value: decodeChurch(out.net, built.root), steps: out.steps, done: out.done };
}

// ------------------------------------------------------------------ layout

export interface NetPose {
  id: number;
  x: number;
  y: number;
  heading: number;
}

const GAP_X = 56;
const GAP_Y = 62;
const MIN_DIST = 44;

type XY = { x: number; y: number };
type Adj = Map<number, Map<PortSlot, PortRef>>;

interface TermNode {
  id: number;
  incoming: PortSlot;
  kids: TermNode[];
}

/**
 * Straight-line drawing of a compiled net.
 *
 * Abstractions and applications form a tree (body / function / argument). That
 * tree is packed with a tidy downward layout, which does not cross itself.
 * Dup/Era trees hang off binders and sit in a side gutter aligned with their
 * use sites, so the extra binder wires run beside the tree rather than through
 * it. A short untangle pass then walks any leftover crossings off the drawing.
 */
export function layoutNet(net: NetSnapshot, root: PortRef, cx: number, cy: number): NetPose[] {
  const kindOf = new Map(net.agents.map((a) => [a.id, a.kind]));
  const adj = adjacency(net);
  const seen = new Set<number>();
  const tree = termTreeFrom(root, adj, kindOf, seen);
  const leftover = net.agents.map((a) => a.id).filter((id) => !seen.has(id));
  const comps = components(leftover, adj);
  const flipped = reverseKids(tree);

  const raw = [
    withLeftovers(placeTidy(tree), comps, adj, kindOf, net.wires),
    withLeftovers(placeTidy(flipped), comps, adj, kindOf, net.wires),
    withLeftovers(placeIndent(tree), comps, adj, kindOf, net.wires),
  ];
  for (const cand of raw) untangle(cand, net.wires);

  let pos = raw[0];
  let best = countXY(pos, net.wires);
  for (let i = 1; i < raw.length; i++) {
    const n = countXY(raw[i], net.wires);
    if (n < best) {
      best = n;
      pos = raw[i];
    }
  }

  untangle(pos, net.wires);
  separate(pos, net.wires);
  if (countXY(pos, net.wires) > 0) untangle(pos, net.wires);
  compact(pos, net.wires);
  separate(pos, net.wires);
  if (countXY(pos, net.wires) > 0) {
    untangle(pos, net.wires);
    nudge(pos, leftover, net.wires);
    if (countXY(pos, net.wires) > 0) nudgeHits(pos, net.wires);
  }
  const poses = posesOf(pos, kindOf, adj);
  centerPoses(poses, cx, cy);
  return poses;
}

/** Proper crossings of agent-center segments. Wires that share an agent do not count. */
export function layoutCrossings(net: NetSnapshot, poses: NetPose[]): number {
  const byId = new Map(poses.map((p) => [p.id, p]));
  const segs: { ax: number; ay: number; bx: number; by: number; a: number; b: number }[] = [];
  for (const w of net.wires) {
    if (w.a.id === w.b.id) continue;
    const A = byId.get(w.a.id);
    const B = byId.get(w.b.id);
    if (!A || !B) continue;
    segs.push({ ax: A.x, ay: A.y, bx: B.x, by: B.y, a: w.a.id, b: w.b.id });
  }
  let n = 0;
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const u = segs[i];
      const v = segs[j];
      if (u.a === v.a || u.a === v.b || u.b === v.a || u.b === v.b) continue;
      if (properCross(u.ax, u.ay, u.bx, u.by, v.ax, v.ay, v.bx, v.by)) n++;
    }
  }
  return n;
}

function adjacency(net: NetSnapshot): Adj {
  const adj: Adj = new Map();
  for (const a of net.agents) adj.set(a.id, new Map());
  for (const w of net.wires) {
    adj.get(w.a.id)!.set(w.a.slot, w.b);
    adj.get(w.b.id)!.set(w.b.slot, w.a);
  }
  return adj;
}

function termTreeFrom(root: PortRef, adj: Adj, kindOf: Map<number, AgentKind>, seen: Set<number>): TermNode {
  seen.add(root.id);
  const kids: TermNode[] = [];
  const to = adj.get(root.id)?.get(root.slot);
  if (to && kindOf.get(to.id) === 'con' && !seen.has(to.id)) {
    kids.push(buildTerm(to.id, to.slot, adj, kindOf, seen));
  }
  return { id: root.id, incoming: 'p', kids };
}

function buildTerm(
  id: number,
  incoming: PortSlot,
  adj: Adj,
  kindOf: Map<number, AgentKind>,
  seen: Set<number>,
): TermNode {
  seen.add(id);
  const kids: TermNode[] = [];
  const tryKid = (slot: PortSlot) => {
    const to = adj.get(id)?.get(slot);
    if (!to || seen.has(to.id) || kindOf.get(to.id) !== 'con') return;
    kids.push(buildTerm(to.id, to.slot, adj, kindOf, seen));
  };
  if (incoming === 'p') tryKid('r');
  else if (incoming === 'l') {
    tryKid('p');
    tryKid('r');
  } else {
    tryKid('p');
    tryKid('l');
  }
  return { id, incoming, kids };
}

function subtreeSpan(n: TermNode): number {
  if (n.kids.length === 0) return 1;
  return n.kids.reduce((s, k) => s + subtreeSpan(k), 0);
}

function placeTidy(tree: TermNode): Map<number, XY> {
  const pos = new Map<number, XY>();
  const walk = (n: TermNode, x: number, y: number) => {
    pos.set(n.id, { x, y });
    if (n.kids.length === 0) return;
    const spans = n.kids.map(subtreeSpan);
    const total = spans.reduce((a, b) => a + b, 0);
    let left = x - (total * GAP_X) / 2;
    for (let i = 0; i < n.kids.length; i++) {
      const w = spans[i] * GAP_X;
      walk(n.kids[i], left + w / 2, y + GAP_Y);
      left += w;
    }
  };
  walk(tree, 0, 0);
  return pos;
}

function placeIndent(tree: TermNode): Map<number, XY> {
  const pos = new Map<number, XY>();
  let row = 0;
  const walk = (n: TermNode, depth: number) => {
    pos.set(n.id, { x: depth * GAP_X, y: row * GAP_Y });
    row++;
    for (const k of n.kids) walk(k, depth + 1);
  };
  walk(tree, 0);
  return pos;
}

function reverseKids(n: TermNode): TermNode {
  return { id: n.id, incoming: n.incoming, kids: n.kids.map(reverseKids).reverse() };
}

function components(ids: number[], adj: Adj): number[][] {
  const leftover = new Set(ids);
  const seen = new Set<number>();
  const out: number[][] = [];
  for (const start of ids) {
    if (seen.has(start)) continue;
    const comp: number[] = [];
    const stack = [start];
    seen.add(start);
    while (stack.length) {
      const u = stack.pop()!;
      comp.push(u);
      for (const to of adj.get(u)?.values() ?? []) {
        if (!leftover.has(to.id) || seen.has(to.id)) continue;
        seen.add(to.id);
        stack.push(to.id);
      }
    }
    out.push(comp);
  }
  return out;
}

function withLeftovers(
  base: Map<number, XY>,
  comps: number[][],
  adj: Adj,
  kindOf: Map<number, AgentKind>,
  wires: NetWire[],
): Map<number, XY> {
  const pos = clonePos(base);
  for (const comp of comps) {
    const left = clonePos(pos);
    const right = clonePos(pos);
    placeDupTree(comp, left, adj, kindOf, -1);
    placeDupTree(comp, right, adj, kindOf, 1);
    const keep = countXY(left, wires) <= countXY(right, wires) ? left : right;
    pos.clear();
    for (const [id, p] of keep) pos.set(id, p);
  }
  return pos;
}

function placeDupTree(
  comp: number[],
  pos: Map<number, XY>,
  adj: Adj,
  kindOf: Map<number, AgentKind>,
  side: number,
): void {
  const inComp = new Set(comp);
  let root = comp[0];
  for (const id of comp) {
    const prin = adj.get(id)?.get('p');
    if (prin && pos.has(prin.id) && kindOf.get(prin.id) === 'con') root = id;
  }

  const placedOthers = () => [...pos.entries()].filter(([id]) => !inComp.has(id));

  const place = (id: number): XY => {
    const l = adj.get(id)?.get('l');
    const r = adj.get(id)?.get('r');
    if (l && inComp.has(l.id)) place(l.id);
    if (r && inComp.has(r.id)) place(r.id);
    if (!l && !r) {
      const prin = adj.get(id)?.get('p');
      const b = prin ? pos.get(prin.id) : undefined;
      const p = { x: (b?.x ?? 0) + side * GAP_X, y: (b?.y ?? 0) + GAP_Y * 0.35 };
      pos.set(id, p);
      return p;
    }
    const pts: XY[] = [];
    for (const to of [l, r]) {
      if (!to) continue;
      const p = pos.get(to.id);
      if (p) pts.push(p);
    }
    const c = pts.length ? centroid(pts) : { x: 0, y: 0 };
    const prin = adj.get(id)?.get('p');
    const binder = prin ? pos.get(prin.id) : undefined;
    const mix = binder
      ? { x: binder.x * 0.35 + c.x * 0.65, y: binder.y * 0.35 + c.y * 0.65 }
      : c;
    let dx = 0;
    let dy = 1;
    if (pts.length >= 2) {
      dx = pts[pts.length - 1].x - pts[0].x;
      dy = pts[pts.length - 1].y - pts[0].y;
    } else if (binder) {
      dx = c.x - binder.x;
      dy = c.y - binder.y;
    }
    if (Math.hypot(dx, dy) < 1e-6) {
      dx = 0;
      dy = 1;
    }
    const len = Math.hypot(dx, dy);
    const px = (-dy / len) * side;
    const py = (dx / len) * side;
    let x = mix.x + px * 16;
    let y = mix.y + py * 16;
    const others = placedOthers();
    for (let k = 0; k < 10; k++) {
      let minD = Infinity;
      for (const [, p] of others) minD = Math.min(minD, Math.hypot(x - p.x, y - p.y));
      if (minD >= MIN_DIST) break;
      x += px * 10;
      y += py * 10;
    }
    const here = { x, y };
    pos.set(id, here);
    return here;
  };
  place(root);
}

function centroid(pts: XY[]): XY {
  let x = 0;
  let y = 0;
  for (const p of pts) {
    x += p.x;
    y += p.y;
  }
  return { x: x / pts.length, y: y / pts.length };
}

function clonePos(pos: Map<number, XY>): Map<number, XY> {
  const out = new Map<number, XY>();
  for (const [id, p] of pos) out.set(id, { x: p.x, y: p.y });
  return out;
}

/** Proper crossing, ignoring grazes that land on a vertex (T-junctions at a node). */
function properCross(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
  dx: number,
  dy: number,
): { ix: number; iy: number } | null {
  if (!segmentsIntersect(ax, ay, bx, by, cx, cy, dx, dy)) return null;
  const hit = closestOnSegments(ax, ay, bx, by, cx, cy, dx, dy);
  const ix = (hit.ax + hit.bx) * 0.5;
  const iy = (hit.ay + hit.by) * 0.5;
  if (
    Math.hypot(ix - ax, iy - ay) < 10 ||
    Math.hypot(ix - bx, iy - by) < 10 ||
    Math.hypot(ix - cx, iy - cy) < 10 ||
    Math.hypot(ix - dx, iy - dy) < 10
  ) {
    return null;
  }
  return { ix, iy };
}

function countXY(pos: Map<number, XY>, wires: NetWire[]): number {
  const segs: { ax: number; ay: number; bx: number; by: number; a: number; b: number }[] = [];
  for (const w of wires) {
    if (w.a.id === w.b.id) continue;
    const A = pos.get(w.a.id);
    const B = pos.get(w.b.id);
    if (!A || !B) continue;
    segs.push({ ax: A.x, ay: A.y, bx: B.x, by: B.y, a: w.a.id, b: w.b.id });
  }
  let n = 0;
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const u = segs[i];
      const v = segs[j];
      if (u.a === v.a || u.a === v.b || u.b === v.a || u.b === v.b) continue;
      if (properCross(u.ax, u.ay, u.bx, u.by, v.ax, v.ay, v.bx, v.by)) n++;
    }
  }
  return n;
}

function listCrossings(
  pos: Map<number, XY>,
  wires: NetWire[],
): { a: number; b: number; c: number; d: number; ix: number; iy: number }[] {
  const segs: { ax: number; ay: number; bx: number; by: number; a: number; b: number }[] = [];
  for (const w of wires) {
    if (w.a.id === w.b.id) continue;
    const A = pos.get(w.a.id);
    const B = pos.get(w.b.id);
    if (!A || !B) continue;
    segs.push({ ax: A.x, ay: A.y, bx: B.x, by: B.y, a: w.a.id, b: w.b.id });
  }
  const hits: { a: number; b: number; c: number; d: number; ix: number; iy: number }[] = [];
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const u = segs[i];
      const v = segs[j];
      if (u.a === v.a || u.a === v.b || u.b === v.a || u.b === v.b) continue;
      const hit = properCross(u.ax, u.ay, u.bx, u.by, v.ax, v.ay, v.bx, v.by);
      if (!hit) continue;
      hits.push({ a: u.a, b: u.b, c: v.a, d: v.b, ix: hit.ix, iy: hit.iy });
    }
  }
  return hits;
}

function reflectOver(p: XY, ax: number, ay: number, bx: number, by: number): XY {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy || 1;
  const t = ((p.x - ax) * dx + (p.y - ay) * dy) / len2;
  const qx = ax + t * dx;
  const qy = ay + t * dy;
  return { x: 2 * qx - p.x, y: 2 * qy - p.y };
}

function untangle(pos: Map<number, XY>, wires: NetWire[]): void {
  for (let iter = 0; iter < 64; iter++) {
    const hits = listCrossings(pos, wires);
    if (hits.length === 0) return;
    const before = hits.length;
    let bestDrop = 0;
    let bestId = -1;
    let bestXY: XY | null = null;
    for (const hit of hits) {
      const ids = [hit.a, hit.b, hit.c, hit.d];
      const other = (id: number): { ax: number; ay: number; bx: number; by: number } => {
        if (id === hit.a || id === hit.b) {
          const C = pos.get(hit.c)!;
          const D = pos.get(hit.d)!;
          return { ax: C.x, ay: C.y, bx: D.x, by: D.y };
        }
        const A = pos.get(hit.a)!;
        const B = pos.get(hit.b)!;
        return { ax: A.x, ay: A.y, bx: B.x, by: B.y };
      };
      for (const id of ids) {
        const p = pos.get(id)!;
        const seg = other(id);
        const flipped = reflectOver(p, seg.ax, seg.ay, seg.bx, seg.by);
        const dx = p.x - hit.ix;
        const dy = p.y - hit.iy;
        const variants: XY[] = [
          flipped,
          { x: flipped.x + (flipped.x - p.x) * 0.25, y: flipped.y + (flipped.y - p.y) * 0.25 },
          { x: hit.ix - dy, y: hit.iy + dx },
          { x: hit.ix + dy, y: hit.iy - dx },
          { x: p.x + (p.x - hit.ix) * 0.6, y: p.y + (p.y - hit.iy) * 0.6 },
        ];
        for (const q of variants) {
          if (!Number.isFinite(q.x) || !Number.isFinite(q.y)) continue;
          pos.set(id, q);
          const drop = before - countXY(pos, wires);
          pos.set(id, p);
          if (drop > bestDrop) {
            bestDrop = drop;
            bestId = id;
            bestXY = q;
          }
        }
      }
    }
    if (bestXY && bestId >= 0) pos.set(bestId, bestXY);
    else return;
  }
}

function nudge(pos: Map<number, XY>, leftover: number[], wires: NetWire[]): void {
  const steps = [0, GAP_X, -GAP_X, GAP_X * 1.6, -GAP_X * 1.6, GAP_X * 0.5, -GAP_X * 0.5];
  for (const id of leftover) {
    const orig = pos.get(id);
    if (!orig) continue;
    let best = countXY(pos, wires);
    if (best === 0) return;
    let bestXY = orig;
    for (const dx of steps) {
      for (const dy of steps) {
        if (dx === 0 && dy === 0) continue;
        pos.set(id, { x: orig.x + dx, y: orig.y + dy });
        const n = countXY(pos, wires);
        if (n < best) {
          best = n;
          bestXY = { x: orig.x + dx, y: orig.y + dy };
        }
      }
    }
    pos.set(id, bestXY);
  }
}

function nudgeHits(pos: Map<number, XY>, wires: NetWire[]): void {
  const steps = [0, 28, -28, 48, -48, 72, -72];
  for (let round = 0; round < 4; round++) {
    const hits = listCrossings(pos, wires);
    if (hits.length === 0) return;
    const ids = new Set<number>();
    for (const h of hits) {
      ids.add(h.a);
      ids.add(h.b);
      ids.add(h.c);
      ids.add(h.d);
    }
    let improved = false;
    for (const id of ids) {
      const orig = pos.get(id)!;
      let best = countXY(pos, wires);
      let bestXY = orig;
      for (const dx of steps) {
        for (const dy of steps) {
          if (dx === 0 && dy === 0) continue;
          pos.set(id, { x: orig.x + dx, y: orig.y + dy });
          const n = countXY(pos, wires);
          if (n < best) {
            best = n;
            bestXY = { x: orig.x + dx, y: orig.y + dy };
          }
        }
      }
      pos.set(id, bestXY);
      if (bestXY !== orig) improved = true;
    }
    if (!improved) return;
  }
}

function compact(pos: Map<number, XY>, wires: NetWire[]): void {
  const lengths: number[] = [];
  for (const w of wires) {
    if (w.a.id === w.b.id) continue;
    const A = pos.get(w.a.id);
    const B = pos.get(w.b.id);
    if (!A || !B) continue;
    lengths.push(Math.hypot(A.x - B.x, A.y - B.y));
  }
  if (lengths.length === 0) return;
  lengths.sort((a, b) => a - b);
  const med = lengths[lengths.length >> 1];
  if (med < 1) return;
  const s = Math.min(1.1, Math.max(0.72, 56 / med));
  if (Math.abs(s - 1) < 0.05) return;
  const c = centroid([...pos.values()]);
  for (const p of pos.values()) {
    p.x = c.x + (p.x - c.x) * s;
    p.y = c.y + (p.y - c.y) * s;
  }
}

function separate(pos: Map<number, XY>, wires?: NetWire[]): void {
  const before = wires ? clonePos(pos) : null;
  const crossed = wires ? countXY(pos, wires) : 0;
  const ids = [...pos.keys()];
  for (let pass = 0; pass < 8; pass++) {
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const a = pos.get(ids[i])!;
        const b = pos.get(ids[j])!;
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let d = Math.hypot(dx, dy);
        if (d >= MIN_DIST) continue;
        if (d < 1e-6) {
          dx = 1;
          dy = 0;
          d = 1;
        }
        const push = (MIN_DIST - d) / 2;
        const ux = dx / d;
        const uy = dy / d;
        a.x += ux * push;
        a.y += uy * push;
        b.x -= ux * push;
        b.y -= uy * push;
      }
    }
  }
  if (before && wires && countXY(pos, wires) > crossed) {
    pos.clear();
    for (const [id, p] of before) pos.set(id, p);
  }
}

function headingForPort(kind: AgentKind, slot: PortSlot, target: number): number {
  const tip = portLocal(kind, slot);
  const root = stemRoot(kind, slot);
  return wrapAngle(target - Math.atan2(tip.y - root.y, tip.x - root.x));
}

function posesOf(pos: Map<number, XY>, kindOf: Map<number, AgentKind>, adj: Adj): NetPose[] {
  const out: NetPose[] = [];
  for (const [id, p] of pos) {
    const kind = kindOf.get(id)!;
    const ports = adj.get(id) ?? new Map();
    const principal = ports.get('p');
    let heading = -Math.PI / 2;
    if (principal) {
      const o = pos.get(principal.id);
      if (o) heading = headingForPort(kind, 'p', Math.atan2(o.y - p.y, o.x - p.x));
    } else {
      const wired = [...ports.entries()][0];
      if (wired) {
        const o = pos.get(wired[1].id);
        if (o) heading = headingForPort(kind, wired[0], Math.atan2(o.y - p.y, o.x - p.x));
      }
    }
    out.push({ id, x: p.x, y: p.y, heading });
  }
  return out;
}

function centerPoses(poses: NetPose[], cx: number, cy: number): void {
  if (poses.length === 0) return;
  let x = 0;
  let y = 0;
  for (const p of poses) {
    x += p.x;
    y += p.y;
  }
  x /= poses.length;
  y /= poses.length;
  for (const p of poses) {
    p.x += cx - x;
    p.y += cy - y;
  }
}

// ------------------------------------------------------------------ injecting

/**
 * Drop a compiled term into a running simulation.
 *
 * Agents land in a planar drawing of the net — the term tree packed downward,
 * sharing nodes in a gutter beside their uses — so wires start uncrossed and
 * the joint solver only has to settle port facing. Every port the term does
 * not use is sealed, because a free port would otherwise latch onto whatever
 * drifts past and quietly turn the term into something else.
 */
export function injectTerm(
  sim: {
    agents: Map<number, Agent>;
    graph: Graph;
    nextId: number;
    w: number;
    h: number;
    wire: (aId: number, aSlot: PortSlot, bId: number, bSlot: PortSlot, p: Params) => void;
    noteRosterChange: () => void;
  },
  term: Term,
  cx: number,
  cy: number,
  params: Params,
): { root: PortRef; ids: number[] } {
  const built = compile(term, sim.nextId);
  const poses = layoutNet(built.net, built.root, cx, cy);
  const byId = new Map(poses.map((p) => [p.id, p]));

  for (const spec of built.net.agents) {
    const pose = byId.get(spec.id);
    const agent = createAgent(
      spec.id,
      spec.kind,
      pose?.x ?? cx,
      pose?.y ?? cy,
      pose?.heading ?? 0,
      params,
    );
    sim.agents.set(agent.id, agent);
  }
  // Written straight into the Map, so the roster caches need telling.
  sim.noteRosterChange();
  sim.nextId = built.nextId;

  for (const w of built.net.wires) {
    sim.wire(w.a.id, w.a.slot, w.b.id, w.b.slot, params);
  }

  for (const spec of built.net.agents) {
    for (const slot of slotsFor(spec.kind)) {
      const ref = { id: spec.id, slot };
      if (sim.graph.isFree(ref)) sim.graph.sealed.add(portKey(ref));
    }
  }

  return { root: built.root, ids: built.net.agents.map((a) => a.id) };
}

/** Read the current state of an injected term straight out of the live graph. */
export function readChurch(
  agents: Map<number, Agent>,
  graph: Graph,
  root: PortRef,
): number | null {
  return decodeChurch(snapshotOf(agents, graph), root);
}
