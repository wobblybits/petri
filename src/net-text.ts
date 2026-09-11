import { portKey, portKeyAt, type AgentKind, type PortRef, type PortSlot } from './agents.ts';
import type { DesignSnapshot } from './net-edit.ts';
import type { Sim } from './sim.ts';

/**
 * HVM2 / hvm-core net IR (HigherOrderCO). A net is one root tree plus redexes:
 *
 *     TERM ::= * | (TERM TERM) | {TERM TERM} | name
 *     NET  ::= TERM | & TERM ~ TERM NET
 *
 * `*` is an eraser, `(a b)` a constructor (aux ports `l`, `r`), `{a b}` a
 * duplicator. A name is a wire and occurs twice in a closed net. `& A ~ B`
 * joins the two trees' principal ports (an active pair).
 *
 * Open nets may have unmatched names, whose ports stay free; a closed net
 * may omit the root, starting at `&`. Pose is not stored.
 */

type Term =
  | { tag: 'var'; name: string }
  | { tag: 'era' }
  | { tag: 'con'; l: Term; r: Term }
  | { tag: 'dup'; l: Term; r: Term };

type Redex = { a: Term; b: Term };

const GAP = 52;

function writeTerm(t: Term): string {
  if (t.tag === 'var') return t.name;
  if (t.tag === 'era') return '*';
  const inner = `${writeTerm(t.l)} ${writeTerm(t.r)}`;
  return t.tag === 'con' ? `(${inner})` : `{${inner}}`;
}

function writeNet(root: Term | null, redexes: Redex[]): string {
  if (!root && redexes.length === 0) return '';
  const pairs = redexes.map((e) => `& ${writeTerm(e.a)} ~ ${writeTerm(e.b)}`);
  if (!root) return pairs.join('\n');
  if (pairs.length === 0) return writeTerm(root);
  return `${writeTerm(root)}\n${pairs.map((p) => `  ${p}`).join('\n')}`;
}

const LETTERS = 'abcdefghijklmnopqrstuvwxyz';

function freshName(n: number): string {
  const k = n % 26;
  const rest = Math.floor(n / 26);
  return rest === 0 ? LETTERS[k]! : `${LETTERS[k]}${rest}`;
}

type GraphView = {
  agents: { id: number; kind: AgentKind }[];
  peer(id: number, slot: PortSlot): PortRef | null;
  isFree(id: number, slot: PortSlot): boolean;
};

function formatView(view: GraphView): string {
  const agents = [...view.agents].sort((a, b) => a.id - b.id);
  if (agents.length === 0) return '';
  const byId = new Map(agents.map((a) => [a.id, a]));

  const names = new Map<number, string>();
  let n = 0;
  const fresh = (): string => freshName(n++);
  const nameOf = (id: number, slot: PortSlot): string => {
    const k = portKeyAt(id, slot);
    const had = names.get(k);
    if (had) return had;
    const nm = fresh();
    names.set(k, nm);
    const peer = view.peer(id, slot);
    if (peer) names.set(portKey(peer), nm);
    return nm;
  };

  const consumed = new Set<number>();
  const treeOf = (id: number): Term => {
    consumed.add(id);
    const agent = byId.get(id);
    if (!agent || agent.kind === 'era') return { tag: 'era' };
    return { tag: agent.kind, l: child(id, 'l'), r: child(id, 'r') };
  };
  const child = (id: number, slot: PortSlot): Term => {
    const peer = view.peer(id, slot);
    if (!peer || !byId.has(peer.id)) return { tag: 'var', name: nameOf(id, slot) };
    if (peer.slot === 'p' && !consumed.has(peer.id)) return treeOf(peer.id);
    return { tag: 'var', name: nameOf(id, slot) };
  };

  const redexes: Redex[] = [];
  const seenPair = new Set<string>();
  for (const agent of agents) {
    const peer = view.peer(agent.id, 'p');
    if (!peer || peer.slot !== 'p' || !byId.has(peer.id)) continue;
    const key = agent.id < peer.id ? `${agent.id}:${peer.id}` : `${peer.id}:${agent.id}`;
    if (seenPair.has(key)) continue;
    seenPair.add(key);
    redexes.push({ a: treeOf(agent.id), b: treeOf(peer.id) });
  }

  let root: Term | null = null;
  for (const agent of agents) {
    if (consumed.has(agent.id)) continue;
    const t = treeOf(agent.id);
    if (view.isFree(agent.id, 'p')) {
      if (!root) root = t;
      else redexes.push({ a: t, b: { tag: 'var', name: fresh() } });
    } else {
      redexes.push({ a: { tag: 'var', name: nameOf(agent.id, 'p') }, b: t });
    }
  }

  return writeNet(root, redexes);
}

/** Encode an agent/wire list as an HVM2 net. Layout is discarded. */
export function formatWires(
  agents: { id: number; kind: AgentKind }[],
  wires: { a: PortRef; b: PortRef }[],
): string {
  const adj = new Map<number, PortRef>();
  for (const w of wires) {
    adj.set(portKey(w.a), w.b);
    adj.set(portKey(w.b), w.a);
  }
  return formatView({
    agents,
    peer: (id, slot) => adj.get(portKeyAt(id, slot)) ?? null,
    isFree: (id, slot) => !adj.has(portKeyAt(id, slot)),
  });
}

/**
 * Encode the live graph as an HVM2 net. Layout is discarded.
 * Pass `ids` to encode a subgraph; wires that leave the set become free ports.
 */
export function formatNet(sim: Sim, ids?: Iterable<number>): string {
  const filter = ids ? new Set(ids) : null;
  const agents = [...sim.agents.values()]
    .filter((a) => !filter || filter.has(a.id))
    .map((a) => ({ id: a.id, kind: a.kind }));
  const wires = [...sim.graph.wires.values()]
    .filter((w) => !filter || (filter.has(w.a.id) && filter.has(w.b.id)))
    .map((w) => ({ a: w.a, b: w.b }));
  return formatWires(agents, wires);
}

class ParseErr extends Error {}

class Parser {
  s: string;
  i = 0;
  constructor(s: string) {
    this.s = s;
  }
  peek(): string {
    return this.s[this.i] ?? '';
  }
  skip(): void {
    for (;;) {
      while (this.i < this.s.length && /\s/.test(this.s[this.i]!)) this.i++;
      if (this.s.startsWith('//', this.i)) {
        while (this.i < this.s.length && this.s[this.i] !== '\n') this.i++;
        continue;
      }
      return;
    }
  }
  eat(ch: string): boolean {
    this.skip();
    if (this.s.startsWith(ch, this.i)) {
      this.i += ch.length;
      return true;
    }
    return false;
  }
  expect(ch: string): void {
    if (!this.eat(ch)) throw new ParseErr(`expected ${ch}`);
  }
  ident(): string {
    this.skip();
    const m = this.s.slice(this.i).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (!m) throw new ParseErr('expected name');
    this.i += m[0].length;
    return m[0];
  }
  digits(): string | null {
    this.skip();
    const m = this.s.slice(this.i).match(/^\d+/);
    if (!m) return null;
    this.i += m[0].length;
    return m[0];
  }
  term(): Term {
    this.skip();
    const c = this.peek();
    if (c === '*') {
      this.i++;
      return { tag: 'era' };
    }
    if (c === '(') {
      this.i++;
      const l = this.term();
      const r = this.term();
      this.expect(')');
      return { tag: 'con', l, r };
    }
    if (c === '[') {
      this.i++;
      const l = this.term();
      const r = this.term();
      this.expect(']');
      return { tag: 'con', l, r };
    }
    if (c === '{') {
      this.i++;
      this.digits();
      const l = this.term();
      const r = this.term();
      this.expect('}');
      return { tag: 'dup', l, r };
    }
    if (c === '@' || c === '#' || c === '<' || c === '?' || c === '$') {
      throw new ParseErr('unsupported HVM node');
    }
    return { tag: 'var', name: this.ident() };
  }
  defName(): void {
    this.skip();
    if (!this.eat('@')) return;
    this.ident();
    this.expect('=');
  }
  net(): { root: Term | null; redexes: Redex[] } {
    this.defName();
    this.skip();
    let root: Term | null = null;
    if (this.peek() && this.peek() !== '&') root = this.term();
    const redexes: Redex[] = [];
    while (this.eat('&')) {
      this.eat('!');
      const a = this.term();
      this.expect('~');
      const b = this.term();
      redexes.push({ a, b });
    }
    this.skip();
    if (this.i < this.s.length) throw new ParseErr('trailing junk');
    return { root, redexes };
  }
}

function spawnPose(index: number): { x: number; y: number } {
  return { x: (index % 8) * GAP, y: Math.floor(index / 8) * GAP };
}

/** Parse an HVM2 net into a designer snapshot. Returns null if the string is not a net. */
export function parseNet(src: string): DesignSnapshot | null {
  const trimmed = src.trim();
  if (!trimmed) return { nextId: 1, agents: [], wires: [] };
  let parsed: { root: Term | null; redexes: Redex[] };
  try {
    parsed = new Parser(trimmed).net();
  } catch {
    return null;
  }

  const agents: DesignSnapshot['agents'] = [];
  const wires: DesignSnapshot['wires'] = [];
  const pending = new Map<string, PortRef | 'free'>();

  const spawn = (kind: AgentKind): number => {
    const id = agents.length + 1;
    const pose = spawnPose(id - 1);
    agents.push({ id, kind, x: pose.x, y: pose.y, heading: 0, extra: 1 });
    return id;
  };

  const intern = (name: string, port: PortRef | 'free'): void => {
    const prev = pending.get(name);
    if (prev === undefined) {
      pending.set(name, port);
      return;
    }
    pending.delete(name);
    if (prev === 'free' || port === 'free') return;
    wires.push({ a: prev, b: port });
  };

  const spawnTree = (t: Term): PortRef => {
    if (t.tag === 'var') throw new ParseErr('var is not a tree');
    if (t.tag === 'era') return { id: spawn('era'), slot: 'p' };
    const id = spawn(t.tag);
    link(t.l, { id, slot: 'l' });
    link(t.r, { id, slot: 'r' });
    return { id, slot: 'p' };
  };

  const link = (t: Term, port: PortRef): void => {
    if (t.tag === 'var') intern(t.name, port);
    else wires.push({ a: port, b: spawnTree(t) });
  };

  const addRoot = (t: Term): void => {
    if (t.tag === 'var') intern(t.name, 'free');
    else spawnTree(t);
  };

  const addRedex = (a: Term, b: Term): void => {
    if (a.tag === 'var' && b.tag === 'var') {
      intern(a.name, 'free');
      intern(b.name, 'free');
      return;
    }
    if (a.tag === 'var') intern(a.name, spawnTree(b));
    else if (b.tag === 'var') intern(b.name, spawnTree(a));
    else wires.push({ a: spawnTree(a), b: spawnTree(b) });
  };

  if (parsed.root) addRoot(parsed.root);
  for (const e of parsed.redexes) addRedex(e.a, e.b);

  return { nextId: agents.length + 1, agents, wires };
}

/** True when `src` looks like an HVM2 net rather than ordinary clipboard text. */
export function looksLikeNet(src: string): boolean {
  const t = src.trim();
  if (!t) return false;
  if (t.startsWith('*') || t.startsWith('(') || t.startsWith('{') || t.startsWith('[') || t.startsWith('&')) {
    return true;
  }
  if (t.startsWith('@') && t.includes('=')) return true;
  return /^[A-Za-z_][A-Za-z0-9_]*\s*&/.test(t);
}
