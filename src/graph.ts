import {
  inSnapArc,
  portKey,
  portWorld,
  slotsFor,
  stemWorld,
  wireCubic,
  type Agent,
  type PortRef,
} from './agents.ts';
import {
  desiredLinks,
  polylineLength,
  reduceChain,
  sampleChain,
  solveChain,
  unwrapPoints,
  type ChainNode,
} from './chain.ts';
import { bezierLength } from './curve.ts';
import { segmentsInterfere, WIRE_RADIUS } from './geom.ts';
import type { Params } from './params.ts';
import { clamp, easeInOut, lerp, wrap, wrapDeltaVec } from './wrap.ts';

export interface Wire {
  id: number;
  a: PortRef;
  b: PortRef;
  latchLen: number;
  lastLen: number;
  rest: number;
  born: number;
  nodes: ChainNode[];
}

export function otherEnd(wire: Wire, port: PortRef): PortRef {
  if (wire.a.id === port.id && wire.a.slot === port.slot) return wire.b;
  return wire.a;
}

export class Graph {
  wires = new Map<number, Wire>();
  portWire = new Map<string, number>();
  nextWireId = 1;

  clear(): void {
    this.wires.clear();
    this.portWire.clear();
  }

  wireAt(port: PortRef): Wire | undefined {
    const id = this.portWire.get(portKey(port));
    if (id === undefined) return undefined;
    return this.wires.get(id);
  }

  isFree(port: PortRef): boolean {
    return !this.portWire.has(portKey(port));
  }

  leftover(port: PortRef, dying: Set<number>): PortRef | null {
    const w = this.wireAt(port);
    if (!w) return null;
    const o = otherEnd(w, port);
    if (dying.has(o.id)) return null;
    return o;
  }

  wiredTogether(aId: number, bId: number): boolean {
    for (const w of this.wires.values()) {
      if (
        (w.a.id === aId && w.b.id === bId) ||
        (w.a.id === bId && w.b.id === aId)
      ) {
        return true;
      }
    }
    return false;
  }

  /** Shortest hop count along wires. Missing entry ⇒ not in the same component. */
  hopDistances(agents: Map<number, Agent>): Map<number, Map<number, number>> {
    const adj = new Map<number, number[]>();
    for (const id of agents.keys()) adj.set(id, []);
    for (const wire of this.wires.values()) {
      if (!agents.has(wire.a.id) || !agents.has(wire.b.id)) continue;
      if (wire.a.id === wire.b.id) continue;
      adj.get(wire.a.id)!.push(wire.b.id);
      adj.get(wire.b.id)!.push(wire.a.id);
    }
    const out = new Map<number, Map<number, number>>();
    for (const start of agents.keys()) {
      const dist = new Map<number, number>();
      dist.set(start, 0);
      const q = [start];
      for (let i = 0; i < q.length; i++) {
        const u = q[i];
        const du = dist.get(u)!;
        for (const v of adj.get(u) ?? []) {
          if (dist.has(v)) continue;
          dist.set(v, du + 1);
          q.push(v);
        }
      }
      out.set(start, dist);
    }
    return out;
  }

  /** Shortest hop counts along wires. Missing / unreachable pairs are absent (weight 0). */
  hopDistance(): Map<number, Map<number, number>> {
    const adj = new Map<number, Set<number>>();
    const link = (a: number, b: number): void => {
      if (a === b) return;
      let set = adj.get(a);
      if (!set) {
        set = new Set();
        adj.set(a, set);
      }
      set.add(b);
    };
    for (const wire of this.wires.values()) {
      link(wire.a.id, wire.b.id);
      link(wire.b.id, wire.a.id);
    }
    const out = new Map<number, Map<number, number>>();
    for (const start of adj.keys()) {
      const dist = new Map<number, number>([[start, 0]]);
      const q = [start];
      for (let i = 0; i < q.length; i++) {
        const u = q[i];
        const du = dist.get(u)!;
        for (const v of adj.get(u) ?? []) {
          if (dist.has(v)) continue;
          dist.set(v, du + 1);
          q.push(v);
        }
      }
      out.set(start, dist);
    }
    return out;
  }

  attach(a: PortRef, b: PortRef, latchLen: number, time: number): Wire | null {
    if (a.id === b.id && a.slot === b.slot) return null;
    if (!this.isFree(a) || !this.isFree(b)) return null;
    const id = this.nextWireId++;
    const len = Math.max(1, latchLen);
    const wire: Wire = { id, a, b, latchLen: len, lastLen: len, rest: len, born: time, nodes: [] };
    this.wires.set(id, wire);
    this.portWire.set(portKey(a), id);
    this.portWire.set(portKey(b), id);
    return wire;
  }

  connect(
    agents: Map<number, Agent>,
    a: PortRef,
    b: PortRef,
    w: number,
    h: number,
    params: Params,
    time: number,
  ): Wire | null {
    const A = agents.get(a.id);
    const B = agents.get(b.id);
    if (!A || !B) return null;
    const c = wireCubic(A, a.slot, B, b.slot, w, h);
    const len = Math.max(bezierLength(c.p0, c.p1, c.p2, c.p3), params.wireMinRest);
    const wire = this.attach(a, b, len, time);
    if (wire) wire.nodes = sampleChain(c, desiredLinks(len), w, h);
    return wire;
  }

  portsFilled(agent: Agent): boolean {
    for (const slot of slotsFor(agent.kind)) {
      if (this.isFree({ id: agent.id, slot })) return false;
    }
    return true;
  }

  curveLength(wire: Wire, agents: Map<number, Agent>, w: number, h: number): number {
    const A = agents.get(wire.a.id);
    const B = agents.get(wire.b.id);
    if (!A || !B) return wire.lastLen;
    const pts = chainPoints(A, B, wire, w, h);
    return polylineLength(pts, w, h);
  }

  restLength(wire: Wire, time: number, params: Params): number {
    const u = clamp((time - wire.born) / Math.max(0.05, params.wireShrink), 0, 1);
    return lerp(wire.latchLen, params.wireMinRest, easeInOut(u));
  }

  detach(id: number): void {
    const w = this.wires.get(id);
    if (!w) return;
    this.portWire.delete(portKey(w.a));
    this.portWire.delete(portKey(w.b));
    this.wires.delete(id);
  }

  detachAgent(agentId: number): void {
    for (const w of [...this.wires.values()]) {
      if (w.a.id === agentId || w.b.id === agentId) this.detach(w.id);
    }
  }

  snap(
    agents: Map<number, Agent>,
    w: number,
    h: number,
    params: Params,
    time: number,
  ): void {
    type Cand = { pa: PortRef; pb: PortRef; dist: number; rank: number };
    const ports: { ref: PortRef; x: number; y: number; principal: boolean }[] = [];
    for (const agent of agents.values()) {
      if (agent.locked || agent.stun > 0) continue;
      for (const slot of slotsFor(agent.kind)) {
        const ref: PortRef = { id: agent.id, slot };
        if (!this.isFree(ref)) continue;
        const p = portWorld(agent, slot, w, h);
        ports.push({ ref, x: p.x, y: p.y, principal: slot === 'p' });
      }
    }

    const cands: Cand[] = [];
    const r = params.snapRadius;
    const r2 = r * r;
    for (let i = 0; i < ports.length; i++) {
      for (let j = i + 1; j < ports.length; j++) {
        const A = ports[i];
        const B = ports[j];
        if (A.ref.id === B.ref.id) continue;
        const d = wrapDeltaVec(A.x, A.y, B.x, B.y, w, h);
        const dist2 = d.x * d.x + d.y * d.y;
        if (dist2 > r2) continue;
        const agentA = agents.get(A.ref.id);
        const agentB = agents.get(B.ref.id);
        if (!agentA || !agentB) continue;
        if (!inSnapArc(agentA, A.ref.slot, B.x, B.y, w, h, r, params.snapArc)) continue;
        if (!inSnapArc(agentB, B.ref.slot, A.x, A.y, w, h, r, params.snapArc)) continue;
        const rank =
          A.principal && B.principal ? 0 : A.principal || B.principal ? 1 : 2;
        cands.push({ pa: A.ref, pb: B.ref, dist: dist2, rank });
      }
    }
    cands.sort((a, b) => a.rank - b.rank || a.dist - b.dist);
    const taken = new Set<string>();
    for (const c of cands) {
      const ka = portKey(c.pa);
      const kb = portKey(c.pb);
      if (taken.has(ka) || taken.has(kb)) continue;
      if (!this.isFree(c.pa) || !this.isFree(c.pb)) continue;
      if (this.wiredTogether(c.pa.id, c.pb.id)) continue;
      if (this.latchCrosses(agents, c.pa, c.pb, w, h)) continue;
      if (this.connect(agents, c.pa, c.pb, w, h, params, time)) {
        taken.add(ka);
        taken.add(kb);
      }
    }
  }

  latchCrosses(
    agents: Map<number, Agent>,
    pa: PortRef,
    pb: PortRef,
    w: number,
    h: number,
  ): boolean {
    const A = agents.get(pa.id);
    const B = agents.get(pb.id);
    if (!A || !B) return true;
    const p0 = stemWorld(A, pa.slot, w, h);
    const p1 = stemWorld(B, pb.slot, w, h);
    const minDist = WIRE_RADIUS * 2;
    for (const wire of this.wires.values()) {
      const WA = agents.get(wire.a.id);
      const WB = agents.get(wire.b.id);
      if (!WA || !WB) continue;
      const pts = unwrapPoints(chainPoints(WA, WB, wire, w, h), w, h);
      const nSeg = pts.length - 1;
      for (let i = 0; i < nSeg; i++) {
        if (i === 0 && (wire.a.id === A.id || wire.a.id === B.id)) continue;
        if (i === nSeg - 1 && (wire.b.id === A.id || wire.b.id === B.id)) continue;
        if (
          segmentsInterfere(
            p0.x,
            p0.y,
            p1.x,
            p1.y,
            pts[i].x,
            pts[i].y,
            pts[i + 1].x,
            pts[i + 1].y,
            minDist,
          )
        ) {
          return true;
        }
      }
    }
    return false;
  }

  componentMass(agents: Map<number, Agent>): Map<number, number> {
    const parent = new Map<number, number>();
    const find = (x: number): number => {
      let r = parent.get(x) ?? x;
      while ((parent.get(r) ?? r) !== r) r = parent.get(r) ?? r;
      let cur = x;
      while ((parent.get(cur) ?? cur) !== r) {
        const p = parent.get(cur) ?? cur;
        parent.set(cur, r);
        cur = p;
      }
      parent.set(x, r);
      return r;
    };
    const union = (a: number, b: number) => {
      const ra = find(a);
      const rb = find(b);
      if (ra !== rb) parent.set(ra, rb);
    };
    for (const id of agents.keys()) parent.set(id, id);
    for (const wire of this.wires.values()) union(wire.a.id, wire.b.id);
    const totals = new Map<number, number>();
    for (const agent of agents.values()) {
      const r = find(agent.id);
      totals.set(r, (totals.get(r) ?? 0) + agent.mass);
    }
    const out = new Map<number, number>();
    for (const agent of agents.values()) {
      out.set(agent.id, totals.get(find(agent.id)) ?? agent.mass);
    }
    return out;
  }

  applySprings(
    agents: Map<number, Agent>,
    w: number,
    h: number,
    params: Params,
    dt: number,
    time: number,
  ): void {
    const snap = new Map<number, { x: number; y: number; heading: number }>();
    for (const agent of agents.values()) {
      snap.set(agent.id, { x: agent.x, y: agent.y, heading: agent.heading });
    }
    const acc = new Map<number, { x: number; y: number; n: number }>();
    const add = (agent: Agent) => {
      if (agent.locked) return;
      let s = acc.get(agent.id);
      if (!s) {
        s = { x: 0, y: 0, n: 0 };
        acc.set(agent.id, s);
      }
      s.x += agent.x;
      s.y += agent.y;
      s.n += 1;
    };
    const restore = (agent: Agent) => {
      const s = snap.get(agent.id);
      if (!s) return;
      agent.x = s.x;
      agent.y = s.y;
      agent.heading = s.heading;
    };

    for (const wire of this.wires.values()) {
      const A = agents.get(wire.a.id);
      const B = agents.get(wire.b.id);
      if (!A || !B) continue;
      if (A.locked && B.locked) continue;
      const rest = this.restLength(wire, time, params);
      wire.rest = rest;
      reduceChain(wire.nodes, rest);
      restore(A);
      restore(B);
      wire.lastLen = solveChain(
        A,
        wire.a.slot,
        B,
        wire.b.slot,
        wire.nodes,
        rest,
        params,
        dt,
        w,
        h,
        0,
        0,
      );
      add(A);
      add(B);
    }

    for (const [id, s] of acc) {
      const agent = agents.get(id);
      if (!agent || s.n < 1) continue;
      agent.x = s.x / s.n;
      agent.y = s.y / s.n;
      const orig = snap.get(id);
      if (orig) agent.heading = orig.heading;
    }
  }
}

function chainPoints(
  A: Agent,
  B: Agent,
  wire: Wire,
  w: number,
  h: number,
): { x: number; y: number }[] {
  return [
    stemWorld(A, wire.a.slot, w, h),
    ...wire.nodes,
    stemWorld(B, wire.b.slot, w, h),
  ];
}

export function wrapPos(agent: Agent, w: number, h: number): void {
  agent.x = wrap(agent.x, w);
  agent.y = wrap(agent.y, h);
}
