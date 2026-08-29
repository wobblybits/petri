import {
  inSnapArc,
  portKey,
  portWorld,
  slotsFor,
  stemWorld,
  wireCubic,
  type Agent,
  type PortRef,
  type PortSlot,
} from './agents.ts';
import {
  desiredLinks,
  polylineLength,
  reduceChain,
  sampleChain,
  solveWire,
  unwrapPoints,
  type ChainNode,
  type WireStiffness,
} from './chain.ts';
import { bezierPoint } from './curve.ts';
import { segmentsInterfere, WIRE_RADIUS } from './geom.ts';
import { PairGrid } from './grid.ts';
import type { Params } from './params.ts';
import { clamp, easeInOut, lerp, wrap, wrapDeltaVec, type Vec2 } from './wrap.ts';
import type { LatchEvent } from './audio/types.ts';

export interface Wire {
  id: number;
  a: PortRef;
  b: PortRef;
  latchLen: number;
  lastLen: number;
  rest: number;
  /** Arc length of the rope. Longer than `rest` when the ports force a detour. */
  ropeLen: number;
  /** Rest shape: where each rope node wants to sit on the port-respecting curve. */
  shape: Vec2[];
  born: number;
  nodes: ChainNode[];
}

const SLOT_ORDER = { p: 0, l: 1, r: 2 } as const;
function slotOrder(slot: PortSlot): number {
  return SLOT_ORDER[slot];
}

export function otherEnd(wire: Wire, port: PortRef): PortRef {
  if (wire.a.id === port.id && wire.a.slot === port.slot) return wire.b;
  return wire.a;
}

export class Graph {
  /** Broad phase for latching: ports only ever pair up within snapRadius. */
  private portGrid = new PairGrid();
  private portX: number[] = [];
  private portY: number[] = [];

  /** Birth length floor, as a fraction of wireMinRest. */
  static BIRTH_FLOOR = 0.2;

  wires = new Map<number, Wire>();
  portWire = new Map<string, number>();
  nextWireId = 1;
  onLatch: ((ev: LatchEvent) => void) | null = null;

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

  attach(a: PortRef, b: PortRef, latchLen: number, time: number): Wire | null {
    if (a.id === b.id && a.slot === b.slot) return null;
    if (!this.isFree(a) || !this.isFree(b)) return null;
    const id = this.nextWireId++;
    const len = Math.max(1, latchLen);
    const wire: Wire = { id, a, b, latchLen: len, lastLen: len, rest: len, ropeLen: len, shape: [], born: time, nodes: [] };
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
    // No heading assignment here. Snapping used to rotate both bodies into
    // alignment on the spot, which moves their stems, which hands the solver a
    // fresh violation and a body overlap to resolve in one substep. The port
    // torques turn them instead, and the wire is born slack enough to allow it.
    const sa = stemWorld(A, a.slot, w, h);
    const sb = stemWorld(B, b.slot, w, h);
    const stemDelta = wrapDeltaVec(sa.x, sa.y, sb.x, sb.y, w, h);
    // Born at the length it actually latched at, so the wire starts satisfied
    // and `restLength` ramps it to wireMinRest over `wireShrink`. Clamping this
    // up to wireMinRest skips the ramp and hands the solver a 30 px violation
    // to resolve in one substep, which reads as a kick.
    const span = Math.hypot(stemDelta.x, stemDelta.y);
    const len = Math.max(span, params.wireMinRest * Graph.BIRTH_FLOOR);
    const c = wireCubic(A, a.slot, B, b.slot, w, h, len);
    const wire = this.attach(a, b, len, time);
    if (wire) {
      wire.nodes = sampleChain(c, desiredLinks(len), w, h);
      this.onLatch?.({
        type: 'latch',
        wireId: wire.id,
        agentA: a.id,
        agentB: b.id,
        slotA: a.slot,
        slotB: b.slot,
        kindA: A.kind,
        kindB: B.kind,
        rest: wire.rest,
        latchLen: len,
      });
    }
    return wire;
  }

  /** Cheap degree test: a few port lookups rather than a scan of every wire. */
  isWired(agent: Agent): boolean {
    for (const slot of slotsFor(agent.kind)) {
      if (!this.isFree({ id: agent.id, slot })) return true;
    }
    return false;
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

  stemSpan(wire: Wire, agents: Map<number, Agent>, w: number, h: number): number {
    const A = agents.get(wire.a.id);
    const B = agents.get(wire.b.id);
    if (!A || !B) return wire.rest;
    const pa = stemWorld(A, wire.a.slot, w, h);
    const pb = stemWorld(B, wire.b.slot, w, h);
    const d = wrapDeltaVec(pa.x, pa.y, pb.x, pb.y, w, h);
    return Math.hypot(d.x, d.y);
  }

  /**
   * Rest length on the shrink curve. The duration stretches with how much wire
   * there is to reel in, so a long latch closes at roughly the same speed as a
   * short one instead of yanking its agents together.
   */
  restLength(wire: Wire, time: number, params: Params): number {
    const travel = Math.abs(wire.latchLen - params.wireMinRest);
    const span = Math.max(1, params.wireMinRest);
    const dur = Math.max(0.05, params.wireShrink) * Math.max(1, travel / span);
    const u = clamp((time - wire.born) / dur, 0, 1);
    return lerp(wire.latchLen, params.wireMinRest, easeInOut(u));
  }

  /** 0 → 1 over the (distance-scaled) shrink window. */
  shrinkProgress(wire: Wire, time: number, params: Params): number {
    const travel = Math.abs(wire.latchLen - params.wireMinRest);
    const span = Math.max(1, params.wireMinRest);
    const dur = Math.max(0.05, params.wireShrink) * Math.max(1, travel / span);
    return clamp((time - wire.born) / dur, 0, 1);
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
    const touchR = 5.5;
    const touchR2 = touchR * touchR;
    // Ports only ever latch within snapRadius, so testing every pair against
    // every other was work the radius check threw away immediately — millions
    // of rejections a frame at a few thousand agents.
    const nPorts = ports.length;
    if (this.portX.length < nPorts) {
      this.portX = new Array(nPorts * 2);
      this.portY = new Array(nPorts * 2);
    }
    for (let i = 0; i < nPorts; i++) {
      this.portX[i] = ports[i].x;
      this.portY[i] = ports[i].y;
    }
    this.portGrid.build(this.portX, this.portY, nPorts, Math.max(1, r));
    this.portGrid.forEachPair((p, q) => {
      // Keep the original lower-index-first ordering: it decides which end
      // becomes wire.a, and the constraint solve is order-sensitive.
      const i = p < q ? p : q;
      const j = p < q ? q : p;
      const A = ports[i];
      const B = ports[j];
      if (A.ref.id === B.ref.id) return;
      const d = wrapDeltaVec(A.x, A.y, B.x, B.y, w, h);
      const dist2 = d.x * d.x + d.y * d.y;
      if (dist2 > r2) return;
      const agentA = agents.get(A.ref.id);
      const agentB = agents.get(B.ref.id);
      if (!agentA || !agentB) return;
      const touching = dist2 <= touchR2;
      if (!touching) {
        if (!inSnapArc(agentA, A.ref.slot, B.x, B.y, w, h, r, params.snapArc)) return;
        if (!inSnapArc(agentB, B.ref.slot, A.x, A.y, w, h, r, params.snapArc)) return;
      }
      const rank = A.principal && B.principal ? 0 : A.principal || B.principal ? 1 : 2;
      cands.push({ pa: A.ref, pb: B.ref, dist: dist2, rank: touching ? rank - 1 : rank });
    });
    // Total order, so the greedy pass below cannot depend on the order
    // candidates happened to be generated in. Rank and distance alone leave
    // exact ties — which mirror-symmetric presets produce — to be broken by
    // Array#sort's stability, i.e. by Map iteration order.
    cands.sort(
      (a, b) =>
        a.rank - b.rank ||
        a.dist - b.dist ||
        a.pa.id - b.pa.id ||
        slotOrder(a.pa.slot) - slotOrder(b.pa.slot) ||
        a.pb.id - b.pb.id ||
        slotOrder(a.pb.slot) - slotOrder(b.pb.slot),
    );
    const taken = new Set<string>();
    for (const c of cands) {
      const ka = portKey(c.pa);
      const kb = portKey(c.pb);
      if (taken.has(ka) || taken.has(kb)) continue;
      if (!this.isFree(c.pa) || !this.isFree(c.pb)) continue;
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

  wireCount(agentId: number): number {
    let n = 0;
    for (const wire of this.wires.values()) {
      if (wire.a.id === agentId || wire.b.id === agentId) n++;
    }
    return n;
  }

  shrinkU(wire: Wire, time: number, params: Params): number {
    return this.shrinkProgress(wire, time, params);
  }

  /** Root id of each agent's connected component. */
  componentIds(agents: Map<number, Agent>): Map<number, number> {
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
    for (const id of agents.keys()) parent.set(id, id);
    for (const wire of this.wires.values()) {
      const ra = find(wire.a.id);
      const rb = find(wire.b.id);
      if (ra !== rb) parent.set(ra, rb);
    }
    const out = new Map<number, number>();
    for (const id of agents.keys()) out.set(id, find(id));
    return out;
  }

  /**
   * Rest length for a wire this frame: the shrink curve toward `wireMinRest`,
   * plus a slow per-wire breath so a settled net keeps moving like tissue
   * instead of freezing solid.
   */
  syncRest(time: number, params: Params): void {
    for (const wire of this.wires.values()) {
      const base = this.restLength(wire, time, params);
      const phase = wire.id * 2.399963;
      const rate = 0.55 + (wire.id % 7) * 0.11;
      const breathe = 1 + params.wireBreathe * Math.sin(time * rate + phase);
      wire.rest = Math.max(4, base * breathe);
      reduceChain(wire.nodes, wire.rest);
    }
  }

  /**
   * The wire's rest shape: the cubic that leaves both ports along their axes —
   * the same curve the renderer draws. Sampling it gives every rope node a
   * target, which is what makes a slack rope well-posed, and its arc length is
   * the rope's length, so links, bending and shape all agree.
   */
  syncRopeShape(agents: Map<number, Agent>, w: number, h: number): void {
    for (const wire of this.wires.values()) {
      const A = agents.get(wire.a.id);
      const B = agents.get(wire.b.id);
      if (!A || !B) continue;
      const n = wire.nodes.length;
      if (n === 0) {
        wire.shape = [];
        wire.ropeLen = wire.rest;
        continue;
      }
      const c = wireCubic(A, wire.a.slot, B, wire.b.slot, w, h, wire.rest);
      const pts: Vec2[] = [];
      for (let i = 1; i <= n; i++) {
        pts.push(bezierPoint(c.p0, c.p1, c.p2, c.p3, i / (n + 1)));
      }
      wire.shape = pts;
      wire.ropeLen = polylineLength([c.p0, ...pts, c.p3], w, h);
    }
  }

  /**
   * Compliance for one wire. A fresh latch is slack — it reaches and settles;
   * an aged latch is firm. This replaces the old shrink/align/organize phase
   * machine with a single continuous parameter.
   */
  private stiffness(wire: Wire, time: number, params: Params): WireStiffness {
    const age = Math.max(0, time - wire.born);
    return {
      scale: 12 / Math.max(1, params.springK),
      slack: 1 + 6 * Math.exp(-age / 0.8),
    };
  }

  /** One XPBD iteration over every wire. Called once per substep. */
  solveWires(
    agents: Map<number, Agent>,
    params: Params,
    h: number,
    time: number,
  ): void {
    for (const wire of this.wires.values()) {
      const A = agents.get(wire.a.id);
      const B = agents.get(wire.b.id);
      if (!A || !B) continue;
      if (A.locked && B.locked) continue;
      solveWire(
        A,
        wire.a.slot,
        B,
        wire.b.slot,
        wire.nodes,
        wire.rest,
        wire.ropeLen,
        wire.shape,
        this.stiffness(wire, time, params),
        h,
      );
    }
  }

  /** Bookkeeping the renderer and rewrite gate read. */
  refreshLengths(agents: Map<number, Agent>, w: number, h: number): void {
    for (const wire of this.wires.values()) {
      wire.lastLen = this.curveLength(wire, agents, w, h);
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
