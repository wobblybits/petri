import {
  inSnapArc,
  portKey,
  portWorld,
  slotsFor,
  stemWorldInto,
  wireCubic,
  type Agent,
  type PortRef,
  type PortSlot,
} from './agents.ts';
import {
  desiredLinks,
  reduceChain,
  sampleChain,
  solveWire,
  solveWireSpan,
  type ChainNode,
  type WireStiffness,
} from './chain.ts';
import { bezierPointInto } from './curve.ts';
import { segmentsInterfere, WIRE_RADIUS } from './geom.ts';
import { PairGrid } from './grid.ts';
import type { Params } from './params.ts';
import { clamp, easeInOut, lerp, wrap, wrapDeltaVec, type Vec2 } from './wrap.ts';
import type { LatchEvent } from './audio/types.ts';

const stemScratchA = { x: 0, y: 0 };
const stemScratchB = { x: 0, y: 0 };

export interface Wire {
  id: number;
  /**
   * How far through a rewrite's pull this wire is, 0..1.
   *
   * A rewrite hauls its two bodies together, and it is the wire that does the
   * hauling — so the wire has to shorten, not go slack. Left alone the rope
   * keeps its material length while the chord closes, which bows it out and
   * sags its pitch; driving the rest length down instead keeps it taut, and
   * it retracts into the pair as they meet.
   */
  collapse: number;
  /**
   * Shortest sounding length this wire will report, half its uncollapsed
   * rest. A rope squeezed below half its rest has buckled rather than
   * tightened, and letting the pitch keep climbing turns a wire retracting
   * into a rewrite into a three-octave squeal.
   */
  pitchFloor: number;
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
  /**
   * Solver path last applied. Young / slack wires stay `'full'`; a taut
   * latch drops shape, then the rope, and comes back if it goes slack.
   */
  ropePath: RopePath;
}

export type RopePath = 'full' | 'no-shape' | 'span';

/**
 * Extra lastLen/rest a coarsened wire may grow before the rope comes back.
 * The taut threshold itself is `params.wireTaut`.
 */
export const ROPE_TAUT_HYSTERESIS = 0.08;

export function ropePathOf(wire: Wire, time: number, params: Params): RopePath {
  const age = Math.max(0, time - wire.born);
  const rest = Math.max(1e-6, wire.rest);
  const ratio = wire.lastLen / rest;
  const tautMax = Math.max(1, params.wireTaut);
  const slackMax = tautMax + ROPE_TAUT_HYSTERESIS;
  const coarsened = wire.ropePath === 'span' || wire.ropePath === 'no-shape';
  const taut = coarsened ? ratio <= slackMax : ratio <= tautMax;
  if (!taut) return 'full';
  const spanAge = params.wireSpanAge;
  const shapeAge = params.wireShapeAge;
  if (spanAge > 0 && age >= spanAge) return 'span';
  if (shapeAge > 0 && age >= shapeAge) return 'no-shape';
  return 'full';
}

/** Live XPBD rope, not a view-FAR chord or an age-span joint. */
export function ropeIsLive(wire: Wire, detailed?: (wire: Wire) => boolean): boolean {
  if (detailed && !detailed(wire)) return false;
  return wire.ropePath !== 'span';
}

const SLOT_ORDER = { p: 0, l: 1, r: 2 } as const;
function slotOrder(slot: PortSlot): number {
  return SLOT_ORDER[slot];
}

export function otherEnd(wire: Wire, port: PortRef): PortRef {
  if (wire.a.id === port.id && wire.a.slot === port.slot) return wire.b;
  return wire.a;
}

function samePort(a: PortRef, b: PortRef): boolean {
  return a.id === b.id && a.slot === b.slot;
}

/** Keep a surviving end on the same delay-line side. */
function orientRebind(
  oldA: PortRef,
  oldB: PortRef,
  na: PortRef,
  nb: PortRef,
): [PortRef, PortRef] {
  if (samePort(oldA, na) || samePort(oldB, nb)) return [na, nb];
  if (samePort(oldA, nb) || samePort(oldB, na)) return [nb, na];
  if (oldA.id === na.id || oldB.id === nb.id) return [na, nb];
  if (oldA.id === nb.id || oldB.id === na.id) return [nb, na];
  return [na, nb];
}

export class Graph {
  /**
   * Ports that must never latch on their own. A compiled lambda term has an
   * interface — the handle on its result — and leaving that free would let it
   * grab the first passing agent and corrupt the term.
   */
  sealed = new Set<string>();

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
  /** Bumps whenever a wire is added or removed. Hop caches key off this. */
  version = 0;
  private hops: Map<number, Map<number, number>> | null = null;
  private hopsVersion = -1;
  private hopsAgents = -1;
  private comps: Map<number, number> | null = null;
  private compsVersion = -1;
  private compsAgents = -1;
  private latchPts: { x: number; y: number }[] = [];

  clear(): void {
    this.wires.clear();
    this.portWire.clear();
    this.sealed.clear();
    this.bump();
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

  private bump(): void {
    this.version++;
    this.hops = null;
    this.comps = null;
  }

  /** Shortest hop count along wires. Missing entry ⇒ not in the same component.
   *  Flocking no longer reads this; it is kept for tests and debug. */
  hopDistances(agents: Map<number, Agent>): Map<number, Map<number, number>> {
    if (this.hops && this.hopsVersion === this.version && this.hopsAgents === agents.size) {
      return this.hops;
    }
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
    this.hops = out;
    this.hopsVersion = this.version;
    this.hopsAgents = agents.size;
    return out;
  }

  attach(a: PortRef, b: PortRef, latchLen: number, time: number): Wire | null {
    if (a.id === b.id && a.slot === b.slot) return null;
    if (!this.isFree(a) || !this.isFree(b)) return null;
    const id = this.nextWireId++;
    const len = Math.max(1, latchLen);
    const wire: Wire = { id, a, b, collapse: 0, pitchFloor: len * 0.5, latchLen: len, lastLen: len, rest: len, ropeLen: len, shape: [], born: time, nodes: [], ropePath: 'full' };
    this.wires.set(id, wire);
    this.portWire.set(portKey(a), id);
    this.portWire.set(portKey(b), id);
    this.bump();
    return wire;
  }

  /**
   * Move a live wire onto new ports without minting an id. The delay line
   * keeps ringing; `restitchChord` then sits the rope on the new stems.
   * Surviving ends stay on the same side of the delay line.
   */
  rebind(id: number, a: PortRef, b: PortRef): boolean {
    const w = this.wires.get(id);
    if (!w) return false;
    if (a.id === b.id && a.slot === b.slot) return false;
    const held = (p: PortRef) => {
      const at = this.portWire.get(portKey(p));
      return at === undefined || at === id;
    };
    if (!held(a) || !held(b)) return false;
    this.portWire.delete(portKey(w.a));
    this.portWire.delete(portKey(w.b));
    const [na, nb] = orientRebind(w.a, w.b, a, b);
    w.a = na;
    w.b = nb;
    this.portWire.set(portKey(w.a), id);
    this.portWire.set(portKey(w.b), id);
    this.bump();
    return true;
  }

  /**
   * Drop the rope onto the current stem chord. Used after `rebind` so a
   * leftover does not keep a polyline that belonged to the dying ports.
   */
  restitchChord(
    id: number,
    agents: Map<number, Agent>,
    w: number,
    h: number,
    time: number,
  ): void {
    const wire = this.wires.get(id);
    if (!wire) return;
    const A = agents.get(wire.a.id);
    const B = agents.get(wire.b.id);
    if (!A || !B) return;
    const sa = stemWorldInto(A, wire.a.slot, w, h, stemScratchA);
    const sax = sa.x;
    const say = sa.y;
    const sb = stemWorldInto(B, wire.b.slot, w, h, stemScratchB);
    const sbx = sb.x;
    const sby = sb.y;
    const dx = sbx - sax;
    const dy = sby - say;
    const span = Math.max(1, Math.hypot(dx, dy));
    wire.nodes = sampleChain(
      {
        p0: { x: sax, y: say },
        p1: { x: sax + dx / 3, y: say + dy / 3 },
        p2: { x: sax + (2 * dx) / 3, y: say + (2 * dy) / 3 },
        p3: { x: sbx, y: sby },
      },
      desiredLinks(span),
      w,
      h,
    );
    wire.lastLen = span;
    wire.ropeLen = span;
    wire.latchLen = span;
    wire.rest = span;
    wire.pitchFloor = span * 0.5;
    wire.collapse = 0;
    wire.born = time;
    wire.shape = [];
    wire.ropePath = 'full';
  }

  connect(
    agents: Map<number, Agent>,
    a: PortRef,
    b: PortRef,
    w: number,
    h: number,
    params: Params,
    time: number,
    opts?: { chord?: boolean; silent?: boolean },
  ): Wire | null {
    const A = agents.get(a.id);
    const B = agents.get(b.id);
    if (!A || !B) return null;
    // No heading assignment here. Snapping used to rotate both bodies into
    // alignment on the spot, which moves their stems, which hands the solver a
    // fresh violation and a body overlap to resolve in one substep. The port
    // torques turn them instead, and the wire is born slack enough to allow it.
    const sa = stemWorldInto(A, a.slot, w, h, stemScratchA);
    const sax = sa.x;
    const say = sa.y;
    const sb = stemWorldInto(B, b.slot, w, h, stemScratchB);
    const sbx = sb.x;
    const sby = sb.y;
    // Born at the length it actually latched at, so the wire starts satisfied
    // and `restLength` ramps it to wireMinRest over `wireShrink`. Clamping this
    // up to wireMinRest skips the ramp and hands the solver a 30 px violation
    // to resolve in one substep, which reads as a kick.
    const stemDx = sbx - sax;
    const stemDy = sby - say;
    const span = Math.hypot(stemDx, stemDy);
    const len = Math.max(span, params.wireMinRest * Graph.BIRTH_FLOOR);
    const c = opts?.chord
      ? {
          p0: { x: sax, y: say },
          p1: { x: sax + stemDx / 3, y: say + stemDy / 3 },
          p2: { x: sax + (2 * stemDx) / 3, y: say + (2 * stemDy) / 3 },
          p3: { x: sbx, y: sby },
        }
      : wireCubic(A, a.slot, B, b.slot, w, h, len);
    const wire = this.attach(a, b, len, time);
    if (wire) {
      wire.nodes = sampleChain(c, desiredLinks(len), w, h);
      if (!opts?.silent) {
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
    const sa = stemWorldInto(A, wire.a.slot, w, h, stemScratchA);
    const sb = stemWorldInto(B, wire.b.slot, w, h, stemScratchB);
    let px = sa.x;
    let py = sa.y;
    let len = 0;
    for (let i = 0; i < wire.nodes.length; i++) {
      const n = wire.nodes[i];
      const dx = n.x - px;
      const dy = n.y - py;
      len += Math.hypot(dx, dy);
      px = n.x;
      py = n.y;
    }
    const dx = sb.x - px;
    const dy = sb.y - py;
    return len + Math.hypot(dx, dy);
  }

  stemSpan(wire: Wire, agents: Map<number, Agent>, w: number, h: number): number {
    const A = agents.get(wire.a.id);
    const B = agents.get(wire.b.id);
    if (!A || !B) return wire.rest;
    const pa = stemWorldInto(A, wire.a.slot, w, h, stemScratchA);
    const pb = stemWorldInto(B, wire.b.slot, w, h, stemScratchB);
    const dx = pb.x - pa.x;
    const dy = pb.y - pa.y;
    return Math.hypot(dx, dy);
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
    this.bump();
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
        if (this.sealed.has(portKey(ref))) continue;
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
    const p0 = stemWorldInto(A, pa.slot, w, h, stemScratchA);
    const p1 = stemWorldInto(B, pb.slot, w, h, stemScratchB);
    const minDist = WIRE_RADIUS * 2;
    const pts = this.latchPts;
    for (const wire of this.wires.values()) {
      const WA = agents.get(wire.a.id);
      const WB = agents.get(wire.b.id);
      if (!WA || !WB) continue;
      const n = wire.nodes.length;
      const need = n + 2;
      while (pts.length < need) pts.push({ x: 0, y: 0 });
      stemWorldInto(WA, wire.a.slot, w, h, pts[0]);
      for (let i = 0; i < n; i++) {
        pts[i + 1].x = wire.nodes[i].x;
        pts[i + 1].y = wire.nodes[i].y;
      }
      stemWorldInto(WB, wire.b.slot, w, h, pts[n + 1]);
      const nSeg = need - 1;
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
    if (this.comps && this.compsVersion === this.version && this.compsAgents === agents.size) {
      return this.comps;
    }
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
    this.comps = out;
    this.compsVersion = this.version;
    this.compsAgents = agents.size;
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
      wire.pitchFloor = wire.rest * 0.5;
      // Applied under the floor on purpose: a collapsing wire has to be able
      // to reach nothing, and 4 px is still a visible thread.
      if (wire.collapse > 0) {
        wire.rest = Math.max(0.5, wire.rest * (1 - wire.collapse));
      }
      if (wire.ropePath !== 'span') reduceChain(wire.nodes, wire.rest);
    }
  }

  /**
   * Age + tautness → solver path. Crossing into span-only leaves the nodes
   * in place (draw uses the port-axis cubic); coming back resamples them so
   * a leftover that goes slack does not teleport.
   */
  applyRopePaths(
    agents: Map<number, Agent>,
    w: number,
    h: number,
    time: number,
    params: Params,
  ): void {
    for (const wire of this.wires.values()) {
      const next = ropePathOf(wire, time, params);
      if (wire.ropePath === 'span' && next !== 'span') {
        this.rebuildRope(wire, agents, w, h);
      }
      wire.ropePath = next;
    }
  }

  private rebuildRope(
    wire: Wire,
    agents: Map<number, Agent>,
    w: number,
    h: number,
  ): void {
    const A = agents.get(wire.a.id);
    const B = agents.get(wire.b.id);
    if (!A || !B) return;
    const c = wireCubic(A, wire.a.slot, B, wire.b.slot, w, h, wire.rest);
    wire.nodes = sampleChain(c, desiredLinks(wire.rest), w, h);
    wire.shape = [];
  }

  /**
   * The wire's rest shape: the cubic that leaves both ports along their axes —
   * the same curve the renderer draws. Sampling it gives every rope node a
   * target, which is what makes a slack rope well-posed, and its arc length is
   * the rope's length, so links, bending and shape all agree.
   */
  syncRopeShape(
    agents: Map<number, Agent>,
    w: number,
    h: number,
    detailed?: (wire: Wire) => boolean,
  ): void {
    for (const wire of this.wires.values()) {
      if (detailed && !detailed(wire)) continue;
      if (wire.ropePath === 'span') {
        wire.shape = [];
        wire.ropeLen = wire.rest;
        continue;
      }
      const A = agents.get(wire.a.id);
      const B = agents.get(wire.b.id);
      if (!A || !B) continue;
      const n = wire.nodes.length;
      if (n === 0 || wire.ropePath === 'no-shape') {
        wire.shape = [];
        wire.ropeLen = wire.rest;
        continue;
      }
      const c = wireCubic(A, wire.a.slot, B, wire.b.slot, w, h, wire.rest);
      const shape = wire.shape;
      if (shape.length !== n) {
        wire.shape = new Array(n);
        for (let i = 0; i < n; i++) wire.shape[i] = { x: 0, y: 0 };
      }
      const pts = wire.shape;
      let rope = 0;
      let px = c.p0.x;
      let py = c.p0.y;
      for (let i = 0; i < n; i++) {
        const p = bezierPointInto(c.p0, c.p1, c.p2, c.p3, (i + 1) / (n + 1), pts[i]);
        rope += Math.hypot(p.x - px, p.y - py);
        px = p.x;
        py = p.y;
      }
      rope += Math.hypot(c.p3.x - px, c.p3.y - py);
      wire.ropeLen = rope;
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

  /** Public for the WASM pack — same numbers the JS XPBD path uses. */
  stiffnessOf(wire: Wire, time: number, params: Params): WireStiffness {
    return this.stiffness(wire, time, params);
  }

  /** One XPBD iteration over every wire. Called once per substep. */
  solveWires(
    agents: Map<number, Agent>,
    params: Params,
    h: number,
    time: number,
    frozen?: Set<number>,
    detailed?: (wire: Wire) => boolean,
  ): void {
    for (const wire of this.wires.values()) {
      const A = agents.get(wire.a.id);
      const B = agents.get(wire.b.id);
      if (!A || !B) continue;
      if (A.locked && B.locked) continue;
      if (frozen && (frozen.has(A.id) || frozen.has(B.id))) continue;
      const stiff = this.stiffness(wire, time, params);
      if (!ropeIsLive(wire, detailed)) {
        solveWireSpan(A, wire.a.slot, B, wire.b.slot, wire.rest, stiff, h);
        continue;
      }
      solveWire(
        A,
        wire.a.slot,
        B,
        wire.b.slot,
        wire.nodes,
        wire.rest,
        wire.ropeLen,
        wire.ropePath === 'full' ? wire.shape : [],
        stiff,
        h,
      );
    }
  }

  /** Bookkeeping the renderer and rewrite gate read. */
  refreshLengths(
    agents: Map<number, Agent>,
    w: number,
    h: number,
    frozen?: Set<number>,
    detailed?: (wire: Wire) => boolean,
  ): void {
    for (const wire of this.wires.values()) {
      if (frozen && frozen.has(wire.a.id) !== frozen.has(wire.b.id)) continue;
      wire.lastLen = ropeIsLive(wire, detailed)
        ? this.curveLength(wire, agents, w, h)
        : this.stemSpan(wire, agents, w, h);
    }
  }
}

export function wrapPos(agent: Agent, w: number, h: number): void {
  agent.x = wrap(agent.x, w);
  agent.y = wrap(agent.y, h);
}
