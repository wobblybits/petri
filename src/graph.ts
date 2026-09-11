import {
  ERA_SLOTS,
  NODE_SLOTS,
  poseHeld,
  portFrameInto,
  portKeyAt,
  slotIndex,
  slotsFor,
  stemWorldInto,
  syncHeadingCosSin,
  wireCubic,
  type Agent,
  type AgentKind,
  type PortFrame,
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
import type { AgentStore } from './agent-store.ts';
import { BoxGrid, PairGrid } from './grid.ts';
import type { Params } from './params.ts';
import { clamp, easeInOut, lerp, wrap, type Vec2 } from './wrap.ts';
import type { LatchEvent } from './audio/types.ts';

const stemScratchA = { x: 0, y: 0 };
const stemScratchB = { x: 0, y: 0 };

export interface Wire {
  id: number;
  /** How far through a rewrite's pull this wire is, 0..1. Drives the rest length down so the
   *  hauling wire stays taut rather than bowing as the chord closes. */
  collapse: number;
  /** Shortest sounding length this wire will report: half its uncollapsed rest, below which a rope has buckled rather than tightened. */
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
  /** Solver path last applied. Young / slack wires stay `'full'`; a taut latch drops shape, then the rope. */
  ropePath: RopePath;
}

export type RopePath = 'full' | 'no-shape' | 'span';

/** Extra lastLen/rest a coarsened wire may grow before the rope comes back; the taut threshold is `params.wireTaut`. */
export const ROPE_TAUT_HYSTERESIS = 0.08;

/** Drop a wire's shape samples in place; every reader tests `length`, and this runs for every wire every frame. */
function clearShape(wire: Wire): void {
  if (wire.shape.length > 0) wire.shape.length = 0;
}
/** Rest lengths past this yank the FAR span joint across the view. */
export const REST_CAP = 2500;

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

/** Slot names by the code `snap` keys its port table on: p, l, r. */
const SLOT_NAME: readonly PortSlot[] = ['p', 'l', 'r'];

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
  /** Ports that must never latch on their own: a compiled term's interface, which a stray latch would corrupt. */
  sealed = new Set<number>();

  /** Broad phase for latching: ports only ever pair up within snapRadius. */
  private portGrid = new PairGrid();
  /** The free ports of the pond, as parallel arrays reused frame to frame: body, id, slot code, tip. */
  private snapAgents: Agent[] = [];
  private portId = new Int32Array(0);
  private snapSlot = new Uint8Array(0);
  private portX = new Float64Array(0);
  private portY = new Float64Array(0);
  /** Outward unit axis of each port, parallel to `portX`/`portY` and written by the same pass. */
  private portAX = new Float64Array(0);
  private portAY = new Float64Array(0);
  /** Candidate pairs, by port-table index, and an index array to sort them. */
  private candA = new Int32Array(0);
  private candB = new Int32Array(0);
  private candDist = new Float64Array(0);
  private candRank = new Int8Array(0);
  private candOrder = new Int32Array(0);
  private nCands = 0;
  /** Scratch refs for `latchCrosses`, which reads them and keeps nothing. */
  private readonly snapRefA: PortRef = { id: 0, slot: 'p' };
  private readonly snapRefB: PortRef = { id: 0, slot: 'p' };
  private readonly snapFrame: PortFrame = { x: 0, y: 0, ax: 0, ay: 0 };

  private growPorts(cap: number): void {
    const id = new Int32Array(cap);
    id.set(this.portId);
    this.portId = id;
    const sl = new Uint8Array(cap);
    sl.set(this.snapSlot);
    this.snapSlot = sl;
    const px = new Float64Array(cap);
    px.set(this.portX);
    this.portX = px;
    const py = new Float64Array(cap);
    py.set(this.portY);
    this.portY = py;
    const ax = new Float64Array(cap);
    ax.set(this.portAX);
    this.portAX = ax;
    const ay = new Float64Array(cap);
    ay.set(this.portAY);
    this.portAY = ay;
  }

  private pushCand(i: number, j: number, dist2: number, rank: number): void {
    const n = this.nCands;
    if (n >= this.candA.length) {
      const cap = n * 2 + 64;
      const a = new Int32Array(cap);
      a.set(this.candA);
      this.candA = a;
      const b = new Int32Array(cap);
      b.set(this.candB);
      this.candB = b;
      const d = new Float64Array(cap);
      d.set(this.candDist);
      this.candDist = d;
      const r = new Int8Array(cap);
      r.set(this.candRank);
      this.candRank = r;
      this.candOrder = new Int32Array(cap);
    }
    this.candA[n] = i;
    this.candB[n] = j;
    this.candDist[n] = dist2;
    this.candRank[n] = rank;
    this.nCands = n + 1;
  }

  constructor(store: AgentStore) {
    this.store = store;
  }

  /** Follow the sim onto a new store; port occupancy lives in the store, so the graph has to be told. */
  useStore(store: AgentStore): void {
    this.store = store;
  }

  /** Birth length floor, as a fraction of wireMinRest. */
  static BIRTH_FLOOR = 0.2;

  wires = new Map<number, Wire>();
  /**
   * Port occupancy lives in `store.portWire`: three entries a body, -1 for free.
   * `isFree(port)` and friends resolve an id through `idToSlot` and serve the
   * mutating paths; `isFreeAtSlot` / `portWireAtSlot` take a slot and serve the
   * loops over every port of every body.
   */
  private store: AgentStore;
  nextWireId = 1;
  onLatch: ((ev: LatchEvent) => void) | null = null;
  /** Bumps whenever a wire is added or removed. Hop caches key off this. */
  version = 0;
  private comps: Map<number, number> | null = null;
  private compsVersion = -1;
  private compsAgents = -1;
  private latchPts: { x: number; y: number }[] = [];

  /** Wire bounding boxes for the latch crossing test, rebuilt each `snap`. Valid only while
   *  `latchIndexed`; otherwise `nearbyWires` falls back to the whole map. */
  private latchGrid = new BoxGrid();
  private latchWires: Wire[] = [];
  private latchMinX = new Float64Array(0);
  private latchMinY = new Float64Array(0);
  private latchMaxX = new Float64Array(0);
  private latchMaxY = new Float64Array(0);
  private latchIndexed = false;
  private latchHits: Wire[] = [];

  /**
   * Bin every wire by the box its polyline occupies, once per `snap`. `wires`, `endA`
   * and `endB` are the caller's resolved list, cut against `agents.values()`; an
   * unresolved endpoint arrives as `undefined` and the wire is skipped.
   */
  private buildLatchIndex(
    wires: Wire[],
    endA: readonly (Agent | undefined)[],
    endB: readonly (Agent | undefined)[],
    w: number,
    h: number,
  ): void {
    const cap = wires.length;
    if (this.latchMinX.length < cap) {
      this.latchMinX = new Float64Array(cap * 2);
      this.latchMinY = new Float64Array(cap * 2);
      this.latchMaxX = new Float64Array(cap * 2);
      this.latchMaxY = new Float64Array(cap * 2);
    }
    this.latchWires.length = 0;
    let k = 0;
    for (let wi = 0; wi < cap; wi++) {
      const wire = wires[wi];
      const WA = endA[wi];
      const WB = endB[wi];
      if (!WA || !WB) continue;
      const sa = stemWorldInto(WA, wire.a.slot, w, h, stemScratchA);
      let lox = sa.x;
      let hix = sa.x;
      let loy = sa.y;
      let hiy = sa.y;
      const sb = stemWorldInto(WB, wire.b.slot, w, h, stemScratchB);
      if (sb.x < lox) lox = sb.x;
      else if (sb.x > hix) hix = sb.x;
      if (sb.y < loy) loy = sb.y;
      else if (sb.y > hiy) hiy = sb.y;
      // The rope bulges off the chord, so the nodes set the box, not the ends.
      for (let i = 0; i < wire.nodes.length; i++) {
        const nx = wire.nodes[i].x;
        const ny = wire.nodes[i].y;
        if (nx < lox) lox = nx;
        else if (nx > hix) hix = nx;
        if (ny < loy) loy = ny;
        else if (ny > hiy) hiy = ny;
      }
      this.latchMinX[k] = lox;
      this.latchMinY[k] = loy;
      this.latchMaxX[k] = hix;
      this.latchMaxY[k] = hiy;
      this.latchWires.push(wire);
      k++;
    }
    this.latchGrid.build(this.latchMinX, this.latchMinY, this.latchMaxX, this.latchMaxY, k);
    this.latchIndexed = true;
  }

  /** Wires whose box the chord reaches, or all of them if nothing is indexed. */
  private nearbyWires(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    pad: number,
  ): Iterable<Wire> {
    if (!this.latchIndexed) return this.wires.values();
    const hits = this.latchHits;
    hits.length = 0;
    const wires = this.latchWires;
    this.latchGrid.forEachNear(
      Math.min(x0, x1) - pad,
      Math.min(y0, y1) - pad,
      Math.max(x0, x1) + pad,
      Math.max(y0, y1) + pad,
      (i) => {
        hits.push(wires[i]);
      },
    );
    return hits;
  }

  clear(): void {
    this.wires.clear();
    this.store.portWire.fill(-1);
    this.sealed.clear();
    this.bump();
  }

  /** The wire on a port named by slot, or -1. The hot form: no id lookup. */
  portWireAtSlot(storeSlot: number, code: number): number {
    return this.store.portWire[storeSlot * 3 + code];
  }

  /** `isFree` for a caller that already holds the body's slot. */
  isFreeAtSlot(storeSlot: number, code: number): boolean {
    return this.store.portWire[storeSlot * 3 + code] < 0;
  }

  /** The wire id on a port, or -1 for a free port or an id nobody lives at. */
  private wireIdAt(id: number, slot: PortSlot): number {
    const at = this.store.slotFor(id);
    if (at === undefined) return -1;
    return this.store.portWire[at * 3 + slotIndex(slot)];
  }

  private setWireAt(id: number, slot: PortSlot, wireId: number): void {
    const at = this.store.slotFor(id);
    if (at === undefined) return;
    this.store.portWire[at * 3 + slotIndex(slot)] = wireId;
  }

  wireAt(port: PortRef): Wire | undefined {
    const id = this.wireIdAt(port.id, port.slot);
    if (id < 0) return undefined;
    return this.wires.get(id);
  }

  isFree(port: PortRef): boolean {
    return this.wireIdAt(port.id, port.slot) < 0;
  }

  /** `isFree` without building a PortRef for it. */
  isFreeAt(id: number, slot: PortSlot): boolean {
    return this.wireIdAt(id, slot) < 0;
  }

  /** `wireAt` without building a PortRef for it. */
  wireAtSlot(id: number, slot: PortSlot): Wire | undefined {
    const wid = this.wireIdAt(id, slot);
    if (wid < 0) return undefined;
    return this.wires.get(wid);
  }

  /** `portsFilled` without allocating the slot list. */
  portsFilledAt(agent: Agent): boolean {
    const PW = this.store.portWire;
    const at = agent.slot * 3;
    if (PW[at] < 0) return false;
    if (agent.kind === 'era') return true;
    return PW[at + 1] >= 0 && PW[at + 2] >= 0;
  }

  /** True when `a` and `b` already share a wire. FAR discs skip those pairs. */
  sharesWire(aId: number, bId: number): boolean {
    if (aId === bId) return false;
    for (const slot of ['p', 'l', 'r'] as const) {
      const w = this.wireAt({ id: aId, slot });
      if (w && (w.a.id === bId || w.b.id === bId)) return true;
    }
    return false;
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
    this.comps = null;
  }

  attach(a: PortRef, b: PortRef, latchLen: number, time: number): Wire | null {
    if (a.id === b.id && a.slot === b.slot) return null;
    if (!this.isFree(a) || !this.isFree(b)) return null;
    const id = this.nextWireId++;
    const len = Math.max(1, latchLen);
    const wire: Wire = { id, a, b, collapse: 0, pitchFloor: len * 0.5, latchLen: len, lastLen: len, rest: len, ropeLen: len, shape: [], born: time, nodes: [], ropePath: 'full' };
    this.wires.set(id, wire);
    this.setWireAt(a.id, a.slot, id);
    this.setWireAt(b.id, b.slot, id);
    this.bump();
    return wire;
  }

  /** Move a live wire onto new ports without minting an id; surviving ends stay on the same
   *  delay-line side. `restitchChord` then sits the rope on the new stems. */
  rebind(id: number, a: PortRef, b: PortRef): boolean {
    const w = this.wires.get(id);
    if (!w) return false;
    if (a.id === b.id && a.slot === b.slot) return false;
    const held = (p: PortRef) => {
      const at = this.wireIdAt(p.id, p.slot);
      return at < 0 || at === id;
    };
    if (!held(a) || !held(b)) return false;
    this.setWireAt(w.a.id, w.a.slot, -1);
    this.setWireAt(w.b.id, w.b.slot, -1);
    const [na, nb] = orientRebind(w.a, w.b, a, b);
    w.a = na;
    w.b = nb;
    this.setWireAt(w.a.id, w.a.slot, id);
    this.setWireAt(w.b.id, w.b.slot, id);
    this.bump();
    return true;
  }

  /**
   * Rotate which wire (or vacancy) sits on which of an agent's ports, one step.
   * Returns false when the agent has fewer than two slots or every slot is empty.
   */
  cycleSlots(agentId: number, kind: AgentKind, dir: 1 | -1 = 1): boolean {
    const slots = slotsFor(kind);
    if (slots.length < 2) return false;
    const n = slots.length;
    const step = dir >= 0 ? 1 : n - 1;
    type Occ = { wire: Wire; end: 'a' | 'b' };
    const occ: (Occ | null)[] = slots.map((slot) => {
      const wire = this.wireAtSlot(agentId, slot);
      if (!wire) return null;
      const end: 'a' | 'b' = wire.a.id === agentId && wire.a.slot === slot ? 'a' : 'b';
      return { wire, end };
    });
    if (occ.every((o) => o === null)) return false;
    for (const slot of slots) this.setWireAt(agentId, slot, -1);
    for (let i = 0; i < n; i++) {
      const o = occ[i];
      if (!o) continue;
      const slot = slots[(i + step) % n];
      if (o.end === 'a') o.wire.a = { id: agentId, slot };
      else o.wire.b = { id: agentId, slot };
      this.setWireAt(agentId, slot, o.wire.id);
    }
    this.bump();
    return true;
  }

  /** Drop the rope onto the current stem chord; used after `rebind`. */
  restitchChord(
    id: number,
    agents: Map<number, Agent>,
    w: number,
    h: number,
    params: Params,
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
    const span = Math.min(REST_CAP, Math.max(1, Math.hypot(dx, dy)));
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
    // Seed the relaxed length: `syncRest` multiplies `latchLen` by the stroke. See `strokeOf`.
    const stroke = this.strokeOf(wire.a, wire.b, agents, this.store.gaitWave, params);
    wire.latchLen = stroke === 1 ? span : Math.min(REST_CAP, span / stroke);
    wire.rest = span;
    wire.pitchFloor = span * 0.5;
    wire.collapse = 0;
    wire.born = time;
    clearShape(wire);
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
    // No heading assignment here: the port torques turn the bodies, and the wire is born slack enough to allow it.
    const sa = stemWorldInto(A, a.slot, w, h, stemScratchA);
    const sax = sa.x;
    const say = sa.y;
    const sb = stemWorldInto(B, b.slot, w, h, stemScratchB);
    const sbx = sb.x;
    const sby = sb.y;
    // Born at the length it latched at, so the wire starts satisfied and `restLength` ramps it toward wireMinRest.
    const stemDx = sbx - sax;
    const stemDy = sby - say;
    const span = Math.hypot(stemDx, stemDy);
    const len = Math.min(REST_CAP, Math.max(span, params.wireMinRest * Graph.BIRTH_FLOOR));
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
      // `len` is the contracted span mid-stroke; `latchLen` holds the relaxed length. See `strokeOf`.
      const stroke = this.strokeOf(a, b, agents, this.store.gaitWave, params);
      if (stroke !== 1) wire.latchLen = Math.min(REST_CAP, len / stroke);
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
    // Shared slot lists rather than `slotsFor`, which allocates; asked once a body a frame.
    const slots = agent.kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
    for (let i = 0; i < slots.length; i++) {
      if (!this.isFreeAt(agent.id, slots[i])) return true;
    }
    return false;
  }

  portsFilled(agent: Agent): boolean {
    const slots = agent.kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
    for (let i = 0; i < slots.length; i++) {
      if (this.isFreeAt(agent.id, slots[i])) return false;
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
   * The gait's stroke on a wire: what its two ends' metabolism multiplies its rest
   * length by, this instant. The mean of the two ends, so a wire is one muscle.
   * One definition read in three places: `syncRest` applies it; `connect` and
   * `restitchChord` divide it back out of the observed (contracted) span they seed a
   * wire at, so the wire is born satisfied and only a *change* of stroke moves it.
   */
  strokeOf(
    a: PortRef,
    b: PortRef,
    agents: Map<number, Agent>,
    wave: Float64Array,
    params: Params,
  ): number {
    const swell = params.gaitSwell;
    if (!(swell > 0)) return 1;
    const A = agents.get(a.id);
    const B = agents.get(b.id);
    const w = ((A ? wave[A.slot] : 0) + (B ? wave[B.slot] : 0)) * 0.5;
    const stroke = 1 + swell * w;
    // Floored: a wire hauling its ends into contact is a rewrite, and this is not one.
    return stroke < 0.4 ? 0.4 : stroke;
  }

  /** Rest length on the shrink curve; the duration scales with the travel, so a long latch closes at the same speed as a short one. */
  restLength(wire: Wire, time: number, params: Params): number {
    const travel = Math.abs(wire.latchLen - params.wireMinRest);
    const span = Math.max(1, params.wireMinRest);
    const dur = Math.max(0.05, params.wireShrink) * Math.max(1, travel / span);
    const u = clamp((time - wire.born) / dur, 0, 1);
    const rest = lerp(wire.latchLen, params.wireMinRest, easeInOut(u));
    if (!Number.isFinite(rest)) return params.wireMinRest;
    return Math.min(REST_CAP, rest);
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
    this.setWireAt(w.a.id, w.a.slot, -1);
    this.setWireAt(w.b.id, w.b.slot, -1);
    this.wires.delete(id);
    this.bump();
  }

  /** Drop every wire touching `agentId`, through the port map: three lookups. A self-wire is
   *  reached through either end and detached once. */
  detachAgent(agentId: number): void {
    const wp = this.wireAtSlot(agentId, 'p');
    if (wp) this.detach(wp.id);
    const wl = this.wireAtSlot(agentId, 'l');
    if (wl) this.detach(wl.id);
    const wr = this.wireAtSlot(agentId, 'r');
    if (wr) this.detach(wr.id);
  }

  /** `wires`, `endA` and `endB` are the caller's resolved wire list (see `buildLatchIndex`);
   *  `Sim.latchPass` is the only caller, because it is what keeps them. */
  snap(
    agents: Map<number, Agent>,
    w: number,
    h: number,
    params: Params,
    time: number,
    wires: Wire[],
    endA: readonly (Agent | undefined)[],
    endB: readonly (Agent | undefined)[],
  ): void {
    if (params.snapRadius <= 0) return;
    const list = this.snapAgents;
    const frame = this.snapFrame;
    // Usually nothing is sealed, and then the per-port lookup is skipped.
    const anySealed = this.sealed.size > 0;
    const PORTWIRE = this.store.portWire;
    let n = 0;
    for (const agent of agents.values()) {
      const store = agent.store;
      const s = agent.slot;
      if (store.locked[s] || store.stun[s] > 0) continue;
      const id = agent.id;
      const kind = agent.kind;
      const slots = kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
      // One heading turn per body, through the store's memo so the GPU probe pack later reuses it.
      const heading = store.heading[s];
      syncHeadingCosSin(store.csHeading, store.csCos, store.csSin, s, heading);
      const cos = store.csCos[s];
      const sin = store.csSin[s];
      const ax = agent.x;
      const ay = agent.y;
      const scale = agent.scale;
      // The slot is in hand, so occupancy is an array read rather than a hash.
      const pw = s * 3;
      for (let k = 0; k < slots.length; k++) {
        const slot = slots[k];
        if (PORTWIRE[pw + k] >= 0) continue;
        if (anySealed && this.sealed.has(portKeyAt(id, slot))) continue;
        if (n >= this.portX.length) this.growPorts(n * 2 + 64);
        portFrameInto(kind, slot, ax, ay, scale, cos, sin, frame);
        list[n] = agent;
        this.portId[n] = id;
        // The slot's index in its list is its code: p, l, r as 0, 1, 2.
        this.snapSlot[n] = k;
        this.portX[n] = frame.x;
        this.portY[n] = frame.y;
        this.portAX[n] = frame.ax;
        this.portAY[n] = frame.ay;
        n++;
      }
    }
    list.length = n;

    const r = params.snapRadius;
    const r2 = r * r;
    const touchR = 5.5;
    const touchR2 = touchR * touchR;
    const arcCos = Math.cos(Math.min(params.snapArc, Math.PI * 0.49));
    // Ports only ever latch within snapRadius, so that is the grid's cell size.
    this.portGrid.build(this.portX, this.portY, n, Math.max(1, r));
    this.nCands = 0;
    const PID = this.portId;
    const SLOT = this.snapSlot;
    const PX = this.portX;
    const PY = this.portY;
    const AX = this.portAX;
    const AY = this.portAY;
    this.portGrid.forEachPair((p, q) => {
      // Keep the original lower-index-first ordering: it decides which end
      // becomes wire.a, and the constraint solve is order-sensitive.
      const i = p < q ? p : q;
      const j = p < q ? q : p;
      if (PID[i] === PID[j]) return;
      const dx = PX[j] - PX[i];
      const dy = PY[j] - PY[i];
      const dist2 = dx * dx + dy * dy;
      if (dist2 > r2) return;
      const touching = dist2 <= touchR2;
      const sa = SLOT[i];
      const sb = SLOT[j];
      if (!touching) {
        // Arc test at both ends. The hypot-then-divide form is what the determinism hashes
        // were taken on; the second end's vector is the first's negated, and IEEE negation is exact.
        const dist = Math.hypot(dx, dy);
        if (dist > r || dist < 1e-6) return;
        if ((dx * AX[i] + dy * AY[i]) / dist < arcCos) return;
        if ((-dx * AX[j] - dy * AY[j]) / dist < arcCos) return;
      }
      const pa = sa === 0;
      const pb = sb === 0;
      const rank = pa && pb ? 0 : pa || pb ? 1 : 2;
      this.pushCand(i, j, dist2, touching ? rank - 1 : rank);
    });
    const nc = this.nCands;
    if (nc === 0) {
      list.length = 0;
      return;
    }
    // Total order, so the greedy pass cannot depend on generation order: rank and distance
    // alone leave exact ties (mirror-symmetric presets) to Map iteration order.
    const order = this.candOrder;
    for (let k = 0; k < nc; k++) order[k] = k;
    const CA = this.candA;
    const CB = this.candB;
    const CD = this.candDist;
    const CR = this.candRank;
    order.subarray(0, nc).sort(
      (x, y) =>
        CR[x] - CR[y] ||
        CD[x] - CD[y] ||
        PID[CA[x]] - PID[CA[y]] ||
        SLOT[CA[x]] - SLOT[CA[y]] ||
        PID[CB[x]] - PID[CB[y]] ||
        SLOT[CB[x]] - SLOT[CB[y]],
    );
    // Index the wires once for this pass; endpoints move next frame.
    this.buildLatchIndex(wires, endA, endB, w, h);
    const refA = this.snapRefA;
    const refB = this.snapRefB;
    for (let k = 0; k < nc; k++) {
      const c = order[k];
      const i = CA[c];
      const j = CB[c];
      const ia = PID[i];
      const ib = PID[j];
      const sla = SLOT_NAME[SLOT[i]];
      const slb = SLOT_NAME[SLOT[j]];
      // A port latched earlier in this pass is no longer free.
      if (this.wireIdAt(ia, sla) >= 0 || this.wireIdAt(ib, slb) >= 0) continue;
      refA.id = ia;
      refA.slot = sla;
      refB.id = ib;
      refB.slot = slb;
      if (this.latchCrosses(agents, refA, refB, w, h)) continue;
      // `connect` keeps its refs, so a latch is the one place this pass allocates.
      this.connect(agents, { id: ia, slot: sla }, { id: ib, slot: slb }, w, h, params, time);
    }
    this.latchIndexed = false;
    // Not held across frames: a body that dies would otherwise stay reachable.
    list.length = 0;
  }

  /**
   * Whether a proposed latch chord would cut across an existing wire. Structural in
   * presets and hand-built terms (`does not latch through an intervening wire` in
   * sim.test.ts). Walks only the boxes the chord reaches when `snap` has built the
   * index, and the whole map otherwise, as when a test calls this directly.
   */
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
    for (const wire of this.nearbyWires(p0.x, p0.y, p1.x, p1.y, minDist)) {
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

  /**
   * Root id of each agent's connected component. Cached on the graph version and on
   * `roster`; the agent count alone misses a death and a birth in the same frame.
   */
  componentIds(agents: Map<number, Agent>, roster = agents.size): Map<number, number> {
    if (this.comps && this.compsVersion === this.version && this.compsAgents === roster) {
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
    this.compsAgents = roster;
    return out;
  }

  /**
   * Rest length for a wire this frame: the shrink curve toward `wireMinRest`, the
   * gait's stroke, and a slow per-wire breath. The stroke is the gait's only actuator:
   * a span correction shared by inverse mass moves no centre of mass, so what carries
   * is the induced velocity, which decays at each body's own grip — a wire whose ends
   * grip differently keeps a step out of every cycle. `grip * fullness` and the `G`
   * head's `anchor` are those differences, and the anchor rides the same cosine.
   */
  syncRest(
    time: number,
    params: Params,
    agents: Map<number, Agent>,
    wave: Float64Array,
    detailed?: (wire: Wire) => boolean,
  ): void {
    for (const wire of this.wires.values()) {
      const base = this.restLength(wire, time, params);
      const phase = wire.id * 2.399963;
      const rate = 0.55 + (wire.id % 7) * 0.11;
      const breathe = 1 + params.wireBreathe * Math.sin(time * rate + phase);
      const quiet = base * breathe;
      const raw = quiet * this.strokeOf(wire.a, wire.b, agents, wave, params);
      wire.rest = Number.isFinite(raw) ? clamp(raw, 4, REST_CAP) : params.wireMinRest;
      wire.pitchFloor = wire.rest * 0.5;
      // Under the floor on purpose: a collapsing wire has to be able to reach nothing.
      if (wire.collapse > 0) {
        wire.rest = Math.max(0.5, wire.rest * (1 - wire.collapse));
      }
      if (wire.ropePath !== 'span' && (!detailed || detailed(wire))) {
        reduceChain(wire.nodes, wire.rest);
      }
    }
  }

  /** Age + tautness → solver path. Coming back from span resamples the nodes so a leftover
   *  that goes slack does not teleport. */
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
    clearShape(wire);
  }

  /** Drop a coarsened rope onto the live stem chord without touching rest; FAR skips XPBD,
   *  so leftover nodes would otherwise freeze in world space. */
  private sitNodesOnChord(
    wire: Wire,
    agents: Map<number, Agent>,
    w: number,
    h: number,
  ): void {
    const n = wire.nodes.length;
    if (n === 0) return;
    const A = agents.get(wire.a.id);
    const B = agents.get(wire.b.id);
    if (!A || !B) return;
    const sa = stemWorldInto(A, wire.a.slot, w, h, stemScratchA);
    const ax = sa.x;
    const ay = sa.y;
    const sb = stemWorldInto(B, wire.b.slot, w, h, stemScratchB);
    const dx = sb.x - ax;
    const dy = sb.y - ay;
    for (let i = 0; i < n; i++) {
      const t = (i + 1) / (n + 1);
      const node = wire.nodes[i];
      node.x = ax + dx * t;
      node.y = ay + dy * t;
      node.vx = 0;
      node.vy = 0;
      node.prevX = node.x;
      node.prevY = node.y;
    }
  }

  /** The wire's rest shape: the cubic leaving both ports along their axes, the curve the
   *  renderer draws. Its samples are the rope nodes' targets and its arc length is `ropeLen`. */
  syncRopeShape(
    agents: Map<number, Agent>,
    w: number,
    h: number,
    detailed?: (wire: Wire) => boolean,
  ): void {
    for (const wire of this.wires.values()) {
      if (detailed && !detailed(wire)) {
        clearShape(wire);
        wire.ropeLen = Number.isFinite(wire.rest) ? wire.rest : 40;
        this.sitNodesOnChord(wire, agents, w, h);
        continue;
      }
      if (wire.ropePath === 'span') {
        clearShape(wire);
        wire.ropeLen = wire.rest;
        continue;
      }
      const A = agents.get(wire.a.id);
      const B = agents.get(wire.b.id);
      if (!A || !B) continue;
      const n0 = wire.nodes.length;
      if (n0 === 0 || wire.ropePath === 'no-shape') {
        clearShape(wire);
        wire.ropeLen = wire.rest;
        continue;
      }
      const bowed = this.curveLength(wire, agents, w, h);
      if (!Number.isFinite(bowed) || bowed > Math.max(wire.rest * 8, 800)) {
        this.rebuildRope(wire, agents, w, h);
      }
      const n = wire.nodes.length;
      if (n === 0) {
        clearShape(wire);
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

  /** Compliance for one wire: a fresh latch is slack, an aged latch is firm. */
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
      if (poseHeld(A) && poseHeld(B)) continue;
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
