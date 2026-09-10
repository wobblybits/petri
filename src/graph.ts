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

/**
 * Drop a wire's shape samples without replacing the array.
 *
 * `wire.shape = []` reads as free and is not: the coarsened branch runs for
 * every wire every frame, so at pond scale it minted fourteen thousand empty
 * arrays a frame purely to say "nothing here". Truncating in place says the
 * same thing to every reader — they all test `length`.
 */
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
  /**
   * Ports that must never latch on their own. A compiled lambda term has an
   * interface — the handle on its result — and leaving that free would let it
   * grab the first passing agent and corrupt the term.
   */
  sealed = new Set<number>();

  /** Broad phase for latching: ports only ever pair up within snapRadius. */
  private portGrid = new PairGrid();
  /**
   * The free ports of the pond, as parallel arrays reused frame to frame:
   * the body, its id, the slot as a small integer, and the tip.
   *
   * This was an array of objects each carrying a fresh `PortRef`, rebuilt
   * every frame, with a second array of candidate objects on top of it — on
   * the order of a hundred thousand short-lived objects a frame at fifty
   * thousand bodies, for a pass whose inputs are a body, a slot and a point.
   * The greedy pass at the end needs a real `PortRef` only for the handful of
   * latches it actually makes.
   */
  private snapAgents: Agent[] = [];
  private portId = new Int32Array(0);
  private snapSlot = new Uint8Array(0);
  private portX = new Float64Array(0);
  private portY = new Float64Array(0);
  /*
   * The outward axis of each port, unit length. Parallel to `portX`/`portY`
   * and written by the same pass, because both come out of one sine and one
   * cosine of the body's heading and the arc test would otherwise take that
   * heading apart again for every candidate pair the port appears in.
   */
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

  /**
   * Follow the sim onto a new store.
   *
   * `Sim.clear` builds a fresh `AgentStore` rather than emptying the old one,
   * and deliberately: an `Agent` is a flyweight over a slot, so anything still
   * holding one from the old pond — a selection, a test comparing before with
   * after — keeps reading the old pond's numbers instead of silently aliasing
   * whichever body lands in that slot next. Port occupancy lives in the store,
   * so the graph has to be told.
   */
  useStore(store: AgentStore): void {
    this.store = store;
  }

  /** Birth length floor, as a fraction of wireMinRest. */
  static BIRTH_FLOOR = 0.2;

  wires = new Map<number, Wire>();
  /**
   * Where port occupancy lives: `store.portWire`, three entries a body, -1
   * for a free port. See the note on that field for why it is stored by slot
   * and not by agent id.
   *
   * The methods below come in two forms on purpose. `isFree(port)` and
   * friends take an id, resolve it through the store's `idToSlot`, and are
   * for the paths that mutate the graph — a latch, a detach, a rewrite — of
   * which there are hundreds a frame. `isFreeAtSlot` and `portWireAtSlot`
   * take a slot the caller already has, and are for the loops that ask about
   * every port of every body, of which there are hundreds of thousands.
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

  /**
   * Wire bounding boxes for the latch crossing test, rebuilt for each `snap`
   * because every endpoint moves every frame. `latchIndexed` says whether it
   * describes the wires as they are right now: outside the greedy pass it does
   * not, and `nearbyWires` falls back to the whole map.
   */
  private latchGrid = new BoxGrid();
  private latchWires: Wire[] = [];
  private latchMinX = new Float64Array(0);
  private latchMinY = new Float64Array(0);
  private latchMaxX = new Float64Array(0);
  private latchMaxY = new Float64Array(0);
  private latchIndexed = false;
  private latchHits: Wire[] = [];

  /**
   * Bin every wire by the box its polyline occupies. Paid once per `snap`,
   * against the `O(candidates x wires)` it replaces -- and it hoists the two
   * `stemWorldInto` calls per wire out of the candidate loop as well, which is
   * most of what the scan cost even before the segment tests.
   *
   * `wires` and its two endpoint lists are the caller's, resolved once a frame
   * and cached on the graph and roster versions. This used to answer
   * `wire.a.id -> body` itself, with two map lookups per wire: at fifty
   * thousand bodies, fifty thousand lookups a frame, measured at 3.7 ms of the
   * latch pass's 11, rebuilding a resolution the caller already kept on
   * exactly the keys that decide when it goes stale. The roster it is cut
   * against is `agents.values()`, so the two agree body for body; an endpoint
   * the caller could not resolve arrives as `undefined`, which is the same
   * wire this used to skip.
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
   * Rotate which wire (or vacancy) sits on which of an agent's ports.
   *
   * The occupancy of `p, l, r` — including empty slots — shifts one step so a
   * designer can try every attachment without tearing the ropes down and
   * rebuilding them. Era has only a principal, so there is nothing to turn.
   * Returns false when the agent has fewer than two slots or every slot is
   * already empty.
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

  /**
   * Drop the rope onto the current stem chord. Used after `rebind` so a
   * leftover does not keep a polyline that belonged to the dying ports.
   */
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
    // The relaxed length, for the same reason `connect` seeds one: a rebind
    // re-seats the ramp at the new stem chord, and `syncRest` will multiply
    // that by the stroke. See `strokeOf`.
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
      // `len` is the span the wire is observed to have, which is the
      // contracted length while its bodies are mid-stroke. `latchLen` starts
      // the shrink ramp that `syncRest` then multiplies by the stroke, so the
      // relaxed length is what belongs in it. See `strokeOf`.
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
    // The shared slot lists rather than `slotsFor`, which allocates: this is
    // asked once a body a frame by the activity LOD.
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
   * Rest length on the shrink curve. The duration stretches with how much wire
   * there is to reel in, so a long latch closes at roughly the same speed as a
   * short one instead of yanking its agents together.
   */
  /**
   * The gait's stroke on a wire: what its two ends' metabolism multiplies its
   * rest length by, this instant.
   *
   * The mean of the two ends, so a wire is one muscle rather than two arguing,
   * and so a phase difference across it shortens the stroke rather than
   * tearing it in half.
   *
   * One definition, read in three places, because the three have to agree.
   * `syncRest` applies it; `connect` and `restitchChord` divide it back out of
   * the length they seed a wire at. A wire is seeded at the span it is
   * observed to have, and a body mid-stroke holds its wires off their relaxed
   * length — so the observed span is the *contracted* one. Storing it as
   * `latchLen` and then multiplying by the stroke again asks for a length
   * nothing is at: a step of up to `gaitSwell` in `rest` on the frame after
   * every latch and every restitch, handed to a span constraint stiff enough
   * to answer it as a kick. Divided out, the wire is born satisfied exactly as
   * `connect` intends, and what moves it afterwards is the stroke *changing*,
   * which is what a muscle is.
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
    // Floored well above nothing: a wire hauling its ends into contact is a
    // rewrite, and this is not one.
    return stroke < 0.4 ? 0.4 : stroke;
  }

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

  /**
   * Drop every wire touching `agentId`.
   *
   * Through the port map, not a scan. This used to copy the whole wire map
   * into an array and walk it — once per death and twice per rewrite — which
   * is nothing at a hundred wires and is most of a frame at seventy thousand:
   * a churning pond kills tens of bodies a frame, and each one paid for every
   * wire in the pond. A body has three ports and a port holds at most one
   * wire, so three lookups find everything the scan did. A self-wire is
   * reached through either of its ends and detached once, since the second
   * lookup finds the port empty.
   */
  detachAgent(agentId: number): void {
    const wp = this.wireAtSlot(agentId, 'p');
    if (wp) this.detach(wp.id);
    const wl = this.wireAtSlot(agentId, 'l');
    if (wl) this.detach(wl.id);
    const wr = this.wireAtSlot(agentId, 'r');
    if (wr) this.detach(wr.id);
  }

  /**
   * `wires`, `endA` and `endB` are the caller's resolved wire list; see
   * `buildLatchIndex`. `Sim.latchPass` is the only thing that should call
   * this, because it is what keeps them.
   */
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
    // Sealing is rare and usually nothing is sealed at all, in which case the
    // second lookup for every port of every body answers no by definition.
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
      // One turn of the heading for all of this body's ports, and for the arc
      // test on every pair they later land in. Through the store's memo, so
      // the GPU probe pack a few phases later gets it for nothing — it wants
      // the same bodies at the same headings.
      const heading = store.heading[s];
      syncHeadingCosSin(store.csHeading, store.csCos, store.csSin, s, heading);
      const cos = store.csCos[s];
      const sin = store.csSin[s];
      const ax = agent.x;
      const ay = agent.y;
      const scale = agent.scale;
      // The slot is in hand, so occupancy is an array read rather than a hash.
      // This loop runs for every port of every body: at thirty thousand it was
      // seventy thousand map lookups a frame, and 3.8 ms of the pass.
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
    // Ports only ever latch within snapRadius, so testing every pair against
    // every other was work the radius check threw away immediately — millions
    // of rejections a frame at a few thousand agents.
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
        /*
         * The deleted `inSnapArcAt`, inlined for both ends, with the two
         * things it recomputed lifted out: the separation, which it measured once per end from
         * coordinates this already has, and each port's outward axis, which
         * it rebuilt from the body's heading with a sine and a cosine. Thirty
         * thousand pairs survive the radius at fifty thousand bodies, so that
         * was sixty thousand hypotenuses and as many sine-cosine pairs.
         *
         * Arithmetic is unchanged on purpose — same `Math.hypot`, same
         * divide, same comparison — so the same pairs latch in the same order
         * and the determinism hashes hold. The second end's vector is the
         * first's negated, and IEEE negation is exact.
         */
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
    // Total order, so the greedy pass below cannot depend on the order
    // candidates happened to be generated in. Rank and distance alone leave
    // exact ties — which mirror-symmetric presets produce — to be broken by
    // the sort's stability, i.e. by Map iteration order.
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
    // Index the wires once here rather than rescanning them for every
    // candidate. Only valid while the pass runs: endpoints move next frame.
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
      // A port latched earlier in this pass is no longer free, which is the
      // whole of what the old `taken` set recorded. Hundreds of candidates a
      // frame rather than tens of thousands of ports, so the id form is fine.
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
   * Whether a proposed latch chord would cut across an existing wire.
   *
   * Costly, and the cost curve runs backwards to the value. The loop is over
   * every wire in the graph, so this is `O(candidates x wires)`: measured at
   * ~6,000 agents and 7,000 wires it was 70% of what `snap` costs, and in a
   * churning 9,000-body soup 9.35ms of snap's 9.72, off four calls a frame.
   *
   * And in a soup that size it is not visibly buying anything -- disabling it
   * for 1,400 frames left the crossing rate flat, 0.94 to 1.03 per thousand
   * wire pairs against 0.96 with it on. Crossings there come from bodies
   * drifting after they latch rather than from the latch, and nothing catches
   * those: `uncrossPrincipals` returns immediately, `params.uncross` being 0.
   *
   * It is still not removable. Where it is cheap is exactly where it matters:
   * a preset or a hand-built term has few enough wires that the loop is
   * nothing, and one latch reaching across a chord there is structural rather
   * than cosmetic. Removing it was tried and `does not latch through an
   * intervening wire` in sim.test.ts caught it immediately -- two bodies at
   * either side of a wall wired straight through it.
   *
   * So it is indexed rather than switched off. `snap` bins every wire's bounding
   * box before the greedy pass and this walks only the boxes the chord reaches,
   * which is a handful. The per-wire geometry below is untouched -- the index
   * decides which wires are examined, never whether one crosses -- and when no
   * index has been built, as when a test calls this directly, it falls back to
   * the whole map and behaves exactly as it always did.
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
   * Root id of each agent's connected component.
   *
   * Cached on the graph version and on `roster`, which a caller that tracks
   * one should pass: keyed on the agent count alone, a death and a birth in
   * the same frame kept the size and handed back a map holding the dead id
   * and missing the live one. The count is the fallback for callers without
   * a version to hand.
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
   * Rest length for a wire this frame: the shrink curve toward `wireMinRest`,
   * the gait's swell, and a slow per-wire breath under both.
   *
   * The swell is the gait, and it is the only actuator it has.
   *
   * `rest` is what this engine already moves things with: `wireShrink` reels
   * a latch in through it, `Wire.collapse` hauls a rewrite's ends together
   * with it, `wireBreathe` makes tissue move with it. All three read on
   * screen, because the span constraint *serves* `rest` rather than fighting
   * it. The gait is the fourth thing that writes it.
   *
   * A correction shared by inverse mass moves both bodies and not their
   * centre, so the swing itself carries nothing. What carries is the velocity
   * it induces, which decays at each body's own rate — so a wire whose two
   * ends grip differently keeps a step out of every cycle, and one whose ends
   * match keeps nothing. `grip * fullness` is one such difference and the
   * `G` head's `anchor` is the other, and the anchor rides the same cosine as
   * this does, so a body grips exactly while its wires pull.
   *
   * This is `wireTug` with a better clock rather than a new mechanism. The
   * tug was driven by whichever packet last crossed, which is why it could
   * only twitch; this is driven by a phase the net owns and couples along its
   * own wires, so what runs down a chain is a wave.
   *
   * Floored well above nothing: a wire hauling its ends into contact is a
   * rewrite, and this is not one.
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
      // Applied under the floor on purpose: a collapsing wire has to be able
      // to reach nothing, and 4 px is still a visible thread.
      if (wire.collapse > 0) {
        wire.rest = Math.max(0.5, wire.rest * (1 - wire.collapse));
      }
      if (wire.ropePath !== 'span' && (!detailed || detailed(wire))) {
        reduceChain(wire.nodes, wire.rest);
      }
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
    clearShape(wire);
  }

  /**
   * Drop a coarsened rope onto the live stem chord without touching rest.
   * FAR skips XPBD, so leftover nodes otherwise freeze in world space and
   * become a several-hundred-pixel fossil — and a whip on the way back.
   */
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
