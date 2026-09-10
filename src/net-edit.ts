import {
  boundRadius,
  createAgent,
  portWorld,
  slotsFor,
  stemWorld,
  type Agent,
  type AgentKind,
  type PortRef,
  type PortSlot,
} from './agents.ts';
import type { Camera } from './camera.ts';
import { closestPointOnSegment, segmentsInterfere } from './geom.ts';
import { pickAgent, pickPort } from './interact.ts';
import { formatNet, parseNet } from './net-text.ts';
import { beginStroke, moveStroke, pickSound, pickWire, type SoundStroke } from './net-sound.ts';
import type { Params } from './params.ts';
import type { Sim } from './sim.ts';
import { rotate, wrapAngle } from './wrap.ts';

/** Same extra reach the demo eraser uses past `boundRadius`. */
const ERASE_BRUSH = 10;

/** Pointer travel past which a press stops counting as a click. */
const CLICK_SLOP = 4;

export const ROTATE_STEP = Math.PI / 12;
export const ROTATE_RIGHT_ANGLE = Math.PI / 2;

export type DesignTool =
  | 'select'
  | 'pan'
  | 'erase'
  | 'paint-era'
  | 'paint-dup'
  | 'paint-con'
  | 'rotate'
  | 'touch';

export interface DesignPose {
  id: number;
  kind: AgentKind;
  x: number;
  y: number;
  heading: number;
  extra: number;
  pinned?: boolean;
}

export interface DesignSnapshot {
  nextId: number;
  agents: DesignPose[];
  wires: { a: PortRef; b: PortRef }[];
}

/** Clipboard fragment in centroid-relative coordinates. */
export interface DesignFragment {
  agents: { kind: AgentKind; x: number; y: number; heading: number; extra: number }[];
  wires: { a: number; aSlot: PortSlot; b: number; bSlot: PortSlot }[];
}

type Gesture =
  | { kind: 'none' }
  | { kind: 'pan'; lastX: number; lastY: number }
  | { kind: 'lasso'; points: { x: number; y: number }[]; additive: boolean; wireId?: number }
  | { kind: 'move'; lastX: number; lastY: number }
  | { kind: 'place'; id: number; x: number; y: number; over: PortRef | null }
  | { kind: 'erase'; x: number; y: number }
  | { kind: 'rotate'; cx: number; cy: number; lastAng: number }
  | { kind: 'wire'; from: PortRef; x: number; y: number; over: PortRef | null }
  | { kind: 'sound'; stroke: SoundStroke };

export function toolKind(tool: DesignTool): AgentKind | null {
  if (tool === 'paint-era') return 'era';
  if (tool === 'paint-dup') return 'dup';
  if (tool === 'paint-con') return 'con';
  return null;
}

export function paintSpacing(params: Params): number {
  return Math.max(8, params.wireMinRest);
}

export function capture(sim: Sim): DesignSnapshot {
  return {
    nextId: sim.nextId,
    agents: [...sim.agents.values()].map((a) => ({
      id: a.id,
      kind: a.kind,
      x: a.x,
      y: a.y,
      heading: a.heading,
      extra: a.extra,
      pinned: a.pinned,
    })),
    wires: [...sim.graph.wires.values()].map((w) => ({ a: { ...w.a }, b: { ...w.b } })),
  };
}

export function restore(sim: Sim, snap: DesignSnapshot, params: Params): void {
  sim.clear();
  for (const rec of snap.agents) {
    const a = createAgent(rec.id, rec.kind, rec.x, rec.y, rec.heading, params, sim.agentStore);
    a.extra = rec.extra;
    a.pinned = rec.pinned ?? false;
    a.drive = 0;
    sim.agents.set(a.id, a);
  }
  sim.nextId = snap.nextId;
  sim.noteRosterChange();
  for (const w of snap.wires) {
    sim.graph.connect(sim.agents, w.a, w.b, sim.w, sim.h, params, sim.time, { silent: true });
  }
}

export function pointInPolygon(x: number, y: number, poly: { x: number; y: number }[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    if (yi === yj) continue;
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function selectionCentroid(sim: Sim, ids: Iterable<number>): { x: number; y: number } | null {
  let n = 0;
  let x = 0;
  let y = 0;
  for (const id of ids) {
    const a = sim.agents.get(id);
    if (!a) continue;
    x += a.x;
    y += a.y;
    n++;
  }
  if (n === 0) return null;
  return { x: x / n, y: y / n };
}

export function still(agent: Agent): void {
  agent.vx = 0;
  agent.vy = 0;
  agent.omega = 0;
  agent.prevX = agent.x;
  agent.prevY = agent.y;
  agent.prevHeading = agent.heading;
}

export function restitchIncident(sim: Sim, ids: Iterable<number>, params: Params): void {
  const set = ids instanceof Set ? ids : new Set(ids);
  for (const w of sim.graph.wires.values()) {
    if (set.has(w.a.id) || set.has(w.b.id)) {
      sim.graph.restitchChord(w.id, sim.agents, sim.w, sim.h, params, sim.time);
    }
  }
}

export function translateAgents(sim: Sim, ids: Iterable<number>, dx: number, dy: number, params: Params): void {
  const set = ids instanceof Set ? ids : new Set(ids);
  for (const id of set) {
    const a = sim.agents.get(id);
    if (!a) continue;
    a.x += dx;
    a.y += dy;
    still(a);
  }
  restitchIncident(sim, set, params);
}

export function rotateAgents(sim: Sim, ids: Iterable<number>, da: number, cx: number, cy: number, params: Params): void {
  const set = ids instanceof Set ? ids : new Set(ids);
  for (const id of set) {
    const a = sim.agents.get(id);
    if (!a) continue;
    const p = rotate(a.x - cx, a.y - cy, da);
    a.x = cx + p.x;
    a.y = cy + p.y;
    a.heading = wrapAngle(a.heading + da);
    still(a);
  }
  restitchIncident(sim, set, params);
}

export function agentTooClose(sim: Sim, x: number, y: number, spacing: number, ignore?: number): boolean {
  const min = spacing * 0.9;
  const min2 = min * min;
  for (const a of sim.agents.values()) {
    if (ignore !== undefined && a.id === ignore) continue;
    const d = (a.x - x) ** 2 + (a.y - y) ** 2;
    if (d < min2) return true;
  }
  return false;
}

export function spawnDesigned(
  sim: Sim,
  kind: AgentKind,
  x: number,
  y: number,
  heading: number,
  params: Params,
): Agent | null {
  const a = sim.spawn(kind, x, y, heading, params, true);
  if (!a) return null;
  a.extra = 1;
  a.drive = 0;
  still(a);
  return a;
}

export function copyFragment(sim: Sim, ids: Iterable<number>): DesignFragment | null {
  const list: Agent[] = [];
  const index = new Map<number, number>();
  for (const id of ids) {
    const a = sim.agents.get(id);
    if (!a) continue;
    index.set(a.id, list.length);
    list.push(a);
  }
  if (list.length === 0) return null;
  let cx = 0;
  let cy = 0;
  for (const a of list) {
    cx += a.x;
    cy += a.y;
  }
  cx /= list.length;
  cy /= list.length;
  const wires: DesignFragment['wires'] = [];
  for (const w of sim.graph.wires.values()) {
    const ai = index.get(w.a.id);
    const bi = index.get(w.b.id);
    if (ai === undefined || bi === undefined) continue;
    wires.push({ a: ai, aSlot: w.a.slot, b: bi, bSlot: w.b.slot });
  }
  return {
    agents: list.map((a) => ({
      kind: a.kind,
      x: a.x - cx,
      y: a.y - cy,
      heading: a.heading,
      extra: a.extra,
    })),
    wires,
  };
}

export function snapshotToFragment(snap: DesignSnapshot): DesignFragment | null {
  if (snap.agents.length === 0) return null;
  let cx = 0;
  let cy = 0;
  for (const a of snap.agents) {
    cx += a.x;
    cy += a.y;
  }
  cx /= snap.agents.length;
  cy /= snap.agents.length;
  const index = new Map(snap.agents.map((a, i) => [a.id, i]));
  const wires: DesignFragment['wires'] = [];
  for (const w of snap.wires) {
    const ai = index.get(w.a.id);
    const bi = index.get(w.b.id);
    if (ai === undefined || bi === undefined) continue;
    wires.push({ a: ai, aSlot: w.a.slot, b: bi, bSlot: w.b.slot });
  }
  return {
    agents: snap.agents.map((a) => ({
      kind: a.kind,
      x: a.x - cx,
      y: a.y - cy,
      heading: a.heading,
      extra: a.extra,
    })),
    wires,
  };
}

export function pasteFragment(
  sim: Sim,
  params: Params,
  frag: DesignFragment,
  x: number,
  y: number,
): number[] {
  const ids: number[] = [];
  for (const rec of frag.agents) {
    const a = spawnDesigned(sim, rec.kind, x + rec.x, y + rec.y, rec.heading, params);
    if (!a) break;
    a.extra = rec.extra;
    ids.push(a.id);
  }
  if (ids.length !== frag.agents.length) {
    for (const id of ids) sim.kill(id);
    return [];
  }
  for (const w of frag.wires) {
    const aId = ids[w.a];
    const bId = ids[w.b];
    if (aId === undefined || bId === undefined) continue;
    if (!sim.graph.isFreeAt(aId, w.aSlot) || !sim.graph.isFreeAt(bId, w.bSlot)) continue;
    sim.graph.connect(
      sim.agents,
      { id: aId, slot: w.aSlot },
      { id: bId, slot: w.bSlot },
      sim.w,
      sim.h,
      params,
      sim.time,
      { silent: true },
    );
  }
  return ids;
}

/**
 * Pointer tools and history for the net designer.
 *
 * Selection, clipboard and undo live here rather than on `Sim` — the pond
 * does not know about lassos, and the page should stay a thin chrome around
 * this. History snapshots topology and pose, not rope nodes or scent; a
 * restore rebuilds wires from the port pairs.
 */
export class NetEditor {
  tool: DesignTool = 'select';
  selection = new Set<number>();
  clipboard: DesignFragment | null = null;
  lastPointer = { x: 0, y: 0 };
  gesture: Gesture = { kind: 'none' };
  /** True once the user has panned; the camera stops chasing. */
  freeCamera = true;

  private undoStack: DesignSnapshot[] = [];
  private redoStack: DesignSnapshot[] = [];
  private dirty = false;
  private mutating = false;
  sim: Sim;
  camera: Camera;
  params: Params;

  constructor(sim: Sim, camera: Camera, params: Params) {
    this.sim = sim;
    this.camera = camera;
    this.params = params;
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }
  get canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  pruneSelection(): void {
    for (const id of [...this.selection]) {
      if (!this.sim.agents.has(id)) this.selection.delete(id);
    }
  }

  checkpoint(): void {
    this.undoStack.push(capture(this.sim));
    if (this.undoStack.length > 80) this.undoStack.shift();
    this.redoStack.length = 0;
  }

  private beginMutate(): void {
    this.checkpoint();
    this.dirty = false;
    this.mutating = true;
  }

  private mark(): void {
    this.dirty = true;
  }

  private endMutate(): void {
    if (!this.mutating) return;
    this.mutating = false;
    if (!this.dirty && this.undoStack.length > 0) this.undoStack.pop();
    this.dirty = false;
  }

  undo(): boolean {
    if (this.undoStack.length === 0) return false;
    const current = capture(this.sim);
    const prev = this.undoStack.pop()!;
    this.redoStack.push(current);
    restore(this.sim, prev, this.params);
    this.pruneSelection();
    return true;
  }

  redo(): boolean {
    if (this.redoStack.length === 0) return false;
    const current = capture(this.sim);
    const next = this.redoStack.pop()!;
    this.undoStack.push(current);
    restore(this.sim, next, this.params);
    this.pruneSelection();
    return true;
  }

  selectAll(): void {
    this.selection = new Set(this.sim.agents.keys());
  }

  clearSelection(): void {
    this.selection.clear();
  }

  copy(): boolean {
    this.pruneSelection();
    const frag = copyFragment(this.sim, this.selection);
    if (!frag) return false;
    this.clipboard = frag;
    return true;
  }

  cut(): boolean {
    this.pruneSelection();
    if (this.selection.size === 0) return false;
    if (!this.copy()) return false;
    this.beginMutate();
    for (const id of [...this.selection]) this.sim.kill(id);
    this.selection.clear();
    this.mark();
    this.endMutate();
    return true;
  }

  paste(x?: number, y?: number): number[] {
    if (!this.clipboard) return [];
    const px = x ?? this.lastPointer.x;
    const py = y ?? this.lastPointer.y;
    this.beginMutate();
    const ids = pasteFragment(this.sim, this.params, this.clipboard, px, py);
    if (ids.length === 0) {
      this.endMutate();
      return [];
    }
    this.selection = new Set(ids);
    this.mark();
    this.endMutate();
    return ids;
  }

  deleteSelection(): boolean {
    if (this.selection.size === 0) return false;
    this.beginMutate();
    for (const id of [...this.selection]) this.sim.kill(id);
    this.selection.clear();
    this.mark();
    this.endMutate();
    return true;
  }

  rotateSelection(da: number): boolean {
    this.pruneSelection();
    if (this.selection.size === 0) return false;
    const c = selectionCentroid(this.sim, this.selection);
    if (!c) return false;
    this.beginMutate();
    rotateAgents(this.sim, this.selection, da, c.x, c.y, this.params);
    this.mark();
    this.endMutate();
    return true;
  }

  cycleSelection(dir: 1 | -1 = 1): boolean {
    this.pruneSelection();
    if (this.selection.size === 0) return false;
    this.beginMutate();
    let any = false;
    for (const id of this.selection) {
      const a = this.sim.agents.get(id);
      if (!a) continue;
      if (this.sim.graph.cycleSlots(id, a.kind, dir)) {
        any = true;
        restitchIncident(this.sim, [id], this.params);
      }
    }
    if (any) this.mark();
    this.endMutate();
    return any;
  }

  /** True when every selected body is pinned. Empty selection is not. */
  selectionIsPinned(): boolean {
    this.pruneSelection();
    if (this.selection.size === 0) return false;
    for (const id of this.selection) {
      const a = this.sim.agents.get(id);
      if (!a || !a.pinned) return false;
    }
    return true;
  }

  /**
   * Pin the selection so Settle cannot move them. If every selected body is
   * already pinned, unpin them instead.
   */
  togglePinSelection(): boolean {
    this.pruneSelection();
    if (this.selection.size === 0) return false;
    const pin = !this.selectionIsPinned();
    this.beginMutate();
    for (const id of this.selection) {
      const a = this.sim.agents.get(id);
      if (!a) continue;
      a.pinned = pin;
      still(a);
    }
    this.mark();
    this.endMutate();
    return true;
  }

  clearWorld(): void {
    this.beginMutate();
    this.sim.clear();
    this.selection.clear();
    this.mark();
    this.endMutate();
  }

  exportText(): string {
    return formatNet(this.sim);
  }

  /** Selection if there is one, otherwise the whole dish. */
  exportPiece(): string {
    this.pruneSelection();
    if (this.selection.size > 0) return formatNet(this.sim, this.selection);
    return formatNet(this.sim);
  }

  importText(src: string): boolean {
    const snap = parseNet(src);
    if (!snap) return false;
    this.beginMutate();
    restore(this.sim, snap, this.params);
    this.selection.clear();
    this.mark();
    this.endMutate();
    return true;
  }

  /** Stamp an HVM2 net at a world point without clearing the dish. */
  insertText(src: string, x?: number, y?: number): number[] {
    const snap = parseNet(src);
    if (!snap) return [];
    const frag = snapshotToFragment(snap);
    if (!frag) return [];
    const px = x ?? this.lastPointer.x;
    const py = y ?? this.lastPointer.y;
    this.beginMutate();
    const ids = pasteFragment(this.sim, this.params, frag, px, py);
    if (ids.length === 0) {
      this.endMutate();
      return [];
    }
    this.selection = new Set(ids);
    this.mark();
    this.endMutate();
    return ids;
  }

  cursorFor(wx: number, wy: number): string {
    const g = this.gesture;
    if (g.kind === 'place') return g.over ? 'crosshair' : 'grabbing';
    if (g.kind === 'pan' || g.kind === 'move' || g.kind === 'rotate') return 'grabbing';
    if (g.kind === 'lasso' || g.kind === 'erase' || g.kind === 'wire') {
      return 'crosshair';
    }
    if (g.kind === 'sound' || this.tool === 'touch') {
      return pickSound(this.sim, wx, wy, this.camera.zoom) || g.kind === 'sound' ? 'pointer' : 'grab';
    }
    if (this.tool === 'pan') return 'grab';
    if (this.tool === 'erase') return 'crosshair';
    if (pickPort(this.sim, wx, wy, this.camera.zoom)) return 'crosshair';
    if (pickAgent(this.sim, wx, wy)) return 'grab';
    if (this.tool === 'rotate' || toolKind(this.tool)) return 'crosshair';
    return 'crosshair';
  }

  begin(
    wx: number,
    wy: number,
    sx: number,
    sy: number,
    opts: { shift?: boolean; pan?: boolean; pressure?: number; now?: number } = {},
  ): void {
    this.lastPointer = { x: wx, y: wy };
    if (opts.pan || this.tool === 'pan') {
      this.gesture = { kind: 'pan', lastX: sx, lastY: sy };
      return;
    }
    if (this.tool === 'touch') {
      const hit = pickSound(this.sim, wx, wy, this.camera.zoom);
      if (hit) {
        this.gesture = {
          kind: 'sound',
          stroke: beginStroke(hit, wx, wy, opts.now ?? 0, opts.pressure ?? 0.5),
        };
        return;
      }
      this.gesture = { kind: 'pan', lastX: sx, lastY: sy };
      return;
    }
    if (this.tool === 'erase') {
      this.beginMutate();
      this.eraseAlong(wx, wy, wx, wy);
      this.gesture = { kind: 'erase', x: wx, y: wy };
      return;
    }
    const port = pickPort(this.sim, wx, wy, this.camera.zoom);
    if (port) {
      this.beginMutate();
      this.gesture = { kind: 'wire', from: port, x: wx, y: wy, over: null };
      return;
    }
    const agent = pickAgent(this.sim, wx, wy);
    if (agent) {
      // Grab always moves the body under the pointer. Shift keeps the rest of
      // the selection so a lassoed net can still be slid as a group.
      if (opts.shift) this.selection.add(agent.id);
      else {
        this.selection.clear();
        this.selection.add(agent.id);
      }
      this.beginMutate();
      this.gesture = { kind: 'move', lastX: wx, lastY: wy };
      return;
    }
    const kind = toolKind(this.tool);
    if (kind) {
      this.beginMutate();
      const placed = this.placeOne(kind, wx, wy);
      if (!placed) {
        this.endMutate();
        return;
      }
      this.selection.clear();
      this.selection.add(placed.id);
      this.gesture = { kind: 'place', id: placed.id, x: wx, y: wy, over: null };
      return;
    }
    if (this.tool === 'rotate') {
      this.ensureHitSelected(wx, wy, opts.shift === true);
      const c = selectionCentroid(this.sim, this.selection);
      if (!c) {
        this.gesture = { kind: 'none' };
        return;
      }
      this.beginMutate();
      this.gesture = { kind: 'rotate', cx: c.x, cy: c.y, lastAng: Math.atan2(wy - c.y, wx - c.x) };
      return;
    }
    const wire = pickWire(this.sim, wx, wy, this.camera.zoom);
    this.gesture = {
      kind: 'lasso',
      points: [{ x: wx, y: wy }],
      additive: opts.shift === true,
      wireId: wire?.id,
    };
  }

  move(
    wx: number,
    wy: number,
    sx: number,
    sy: number,
    opts: { pressure?: number; now?: number } = {},
  ): void {
    this.lastPointer = { x: wx, y: wy };
    const g = this.gesture;
    if (g.kind === 'sound') {
      moveStroke(g.stroke, wx, wy, opts.now ?? g.stroke.lastT, opts.pressure ?? g.stroke.pressure);
      return;
    }
    if (g.kind === 'pan') {
      const dx = sx - g.lastX;
      const dy = sy - g.lastY;
      this.camera.x -= dx / this.camera.zoom;
      this.camera.y -= dy / this.camera.zoom;
      g.lastX = sx;
      g.lastY = sy;
      this.freeCamera = true;
      return;
    }
    if (g.kind === 'erase') {
      this.eraseAlong(g.x, g.y, wx, wy);
      g.x = wx;
      g.y = wy;
      return;
    }
    if (g.kind === 'place') {
      this.aimPlaced(g, wx, wy);
      return;
    }
    if (g.kind === 'lasso') {
      const last = g.points[g.points.length - 1];
      if (!last || Math.hypot(wx - last.x, wy - last.y) > 1 / this.camera.zoom) {
        g.points.push({ x: wx, y: wy });
      }
      return;
    }
    if (g.kind === 'move') {
      const dx = wx - g.lastX;
      const dy = wy - g.lastY;
      if (dx !== 0 || dy !== 0) {
        translateAgents(this.sim, this.selection, dx, dy, this.params);
        this.mark();
      }
      g.lastX = wx;
      g.lastY = wy;
      return;
    }
    if (g.kind === 'rotate') {
      const ang = Math.atan2(wy - g.cy, wx - g.cx);
      const da = ang - g.lastAng;
      if (da !== 0) {
        rotateAgents(this.sim, this.selection, da, g.cx, g.cy, this.params);
        this.mark();
      }
      g.lastAng = ang;
      return;
    }
    if (g.kind === 'wire') {
      g.x = wx;
      g.y = wy;
      const over = pickPort(this.sim, wx, wy, this.camera.zoom);
      g.over = over && over.id !== g.from.id ? over : null;
    }
  }

  end(wx: number, wy: number): void {
    this.lastPointer = { x: wx, y: wy };
    const g = this.gesture;
    this.gesture = { kind: 'none' };
    if (g.kind === 'lasso') {
      this.commitLasso(g.points, g.additive, g.wireId);
      return;
    }
    if (g.kind === 'place') {
      this.aimPlaced(g, wx, wy);
      const target = this.pickAttach(wx, wy, g.id);
      if (target) this.wirePorts({ id: g.id, slot: 'p' }, target);
      this.endMutate();
      return;
    }
    if (g.kind === 'wire') {
      const target = pickPort(this.sim, wx, wy, this.camera.zoom);
      if (target && target.id !== g.from.id) this.wirePorts(g.from, target);
      this.endMutate();
      return;
    }
    if (g.kind === 'erase' || g.kind === 'move' || g.kind === 'rotate') {
      this.endMutate();
    }
  }

  cancel(): void {
    const g = this.gesture;
    this.gesture = { kind: 'none' };
    if (g.kind === 'place' || g.kind === 'erase' || g.kind === 'move' || g.kind === 'rotate' || g.kind === 'wire') {
      this.endMutate();
    }
  }

  drawOverlay(ctx: CanvasRenderingContext2D): void {
    ctx.save();
    this.camera.apply(ctx);
    const zoom = this.camera.zoom;
    const lw = 1.5 / zoom;
    ctx.lineWidth = lw;
    for (const a of this.sim.agents.values()) {
      if (!a.pinned) continue;
      ctx.strokeStyle = '#f0c87e';
      ctx.setLineDash([3 / zoom, 3 / zoom]);
      ctx.beginPath();
      ctx.arc(a.x, a.y, boundRadius(a) + 6 / zoom, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (this.selection.size > 0) {
      ctx.strokeStyle = '#7ef0c8';
      for (const id of this.selection) {
        const a = this.sim.agents.get(id);
        if (!a) continue;
        ctx.beginPath();
        ctx.arc(a.x, a.y, boundRadius(a) + 4 / zoom, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    const g = this.gesture;
    if (g.kind === 'lasso' && g.points.length > 1) {
      ctx.strokeStyle = '#cfd8e3';
      ctx.setLineDash([4 / zoom, 4 / zoom]);
      ctx.beginPath();
      ctx.moveTo(g.points[0].x, g.points[0].y);
      for (let i = 1; i < g.points.length; i++) ctx.lineTo(g.points[i].x, g.points[i].y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (g.kind === 'place') {
      const a = this.sim.agents.get(g.id);
      if (a) {
        const from = portWorld(a, 'p', this.sim.w, this.sim.h);
        const overAg = g.over ? this.sim.agents.get(g.over.id) : null;
        const to = overAg && g.over
          ? portWorld(overAg, g.over.slot, this.sim.w, this.sim.h)
          : { x: g.x, y: g.y };
        ctx.strokeStyle = g.over ? '#7ef0c8' : '#8899aa';
        ctx.setLineDash(g.over ? [] : [4 / zoom, 4 / zoom]);
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(from.x, from.y, 5 / zoom, 0, Math.PI * 2);
        ctx.stroke();
        if (g.over && overAg) {
          ctx.strokeStyle = '#7ef0c8';
          ctx.beginPath();
          ctx.arc(to.x, to.y, 5 / zoom, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
    if (g.kind === 'wire') {
      const from = this.sim.agents.get(g.from.id);
      if (from) {
        const a = portWorld(from, g.from.slot, this.sim.w, this.sim.h);
        ctx.strokeStyle = g.over ? '#7ef0c8' : '#8899aa';
        ctx.setLineDash(g.over ? [] : [4 / zoom, 4 / zoom]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(g.x, g.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    const sound = g.kind === 'sound' ? g.stroke.hit : this.tool === 'touch'
      ? pickSound(this.sim, this.lastPointer.x, this.lastPointer.y, zoom)
      : null;
    if (sound) this.drawSoundHit(ctx, sound, g.kind === 'sound' ? g.stroke : null, zoom);
    if (
      (this.tool === 'select' || this.tool === 'erase') &&
      (g.kind === 'none' || g.kind === 'erase') &&
      !pickPort(this.sim, this.lastPointer.x, this.lastPointer.y, zoom) &&
      !pickAgent(this.sim, this.lastPointer.x, this.lastPointer.y)
    ) {
      const wire = pickWire(this.sim, this.lastPointer.x, this.lastPointer.y, zoom);
      if (wire) {
        ctx.strokeStyle = this.tool === 'erase' ? '#e88' : '#cfd8e3';
        ctx.fillStyle = this.tool === 'erase' ? 'rgba(238, 136, 136, 0.35)' : 'rgba(207, 216, 227, 0.28)';
        ctx.beginPath();
        ctx.arc(wire.x, wire.y, 6 / zoom, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  private drawSoundHit(
    ctx: CanvasRenderingContext2D,
    hit: SoundStroke['hit'],
    stroke: SoundStroke | null,
    zoom: number,
  ): void {
    ctx.strokeStyle = '#f4b183';
    ctx.fillStyle = 'rgba(244, 177, 131, 0.28)';
    const r = 6 / zoom;
    if (hit.kind === 'body') {
      const a = this.sim.agents.get(hit.id);
      if (!a) return;
      ctx.beginPath();
      ctx.arc(a.x, a.y, boundRadius(a) + 5 / zoom, 0, Math.PI * 2);
      ctx.stroke();
    } else if (hit.kind === 'port') {
      ctx.beginPath();
      ctx.arc(hit.x, hit.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(hit.x, hit.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      if (stroke) {
        ctx.setLineDash([3 / zoom, 3 / zoom]);
        ctx.beginPath();
        ctx.moveTo(hit.x, hit.y);
        ctx.lineTo(stroke.x, stroke.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  private ensureHitSelected(wx: number, wy: number, additive: boolean): void {
    const agent = pickAgent(this.sim, wx, wy);
    if (!agent) return;
    if (additive) this.selection.add(agent.id);
    else if (!this.selection.has(agent.id)) {
      this.selection.clear();
      this.selection.add(agent.id);
    }
  }

  private commitLasso(points: { x: number; y: number }[], additive: boolean, wireId?: number): void {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    const slop = CLICK_SLOP / Math.max(0.2, this.camera.zoom);
    const click = points.length < 3 || (maxX - minX < slop && maxY - minY < slop);
    if (click) {
      const p = points[0];
      const hit = p ? pickAgent(this.sim, p.x, p.y) : null;
      if (hit) {
        if (!additive) this.selection.clear();
        if (additive && this.selection.has(hit.id)) this.selection.delete(hit.id);
        else this.selection.add(hit.id);
        return;
      }
      if (wireId !== undefined && this.sim.graph.wires.has(wireId)) {
        this.beginMutate();
        this.sim.graph.detach(wireId);
        this.mark();
        this.endMutate();
        return;
      }
      if (!additive) this.selection.clear();
      return;
    }
    if (!additive) this.selection.clear();
    for (const a of this.sim.agents.values()) {
      if (pointInPolygon(a.x, a.y, points)) this.selection.add(a.id);
    }
  }

  private wirePorts(a: PortRef, b: PortRef): boolean {
    if (a.id === b.id) return false;
    const wire = this.sim.graph.connect(
      this.sim.agents,
      a,
      b,
      this.sim.w,
      this.sim.h,
      this.params,
      this.sim.time,
      { silent: true },
    );
    if (!wire) return false;
    this.mark();
    return true;
  }

  /** Free port under the pointer, preferring a real nub but accepting the body. */
  private pickAttach(x: number, y: number, ignoreId: number): PortRef | null {
    const port = pickPort(this.sim, x, y, this.camera.zoom);
    if (port && port.id !== ignoreId) return port;
    const agent = pickAgent(this.sim, x, y);
    if (!agent || agent.id === ignoreId) return null;
    let best: PortRef | null = null;
    let bestD = Infinity;
    for (const slot of slotsFor(agent.kind)) {
      if (!this.sim.graph.isFreeAt(agent.id, slot)) continue;
      const p = portWorld(agent, slot, this.sim.w, this.sim.h);
      const d = (p.x - x) ** 2 + (p.y - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = { id: agent.id, slot };
      }
    }
    return best;
  }

  private placeOne(kind: AgentKind, x: number, y: number): Agent | null {
    const spacing = paintSpacing(this.params);
    if (agentTooClose(this.sim, x, y, spacing)) return null;
    const a = spawnDesigned(this.sim, kind, x, y, 0, this.params);
    if (!a) return null;
    this.mark();
    return a;
  }

  private aimPlaced(g: Extract<Gesture, { kind: 'place' }>, wx: number, wy: number): void {
    g.x = wx;
    g.y = wy;
    g.over = this.pickAttach(wx, wy, g.id);
    const a = this.sim.agents.get(g.id);
    if (!a) return;
    let tx = wx;
    let ty = wy;
    if (g.over) {
      const other = this.sim.agents.get(g.over.id);
      if (other) {
        const p = portWorld(other, g.over.slot, this.sim.w, this.sim.h);
        tx = p.x;
        ty = p.y;
      }
    }
    const dx = tx - a.x;
    const dy = ty - a.y;
    if (Math.hypot(dx, dy) < CLICK_SLOP / Math.max(0.2, this.camera.zoom)) return;
    a.heading = Math.atan2(dy, dx);
    still(a);
  }

  private eraseAlong(x0: number, y0: number, x1: number, y1: number): void {
    const reach = ERASE_BRUSH / Math.max(0.2, this.camera.zoom);
    const dead: number[] = [];
    for (const agent of this.sim.agents.values()) {
      const p = closestPointOnSegment(agent.x, agent.y, x0, y0, x1, y1);
      const d = Math.hypot(agent.x - p.x, agent.y - p.y);
      if (d <= boundRadius(agent) + reach) dead.push(agent.id);
    }
    const cut: number[] = [];
    for (const w of this.sim.graph.wires.values()) {
      const A = this.sim.agents.get(w.a.id);
      const B = this.sim.agents.get(w.b.id);
      if (!A || !B) continue;
      const a = stemWorld(A, w.a.slot, this.sim.w, this.sim.h);
      const b = stemWorld(B, w.b.slot, this.sim.w, this.sim.h);
      if (segmentsInterfere(x0, y0, x1, y1, a.x, a.y, b.x, b.y, reach)) cut.push(w.id);
    }
    if (dead.length === 0 && cut.length === 0) return;
    for (const id of dead) this.sim.kill(id);
    for (const id of cut) {
      if (this.sim.graph.wires.has(id)) this.sim.graph.detach(id);
    }
    this.mark();
    this.pruneSelection();
  }
}
