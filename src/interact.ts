import { boundRadius, portWorld, slotsFor, type Agent, type PortRef } from './agents.ts';
import type { Camera } from './camera.ts';
import { closestPointOnSegment } from './geom.ts';
import type { Sim } from './sim.ts';

/** How close the pointer must be to a port, in world units, to grab it. */
const PORT_PICK = 9;

/**
 * Extra reach the eraser brush gets past an agent's own `boundRadius`.
 *
 * `boundRadius` bounds the triangle body, not the port stems reaching past
 * it — a Con or Dup's principal tip sits about 25px from centre against a
 * ~18px bound — so brushing the visible port nub of a body should still
 * erase it. The rest is slack for a fast drag across closely-spaced bodies,
 * whose centres the segment can miss by a pixel or two between samples.
 */
const ERASE_BRUSH = 10;

/** Pointer travel past which a press stops counting as a click. */
const CLICK_SLOP = 4;

/** Energy added to a cell the first time a paint stroke covers it. */
export const PAINT_DEPOSIT = 1;

export type Tool = 'none' | 'erase' | 'paint' | 'splat';

export type Gesture =
  | { kind: 'none' }
  | { kind: 'pan'; lastX: number; lastY: number; moved: number }
  | { kind: 'drag'; id: number }
  | { kind: 'wire'; from: PortRef; x: number; y: number; over: PortRef | null }
  | { kind: 'erase'; x: number; y: number }
  | { kind: 'paint'; x: number; y: number; visited: Set<number> }
  | { kind: 'splat' };

/** Free port nearest the pointer, if one is close enough to mean it. */
export function pickPort(sim: Sim, x: number, y: number, zoom: number): PortRef | null {
  const reach = PORT_PICK / Math.max(0.2, zoom);
  let best: PortRef | null = null;
  let bestD = reach * reach;
  for (const agent of sim.agents.values()) {
    if (agent.locked) continue;
    for (const slot of slotsFor(agent.kind)) {
      const ref: PortRef = { id: agent.id, slot };
      if (!sim.graph.isFree(ref)) continue;
      const p = portWorld(agent, slot, sim.w, sim.h);
      const d = (p.x - x) ** 2 + (p.y - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = ref;
      }
    }
  }
  return best;
}

/** Body under the pointer, if any. */
export function pickAgent(sim: Sim, x: number, y: number): Agent | null {
  let best: Agent | null = null;
  let bestD = Infinity;
  for (const agent of sim.agents.values()) {
    const r = boundRadius(agent);
    const d = (agent.x - x) ** 2 + (agent.y - y) ** 2;
    if (d <= r * r && d < bestD) {
      bestD = d;
      best = agent;
    }
  }
  return best;
}

/**
 * Pointer gestures over the canvas, chosen by what is under the cursor rather
 * than by a mode switch: a free port starts a wire, a body drags it, and empty
 * space pans. A press that never really moves is still a click, so spawning
 * keeps working.
 *
 * `tool` overrides that pick: erase and paint are strokes, splat is one drop
 * per press (the caller supplies `onSplat`).
 */
export class Interaction {
  gesture: Gesture = { kind: 'none' };
  /** True once the user has panned; the camera stops chasing the flock. */
  freeCamera = false;
  tool: Tool = 'none';
  /** Fired once on pointer-down while the splat tool is selected. */
  onSplat: ((x: number, y: number) => void) | null = null;

  /**
   * Lab checkbox compatibility. Setting false always returns to the default
   * pick, even if paint/splat was active — the lab only ever toggles erase.
   */
  get eraserMode(): boolean {
    return this.tool === 'erase';
  }
  set eraserMode(on: boolean) {
    this.tool = on ? 'erase' : 'none';
  }

  private sim: Sim;
  private camera: Camera;

  constructor(sim: Sim, camera: Camera) {
    this.sim = sim;
    this.camera = camera;
  }

  begin(wx: number, wy: number, sx: number, sy: number): void {
    if (this.tool === 'erase') {
      this.eraseAlong(wx, wy, wx, wy);
      this.gesture = { kind: 'erase', x: wx, y: wy };
      return;
    }
    if (this.tool === 'paint') {
      const visited = new Set<number>();
      this.paintAlong(wx, wy, wx, wy, visited);
      this.gesture = { kind: 'paint', x: wx, y: wy, visited };
      return;
    }
    if (this.tool === 'splat') {
      this.onSplat?.(wx, wy);
      this.gesture = { kind: 'splat' };
      return;
    }
    const port = pickPort(this.sim, wx, wy, this.camera.zoom);
    if (port) {
      this.gesture = { kind: 'wire', from: port, x: wx, y: wy, over: null };
      return;
    }
    const agent = pickAgent(this.sim, wx, wy);
    if (agent) {
      this.sim.grabbed = { id: agent.id, x: wx, y: wy };
      this.gesture = { kind: 'drag', id: agent.id };
      return;
    }
    this.gesture = { kind: 'pan', lastX: sx, lastY: sy, moved: 0 };
  }

  move(wx: number, wy: number, sx: number, sy: number): void {
    const g = this.gesture;
    if (g.kind === 'erase') {
      this.eraseAlong(g.x, g.y, wx, wy);
      g.x = wx;
      g.y = wy;
      return;
    }
    if (g.kind === 'paint') {
      this.paintAlong(g.x, g.y, wx, wy, g.visited);
      g.x = wx;
      g.y = wy;
      return;
    }
    if (g.kind === 'splat') return;
    if (g.kind === 'pan') {
      const dx = sx - g.lastX;
      const dy = sy - g.lastY;
      g.moved += Math.abs(dx) + Math.abs(dy);
      if (g.moved > CLICK_SLOP) {
        this.freeCamera = true;
        this.camera.x -= dx / this.camera.zoom;
        this.camera.y -= dy / this.camera.zoom;
      }
      g.lastX = sx;
      g.lastY = sy;
      return;
    }
    if (g.kind === 'drag') {
      if (this.sim.grabbed) {
        this.sim.grabbed.x = wx;
        this.sim.grabbed.y = wy;
      }
      return;
    }
    if (g.kind === 'wire') {
      g.x = wx;
      g.y = wy;
      const over = pickPort(this.sim, wx, wy, this.camera.zoom);
      g.over = over && over.id !== g.from.id ? over : null;
    }
  }

  /** Returns the world point to spawn at when the press was really a click. */
  end(wx: number, wy: number, connect: (a: PortRef, b: PortRef) => void): { x: number; y: number } | null {
    const g = this.gesture;
    this.gesture = { kind: 'none' };
    this.sim.grabbed = null;
    if (g.kind === 'pan') return g.moved <= CLICK_SLOP ? { x: wx, y: wy } : null;
    if (g.kind === 'wire') {
      const target = pickPort(this.sim, wx, wy, this.camera.zoom);
      if (target && target.id !== g.from.id) connect(g.from, target);
    }
    return null;
  }

  cancel(): void {
    this.gesture = { kind: 'none' };
    this.sim.grabbed = null;
  }

  /**
   * Kill every agent whose glyph the brush swept between the last point and
   * this one. Collected before killing rather than removed mid-scan: `kill`
   * mutates the same agents map this is iterating, and a starved body's own
   * death yield lands back on the grid, which is easier to reason about as a
   * clean batch than interleaved with the scan that found it.
   */
  private eraseAlong(x0: number, y0: number, x1: number, y1: number): void {
    const reach = ERASE_BRUSH / Math.max(0.2, this.camera.zoom);
    const dead: number[] = [];
    for (const agent of this.sim.agents.values()) {
      const p = closestPointOnSegment(agent.x, agent.y, x0, y0, x1, y1);
      const d = Math.hypot(agent.x - p.x, agent.y - p.y);
      if (d <= boundRadius(agent) + reach) dead.push(agent.id);
    }
    for (const id of dead) this.sim.kill(id);
  }

  /**
   * Deposit `PAINT_DEPOSIT` into every energy cell whose centre is within one
   * cell of the stroke. `visited` is the stroke's own set — hovering the same
   * cell does not stack, a new press does.
   */
  private paintAlong(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    visited: Set<number>,
  ): void {
    const grid = this.sim.energy;
    const size = grid.cellSize;
    const radius = size;
    const { x: ox, y: oy } = grid.lattice;
    const pad = radius + size;
    const minX = Math.min(x0, x1) - pad;
    const maxX = Math.max(x0, x1) + pad;
    const minY = Math.min(y0, y1) - pad;
    const maxY = Math.max(y0, y1) + pad;
    const i0 = Math.floor((minX - ox) / size);
    const i1 = Math.ceil((maxX - ox) / size);
    const j0 = Math.floor((minY - oy) / size);
    const j1 = Math.ceil((maxY - oy) / size);
    for (let i = i0; i < i1; i++) {
      for (let j = j0; j < j1; j++) {
        const cx = ox + (i + 0.5) * size;
        const cy = oy + (j + 0.5) * size;
        const p = closestPointOnSegment(cx, cy, x0, y0, x1, y1);
        if (Math.hypot(cx - p.x, cy - p.y) > radius) continue;
        const { key } = grid.index(cx, cy);
        if (visited.has(key)) continue;
        visited.add(key);
        grid.addAt(cx, cy, PAINT_DEPOSIT);
      }
    }
  }

  /** Cursor hint for the current hover. */
  cursorFor(wx: number, wy: number): string {
    if (this.tool !== 'none') return 'crosshair';
    if (this.gesture.kind === 'pan') return 'grabbing';
    if (this.gesture.kind !== 'none') return 'grabbing';
    if (pickPort(this.sim, wx, wy, this.camera.zoom)) return 'crosshair';
    if (pickAgent(this.sim, wx, wy)) return 'grab';
    return 'default';
  }
}
