import { boundRadius, portWorld, slotsFor, type Agent, type PortRef } from './agents.ts';
import type { Camera } from './camera.ts';
import type { Sim } from './sim.ts';

/** How close the pointer must be to a port, in world units, to grab it. */
const PORT_PICK = 9;

/** Pointer travel past which a press stops counting as a click. */
const CLICK_SLOP = 4;

export type Gesture =
  | { kind: 'none' }
  | { kind: 'pan'; lastX: number; lastY: number; moved: number }
  | { kind: 'drag'; id: number }
  | { kind: 'wire'; from: PortRef; x: number; y: number; over: PortRef | null };

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
 */
export class Interaction {
  gesture: Gesture = { kind: 'none' };
  /** True once the user has panned; the camera stops chasing the flock. */
  freeCamera = false;

  private sim: Sim;
  private camera: Camera;

  constructor(sim: Sim, camera: Camera) {
    this.sim = sim;
    this.camera = camera;
  }

  begin(wx: number, wy: number, sx: number, sy: number): void {
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

  /** Cursor hint for the current hover. */
  cursorFor(wx: number, wy: number): string {
    if (this.gesture.kind === 'pan') return 'grabbing';
    if (this.gesture.kind !== 'none') return 'grabbing';
    if (pickPort(this.sim, wx, wy, this.camera.zoom)) return 'crosshair';
    if (pickAgent(this.sim, wx, wy)) return 'grab';
    return 'default';
  }
}
