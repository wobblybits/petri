import type { PanView } from './types.ts';
import { CAMERA_MIN_ZOOM } from '../camera.ts';

/**
 * How much of an object the listener is entitled to hear in full. The
 * selector is apparent size on screen, not loudness, which lines up the audio,
 * render and pick budgets. Thresholds decide what deserves detail and the
 * ranked budget in `assign` decides how much the frame can afford; wires
 * overlap, so the screen alone does not bound the cost. Nothing is ever
 * silenced: an object that misses the budget drops a tier and is still heard.
 */
export const LOD_NEAR = 0;
export const LOD_MID = 1;
export const LOD_FAR = 2;

export type LodTier = typeof LOD_NEAR | typeof LOD_MID | typeof LOD_FAR;

export interface LodBand {
  /** Apparent size, in screen px, at or above which an object is NEAR. */
  near: number;
  /** Apparent size, in screen px, at or above which an object is MID. */
  mid: number;
}

/** Wires earn full treatment while their travelling wave is legible; bodies hold detail down to a smaller footprint. */
export const WIRE_BAND: LodBand = { near: 48, mid: 12 };
export const AGENT_BAND: LodBand = { near: 20, mid: 6 };

/** World-space stroke used for wires. Screen width is this times zoom. */
export const WIRE_STROKE_PX = 1.35;
/**
 * Below this many screen pixels a wire is a hairline: skip drawing it, and
 * drop the live rope for a chord span. Set to the stroke width at
 * `CAMERA_MIN_ZOOM` so a fully zoomed-out net still draws as a net.
 */
export const WIRE_HAIRLINE_PX = CAMERA_MIN_ZOOM * WIRE_STROKE_PX;

/** False when zoom has shrunk the stroke below a readable hairline. */
export function wiresDrawable(zoom: number): boolean {
  if (!(zoom > 0)) return true;
  return zoom * WIRE_STROKE_PX >= WIRE_HAIRLINE_PX;
}

/**
 * Fraction of a threshold an object must clear to move up a tier, and fall
 * short of to move down; without it a boundary object flutters every frame.
 */
export const HYSTERESIS = 0.25;

/** Apparent size in screen px of something `worldSize` px across. */
export function apparentPx(worldSize: number, view: PanView | null | undefined): number {
  if (!view || !(view.zoom > 0)) return worldSize;
  return worldSize * view.zoom;
}

/** Whether any part of a box `worldSize` across, centred at (wx, wy), falls inside the viewport. */
export function onScreen(
  wx: number,
  wy: number,
  worldSize: number,
  view: PanView | null | undefined,
): boolean {
  if (!view || !(view.zoom > 0) || !(view.viewW > 0) || !(view.viewH > 0)) return true;
  const sx = Math.abs(wx - view.x) * view.zoom;
  const sy = Math.abs(wy - view.y) * view.zoom;
  const margin = Math.max(0, worldSize) * view.zoom * 0.5;
  return sx <= view.viewW * 0.5 + margin && sy <= view.viewH * 0.5 + margin;
}

/** Tier for an object of a given apparent size, ignoring hysteresis. */
export function tierFor(px: number, visible: boolean, band: LodBand): LodTier {
  if (!visible) return LOD_FAR;
  if (px >= band.near) return LOD_NEAR;
  if (px >= band.mid) return LOD_MID;
  return LOD_FAR;
}

/**
 * Tiering with memory: promotion needs the size to clear the threshold by
 * `HYSTERESIS`; demotion needs it to fall the same margin below.
 */
export class LodSelector {
  private prev = new Map<number, LodTier>();
  private live = new Set<number>();

  /** Tier for one object. `key` must be unique across the whole net. */
  tier(key: number, px: number, visible: boolean, band: LodBand): LodTier {
    this.live.add(key);
    const was = this.prev.get(key);
    if (was === undefined) {
      const fresh = tierFor(px, visible, band);
      this.prev.set(key, fresh);
      return fresh;
    }
    if (!visible) {
      this.prev.set(key, LOD_FAR);
      return LOD_FAR;
    }
    const up = 1 + HYSTERESIS;
    const down = 1 - HYSTERESIS;
    let next = was;
    if (was === LOD_FAR) {
      if (px >= band.near * up) next = LOD_NEAR;
      else if (px >= band.mid * up) next = LOD_MID;
    } else if (was === LOD_MID) {
      if (px >= band.near * up) next = LOD_NEAR;
      else if (px < band.mid * down) next = LOD_FAR;
    } else {
      if (px < band.mid * down) next = LOD_FAR;
      else if (px < band.near * down) next = LOD_MID;
    }
    this.prev.set(key, next);
    return next;
  }

  /** What this key was last given, without recording a fresh visit. */
  peek(key: number): LodTier | undefined {
    return this.prev.get(key);
  }

  /** Overwrite what a key holds — how the budget records a forced demotion. */
  force(key: number, tier: LodTier): void {
    this.prev.set(key, tier);
    this.live.add(key);
  }

  /** Forget objects that were not tiered this pass. Called once a frame. */
  sweep(): void {
    if (this.live.size === this.prev.size) {
      this.live.clear();
      return;
    }
    for (const key of this.prev.keys()) {
      if (!this.live.has(key)) this.prev.delete(key);
    }
    this.live.clear();
  }

  clear(): void {
    this.prev.clear();
    this.live.clear();
  }

  get size(): number {
    return this.prev.size;
  }
}

/** Wire ids and agent ids are separate spaces; keep them apart in one map. */
export function wireKey(id: number): number {
  return id * 2;
}

export function agentKey(id: number): number {
  return id * 2 + 1;
}

/**
 * What one awake object costs per 128-sample quantum, in microseconds
 * (`perf-probe.perf.test.ts`). They set the shape of the ranking; the budget
 * below absorbs the error.
 */
export const COST_US = {
  nearWire: 19.6,
  midWire: 3.2,
  nearAgent: 8.3,
  midAgent: 2,
};

/** One render quantum at 48 kHz. */
export const QUANTUM_US = (128 / 48000) * 1e6;

/**
 * Share of the quantum the tiered voices may spend. The rest pays for the
 * parts that do not tier: the floor, contacts, air, the ensemble and the
 * master chain, so a busy frame degrades detail instead of missing the deadline.
 */
export const VOICE_BUDGET_US = QUANTUM_US * 0.6;

export interface LodCandidate {
  key: number;
  /** Apparent size in screen px. */
  px: number;
  visible: boolean;
  band: LodBand;
  /** Cost of this object at NEAR, in us per quantum. */
  nearUs: number;
  /** Cost of this object at MID. */
  midUs: number;
}

/**
 * Tier a whole frame's worth of objects against a cost budget. Candidates
 * are ranked by apparent size and promoted from the top down, so the
 * smallest things lose detail first. Missing the budget is a demotion, never
 * a mute: the object still sounds as part of the ensemble.
 */
export function assign(
  cands: LodCandidate[],
  selector: LodSelector | null = null,
  budgetUs = VOICE_BUDGET_US,
): Map<number, LodTier> {
  const out = new Map<number, LodTier>();
  const ranked: LodCandidate[] = [];

  for (const c of cands) {
    // Below the mid threshold, or not on screen: these never compete for budget.
    const deserved = selector
      ? selector.tier(c.key, c.px, c.visible, c.band)
      : tierFor(c.px, c.visible, c.band);
    if (deserved === LOD_FAR) {
      out.set(c.key, LOD_FAR);
      continue;
    }
    ranked.push(c);
  }

  ranked.sort((a, b) => b.px - a.px);

  let spent = 0;
  for (const c of ranked) {
    const deserved = selector ? (selector.peek(c.key) ?? LOD_MID) : tierFor(c.px, true, c.band);
    if (deserved === LOD_NEAR && spent + c.nearUs <= budgetUs) {
      spent += c.nearUs;
      out.set(c.key, LOD_NEAR);
    } else if (spent + c.midUs <= budgetUs) {
      spent += c.midUs;
      out.set(c.key, LOD_MID);
    } else {
      out.set(c.key, LOD_FAR);
    }
  }

  // Remember what actually plays, so a budget-forced demotion does not flip back next frame.
  if (selector) for (const [key, tier] of out) selector.force(key, tier);
  return out;
}

/** Total cost of an assignment, in us per quantum. Used by tests and probes. */
export function assignedCostUs(cands: LodCandidate[], tiers: Map<number, LodTier>): number {
  let us = 0;
  for (const c of cands) {
    const t = tiers.get(c.key);
    if (t === LOD_NEAR) us += c.nearUs;
    else if (t === LOD_MID) us += c.midUs;
  }
  return us;
}
