import type { PanView } from './types.ts';
import { CAMERA_MIN_ZOOM } from '../camera.ts';

/**
 * How much of an object the listener is entitled to hear in full.
 *
 * The selector is apparent size on screen, not loudness. Loudness looks like
 * the obvious criterion and is the wrong one here: a quiet wire you are
 * staring at would be demoted while a loud one somewhere off-screen was
 * promoted, so the thing under the cursor is the thing being approximated.
 * Size does not have that failure, and it also lines up the three budgets
 * that have to agree — a wire too small to render a travelling wave is too
 * small to pick out of the mix and too small to grab with the mouse.
 *
 * Size gets the policy right but does not, on its own, bound the cost. It is
 * tempting to argue that the screen bounds it — so many pixels, so many
 * resolvable wires — and that is wrong, because wires overlap. A dense tangle
 * of long wires in one view is legitimately all NEAR at once, and a fixed
 * threshold would happily promote every one of them.
 *
 * So thresholds decide what *deserves* detail, and the ranked budget in
 * `assign` decides how much of that the frame can afford: candidates sorted
 * by apparent size, promoted from the top until the budget runs out. Nothing
 * is ever silenced — an object that misses the budget drops a tier and is
 * still heard, which is the difference between this and a voice cap.
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

/**
 * Wires carry a visible travelling wave, so they earn full treatment while
 * that wave is still legible — roughly a wire long enough to show a couple of
 * arcs. Bodies are small by nature and are the voice of the net, so they hold
 * their detail down to a smaller footprint.
 */
export const WIRE_BAND: LodBand = { near: 48, mid: 12 };
export const AGENT_BAND: LodBand = { near: 20, mid: 6 };

/** World-space stroke used for wires. Screen width is this times zoom. */
export const WIRE_STROKE_PX = 1.35;
/**
 * Below this many screen pixels a wire is a hairline — skip drawing it, and
 * drop the live rope for a chord span. Canvas will still rasterize a 0.05 px
 * cubic as a device-pixel streak across the view.
 *
 * Set to the stroke width at `CAMERA_MIN_ZOOM` so a fully zoomed-out net still
 * draws as a net. Live ropes do not come along for the ride: they also need a
 * detailed body, and bodies have already gone FAR by then.
 */
export const WIRE_HAIRLINE_PX = CAMERA_MIN_ZOOM * WIRE_STROKE_PX;

/** False when zoom has shrunk the stroke below a readable hairline. */
export function wiresDrawable(zoom: number): boolean {
  if (!(zoom > 0)) return true;
  return zoom * WIRE_STROKE_PX >= WIRE_HAIRLINE_PX;
}

/**
 * Fraction of a threshold an object must clear to move up a tier, and fall
 * short of to move down. Without it an object hovering exactly on a boundary
 * changes representation every frame, which is audible as a flutter even when
 * each individual crossfade is clean.
 */
export const HYSTERESIS = 0.25;

/** Apparent size in screen px of something `worldSize` px across. */
export function apparentPx(worldSize: number, view: PanView | null | undefined): number {
  if (!view || !(view.zoom > 0)) return worldSize;
  return worldSize * view.zoom;
}

/**
 * Whether any part of a box `worldSize` across, centred at (wx, wy), falls
 * inside the viewport. Off-screen is the strongest demotion signal there is:
 * nothing off-screen can be looked at, grabbed, or read off the wave display.
 */
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
 * Tiering with memory, so an object sitting on a boundary keeps whatever it
 * had rather than oscillating. Promotion needs the size to clear the
 * threshold by `HYSTERESIS`; demotion needs it to fall the same margin below.
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

  /**
   * Forget objects that were not tiered this pass. Called once a frame so a
   * net that churns through thousands of ids does not accumulate them.
   */
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
 * What one awake object costs per 128-sample quantum, in microseconds.
 *
 * NEAR figures are measured on this machine: a ringing waveguide wire is
 * ~19.6 us and a ringing three-mode body ~8.3 us, taken as the marginal cost
 * of the Nth object with the rest of the net held still. MID figures are the
 * modal stand-ins, which are a small fixed filter bank rather than a delay
 * line with interpolation, dispersion and a loop filter.
 *
 * They do not need to be exact. They set the shape of the ranking — a wire
 * costs a couple of bodies — and the budget below absorbs the error.
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
 * Share of the quantum the tiered voices may spend.
 *
 * The rest pays for the parts that do not tier: the ~500 us floor a live net
 * costs with nothing sounding, contacts, air, the ensemble, and the master
 * chain. Sitting at 60% leaves real headroom, which is the point — the budget
 * exists so a busy frame degrades detail instead of missing the deadline.
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
 * Tier a whole frame's worth of objects against a cost budget.
 *
 * Thresholds say what an object deserves; the budget says how much of that
 * the frame can pay for. Candidates are ranked by apparent size and promoted
 * from the top down, so when the budget binds it is the smallest things that
 * lose detail — which is both the cheapest place to lose it and the place
 * nobody is looking.
 *
 * Missing the budget is a demotion, never a mute. An object that gets nothing
 * here still sounds, as part of the ensemble.
 */
export function assign(
  cands: LodCandidate[],
  selector: LodSelector | null = null,
  budgetUs = VOICE_BUDGET_US,
): Map<number, LodTier> {
  const out = new Map<number, LodTier>();
  const ranked: LodCandidate[] = [];

  for (const c of cands) {
    // Below the mid threshold, or not on screen: no amount of budget buys
    // detail worth having, so these never compete for it.
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

  // Remember what actually plays, not what was deserved, so a budget-forced
  // demotion does not flip straight back next frame.
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
