import { boundRadius, ERA_RADIUS, ERA_SLOTS, NODE_SLOTS, portLocal, stemRoot, stemWorld, stemWorldInto, triangleLocal, wireCubic, Agent, type AgentKind } from './agents.ts';
import { AgentStore } from './agent-store.ts';
import { EXTRA_CAP, EXTRA_FLOOR, REQUEST_DECAY, REQUEST_FULL } from './energy.ts';
import { WIRE_STROKE_PX, wiresDrawable } from './audio/lod.ts';
import { clampPolylineToChord, WAVE_DISP_PX, wireBowBudget } from './geom.ts';
import type { WaveSnapshot } from './audio/types.ts';
import type { Camera } from './camera.ts';
import { catmullSegment, unwrapPoints } from './chain.ts';
import { bezierPoint, bezierPointInto } from './curve.ts';
import type { Fields } from './fields.ts';
import type { Graph, Wire } from './graph.ts';
import {
  COMMUTE_K22,
  commuteGhost,
  rewriteHandoffStems,
  stemFromGhost,
  type Ghost,
  type Rewrite,
} from './rewrite.ts';
import type { Sim } from './sim.ts';

export interface ViewOpts {
  overlay: boolean;
  energyGrid: boolean;
  energyCircles: boolean;
  /** Kind hues with energy as saturation, instead of a ring around each body. */
  kindColors: boolean;
  /**
   * When true, FAR-tier agents are skipped here entirely — the caller is
   * expected to have already drawn them as instanced dots on a WebGPU
   * canvas layered *on top* of this one (see buildFarInstances / AgentsGpu),
   * which is one draw call regardless of count instead of one drawAgent()
   * call per agent, and sits above everything here — including wires and
   * whatever this draws for the energy grid — so a FAR-tier dot can never
   * end up buried under either. False draws everyone here, exactly as
   * before the GPU layer existed — the safe default when that layer isn't
   * ready.
   */
  gpuAgents: boolean;
}

/** Floats per instance in the GPU layer's buffer — see agents.wgsl. */
export const FAR_INSTANCE_STRIDE = 9;

/** Era is round; Con and Dup get the same triangle drawTriangle() draws. */
function shapeFor(kind: AgentKind): number {
  return kind === 'era' ? 0 : 1;
}

/**
 * Fills `out` with one FAR_INSTANCE_STRIDE-float record per FAR-tier agent
 * and returns how many were written. `out` must be at least
 * `sim.agents.size * FAR_INSTANCE_STRIDE` long; the caller owns growing it
 * (main.ts keeps one reusable buffer, matching the pattern GravitatePool's
 * scratch array already uses).
 */
export function buildFarInstances(sim: Sim, out: Float32Array, kindColors: boolean): number {
  let n = 0;
  for (const agent of sim.agents.values()) {
    if (!sim.isFarTier(agent.id)) continue;
    const base = n * FAR_INSTANCE_STRIDE;
    if (base + FAR_INSTANCE_STRIDE > out.length) break;
    // Numeric fields read straight off the store by slot, bypassing Agent's
    // getters — this walks the whole FAR-tier population every frame. `kind`
    // stays a getter call: kindFillRgb/KIND_RGB/shapeFor all key on the
    // string, and boundRadius needs the Agent shape regardless.
    const store = agent.store;
    const slot = agent.slot;
    const kind = agent.kind;
    const extra = store.extra[slot];
    const rgb = agentFillRgb(kind, extra, kindColors);
    out[base + 0] = store.x[slot];
    out[base + 1] = store.y[slot];
    out[base + 2] = boundRadius(agent);
    out[base + 3] = store.heading[slot];
    out[base + 4] = shapeFor(kind);
    out[base + 5] = rgb[0] / 255;
    out[base + 6] = rgb[1] / 255;
    out[base + 7] = rgb[2] / 255;
    out[base + 8] = store.alpha[slot];
    n++;
  }
  return n;
}

let overlayCanvas: HTMLCanvasElement | null = null;
let overlayCtx: CanvasRenderingContext2D | null = null;

function overlayFor(fields: Fields): CanvasRenderingContext2D {
  if (!overlayCanvas || overlayCanvas.width !== fields.cols || overlayCanvas.height !== fields.rows) {
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.width = fields.cols;
    overlayCanvas.height = fields.rows;
    overlayCtx = overlayCanvas.getContext('2d');
  }
  return overlayCtx!;
}

export function render(
  ctx: CanvasRenderingContext2D,
  sim: Sim,
  camera: Camera,
  opts: ViewOpts,
  waves: WaveSnapshot | null = null,
): void {
  ctx.save();
  // The background fill. This also doubles as Canvas2D's per-frame clear —
  // it never clears itself between draws, so this has to run unconditionally
  // every frame regardless of whether the GPU dot layer is active.
  //
  // This canvas is the *bottom* layer (#view-gpu sits on top of it, not
  // under — see #viewport in style.css): wires, near-tier agents, and
  // whatever else this draws should never be able to bury a FAR-tier dot,
  // which an opaque fill on top of it would. The GPU layer's own clear
  // (agents-gpu.ts) is to transparent, so it shows this canvas through
  // everywhere it isn't drawing a dot.
  ctx.fillStyle = '#0c0d10';
  ctx.fillRect(0, 0, camera.viewW, camera.viewH);

  ctx.save();
  camera.apply(ctx);

  /*
   * Both of the views below paint from `sim.fields.data`, which is a stale
   * copy when the field lives on the GPU. Asking for it here rather than
   * inside them means the sim does one copy a frame while either is open and
   * none at all when neither is — and it is asked for a frame ahead, since
   * the copy happens during the step and this runs after it.
   */
  sim.wantFieldReadback = opts.energyGrid === true || opts.overlay === true;

  if (opts.energyGrid) drawEnergyGrid(ctx, sim, camera);

  if (opts.overlay) {
    const octx = overlayFor(sim.fields);
    const img = octx.createImageData(sim.fields.cols, sim.fields.rows);
    sim.fields.paintOverlay(img);
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = 0.85;
    ctx.drawImage(
      overlayCanvas!,
      sim.fields.originX,
      sim.fields.originY,
      sim.fields.worldW,
      sim.fields.worldH,
    );
    ctx.globalAlpha = 1;
  }

  if (wiresDrawable(camera.zoom)) {
    drawWires(ctx, sim, waves, camera.zoom);
    for (const rw of sim.rewrites) {
      drawCommuteGhostWires(ctx, rw, sim.w, sim.h);
    }
  }
  for (const rw of sim.rewrites) {
    for (const g of rw.ghosts) drawAgent(ctx, ghostAsAgent(g), 1, undefined, opts.kindColors);
  }
  for (const agent of sim.agents.values()) {
    if (opts.gpuAgents && sim.isFarTier(agent.id)) continue;
    if (opts.energyCircles && !opts.kindColors) drawEnergySlot(ctx, agent);
    drawAgent(ctx, agent, agent.alpha, sim.graph, opts.kindColors);
  }
  ctx.restore();
  ctx.restore();
}

function drawEnergyGrid(ctx: CanvasRenderingContext2D, sim: Sim, camera: Camera): void {
  const grid = sim.energy;
  const size = grid.cellSize;
  const { x: originX, y: originY } = grid.lattice;
  const ambient = Math.max(0.001, grid.ambient);
  const pad = size;
  const left = camera.x - camera.coverWidth() * 0.5 - pad;
  const top = camera.y - camera.coverHeight() * 0.5 - pad;
  const right = camera.x + camera.coverWidth() * 0.5 + pad;
  const bottom = camera.y + camera.coverHeight() * 0.5 + pad;
  // Cell (i, j) covers world space [originX + i*size, ...), not [i*size, ...)
  // — the grid's lattice is anchored to the scent field's origin, not to the
  // world origin, so the cull window and the draw position both need it.
  let i0 = Math.floor((left - originX) / size);
  let i1 = Math.ceil((right - originX) / size);
  let j0 = Math.floor((top - originY) / size);
  let j1 = Math.ceil((bottom - originY) / size);
  /*
   * And clamped to the dish, which the camera window alone is not.
   *
   * At the minimum zoom the cover is 72,000 world units across — 1802 x 1127
   * cells, two million of them, and all but the 256 x 256 that sit over the
   * field read 0 and so draw a black rect each. That is exactly the freeze the
   * comment below warns about. The draw is already clipped to this same disk,
   * so every cell dropped here was invisible anyway; the bounding box is a
   * superset of the disk, so nothing inside it is lost either.
   */
  if (sim.worldR > 0) {
    i0 = Math.max(i0, Math.floor((sim.worldX - sim.worldR - originX) / size));
    i1 = Math.min(i1, Math.ceil((sim.worldX + sim.worldR - originX) / size));
    j0 = Math.max(j0, Math.floor((sim.worldY - sim.worldR - originY) / size));
    j1 = Math.min(j1, Math.ceil((sim.worldY + sim.worldR - originY) / size));
  }
  const gold = (e: number): string => {
    const a = Math.min(0.28, 0.06 + 0.12 * (e / ambient));
    return `rgba(232, 196, 88, ${a})`;
  };
  ctx.save();
  if (sim.worldR > 0) {
    ctx.beginPath();
    ctx.arc(sim.worldX, sim.worldY, sim.worldR, 0, Math.PI * 2);
    ctx.clip();
  }
  // Uniform ambient as one fill — walking every cell in a far-zoom cover is
  // hundreds of thousands of fillRects and freezes the tab.
  ctx.fillStyle = gold(grid.ambient);
  ctx.fillRect(left, top, right - left, bottom - top);
  /*
   * Over the cells the camera can see, rather than over the cells that have
   * been touched.
   *
   * The grid used to be sparse — an implicit ambient everywhere, with only
   * grazed cells stored — so drawing what was stored *was* drawing the
   * interesting part, and there was nothing else to walk. Now the ground
   * diffuses and regrows, so every cell in the dish differs from ambient by
   * something, and "stored" would mean all eight hundred thousand of them.
   * The cull window is what keeps this bounded instead: a screen of 40-unit
   * cells is a couple of thousand rects however far the pond has spread.
   */
  const cap = Math.max(1e-9, grid.ambient);
  for (let j = j0; j < j1; j++) {
    for (let i = i0; i < i1; i++) {
      const e = grid.getCell(i, j);
      if (Math.abs(e - grid.ambient) < cap * 1e-3) continue;
      const x = originX + i * size;
      const y = originY + j * size;
      if (e <= cap * 1e-3) {
        ctx.fillStyle = '#0c0d10';
        ctx.fillRect(x, y, size, size);
        continue;
      }
      ctx.fillStyle = gold(e);
      ctx.fillRect(x, y, size, size);
    }
  }
  ctx.restore();
}

function drawEnergySlot(ctx: CanvasRenderingContext2D, agent: Agent): void {
  const r = boundRadius(agent) + 6;
  ctx.save();
  if (agent.request > 0) {
    const a = Math.min(0.4, 0.06 + 0.35 * Math.min(1, agent.request / REQUEST_FULL));
    ctx.beginPath();
    ctx.arc(agent.x, agent.y, r + 4, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(232, 196, 88, ${a})`;
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.arc(agent.x, agent.y, r, 0, Math.PI * 2);
  if (agent.extra > 0) {
    ctx.fillStyle = `rgba(232, 196, 88, ${0.12 + 0.3 * Math.min(1, agent.extra)})`;
    ctx.fill();
    ctx.strokeStyle = 'rgba(232, 196, 88, 0.95)';
  } else if (agent.extra < 0) {
    ctx.strokeStyle = `rgba(196, 92, 72, ${0.35 + 0.4 * Math.min(1, -agent.extra)})`;
  } else {
    ctx.strokeStyle = 'rgba(232, 196, 88, 0.4)';
  }
  ctx.lineWidth = 1.6;
  ctx.stroke();
  ctx.restore();
}

type Rgb = readonly [number, number, number];
type Hsl = { h: number; s: number; l: number };

/** Dup red, Con blue, Era yellow — full saturation at `EXTRA_CAP`. */
const KIND_RGB: Record<AgentKind, Rgb> = {
  dup: [255, 0, 0],
  con: [0, 0, 255],
  era: [255, 255, 0],
};

/**
 * Canvas2D's un-tinted fills when kind colors are off. GPU FAR dots have to
 * pack the same RGB — they used to fall through to KIND_RGB, so unchecking
 * kind colors only greyscaled the NEAR-tier Canvas2D bodies and left the
 * zoomed-out population in full hue.
 */
export const KIND_BW_RGB: Record<AgentKind, Rgb> = {
  dup: [0x11, 0x12, 0x13],
  era: [0xf3, 0xf3, 0xf3],
  con: [0xf4, 0xf4, 0xf4],
};

/** Kind hue (energy as saturation) or the grayscale fill, matching drawAgent. */
export function agentFillRgb(kind: AgentKind, extra: number, kindColors: boolean): Rgb {
  return kindColors ? kindFillRgb(kind, extra) : KIND_BW_RGB[kind];
}

const KIND_HSL: Record<AgentKind, Hsl> = {
  dup: rgbToHsl(...KIND_RGB.dup),
  con: rgbToHsl(...KIND_RGB.con),
  era: rgbToHsl(...KIND_RGB.era),
};

function rgbToHsl(r: number, g: number, b: number): Hsl {
  const rr = r / 255;
  const gg = g / 255;
  const bb = b / 255;
  const max = Math.max(rr, gg, bb);
  const min = Math.min(rr, gg, bb);
  const l = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = 0;
  if (max === rr) h = (gg - bb) / d + (gg < bb ? 6 : 0);
  else if (max === gg) h = (bb - rr) / d + 2;
  else h = (rr - gg) / d + 4;
  return { h: h * 60, s, l };
}

function hueToRgb(p: number, q: number, t: number): number {
  let tt = t;
  if (tt < 0) tt += 1;
  if (tt > 1) tt -= 1;
  if (tt < 1 / 6) return p + (q - p) * 6 * tt;
  if (tt < 1 / 2) return q;
  if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
  return p;
}

function hslToRgb(h: number, s: number, l: number): Rgb {
  const hh = ((h % 360) + 360) % 360 / 360;
  if (s <= 0) {
    const g = Math.round(l * 255);
    return [g, g, g];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    Math.round(hueToRgb(p, q, hh + 1 / 3) * 255),
    Math.round(hueToRgb(p, q, hh) * 255),
    Math.round(hueToRgb(p, q, hh - 1 / 3) * 255),
  ];
}

function cssRgb(rgb: Rgb): string {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function chroma(rgb: Rgb): number {
  return Math.max(rgb[0], rgb[1], rgb[2]) - Math.min(rgb[0], rgb[1], rgb[2]);
}

function energySat(extra: number, floor = EXTRA_FLOOR, cap = EXTRA_CAP): number {
  const span = cap - floor;
  if (!(span > 0)) return extra >= cap ? 1 : 0;
  return Math.max(0, Math.min(1, (extra - floor) / span));
}

/** Kind fill with energy as saturation. Full tank is the named hue; empty is gray. */
export function kindFillRgb(
  kind: AgentKind,
  extra: number,
  floor = EXTRA_FLOOR,
  cap = EXTRA_CAP,
): Rgb {
  const t = energySat(extra, floor, cap);
  if (t >= 1) return KIND_RGB[kind];
  const { h, s, l } = KIND_HSL[kind];
  return hslToRgb(h, s * t, l);
}

export function kindFillCss(
  kind: AgentKind,
  extra: number,
  floor = EXTRA_FLOOR,
  cap = EXTRA_CAP,
): string {
  return cssRgb(kindFillRgb(kind, extra, floor, cap));
}

export function kindChroma(kind: AgentKind, extra: number): number {
  return chroma(kindFillRgb(kind, extra));
}

/**
 * One scratch slot reused for every ghost, rather than a fresh `AgentStore`
 * per call — a ghost is drawn and discarded within the same expression
 * (`drawAgent(ctx, ghostAsAgent(g), ...)`, never stashed), so there is
 * nothing to isolate between calls, and reusing the slot avoids allocating
 * a whole typed-array table per preview frame.
 */
const ghostStore = new AgentStore(1);
const ghostAgent = new Agent(ghostStore, ghostStore.allocate(-1));

/*
 * Everything about the scratch ghost that does not vary from ghost to ghost,
 * written once.
 *
 * `ghostStore` is private to this module and `ghostAgent` its only occupant,
 * so nothing between two draws can disturb these — and a rewrite-heavy dish
 * draws a lot of ghosts. At 1,469 rewrites in flight that loop was running
 * about 5,900 times a frame, and `chem.fill(0)` alone was zeroing 134 floats
 * each time, on an array that has been zero since the first call.
 */
function initGhostAgent(): void {
  const a = ghostAgent;
  // Never read before being rewritten — `csHeading` is set to NaN per ghost, so
  // the memo in `stemOffsetInto` always misses and recomputes — but set once so
  // the scratch starts in exactly the state it used to start every call in.
  a.csCos = 1;
  a.csSin = 0;
  // A ghost is drawn, never simulated, so it neither emits nor smells.
  a.chem.fill(0);
  // A ghost is drawn, never flocked.
  a.flockAlign = 0;
  a.flockSep = 0;
  a.vx = 0;
  a.vy = 0;
  a.omega = 0;
  a.mass = 1;
  a.locked = true;
  a.stun = 0;
  a.drive = 0;
  a.trail = 0;
  a.extra = 0;
  a.request = 0;
  a.recovering = false;
  // A ghost is preview art, not a simulated body — never bred, never billed.
  a.requestDecay = REQUEST_DECAY;
  a.energyCap = EXTRA_CAP;
  a.debtCap = EXTRA_FLOOR;
  a.rescueTo = 0.9;
}
initGhostAgent();

/** The eight fields that actually differ between one ghost and the next. */
function ghostAsAgent(g: Ghost): Agent {
  const a = ghostAgent;
  a.kind = g.kind;
  a.x = g.x;
  a.y = g.y;
  // Cold: a ghost is built fresh each time, so there is nothing to reuse.
  a.csHeading = NaN;
  a.heading = g.heading;
  a.alpha = g.alpha;
  a.scale = g.scale;
  a.prevX = g.x;
  a.prevY = g.y;
  a.prevHeading = g.heading;
  return a;
}

/** World-space px of displacement at |sample/env| = 1 after AGC. */
const WAVE_SCALE = WAVE_DISP_PX * 0.75;
/** Below this envelope the wire is drawn as its rest pose. */
const ENV_DEAD = 4e-4;
/** Envelope at which AGC fade reaches 1. */
const ENV_FULL = 0.04;

/**
 * Visible offset from a traveling-wave sample pair.
 * Divides by the pickup envelope so a quiet pluck and a loud ring occupy
 * similar pixels; the fade keeps silence from being amplified into noise.
 */
export function waveDisplace(fwd: number, back: number, env: number, pin: number): number {
  if (!(env > ENV_DEAD) || pin === 0) return 0;
  if (!Number.isFinite(fwd) || !Number.isFinite(back) || !Number.isFinite(env)) return 0;
  const t = Math.max(0, Math.min(1, (env - ENV_DEAD) / (ENV_FULL - ENV_DEAD)));
  const fade = t * t * (3 - 2 * t);
  const unit = (fwd + back) / env;
  const a = Math.max(-1.7, Math.min(1.7, unit));
  return a * WAVE_SCALE * pin * fade;
}

/**
 * Every wire in one path, stroked once.
 *
 * They all share the same white stroke, so per-wire `beginPath`/`stroke` was
 * buying nothing and costing a draw call each: at ~5,500 wires that is 5,500
 * strokes a frame, which is the same ceiling the GPU dot layer exists to get
 * the population off. Subpaths are what `moveTo` starts, so one path holds
 * them all and the caps and joins come out identical.
 */
/**
 * Body id -> index of the first rewrite in this frame's list consuming it.
 *
 * A set of dying ids was not enough. It skipped the rewrite scan for wires that
 * touch nothing dying, which is most of them in a young pond — but in a grown
 * one a lot of wires touch something dying, and each of those still walked the
 * whole rewrite list. Measured on a pond of 16,765 wires with 1,475 rewrites in
 * flight: 3,621 wires touching a dying body, 5.3 million `rewriteHandoffStems`
 * calls a frame, **64.6 ms** — four fifths of the entire wire pass.
 *
 * `rewriteHandoffStems` returns null unless the wire has an end on that
 * rewrite's own two bodies, so only the rewrites indexed here can ever answer
 * yes: a wire has two ends, so at most two candidates. The index is the
 * position in `sim.rewrites` so they can still be tried in the order the scan
 * would have tried them, which is what makes this exactly equivalent rather
 * than merely close.
 */
const dyingScratch = new Map<number, number>();

function drawWires(
  ctx: CanvasRenderingContext2D,
  sim: Sim,
  waves: WaveSnapshot | null,
  zoom: number,
): void {
  ctx.lineWidth = WIRE_STROKE_PX;
  ctx.strokeStyle = '#ffffff';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  const graph = sim.graph;
  const agents = sim.agents;
  const w = sim.w;
  const h = sim.h;
  const rewrites = sim.rewrites;
  // A handoff only exists for a wire touching an agent a rewrite is consuming,
  // so collect those ids once instead of asking every rewrite about every wire.
  // That inner loop was the whole cross product -- 5,500 wires against 30
  // rewrites is 165,000 calls a frame to answer "no" 164,900 times.
  dyingScratch.clear();
  for (let i = 0; i < rewrites.length; i++) {
    const rw = rewrites[i];
    if (!dyingScratch.has(rw.a)) dyingScratch.set(rw.a, i);
    if (!dyingScratch.has(rw.b)) dyingScratch.set(rw.b, i);
  }
  for (const wire of graph.wires.values()) {
    const A = agents.get(wire.a.id);
    const B = agents.get(wire.b.id);
    if (!A || !B) continue;
    let stemA: { x: number; y: number } | undefined;
    let stemB: { x: number; y: number } | undefined;
    const ia = dyingScratch.get(wire.a.id);
    const ib = dyingScratch.get(wire.b.id);
    if (ia !== undefined || ib !== undefined) {
      // Whichever of the two candidates the old scan would have reached first,
      // then the other. Anything else in the list answers null by definition.
      const lo = ia === undefined ? ib! : ib === undefined ? ia : ia < ib ? ia : ib;
      const hi = ia === undefined || ib === undefined ? -1 : ia < ib ? ib : ia;
      let handoff = rewriteHandoffStems(rewrites[lo], wire, agents, w, h);
      if (!handoff && hi >= 0 && hi !== lo) {
        handoff = rewriteHandoffStems(rewrites[hi], wire, agents, w, h);
      }
      if (handoff) {
        stemA = { x: handoff.ax, y: handoff.ay };
        stemB = { x: handoff.bx, y: handoff.by };
      }
    }
    const rec = waves?.index.get(wire.id);
    const rope = sim.wireSimulatesRope(wire);
    if (!(rec !== undefined && strokeOffsetWire(ctx, A, B, wire, w, h, waves!.packed, rec, stemA, stemB, rope))) {
      strokeWire(ctx, A, B, wire, w, h, stemA, stemB, rope, zoom);
    }
  }
  ctx.stroke();
}

function drawCommuteGhostWires(
  ctx: CanvasRenderingContext2D,
  rw: Rewrite,
  w: number,
  h: number,
): void {
  if (rw.rule !== 'commute' || rw.ghosts.length !== 4) return;
  const alpha = rw.ghosts[0]?.alpha ?? 0;
  if (alpha < 0.02) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineWidth = WIRE_STROKE_PX;
  ctx.strokeStyle = '#ffffff';
  ctx.lineCap = 'round';
  for (const link of COMMUTE_K22) {
    const ga = commuteGhost(rw, link.a);
    const gb = commuteGhost(rw, link.b);
    if (!ga || !gb) continue;
    const sa = stemFromGhost(ga, link.aSlot, w, h);
    const sb = stemFromGhost(gb, link.bSlot, w, h);
    ctx.beginPath();
    ctx.moveTo(sa.x, sa.y);
    ctx.lineTo(sb.x, sb.y);
    ctx.stroke();
  }
  ctx.restore();
}

/** Most samples a chord wire is ever cut into. Matches `wireControlPoints`. */
const CHORD_STEPS = 16;

/**
 * Screen pixels a chord segment is worth drawing for.
 *
 * Sixteen segments is the right count for a wire that fills a good part of the
 * view and absurd for one that is a pixel long, which is what nearly every
 * wire is once the camera pulls back far enough to see a whole dish. A 60 s
 * trace of a grown pond put `drawWires` at 13.3% of all CPU, most of it here.
 *
 * The cubic's bow is bounded by its handles, which `wireCubic` caps at a third
 * of the span, so the sagitta is under about a quarter of the span. At five
 * screen pixels a segment, the straight-line case is entering at a span whose
 * whole curve could deviate by roughly one pixel — under the 1.35 px stroke.
 */
const CHORD_PX_PER_SEGMENT = 5;

/** Scratch for the one-segment case, which needs the two stems and nothing else. */
const chordEndA = { x: 0, y: 0 };
const chordEndB = { x: 0, y: 0 };

/**
 * How many segments this wire is worth at this zoom, from 1 to `CHORD_STEPS`.
 *
 * `lastLen` is the length the step already measured, so this costs a multiply.
 */
function chordSteps(wire: Wire, zoom: number): number {
  const span = wire.lastLen > 0 ? wire.lastLen : wire.rest;
  if (!(span > 0) || !(zoom > 0)) return CHORD_STEPS;
  const want = Math.ceil((span * zoom) / CHORD_PX_PER_SEGMENT);
  if (!(want > 1)) return 1;
  return want < CHORD_STEPS ? want : CHORD_STEPS;
}

/**
 * Reused sample buffer for the chord case, which is every wire that is not
 * running a live rope -- i.e. the whole FAR-tier pond.
 *
 * `wireControlPoints` builds that polyline with `push` and a fresh point per
 * sample: seventeen objects per wire per frame, ~93,000 a frame at 5,500
 * wires, all of it discarded before the next one. The length never varies
 * here, so one buffer serves every wire and the objects are written through.
 */
const chordScratch: { x: number; y: number }[] = [];

/** Writes `steps + 1` points into `chordScratch` and returns that count. */
function chordPolyline(
  A: Agent,
  B: Agent,
  wire: Wire,
  w: number,
  h: number,
  steps: number,
): number {
  while (chordScratch.length <= CHORD_STEPS) chordScratch.push({ x: 0, y: 0 });
  const c = wireCubic(A, wire.a.slot, B, wire.b.slot, w, h, wire.rest);
  for (let i = 0; i <= steps; i++) {
    bezierPointInto(c.p0, c.p1, c.p2, c.p3, i / steps, chordScratch[i]);
  }
  const last = chordScratch[steps];
  const first = chordScratch[0];
  const span = Math.hypot(last.x - first.x, last.y - first.y);
  clampPolylineToChord(chordScratch, wireBowBudget(span, wire.rest), steps + 1);
  return steps + 1;
}

function strokeWire(
  ctx: CanvasRenderingContext2D,
  A: Agent,
  B: Agent,
  wire: Wire,
  w: number,
  h: number,
  stemA?: { x: number; y: number },
  stemB?: { x: number; y: number },
  rope = true,
  zoom = 0,
): void {
  // The chord case ignores the stems anyway -- `wireControlPoints` reads the
  // ports off the bodies -- so the pooled path is the same geometry.
  if (!rope || wire.nodes.length === 0) {
    const steps = chordSteps(wire, zoom);
    if (steps <= 1) {
      /*
       * A wire this short on screen is a line. Taking it here skips the whole
       * cubic — `wireCubic` alone builds about eleven objects a call, and the
       * sampling built seventeen points for something under five pixels long.
       */
      stemWorldInto(A, wire.a.slot, w, h, chordEndA);
      stemWorldInto(B, wire.b.slot, w, h, chordEndB);
      ctx.moveTo(chordEndA.x, chordEndA.y);
      ctx.lineTo(chordEndB.x, chordEndB.y);
      return;
    }
    const n = chordPolyline(A, B, wire, w, h, steps);
    ctx.moveTo(chordScratch[0].x, chordScratch[0].y);
    for (let i = 1; i < n; i++) ctx.lineTo(chordScratch[i].x, chordScratch[i].y);
    return;
  }
  const pts = wireStrokePoints(A, B, wire, w, h, stemA, stemB, rope);
  if (pts.length === 0) return;
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
}

function strokeOffsetWire(
  ctx: CanvasRenderingContext2D,
  A: Agent,
  B: Agent,
  wire: Wire,
  w: number,
  h: number,
  packed: Float32Array,
  rec: number,
  stemA?: { x: number; y: number },
  stemB?: { x: number; y: number },
  rope = true,
): boolean {
  const bins = packed[1] | 0;
  if (bins < 2) return false;
  const samples = pathSamples(wireStrokePoints(A, B, wire, w, h, stemA, stemB, rope), bins);
  if (samples.length < 2) return false;
  const env = packed[rec + 1];
  if (!(env > ENV_DEAD)) return false;
  const fwd0 = rec + 2;
  const back0 = fwd0 + bins;
  const last = samples.length - 1;
  const xs = new Array<number>(last + 1);
  const ys = new Array<number>(last + 1);
  for (let i = 0; i <= last; i++) {
    const pin = i === 0 || i === last ? 0 : Math.sin((Math.PI * i) / last);
    if (!Number.isFinite(samples[i].x) || !Number.isFinite(samples[i].y)) {
      return false;
    }
    const d = waveDisplace(packed[fwd0 + i], packed[back0 + i], env, pin);
    xs[i] = samples[i].x + samples[i].nx * d;
    ys[i] = samples[i].y + samples[i].ny * d;
    if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) return false;
  }
  for (let i = 0; i <= last; i++) {
    if (i === 0) ctx.moveTo(xs[i], ys[i]);
    else ctx.lineTo(xs[i], ys[i]);
  }
  return true;
}

interface PathSample {
  x: number;
  y: number;
  nx: number;
  ny: number;
}

/** Polyline the renderer strokes for a wire — cubic if no rope, Catmull otherwise. */
export function wireStrokePoints(
  A: Agent,
  B: Agent,
  wire: Wire,
  w: number,
  h: number,
  stemA?: { x: number; y: number },
  stemB?: { x: number; y: number },
  rope = true,
): { x: number; y: number }[] {
  const pts = wireControlPoints(A, B, wire, w, h, stemA, stemB, rope);
  if (pts.length < 2) return pts;
  const span = Math.hypot(pts[pts.length - 1].x - pts[0].x, pts[pts.length - 1].y - pts[0].y);
  const budget = wireBowBudget(span, wire.rest);
  if (!rope || wire.nodes.length === 0) {
    clampPolylineToChord(pts, budget);
    return pts;
  }
  const n = pts.length;
  const out: { x: number; y: number }[] = [];
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(n - 1, i + 2)];
    const c = catmullSegment(
      i === 0 ? { x: 2 * p1.x - p2.x, y: 2 * p1.y - p2.y } : p0,
      p1,
      p2,
      i === n - 2 ? { x: 2 * p2.x - p1.x, y: 2 * p2.y - p1.y } : p3,
    );
    const steps = 4;
    const start = i === 0 ? 0 : 1;
    for (let k = start; k <= steps; k++) {
      out.push(bezierPoint(c.p0, c.p1, c.p2, c.p3, k / steps));
    }
  }
  clampPolylineToChord(out, budget);
  return out;
}

function wireControlPoints(
  A: Agent,
  B: Agent,
  wire: Wire,
  w: number,
  h: number,
  stemA?: { x: number; y: number },
  stemB?: { x: number; y: number },
  rope = true,
): { x: number; y: number }[] {
  if (!rope || wire.nodes.length === 0) {
    const c = wireCubic(A, wire.a.slot, B, wire.b.slot, w, h, wire.rest);
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= 16; i++) pts.push(bezierPoint(c.p0, c.p1, c.p2, c.p3, i / 16));
    return pts;
  }
  const raw = [
    stemA ?? stemWorld(A, wire.a.slot, w, h),
    ...wire.nodes,
    stemB ?? stemWorld(B, wire.b.slot, w, h),
  ];
  const pts = unwrapPoints(raw, w, h);
  const span = Math.hypot(pts[pts.length - 1].x - pts[0].x, pts[pts.length - 1].y - pts[0].y);
  clampPolylineToChord(pts, wireBowBudget(span, wire.rest));
  return pts;
}

function pathSamples(pts: { x: number; y: number }[], n: number): PathSample[] {
  const out: PathSample[] = [];
  if (n <= 0 || pts.length === 0) return out;
  if (pts.length === 1) {
    for (let i = 0; i < n; i++) out.push({ x: pts[0].x, y: pts[0].y, nx: 0, ny: -1 });
    return out;
  }
  const seg: number[] = [];
  let total = 0;
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    seg.push(d);
    total += d;
  }
  if (total < 1e-6) {
    for (let i = 0; i < n; i++) out.push({ x: pts[0].x, y: pts[0].y, nx: 0, ny: -1 });
    return out;
  }
  for (let i = 0; i < n; i++) {
    const target = n === 1 ? 0 : (i / (n - 1)) * total;
    let acc = 0;
    let s = 0;
    while (s < seg.length - 1 && acc + seg[s] < target) {
      acc += seg[s];
      s++;
    }
    const span = seg[s];
    const u = span < 1e-9 ? 0 : (target - acc) / span;
    const a = pts[s];
    const b = pts[s + 1];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    out.push({
      x: a.x + dx * u,
      y: a.y + dy * u,
      nx: -dy / len,
      ny: dx / len,
    });
  }
  for (let i = 1; i < out.length; i++) {
    if (out[i].nx * out[i - 1].nx + out[i].ny * out[i - 1].ny < 0) {
      out[i].nx *= -1;
      out[i].ny *= -1;
    }
  }
  return out;
}

function drawAgent(
  ctx: CanvasRenderingContext2D,
  agent: Agent,
  alpha: number,
  graph?: Graph,
  kindColors = false,
): void {
  ctx.save();
  ctx.translate(agent.x, agent.y);
  ctx.rotate(agent.heading);
  ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
  ctx.scale(agent.scale, agent.scale);
  const fill = kindColors ? kindFillCss(agent.kind, agent.extra, agent.debtCap, agent.energyCap) : undefined;
  if (agent.kind === 'era') drawEra(ctx, fill);
  else drawTriangle(ctx, agent.kind, fill);
  drawPortStems(ctx, agent, graph, fill);
  ctx.restore();
}

function drawEra(ctx: CanvasRenderingContext2D, fill?: string): void {
  ctx.beginPath();
  ctx.arc(0, 0, ERA_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = fill ?? '#f3f3f3';
  ctx.fill();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = fill ?? '#ffffff';
  ctx.stroke();
}

/*
 * The unit triangle, once.
 *
 * `triangleLocal(1)` builds three points a call, and `drawAgent` scales the
 * context rather than the geometry — so the argument is always 1 and the
 * answer always the same three points. A rewrite-heavy dish draws thousands
 * of ghosts a frame through here.
 */
const TRIANGLE_UNIT = triangleLocal(1);

function drawTriangle(ctx: CanvasRenderingContext2D, kind: AgentKind, fill?: string): void {
  const [a, b, c] = TRIANGLE_UNIT;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineTo(c.x, c.y);
  ctx.closePath();
  ctx.fillStyle = fill ?? (kind === 'dup' ? '#111213' : '#f4f4f4');
  ctx.strokeStyle = fill ?? '#ffffff';
  ctx.fill();
  ctx.lineWidth = 1.4;
  ctx.stroke();
}

/**
 * Stem inner and outer point per slot, flat: `[ix, iy, ox, oy]` a slot, in
 * `ERA_SLOTS` / `NODE_SLOTS` order. Pure geometry of the kind, so it is built
 * once from the same functions that used to be called per port.
 */
function stemTable(kind: AgentKind): Float64Array {
  const slots = kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
  const out = new Float64Array(slots.length * 4);
  for (let k = 0; k < slots.length; k++) {
    const inner = stemRoot(kind, slots[k]);
    const outer = portLocal(kind, slots[k]);
    out[k * 4] = inner.x;
    out[k * 4 + 1] = inner.y;
    out[k * 4 + 2] = outer.x;
    out[k * 4 + 3] = outer.y;
  }
  return out;
}
const ERA_STEMS = stemTable('era');
const DUP_STEMS = stemTable('dup');
const CON_STEMS = stemTable('con');

function drawPortStems(
  ctx: CanvasRenderingContext2D,
  agent: Agent,
  graph?: Graph,
  stroke?: string,
): void {
  ctx.beginPath();
  let any = false;
  /*
   * The stem geometry is a function of kind and slot alone — the body's own
   * transform is already on the context — so it is a small fixed table rather
   * than four objects a port. `slotsFor` also built a fresh array a call;
   * `ERA_SLOTS`/`NODE_SLOTS` are the frozen ones it exists to avoid.
   */
  const kind = agent.kind;
  const slots = kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
  const stems = kind === 'era' ? ERA_STEMS : kind === 'dup' ? DUP_STEMS : CON_STEMS;
  const at = agent.slot;
  for (let k = 0; k < slots.length; k++) {
    if (graph && !graph.isFreeAtSlot(at, k)) continue;
    const o = k * 4;
    ctx.moveTo(stems[o], stems[o + 1]);
    ctx.lineTo(stems[o + 2], stems[o + 3]);
    any = true;
  }
  if (!any) return;
  ctx.lineWidth = 1.6;
  ctx.lineCap = 'round';
  ctx.strokeStyle = stroke ?? '#ffffff';
  ctx.stroke();
}
