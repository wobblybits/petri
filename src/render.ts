import { ERA_RADIUS, portLocal, slotsFor, stemRoot, stemWorld, triangleLocal, wireCubic, type Agent, type AgentKind } from './agents.ts';
import { WAVE_DISP_PX } from './geom.ts';
import type { WaveSnapshot } from './audio/types.ts';
import type { Camera } from './camera.ts';
import { catmullSegment, unwrapPoints } from './chain.ts';
import { bezierPoint } from './curve.ts';
import type { Fields } from './fields.ts';
import type { Graph, Wire } from './graph.ts';
import type { Ghost } from './rewrite.ts';
import type { Sim } from './sim.ts';

export interface ViewOpts {
  overlay: boolean;
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
  ctx.fillStyle = '#0c0d10';
  ctx.fillRect(0, 0, camera.viewW, camera.viewH);

  ctx.save();
  camera.apply(ctx);

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

  drawWires(ctx, sim.graph, sim.agents, sim.w, sim.h, waves);
  for (const rw of sim.rewrites) {
    for (const g of rw.ghosts) drawAgent(ctx, ghostAsAgent(g), 1);
  }
  for (const agent of sim.agents.values()) {
    drawAgent(ctx, agent, agent.alpha, sim.graph);
  }
  ctx.restore();
  ctx.restore();
}

function ghostAsAgent(g: Ghost): Agent {
  return {
    id: -1,
    kind: g.kind,
    x: g.x,
    y: g.y,
    vx: 0,
    vy: 0,
    heading: g.heading,
    omega: 0,
    mass: 1,
    alpha: g.alpha,
    scale: g.scale,
    locked: true,
    stun: 0,
    drive: 0,
    trail: 0,
    prevX: g.x,
    prevY: g.y,
    prevHeading: g.heading,
    integVx: 0,
    integVy: 0,
    integOmega: 0,
  };
}

/** World-space px of displacement at |sample/env| = 1 after AGC. */
const WAVE_SCALE = WAVE_DISP_PX;
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

function drawWires(
  ctx: CanvasRenderingContext2D,
  graph: Graph,
  agents: Map<number, Agent>,
  w: number,
  h: number,
  waves: WaveSnapshot | null,
): void {
  ctx.lineWidth = 1.35;
  ctx.strokeStyle = '#ffffff';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const wire of graph.wires.values()) {
      const A = agents.get(wire.a.id);
      const B = agents.get(wire.b.id);
      if (!A || !B) continue;
      ctx.beginPath();
      const rec = waves?.index.get(wire.id);
      if (!(rec !== undefined && strokeOffsetWire(ctx, A, B, wire, w, h, waves!.packed, rec))) {
        strokeWire(ctx, A, B, wire, w, h);
      }
      ctx.stroke();
    }
}

function strokeWire(
  ctx: CanvasRenderingContext2D,
  A: Agent,
  B: Agent,
  wire: Wire,
  w: number,
  h: number,
): void {
  if (wire.nodes.length === 0) {
    strokeCubic(ctx, wireCubic(A, wire.a.slot, B, wire.b.slot, w, h, wire.rest));
    return;
  }
  const raw = [
    stemWorld(A, wire.a.slot, w, h),
    ...wire.nodes,
    stemWorld(B, wire.b.slot, w, h),
  ];
  const pts = unwrapPoints(raw, w, h);
  const n = pts.length;
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(n - 1, i + 2)];
    strokeCubic(
      ctx,
      catmullSegment(
        i === 0 ? { x: 2 * p1.x - p2.x, y: 2 * p1.y - p2.y } : p0,
        p1,
        p2,
        i === n - 2 ? { x: 2 * p2.x - p1.x, y: 2 * p2.y - p1.y } : p3,
      ),
    );
  }
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
): boolean {
  const bins = packed[1] | 0;
  if (bins < 2) return false;
  const samples = pathSamples(wireStrokePoints(A, B, wire, w, h), bins);
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

function wireStrokePoints(A: Agent, B: Agent, wire: Wire, w: number, h: number): { x: number; y: number }[] {
  if (wire.nodes.length === 0) {
    const c = wireCubic(A, wire.a.slot, B, wire.b.slot, w, h, wire.rest);
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i <= 16; i++) pts.push(bezierPoint(c.p0, c.p1, c.p2, c.p3, i / 16));
    return pts;
  }
  const raw = [
    stemWorld(A, wire.a.slot, w, h),
    ...wire.nodes,
    stemWorld(B, wire.b.slot, w, h),
  ];
  const pts = unwrapPoints(raw, w, h);
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
  return out;
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

function strokeCubic(
  ctx: CanvasRenderingContext2D,
  c: { p0: { x: number; y: number }; p1: { x: number; y: number }; p2: { x: number; y: number }; p3: { x: number; y: number } },
): void {
  ctx.moveTo(c.p0.x, c.p0.y);
  ctx.bezierCurveTo(c.p1.x, c.p1.y, c.p2.x, c.p2.y, c.p3.x, c.p3.y);
}

function drawAgent(ctx: CanvasRenderingContext2D, agent: Agent, alpha: number, graph?: Graph): void {
  ctx.save();
  ctx.translate(agent.x, agent.y);
  ctx.rotate(agent.heading);
  ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
  ctx.scale(agent.scale, agent.scale);
  if (agent.kind === 'era') drawEra(ctx);
  else drawTriangle(ctx, agent.kind);
  drawPortStems(ctx, agent, graph);
  ctx.restore();
}

function drawEra(ctx: CanvasRenderingContext2D): void {
  ctx.beginPath();
  ctx.arc(0, 0, ERA_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = '#f3f3f3';
  ctx.fill();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = '#ffffff';
  ctx.stroke();
}

function drawTriangle(ctx: CanvasRenderingContext2D, kind: AgentKind): void {
  const [a, b, c] = triangleLocal(1);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineTo(c.x, c.y);
  ctx.closePath();
  ctx.fillStyle = kind === 'dup' ? '#111213' : '#f4f4f4';
  ctx.strokeStyle = '#ffffff';
  ctx.fill();
  ctx.lineWidth = 1.4;
  ctx.stroke();
}

function drawPortStems(ctx: CanvasRenderingContext2D, agent: Agent, graph?: Graph): void {
  ctx.beginPath();
  let any = false;
  for (const slot of slotsFor(agent.kind)) {
    if (graph && !graph.isFree({ id: agent.id, slot })) continue;
    const inner = stemRoot(agent.kind, slot);
    const outer = portLocal(agent.kind, slot);
    ctx.moveTo(inner.x, inner.y);
    ctx.lineTo(outer.x, outer.y);
    any = true;
  }
  if (!any) return;
  ctx.lineWidth = 1.6;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#ffffff';
  ctx.stroke();
}
