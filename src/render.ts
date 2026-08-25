import { ERA_RADIUS, portLocal, slotsFor, stemRoot, stemWorld, triangleLocal, wireCubic, type Agent, type AgentKind } from './agents.ts';
import type { Camera } from './camera.ts';
import { catmullSegment, unwrapPoints } from './chain.ts';
import type { Fields } from './fields.ts';
import type { Graph } from './graph.ts';
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

export function render(ctx: CanvasRenderingContext2D, sim: Sim, camera: Camera, opts: ViewOpts): void {
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

  drawWires(ctx, sim.graph, sim.agents, sim.w, sim.h);
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
    prevX: g.x,
    prevY: g.y,
    prevHeading: g.heading,
    integVx: 0,
    integVy: 0,
    integOmega: 0,
  };
}

function drawWires(
  ctx: CanvasRenderingContext2D,
  graph: Graph,
  agents: Map<number, Agent>,
  w: number,
  h: number,
): void {
  ctx.lineWidth = 1.35;
  ctx.strokeStyle = '#ffffff';
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const wire of graph.wires.values()) {
    const A = agents.get(wire.a.id);
    const B = agents.get(wire.b.id);
    if (!A || !B) continue;
    if (wire.nodes.length === 0) {
      const c = wireCubic(A, wire.a.slot, B, wire.b.slot, w, h, wire.rest);
      strokeCubic(ctx, c);
      continue;
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
      const c = catmullSegment(
        i === 0 ? { x: 2 * p1.x - p2.x, y: 2 * p1.y - p2.y } : p0,
        p1,
        p2,
        i === n - 2 ? { x: 2 * p2.x - p1.x, y: 2 * p2.y - p1.y } : p3,
      );
      strokeCubic(ctx, c);
    }
  }
  ctx.stroke();
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
