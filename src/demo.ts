import './style.css';
import { Camera } from './camera.ts';
import { DemoChart } from './demo-chart.ts';
import { Interaction, type Tool } from './interact.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset, splatter } from './presets.ts';
import { render, buildFarInstances, FAR_INSTANCE_STRIDE } from './render.ts';
import { Sim } from './sim.ts';
import { agentsGpu } from './gpu/agents-gpu.ts';
import { farGpu } from './gpu/far-gpu.ts';
import { nativeSolver } from './native/solver.ts';

const DEMO_SOUP = 500;

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('missing #app');

app.classList.add('demo');
app.innerHTML = `
  <div id="viewport">
    <canvas id="view"></canvas>
    <canvas id="view-gpu"></canvas>
    <canvas id="chart"></canvas>
    <div id="hud">
      <div class="hud-tools">
        <label class="hud-count">
          <input type="number" id="count" min="1" step="1" value="${DEMO_SOUP}" />
        </label>
        <button type="button" id="reset" class="icon-btn" title="Reset soup" aria-label="Reset">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4v7h7"/><path d="M20 20v-7h-7"/><path d="M5.2 9.2A8 8 0 0 1 19 8.5M18.8 14.8A8 8 0 0 1 5 15.5"/></svg>
        </button>
        <button type="button" id="erase" class="icon-btn" data-tool="erase" title="Erase" aria-label="Erase">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h16"/><path d="M5.5 13.5 14 5a2.1 2.1 0 0 1 3 0l2 2a2.1 2.1 0 0 1 0 3l-8.5 8.5H5.5z"/></svg>
        </button>
        <button type="button" id="paint" class="icon-btn" data-tool="paint" title="Paint energy" aria-label="Paint energy">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3s6 6.4 6 11a6 6 0 1 1-12 0c0-4.6 6-11 6-11z"/></svg>
        </button>
        <button type="button" id="splat" class="icon-btn" data-tool="splat" title="Splatter agents" aria-label="Splatter agents">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="2.1"/><circle cx="6.5" cy="8" r="1.5"/><circle cx="17.5" cy="9" r="1.4"/><circle cx="8" cy="16.5" r="1.5"/><circle cx="16.5" cy="16" r="1.35"/><circle cx="12" cy="5.5" r="1.1"/></svg>
        </button>
        <button type="button" id="color" class="icon-btn on" title="Color / black and white" aria-label="Toggle color">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 0 0 16z" fill="currentColor" stroke="none"/></svg>
        </button>
      </div>
    </div>
    <aside id="about">
      <h1>Nets</h1>
      <p>Era, Dup, and Con forage on an energy grid, latch ports, and rewrite like Lafont interaction nets. Annihilation releases energy; a Con–Dup commute spends it.</p>
      <p class="hint">Drag to pan, scroll to zoom. Reset reseeds the soup. Erase, paint energy, or splatter a handful of agents.</p>
      <p class="hint"><a href="https://github.com/wobblybits/swimmers">Source</a></p>
    </aside>
  </div>
`;

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const maybeCtx = canvas.getContext('2d');
if (!maybeCtx) throw new Error('no 2d context');
const ctx: CanvasRenderingContext2D = maybeCtx;
const gpuCanvas = document.querySelector<HTMLCanvasElement>('#view-gpu')!;
const chartCanvas = document.querySelector<HTMLCanvasElement>('#chart')!;
const countInput = document.querySelector<HTMLInputElement>('#count')!;
const resetBtn = document.querySelector<HTMLButtonElement>('#reset')!;
const colorBtn = document.querySelector<HTMLButtonElement>('#color')!;

const params: Params = defaultParams();
params.soupCount = DEMO_SOUP;
params.spawnInterval = 0;
countInput.max = String(params.maxAgents);

const sim = new Sim(800, 600);
const camera = new Camera();
const interaction = new Interaction(sim, camera);
const chart = new DemoChart(chartCanvas);
const view = { overlay: false, energyGrid: false, energyCircles: false, kindColors: true, gpuAgents: false };
let farInstances = new Float32Array(0);
let paused = false;
let snapCamera = true;

interaction.onSplat = (x, y) => {
  splatter(sim, x, y, params);
};

function cssSize(): { w: number; h: number } {
  const rect = canvas.getBoundingClientRect();
  return { w: Math.max(1, rect.width), h: Math.max(1, rect.height) };
}

function sizeCanvas(): void {
  const { w, h } = cssSize();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.floor(w * dpr));
  canvas.height = Math.max(1, Math.floor(h * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  gpuCanvas.width = canvas.width;
  gpuCanvas.height = canvas.height;
  sim.resize(w, h);
  camera.setView(w, h);
  chart.resize();
}

function syncCamera(dt: number): void {
  const com = sim.centerOfMass();
  if (com && !interaction.freeCamera) {
    if (snapCamera) {
      camera.snap(com.x, com.y);
      snapCamera = false;
    } else {
      camera.follow(com.x, com.y, dt);
    }
  }
  sim.setViewExtent(camera.coverWidth() * 1.7, camera.coverHeight() * 1.7);
}

function readCount(): number {
  const n = Math.floor(Number(countInput.value));
  if (!Number.isFinite(n)) return DEMO_SOUP;
  return Math.max(1, Math.min(params.maxAgents, n));
}

function resetSoup(): void {
  params.soupCount = readCount();
  countInput.value = String(params.soupCount);
  sizeCanvas();
  loadPreset(sim, 'soup', params);
  snapCamera = true;
  interaction.freeCamera = false;
  chart.clear();
  chart.sample(sim);
  syncCamera(0);
}

function setTool(next: Tool): void {
  interaction.tool = interaction.tool === next ? 'none' : next;
  view.energyGrid = interaction.tool === 'paint';
  for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-tool]')) {
    btn.classList.toggle('on', btn.dataset.tool === interaction.tool);
  }
}

function paint(): void {
  view.gpuAgents = agentsGpu.ready;
  if (view.gpuAgents) {
    const need = sim.agents.size * FAR_INSTANCE_STRIDE;
    if (farInstances.length < need) farInstances = new Float32Array(need);
    const count = buildFarInstances(sim, farInstances, view.kindColors);
    agentsGpu.render(farInstances, count, camera);
  }
  render(ctx, sim, camera, view);
}

resetBtn.addEventListener('click', () => resetSoup());
countInput.addEventListener('change', () => {
  countInput.value = String(readCount());
});
colorBtn.addEventListener('click', () => {
  view.kindColors = !view.kindColors;
  colorBtn.classList.toggle('on', view.kindColors);
  paint();
});
for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-tool]')) {
  btn.addEventListener('click', () => setTool(btn.dataset.tool as Tool));
}

function pointerWorld(ev: PointerEvent): { wx: number; wy: number; sx: number; sy: number } {
  const rect = canvas.getBoundingClientRect();
  const sx = ev.clientX - rect.left;
  const sy = ev.clientY - rect.top;
  const w = camera.worldFromScreen(sx, sy);
  return { wx: w.x, wy: w.y, sx, sy };
}

canvas.addEventListener('pointerdown', (ev) => {
  canvas.setPointerCapture(ev.pointerId);
  const { wx, wy, sx, sy } = pointerWorld(ev);
  interaction.begin(wx, wy, sx, sy);
});
canvas.addEventListener('pointermove', (ev) => {
  const { wx, wy, sx, sy } = pointerWorld(ev);
  interaction.move(wx, wy, sx, sy);
  canvas.style.cursor = interaction.cursorFor(wx, wy);
});
canvas.addEventListener('pointerup', (ev) => {
  const { wx, wy } = pointerWorld(ev);
  interaction.end(wx, wy, () => {});
});
canvas.addEventListener('pointercancel', () => interaction.cancel());
canvas.addEventListener(
  'wheel',
  (ev) => {
    ev.preventDefault();
    camera.zoomBy(Math.exp(-ev.deltaY * 0.0015));
    paint();
  },
  { passive: false },
);

window.addEventListener('keydown', (ev) => {
  if (ev.target instanceof HTMLInputElement) return;
  if (ev.code === 'Space') {
    ev.preventDefault();
    paused = !paused;
  }
  if (ev.key === 'x' || ev.key === 'X') setTool('erase');
});

void sim.openFieldGpu().then((ok) => {
  if (ok) console.info('scent field: GPU');
});

new ResizeObserver(() => sizeCanvas()).observe(canvas);

let last = performance.now();
let ticking = false;
async function tick(now: number): Promise<void> {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  try {
    if (paused) sim.dragStep(params, dt, camera);
    else await sim.stepAsync(dt, params, camera);
    syncCamera(dt);
    paint();
  } catch (err) {
    console.error(err);
  }
}

function frame(now: number): void {
  if (ticking) return;
  ticking = true;
  void tick(now).finally(() => {
    ticking = false;
    requestAnimationFrame(frame);
  });
}

sizeCanvas();
resetSoup();
void nativeSolver.init();
void farGpu.init();
void agentsGpu.init(gpuCanvas);
void sim.startBackgroundConfine();
window.setInterval(() => chart.sample(sim), 1000);
requestAnimationFrame(frame);
