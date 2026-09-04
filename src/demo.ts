import './style.css';
import { Camera } from './camera.ts';
import { Interaction, type Tool } from './interact.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset, splatter } from './presets.ts';
import { render, buildFarInstances, FAR_INSTANCE_STRIDE } from './render.ts';
import { Sim } from './sim.ts';
import { agentsGpu } from './gpu/agents-gpu.ts';
import { farGpu } from './gpu/far-gpu.ts';
import { nativeSolver } from './native/solver.ts';

const DEMO_SOUP = 5000;

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('missing #app');

app.classList.add('demo');
app.innerHTML = `
  <div id="viewport">
    <canvas id="view"></canvas>
    <canvas id="view-gpu"></canvas>
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
        <a class="icon-btn" id="github" href="https://github.com/wobblybits/petri" target="_blank" rel="noopener noreferrer" title="Source" aria-label="Source on GitHub">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" stroke="none" d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.157-1.11-1.465-1.11-1.465-.908-.62.069-.608.069-.608 1.004.07 1.532 1.03 1.532 1.03.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0 1 12 6.844a9.56 9.56 0 0 1 2.504.337c1.909-1.294 2.748-1.025 2.748-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482C19.138 20.162 22 16.416 22 12c0-5.523-4.477-10-10-10z"/></svg>
        </a>
      </div>
    </div>
    <aside id="about">
      <img class="logo" src="/petri-logo.png" alt="petri" />
    </aside>
  </div>
`;

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const maybeCtx = canvas.getContext('2d');
if (!maybeCtx) throw new Error('no 2d context');
const ctx: CanvasRenderingContext2D = maybeCtx;
const gpuCanvas = document.querySelector<HTMLCanvasElement>('#view-gpu')!;
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
requestAnimationFrame(frame);
