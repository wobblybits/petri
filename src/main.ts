import './style.css';
import { Camera } from './camera.ts';
import { defaultParams, SLIDERS, type Params } from './params.ts';
import { loadPreset, type PresetName } from './presets.ts';
import { render } from './render.ts';
import { Sim } from './sim.ts';
import type { AgentKind } from './agents.ts';
import { audio } from './audio/engine.ts';
import { getWaveSpeed, setWaveSpeed } from './audio/presets.ts';

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('missing #app');

app.innerHTML = `
  <canvas id="view"></canvas>
  <aside id="panel">
    <h1>Interaction nets</h1>
    <p class="lede">Era, Dup, and Con forage on an infinite plane, latch ports, and rewrite like Lafont combinators.</p>
    <div class="row">
      <button type="button" id="pause">Pause</button>
      <button type="button" id="step">Step</button>
      <button type="button" id="reset">Reset</button>
    </div>
    <label class="check"><input type="checkbox" id="overlay" /> Field overlay</label>
    <label class="check"><input type="checkbox" id="sound" checked /> Sound</label>
    <p class="hint sound-hint" id="sound-hint">Click anywhere to enable sound.</p>
    <label class="slider">
      <span>Wave speed</span>
      <input type="range" id="wave-speed" min="0.35" max="2.8" step="0.05" value="1" />
      <span class="val" id="wave-speed-val">1</span>
    </label>
    <div class="spawn">
      <span>Spawn</span>
      <button type="button" data-kind="era" class="on">Era</button>
      <button type="button" data-kind="dup">Dup</button>
      <button type="button" data-kind="con">Con</button>
    </div>
    <div class="presets">
      <button type="button" data-preset="soup">Soup</button>
      <button type="button" data-preset="commute">γ–δ</button>
      <button type="button" data-preset="annihilate-con">γ–γ</button>
      <button type="button" data-preset="annihilate-dup">δ–δ</button>
      <button type="button" data-preset="oscillator">Oscillator</button>
    </div>
    <p class="hint">Click the canvas to spawn. New agents arrive every ~10s. Scroll to zoom. Keys E / D / C select type. Space pauses.</p>
    <p class="stats" id="stats"></p>
    <div id="sliders"></div>
  </aside>
`;

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const maybeCtx = canvas.getContext('2d');
if (!maybeCtx) throw new Error('no 2d context');
const ctx: CanvasRenderingContext2D = maybeCtx;

const params: Params = defaultParams();
const sim = new Sim(800, 600);
const camera = new Camera();
const view = { overlay: false };
let paused = false;
let spawnKind: AgentKind = 'era';
let currentPreset: PresetName = 'soup';
let snapCamera = true;

const pauseBtn = document.querySelector<HTMLButtonElement>('#pause')!;
const statsEl = document.querySelector<HTMLParagraphElement>('#stats')!;
const sliderRoot = document.querySelector<HTMLDivElement>('#sliders')!;
const soundHint = document.querySelector<HTMLParagraphElement>('#sound-hint')!;
const soundCheck = document.querySelector<HTMLInputElement>('#sound')!;
const waveSpeedInput = document.querySelector<HTMLInputElement>('#wave-speed')!;
const waveSpeedVal = document.querySelector<HTMLSpanElement>('#wave-speed-val')!;
let audioReady = false;

waveSpeedInput.value = String(getWaveSpeed());
waveSpeedVal.textContent = format(getWaveSpeed());
waveSpeedInput.addEventListener('input', () => {
  const n = Number(waveSpeedInput.value);
  setWaveSpeed(n);
  waveSpeedVal.textContent = format(n);
  audio.invalidateTopology();
});

for (const spec of SLIDERS) {
  const row = document.createElement('label');
  row.className = 'slider';
  const name = document.createElement('span');
  name.textContent = spec.label;
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(spec.min);
  input.max = String(spec.max);
  input.step = String(spec.step);
  input.value = String(params[spec.key]);
  const value = document.createElement('span');
  value.className = 'val';
  value.textContent = format(params[spec.key]);
  input.addEventListener('input', () => {
    const n = Number(input.value);
    (params[spec.key] as number) = n;
    value.textContent = format(n);
  });
  row.append(name, input, value);
  sliderRoot.append(row);
}

function format(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(n < 0.1 ? 3 : 2);
}

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
  sim.resize(w, h);
  camera.setView(w, h);
}

function syncCamera(dt: number): void {
  const com = sim.centerOfMass();
  if (com) {
    if (snapCamera) {
      camera.snap(com.x, com.y);
      snapCamera = false;
    } else {
      camera.follow(com.x, com.y, dt);
    }
  }
  sim.setFieldCover(camera.coverWidth() * 1.7, camera.coverHeight() * 1.7);
  sim.fields.cover(com?.x ?? camera.x, com?.y ?? camera.y, sim.coverW, sim.coverH);
}

function setPaused(next: boolean): void {
  paused = next;
  pauseBtn.textContent = paused ? 'Run' : 'Pause';
}

function setSpawn(kind: AgentKind): void {
  spawnKind = kind;
  for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-kind]')) {
    btn.classList.toggle('on', btn.dataset.kind === kind);
  }
}

function applyPreset(name: PresetName): void {
  currentPreset = name;
  sizeCanvas();
  loadPreset(sim, name, params);
  snapCamera = true;
  syncCamera(0);
}

pauseBtn.addEventListener('click', () => setPaused(!paused));
document.querySelector('#step')!.addEventListener('click', () => {
  sim.step(1 / 60, params);
  syncCamera(1 / 60);
  paint();
});
document.querySelector('#reset')!.addEventListener('click', () => applyPreset(currentPreset));
document.querySelector('#overlay')!.addEventListener('change', (ev) => {
  view.overlay = (ev.target as HTMLInputElement).checked;
});
soundCheck.addEventListener('change', () => {
  audio.setMuted(!soundCheck.checked);
});

async function bootAudio(): Promise<void> {
  if (audioReady) return;
  const ok = await audio.boot();
  audioReady = ok;
  soundHint.hidden = ok;
  soundHint.textContent = ok
    ? ''
    : 'Audio failed to start — check the browser console.';
  if (ok) audio.setMuted(!soundCheck.checked);
}

function armAudio(): void {
  void bootAudio();
}

document.querySelector('#app')!.addEventListener('pointerdown', armAudio, { once: true });
document.querySelector('#app')!.addEventListener('keydown', armAudio, { once: true });

for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-kind]')) {
  btn.addEventListener('click', () => setSpawn(btn.dataset.kind as AgentKind));
}
for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-preset]')) {
  btn.addEventListener('click', () => applyPreset(btn.dataset.preset as PresetName));
}

canvas.addEventListener('click', (ev) => {
  armAudio();
  const rect = canvas.getBoundingClientRect();
  const world = camera.worldFromScreen(ev.clientX - rect.left, ev.clientY - rect.top);
  sim.spawn(spawnKind, world.x, world.y, Math.random() * Math.PI * 2, params);
});

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
    setPaused(!paused);
  }
  if (ev.key === 'e' || ev.key === 'E') setSpawn('era');
  if (ev.key === 'd' || ev.key === 'D') setSpawn('dup');
  if (ev.key === 'c' || ev.key === 'C') setSpawn('con');
});

new ResizeObserver(() => sizeCanvas()).observe(canvas);

function paint(): void {
  render(ctx, sim, camera, view, audio.waves);
  statsEl.textContent = `${sim.agents.size} agents · ${sim.graph.wires.size} wires · ${sim.rewrites.length} rewrites`;
}

let last = performance.now();
function frame(now: number): void {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  try {
    if (!paused) sim.step(dt, params);
    audio.frame(sim.graph, sim.agents, dt, camera);
    syncCamera(dt);
    paint();
  } catch (err) {
    console.error(err);
  }
  requestAnimationFrame(frame);
}

sizeCanvas();
applyPreset('soup');
requestAnimationFrame(frame);
