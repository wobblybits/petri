import './style.css';
import { Camera } from './camera.ts';
import { defaultParams, SLIDERS, type Params } from './params.ts';
import { loadPreset, type PresetName } from './presets.ts';
import { render } from './render.ts';
import { Interaction } from './interact.ts';
import { ap, church, injectTerm, readChurch, PLUS, type Term } from './lambda.ts';
import type { PortRef } from './agents.ts';
import { portWorld } from './agents.ts';
import { Sim } from './sim.ts';
import type { AgentKind } from './agents.ts';
import { audio } from './audio/engine.ts';
import { getWaveSpeed, setWaveSpeed } from './audio/presets.ts';
import { farGpu } from './gpu/far-gpu.ts';
import { nativeSolver } from './native/solver.ts';

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('missing #app');

app.innerHTML = `
  <canvas id="view"></canvas>
  <aside id="panel">
    <h1>Interaction nets</h1>
    <p class="lede">Era, Dup, and Con forage on an energy grid, latch ports, and rewrite like Lafont combinators. Annihilation releases energy; a Con–Dup commute spends it.</p>
    <div class="row">
      <button type="button" id="pause">Pause</button>
      <button type="button" id="step">Step</button>
      <button type="button" id="reset">Reset</button>
      <button type="button" id="recentre">Recentre</button>
    </div>
    <label class="check"><input type="checkbox" id="overlay" /> Field overlay</label>
    <label class="check"><input type="checkbox" id="energy-circles" /> Energy circles</label>
    <label class="check"><input type="checkbox" id="kind-colors" checked /> Kind colors</label>
    <label class="check"><input type="checkbox" id="energy-grid" /> Energy grid</label>
    <label class="check"><input type="checkbox" id="sound" /> Sound</label>
    <p class="hint sound-hint" id="sound-hint">Tick Sound to start the synth.</p>
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
      <button type="button" id="eraser">Eraser</button>
    </div>
    <div class="presets lambda">
      <span>λ</span>
      <button type="button" data-lambda="0">0</button>
      <button type="button" data-lambda="1">1</button>
      <button type="button" data-lambda="+">+</button>
      <button type="button" data-lambda="2+3">2+3</button>
      <button type="button" data-lambda="1+1">1+1</button>
      <button type="button" data-lambda="4+5">4+5</button>
      <button type="button" data-lambda="0+3">0+3</button>
    </div>
    <p class="hint" id="lambda-out"></p>
    <div class="presets">
      <button type="button" data-preset="soup">Soup</button>
      <button type="button" data-preset="commute">γ–δ</button>
      <button type="button" data-preset="annihilate-con">γ–γ</button>
      <button type="button" data-preset="annihilate-dup">δ–δ</button>
      <button type="button" data-preset="oscillator">Oscillator</button>
    </div>
    <p class="hint">Click empty space to spawn, drag it to pan. Drag a body to move it; drag from one free port to another to wire them. Eraser: click and drag to remove bodies under the cursor. Scroll to zoom. Keys E / D / C select type, X toggles the eraser. Space pauses.</p>
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
const interaction = new Interaction(sim, camera);
const view = { overlay: false, energyGrid: false, energyCircles: false, kindColors: true };
let paused = false;
let spawnKind: AgentKind = 'era';
let currentPreset: PresetName = 'soup';
let snapCamera = true;

const pauseBtn = document.querySelector<HTMLButtonElement>('#pause')!;
const eraserBtn = document.querySelector<HTMLButtonElement>('#eraser')!;
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
  if (com && !interaction.freeCamera) {
    if (snapCamera) {
      camera.snap(com.x, com.y);
      snapCamera = false;
    } else {
      camera.follow(com.x, com.y, dt);
    }
  }
  // The field follows `home` from inside beginFrame now, so there is nothing
  // to do here but tell the sim how much world is on screen for auto-spawn.
  sim.setViewExtent(camera.coverWidth() * 1.7, camera.coverHeight() * 1.7);
}

function setPaused(next: boolean): void {
  paused = next;
  pauseBtn.textContent = paused ? 'Run' : 'Pause';
}

function setSpawn(kind: AgentKind): void {
  spawnKind = kind;
  interaction.eraserMode = false;
  eraserBtn.classList.remove('on');
  for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-kind]')) {
    btn.classList.toggle('on', btn.dataset.kind === kind);
  }
}

function setEraser(on: boolean): void {
  interaction.eraserMode = on;
  eraserBtn.classList.toggle('on', on);
  if (on) {
    for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-kind]')) {
      btn.classList.remove('on');
    }
  } else {
    for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-kind]')) {
      btn.classList.toggle('on', btn.dataset.kind === spawnKind);
    }
  }
}

function applyPreset(name: PresetName): void {
  lambdaRoot = null;
  currentPreset = name;
  sizeCanvas();
  loadPreset(sim, name, params);
  snapCamera = true;
  syncCamera(0);
}

pauseBtn.addEventListener('click', () => setPaused(!paused));
document.querySelector('#step')!.addEventListener('click', () => {
  void sim.stepAsync(1 / 60, params, camera).then(() => {
    syncCamera(1 / 60);
    paint();
  });
});
document.querySelector('#reset')!.addEventListener('click', () => applyPreset(currentPreset));
document.querySelector('#recentre')!.addEventListener('click', () => {
  interaction.freeCamera = false;
  snapCamera = true;
});
document.querySelector('#overlay')!.addEventListener('change', (ev) => {
  view.overlay = (ev.target as HTMLInputElement).checked;
});
document.querySelector('#energy-circles')!.addEventListener('change', (ev) => {
  view.energyCircles = (ev.target as HTMLInputElement).checked;
});
document.querySelector('#kind-colors')!.addEventListener('change', (ev) => {
  view.kindColors = (ev.target as HTMLInputElement).checked;
});
document.querySelector('#energy-grid')!.addEventListener('change', (ev) => {
  view.energyGrid = (ev.target as HTMLInputElement).checked;
});
soundCheck.addEventListener('change', () => {
  // Booting builds the whole synthesis graph, so a pond nobody is listening to
  // never pays for one. Ticking the box is the gesture that boots it, and it
  // counts as the user activation an AudioContext needs.
  if (soundCheck.checked && !audioReady) {
    void bootAudio();
    return;
  }
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
  if (soundCheck.checked) void bootAudio();
}

// Dev-only handle for checking what the audio path is actually doing:
// whether synthesis moved to the worker, how full the ring is, and whether
// the callback has had to pad. None of it is reachable from the UI, and none
// of it ships.
if (import.meta.env.DEV) {
  (globalThis as unknown as { swimmers: unknown }).swimmers = {
    audio,
    sim,
    params,
    get ring() {
      return { fill: audio.ringFill, underruns: audio.underruns, shards: audio.shardFills, sharded: audio.sharded };
    },
    boot: () => bootAudio(),
  };
}

document.querySelector('#app')!.addEventListener('pointerdown', armAudio, { once: true });
document.querySelector('#app')!.addEventListener('keydown', armAudio, { once: true });

for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-kind]')) {
  btn.addEventListener('click', () => setSpawn(btn.dataset.kind as AgentKind));
}
eraserBtn.addEventListener('click', () => setEraser(!interaction.eraserMode));
for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-preset]')) {
  btn.addEventListener('click', () => applyPreset(btn.dataset.preset as PresetName));
}

function pointerWorld(ev: PointerEvent): { wx: number; wy: number; sx: number; sy: number } {
  const rect = canvas.getBoundingClientRect();
  const sx = ev.clientX - rect.left;
  const sy = ev.clientY - rect.top;
  const w = camera.worldFromScreen(sx, sy);
  return { wx: w.x, wy: w.y, sx, sy };
}

canvas.addEventListener('pointerdown', (ev) => {
  armAudio();
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
  const spawnAt = interaction.end(wx, wy, (a, b) => {
    const A = sim.agents.get(a.id);
    const B = sim.agents.get(b.id);
    if (!A || !B) return;
    sim.wire(a.id, a.slot, b.id, b.slot, params);
  });
  if (spawnAt) sim.spawn(spawnKind, spawnAt.x, spawnAt.y, Math.random() * Math.PI * 2, params);
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
    setPaused(!paused);
  }
  if (ev.key === 'e' || ev.key === 'E') setSpawn('era');
  if (ev.key === 'd' || ev.key === 'D') setSpawn('dup');
  if (ev.key === 'c' || ev.key === 'C') setSpawn('con');
  if (ev.key === 'x' || ev.key === 'X') setEraser(!interaction.eraserMode);
});

new ResizeObserver(() => sizeCanvas()).observe(canvas);

function drawGesture(): void {
  const g = interaction.gesture;
  if (g.kind !== 'wire') return;
  const from = sim.agents.get(g.from.id);
  if (!from) return;
  const a = portWorld(from, g.from.slot, sim.w, sim.h);
  ctx.save();
  camera.apply(ctx);
  ctx.lineWidth = 1.5 / camera.zoom;
  ctx.strokeStyle = g.over ? '#7ef0c8' : '#8899aa';
  ctx.setLineDash(g.over ? [] : [4 / camera.zoom, 4 / camera.zoom]);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(g.x, g.y);
  ctx.stroke();
  ctx.setLineDash([]);
  for (const [ref, hot] of [[g.from, false], [g.over, true]] as const) {
    if (!ref) continue;
    const agent = sim.agents.get(ref.id);
    if (!agent) continue;
    const p = portWorld(agent, ref.slot, sim.w, sim.h);
    ctx.strokeStyle = hot ? '#7ef0c8' : '#cfd8e3';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5 / camera.zoom, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function paint(): void {
  render(ctx, sim, camera, view, audio.waves);
  drawGesture();
  statsEl.textContent = `${sim.agents.size} agents · ${sim.graph.wires.size} wires · ${sim.rewrites.length} rewrites · ${format(sim.totalFree())} extra / ${sim.totalBound()} bound`;
  if (lambdaRoot) {
    const value = readChurch(sim.agents, sim.graph, lambdaRoot);
    if (value !== null) lambdaOut.textContent = `${lambdaLabel} = ${value}`;
    else if (lambdaExpectNumeral) lambdaOut.textContent = `${lambdaLabel} → reducing…`;
    else lambdaOut.textContent = lambdaLabel;
  } else {
    lambdaOut.textContent = '';
  }
}

let lambdaRoot: PortRef | null = null;
let lambdaLabel = '';
let lambdaExpectNumeral = false;
const lambdaOut = document.querySelector<HTMLParagraphElement>('#lambda-out')!;

function lambdaSpec(spec: string): { term: Term; label: string; numeral: boolean } {
  if (spec === '0') return { term: church(0), label: '0', numeral: true };
  if (spec === '1') return { term: church(1), label: '1', numeral: true };
  if (spec === '+') return { term: PLUS, label: '+', numeral: false };
  const [a, b] = spec.split('+').map(Number);
  return { term: ap(PLUS, church(a), church(b)), label: `${a} + ${b}`, numeral: true };
}

/**
 * Drop a compiled term into an empty world. Numerals and sums are read back as
 * they reduce; combinators just sit as nets.
 */
function runLambda(spec: string): void {
  const { term, label, numeral } = lambdaSpec(spec);
  sim.clear();
  params.spawnInterval = 0;
  lambdaLabel = label;
  lambdaExpectNumeral = numeral;
  const { root } = injectTerm(sim, term, sim.w * 0.5, sim.h * 0.5, params);
  lambdaRoot = root;
  interaction.freeCamera = false;
  snapCamera = true;
  currentPreset = 'soup';
}

for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-lambda]')) {
  btn.addEventListener('click', () => {
    armAudio();
    runLambda(btn.dataset.lambda!);
  });
}

let last = performance.now();
let ticking = false;
async function tick(now: number): Promise<void> {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  try {
    if (paused) sim.dragStep(params, dt, camera);
    else await sim.stepAsync(dt, params, camera);
    audio.frame(sim.graph, sim.agents, dt, camera);
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
applyPreset('soup');
void nativeSolver.init();
void farGpu.init();
requestAnimationFrame(frame);
