import './style.css';
import { Camera } from './camera.ts';
import { NetEditor, ROTATE_RIGHT_ANGLE, ROTATE_STEP, type DesignTool } from './net-edit.ts';
import { looksLikeNet } from './net-text.ts';
import { attackSound, holdSound, moveStroke, releaseSound } from './net-sound.ts';
import { GALLERY_TABS, catalogPieces, galleryHint, type GalleryTabId } from './net-catalog.ts';
import {
  addPiece,
  loadGallery,
  removePiece,
  renamePiece,
  storeGallery,
} from './net-gallery.ts';
import { audio } from './audio/engine.ts';
import { defaultParams, type Params } from './params.ts';
import { render, buildFarInstances, FAR_INSTANCE_STRIDE } from './render.ts';
import { Sim } from './sim.ts';
import { agentsGpu } from './gpu/agents-gpu.ts';
import { farGpu } from './gpu/far-gpu.ts';
import { nativeSolver } from './native/solver.ts';

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('missing #app');

app.classList.add('demo', 'design');
document.documentElement.classList.add('demo', 'design');
app.innerHTML = `
  <div class="demo-chrome">
    <aside id="about">
      <p class="design-title">Net designer</p>
    </aside>
    <div id="hud">
      <div class="hud-tools">
        <button type="button" class="icon-btn on" data-tool="select" data-tip="Select" data-kbd="V" aria-label="Select">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6 14 4l2 10-4 1-2 5-2-1 2-5z"/></svg>
        </button>
        <button type="button" class="icon-btn" data-tool="pan" data-tip="Pan" data-kbd="H" aria-label="Pan">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 11V7a2 2 0 1 1 4 0v4"/><path d="M12 11V6a2 2 0 1 1 4 0v5"/><path d="M16 12V8a2 2 0 1 1 4 0v6c0 4-2.5 7-8 7h-1c-4 0-7-2-7-6v-5a2 2 0 1 1 4 0v3"/></svg>
        </button>
        <span class="hud-sep"></span>
        <button type="button" class="icon-btn kind-era" data-tool="paint-era" data-tip="Era" data-kbd="E" aria-label="Era">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="6.5"/></svg>
        </button>
        <button type="button" class="icon-btn kind-dup" data-tool="paint-dup" data-tip="Dup" data-kbd="D" aria-label="Dup">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4 20 19H4z"/></svg>
        </button>
        <button type="button" class="icon-btn kind-con" data-tool="paint-con" data-tip="Con" data-kbd="C" aria-label="Con">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4 20 19H4z"/></svg>
        </button>
        <button type="button" class="icon-btn" data-tool="erase" data-tip="Erase bodies and wires" data-kbd="X" aria-label="Erase bodies and wires">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h16"/><path d="M5.5 13.5 14 5a2.1 2.1 0 0 1 3 0l2 2a2.1 2.1 0 0 1 0 3l-8.5 8.5H5.5z"/></svg>
        </button>
        <button type="button" class="icon-btn" data-tool="rotate" data-tip="Rotate" data-kbd="R" aria-label="Rotate">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.3-5.6"/><path d="M4 4v5h5"/></svg>
        </button>
        <button type="button" class="icon-btn" data-tool="touch" data-tip="Touch" data-kbd="T" aria-label="Touch">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="8.5" cy="8.5" r="3.2"/><path d="M10.8 10.8 20 20"/></svg>
        </button>
        <button type="button" class="icon-btn" id="cycle" data-tip="Cycle ports" data-kbd="Tab" aria-label="Cycle ports">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="6.5" r="2"/><circle cx="6.8" cy="16.2" r="2"/><circle cx="17.2" cy="16.2" r="2"/><path d="M12 8.5 7.8 14.2M12 8.5l4.2 5.7"/></svg>
        </button>
        <button type="button" class="icon-btn" id="pin" data-tip="Lock position" data-kbd="L" aria-label="Lock position">
          <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="11" width="12" height="10" rx="1.5"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>
        </button>
        <span class="hud-sep"></span>
        <button type="button" class="icon-btn" id="undo" data-tip="Undo" data-kbd="⌘Z" aria-label="Undo">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 8H5V4"/><path d="M5 8a9 9 0 1 1-1.2 6"/></svg>
        </button>
        <button type="button" class="icon-btn" id="redo" data-tip="Redo" data-kbd="⇧⌘Z" aria-label="Redo">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 8h4V4"/><path d="M19 8a9 9 0 1 0 1.2 6"/></svg>
        </button>
        <button type="button" class="icon-btn" id="cut" data-tip="Cut" data-kbd="⌘X" aria-label="Cut">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="18" r="2.2"/><circle cx="18" cy="18" r="2.2"/><path d="M8 16.2 20 4M16 16.2 4 4"/></svg>
        </button>
        <button type="button" class="icon-btn" id="copy" data-tip="Copy" data-kbd="⌘C" aria-label="Copy">
          <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="1.5"/><path d="M5 16V5h11"/></svg>
        </button>
        <button type="button" class="icon-btn" id="paste" data-tip="Paste" data-kbd="⌘V" aria-label="Paste">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5h8"/><path d="M9 5.5V4h6v1.5"/><rect x="6" y="5.5" width="12" height="14.5" rx="1.5"/></svg>
        </button>
        <span class="hud-sep"></span>
        <button type="button" class="icon-btn" id="save-net" data-tip="Save net" data-kbd="⌘S" aria-label="Save net">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h11l3 3v11H5z"/><path d="M8 5v5h8V5"/><path d="M8 16h8v3H8z"/></svg>
        </button>
        <button type="button" class="icon-btn" id="load-net" data-tip="Load net" data-kbd="⌘O" aria-label="Load net">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h5l2 2h9v8H4z"/><path d="M4 8V6h4l1.5 2"/></svg>
        </button>
        <input type="file" id="load-net-file" accept=".hvm,.hvm2,.txt,text/plain" hidden />
        <span class="hud-sep"></span>
        <button type="button" class="icon-btn" id="play" data-tip="Play" data-kbd="P" aria-label="Play">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/></svg>
        </button>
        <button type="button" class="icon-btn" id="settle" data-tip="Settle" data-kbd="Space" aria-label="Settle">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/><path d="M5 20h14"/></svg>
        </button>
        <button type="button" class="icon-btn" id="listen" data-tip="Listen" data-kbd="M" aria-label="Listen">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 9v6h4l5 4V5L9 9H5z"/><path d="M16.2 8.8a4.2 4.2 0 0 1 0 6.4"/><path d="M18.5 6.4a7.5 7.5 0 0 1 0 11.2"/></svg>
        </button>
        <button type="button" class="icon-btn" id="clear" data-tip="Clear" aria-label="Clear">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4v7h7"/><path d="M20 20v-7h-7"/><path d="M5.2 9.2A8 8 0 0 1 19 8.5M18.8 14.8A8 8 0 0 1 5 15.5"/></svg>
        </button>
      </div>
    </div>
  </div>
  <div id="viewport">
    <canvas id="view"></canvas>
    <canvas id="view-gpu"></canvas>
  </div>
  <aside id="gallery">
    <header class="gallery-head">
      <h1>Gallery</h1>
      <button type="button" id="gallery-add" data-tip="Save piece">Save</button>
    </header>
    <nav class="gallery-tabs" id="gallery-tabs"></nav>
    <p class="gallery-hint" id="gallery-hint">Saves the selection, or the whole dish. Click a piece to stamp it.</p>
    <ul id="gallery-list"></ul>
  </aside>
`;

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const maybeCtx = canvas.getContext('2d');
if (!maybeCtx) throw new Error('no 2d context');
const ctx: CanvasRenderingContext2D = maybeCtx;
const gpuCanvas = document.querySelector<HTMLCanvasElement>('#view-gpu')!;
const undoBtn = document.querySelector<HTMLButtonElement>('#undo')!;
const redoBtn = document.querySelector<HTMLButtonElement>('#redo')!;
const playBtn = document.querySelector<HTMLButtonElement>('#play')!;
const settleBtn = document.querySelector<HTMLButtonElement>('#settle')!;
const listenBtn = document.querySelector<HTMLButtonElement>('#listen')!;
const pinBtn = document.querySelector<HTMLButtonElement>('#pin')!;
const loadNetFile = document.querySelector<HTMLInputElement>('#load-net-file')!;
const galleryList = document.querySelector<HTMLUListElement>('#gallery-list')!;
const galleryAdd = document.querySelector<HTMLButtonElement>('#gallery-add')!;
const galleryTabs = document.querySelector<HTMLElement>('#gallery-tabs')!;
const galleryHintEl = document.querySelector<HTMLParagraphElement>('#gallery-hint')!;

const live = defaultParams();
const params: Params = defaultParams();
params.spawnInterval = 0;
params.snapRadius = 0;
params.rewriteDuration = 0;
params.stepSpeed = 0;
params.swimNoise = 0;
params.upkeep = 0;
params.emitCost = 0;
params.ambientEnergy = 1e6;

const sim = new Sim(800, 600);
sim.energy.inexhaustible = true;
sim.breed = false;
const camera = new Camera();
const editor = new NetEditor(sim, camera, params);
const view = { overlay: false, energyGrid: false, energyCircles: false, kindColors: true, gpuAgents: false };
let farInstances = new Float32Array(0);
type RunMode = 'stop' | 'settle' | 'play' | 'listen';
let runMode: RunMode = 'stop';
let audioReady = false;
/** Pluck existing wires once Listen is armed; topology alone is silent. */
let listenKick = false;

function cssSize(): { w: number; h: number } {
  const rect = canvas.getBoundingClientRect();
  return { w: Math.max(1, rect.width), h: Math.max(1, rect.height) };
}

function ensureDish(): void {
  if (sim.worldR > 0) return;
  const { w, h } = cssSize();
  const com = sim.centerOfMass();
  sim.pinWorld(com?.x ?? w * 0.5, com?.y ?? h * 0.5, params);
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
  ensureDish();
}

function setTool(next: DesignTool): void {
  editor.tool = next;
  for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-tool]')) {
    btn.classList.toggle('on', btn.dataset.tool === editor.tool);
  }
}

function setRunMode(mode: RunMode, opts: { force?: boolean; kick?: boolean } = {}): void {
  const next: RunMode = !opts.force && runMode === mode ? 'stop' : mode;
  if ((next === 'play' || next === 'listen') && runMode !== next) editor.checkpoint();
  runMode = next;
  // Play rewrites existing principal pairs. Snap stays off so free ports
  // cannot grow new wires; leftover connections from a rewrite still appear.
  params.snapRadius = 0;
  params.rewriteDuration = runMode === 'play' ? live.rewriteDuration : 0;
  params.stepSpeed = runMode === 'play' ? live.stepSpeed : 0;
  playBtn.classList.toggle('on', runMode === 'play');
  settleBtn.classList.toggle('on', runMode === 'settle');
  listenBtn.classList.toggle('on', runMode === 'listen');
  audio.setMuted(runMode !== 'listen');
  listenKick = runMode === 'listen' && opts.kick !== false;
  if (runMode === 'listen') void bootAudio();
}

/** First mallet hit is the user-activation AudioContext wants. */
function ensureListen(): void {
  if (runMode === 'listen') {
    void bootAudio();
    return;
  }
  setRunMode('listen', { force: true, kick: false });
}

async function bootAudio(): Promise<void> {
  if (audioReady) {
    audio.setMuted(runMode !== 'listen');
    return;
  }
  const ok = await audio.boot();
  audioReady = ok;
  if (!ok) {
    console.warn('Audio failed to start — check the browser console.');
    listenBtn.classList.remove('on');
    if (runMode === 'listen') runMode = 'stop';
    return;
  }
  audio.setMuted(runMode !== 'listen');
}

/** Topology does not hum. A latch (or spawn) is the impulse that starts it. */
function kickListen(): void {
  let n = 0;
  for (const w of sim.graph.wires.values()) {
    const A = sim.agents.get(w.a.id);
    const B = sim.agents.get(w.b.id);
    if (!A || !B) continue;
    audio.push(
      {
        type: 'latch',
        wireId: w.id,
        agentA: A.id,
        agentB: B.id,
        slotA: w.a.slot,
        slotB: w.b.slot,
        kindA: A.kind,
        kindB: B.kind,
        rest: w.rest,
        latchLen: sim.graph.stemSpan(w, sim.agents, sim.w, sim.h),
      },
      sim.graph,
      sim.agents,
    );
    if (++n >= 5) return;
  }
  if (n > 0) return;
  for (const a of sim.agents.values()) {
    audio.push({ type: 'spawn', agent: a.id, kind: a.kind }, sim.graph, sim.agents);
    if (++n >= 5) return;
  }
}

function syncHistory(): void {
  undoBtn.disabled = !editor.canUndo;
  redoBtn.disabled = !editor.canRedo;
  pinBtn.classList.toggle('on', editor.selectionIsPinned());
}

function paint(): void {
  view.gpuAgents = agentsGpu.ready;
  if (view.gpuAgents) {
    const need = sim.agents.size * FAR_INSTANCE_STRIDE;
    if (farInstances.length < need) farInstances = new Float32Array(need);
    const count = buildFarInstances(sim, farInstances, view.kindColors);
    agentsGpu.render(farInstances, count, camera.gpuView());
  }
  render(ctx, sim, camera, view, runMode === 'listen' ? audio.waves : null);
  editor.drawOverlay(ctx);
  syncHistory();
  galleryAdd.disabled = sim.agents.size === 0;
}

for (const btn of document.querySelectorAll<HTMLButtonElement>('[data-tool]')) {
  btn.addEventListener('click', () => setTool(btn.dataset.tool as DesignTool));
}
document.querySelector('#cycle')!.addEventListener('click', () => {
  editor.cycleSelection(1);
  paint();
});
pinBtn.addEventListener('click', () => {
  editor.togglePinSelection();
  paint();
});
undoBtn.addEventListener('click', () => {
  editor.undo();
  ensureDish();
  paint();
});
redoBtn.addEventListener('click', () => {
  editor.redo();
  ensureDish();
  paint();
});
document.querySelector('#cut')!.addEventListener('click', () => {
  editor.cut();
  paint();
});
document.querySelector('#copy')!.addEventListener('click', () => {
  editor.copy();
});
document.querySelector('#paste')!.addEventListener('click', () => {
  editor.paste();
  paint();
});
document.querySelector('#save-net')!.addEventListener('click', () => saveNet());
document.querySelector('#load-net')!.addEventListener('click', () => loadNetFile.click());
loadNetFile.addEventListener('change', () => {
  void openNetFile();
});
playBtn.addEventListener('click', () => setRunMode('play'));
settleBtn.addEventListener('click', () => setRunMode('settle'));
listenBtn.addEventListener('click', () => setRunMode('listen'));
document.querySelector('#clear')!.addEventListener('click', () => {
  editor.clearWorld();
  ensureDish();
  paint();
});

function pointerWorld(ev: PointerEvent): { wx: number; wy: number; sx: number; sy: number } {
  const rect = canvas.getBoundingClientRect();
  const sx = ev.clientX - rect.left;
  const sy = ev.clientY - rect.top;
  const w = camera.worldFromScreen(sx, sy);
  return { wx: w.x, wy: w.y, sx, sy };
}

const pointers = new Map<number, { sx: number; sy: number }>();
let pinch: { dist: number; zoom: number } | null = null;

function pinchPair(): { ax: number; ay: number; bx: number; by: number } | null {
  if (pointers.size < 2) return null;
  const [a, b] = [...pointers.values()];
  if (!a || !b) return null;
  return { ax: a.sx, ay: a.sy, bx: b.sx, by: b.sy };
}

canvas.addEventListener('pointerdown', (ev) => {
  const { wx, wy, sx, sy } = pointerWorld(ev);
  pointers.set(ev.pointerId, { sx, sy });
  if (pointers.size >= 2) {
    editor.cancel();
    const pair = pinchPair();
    if (pair) {
      pinch = {
        dist: Math.hypot(pair.bx - pair.ax, pair.by - pair.ay),
        zoom: camera.zoom,
      };
    }
    return;
  }
  try {
    canvas.setPointerCapture(ev.pointerId);
  } catch {
    /* Synthetic events have no hardware pointer. */
  }
  const pan = ev.altKey || ev.button === 1;
  editor.begin(wx, wy, sx, sy, {
    shift: ev.shiftKey,
    pan,
    pressure: ev.pressure,
    now: performance.now(),
  });
  if (editor.gesture.kind === 'sound') {
    ensureListen();
    attackSound(sim, audio, editor.gesture.stroke);
  }
});
canvas.addEventListener('pointermove', (ev) => {
  const { wx, wy, sx, sy } = pointerWorld(ev);
  if (pointers.has(ev.pointerId)) pointers.set(ev.pointerId, { sx, sy });
  const pair = pinch && pinchPair();
  if (pair && pinch && pinch.dist > 1) {
    const dist = Math.hypot(pair.bx - pair.ax, pair.by - pair.ay);
    const mx = (pair.ax + pair.bx) * 0.5;
    const my = (pair.ay + pair.by) * 0.5;
    editor.freeCamera = true;
    camera.zoomAt((pinch.zoom * (dist / pinch.dist)) / camera.zoom, mx, my);
    paint();
    return;
  }
  editor.move(wx, wy, sx, sy, { pressure: ev.pressure, now: performance.now() });
  canvas.style.cursor = editor.cursorFor(wx, wy);
});
canvas.addEventListener('pointerup', (ev) => {
  const { wx, wy } = pointerWorld(ev);
  pointers.delete(ev.pointerId);
  if (pinch && pointers.size < 2) {
    pinch = null;
    if (pointers.size === 1) {
      const left = [...pointers.values()][0];
      const w = camera.worldFromScreen(left.sx, left.sy);
      editor.begin(w.x, w.y, left.sx, left.sy);
    }
    return;
  }
  if (editor.gesture.kind === 'sound') {
    const stroke = editor.gesture.stroke;
    moveStroke(stroke, wx, wy, performance.now(), ev.pressure);
    releaseSound(sim, audio, stroke);
  }
  editor.end(wx, wy);
  paint();
});
canvas.addEventListener('pointercancel', (ev) => {
  pointers.delete(ev.pointerId);
  pinch = null;
  if (editor.gesture.kind === 'sound') releaseSound(sim, audio, editor.gesture.stroke);
  editor.cancel();
});
canvas.addEventListener(
  'wheel',
  (ev) => {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    camera.zoomAt(Math.exp(-ev.deltaY * 0.0015), ev.clientX - rect.left, ev.clientY - rect.top);
    paint();
  },
  { passive: false },
);
canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());

function swallowPageZoom(ev: Event): void {
  ev.preventDefault();
}
window.addEventListener('gesturestart', swallowPageZoom, { capture: true, passive: false });
window.addEventListener('gesturechange', swallowPageZoom, { capture: true, passive: false });
window.addEventListener('gestureend', swallowPageZoom, { capture: true, passive: false });
window.addEventListener(
  'touchmove',
  (ev) => {
    if (ev.touches.length > 1) ev.preventDefault();
  },
  { capture: true, passive: false },
);
window.addEventListener(
  'wheel',
  (ev) => {
    if (ev.ctrlKey || ev.metaKey) ev.preventDefault();
  },
  { capture: true, passive: false },
);

function isBrowserZoomKey(ev: KeyboardEvent): boolean {
  if (!(ev.ctrlKey || ev.metaKey)) return false;
  return (
    ev.key === '+' ||
    ev.key === '-' ||
    ev.key === '=' ||
    ev.key === '_' ||
    ev.key === '0' ||
    ev.code === 'NumpadAdd' ||
    ev.code === 'NumpadSubtract' ||
    ev.code === 'Numpad0'
  );
}

function isTyping(ev: KeyboardEvent): boolean {
  const el = ev.target;
  if (!(el instanceof HTMLElement)) return false;
  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) return true;
  if (el.isContentEditable) return true;
  return Boolean(el.closest('#gallery'));
}

function applyImportedNet(): void {
  ensureDish();
  const com = sim.centerOfMass();
  if (com) camera.snap(com.x, com.y);
  paint();
}

function saveNet(): void {
  const text = editor.exportText();
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'net.hvm';
  a.click();
  URL.revokeObjectURL(url);
}

async function openNetFile(): Promise<void> {
  const file = loadNetFile.files?.[0];
  loadNetFile.value = '';
  if (!file) return;
  const text = await file.text();
  if (!editor.importText(text)) return;
  applyImportedNet();
}

function loadHash(): void {
  const raw = location.hash.startsWith('#') ? location.hash.slice(1) : '';
  if (!raw) return;
  let text = raw;
  try {
    text = decodeURIComponent(raw);
  } catch {
    /* keep raw */
  }
  if (!looksLikeNet(text)) return;
  if (!editor.importText(text)) return;
  applyImportedNet();
}

let gallery = loadGallery();
let galleryTab: GalleryTabId = 'yours';

function persistGallery(): void {
  storeGallery(gallery);
}

function stampPoint(): { x: number; y: number } {
  return camera.worldFromScreen(camera.screenCX, camera.screenCY);
}

function previewNet(text: string): string {
  const compact = text.replace(/\s+/g, ' ').trim();
  if (compact.length <= 80) return compact;
  return `${compact.slice(0, 79)}…`;
}

function stampPiece(text: string): void {
  const at = stampPoint();
  editor.insertText(text, at.x, at.y);
  paint();
}

function renderGallery(): void {
  const yours = galleryTab === 'yours';
  galleryAdd.hidden = !yours;
  galleryHintEl.textContent = galleryHint(galleryTab);

  galleryTabs.replaceChildren();
  for (const tab of GALLERY_TABS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'gallery-tab';
    btn.classList.toggle('on', tab.id === galleryTab);
    btn.textContent = tab.label;
    btn.addEventListener('click', () => {
      galleryTab = tab.id;
      renderGallery();
    });
    galleryTabs.append(btn);
  }

  galleryList.replaceChildren();
  if (yours) {
    if (gallery.length === 0) {
      const empty = document.createElement('li');
      empty.className = 'gallery-empty';
      empty.textContent = 'Nothing saved yet.';
      galleryList.append(empty);
      return;
    }
    for (const piece of gallery) {
      galleryList.append(userPieceItem(piece.id, piece.name, piece.text));
    }
    return;
  }

  for (const piece of catalogPieces(galleryTab)) {
    galleryList.append(catalogPieceItem(piece.name, piece.text));
  }
}

function userPieceItem(id: string, name: string, text: string): HTMLLIElement {
  const item = document.createElement('li');
  item.className = 'gallery-item';
  item.dataset.id = id;

  const input = document.createElement('input');
  input.className = 'gallery-name';
  input.type = 'text';
  input.value = name;
  input.autocomplete = 'off';
  input.spellcheck = false;
  input.ariaLabel = 'Piece name';
  input.addEventListener('change', () => {
    gallery = renamePiece(gallery, id, input.value);
    persistGallery();
    renderGallery();
  });
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      input.blur();
    }
  });

  const del = document.createElement('button');
  del.type = 'button';
  del.className = 'gallery-delete';
  del.setAttribute('aria-label', `Remove ${name}`);
  del.textContent = '×';
  del.addEventListener('click', () => {
    gallery = removePiece(gallery, id);
    persistGallery();
    renderGallery();
  });

  const top = document.createElement('div');
  top.className = 'gallery-item-top';
  top.append(input, del);
  item.append(top, stampButton(text, text));
  return item;
}

function catalogPieceItem(name: string, text: string): HTMLLIElement {
  const item = document.createElement('li');
  item.className = 'gallery-item';
  const label = document.createElement('span');
  label.className = 'gallery-label';
  label.textContent = name;
  const top = document.createElement('div');
  top.className = 'gallery-item-top';
  top.append(label);
  item.append(top, stampButton(text, previewNet(text)));
  return item;
}

function stampButton(text: string, preview: string): HTMLButtonElement {
  const stamp = document.createElement('button');
  stamp.type = 'button';
  stamp.className = 'gallery-stamp';
  stamp.title = 'Stamp onto the dish';
  const src = document.createElement('code');
  src.textContent = preview;
  stamp.append(src);
  stamp.addEventListener('click', () => stampPiece(text));
  return stamp;
}

galleryAdd.addEventListener('click', () => {
  const text = editor.exportPiece();
  const next = addPiece(gallery, text);
  if (next === gallery) return;
  gallery = next;
  persistGallery();
  renderGallery();
});

renderGallery();

window.addEventListener('keydown', (ev) => {
  if (isBrowserZoomKey(ev)) {
    ev.preventDefault();
    return;
  }
  if (isTyping(ev)) return;
  const mod = ev.metaKey || ev.ctrlKey;
  if (mod && ev.key.toLowerCase() === 'z') {
    ev.preventDefault();
    if (ev.shiftKey) editor.redo();
    else editor.undo();
    ensureDish();
    paint();
    return;
  }
  if (mod && ev.key.toLowerCase() === 'y') {
    ev.preventDefault();
    editor.redo();
    ensureDish();
    paint();
    return;
  }
  if (mod && ev.key.toLowerCase() === 'c') {
    ev.preventDefault();
    editor.copy();
    return;
  }
  if (mod && ev.key.toLowerCase() === 'x') {
    ev.preventDefault();
    editor.cut();
    paint();
    return;
  }
  if (mod && ev.key.toLowerCase() === 'v') {
    ev.preventDefault();
    editor.paste();
    paint();
    return;
  }
  if (mod && ev.key.toLowerCase() === 's') {
    ev.preventDefault();
    saveNet();
    return;
  }
  if (mod && ev.key.toLowerCase() === 'o') {
    ev.preventDefault();
    loadNetFile.click();
    return;
  }
  if (mod && ev.key.toLowerCase() === 'a') {
    ev.preventDefault();
    editor.selectAll();
    paint();
    return;
  }
  if (ev.code === 'Space') {
    ev.preventDefault();
    setRunMode('settle');
    return;
  }
  if (ev.key === 'p' || ev.key === 'P') {
    ev.preventDefault();
    setRunMode('play');
    return;
  }
  if (ev.key === 'm' || ev.key === 'M') {
    ev.preventDefault();
    setRunMode('listen');
    return;
  }
  if (ev.key === 'Tab') {
    ev.preventDefault();
    editor.cycleSelection(ev.shiftKey ? -1 : 1);
    paint();
    return;
  }
  if (ev.key === 'l' || ev.key === 'L') {
    ev.preventDefault();
    editor.togglePinSelection();
    paint();
    return;
  }
  if (ev.key === '[' || ev.key === ']') {
    ev.preventDefault();
    const step = ev.shiftKey ? ROTATE_RIGHT_ANGLE : ROTATE_STEP;
    editor.rotateSelection(ev.key === ']' ? step : -step);
    paint();
    return;
  }
  if (ev.key === 'Escape') {
    editor.cancel();
    editor.clearSelection();
    paint();
    return;
  }
  if (ev.key === 'Backspace' || ev.key === 'Delete') {
    ev.preventDefault();
    editor.deleteSelection();
    paint();
    return;
  }
  if (ev.key === 'v' || ev.key === 'V') setTool('select');
  if (ev.key === 'h' || ev.key === 'H') setTool('pan');
  if (ev.key === 'e' || ev.key === 'E') setTool('paint-era');
  if (ev.key === 'd' || ev.key === 'D') setTool('paint-dup');
  if (ev.key === 'c' || ev.key === 'C') setTool('paint-con');
  if (ev.key === 'x' || ev.key === 'X') setTool('erase');
  if (ev.key === 'r' || ev.key === 'R') setTool('rotate');
  if (ev.key === 't' || ev.key === 'T') setTool('touch');
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
    if (runMode === 'settle' || runMode === 'play' || runMode === 'listen') {
      await sim.stepAsync(dt, params, camera);
    }
    if (runMode === 'listen') {
      if (editor.gesture.kind === 'sound') holdSound(sim, audio, editor.gesture.stroke, now);
      if (listenKick && audio.isArmed) {
        kickListen();
        listenKick = false;
      }
      audio.frame(sim.graph, sim.agents, dt, camera);
    }
    sim.setViewExtent(camera.coverWidth() * 1.7, camera.coverHeight() * 1.7);
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
camera.zoom = 1;
camera.snap(sim.worldX, sim.worldY);
editor.lastPointer = camera.worldFromScreen(camera.viewW * 0.5, camera.viewH * 0.5);
loadHash();
void nativeSolver.init();
void farGpu.init();
void agentsGpu.init(gpuCanvas);
requestAnimationFrame(frame);
