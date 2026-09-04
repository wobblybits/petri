/*
 * Runs the WebGPU FAR solve against its CPU twin and reports where they
 * disagree. Not a vitest file on purpose: there is no WebGPU under Node,
 * so `farGpu.ready` is false there and every comparison would silently be
 * the twin against itself. This is the check `Sim.gpuFirst` says is
 * missing, and it has to run in a browser.
 *
 *   npm run dev, then open /gpu-check.html
 *
 * `runGpuCheck` does single frames over scenes chosen to isolate parts of
 * the kernel; `runDriftCheck` runs one scene for hundreds of frames, which
 * is the question that actually gates `gpuFirst` — not whether the two
 * agree exactly (they cannot: far-kernel.ts solves spans sequentially and
 * the shader solves them in parallel, Gauss-Seidel against Jacobi) but
 * whether the difference stays bounded or compounds.
 *
 * Measured 2026-09-04 on this machine's GPU: isolated pairs agree to the
 * bit, a 200-body lattice with mild overlap and 15%-stretched wires holds
 * both paths to the same peak speed within 0.1% over 300 frames with no
 * non-finite values, and the position difference flattens around 25px
 * (about one lattice cell) rather than growing. Note what that does *not*
 * cover: this drives `farGpu.step` directly, so it exercises the kernel
 * and not `packFar` — which is where the bug that actually sent wires to
 * infinite length lived. Turning `gpuFirst` on still wants an end-to-end
 * run in a browser, and on more than one vendor's GPU.
 */
import { farGpu } from './far-gpu.ts';
import {
  FAR,
  FAR_STRIDE,
  FAR_SUBSTEPS,
  FAR_WIRE_STRIDE,
  packFarWire,
  stepFarKernel,
} from './far-kernel.ts';

const FIELD_NAMES = Object.keys(FAR) as (keyof typeof FAR)[];

function particle(
  data: Float32Array,
  i: number,
  x: number,
  y: number,
  radius: number,
  mass = 1,
  heading = 0,
): void {
  const o = i * FAR_STRIDE;
  data[o + FAR.x] = x;
  data[o + FAR.y] = y;
  data[o + FAR.vx] = 0;
  data[o + FAR.vy] = 0;
  data[o + FAR.heading] = heading;
  data[o + FAR.omega] = 0;
  data[o + FAR.invMass] = 1 / mass;
  data[o + FAR.radius] = radius;
  data[o + FAR.locked] = 0;
}

function wiresOf(rows: number[][]): Float32Array {
  const wires = new Float32Array(Math.max(1, rows.length) * FAR_WIRE_STRIDE);
  for (let k = 0; k < rows.length; k++) {
    const [a, b, rest, oax = 0, oay = 0, obx = 0, oby = 0] = rows[k];
    packFarWire(wires, k, a, b, rest, oax, oay, obx, oby);
  }
  return wires;
}

interface Scene {
  name: string;
  data: Float32Array;
  n: number;
  wires: Float32Array;
  nWires: number;
}

/** Deterministic: a seeded LCG, so a rerun compares the same scene. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function scenes(): Scene[] {
  const out: Scene[] = [];

  {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 2, 0, 10);
    out.push({ name: 'two overlapping discs, no wires', data, n: 2, wires: wiresOf([]), nWires: 0 });
  }

  {
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 15);
    particle(data, 1, 20, 0, 15);
    out.push({ name: 'wired pair, rest inside the discs', data, n: 2, wires: wiresOf([[0, 1, 20]]), nWires: 1 });
  }

  {
    // A wire stretched well past rest — the span constraint under load is
    // the suspect for "wires went to infinite length".
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 400, 0, 10);
    out.push({ name: 'wired pair stretched 20x rest', data, n: 2, wires: wiresOf([[0, 1, 20]]), nWires: 1 });
  }

  {
    // Offset attachments, so the span applies torque as well as pull.
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 12, 1, 0.7);
    particle(data, 1, 60, 10, 12, 1, -1.2);
    out.push({
      name: 'wired pair, offset attachments (torque)',
      data,
      n: 2,
      wires: wiresOf([[0, 1, 40, 10, 4, -10, -4]]),
      nWires: 1,
    });
  }

  {
    /*
     * Three wires between the *same* pair, which the graph allows: a Con and
     * a Dup have three ports each, so two bodies can be joined more than
     * once, and packFar does not collapse them.
     *
     * This is the case a mesh does not cover. Solved in sequence the second
     * wire sees the first one's correction and only has to nudge; summed in
     * parallel all three compute the full correction against the same start
     * pose and the pair gets pulled three times as far as any one of them
     * asked for. Found in a live pond — body 168 wired to body 756 three
     * times, rests 46.3 / 47.8 / 47.0 — where it took the GPU path from
     * 2,720 to 6.7 million peak speed in a single frame.
     */
    const data = new Float32Array(2 * FAR_STRIDE);
    particle(data, 0, 0, 0, 10);
    particle(data, 1, 100, 0, 10);
    out.push({
      name: 'THREE wires between the same pair',
      data,
      n: 2,
      wires: wiresOf([
        [0, 1, 46.335],
        [0, 1, 47.769],
        [0, 1, 46.975],
      ]),
      nWires: 3,
    });
  }

  {
    // A chain: several wires sharing bodies, which is where a Jacobi-style
    // GPU pass and a sequential CPU pass are most likely to part ways.
    const n = 12;
    const data = new Float32Array(n * FAR_STRIDE);
    const rows: number[][] = [];
    for (let i = 0; i < n; i++) particle(data, i, i * 26, (i % 3) * 4, 11);
    for (let i = 0; i + 1 < n; i++) rows.push([i, i + 1, 24]);
    out.push({ name: '12-body chain, 11 shared wires', data, n, wires: wiresOf(rows), nWires: rows.length });
  }

  {
    // A crowd: many overlapping discs, the contact path under real load.
    const rng = makeRng(20260904);
    const n = 256;
    const data = new Float32Array(n * FAR_STRIDE);
    for (let i = 0; i < n; i++) {
      particle(data, i, rng() * 400, rng() * 400, 10 + rng() * 6, 0.5 + rng(), rng() * 6.28);
    }
    out.push({ name: '256-body crowd, no wires', data, n, wires: wiresOf([]), nWires: 0 });
  }

  {
    // Crowd plus a wire mesh — closest to the real FAR-tier pond.
    const rng = makeRng(77771);
    const n = 200;
    const data = new Float32Array(n * FAR_STRIDE);
    for (let i = 0; i < n; i++) {
      particle(data, i, rng() * 300, rng() * 300, 9 + rng() * 5, 0.5 + rng(), rng() * 6.28);
    }
    const rows: number[][] = [];
    for (let i = 0; i + 1 < n; i += 2) rows.push([i, i + 1, 30 + rng() * 20]);
    out.push({ name: '200-body crowd + 100 wires', data, n, wires: wiresOf(rows), nWires: rows.length });
  }

  {
    // A pile, which is what the broadphase cannot escape by any grid sizing:
    // cells are already as small as the widest contact gap allows, so density
    // this high overruns CELL_CAP and the cell stops recording bodies. Missed
    // contacts read as a pile that separates too slowly rather than as a blow
    // up, which is exactly the kind of wrong that goes unnoticed -- so measure
    // it against the twin, which tests every pair and cannot miss any.
    const rng = makeRng(31337);
    const n = 400;
    const data = new Float32Array(n * FAR_STRIDE);
    for (let i = 0; i < n; i++) {
      particle(data, i, 200 + rng() * 60, 200 + rng() * 60, 10.34, 1, rng() * 6.28);
    }
    out.push({ name: '400-body pile in one cell-width', data, n, wires: wiresOf([]), nWires: 0 });
  }

  return out;
}

export interface FieldDiff {
  field: string;
  maxAbs: number;
  atIndex: number;
  cpu: number;
  gpu: number;
}

export interface SceneResult {
  name: string;
  n: number;
  nWires: number;
  gpuActuallyRan: boolean;
  worst: FieldDiff;
  perField: FieldDiff[];
  cpuNonFinite: number;
  gpuNonFinite: number;
}

function compare(name: string, sc: Scene, cpu: Float32Array, gpu: Float32Array, ran: boolean): SceneResult {
  const perField: FieldDiff[] = [];
  let cpuNonFinite = 0;
  let gpuNonFinite = 0;
  for (const field of FIELD_NAMES) {
    const off = FAR[field];
    let maxAbs = 0;
    let atIndex = -1;
    let cv = 0;
    let gv = 0;
    for (let i = 0; i < sc.n; i++) {
      const a = cpu[i * FAR_STRIDE + off];
      const b = gpu[i * FAR_STRIDE + off];
      if (!Number.isFinite(a)) cpuNonFinite++;
      if (!Number.isFinite(b)) gpuNonFinite++;
      const d = Math.abs(a - b);
      if (d > maxAbs || (!Number.isFinite(d) && atIndex < 0)) {
        maxAbs = d;
        atIndex = i;
        cv = a;
        gv = b;
      }
    }
    perField.push({ field, maxAbs, atIndex, cpu: cv, gpu: gv });
  }
  const worst = perField.reduce((m, f) => (f.maxAbs > m.maxAbs || !Number.isFinite(f.maxAbs) ? f : m), perField[0]);
  return { name, n: sc.n, nWires: sc.nWires, gpuActuallyRan: ran, worst, perField, cpuNonFinite, gpuNonFinite };
}

export interface DriftSample {
  frame: number;
  maxPosDiff: number;
  cpuMaxSpeed: number;
  gpuMaxSpeed: number;
  cpuNonFinite: number;
  gpuNonFinite: number;
}

/**
 * The question `Sim.gpuFirst` actually hangs on.
 *
 * A per-frame difference between Jacobi and Gauss-Seidel is expected and
 * documented (see far-span-jacobi.test.ts). What matters is whether it
 * settles or compounds: two solvers that both converge to rest can differ
 * on the way there and still agree about where the pond ends up, but a
 * divergence that grows without bound is a pond that flies apart on one
 * path and not the other. Node cannot answer this — there is no WebGPU —
 * so it is asked here, over enough frames for a trend to show.
 */
export async function runDriftCheck(frames = 300, sampleEvery = 25): Promise<DriftSample[]> {
  /*
   * A settled pond, not a detonation. Bodies sit on a lattice wider than
   * their own diameter so contacts start quiet, and each wired pair starts
   * at exactly its rest length. A scene that is violent on both paths
   * answers nothing: two trajectories through a chaotic contact pile
   * separate whatever the solvers do, so the divergence measured there is
   * the chaos, not the shader.
   */
  /*
   * Perturbed, but convergent — the regime the FAR tier actually runs in.
   *
   * Two earlier scenes each answered nothing. A random pile at ~85% areal
   * packing is chaotic on both paths, so what separates the trajectories
   * there is the chaos, not the shader. A lattice with wires at exact rest
   * is inert: nothing moves, both solvers agree on doing nothing. This sits
   * between them — a lattice with mild overlap and wires ~15% stretched, so
   * contacts and spans both do real work while the system relaxes toward
   * equilibrium instead of flying apart.
   */
  const rng = makeRng(4242);
  const n = 200;
  const cols = 20;
  const pitch = 20; // against radii of 9-14, a few px of overlap: gentle contact
  const cpu = new Float32Array(n * FAR_STRIDE);
  for (let i = 0; i < n; i++) {
    const cx = (i % cols) * pitch;
    const cy = Math.floor(i / cols) * pitch;
    particle(cpu, i, cx, cy, 9 + rng() * 5, 0.5 + rng(), rng() * 6.28);
  }
  const rows: number[][] = [];
  for (let i = 0; i + 1 < n; i += 2) {
    const oa = i * FAR_STRIDE;
    const ob = (i + 1) * FAR_STRIDE;
    const gap = Math.hypot(cpu[oa + FAR.x] - cpu[ob + FAR.x], cpu[oa + FAR.y] - cpu[ob + FAR.y]);
    rows.push([i, i + 1, gap * 0.85]); // stretched, so the span pulls
  }
  const wires = wiresOf(rows);
  const nWires = rows.length;
  const gpu = new Float32Array(cpu);

  const out: DriftSample[] = [];
  for (let f = 1; f <= frames; f++) {
    stepFarKernel(cpu, n, wires, nWires, 1 / 60, FAR_SUBSTEPS);
    await farGpu.step(gpu, n, wires, nWires, 1 / 60, FAR_SUBSTEPS);
    if (f % sampleEvery !== 0 && f !== frames) continue;
    let maxPosDiff = 0;
    let cpuMaxSpeed = 0;
    let gpuMaxSpeed = 0;
    let cpuNonFinite = 0;
    let gpuNonFinite = 0;
    for (let i = 0; i < n; i++) {
      const o = i * FAR_STRIDE;
      const dx = cpu[o + FAR.x] - gpu[o + FAR.x];
      const dy = cpu[o + FAR.y] - gpu[o + FAR.y];
      const d = Math.hypot(dx, dy);
      if (d > maxPosDiff) maxPosDiff = d;
      const cs = Math.hypot(cpu[o + FAR.vx], cpu[o + FAR.vy]);
      const gs = Math.hypot(gpu[o + FAR.vx], gpu[o + FAR.vy]);
      if (cs > cpuMaxSpeed) cpuMaxSpeed = cs;
      if (gs > gpuMaxSpeed) gpuMaxSpeed = gs;
      for (const k of [FAR.x, FAR.y, FAR.vx, FAR.vy]) {
        if (!Number.isFinite(cpu[o + k])) cpuNonFinite++;
        if (!Number.isFinite(gpu[o + k])) gpuNonFinite++;
      }
    }
    out.push({ frame: f, maxPosDiff, cpuMaxSpeed, gpuMaxSpeed, cpuNonFinite, gpuNonFinite });
  }
  return out;
}

export async function runGpuCheck(): Promise<{
  gpuAvailable: boolean;
  initError: string;
  results: SceneResult[];
}> {
  let gpuAvailable = false;
  let initError = '';
  try {
    gpuAvailable = await farGpu.init();
    // A shader that fails to compile still yields a pipeline, so every scene
    // below would report `gpuActuallyRan` and then compare the untouched pack
    // against a solved one -- a page full of loud differences whose real cause
    // is one line of WGSL. Say so instead.
    if (!gpuAvailable && farGpu.initError) initError = farGpu.initError;
  } catch (err) {
    initError = String(err);
  }
  const results: SceneResult[] = [];
  if (!gpuAvailable) return { gpuAvailable, initError, results };

  for (const sc of scenes()) {
    const cpu = new Float32Array(sc.data);
    const gpu = new Float32Array(sc.data);
    stepFarKernel(cpu, sc.n, sc.wires, sc.nWires, 1 / 60, FAR_SUBSTEPS);
    // Returns false when it fell back to the CPU twin internally, which
    // would make the comparison compare the twin against itself.
    const ran = await farGpu.step(gpu, sc.n, sc.wires, sc.nWires, 1 / 60, FAR_SUBSTEPS);
    results.push(compare(sc.name, sc, cpu, gpu, ran));
  }
  return { gpuAvailable, initError, results };
}
