import { describe, expect, it } from 'vitest';
import shader from './field.wgsl?raw';
import { CH, CHANNELS, Fields } from '../fields.ts';
import {
  EnergyGrid,
  HARVEST_GOT,
  HARVEST_KS,
  HARVEST_ROOM,
  HARVEST_STRIDE,
  HARVEST_TOTAL,
  HarvestPlan,
  runHarvestPlan,
  uptakeRate,
  type UptakeKinetics,
} from '../energy.ts';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { CHEM_SPECIES } from '../chem-layout.ts';

/**
 * The field shaders' arithmetic, checked against `Fields`.
 *
 * There is no WebGPU under Node, so field.wgsl cannot be executed here. What
 * can be executed is a line-for-line mirror of each entry point, and that is
 * what these are. It catches the class of bug that actually happens — a
 * boundary handled differently, an index off by a row, a normalisation applied
 * on one side and not the other — without pretending to have run the shader.
 *
 * The precedent is far.wgsl, which had never executed once in the whole life
 * of the project and turned out to be wrong in a way no test could see.
 */

const FIXED_SCALE = 1e4;

/**
 * `grow`, line for line. Logistic on one channel, optionally catalysed.
 */
function mirrorGrow(
  src: Float32Array,
  n: number,
  ch: number,
  r: number,
  cap: number,
  catCh: number,
  gamma: number,
): Float32Array {
  const out = Float32Array.from(src);
  if (r <= 0 || cap <= 0) return out;
  for (let i = 0; i < n; i++) {
    const k = i * CHANNELS + ch;
    const e = out[k];
    if (e <= 0 || e >= cap) continue;
    let rr = r;
    if (catCh >= 0 && catCh !== ch && gamma !== 0) {
      rr = r * (1 + gamma * out[i * CHANNELS + catCh]);
    }
    if (rr <= 0) continue;
    const next = e + rr * e * (1 - e / cap);
    out[k] = next > cap ? cap : next;
  }
  return out;
}

/** `react`, line for line. Gray-Scott between two channels. */
function mirrorReact(
  src: Float32Array,
  n: number,
  uc: number,
  vc: number,
  feed: number,
  kill: number,
  dt: number,
): Float32Array {
  const out = Float32Array.from(src);
  if (dt <= 0 || uc === vc) return out;
  if (feed <= 0 && kill <= 0) return out;
  const f = feed * dt;
  const kv = (feed + kill) * dt;
  for (let i = 0; i < n; i++) {
    const u = out[i * CHANNELS + uc];
    const v = out[i * CHANNELS + vc];
    const uvv = u * v * v * dt;
    const nu = u - uvv + f * (1 - u);
    const nv = v + uvv - kv * v;
    out[i * CHANNELS + uc] = nu > 0 ? nu : 0;
    out[i * CHANNELS + vc] = nv > 0 ? nv : 0;
  }
  return out;
}

function mirrorDiffuse(
  src: Float32Array,
  cols: number,
  rows: number,
  m: number | number[],
): Float32Array {
  const dst = new Float32Array(src.length);
  for (let idx = 0; idx < cols * rows; idx++) {
    const i = idx % cols;
    const j = (idx / cols) | 0;
    const at = idx * CHANNELS;
    for (let c = 0; c < CHANNELS; c++) {
      const self = src[at + c];
      const a = i > 0 ? src[(idx - 1) * CHANNELS + c] : self;
      const b = i + 1 < cols ? src[(idx + 1) * CHANNELS + c] : self;
      const u = j > 0 ? src[(idx - cols) * CHANNELS + c] : self;
      const d = j + 1 < rows ? src[(idx + cols) * CHANNELS + c] : self;
      const mc = Array.isArray(m) ? m[c] : m;
      dst[at + c] = (1 - mc) * self + mc * (a + b + u + d) * 0.25;
    }
  }
  return dst;
}

/** `scatter` then `applyAcc`, including the fixed-point round trip. */
function mirrorScatter(
  field: Float32Array,
  cols: number,
  rows: number,
  originX: number,
  originY: number,
  extent: number,
  deposits: { x: number; y: number; w: number[] }[],
): void {
  const acc = new Int32Array(cols * rows * CHANNELS);
  for (const d of deposits) {
    const gx = ((d.x - originX) / extent) * cols;
    const gy = ((d.y - originY) / extent) * cols;
    const i0 = Math.floor(gx);
    const j0 = Math.floor(gy);
    const tx = gx - i0;
    const ty = gy - j0;
    for (let dj = 0; dj < 2; dj++) {
      for (let di = 0; di < 2; di++) {
        const i = i0 + di;
        const j = j0 + dj;
        if (i < 0 || j < 0 || i >= cols || j >= rows) continue;
        const w = (di ? tx : 1 - tx) * (dj ? ty : 1 - ty);
        const base = (j * cols + i) * CHANNELS;
        for (let c = 0; c < CHANNELS; c++) {
          const v = d.w[c] * w;
          // Rounded, matching the shader: truncation biases every
          // contribution the same way and the error accumulates per cell.
          if (v !== 0) acc[base + c] += Math.round(v * FIXED_SCALE);
        }
      }
    }
  }
  for (let i = 0; i < cols * rows * CHANNELS; i++) field[i] += acc[i] / FIXED_SCALE;
}

function mirrorSample(
  field: Float32Array,
  cols: number,
  rows: number,
  originX: number,
  originY: number,
  extent: number,
  x: number,
  y: number,
): number[] {
  const gx = ((x - originX) / extent) * cols;
  const gy = ((y - originY) / extent) * cols;
  const i0 = Math.floor(gx);
  const j0 = Math.floor(gy);
  const tx = gx - i0;
  const ty = gy - j0;
  const out = [0, 0, 0, 0];
  for (let dj = 0; dj < 2; dj++) {
    for (let di = 0; di < 2; di++) {
      const i = i0 + di;
      const j = j0 + dj;
      if (i < 0 || j < 0 || i >= cols || j >= rows) continue;
      const w = (di ? tx : 1 - tx) * (dj ? ty : 1 - ty);
      const base = (j * cols + i) * CHANNELS;
      for (let c = 0; c < CHANNELS; c++) out[c] += field[base + c] * w;
    }
  }
  return out;
}

/**
 * The shader's `gather`, which packs two different consumers into one
 * readback: three taste-collapsed sensor readings for the solver, then the
 * raw four channels under the body for the genome's `Wx` sense columns.
 *
 * The second half is the part worth mirroring. It is what lets
 * `Sim.updateState` stop calling `Fields.sampleAll` — reading a CPU array the
 * GPU never writes to was the quietest of the four reasons `openFieldGpu`
 * refuses, and the only one that fails silently, on machines with a device,
 * and only once evolution has moved a sense gene off its seed.
 */
function mirrorGather(
  field: Float32Array,
  cols: number,
  rows: number,
  originX: number,
  originY: number,
  extent: number,
  p: { lx: number; ly: number; rx: number; ry: number; ox: number; oy: number; taste: number[] },
): number[] {
  const at = (x: number, y: number): number[] =>
    mirrorSample(field, cols, rows, originX, originY, extent, x, y);
  const dot = (a: number[], b: number[]): number =>
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const own = at(p.ox, p.oy);
  return [
    dot(p.taste, at(p.lx, p.ly)),
    dot(p.taste, at(p.rx, p.ry)),
    dot(p.taste, own),
    0,
    own[0],
    own[1],
    own[2],
    own[3],
  ];
}

/**
 * The shader's `harvest`, transcribed.
 *
 * This is the mirror that matters most in the file. Every other kernel is a
 * stencil or a scale — arithmetic that is obviously the same on both sides.
 * This one is a hand-port of a *sequential* CPU loop, with an early exit and
 * a nested scan whose order is the whole point, and there is no WebGPU under
 * Node to check it against. Drains the block in raster order, restarting the
 * scan per body, exactly as `EnergyGrid.take` does.
 */
function mirrorHarvest(
  field: Float32Array,
  cols: number,
  ch: number,
  blk: { fi: number; fj: number; wi: number; wj: number },
  rooms: number[],
  uptake: { cap: number; ks: number } = { cap: 0, ks: 0 },
): number[] {
  const FLOW_EPS = 1e-9;
  const got: number[] = [];
  // The shader's Monod pre-pass: the block's mean density, read before anybody
  // eats, decides the rate everybody in it draws at. `1e30` is the shader's
  // stand-in for unmetered; `min` against it returns the room unchanged.
  let rate = 1e30;
  if (uptake.cap > 0) {
    let sum = 0;
    for (let y = 0; y < blk.wj; y++) {
      for (let x = 0; x < blk.wi; x++) {
        sum += field[((blk.fj + y) * cols + (blk.fi + x)) * CHANNELS + ch];
      }
    }
    const cells = blk.wi * blk.wj;
    const density = cells > 0 ? sum / cells : 0;
    rate = density <= 0 ? 0 : (uptake.cap * density) / (uptake.ks + density);
  }
  for (let e = 0; e < rooms.length; e++) {
    let want = Math.min(rooms[e], rate);
    if (want <= FLOW_EPS) {
      got.push(0);
      continue;
    }
    let g = 0;
    for (let y = 0; y < blk.wj && want > FLOW_EPS; y++) {
      for (let x = 0; x < blk.wi && want > FLOW_EPS; x++) {
        const idx = ((blk.fj + y) * cols + (blk.fi + x)) * CHANNELS + ch;
        const have = field[idx];
        if (have <= 0) continue;
        const take = have < want ? have : want;
        field[idx] = have - take;
        g += take;
        want -= take;
      }
    }
    got.push(g);
    if (g <= 0) {
      while (got.length < rooms.length) got.push(0);
      break;
    }
  }
  return got;
}

/**
 * `scatter` in conserve mode, which is the energy add rather than the scent
 * one, plus the disk mask the plain mirror above does not model.
 *
 * Returns the total that actually landed, because that is the property under
 * test: a *quantity* has to survive the rim, where a point inside the disk can
 * straddle cells whose centres are outside it.
 */
function mirrorScatterConserve(
  f: Fields,
  ch: number,
  x: number,
  y: number,
  amount: number,
): void {
  const cols = f.cols;
  const gx = ((x - f.originX) / f.worldW) * cols;
  const gy = ((y - f.originY) / f.worldW) * cols;
  const i0 = Math.floor(gx);
  const j0 = Math.floor(gy);
  const tx = gx - i0;
  const ty = gy - j0;
  const legalW = (di: number, dj: number): number => {
    const i = i0 + di;
    const j = j0 + dj;
    if (i < 0 || j < 0 || i >= cols || j >= f.rows) return 0;
    if (f.cellOut(i, j)) return 0;
    return (di ? tx : 1 - tx) * (dj ? ty : 1 - ty);
  };
  let legal = 0;
  for (let dj = 0; dj < 2; dj++) for (let di = 0; di < 2; di++) legal += legalW(di, dj);
  if (legal <= 0) return;
  const norm = 1 / legal;
  for (let dj = 0; dj < 2; dj++) {
    for (let di = 0; di < 2; di++) {
      const w = legalW(di, dj) * norm;
      if (w === 0) continue;
      const k = ((j0 + dj) * cols + (i0 + di)) * CHANNELS + ch;
      // Through the same fixed point the shader's atomics force.
      f.data[k] += Math.round(amount * w * FIXED_SCALE) / FIXED_SCALE;
    }
  }
}

/** A small field with a few blobs in it, so the passes have work to do. */
/**
 * Open the live box over the whole grid.
 *
 * `Fields` skips cells outside its box; the shader has no box at all and walks
 * every cell. That is equivalent only where the box covers the grid — which it
 * does in production, within seconds of a pond starting, because the field
 * fills the dish. These comparisons force it so the two are looking at the
 * same cells.
 *
 * `react` is the one pass where the difference would be visible rather than
 * academic: its feed term is `f * (1 - u)`, which is non-zero at `u = 0`, so
 * it puts substrate into empty cells. Everything else multiplies or scales
 * what is already there and leaves a zero cell at zero.
 */
function openBox(f: Fields): Fields {
  const cell = f.cellSize;
  f.touchWorld(f.originX + cell * 0.5, f.originY + cell * 0.5);
  f.touchWorld(f.originX + (f.cols - 0.5) * cell, f.originY + (f.rows - 0.5) * cell);
  return f;
}

function seeded(): Fields {
  const f = new Fields(64);
  for (const [x, y, ch] of [
    [200, 200, 0],
    [400, 300, 1],
    [260, 420, 3],
  ] as number[][]) {
    for (let r = 0; r < 60; r += 10) {
      for (let a = 0; a < 6.283; a += 0.4) {
        // Asked for as a field density: at 64 cells over the world extent a
        // cell is huge and the normalisation is tiny, so a raw amount lands
        // near nothing and there is no signal to compare.
        f.deposit(ch, x + Math.cos(a) * r, y + Math.sin(a) * r, 5 / (1 + r / 20) / f.depositScale);
      }
    }
  }
  return f;
}

describe('field shader arithmetic', () => {
  it('diffuses the way Fields does, boundary included', () => {
    const f = seeded();
    const before = Float32Array.from(f.data);
    const mine = mirrorDiffuse(before, f.cols, f.rows, 0.28);
    f.diffuse(0.28);
    let worst = 0;
    let peak = 0;
    for (let i = 0; i < f.data.length; i++) {
      peak = Math.max(peak, Math.abs(f.data[i]));
      worst = Math.max(worst, Math.abs(f.data[i] - mine[i]));
    }
    expect(peak, 'nothing to compare').toBeGreaterThan(0.5);
    expect(worst / peak, `worst cell off by ${((worst / peak) * 100).toFixed(4)}%`)
      .toBeLessThan(1e-6);
  });

  it('decays the way Fields does', () => {
    const f = seeded();
    const mine = Float32Array.from(f.data);
    const keep = 1 - 0.018;
    for (let i = 0; i < mine.length; i++) mine[i] *= keep;
    f.decay(0.018);
    let worst = 0;
    for (let i = 0; i < f.data.length; i++) worst = Math.max(worst, Math.abs(f.data[i] - mine[i]));
    expect(worst).toBeLessThan(1e-6);
  });

  it('diffuses each channel at its own rate, the way Fields does', () => {
    // The shader takes `mix` as a vec4f and the host resolves slider times
    // per-channel rate into it. This is the same resolution on the CPU side,
    // and it is the one that matters: a channel diffusing at the wrong rate
    // looks like a plausible field rather than an error.
    const f = seeded();
    f.diffuseRate[CH.energy] = 0.25;
    f.diffuseRate[CH.aux] = 0;
    const mix = 0.28;
    const per = [0, 1, 2, 3].map((c) => Math.min(1, mix * f.diffuseRate[c]));
    const mine = mirrorDiffuse(Float32Array.from(f.data), f.cols, f.rows, per);
    f.diffuse(mix);
    let worst = 0;
    let peak = 0;
    for (let i = 0; i < f.data.length; i++) {
      peak = Math.max(peak, Math.abs(f.data[i]));
      worst = Math.max(worst, Math.abs(f.data[i] - mine[i]));
    }
    expect(peak).toBeGreaterThan(0.5);
    expect(worst / peak).toBeLessThan(1e-6);
  });

  it('decays each channel at its own rate', () => {
    const f = seeded();
    f.decayRate[CH.energy] = 0;
    const rate = 0.018;
    const mine = Float32Array.from(f.data);
    for (let i = 0; i < mine.length; i++) {
      const c = i % CHANNELS;
      const r = rate * f.decayRate[c];
      mine[i] *= r >= 1 ? 0 : 1 - r;
    }
    f.decay(rate);
    let worst = 0;
    for (let i = 0; i < f.data.length; i++) worst = Math.max(worst, Math.abs(f.data[i] - mine[i]));
    expect(worst).toBeLessThan(1e-6);
  });

  it('grows the way Fields does, catalyst included', () => {
    const f = openBox(seeded());
    const cap = 0.4;
    // Something below capacity everywhere, and a catalyst that varies.
    for (let i = CH.energy; i < f.data.length; i += CHANNELS) f.data[i] = cap * 0.3;
    const before = Float32Array.from(f.data);
    const mine = mirrorGrow(before, f.cols * f.rows, CH.energy, 0.05, cap, CH.conP, 3);
    f.grow(CH.energy, 0.05, cap, CH.conP, 3);
    let worst = 0;
    for (let i = 0; i < f.data.length; i++) worst = Math.max(worst, Math.abs(f.data[i] - mine[i]));
    expect(worst, 'grow diverged from its mirror').toBeLessThan(1e-6);
    // And it did something, or the comparison proves nothing.
    let moved = 0;
    for (let i = CH.energy; i < f.data.length; i += CHANNELS) {
      if (Math.abs(f.data[i] - before[i]) > 1e-9) moved++;
    }
    expect(moved).toBeGreaterThan(100);
  });

  it('reacts the way Fields does', () => {
    const f = openBox(seeded());
    const before = Float32Array.from(f.data);
    const mine = mirrorReact(before, f.cols * f.rows, CH.conP, CH.dupP, 0.037, 0.06, 0.5);
    f.react(CH.conP, CH.dupP, 0.037, 0.06, 0.5);
    let worst = 0;
    for (let i = 0; i < f.data.length; i++) worst = Math.max(worst, Math.abs(f.data[i] - mine[i]));
    expect(worst, 'react diverged from its mirror').toBeLessThan(1e-6);
  });

  it('scatters where Fields deposits, within fixed-point', () => {
    const ref = new Fields(64);
    const mineField = new Float32Array(ref.data.length);
    const pts: { x: number; y: number; w: number[] }[] = [];
    for (let k = 0; k < 200; k++) {
      const x = 100 + (k * 37) % 900;
      const y = 100 + (k * 53) % 900;
      const w = [0.4, 0.1, 0, 0.5];
      pts.push({ x, y, w });
      // The host applies the density normalisation before handing weights
      // over, so the reference gets the same already-scaled amounts.
      for (let c = 0; c < CHANNELS; c++) {
        if (w[c] !== 0) ref.deposit(c, x, y, w[c] / ref.depositScale);
      }
    }
    mirrorScatter(mineField, ref.cols, ref.rows, ref.originX, ref.originY, ref.worldW, pts);
    let worst = 0;
    let peak = 0;
    for (let i = 0; i < mineField.length; i++) {
      peak = Math.max(peak, Math.abs(ref.data[i]));
      worst = Math.max(worst, Math.abs(ref.data[i] - mineField[i]));
    }
    expect(peak, 'nothing landed').toBeGreaterThan(0.1);
    // Fixed point at 1e4 gives four decimals; the tolerance is the truncation.
    expect(worst, `worst cell off by ${worst}`).toBeLessThan(1e-3);
  });

  it('samples where Fields samples', () => {
    const f = seeded();
    let worst = 0;
    let peak = 0;
    for (let k = 0; k < 60; k++) {
      const x = 150 + (k * 41) % 500;
      const y = 150 + (k * 67) % 500;
      const mine = mirrorSample(f.data, f.cols, f.rows, f.originX, f.originY, f.worldW, x, y);
      for (let c = 0; c < CHANNELS; c++) {
        const ref = f.sample(c, x, y);
        peak = Math.max(peak, Math.abs(ref));
        worst = Math.max(worst, Math.abs(ref - mine[c]));
      }
    }
    expect(peak, 'sampled only empty space').toBeGreaterThan(0.5);
    expect(worst / peak, `worst sample off by ${((worst / peak) * 100).toFixed(4)}%`)
      .toBeLessThan(1e-6);
  });

  it('hands back the raw channels the genome reads, not just the taste scalars', () => {
    // The parity that lets `updateState` stop sampling on the CPU. A
    // taste-collapsed scalar cannot stand in for these: `Wx` multiplies each
    // channel by its own weight, so the four have to survive separately.
    const f = seeded();
    const taste = [0.7, -1.3, 0.2, 2.1];
    const ref = new Float64Array(4);
    let worstRaw = 0;
    let worstSteer = 0;
    let peak = 0;
    for (let k = 0; k < 40; k++) {
      const ox = 150 + ((k * 41) % 500);
      const oy = 150 + ((k * 67) % 500);
      const got = mirrorGather(f.data, f.cols, f.rows, f.originX, f.originY, f.worldW, {
        lx: ox - 18,
        ly: oy - 12,
        rx: ox + 18,
        ry: oy - 12,
        ox,
        oy,
        taste,
      });

      // Second half against the call it replaces, channel for channel.
      f.sampleAll(ox, oy, ref, 0);
      for (let c = 0; c < CHANNELS; c++) {
        peak = Math.max(peak, Math.abs(ref[c]));
        worstRaw = Math.max(worstRaw, Math.abs(ref[c] - got[4 + c]));
      }

      // And the first half still says what steering was already told, so the
      // widening did not disturb the consumer that was already there.
      let own = 0;
      for (let c = 0; c < CHANNELS; c++) own += ref[c] * taste[c];
      worstSteer = Math.max(worstSteer, Math.abs(own - got[2]));
    }
    expect(peak, 'sampled only empty space').toBeGreaterThan(0.5);
    expect(worstRaw / peak, `raw channels off by ${((worstRaw / peak) * 100).toFixed(4)}%`)
      .toBeLessThan(1e-6);
    expect(worstSteer / peak, 'the steering scalar moved').toBeLessThan(1e-6);
  });

  it('lands the whole of a quantity at the rim, the way Fields.addAt does', () => {
    // The difference between the two adds `Fields` has, and the one that only
    // shows at the wall. A scent deposit may lose the share that falls in a
    // cell the disk rejects; energy may not, and the leak is unbounded and
    // worse the closer to the edge a body dies.
    //
    // Note what the reference does *not* claim: a point whose four cells are
    // all outside the mask is dropped outright, by both, and deliberately —
    // it is outside in every sense. So the assertion is parity with `addAt`,
    // not conservation in the abstract.
    const ref = new Fields();
    const mine = new Fields();
    const naive = new Fields();
    const cx = ref.originX + ref.worldW * 0.5;
    const cy = ref.originY + ref.worldW * 0.5;
    const r = ref.worldW * 0.25;
    for (const f of [ref, mine, naive]) f.setWorldBound(cx, cy, r);

    const N = 400;
    const AMT = 0.75;
    for (let k = 0; k < N; k++) {
      const a = (k / N) * Math.PI * 2;
      // Just inside the wall, where the four straddled cells disagree about
      // whether they are in the disk.
      const px = cx + Math.cos(a) * (r - 8);
      const py = cy + Math.sin(a) * (r - 8);
      ref.addAt(CH.energy, px, py, AMT);
      mirrorScatterConserve(mine, CH.energy, px, py, AMT);
      // What the shader did before the conserve branch: drop the share that
      // falls outside instead of spreading it over what is left.
      naive.deposit(CH.energy, px, py, AMT / naive.depositScale);
    }

    const total = (f: Fields): number => {
      let t = 0;
      for (let i = CH.energy; i < f.data.length; i += CHANNELS) t += f.data[i];
      return t;
    };
    const refTotal = total(ref);
    let worst = 0;
    for (let i = CH.energy; i < ref.data.length; i += CHANNELS) {
      worst = Math.max(worst, Math.abs(ref.data[i] - mine.data[i]));
    }

    expect(refTotal, 'nothing landed, so there is nothing being compared')
      .toBeGreaterThan(N * AMT * 0.5);
    expect(total(mine), 'the shader would have banked a different amount')
      .toBeCloseTo(refTotal, 2);
    expect(worst, 'it landed in different cells').toBeLessThan(1e-3);
    // And the test can tell the difference: without the renormalisation this
    // rim loses energy, which is the whole reason the branch exists.
    expect(total(naive), 'the un-renormalised add leaked nothing, so this proves nothing')
      .toBeLessThan(refTotal - 1);
  });

  it('grazes a block exactly as EnergyGrid.take does, body after body', () => {
    // Two fields seeded alike, one drained through `take` and one through the
    // mirror, cell for cell afterwards. Anything the port got wrong about
    // order shows up as a different *floor*, not a different total: a flat
    // proportional share takes the same energy and leaves the block uniform,
    // which is the one outcome this ordering exists to prevent.
    const params = defaultParams();
    params.ambientEnergy = 1;
    // Full resolution, not the coarse field the arithmetic tests use: an
    // energy cell is `energyCell / FIELD_CELL` field cells across, so the two
    // lattices only agree when the field has its real cell size.
    const ref = new Fields();
    const mine = new Fields();
    const grid = new EnergyGrid(params.energyCell, params.ambientEnergy);
    grid.bind(ref);
    grid.configure(params.energyCell, params.ambientEnergy);
    // Seeded through `setCell`, which spreads a cell's worth evenly over its
    // block, rather than through `fillDisk` — the disk mask needs bounds this
    // test has no reason to pin, and an empty block would make the comparison
    // below pass by having nothing to compare.
    const { i, j, key } = grid.index(ref.originX + ref.worldW * 0.5, ref.originY + ref.worldW * 0.5);
    grid.setCell(i, j, 1);
    mine.data.set(ref.data);
    const rect = grid.blockRect(i, j)!;
    expect(rect.wi * rect.wj, 'the block should span more than one field cell')
      .toBeGreaterThan(1);

    // More appetite than the block holds, so the early exit is exercised too.
    const rooms = [0.02, 0.5, 0.003, 4, 0.1];
    expect(grid.getCell(i, j), 'the block should have something in it').toBeCloseTo(1, 6);
    const got = mirrorHarvest(mine.data, mine.cols, CH.energy, rect, rooms);
    const want = rooms.map((r) => grid.take(key, r));

    for (let e = 0; e < rooms.length; e++) {
      expect(got[e], `body ${e} took a different amount`).toBeCloseTo(want[e], 6);
    }
    let worst = 0;
    for (let y = 0; y < rect.wj; y++) {
      for (let x = 0; x < rect.wi; x++) {
        const k = ((rect.fj + y) * ref.cols + (rect.fi + x)) * CHANNELS + CH.energy;
        worst = Math.max(worst, Math.abs(ref.data[k] - mine.data[k]));
      }
    }
    expect(worst, 'the block was left in a different state').toBeLessThan(1e-7);
  });

  it('meters a block the way runHarvestPlan does, at the same rate', () => {
    /*
     * The Monod half of `harvest`, on both sides. `energy.ts` reads the
     * density off `EnergyGrid.density` and the shader sums the block itself;
     * this is what says the two produce the same rate, and therefore the same
     * draw, for the same block.
     *
     * Also the guard on the ordering claim: metered, everyone in the block
     * gets the *same* rate, so a body's share stops depending on its id — which
     * is the artifact `docs/energy-chemistry-plan.md` §4 exists to remove.
     */
    const params = defaultParams();
    params.ambientEnergy = 1;
    const ref = new Fields();
    const mine = new Fields();
    const grid = new EnergyGrid(params.energyCell, params.ambientEnergy);
    grid.bind(ref);
    grid.configure(params.energyCell, params.ambientEnergy);
    const { i, j, key } = grid.index(ref.originX + ref.worldW * 0.5, ref.originY + ref.worldW * 0.5);
    grid.setCell(i, j, 1);
    mine.data.set(ref.data);
    const rect = grid.blockRect(i, j)!;

    const uptake = { cap: 0.05, ks: 0.25 };
    const density = grid.density(key);
    expect(density, 'the block should have a standing stock').toBeGreaterThan(0);
    const rate = uptakeRate(density, uptake.cap, uptake.ks);
    expect(rate).toBeLessThan(uptake.cap);

    const rooms = [4, 4, 4];
    const got = mirrorHarvest(mine.data, mine.cols, CH.energy, rect, rooms, uptake);
    // The CPU side, through the grid, at the rate read before anybody ate.
    const want = rooms.map((r) => grid.take(key, Math.min(r, rate)));

    for (let e = 0; e < rooms.length; e++) {
      expect(got[e], `body ${e} took a different amount`).toBeCloseTo(want[e], 6);
      // Three bodies, one rate, identical appetites: identical draws. Without
      // metering the first would have taken the block and left the other two
      // nothing at all.
      expect(got[e]).toBeCloseTo(rate, 6);
    }
    let worst = 0;
    for (let y = 0; y < rect.wj; y++) {
      for (let x = 0; x < rect.wi; x++) {
        const k = ((rect.fj + y) * ref.cols + (rect.fi + x)) * CHANNELS + CH.energy;
        worst = Math.max(worst, Math.abs(ref.data[k] - mine.data[k]));
      }
    }
    expect(worst, 'the block was left in a different state').toBeLessThan(1e-7);
  });

  it('takes the uptake dials out of the slots the uniform used to pad', () => {
    /*
     * The host hand-packs `FieldParams` by index, so a field added on one side
     * and not the other reads as a plausible number rather than an error —
     * the same hazard `genome-kernel.test.ts` pins for its own uniform. These
     * two went into `pad3` and `pad4`, and the struct's size must not have
     * moved: a uniform buffer is a whole number of sixteen-byte blocks.
     */
    const body = /struct\s+FieldParams\s*\{([\s\S]*?)\n\}/.exec(shader);
    expect(body, 'no FieldParams in field.wgsl').not.toBeNull();
    const fields = [...body![1].matchAll(/^\s*([A-Za-z_]\w*)\s*:/gm)].map((m) => m[1]);
    expect(fields).not.toContain('pad3');
    expect(fields).not.toContain('pad4');
    // `uptakeCap` and `uptakeKs` took the two slots that used to pad; `hillN`
    // the third, and the fourth carried `coSubstrate` until catabolism moved
    // to the host, so it is a pad again.
    // needed a fourth block, which is why there are three pads after it again.
    expect(fields.slice(-6)).toEqual(['uptakeCap', 'uptakeKs', 'hillN', 'pad5', 'pad6', 'pad7']);
    // Three vec4f (mix, mix2, keep) count as four slots each; everything else
    // is a scalar, so index and slot are the same thing. `UNIFORM_BYTES` is
    // 176, which is eleven sixteen-byte blocks.
    expect(fields.length).toBe(176 / 4 - 3 * 3);
  });

  it('pins the harvest row layout to the shader\'s literals', () => {
    // The shader cannot import, so its `HARVEST_*` are literals, and a stride
    // that moves on one side and not the other reads every body's affinities
    // as someone else's room. The host's exports are the source of truth.
    const literal = (name: string): number => {
      const m = new RegExp(`const HARVEST_${name}: u32 = (\\d+)u;`).exec(shader);
      expect(m, `no HARVEST_${name} in field.wgsl`).not.toBeNull();
      return Number(m![1]);
    };
    expect(literal('ROOM')).toBe(HARVEST_ROOM);
    expect(literal('TOTAL')).toBe(HARVEST_TOTAL);
    expect(literal('KS')).toBe(HARVEST_KS);
    expect(literal('GOT')).toBe(HARVEST_GOT);
    expect(literal('STRIDE')).toBe(HARVEST_STRIDE);
  });

  it('draws a sampled mouthful the way runHarvestPlan does', () => {
    /*
     * The metered half of `harvest`, transcribed: one budget a body, shared
     * across four species by the block's composition, each species at most
     * its share and at most its Monod rate, `left` the gut room decremented
     * as it goes, index order, `FLOW_EPS` between species. Run against
     * `runHarvestPlan` on the same plan and the same field, and compared on
     * what each body got of each species — which is what lands in the gut on
     * the host and what `creditHarvest` reads back from `hFlow` on the device.
     */
    const params = defaultParams();
    params.soupCount = 6;
    params.spawnInterval = 0;
    params.energyRegrow = 0;
    params.decay = 0;
    params.diffuse = 0;
    params.uptakeVmax = 6;
    params.learnRate = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    sim.step(1 / 60, params);
    // A mixed stock under every body, different per species so a share and a
    // rate can disagree, and two bodies sharing one block where the preset
    // happened to put them.
    const per = [0.3, 0.05, 0.4, 0.15];
    const f = sim.fields;
    f.data.fill(0);
    for (const a of sim.agents.values()) {
      const { i, j } = sim.energy.index(a.x, a.y);
      const r = sim.energy.blockRect(i, j);
      if (!r) continue;
      for (let y = 0; y < r.wj; y++) {
        for (let x = 0; x < r.wi; x++) {
          const k = ((r.fj + y) * f.cols + (r.fi + x)) * CHANNELS;
          for (let c = 0; c < CHANNELS; c++) f.data[k + c] = per[c];
        }
      }
    }
    const kinetics: UptakeKinetics = { cap: 0.1, ks: params.uptakeKs, yDirect: 1, yEra: 1, hillN: 1, gutSize: 1 };
    const plan = new HarvestPlan();
    plan.build(sim.agents.values(), sim.agentStore, sim.energy, kinetics);
    expect(plan.nEntries, 'nobody was hungry').toBeGreaterThan(0);

    // The shader, on a copy of the field.
    const src = Float32Array.from(f.data);
    const FLOW_EPS = 1e-9;
    const got = new Float64Array(plan.nEntries * CHANNELS);
    for (let b = 0; b < plan.nBlocks; b++) {
      const bo = b * 6;
      const fi = plan.blocks[bo];
      const fj = plan.blocks[bo + 1];
      const wi = plan.blocks[bo + 2];
      const wj = plan.blocks[bo + 3];
      const first = plan.blocks[bo + 4];
      const count = plan.blocks[bo + 5];
      const density = [0, 0, 0, 0];
      const cells = wi * wj;
      if (cells > 0) {
        for (let y = 0; y < wj; y++) {
          for (let x = 0; x < wi; x++) {
            const k = ((fj + y) * f.cols + (fi + x)) * CHANNELS;
            for (let c = 0; c < CHANNELS; c++) density[c] += src[k + c];
          }
        }
        for (let c = 0; c < CHANNELS; c++) density[c] /= cells;
      }
      let stock = 0;
      for (let c = 0; c < CHANNELS; c++) stock += Math.max(density[c], 0);
      for (let e = 0; e < count; e++) {
        const ro = (first + e) * HARVEST_STRIDE;
        let left = plan.rooms[ro + HARVEST_ROOM];
        const total = plan.rooms[ro + HARVEST_TOTAL];
        const ks = [0, 1, 2, 3].map((c) => plan.rooms[ro + HARVEST_KS + c]);
        for (let c = 0; c < CHANNELS; c++) {
          let g = 0;
          if (left > FLOW_EPS && total > 0 && density[c] > 0 && stock > 0) {
            const rate = (total * density[c]) / (ks[c] + density[c]);
            const share = (total * density[c]) / stock;
            let want = Math.min(rate, share);
            if (left < want) want = left;
            if (want > FLOW_EPS) {
              // `drain`: raster order over the block, channel `c`.
              for (let y = 0; y < wj && want > FLOW_EPS; y++) {
                for (let x = 0; x < wi && want > FLOW_EPS; x++) {
                  const k = ((fj + y) * f.cols + (fi + x)) * CHANNELS + c;
                  const have = src[k];
                  if (have <= 0) continue;
                  const take = have < want ? have : want;
                  src[k] = have - take;
                  g += take;
                  want -= take;
                }
              }
              left -= g;
            }
          }
          got[(first + e) * CHANNELS + c] = g;
        }
      }
    }

    // The host, on the real field, into the gut.
    const GUT = sim.agentStore.gut;
    const before = Float64Array.from(GUT);
    runHarvestPlan(plan, sim.agentStore, sim.energy, kinetics);
    let drew = 0;
    for (let e = 0; e < plan.nEntries; e++) {
      const slot = plan.slots[e];
      for (let c = 0; c < CHANNELS; c++) {
        const host = GUT[slot * CHEM_SPECIES + c] - before[slot * CHEM_SPECIES + c];
        expect(host, `entry ${e} species ${c}`).toBeCloseTo(got[e * CHANNELS + c], 7);
        drew += host;
      }
    }
    expect(drew, 'the mirror should have had something to compare').toBeGreaterThan(0);
    // And the field itself, cell for cell.
    let worst = 0;
    for (let k = 0; k < src.length; k++) worst = Math.max(worst, Math.abs(src[k] - f.data[k]));
    expect(worst, 'the block was left in a different state').toBeLessThan(1e-7);
  });

  it('leaves an uneven floor, which is the reason for the ordering', () => {
    // Guarding the mechanism rather than the arithmetic. `take`'s comment says
    // a flat share would keep the block uniform and leave diffusion nothing to
    // work against; this is what makes that claim testable, and what would
    // fail if the shader were ever "simplified" into a proportional share.
    const params = defaultParams();
    params.ambientEnergy = 1;
    const f = new Fields();
    const grid = new EnergyGrid(params.energyCell, params.ambientEnergy);
    grid.bind(f);
    grid.configure(params.energyCell, params.ambientEnergy);
    const { i, j } = grid.index(f.originX + f.worldW * 0.5, f.originY + f.worldW * 0.5);
    grid.setCell(i, j, 1);
    const rect = grid.blockRect(i, j)!;

    const before: number[] = [];
    for (let y = 0; y < rect.wj; y++) {
      for (let x = 0; x < rect.wi; x++) {
        before.push(f.data[((rect.fj + y) * f.cols + (rect.fi + x)) * CHANNELS + CH.energy]);
      }
    }
    // A nibble: less than the block holds, so some cells survive untouched.
    mirrorHarvest(f.data, f.cols, CH.energy, rect, [before[0] * 2.5]);
    const after: number[] = [];
    for (let y = 0; y < rect.wj; y++) {
      for (let x = 0; x < rect.wi; x++) {
        after.push(f.data[((rect.fj + y) * f.cols + (rect.fi + x)) * CHANNELS + CH.energy]);
      }
    }
    expect(Math.min(...after), 'nothing was emptied').toBe(0);
    expect(Math.max(...after), 'everything was emptied').toBeGreaterThan(0);
  });
});
