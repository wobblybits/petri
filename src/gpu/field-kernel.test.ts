import { describe, expect, it } from 'vitest';
import { CH, CHANNELS, Fields } from '../fields.ts';

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
});
