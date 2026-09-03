import { describe, expect, it } from 'vitest';
import { CHANNELS, Fields } from '../fields.ts';

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

function mirrorDiffuse(src: Float32Array, cols: number, rows: number, m: number): Float32Array {
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
      dst[at + c] = (1 - m) * self + m * (a + b + u + d) * 0.25;
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
