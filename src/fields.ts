import { nativeSolver } from './native/solver.ts';

export const CH = {
  conP: 0,
  dupP: 1,
  eraP: 2,
  aux: 3,
} as const;

export const CHANNELS = 4;

export class Fields {
  cols: number;
  rows: number;
  worldW: number;
  worldH: number;
  originX = 0;
  originY = 0;
  data: Float32Array;
  tmp: Float32Array;
  /** Cells occupied by wires; scent will not diffuse through them. */
  walls: Uint8Array;

  constructor(worldW: number, worldH: number) {
    this.worldW = Math.max(1, worldW);
    this.worldH = Math.max(1, worldH);
    this.cols = 160;
    this.rows = Math.max(8, Math.round((160 * this.worldH) / this.worldW));
    this.data = new Float32Array(this.cols * this.rows * CHANNELS);
    this.tmp = new Float32Array(this.data.length);
    this.walls = new Uint8Array(this.cols * this.rows);
  }

  /** Slide / resize the scent window so it stays centered on (cx, cy). */
  cover(cx: number, cy: number, coverW: number, coverH: number): void {
    const newW = Math.max(32, coverW);
    const newH = Math.max(32, coverH);
    const newRows = Math.max(8, Math.round((this.cols * newH) / newW));
    const newOx = cx - newW * 0.5;
    const newOy = cy - newH * 0.5;

    const resized =
      Math.abs(newW - this.worldW) / this.worldW > 0.18 ||
      Math.abs(newH - this.worldH) / this.worldH > 0.18 ||
      newRows !== this.rows;

    if (resized) {
      this.worldW = newW;
      this.worldH = newH;
      this.originX = newOx;
      this.originY = newOy;
      if (newRows !== this.rows) {
        this.rows = newRows;
        this.data = new Float32Array(this.cols * this.rows * CHANNELS);
        this.tmp = new Float32Array(this.data.length);
        this.walls = new Uint8Array(this.cols * this.rows);
      } else {
        this.data.fill(0);
        this.tmp.fill(0);
        this.walls.fill(0);
      }
      return;
    }

    const cellW = this.worldW / this.cols;
    const cellH = this.worldH / this.rows;
    const di = Math.round((newOx - this.originX) / cellW);
    const dj = Math.round((newOy - this.originY) / cellH);
    if (di !== 0 || dj !== 0) {
      this.shiftCells(di, dj);
      this.originX += di * cellW;
      this.originY += dj * cellH;
    }
  }

  clear(): void {
    this.data.fill(0);
    this.walls.fill(0);
  }

  clearWalls(): void {
    this.walls.fill(0);
  }

  private markCell(i: number, j: number): void {
    if (i < 0 || j < 0 || i >= this.cols || j >= this.rows) return;
    this.walls[j * this.cols + i] = 1;
  }

  /** Stamp a world-space segment onto the scent wall mask. */
  markSegment(x0: number, y0: number, x1: number, y1: number): void {
    const cellW = this.worldW / this.cols;
    const cellH = this.worldH / this.rows;
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = Math.hypot(dx, dy);
    const step = Math.max(0.35 * Math.min(cellW, cellH), 0.5);
    const n = Math.max(1, Math.ceil(len / step));
    for (let k = 0; k <= n; k++) {
      const t = k / n;
      const { gx, gy } = this.toGrid(x0 + dx * t, y0 + dy * t);
      const i = Math.floor(gx);
      const j = Math.floor(gy);
      this.markCell(i, j);
      this.markCell(i - 1, j);
      this.markCell(i + 1, j);
      this.markCell(i, j - 1);
      this.markCell(i, j + 1);
    }
  }

  private shiftCells(di: number, dj: number): void {
    const { cols, rows, data } = this;
    const next = new Float32Array(data.length);
    for (let j = 0; j < rows; j++) {
      const sj = j + dj;
      if (sj < 0 || sj >= rows) continue;
      for (let i = 0; i < cols; i++) {
        const si = i + di;
        if (si < 0 || si >= cols) continue;
        const src = (sj * cols + si) * CHANNELS;
        const dst = (j * cols + i) * CHANNELS;
        for (let ch = 0; ch < CHANNELS; ch++) next[dst + ch] = data[src + ch];
      }
    }
    data.set(next);
  }

  private cell(i: number, j: number, ch: number): number {
    if (i < 0 || j < 0 || i >= this.cols || j >= this.rows) return -1;
    return (j * this.cols + i) * CHANNELS + ch;
  }

  private toGrid(x: number, y: number): { gx: number; gy: number } {
    return {
      gx: ((x - this.originX) / this.worldW) * this.cols,
      gy: ((y - this.originY) / this.worldH) * this.rows,
    };
  }

  deposit(ch: number, x: number, y: number, amount: number): void {
    const { gx, gy } = this.toGrid(x, y);
    const i0 = Math.floor(gx);
    const j0 = Math.floor(gy);
    const tx = gx - i0;
    const ty = gy - j0;
    this.add(i0, j0, ch, amount * (1 - tx) * (1 - ty));
    this.add(i0 + 1, j0, ch, amount * tx * (1 - ty));
    this.add(i0, j0 + 1, ch, amount * (1 - tx) * ty);
    this.add(i0 + 1, j0 + 1, ch, amount * tx * ty);
  }

  private add(i: number, j: number, ch: number, amount: number): void {
    const idx = this.cell(i, j, ch);
    if (idx >= 0) this.data[idx] += amount;
  }

  private at(i: number, j: number, ch: number): number {
    const idx = this.cell(i, j, ch);
    return idx < 0 ? 0 : this.data[idx];
  }

  /**
   * One diffusion pass. Walls hold their value and are treated as reflecting
   * for their neighbours, so scent cannot cross a wire.
   *
   * Written the long way on purpose: the readable version called a closure four
   * times per channel per cell, which on a 160x107x4 grid is half a million
   * calls with bounds checks, and it dominated the frame at low agent counts.
   * Neighbour offsets are resolved once per cell, and the buffers ping-pong
   * rather than copying a quarter-megabyte each pass.
   */
  diffuse(mix: number): void {
    if (mix <= 0) return;
    if (nativeSolver.scentDiffuse(this, mix)) return;
    const m = mix;
    const keep = 1 - m;
    const { cols, rows, walls } = this;
    const src = this.data;
    const dst = this.tmp;
    const rowStride = cols * CHANNELS;

    for (let j = 0; j < rows; j++) {
      const hasUp = j > 0;
      const hasDown = j < rows - 1;
      let cell = j * cols;
      let base = cell * CHANNELS;
      for (let i = 0; i < cols; i++, cell++, base += CHANNELS) {
        if (walls[cell]) {
          dst[base] = src[base];
          dst[base + 1] = src[base + 1];
          dst[base + 2] = src[base + 2];
          dst[base + 3] = src[base + 3];
          continue;
        }
        // -1 means "reflect": fall back to this cell's own value.
        const left = i > 0 && !walls[cell - 1] ? base - CHANNELS : -1;
        const right = i < cols - 1 && !walls[cell + 1] ? base + CHANNELS : -1;
        const up = hasUp && !walls[cell - cols] ? base - rowStride : -1;
        const down = hasDown && !walls[cell + cols] ? base + rowStride : -1;
        for (let ch = 0; ch < CHANNELS; ch++) {
          const k = base + ch;
          const self = src[k];
          const a = left >= 0 ? src[left + ch] : self;
          const b = right >= 0 ? src[right + ch] : self;
          const c = up >= 0 ? src[up + ch] : self;
          const e = down >= 0 ? src[down + ch] : self;
          dst[k] = keep * self + m * (a + b + c + e) * 0.25;
        }
      }
    }

    this.data = dst;
    this.tmp = src;
  }

  decay(rate: number): void {
    const k = Math.max(0, 1 - rate);
    if (nativeSolver.scentDecay(this, k)) return;
    const d = this.data;
    for (let i = 0; i < d.length; i++) d[i] *= k;
  }

  sample(ch: number, x: number, y: number): number {
    const { gx, gy } = this.toGrid(x, y);
    const i0 = Math.floor(gx);
    const j0 = Math.floor(gy);
    const tx = gx - i0;
    const ty = gy - j0;
    const a = this.at(i0, j0, ch);
    const b = this.at(i0 + 1, j0, ch);
    const c = this.at(i0, j0 + 1, ch);
    const d = this.at(i0 + 1, j0 + 1, ch);
    return a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty;
  }

  gradient(ch: number, x: number, y: number): { x: number; y: number } {
    const epsX = this.worldW / this.cols;
    const epsY = this.worldH / this.rows;
    const dx = this.sample(ch, x + epsX, y) - this.sample(ch, x - epsX, y);
    const dy = this.sample(ch, x, y + epsY) - this.sample(ch, x, y - epsY);
    return { x: dx / (2 * epsX), y: dy / (2 * epsY) };
  }

  peak(): number {
    let m = 0;
    const d = this.data;
    for (let i = 0; i < d.length; i++) if (d[i] > m) m = d[i];
    return m;
  }

  paintOverlay(image: ImageData): void {
    const { cols, rows, data } = this;
    const pix = image.data;
    const peak = Math.max(0.08, this.peak());
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const base = (j * cols + i) * CHANNELS;
        const con = data[base + CH.conP] / peak;
        const dup = data[base + CH.dupP] / peak;
        const era = data[base + CH.eraP] / peak;
        const aux = data[base + CH.aux] / peak;
        const p = (j * cols + i) * 4;
        pix[p] = Math.min(255, (con * 210 + aux * 70) | 0);
        pix[p + 1] = Math.min(255, (aux * 90 + era * 40) | 0);
        pix[p + 2] = Math.min(255, (dup * 200 + era * 160 + aux * 40) | 0);
        pix[p + 3] = Math.min(180, ((con + dup + era + aux) * 140) | 0);
      }
    }
  }
}
