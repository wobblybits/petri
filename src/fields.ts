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

  constructor(worldW: number, worldH: number) {
    this.worldW = Math.max(1, worldW);
    this.worldH = Math.max(1, worldH);
    this.cols = 160;
    this.rows = Math.max(8, Math.round((160 * this.worldH) / this.worldW));
    this.data = new Float32Array(this.cols * this.rows * CHANNELS);
    this.tmp = new Float32Array(this.data.length);
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
      } else {
        this.data.fill(0);
        this.tmp.fill(0);
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

  diffuse(mix: number): void {
    const m = mix;
    const keep = 1 - m;
    const { cols, rows, data, tmp } = this;
    tmp.set(data);
    for (let j = 0; j < rows; j++) {
      const jm = j === 0 ? 0 : j - 1;
      const jp = j === rows - 1 ? rows - 1 : j + 1;
      for (let i = 0; i < cols; i++) {
        const im = i === 0 ? 0 : i - 1;
        const ip = i === cols - 1 ? cols - 1 : i + 1;
        const base = (j * cols + i) * CHANNELS;
        for (let ch = 0; ch < CHANNELS; ch++) {
          const avg =
            (tmp[(jm * cols + i) * CHANNELS + ch] +
              tmp[(jp * cols + i) * CHANNELS + ch] +
              tmp[(j * cols + im) * CHANNELS + ch] +
              tmp[(j * cols + ip) * CHANNELS + ch]) *
            0.25;
          data[base + ch] = keep * tmp[base + ch] + m * avg;
        }
      }
    }
  }

  decay(rate: number): void {
    const k = Math.max(0, 1 - rate);
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
