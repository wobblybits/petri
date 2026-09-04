import { nativeSolver } from './native/solver.ts';

export const CH = {
  conP: 0,
  dupP: 1,
  eraP: 2,
  aux: 3,
} as const;

export const CHANNELS = 4;

/*
 * The world grid, shared by the scent field and the energy grid.
 *
 * Cell size is the parameter that matters, and it is not free. Steering
 * compares two sensors `sensorDist` apart — 22 world units — against a dead
 * zone of 5% of the signal, and that dead zone is doing real work: it rejects
 * the sampling asymmetry an agent reads off its own trail, which is
 * proportional to signal strength in the same way. So a real gradient is only
 * visible when the sensors straddle something close to a whole cell.
 *
 * At 20 units they straddle about 1.1, which is where the steering constants
 * were tuned — the old field sized itself to the camera and was roughly this
 * fine when zoomed in. Measured at 160 units the sensors straddled 0.14 of a
 * cell, a real gradient read 0.043 against a dead zone of 0.237, and nothing
 * turned at all. Shrinking the dead zone does not rescue it: then the
 * self-trail artifact steers instead, at every resolution tried.
 *
 * Extent is the other side of the same coin, and it is the cheap side: cost
 * scales with the number of cells, so a finer cell costs nothing and a wider
 * world costs everything. 256 squared at 20 units is 65,536 cells — about a
 * millisecond for two diffusion passes and a decay — over a world 5,120 units
 * across. At the ~52-unit spacing the pond actually settles to, that holds
 * something like 9,700 bodies packed solid.
 *
 * One grid, not two. An earlier plan had a fine GPU field with a coarse CPU
 * fallback, which would have meant agents steering differently depending on
 * whether WebGPU turned up — a second simulation rather than a second code
 * path. A GPU field is still worth having; it buys extent, not fidelity.
 */
export const FIELD_CELLS = 1024;
export const FIELD_CELL = 10;
export const FIELD_EXTENT = FIELD_CELLS * FIELD_CELL;
export const FIELD_HALF = FIELD_EXTENT / 2;

export class Fields {
  cols: number;
  rows: number;
  worldW: number;
  worldH: number;
  originX = 0;
  originY = 0;
  data: Float32Array;
  tmp: Float32Array;

  constructor(cells = FIELD_CELLS) {
    this.cols = cells;
    this.rows = cells;
    this.worldW = FIELD_EXTENT;
    this.worldH = FIELD_EXTENT;
    this.data = new Float32Array(this.cols * this.rows * CHANNELS);
    this.tmp = new Float32Array(this.data.length);
  }

  /**
   * Scratch for `shiftCells`, which used to allocate a fresh buffer per scroll.
   * Lazy: at 1024 squared this is 16 MB, and a Sim that never scrolls its field
   * — every Sim in the test suite — should not pay for it.
   */
  private scroll: Float32Array | null = null;

  /*
   * The rectangle of cells that have ever been deposited into, plus a margin.
   *
   * The grid is a million cells and a pond touches a small patch of it, so
   * diffusing and decaying the whole thing spends almost all of its time on
   * zeros — which stay zero however many times they are averaged with their
   * neighbours. Tracking where the scent actually is turns both passes into
   * work proportional to the pond rather than to the world.
   *
   * The margin is what diffusion is allowed to spread into before a deposit
   * extends the box again. Two passes at the default mix move a meaningful
   * amount about a cell, so sixteen is many frames of headroom; beyond it a
   * vanishing tail is clipped, and decay was taking it to nothing anyway.
   *
   * The box only ever grows, which keeps the invariant the ping-pong needs:
   * outside it both buffers are exactly zero, so a cell entering the box for
   * the first time is zero in whichever buffer is about to be read.
   */
  private static readonly MARGIN = 16;
  private loI = 0;
  private loJ = 0;
  private hiI = -1;
  private hiJ = -1;

  /** World units per cell. Fixed — this is the whole point of the rework. */
  get cellSize(): number {
    return FIELD_EXTENT / this.cols;
  }

  /**
   * Slide the field so it stays centred on (cx, cy) — normally `home`.
   *
   * Slide only. The old version also resized to track the camera, which is
   * what made a cell 340 world units when zoomed out and 40 when zoomed in,
   * and what made a zoom of more than 18% throw the entire field away. Extent
   * and cell size are now constants, so scrolling by whole cells is the only
   * thing left to do and nothing is ever discarded except what scrolls off.
   */
  cover(cx: number, cy: number): void {
    const cell = this.cellSize;
    const di = Math.round((cx - FIELD_HALF - this.originX) / cell);
    const dj = Math.round((cy - FIELD_HALF - this.originY) / cell);
    if (di === 0 && dj === 0) return;
    this.shiftCells(di, dj);
    this.originX += di * cell;
    this.originY += dj * cell;
    // The box indexes cells, and the cells just moved under it.
    if (this.hiI >= this.loI) {
      this.loI = Math.max(0, this.loI - di);
      this.hiI = Math.min(this.cols - 1, this.hiI - di);
      this.loJ = Math.max(0, this.loJ - dj);
      this.hiJ = Math.min(this.rows - 1, this.hiJ - dj);
      if (this.hiI < this.loI || this.hiJ < this.loJ) this.hiI = this.loI - 1;
    }
  }

  clear(): void {
    this.data.fill(0);
    this.tmp.fill(0);
    this.hiI = -1;
    this.hiJ = -1;
    this.loI = 0;
    this.loJ = 0;
  }

  private shiftCells(di: number, dj: number): void {
    const { cols, rows, data } = this;
    let next = this.scroll;
    if (!next || next.length !== data.length) {
      next = new Float32Array(data.length);
      this.scroll = next;
    } else {
      next.fill(0);
    }
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

  /**
   * Reference cell size the scent response curves were tuned against.
   *
   * `slow_factor` and `turn_boost` in the solver, and the dead zone in the
   * steering comparison, all read raw cell values. This is the cell size they
   * are calibrated for, so at the grid's own resolution the scale is 1 and a
   * deposit lands exactly as it always did.
   *
   * A cell holds whatever was deposited into it, so it is a total and not a
   * density: widen the cell and the same pond's bodies all deposit into fewer,
   * bigger cells, and every reading scales with the cell's area. Going from
   * 17-unit cells to 160-unit ones multiplied the trail an agent reads by
   * about ninety, which drove `slow_factor` to nearly zero — agents stopped
   * dead — and inflated the dead zone until the difference between two sensors
   * could never clear it. Normalising by area makes a reading mean the same
   * thing at any resolution, which is the whole point of a fixed grid.
   */
  static readonly REF_CELL = FIELD_CELL;

  /** Area scale that turns a deposit into a density at this resolution. */
  get depositScale(): number {
    const c = Fields.REF_CELL / this.cellSize;
    return c * c;
  }

  deposit(ch: number, x: number, y: number, amount: number): void {
    amount *= this.depositScale;
    const { gx, gy } = this.toGrid(x, y);
    this.touch(Math.floor(gx), Math.floor(gy));
    const i0 = Math.floor(gx);
    const j0 = Math.floor(gy);
    const tx = gx - i0;
    const ty = gy - j0;
    this.add(i0, j0, ch, amount * (1 - tx) * (1 - ty));
    this.add(i0 + 1, j0, ch, amount * tx * (1 - ty));
    this.add(i0, j0 + 1, ch, amount * (1 - tx) * ty);
    this.add(i0 + 1, j0 + 1, ch, amount * tx * ty);
  }

  /** The live box, as inclusive cell indices. Empty when hi < lo. */
  get boxLoI(): number { return this.loI; }
  get boxHiI(): number { return this.hiI; }
  get boxLoJ(): number { return this.loJ; }
  get boxHiJ(): number { return this.hiJ; }

  /** Extend the live box to cover a world position — for the native path,
   *  which deposits inside the solver and so never calls `deposit` here. */
  touchWorld(x: number, y: number): void {
    const { gx, gy } = this.toGrid(x, y);
    this.touch(Math.floor(gx), Math.floor(gy));
  }

  /** Extend the live box to cover this cell and its margin. */
  private touch(i: number, j: number): void {
    const m = Fields.MARGIN;
    const lo = (v: number, n: number) => (v < 0 ? 0 : v > n - 1 ? n - 1 : v);
    const a = lo(i - m, this.cols);
    const b = lo(i + m, this.cols);
    const c = lo(j - m, this.rows);
    const d = lo(j + m, this.rows);
    if (this.hiI < this.loI) {
      this.loI = a;
      this.hiI = b;
      this.loJ = c;
      this.hiJ = d;
      return;
    }
    if (a < this.loI) this.loI = a;
    if (b > this.hiI) this.hiI = b;
    if (c < this.loJ) this.loJ = c;
    if (d > this.hiJ) this.hiJ = d;
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
   * One diffusion pass.
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
    const { cols, rows } = this;
    const src = this.data;
    const dst = this.tmp;
    const rowStride = cols * CHANNELS;

    if (this.hiI < this.loI) return;
    /*
     * Widen by a cell before diffusing, because a diffusion pass moves scent
     * exactly one cell and the box has to be somewhere for it to move into.
     *
     * Without this the box was a hard ceiling on how far scent could ever
     * spread — sixteen cells from the nearest deposit, whatever the diffusion
     * rate — so turning diffusion up flattened the peak without extending the
     * reach, which looks exactly like turning it down. The box is meant to skip
     * cells that are zero and will stay zero, not to clip the physics.
     *
     * It grows to the whole grid if scent genuinely reaches the whole grid,
     * which is the correct cost of that happening. Decay is what keeps it
     * bounded in practice.
     */
    if (this.loI > 0) this.loI--;
    if (this.loJ > 0) this.loJ--;
    if (this.hiI < this.cols - 1) this.hiI++;
    if (this.hiJ < this.rows - 1) this.hiJ++;
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const hasUp = j > 0;
      const hasDown = j < rows - 1;
      let base = (j * cols + this.loI) * CHANNELS;
      for (let i = this.loI; i <= this.hiI; i++, base += CHANNELS) {
        // -1 means "reflect off the edge": fall back to this cell's own value,
        // so the boundary neither absorbs scent nor invents it.
        const left = i > 0 ? base - CHANNELS : -1;
        const right = i < cols - 1 ? base + CHANNELS : -1;
        const up = hasUp ? base - rowStride : -1;
        const down = hasDown ? base + rowStride : -1;
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
    if (this.hiI < this.loI) return;
    const d = this.data;
    const { cols } = this;
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const row = j * cols;
      for (let i = this.loI; i <= this.hiI; i++) {
        const base = (row + i) * CHANNELS;
        d[base] *= k;
        d[base + 1] *= k;
        d[base + 2] *= k;
        d[base + 3] *= k;
      }
    }
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
        const posCon = Math.max(0, data[base + CH.conP]);
        const posDup = Math.max(0, data[base + CH.dupP]);
        const posEra = Math.max(0, data[base + CH.eraP]);
        const posAux = Math.max(0, data[base + CH.aux]);
        const neg =
          Math.max(0, -data[base + CH.conP]) +
          Math.max(0, -data[base + CH.dupP]) +
          Math.max(0, -data[base + CH.eraP]) +
          Math.max(0, -data[base + CH.aux]);
        const con = posCon / peak;
        const dup = posDup / peak;
        const era = posEra / peak;
        const aux = posAux / peak;
        const p = (j * cols + i) * 4;
        pix[p] = Math.min(255, (con * 210 + aux * 70 + (neg / peak) * 140) | 0);
        pix[p + 1] = Math.min(255, (aux * 90 + era * 40) | 0);
        pix[p + 2] = Math.min(255, (dup * 200 + era * 160 + aux * 40) | 0);
        pix[p + 3] = Math.min(180, ((con + dup + era + aux + neg / peak) * 140) | 0);
      }
    }
  }
}
