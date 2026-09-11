export const CH = {
  conP: 0,
  dupP: 1,
  /**
   * Energy. Not a signal — the stuff itself. Nothing deposits into it
   * through `emit`; it arrives by dying, by a rewrite's leftovers, by a full
   * producer spilling, by excretion, and by growing.
   */
  energy: 2,
  aux: 3,
} as const;

export const CHANNELS = 4;

/**
 * The channels that carry a signal — everything but the ground. The emit
 * budget is still four wide (see `effEmit`), but `peak` and anything asking
 * "is this pond audible" wants these three.
 */
export const VOICE = [CH.conP, CH.dupP, CH.aux] as const;

/**
 * The order a body digests what it swallowed: the three signalling species,
 * then the ground. The ground is the co-substrate the other three are
 * converted with (`Sim.runDigestion`), so it must be paired off last or a
 * body digests it out from under its own catabolism in the same frame.
 */
export const DIGEST_ORDER = [CH.conP, CH.dupP, CH.aux, CH.energy] as const;

/**
 * The channel whose presence accelerates the ground's regrowth — the
 * fertiliser signal. See `params.fertilise`. A voice channel rather than
 * `CH.aux`, so fertilising is something a lineage spends its voice budget
 * on rather than a property of having free ports.
 */
export const FERTILISE_CH: number = CH.conP;

/*
 * The world grid, shared by the scent field and the energy grid. Cell size
 * is not free: steering compares two sensors `sensorDist` apart against a
 * dead zone that rejects the self-trail artifact, so a real gradient is only
 * visible when the sensors straddle close to a whole cell. Cost scales with
 * the number of cells. One grid on both field paths, so agents steer the
 * same whether or not WebGPU turned up.
 */
export const FIELD_CELLS = 1024;
export const FIELD_CELL = 10;
export const FIELD_EXTENT = FIELD_CELLS * FIELD_CELL;
export const FIELD_HALF = FIELD_EXTENT / 2;
/**
 * One scent cell of padding so the live disk sits inside the snapped field
 * rectangle. Body radius is subtracted again at the wall, so glyphs do not
 * clip through the rim into a cell the field does not own.
 */
export const WORLD_BOUND_INSET = FIELD_CELL;

/**
 * Radius of the largest disk centred on `(cx, cy)` that sits inside the field
 * window `[origin, origin + FIELD_EXTENT)` by `WORLD_BOUND_INSET`. The one
 * number the wall, the scent mask, the energy bound, and soup spawn share.
 */
export function worldBoundRadius(
  cx: number,
  cy: number,
  originX: number,
  originY: number,
  extent = FIELD_EXTENT,
): number {
  const inscribed = Math.min(
    cx - originX,
    originX + extent - cx,
    cy - originY,
    originY + extent - cy,
  );
  return Math.max(0, inscribed - WORLD_BOUND_INSET);
}

export class Fields {
  cols: number;
  rows: number;
  worldW: number;
  worldH: number;
  originX = 0;
  originY = 0;
  /** Live-disk centre; unused until `setWorldBound`. */
  boundX = 0;
  boundY = 0;
  /** Live-disk radius. `<= 0` means no Dirichlet mask and no wall. */
  boundR = 0;
  data: Float32Array;
  /** The diffusion ping-pong's other half, allocated the first time a pass needs it. */
  private tmp: Float32Array | null = null;

  /**
   * Per-channel multipliers on the global diffuse and decay rates: a signal
   * spreads and fades, a conserved quantity spreads and stays, and a
   * reaction-diffusion pair only patterns when its species move at different
   * speeds. All ones is the only shape the solver's SIMD kernel can take.
   * Float64, so a rate of 1 leaves a mix bit-identical.
   */
  readonly diffuseRate = new Float64Array([1, 1, 1, 1]);
  readonly decayRate = new Float64Array([1, 1, 1, 1]);

  /**
   * `cells` a side over `extent` world units. Separate so a smaller dish can
   * keep the same cell: the steering constants are tuned to `FIELD_CELL`.
   * `Sim` passes `cells * FIELD_CELL`; the default gives the kernel tests a
   * coarse grid over the whole world.
   */
  constructor(cells = FIELD_CELLS, extent = FIELD_EXTENT) {
    this.cols = cells;
    this.rows = cells;
    this.worldW = extent;
    this.worldH = extent;
    this.data = new Float32Array(this.cols * this.rows * CHANNELS);
  }

  /*
   * The rectangle of cells that have ever been deposited into, plus a
   * margin, so the passes skip cells that are zero and will stay zero. The
   * box only ever grows, which keeps the invariant the ping-pong needs:
   * outside it both buffers are exactly zero. In practice the field fills
   * the dish within seconds; the disk mask is what keeps the passes cheap.
   */
  private static readonly MARGIN = 16;
  private loI = 0;
  private loJ = 0;
  private hiI = -1;
  private hiJ = -1;

  /** World units per cell. Fixed. */
  get cellSize(): number {
    return this.worldW / this.cols;
  }

  /**
   * Slide the field by whole cells so it stays centred on (cx, cy) —
   * normally `home`. Nothing is discarded except what scrolls off.
   */
  cover(cx: number, cy: number): void {
    const cell = this.cellSize;
    const half = this.worldW * 0.5;
    const di = Math.round((cx - half - this.originX) / cell);
    const dj = Math.round((cy - half - this.originY) / cell);
    if (di === 0 && dj === 0) return;
    // Only a field holding something has anything to move.
    if (this.hiI >= this.loI) this.shiftCells(di, dj);
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
    this.tmp?.fill(0);
    this.hiI = -1;
    this.hiJ = -1;
    this.loI = 0;
    this.loJ = 0;
    this.boundR = 0;
  }

  /**
   * Dirichlet disk: cell centres outside this radius read and write as 0, so
   * scent decays toward the rim rather than reflecting off it. `r <= 0` turns
   * the mask off (Neumann at the square grid edge).
   */
  setWorldBound(cx: number, cy: number, r: number): void {
    this.boundX = cx;
    this.boundY = cy;
    this.boundR = r > 0 ? r : 0;
    this.spanCols = -1;
  }

  /*
   * The disk, as one inclusive column span per row: a disk is convex, so its
   * intersection with a row is one contiguous run, and every `cellOut` in
   * the hot path becomes an integer compare. With no bound the spans are
   * full rows, so the masked and unmasked paths are the same loop.
   */
  private spanLo: Int32Array | null = null;
  private spanHi: Int32Array | null = null;
  /** Inputs the cached spans were built from. */
  private spanBX = NaN;
  private spanBY = NaN;
  private spanBR = NaN;
  private spanOX = NaN;
  private spanOY = NaN;
  private spanCols = -1;

  private spansFresh(): boolean {
    return (
      this.spanBX === this.boundX &&
      this.spanBY === this.boundY &&
      this.spanBR === this.boundR &&
      this.spanOX === this.originX &&
      this.spanOY === this.originY &&
      this.spanCols === this.cols
    );
  }

  private spans(): { lo: Int32Array; hi: Int32Array } {
    let lo = this.spanLo;
    let hi = this.spanHi;
    if (!lo || !hi || lo.length !== this.rows) {
      lo = this.spanLo = new Int32Array(this.rows);
      hi = this.spanHi = new Int32Array(this.rows);
      this.spanCols = -1;
    }
    if (this.spansFresh()) return { lo, hi };
    this.spanBX = this.boundX;
    this.spanBY = this.boundY;
    this.spanBR = this.boundR;
    this.spanOX = this.originX;
    this.spanOY = this.originY;
    this.spanCols = this.cols;
    const { cols, rows } = this;
    if (this.boundR <= 0) {
      lo.fill(0);
      hi.fill(cols - 1);
      return { lo, hi };
    }
    const cs = this.cellSize;
    const r = this.boundR;
    for (let j = 0; j < rows; j++) {
      const dy = this.originY + (j + 0.5) * cs - this.boundY;
      const inside = r * r - dy * dy;
      if (inside <= 0) {
        // Empty row: hi < lo, which every consumer already reads as "skip".
        lo[j] = 0;
        hi[j] = -1;
        continue;
      }
      const hw = Math.sqrt(inside);
      // Cell i's centre is origin + (i + 0.5) * cs, so the run is the cells
      // whose centres land within hw of the disk's own centre.
      let a = Math.ceil((this.boundX - hw - this.originX) / cs - 0.5);
      let b = Math.floor((this.boundX + hw - this.originX) / cs - 0.5);
      if (a < 0) a = 0;
      if (b > cols - 1) b = cols - 1;
      /*
       * Then walk the ends onto the exact run `cellOut` describes: the span
       * comes through a square root and `cellOut` is a squared compare, and
       * `add` still asks `cellOut` directly, so the two must agree on the rim
       * or a deposit lands in a cell diffusion treats as outside.
       */
      while (a <= b && this.cellOut(a, j)) a++;
      while (b >= a && this.cellOut(b, j)) b--;
      while (a > 0 && !this.cellOut(a - 1, j)) a--;
      while (b < cols - 1 && !this.cellOut(b + 1, j)) b++;
      lo[j] = a;
      hi[j] = b;
      if (b < a) hi[j] = a - 1;
    }
    return { lo, hi };
  }

  /** True when this cell's centre sits outside the live disk. */
  cellOut(i: number, j: number): boolean {
    if (this.boundR <= 0) return false;
    const cs = this.cellSize;
    const dx = this.originX + (i + 0.5) * cs - this.boundX;
    const dy = this.originY + (j + 0.5) * cs - this.boundY;
    return dx * dx + dy * dy > this.boundR * this.boundR;
  }

  /**
   * `new[j][i] = old[j + dj][i + di]`, in place. Rows are moved in the order
   * that reads each source row before anything overwrites it. The scratch
   * buffer is zeroed rather than moved: outside the live box both buffers
   * must be exactly zero.
   */
  private shiftCells(di: number, dj: number): void {
    const { cols, rows, data } = this;
    const rowLen = cols * CHANNELS;
    const iLo = Math.max(0, -di);
    const iHi = Math.min(cols, cols - di);
    const j0 = dj > 0 ? 0 : rows - 1;
    const j1 = dj > 0 ? rows : -1;
    const step = dj > 0 ? 1 : -1;
    for (let j = j0; j !== j1; j += step) {
      const sj = j + dj;
      const dst = j * rowLen;
      if (sj < 0 || sj >= rows || iHi <= iLo) {
        data.fill(0, dst, dst + rowLen);
        continue;
      }
      const src = sj * rowLen;
      data.copyWithin(dst + iLo * CHANNELS, src + (iLo + di) * CHANNELS, src + (iHi + di) * CHANNELS);
      if (iLo > 0) data.fill(0, dst, dst + iLo * CHANNELS);
      if (iHi < cols) data.fill(0, dst + iHi * CHANNELS, dst + rowLen);
    }
    this.tmp?.fill(0);
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
   * Reference cell size the scent response curves were tuned against:
   * `slow_factor`, `turn_boost` and the steering dead zone all read raw cell
   * values. A cell holds a total, not a density, so a deposit is normalised
   * by cell area to read the same at any resolution.
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

  /**
   * Add a quantity at a world point, spread bilinearly over the four cells
   * that straddle it. `deposit`'s twin without `depositScale`: a scent
   * deposit is a density, energy is a count and a unit must stay a unit
   * however the grid is cut.
   */
  addAt(ch: number, x: number, y: number, amount: number): void {
    if (amount === 0) return;
    const { gx, gy } = this.toGrid(x, y);
    const i0 = Math.floor(gx);
    const j0 = Math.floor(gy);
    this.touch(i0, j0);
    const tx = gx - i0;
    const ty = gy - j0;
    /*
     * Renormalised over the cells that will actually take it: a point inside
     * the disk can straddle cells whose centres are outside it, and `add`
     * drops those. Spreading the whole amount over the legal cells keeps
     * `addAt` conservative for any point the disk accepts at all.
     */
    const wi = [1 - tx, tx, 1 - tx, tx];
    const wj = [1 - ty, 1 - ty, ty, ty];
    const ii = [i0, i0 + 1, i0, i0 + 1];
    const jj = [j0, j0, j0 + 1, j0 + 1];
    let take = 0;
    for (let k = 0; k < 4; k++) {
      if (this.cell(ii[k], jj[k], ch) < 0 || this.cellOut(ii[k], jj[k])) {
        wi[k] = 0;
        continue;
      }
      take += wi[k] * wj[k];
    }
    if (take <= 0) return;
    const scale = amount / take;
    for (let k = 0; k < 4; k++) {
      const w = wi[k] * wj[k];
      if (w > 0) this.add(ii[k], jj[k], ch, w * scale);
    }
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
    if (this.cellOut(i, j)) return;
    const idx = this.cell(i, j, ch);
    if (idx >= 0) this.data[idx] += amount;
  }

  private at(i: number, j: number, ch: number): number {
    const idx = this.cell(i, j, ch);
    return idx < 0 ? 0 : this.data[idx];
  }

  /**
   * One diffusion pass over the live disk: the most expensive thing in the
   * frame at the sizes anyone runs, and a fixed cost, since the field fills
   * the dish. Written the long way: neighbour offsets are resolved once per
   * cell and the buffers ping-pong rather than copy.
   */
  diffuse(mix: number): void {
    if (mix <= 0) return;
    const dr = this.diffuseRate;
    const rate = (ch: number): number => {
      const r = mix * (dr[ch] > 0 ? dr[ch] : 0);
      return r > 1 ? 1 : r;
    };
    const m0 = rate(0);
    const m1 = rate(1);
    const m2 = rate(2);
    const m3 = rate(3);
    const k0 = 1 - m0;
    const k1 = 1 - m1;
    const k2 = 1 - m2;
    const k3 = 1 - m3;
    const { cols, rows } = this;
    if (this.hiI < this.loI) return;
    const src = this.data;
    const dst = this.tmp ?? (this.tmp = new Float32Array(src.length));
    const rowStride = cols * CHANNELS;

    /*
     * Widen by a cell before diffusing: a pass moves scent exactly one cell,
     * and the box must not be a ceiling on how far scent can spread.
     */
    if (this.loI > 0) this.loI--;
    if (this.loJ > 0) this.loJ--;
    if (this.hiI < this.cols - 1) this.hiI++;
    if (this.hiJ < this.rows - 1) this.hiJ++;
    /*
     * No bound: -1 reflects off the square edge. Bound set: a neighbour
     * outside the disk is 0, so scent leaks into the rim — except
     * `CH.energy`, which always reflects: a substance is conserved and a
     * signal is not, and an absorbing rim would drain the ground as
     * perimeter over area. Hard-coded, not a `Params` flag;
     * `field-kernel.test.ts` pins the two implementations together.
     */
    const dirichlet = this.boundR > 0;
    const { lo: sLo, hi: sHi } = this.spans();
    for (let j = this.loJ; j <= this.hiJ; j++) {
      // The disk's run on this row, clipped to the box. An empty row
      // (hi < lo) fails every `>= lo && <= hi` below, so no branch is needed.
      const a0 = sLo[j] > this.loI ? sLo[j] : this.loI;
      const b0 = sHi[j] < this.hiI ? sHi[j] : this.hiI;
      const upLo = j > 0 ? sLo[j - 1] : 0;
      const upHi = j > 0 ? sHi[j - 1] : -1;
      const dnLo = j < rows - 1 ? sLo[j + 1] : 0;
      const dnHi = j < rows - 1 ? sHi[j + 1] : -1;
      // Cells in the box but outside the disk are written zero: `dst` must
      // be fully defined over the box.
      const rowBase = j * cols * CHANNELS;
      if (a0 > this.loI) dst.fill(0, rowBase + this.loI * CHANNELS, rowBase + a0 * CHANNELS);
      if (b0 < this.hiI) dst.fill(0, rowBase + (b0 + 1) * CHANNELS, rowBase + (this.hiI + 1) * CHANNELS);
      if (b0 < a0) continue;
      /*
       * Split into edge and interior: `iLo`..`iHi` is the run where a cell
       * provably has all four neighbours. The arithmetic is identical and in
       * the same order on both paths, so this is the same pass.
       */
      const rowLo = sLo[j];
      const rowHi = sHi[j];
      let iLo = a0 > rowLo + 1 ? a0 : rowLo + 1;
      if (iLo < upLo) iLo = upLo;
      if (iLo < dnLo) iLo = dnLo;
      let iHi = b0 < rowHi - 1 ? b0 : rowHi - 1;
      if (iHi > upHi) iHi = upHi;
      if (iHi > dnHi) iHi = dnHi;
      if (iHi < iLo) {
        iLo = a0;
        iHi = a0 - 1;
      }
      for (let i = a0, base = rowBase + (a0) * CHANNELS; i <= iLo - 1; i++, base += CHANNELS) {
        const left = i > rowLo ? base - CHANNELS : -1;
        const right = i < rowHi ? base + CHANNELS : -1;
        const up = i >= upLo && i <= upHi ? base - rowStride : -1;
        const down = i >= dnLo && i <= dnHi ? base + rowStride : -1;
        // Unrolled over the four channels with the rates as locals, in the
        // same order as the interior loop.
        const s0 = src[base];
        const s1 = src[base + 1];
        const s2 = src[base + 2];
        const s3 = src[base + 3];
        const l = left >= 0;
        const r = right >= 0;
        const u = up >= 0;
        const w = down >= 0;
        dst[base] =
          k0 * s0 +
          m0 *
            ((l ? src[left] : dirichlet ? 0 : s0) +
              (r ? src[right] : dirichlet ? 0 : s0) +
              (u ? src[up] : dirichlet ? 0 : s0) +
              (w ? src[down] : dirichlet ? 0 : s0)) *
            0.25;
        dst[base + 1] =
          k1 * s1 +
          m1 *
            ((l ? src[left + 1] : dirichlet ? 0 : s1) +
              (r ? src[right + 1] : dirichlet ? 0 : s1) +
              (u ? src[up + 1] : dirichlet ? 0 : s1) +
              (w ? src[down + 1] : dirichlet ? 0 : s1)) *
            0.25;
        dst[base + 2] =
          k2 * s2 +
          m2 *
            ((l ? src[left + 2] : s2) +
              (r ? src[right + 2] : s2) +
              (u ? src[up + 2] : s2) +
              (w ? src[down + 2] : s2)) *
            0.25;
        dst[base + 3] =
          k3 * s3 +
          m3 *
            ((l ? src[left + 3] : dirichlet ? 0 : s3) +
              (r ? src[right + 3] : dirichlet ? 0 : s3) +
              (u ? src[up + 3] : dirichlet ? 0 : s3) +
              (w ? src[down + 3] : dirichlet ? 0 : s3)) *
            0.25;
      }
      for (let i = iLo, base = rowBase + iLo * CHANNELS; i <= iHi; i++, base += CHANNELS) {
        const l = base - CHANNELS;
        const r = base + CHANNELS;
        const u = base - rowStride;
        const w = base + rowStride;
        dst[base] = k0 * src[base] + m0 * (src[l] + src[r] + src[u] + src[w]) * 0.25;
        dst[base + 1] =
          k1 * src[base + 1] + m1 * (src[l + 1] + src[r + 1] + src[u + 1] + src[w + 1]) * 0.25;
        dst[base + 2] =
          k2 * src[base + 2] + m2 * (src[l + 2] + src[r + 2] + src[u + 2] + src[w + 2]) * 0.25;
        dst[base + 3] =
          k3 * src[base + 3] + m3 * (src[l + 3] + src[r + 3] + src[u + 3] + src[w + 3]) * 0.25;
      }
      for (let i = iHi + 1, base = rowBase + (iHi + 1) * CHANNELS; i <= b0; i++, base += CHANNELS) {
        const left = i > rowLo ? base - CHANNELS : -1;
        const right = i < rowHi ? base + CHANNELS : -1;
        const up = i >= upLo && i <= upHi ? base - rowStride : -1;
        const down = i >= dnLo && i <= dnHi ? base + rowStride : -1;
        // Same edge stencil as above.
        const s0 = src[base];
        const s1 = src[base + 1];
        const s2 = src[base + 2];
        const s3 = src[base + 3];
        const l = left >= 0;
        const r = right >= 0;
        const u = up >= 0;
        const w = down >= 0;
        dst[base] =
          k0 * s0 +
          m0 *
            ((l ? src[left] : dirichlet ? 0 : s0) +
              (r ? src[right] : dirichlet ? 0 : s0) +
              (u ? src[up] : dirichlet ? 0 : s0) +
              (w ? src[down] : dirichlet ? 0 : s0)) *
            0.25;
        dst[base + 1] =
          k1 * s1 +
          m1 *
            ((l ? src[left + 1] : dirichlet ? 0 : s1) +
              (r ? src[right + 1] : dirichlet ? 0 : s1) +
              (u ? src[up + 1] : dirichlet ? 0 : s1) +
              (w ? src[down + 1] : dirichlet ? 0 : s1)) *
            0.25;
        dst[base + 2] =
          k2 * s2 +
          m2 *
            ((l ? src[left + 2] : s2) +
              (r ? src[right + 2] : s2) +
              (u ? src[up + 2] : s2) +
              (w ? src[down + 2] : s2)) *
            0.25;
        dst[base + 3] =
          k3 * s3 +
          m3 *
            ((l ? src[left + 3] : dirichlet ? 0 : s3) +
              (r ? src[right + 3] : dirichlet ? 0 : s3) +
              (u ? src[up + 3] : dirichlet ? 0 : s3) +
              (w ? src[down + 3] : dirichlet ? 0 : s3)) *
            0.25;
      }
    }

    this.data = dst;
    this.tmp = src;
  }

  decay(rate: number): void {
    this.decayCells(rate);
  }

  private decayCells(rate: number): void {
    if (this.hiI < this.loI) return;
    const d = this.data;
    const { cols } = this;
    const dr = this.decayRate;
    const keep = (ch: number): number => {
      const r = rate * (dr[ch] > 0 ? dr[ch] : 0);
      return r >= 1 ? 0 : 1 - r;
    };
    const k0 = keep(0);
    const k1 = keep(1);
    const k2 = keep(2);
    const k3 = keep(3);
    const { lo: sLo, hi: sHi } = this.spans();
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const rowBase = j * cols * CHANNELS;
      const a0 = sLo[j] > this.loI ? sLo[j] : this.loI;
      const b0 = sHi[j] < this.hiI ? sHi[j] : this.hiI;
      // Outside the disk is zeroed, not decayed — same two `fill`s as diffuse.
      if (a0 > this.loI) d.fill(0, rowBase + this.loI * CHANNELS, rowBase + a0 * CHANNELS);
      if (b0 < this.hiI) d.fill(0, rowBase + (b0 + 1) * CHANNELS, rowBase + (this.hiI + 1) * CHANNELS);
      for (let i = a0, base = rowBase + a0 * CHANNELS; i <= b0; i++, base += CHANNELS) {
        d[base] *= k0;
        d[base + 1] *= k1;
        d[base + 2] *= k2;
        d[base + 3] *= k3;
      }
    }
  }

  /**
   * Logistic regrowth on one channel: `E += r * E * (1 - E / cap)`. Zero is
   * a fixed point, so a cell grazed to nothing only comes back by diffusion
   * from its rim. One-directional: a cell above `cap` is left alone, never
   * pulled down, because the ground is conserved. The result is clamped at
   * `cap` because the fertiliser term can make a single explicit step
   * overshoot. `r` is per frame, already multiplied by dt by the caller.
   */
  grow(ch: number, r: number, cap: number, catCh = -1, gamma = 0): void {
    if (!(r > 0) || !(cap > 0)) return;
    if (this.hiI < this.loI) return;
    const d = this.data;
    const { cols } = this;
    const { lo: sLo, hi: sHi } = this.spans();
    const invCap = 1 / cap;
    const catalysed = catCh >= 0 && catCh !== ch && gamma !== 0;
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const rowBase = j * cols * CHANNELS;
      const a0 = sLo[j] > this.loI ? sLo[j] : this.loI;
      const b0 = sHi[j] < this.hiI ? sHi[j] : this.hiI;
      // Two loops rather than a loop-invariant branch per cell.
      if (catalysed) {
        const off = catCh - ch;
        for (let i = a0, k = rowBase + a0 * CHANNELS + ch; i <= b0; i++, k += CHANNELS) {
          const e = d[k];
          if (e <= 0 || e >= cap) continue;
          // The catalyst scales the rate, never the outcome. Clamped at zero so
          // an inhibitor (negative gamma) can stall regrowth but never run it backwards.
          const rr = r * (1 + gamma * d[k + off]);
          if (rr <= 0) continue;
          const next = e + rr * e * (1 - e * invCap);
          d[k] = next > cap ? cap : next;
        }
      } else {
        for (let i = a0, k = rowBase + a0 * CHANNELS + ch; i <= b0; i++, k += CHANNELS) {
          const e = d[k];
          if (e <= 0 || e >= cap) continue;
          const next = e + r * e * (1 - e * invCap);
          d[k] = next > cap ? cap : next;
        }
      }
    }
  }

  /**
   * Gray-Scott between two channels: `u + 2v -> 3v`, fed and killed.
   *
   *     uvv = u * v * v
   *     u  +=  -uvv + feed * (1 - u)
   *     v  +=   uvv - (feed + kill) * v
   *
   * Puts local maxima where nobody is standing, so signal comes apart from
   * source. Needs the species to diffuse at different rates (`diffuseRate`;
   * substrate about twice the activator) and `feed`/`kill` inside roughly
   * F in [0.01, 0.09], k in [0.045, 0.07]; outside that it is a uniform wash,
   * which is why it is off unless deliberately turned on. `u` is normalised
   * toward 1 by the feed term, so it expects a channel whose natural scale
   * is about 1.
   */
  react(uCh: number, vCh: number, feed: number, kill: number, dt: number): void {
    if (!(dt > 0) || uCh === vCh) return;
    if (!(feed > 0) && !(kill > 0)) return;
    if (this.hiI < this.loI) return;
    const d = this.data;
    const { cols } = this;
    const { lo: sLo, hi: sHi } = this.spans();
    const f = feed * dt;
    const kv = (feed + kill) * dt;
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const rowBase = j * cols * CHANNELS;
      const a0 = sLo[j] > this.loI ? sLo[j] : this.loI;
      const b0 = sHi[j] < this.hiI ? sHi[j] : this.hiI;
      for (let i = a0, k = rowBase + a0 * CHANNELS; i <= b0; i++, k += CHANNELS) {
        const u = d[k + uCh];
        const v = d[k + vCh];
        const uvv = u * v * v * dt;
        const nu = u - uvv + f * (1 - u);
        const nv = v + uvv - kv * v;
        d[k + uCh] = nu > 0 ? nu : 0;
        d[k + vCh] = nv > 0 ? nv : 0;
      }
    }
  }

  /**
   * Set one channel to `value` across every cell inside the live disk, and
   * open the box over it. How a world starts with ground in it.
   */
  fillDisk(ch: number, value: number): void {
    if (this.boundR <= 0) return;
    const { lo, hi } = this.spans();
    const d = this.data;
    const stride = this.cols * CHANNELS;
    let loJ = -1;
    let hiJ = -1;
    let loI = this.cols;
    let hiI = -1;
    for (let j = 0; j < this.rows; j++) {
      if (hi[j] < lo[j]) continue;
      if (loJ < 0) loJ = j;
      hiJ = j;
      if (lo[j] < loI) loI = lo[j];
      if (hi[j] > hiI) hiI = hi[j];
      for (let i = lo[j], k = j * stride + lo[j] * CHANNELS + ch; i <= hi[j]; i++, k += CHANNELS) {
        d[k] = value;
      }
    }
    if (hiJ < 0) return;
    // The box has to cover what was just written, or the passes skip it.
    if (this.hiI < this.loI) {
      this.loI = loI;
      this.hiI = hiI;
      this.loJ = loJ;
      this.hiJ = hiJ;
      return;
    }
    if (loI < this.loI) this.loI = loI;
    if (hiI > this.hiI) this.hiI = hiI;
    if (loJ < this.loJ) this.loJ = loJ;
    if (hiJ > this.hiJ) this.hiJ = hiJ;
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

  /** All four channels at one point, into `out[off .. off+4)`: `sample` once, four lerps. */
  sampleAll(x: number, y: number, out: Float64Array, off = 0): void {
    const { gx, gy } = this.toGrid(x, y);
    const i0 = Math.floor(gx);
    const j0 = Math.floor(gy);
    const tx = gx - i0;
    const ty = gy - j0;
    const w00 = (1 - tx) * (1 - ty);
    const w10 = tx * (1 - ty);
    const w01 = (1 - tx) * ty;
    const w11 = tx * ty;
    const a = this.cell(i0, j0, 0);
    const b = this.cell(i0 + 1, j0, 0);
    const c = this.cell(i0, j0 + 1, 0);
    const e = this.cell(i0 + 1, j0 + 1, 0);
    const d = this.data;
    for (let ch = 0; ch < CHANNELS; ch++) {
      out[off + ch] =
        (a < 0 ? 0 : d[a + ch] * w00) +
        (b < 0 ? 0 : d[b + ch] * w10) +
        (c < 0 ? 0 : d[c + ch] * w01) +
        (e < 0 ? 0 : d[e + ch] * w11);
    }
  }

  /**
   * The loudest thing anything is saying — the signal channels only, since
   * the ground sits at capacity across the dish. Pass a channel to ask about
   * one specifically, the ground included.
   */
  peak(ch?: number): number {
    let m = 0;
    const d = this.data;
    if (ch !== undefined) {
      for (let i = ch; i < d.length; i += CHANNELS) if (d[i] > m) m = d[i];
      return m;
    }
    // `VOICE`, unrolled: a `for...of` allocates an iterator per cell.
    for (let i = 0; i < d.length; i += CHANNELS) {
      if (d[i + CH.conP] > m) m = d[i + CH.conP];
      if (d[i + CH.dupP] > m) m = d[i + CH.dupP];
      if (d[i + CH.aux] > m) m = d[i + CH.aux];
    }
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
        const posEra = Math.max(0, data[base + CH.energy]);
        const posAux = Math.max(0, data[base + CH.aux]);
        const neg =
          Math.max(0, -data[base + CH.conP]) +
          Math.max(0, -data[base + CH.dupP]) +
          Math.max(0, -data[base + CH.energy]) +
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
