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
/**
 * One scent cell of padding so the live disk sits inside the snapped field
 * rectangle. Body radius is subtracted again at the wall, so glyphs do not
 * clip through the rim into a cell the field does not own.
 */
export const WORLD_BOUND_INSET = FIELD_CELL;

/**
 * Radius of the largest disk centred on `(cx, cy)` that sits inside the field
 * window `[origin, origin + FIELD_EXTENT)` by `WORLD_BOUND_INSET`.
 *
 * `Fields.cover` snaps the origin to a whole cell, so a circle of radius
 * `FIELD_HALF` can poke up to half a cell past the tight edge. This is the
 * number the wall, the scent mask, the energy bound, and soup spawn share.
 */
export function worldBoundRadius(cx: number, cy: number, originX: number, originY: number): number {
  const inscribed = Math.min(
    cx - originX,
    originX + FIELD_EXTENT - cx,
    cy - originY,
    originY + FIELD_EXTENT - cy,
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
   * The rectangle of cells scent is actually in, plus a small margin.
   *
   * The grid is a million cells and a pond touches a small patch of it, so
   * diffusing and decaying the whole thing spends almost all of its time on
   * zeros — which stay zero however many times they are averaged with their
   * neighbours. Tracking where the scent actually is turns both passes into
   * work proportional to the pond rather than to the world.
   *
   * It has to track *both* ways to do that, which it did not.
   *
   * The box used only ever to grow, and `diffuse` widened it by a cell on
   * every side of every call whether or not anything had reached the edge.
   * Two passes a frame is two cells a side a frame, so from a centred pond
   * the box reached the grid's own edge in about 256 frames and stayed
   * there. Measured, a 198-body pond and a 20,000-body one both paid the
   * same 44 ms a frame after four seconds — the whole point of the box,
   * given away on a timer. Widening is now conditional on the edge actually
   * holding something (`diffuse`), and `trim` pulls the sides back in as
   * decay empties them.
   *
   * Shrinking has one obligation. The ping-pong needs both buffers to be
   * exactly zero outside the box, so that a cell entering it for the first
   * time reads zero in whichever buffer is about to be read; growth
   * preserved that for free, and shrinking does not. `trim` therefore zeroes
   * every row and column it abandons, in both buffers. That also throws away
   * the vanishing tail below `EMPTY`, which decay was taking to nothing
   * anyway.
   *
   * The margin is what a deposit claims beyond itself. It was sixteen, as
   * headroom so that diffusion had somewhere to spread before the next
   * deposit extended the box again — a job that belongs to the conditional
   * widening now. What is left is the deposit's own bilinear splat (two
   * cells) and the frame's two diffusion passes (one each), so four covers
   * it with room to spare. Keeping it at sixteen would only mean `trim`
   * clearing twelve rows a side a frame that nothing had ever reached.
   */
  private static readonly MARGIN = 4;

  /**
   * Magnitude at or below which a cell counts as empty for the box.
   *
   * Only the box reads this; the field itself keeps whatever it holds. It
   * wants to sit far enough below anything that can change behaviour that
   * trimming is invisible, and far enough above denormals to actually
   * terminate. Steering compares two sensors against a dead zone of 5% of
   * the signal, and `scentSlowFactor` starts to bite around a trail of 1,
   * so a cell four orders of magnitude below that cannot move any part of
   * the sim.
   */
  private static readonly EMPTY = 1e-4;
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
    this.boundR = 0;
  }

  /**
   * Dirichlet disk: cell centres outside this radius read and write as 0, so
   * scent decays toward the rim rather than reflecting off it. `r <= 0` turns
   * the mask off (Neumann at the square grid edge, the old behaviour).
   */
  setWorldBound(cx: number, cy: number, r: number): void {
    this.boundX = cx;
    this.boundY = cy;
    this.boundR = r > 0 ? r : 0;
    this.spanStamp = '';
  }

  /*
   * The disk, as one inclusive column span per row, instead of a predicate.
   *
   * `cellOut` is a world-space distance — two subtractions, two multiplies
   * and a compare — and `diffuse` asked it five times per cell, once for the
   * cell and once for each neighbour it might read from. Measured over the
   * whole grid that was 11.6 ms of a 52.3 ms field frame: the mask cost more
   * than a fifth of the work, to exclude a fifth of the grid that the loops
   * then iterated anyway.
   *
   * A disk is convex, so its intersection with a row is one contiguous run,
   * and the whole mask is two integers per row. Rows above and below it are
   * skipped outright, the inner loop runs the span instead of the width, and
   * every `cellOut` in the hot path becomes an integer compare against the
   * span of the row being read from.
   *
   * With no bound the spans are full rows, so the masked and unmasked paths
   * are the same loop and neither pays a branch for the other.
   */
  private spanLo: Int32Array | null = null;
  private spanHi: Int32Array | null = null;
  /** Inputs the cached spans were built from; `''` forces a rebuild. */
  private spanStamp = '';

  private spans(): { lo: Int32Array; hi: Int32Array } {
    const stamp = `${this.boundX},${this.boundY},${this.boundR},${this.originX},${this.originY},${this.cols}`;
    let lo = this.spanLo;
    let hi = this.spanHi;
    if (!lo || !hi || lo.length !== this.rows) {
      lo = this.spanLo = new Int32Array(this.rows);
      hi = this.spanHi = new Int32Array(this.rows);
      this.spanStamp = '';
    }
    if (this.spanStamp === stamp) return { lo, hi };
    this.spanStamp = stamp;
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

  /**
   * True when every cell of this row, across the box's current width, is
   * within `EMPTY` of zero. Signed: a channel is allowed below zero, and a
   * strong negative is as much "something is here" as a strong positive.
   */
  private rowEmpty(j: number): boolean {
    const e = Fields.EMPTY;
    const d = this.data;
    let base = (j * this.cols + this.loI) * CHANNELS;
    for (let i = this.loI; i <= this.hiI; i++, base += CHANNELS) {
      if (d[base] > e || d[base] < -e) return false;
      if (d[base + 1] > e || d[base + 1] < -e) return false;
      if (d[base + 2] > e || d[base + 2] < -e) return false;
      if (d[base + 3] > e || d[base + 3] < -e) return false;
    }
    return true;
  }

  /** `rowEmpty` down a column, across the box's current height. */
  private colEmpty(i: number): boolean {
    const e = Fields.EMPTY;
    const d = this.data;
    const stride = this.cols * CHANNELS;
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const base = j * stride + i * CHANNELS;
      if (d[base] > e || d[base] < -e) return false;
      if (d[base + 1] > e || d[base + 1] < -e) return false;
      if (d[base + 2] > e || d[base + 2] < -e) return false;
      if (d[base + 3] > e || d[base + 3] < -e) return false;
    }
    return true;
  }

  /**
   * Zero a row across the box's width, in *both* buffers.
   *
   * The second buffer is the whole point. Outside the box the ping-pong
   * assumes both are exactly zero, so a row handed back to the outside has
   * to be left that way in the one that is not currently live as well —
   * otherwise a later deposit that grows the box back over it reads a stale
   * value out of `tmp` on the next diffuse.
   */
  private clearRow(j: number): void {
    const from = (j * this.cols + this.loI) * CHANNELS;
    const to = (j * this.cols + this.hiI + 1) * CHANNELS;
    this.data.fill(0, from, to);
    this.tmp.fill(0, from, to);
  }

  /** `clearRow` down a column. Strided, so no `fill` to lean on. */
  private clearCol(i: number): void {
    const stride = this.cols * CHANNELS;
    const d = this.data;
    const t = this.tmp;
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const base = j * stride + i * CHANNELS;
      d[base] = d[base + 1] = d[base + 2] = d[base + 3] = 0;
      t[base] = t[base + 1] = t[base + 2] = t[base + 3] = 0;
    }
  }

  /**
   * Pull the box in past the sides decay has emptied.
   *
   * The other half of tracking where the scent is; `diffuse`'s conditional
   * widening is the first. Without this the box is a high-water mark — a
   * pond that swims away leaves the cost of everywhere it has ever been
   * behind it, forever.
   *
   * Rows before columns, so the column scans are over an already-shortened
   * height and a box that has gone entirely empty is usually settled by the
   * first loop alone.
   *
   * Budgeted per side per frame. Decay empties an edge a row at a time, so
   * the loops almost always run once or not at all, and the bound is only
   * there so that a field emptied all at once — a `clear`, a preset swap, a
   * decay slider yanked to the top — trims over a few frames instead of
   * walking the whole box in one. A margin's worth converges fast enough
   * that nothing perceives it.
   */
  private trim(): void {
    if (this.hiI < this.loI) return;
    const budget = Fields.MARGIN;
    let n = budget;
    while (n-- > 0 && this.loJ <= this.hiJ && this.rowEmpty(this.loJ)) this.clearRow(this.loJ++);
    // Everything in the box was empty, so there is no box. `touch` seeds all
    // four edges again from scratch the next time anything is deposited.
    if (this.loJ > this.hiJ) {
      this.hiI = this.loI - 1;
      return;
    }
    n = budget;
    while (n-- > 0 && this.hiJ > this.loJ && this.rowEmpty(this.hiJ)) this.clearRow(this.hiJ--);
    n = budget;
    while (n-- > 0 && this.loI <= this.hiI && this.colEmpty(this.loI)) this.clearCol(this.loI++);
    if (this.loI > this.hiI) {
      this.hiI = this.loI - 1;
      return;
    }
    n = budget;
    while (n-- > 0 && this.hiI > this.loI && this.colEmpty(this.hiI)) this.clearCol(this.hiI--);
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
     * spread — a margin's width from the nearest deposit, whatever the
     * diffusion rate — so turning diffusion up flattened the peak without
     * extending the reach, which looks exactly like turning it down. The box
     * is meant to skip cells that are zero and will stay zero, not to clip
     * the physics.
     *
     * Per side, and only where there is something to spread. An edge that is
     * empty has nothing to push outward, so widening past it buys a row of
     * zeros being averaged with zeros — which is precisely the work the box
     * exists to skip. Done unconditionally, as it was, this alone walked the
     * box out to the whole grid in four seconds however little scent was in
     * it. The scan is a perimeter, not an area.
     */
    if (this.loI > 0 && !this.colEmpty(this.loI)) this.loI--;
    if (this.loJ > 0 && !this.rowEmpty(this.loJ)) this.loJ--;
    if (this.hiI < this.cols - 1 && !this.colEmpty(this.hiI)) this.hiI++;
    if (this.hiJ < this.rows - 1 && !this.rowEmpty(this.hiJ)) this.hiJ++;
    // No bound: -1 reflects off the square edge (neither absorbs nor invents).
    // Bound set: a neighbour outside the disk is 0, so scent leaks into the rim.
    const dirichlet = this.boundR > 0;
    const { lo: sLo, hi: sHi } = this.spans();
    for (let j = this.loJ; j <= this.hiJ; j++) {
      // The disk's run on this row, clipped to the box. Its neighbours' runs
      // are read once here rather than per cell; an empty row (hi < lo) fails
      // every `>= lo && <= hi` below, so no branch is needed for it.
      const a0 = sLo[j] > this.loI ? sLo[j] : this.loI;
      const b0 = sHi[j] < this.hiI ? sHi[j] : this.hiI;
      const upLo = j > 0 ? sLo[j - 1] : 0;
      const upHi = j > 0 ? sHi[j - 1] : -1;
      const dnLo = j < rows - 1 ? sLo[j + 1] : 0;
      const dnHi = j < rows - 1 ? sHi[j + 1] : -1;
      /*
       * Cells of this row that are in the box but outside the disk still have
       * to be written zero — the ping-pong's contract is that `dst` is fully
       * defined over the box, and the mask used to satisfy it a cell at a
       * time. Two `fill`s do it as a memset instead, which is most of why
       * dropping `cellOut` pays twice: the distance math goes, and so does
       * the per-cell store it was guarding.
       */
      const rowBase = j * cols * CHANNELS;
      if (a0 > this.loI) dst.fill(0, rowBase + this.loI * CHANNELS, rowBase + a0 * CHANNELS);
      if (b0 < this.hiI) dst.fill(0, rowBase + (b0 + 1) * CHANNELS, rowBase + (this.hiI + 1) * CHANNELS);
      if (b0 < a0) continue;
      let base = rowBase + a0 * CHANNELS;
      for (let i = a0; i <= b0; i++, base += CHANNELS) {
        const left = i > sLo[j] ? base - CHANNELS : -1;
        const right = i < sHi[j] ? base + CHANNELS : -1;
        const up = i >= upLo && i <= upHi ? base - rowStride : -1;
        const down = i >= dnLo && i <= dnHi ? base + rowStride : -1;
        for (let ch = 0; ch < CHANNELS; ch++) {
          const k = base + ch;
          const self = src[k];
          const a = left >= 0 ? src[left + ch] : dirichlet ? 0 : self;
          const b = right >= 0 ? src[right + ch] : dirichlet ? 0 : self;
          const c = up >= 0 ? src[up + ch] : dirichlet ? 0 : self;
          const e = down >= 0 ? src[down + ch] : dirichlet ? 0 : self;
          dst[k] = keep * self + m * (a + b + c + e) * 0.25;
        }
      }
    }

    this.data = dst;
    this.tmp = src;
  }

  /**
   * Decay, then pull the box in behind it.
   *
   * `trim` hangs off the end of decay rather than off its own call in `Sim`
   * because decay is the pass that empties a cell — it is the only thing
   * that can newly make an edge trimmable, and it is the last field pass of
   * the frame. It runs whichever decay path did, so the native one does not
   * quietly leave the box at its high-water mark.
   */
  decay(rate: number): void {
    const k = Math.max(0, 1 - rate);
    if (!nativeSolver.scentDecay(this, k)) this.decayCells(k);
    this.trim();
  }

  private decayCells(k: number): void {
    if (this.hiI < this.loI) return;
    const d = this.data;
    const { cols } = this;
    const { lo: sLo, hi: sHi } = this.spans();
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const rowBase = j * cols * CHANNELS;
      const a0 = sLo[j] > this.loI ? sLo[j] : this.loI;
      const b0 = sHi[j] < this.hiI ? sHi[j] : this.hiI;
      // Outside the disk is zeroed, not decayed — same two `fill`s as diffuse.
      if (a0 > this.loI) d.fill(0, rowBase + this.loI * CHANNELS, rowBase + a0 * CHANNELS);
      if (b0 < this.hiI) d.fill(0, rowBase + (b0 + 1) * CHANNELS, rowBase + (this.hiI + 1) * CHANNELS);
      for (let i = a0, base = rowBase + a0 * CHANNELS; i <= b0; i++, base += CHANNELS) {
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
