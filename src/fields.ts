import { nativeSolver } from './native/solver.ts';

export const CH = {
  conP: 0,
  dupP: 1,
  /**
   * Energy. Not a signal — the stuff itself, stored where everything else
   * about this world is stored.
   *
   * It was the channel an Era's principal laid into, and it was the one
   * channel nothing listened to: `seedChem` gives no kind a taste for it, so
   * Eras spent their whole unit of voice shouting into a band with no
   * receivers. That made it the obvious place to put a substance, and putting
   * it here is what turns "how much food is at this spot" into something the
   * same diffusion, the same disk mask and the same sampling already answer
   * for everything else.
   *
   * Nothing deposits into it through `emit`. A body's voice is spent across
   * the three signalling channels; energy arrives by dying, by a rewrite's
   * leftovers, by a full producer spilling, and by growing.
   */
  energy: 2,
  aux: 3,
} as const;

export const CHANNELS = 4;

/** The channels a body's voice is spread across — everything but `energy`. */
export const VOICE = [CH.conP, CH.dupP, CH.aux] as const;

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

  /**
   * Per-channel multipliers on the global diffuse and decay rates.
   *
   * The sliders stay one number each, because "how volatile is this world"
   * is one property of the world. What a channel *is* then scales it: these
   * say how fast this particular substance moves and how fast it goes away,
   * relative to everything else.
   *
   * They exist because the four channels stopped being four of the same
   * thing. A signal wants to spread and fade — that is what makes a trail a
   * trail. A conserved quantity wants to spread and stay, because energy
   * that evaporates is energy the economy has to mint back. And a
   * reaction-diffusion pair only patterns at all when the two species move
   * at different speeds: Gray-Scott wants the substrate about twice the
   * activator, and at one rate for everything there is no instability to
   * find.
   *
   * All ones is exactly the old behaviour, which is also the only shape the
   * SIMD kernel in the solver can take — it splats one rate across the whole
   * register. `uniform` is what decides whether the native path is still
   * allowed to answer.
   *
   * Float64, so that a rate of 1 multiplies a mix of 0.6 back into 0.6.
   * Float32 rounds both of the mixes a frame actually uses — 0.6 and 0.39 —
   * and the pass would then land every cell a few bits off the one it
   * replaces, for a change that is supposed to do nothing until a rate moves.
   * Four numbers; there is nothing to save by narrowing them.
   */
  readonly diffuseRate = new Float64Array([1, 1, 1, 1]);
  readonly decayRate = new Float64Array([1, 1, 1, 1]);

  private uniform(rate: Float64Array): boolean {
    return rate[0] === rate[1] && rate[1] === rate[2] && rate[2] === rate[3];
  }

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
   *
   * It briefly grew a shrinking half — a `trim` that pulled the sides back
   * past rows decay had emptied, and a widening conditional on the edge
   * holding something. Both are gone, and the reason is worth keeping. At
   * this world's rates the field fills the dish within seconds however few
   * bodies are in it — measured, a 198-body pond and a 20,000-body one both
   * occupy the whole grid — so there is never anything to reclaim. What the
   * trim did instead was fold sub-threshold mass near the rim back inward,
   * against a Dirichlet boundary whose whole job is to absorb it, and that
   * moved a settled pond for no measurable gain. The disk mask below is what
   * actually keeps this cheap.
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
      /*
       * Then walk the ends onto the exact run `cellOut` describes.
       *
       * The span is derived through a square root and `cellOut` is a squared
       * compare, so on a cell sitting all but exactly on the rim the two can
       * disagree in the last bit. That would be harmless if the span were the
       * only answer, but `add` and `at` still ask `cellOut` directly — a
       * deposit would land in a cell the diffusion then treated as outside.
       * One question, one answer: these loops move each end by a cell at
       * most, once per rebuild, and cost nothing per frame.
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

  /**
   * Add a *quantity* at a world point, spread bilinearly over the four cells
   * that straddle it.
   *
   * `deposit`'s twin, and the difference is `depositScale`. A scent deposit is
   * a density: widen the cell and the same emitter should still read the same
   * strength, so the amount is scaled by the cell's area. Energy is a count of
   * things, and a unit of it has to stay a unit however the grid is cut, or
   * the economy's totals move when the resolution does.
   *
   * Bilinear rather than into one cell, so that what lands is independent of
   * where inside a cell it landed — a corpse a hair either side of a boundary
   * should not be worth a different amount.
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
     * Renormalised over the cells that will actually take it, which matters
     * only at the rim and matters a lot there.
     *
     * A point inside the disk can still straddle cells whose *centres* are
     * outside it, because `inBounds` asks about the point and `cellOut` asks
     * about the centre. `add` drops those, so a corpse near the wall used to
     * lose whatever share of itself landed in them — silently, and worse the
     * closer to the edge it died. A signal channel would never notice; a
     * conserved one is exactly where a small unbounded leak is unaffordable.
     *
     * Spreading the whole amount over whichever of the four are legal keeps
     * `addAt` conservative for any point the disk accepts at all. When none of
     * them are, the point is outside in every sense and the deposit is
     * genuinely dropped.
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
    if (this.uniform(this.diffuseRate) && nativeSolver.scentDiffuse(this, mix)) return;
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
     * reach, which looks exactly like turning it down. The box is meant to
     * skip cells that are zero and will stay zero, not to clip the physics.
     *
     * It grows to the whole grid if scent genuinely reaches the whole grid,
     * which is the correct cost of that happening — and at this world's rates
     * it does, within seconds. Decay does not bound it in practice; the disk
     * mask is what keeps the work down.
     */
    if (this.loI > 0) this.loI--;
    if (this.loJ > 0) this.loJ--;
    if (this.hiI < this.cols - 1) this.hiI++;
    if (this.hiJ < this.rows - 1) this.hiJ++;
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
        /*
         * Unrolled over the four channels, because they no longer share a
         * rate and reading one out of an array per channel per cell is the
         * whole of what per-channel rates would otherwise cost. Measured over
         * a 1024^2 pass: 15.9 ms through a Float32Array, 14.9 ms through a
         * Float64Array, 9.5 ms with the rates as locals. The array is small
         * enough to sit in L1 either way — what it costs is the load itself,
         * eight of them per cell, and unrolling is how they go away.
         *
         * Same shape as the one-rate version it replaces, in the same order,
         * so at a rate of 1 every cell lands on the same bits it used to.
         */
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
            ((l ? src[left + 2] : dirichlet ? 0 : s2) +
              (r ? src[right + 2] : dirichlet ? 0 : s2) +
              (u ? src[up + 2] : dirichlet ? 0 : s2) +
              (w ? src[down + 2] : dirichlet ? 0 : s2)) *
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
    const k = Math.max(0, 1 - rate);
    if (this.uniform(this.decayRate) && nativeSolver.scentDecay(this, k)) return;
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
   * Logistic regrowth on one channel: `E += r * E * (1 - E / cap)`.
   *
   * The world's productivity, and the shape of it matters more than the rate.
   *
   * Growth is proportional to what is already there, so **a cell grazed to
   * exactly nothing does not come back**. Zero is a fixed point of the
   * logistic, and the only thing that can recolonise a dead cell is diffusion
   * from a neighbour that still has something — which spreads at a finite
   * speed, from the edges inward. That is the whole anti-strip-mine argument
   * in one line: overgraze a patch and you have made a scar that heals slowly
   * and from its rim, rather than a cell that refills on a timer wherever you
   * happen to be standing.
   *
   * Bounded by `cap`, so this is not a free-energy tap: it is the carrying
   * capacity of the ground, and the most the whole dish can hold is fixed.
   *
   * One-directional. A cell above capacity — a corpse's whole worth dropped
   * in one place — is left alone rather than pulled back down, because this
   * is a resource and not a signal, and the tidy-looking symmetric version
   * would quietly destroy energy the economy is careful to conserve. Excess
   * spreads out by diffusion instead, and stops growing until it is under
   * capacity again.
   *
   * `r` is per frame, already multiplied by dt by the caller — this pass has
   * no idea what a second is.
   */
  grow(ch: number, r: number, cap: number): void {
    if (!(r > 0) || !(cap > 0)) return;
    if (this.hiI < this.loI) return;
    const d = this.data;
    const { cols } = this;
    const { lo: sLo, hi: sHi } = this.spans();
    const invCap = 1 / cap;
    for (let j = this.loJ; j <= this.hiJ; j++) {
      const rowBase = j * cols * CHANNELS;
      const a0 = sLo[j] > this.loI ? sLo[j] : this.loI;
      const b0 = sHi[j] < this.hiI ? sHi[j] : this.hiI;
      for (let i = a0, k = rowBase + a0 * CHANNELS + ch; i <= b0; i++, k += CHANNELS) {
        const e = d[k];
        if (e <= 0 || e >= cap) continue;
        d[k] = e + r * e * (1 - e * invCap);
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
   * What it buys is the one thing four independent decaying blobs cannot do.
   * As it stands every channel is a hill around whoever is emitting, so what a
   * body smells is always *who is there* — the field carries information but
   * does not hold any of its own. A reaction puts local maxima where nobody is
   * standing, travelling fronts, and regions that have just been used up and
   * are briefly unusable. Signal comes apart from source, and "over there" can
   * mean something no emitter is saying.
   *
   * Two things it needs to work at all, both of which are the caller's job.
   * The species must diffuse at different rates — `diffuseRate` exists partly
   * for this, and Gray-Scott wants the substrate at roughly twice the
   * activator; equal rates have no instability to find and simply blur. And
   * `feed`/`kill` live in a thin sliver of their own plane, roughly F in
   * [0.01, 0.09] and k in [0.045, 0.07], with the interesting behaviour in a
   * fraction of that. Outside it the pattern is a uniform wash either way,
   * which is why this is off unless someone deliberately turns it on rather
   * than something with a plausible-looking default.
   *
   * `u` is normalised toward 1 by the feed term, so this expects a channel
   * whose natural scale is about 1 — `CH.energy` read against `cellCap`, or a
   * signal channel that is not also carrying deposits at peaks of ten. Handed
   * a raw signal channel it will not explode, because `soft`-free arithmetic
   * on bounded inputs stays bounded, but the pattern will sit outside its
   * regime and do nothing interesting.
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
   * open the box over it.
   *
   * How a world starts with ground in it. The sparse grid this replaces could
   * answer "ambient" for a cell nobody had touched; a field holds what it
   * holds, so the ground has to actually be put there — once, when the world
   * is pinned and the disk first exists.
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

  /**
   * The loudest thing anything is *saying* — the signal channels only.
   *
   * `CH.energy` is deliberately not in it. The ground sits at capacity across
   * the whole dish, so including it would make this a constant a bit under
   * `cellCap` no matter what the pond was doing, which is useless as a
   * normaliser for the scent overlay and wrong as an answer to "is anything
   * being emitted". Pass a channel to ask about one specifically, the ground
   * included.
   */
  /**
   * All four channels at one point, into `out[off .. off+4)`.
   *
   * `sample` four times over would redo the grid transform, the two floors and
   * the four corner indices each time, for readings that always come from the
   * same four cells. This is the same arithmetic once and four lerps after it.
   *
   * It exists because the state vector needs the channels *separately*. A body
   * has only ever had `trail` — the taste-weighted sum — which is one number,
   * so nothing could condition on which channel it was smelling, only on how
   * much it liked the mixture. Emitting on one channel because you smell
   * another was outside the language.
   */
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

  peak(ch?: number): number {
    let m = 0;
    const d = this.data;
    if (ch !== undefined) {
      for (let i = ch; i < d.length; i += CHANNELS) if (d[i] > m) m = d[i];
      return m;
    }
    for (let i = 0; i < d.length; i += CHANNELS) {
      for (const c of VOICE) if (d[i + c] > m) m = d[i + c];
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
