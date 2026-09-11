/**
 * Uniform spatial grid for broad-phase pair queries: bucket by cell, visit a cell
 * and its forward neighbours, so every close pair is yielded exactly once.
 * Counting-sort layout: a rebuild allocates nothing once the arrays are big enough.
 */
export class PairGrid {
  private cols = 1;
  private rows = 1;
  private originX = 0;
  private originY = 0;
  private inv = 1;
  private count = 0;
  private cellOf = new Int32Array(0);
  private start = new Int32Array(0);
  private order = new Int32Array(0);

  /** Bucket `count` points. `cellSize` must be at least the largest interaction radius. */
  build(xs: ArrayLike<number>, ys: ArrayLike<number>, count: number, cellSize: number): void {
    this.count = count;
    if (count === 0) return;
    if (this.cellOf.length < count) {
      this.cellOf = new Int32Array(count * 2);
      this.order = new Int32Array(count * 2);
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < count; i++) {
      const x = xs[i];
      const y = ys[i];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }

    // Non-finite bounds would keep the coarsen loop below from ever exiting.
    if (
      !Number.isFinite(minX) ||
      !Number.isFinite(minY) ||
      !Number.isFinite(maxX) ||
      !Number.isFinite(maxY)
    ) {
      minX = 0;
      minY = 0;
      maxX = 0;
      maxY = 0;
    }

    // Cap the bucket count, preferring more cells (tighter buckets) up to the cap.
    let cell = Math.max(1e-3, Number.isFinite(cellSize) ? cellSize : 1);
    const maxCells = Math.min(65536, Math.max(256, count * 32));
    let cols = Math.floor((maxX - minX) / cell) + 1;
    let rows = Math.floor((maxY - minY) / cell) + 1;
    let guard = 0;
    while (cols * rows > maxCells && guard++ < 64) {
      cell *= 2;
      if (!Number.isFinite(cell) || cell <= 0) break;
      cols = Math.floor((maxX - minX) / cell) + 1;
      rows = Math.floor((maxY - minY) / cell) + 1;
    }
    if (!Number.isFinite(cols) || cols < 1) cols = 1;
    if (!Number.isFinite(rows) || rows < 1) rows = 1;

    this.cols = cols;
    this.rows = rows;
    this.originX = minX;
    this.originY = minY;
    this.inv = 1 / cell;

    const cells = cols * rows;
    if (this.start.length < cells + 1) this.start = new Int32Array((cells + 1) * 2);
    const start = this.start;
    start.fill(0, 0, cells + 1);

    for (let i = 0; i < count; i++) {
      const cx = ((xs[i] - minX) * this.inv) | 0;
      const cy = ((ys[i] - minY) * this.inv) | 0;
      const c = cy * cols + cx;
      this.cellOf[i] = c;
      start[c + 1]++;
    }
    for (let c = 0; c < cells; c++) start[c + 1] += start[c];

    // start[] is consumed as a cursor here, then rebuilt by the shift below.
    for (let i = 0; i < count; i++) this.order[start[this.cellOf[i]]++] = i;
    for (let c = cells; c > 0; c--) start[c] = start[c - 1];
    start[0] = 0;
  }

  /** Visit every pair sharing a cell or in adjacent cells, once each; callers still distance-test. */
  forEachPair(fn: (i: number, j: number) => void): void {
    if (this.count === 0) return;
    const { cols, rows, start, order } = this;
    for (let cy = 0; cy < rows; cy++) {
      for (let cx = 0; cx < cols; cx++) {
        const c = cy * cols + cx;
        const a0 = start[c];
        const a1 = start[c + 1];
        if (a0 === a1) continue;

        for (let a = a0; a < a1; a++) {
          for (let b = a + 1; b < a1; b++) fn(order[a], order[b]);
        }

        // Forward half of the neighbourhood only, so no pair is visited twice.
        for (let k = 0; k < 4; k++) {
          const nx = cx + (k === 0 ? 1 : k === 1 ? -1 : k === 2 ? 0 : 1);
          const ny = cy + (k === 0 ? 0 : 1);
          if (nx < 0 || nx >= cols || ny >= rows) continue;
          const n = ny * cols + nx;
          const b0 = start[n];
          const b1 = start[n + 1];
          for (let a = a0; a < a1; a++) {
            for (let b = b0; b < b1; b++) fn(order[a], order[b]);
          }
        }
      }
    }
  }

  /** Visit every point within `radius` of (x, y). */
  forEachNear(x: number, y: number, radius: number, fn: (i: number) => void): void {
    if (this.count === 0) return;
    const { cols, rows, start, order } = this;
    const lo = Math.max(0, ((x - radius - this.originX) * this.inv) | 0);
    const hi = Math.min(cols - 1, ((x + radius - this.originX) * this.inv) | 0);
    const bo = Math.max(0, ((y - radius - this.originY) * this.inv) | 0);
    const bi = Math.min(rows - 1, ((y + radius - this.originY) * this.inv) | 0);
    for (let cy = bo; cy <= bi; cy++) {
      for (let cx = lo; cx <= hi; cx++) {
        const c = cy * cols + cx;
        for (let a = start[c]; a < start[c + 1]; a++) fn(order[a]);
      }
    }
  }
}

/**
 * Uniform grid over axis-aligned boxes, for "what is near this segment". Each box
 * goes into every cell it overlaps; a query yields each entry once however many
 * cells it spans. Counting-sort layout, so a steady state allocates nothing.
 */
export class BoxGrid {
  private cols = 1;
  private rows = 1;
  private originX = 0;
  private originY = 0;
  private inv = 1;
  private count = 0;
  private start = new Int32Array(0);
  private items = new Int32Array(0);
  private cellLo = new Int32Array(0);
  private cellHi = new Int32Array(0);
  /** Entries too sprawling to bin, tested by every query instead. */
  private oversized: number[] = [];
  private seen = new Int32Array(0);
  private stamp = 0;

  /** Cells a single box may occupy before it is treated as oversized. */
  private static readonly SPAN_CAP = 64;

  build(
    minX: Float64Array,
    minY: Float64Array,
    maxX: Float64Array,
    maxY: Float64Array,
    count: number,
  ): void {
    this.count = count;
    this.oversized.length = 0;
    if (count === 0) return;
    if (this.cellLo.length < count * 2) {
      this.cellLo = new Int32Array(count * 4);
      this.cellHi = new Int32Array(count * 4);
      this.seen = new Int32Array(count * 2);
      this.stamp = 0;
    }

    let lox = Infinity;
    let loy = Infinity;
    let hix = -Infinity;
    let hiy = -Infinity;
    let extent = 0;
    for (let i = 0; i < count; i++) {
      if (minX[i] < lox) lox = minX[i];
      if (minY[i] < loy) loy = minY[i];
      if (maxX[i] > hix) hix = maxX[i];
      if (maxY[i] > hiy) hiy = maxY[i];
      extent += maxX[i] - minX[i] + (maxY[i] - minY[i]);
    }
    if (!Number.isFinite(lox) || !Number.isFinite(loy) || !Number.isFinite(hix) || !Number.isFinite(hiy)) {
      lox = 0;
      loy = 0;
      hix = 0;
      hiy = 0;
      extent = 0;
    }

    // Cell to the mean box, not the largest, so one stretched wire cannot coarsen the whole grid.
    let cell = Math.max(1e-3, extent / Math.max(1, count * 2));
    const maxCells = Math.min(65536, Math.max(256, count * 8));
    let cols = Math.floor((hix - lox) / cell) + 1;
    let rows = Math.floor((hiy - loy) / cell) + 1;
    let guard = 0;
    while (cols * rows > maxCells && guard++ < 64) {
      cell *= 2;
      cols = Math.floor((hix - lox) / cell) + 1;
      rows = Math.floor((hiy - loy) / cell) + 1;
    }
    cols = Math.max(1, Math.min(cols, maxCells));
    rows = Math.max(1, Math.min(rows, Math.max(1, Math.floor(maxCells / cols))));

    this.cols = cols;
    this.rows = rows;
    this.originX = lox;
    this.originY = loy;
    this.inv = 1 / cell;
    const cells = cols * rows;
    if (this.start.length < cells + 1) this.start = new Int32Array((cells + 1) * 2);
    const start = this.start;
    start.fill(0, 0, cells + 1);

    // Pass one: each box's cell span, and how many entries each cell will hold.
    let total = 0;
    for (let i = 0; i < count; i++) {
      const x0 = this.clampCol(minX[i]);
      const x1 = this.clampCol(maxX[i]);
      const y0 = this.clampRow(minY[i]);
      const y1 = this.clampRow(maxY[i]);
      const span = (x1 - x0 + 1) * (y1 - y0 + 1);
      if (span > BoxGrid.SPAN_CAP) {
        this.cellLo[i * 2] = -1;
        this.oversized.push(i);
        continue;
      }
      this.cellLo[i * 2] = x0;
      this.cellLo[i * 2 + 1] = y0;
      this.cellHi[i * 2] = x1;
      this.cellHi[i * 2 + 1] = y1;
      for (let cy = y0; cy <= y1; cy++) {
        for (let cx = x0; cx <= x1; cx++) start[cy * cols + cx + 1]++;
      }
      total += span;
    }
    for (let c = 0; c < cells; c++) start[c + 1] += start[c];
    if (this.items.length < total) this.items = new Int32Array(Math.max(16, total * 2));

    // Pass two: fill, consuming start[] as a cursor, then shift it back.
    const items = this.items;
    for (let i = 0; i < count; i++) {
      if (this.cellLo[i * 2] < 0) continue;
      const x0 = this.cellLo[i * 2];
      const y0 = this.cellLo[i * 2 + 1];
      const x1 = this.cellHi[i * 2];
      const y1 = this.cellHi[i * 2 + 1];
      for (let cy = y0; cy <= y1; cy++) {
        for (let cx = x0; cx <= x1; cx++) items[start[cy * cols + cx]++] = i;
      }
    }
    for (let c = cells; c > 0; c--) start[c] = start[c - 1];
    start[0] = 0;
  }

  private clampCol(x: number): number {
    const c = ((x - this.originX) * this.inv) | 0;
    return c < 0 ? 0 : c >= this.cols ? this.cols - 1 : c;
  }

  private clampRow(y: number): number {
    const r = ((y - this.originY) * this.inv) | 0;
    return r < 0 ? 0 : r >= this.rows ? this.rows - 1 : r;
  }

  /** Visit every entry whose box may overlap this one, each at most once. */
  forEachNear(
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    fn: (i: number) => void,
  ): void {
    if (this.count === 0) return;
    for (let k = 0; k < this.oversized.length; k++) fn(this.oversized[k]);
    if (!Number.isFinite(minX) || !Number.isFinite(minY)) return;
    if (!Number.isFinite(maxX) || !Number.isFinite(maxY)) return;
    const { cols, start, items, seen } = this;
    const stamp = ++this.stamp;
    const x0 = this.clampCol(minX);
    const x1 = this.clampCol(maxX);
    const y0 = this.clampRow(minY);
    const y1 = this.clampRow(maxY);
    for (let cy = y0; cy <= y1; cy++) {
      for (let cx = x0; cx <= x1; cx++) {
        const c = cy * cols + cx;
        for (let a = start[c]; a < start[c + 1]; a++) {
          const i = items[a];
          // A box in four cells is one entry, and the caller sees it once.
          if (seen[i] === stamp) continue;
          seen[i] = stamp;
          fn(i);
        }
      }
    }
  }
}
