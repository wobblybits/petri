/**
 * Uniform spatial grid for broad-phase pair queries.
 *
 * The all-pairs loops this replaces were the whole scaling story: at 382 agents
 * the contact solver tested 72,771 pairs to find 21 actual contacts, eight times
 * a frame. Bucketing by cell and only visiting a cell and its forward neighbours
 * yields every close pair exactly once, and skips the 99.9% that are nowhere
 * near each other.
 *
 * Counting-sort layout, so a rebuild allocates nothing once the arrays are big
 * enough — which matters because contacts rebuild it every substep.
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

  /**
   * Bucket `count` points. `cellSize` should be at least the largest interaction
   * radius, so every interacting pair lands in the same cell or an adjacent one.
   */
  build(xs: readonly number[], ys: readonly number[], count: number, cellSize: number): void {
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

    // NaN/Inf bounds make cols*rows stay non-finite, so the coarsen loop
    // never exits and the tab freezes.
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

    // A scattered swarm could otherwise ask for billions of empty buckets.
    // Prefer more cells (tighter buckets) until the cap: coarsening toward
    // all-pairs is how a spread of 400 became as expensive as 400 stacked.
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

  /**
   * Visit every pair of points sharing a cell or lying in adjacent cells, once
   * each. Callers still do their own distance test — this only narrows the field.
   */
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
