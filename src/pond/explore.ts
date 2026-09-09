/*
 * Reading a library of runs: what varies, what moves together, what regimes
 * the thing has, and which effects are conditional.
 *
 * Four passes, because no one of them answers the question on its own.
 *
 *   1. **Joint PCA** over standardised `[params | outcomes]`. Descriptive, and
 *      its virtue is that it looks at every outcome at once — the productivity
 *      confound in the ground sweeps went unnoticed for two rounds because
 *      only the outcomes expected to move were being read. A method that puts
 *      them all in one picture shows a total diverging without anyone having
 *      to suspect it.
 *   2. **PLS**, which is the tool for the job PCA is usually asked to do here.
 *      PCA maximises *total* variance and can spend its leading components on
 *      within-outcome structure that no parameter drives; PLS maximises the
 *      *cross-block* covariance, which is the actual question.
 *   3. **Clustering on the outcomes**, because ponds do not vary continuously.
 *      There is a sterile regime and a fertile one with a sharp transition,
 *      and "which region of parameter space gives which kind of pond" is a far
 *      more useful artefact than a best-parameters vector — the objectives
 *      trade off, so there is no single best.
 *   4. **Conditional effects**, because the other three are linear and additive
 *      and the interesting couplings here are none of those things.
 *      `senseScale` is not correlated with `excreteRate`; it *means* something
 *      different depending on it. A linear method reports such a parameter as
 *      weak and says nothing about the conditionality, which is exactly the
 *      wrong conclusion and exactly the mistake this pipeline exists to stop.
 *
 * No dependencies, so the arithmetic is here. It is tested against cases with
 * known answers, because an analysis tool that is subtly wrong does not fail —
 * it produces a plausible number and someone believes it.
 */

export interface Matrix {
  /** Row-major, `rows * cols`. */
  data: Float64Array;
  rows: number;
  cols: number;
  names: string[];
}

export function matrix(rows: number[][], names: string[]): Matrix {
  const r = rows.length;
  const c = names.length;
  const data = new Float64Array(r * c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) data[i * c + j] = rows[i][j];
  return { data, rows: r, cols: c, names };
}

/**
 * Centre and scale every column to unit variance, dropping the ones that do
 * not vary.
 *
 * Dropping matters more than it looks: a library holds 80 parameters of which
 * a handful were ever moved, and a constant column has no correlation with
 * anything — kept in, it contributes a zero row to every loading and an
 * eigenvalue of zero, and it makes the output long enough that the six things
 * that did vary are hard to find in it.
 */
/**
 * Whether a column carries a real spread, or only floating-point dust.
 *
 * Relative, because an absolute floor is wrong at both ends. A constant
 * column is rarely exactly constant: nine identical doubles summed and
 * divided by nine need not give the double back, and the residue scales with
 * the magnitude — dust on `maxAgents` at 100000 is around 1e-11, which any
 * fixed 1e-12 floor waves through as signal.
 *
 * This project has now produced three confident numbers from that residue:
 * an eta-squared of 1.0 for a metric defined as a constant, a "the dropped
 * runs differ" warning naming three parameters that had not moved, and a
 * standardised column of pure noise waiting to happen on the wide dials.
 */
export function varies(mean: number, sd: number): boolean {
  return sd > Math.max(Math.abs(mean), 1) * 1e-12;
}

export function standardise(m: Matrix): Matrix {
  const keep: number[] = [];
  const mean: number[] = [];
  const sd: number[] = [];
  for (let j = 0; j < m.cols; j++) {
    let s = 0;
    for (let i = 0; i < m.rows; i++) s += m.data[i * m.cols + j];
    const mu = s / m.rows;
    let v = 0;
    for (let i = 0; i < m.rows; i++) {
      const d = m.data[i * m.cols + j] - mu;
      v += d * d;
    }
    const sigma = Math.sqrt(v / m.rows);
    if (varies(mu, sigma)) {
      keep.push(j);
      mean.push(mu);
      sd.push(sigma);
    }
  }
  const out = new Float64Array(m.rows * keep.length);
  for (let i = 0; i < m.rows; i++) {
    for (let k = 0; k < keep.length; k++) {
      out[i * keep.length + k] = (m.data[i * m.cols + keep[k]] - mean[k]) / sd[k];
    }
  }
  return { data: out, rows: m.rows, cols: keep.length, names: keep.map((j) => m.names[j]) };
}

export interface Component {
  /** One loading per column of the input, in its order. */
  loading: number[];
  /** Share of the total variance this component accounts for. */
  explained: number;
}

/** NIPALS: one component at a time, deflating as it goes. */
export function pca(m: Matrix, want: number): Component[] {
  const X = Float64Array.from(m.data);
  const { rows, cols } = m;
  let total = 0;
  for (let i = 0; i < X.length; i++) total += X[i] * X[i];
  const out: Component[] = [];
  const t = new Float64Array(rows);
  const p = new Float64Array(cols);
  for (let k = 0; k < want && k < Math.min(rows, cols); k++) {
    // Start from the column with the most variance left in it.
    let best = 0;
    let bestVar = -1;
    for (let j = 0; j < cols; j++) {
      let v = 0;
      for (let i = 0; i < rows; i++) v += X[i * cols + j] * X[i * cols + j];
      if (v > bestVar) {
        bestVar = v;
        best = j;
      }
    }
    if (bestVar <= 1e-18) break;
    for (let i = 0; i < rows; i++) t[i] = X[i * cols + best];
    for (let iter = 0; iter < 400; iter++) {
      let tt = 0;
      for (let i = 0; i < rows; i++) tt += t[i] * t[i];
      if (tt <= 1e-18) break;
      for (let j = 0; j < cols; j++) {
        let s = 0;
        for (let i = 0; i < rows; i++) s += X[i * cols + j] * t[i];
        p[j] = s / tt;
      }
      let pn = 0;
      for (let j = 0; j < cols; j++) pn += p[j] * p[j];
      pn = Math.sqrt(pn);
      if (pn <= 1e-18) break;
      for (let j = 0; j < cols; j++) p[j] /= pn;
      let delta = 0;
      for (let i = 0; i < rows; i++) {
        let s = 0;
        for (let j = 0; j < cols; j++) s += X[i * cols + j] * p[j];
        delta += (s - t[i]) * (s - t[i]);
        t[i] = s;
      }
      if (delta < 1e-20) break;
    }
    let tt = 0;
    for (let i = 0; i < rows; i++) tt += t[i] * t[i];
    out.push({ loading: [...p], explained: total > 0 ? tt / total : 0 });
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) X[i * cols + j] -= t[i] * p[j];
  }
  return out;
}

export interface CrossComponent {
  /** Loading on the parameter block, in `x.names` order. */
  xLoading: number[];
  /** Loading on the outcome block, in `y.names` order. */
  yLoading: number[];
  /** Correlation between the two blocks' scores on this component. */
  correlation: number;
}

/**
 * PLS2, which finds the directions in each block that covary most with the
 * other — the thing joint PCA is usually being asked for and does not
 * optimise. Both blocks must already be standardised.
 */
export function pls(x: Matrix, y: Matrix, want: number): CrossComponent[] {
  const X = Float64Array.from(x.data);
  const Y = Float64Array.from(y.data);
  const n = x.rows;
  const out: CrossComponent[] = [];
  const w = new Float64Array(x.cols);
  const c = new Float64Array(y.cols);
  const t = new Float64Array(n);
  const u = new Float64Array(n);
  for (let k = 0; k < want; k++) {
    for (let i = 0; i < n; i++) u[i] = Y[i * y.cols];
    let ok = false;
    for (let iter = 0; iter < 400; iter++) {
      let uu = 0;
      for (let i = 0; i < n; i++) uu += u[i] * u[i];
      if (uu <= 1e-18) break;
      for (let j = 0; j < x.cols; j++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += X[i * x.cols + j] * u[i];
        w[j] = s / uu;
      }
      let wn = 0;
      for (let j = 0; j < x.cols; j++) wn += w[j] * w[j];
      wn = Math.sqrt(wn);
      if (wn <= 1e-18) break;
      for (let j = 0; j < x.cols; j++) w[j] /= wn;
      let delta = 0;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < x.cols; j++) s += X[i * x.cols + j] * w[j];
        delta += (s - t[i]) * (s - t[i]);
        t[i] = s;
      }
      let tt = 0;
      for (let i = 0; i < n; i++) tt += t[i] * t[i];
      if (tt <= 1e-18) break;
      for (let j = 0; j < y.cols; j++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += Y[i * y.cols + j] * t[i];
        c[j] = s / tt;
      }
      let cn = 0;
      for (let j = 0; j < y.cols; j++) cn += c[j] * c[j];
      cn = Math.sqrt(cn);
      if (cn <= 1e-18) break;
      for (let j = 0; j < y.cols; j++) c[j] /= cn;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < y.cols; j++) s += Y[i * y.cols + j] * c[j];
        u[i] = s;
      }
      ok = true;
      if (delta < 1e-20 && iter > 1) break;
    }
    if (!ok) break;
    let tt = 0;
    let uu = 0;
    let tu = 0;
    for (let i = 0; i < n; i++) {
      tt += t[i] * t[i];
      uu += u[i] * u[i];
      tu += t[i] * u[i];
    }
    if (tt <= 1e-18) break;
    out.push({
      xLoading: [...w],
      yLoading: [...c],
      correlation: tt > 0 && uu > 0 ? tu / Math.sqrt(tt * uu) : 0,
    });
    // Deflate both blocks by the shared score.
    for (let j = 0; j < x.cols; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i * x.cols + j] * t[i];
      const pj = s / tt;
      for (let i = 0; i < n; i++) X[i * x.cols + j] -= t[i] * pj;
    }
    for (let j = 0; j < y.cols; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Y[i * y.cols + j] * t[i];
      const qj = s / tt;
      for (let i = 0; i < n; i++) Y[i * y.cols + j] -= t[i] * qj;
    }
  }
  return out;
}

export interface Clustering {
  /** Cluster index per row. */
  label: number[];
  /** Cluster centres, in the input's column space. */
  centre: number[][];
  /** Within-cluster sum of squares; lower is tighter. */
  inertia: number;
}

/** k-means with k-means++ starts, best of `restarts`, from a seeded stream. */
export function kmeans(m: Matrix, k: number, seed = 1, restarts = 8): Clustering {
  let s = seed >>> 0;
  const rnd = (): number => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const dist2 = (row: number, centre: number[]): number => {
    let d = 0;
    for (let j = 0; j < m.cols; j++) {
      const x = m.data[row * m.cols + j] - centre[j];
      d += x * x;
    }
    return d;
  };
  let best: Clustering | null = null;
  for (let attempt = 0; attempt < restarts; attempt++) {
    const centre: number[][] = [];
    const first = Math.floor(rnd() * m.rows);
    centre.push([...m.data.subarray(first * m.cols, (first + 1) * m.cols)]);
    while (centre.length < k) {
      const d = new Float64Array(m.rows);
      let total = 0;
      for (let i = 0; i < m.rows; i++) {
        let nearest = Infinity;
        for (const c of centre) nearest = Math.min(nearest, dist2(i, c));
        d[i] = nearest;
        total += nearest;
      }
      let pick = m.rows - 1;
      let target = rnd() * total;
      for (let i = 0; i < m.rows; i++) {
        target -= d[i];
        if (target <= 0) {
          pick = i;
          break;
        }
      }
      centre.push([...m.data.subarray(pick * m.cols, (pick + 1) * m.cols)]);
    }
    const label = new Array<number>(m.rows).fill(0);
    for (let iter = 0; iter < 100; iter++) {
      let moved = false;
      for (let i = 0; i < m.rows; i++) {
        let bestK = 0;
        let bestD = Infinity;
        for (let c = 0; c < k; c++) {
          const d = dist2(i, centre[c]);
          if (d < bestD) {
            bestD = d;
            bestK = c;
          }
        }
        if (label[i] !== bestK) moved = true;
        label[i] = bestK;
      }
      for (let c = 0; c < k; c++) {
        const sum = new Float64Array(m.cols);
        let n = 0;
        for (let i = 0; i < m.rows; i++) {
          if (label[i] !== c) continue;
          n++;
          for (let j = 0; j < m.cols; j++) sum[j] += m.data[i * m.cols + j];
        }
        if (n > 0) for (let j = 0; j < m.cols; j++) centre[c][j] = sum[j] / n;
      }
      if (!moved) break;
    }
    let inertia = 0;
    for (let i = 0; i < m.rows; i++) inertia += dist2(i, centre[label[i]]);
    if (!best || inertia < best.inertia) best = { label, centre: centre.map((c) => [...c]), inertia };
  }
  return best!;
}

/** Pearson correlation of two equal-length columns. */
export function correlate(a: number[], b: number[]): number {
  const n = a.length;
  if (n < 2) return 0;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}

export interface Conditional {
  driver: string;
  condition: string;
  outcome: string;
  /** Correlation of driver with outcome, below and above the condition's median. */
  low: number;
  high: number;
  /** How much the effect changes across the split. The thing being screened for. */
  swing: number;
  nLow: number;
  nHigh: number;
}

/**
 * Where does one parameter's effect *depend* on another's setting?
 *
 * The screen for the failure mode a linear model cannot express: an effect
 * that exists in half the parameter space and not the other half. Split the
 * runs at the median of the condition, correlate the driver with the outcome
 * within each half, and report the difference. A large swing is an
 * interaction, and it is exactly the shape of `senseScale` mattering only
 * where `excreteRate` has switched the minting off.
 *
 * A median split rather than a fitted product term because it needs no
 * assumption about the shape of the dependence — a threshold, a switch and a
 * smooth ramp all show up — and because with per-run noise this large, a
 * two-bin estimate is about as much resolution as the data can carry.
 */
export function conditionalEffects(
  params: Matrix,
  outcomes: Matrix,
  minPerHalf = 8,
): Conditional[] {
  const col = (m: Matrix, j: number): number[] => {
    const out: number[] = [];
    for (let i = 0; i < m.rows; i++) out.push(m.data[i * m.cols + j]);
    return out;
  };
  const median = (xs: number[]): number => {
    const s = [...xs].sort((a, b) => a - b);
    return s[s.length >> 1];
  };
  const found: Conditional[] = [];
  for (let d = 0; d < params.cols; d++) {
    const drv = col(params, d);
    for (let c = 0; c < params.cols; c++) {
      if (c === d) continue;
      const cond = col(params, c);
      const mid = median(cond);
      const lowRows: number[] = [];
      const highRows: number[] = [];
      for (let i = 0; i < params.rows; i++) (cond[i] < mid ? lowRows : highRows).push(i);
      if (lowRows.length < minPerHalf || highRows.length < minPerHalf) continue;
      for (let o = 0; o < outcomes.cols; o++) {
        const out = col(outcomes, o);
        const lo = correlate(lowRows.map((i) => drv[i]), lowRows.map((i) => out[i]));
        const hi = correlate(highRows.map((i) => drv[i]), highRows.map((i) => out[i]));
        found.push({
          driver: params.names[d],
          condition: params.names[c],
          outcome: outcomes.names[o],
          low: lo,
          high: hi,
          swing: Math.abs(hi - lo),
          nLow: lowRows.length,
          nHigh: highRows.length,
        });
      }
    }
  }
  return found.sort((a, b) => b.swing - a.swing);
}
