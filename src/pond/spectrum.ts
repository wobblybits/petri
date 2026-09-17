import type { Params } from '../params.ts';

/**
 * The metabolism, answered in closed form instead of by running a pond.
 *
 * Two questions this pond keeps asking, and both have algebra rather than a
 * sweep behind them:
 *
 *   *Does one body oscillate, and at what period?* A Hopf bifurcation is a
 *   statement about eigenvalues. `A`'s row decouples — it is fed from outside
 *   and only feeds `B` — so the dynamics are the 3x3 over (B, C, D), a
 *   three-stage negative feedback carrying a positive self-loop on C, and
 *   Routh-Hurwitz on its characteristic polynomial answers it exactly. See
 *   `reactorBands`.
 *
 *   *Does a chain of them carry a wave?* Not an impulse question. A relay is
 *   driven at its pacemaker's frequency, so what decides whether the wave
 *   sustains is the catalyst channel's **transfer function** at that
 *   frequency, times the coupling. See `perHopGain`, and `relayBand` for the
 *   fuel window it puts a body in.
 *
 * And one about a net rather than a body: because a body broadcasts out of
 * its **principal** port only, and a body has one principal, the signalling
 * graph is *functional* — out-degree at most one. Its adjacency is therefore
 * a permutation on its cycles and nilpotent everywhere else, so the directed
 * Laplacian's spectrum is exactly the roots of unity around each cycle plus
 * zeros. **That needs no eigensolver: finding the cycles is the whole
 * calculation**, and `netSpectrum` does it in one pass.
 *
 * Every number here is a per-*reactor* unit unless it says otherwise. The
 * reactor advances by `h = metabolicRate * dt` while the coupling moves by
 * `k = metabolicDiffuse * dt` (`Sim.advanceGait`) — two clocks — so the
 * coupling strength the chemistry actually feels is
 * `metabolicDiffuse / metabolicRate`, which is what `epsilon` returns and why
 * the two dials cannot be changed independently.
 */

/** The coupling rate in reactor time. See the note on two clocks above. */
export function epsilon(p: Params): number {
  return p.metabolicRate > 0 ? p.metabolicDiffuse / p.metabolicRate : 0;
}

/** The reactor's fixed point, parameterised by its catalyst level. */
interface FixedPoint {
  /** Active primer. */
  B: number;
  /** Saturated catalyst. */
  C: number;
  /** Reset inhibitor. */
  D: number;
  /** The influx of B that holds this point — the one thing a lineage moves. */
  j: number;
}

/**
 * The fixed point is easiest to walk backwards: pick `C`, and `B`, `D` and the
 * influx that sustains them follow without a solve. `dC/dt = 0` gives B,
 * `dD/dt = 0` gives D, and `dB/dt = 0` gives the influx.
 */
function fixedPoint(p: Params, C: number): FixedPoint {
  const d = p.metabolicDecay;
  const B = ((p.metabolicReset + d) * C * (1 + p.metabolicSigma * C)) / (p.metabolicCat * (p.metabolicBase + C));
  const D = (p.metabolicReset * C) / d;
  return { B, C, D, j: p.metabolicQuench * B * D + d * B };
}

/**
 * The characteristic polynomial's coefficients at a fixed point, in the
 * spec's own letters: `(x+p)(x+q)(x+r) + L`.
 *
 * `q` is C's *net* removal — its own decay less what the autocatalysis gives
 * back — and it is the only one that can approach zero, which is why it is
 * the term the Hopf condition turns on.
 */
export function loop(par: Params, fp: FixedPoint): { p: number; q: number; r: number; L: number } {
  const d = par.metabolicDecay;
  const den = 1 + par.metabolicSigma * fp.C;
  const dr2dB = (par.metabolicCat * (par.metabolicBase + fp.C)) / den;
  const dr2dC = (par.metabolicCat * fp.B * (den - (par.metabolicBase + fp.C) * par.metabolicSigma)) / (den * den);
  return {
    p: par.metabolicQuench * fp.D + d,
    q: par.metabolicReset + d - dr2dC,
    r: d,
    L: par.metabolicReset * dr2dB * par.metabolicQuench * fp.B,
  };
}

/**
 * Routh-Hurwitz: a complex pair crosses into the right half plane when
 * `L > (p+q+r)(pq+pr+qr) - pqr`. Positive means oscillating.
 */
function hopfMargin(par: Params, C: number): number {
  const { p, q, r, L } = loop(par, fixedPoint(par, C));
  return L - ((p + q + r) * (p * q + p * r + q * r) - p * q * r);
}

function bisect(f: (x: number) => number, lo: number, hi: number, steps = 200): number {
  let a = lo;
  let b = hi;
  const fa = f(a) > 0;
  for (let i = 0; i < steps; i++) {
    const mid = 0.5 * (a + b);
    if (f(mid) > 0 === fa) a = mid;
    else b = mid;
  }
  return 0.5 * (a + b);
}

interface Bands {
  /** Influx below which the reactor rests quiet, and above which it cycles. */
  hopfLow: number;
  /** Influx above which it saturates and goes still again. */
  hopfHigh: number;
  /**
   * The spec's excitability line, `k2*B > k3+d`. On the quiet branch `B = j/d`,
   * so it is an influx too — kept because the specification states it, and see
   * `relayBand` for why it is not the line that decides a relay.
   */
  excite: number;
  /** The pacemaker's frequency at onset, `sqrt(pq+pr+qr)`, in reactor units. */
  omega: number;
  /** That period in wall seconds, which is what the eye sees. */
  periodSeconds: number;
}

/**
 * The fuel window, both edges, and the period at the lower one.
 *
 * "A starving body does not undulate" is not a rule anybody wrote: it is
 * where the lower boundary is. So is "a full body strokes faster" — the
 * period shortens across the window.
 */
export function reactorBands(par: Params): Bands {
  const d = par.metabolicDecay;
  const lo = bisect((C) => hopfMargin(par, C), 0.02, 0.4);
  const hi = bisect((C) => hopfMargin(par, C), 1.5, 8);
  const { p, q, r } = loop(par, fixedPoint(par, lo));
  const omega = Math.sqrt(p * q + p * r + q * r);
  return {
    hopfLow: fixedPoint(par, lo).j,
    hopfHigh: fixedPoint(par, hi).j,
    excite: (d * (par.metabolicReset + d)) / par.metabolicCat,
    omega,
    periodSeconds: par.metabolicRate > 0 ? (2 * Math.PI) / (omega * par.metabolicRate) : Infinity,
  };
}

/** The influx that holds a body at a given catalyst level, inverted. */
function influxTo(par: Params, j: number): number {
  return bisect((C) => fixedPoint(par, C).j - j, 1e-9, 60);
}

/**
 * **The number that decides whether a chain carries a wave.**
 *
 * Eliminating B and D from `(sI - J)` leaves the catalyst channel's transfer
 * function — the specification's own characteristic polynomial with the C
 * cofactor on top:
 *
 *     G(s) = (s+p)(s+r) / [ (s+p)(s+q)(s+r) + L ]
 *
 * A relay is driven at its pacemaker's frequency, and a hop multiplies the
 * catalyst by `epsilon * |G(jw)|`. At or above 1 the wave sustains; below it
 * the wave dies within `1/log10` hops however sharp the chemistry looks.
 *
 * This is why an impulse is the wrong instrument. The impulse response of
 * this reactor peaks at t = 0 and decays at *every* stable fixed point — C's
 * only positive feedback is its own diagonal entry, and the loop leaving it,
 * `C -> D -| B -> C`, is negative — so a pulse says "no amplification" and a
 * standing drive at the right frequency says 26x. A body just under its Hopf
 * boundary is a high-Q resonator, not a dead cable.
 */
export function perHopGain(par: Params, j: number): number {
  const fp = fixedPoint(par, influxTo(par, j));
  const { p, q, r, L } = loop(par, fp);
  const w = reactorBands(par).omega;
  const numRe = p * r - w * w;
  const numIm = w * (p + r);
  const denRe = L + p * q * r - (p + q + r) * w * w;
  const denIm = w * (p * q + p * r + q * r) - w * w * w;
  return (epsilon(par) * Math.hypot(numRe, numIm)) / Math.hypot(denRe, denIm);
}

/**
 * The band a **relay** has to sit in: fed enough that a hop does not lose the
 * wave, not so fed that it paces on its own.
 *
 * The specification asks for `(E)` satisfied and the Hopf condition not. `(E)`
 * is the wrong lower edge — at the shipped constants a body sitting exactly on
 * it has a per-hop gain of 0.58, so it attenuates — and the right one is
 * `perHopGain >= 1`. Both are reported so the difference stays visible.
 */
export function relayBand(par: Params): { low: number; high: number; atExcite: number } {
  const b = reactorBands(par);
  const low = bisect((j) => perHopGain(par, j) - 1, 0.02, b.hopfLow * (1 - 1e-9));
  return { low, high: b.hopfLow, atExcite: perHopGain(par, b.excite) };
}

/** A body, as this analysis needs it: its kind, and where its principal points. */
export interface Node {
  kind: 'con' | 'dup' | 'era';
  /** Index of the body this one's principal port is wired to, or -1. */
  out: number;
}

interface NetSpectrum {
  bodies: number;
  /** Bodies whose principal is wired, so the out-degree of the signalling graph. */
  speaking: number;
  /** One entry per directed cycle, its length. */
  cycles: number[];
  /**
   * The directed Laplacian's eigenvalues, exactly. A functional graph's
   * adjacency is a permutation on its cycles and nilpotent elsewhere, so
   * `D_out - A` has `1 - e^{2*pi*i*m/len}` around each cycle and 1 or 0 for
   * everything else. **A travelling eigenmode needs a complex one**, so a
   * cycle of length 3 or more is the only thing that can supply it.
   */
  laplacianComplex: number;
  /** Hop distance from the nearest Era along principal edges; -1 for none. */
  hops: number[];
}

/**
 * The signalling graph, its cycles, and how far an Era's primer reaches.
 *
 * The primer reach is the half of this that decides Era-only intake. An Era
 * broadcasts A and B, a Con broadcasts C and a Dup broadcasts D, so **the
 * primer stops one hop out**: a body at hop 2 has `B = 0`, and with no primer
 * the autocatalytic step makes nothing, so its transfer function collapses to
 * `1/(s + k3 + d)` and it attenuates whatever it is sent. Count the bodies at
 * hop 2 and beyond before believing a net will light up.
 */
export function netSpectrum(nodes: readonly Node[]): NetSpectrum {
  const n = nodes.length;
  const colour = new Int8Array(n);
  const cycles: number[] = [];
  for (let s = 0; s < n; s++) {
    if (colour[s] !== 0) continue;
    const path: number[] = [];
    const at = new Map<number, number>();
    let v = s;
    while (v >= 0 && colour[v] === 0) {
      colour[v] = 1;
      at.set(v, path.length);
      path.push(v);
      v = nodes[v].out;
    }
    if (v >= 0 && colour[v] === 1) cycles.push(path.length - (at.get(v) ?? 0));
    for (const u of path) colour[u] = 2;
  }
  // Roots of unity around a cycle are real only for lengths 1 and 2.
  let laplacianComplex = 0;
  for (const len of cycles) if (len > 2) laplacianComplex += len - (len % 2 === 0 ? 2 : 1);

  const hops = new Array<number>(n).fill(-1);
  const queue: number[] = [];
  for (let i = 0; i < n; i++)
    if (nodes[i].kind === 'era') {
      hops[i] = 0;
      queue.push(i);
    }
  for (let head = 0; head < queue.length; head++) {
    const i = queue[head];
    const k = nodes[i].out;
    if (k >= 0 && hops[k] < 0) {
      hops[k] = hops[i] + 1;
      queue.push(k);
    }
  }
  return {
    bodies: n,
    speaking: nodes.reduce((a, b) => a + (b.out >= 0 ? 1 : 0), 0),
    cycles,
    laplacianComplex,
    hops,
  };
}
