/*
 * How long a body spends alone before it joins anything.
 *
 * The question this answers is whether trophic dependency is survivable. A
 * fresh spawn carries `EXTRA_CAP / upkeep` seconds of tank — about 83 at the
 * defaults — and `yDirect = 0` makes a body that cannot reach a net's
 * generosity a body that dies. If the median time to first latch is well
 * under the tank, obligate dependency *structures* the soup; if it is not,
 * obligate dependency simply kills it, and the plan says to measure this
 * before `yDirect` moves far (`docs/history/energy-chemistry-plan.md` §5).
 *
 * The measure is a histogram rather than a list because a ten-minute pond
 * makes tens of thousands of bodies and only three numbers are ever read off
 * them. Bins are log-spaced, because the interesting range spans from a
 * newborn latching on the frame it is made to a drifter that never does, and
 * a linear bin fine enough for the first is absurd for the second.
 */

/** Bins per decade of seconds. 8 puts the p50 within about 9 per cent. */
const PER_DECADE = 8;
/**
 * 10 ms to 1000 s, five decades. Anything longer lands in the top bin, whose
 * centre reads about 870 s — far past the ~83 s tank the window is compared
 * against, so "longer than this" and "much longer than this" are the same
 * answer to the only question being asked.
 */
const MIN_S = 0.01;
const DECADES = 5;
const BINS = PER_DECADE * DECADES;

export interface LarvalReading {
  /** Bodies that latched at least once, and the seconds they took. */
  latched: number;
  p50: number | null;
  p90: number | null;
  /**
   * Bodies that died having never latched.
   *
   * A rewrite consumes its parents, but a rewrite happens between wired
   * bodies, so anything a rewrite takes was counted as latched long before —
   * this is starvation and collision, which is what the question is about.
   */
  died: number;
  /** `died / (died + latched)`: the share of arrivals that never joined anything. */
  loneliness: number | null;
}

export class LarvalWindow {
  private readonly bins = new Int32Array(BINS);
  private latched = 0;
  private died = 0;

  /** Called once per body, the first time it acquires a wire. */
  latchedAfter(seconds: number): void {
    this.latched++;
    const d = Math.log10(Math.max(MIN_S, seconds) / MIN_S) / DECADES;
    const i = Math.floor(d * BINS);
    this.bins[i < 0 ? 0 : i >= BINS ? BINS - 1 : i]++;
  }

  /** Called when a body dies having never latched. */
  diedAlone(): void {
    this.died++;
  }

  reset(): void {
    this.bins.fill(0);
    this.latched = 0;
    this.died = 0;
  }

  read(): LarvalReading {
    const total = this.latched + this.died;
    return {
      latched: this.latched,
      p50: this.quantile(0.5),
      p90: this.quantile(0.9),
      died: this.died,
      loneliness: total > 0 ? this.died / total : null,
    };
  }

  /*
   * Over the latched bodies only. A body that never latched has no
   * time-to-latch, and giving it one — the run's length, say — would make the
   * median a function of how long the run happened to be.
   */
  private quantile(q: number): number | null {
    if (this.latched === 0) return null;
    const want = q * this.latched;
    let seen = 0;
    for (let i = 0; i < BINS; i++) {
      seen += this.bins[i];
      if (seen >= want) {
        // The bin's geometric centre, which is its midpoint in log space.
        return MIN_S * 10 ** (((i + 0.5) / BINS) * DECADES);
      }
    }
    return null;
  }
}
