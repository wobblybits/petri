/*
 * Time from spawn to first latch, as a log-binned histogram. Measures whether
 * trophic dependency (`yDirect = 0`) is survivable: compare the median against
 * the starting tank, `EXTRA_CAP / upkeep` seconds.
 */

/** Bins per decade of seconds. 8 puts the p50 within about 9 per cent. */
const PER_DECADE = 8;
/** 10 ms to 1000 s, five decades; anything longer lands in the top bin. */
const MIN_S = 0.01;
const DECADES = 5;
const BINS = PER_DECADE * DECADES;

export interface LarvalReading {
  /** Bodies that latched at least once, and the seconds they took. */
  latched: number;
  p50: number | null;
  p90: number | null;
  /** Bodies that died having never latched: starvation and collision, since rewrites only take wired bodies. */
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

  /* Over latched bodies only; an unlatched body has no time-to-latch. */
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
