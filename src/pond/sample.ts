import { SLIDERS, type Params } from '../params.ts';

/*
 * Quasi-random parameter sampling, for asking about many dials at once.
 *
 * A grid over two axes at five seeds spends thirty runs learning about two
 * dimensions. Thirty samples over ten axes learn about ten — Bergstra and
 * Bengio's argument, whose precondition is that only a few dimensions matter,
 * which is what this pond keeps demonstrating: `excreteRate` explained 68 to
 * 93 per cent of the variance in a sweep where `uptakeVmax` explained two to
 * eight and was unresolved.
 *
 * The decisive practical difference is that samples **compose**. A grid point
 * informs its own grid and nothing else; a random draw joins every regression
 * ever run over the library. Which is the whole reason the runs live in one
 * database rather than in a file per experiment.
 *
 * Ranges come from `SLIDERS`, which already declares them for 73 of the 80
 * parameters — with the caveat the `learnDiscount` episode taught: a slider
 * range is an *assumption*, and a search bounded by one inherits it. Its
 * maximum of 0.995 is a 3.3-second horizon, which measurement says is too
 * short for the pond to learn anything about foraging. Override the bounds
 * when the question is whether the assumption holds.
 */

export interface SampleAxis {
  key: keyof Params;
  min: number;
  max: number;
  /**
   * Sampled log-uniform rather than uniform.
   *
   * A rate spanning `0.001` to `0.5` sampled uniformly puts ninety per cent of
   * its draws in the top decade, which is the wrong half to spend a budget on
   * when the interesting behaviour is a transition somewhere below. Applied
   * where a range spans at least a factor of ten and its floor is above zero;
   * a floor at zero is usually a switch rather than a scale, and there are 34
   * of those.
   */
  log: boolean;
  /**
   * Snap draws to this multiple.
   *
   * Set from the slider when it declares a step of one or more, which is how
   * this codebase says "a count": `groundPatches` is a number of patches and
   * the sim floors it, so an unsnapped draw of 5.2 stores a parameter value
   * that never ran. Finer steps are left alone — they are display precision,
   * not quantisation, and rounding to them only throws away coverage.
   */
  step?: number;
}

/** `k`, or `k=lo..hi` to override the declared range. */
export function parseAxis(spec: string): SampleAxis {
  const eq = spec.indexOf('=');
  const key = (eq < 0 ? spec : spec.slice(0, eq)).trim() as keyof Params;
  const declared = SLIDERS.find((s) => s.key === key);
  let min: number;
  let max: number;
  if (eq < 0) {
    if (!declared) {
      throw new Error(`pond: ${String(key)} has no declared range; give one as ${String(key)}=lo..hi`);
    }
    min = declared.min;
    max = declared.max;
  } else {
    const m = /^(-?[\d.eE+-]+)\.\.(-?[\d.eE+-]+)$/.exec(spec.slice(eq + 1).trim());
    if (!m) throw new Error(`pond: --sample wants key or key=lo..hi, got ${JSON.stringify(spec)}`);
    min = Number(m[1]);
    max = Number(m[2]);
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || !(max > min)) {
    throw new Error(`pond: --sample ${String(key)} has an empty or backwards range`);
  }
  const step = declared && declared.step >= 1 ? declared.step : undefined;
  return { key, min, max, log: min > 0 && max / min >= 10, ...(step ? { step } : {}) };
}

/**
 * `count` points over `axes`, from a seeded stream so a sample is a
 * reproducible thing the way a grid is.
 *
 * Stratified per axis rather than independent uniform: each axis's range is
 * cut into `count` equal bins, one draw taken from each, and the bins shuffled
 * independently. That is a Latin hypercube, and it costs nothing over plain
 * random while guaranteeing the whole of every range is covered — with
 * independent draws, a 20-point sample leaves gaps in each axis by chance, and
 * the gaps are where a transition hides.
 */
export function latinHypercube(axes: SampleAxis[], count: number, seed: number): Record<string, number>[] {
  let s = seed >>> 0;
  const rnd = (): number => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const out: Record<string, number>[] = [];
  for (let i = 0; i < count; i++) out.push({});
  for (const axis of axes) {
    const bins: number[] = [];
    for (let i = 0; i < count; i++) {
      const u = (i + rnd()) / count;
      bins.push(
        axis.log
          ? Math.exp(Math.log(axis.min) + u * (Math.log(axis.max) - Math.log(axis.min)))
          : axis.min + u * (axis.max - axis.min),
      );
    }
    for (let i = bins.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const t = bins[i];
      bins[i] = bins[j];
      bins[j] = t;
    }
    for (let i = 0; i < count; i++) {
      out[i][axis.key as string] = axis.step ? Math.round(bins[i] / axis.step) * axis.step : bins[i];
    }
  }
  return out;
}
