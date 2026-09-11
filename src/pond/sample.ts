import { SLIDERS, type Params } from '../params.ts';

/*
 * Quasi-random parameter sampling, for asking about many dials at once
 * (Bergstra and Bengio: worthwhile when only a few dimensions matter).
 * Samples compose: a random draw joins every regression ever run over the
 * library. Ranges come from `SLIDERS`; a slider range is an assumption, so
 * override the bounds when the question is whether the assumption holds.
 */

export interface SampleAxis {
  key: keyof Params;
  min: number;
  max: number;
  /**
   * Sampled log-uniform rather than uniform. Applied where a range spans at
   * least a factor of ten and its floor is above zero; a floor at zero is
   * usually a switch rather than a scale.
   */
  log: boolean;
  /**
   * Snap draws to this multiple. Set from the slider when it declares a step
   * of one or more (a count), so a stored value is one that actually ran;
   * finer steps are display precision and are left alone.
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
 * `count` points over `axes`, from a seeded stream so a sample is
 * reproducible. A Latin hypercube: each axis's range is cut into `count`
 * equal bins, one draw taken from each, and the bins shuffled independently,
 * so the whole of every range is covered.
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
