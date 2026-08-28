import type { AgentKind, PortSlot } from '../agents.ts';

export const MAX_WIRES = 64;
export const MAX_DELAY = 4096;

/**
 * Output sample rate. The AudioContext reports the real device rate at boot;
 * everything pitch-related is derived from this rather than assuming 48 kHz.
 */
let sampleRate = 48000;

export function setSampleRate(rate: number): void {
  if (Number.isFinite(rate) && rate > 8000) sampleRate = rate;
}

export function getSampleRate(): number {
  return sampleRate;
}

/** Rest-tension wave speed along a rope, in world pixels per second.
 *  A 40 px rest (the default min length) then sounds at 400 Hz. */
export const WAVE_SPEED = 32000;

let speedScale = 1;

/** Global wave-speed multiplier. 1 is the default; lower = the whole pond is flatter. */
export function setWaveSpeed(scale: number): void {
  if (!Number.isFinite(scale)) return;
  speedScale = Math.max(0.35, Math.min(2.8, scale));
}

export function getWaveSpeed(): number {
  return speedScale;
}

/**
 * Wave speed along the rope. Tautness raises it as √tension: a yank is sharper
 * because the material got faster, not because we nudged a mapped pitch.
 */
export function waveSpeed(taut = 0): number {
  return WAVE_SPEED * speedScale * Math.sqrt(1 + Math.max(0, taut) * 2);
}

/**
 * Sounding pitch of a rope: travel time for a round trip. Long is low.
 */
export function wirePitchHz(pathPx: number, taut = 0): number {
  return waveSpeed(taut) / (2 * Math.max(1e-3, pathPx));
}

/** DC group delay the dispersion allpass adds on each traversal, in samples. */
function dispersionDelay(disp: number): number {
  return (1 - disp) / (1 + disp);
}

/** Delay line length for a pitch, minus whatever the allpass already adds. */
export function delaySamplesForHz(hz: number, disp = 0): number {
  const raw = sampleRate / (2 * Math.max(1e-3, hz)) - (disp !== 0 ? dispersionDelay(disp) : 0);
  if (!Number.isFinite(raw)) return 64;
  return Math.max(8, Math.min(MAX_DELAY - 1, raw));
}

/** Delay along a rope of `pathPx` at the current tautness. */
export function delaySamplesForPath(pathPx: number, taut = 0, disp = 0): number {
  const path = Number.isFinite(pathPx) && pathPx > 0 ? pathPx : 40;
  const t = Number.isFinite(taut) ? taut : 0;
  return delaySamplesForHz(wirePitchHz(path, t), disp);
}

export function portImpedance(kind: AgentKind, slot: PortSlot): number {
  const base = kind === 'era' ? 1 : kind === 'dup' ? 0.72 : 1.35;
  return slot === 'p' ? base : base * 0.62;
}

/**
 * Magnitude of the one-pole damping filter `y += damp * (x - y)` at `hz`.
 * Unity at DC by construction, falling with frequency — which is the whole
 * point, but it means the fundamental is attenuated too and the loss design
 * has to divide that back out or every note comes up short.
 */
export function dampGainAt(damp: number, hz: number): number {
  const w = (2 * Math.PI * hz) / sampleRate;
  const a = 1 - damp;
  const re = 1 - a * Math.cos(w);
  const im = a * Math.sin(w);
  return damp / Math.max(1e-6, Math.hypot(re, im));
}

/**
 * Raise `damp` until the loop can actually reach `t60`.
 *
 * The damping filter is not free: its own magnitude at the fundamental is a
 * per-traversal loss, and on a short delay line that alone can exceed the whole
 * T60 budget. Asking for both a dark tone and a long ring is then unsatisfiable
 * — the loss term would have to exceed 1, which is not passive — so the tone is
 * the thing that gives. Returns the darkest damp that still hits the target.
 */
export function feasibleDamp(damp: number, lengthSamples: number, t60: number): number {
  const f0 = sampleRate / (2 * Math.max(1, lengthSamples));
  const target = Math.pow(10, (-3 * lengthSamples) / (sampleRate * Math.max(0.02, t60)));
  if (dampGainAt(damp, f0) >= target) return damp;
  let lo = damp;
  let hi = 1;
  // dampGainAt is monotonic in damp, so bisection converges quickly.
  for (let i = 0; i < 24; i++) {
    const mid = (lo + hi) * 0.5;
    if (dampGainAt(mid, f0) >= target) hi = mid;
    else lo = mid;
  }
  return hi;
}

/**
 * Per-traversal gain for a target T60 at the fundamental. A wire rings for
 * `t60` seconds no matter what it is tuned to: the loop makes
 * sampleRate/(2*length) round trips per second, and amplitude has to fall by
 * 1000 over T60 seconds of them. `damp` is compensated for so that the tone
 * control changes the colour of the decay without changing its length.
 */
export function lossForT60(lengthSamples: number, t60: number, damp = 1): number {
  const g = Math.pow(10, (-3 * lengthSamples) / (sampleRate * Math.max(0.02, t60)));
  const f0 = sampleRate / (2 * Math.max(1, lengthSamples));
  return Math.max(0.5, Math.min(0.99999, g / dampGainAt(damp, f0)));
}

/** Slack and node count darken a wire rather than just attenuating it. */
export function bendLoss(nodeCount: number, ropeLen: number, rest: number): number {
  const slack = Math.max(0, ropeLen / Math.max(1, rest) - 1);
  const nodes = Math.min(1, nodeCount * 0.08);
  return Math.min(0.28, 0.03 + slack * 0.1 + nodes * 0.06);
}

/**
 * How far the rope is pulled past its unstretched length. Slack detours have
 * `ropeLen > rest` with `lastLen ≈ ropeLen`, so those stay at 0; a yank has
 * `lastLen` past both and reads as tension.
 */
export function tautness(lastLen: number, rest: number, ropeLen = rest): number {
  const unstretched = Math.max(1, rest, ropeLen);
  return Math.max(0, Math.min(0.6, lastLen / unstretched - 1));
}

/** Taut ropes brighten: more high partials survive each traversal. */
export function tautBrighten(damp: number, taut: number): number {
  return Math.min(1, damp + taut * 0.45);
}

export function latchGain(latchLen: number): number {
  return Math.min(1.8, 0.85 + latchLen / 70);
}

export function collisionGain(impact: number, overlap: number): number {
  const speed = Math.sqrt(Math.max(0, impact) / 120);
  const crush = Math.min(1, overlap / 3);
  return Math.min(1.15, 0.12 + speed * 0.7 * crush);
}

/**
 * Hertzian contact duration, in seconds.
 *
 * For two elastic bodies the contact time goes as (m^2 / (k^2 v))^(1/5): a
 * heavier pair stays in contact longer, and a *faster* impact is over sooner.
 * That last term is the one worth having — it means a hard knock is not merely
 * a louder soft knock, it is a shorter one, and therefore a brighter one. It is
 * the single most recognisable thing about how struck objects actually sound.
 */
export function contactSeconds(effMass: number, vN: number): number {
  const m = Math.max(0.02, effMass);
  const v = Math.max(2, Math.abs(vN));
  return Math.max(0.00025, Math.min(0.02, 0.0016 * Math.pow(m, 0.4) * Math.pow(60 / v, 0.2)));
}

/**
 * Peak contact force from the momentum that has to be reversed.
 *
 * A half-sine of duration `tau` carrying impulse J peaks at pi*J / (2*tau), and
 * that is what the contact really does — but J spans about 60:1 across the
 * speeds this sim produces, and force grows faster than linearly on top of
 * that. The exponent compresses it to roughly 20 dB, which is the range
 * loudness actually occupies. Timbre is left entirely to `tau`, so nothing
 * about the strike's character is being fudged here, only its level.
 */
export function contactPeak(effMass: number, vN: number, tau: number): number {
  const j = Math.max(0, effMass * Math.abs(vN));
  const raw = (Math.PI * j) / (2 * Math.max(1e-4, tau));
  // The quietest contact this sim produces sits near raw = 3600.
  return Math.min(2.2, 0.22 * Math.pow(raw / 3600, 0.4));
}

/**
 * Signed sliding speed (px/s) to bow velocity in the worklet's surface-velocity
 * units. Sign is the tangent of A relative to B; friction is equal-and-opposite.
 *
 * This is not a sounding pitch. The resonator is the clock; this is how fast
 * one surface tries to drag the other. Typical slides land near modal velocity
 * so stick-slip can lock instead of always slipping.
 */
export function bowSpeed(vT: number): number {
  const s = vT * 0.00022;
  if (s > 0.05) return 0.05;
  if (s < -0.05) return -0.05;
  return s;
}

/**
 * Air is faster than the rope so a gap is a short slap, not a second string.
 * 80k px/s: 40 px is ~0.5 ms, a 400 px view is ~5 ms.
 */
export const AIR_SPEED = 80000;
export const AIR_CUTOFF_PX = 240;
export const AIR_MIN_PX = 16;
export const MAX_AIR_PATHS = 48;
export const MAX_AIR_DELAY = 512;

export function airDelaySamples(distPx: number): number {
  const raw = (distPx * getSampleRate()) / AIR_SPEED;
  if (!Number.isFinite(raw)) return 16;
  return Math.max(4, Math.min(MAX_AIR_DELAY - 1, raw));
}

/** Pressure gain along a direct path. 1/r-ish, plus absorption so far hops die. */
export function airGain(distPx: number): number {
  const d = Math.max(AIR_MIN_PX, distPx);
  return (0.14 / (1 + d / 70)) * Math.exp(-d / 200);
}

/** One-pole damp: 1 is bright, toward 0 is a distant thud. */
export function airDamp(distPx: number): number {
  const d = Math.max(AIR_MIN_PX, distPx);
  return Math.max(0.08, Math.min(0.85, 0.88 - d / 420));
}

export function rewriteBeginGain(rule: string): number {
  if (rule === 'commute') return 0.72;
  if (rule === 'erase') return 0.64;
  return 0.58;
}

export function rewriteCommitGain(rule: string): number {
  if (rule === 'commute') return 1.15;
  if (rule === 'erase') return 0.95;
  return 0.82;
}

export function openPortRadiation(openPorts: number): number {
  return 0.12 + openPorts * 0.09;
}
