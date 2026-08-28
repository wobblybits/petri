import { boundRadius, momentOfInertia, type Agent } from '../agents.ts';
import { albedo } from './voice.ts';

/**
 * The resonance of an agent's own body.
 *
 * Without this a collision can only be heard through a wire that happens to
 * exist, so in a loose soup almost every knock is silent and collisions are
 * second-class next to latches and rewrites. A struck body rings whether or not
 * anything is tied to it.
 *
 * Frequencies scale as 1/size, so what you see is what you hear: the small
 * circle pings and the big triangle thuds an octave below it. The mode ratios
 * are the inharmonic ones of a plate, not a harmonic series — a knocked body is
 * not a string.
 */
export const BODY_MODES = 3;

/** Free circular plate: near-harmonic, clean, bell-like. */
const DISC_RATIOS = [1, 2.09, 3.44];
/** Triangular plate: wider spacing, more clatter. */
const PLATE_RATIOS = [1, 1.71, 2.62];

export interface BodyTone {
  /** Mode frequencies in Hz, low to high. */
  freq: number[];
  /** Per-mode T60 in seconds. Higher modes always die first. */
  decay: number[];
  /** Per-mode amplitude. */
  gain: number[];
}

export function bodyTone(agent: Agent): BodyTone {
  const size = boundRadius(agent);
  // A knock is a bar-like mode, so pitch tracks 1/size: Era's 9 px circle sits
  // about an octave over the triangle's 18 px.
  const f0 = Math.max(70, Math.min(2200, 2600 / size));
  const ratios = agent.kind === 'era' ? DISC_RATIOS : PLATE_RATIOS;
  // Heavier bodies hold their energy; a knock is short either way.
  const base = Math.max(0.08, Math.min(0.9, 0.1 + agent.mass * 0.22));
  // Light bodies read as bright and ring on; dark ones are dull and damped.
  const light = albedo(agent.kind);
  const freq: number[] = [];
  const decay: number[] = [];
  const gain: number[] = [];
  for (let i = 0; i < BODY_MODES; i++) {
    freq.push(f0 * ratios[i]);
    decay.push(base / (1 + i * 1.1) * (0.6 + light * 0.7));
    gain.push((1 / (1 + i * 1.4)) * (0.55 + light * 0.5));
  }
  return { freq, decay, gain };
}

/**
 * How strongly a body couples into a wire tied to it. A light body driven by
 * the same force moves further, so it feeds the wire harder.
 */
export function bodyCoupling(agent: Agent): number {
  const inertia = Math.max(0.05, momentOfInertia(agent));
  return Math.max(0.15, Math.min(1.4, 1.6 / Math.sqrt(Math.max(0.08, agent.mass) * inertia) ));
}
