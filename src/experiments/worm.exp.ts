import { describe, expect, it } from 'vitest';
import type { Agent } from '../agents.ts';
import { EXTRA_FLOOR } from '../energy.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';

/*
 * One worm on a bench, with its energy under external control.
 *
 * Everything measured in the soup so far has been unable to answer the
 * question it was asked. `grip` changed how mobile the whole dish was, so
 * `net_drift` moved for reasons that had nothing to do with swimming; `net_coherence` moved with net size; quantising transport
 * changed almost nothing. The confounds were not incidental — a pond decides
 * its own population, its own net sizes and its own energy throughput, and
 * every one of those moves when a physics dial moves.
 *
 * So: no soup. One net, built by hand, held together, with no sensing, no
 * steering, no grazing, no rewriting, no immigration and no signalling. The
 * only things that happen are the wire constraint, the drag law, and energy
 * moving along wires with the recoil that carries. Every body's tank is
 * written each frame from a policy, which makes throughput an independent
 * variable instead of whatever the economy happened to supply.
 *
 * The worm is three strands of 40, cross-braided, with an Era hanging off the
 * periphery at every segment. An Era has one port, so it can only be a leaf;
 * it is also 0.45 to a node's 1, which puts the light mass on the outside.
 *
 *     npm run experiment -- worm
 *
 * The measured quantity is the worm's centre of mass. Under one drag law and
 * no thrust it cannot move at all, whatever energy does inside — so any
 * displacement here is the thing the whole exercise is about.
 */

/** Segments along the worm. */
const SEGMENTS = 40;
/** Strands: A and C are the periphery, B the core. */
const STRANDS = 3;
/**
 * Spacing along the worm and between strands, and the wires' rest length.
 *
 * Wider than the pond's default 48 on purpose. At 48 the bodies are in
 * permanent contact, and contact radiates: `noteRadiation` books an equal and
 * opposite impulse, but `applyRadiationLoss` then clamps each body's share
 * against *its own* speed, so the pair scales unequally and the centre of mass
 * moves. On the first run of this bench that showed up as the `flat` control
 * drifting 13 px in twenty seconds with not one unit of energy moved. At 72
 * the bodies clear each other and the control sits still, which is the whole
 * requirement for a bench.
 */
const PITCH = 72;
const GAUGE = 72;

export interface Worm {
  /** `nodes[strand][segment]`, Con on the core and Dup on the flanks. */
  nodes: Agent[][];
  /** One per segment, alternating flanks. The light bodies. */
  eras: Agent[];
  all: Agent[];
  /** Which segment each body sits at, indexed by agent id. */
  segOf: Map<number, number>;
}

/**
 * A bench pond: the wire constraint and the drag law, and nothing else.
 *
 * `rewriteDuration` 0 is the hard kill on rewriting — `collectReadyRedexes`
 * returns immediately — and the worm is wired so that no two principal ports
 * ever face each other anyway, which is the condition for a redex. Belt and
 * braces, because a worm that rewrites is not the worm that was built.
 */
export function rigParams(over: Partial<Params> = {}): Params {
  const p = defaultParams();
  // No ecology: nothing arrives, nothing grows, nothing is eaten or billed.
  p.spawnInterval = 0;
  p.soupCount = 0;
  p.ambientEnergy = 0;
  p.energyRegrow = 0;
  p.energyDiffuse = 0;
  p.upkeep = 0;
  p.forageAsk = 0;
  p.fertilise = 0;
  p.uptakeVmax = 0;
  p.swimCost = 0;
  p.contactCost = 0;
  /*
   * **And the reactor, which is what killed this rig.**
   *
   * The rig feeds nothing — `ambientEnergy`, `energyRegrow` and `uptakeVmax`
   * are all zero, because the question is the drag law and not the economy.
   * When the gait shipped on (`metabolicRate` 15), every body here began
   * running a reactor with no gut to fill it, so its primer sat at zero, its
   * starvation window ran out at `starveTime` 25 s, and the whole worm died a
   * quarter of the way through the 120 s settle. Every locomotion number this
   * file printed after that was measuring a corpse: 0.000 px/s at every
   * setting, and `moved` stuck at the head's first fill.
   *
   * Pinned here for the reason `fixedParams` pins it in the suite — a rig that
   * is not about the reactor says so — and an arm that *is* about it turns it
   * on and feeds the worm. See the grip-swing arm, which drives the wave from
   * a policy instead so the phase is an independent variable.
   */
  p.metabolicRate = 0;
  // No rewriting, and no latching that could start one.
  p.rewriteDuration = 0;
  p.snapRadius = 0;
  p.snapWell = 0;
  p.faceAttract = 0;
  p.faceRadius = 0;
  // No sensing and nothing to sense.
  p.deposit = 0;
  p.diffuse = 0;
  p.decay = 0;
  p.sense = 0;
  // No steering, no self-propulsion, no shoaling. The genome's cruise and
  // turn heads seed from these, so at zero the locomotion head is inert.
  p.stepSpeed = 0;
  p.turnRate = 0;
  p.swimNoise = 0;
  p.flockAlign = 0;
  p.flockSep = 0;
  p.declutter = 0;
  // Fixed weights, and a wire that holds its length at the bench's spacing
  // rather than shrinking the worm back into self-contact.
  p.learnRate = 0;
  p.wireBreathe = 0;
  p.wireSnap = 0;
  p.wireMinRest = PITCH;
  // The reactionless drive off, so nothing here can move by minting momentum.
  p.transportThrust = 0;
  return { ...p, ...over };
}

/**
 * Three strands, cross-braided, with a leaf Era at every segment.
 *
 * Along a strand, segment `i`'s principal joins segment `i+1`'s left aux —
 * principal to aux, so it is not a redex. The right aux carries the rungs:
 * A-B on even segments and B-C on odd ones, which is the most a three-strand
 * ladder can do when a body has three ports and two are already spent. That
 * leaves exactly one free aux per segment, alternating flanks, and an Era
 * goes there.
 */
export function buildWorm(sim: Sim, params: Params, cx: number, cy: number): Worm {
  const nodes: Agent[][] = [];
  const segOf = new Map<number, number>();
  const x0 = cx - ((SEGMENTS - 1) * PITCH) / 2;
  for (let s = 0; s < STRANDS; s++) {
    const row: Agent[] = [];
    const y = cy + (s - 1) * GAUGE;
    for (let i = 0; i < SEGMENTS; i++) {
      const kind = s === 1 ? 'con' : 'dup';
      const a = sim.spawn(kind, x0 + i * PITCH, y, 0, params, true)!;
      segOf.set(a.id, i);
      row.push(a);
    }
    nodes.push(row);
  }
  for (let s = 0; s < STRANDS; s++) {
    for (let i = 0; i + 1 < SEGMENTS; i++) {
      sim.wire(nodes[s][i].id, 'p', nodes[s][i + 1].id, 'l', params);
    }
  }
  /*
   * Rungs at two segments only, which makes the wire graph a tree.
   *
   * A rung at every segment is the obvious braid and it cannot be measured.
   * Every rung closes a cycle, and a cycle of distance constraints has to be
   * satisfiable in the plane or the net is *frustrated*: the wires are born at
   * their stem spans, ramp to `wireMinRest`, and no configuration meets all of
   * them at once, so the worm creeps for as long as it is run. Measured on the
   * braided version, it was still moving several px/s after a minute, the
   * creep was chaotic, and the stroke came out the same size at recoil 1 and
   * recoil 400 — two diverging trajectories rather than a force.
   *
   * A tree has no cycles and therefore nothing to frustrate. Two rungs join
   * the three strands into one net and every other free aux takes an Era.
   */
  const eras: Agent[] = [];
  const rungAB = Math.floor(SEGMENTS / 3);
  const rungBC = Math.floor((2 * SEGMENTS) / 3);
  sim.wire(nodes[0][rungAB].id, 'r', nodes[1][rungAB].id, 'r', params);
  sim.wire(nodes[1][rungBC].id, 'r', nodes[2][rungBC].id, 'r', params);
  for (let i = 0; i < SEGMENTS; i++) {
    for (const [s, side] of [[0, -1], [2, 1]] as const) {
      if ((s === 0 && i === rungAB) || (s === 2 && i === rungBC)) continue;
      const host = nodes[s][i];
      const era = sim.spawn('era', host.x, host.y + side * GAUGE, 0, params, true)!;
      sim.wire(era.id, 'p', host.id, 'r', params);
      segOf.set(era.id, i);
      eras.push(era);
    }
  }
  const all = [...nodes.flat(), ...eras];
  return { nodes, eras, all, segOf };
}

/** Mass-weighted centre of the worm. */
export function centre(worm: Worm): { x: number; y: number; m: number } {
  let x = 0;
  let y = 0;
  let m = 0;
  for (const a of worm.all) {
    x += a.x * a.mass;
    y += a.y * a.mass;
    m += a.mass;
  }
  return { x: x / m, y: y / m, m };
}

/**
 * A charge in [0, 1] for a body, from where it sits and what time it is.
 *
 * 0 is not an empty tank, it is `EXTRA_FLOOR` — as deep in debt as the pond
 * allows. That matters, and it is what the first version of this bench got
 * wrong: with rewriting off and nothing starving, the only claim left in
 * `pulseRequests` is `rescueNeed`, and `hungerNeed` is `-extra` clamped at
 * zero. A body holding *any* non-negative amount asks for nothing, the demand
 * field stays flat, `flowCharges` finds no neighbour needier than its donor,
 * and not one unit moves all run. Every condition then measures the same
 * settling twitch, which is exactly what it did: `flat` and `wave` came back
 * identical to the last decimal.
 *
 * So a charge spans debt to full, and a drive that wants transport has to put
 * one end of the worm under water.
 *
 * This is the external supply: it is written over every body every frame, so
 * what transport does with it never feeds back into what is available. That
 * is the point — the soup could not separate "does a stroke move a net" from
 * "how much energy was there to move".
 */
export type Drive = (seg: number, t: number, isEra: boolean) => number;

export const DRIVES: Record<string, Drive> = {
  /** Control: everyone full, no gradient, nothing to transport. */
  flat: () => 1,
  /** A fixed gradient, head to tail. What a net holding a standing shortage has. */
  standing: (seg) => 1 - seg / (SEGMENTS - 1),
  /** The same gradient, travelling: peristalsis, at one wavelength per worm. */
  wave: (seg, t) => 0.5 + 0.5 * Math.sin(2 * Math.PI * (seg / SEGMENTS - t / WAVE_PERIOD)),
  /** Four waves along the body at once, which is what a crawler actually does. */
  ripple: (seg, t) => 0.5 + 0.5 * Math.sin(2 * Math.PI * ((4 * seg) / SEGMENTS - t / WAVE_PERIOD)),
};

/** Seconds for one full pass of a travelling wave. */
const WAVE_PERIOD = 2;

export interface RunSpec {
  drive: keyof typeof DRIVES;
  seed?: number;
  params?: Partial<Params>;
  /** Seconds to let the shape relax before the drive starts and the clock runs. */
  settle?: number;
  seconds?: number;
}

export interface RunResult {
  dx: number;
  dy: number;
  /** Along-body speed, px/s, over the measured window. */
  speed: number;
  /** Largest distance any body ended up from where the worm's centre says it should be. */
  spread: number;
  /** Units of energy transport actually delivered, over the whole window. */
  moved: number;
  /** Centre-of-mass speed still left over from construction, px/s. */
  residual: number;
  bodies: number;
}

/** The suite's LCG. The experiments project seeds nothing, so a bench must. */
function seeded(n: number): () => number {
  let s = n >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function runWorm(spec: RunSpec): RunResult {
  const real = Math.random;
  Math.random = seeded(spec.seed ?? 1);
  try {
    return runWormInner(spec);
  } finally {
    Math.random = real;
  }
}

function runWormInner(spec: RunSpec): RunResult {
  const params = rigParams(spec.params);
  const dt = 1 / 60;
  const settle = spec.settle ?? 120;
  const seconds = spec.seconds ?? 20;
  const sim = new Sim(4000, 4000, 512);
  const worm = buildWorm(sim, params, 2000, 2000);
  /*
   * Relax the shape first, hard.
   *
   * A hand-built worm is not at mechanical equilibrium: a wire is born at the
   * span between its two stems and then ramps to `wireMinRest`, which is not
   * the spacing the bodies were placed at, so the whole body inflates. At the
   * pond's own drag that relaxation is still running minutes later at several
   * px/s, and it is chaotic — measured, the difference between recoil 1 and
   * recoil 400 was the same 20 px, which is not a force scaling with impulse
   * but two trajectories diverging.
   *
   * So the settle runs heavily overdamped, which reaches the same
   * configuration without the ringing, and the measurement only starts once
   * the worm has stopped moving on its own.
   */
  for (const a of worm.all) a.extra = 0;
  const thick = { ...params, drag: 20, angDrag: 20 };
  for (let f = 0; f < Math.round(settle / dt); f++) sim.step(dt, thick);

  // What the shape is still doing when the clock starts. Under one drag law
  // and no thrust the centre cannot move at all, so any residue here is
  // construction momentum that would be counted as travel.
  let residual = 0;
  {
    const a = centre(worm);
    for (let f = 0; f < 30; f++) sim.step(dt, params);
    const b = centre(worm);
    residual = Math.hypot(b.x - a.x, b.y - a.y) / (30 * dt);
  }

  const before = centre(worm);
  const frames = Math.round(seconds / dt);
  const written = new Float64Array(worm.all.length);
  let moved = 0;
  for (let f = 0; f < frames; f++) {
    const t = f * dt;
    for (let i = 0; i < worm.all.length; i++) {
      const a = worm.all[i];
      const seg = worm.segOf.get(a.id) ?? 0;
      const c = DRIVES[spec.drive](seg, t, a.kind === 'era');
      const level = c <= 0 ? 0 : c >= 1 ? 1 : c;
      a.extra = EXTRA_FLOOR + level * (a.energyCap - EXTRA_FLOOR);
      written[i] = a.extra;
    }
    sim.step(dt, params);
    // Arrivals, which is what transport delivered this frame. Measured rather
    // than assumed: throughput is the independent variable the soup could
    // never hold still, and a condition that moved nothing explains itself.
    for (let i = 0; i < worm.all.length; i++) {
      const d = worm.all[i].extra - written[i];
      if (d > 0) moved += d;
    }
  }
  const after = centre(worm);
  let spread = 0;
  for (const a of worm.all) {
    spread = Math.max(spread, Math.hypot(a.x - after.x, a.y - after.y));
  }
  const dx = after.x - before.x;
  const dy = after.y - before.y;
  return { dx, dy, speed: Math.hypot(dx, dy) / seconds, spread, moved, residual, bodies: worm.all.length };
}


/*
 * Free running: the worm's own transport, not an imposed pattern.
 *
 * The driven bench answered "if a wave of fullness ran down the body, would
 * it swim" — yes, and three times better than a standing gradient. It could
 * not answer where a wave would come from, because the wave was written in by
 * hand every frame.
 *
 * This is the other half. Only the two ends are held: the head at a full tank
 * and the tail in debt, which is a source and a sink and nothing else. Every
 * body between them floats on what transport brings it. A body with a quantum
 * accumulates until it holds a packet and then fires, which makes it a
 * relaxation oscillator whose natural period is its quantum over its income —
 * so a row of them with *different* quanta is a chain of coupled oscillators
 * at different natural frequencies. That is the standard account of how gut
 * peristalsis comes to travel in one direction rather than pulsing in place.
 *
 * Whether it does so here is the question, and the arrangement is the thing
 * to vary: identical quanta everywhere, a difference across the worm (core
 * against flanks, which is what "Con and Dup differ" comes to on this
 * skeleton), a difference scattered body by body, and a monotonic gradient
 * along the axis, which is the arrangement the gut actually has.
 */
export type Quanta = (seg: number, strand: number, isEra: boolean) => number;

export const QUANTA: Record<string, Quanta> = {
  /** Continuous transport: the pond as it ships. */
  none: () => 0,
  /** Every body the same. A row of identical oscillators, which synchronise. */
  uniform: () => 0.5,
  /** Core against flanks — Con against Dup, on this worm. Across, not along. */
  kind: (_seg, strand) => (strand === 1 ? 0.5 : 1),
  /** The same two values, scattered, which is what a bred net would look like. */
  mixed: (seg, strand) => ((seg * 7 + strand * 3) % 5 < 2 ? 0.5 : 1),
  /** A monotonic gradient head to tail: the arrangement the gut has. */
  gradient: (seg) => 0.35 + (0.9 * seg) / (SEGMENTS - 1),
};

export interface FreeSpec {
  quanta: keyof typeof QUANTA;
  params?: Partial<Params>;
  seed?: number;
  seconds?: number;
  /**
   * Units of ground written into every body's gut each frame, so the reactor
   * runs without the worm having to eat.
   *
   * The rig cannot simply be fed. Give bodies ground and `uptakeVmax` and each
   * one earns its own income, every tank fills, `hungerNeed` goes to zero
   * everywhere and the head-to-tail gradient this bench exists to control
   * stops existing — measured, `moved` falls from about 2600 to 1.
   *
   * So the gut is written from a policy, the way the tanks are. Keep it small
   * enough that digestion's whole mouthful is inside what `intake` routes to
   * the reactor (`gut * (1 - exp(-digestRate*dt))` under `intake * dt`) and
   * nothing reaches the tank: the reactor is fed at exactly its own rate and
   * the transport gradient is untouched.
   */
  primeGut?: number;
  /**
   * Units per second fed to the head and taken from the tail. Omit for the
   * unmetered bench, which pins the head at a full tank and the tail in debt
   * and so supplies whatever the physics will take.
   *
   * The unmetered version answers "can this swim at all" and nothing about
   * whether a pond could pay for it. Metering makes the through-flux an
   * independent variable: at steady state exactly this many units cross the
   * worm each second, whatever the wires do. A worm of `n` bodies at the
   * pond's own `upkeep` needs `n * upkeep` units a second to stand still —
   * 3.0 at the default and 15.8 in the starved regime the sweeps used — so
   * those are the numbers a real economy could put through it.
   */
  income?: number;
}

export interface FreeResult {
  /** Mean frames of lag between one segment's fullness and the next one's. */
  lag: number;
  /** How much of the lag estimate agrees on a direction, 0 to 1. */
  agree: number;
  /** Peak-to-trough swing of segment fullness, averaged. 0 = nothing oscillates. */
  swing: number;
  moved: number;
  dx: number;
  dy: number;
  /** Frames of lag between one segment's *gait phase* and the next one's. */
  gaitLag: number;
  /** How much of that lag agrees on a direction, 0 to 1. */
  gaitAgree: number;
  /** Peak-to-trough swing of segment gait phase. 0 = the reactor is not running. */
  gaitSwingSeen: number;
}

/**
 * Lag between neighbouring segments' fullness, by cross-correlation.
 *
 * A standing pulse has every segment rising and falling together, so the lag
 * that best matches one segment to the next is zero. A travelling wave has a
 * consistent non-zero lag, and its sign is the direction of travel. The
 * fraction of pairs agreeing on that sign is reported beside it, because a
 * mean lag near zero means "standing" and "half of it going each way" alike.
 */
function waveLag(series: Float64Array, segments: number, frames: number): { lag: number; agree: number } {
  const MAXLAG = 45;
  const at = (seg: number, f: number): number => series[seg * frames + f];
  let sum = 0;
  let n = 0;
  let pos = 0;
  for (let seg = 0; seg + 1 < segments; seg++) {
    let mA = 0;
    let mB = 0;
    for (let f = 0; f < frames; f++) {
      mA += at(seg, f);
      mB += at(seg + 1, f);
    }
    mA /= frames;
    mB /= frames;
    let best = 0;
    let bestScore = -Infinity;
    for (let L = -MAXLAG; L <= MAXLAG; L++) {
      let acc = 0;
      let count = 0;
      for (let f = 0; f < frames; f++) {
        const g = f + L;
        if (g < 0 || g >= frames) continue;
        acc += (at(seg, f) - mA) * (at(seg + 1, g) - mB);
        count++;
      }
      if (count === 0) continue;
      const score = acc / count;
      if (score > bestScore) {
        bestScore = score;
        best = L;
      }
    }
    sum += best;
    n++;
    if (best > 0) pos++;
    else if (best < 0) pos--;
  }
  return { lag: n > 0 ? sum / n : 0, agree: n > 0 ? Math.abs(pos) / n : 0 };
}

export function runFree(spec: FreeSpec): FreeResult {
  const real = Math.random;
  Math.random = seeded(spec.seed ?? 1);
  try {
    const params = rigParams(spec.params);
    const dt = 1 / 60;
    const seconds = spec.seconds ?? 30;
    const sim = new Sim(4000, 4000, 512);
    const worm = buildWorm(sim, params, 2000, 2000);
    for (const a of worm.all) a.extra = 0;
    const thick = { ...params, drag: 20, angDrag: 20 };
    const prime = spec.primeGut;
    const GUT = sim.agentStore.gut;
    /*
     * The settle is primed too, and it has to be: it is 120 s long and the
     * starvation window is 25, so a reactor left unfed here kills the whole
     * worm before the clock even starts. That is exactly how this rig died
     * quietly when the gait shipped on.
     */
    for (let f = 0; f < Math.round(120 / dt); f++) {
      if (prime !== undefined) for (const a of worm.all) GUT[a.slot] = prime;
      sim.step(dt, thick);
    }

    // Quanta by arrangement, and the two ends that are held.
    const pick = QUANTA[spec.quanta];
    for (let st = 0; st < STRANDS; st++) {
      for (let i = 0; i < SEGMENTS; i++) worm.nodes[st][i].transportQuantum = pick(i, st, false);
    }
    for (const e of worm.eras) e.transportQuantum = pick(worm.segOf.get(e.id) ?? 0, 1, true);
    const head = worm.nodes.map((row) => row[0]);
    const tail = worm.nodes.map((row) => row[SEGMENTS - 1]);

    const frames = Math.round(seconds / dt);
    const series = new Float64Array(SEGMENTS * frames);
    // The same cross-correlation, on the gait's own phase rather than on
    // fullness: is there a wave running along this worm at all?
    const gait = new Float64Array(SEGMENTS * frames);
    const WAVE = sim.agentStore.gaitWave;
    const before = centre(worm);
    let moved = 0;
    const last = new Map<number, number>();
    for (const a of worm.all) last.set(a.id, a.extra);
    const income = spec.income;
    for (let f = 0; f < frames; f++) {
      // The reactor's own supply, written rather than eaten. See `primeGut`.
      if (prime !== undefined) for (const a of worm.all) GUT[a.slot] = prime;
      if (income === undefined) {
        // Unmetered: an infinite source and an infinite sink. Whatever the
        // physics will carry, it gets.
        for (const a of head) a.extra = a.energyCap;
        for (const a of tail) a.extra = EXTRA_FLOOR;
      } else {
        // Metered: a fixed number of units a second in at the head and the
        // same out at the tail, which is what an economy would supply.
        const per = (income * dt) / head.length;
        for (const a of head) a.extra = Math.min(a.energyCap, a.extra + per);
        for (const a of tail) a.extra = Math.max(EXTRA_FLOOR, a.extra - per);
      }
      sim.step(dt, params);
      const fill = new Float64Array(SEGMENTS);
      const count = new Float64Array(SEGMENTS);
      for (const a of worm.all) {
        const seg = worm.segOf.get(a.id) ?? 0;
        const c = (a.extra - EXTRA_FLOOR) / (a.energyCap - EXTRA_FLOOR);
        fill[seg] += c <= 0 ? 0 : c >= 1 ? 1 : c;
        count[seg] += 1;
        const d = a.extra - (last.get(a.id) ?? 0);
        if (d > 0) moved += d;
        last.set(a.id, a.extra);
      }
      for (let seg = 0; seg < SEGMENTS; seg++) series[seg * frames + f] = count[seg] > 0 ? fill[seg] / count[seg] : 0;
      const gw = new Float64Array(SEGMENTS);
      const gn = new Float64Array(SEGMENTS);
      for (const a of worm.all) {
        const seg = worm.segOf.get(a.id) ?? 0;
        gw[seg] += WAVE[a.slot];
        gn[seg] += 1;
      }
      for (let seg = 0; seg < SEGMENTS; seg++) gait[seg * frames + f] = gn[seg] > 0 ? gw[seg] / gn[seg] : 0;
    }
    const after = centre(worm);
    const { lag, agree } = waveLag(series, SEGMENTS, frames);
    const g = waveLag(gait, SEGMENTS, frames);
    let gaitSwingSeen = 0;
    for (let seg = 0; seg < SEGMENTS; seg++) {
      let lo = Infinity;
      let hi = -Infinity;
      for (let f = 0; f < frames; f++) {
        const v = gait[seg * frames + f];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      gaitSwingSeen += hi - lo;
    }
    gaitSwingSeen /= SEGMENTS;
    let swing = 0;
    for (let seg = 0; seg < SEGMENTS; seg++) {
      let lo = Infinity;
      let hi = -Infinity;
      for (let f = 0; f < frames; f++) {
        const v = series[seg * frames + f];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      swing += hi - lo;
    }
    return {
      lag,
      agree,
      swing: swing / SEGMENTS,
      moved,
      dx: after.x - before.x,
      dy: after.y - before.y,
      gaitLag: g.lag,
      gaitAgree: g.agree,
      gaitSwingSeen,
    };
  } finally {
    Math.random = real;
  }
}

describe('experiment: one worm on a bench', () => {
  it('builds a worm that holds together and does not rewrite', () => {
    const params = rigParams();
    const sim = new Sim(4000, 4000, 512);
    const worm = buildWorm(sim, params, 2000, 2000);
    expect(worm.all.length).toBe(STRANDS * SEGMENTS + 2 * SEGMENTS - 2);
    // No principal faces a principal, so nothing here is a redex even before
    // `rewriteDuration` 0 takes rewriting away.
    for (const w of sim.graph.wires.values()) {
      expect(w.a.slot === 'p' && w.b.slot === 'p', `wire ${w.id} is a redex`).toBe(false);
    }
    for (let f = 0; f < 300; f++) sim.step(1 / 60, params);
    expect(sim.agents.size, 'the worm lost bodies').toBe(worm.all.length);
    expect(sim.tally.commutes + sim.tally.annihilations + sim.tally.erases).toBe(0);
    expect(sim.graph.componentIds(sim.agents, sim.rosterVersion).size, 'it came apart')
      .toBe(worm.all.length);
  });

  /*
   * The stroke, as a difference against the same worm with the kicks off.
   *
   * A hand-built worm does not start at rest. It relaxes for a long time —
   * the wires are born at their stem span and ramp to `wireMinRest`, which is
   * not the spacing the bodies were placed at, so the whole body inflates and
   * keeps inflating well past a minute — and that relaxation moves the centre
   * by pixels, which is the size of the effect being looked for. Settling
   * longer does not fix it and a quiescent bench is not cheaply available.
   *
   * `transportRecoil` 0 is the control that makes it irrelevant. The run is
   * seeded and deterministic, so the twin has the identical geometry, the
   * identical relaxation and the identical drag pattern — the energy still
   * moves, `grip` still reads the same tank levels — and the only thing
   * removed is the impulse a transfer carries. The difference between the two
   * is the momentum transport produced, and nothing else.
   */
  function stroke(spec: RunSpec): { ddx: number; ddy: number; px: number; moved: number } {
    const on = runWorm({ ...spec, params: { ...spec.params, transportRecoil: 100 } });
    const off = runWorm({ ...spec, params: { ...spec.params, transportRecoil: 0 } });
    const ddx = on.dx - off.dx;
    const ddy = on.dy - off.dy;
    const seconds = spec.seconds ?? 20;
    return { ddx, ddy, px: Math.hypot(ddx, ddy) / seconds, moved: on.moved };
  }


  it('asks whether unequal quanta make a wave on their own', () => {
    /*
     * The driven bench showed a travelling wave swims and a standing gradient
     * barely does. This asks where a wave could come from without one being
     * written in: hold a source at the head and a sink at the tail, give the
     * bodies different quanta, and see whether the fullness pattern travels.
     *
     * `lag` is frames of delay between one segment and the next, `agree` how
     * much of the body agrees on a direction, `swing` whether anything is
     * oscillating at all — a flat profile has no phase to measure and its lag
     * is noise.
     */
    console.log('\n  free running: source at the head, sink at the tail, grip 2');
    console.log('\n  quanta      lag  agree  swing     moved   stroke px/s');
    console.log('  -------- ------- ------ ------ --------- -------- ------');
    for (const q of ['none', 'uniform', 'kind', 'mixed', 'gradient'] as const) {
      const on = runFree({ quanta: q, params: { grip: 2, transportRecoil: 100 } });
      const off = runFree({ quanta: q, params: { grip: 2, transportRecoil: 0 } });
      // The same control the driven bench uses: identical worm, identical
      // energy, the impulse removed. What is left is the momentum transport
      // made, with the settling creep and everything else subtracted out.
      const d = Math.hypot(on.dx - off.dx, on.dy - off.dy);
      console.log(
        `  ${q.padEnd(8)} ${on.lag.toFixed(2).padStart(7)} ${on.agree.toFixed(2).padStart(6)} ` +
          `${on.swing.toFixed(3).padStart(6)} ${on.moved.toFixed(0).padStart(9)} ` +
          `${d.toFixed(2).padStart(8)} ${(d / 30).toFixed(3).padStart(6)}`,
      );
      expect(Number.isFinite(on.lag)).toBe(true);
    }
  });


  it('asks what a pond could actually pay for', () => {
    /*
     * The unmetered bench supplies whatever the physics will take, which is
     * about 1,200 units of wire-crossing a second. A worm of 198 bodies at
     * the pond's own upkeep needs 3.0 units a second to stand still, 15.8 in
     * the starved regime, and each of those units crosses roughly twenty
     * wires on its way down the body — so a real economy puts something like
     * 60 to 320 units of crossing a second through it, not 1,200.
     *
     * This meters the source and sink to a fixed flux and asks what is left
     * of the stroke. The floor is the settling creep, ~0.015 px/s.
     */
    console.log('\n  metered: gradient quanta, grip 2, 30 s');
    console.log('\n  income  what it is              moved   px/s (3 seeds)');
    console.log('  ------ ---------------------- --------- ------');
    const rows: [number | undefined, string][] = [
      [3, 'a default pond, exactly'],
      [8, 'between the two'],
      [16, 'a starved pond, exactly'],
      [60, 'four times starved'],
      [240, 'sixteen times starved'],
      [undefined, 'unmetered, for scale'],
    ];
    for (const [income, what] of rows) {
      // Three seeds, which buy nothing and are kept to show that. The rig
      // has no stochastic part left — no swim noise, no spawning, no
      // rewriting, and a hand-placed worm — so the seed changes nothing and
      // the spread is exactly zero at every point. Any structure in this
      // curve is therefore real and reproducible rather than noise.
      const px: number[] = [];
      let moved = 0;
      for (const seed of [1, 2, 3]) {
        const r = runFree({
          quanta: 'gradient',
          income,
          seed,
          params: { grip: 2 },
        });
        px.push(Math.hypot(r.dx, r.dy) / 30);
        moved += r.moved / 3;
      }
      const mean = px.reduce((a, b) => a + b, 0) / px.length;
      const sd = Math.sqrt(px.reduce((a, b) => a + (b - mean) ** 2, 0) / px.length);
      console.log(
        `  ${(income === undefined ? 'inf' : String(income)).padStart(6)} ${what.padEnd(22)} ` +
          `${moved.toFixed(0).padStart(9)} ${mean.toFixed(2).padStart(6)} ±${sd.toFixed(2)}`,
      );
      expect(Number.isFinite(mean)).toBe(true);
    }
  });


  it('separates thrust from recoil', () => {
    /*
     * Two things push a pair along its own wire, and they are not
     * independent. Within a frame the order is drag, then wires, then
     * transport — so the drag law always reads the fullness a transfer has
     * *already* produced: the sender is the empty one and slides, the
     * receiver is the full one and holds.
     *
     * `transportRecoil` is the impulse, and it shoves the pair apart along
     * the wire; `grip` is what stops the two ends sharing that impulse
     * equally, so the pair's centre keeps a share of it.
     * `transportThrust` withholds the receiver's half of the same impulse,
     * which is the reactionless part and the only one that mints momentum.
     * It is retired at 0 and swept here to keep the comparison readable.
     *
     * Signed dx, because the direction is the point: the worm lies along x
     * with its source at low x, so energy flows +x and a negative dx is the
     * worm walking back toward its supply.
     */
    console.log('\n  grip 2, gradient quanta, source at low x so energy flows +x');
    console.log('  metered to 3 units/s, which is this worm at its whole upkeep on the');
    console.log('  pond default — the unmetered bench supplies ~1180, and thrust');
    console.log('  scales with the amount moved, so it flatters itself there.');
    console.log('\n  recoil thrust       dx      dy   px/s     moved');
    console.log('  ------ ------ -------- ------- ------ ---------');
    const rows: [number, number][] = [
      [0, 0],
      [100, 0],
      [100, 1],
    ];
    for (const [transportRecoil, transportThrust] of rows) {
      const r = runFree({
        quanta: 'gradient',
        income: 3,
        params: { grip: 2, transportRecoil, transportThrust },
      });
      const px = Math.hypot(r.dx, r.dy) / 30;
      console.log(
        `  ${String(transportRecoil).padStart(6)} ${transportThrust.toFixed(1).padStart(6)} ` +
          `${r.dx.toFixed(2).padStart(8)} ${r.dy.toFixed(2).padStart(7)} ${px.toFixed(3).padStart(6)} ${r.moved.toFixed(0).padStart(9)}`,
      );
      expect(Number.isFinite(r.dx)).toBe(true);
    }
  });


  it('asks what would make the stroke an order of magnitude larger', () => {
    /*
     * The whole stroke is now: a transfer kicks the pair apart along the wire
     * with an equal and opposite impulse, and the two ends coast different
     * distances from it because `grip` damps them at different rates. The
     * pair's centre keeps `|p| * (1/k_sender - 1/k_receiver) / (m_s + m_r)`.
     * So the step is set by the *ratio* of the anchored end's rate to the
     * sliding end's, not by the level of either — which is why turning `grip`
     * up alone reads as paralysis.
     *
     * `drag` is the floor under that ratio. At the shipped 0.55 against grip
     * 2 a full body damps at 2.55 and an empty one at 0.55, which is only
     * 4.6 to 1, and the sliding end is being held back nearly as hard as the
     * anchored one. Lowering `drag` while keeping `grip` widens the ratio and
     * lets the free end actually travel.
     *
     * The ratio is what the bench optimises and the pond cannot have: `drag`
     * 0.05 and `grip` 4 each stall the calculus inside a simulated minute.
     * See `docs/experiments.md` on why travel is not what the pond is for.
     */
    console.log('\n  thrust 0, recoil 100, income 3: what widens the stroke');
    console.log('\n   drag  grip   ratio       dx   px/s     moved');
    console.log('  ----- ----- ------- -------- ------ ---------');
    const rows: [number, number][] = [
      [0.55, 2],
      [0.55, 8],
      [0.15, 2],
      [0.15, 8],
      [0.05, 4],
      [0.05, 12],
    ];
    for (const [drag, grip] of rows) {
      const r = runFree({
        quanta: 'gradient',
        income: 3,
        params: { drag, grip, transportRecoil: 100, transportThrust: 0 },
      });
      const px = Math.hypot(r.dx, r.dy) / 30;
      const ratio = (drag + grip) / drag;
      console.log(
        `  ${drag.toFixed(2).padStart(5)} ${grip.toFixed(0).padStart(5)} ` +
          `${ratio.toFixed(1).padStart(7)} ${r.dx.toFixed(2).padStart(8)} ${px.toFixed(3).padStart(6)} ${r.moved.toFixed(0).padStart(9)}`,
      );
      expect(Number.isFinite(r.dx)).toBe(true);
    }
  });

  it('is a tripwire on the rig itself', () => {
    /*
     * **This rig was dead for a while and nothing said so.** When the gait
     * shipped on (`metabolicRate` 15), every body here started running a
     * reactor on a bench that deliberately feeds nothing, so the primer sat at
     * zero, the starvation window ran out at 25 s, and the worm died a quarter
     * of the way through the 120 s settle. Every arm went on printing numbers:
     * 0.000 px/s at every setting, including the `drag` 0.05 / `grip` 12 row
     * the stroke arm calls the widest ratio it can reach. A column of zeros
     * reads like a null result rather than a corpse.
     *
     * `moved` is the tell, and it is why this arm asserts on that rather than
     * on travel. It counts increases in a body's tank, so with the head pinned
     * at its cap and income arriving every frame it climbs all run whether or
     * not the worm goes anywhere. Dead it read **4** — three strands of head
     * filling once, then nothing. Alive it reads about **2600**.
     *
     * So: a tripwire on the *rig*, not on the pond. It fails when the worm
     * cannot pass a packet along, which is the precondition every other arm in
     * this file assumes and none of them checks.
     */
    const r = runFree({ quanta: 'gradient', income: 3, params: { grip: 2, transportRecoil: 100 } });
    console.log(`\n  rig tripwire: moved ${r.moved.toFixed(1)}, swing ${r.swing.toFixed(4)}, dx ${r.dx.toFixed(3)}`);
    // Three strands of head, filling once, is about 4. Anything at or below
    // that is a worm that never passed a packet along.
    expect(
      r.moved,
      'the rig transported nothing: every locomotion number from this file is measuring a dead worm',
    ).toBeGreaterThan(10);
  });

  it('asks whether a D-driven grip swing is worth anything', () => {
    /*
     * The stroke and the anchor are both driven by `wave(C)`, so the worm
     * deforms and undeforms through the same shapes: a reciprocal cycle, which
     * nets zero displacement however hard it is driven. That is Purcell's
     * scallop theorem, and it is the same objection that retired `wireTug` —
     * "one degree of freedom and a race".
     *
     * `gripSwing` is the second degree of freedom, and the angle between them
     * is the reactor's own: `dD/dt = k3*C - d*D` is a first-order lag, so D
     * trails C by `atan(w/d)` — 51 degrees at the bottom of the fuel window,
     * 65 at the top, measured at 53.5 on a lone fed body. Nobody chose those
     * numbers.
     *
     * **This arm primes the worm's guts**, which the rest of the file
     * deliberately does not: the reactor has to be running for there to be a
     * second actuator at all. So it is not comparable with the arms above, and
     * the control is `gripSwing` 0 with everything else identical.
     *
     * **Measured, and it costs:** 4.22 px/s at swing 0, against 2.62 at 0.5
     * and 1.88 at 1. Turning the second actuator on makes this worm slower,
     * and the reason is that it has no travelling wave to ride. Every body
     * here is primed identically, so every reactor runs at the same phase and
     * the grip swings *together* at both ends of every wire. Travel per kick
     * is `|p| * (1/k_a - 1/k_b) / (m_a + m_b)`, which is nonlinear in `k`, so
     * scaling both ends by the same factor shrinks the difference on average
     * rather than leaving it alone. A phase offset between the reactor's own
     * species is not enough; the offset has to be *along the body*.
     *
     * **That explanation was wrong and the rig says so.** `gaitLag` reads 6.64
     * frames a segment with 0.67 of pairs agreeing on the direction, against a
     * period of about 128 — a real travelling wave, roughly 19 degrees a
     * segment, even with every gut primed identically, because the broadcast
     * coupling makes the gradient rather than the priming. The swing cost
     * anyway.
     *
     * **The real reason was that nothing timed the kicks.** `Sim.recoil` was
     * reachable from exactly one place, the demand gradient's
     * `flowChargesFast` — so the only impulse in the pond arrived on the
     * economy's clock while the grip swung on the chemistry's, and an
     * uncorrelated zero-mean modulation on a steady asymmetry is a loss,
     * because travel goes as `1/k` and that is convex. The gait's own
     * broadcast moved matter along the same wires and kicked nothing. It does
     * now, and the swing turns from a cost into a gain:
     *
     *     gripSwing      before      after
     *          0.00    4.217       3.969   <- control
     *          0.50    2.623       4.032
     *          0.75    3.462       5.091   <- +28% on the control
     *          1.00    1.876       2.719
     *
     * **One seed, and that is not a shortcut.** Nothing in this rig calls
     * `Math.random`: the worm is built to a plan, there is no noise, no
     * spawning and no immigration, and `runFree` swaps in a seeded generator
     * that nothing draws from. Eight seeds return the same numbers as three to
     * four significant figures. So these are exact readings of a deterministic
     * rig rather than a sample of a noisy one — which means the caveat is
     * generalisation, not variance, and that averaging over seeds here costs
     * time and buys nothing.
     */
    const fed: Partial<Params> = {
      grip: 2,
      transportRecoil: 100,
      transportThrust: 0,
      // The reactor on, still eating nothing: `primeGut` supplies it.
      metabolicRate: defaultParams().metabolicRate,
    };
    // Small enough that digestion's whole mouthful is inside what `intake`
    // routes to the reactor, so the tank never sees any of it.
    const PRIME = 0.004;
    console.log('\n  gut primed at ' + PRIME + ', gradient quanta, income 3, recoil 100, 30 s');
    console.log('\n  gripSwing      dx     dy   px/s      moved  gaitLag  agree  gaitSwing');
    console.log('  --------- ------- ------ ------ ---------- -------- ------ ----------');
    // One seed, because there is nothing for a seed to vary — see below.
    const seeds = [1];
    for (const swing of [0, 0.25, 0.5, 0.75, 1]) {
      let px = 0;
      let dx = 0;
      let dy = 0;
      let moved = 0;
      let glag = 0;
      let gagree = 0;
      let gswing = 0;
      for (const seed of seeds) {
        const r = runFree({
          quanta: 'gradient', income: 3, seed, primeGut: PRIME,
          params: { ...fed, gripSwing: swing },
        });
        px += Math.hypot(r.dx, r.dy) / 30;
        dx += r.dx;
        dy += r.dy;
        moved += r.moved;
        glag += r.gaitLag;
        gagree += r.gaitAgree;
        gswing += r.gaitSwingSeen;
        expect(Number.isFinite(r.dx)).toBe(true);
      }
      const n = seeds.length;
      console.log(
        `  ${swing.toFixed(2).padStart(9)} ${(dx / n).toFixed(2).padStart(7)} ${(dy / n).toFixed(2).padStart(6)} ` +
          `${(px / n).toFixed(3).padStart(6)} ${(moved / n).toFixed(0).padStart(10)} ` +
          `${(glag / n).toFixed(2).padStart(8)} ${(gagree / n).toFixed(2).padStart(6)} ${(gswing / n).toFixed(3).padStart(10)}`,
      );
    }
    console.log('\n  the control is the 0 row: same fed worm, no swing on the grip.');
  });

  it('runs the hypotheses', () => {
    const CASES: { name: string; spec: RunSpec }[] = [
      { name: 'flat     no grip   ', spec: { drive: 'flat', params: {} } },
      { name: 'wave     no grip   ', spec: { drive: 'wave', params: {} } },
      { name: 'ripple   no grip   ', spec: { drive: 'ripple', params: {} } },
      { name: 'standing grip 2          ', spec: { drive: 'standing', params: { grip: 2 } } },
      { name: 'wave     grip 2          ', spec: { drive: 'wave', params: { grip: 2 } } },
      { name: 'ripple   grip 2          ', spec: { drive: 'ripple', params: { grip: 2 } } },
      { name: 'ripple   grip -1         ', spec: { drive: 'ripple', params: { grip: -1 } } },
    ];
    console.log('\n  the stroke: (recoil 100) - (recoil 0), same worm, same energy');
    console.log('\n  drive    drag              ddx     ddy   px/s     moved');
    console.log('  ------------------------ ------- ------- ------ ---------');
    for (const c of CASES) {
      const r = stroke(c.spec);
      console.log(
        `  ${c.name} ${r.ddx.toFixed(2).padStart(7)} ${r.ddy.toFixed(2).padStart(7)} ` +
          `${r.px.toFixed(3).padStart(6)} ${r.moved.toFixed(0).padStart(9)}`,
      );
      expect(Number.isFinite(r.ddx)).toBe(true);
    }
  });

  it('asks how long the worm takes to stop moving', () => {
    // A tree of distance constraints has a satisfying configuration, so the
    // worm should reach it and stay. How long that takes decides whether a
    // stroke can be measured at all: the creep is the noise floor.
    const params = rigParams();
    const thick = { ...params, drag: 20, angDrag: 20 };
    const sim = new Sim(4000, 4000, 512);
    const worm = buildWorm(sim, params, 2000, 2000);
    for (const a of worm.all) a.extra = 0;
    console.log('\n  settle   com px/s   fastest body px/s');
    console.log('  ------ ---------- -------------------');
    for (const upto of [30, 60, 120, 240]) {
      while (sim.time < upto) sim.step(1 / 60, thick);
      const a = centre(worm);
      for (let f = 0; f < 60; f++) sim.step(1 / 60, params);
      const b = centre(worm);
      let maxv = 0;
      for (const g of worm.all) maxv = Math.max(maxv, Math.hypot(g.vx, g.vy));
      console.log(
        `  ${String(upto).padStart(6)} ${Math.hypot(b.x - a.x, b.y - a.y).toFixed(4).padStart(10)} ${maxv.toFixed(3).padStart(19)}`,
      );
    }
  });

  it('asks whether the kick conserves momentum at all', () => {
    /*
     * With `grip` at zero, one drag rate applies to every body,
     * and `applyTransportRecoil` at `transportThrust` 0 is exactly equal and
     * opposite. Under those two facts the worm's centre cannot move however
     * much energy runs through it — that is the whole reason the two drag
     * terms were built. So the `wave, no grip` row above must be zero,
     * and it is not.
     *
     * If the leak is a real force proportional to the impulse, the stroke per
     * unit of recoil is flat. If it is the solver clamping a correction it
     * cannot apply in one substep — and there are such clamps, sized for
     * pixels per substep — it appears only once the kicks are large, and the
     * ratio climbs with recoil.
     */
    console.log('\n  is the recoil conservative? (no grip, wave drive)');
    console.log('\n  recoil     ddx     ddy   px/s   px/s per unit recoil');
    console.log('  ------ ------- ------- ------ ----------------------');
    for (const recoil of [1, 10, 100, 400]) {
      const on = runWorm({ drive: 'wave', params: { transportRecoil: recoil } });
      const off = runWorm({ drive: 'wave', params: { transportRecoil: 0 } });
      const ddx = on.dx - off.dx;
      const ddy = on.dy - off.dy;
      const px = Math.hypot(ddx, ddy) / 20;
      console.log(
        `  ${String(recoil).padStart(6)} ${ddx.toFixed(2).padStart(7)} ${ddy.toFixed(2).padStart(7)} ` +
          `${px.toFixed(3).padStart(6)} ${(px / recoil).toExponential(2).padStart(22)}`,
      );
    }
  });

  it('is a control that reads zero when nothing is transported', () => {
    // `flat` puts every tank at the same level, so no neighbour is needier
    // than its donor and not one unit crosses. With no transfers there are no
    // kicks, so the two arms of the difference are the same run twice and the
    // stroke has to be identically zero. If this ever reads non-zero the
    // difference is measuring something other than transport.
    const r = stroke({ drive: 'flat' });
    expect(r.moved).toBe(0);
    expect(r.ddx).toBe(0);
    expect(r.ddy).toBe(0);
  });
});
