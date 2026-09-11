import { describe, expect, it } from 'vitest';
import type { Agent } from '../agents.ts';
import { EXTRA_FLOOR } from '../energy.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';

/*
 * One worm on a bench, with its energy under external control: one net,
 * built by hand, with no sensing, steering, grazing, rewriting, immigration
 * or signalling. Only the wire constraint, the drag law, and energy moving
 * along wires with the recoil that carries. Every body's tank is written
 * each frame from a policy, so throughput is an independent variable.
 *
 * Three strands of 40 with a leaf Era at every segment (an Era has one port
 * and is 0.45 to a node's 1, so the light mass is on the outside).
 *
 *     npm run experiment -- worm
 *
 * The measured quantity is the worm's centre of mass; under one drag law and
 * no thrust it cannot move at all, whatever energy does inside.
 */

/** Segments along the worm. */
const SEGMENTS = 40;
/** Strands: A and C are the periphery, B the core. */
const STRANDS = 3;
/**
 * Spacing along the worm and between strands, and the wires' rest length.
 * Wide enough that bodies never touch: contact radiates, and
 * `applyRadiationLoss` clamps each body's share against its own speed, which
 * moves the centre of mass and would foul the `flat` control.
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
 * `rewriteDuration` 0 is the hard kill on rewriting; the worm is also wired
 * so no two principal ports face each other.
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
  p.fertilise = 0;
  p.uptakeVmax = 0;
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
  p.excreteRate = 0;
  p.sense = 0;
  // No steering, no self-propulsion, no shoaling; at zero the locomotion head is inert.
  p.stepSpeed = 0;
  p.turnRate = 0;
  p.swimNoise = 0;
  p.flockAlign = 0;
  p.flockSep = 0;
  p.declutter = 0;
  // Fixed weights, and a wire that holds its length at the bench's spacing.
  p.learnRate = 0;
  p.wireBreathe = 0;
  p.wireSnap = 0;
  p.wireMinRest = PITCH;
  return { ...p, ...over };
}

/**
 * Three strands with a leaf Era at every segment. Along a strand, segment
 * `i`'s principal joins segment `i+1`'s left aux (principal to aux, so not a
 * redex); the right aux carries a rung or an Era.
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
   * Rungs at two segments only, so the wire graph is a tree. Every rung
   * closes a cycle of distance constraints, and a frustrated cycle makes the
   * worm creep for as long as it is run.
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
 * 0 is `EXTRA_FLOOR`, as deep in debt as the pond allows, not an empty tank:
 * with rewriting off the only claim left is `rescueNeed`, so a body holding
 * any non-negative amount asks for nothing and a drive that wants transport
 * has to put one end of the worm under water. Written over every body every
 * frame, so transport never feeds back into what is available.
 */
export type Drive = (seg: number, t: number, isEra: boolean) => number;

export const DRIVES: Record<string, Drive> = {
  /** Control: everyone full, no gradient, nothing to transport. */
  flat: () => 1,
  /** A fixed gradient, head to tail. */
  standing: (seg) => 1 - seg / (SEGMENTS - 1),
  /** The same gradient, travelling: peristalsis, at one wavelength per worm. */
  wave: (seg, t) => 0.5 + 0.5 * Math.sin(2 * Math.PI * (seg / SEGMENTS - t / WAVE_PERIOD)),
  /** Four waves along the body at once. */
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
  /** Largest distance of any body from the worm's centre. */
  spread: number;
  /** Units of energy transport actually delivered, over the whole window. */
  moved: number;
  /** Centre-of-mass speed still left over from construction, px/s. */
  residual: number;
  bodies: number;
}

/** The suite's LCG; the experiments project seeds nothing, so a bench must. */
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
   * Relax the shape first, heavily overdamped: a hand-built worm is not at
   * mechanical equilibrium (wires are born at their stem span and ramp to
   * `wireMinRest`), and at the pond's own drag the relaxation rings for minutes.
   */
  for (const a of worm.all) a.extra = 0;
  const thick = { ...params, drag: 20, angDrag: 20 };
  for (let f = 0; f < Math.round(settle / dt); f++) sim.step(dt, thick);

  // What the shape is still doing when the clock starts: construction
  // momentum that would otherwise be counted as travel.
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
    // Arrivals: what transport delivered this frame.
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
 * Free running: the worm's own transport, not an imposed pattern. Only the
 * two ends are held, the head at a full tank and the tail in debt; every
 * body between floats on what transport brings it. A body with a quantum
 * accumulates until it holds a packet and then fires, a relaxation oscillator
 * with period quantum over income, so a row with different quanta is a chain
 * of coupled oscillators at different frequencies. The arrangement of quanta
 * is the thing to vary.
 */
export type Quanta = (seg: number, strand: number, isEra: boolean) => number;

export const QUANTA: Record<string, Quanta> = {
  /** Continuous transport: the pond as it ships. */
  none: () => 0,
  /** Every body the same. */
  uniform: () => 0.5,
  /** Core against flanks: Con against Dup, on this worm. */
  kind: (_seg, strand) => (strand === 1 ? 0.5 : 1),
  /** The same two values, scattered. */
  mixed: (seg, strand) => ((seg * 7 + strand * 3) % 5 < 2 ? 0.5 : 1),
  /** A monotonic gradient head to tail. */
  gradient: (seg) => 0.35 + (0.9 * seg) / (SEGMENTS - 1),
};

export interface FreeSpec {
  quanta: keyof typeof QUANTA;
  params?: Partial<Params>;
  seed?: number;
  seconds?: number;
  /**
   * Units per second fed to the head and taken from the tail. Omit for the
   * unmetered bench, which pins the head full and the tail in debt and so
   * supplies whatever the physics will take. A worm of `n` bodies needs
   * `n * upkeep` units a second to stand still.
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
}

/**
 * Lag between neighbouring segments' fullness, by cross-correlation. A
 * standing pulse has lag zero; a travelling wave a consistent non-zero lag
 * whose sign is the direction. `agree` is the fraction of pairs agreeing on
 * the sign, since a mean lag near zero is also "half going each way".
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
    for (let f = 0; f < Math.round(120 / dt); f++) sim.step(dt, thick);

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
    const before = centre(worm);
    let moved = 0;
    const last = new Map<number, number>();
    for (const a of worm.all) last.set(a.id, a.extra);
    const income = spec.income;
    for (let f = 0; f < frames; f++) {
      if (income === undefined) {
        // Unmetered: an infinite source and an infinite sink.
        for (const a of head) a.extra = a.energyCap;
        for (const a of tail) a.extra = EXTRA_FLOOR;
      } else {
        // Metered: a fixed number of units a second in at the head, the same out at the tail.
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
    }
    const after = centre(worm);
    const { lag, agree } = waveLag(series, SEGMENTS, frames);
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
    // No principal faces a principal, so nothing here is a redex.
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
   * The run is seeded and deterministic, so the `transportRecoil` 0 twin has
   * identical geometry, relaxation and drag; the difference is the momentum
   * transport produced, with the settling creep subtracted out.
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
     * Source at the head, sink at the tail, different quanta: does the
     * fullness pattern travel? A flat profile (`swing` near 0) has no phase
     * to measure and its lag is noise.
     */
    console.log('\n  free running: source at the head, sink at the tail, grip 2');
    console.log('\n  quanta      lag  agree  swing     moved   stroke px/s');
    console.log('  -------- ------- ------ ------ --------- -------- ------');
    for (const q of ['none', 'uniform', 'kind', 'mixed', 'gradient'] as const) {
      const on = runFree({ quanta: q, params: { grip: 2, transportRecoil: 100 } });
      const off = runFree({ quanta: q, params: { grip: 2, transportRecoil: 0 } });
      // The same control the driven bench uses: the impulse removed, the creep subtracted.
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
     * Meters the source and sink to the flux a pond could pay (`n * upkeep`
     * units a second to stand still) and asks what is left of the stroke.
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
      // Three seeds, kept to show the rig has no stochastic part: the spread is zero.
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
     * Within a frame the order is drag, then wires, then transport, so the
     * drag law reads the fullness a transfer has already produced: the
     * sender slides, the receiver holds. `transportRecoil` is the impulse;
     * `grip` stops the two ends sharing it equally. Signed dx: energy flows
     * +x, so a negative dx is the worm walking back toward its supply.
     */
    console.log('\n  grip 2, gradient quanta, source at low x so energy flows +x');
    console.log('  metered to 3 units/s, which is this worm at its whole upkeep on the');
    console.log('  pond default; the unmetered bench supplies ~1180.');
    console.log('\n  recoil       dx      dy   px/s     moved');
    console.log('  ------ -------- ------- ------ ---------');
    const rows: number[] = [0, 100];
    for (const transportRecoil of rows) {
      const r = runFree({
        quanta: 'gradient',
        income: 3,
        params: { grip: 2, transportRecoil },
      });
      const px = Math.hypot(r.dx, r.dy) / 30;
      console.log(
        `  ${String(transportRecoil).padStart(6)} ` +
          `${r.dx.toFixed(2).padStart(8)} ${r.dy.toFixed(2).padStart(7)} ${px.toFixed(3).padStart(6)} ${r.moved.toFixed(0).padStart(9)}`,
      );
      expect(Number.isFinite(r.dx)).toBe(true);
    }
  });


  it('asks what would make the stroke an order of magnitude larger', () => {
    /*
     * The pair's centre keeps `|p| * (1/k_sender - 1/k_receiver) / (m_s + m_r)`,
     * so the step is set by the ratio of the anchored end's damping to the
     * sliding end's, `(drag + grip) / drag`, not by the level of either.
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
        params: { drag, grip, transportRecoil: 100 },
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
    // The settling creep is the noise floor under any stroke.
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
     * With `grip` at zero one drag rate applies to every body and the recoil
     * is exactly equal and opposite, so the centre cannot move. A real leak
     * gives a flat stroke per unit of recoil; a solver clamp appears only
     * once the kicks are large, and the ratio climbs with recoil.
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
    // `flat` moves no units, so the two arms are the same run twice and the
    // stroke must be identically zero.
    const r = stroke({ drive: 'flat' });
    expect(r.moved).toBe(0);
    expect(r.ddx).toBe(0);
    expect(r.ddy).toBe(0);
  });
});
