import { describe, it } from 'vitest';
import type { Params } from '../params.ts';
import type { Sim } from '../sim.ts';
import {
  dress,
  oscillator,
  oscillatorPhase,
  gaitOf,
  gaitTable,
  motorsOff,
  runOrganism,
  type Gait,
  type OrganismSpec,
} from './organism.ts';

/*
 * A worm that swims on its own metabolism.
 *
 * `applyTransportRecoil` is the one force in the sim a net can only generate
 * by moving energy through itself: pumping a charge along a wire shoves the
 * sender back down it, and the receiver's `transportThrust` decides how much
 * of that kick it hands back. At thrust 1 the receiver cancels nothing, so the
 * pair keeps the whole impulse and it points from receiver toward sender — a
 * chain that takes energy in at the nose and spends it at the tail is pushed
 * nose-first. That is a muscle in the only sense this world offers one: the
 * stroke costs energy, it is driven by the internal demand field rather than by
 * any body's own port, and no single body can perform it.
 *
 * Its own comment says the effect has only ever been seen on events — a latch,
 * a rescue, a refill — because a topped-up soup moves ~4e-4 a frame, and that
 * the case nobody has run is "a net held under a real gradient". This is that
 * case.
 *
 * Every other motor is off (see `motorsOff`), so displacement is the pump.
 * Two controls hold it to that:
 *
 *   recoil 0   no kick at all. Energy still flows; the worm must not move.
 *   thrust 0   the receiver cancels the sender exactly. Energy still flows and
 *              both ends are still kicked; only the *net* is zero. The sharper
 *              of the two: it fails if the motion were coming from transfers
 *              disturbing the solver rather than from momentum they inject.
 *
 *     npm run experiment -- worm
 *     EXP_SECONDS=120 EXP_SEGMENTS=12 npm run experiment -- worm
 */

const SEGMENTS = Number(process.env.EXP_SEGMENTS ?? 8);
const SECONDS = Number(process.env.EXP_SECONDS ?? 40);
/**
 * Seeds per condition, which the build jitter makes mean something.
 *
 * Load-bearing rather than diligence: with one seed a tumbling worm and a
 * swimming one are indistinguishable, because a tumbler still ends up
 * somewhere and that somewhere is as likely to be forward as not. The
 * separation is in the spread, not the mean.
 */
const SEEDS = [1, 2, 3, 4, 5];

/*
 * Why the ground is switched off, which is the first thing that surprised this
 * bench.
 *
 * The obvious way to build the gradient is metabolic: give the tail the farm
 * gene so it dumps stock onto the dish, let the head graze, and the worm eats
 * at one end and excretes at the other. Measured, that moves *nothing at all*
 * — `moved` and `hops` both flat zero over ten seconds, in every condition.
 *
 * The reason is that farming deposits at `a.x, a.y` and harvesting takes from
 * the cell a body is standing in, so a farming body is standing in its own
 * excretion and grazes it straight back the next frame. Worse, every segment
 * is on ground of its own, so every segment tops itself up to cap for free and
 * *no* segment can be in deficit. `flowCharges` says a flat field is a stall
 * by design, and a pond where everybody is standing on food is a flat field.
 *
 * A metabolic gradient in this sim therefore needs the sink to be somewhere
 * the source is not, and the ground is everywhere. So the bench does it the
 * declared way instead — no ground at all, a puppet feeder pinning the head at
 * its cap, and upkeep as the sink — and the ecological version is left as its
 * own question. The `ground` row keeps the null result where it can be seen.
 */
function bench(): Partial<Params> {
  return {
    ...motorsOff(),
    ambientEnergy: 0,
    energyRegrow: 0,
    farmRate: 0,
    upkeep: 0.15,
  };
}

interface Condition {
  label: string;
  /**
   * Metered uptake, and with it the ground as a supply. At 0 the old
   * take-what-fits path runs and a body on ground tops up to cap in one frame,
   * which is the flat field the `ground on` control shows stalling.
   */
  uptakeVmax?: number;
  /** Expression: which segments carry a mouth, counted from the head. 0 = every body eats alike. */
  mouths?: number;
  /** Initial speed along the build axis, to ask whether a gait sustains as against self-starts. */
  kick?: number;
  /**
   * The clock. `phase` is the phase step per segment from tail to head, in
   * radians — 0 is every segment spending together, positive and negative are
   * the two directions a wave can run.
   */
  cpg?: { gain: number; step: number; amplitude: number; phase: number; drive: number };
  /**
   * Where the one segment that can eat sits.
   *
   * `head` gives the worm a standing head-to-tail gradient whatever the clock
   * is doing, and that gradient turns out to dominate — the source is fixed, so
   * every transfer ultimately runs rearward and the clock only jitters the
   * timing. `middle` removes it: energy enters amidships with equal distances
   * to both ends, the standing gradient is symmetric and cancels, and the only
   * thing left that can decide which way matter travels is the phase of the
   * wave. That is the condition under which the sign of `phase` is a
   * prediction rather than a hope.
   */
  mouthAt?: 'head' | 'middle';
  excreteRate?: number;
  /**
   * The stroke: every segment pushes out of one aux port, gated by its own
   * clock, so the wave along the body is a wave of *thrust* rather than a wave
   * of shortage. `pushRate` scales it; `pushSlot` picks the port and so the
   * direction (0 = principal, forward-facing; 1 = an aux, rearward-facing).
   */
  pushRate?: number;
  pushSlot?: number;
  transportSpeed?: number;
  thrust?: number;
  recoil?: number;
  segments?: number;
  /** Where a rescue aims, and so how big one packet is. Lower is a finer stroke. */
  rescueTo?: number;
  portStiff?: number;
  params?: Partial<Params>;
  /** Pin the head at its cap every frame, so the fuel supply is not what is under test. */
  feed?: boolean;
  /**
   * Pin *every* segment at its cap every frame.
   *
   * The actuator with the economy taken out from under it. Push is bounded by
   * what a body holds above break-even, so a starving segment cannot pump —
   * and in the obligate configurations it starves, which confounds the one
   * question this is asking. With every tank held full nobody is short, so
   * `flowCharges` has nothing to do and every transfer in the run is a push.
   * Whatever moves, the `PUSH` head moved.
   */
  feedAll?: boolean;
}

function run(c: Condition, seed: number): Gait {
  const segments = c.segments ?? SEGMENTS;
  const thrust = c.thrust ?? 1;
  const recoil = c.recoil ?? 100;
  // Ground-fed conditions name a `uptakeVmax`; the motor bench feeds by puppet.
  const feed = c.feedAll ? false : (c.feed ?? c.uptakeVmax === undefined);
  const spec: OrganismSpec = {
    seconds: SECONDS,
    seed,
    sampleEvery: c.cpg ? 0.1 : 2,
    worm: { segments, kinds: 'con', heading: 0, jitter: 0.08 },
    params: {
      ...bench(),
      portStiff: c.portStiff ?? 2,
      ...(c.uptakeVmax !== undefined
        ? { uptakeVmax: c.uptakeVmax, ambientEnergy: 1, energyRegrow: 0.04, upkeep: 0.015 }
        : {}),
      ...(c.excreteRate !== undefined ? { excreteRate: c.excreteRate } : {}),
      ...(c.pushRate !== undefined ? { pushRate: c.pushRate } : {}),
      ...(c.transportSpeed !== undefined ? { transportSpeed: c.transportSpeed } : {}),
      ...(c.params ?? {}),
    },
    dressWorm: (sim: Sim, ids: number[]) => {
      for (let i = 0; i < ids.length; i++) {
        const a = sim.agents.get(ids[i]!);
        if (!a) continue;
        const isHead = i === ids.length - 1;
        const isMouth = c.mouths !== undefined && c.mouths > 0 && i >= ids.length - c.mouths;
        dress(a, {
          // Mute and tasteless. Steering is off anyway, but an empty field
          // keeps the state pass reading zeros, so `h` stays at phi(0) and
          // every head is exactly the constant written here.
          emit: {},
          taste: {},
          thrust,
          recoil,
          cruise: 0,
          turn: 0,
          align: 0,
          sep: 0,
          ...(c.rescueTo !== undefined ? { rescueTo: c.rescueTo } : {}),
          /*
           * A mouth is one row of the expression simplex. Writing it silences
           * the other seven, which is the point: a segment that is not a mouth
           * expresses nothing on uptake, cannot draw from the ground under it,
           * and has to be fed through the wires or die. That is §5's obligate
           * trophic dependency, built rather than dialled.
           */
          /*
           * The clock, the mouth, and the two rows the clock moves effort
           * between.
           *
           * Three things had to line up, and each was found by its failure.
           *
           * A rhythm has to move share *between* rows: `expressVector`
           * normalises over the eight, so one oscillating row keeps a share of
           * 1 and does nothing at all. So uptake and excretion are driven in
           * antiphase off the same state dim — as `h[2]` rises the segment
           * shifts from taking the ground in to putting `aux` back out.
           *
           * The body cannot be allowed to eat. With every segment on ambient
           * ground the clock ran perfectly and changed nothing: uptake at
           * `vmax * ROW_COUNT * share` is around 9.6 a second against an
           * excretion of 0.35, so the tank sat pinned at cap and no demand ever
           * appeared. That is §0's finding — a body on ambient ground is a
           * complete self-sufficient organism — reappearing inside the muscle.
           * So every segment but the mouth gets a deliberately terrible
           * affinity gene, which is the `(vmax, ks)` axis pushed to one end: it
           * still expresses a transporter, the transporter is just no good, and
           * its only real income is what the wires bring.
           *
           * And the tank has to be small. Excretion is mass action on the tank,
           * so it asymptotes at zero and can never by itself put a body into
           * debt — and `rescueNeed`, which is what the demand field actually
           * listens to, only latches below zero. A full 1.25 tank drains toward
           * zero and then waits on upkeep at 0.015 a second, which is a minute
           * of nothing. The cycle time has to be brought to the clock, not the
           * other way round.
           */
          ...(c.cpg
            ? (c.mouthAt === 'middle' ? i === Math.floor(ids.length / 2) : isHead)
              ? { uptake: { energy: 1 } }
              : {
                  wh: oscillator(c.cpg.gain, c.cpg.step),
                  h: oscillatorPhase(c.cpg.amplitude, i * c.cpg.phase),
                  ks: { energy: 2000 },
                  energyCap: 0.4,
                  debtCap: -0.2,
                  uptake: { energy: 1 },
                  excrete: { aux: 1 },
                  uptakeGain: { energy: [0, 0, -c.cpg.drive, 0] },
                  excreteGain: { aux: [0, 0, c.cpg.drive, 0] },
                  ...(c.pushRate
                    ? {
                        // Base at half, driven by the same state dim the
                        // metabolism is on, so a segment pushes hardest at one
                        // point in its cycle and not at all at the other. The
                        // phase offset between segments is then a phase offset
                        // between strokes.
                        push: [0, 0, 0].map((_, k) => (k === (c.pushSlot ?? 1) ? 0.5 : 0)),
                        pushGain: [0, 1, 2].map((k) =>
                          k === (c.pushSlot ?? 1) ? [0, 0, 1.5, 0] : [0, 0, 0, 0],
                        ),
                      }
                    : {}),
                }
            : {}),
          ...(c.mouths !== undefined && c.mouths > 0
            ? isMouth
              ? { uptake: { energy: 1 } }
              : { excrete: { aux: 1 } }
            : {}),
          // Primed: the body of the worm starts a hair in debt, which latches
          // `recovering` on frame one. Otherwise upkeep takes 80 s to walk a
          // full tank under break-even and the run measures the wait.
          extra: c.cpg
            ? (c.mouthAt === 'middle' ? i === Math.floor(ids.length / 2) : isHead)
              ? 1.25
              : 0.4
            : isHead
              ? 1.25
              : -0.05,
        });
        if (c.kick) {
          a.vx = c.kick;
        }
      }
    },
    drive: c.feedAll
      ? (sim: Sim, ids: number[]) => {
          for (const id of ids) {
            const a = sim.agents.get(id);
            if (a) a.extra = a.energyCap * 0.5;
          }
        }
      : !feed
        ? undefined
        : feed
        ? (sim: Sim, ids: number[]) => {
            const head = sim.agents.get(ids[ids.length - 1]!);
            if (head) head.extra = head.energyCap;
          }
        : undefined,
  };
  return gaitOf(runOrganism(spec));
}

/**
 * Mean of a condition over the seeds, plus the one number a mean cannot carry.
 *
 * `speedSd` is the spread of the *speed* across seeds. A directed swimmer
 * covers about the same ground however it was nudged at build; a tumbler's
 * distance is a random walk and its spread is the same order as its mean.
 * Read it beside `headward`, which says whether the ground it covered was in
 * the direction the body points.
 */
function meanGait(c: Condition): { gait: Gait; speedSd: number } {
  const gaits = SEEDS.map((seed) => run(c, seed));
  const n = gaits.length;
  const avg = (f: (g: Gait) => number): number => gaits.reduce((s, g) => s + f(g), 0) / n;
  const speed = avg((g) => g.speed);
  const speedSd = Math.sqrt(Math.max(0, avg((g) => g.speed * g.speed) - speed * speed));
  return {
    gait: {
      along: avg((g) => g.along),
      perp: avg((g) => g.perp),
      speed,
      straightness: avg((g) => g.straightness),
      moved: avg((g) => g.moved),
      hops: Math.round(avg((g) => g.hops)),
      pumpImpulse: avg((g) => g.pumpImpulse),
      costOfTransport: avg((g) => (Number.isFinite(g.costOfTransport) ? g.costOfTransport : 0)),
      pumpEfficiency: avg((g) => g.pumpEfficiency),
      headward: avg((g) => g.headward),
      demandTilt: avg((g) => g.demandTilt),
      tankSwing: avg((g) => g.tankSwing),
      waveLag: avg((g) => g.waveLag),
      bendSwing: avg((g) => g.bendSwing),
      gapSwing: avg((g) => g.gapSwing),
      intact: gaits.every((g) => g.intact),
    },
    speedSd,
  };
}

function report(title: string, conditions: Condition[]): void {
  const rows: { label: string; gait: Gait }[] = [];
  for (const c of conditions) {
    const { gait, speedSd } = meanGait(c);
    rows.push({ label: c.label, gait });
    console.log(
      `  ${c.label.padEnd(18)} ${gait.speed.toFixed(2).padStart(6)} +/- ${speedSd.toFixed(2).padStart(5)} px/s  ` +
        `headward=${gait.headward.toFixed(2).padStart(5)} straight=${gait.straightness.toFixed(2)} intact=${gait.intact}`,
    );
  }
  console.log(`\n${title} (mean of ${SEEDS.length} seeds)\n${gaitTable(rows)}\n`);
}

describe('experiment: a worm that swims on transport', () => {
  it('pumps, and stops when either half of the pump is removed', () => {
    report('controls', [
      { label: 'pump' },
      { label: '  thrust 0', thrust: 0 },
      { label: '  recoil 0', recoil: 0 },
      { label: '  no feeder', feed: false },
      { label: '  ground on', params: { ambientEnergy: 1, energyRegrow: 0.04, farmRate: 1 } },
    ]);
  });

  it('sweeps the stroke', () => {
    report('gain', [
      { label: 'recoil 25', recoil: 25 },
      { label: 'recoil 50', recoil: 50 },
      { label: 'recoil 100', recoil: 100 },
      { label: 'recoil 200', recoil: 200 },
    ]);
    report('packet size (rescueTo)', [
      { label: 'rescueTo 0.2', rescueTo: 0.2 },
      { label: 'rescueTo 0.5', rescueTo: 0.5 },
      { label: 'rescueTo 0.9', rescueTo: 0.9 },
    ]);
    report('backbone (portStiff)', [
      { label: 'stiff 2', portStiff: 2 },
      { label: 'stiff 8', portStiff: 8 },
      { label: 'stiff 20', portStiff: 20 },
    ]);
    report('length', [
      { label: '4 segments', segments: 4 },
      { label: '8 segments', segments: 8 },
      { label: '16 segments', segments: 16 },
    ]);
  });
});

describe('experiment: can the ground drive the pump?', () => {
  /*
   * The gradient without a puppet.
   *
   * Metered uptake (`uptakeVmax > 0`) is the change that makes this askable at
   * all. Under the old take-what-fits harvest a body on ground filled to cap in
   * one frame, so no segment could be in deficit and the demand field was flat
   * — the `ground on` control above, at zero hops. Monod makes income a rate,
   * so income collapses where the ground is thin.
   *
   * The hypothesis: a *moving* worm strips the ground it passes over, so its
   * rear sits in its own grazed wake while its nose is over fresh ground. That
   * is a head-to-tail gradient made by nothing but motion, and the recoil it
   * produces points nose-first — which makes more wake. Read `tilt` first: it
   * is the cause, and speed without a positive tilt is not this mechanism.
   *
   * Three questions, because they have different answers:
   *   self-start  a still worm. Is there any asymmetry to amplify?
   *   sustain     given an initial shove. Does the wake keep it going?
   *   mouth       expression makes the head the only segment that can eat, so
   *               the gradient does not wait on motion at all.
   */
  it('grazes its own wake', () => {
    report('self-start (still, uniform)', [
      { label: 'vmax 0.25', uptakeVmax: 0.25 },
      { label: 'vmax 1', uptakeVmax: 1 },
      { label: 'vmax 4', uptakeVmax: 4 },
    ]);
    report('sustain (kicked)', [
      { label: 'vmax 1 kick 60', uptakeVmax: 1, kick: 60 },
      { label: 'vmax 4 kick 60', uptakeVmax: 4, kick: 60 },
      { label: 'vmax 4 kick 120', uptakeVmax: 4, kick: 120 },
    ]);
    report('mouth (expression)', [
      { label: '1 mouth', uptakeVmax: 4, mouths: 1 },
      { label: '2 mouths', uptakeVmax: 4, mouths: 2 },
      { label: '1 mouth vmax 16', uptakeVmax: 16, mouths: 1 },
    ]);
  });
});

describe('experiment: a muscle the net drives itself', () => {
  /*
   * Everything above is the net conducting a gradient somebody else made — a
   * feeder at the nose, or ground that happens to be thinner behind. The net is
   * a pipe, and locomotion is a side effect of eating. A muscle is the other
   * thing: the net decides when to spend, on a clock it keeps itself, and the
   * spending is what moves it.
   *
   * The clock is `Wh` as a rotation with gain, which limit-cycles under `phi`
   * (see `oscillator`). It runs in state dims 2 and 3, where nothing else
   * lives. The expression head turns that clock into metabolism: each segment
   * shifts effort between taking energy in and putting `aux` back out, in
   * antiphase, so it alternately fills and empties. A phase offset per segment
   * makes that a wave along the body, and a wave means that at any instant some
   * segments are flush and others are short — which is a demand gradient the
   * net made, on its own schedule, out of its own state.
   *
   * The prediction that makes this falsifiable: `phase 0` is every segment
   * spending at once, so the field stays flat and there is no stroke. A
   * positive and a negative phase step are the same worm with the wave running
   * the other way, and should swim in opposite directions. `lag` is the metric
   * that shows the wave, `tank` that there is a rhythm at all, and `headward`
   * which way it went.
   */
  const CPG = { gain: 1.2, step: 0.05, amplitude: 0.3, drive: 1.5, phase: 0 };
  const base = { uptakeVmax: 4, excreteRate: 0.2, params: { upkeep: 0.05 } };

  it('pushes on a clock, and the phase decides the direction', () => {
    const A = {
      recoil: 25,
      pushRate: 1.5,
      feedAll: true,
      params: { upkeep: 0, ambientEnergy: 0 },
    };
    report('the actuator alone: every tank held full, so every transfer is a push', [
      { label: 'aux, phase 0', ...A, pushSlot: 1, cpg: { ...CPG, phase: 0 } },
      { label: 'aux, +pi/2', ...A, pushSlot: 1, cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'aux, -pi/2', ...A, pushSlot: 1, cpg: { ...CPG, phase: -Math.PI / 2 } },
      { label: 'nose, phase 0', ...A, pushSlot: 0, cpg: { ...CPG, phase: 0 } },
      { label: 'nose, +pi/2', ...A, pushSlot: 0, cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'push off', ...A, pushRate: 0, pushSlot: 1, cpg: { ...CPG, phase: Math.PI / 2 } },
    ]);
    const P = { ...base, recoil: 25, pushRate: 1.5, transportSpeed: 6 };
    report('directed push, mouth amidships', [
      { label: 'push aux, phase 0', ...P, mouthAt: 'middle', pushSlot: 1, cpg: { ...CPG, phase: 0 } },
      { label: 'push aux, +pi/2', ...P, mouthAt: 'middle', pushSlot: 1, cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'push aux, -pi/2', ...P, mouthAt: 'middle', pushSlot: 1, cpg: { ...CPG, phase: -Math.PI / 2 } },
      { label: 'push nose, +pi/2', ...P, mouthAt: 'middle', pushSlot: 0, cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'no push, +pi/2', ...base, recoil: 25, mouthAt: 'middle', cpg: { ...CPG, phase: Math.PI / 2 } },
    ]);
  });

  it('runs a wave along itself', () => {
    report('phase gradient, mouth amidships, recoil 25', [
      { label: 'phase 0 (sync)', ...base, recoil: 25, mouthAt: 'middle', cpg: { ...CPG, phase: 0 } },
      { label: 'phase +pi/2', ...base, recoil: 25, mouthAt: 'middle', cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'phase -pi/2', ...base, recoil: 25, mouthAt: 'middle', cpg: { ...CPG, phase: -Math.PI / 2 } },
    ]);
    report('phase gradient, mouth at the nose, recoil 25', [
      { label: 'phase 0 (sync)', ...base, recoil: 25, cpg: { ...CPG, phase: 0 } },
      { label: 'phase +pi/2', ...base, recoil: 25, cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'phase -pi/2', ...base, recoil: 25, cpg: { ...CPG, phase: -Math.PI / 2 } },
    ]);
    report('phase gradient, recoil 100', [
      { label: 'phase 0 (sync)', ...base, cpg: { ...CPG, phase: 0 } },
      { label: 'phase +pi/2', ...base, cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'phase -pi/2', ...base, cpg: { ...CPG, phase: -Math.PI / 2 } },
      { label: 'phase +pi/4', ...base, cpg: { ...CPG, phase: Math.PI / 4 } },
      { label: 'no clock (mouth)', ...base, mouths: 1 },
    ]);
    report('spend rate', [
      { label: 'excrete 0.05', ...base, excreteRate: 0.05, cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'excrete 0.2', ...base, excreteRate: 0.2, cpg: { ...CPG, phase: Math.PI / 2 } },
      { label: 'excrete 0.6', ...base, excreteRate: 0.6, cpg: { ...CPG, phase: Math.PI / 2 } },
    ]);
  });
});
