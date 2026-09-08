import { describe, it } from 'vitest';
import type { Params } from '../params.ts';
import type { Sim } from '../sim.ts';
import {
  dress,
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
  thrust?: number;
  recoil?: number;
  segments?: number;
  /** Where a rescue aims, and so how big one packet is. Lower is a finer stroke. */
  rescueTo?: number;
  portStiff?: number;
  params?: Partial<Params>;
  /** Pin the head at its cap every frame, so the fuel supply is not what is under test. */
  feed?: boolean;
}

function run(c: Condition, seed: number): Gait {
  const segments = c.segments ?? SEGMENTS;
  const thrust = c.thrust ?? 1;
  const recoil = c.recoil ?? 100;
  const feed = c.feed ?? true;
  const spec: OrganismSpec = {
    seconds: SECONDS,
    seed,
    sampleEvery: 2,
    worm: { segments, kinds: 'con', heading: 0, jitter: 0.08 },
    params: { ...bench(), portStiff: c.portStiff ?? 2, ...(c.params ?? {}) },
    dressWorm: (sim: Sim, ids: number[]) => {
      for (let i = 0; i < ids.length; i++) {
        const a = sim.agents.get(ids[i]!);
        if (!a) continue;
        const isHead = i === ids.length - 1;
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
          // Primed: the body of the worm starts a hair in debt, which latches
          // `recovering` on frame one. Otherwise upkeep takes 80 s to walk a
          // full tank under break-even and the run measures the wait.
          extra: isHead ? 1.25 : -0.05,
        });
      }
    },
    drive: feed
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
