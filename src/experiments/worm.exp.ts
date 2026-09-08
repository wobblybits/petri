import { describe, it } from 'vitest';
import type { Params } from '../params.ts';
import type { Sim } from '../sim.ts';
import {
  dress,
  gaitOf,
  gaitTable,
  motorsOff,
  oscillator,
  oscillatorPhase,
  runOrganism,
  type Gait,
  type OrganismSpec,
} from './organism.ts';

/*
 * A worm with one actuator.
 *
 * Everything before this tried to make a net swim by moving matter about
 * inside it — a transport recoil, a momentum-splitting coefficient, a directed
 * push. All three are gone, and the reason is in `docs/transport-worm.md`: an
 * internal transfer cannot create momentum, so the worm that appeared to swim
 * at 36 px/s was riding a momentum pump, and the control that read as a clean
 * falsification was the physically correct configuration going nowhere.
 *
 * What is left is the only honest arrangement. `ANGLE` sets a joint's rest
 * angle, which changes the net's *shape* and conserves its momentum exactly.
 * `dragAniso` is the medium noticing which way a body lies, which is what turns
 * a shape change into travel. A clock on the state matrices makes the shape
 * change periodic, and a phase offset per segment makes it a wave.
 *
 * The prediction, and the reason this is worth running: with an isotropic
 * medium a phase gradient must do nothing, because displacement cannot depend
 * on *when* an internal force acted. With an anisotropic one it must, and the
 * two directions of the wave must swim opposite ways. That is the first time
 * in this work that phase has had a mechanism to matter through.
 *
 *     npm run experiment -- worm
 */

const SEGMENTS = Number(process.env.EXP_SEGMENTS ?? 8);
const SECONDS = Number(process.env.EXP_SECONDS ?? 45);
const SEEDS = [1, 2, 3, 4, 5];

/** The clock: gain 1.2 and 0.05 rad a frame is a 4 s cycle. See `oscillator`. */
const CPG = { gain: 1.2, step: 0.05, amplitude: 0.3 };

interface Condition {
  label: string;
  /** Radians of rest angle commanded at the peak of the cycle. */
  amplitude?: number;
  /** Phase step per segment, tail to head. 0 is every joint bending together. */
  phase?: number;
  jointStiff?: number;
  dragAniso?: number;
  bendCost?: number;
  segments?: number;
  portStiff?: number;
  params?: Partial<Params>;
}

function run(c: Condition, seed: number): Gait {
  const segments = c.segments ?? SEGMENTS;
  const amplitude = c.amplitude ?? 0.6;
  const spec: OrganismSpec = {
    seconds: SECONDS,
    seed,
    sampleEvery: 0.1,
    worm: { segments, kinds: 'con', heading: 0, jitter: 0.05 },
    params: {
      ...motorsOff(),
      upkeep: 0,
      ambientEnergy: 0,
      jointStiff: c.jointStiff ?? 1,
      dragAniso: c.dragAniso ?? 1,
      bendCost: c.bendCost ?? 0,
      ...(c.portStiff !== undefined ? { portStiff: c.portStiff } : {}),
      ...(c.params ?? {}),
    },
    dressWorm: (sim: Sim, ids: number[]) => {
      for (let i = 0; i < ids.length; i++) {
        const a = sim.agents.get(ids[i]!);
        if (!a) continue;
        dress(a, {
          emit: {},
          taste: {},
          cruise: 0,
          turn: 0,
          align: 0,
          sep: 0,
          extra: a.energyCap,
          wh: oscillator(CPG.gain, CPG.step),
          h: oscillatorPhase(CPG.amplitude, i * (c.phase ?? 0)),
          /*
           * The muscle. Row 0 is the principal, which is the port every segment
           * but the head is wired forward through, so commanding it bends this
           * segment against the one ahead. Driven off state dim 2, where the
           * clock runs, with no base — so the joint's *rest* position is
           * straight and the cycle swings it either side of that.
           */
          angleGain: [[0, 0, amplitude / CPG.amplitude, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        });
      }
    },
  };
  return gaitOf(runOrganism(spec));
}

function meanGait(c: Condition): { gait: Gait; speedSd: number } {
  const gaits = SEEDS.map((seed) => run(c, seed));
  const n = gaits.length;
  const avg = (f: (g: Gait) => number): number => gaits.reduce((s, g) => s + f(g), 0) / n;
  const speed = avg((g) => g.speed);
  return {
    gait: {
      along: avg((g) => g.along),
      perp: avg((g) => g.perp),
      speed,
      straightness: avg((g) => g.straightness),
      moved: avg((g) => g.moved),
      hops: Math.round(avg((g) => g.hops)),
      costOfTransport: avg((g) => (Number.isFinite(g.costOfTransport) ? g.costOfTransport : 0)),
      headward: avg((g) => g.headward),
      demandTilt: avg((g) => g.demandTilt),
      tankSwing: avg((g) => g.tankSwing),
      jointSwing: avg((g) => g.jointSwing),
      jointDrive: avg((g) => g.jointDrive),
      jointTrack: avg((g) => g.jointTrack),
      jointLag: avg((g) => g.jointLag),
      bendSwing: avg((g) => g.bendSwing),
      gapSwing: avg((g) => g.gapSwing),
      intact: gaits.every((g) => g.intact),
    },
    speedSd: Math.sqrt(Math.max(0, avg((g) => g.speed * g.speed) - speed * speed)),
  };
}

function report(title: string, conditions: Condition[]): void {
  const rows: { label: string; gait: Gait }[] = [];
  for (const c of conditions) {
    const { gait, speedSd } = meanGait(c);
    rows.push({ label: c.label, gait });
    console.log(
      `  ${c.label.padEnd(20)} ${gait.speed.toFixed(2).padStart(6)} +/- ${speedSd.toFixed(2).padStart(5)} px/s  ` +
        `along=${gait.along.toFixed(0).padStart(6)} headward=${gait.headward.toFixed(2).padStart(5)} ` +
        `bend=${gait.bendSwing.toFixed(2)}`,
    );
  }
  console.log(`\n${title} (mean of ${SEEDS.length} seeds)\n${gaitTable(rows)}\n`);
}

describe('experiment: a worm with one actuator', () => {
  it('sweeps the muscle against the servo that holds the chain straight', () => {
    /*
     * `portTorques` runs at `portStiff * 320`, so the shipped 2 is a gain of
     * 640 aiming every port at its neighbour — against which a joint stiffness
     * of 300 is not obviously the louder voice. If the aiming servo is
     * swallowing the stroke, softening it is what lets the stroke out.
     */
    for (const portStiff of [2, 0.25]) {
      report(`portStiff ${portStiff}, aniso 3`, [
        { label: 'rigid 0.3, +pi/2', portStiff, jointStiff: 0.3, dragAniso: 3, phase: Math.PI / 2 },
        { label: 'rigid 0.3, -pi/2', portStiff, jointStiff: 0.3, dragAniso: 3, phase: -Math.PI / 2 },
        { label: 'rigid 1.0, +pi/2', portStiff, jointStiff: 1, dragAniso: 3, phase: Math.PI / 2 },
        { label: 'rigid 1.0, -pi/2', portStiff, jointStiff: 1, dragAniso: 3, phase: -Math.PI / 2 },
        { label: 'rigid 1.0, no muscle', portStiff, jointStiff: 1, dragAniso: 3, amplitude: 0 },
      ]);
    }
  });

  it('asks whether phase can matter now that the medium notices', () => {
    /*
     * `along` is the number to read, with its sign: the two phase directions
     * should swim opposite ways if the wave is doing the work. Speed alone
     * cannot tell a gait from a thrash, and `headward` cannot either when the
     * body is turning.
     */
    report('isotropic medium (dragAniso 1)', [
      { label: 'phase 0', dragAniso: 1, phase: 0 },
      { label: 'phase +pi/2', dragAniso: 1, phase: Math.PI / 2 },
      { label: 'phase -pi/2', dragAniso: 1, phase: -Math.PI / 2 },
    ]);
    report('anisotropic medium (dragAniso 3)', [
      { label: 'phase 0', dragAniso: 3, phase: 0 },
      { label: 'phase +pi/2', dragAniso: 3, phase: Math.PI / 2 },
      { label: 'phase -pi/2', dragAniso: 3, phase: -Math.PI / 2 },
      { label: 'phase +pi/4', dragAniso: 3, phase: Math.PI / 4 },
    ]);
    report('controls, aniso 3 and +pi/2', [
      { label: 'no joint', dragAniso: 3, phase: Math.PI / 2, jointStiff: 0 },
      { label: 'no muscle', dragAniso: 3, phase: Math.PI / 2, amplitude: 0 },
      { label: 'full', dragAniso: 3, phase: Math.PI / 2 },
    ]);
  });
});
