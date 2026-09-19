import { describe, it } from 'vitest';
import { CH, CHANNELS } from '../fields.ts';
import { TASTE, type Agent } from '../agents.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { seededRandom } from './harness.ts';

/*
 * Does a net crawl toward a smell?
 *
 *     npm run experiment -- crawl
 *
 * One chain on a bench, with the only route from the field to its motion being
 * the phase of its own gait. The claim under test is the inchworm's: that a
 * phase offset which is a *function of a food-aligned body coordinate* aims a
 * travelling wave, where a free offset per body cannot. On that bench three
 * profile coefficients beat twenty-eight free offsets 35 captures to 3, and
 * the profile windows were flat where the free ones scattered and went
 * negative, because the frame re-aims itself instead of being relearned.
 *
 * **No steering and no thrust.** `stepSpeed` and `turnRate` are zero, so a
 * body cannot swim at a gradient and cannot turn toward one. What is left is
 * the drag law, the wire constraint, energy moving along wires with the recoil
 * that carries, and the gait — whose phase per body is `gaitProfile` times
 * `store.depth`, and whose depth comes from a relay over a smell the rig
 * paints by hand. If the chain travels, the coordinate is carrying it.
 *
 * **The smell is painted and static.** `groundSmell` and the field's own
 * diffusion and decay are off, so the gradient does not move, does not get
 * grazed, and is the same on every arm. How a halo gets there is Phase 2's
 * question and would be three more confounds here.
 *
 * The control is the same rig at `gaitProfile` 0 — one phase for the whole
 * net, which is the pond before the relay — and the reversal arm puts the
 * smell at the other end, where a working mechanism has to change sign.
 */

const N = 10;
const PITCH = 60;
const AT = { x: 3000, y: 3000 };
/** How far past the chain's end the smell peaks, and how far it reaches. */
const FOOD_OFFSET = 500;
const SMELL_REACH = 2600;

function benchParams(): Params {
  const p = defaultParams();
  // No ecology beyond what the reactor has to eat.
  p.spawnInterval = 0;
  p.soupCount = 0;
  p.groundPatches = 0;
  p.groundDropEvery = 0;
  p.energyRegrow = 0;
  p.energyDiffuse = 0;
  // The reactor is the clock, so the chain has to eat. Ground everywhere, and
  // every body puts its reactor's fuel back under itself, so a bench that runs
  // for a minute does not turn into a starvation measurement halfway through.
  p.ambientEnergy = 0.5;
  p.upkeepExcrete = 1;
  p.upkeep = 0;
  p.swimCost = 0;
  // The field holds exactly what the rig paints.
  p.groundSmell = 0;
  p.decay = 0;
  p.diffuse = 0;
  p.deposit = 0;
  p.portLeak = 0;
  p.auxLeak = 0;
  // Nothing latches, nothing rewrites, nothing crowds.
  p.rewriteDuration = 0;
  p.snapRadius = 0;
  p.declutter = 0;
  p.flockAlign = 0;
  p.flockSep = 0;
  // **And nothing swims or steers.** The gait is the only way out of the field.
  p.stepSpeed = 0;
  p.turnRate = 0;
  p.swimNoise = 0;
  p.learnRate = 0;
  /*
   * **And the stroke is turned up, which is the whole reason this is a bench.**
   *
   * At the shipped `gaitSwell` of 0.04 this chain travels about 0.18 px a
   * second — measured — and forty seconds of it is seven pixels against a
   * settle drift of six. That is not a null result, it is an instrument with
   * no resolution. The pond ships at 0.04 because above it a freshly latched
   * pair breaks the spin bench, which is a constraint on the *pond* and not on
   * the question "does a phase profile over depth aim a wave". So the bench
   * asks at a stroke it can see and the pond's own setting is a separate
   * argument.
   */
  p.gaitSwell = 0.3;
  return p;
}

/** A chain, mouth to ear, an Era at the head and Cons behind it. */
function chain(sim: Sim, p: Params): Agent[] {
  const out: Agent[] = [];
  for (let i = 0; i < N; i++) {
    const a = sim.spawn(i === 0 ? 'era' : 'con', AT.x + i * PITCH, AT.y, 0, p, true)!;
    for (let k = 0; k < 4; k++) a.chem[TASTE + k] = 0;
    a.chem[TASTE + CH.aux] = 1;
    out.push(a);
  }
  for (let i = 0; i + 1 < N; i++) {
    sim.graph.connect(
      sim.agents,
      { id: out[i].id, slot: 'p' },
      { id: out[i + 1].id, slot: 'l' },
      sim.w,
      sim.h,
      p,
      sim.time,
    );
  }
  return out;
}

/** A smell that falls off with distance from a point, painted once. */
function paint(sim: Sim, x: number, y: number): void {
  const f = sim.fields;
  const d = f.data;
  const cell = f.worldW / f.cols;
  for (let j = 0; j < f.rows; j++) {
    const cy = f.originY + (j + 0.5) * cell;
    for (let i = 0; i < f.cols; i++) {
      const cx = f.originX + (i + 0.5) * cell;
      const r = Math.hypot(cx - x, cy - y);
      d[(j * f.cols + i) * CHANNELS + CH.aux] = Math.max(0, 1 - r / SMELL_REACH);
    }
  }
  f.touchWorld(f.originX + cell * 0.5, f.originY + cell * 0.5);
  f.touchWorld(f.originX + (f.cols - 0.5) * cell, f.originY + (f.rows - 0.5) * cell);
}

const centre = (bodies: Agent[]): { x: number; y: number } => {
  let x = 0;
  let y = 0;
  for (const a of bodies) {
    x += a.x;
    y += a.y;
  }
  return { x: x / bodies.length, y: y / bodies.length };
};

interface Run {
  /** Displacement along the direction of the smell, px. Positive is toward. */
  toward: number;
  /** Displacement across it, px. The confound this rig cannot remove. */
  across: number;
  /** Mean depth spread across the chain: 0 means the relay said nothing. */
  depthSpan: number;
  /** Mean |stroke| over the run, so a still reactor is visible as one. */
  swing: number;
}

function run(profile: number, broadcast: number, headward: boolean, seed: number, seconds: number): Run {
  const real = Math.random;
  Math.random = seededRandom(seed);
  try {
    const p = benchParams();
    p.gaitProfile = profile;
    p.metabolicDiffuse = broadcast;
    const sim = new Sim(6000, 6000);
    const bodies = chain(sim, p);
    // Past the head end, or past the tail end.
    const fx = headward ? AT.x - FOOD_OFFSET : AT.x + (N - 1) * PITCH + FOOD_OFFSET;
    paint(sim, fx, AT.y);
    // Settle: let the wires relax and the relay fill in before anything counts.
    for (let f = 0; f < 60 * 10; f++) sim.step(1 / 60, p);
    const from = centre(bodies);
    const st = sim.agentStore;
    let swing = 0;
    let span = 0;
    let n = 0;
    for (let f = 0; f < 60 * seconds; f++) {
      sim.step(1 / 60, p);
      if (f % 30 === 0) {
        let lo = 1;
        let hi = 0;
        for (const a of bodies) {
          const d = st.depth[a.slot];
          if (d < lo) lo = d;
          if (d > hi) hi = d;
          swing += Math.abs(st.gaitWave[a.slot]);
          n++;
        }
        span += hi - lo;
      }
    }
    const to = centre(bodies);
    const ux = headward ? -1 : 1;
    const samples = Math.max(1, n / bodies.length);
    return {
      toward: (to.x - from.x) * ux,
      across: to.y - from.y,
      depthSpan: span / samples,
      swing: swing / Math.max(1, n),
    };
  } finally {
    Math.random = real;
  }
}

describe('experiment: does a net crawl toward a smell', () => {
  it('sweeps the gait profile against the broadcast it competes with', () => {
    /*
     * Two things set a chain's phase gradient and they are not the same thing.
     * `metabolicDiffuse` is the broadcast, and a body speaks out of its
     * principal only, so a chain wired mouth-to-ear has a lag per wire that
     * runs in the *wiring order* — a travelling wave whose direction is the
     * accident of who is upstream. `gaitProfile` is the depth profile, which
     * runs from whichever end smells best. Whichever is larger decides.
     *
     * At the shipped broadcast the lag is about a tenth of a cycle a wire, so
     * over nine wires it is most of a full turn — against a profile that spans
     * at most `gaitProfile` radians end to end. So the column to read is not
     * whether the profile does anything, but whether it wins.
     */
    const SECONDS = 60;
    const SEED = 1;
    const profiles = [0, 2.2, 4.4, 6.3];
    console.log(`\n  one chain of ${N}, no steering, no thrust, ${SECONDS} s, seed ${SEED}`);
    console.log('  toward = px along the smell. A mechanism that aims is positive in BOTH');
    console.log('  smell columns; one that just picks a direction is positive in one.\n');
    console.log('  broadcast profile   head-end      tail-end      depth   swing');
    console.log('                      toward        toward         span        ');
    console.log('  --------- ------- --------      --------      ------  ------');
    for (const broadcast of [1, 0.2]) {
      for (const profile of profiles) {
        const h = run(profile, broadcast, true, SEED, SECONDS);
        const t = run(profile, broadcast, false, SEED, SECONDS);
        console.log(
          `  ${broadcast.toFixed(2).padStart(9)}` +
            ` ${profile.toFixed(2).padStart(7)}` +
            ` ${h.toward.toFixed(1).padStart(8)}      ` +
            ` ${t.toward.toFixed(1).padStart(8)}      ` +
            ` ${((h.depthSpan + t.depthSpan) / 2).toFixed(3).padStart(6)}` +
            ` ${((h.swing + t.swing) / 2).toFixed(3).padStart(7)}`,
        );
      }
    }
  });

  it('puts seeds on the corner where depth wins', () => {
    /*
     * One cell of the sweep above is positive in both smell columns, which is
     * the signature of a mechanism that *aims* rather than one that picks a
     * direction: a full turn of profile across the body against a broadcast
     * turned down to a fifth. That is one seed, and one seed in this project
     * is not a finding — seed variance is most of the variance. So: the same
     * corner and its control, three seeds each, both ways round.
     */
    const SECONDS = 60;
    const SEEDS = [1, 2, 3];
    const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
    console.log('\n  broadcast 0.2, three seeds, both ways round');
    console.log('  a mechanism that aims is positive in both columns, every seed\n');
    console.log('  profile   head-end          tail-end          both');
    console.log('             mean   seeds      mean   seeds      +ve?');
    console.log('  ------- ------- -------    ------- -------    -----');
    for (const profile of [0, 6.3]) {
      const h = SEEDS.map((s) => run(profile, 0.2, true, s, SECONDS).toward);
      const t = SEEDS.map((s) => run(profile, 0.2, false, s, SECONDS).toward);
      const both = h.filter((v, i) => v > 0 && t[i] > 0).length;
      console.log(
        `  ${profile.toFixed(2).padStart(7)}` +
          ` ${mean(h).toFixed(1).padStart(7)}` +
          ` ${h.map((v) => (v > 0 ? '+' : '-')).join('').padStart(7)}    ` +
          ` ${mean(t).toFixed(1).padStart(7)}` +
          ` ${t.map((v) => (v > 0 ? '+' : '-')).join('').padStart(7)}    ` +
          ` ${both}/${SEEDS.length}`,
      );
    }
  });
});
