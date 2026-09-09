import { describe, it } from 'vitest';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { buildWorm, dress, motorsOff } from './organism.ts';
import { seededRandom } from './harness.ts';

/*
 * Is division of labour something selection could find?
 *
 * Everything built so far — the polarised worm with a mouth, the obligate
 * chain fed through its wires — is *buildable*. Whether it is *reachable* is a
 * different question and the plan already answers it in the abstract
 * (`docs/energy-chemistry-plan.md` §3): specialisation beats splitting only
 * when
 *
 *     f(1) > 2 * f(1/2)
 *
 * and with a linear budget — which the expression simplex is — against a
 * concave payoff — which Monod saturation is — the optimum is interior and
 * every body becomes a generalist. Pooling across wires does not rescue it.
 * The superadditivity has to come from somewhere, and the plan ships two
 * candidates at their neutral values: `rowCost`, a fixed price per expressed
 * row, and `hillN`, which makes uptake convex at low density.
 *
 * This measures the claim rather than trusting it. Two nets of the same length
 * on the same ground:
 *
 *   generalist   every segment at the seed, which is a flat eighth on all
 *                eight rows, so every one of them eats from its own cell
 *   specialist   one mouth expressing the ground's uptake row alone, so it
 *                draws at eight times the rate; every other segment expresses
 *                a single inert row and has to be fed through the wires
 *
 * Both pay upkeep and both pay `rowCost` per expressed row — eight rows a body
 * for the generalist against one for the specialist, which is the whole of the
 * advantage on offer. If the generalist wins at the shipped neutral dials and
 * loses as `rowCost` or `hillN` rises, the plan's argument is right and those
 * two numbers are the precondition for any of the body-plan work being
 * evolvable rather than merely constructible.
 *
 *     npm run experiment -- labour
 */

const SEGMENTS = Number(process.env.EXP_SEGMENTS ?? 8);
const SECONDS = Number(process.env.EXP_SECONDS ?? 120);
const SEEDS = [1, 2, 3];

/**
 * A body plan, as a share of expression per segment.
 *
 * `flat` is the seed and what an unevolved body actually is: every row at zero,
 * which `expressVector` turns into an eighth each, so it pays for all eight
 * rows and does every job badly.
 *
 * The other two are the real comparison, and getting to them meant fixing the
 * first version of this experiment. With `excreteRate` at zero there is exactly
 * **one** economically live row in the whole reaction table — the ground's
 * uptake — so there is no labour to divide, and the only "specialist" a net can
 * form is one body doing the job while the rest freeload. That is not a
 * division of labour, it is a passenger list, and it is why the first
 * specialist here was one mouth and seven inert segments.
 *
 * Turning on `excreteRate` and `fertilise` gives a second job that is genuinely
 * worth doing: a body excreting the fertiliser channel makes the ground it
 * stands on regrow faster, which is the mutualism `params.fertilise` was added
 * for. Now there are two jobs, and the question is real.
 */
interface Plan {
  label: string;
  kind: 'flat' | 'generalist' | 'specialist';
}

interface Outcome {
  /** Mean tank across the surviving bodies at the end. The fitness proxy. */
  meanExtra: number;
  /** Bodies still alive. A design that cannot feed itself loses its far end. */
  alive: number;
  /** Energy that crossed a wire over the run: how hard the net had to redistribute. */
  moved: number;
}

function run(plan: Plan, rowCost: number, hillN: number, seed: number, fertilise = 3): Outcome {
  const realRandom = Math.random;
  Math.random = seededRandom(seed);
  try {
    const params: Params = {
      ...defaultParams(),
      ...motorsOff(),
      uptakeVmax: 2,
      rowCost,
      hillN,
      /*
       * Food-limited on purpose, and this had to be found the hard way. At the
       * shipped ambient of 1 a cell holds two and a half tanks and every body
       * sits pinned at cap — `meanExtra` 0.398 against a cap of 0.4 — so the
       * ground is not the binding constraint and *nothing that increases the
       * supply of ground can matter*. Measured: `fertilise` at 0, 3 and 12 gave
       * byte-identical ponds. Half the specialist's bodies were doing a job
       * worth exactly zero, so it losing proved only that.
       *
       * A quarter of a unit a cell against a body needing 0.9 over the trial
       * puts the ground in charge, which is the only regime in which a second
       * job exists to divide.
       */
      ambientEnergy: 0.25,
      energyRegrow: 0.08,
      upkeep: 0.015,
      // The second job. Without these the reaction table has one live row and
      // there is nothing to divide.
      excreteRate: 0.05,
      fertilise,
    };
    const sim = new Sim(4000, 4000, 256);
    sim.pinWorld(2000, 2000, params);
    const ids = buildWorm(sim, params, {
      segments: SEGMENTS,
      kinds: 'con',
      heading: 0,
      jitter: 0.02,
      x: 2000,
      y: 2000,
    });
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
        /*
         * Small tanks, so the run reaches its steady state instead of
         * measuring the drain down to it. At the shipped cap of 1.25 against
         * an upkeep of 0.015 a body takes 83 s just to reach break-even, and a
         * first pass at 60 s read `moved` 0.00 — the specialist's segments had
         * not yet gone short, so nothing had asked and no transport had
         * happened at all. The comparison was of two nets coasting on the
         * energy they were built with.
         */
        energyCap: 0.4,
        debtCap: -0.2,
        extra: 0.4,
        /*
         * The generalist writes no expression at all, which is the seed: every
         * row at zero, `expressVector`'s flat fallback, an eighth each, and all
         * eight rows charged. The specialist writes exactly one row — the
         * ground's uptake for its mouth, and an inert one for everybody else,
         * since a body expressing literally nothing would take the flat
         * fallback and pay for all eight.
         */
        /*
         * Two jobs: eat the ground, and fertilise it so it comes back faster.
         * The generalist does both at half a share and pays for two rows; the
         * specialist does one at a whole share and pays for one. Interleaved
         * rather than split end to end, because fertiliser acts on the cell it
         * lands in and a fertiliser at the far end of the worm would be
         * manuring ground nobody is grazing.
         */
        ...(plan.kind === 'generalist'
          ? { uptake: { energy: 1 }, excrete: { conP: 1 } }
          : plan.kind === 'specialist'
            ? i % 2 === 0
              ? { uptake: { energy: 1 } }
              : { excrete: { conP: 1 } }
            : {}),
      });
    }
    const dt = 1 / 60;
    for (let f = 0; f < SECONDS * 60; f++) sim.step(dt, params);
    let sum = 0;
    let alive = 0;
    for (const id of ids) {
      const a = sim.agents.get(id);
      if (!a) continue;
      sum += a.extra;
      alive++;
    }
    return { meanExtra: alive > 0 ? sum / alive : 0, alive, moved: sim.tally.moved };
  } finally {
    Math.random = realRandom;
  }
}

function mean(plan: Plan, rowCost: number, hillN: number, fertilise = 3): Outcome {
  const runs = SEEDS.map((s) => run(plan, rowCost, hillN, s, fertilise));
  const avg = (f: (o: Outcome) => number) => runs.reduce((a, o) => a + f(o), 0) / runs.length;
  return {
    meanExtra: avg((o) => o.meanExtra),
    alive: avg((o) => o.alive),
    moved: avg((o) => o.moved),
  };
}

function table(title: string, rows: { label: string; o: Outcome }[]): void {
  console.log(`\n${title} (mean of ${SEEDS.length} seeds, ${SECONDS}s, ${SEGMENTS} segments)`);
  console.log(`${'condition'.padEnd(26)} ${'meanExtra'.padStart(10)} ${'alive'.padStart(6)} ${'moved'.padStart(9)}`);
  for (const r of rows) {
    console.log(
      `${r.label.padEnd(26)} ${r.o.meanExtra.toFixed(3).padStart(10)} ` +
        `${r.o.alive.toFixed(1).padStart(6)} ${r.o.moved.toFixed(2).padStart(9)}`,
    );
  }
}

const FLAT: Plan = { label: 'flat (the seed)', kind: 'flat' };
const GENERALIST: Plan = { label: 'generalist', kind: 'generalist' };
const SPECIALIST: Plan = { label: 'specialist', kind: 'specialist' };

describe('experiment: can division of labour pay?', () => {
  it('checks the two plans differ at all before reading anything into them', () => {
    /*
     * The null control, first and on purpose. Three times in this work a
     * "nothing happened" result turned out to be a disconnected instrument
     * rather than a fact about the pond, and each time the tell was a control
     * that should have differed and did not. So: do these two body plans
     * produce different ponds at all? If not, nothing below means anything.
     */
    const g = mean(GENERALIST, 0, 1);
    const s = mean(SPECIALIST, 0, 1);
    table('null control, neutral dials', [
      { label: 'flat (the seed)', o: mean(FLAT, 0, 1) },
      { label: 'generalist', o: g },
      { label: 'specialist', o: s },
    ]);
    const differ =
      Math.abs(g.meanExtra - s.meanExtra) > 1e-6 ||
      Math.abs(g.alive - s.alive) > 1e-6 ||
      Math.abs(g.moved - s.moved) > 1e-6;
    console.log(differ ? '\n  OK: the plans are distinguishable.' : '\n  DEAD INSTRUMENT: identical.');
  });

  it('checks the second job is worth doing at all', () => {
    /*
     * The control the fair-champion comparison depends on, and the one whose
     * absence would void it. A specialist net puts half its bodies on
     * fertilising; if fertilising is worth nothing then half of it is idle by
     * construction and losing proves only that. So: does the fertiliser
     * channel actually buy anything? Run the same plans with `fertilise` off
     * and on, and read the difference.
     *
     * If the two columns match, the second job is a fiction and the whole
     * division-of-labour question has not been asked yet.
     */
    for (const plan of [FLAT, GENERALIST, SPECIALIST]) {
      table(`${plan.label}: does fertilising pay?`, [
        { label: '  fertilise 0', o: mean(plan, 0, 1, 0) },
        { label: '  fertilise 12', o: mean(plan, 0, 1, 12) },
        { label: '  fertilise 200', o: mean(plan, 0, 1, 200) },
        { label: '  fertilise 5000', o: mean(plan, 0, 1, 5000) },
      ]);
    }
  });

  it('sweeps the row cost finely, against a fair champion', () => {
    /*
     * Finely, and with `alive` beside the tank, because the first pass found
     * the crossover sitting at the top of the slider in a region where *both*
     * plans were collapsing — the specialist mostly won by dying less. What
     * matters is whether there is a cost at which specialisation pays while
     * the economy still works, and that needs resolution between 0.01 and 0.02
     * rather than a jump across it.
     */
    for (const hillN of [1, 2]) {
      table(
        `hillN ${hillN}`,
        [0, 0.005, 0.01, 0.0125, 0.015, 0.02].flatMap((rowCost) => [
          { label: `  rowCost ${rowCost} flat`, o: mean(FLAT, rowCost, hillN) },
          { label: `  rowCost ${rowCost} generalist`, o: mean(GENERALIST, rowCost, hillN) },
          { label: `  rowCost ${rowCost} specialist`, o: mean(SPECIALIST, rowCost, hillN) },
        ]),
      );
    }
  });
});
