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

interface Plan {
  label: string;
  specialist: boolean;
}

interface Outcome {
  /** Mean tank across the surviving bodies at the end. The fitness proxy. */
  meanExtra: number;
  /** Bodies still alive. A design that cannot feed itself loses its far end. */
  alive: number;
  /** Energy that crossed a wire over the run: how hard the net had to redistribute. */
  moved: number;
}

function run(plan: Plan, rowCost: number, hillN: number, seed: number): Outcome {
  const realRandom = Math.random;
  Math.random = seededRandom(seed);
  try {
    const params: Params = {
      ...defaultParams(),
      ...motorsOff(),
      uptakeVmax: 2,
      rowCost,
      hillN,
      ambientEnergy: 1,
      energyRegrow: 0.04,
      upkeep: 0.015,
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
      const isMouth = i === ids.length - 1;
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
        ...(plan.specialist ? (isMouth ? { uptake: { energy: 1 } } : { uptake: { aux: 1 } }) : {}),
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

function mean(plan: Plan, rowCost: number, hillN: number): Outcome {
  const runs = SEEDS.map((s) => run(plan, rowCost, hillN, s));
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

const GENERALIST: Plan = { label: 'generalist', specialist: false };
const SPECIALIST: Plan = { label: 'specialist', specialist: true };

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
      { label: 'generalist', o: g },
      { label: 'specialist', o: s },
    ]);
    const differ =
      Math.abs(g.meanExtra - s.meanExtra) > 1e-6 ||
      Math.abs(g.alive - s.alive) > 1e-6 ||
      Math.abs(g.moved - s.moved) > 1e-6;
    console.log(differ ? '\n  OK: the plans are distinguishable.' : '\n  DEAD INSTRUMENT: identical.');
  });

  it('sweeps the two superadditivity dials', () => {
    for (const hillN of [1, 2]) {
      table(
        `hillN ${hillN}`,
        [0, 0.005, 0.01, 0.02].flatMap((rowCost) => [
          { label: `  rowCost ${rowCost} generalist`, o: mean(GENERALIST, rowCost, hillN) },
          { label: `  rowCost ${rowCost} specialist`, o: mean(SPECIALIST, rowCost, hillN) },
        ]),
      );
    }
  });
});
