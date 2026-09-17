import { describe, it } from 'vitest';
import { runSweep, summarize, writeSweep } from './harness.ts';

/*
 * The dials that ship at zero, one at a time.
 *
 * `swimCost` is the last of them. It had two companions here — `forageAsk`,
 * which made a body's energy bill directional, and `fertilise`, which let a
 * signal accelerate the ground's regrowth — and both were removed along with
 * the Gray-Scott reaction and three other dials: a mechanism nobody has run
 * at a visible setting is not a mechanism the pond has, and `CLAUDE.md` says
 * a dial ships on or it goes. `swimCost` stays because what it prices —
 * a net paying for its own locomotion — has no other expression.
 *
 * It is swept alone against the defaults so its effect is its own:
 * population, deaths, how much of the pond can afford to breed, and how far
 * the genome has drifted — which is the first thing that would show a dial
 * creating selection where there was none.
 *
 *     npx vitest run --project experiments economy --disableConsoleIntercept
 */
const AXES: Record<string, number[]> = {
  swimCost: [0, 0.0002, 0.0004, 0.0008],
};

describe('experiment: economy dials', () => {
  for (const [key, values] of Object.entries(AXES)) {
    it(`sweeps ${key}`, () => {
      const rows = runSweep(
        {
          name: `economy-${key}`,
          grid: { [key]: values },
          seeds: [1, 2],
          base: {
            seconds: Number(process.env.EXP_SECONDS ?? 60),
            soupCount: Number(process.env.EXP_BODIES ?? 300),
            sampleEvery: 10,
            // The breeding sweep says these two are what keeps the pond from
            // meeting; the economy dials only mean anything on a pond that
            // reproduces, so they are off here.
            params: { declutter: 0, flockAlign: 0 },
          },
        },
        (row, done, total) => {
          const last = row.trial.samples[row.trial.samples.length - 1];
          console.log(
            `[${done}/${total}] ${key}=${row.point[key as keyof typeof row.point]} seed=${row.seed} ` +
              `bodies=${last.bodies} died=${last.died} bornMean=${last.bornMean.toFixed(2)} ` +
              `canPay=${last.canPay.toFixed(2)} drift=${last.matrixDrift.toFixed(4)}`,
          );
        },
      );
      const path = writeSweep(`economy-${key}`, rows);
      console.log('\n' + summarize(rows, ['bodies', 'died', 'bornMean', 'lines', 'canPay', 'meanExtra', 'ground', 'matrixDrift']));
      console.log(`\nwrote ${path}`);
    });
  }
});
