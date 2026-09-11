import { describe, it } from 'vitest';
import { runSweep, summarize, writeSweep } from './harness.ts';

/*
 * The dials that ship at zero, each swept alone against the defaults.
 * `excreteRate` is not here: its two regimes are the arms in `pond/protocol.ts`.
 *
 *     npx vitest run --project experiments economy --disableConsoleIntercept
 */
const AXES: Record<string, number[]> = {
  fertilise: [0, 2, 4],
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
            // Off so the pond reproduces; the economy dials mean nothing otherwise.
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
