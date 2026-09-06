import { describe, it } from 'vitest';
import { runSweep, summarize, writeSweep } from './harness.ts';

/*
 * How much does the pond breed, and what does each dial do to it?
 *
 * The memory of this project says `declutter` and `flockAlign` together cost
 * eighteen times the generation depth, and that the soup barely breeds at the
 * shipped defaults. That was measured once, by hand, over 45 seconds. This is
 * the same question asked properly: a grid over the two sliders and the
 * immigration rate, three seeds each, a minute of simulated time, sampled
 * every ten seconds so the warm-up can be told from the steady state.
 *
 * Read `bornMean` and `lines` together. Under selection `lines` falls while
 * `bornMean` climbs; under drift with immigration `lines` stays near the
 * number that arrived. `commutes` is the raw reproduction rate, `conDupWires`
 * the number of commutes waiting to happen.
 *
 *     npx vitest run --project experiments breeding --disableConsoleIntercept
 */
describe('experiment: breeding rate', () => {
  it('sweeps declutter x flockAlign x spawnInterval', () => {
    const rows = runSweep(
      {
        name: 'breeding',
        grid: {
          declutter: [0, 1.4],
          flockAlign: [0, 5.5],
          spawnInterval: [0, 0.5],
        },
        seeds: [1, 2, 3],
        base: {
          seconds: Number(process.env.EXP_SECONDS ?? 60),
          soupCount: Number(process.env.EXP_BODIES ?? 300),
          sampleEvery: 10,
        },
      },
      (row, done, total) => {
        const last = row.trial.samples[row.trial.samples.length - 1];
        console.log(
          `[${done}/${total}] ${JSON.stringify(row.point)} seed=${row.seed} ` +
            `bornMean=${last.bornMean.toFixed(2)} lines=${last.lines} commutes=${last.commutes} ` +
            `died=${last.died} ${(row.trial.wallMs / 1000).toFixed(1)}s`,
        );
      },
    );
    const path = writeSweep('breeding', rows);
    console.log('\n' + summarize(rows, ['bodies', 'lines', 'bornMean', 'commutes', 'died', 'conDupWires', 'meanExtra', 'matrixDrift']));
    console.log(`\nwrote ${path}`);
  });
});
