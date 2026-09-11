import { describe, it } from 'vitest';
import { runSweep, summarize, writeSweep } from './harness.ts';

/*
 * A grid over `declutter`, `flockAlign` and `spawnInterval`, three seeds each.
 * Under selection `lines` falls while `bornMean` climbs; under drift with
 * immigration `lines` stays near the number that arrived.
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
