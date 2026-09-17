import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { closeWebGpu, openWebGpu } from '../pond/webgpu-node.ts';
import { fieldGpu } from './field-gpu.ts';
import { runGpuCheck } from './far-gpu-check.ts';

/*
 * The FAR shader, run.
 *
 * `far.wgsl` was the only shader in the program with no test on a device.
 * `far-gpu-check.ts` next door has the scenes and the comparison, but it was
 * written as a browser page on the grounds that "there is no WebGPU under
 * Node" — which stopped being true: `field-device.test.ts` and
 * `genome-device.test.ts` both run against Dawn in this suite. So the scenes
 * existed, the twin existed, and the only thing missing was the eight lines
 * that run them where the rest of the suite runs.
 *
 * What that left uncovered is not hypothetical. The contact tier's skin and
 * its wired pairs were changed in `far.wgsl` and in `far-kernel.ts` by hand,
 * in two languages, and nothing in the suite could tell whether the two edits
 * agreed.
 *
 * Skips where Dawn is not installed, like the other two, and for the same
 * reason: a skip says the check did not happen.
 */

let device = false;

beforeAll(async () => {
  const open = await openWebGpu();
  device = open.ok;
});

afterAll(() => {
  closeWebGpu(fieldGpu.gpuDevice);
});

describe('far.wgsl on a device', () => {
  it('solves every scene the way its CPU twin does', async ({ skip }) => {
    if (!device) skip();
    const { gpuAvailable, initError, results } = await runGpuCheck();
    expect(gpuAvailable, `far.wgsl did not compile: ${initError}`).toBe(true);
    expect(results.length, 'no scenes ran').toBeGreaterThan(0);

    const label = results
      .map((r) => `${r.name}: worst ${r.worst.field} ${r.worst.maxAbs.toExponential(2)}`)
      .join('\n  ');

    for (const r of results) {
      // False means the kernel fell back internally, which would have this
      // comparing the twin against itself — the exact failure the check's own
      // header warns about.
      expect(r.gpuActuallyRan, `${r.name}: fell back to the CPU twin\n  ${label}`).toBe(true);
      // Finiteness is the one thing every scene owes, crowded or not. It is
      // also the shape the historical failure took: wires to infinite length.
      expect(r.cpuNonFinite, `${r.name}: the twin produced non-finite state`).toBe(0);
      expect(r.gpuNonFinite, `${r.name}: the device produced non-finite state`).toBe(0);
    }

    /*
     * Exact agreement is owed only where one constraint acts at a time.
     *
     * The device solves a substep in parallel — a thread a body, each
     * accumulating its own pushes — and the twin accumulates them in index
     * order, so anything with two constraints on one body is Jacobi against
     * Gauss-Seidel and the two are entitled to differ. Measured here, on
     * Dawn: the four isolated scenes sit at 6e-5 to 2.6e-4, and the crowded
     * ones run to 8.4e+3 — the 400-body pile most of all, because
     * `CELL_CAP` is 64 and a cell past it drops contacts on the device that
     * the twin still solves. That is a documented limit of the kernel, not a
     * disagreement about what it computes.
     *
     * So the isolated scenes are the assertion, and they are also the ones
     * that carry the contact geometry — the skin, the radius, and whether a
     * wired pair collides. A hand-edit to `far.wgsl` that misses
     * `far-kernel.ts` shows up here.
     */
    const isolated = results.filter((r) => r.n <= 2 && r.nWires <= 1);
    expect(isolated.length, 'no single-constraint scene to compare').toBeGreaterThanOrEqual(4);
    for (const r of isolated) {
      expect(r.worst.maxAbs, `${r.name}: ${r.worst.field}\n  ${label}`).toBeLessThan(1e-3);
    }
  });
});
