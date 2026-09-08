import { beforeAll } from 'vitest';
import { nativeSolver } from '../native/solver.ts';

/*
 * Turn the solver on for every sweep.
 *
 * `Sim.nativeForces` defaults to true, but `nativeSolver.ready` does not:
 * the wasm is instantiated by an explicit `init()`, which `main.ts`, `demo.ts`
 * and `design.ts` each call at startup and which nothing under Node did. Every
 * guarded call therefore fell through to the JavaScript twin — silently, and
 * correctly, because the twin is the reference the wasm was written against.
 * The sweeps were right about the pond and wrong about the machine, and they
 * were measuring an implementation that does not ship.
 *
 * Measured back to back on a 1,800-body pond: 57.0 ms a frame on the twin,
 * 16.9 ms with the solver, the solve phase alone 37.6 -> 4.8 ms.
 *
 * `test-setup.ts` deliberately does not do this. The correctness projects
 * contain the parity suites that compare the two paths, and several of them
 * turn the solver on themselves; a global init there would decide for them.
 */
beforeAll(async () => {
  const ok = await nativeSolver.init();
  if (!ok) {
    throw new Error(`experiments: wasm solver unavailable: ${nativeSolver.lastError}`);
  }
});
