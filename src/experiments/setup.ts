import { beforeAll } from 'vitest';
import { nativeSolver } from '../native/solver.ts';

/*
 * Turn the wasm solver on for every sweep. `nativeSolver.ready` is only set by
 * an explicit `init()`; without it every guarded call falls through to the
 * JavaScript twin, which is not the implementation that ships.
 *
 * `test-setup.ts` deliberately does not do this: the parity suites turn the
 * solver on themselves.
 */
beforeAll(async () => {
  const ok = await nativeSolver.init();
  if (!ok) {
    throw new Error(`experiments: wasm solver unavailable: ${nativeSolver.lastError}`);
  }
});
