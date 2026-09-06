import { afterEach, beforeEach } from 'vitest';
import { Sim } from './sim.ts';
import { nativeSolver } from './native/solver.ts';

/**
 * Per-test isolation, applied to every file.
 *
 * Vitest reuses a worker process across test files, so anything living on a
 * module global or a class static outlives the file that set it. Two things
 * here do, and both bit hard enough to be worth a setup file.
 *
 * `Math.random` is replaced by several suites and restored by each of them
 * individually — which works right up until one throws, or until a file that
 * *does not* seed inherits the stream position left by whichever file ran
 * before it. Tests that measure settled geometry to a few pixels then pass or
 * fail on scheduling. Seeding every test from the same value makes each one
 * answer the same question however the runner orders them; a suite that wants
 * its own stream still just overwrites this.
 *
 * `Sim.nativeForces` and `Sim.auditForceBlock` are statics that the parity and
 * audit tests flip. A leaked `false` silently moves other files onto the JS
 * reference path, where they can pass while the shipped path is broken.
 */

const realRandom = Math.random;

/** Same LCG the seeded suites use, so nothing changes behaviour when they set it. */
function seeded(n: number): () => number {
  let s = n >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export const TEST_SEED = 20260902;

beforeEach(() => {
  Math.random = seeded(TEST_SEED);
  Sim.nativeForces = true;
  Sim.auditForceBlock = false;
  // The wasm module is the third thing that outlives a file, and the one
  // whose leaks are not a flag but a buffer: see `NativeSolver.resetCaches`.
  nativeSolver.resetCaches();
});

afterEach(() => {
  Math.random = realRandom;
  Sim.nativeForces = true;
  Sim.auditForceBlock = false;
});
