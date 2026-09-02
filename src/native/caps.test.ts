import { describe, expect, it } from 'vitest';
import { nativeSolver } from './solver.ts';

/**
 * The solver's population caps and the heap that has to hold them.
 *
 * Every array in native/solver.c is static, so the caps fix the module's whole
 * footprint at instantiation — there is no allocator to grow into, and
 * `-sALLOW_MEMORY_GROWTH=0` means the wasm memory never gets any bigger than
 * `INITIAL_MEMORY`. Raising `MAX_BODIES` without raising the heap does not fail
 * loudly at build time; it fails at link time with a message about the data
 * segment, or worse, quietly leaves so little slack that the next array added
 * pushes it over. So the module reports its own footprint and this asserts it.
 *
 * The ratios matter as much as the absolute numbers. `MAX_WIRES` used to equal
 * `MAX_BODIES`, which is *below* the structural bound: three ports per body,
 * two ports per wire, so a fully wired pond wants 1.5 wires per body. A pond
 * that reached that ratio silently dropped off the native path onto the TS
 * twin — correct, ~30% slower, and invisible.
 */

describe('solver capacity', () => {
  it('fits its statics in the heap with room to spare', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const used = nativeSolver.staticBytes;
    const heap = nativeSolver.heapBytes;
    console.log(
      `\nsolver footprint\n` +
        `  statics + stack  ${(used / 1048576).toFixed(2)} MB\n` +
        `  heap             ${(heap / 1048576).toFixed(2)} MB\n` +
        `  free             ${((heap - used) / 1048576).toFixed(2)} MB\n`,
    );
    expect(used).toBeGreaterThan(0);
    expect(used, `statics ${used} exceed heap ${heap}`).toBeLessThan(heap);
    // Slack, not just fit: something has to be left for the next array added.
    expect(heap - used, 'less than 1 MB of headroom left in the wasm heap').toBeGreaterThan(
      1 << 20,
    );
  });

  it('caps the derived arrays at their structural ratios', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const b = nativeSolver.bodyCap;
    expect(b).toBeGreaterThanOrEqual(32768);
    // 3 ports per body, 2 ports per wire.
    expect(nativeSolver.wireCap).toBeGreaterThanOrEqual((b * 3) / 2);
    // NEAR rope nodes, 8 per body at the finest subdivision.
    expect(nativeSolver.nodeCap).toBeGreaterThanOrEqual(b * 8);
    // Adjacency holds both endpoints of every wire.
    expect(nativeSolver.adjCap).toBeGreaterThanOrEqual(nativeSolver.wireCap * 2);
  });
});
