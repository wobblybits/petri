import { afterEach, describe, expect, it } from 'vitest';
import { nativeSolver } from './native/solver.ts';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';
import { stateHash } from './state-hash.ts';

export { stateHash };

/**
 * A reproducibility harness for the solver ports.
 *
 * Two jobs. It fails if the sim is nondeterministic under a fixed seed — the
 * classic causes being iteration order that depends on object identity, or
 * reading WASM memory a pass did not write. And it prints a state hash, so a
 * refactor that is meant to be bit-identical can be *shown* to be rather than
 * argued about, and one that is expected to drift can have that drift measured
 * against a known-good baseline instead of discovered when a test fails.
 *
 * Run the printed hashes before and after a change:
 *   npx vitest run src/determinism.test.ts --silent=false
 */

const realRandom = Math.random;

function seed(n: number): void {
  let s = n >>> 0;
  Math.random = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}


export interface Scenario {
  name: string;
  frames: number;
  build: (sim: Sim, params: Params) => void;
  tune?: (params: Params) => void;
  view?: { x: number; y: number; zoom: number; viewW: number; viewH: number } | null;
}

/** Run one scenario from a fixed seed and hash the result. */
export function runScenario(sc: Scenario, sd = 20260902): string {
  seed(sd);
  try {
    const sim = new Sim(1200, 800);
    const params = defaultParams();
    sc.tune?.(params);
    sc.build(sim, params);
    for (let f = 0; f < sc.frames; f++) sim.step(1 / 60, params, sc.view ?? null);
    return stateHash(sim);
  } finally {
    Math.random = realRandom;
  }
}

/** A grid-wired brick, the shape a grown net actually takes. */
function brick(sim: Sim, params: Params, side: number): void {
  const ids: number[] = [];
  for (let i = 0; i < side * side; i++) {
    const c = i % side;
    const r = (i / side) | 0;
    const kind = i % 3 === 0 ? 'era' : i % 3 === 1 ? 'con' : 'dup';
    const a = sim.spawn(kind, 200 + c * 52, 200 + r * 52, (i * 0.7) % 6.28, params, true);
    ids.push(a ? a.id : -1);
  }
  for (let i = 0; i < side * side; i++) {
    const c = i % side;
    const r = (i / side) | 0;
    if (c + 1 < side && ids[i] > 0 && ids[i + 1] > 0) {
      sim.wire(ids[i], 'r', ids[i + 1], 'l', params);
    }
    if (r + 1 < side && r % 2 === 0 && ids[i] > 0 && ids[i + side] > 0) {
      sim.wire(ids[i], 'p', ids[i + side], 'p', params);
    }
  }
}

export const SCENARIOS: Scenario[] = [
  {
    name: 'soup, no view',
    frames: 300,
    build: (sim, params) => loadPreset(sim, 'soup', params),
    tune: (p) => {
      p.soupCount = 60;
    },
  },
  {
    name: 'soup, zoomed in (NEAR)',
    frames: 300,
    build: (sim, params) => loadPreset(sim, 'soup', params),
    tune: (p) => {
      p.soupCount = 60;
    },
    view: { x: 600, y: 400, zoom: 1, viewW: 1200, viewH: 800 },
  },
  {
    name: 'soup, zoomed out (FAR)',
    frames: 300,
    build: (sim, params) => loadPreset(sim, 'soup', params),
    tune: (p) => {
      p.soupCount = 60;
    },
    view: { x: 600, y: 400, zoom: 0.06, viewW: 1200, viewH: 800 },
  },
  {
    name: 'oscillator, rewriting',
    frames: 600,
    build: (sim, params) => loadPreset(sim, 'oscillator', params),
    tune: (p) => {
      p.spawnInterval = 0;
    },
  },
  {
    name: '8x8 brick, settled tissue',
    frames: 300,
    build: (sim, params) => brick(sim, params, 8),
    tune: (p) => {
      p.spawnInterval = 0;
      p.snapRadius = 0;
    },
    view: { x: 400, y: 400, zoom: 0.6, viewW: 1200, viewH: 800 },
  },
];

afterEach(() => {
  // See the note in solver-extra.test.ts: this static leaks across files.
  Sim.nativeForces = true;
  Sim.auditForceBlock = false;
});

describe('determinism', () => {
  it('gives the same state twice from the same seed', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const lines: string[] = [];
    const seen = new Map<string, string>();
    for (const sc of SCENARIOS) {
      const a = runScenario(sc);
      const b = runScenario(sc);
      lines.push(`  ${sc.name.padEnd(26)} ${a}`);
      expect(a, `${sc.name} is not reproducible`).toBe(b);
      /*
       * Distinctness is a separate check from reproducibility, and it catches
       * a different failure: a pass that has been silently switched off.
       * Caching the port-torque wire list against the wrong key left the list
       * permanently empty, which reads as "no wires" and disables the pass —
       * and two scenarios that should settle differently collapsed onto the
       * same hash. Reproducibility was perfect throughout.
       */
      const clash = seen.get(a);
      expect(
        clash,
        `${sc.name} and ${clash} settle to the identical state (${a}). ` +
          'Two different scenes agreeing bit-for-bit usually means a pass ' +
          'stopped running rather than that they genuinely converged.',
      ).toBeUndefined();
      seen.set(a, sc.name);
    }
    console.log(`\nstate hashes @ ${SCENARIOS[0].frames}+ frames\n${lines.join('\n')}\n`);
  }, 180_000);

  it('leaves the JS reference path alone', async () => {
    // A control. With the ported passes switched off the sim runs the same
    // JavaScript it always did, so these hashes are a fixed point that the
    // WASM work must not touch. Printed rather than pinned: the value depends
    // on the whole sim, and a deliberate change to the physics should not
    // require editing a magic constant here.
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    Sim.nativeForces = false;
    const lines: string[] = [];
    try {
      for (const sc of SCENARIOS) {
        const a = runScenario(sc);
        expect(a, `${sc.name} is not reproducible`).toBe(runScenario(sc));
        lines.push(`  ${sc.name.padEnd(26)} ${a}`);
      }
    } finally {
      Sim.nativeForces = true;
    }
    console.log(`\nJS-forces reference hashes\n${lines.join('\n')}\n`);
  }, 180_000);

  it('never lets a body move between the force passes and the solve', async () => {
    // The packed bodies are handed straight to the solver rather than copied
    // again, which is only sound while nothing in between writes a position.
    // With the audit on, a violation throws instead of being quietly unpacked
    // over by syncForces.
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    Sim.auditForceBlock = true;
    try {
      for (const sc of SCENARIOS) {
        expect(() => runScenario(sc), sc.name).not.toThrow();
      }
    } finally {
      Sim.auditForceBlock = false;
    }
  }, 180_000);

  it('the audit actually catches a body that moves', async () => {
    // Guards the guard. Nudging an agent mid-frame has to be reported.
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const sim = new Sim(600, 400);
    const params = defaultParams();
    params.spawnInterval = 0;
    for (let i = 0; i < 6; i++) {
      sim.spawn('con', 200 + i * 40, 200, i * 0.5, params, true);
    }
    const view = { x: 300, y: 200, zoom: 0.06, viewW: 600, viewH: 400 };
    Sim.auditForceBlock = true;
    try {
      sim.step(1 / 60, params, view);
      const meddle = { ...view };
      // Move a body after the force passes have packed it. `syncRest` is the
      // nearest real hook; a monkey-patch stands in for a future pass that
      // forgets the rule.
      const graph = sim.graph as never as Record<string, (...a: unknown[]) => unknown>;
      const proto = Object.getPrototypeOf(sim.graph) as Record<string, (...a: unknown[]) => unknown>;
      const real = proto.syncRest;
      graph.syncRest = function (...a: unknown[]) {
        const r = real.apply(this, a);
        [...sim.agents.values()][0].x += 1;
        return r;
      };
      expect(() => sim.step(1 / 60, params, meddle)).toThrow(/moved after the force block/);
      delete (graph as Record<string, unknown>).syncRest;
    } finally {
      Sim.auditForceBlock = false;
    }
  }, 60_000);

  it('notices when state actually differs', () => {
    // Guards the guard: a hash that ignored a field would pass everything.
    const sim = new Sim(400, 300);
    const params = defaultParams();
    params.spawnInterval = 0;
    const a = sim.spawn('con', 100, 100, 0, params, true)!;
    const before = stateHash(sim);
    a.vx += 1e-12;
    expect(stateHash(sim)).not.toBe(before);
    a.vx -= 1e-12;
    expect(stateHash(sim)).toBe(before);
  });
});
