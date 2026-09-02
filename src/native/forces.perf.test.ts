import { afterEach, describe, expect, it } from 'vitest';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { nativeSolver } from './solver.ts';

/**
 * Large-net frame cost, JS force passes against the WASM ones.
 *
 * The shape that matters for a grown pond: several thousand bodies, most of
 * them inside a handful of big connected nets, viewed from far enough out that
 * the whole scene is on the cheap physics tier. Net *size* turns out not to
 * matter — 4800 bodies in one net and in 48 nets cost the same, so nothing
 * here is quadratic in component size — but total body count does, linearly.
 *
 * Measured by alternating blocks on a *single* sim rather than building one
 * per configuration. Building a fresh 9600-body sim per measurement puts the
 * result at the mercy of allocation and GC: that version reported a 40 % spread
 * between repeats of the same configuration, which is wider than the effect.
 */

function bigNets(sim: Sim, params: Params, netSizes: number[]): void {
  let ox = 400;
  for (const size of netSizes) {
    const ids: number[] = [];
    const cols = Math.max(2, Math.round(Math.sqrt(size)));
    for (let i = 0; i < size; i++) {
      const c = i % cols;
      const r = (i / cols) | 0;
      const kind = i % 3 === 0 ? 'era' : i % 3 === 1 ? 'con' : 'dup';
      const a = sim.spawn(kind, ox + c * 52, 400 + r * 52, (i * 0.7) % 6.28, params);
      ids.push(a ? a.id : -1);
    }
    for (let i = 0; i < size; i++) {
      const c = i % cols;
      const r = (i / cols) | 0;
      if (c + 1 < cols && i + 1 < size && ids[i] > 0 && ids[i + 1] > 0) {
        sim.wire(ids[i], 'r', ids[i + 1], 'l', params);
      }
      if (i + cols < size && r % 2 === 0 && ids[i] > 0 && ids[i + cols] > 0) {
        sim.wire(ids[i], 'p', ids[i + cols], 'p', params);
      }
    }
    ox += cols * 52 + 400;
  }
}

function median(a: number[]): number {
  return a.slice().sort((x, y) => x - y)[a.length >> 1];
}

/** Alternating blocks on one sim, so both configurations see the same heap. */
function compare(netSizes: number[]): { sim: Sim; off: number[]; on: number[] } {
  const params = defaultParams();
  params.maxAgents = 100_000;
  params.spawnInterval = 0;
  params.rewriteDuration = 0;
  params.snapRadius = 0;
  const sim = new Sim(40_000, 40_000);
  bigNets(sim, params, netSizes);
  const view = { x: 4000, y: 2000, zoom: 0.08, viewW: 1600, viewH: 900 };
  for (let f = 0; f < 30; f++) sim.step(1 / 60, params, view);
  const off: number[] = [];
  const on: number[] = [];
  try {
    for (let block = 0; block < 12; block++) {
      Sim.nativeForces = block % 2 === 1;
      const ts: number[] = [];
      for (let f = 0; f < 15; f++) {
        const t0 = performance.now();
        sim.step(1 / 60, params, view);
        ts.push(performance.now() - t0);
      }
      ts.sort((a, b) => a - b);
      (Sim.nativeForces ? on : off).push(ts[ts.length >> 1]);
    }
  } finally {
    Sim.nativeForces = true;
  }
  return { sim, off, on };
}

afterEach(() => {
  Sim.nativeForces = true;
});

describe('force passes at pond scale', () => {
  it('runs a large pond faster in WASM than in JS', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const rows: string[] = [];
    let checked = 0;
    for (const [label, nets] of [
      ['4800', [800, 800, 800, 800, 800, 800]],
      ['9600', [1600, 1600, 1600, 1600, 1600, 1600]],
    ] as [string, number[]][]) {
      const { sim, off, on } = compare(nets);
      const jsMs = median(off);
      const wasmMs = median(on);
      rows.push(
        `  ${label} agents, ${sim.graph.wires.size} wires\n` +
          `    JS   ${off.map((v) => v.toFixed(1)).join(' ')}   median ${jsMs.toFixed(1)}ms\n` +
          `    WASM ${on.map((v) => v.toFixed(1)).join(' ')}   median ${wasmMs.toFixed(1)}ms\n` +
          `    ${(((jsMs - wasmMs) / jsMs) * 100).toFixed(0)}% faster`,
      );
      // Generous: the point is to catch the ports regressing to parity or
      // worse, not to pin a number that varies with the machine.
      expect(wasmMs, `${label}: WASM ${wasmMs.toFixed(1)}ms vs JS ${jsMs.toFixed(1)}ms`)
        .toBeLessThan(jsMs * 0.97);
      checked++;
    }
    expect(checked).toBe(2);
    console.log(`\nlarge-net frame cost\n${rows.join('\n')}\n`);
  }, 900_000);
});
