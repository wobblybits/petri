import { describe, expect, it } from 'vitest';
import type { Agent } from '../agents.ts';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { nativeSolver } from './solver.ts';

const FRAME = 1 / 60;
const BUDGET_60 = 1000 / 60;
const BUDGET_30 = 1000 / 30;

/** Settled Dup/Con tissue: no new latches, no rewrites, still a living NEAR step. */
function meshParams(): Params {
  const p = defaultParams();
  p.maxAgents = 20000;
  p.spawnInterval = 0;
  p.rewriteDuration = 0;
  p.snapRadius = 0;
  p.snapWell = 0;
  p.faceAttract = 0;
  return p;
}

/**
 * 3-regular brick mesh. Vertical p–l (not a redex) plus every other horizontal
 * r–r. Interior agents fill all three ports. Growing a side adds a ring of
 * agents wired into the existing net.
 */
class GrowingMesh {
  readonly sim: Sim;
  readonly params: Params;
  readonly spacing: number;
  cols = 0;
  rows = 0;
  grid: Agent[][] = [];

  constructor(params: Params, world = 8000) {
    this.params = params;
/*
     * A quarter-size dish, because the field is not what this measures.
     *
     * `bench` keeps the production 1024 cells on purpose — a frame budget over a
     * small field would not be a budget of the thing that ships. But the CPU
     * field's two diffusions, decay and grow are a *fixed* cost of about 18 ms at
     * that size, independent of bodies, wires and zoom: measured here at 18.45 ms
     * with zero bodies detailed, against a 60 fps budget of 16.67. So the budget
     * was failing before any of the physics this file is about had run, and the
     * LOD saving it exists to demonstrate — 21.56 ms in to 18.45 ms out — was
     * three milliseconds inside a frame that was five-sixths field.
     *
     * The field has its own ledger in `native/field-extent.perf.test.ts`, and on
     * the path that ships it is on the GPU anyway. Sixteenth the field work leaves
     * the physics, which is the thing under budget.
     */
    this.sim = new Sim(world, world, 256);
    this.spacing = params.wireMinRest + 16;
  }

  seed(side: number): void {
    for (let j = 0; j < side; j++) this.grid.push(this.spawnRow(j, side));
    this.cols = side;
    this.rows = side;
    for (let j = 0; j < this.rows; j++) {
      for (let i = 0; i < this.cols; i++) {
        if (j + 1 < this.rows) this.vert(i, j);
        if (i + 1 < this.cols && (i + j) % 2 === 0) this.horiz(i, j);
      }
    }
  }

  growTo(side: number): void {
    while (this.cols < side) this.addCol();
    while (this.rows < side) this.addRow();
  }

  get agents(): number {
    return this.sim.agents.size;
  }

  get wires(): number {
    return this.sim.graph.wires.size;
  }

  private spawnRow(j: number, cols: number): Agent[] {
    const row: Agent[] = [];
    for (let i = 0; i < cols; i++) row.push(this.spawnAt(i, j));
    return row;
  }

  private spawnAt(i: number, j: number): Agent {
    const kind = (i + j) % 2 === 0 ? 'con' : 'dup';
    const heading = j % 2 === 0 ? Math.PI / 2 : -Math.PI / 2;
    return this.sim.spawn(
      kind,
      400 + i * this.spacing,
      400 + j * this.spacing,
      heading,
      this.params,
      true,
    )!;
  }

  private vert(i: number, j: number): void {
    this.sim.wire(this.grid[j][i].id, 'p', this.grid[j + 1][i].id, 'l', this.params);
  }

  private horiz(i: number, j: number): void {
    this.sim.wire(this.grid[j][i].id, 'r', this.grid[j][i + 1].id, 'r', this.params);
  }

  private addCol(): void {
    const i = this.cols;
    for (let j = 0; j < this.rows; j++) {
      this.grid[j].push(this.spawnAt(i, j));
      if (j > 0) this.vert(i, j - 1);
      if ((i - 1 + j) % 2 === 0) this.horiz(i - 1, j);
    }
    this.cols++;
  }

  private addRow(): void {
    const j = this.rows;
    const row: Agent[] = [];
    for (let i = 0; i < this.cols; i++) row.push(this.spawnAt(i, j));
    this.grid.push(row);
    for (let i = 0; i < this.cols; i++) {
      this.vert(i, j - 1);
      if (i + 1 < this.cols && (i + j) % 2 === 0) this.horiz(i, j);
    }
    this.rows++;
  }
}

function percentile(xs: number[], p: number): number {
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.ceil(p * s.length) - 1));
  return s[i];
}

function timeSteps(sim: Sim, params: Params, frames: number): number[] {
  const ms: number[] = [];
  for (let i = 0; i < frames; i++) {
    const t0 = performance.now();
    sim.step(FRAME, params);
    ms.push(performance.now() - t0);
  }
  return ms;
}

describe('NEAR densely connected growth', () => {
  it('seeds a 3-regular mesh and finds the 60 fps cliff', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    expect(nativeSolver.ready).toBe(true);

    const params = meshParams();
    const mesh = new GrowingMesh(params);
    mesh.seed(4);
    expect(mesh.agents).toBe(16);
    expect(mesh.wires).toBeGreaterThan(16);

    const sizes = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64];
    const rows: { side: number; n: number; wires: number; p50: number; p95: number }[] = [];
    let cliff60 = 0;
    let cliff30 = 0;

    for (const side of sizes) {
      mesh.growTo(side);
      let nNodes = 0;
      for (const w of mesh.sim.graph.wires.values()) nNodes += w.nodes.length;
      if (!nativeSolver.canNear(mesh.agents, mesh.wires, nNodes)) break;
      timeSteps(mesh.sim, params, 3);
      const samples = timeSteps(mesh.sim, params, 10);
      const p50 = percentile(samples, 0.5);
      const p95 = percentile(samples, 0.95);
      rows.push({ side, n: mesh.agents, wires: mesh.wires, p50, p95 });
      if (p50 <= BUDGET_60) cliff60 = mesh.agents;
      if (p50 <= BUDGET_30) cliff30 = mesh.agents;
      if (p50 > BUDGET_30 && side >= 20) break;
    }

    const table = rows
      .map(
        (r) =>
          `${r.side}x${r.side}  n=${r.n}  wires=${r.wires}  p50=${r.p50.toFixed(2)}ms  p95=${r.p95.toFixed(2)}ms`,
      )
      .join('\n');
    console.log(
      `\nNEAR dense mesh (no view, WASM SAT+XPBD)\n${table}\n60fps cliff: ${cliff60} agents\n30fps cliff: ${cliff30} agents\n`,
    );

    const at400 = rows.find((r) => r.n >= 400);
    expect(rows[0].p50).toBeLessThan(BUDGET_60);
    expect(at400, 'mesh should grow to 400 agents').toBeTruthy();
    expect(at400!.p50, `400 NEAR p50 ${at400!.p50.toFixed(2)}ms`).toBeLessThan(BUDGET_60);
    expect(cliff60).toBeGreaterThanOrEqual(576);
  }, 180_000);

  it('shows which leftovers still eat the frame at 576 NEAR', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const variants: { name: string; apply: (p: Params) => void }[] = [
      { name: 'full', apply: () => {} },
      {
        name: 'no flock',
        apply: (p) => {
          p.flockAlign = 0;
          p.flockSep = 0;
        },
      },
      {
        name: 'no scent',
        apply: (p) => {
          p.deposit = 0;
          p.diffuse = 0;
          p.decay = 0;
        },
      },
      {
        name: 'no clear',
        apply: (p) => {
          p.wireClear = 0;
        },
      },
      {
        name: 'no motors',
        apply: (p) => {
          p.stepSpeed = 0;
          p.swimNoise = 0;
          p.portStiff = 0;
          p.declutter = 0;
        },
      },
    ];
    for (const v of variants) {
      const params = meshParams();
      v.apply(params);
      const mesh = new GrowingMesh(params);
      mesh.seed(24);
      timeSteps(mesh.sim, params, 2);
      const samples = timeSteps(mesh.sim, params, 8);
      const p50 = percentile(samples, 0.5);
      console.log(`576 NEAR ${v.name}: p50=${p50.toFixed(2)}ms  wires=${mesh.wires}`);
    }
  }, 120_000);
});
