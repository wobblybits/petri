import { SOLVER_WASM_B64 } from './solver.b64.ts';
import { FAR_STRIDE, FAR_SUBSTEPS, stepFarKernel } from '../gpu/far-kernel.ts';
import type { Fields } from '../fields.ts';

export const WIRE_NEAR_STRIDE = 12;
export const NODE_STRIDE = 8;
export const HIT_STRIDE = 7;
export const WF_FULL = 1;
export const WF_SKIP = 2;
export const WF_SHAPE = 4;
export const WF_HOLD = 8;

export const WN = {
  a: 0,
  b: 1,
  rest: 2,
  rope: 3,
  scale: 4,
  slack: 5,
  aSlot: 6,
  bSlot: 7,
  node0: 8,
  nNodes: 9,
  flags: 10,
} as const;

export const ND = {
  x: 0,
  y: 1,
  vx: 2,
  vy: 3,
  prevX: 4,
  prevY: 5,
  shapeX: 6,
  shapeY: 7,
} as const;

export const KIND_ERA = 0;
export const KIND_DUP = 1;
export const KIND_CON = 2;

type Exp = {
  memory: WebAssembly.Memory;
  solver_bodies(): number;
  solver_wires(): number;
  solver_nodes(): number;
  solver_inv_inertia(): number;
  solver_scale(): number;
  solver_kind(): number;
  solver_detailed(): number;
  solver_pair_a(): number;
  solver_pair_b(): number;
  solver_cap(): number;
  solver_wire_cap(): number;
  solver_node_cap(): number;
  solver_pair_cap(): number;
  solver_pair_count(): number;
  solver_wire_near_stride(): number;
  solver_node_stride(): number;
  solver_step_far(n: number, nWires: number, dt: number, substeps: number): void;
  solver_near_integrate(n: number, nWires: number, h: number): void;
  solver_near_wires(n: number, nWires: number, h: number): void;
  solver_near_disc(n: number, h: number): number;
  solver_near_finalize(
    n: number,
    nWires: number,
    h: number,
    ropeKeep: number,
    held: number,
    grabMax: number,
  ): void;
  solver_near_contacts(n: number, h: number): number;
  solver_step_near(
    n: number,
    nWires: number,
    dt: number,
    substeps: number,
    ropeKeep: number,
    held: number,
    grabMax: number,
    gx: number,
    gy: number,
  ): void;
  solver_hits(): number;
  solver_hit_count(): number;
  solver_hit_stride(): number;
  solver_hit_cap(): number;
  solver_scent(): number;
  solver_scent_tmp(): number;
  solver_walls(): number;
  solver_scent_cap(): number;
  solver_wall_cap(): number;
  solver_scent_diffuse(cols: number, rows: number, mix: number): void;
  solver_scent_decay(n: number, keep: number): void;
  _initialize?: () => void;
};

/**
 * C FAR solver, NEAR XPBD, and scent field, compiled to WASM. Same SoA as
 * the TS/GPU kernel for bodies. After `init`, steps are synchronous.
 */
export class NativeSolver {
  ready = false;
  lastError = '';
  bodyCap = 0;
  wireCap = 0;
  nodeCap = 0;
  pairCap = 0;
  bodies: Float32Array | null = null;
  wires: Float32Array | null = null;
  wiresNear: Float32Array | null = null;
  nodes: Float32Array | null = null;
  invInertia: Float32Array | null = null;
  scale: Float32Array | null = null;
  kind: Uint8Array | null = null;
  detailed: Uint8Array | null = null;
  pairA: Int32Array | null = null;
  pairB: Int32Array | null = null;
  hits: Float32Array | null = null;
  hitCap = 0;
  private exp: Exp | null = null;
  private scent: Float32Array | null = null;
  private walls: Uint8Array | null = null;
  private scentCap = 0;
  private wallCap = 0;

  async init(source?: ArrayBuffer | Uint8Array): Promise<boolean> {
    try {
      const copy = source
        ? source instanceof ArrayBuffer
          ? source
          : (source.buffer as ArrayBuffer).slice(source.byteOffset, source.byteOffset + source.byteLength)
        : decodeWasm();
      const module = await WebAssembly.compile(copy);
      const instance = await WebAssembly.instantiate(module, {});
      const exp = instance.exports as unknown as Exp;
      exp._initialize?.();
      const mem = exp.memory;
      this.bodyCap = exp.solver_cap();
      this.wireCap = exp.solver_wire_cap();
      this.nodeCap = exp.solver_node_cap();
      this.pairCap = exp.solver_pair_cap();
      this.scentCap = exp.solver_scent_cap();
      this.wallCap = exp.solver_wall_cap();
      this.bodies = new Float32Array(mem.buffer, exp.solver_bodies(), this.bodyCap * FAR_STRIDE);
      this.wires = new Float32Array(mem.buffer, exp.solver_wires(), this.wireCap * 4);
      this.wiresNear = new Float32Array(mem.buffer, exp.solver_wires(), this.wireCap * WIRE_NEAR_STRIDE);
      this.nodes = new Float32Array(mem.buffer, exp.solver_nodes(), this.nodeCap * NODE_STRIDE);
      this.invInertia = new Float32Array(mem.buffer, exp.solver_inv_inertia(), this.bodyCap);
      this.scale = new Float32Array(mem.buffer, exp.solver_scale(), this.bodyCap);
      this.kind = new Uint8Array(mem.buffer, exp.solver_kind(), this.bodyCap);
      this.detailed = new Uint8Array(mem.buffer, exp.solver_detailed(), this.bodyCap);
      this.pairA = new Int32Array(mem.buffer, exp.solver_pair_a(), this.pairCap);
      this.pairB = new Int32Array(mem.buffer, exp.solver_pair_b(), this.pairCap);
      this.hitCap = exp.solver_hit_cap();
      this.hits = new Float32Array(mem.buffer, exp.solver_hits(), this.hitCap * HIT_STRIDE);
      this.scent = new Float32Array(mem.buffer, exp.solver_scent(), this.scentCap);
      this.walls = new Uint8Array(mem.buffer, exp.solver_walls(), this.wallCap);
      this.exp = exp;
      this.ready = true;
      this.lastError = '';
      return true;
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      this.ready = false;
      this.exp = null;
      return false;
    }
  }

  canNear(n: number, nWires: number, nNodes: number): boolean {
    return this.ready && n <= this.bodyCap && nWires <= this.wireCap && nNodes <= this.nodeCap;
  }

  /**
   * Run the FAR kernel in place on `data` / `wires`. Uses WASM when loaded
   * and the pack fits; otherwise the TS twin. Always mutates `data`.
   */
  stepFar(
    data: Float32Array,
    n: number,
    wires: Float32Array,
    nWires: number,
    dt: number,
    substeps = FAR_SUBSTEPS,
  ): boolean {
    if (n <= 0 || dt <= 0) return true;
    const exp = this.exp;
    if (!this.ready || !exp || !this.bodies || !this.wires || n > this.bodyCap || nWires > this.wireCap) {
      stepFarKernel(data, n, wires, nWires, dt, substeps);
      return false;
    }
    this.bodies.set(data.subarray(0, n * FAR_STRIDE));
    if (nWires > 0) this.wires.set(wires.subarray(0, nWires * 4));
    exp.solver_step_far(n, nWires, dt, substeps);
    data.set(this.bodies.subarray(0, n * FAR_STRIDE));
    return true;
  }

  nearIntegrate(n: number, nWires: number, h: number): void {
    this.exp?.solver_near_integrate(n, nWires, h);
  }

  nearWires(n: number, nWires: number, h: number): void {
    this.exp?.solver_near_wires(n, nWires, h);
  }

  nearDisc(n: number, h: number): number {
    return this.exp?.solver_near_disc(n, h) ?? 0;
  }

  nearFinalize(n: number, nWires: number, h: number, ropeKeep: number, held: number, grabMax: number): void {
    this.exp?.solver_near_finalize(n, nWires, h, ropeKeep, held, grabMax);
  }

  nearContacts(n: number, h: number): number {
    return this.exp?.solver_near_contacts(n, h) ?? 0;
  }

  /**
   * Eight-substep NEAR pass in WASM: integrate, XPBD, grab, SAT (detailed)
   * / disc (FAR-FAR), finalize. Pack state before, unpack after.
   */
  stepNear(
    n: number,
    nWires: number,
    dt: number,
    substeps: number,
    ropeKeep: number,
    held: number,
    grabMax: number,
    gx: number,
    gy: number,
  ): boolean {
    const exp = this.exp;
    if (!this.ready || !exp || n <= 0 || dt <= 0) return false;
    exp.solver_step_near(n, nWires, dt, substeps, ropeKeep, held, grabMax, gx, gy);
    return true;
  }

  pairCount(): number {
    return this.exp?.solver_pair_count() ?? 0;
  }

  hitCount(): number {
    return this.exp?.solver_hit_count() ?? 0;
  }

  scentDiffuse(fields: Fields, mix: number): boolean {
    const exp = this.exp;
    if (!this.ready || !exp || !this.scent || !this.walls) return false;
    const n = fields.cols * fields.rows;
    if (n * 4 > this.scentCap || n > this.wallCap) return false;
    this.scent.set(fields.data.subarray(0, n * 4));
    this.walls.set(fields.walls.subarray(0, n));
    exp.solver_scent_diffuse(fields.cols, fields.rows, mix);
    fields.data.set(this.scent.subarray(0, n * 4));
    return true;
  }

  scentDecay(fields: Fields, keep: number): boolean {
    const exp = this.exp;
    if (!this.ready || !exp || !this.scent) return false;
    const n = fields.cols * fields.rows * 4;
    if (n > this.scentCap) return false;
    this.scent.set(fields.data.subarray(0, n));
    exp.solver_scent_decay(n, keep);
    fields.data.set(this.scent.subarray(0, n));
    return true;
  }
}

function decodeWasm(): ArrayBuffer {
  const bin = atob(SOLVER_WASM_B64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

export const nativeSolver = new NativeSolver();
