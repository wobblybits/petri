import { SOLVER_WASM_B64 } from './solver.b64.ts';
import { FAR_STRIDE, FAR_SUBSTEPS, FAR_WIRE_STRIDE, stepFarKernel } from '../gpu/far-kernel.ts';
import type { Fields } from '../fields.ts';

export const WIRE_NEAR_STRIDE = 12;
export const NODE_STRIDE = 8;
export const HIT_STRIDE = 10;
export const HIT = {
  a: 0,
  b: 1,
  nx: 2,
  ny: 3,
  overlap: 4,
  px: 5,
  py: 6,
  vN: 7,
  vT: 8,
  effMass: 9,
} as const;
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
  solver_static_bytes(): number;
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
  solver_near_contacts(n: number, nWires: number, h: number): number;
  solver_port_torques(n: number, nWires: number, gain: number, splay: number, dt: number): void;
  solver_steer(n: number, dt: number): void;
  solver_steer_params(): number;
  solver_steer_flags(): number;
  solver_steer_pwire(): number;
  solver_steer_noise(): number;
  solver_body_drive(): number;
  solver_body_trail(): number;
  solver_scent_frame(cols: number, rows: number, ox: number, oy: number, ww: number, wh: number): void;
  solver_deposit(n: number, amount: number): void;
  solver_port_free(): number;
  solver_declutter(n: number, reach: number, atReach: number, cutoff: number, floorFrac: number, dt: number): void;
  solver_gravitate(n: number, cx: number, cy: number, base: number, reach: number, maxComp: number, dt: number, half: number, edge: number): void;
  solver_decl_comp(): number;
  solver_decl_sat(): number;
  solver_grav_sat(): number;
  solver_body_mass(): number;
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
  solver_adj_off(): number;
  solver_adj_nei(): number;
  solver_flock_id(): number;
  solver_flock_mass(): number;
  solver_swim(): number;
  solver_adj_cap(): number;
  solver_flock(
    n: number,
    align: number,
    sep: number,
    dt: number,
    turnRate: number,
    desired: number,
    maxHops: number,
    reuse: number,
  ): void;
  solver_flock_pairs(): number;
  solver_scent(): number;
  solver_scent_tmp(): number;
  solver_scent_cap(): number;
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
  /** Bytes of wasm memory the module's statics and stack occupy. */
  staticBytes = 0;
  /** Total wasm memory, from INITIAL_MEMORY in native/build.sh. */
  heapBytes = 0;
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
  adjOff: Int32Array | null = null;
  adjNei: Int32Array | null = null;
  flockId: Int32Array | null = null;
  declComp: Int32Array | null = null;
  steerParams: Float32Array | null = null;
  portFree: Uint8Array | null = null;
  steerFlags: Uint8Array | null = null;
  steerPwire: Int32Array | null = null;
  steerNoise: Float32Array | null = null;
  bodyDrive: Float32Array | null = null;
  bodyTrail: Float32Array | null = null;
  declSat: Uint8Array | null = null;
  gravSat: Uint8Array | null = null;
  bodyMass: Float32Array | null = null;
  flockMass: Float32Array | null = null;
  swim: Uint8Array | null = null;
  adjCap = 0;
  private exp: Exp | null = null;
  private scent: Float32Array | null = null;
  private scentCap = 0;

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
      this.fkSim = -1;
      this.scentOwner = null;
      this.skSim = -1;
      this.staticBytes = exp.solver_static_bytes();
      this.heapBytes = mem.buffer.byteLength;
      this.bodyCap = exp.solver_cap();
      this.wireCap = exp.solver_wire_cap();
      this.nodeCap = exp.solver_node_cap();
      this.pairCap = exp.solver_pair_cap();
      this.scentCap = exp.solver_scent_cap();
      this.bodies = new Float32Array(mem.buffer, exp.solver_bodies(), this.bodyCap * FAR_STRIDE);
      this.wires = new Float32Array(mem.buffer, exp.solver_wires(), this.wireCap * FAR_WIRE_STRIDE);
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
      this.adjCap = exp.solver_adj_cap();
      this.adjOff = new Int32Array(mem.buffer, exp.solver_adj_off(), this.bodyCap + 1);
      this.adjNei = new Int32Array(mem.buffer, exp.solver_adj_nei(), this.adjCap);
      this.flockId = new Int32Array(mem.buffer, exp.solver_flock_id(), this.bodyCap);
      this.declComp = new Int32Array(mem.buffer, exp.solver_decl_comp(), this.bodyCap);
      this.steerParams = new Float32Array(mem.buffer, exp.solver_steer_params(), 32);
      this.portFree = new Uint8Array(mem.buffer, exp.solver_port_free(), this.bodyCap);
      this.steerFlags = new Uint8Array(mem.buffer, exp.solver_steer_flags(), this.bodyCap);
      this.steerPwire = new Int32Array(mem.buffer, exp.solver_steer_pwire(), this.bodyCap * 2);
      this.steerNoise = new Float32Array(mem.buffer, exp.solver_steer_noise(), this.bodyCap * 3);
      this.bodyDrive = new Float32Array(mem.buffer, exp.solver_body_drive(), this.bodyCap);
      this.bodyTrail = new Float32Array(mem.buffer, exp.solver_body_trail(), this.bodyCap);
      this.declSat = new Uint8Array(mem.buffer, exp.solver_decl_sat(), this.bodyCap);
      this.gravSat = new Uint8Array(mem.buffer, exp.solver_grav_sat(), this.bodyCap);
      this.bodyMass = new Float32Array(mem.buffer, exp.solver_body_mass(), this.bodyCap);
      this.flockMass = new Float32Array(mem.buffer, exp.solver_flock_mass(), this.bodyCap);
      this.swim = new Uint8Array(mem.buffer, exp.solver_swim(), this.bodyCap);
      this.scent = new Float32Array(mem.buffer, exp.solver_scent(), this.scentCap);
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
    if (nWires > 0) this.wires.set(wires.subarray(0, nWires * FAR_WIRE_STRIDE));
    exp.solver_step_far(n, nWires, dt, substeps);
    data.set(this.bodies.subarray(0, n * FAR_STRIDE));
    return true;
  }

  /**
   * The same FAR step against bodies the caller has already written into
   * `bodies` and `wires`. Saves a copy of the whole scene in and another out;
   * `stepFar` above stays for the GPU path, which packs into its own array.
   */
  stepFarInPlace(n: number, nWires: number, dt: number, substeps = FAR_SUBSTEPS): boolean {
    const exp = this.exp;
    if (!this.ready || !exp || !this.bodies || !this.wires) return false;
    if (n > this.bodyCap || nWires > this.wireCap) return false;
    exp.solver_step_far(n, nWires, dt, substeps);
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

  /**
   * Per-wire port torques, in place on the packed bodies. Only `omega` moves,
   * so the caller can unpack that alone.
   */
  portTorques(n: number, nWires: number, gain: number, splay: number, dt: number): void {
    this.exp?.solver_port_torques(n, nWires, gain, splay, dt);
  }

  /**
   * Copy the live scent window in and tell the solver where it sits in the
   * world, so steering can sample it. Returns false when it will not fit.
   */
  /*
   * Only the live box crosses, not the whole grid.
   *
   * At 1024 squared the field is 16 MB, and copying it in and back out every
   * frame is 2 GB/s of memcpy at 60 fps — more than everything else in the
   * frame put together. The box is where the scent is, so the rows outside it
   * are zeros being copied back and forth.
   */
  /** Which field the resident scent buffer currently mirrors. */
  private scentOwner: Fields | null = null;

  loadScent(fields: Fields): boolean {
    if (!this.ready || !this.exp || !this.scent) return false;
    const n = fields.cols * fields.rows;
    if (n * 4 > this.scentCap) return false;
    /*
     * Only the box crosses, so everything outside it has to already agree —
     * and for a field the module has not seen before, it does not: the buffer
     * still holds whatever the last Sim left there, and `solver_deposit` and
     * the steering samples read the whole array. That made a scenario depend
     * on which sim ran before it, which the determinism harness caught as a
     * run failing to reproduce itself.
     *
     * Zeroing on handover restores the invariant the box copy needs: outside
     * the box both sides are zero. It costs one memset per owner change, and
     * the owner changes when a Sim is constructed, not when a frame runs.
     */
    if (this.scentOwner !== fields) {
      this.scent.fill(0);
      this.scentOwner = fields;
    }
    this.copyScentRows(fields, fields.data, this.scent);
    this.exp.solver_scent_frame(
      fields.cols,
      fields.rows,
      fields.originX,
      fields.originY,
      fields.worldW,
      fields.worldH,
    );
    return true;
  }

  steer(n: number, dt: number): void {
    this.exp?.solver_steer(n, dt);
  }

  /** Copy the solver's scent box back out to the host field. */
  storeScent(fields: Fields): void {
    if (!this.scent) return;
    this.copyScentRows(fields, this.scent, fields.data);
  }

  /** Row-by-row over the live box only. */
  private copyScentRows(fields: Fields, src: Float32Array, dst: Float32Array): void {
    const loI = fields.boxLoI;
    const hiI = fields.boxHiI;
    if (hiI < loI) return;
    const width = (hiI - loI + 1) * 4;
    const stride = fields.cols * 4;
    for (let j = fields.boxLoJ; j <= fields.boxHiJ; j++) {
      const at = j * stride + loI * 4;
      dst.set(src.subarray(at, at + width), at);
    }
  }

  deposit(n: number, amount: number): void {
    this.exp?.solver_deposit(n, amount);
  }

  declutter(n: number, reach: number, atReach: number, cutoff: number, floorFrac: number, dt: number): void {
    this.exp?.solver_declutter(n, reach, atReach, cutoff, floorFrac, dt);
  }

  gravitate(
    n: number, cx: number, cy: number, base: number, reach: number,
    maxComp: number, dt: number, half: number, edge: number,
  ): void {
    this.exp?.solver_gravitate(n, cx, cy, base, reach, maxComp, dt, half, edge);
  }

  /** Standalone contact pass. `nWires` lets it skip wired pairs the way
   *  `stepNear` does; pass 0 for a scene with no wires packed. */
  nearContacts(n: number, nWires: number, h: number): number {
    return this.exp?.solver_near_contacts(n, nWires, h) ?? 0;
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

  canFlock(n: number, nAdj: number): boolean {
    return this.ready && n <= this.bodyCap && nAdj <= this.adjCap;
  }

  /*
   * Owner of the flocking neighbourhood cache inside the wasm module.
   *
   * The module is a singleton and a process can hold several Sims — the test
   * suite routinely does. The cached pair list is indices into whichever Sim
   * packed the adjacency last, so replaying it against a different Sim's
   * bodies would be quietly, thoroughly wrong. Tracking the owner here rather
   * than in C keeps one authority: a caller whose key does not match has to
   * repack the adjacency, and the repack and the rebuild then happen together.
   */
  private fkSim = -1;
  private fkGraph = -1;
  private fkRoster = -1;
  private fkN = -1;
  private fkHops = -1;

  /*
   * Owner of the per-body scratch arrays that the force passes read —
   * steerFlags, steerPwire, declComp, declSat. Same problem and same shape as
   * the flocking cache above: all of it is derived from the topology and the
   * roster, none of it from the pose, and all of it lives in wasm memory that
   * every Sim in the process shares.
   */
  private skSim = -1;
  private skGraph = -1;
  private skRoster = -1;
  private skN = -1;

  /** True when the per-body scratch is this caller's and still current. */
  scratchHolds(simId: number, graphVersion: number, rosterVersion: number, n: number): boolean {
    return (
      this.ready &&
      this.skSim === simId &&
      this.skGraph === graphVersion &&
      this.skRoster === rosterVersion &&
      this.skN === n
    );
  }

  claimScratch(simId: number, graphVersion: number, rosterVersion: number, n: number): void {
    this.skSim = simId;
    this.skGraph = graphVersion;
    this.skRoster = rosterVersion;
    this.skN = n;
  }

  /** True when the cached pair list is this caller's and still current. */
  flockCacheHolds(
    simId: number,
    graphVersion: number,
    rosterVersion: number,
    n: number,
    maxHops: number,
  ): boolean {
    return (
      this.ready &&
      this.fkSim === simId &&
      this.fkGraph === graphVersion &&
      this.fkRoster === rosterVersion &&
      this.fkN === n &&
      this.fkHops === maxHops
    );
  }

  private claimFlockCache(
    simId: number,
    graphVersion: number,
    rosterVersion: number,
    n: number,
    maxHops: number,
  ): void {
    this.fkSim = simId;
    this.fkGraph = graphVersion;
    this.fkRoster = rosterVersion;
    this.fkN = n;
    this.fkHops = maxHops;
  }

  flock(
    n: number,
    align: number,
    sep: number,
    dt: number,
    turnRate: number,
    desired: number,
    maxHops: number,
    simId: number,
    graphVersion: number,
    rosterVersion: number,
  ): boolean {
    const exp = this.exp;
    if (!this.ready || !exp || n <= 0 || dt <= 0) return false;
    const reuse = this.flockCacheHolds(simId, graphVersion, rosterVersion, n, maxHops);
    exp.solver_flock(n, align, sep, dt, turnRate, desired, maxHops, reuse ? 1 : 0);
    this.claimFlockCache(simId, graphVersion, rosterVersion, n, maxHops);
    return true;
  }

  /**
   * Pairs held in the flocking neighbourhood cache, or -1 when it is not
   * valid — which means the pair list overran and every frame is searching.
   */
  flockPairs(): number {
    return this.exp?.solver_flock_pairs() ?? -1;
  }

  /*
   * Declined for a grid this size. These walk every cell and need the field
   * copied both ways to do it, which on a million-cell grid costs far more
   * than the arithmetic saves; the TS twin walks only the live box instead.
   * Kept for small grids, and for whatever the GPU path ends up leaving here.
   */
  scentDiffuse(fields: Fields, mix: number): boolean {
    const exp = this.exp;
    if (!this.ready || !exp || !this.scent) return false;
    const n = fields.cols * fields.rows;
    if (n > 65_536) return false;
    if (n * 4 > this.scentCap) return false;
    this.scent.set(fields.data.subarray(0, n * 4));
    exp.solver_scent_diffuse(fields.cols, fields.rows, mix);
    fields.data.set(this.scent.subarray(0, n * 4));
    return true;
  }

  scentDecay(fields: Fields, keep: number): boolean {
    const exp = this.exp;
    if (!this.ready || !exp || !this.scent) return false;
    if (fields.cols * fields.rows > 65_536) return false;
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
