import { GRAVITATE_THREADED_WASM_B64 } from './gravitate-threaded.b64.ts';

/*
 * Runs `solver_gravitate_range` (native/solver.c) across a pool of Workers,
 * all executing the same tiny WASM module against one shared
 * `WebAssembly.Memory`. This is the hand-rolled alternative to Emscripten's
 * `-pthread` runtime: that runtime needs `-sSTANDALONE_WASM` off *and* its
 * own generated JS glue to spawn workers, which is more machinery than this
 * project's minimal hand-instantiated build style wants. This pool
 * hand-instantiates gravitate-threaded.wasm directly, in every worker and
 * on the main thread, using a ~5-function import object — see
 * native/build-gravitate-threaded.sh for why that module (not the main
 * standalone one) is the one built this way.
 *
 * Deliberately separate from the main solver's own memory. Reusing that
 * memory for a shared-across-threads module would mean rebuilding the whole
 * solver non-standalone, which changes the loading story for every other
 * pass, not just this one. This pool keeps its own small copy of the body
 * fields gravitate reads and writes, filled in and drained out around each
 * dispatch — an extra copy, but one that keeps the blast radius of "add
 * threading" to exactly this pass.
 *
 * Only `solver_gravitate` is safe to split this way today: every body it
 * touches only reads shared scalars and its own slot, and only ever writes
 * its own velocity, so disjoint index ranges never race and the result does
 * not depend on scheduling. `flock_apply` and `near_contacts` mutate both
 * ends of a pair per iteration and are not safe to hand this same pool
 * without per-thread accumulators and a reduction step first.
 */

const STRIDE = 12;

const CTL_N = 0;
const CTL_REMAINING = 1;
const CTL_MAX_COMP = 2;
const CTL_GEN_BASE = 3;

const PARAMS_CX = 0;
const PARAMS_CY = 1;
const PARAMS_BASE = 2;
const PARAMS_REACH = 3;
const PARAMS_DT = 4;
const PARAMS_HALF = 5;
const PARAMS_EDGE = 6;
const PARAMS_LEN = 7;

/** SharedArrayBuffer needs a cross-origin-isolated page (COOP + COEP). */
function sharedMemoryAvailable(): boolean {
  return (
    typeof SharedArrayBuffer !== 'undefined' &&
    typeof Worker !== 'undefined' &&
    typeof globalThis.crossOriginIsolated === 'boolean' &&
    globalThis.crossOriginIsolated
  );
}

function decodeWasm(): ArrayBuffer {
  const bin = atob(GRAVITATE_THREADED_WASM_B64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

/** The 5-function import object every instance (main thread and worker) needs. */
export function gravitateImports(memory: WebAssembly.Memory): WebAssembly.Imports {
  return {
    wasi_snapshot_preview1: {
      proc_exit: (code: number) => {
        throw new Error(`gravitate-threaded: proc_exit(${code})`);
      },
    },
    env: {
      memory,
      _emscripten_runtime_keepalive_clear: () => {},
      _setitimer_js: () => 0,
      _abort_js: () => {
        throw new Error('gravitate-threaded: abort');
      },
    },
  };
}

type GravitateExports = {
  solver_bodies(): number;
  solver_grav_sat(): number;
  solver_gravitate_range(
    start: number,
    end: number,
    cx: number,
    cy: number,
    base: number,
    reach: number,
    maxComp: number,
    dt: number,
    half: number,
    edge: number,
  ): void;
  solver_cap(): number;
  __wasm_call_ctors?: () => void;
};

export class GravitatePool {
  ready = false;
  private workerCount = 0;
  private workers: Worker[] = [];
  private ctl: Int32Array | null = null;
  private params: Float32Array | null = null;
  private sharedBodies: Float32Array | null = null;
  private sharedGravSat: Uint8Array | null = null;
  private cap = 0;
  private generation = 0;

  /**
   * `workerCount` workers, each instantiating gravitate-threaded.wasm
   * against one shared memory. Resolves false (and leaves the pool unready)
   * if SharedArrayBuffer/cross-origin isolation isn't available — callers
   * should keep using the single-threaded `nativeSolver.gravitate` then.
   */
  async init(workerCount: number): Promise<boolean> {
    if (!sharedMemoryAvailable() || workerCount < 1) return false;
    const wasmBytes = decodeWasm();
    const memory = new WebAssembly.Memory({
      initial: 83886080 / 65536,
      maximum: 83886080 / 65536,
      shared: true,
    });
    const module = await WebAssembly.compile(wasmBytes);
    const instance = await WebAssembly.instantiate(module, gravitateImports(memory));
    const exp = instance.exports as unknown as GravitateExports;
    exp.__wasm_call_ctors?.();
    this.cap = exp.solver_cap();
    this.sharedBodies = new Float32Array(memory.buffer, exp.solver_bodies(), this.cap * STRIDE);
    this.sharedGravSat = new Uint8Array(memory.buffer, exp.solver_grav_sat(), this.cap);

    const ctlBuf = new SharedArrayBuffer((CTL_GEN_BASE + workerCount) * 4);
    const paramsBuf = new SharedArrayBuffer(PARAMS_LEN * 4);
    this.ctl = new Int32Array(ctlBuf);
    this.params = new Float32Array(paramsBuf);

    const ready = await Promise.all(
      Array.from({ length: workerCount }, (_, i) =>
        this.spawnWorker(i, workerCount, memory, wasmBytes, ctlBuf, paramsBuf),
      ),
    );
    if (ready.some((ok) => !ok)) {
      this.dispose();
      return false;
    }
    this.workerCount = workerCount;
    this.ready = true;
    return true;
  }

  private spawnWorker(
    index: number,
    workerCount: number,
    memory: WebAssembly.Memory,
    wasmBytes: ArrayBuffer,
    ctlBuf: SharedArrayBuffer,
    paramsBuf: SharedArrayBuffer,
  ): Promise<boolean> {
    return new Promise((resolve) => {
      let worker: Worker;
      try {
        worker = new Worker(new URL('./gravitate-worker.ts', import.meta.url), { type: 'module' });
      } catch {
        resolve(false);
        return;
      }
      worker.onmessage = (ev: MessageEvent) => {
        if (ev.data?.type === 'ready') resolve(true);
      };
      worker.onerror = () => resolve(false);
      worker.postMessage({
        type: 'init',
        index,
        workerCount,
        memory,
        wasmBytes,
        ctlBuf,
        paramsBuf,
      });
      this.workers.push(worker);
    });
  }

  /**
   * Runs gravitate for bodies [0, n) split across the pool. `bodies` is the
   * caller's own Float32Array (STRIDE-major, same layout as native/solver.c)
   * and `gravSat` its per-body saturation flags; both are read in full and
   * `bodies`' velocity fields are updated in place on return.
   */
  async run(
    n: number,
    cx: number,
    cy: number,
    base: number,
    reach: number,
    maxComp: number,
    dt: number,
    half: number,
    edge: number,
    bodies: Float32Array,
    gravSat: Uint8Array,
  ): Promise<void> {
    if (!this.ready || !this.ctl || !this.params || !this.sharedBodies || !this.sharedGravSat) {
      throw new Error('GravitatePool.run called before a successful init()');
    }
    if (n > this.cap) throw new Error(`GravitatePool: ${n} bodies exceeds pool capacity ${this.cap}`);
    this.sharedBodies.set(bodies.subarray(0, n * STRIDE));
    this.sharedGravSat.set(gravSat.subarray(0, n));
    this.params[PARAMS_CX] = cx;
    this.params[PARAMS_CY] = cy;
    this.params[PARAMS_BASE] = base;
    this.params[PARAMS_REACH] = reach;
    this.params[PARAMS_DT] = dt;
    this.params[PARAMS_HALF] = half;
    this.params[PARAMS_EDGE] = edge;
    Atomics.store(this.ctl, CTL_N, n);
    Atomics.store(this.ctl, CTL_MAX_COMP, maxComp);
    Atomics.store(this.ctl, CTL_REMAINING, this.workerCount);
    const gen = ++this.generation;
    for (let i = 0; i < this.workerCount; i++) {
      Atomics.store(this.ctl, CTL_GEN_BASE + i, gen);
      Atomics.notify(this.ctl, CTL_GEN_BASE + i);
    }
    await this.awaitCompletion();
    bodies.set(this.sharedBodies.subarray(0, n * STRIDE));
  }

  /**
   * Polls `CTL_REMAINING` down to 0 via `Atomics.waitAsync`, the one Atomics
   * wait the main thread is allowed to use (a blocking `Atomics.wait` on the
   * main thread throws — that's reserved for workers). Loops rather than
   * awaiting once: a wake fires on any change to the slot, including a
   * decrement that hasn't yet reached zero.
   */
  private async awaitCompletion(): Promise<void> {
    const ctl = this.ctl!;
    for (;;) {
      const cur = Atomics.load(ctl, CTL_REMAINING);
      if (cur === 0) return;
      const res = Atomics.waitAsync(ctl, CTL_REMAINING, cur);
      if (res.async) await res.value;
    }
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    this.ctl = null;
    this.params = null;
    this.sharedBodies = null;
    this.sharedGravSat = null;
    this.ready = false;
  }
}
