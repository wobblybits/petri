import { CONFINE_THREADED_WASM_B64 } from './confine-threaded.b64.ts';

/*
 * Runs `solver_confine_range` (native/solver.c) across a pool of Workers,
 * all executing the same tiny WASM module against one shared
 * `WebAssembly.Memory`. This is the hand-rolled alternative to Emscripten's
 * `-pthread` runtime: that runtime needs `-sSTANDALONE_WASM` off *and* its
 * own generated JS glue to spawn workers, which is more machinery than this
 * project's minimal hand-instantiated build style wants. This pool
 * hand-instantiates confine-threaded.wasm directly, in every worker and on
 * the main thread, using a ~5-function import object — see
 * native/build-confine-threaded.sh for why that module (not the main
 * standalone one) is the one built this way.
 *
 * Deliberately separate from the main solver's own memory. Reusing that
 * memory for a shared-across-threads module would mean rebuilding the whole
 * solver non-standalone, which changes the loading story for every other
 * pass, not just this one. This pool keeps its own small copy of the body
 * fields confinement reads and writes, filled in and drained out around each
 * dispatch — an extra copy, but one that keeps the blast radius of "add
 * threading" to exactly this pass.
 *
 * Confinement is a loose failsafe, not a per-frame-exact force, so this pool
 * is not called from Sim's synchronous step()/beginFrame() at all — see
 * Sim.runConfineLoop, which dispatches on its own cadence and applies
 * whatever result lands whenever it lands. That sidesteps a real
 * architectural mismatch: a threaded dispatch is inherently async (there is
 * no legal blocking wait on the main thread), while beginFrame is
 * synchronous and shared by both `step()` (called synchronously from ~40
 * test files) and `stepAsync()`. Solving that properly — keeping a threaded
 * pass in exact lockstep with a synchronous frame — is real work this pass
 * does not need, because a body being nudged home a fraction of a second
 * late is not observable.
 */

const STRIDE = 12;
const FAR_X = 0;
const FAR_Y = 1;
const FAR_VX = 2;
const FAR_VY = 3;
const FAR_LOCKED = 8;

const CTL_N = 0;
const CTL_REMAINING = 1;
const CTL_GEN_BASE = 2;

const PARAMS_CX = 0;
const PARAMS_CY = 1;
const PARAMS_DT = 2;
const PARAMS_HALF = 3;
const PARAMS_EDGE = 4;
const PARAMS_LEN = 5;

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
  const bin = atob(CONFINE_THREADED_WASM_B64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

/** The 5-function import object every instance (main thread and worker) needs. */
export function confineImports(memory: WebAssembly.Memory): WebAssembly.Imports {
  return {
    wasi_snapshot_preview1: {
      proc_exit: (code: number) => {
        throw new Error(`confine-threaded: proc_exit(${code})`);
      },
    },
    env: {
      memory,
      _emscripten_runtime_keepalive_clear: () => {},
      _setitimer_js: () => 0,
      _abort_js: () => {
        throw new Error('confine-threaded: abort');
      },
    },
  };
}

type ConfineExports = {
  solver_bodies(): number;
  solver_confine_range(start: number, end: number, cx: number, cy: number, dt: number, half: number, edge: number): void;
  solver_cap(): number;
  __wasm_call_ctors?: () => void;
};

/** The body fields solver_confine_range reads and writes. */
export interface ConfineAgent {
  x: number;
  y: number;
  vx: number;
  vy: number;
  locked: boolean;
  pinned?: boolean;
}

export class ConfinePool {
  ready = false;
  private workerCount = 0;
  private workers: Worker[] = [];
  private ctl: Int32Array | null = null;
  private params: Float32Array | null = null;
  private sharedBodies: Float32Array | null = null;
  private cap = 0;
  private generation = 0;
  private scratch = new Float32Array(0);

  /**
   * `workerCount` workers, each instantiating confine-threaded.wasm against
   * one shared memory. Resolves false (and leaves the pool unready) if
   * SharedArrayBuffer/cross-origin isolation isn't available — callers
   * should keep using the single-threaded path then.
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
    const instance = await WebAssembly.instantiate(module, confineImports(memory));
    const exp = instance.exports as unknown as ConfineExports;
    exp.__wasm_call_ctors?.();
    this.cap = exp.solver_cap();
    this.sharedBodies = new Float32Array(memory.buffer, exp.solver_bodies(), this.cap * STRIDE);

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
        worker = new Worker(new URL('./confine-worker.ts', import.meta.url), { type: 'module' });
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

  /** Largest population one dispatch can carry. */
  get capacity(): number {
    return this.cap;
  }

  /**
   * Confines `agents` in place: gathers x/y/vx/vy/locked into the pool's
   * shared buffer, runs solver_confine_range split across the pool, and
   * writes the resulting vx/vy back into the same agent objects. `agents`
   * beyond `capacity` are left untouched — callers running populations that
   * large should be sharding across pools, not growing one past its build
   * size.
   */
  async runFromAgents(
    agents: readonly ConfineAgent[],
    cx: number,
    cy: number,
    dt: number,
    half: number,
    edge: number,
  ): Promise<void> {
    if (!this.ready || !this.ctl || !this.params || !this.sharedBodies) {
      throw new Error('ConfinePool.runFromAgents called before a successful init()');
    }
    const n = Math.min(agents.length, this.cap);
    if (n === 0) return;
    if (this.scratch.length < n * STRIDE) this.scratch = new Float32Array(n * STRIDE);
    const scratch = this.scratch;
    for (let i = 0; i < n; i++) {
      const a = agents[i];
      scratch[i * STRIDE + FAR_X] = a.x;
      scratch[i * STRIDE + FAR_Y] = a.y;
      scratch[i * STRIDE + FAR_VX] = a.vx;
      scratch[i * STRIDE + FAR_VY] = a.vy;
      scratch[i * STRIDE + FAR_LOCKED] = a.locked || a.pinned ? 1 : 0;
    }
    this.sharedBodies.set(scratch.subarray(0, n * STRIDE));
    this.params[PARAMS_CX] = cx;
    this.params[PARAMS_CY] = cy;
    this.params[PARAMS_DT] = dt;
    this.params[PARAMS_HALF] = half;
    this.params[PARAMS_EDGE] = edge;
    Atomics.store(this.ctl, CTL_N, n);
    Atomics.store(this.ctl, CTL_REMAINING, this.workerCount);
    const gen = ++this.generation;
    for (let i = 0; i < this.workerCount; i++) {
      Atomics.store(this.ctl, CTL_GEN_BASE + i, gen);
      Atomics.notify(this.ctl, CTL_GEN_BASE + i);
    }
    await this.awaitCompletion();
    /*
     * Applied as a delta against the snapshot (`scratch`, untouched by the
     * dispatch — only the shared buffer's copy was mutated), not written
     * back outright. The dispatch takes real wall-clock time (worker
     * compute, an Atomics round trip), and every other force keeps running
     * on `agents` while it's in flight — steering, locomotion, swimming.
     * Writing the shared buffer's value back directly would stomp all of
     * that with the pre-dispatch snapshot for every body confinement left
     * untouched (which is most of them, most of the time), fighting the
     * agent's own motion instead of nudging it home.
     */
    for (let i = 0; i < n; i++) {
      const a = agents[i];
      a.vx += this.sharedBodies[i * STRIDE + FAR_VX] - scratch[i * STRIDE + FAR_VX];
      a.vy += this.sharedBodies[i * STRIDE + FAR_VY] - scratch[i * STRIDE + FAR_VY];
    }
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
    this.ready = false;
  }
}
