import { confineImports } from './confine-pool.ts';

/*
 * One shard of ConfinePool. Instantiates the same confine-threaded.wasm as
 * every other shard and the main thread, against the memory it's handed —
 * so `solver_bodies()` resolves to the same byte offset everywhere and no
 * data ever needs copying between workers, only into and out of the pool
 * once per dispatch (see confine-pool.ts).
 */

const CTL_N = 0;
const CTL_REMAINING = 1;
const CTL_GEN_BASE = 2;

const PARAMS_CX = 0;
const PARAMS_CY = 1;
const PARAMS_DT = 2;
const PARAMS_HALF = 3;
const PARAMS_EDGE = 4;

type ConfineExports = {
  solver_confine_range(start: number, end: number, cx: number, cy: number, dt: number, half: number, edge: number): void;
  __wasm_call_ctors?: () => void;
};

type InitMessage = {
  type: 'init';
  index: number;
  workerCount: number;
  memory: WebAssembly.Memory;
  wasmBytes: ArrayBuffer;
  ctlBuf: SharedArrayBuffer;
  paramsBuf: SharedArrayBuffer;
};

self.onmessage = async (ev: MessageEvent<InitMessage>) => {
  const msg = ev.data;
  if (msg.type !== 'init') return;
  const { index, workerCount, memory, wasmBytes, ctlBuf, paramsBuf } = msg;
  const module = await WebAssembly.compile(wasmBytes);
  const instance = await WebAssembly.instantiate(module, confineImports(memory));
  const exp = instance.exports as unknown as ConfineExports;
  exp.__wasm_call_ctors?.();
  const ctl = new Int32Array(ctlBuf);
  const params = new Float32Array(paramsBuf);

  (self as unknown as Worker).postMessage({ type: 'ready' });

  /*
   * Blocks on its own generation slot — `Atomics.wait` is legal here because
   * this is a worker, not the main thread (the main thread uses the async
   * form; see ConfinePool.awaitCompletion). A negative generation is the
   * shutdown signal.
   */
  let lastGen = 0;
  for (;;) {
    const cur = Atomics.load(ctl, CTL_GEN_BASE + index);
    if (cur === lastGen) {
      Atomics.wait(ctl, CTL_GEN_BASE + index, cur);
      continue;
    }
    lastGen = cur;
    if (cur < 0) return;
    const n = Atomics.load(ctl, CTL_N);
    const start = Math.floor((index * n) / workerCount);
    const end = Math.floor(((index + 1) * n) / workerCount);
    if (end > start) {
      exp.solver_confine_range(
        start,
        end,
        params[PARAMS_CX],
        params[PARAMS_CY],
        params[PARAMS_DT],
        params[PARAMS_HALF],
        params[PARAMS_EDGE],
      );
    }
    Atomics.sub(ctl, CTL_REMAINING, 1);
    Atomics.notify(ctl, CTL_REMAINING);
  }
};
