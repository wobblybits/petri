/**
 * The bit of DedicatedWorkerGlobalScope net-worker.ts uses.
 *
 * The worker is bundled import-free and loaded with importScripts, after a
 * bootstrap has set `sampleRate` — so it is neither a DOM script nor a module,
 * and the lib.dom `self` is the wrong shape for it.
 */
declare const self: {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage(message: unknown): void;
};
