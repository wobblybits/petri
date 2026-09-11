/*
 * A WebGPU device for Node, out of Dawn through the `webgpu` package, so the
 * headless pond can run the shaders the page runs. A global `navigator.gpu`
 * shim rather than an injected device, so `src/gpu/` runs unmodified;
 * `navigator` is proxied rather than replaced so Node's own accessors
 * survive. `webgpu` is an optional dependency: every entry point here
 * degrades to the CPU field, and the import is dynamic so nothing on the
 * browser's module graph can reach it.
 */

export interface GpuOpen {
  ok: boolean;
  /** Why not, when `ok` is false. */
  error?: string;
  /** What Dawn picked, when it says. */
  adapter?: string;
}

let opened: GpuOpen | null = null;

/** Install `navigator.gpu` and the WebGPU globals, once per process. A no-op where `navigator.gpu` exists. */
export async function openWebGpu(): Promise<GpuOpen> {
  if (opened) return opened;
  const nav = (globalThis as { navigator?: { gpu?: unknown } }).navigator;
  if (nav?.gpu) {
    opened = { ok: true };
    return opened;
  }
  try {
    // Dynamic and by variable, so Vite's browser build cannot follow it.
    const mod: unknown = await import(/* @vite-ignore */ 'webgpu');
    const { create, globals } = mod as {
      create(options: string[]): unknown;
      globals: Record<string, unknown>;
    };
    // The shader hosts read `GPUShaderStage` and the rest as globals.
    Object.assign(globalThis, globals);
    const gpu = create([]);

    const real = globalThis.navigator as object | undefined;
    const shim = real
      ? new Proxy(real, {
          // The receiver has to be the real navigator, not the proxy: Node's
          // accessors are brand-checked and throw on a foreign `this`.
          get: (t, p) => (p === 'gpu' ? gpu : Reflect.get(t, p, t)),
          has: (t, p) => p === 'gpu' || Reflect.has(t, p),
        })
      : { gpu };
    Object.defineProperty(globalThis, 'navigator', {
      value: shim,
      configurable: true,
      writable: true,
    });

    const adapter = await (gpu as { requestAdapter(): Promise<unknown> }).requestAdapter();
    if (!adapter) {
      opened = { ok: false, error: 'Dawn found no adapter' };
      return opened;
    }
    const info = (adapter as { info?: { device?: string; description?: string } }).info;
    opened = { ok: true, adapter: info?.device ?? info?.description ?? 'Dawn' };
    return opened;
  } catch (e) {
    opened = {
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    };
    return opened;
  }
}

/** Let go of the device so the process can exit: a live `GPUDevice` holds the event loop open. */
export function closeWebGpu(device: { destroy(): void } | null | undefined): void {
  try {
    device?.destroy();
  } catch {
  }
}
