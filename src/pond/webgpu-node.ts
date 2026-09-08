/*
 * A WebGPU device for Node, so the headless pond can run the shaders the page
 * already runs.
 *
 * `field.wgsl` and `genome.wgsl` ship in the browser and are checked against
 * their CPU twins by `field-kernel.test.ts` and `genome-kernel.test.ts`. The
 * only reason a headless run could not use them is that Node has no
 * `navigator.gpu`. That is what this supplies, out of Dawn — the same
 * implementation Chrome's WebGPU is built on — through the `webgpu` package.
 *
 * ## Why a global shim and not an injected device
 *
 * `FieldGpu.init` reaches for `navigator.gpu` itself, as browser code should.
 * Rewriting it to take an injected adapter would mean changing the shipping
 * render path to serve a headless tool, and every such change is a chance to
 * break the thing that ships for the thing that does not. Supplying the API
 * Dawn exists to supply leaves `src/gpu/` untouched: on this path the page's
 * code runs unmodified, which is the property that makes a lineage grown here
 * one the page can open.
 *
 * `navigator` is proxied rather than replaced, so Node's own
 * `hardwareConcurrency` and the rest survive. Nothing in `src/` reads any of
 * them today; that is not a reason to break them.
 *
 * ## The dependency
 *
 * `webgpu` is an *optional* dependency: about 21 MB of Dawn per platform, and
 * a machine that cannot install it should still be able to run a pond. Every
 * entry point here degrades to the CPU field rather than failing, and the
 * import is dynamic so nothing on the browser's module graph can reach it.
 */

export interface GpuOpen {
  ok: boolean;
  /** Why not, when `ok` is false. */
  error?: string;
  /** What Dawn picked, when it says. */
  adapter?: string;
}

let opened: GpuOpen | null = null;

/**
 * Install `navigator.gpu` and the WebGPU globals, once per process.
 *
 * Idempotent, and a no-op where `navigator.gpu` already exists — which is the
 * browser, and also a second call here.
 */
export async function openWebGpu(): Promise<GpuOpen> {
  if (opened) return opened;
  const nav = (globalThis as { navigator?: { gpu?: unknown } }).navigator;
  if (nav?.gpu) {
    opened = { ok: true };
    return opened;
  }
  try {
    // Dynamic and by variable, so neither Vite's browser build nor a reader
    // of the module graph can be tempted to follow it.
    const mod: unknown = await import(/* @vite-ignore */ 'webgpu');
    const { create, globals } = mod as {
      create(options: string[]): unknown;
      globals: Record<string, unknown>;
    };
    // `GPUShaderStage`, `GPUBufferUsage`, `GPUMapMode` and the rest: the
    // shader modules read them as globals, exactly as a page would.
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

/**
 * Let go of the device so the process can exit.
 *
 * A live `GPUDevice` holds the event loop open — a run that finished would sit
 * there with the database closed and nothing left to do. Called on the way
 * out; failing here is not worth reporting, since the only thing left is to
 * exit anyway.
 */
export function closeWebGpu(device: { destroy(): void } | null | undefined): void {
  try {
    device?.destroy();
  } catch {
    // Nothing useful to do about it at this point.
  }
}
