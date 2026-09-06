import shader from './field.wgsl?raw';
import { CHANNELS, type Fields } from '../fields.ts';

/**
 * WebGPU host for the scent field.
 *
 * The field lives here for the whole session once a device exists — it is
 * never read back except to draw the overlay. That is the point: the two
 * diffusion passes and the decay are 13.7ms a frame on the CPU at a million
 * cells, which is the second largest fixed cost in the frame after the solve,
 * and they are a bandwidth-bound stencil, which is the one thing a GPU is
 * unambiguously for.
 *
 * What crosses each frame is small and goes both ways: a list of world
 * positions to deposit at, a list of world positions to sample, and the
 * handful of scalars per body that come back. The host owns all the geometry;
 * see the note at the top of field.wgsl for why.
 */

/*
 * 144, and the layout is dictated by WGSL rather than by taste: a `vec4f` must
 * sit on a sixteen-byte boundary, so the scalars are grouped in fours. See
 * `FieldParams` in field.wgsl, which this has to match exactly — a field
 * written at the wrong offset reads as a plausible number rather than an
 * error, which is the whole hazard of hand-packing a uniform.
 */
const UNIFORM_BYTES = 144;
/** Floats per Deposit and per Probe in the shader's layout. */
const DEPOSIT_FLOATS = 8;
const PROBE_FLOATS = 12;
/**
 * Fixed point for the deposit accumulator. WGSL atomics are integer only, so
 * a scatter with collisions has to accumulate in integers. Scent values run to
 * a few tens, deposits to fractions of one, and an i32 has nine digits — 1e4
 * leaves four decimal places and five orders of headroom before overflow.
 */
const FIXED_SCALE = 1e4;

type Entry =
  | 'clearAcc'
  | 'scatter'
  | 'applyAcc'
  | 'diffuse'
  | 'diffuse2'
  | 'react'
  | 'decay'
  | 'grow'
  | 'gather';

export class FieldGpu {
  ready = false;
  lastError = '';
  private device: GPUDevice | null = null;
  private layout: GPUBindGroupLayout | null = null;
  private pipelines = new Map<Entry, GPUComputePipeline>();
  private uniform: GPUBuffer | null = null;
  private fieldA: GPUBuffer | null = null;
  private fieldB: GPUBuffer | null = null;
  private acc: GPUBuffer | null = null;
  private deposits: GPUBuffer | null = null;
  private probes: GPUBuffer | null = null;
  private samples: GPUBuffer | null = null;
  private readback: GPUBuffer | null = null;
  private cells = 0;
  private depositCap = 0;
  private probeCap = 0;
  /** True when `fieldA` is the live one; the diffusion pass ping-pongs. */
  private aLive = true;
  /** Host-side staging, reused so a frame allocates nothing. */
  depositData = new Float32Array(0);
  probeData = new Float32Array(0);
  sampleData = new Float32Array(0);

  async init(cells: number): Promise<boolean> {
    const nav = navigator as Navigator & { gpu?: GPU };
    if (!nav.gpu) {
      this.lastError = 'no navigator.gpu';
      return false;
    }
    try {
      const adapter = await nav.gpu.requestAdapter();
      if (!adapter) {
        this.lastError = 'no adapter';
        return false;
      }
      const device = await adapter.requestDevice();
      device.lost.then(() => {
        this.ready = false;
        this.device = null;
      });
      /*
       * WebGPU reports validation failures asynchronously and then silently
       * drops the work. A bad binding therefore looks exactly like a shader
       * that computed nothing, which is a bad way to spend an afternoon.
       */
      device.addEventListener('uncapturederror', (e) => {
        const err = (e as GPUUncapturedErrorEvent).error;
        this.lastError = String(err.message ?? err);
        console.error('field-gpu:', this.lastError);
      });
      this.device = device;
      const module = device.createShaderModule({ code: shader });
      // Compilation failures are reported here, not thrown: without this a bad
      // shader yields an invalid pipeline that silently drops every dispatch.
      const info = await module.getCompilationInfo();
      const bad = info.messages.filter((m) => m.type === 'error');
      if (bad.length > 0) {
        this.lastError = bad.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
        console.error('field.wgsl:\n' + this.lastError);
        this.ready = false;
        return false;
      }
      const storage = { type: 'storage' } as const;
      const readonly = { type: 'read-only-storage' } as const;
      this.layout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: storage },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: storage },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: storage },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: storage },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
      for (const name of [
        'clearAcc',
        'scatter',
        'applyAcc',
        'diffuse',
        'diffuse2',
        'react',
        'decay',
        'grow',
        'gather',
      ] as const) {
        this.pipelines.set(
          name,
          device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint: name },
          }),
        );
      }
      this.uniform = device.createBuffer({
        size: UNIFORM_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.allocField(cells);
      this.ready = true;
      return true;
    } catch (e) {
      this.lastError = String(e);
      this.ready = false;
      this.device = null;
      return false;
    }
  }

  private allocField(cells: number): void {
    const device = this.device!;
    this.cells = cells * cells;
    const bytes = this.cells * CHANNELS * 4;
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.fieldA?.destroy();
    this.fieldB?.destroy();
    this.acc?.destroy();
    this.fieldA = device.createBuffer({ size: bytes, usage: st });
    this.fieldB = device.createBuffer({ size: bytes, usage: st });
    this.acc = device.createBuffer({ size: bytes, usage: st });
    this.aLive = true;
  }

  /** Size the staging arrays before the caller writes into them. */
  reserve(nDeposit: number, nProbe: number): void {
    if (this.device) this.ensureLists(nDeposit, nProbe);
  }

  private ensureLists(nDeposit: number, nProbe: number): void {
    const device = this.device!;
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    if (nDeposit > this.depositCap) {
      this.depositCap = Math.max(1024, nDeposit * 2);
      this.deposits?.destroy();
      this.deposits = device.createBuffer({
        size: this.depositCap * DEPOSIT_FLOATS * 4,
        usage: st,
      });
      this.depositData = new Float32Array(this.depositCap * DEPOSIT_FLOATS);
    }
    if (nProbe > this.probeCap) {
      this.probeCap = Math.max(1024, nProbe * 2);
      this.probes?.destroy();
      this.samples?.destroy();
      this.readback?.destroy();
      this.probes = device.createBuffer({ size: this.probeCap * PROBE_FLOATS * 4, usage: st });
      this.samples = device.createBuffer({
        size: this.probeCap * 4 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.readback = device.createBuffer({
        size: this.probeCap * 4 * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.probeData = new Float32Array(this.probeCap * PROBE_FLOATS);
      this.sampleData = new Float32Array(this.probeCap * 4);
    }
  }

  /**
   * One frame of field: scatter the deposits, fold them in, diffuse twice,
   * decay, and gather the probes. Returns false if the GPU is not available,
   * in which case the caller keeps doing it on the CPU.
   *
   * The gathered samples are read back one frame late by design — mapping a
   * buffer stalls the pipeline, so the read is issued now and collected on the
   * next call. Sensing a frame behind is invisible at 60fps and worth far more
   * than the stall costs.
   */
  async step(
    fields: Fields,
    nDeposit: number,
    nProbe: number,
    mix: number,
    mix2: number,
    decayRate: number,
    grow: { ch: number; r: number; cap: number; catCh: number; gamma: number },
    react: { u: number; v: number; feed: number; kill: number; dt: number },
  ): Promise<boolean> {
    const device = this.device;
    if (!this.ready || !device || !this.fieldA || !this.fieldB || !this.acc) return false;
    try {
      this.ensureLists(nDeposit, nProbe);
      const u = new ArrayBuffer(UNIFORM_BYTES);
      const u32 = new Uint32Array(u);
      const f32 = new Float32Array(u);
      u32[0] = fields.cols;
      u32[1] = fields.rows;
      u32[2] = nDeposit;
      u32[3] = nProbe;
      f32[4] = fields.originX;
      f32[5] = fields.originY;
      f32[6] = fields.worldW;
      f32[7] = FIXED_SCALE;
      f32[8] = fields.boundX;
      f32[9] = fields.boundY;
      f32[10] = fields.boundR;
      f32[11] = grow.ch;
      /*
       * Per-channel rates, resolved the same way `Fields` does: the slider is
       * one number for the world and each channel scales it. Clamped at one,
       * because a diffusion mix above it is a cell overshooting its own
       * neighbours, which oscillates and then blows up.
       */
      for (let c = 0; c < 4; c++) {
        const dr = fields.diffuseRate[c] > 0 ? fields.diffuseRate[c] : 0;
        const kr = fields.decayRate[c] > 0 ? fields.decayRate[c] : 0;
        const m1 = mix * dr;
        const m2 = mix2 * dr;
        const k = decayRate * kr;
        f32[12 + c] = m1 > 1 ? 1 : m1;
        f32[16 + c] = m2 > 1 ? 1 : m2;
        f32[20 + c] = k >= 1 ? 0 : 1 - k;
      }
      f32[24] = grow.r;
      f32[25] = grow.cap;
      f32[26] = grow.gamma;
      f32[27] = grow.catCh;
      f32[28] = react.feed * react.dt;
      f32[29] = (react.feed + react.kill) * react.dt;
      f32[30] = react.dt;
      f32[32] = react.u;
      f32[33] = react.v;
      device.queue.writeBuffer(this.uniform!, 0, u);
      if (nDeposit > 0) {
        device.queue.writeBuffer(
          this.deposits!,
          0,
          this.depositData.buffer,
          this.depositData.byteOffset,
          nDeposit * DEPOSIT_FLOATS * 4,
        );
      }
      if (nProbe > 0) {
        device.queue.writeBuffer(
          this.probes!,
          0,
          this.probeData.buffer,
          this.probeData.byteOffset,
          nProbe * PROBE_FLOATS * 4,
        );
      }

      const enc = device.createCommandEncoder();
      const groups = (n: number): number => Math.max(1, Math.ceil(n / 64));
      const run = (name: Entry, n: number, live: GPUBuffer, other: GPUBuffer): void => {
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipelines.get(name)!);
        pass.setBindGroup(
          0,
          device.createBindGroup({
            layout: this.layout!,
            entries: [
              { binding: 0, resource: { buffer: this.uniform! } },
              { binding: 1, resource: { buffer: live } },
              { binding: 2, resource: { buffer: other } },
              { binding: 3, resource: { buffer: this.acc! } },
              { binding: 4, resource: { buffer: this.deposits! } },
              { binding: 5, resource: { buffer: this.probes! } },
              { binding: 6, resource: { buffer: this.samples! } },
            ],
          }),
        );
        pass.dispatchWorkgroups(groups(n));
        pass.end();
      };

      let live = this.aLive ? this.fieldA : this.fieldB;
      let other = this.aLive ? this.fieldB : this.fieldA;
      if (nDeposit > 0) {
        run('scatter', nDeposit, live, other);
        run('applyAcc', this.cells, live, other);
      }
      // Two passes at different mixes then a decay, matching the CPU order.
      run('diffuse', this.cells, live, other);
      [live, other] = [other, live];
      run('diffuse2', this.cells, live, other);
      [live, other] = [other, live];
      // Same order as the CPU: spread, react, decay, grow. `react` and `grow`
      // both work in place on the live buffer, so neither swaps.
      run('react', this.cells, live, other);
      run('decay', this.cells, live, other);
      run('grow', this.cells, live, other);
      if (nProbe > 0) run('gather', nProbe, live, other);
      // Two swaps, so the live buffer is back where it started.
      void other;

      if (nProbe > 0) {
        enc.copyBufferToBuffer(this.samples!, 0, this.readback!, 0, nProbe * 16);
      }
      device.queue.submit([enc.finish()]);
      if (nProbe > 0) {
        await this.readback!.mapAsync(GPUMapMode.READ, 0, nProbe * 16);
        this.sampleData.set(new Float32Array(this.readback!.getMappedRange(0, nProbe * 16)));
        this.readback!.unmap();
      }
      return true;
    } catch (e) {
      this.lastError = String(e);
      return false;
    }
  }

  /** Debug: pull the live field back and report what is in it. */
  async debugField(): Promise<{ nonZero: number; max: number; total: number }> {
    const device = this.device;
    const live = this.aLive ? this.fieldA : this.fieldB;
    if (!device || !live) return { nonZero: 0, max: 0, total: 0 };
    const bytes = this.cells * CHANNELS * 4;
    const rb = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(live, 0, rb, 0, bytes);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange());
    let nonZero = 0;
    let max = 0;
    let total = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== 0) nonZero++;
      max = Math.max(max, Math.abs(a[i]));
      total += a[i];
    }
    rb.unmap();
    rb.destroy();
    return { nonZero, max, total };
  }

  get depositStride(): number {
    return DEPOSIT_FLOATS;
  }

  get probeStride(): number {
    return PROBE_FLOATS;
  }
}

export const fieldGpu = new FieldGpu();
