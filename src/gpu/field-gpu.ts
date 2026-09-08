import shader from './field.wgsl?raw';
import { HARVEST_STRIDE } from '../energy.ts';
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
 * 160, and the layout is dictated by WGSL rather than by taste: a `vec4f` must
 * sit on a sixteen-byte boundary, so the scalars are grouped in fours. See
 * `FieldParams` in field.wgsl, which this has to match exactly — a field
 * written at the wrong offset reads as a plausible number rather than an
 * error, which is the whole hazard of hand-packing a uniform.
 */
const UNIFORM_BYTES = 160;
/** Floats per Deposit and per Probe in the shader's layout. */
const DEPOSIT_FLOATS = 8;
const PROBE_FLOATS = 12;
/**
 * Floats per body coming back: three taste-collapsed sensor readings for
 * steering, then the raw four channels under the body for the genome's sense
 * columns. See `gather` in field.wgsl for why both, and why the second is
 * free.
 */
const SAMPLE_FLOATS = 8;
/** u32 per HarvestBlock in the shader's layout. */
const BLOCK_WORDS = 8;
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
  | 'harvest'
  | 'fill'
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
  private hBlocks: GPUBuffer | null = null;
  private hFlow: GPUBuffer | null = null;
  private hRead: GPUBuffer | null = null;
  private blockCap = 0;
  private entryCap = 0;
  private cells = 0;
  private depositCap = 0;
  private probeCap = 0;
  /** True when `fieldA` is the live one; the diffusion pass ping-pongs. */
  private aLive = true;
  /** Host-side staging, reused so a frame allocates nothing. */
  depositData = new Float32Array(0);
  probeData = new Float32Array(0);
  sampleData = new Float32Array(0);
  /**
   * Harvest staging, filled by the caller before `step`. One entry per
   * (block, body) pair in the order that block feeds its bodies; `blockData`
   * indexes into it. See `harvest` in field.wgsl.
   */
  blockData = new Uint32Array(0);
  roomData = new Float32Array(0);
  gotData = new Float32Array(0);

  /**
   * Build the field on a device, once.
   *
   * The guard is not an optimisation. Without it every call requested a fresh
   * adapter and device — and `Sim.openFieldGpu` runs once per Sim, so a preset
   * reload or a second pond got device B while every capacity counter here
   * still said the buffers were big enough, so they were never rebuilt and the
   * field went on using device A's memory. Worse, `GenomeGpu` then held a
   * different device again, and WebGPU rejects a bind group that mixes two:
   * "[Buffer] is associated with [Device], and cannot be used with [Device]".
   * That lands in the uncaptured-error scope rather than any try/catch, so the
   * dispatch quietly does nothing and every body reads a genome of zeros —
   * on the second pond of a session and never the first.
   *
   * A device is also not free to abandon: the old one and all its buffers
   * stayed alive, once per Sim, for the life of the tab.
   */
  async init(cells: number): Promise<boolean> {
    // `this.cells` is the total, `cells` the side; comparing them directly
    // would rebuild every time on any grid but a 1x1.
    if (this.ready && this.device && this.cells === cells * cells) return true;
    const nav = navigator as Navigator & { gpu?: GPU };
    if (!nav.gpu) {
      this.lastError = 'no navigator.gpu';
      return false;
    }
    // A rebuild is a new device, so nothing allocated against the old one may
    // be reused. Zeroing the caps is what forces `ensureLists` to notice.
    this.depositCap = 0;
    this.probeCap = 0;
    this.blockCap = 0;
    this.entryCap = 0;
    this.deposits = null;
    this.probes = null;
    this.samples = null;
    this.readback = null;
    this.hBlocks = null;
    this.hFlow = null;
    this.hRead = null;
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
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: storage },
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
        'harvest',
        'fill',
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
  reserve(nDeposit: number, nProbe: number, nBlocks = 0, nEntries = 0): void {
    if (!this.device) return;
    this.ensureLists(nDeposit, nProbe);
    this.ensureHarvest(nBlocks, nEntries);
  }

  get blockStride(): number {
    return BLOCK_WORDS;
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
        size: this.probeCap * SAMPLE_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.readback = device.createBuffer({
        size: this.probeCap * SAMPLE_FLOATS * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.probeData = new Float32Array(this.probeCap * PROBE_FLOATS);
      this.sampleData = new Float32Array(this.probeCap * SAMPLE_FLOATS);
    }
  }

  /**
   * Harvest lists. Separate from `ensureLists` because they are sized by
   * occupied blocks and hungry bodies rather than by population — a full pond
   * of sated bodies harvests nothing at all — and because they must exist at
   * some non-zero size even then: the bind group is built once and every
   * dispatch uses it, so a null binding would fail every pass and not just
   * this one.
   */
  private ensureHarvest(nBlocks: number, nEntries: number): void {
    const device = this.device!;
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    if (nBlocks > this.blockCap || !this.hBlocks) {
      this.blockCap = Math.max(1024, nBlocks * 2);
      this.hBlocks?.destroy();
      this.hBlocks = device.createBuffer({ size: this.blockCap * BLOCK_WORDS * 4, usage: st });
      this.blockData = new Uint32Array(this.blockCap * BLOCK_WORDS);
    }
    if (nEntries > this.entryCap || !this.hFlow) {
      this.entryCap = Math.max(1024, nEntries * 2);
      this.hFlow?.destroy();
      this.hRead?.destroy();
      // `HARVEST_STRIDE` floats an entry: room, then a rate and an affinity
      // per species. Wider rather than a second buffer, because this pass
      // binds eight storage buffers of a guaranteed eight. See `energy.ts`.
      const floats = this.entryCap * HARVEST_STRIDE;
      this.hFlow = device.createBuffer({
        size: floats * 4,
        usage: st | GPUBufferUsage.COPY_SRC,
      });
      this.hRead = device.createBuffer({
        size: floats * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.roomData = new Float32Array(floats);
      this.gotData = new Float32Array(floats);
    }
  }

  /**
   * Zero the field on the device: a new pond in the same session.
   *
   * `Fields.clear` only empties the CPU mirror, and on this path nothing
   * reads that. Without this the old pond's scent was still in the buffers
   * when the new one pinned its world, and its bodies were steering on trails
   * nobody had laid.
   */
  clear(): void {
    const device = this.device;
    if (!this.ready || !device || !this.fieldA || !this.fieldB || !this.acc) return;
    const enc = device.createCommandEncoder();
    enc.clearBuffer(this.fieldA);
    enc.clearBuffer(this.fieldB);
    enc.clearBuffer(this.acc);
    device.queue.submit([enc.finish()]);
    this.aLive = true;
  }

  /** What the last `submit` asked for, so `collect` knows what to wait on. */
  private pendingProbe = 0;
  private pendingEntries = 0;

  /**
   * One frame of field: scatter the deposits, fold them in, diffuse twice,
   * decay, and gather the probes. Returns false if the GPU is not available,
   * in which case the caller keeps doing it on the CPU.
   *
   * `submit` and `collect` are separable so a second pipeline on the same
   * device — the genome's — can be queued behind this one before either
   * readback is waited on, which turns two round trips a frame into one.
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
    harvest: { ch: number; blocks: number; entries: number; uptakeCap: number; uptakeKs: number },
    fill: { ch: number; value: number } | null,
  ): Promise<boolean> {
    if (!this.submit(fields, nDeposit, nProbe, mix, mix2, decayRate, grow, react, harvest, fill)) {
      return false;
    }
    return this.collect();
  }

  /** Encode and submit a frame's passes. False if the device is not there. */
  submit(
    fields: Fields,
    nDeposit: number,
    nProbe: number,
    mix: number,
    mix2: number,
    decayRate: number,
    grow: { ch: number; r: number; cap: number; catCh: number; gamma: number },
    react: { u: number; v: number; feed: number; kill: number; dt: number },
    harvest: { ch: number; blocks: number; entries: number; uptakeCap: number; uptakeKs: number },
    fill: { ch: number; value: number } | null,
  ): boolean {
    const device = this.device;
    if (!this.ready || !device || !this.fieldA || !this.fieldB || !this.acc) return false;
    try {
      this.ensureLists(nDeposit, nProbe);
      this.ensureHarvest(harvest.blocks, harvest.entries);
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
      u32[34] = harvest.blocks;
      f32[35] = harvest.ch;
      f32[36] = fill ? fill.ch : 0;
      f32[37] = fill ? fill.value : 0;
      // The two slots the struct used to pad with. See `harvest` in field.wgsl.
      f32[38] = harvest.uptakeCap;
      f32[39] = harvest.uptakeKs;
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
      if (harvest.blocks > 0) {
        device.queue.writeBuffer(
          this.hBlocks!,
          0,
          this.blockData.buffer,
          this.blockData.byteOffset,
          harvest.blocks * BLOCK_WORDS * 4,
        );
        device.queue.writeBuffer(
          this.hFlow!,
          0,
          this.roomData.buffer,
          this.roomData.byteOffset,
          harvest.entries * HARVEST_STRIDE * 4,
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
              { binding: 7, resource: { buffer: this.hBlocks! } },
              { binding: 8, resource: { buffer: this.hFlow! } },
            ],
          }),
        );
        pass.dispatchWorkgroups(groups(n));
        pass.end();
      };

      let live = this.aLive ? this.fieldA : this.fieldB;
      let other = this.aLive ? this.fieldB : this.fieldA;
      // Before everything: a seed is the world being laid down, not a thing
      // that happens to it, so this frame's passes should act on the result.
      if (fill) run('fill', this.cells, live, other);
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
      /*
       * Last, and after the field passes rather than before them.
       *
       * On the CPU harvest is the first thing in `endFrame` and the field
       * passes are the last, so a frame's grazing sees the field as the
       * previous frame's passes left it. This runs at the end of a frame and
       * is credited at the top of the next, so putting it after the passes
       * here lands on exactly the same field state — the pipelining is exact
       * rather than approximate, which is the whole reason it is safe.
       */
      if (harvest.blocks > 0) run('harvest', harvest.blocks, live, other);
      if (nProbe > 0) run('gather', nProbe, live, other);
      // Two swaps, so the live buffer is back where it started.
      void other;

      if (nProbe > 0) {
        enc.copyBufferToBuffer(this.samples!, 0, this.readback!, 0, nProbe * SAMPLE_FLOATS * 4);
      }
      if (harvest.entries > 0) {
        enc.copyBufferToBuffer(this.hFlow!, 0, this.hRead!, 0, harvest.entries * HARVEST_STRIDE * 4);
      }
      device.queue.submit([enc.finish()]);
      this.pendingProbe = nProbe;
      this.pendingEntries = harvest.entries;
      return true;
    } catch (e) {
      this.lastError = String(e);
      return false;
    }
  }

  /**
   * Wait for the last `submit`'s readbacks and unpack them into `sampleData`
   * and `gotData`. Both maps are issued before either is awaited, since they
   * ride the same submission and finish together.
   */
  async collect(): Promise<boolean> {
    const nProbe = this.pendingProbe;
    const entries = this.pendingEntries;
    this.pendingProbe = 0;
    this.pendingEntries = 0;
    if (!this.ready || !this.device) return false;
    try {
      const sampleBytes = nProbe * SAMPLE_FLOATS * 4;
      const gotBytes = entries * HARVEST_STRIDE * 4;
      const waits: Promise<void>[] = [];
      if (nProbe > 0) waits.push(this.readback!.mapAsync(GPUMapMode.READ, 0, sampleBytes));
      if (entries > 0) waits.push(this.hRead!.mapAsync(GPUMapMode.READ, 0, gotBytes));
      if (waits.length > 0) await Promise.all(waits);
      if (nProbe > 0) {
        this.sampleData.set(new Float32Array(this.readback!.getMappedRange(0, sampleBytes)));
        this.readback!.unmap();
      }
      if (entries > 0) {
        this.gotData.set(new Float32Array(this.hRead!.getMappedRange(0, gotBytes)));
        this.hRead!.unmap();
      }
      return true;
    } catch (e) {
      this.lastError = String(e);
      return false;
    }
  }

  /**
   * Copy the live field into a `Fields`, for the CPU-side views that read it.
   *
   * Sixteen megabytes, so this is not something to do every frame — it exists
   * because the scent overlay and the energy-grid overlay paint from
   * `fields.data`, and both are opt-in. When nobody has them open nothing here
   * runs, and when somebody does, one copy a frame is the price of looking.
   *
   * Allocates its own staging buffer rather than keeping one alive: holding
   * sixteen megabytes of mapped memory for a debug view nobody has opened is
   * the wrong default, and the allocation is nothing against the copy.
   */
  async readInto(fields: Fields): Promise<boolean> {
    const device = this.device;
    const live = this.aLive ? this.fieldA : this.fieldB;
    if (!this.ready || !device || !live) return false;
    const bytes = this.cells * CHANNELS * 4;
    if (fields.data.length * 4 !== bytes) return false;
    const rb = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(live, 0, rb, 0, bytes);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      fields.data.set(new Float32Array(rb.getMappedRange()));
      rb.unmap();
      return true;
    } catch (e) {
      this.lastError = String(e);
      return false;
    } finally {
      rb.destroy();
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

  get sampleStride(): number {
    return SAMPLE_FLOATS;
  }

  /**
   * The device and the probe's output buffer, for the genome pass.
   *
   * It is a second pipeline on the same device and it reads the raw channel
   * readings `gather` leaves here — the whole point of that being a shared
   * buffer rather than a readback is that the pass which needs it runs on the
   * same side of the bus.
   */
  get gpuDevice(): GPUDevice | null {
    return this.device;
  }

  get sampleBuffer(): GPUBuffer | null {
    return this.samples;
  }
}

export const fieldGpu = new FieldGpu();
