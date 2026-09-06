import shader from './genome.wgsl?raw';
import { CHEM_LEN, HEAD_SCALE, SENSE_SCALE } from '../chem-layout.ts';

/**
 * WebGPU host for the genome pass.
 *
 * A separate pipeline from the field, sharing its device and its sample
 * buffer. Separate because of a hard limit rather than taste: WebGPU
 * guarantees only eight storage buffers per compute stage, counted across the
 * whole pipeline layout, and the field already uses all eight. Two pipelines
 * each count their own.
 *
 * What it reads is mostly already here. `samples` is the field probe's output,
 * so a body's four channel readings never cross the bus at all — the pass that
 * needs them runs on the same device that produced them. `chem` is a body's
 * genome, which changes only at birth, so it is uploaded when it changes and
 * not per frame.
 *
 * Everything is in list order, matching `WireAdjacency`, so the shader never
 * has to know what a slot is except to index the genome.
 */

/** Floats written per body: h(4), emit(4), taste(4), six heads. */
const OUT_STRIDE = 18;
/** 48 bytes: twelve f32/u32, and nothing here is a vec so nothing has to align. */
const UNIFORM_BYTES = 48;

export class GenomeGpu {
  ready = false;
  lastError = '';
  private device: GPUDevice | null = null;
  private layout: GPUBindGroupLayout | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private uniform: GPUBuffer | null = null;
  private chem: GPUBuffer | null = null;
  private hPrev: GPUBuffer | null = null;
  private inputs: GPUBuffer | null = null;
  private adjOff: GPUBuffer | null = null;
  private adjNei: GPUBuffer | null = null;
  private out: GPUBuffer | null = null;
  private read: GPUBuffer | null = null;
  private bodyCap = 0;
  private neiCap = 0;
  private chemCap = 0;

  /** Staging, filled by the caller before `run`. */
  hData = new Float32Array(0);
  inputData = new Float32Array(0);
  offData = new Uint32Array(0);
  neiData = new Uint32Array(0);
  outData = new Float32Array(0);

  get outStride(): number {
    return OUT_STRIDE;
  }

  /**
   * Build the pipeline on `device`, rebuilding if it is not the one we hold.
   *
   * The device check is the whole point of this signature. `FieldGpu.init`
   * has no ready guard — every call requests a fresh adapter and device and
   * rebuilds its buffers — and `Sim.openFieldGpu` calls it once per Sim, so a
   * preset reload or a second Sim gives the field a new device while this
   * kept the first. The bind group then mixes the two, and WebGPU rejects it
   * with "[Buffer] is associated with [Device], and cannot be used with
   * [Device]" — into the uncaptured-error scope, not into any try/catch here,
   * so `run` returns true having dispatched nothing and every body reads a
   * genome of zeros. Silent, and only after the second pond.
   */
  async init(device: GPUDevice): Promise<boolean> {
    if (this.ready && this.device === device) return true;
    this.ready = false;
    this.bodyCap = 0;
    this.neiCap = 0;
    this.chemCap = 0;
    this.chem = null;
    this.hPrev = null;
    this.inputs = null;
    this.adjOff = null;
    this.adjNei = null;
    this.out = null;
    this.read = null;
    try {
      const module = device.createShaderModule({ code: shader });
      const storage = { type: 'storage' } as const;
      const readonly = { type: 'read-only-storage' } as const;
      this.layout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: readonly },
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: storage },
        ],
      });
      this.pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
        compute: { module, entryPoint: 'state' },
      });
      this.uniform = device.createBuffer({
        size: UNIFORM_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device = device;
      this.ready = true;
      return true;
    } catch (e) {
      this.lastError = String(e);
      return false;
    }
  }

  /**
   * Grow the buffers to fit, and report whether the genome table moved.
   *
   * `chem` is the big one — 134 floats a body — and it is the one that does
   * not change from frame to frame, so the caller uploads it only when it has
   * to. A reallocation is one of those times.
   */
  reserve(n: number, nei: number, chemFloats: number): boolean {
    const device = this.device;
    if (!device) return false;
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    let chemMoved = false;
    if (chemFloats > this.chemCap) {
      this.chemCap = Math.max(4096, chemFloats * 2);
      this.chem?.destroy();
      this.chem = device.createBuffer({ size: this.chemCap * 4, usage: st });
      chemMoved = true;
    }
    if (n > this.bodyCap) {
      this.bodyCap = Math.max(1024, n * 2);
      this.hPrev?.destroy();
      this.inputs?.destroy();
      this.adjOff?.destroy();
      this.out?.destroy();
      this.read?.destroy();
      this.hPrev = device.createBuffer({ size: this.bodyCap * 4 * 4, usage: st });
      this.inputs = device.createBuffer({ size: this.bodyCap * 8 * 4, usage: st });
      this.adjOff = device.createBuffer({ size: (this.bodyCap + 1) * 4, usage: st });
      this.out = device.createBuffer({
        size: this.bodyCap * OUT_STRIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.read = device.createBuffer({
        size: this.bodyCap * OUT_STRIDE * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.hData = new Float32Array(this.bodyCap * 4);
      this.inputData = new Float32Array(this.bodyCap * 8);
      this.offData = new Uint32Array(this.bodyCap + 1);
      this.outData = new Float32Array(this.bodyCap * OUT_STRIDE);
    }
    if (nei > this.neiCap || !this.adjNei) {
      this.neiCap = Math.max(2048, nei * 2);
      this.adjNei?.destroy();
      this.adjNei = device.createBuffer({ size: this.neiCap * 4, usage: st });
      this.neiData = new Uint32Array(this.neiCap);
    }
    return chemMoved;
  }

  /** Push the genome table. Only when it changed — see `reserve`. */
  uploadChem(chem: Float32Array, floats: number): void {
    const device = this.device;
    if (!device || !this.chem) return;
    device.queue.writeBuffer(this.chem, 0, chem.buffer, chem.byteOffset, floats * 4);
  }

  /**
   * One frame of genome. `samples` is the field's probe buffer, borrowed.
   *
   * Returns false if the device has gone, in which case the caller keeps
   * doing it on the CPU.
   */
  async run(
    samples: GPUBuffer,
    n: number,
    nei: number,
    groundScale: number,
    energyCh: number,
  ): Promise<boolean> {
    const device = this.device;
    if (!this.ready || !device || !this.pipeline) return false;
    if (n === 0) return true;
    try {
      const u = new ArrayBuffer(UNIFORM_BYTES);
      const u32 = new Uint32Array(u);
      const f32 = new Float32Array(u);
      u32[0] = n;
      u32[1] = CHEM_LEN;
      f32[2] = 1 / SENSE_SCALE;
      f32[3] = groundScale;
      f32[4] = HEAD_SCALE.cruise;
      f32[5] = HEAD_SCALE.turn;
      f32[6] = HEAD_SCALE.align;
      f32[7] = HEAD_SCALE.sep;
      f32[8] = HEAD_SCALE.thrust;
      f32[9] = HEAD_SCALE.recoil;
      f32[10] = energyCh;
      device.queue.writeBuffer(this.uniform!, 0, u);
      device.queue.writeBuffer(this.hPrev!, 0, this.hData.buffer, this.hData.byteOffset, n * 4 * 4);
      device.queue.writeBuffer(
        this.inputs!,
        0,
        this.inputData.buffer,
        this.inputData.byteOffset,
        n * 8 * 4,
      );
      device.queue.writeBuffer(
        this.adjOff!,
        0,
        this.offData.buffer,
        this.offData.byteOffset,
        (n + 1) * 4,
      );
      if (nei > 0) {
        device.queue.writeBuffer(
          this.adjNei!,
          0,
          this.neiData.buffer,
          this.neiData.byteOffset,
          nei * 4,
        );
      }

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: this.layout!,
          entries: [
            { binding: 0, resource: { buffer: this.uniform! } },
            { binding: 1, resource: { buffer: samples } },
            { binding: 2, resource: { buffer: this.chem! } },
            { binding: 3, resource: { buffer: this.hPrev! } },
            { binding: 4, resource: { buffer: this.inputs! } },
            { binding: 5, resource: { buffer: this.adjOff! } },
            { binding: 6, resource: { buffer: this.adjNei! } },
            { binding: 7, resource: { buffer: this.out! } },
          ],
        }),
      );
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / 64)));
      pass.end();
      const bytes = n * OUT_STRIDE * 4;
      enc.copyBufferToBuffer(this.out!, 0, this.read!, 0, bytes);
      device.queue.submit([enc.finish()]);
      await this.read!.mapAsync(GPUMapMode.READ, 0, bytes);
      this.outData.set(new Float32Array(this.read!.getMappedRange(0, bytes)));
      this.read!.unmap();
      return true;
    } catch (e) {
      this.lastError = String(e);
      return false;
    }
  }
}

export const genomeGpu = new GenomeGpu();
