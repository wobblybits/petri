import shader from './genome.wgsl?raw';
import { CHEM_LEN, HEAD_SCALE, LEARN_STRIDE } from '../chem-layout.ts';

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
/** 80 bytes: twenty f32/u32, and nothing here is a vec so nothing has to align. */
const UNIFORM_BYTES = 80;
/** Most learning rows read back in one frame. A frame that begins more
 *  rewrites than this leaves the rest marked for the next one. */
const LEARN_READ_CAP = 64;

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
  /** Resident learning state, one `LEARN_STRIDE` row a slot. */
  private learn: GPUBuffer | null = null;
  private learnRead: GPUBuffer | null = null;
  private bodyCap = 0;
  private neiCap = 0;
  private chemCap = 0;
  private learnCap = 0;

  /** Staging, filled by the caller before `run`. */
  hData = new Float32Array(0);
  inputData = new Float32Array(0);
  offData = new Uint32Array(0);
  neiData = new Uint32Array(0);
  outData = new Float32Array(0);
  /** Rows to push, filled by the caller before `pushLearn`. */
  learnUpData = new Float32Array(0);
  /** Rows that came back, in the order `readLearn` asked for them. */
  learnOutData = new Float32Array(0);
  private learnWant = new Int32Array(LEARN_READ_CAP);
  private learnWantN = 0;

  get outStride(): number {
    return OUT_STRIDE;
  }

  get learnStride(): number {
    return LEARN_STRIDE;
  }

  /** Rows `collect` brought back, in the order `readLearn` asked for them. */
  get learnCount(): number {
    return this.learnGot;
  }

  private learnGot = 0;

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
    this.learn = null;
    this.learnRead = null;
    this.learnCap = 0;
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
          { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: storage },
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
  /**
   * Grow to fit, and report whether the genome table moved.
   *
   * `learnSlots` sizes the resident learning state, which is indexed by slot
   * like the genome. Growing that one loses every body's learning, so it is
   * grown generously and the caller re-pushes what it has.
   */
  reserve(n: number, nei: number, chemFloats: number, learnSlots = 0): boolean {
    const device = this.device;
    if (!device) return false;
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    if (learnSlots > this.learnCap || !this.learn) {
      const oldLearn = this.learn;
      const oldBytes = this.learnCap * LEARN_STRIDE * 4;
      this.learnCap = Math.max(1024, learnSlots * 2);
      this.learnRead?.destroy();
      this.learn = device.createBuffer({
        size: this.learnCap * LEARN_STRIDE * 4,
        usage: st | GPUBufferUsage.COPY_SRC,
      });
      /*
       * Carried over rather than started again. This is the one buffer here
       * whose contents are not recomputed every frame — it is what every
       * body has learned — and a pond passes a thousand bodies in seconds,
       * so a doubling that dropped it would wipe the pond's memory
       * repeatedly and look like learning that does not stick.
       */
      if (oldLearn && oldBytes > 0) {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(oldLearn, 0, this.learn, 0, oldBytes);
        device.queue.submit([enc.finish()]);
        oldLearn.destroy();
      }
      this.learnRead = device.createBuffer({
        size: LEARN_READ_CAP * LEARN_STRIDE * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.learnUpData = new Float32Array(this.learnCap * LEARN_STRIDE);
      this.learnOutData = new Float32Array(LEARN_READ_CAP * LEARN_STRIDE);
    }
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

  /**
   * Push `count` learning rows starting at slot `lo`, from `learnUpData`.
   *
   * The host writes this only to zero a slot that has been recycled, which
   * matters more than it sounds: a slot handed on with the device copy still
   * in it would give a newborn the previous occupant's experience.
   */
  pushLearn(lo: number, count: number): void {
    const device = this.device;
    if (!device || !this.learn || count <= 0) return;
    device.queue.writeBuffer(
      this.learn,
      lo * LEARN_STRIDE * 4,
      this.learnUpData.buffer,
      this.learnUpData.byteOffset,
      count * LEARN_STRIDE * 4,
    );
  }

  /**
   * Ask for these slots' learning rows on the next `collect`.
   *
   * Only the two bodies of a rewrite need this, and only so the CPU can
   * consolidate what they learned into their children's genome, where
   * inheritance lives. `beginRewrite` gives about forty frames of notice,
   * which is why a handful of small copies is enough and the whole buffer
   * never has to come back.
   */
  readLearn(slots: ArrayLike<number>, count: number): number {
    const take = Math.min(count, LEARN_READ_CAP);
    for (let i = 0; i < take; i++) this.learnWant[i] = slots[i];
    this.learnWantN = take;
    return take;
  }

  /** Push the whole genome table. Only when it moved — see `reserve`. */
  uploadChem(chem: Float32Array, floats: number): void {
    this.uploadChemRange(chem, 0, floats);
  }

  /**
   * Push the floats `[lo, hi)` of the genome table: the slots that changed
   * since the last upload, which `AgentStore` tracks as a dirty range. A frame
   * with one birth uploads one genome rather than every one in the pond.
   */
  uploadChemRange(chem: Float32Array, lo: number, hi: number): void {
    const device = this.device;
    if (!device || !this.chem || hi <= lo) return;
    device.queue.writeBuffer(this.chem, lo * 4, chem.buffer, chem.byteOffset + lo * 4, (hi - lo) * 4);
  }

  /** Bytes the last `submit` copied out, for `collect` to map. */
  private pendingBytes = 0;
  /** Learning rows the last `submit` copied out. */
  private pendingLearn = 0;

  /**
   * One frame of genome. `samples` is the field's probe buffer, borrowed.
   *
   * Returns false if the device has gone, in which case the caller keeps
   * doing it on the CPU. `submit` and `collect` are separable so the dispatch
   * can be queued behind the field's before either readback is waited on.
   */
  async run(
    samples: GPUBuffer,
    n: number,
    nei: number,
    groundScale: number,
    energyCh: number,
    senseScale: number,
    learn: { rate: number; critic: number; trace: number; discount: number; maxWeight: number },
  ): Promise<boolean> {
    if (!this.submit(samples, n, nei, groundScale, energyCh, senseScale, learn)) return false;
    return this.collect();
  }

  /** Encode and submit a frame's dispatch. False if the device is not there. */
  submit(
    samples: GPUBuffer,
    n: number,
    nei: number,
    groundScale: number,
    energyCh: number,
    senseScale: number,
    learn: { rate: number; critic: number; trace: number; discount: number; maxWeight: number },
  ): boolean {
    const device = this.device;
    if (!this.ready || !device || !this.pipeline) return false;
    if (n === 0) {
      this.pendingBytes = 0;
      return true;
    }
    try {
      const u = new ArrayBuffer(UNIFORM_BYTES);
      const u32 = new Uint32Array(u);
      const f32 = new Float32Array(u);
      u32[0] = n;
      u32[1] = CHEM_LEN;
      f32[2] = 1 / senseScale;
      f32[3] = groundScale;
      f32[4] = HEAD_SCALE.cruise;
      f32[5] = HEAD_SCALE.turn;
      f32[6] = HEAD_SCALE.align;
      f32[7] = HEAD_SCALE.sep;
      f32[8] = HEAD_SCALE.recoil;
      f32[9] = energyCh;
      f32[10] = learn.rate;
      f32[11] = learn.critic;
      f32[12] = learn.trace;
      f32[13] = learn.discount;
      f32[14] = learn.maxWeight;
      f32[15] = HEAD_SCALE.anchor;
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
            { binding: 8, resource: { buffer: this.learn! } },
          ],
        }),
      );
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / 64)));
      pass.end();
      const bytes = n * OUT_STRIDE * 4;
      enc.copyBufferToBuffer(this.out!, 0, this.read!, 0, bytes);
      /*
       * And a row apiece for whoever asked. `LEARN_STRIDE` floats is 536
       * bytes, so every row starts on a four-byte boundary, which is all a
       * buffer-to-buffer copy asks for.
       */
      const rowBytes = LEARN_STRIDE * 4;
      for (let k = 0; k < this.learnWantN; k++) {
        enc.copyBufferToBuffer(
          this.learn!,
          this.learnWant[k] * rowBytes,
          this.learnRead!,
          k * rowBytes,
          rowBytes,
        );
      }
      this.pendingLearn = this.learnWantN;
      this.learnWantN = 0;
      device.queue.submit([enc.finish()]);
      this.pendingBytes = bytes;
      return true;
    } catch (e) {
      this.lastError = String(e);
      return false;
    }
  }

  /** Wait for the last `submit`'s readbacks and unpack them. */
  /**
   * Copy every resident learning row back to the host, in one go.
   *
   * The per-frame path (`readLearn`) is a trickle by design: only a rewrite's
   * two parents need their learning on the CPU, `beginRewrite` gives forty
   * frames of notice, and `LEARN_READ_CAP` is sized for that. Which means the
   * host's copy of everybody else's learning is whatever it was when the pond
   * moved to the device — usually zero.
   *
   * That is fine for the simulation and wrong for anything that *measures*
   * it. `docs/plasticity-plan.md` phase 5 says as much: an instrument either
   * reads this back deliberately or is quietly sampling rewrite parents. This
   * is the deliberate read — the headless harvest uses it before storing a
   * net, since a stored genome without what its bodies learned is a record of
   * half the animal.
   *
   * One staging buffer per call, destroyed on the way out: this runs at a
   * harvest and not in a frame, so a resident buffer the size of the whole
   * learning table would be megabytes held for something that happens once a
   * minute.
   */
  async drainLearn(slots: number): Promise<Float32Array | null> {
    const device = this.device;
    if (!this.ready || !device || !this.learn || slots <= 0) return null;
    const rows = Math.min(slots, this.learnCap);
    const bytes = rows * LEARN_STRIDE * 4;
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(this.learn, 0, staging, 0, bytes);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ, 0, bytes);
      const out = new Float32Array(new Float32Array(staging.getMappedRange(0, bytes)));
      staging.unmap();
      return out;
    } catch (e) {
      this.lastError = String(e);
      return null;
    } finally {
      staging.destroy();
    }
  }

  async collect(): Promise<boolean> {
    const bytes = this.pendingBytes;
    const rows = this.pendingLearn;
    this.pendingBytes = 0;
    this.pendingLearn = 0;
    this.learnGot = 0;
    if (!this.ready || !this.device) return false;
    if (bytes === 0 && rows === 0) return true;
    try {
      // Both maps issued before either is awaited: they ride one submission
      // and finish together.
      const waits: Promise<void>[] = [];
      if (bytes > 0) waits.push(this.read!.mapAsync(GPUMapMode.READ, 0, bytes));
      const learnBytes = rows * LEARN_STRIDE * 4;
      if (rows > 0) waits.push(this.learnRead!.mapAsync(GPUMapMode.READ, 0, learnBytes));
      await Promise.all(waits);
      if (bytes > 0) {
        this.outData.set(new Float32Array(this.read!.getMappedRange(0, bytes)));
        this.read!.unmap();
      }
      if (rows > 0) {
        this.learnOutData.set(new Float32Array(this.learnRead!.getMappedRange(0, learnBytes)));
        this.learnRead!.unmap();
        this.learnGot = rows;
      }
      return true;
    } catch (e) {
      this.lastError = String(e);
      return false;
    }
  }
}

export const genomeGpu = new GenomeGpu();
