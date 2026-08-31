import shader from './far.wgsl?raw';
import {
  FAR,
  FAR_CONTACT_COMP,
  FAR_SLOP,
  FAR_SPAN_COMP,
  FAR_STRIDE,
  FAR_SUBSTEPS,
  stepFarKernel,
} from './far-kernel.ts';

const UNIFORM_BYTES = 256;
const PARTICLE_BYTES = FAR_STRIDE * 4;

type GpuEntry = 'integrate' | 'disc' | 'span' | 'apply' | 'finalize';

/**
 * WebGPU host for the FAR kernel. One submit per frame, one readback.
 * Falls back to the CPU twin when the adapter is missing or a pass fails.
 */
export class FarGpu {
  ready = false;
  private device: GPUDevice | null = null;
  private module: GPUShaderModule | null = null;
  private pipelines = new Map<GpuEntry, GPUComputePipeline>();
  private bindLayout: GPUBindGroupLayout | null = null;
  private cap = 0;
  private wireCap = 0;
  private uniform: GPUBuffer | null = null;
  private particles: GPUBuffer | null = null;
  private delta: GPUBuffer | null = null;
  private wires: GPUBuffer | null = null;
  private readback: GPUBuffer | null = null;
  private cpuData = new Float32Array(0);
  private cpuWires = new Float32Array(0);

  async init(): Promise<boolean> {
    const nav = navigator as Navigator & { gpu?: GPU };
    if (!nav.gpu) return false;
    try {
      const adapter = await nav.gpu.requestAdapter();
      if (!adapter) return false;
      const device = await adapter.requestDevice();
      device.lost.then(() => {
        this.ready = false;
        this.device = null;
      });
      this.device = device;
      this.module = device.createShaderModule({ code: shader });
      this.bindLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
      });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] });
      for (const name of ['integrate', 'disc', 'span', 'apply', 'finalize'] as const) {
        this.pipelines.set(
          name,
          device.createComputePipeline({
            layout,
            compute: { module: this.module, entryPoint: name },
          }),
        );
      }
      this.uniform = device.createBuffer({
        size: UNIFORM_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.ready = true;
      return true;
    } catch {
      this.ready = false;
      this.device = null;
      return false;
    }
  }

  /**
   * Run `substeps` of FAR physics. `data` is packed `FAR_STRIDE` floats per
   * body; `wires` is `[a, b, rest, 0] * nWires`. Mutates `data` in place.
   * Returns false if the CPU twin ran instead.
   */
  async step(
    data: Float32Array,
    n: number,
    wires: Float32Array,
    nWires: number,
    dt: number,
    substeps = FAR_SUBSTEPS,
  ): Promise<boolean> {
    if (n <= 0 || dt <= 0) return true;
    const device = this.device;
    if (!this.ready || !device || !this.bindLayout) {
      stepFarKernel(data, n, wires, nWires, dt, substeps);
      return false;
    }
    try {
      this.ensureCap(n, Math.max(1, nWires));
      const h = dt / substeps;
      const uniform = new ArrayBuffer(UNIFORM_BYTES);
      const u32 = new Uint32Array(uniform);
      const f32 = new Float32Array(uniform);
      u32[0] = n;
      u32[1] = nWires;
      f32[4] = h;
      f32[5] = FAR_SLOP;
      f32[6] = FAR_CONTACT_COMP;
      f32[7] = FAR_SPAN_COMP;
      device.queue.writeBuffer(this.uniform!, 0, uniform);
      device.queue.writeBuffer(this.particles!, 0, data.buffer, data.byteOffset, n * PARTICLE_BYTES);
      if (nWires > 0) {
        device.queue.writeBuffer(this.wires!, 0, wires.buffer, wires.byteOffset, nWires * 16);
      } else {
        device.queue.writeBuffer(this.wires!, 0, new Float32Array(4));
      }

      const encoder = device.createCommandEncoder();
      const bind = device.createBindGroup({
        layout: this.bindLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform! } },
          { binding: 1, resource: { buffer: this.particles! } },
          { binding: 2, resource: { buffer: this.delta! } },
          { binding: 3, resource: { buffer: this.wires! } },
        ],
      });
      const groups = Math.ceil(n / 64);
      const pass = (name: GpuEntry) => {
        const p = encoder.beginComputePass();
        p.setPipeline(this.pipelines.get(name)!);
        p.setBindGroup(0, bind);
        p.dispatchWorkgroups(groups);
        p.end();
      };
      for (let s = 0; s < substeps; s++) {
        pass('integrate');
        pass('disc');
        pass('apply');
        if (nWires > 0) {
          pass('span');
          pass('apply');
        }
        pass('finalize');
      }
      encoder.copyBufferToBuffer(this.particles!, 0, this.readback!, 0, n * PARTICLE_BYTES);
      device.queue.submit([encoder.finish()]);
      await this.readback!.mapAsync(GPUMapMode.READ, 0, n * PARTICLE_BYTES);
      data.set(new Float32Array(this.readback!.getMappedRange(0, n * PARTICLE_BYTES)));
      this.readback!.unmap();
      return true;
    } catch {
      stepFarKernel(data, n, wires, nWires, dt, substeps);
      return false;
    }
  }

  /** Scratch packed arrays the sim can reuse. */
  packTarget(n: number, nWires: number): { data: Float32Array; wires: Float32Array } {
    if (this.cpuData.length < n * FAR_STRIDE) this.cpuData = new Float32Array(n * FAR_STRIDE * 2);
    if (this.cpuWires.length < Math.max(1, nWires) * 4) {
      this.cpuWires = new Float32Array(Math.max(4, nWires * 8));
    }
    return { data: this.cpuData, wires: this.cpuWires };
  }

  get stride(): number {
    return FAR_STRIDE;
  }

  get fields(): typeof FAR {
    return FAR;
  }

  private ensureCap(n: number, nWires: number): void {
    const device = this.device!;
    if (n > this.cap) {
      this.cap = Math.max(64, n * 2);
      const partSize = this.cap * PARTICLE_BYTES;
      this.particles?.destroy();
      this.delta?.destroy();
      this.readback?.destroy();
      this.particles = device.createBuffer({
        size: partSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.delta = device.createBuffer({
        size: this.cap * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.readback = device.createBuffer({
        size: partSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    if (nWires > this.wireCap) {
      this.wireCap = Math.max(16, nWires * 2);
      this.wires?.destroy();
      this.wires = device.createBuffer({
        size: this.wireCap * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    if (!this.wires) {
      this.wireCap = 16;
      this.wires = device.createBuffer({
        size: this.wireCap * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
  }
}

export const farGpu = new FarGpu();
