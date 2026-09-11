import shader from './far.wgsl?raw';
import {
  FAR,
  FAR_CONTACT_COMP,
  FAR_SLOP,
  FAR_SPAN_COMP,
  FAR_STRIDE,
  FAR_SUBSTEPS,
  FAR_WIRE_STRIDE,
  stepFarKernel,
} from './far-kernel.ts';

const UNIFORM_BYTES = 256;
const PARTICLE_BYTES = FAR_STRIDE * 4;
/** Must match `CELL_CAP` and `NEI_CAP` in far.wgsl. */
const CELL_CAP = 64;
const NEI_CAP = 4;
/** Grid ceiling, so the broadphase is bounded however far the pond drifts; `collect_pairs` in native/solver.c does the same. */
const MAX_CELLS = 131072;

/** Cells to aim for at a given body count: four a body keeps occupancy under CELL_CAP. */
function cellTarget(n: number): number {
  return Math.max(256, Math.min(MAX_CELLS, n * 4));
}

type GpuEntry =
  | 'integrate'
  | 'disc'
  | 'span'
  | 'apply'
  | 'finalize'
  | 'wall'
  | 'clearGrid'
  | 'binBodies'
  | 'clearNei'
  | 'buildNei';

/** WebGPU host for the FAR kernel. One submit per frame, one readback. Falls back to the CPU twin. */
export class FarGpu {
  ready = false;
  private device: GPUDevice | null = null;
  private module: GPUShaderModule | null = null;
  private pipelines = new Map<GpuEntry, GPUComputePipeline>();
  private bindLayout: GPUBindGroupLayout | null = null;
  private cap = 0;
  private wireCap = 0;
  private cellCap = 0;
  private uniform: GPUBuffer | null = null;
  private particles: GPUBuffer | null = null;
  private delta: GPUBuffer | null = null;
  private wires: GPUBuffer | null = null;
  private readback: GPUBuffer | null = null;
  private cellCount: GPUBuffer | null = null;
  private cellBodies: GPUBuffer | null = null;
  private nei: GPUBuffer | null = null;
  private neiCount: GPUBuffer | null = null;
  private cpuData = new Float32Array(0);
  private cpuWires = new Float32Array(0);
  /** Why `init` returned false, when the reason was the shader itself. */
  initError = '';

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
      // A WGSL error does not throw; the invalid pipeline quietly drops every dispatch.
      const info = await this.module.getCompilationInfo?.();
      const errors = info ? info.messages.filter((m) => m.type === 'error') : [];
      if (errors.length > 0) {
        this.initError = errors
          .map((m) => `far.wgsl:${m.lineNum}:${m.linePos}: ${m.message}`)
          .join('\n');
        this.ready = false;
        this.device = null;
        return false;
      }
      this.bindLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] });
      for (const name of [
        'integrate',
        'disc',
        'span',
        'apply',
        'finalize',
        'wall',
        'clearGrid',
        'binBodies',
        'clearNei',
        'buildNei',
      ] as const) {
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
   * Run `substeps` of FAR physics in place on `data` (`FAR_STRIDE` floats per
   * body); `wires` is `[a, b, rest, pad, oax, oay, obx, oby] * nWires`. False if the CPU twin ran.
   */
  async step(
    data: Float32Array,
    n: number,
    wires: Float32Array,
    nWires: number,
    dt: number,
    substeps = FAR_SUBSTEPS,
    cx = 0,
    cy = 0,
    boundR = 0,
  ): Promise<boolean> {
    if (n <= 0 || dt <= 0) return true;
    const device = this.device;
    if (!this.ready || !device || !this.bindLayout) {
      stepFarKernel(data, n, wires, nWires, dt, substeps, cx, cy, boundR);
      return false;
    }
    try {
      this.ensureCap(n, Math.max(1, nWires));
      const h = dt / substeps;
      const uniform = new ArrayBuffer(UNIFORM_BYTES);
      const u32 = new Uint32Array(uniform);
      const f32 = new Float32Array(uniform);
      const grid = this.computeGrid(data, n);
      u32[0] = n;
      u32[1] = nWires;
      u32[2] = grid.cols;
      u32[3] = grid.rows;
      f32[4] = h;
      f32[5] = FAR_SLOP;
      f32[6] = FAR_CONTACT_COMP;
      f32[7] = FAR_SPAN_COMP;
      f32[8] = grid.minX;
      f32[9] = grid.minY;
      f32[10] = grid.invCell;
      f32[12] = cx;
      f32[13] = cy;
      f32[14] = boundR;
      device.queue.writeBuffer(this.uniform!, 0, uniform);
      device.queue.writeBuffer(this.particles!, 0, data.buffer, data.byteOffset, n * PARTICLE_BYTES);
      if (nWires > 0) {
        device.queue.writeBuffer(this.wires!, 0, wires.buffer, wires.byteOffset, nWires * FAR_WIRE_STRIDE * 4);
      } else {
        device.queue.writeBuffer(this.wires!, 0, new Float32Array(FAR_WIRE_STRIDE));
      }

      const encoder = device.createCommandEncoder();
      const bind = device.createBindGroup({
        layout: this.bindLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniform! } },
          { binding: 1, resource: { buffer: this.particles! } },
          { binding: 2, resource: { buffer: this.delta! } },
          { binding: 3, resource: { buffer: this.wires! } },
          { binding: 4, resource: { buffer: this.cellCount! } },
          { binding: 5, resource: { buffer: this.cellBodies! } },
          { binding: 6, resource: { buffer: this.nei! } },
          { binding: 7, resource: { buffer: this.neiCount! } },
        ],
      });
      const groups = Math.ceil(n / 64);
      const cellGroups = Math.ceil((grid.cols * grid.rows) / 64);
      const wireGroups = Math.ceil(Math.max(1, nWires) / 64);
      const pass = (name: GpuEntry, count = groups) => {
        const p = encoder.beginComputePass();
        p.setPipeline(this.pipelines.get(name)!);
        p.setBindGroup(0, bind);
        p.dispatchWorkgroups(count);
        p.end();
      };
      // Wires do not move between substeps, so this table is built once.
      pass('clearNei');
      if (nWires > 0) pass('buildNei', wireGroups);
      for (let s = 0; s < substeps; s++) {
        pass('integrate');
        // After integrate: the grid has to describe where bodies are now.
        pass('clearGrid', cellGroups);
        pass('binBodies');
        pass('disc');
        pass('apply');
        if (nWires > 0) {
          pass('span');
          pass('apply');
        }
        pass('finalize');
        pass('wall');
      }
      encoder.copyBufferToBuffer(this.particles!, 0, this.readback!, 0, n * PARTICLE_BYTES);
      device.queue.submit([encoder.finish()]);
      await this.readback!.mapAsync(GPUMapMode.READ, 0, n * PARTICLE_BYTES);
      data.set(new Float32Array(this.readback!.getMappedRange(0, n * PARTICLE_BYTES)));
      this.readback!.unmap();
      return true;
    } catch {
      stepFarKernel(data, n, wires, nWires, dt, substeps, cx, cy, boundR);
      return false;
    }
  }

  /**
   * Grid to bin the pack into. Cell size starts at the widest contact gap,
   * then doubles until the grid fits MAX_CELLS; `cellXY` clamps drift.
   */
  private computeGrid(
    data: Float32Array,
    n: number,
  ): { cols: number; rows: number; minX: number; minY: number; invCell: number } {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxR = 0;
    for (let i = 0; i < n; i++) {
      const o = i * FAR_STRIDE;
      const x = data[o + FAR.x];
      const y = data[o + FAR.y];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      const r = data[o + FAR.radius];
      if (Number.isFinite(r) && r > maxR) maxR = r;
    }
    // Every body non-finite, or one body: a 1x1 grid is correct and trivial.
    if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
      return { cols: 1, rows: 1, minX: 0, minY: 0, invCell: 1 };
    }
    let cell = Math.max(2 * maxR, 1e-3);
    minX -= cell;
    minY -= cell;
    const w = maxX - minX + cell;
    const h = maxY - minY + cell;
    let cols = Math.floor(w / cell) + 1;
    let rows = Math.floor(h / cell) + 1;
    const budget = cellTarget(n);
    let guard = 0;
    while (cols * rows > budget && guard++ < 64) {
      cell *= 2;
      cols = Math.floor(w / cell) + 1;
      rows = Math.floor(h / cell) + 1;
    }
    cols = Math.max(1, Math.min(cols, budget));
    rows = Math.max(1, Math.min(rows, Math.floor(budget / cols)));
    return { cols, rows, minX, minY, invCell: 1 / cell };
  }

  /** Scratch packed arrays the sim can reuse. */
  packTarget(n: number, nWires: number): { data: Float32Array; wires: Float32Array } {
    if (this.cpuData.length < n * FAR_STRIDE) this.cpuData = new Float32Array(n * FAR_STRIDE * 2);
    if (this.cpuWires.length < Math.max(1, nWires) * FAR_WIRE_STRIDE) {
      this.cpuWires = new Float32Array(Math.max(FAR_WIRE_STRIDE, nWires * FAR_WIRE_STRIDE * 2));
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
      this.nei?.destroy();
      this.neiCount?.destroy();
      this.nei = device.createBuffer({
        size: this.cap * NEI_CAP * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.neiCount = device.createBuffer({
        size: this.cap * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    const cells = cellTarget(this.cap);
    if (cells > this.cellCap) {
      this.cellCap = cells;
      this.cellCount?.destroy();
      this.cellBodies?.destroy();
      this.cellCount = device.createBuffer({
        size: cells * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.cellBodies = device.createBuffer({
        size: cells * CELL_CAP * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    if (nWires > this.wireCap) {
      this.wireCap = Math.max(16, nWires * 2);
      this.wires?.destroy();
      this.wires = device.createBuffer({
        size: this.wireCap * FAR_WIRE_STRIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    if (!this.wires) {
      this.wireCap = 16;
      this.wires = device.createBuffer({
        size: this.wireCap * FAR_WIRE_STRIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
  }
}

export const farGpu = new FarGpu();
