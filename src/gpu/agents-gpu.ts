import shader from './agents.wgsl?raw';

const UNIFORM_BYTES = 16; // 4 f32: originX, originY, scaleX, scaleY
const INSTANCE_STRIDE = 9; // floats per instance — see agents.wgsl

/**
 * Instanced-dot layer for the FAR-tier population Canvas2D skips (see
 * render.ts): one draw call regardless of count. A separate canvas layered
 * on top, clearing to transparent, so a FAR-tier dot is never buried under a
 * wire; a canvas is either a 2D context or a WebGPU context, never both.
 */
export class AgentsGpu {
  ready = false;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private bindLayout: GPUBindGroupLayout | null = null;
  private uniform: GPUBuffer | null = null;
  private instances: GPUBuffer | null = null;
  private cap = 0;

  async init(canvas: HTMLCanvasElement): Promise<boolean> {
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
      const context = canvas.getContext('webgpu');
      if (!context) return false;
      const format = nav.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'premultiplied' });

      const module = device.createShaderModule({ code: shader });
      const bindLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        ],
      });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
      const pipeline = device.createRenderPipeline({
        layout,
        vertex: { module, entryPoint: 'vs_main' },
        fragment: {
          module,
          entryPoint: 'fs_main',
          targets: [
            {
              format,
              // Premultiplied source, matching the shader's output and the
              // canvas's alphaMode: srcFactor 'one', or alpha applies twice.
              blend: {
                color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              },
            },
          ],
        },
        primitive: { topology: 'triangle-list' },
      });

      this.device = device;
      this.context = context;
      this.pipeline = pipeline;
      this.bindLayout = bindLayout;
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

  private ensureCap(n: number): void {
    if (n <= this.cap) return;
    this.cap = Math.max(64, n * 2);
    this.instances?.destroy();
    this.instances = this.device!.createBuffer({
      size: this.cap * INSTANCE_STRIDE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Uploads `count` instances (INSTANCE_STRIDE floats each, packed as
   * agents.wgsl expects) and draws them. `camera` matches Camera's own fields.
   */
  render(
    instances: Float32Array,
    count: number,
    camera: { x: number; y: number; zoom: number; viewW: number; viewH: number },
  ): void {
    const device = this.device;
    if (!this.ready || !device || !this.context || !this.pipeline || !this.bindLayout) return;
    if (count <= 0) {
      // Still clear the canvas: an empty pond should not show the last frame.
      const encoder = device.createCommandEncoder();
      encoder
        .beginRenderPass({
          colorAttachments: [
            { view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } },
          ],
        })
        .end();
      device.queue.submit([encoder.finish()]);
      return;
    }
    this.ensureCap(count);
    device.queue.writeBuffer(this.instances!, 0, instances.buffer, instances.byteOffset, count * INSTANCE_STRIDE * 4);
    const uniform = new Float32Array(UNIFORM_BYTES / 4);
    uniform[0] = camera.x;
    uniform[1] = camera.y;
    uniform[2] = (2 * camera.zoom) / camera.viewW;
    uniform[3] = (2 * camera.zoom) / camera.viewH;
    device.queue.writeBuffer(this.uniform!, 0, uniform);

    const bind = device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniform! } },
        { binding: 1, resource: { buffer: this.instances! } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.draw(6, count);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
}

export const agentsGpu = new AgentsGpu();
