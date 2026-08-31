/** Minimal WebGPU types so the FAR solver typechecks without @webgpu/types. */

interface GPU {
  requestAdapter(): Promise<GPUAdapter | null>;
}

interface GPUAdapter {
  requestDevice(): Promise<GPUDevice>;
}

interface GPUDevice {
  readonly queue: GPUQueue;
  readonly lost: Promise<unknown>;
  createShaderModule(desc: { code: string }): GPUShaderModule;
  createBindGroupLayout(desc: unknown): GPUBindGroupLayout;
  createPipelineLayout(desc: { bindGroupLayouts: GPUBindGroupLayout[] }): GPUPipelineLayout;
  createComputePipeline(desc: unknown): GPUComputePipeline;
  createBuffer(desc: { size: number; usage: number }): GPUBuffer;
  createBindGroup(desc: unknown): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
}

interface GPUQueue {
  writeBuffer(buffer: GPUBuffer, offset: number, data: ArrayBuffer | ArrayBufferView, srcOffset?: number, size?: number): void;
  submit(buffers: GPUCommandBuffer[]): void;
}

interface GPUShaderModule {}
interface GPUBindGroupLayout {}
interface GPUPipelineLayout {}
interface GPUComputePipeline {}
interface GPUBindGroup {}
interface GPUCommandBuffer {}

interface GPUBuffer {
  destroy(): void;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
}

interface GPUCommandEncoder {
  beginComputePass(): GPUComputePassEncoder;
  copyBufferToBuffer(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number): void;
  finish(): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, group: GPUBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface GPUShaderStageCtor {
  COMPUTE: number;
}

interface GPUBufferUsageCtor {
  UNIFORM: number;
  STORAGE: number;
  COPY_DST: number;
  COPY_SRC: number;
  MAP_READ: number;
}

interface GPUMapModeCtor {
  READ: number;
}

declare const GPUShaderStage: GPUShaderStageCtor;
declare const GPUBufferUsage: GPUBufferUsageCtor;
declare const GPUMapMode: GPUMapModeCtor;

interface Navigator {
  gpu?: GPU;
}
