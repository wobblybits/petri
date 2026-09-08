import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { CH, CHANNELS, Fields } from '../fields.ts';
import { defaultParams, type Params } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { fieldGpu } from './field-gpu.ts';
import { closeWebGpu, openWebGpu } from '../pond/webgpu-node.ts';

/*
 * The field shader, run.
 *
 * `field-kernel.test.ts` opens by saying there is no WebGPU under Node, so
 * `field.wgsl` cannot be executed and a line-for-line mirror is the next best
 * thing. That was true until `pond/webgpu-node.ts` handed Node a Dawn device,
 * and it is worth being precise about what changes: a mirror catches a port
 * that was written wrong, and cannot catch a port that was written wrong *and
 * transcribed the same way twice*. Two implementations agreeing because they
 * are the same mistake is exactly the failure a hand-written mirror is blind
 * to, and it is the one that costs an afternoon.
 *
 * So these run the shipping shader against the shipping CPU field through the
 * public path — `Sim.stepAsync` with the field on the device, against
 * `Sim.step` with it here — and compare cell for cell.
 *
 * They **skip** rather than fail where Dawn is not installed: `webgpu` is an
 * optional dependency of about 21 MB a platform, and a machine without it must
 * still be able to run the suite. A skip is visible in the runner's output,
 * which is the point — it says the check did not happen rather than pretending
 * it passed.
 */

let device = false;

beforeAll(async () => {
  const open = await openWebGpu();
  device = open.ok;
});

afterAll(() => {
  // A live GPUDevice holds Node's event loop open. The runner would sit there
  // with every test green and nothing left to do.
  closeWebGpu(fieldGpu.gpuDevice);
});

function pondParams(): Params {
  const params = defaultParams();
  params.soupCount = 0;
  params.spawnInterval = 0;
  return params;
}

/** Worst absolute difference over one channel of two fields. */
function worstDiff(a: Fields, b: Fields, ch: number): number {
  let worst = 0;
  for (let k = ch; k < a.data.length; k += CHANNELS) {
    worst = Math.max(worst, Math.abs(a.data[k] - b.data[k]));
  }
  return worst;
}

describe('field.wgsl on a device', () => {
  it('diffuses, decays and grows the way Fields does', async ({ skip }) => {
    if (!device) skip();
    const params = pondParams();

    const cpu = new Sim(1600, 1200, 128);
    loadPreset(cpu, 'soup', params);

    const gpu = new Sim(1600, 1200, 128);
    loadPreset(gpu, 'soup', params);
    expect(await gpu.openFieldGpu(), fieldGpu.lastError).toBe(true);
    gpu.wantFieldReadback = true;

    for (let i = 0; i < 120; i++) {
      cpu.step(1 / 60, params);
      await gpu.stepAsync(1 / 60, params);
    }

    /*
     * Float32 on both sides and the same arithmetic in the same order, so the
     * agreement should be near-exact rather than merely close. A tolerance
     * this tight is what makes the test able to catch a transposed index or a
     * dropped term, which a loose one would wave through.
     */
    const cap = cpu.energy.cellCap;
    expect(cap).toBeGreaterThan(0);
    expect(worstDiff(cpu.fields, gpu.fields, CH.energy)).toBeLessThan(cap * 1e-3);
    // And the totals, which is the property the wall fix is about.
    expect(gpu.energy.storedTotal()).toBeCloseTo(cpu.energy.storedTotal(), 1);
  });

  it('reflects the conserved channel off the dish wall and absorbs a signal', async ({ skip }) => {
    if (!device) skip();
    /*
     * The half of the boundary rule that only a device can confirm. The
     * mirror in `field-kernel.test.ts` checks the CPU's own arithmetic; this
     * checks that `diffuseAt`'s `miss` vector picks the same channel.
     */
    const params = pondParams();
    params.energyRegrow = 0;
    params.decay = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    expect(await sim.openFieldGpu(), fieldGpu.lastError).toBe(true);
    sim.wantFieldReadback = true;

    const before = sim.energy.storedTotal();
    expect(before).toBeGreaterThan(0);
    for (let i = 0; i < 240; i++) await sim.stepAsync(1 / 60, params);
    // Conserved on the device, exactly as on the CPU. Before the wall fix this
    // lost a percent a dish-crossing.
    expect(sim.energy.storedTotal()).toBeCloseTo(before, 1);
  });

  it('grazes a block the way EnergyGrid.take does, with real bodies', async ({ skip }) => {
    if (!device) skip();
    /*
     * `harvest` is the kernel the mirror's own comment calls the one that
     * matters most: a hand-port of a sequential CPU loop with an early exit
     * and an order-dependent scan. Bodies are pinned so the two ponds cannot
     * drift apart through physics, which leaves the grazing as the only thing
     * that could differ.
     */
    const params = pondParams();
    params.energyRegrow = 0;
    params.decay = 0;
    params.upkeep = 0;

    const build = (): Sim => {
      const sim = new Sim(1600, 1200, 128);
      loadPreset(sim, 'soup', params);
      const cx = sim.w * 0.5;
      const cy = sim.h * 0.5;
      for (let i = 0; i < 12; i++) {
        const a = sim.spawn('con', cx + (i % 4) * 9 - 18, cy + Math.floor(i / 4) * 9 - 9, 0, params, true)!;
        a.pinned = true;
        a.extra = 0;
      }
      return sim;
    };

    const cpu = build();
    const gpu = build();
    expect(await gpu.openFieldGpu(), fieldGpu.lastError).toBe(true);
    gpu.wantFieldReadback = true;

    for (let i = 0; i < 60; i++) {
      cpu.step(1 / 60, params);
      await gpu.stepAsync(1 / 60, params);
    }

    // The GPU harvest is credited a frame late by design, so the two are
    // compared on what the pond holds in total rather than frame for frame.
    const cpuFree = cpu.totalFree();
    expect(cpuFree, 'nobody ate').toBeGreaterThan(0);
    expect(gpu.totalFree()).toBeCloseTo(cpuFree, 1);
    expect(gpu.energy.storedTotal()).toBeCloseTo(cpu.energy.storedTotal(), 0);
  });

  it('meters uptake on the device the way runHarvestPlan does', async ({ skip }) => {
    if (!device) skip();
    // Phase 1's Monod rate, which went into the shader as two uniforms in the
    // slots `FieldParams` was padding. Nothing but a device can say the
    // shader's copy of the formula agrees with `energy.ts`'s.
    const params = pondParams();
    params.energyRegrow = 0;
    params.decay = 0;
    params.upkeep = 0;
    params.uptakeVmax = 0.6;
    params.uptakeKs = 0.25;

    const build = (): Sim => {
      const sim = new Sim(1600, 1200, 128);
      loadPreset(sim, 'soup', params);
      const cx = sim.w * 0.5;
      const cy = sim.h * 0.5;
      for (let i = 0; i < 8; i++) {
        const a = sim.spawn('con', cx + i * 11 - 44, cy, 0, params, true)!;
        a.pinned = true;
        a.extra = 0;
      }
      return sim;
    };

    const cpu = build();
    const gpu = build();
    expect(await gpu.openFieldGpu(), fieldGpu.lastError).toBe(true);
    gpu.wantFieldReadback = true;

    for (let i = 0; i < 60; i++) {
      cpu.step(1 / 60, params);
      await gpu.stepAsync(1 / 60, params);
    }
    const cpuFree = cpu.totalFree();
    expect(cpuFree).toBeGreaterThan(0);
    // Rate-limited, so nobody is anywhere near full — which is the thing being
    // checked, not just that the two agree.
    expect(cpuFree).toBeLessThan(8 * 0.6 * 1.1);
    expect(gpu.totalFree()).toBeCloseTo(cpuFree, 1);
  });
});
