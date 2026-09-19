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
  /*
   * Flat. Every test in this file reads a named cell — the one a body stands
   * in, the one against the wall — and asks whether host and device agree
   * about it. On the shipped patchy dish those coordinates are bare four
   * times out of five, and "they agree that nothing happened" is not parity.
   */
  params.groundPatches = 0;
  /*
   * And no exploration. Every test here compares two implementations of the
   * same field, and node perturbation displaces each body's heads by a draw
   * scaled in `f64` on the host and `f32` on the device — identical draws,
   * differing in the last bits once scaled, and then fed straight back into
   * where the body swims. That is a divergence amplifier bolted to the thing
   * under test, and it is not what the test is asking about. The precedent is
   * the gait and the pair flock, pinned in the solver parity rigs for the same
   * reason.
   */
  params.learnExplore = 0;
  return params;
}

/** Sum of one channel over a whole field. */
function totalOf(f: Fields, ch: number): number {
  let s = 0;
  for (let k = ch; k < f.data.length; k += CHANNELS) s += f.data[k];
  return s;
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

  it('lays the shipped patchy dish on the device, at the same mass', async ({ skip }) => {
    if (!device) skip();
    /*
     * The one thing about the shipped ground that only a device can confirm,
     * and the reason the rest of this file pins `groundPatches` to 0: a patchy
     * seed is the only layout that does not go through `fill`. It clears the
     * dish through `fill` at zero and then puts the mass back as three
     * thousand deferred conserved adds, which reach the device as `scatter`
     * deposits — a different kernel, on a different code path, with the rim
     * behaviour of `addAt` rather than of `deposit`.
     *
     * So this asks the two questions that path can get wrong: does the mass
     * arrive at all (the passes run `fill` before `scatter`, and the other
     * order would wipe it), and does it arrive *once* (`configure` notices the
     * count and lays the dish, then `pinWorld` asks again, and the seed drops
     * its own queue so the two agree).
     */
    const params = pondParams();
    params.groundPatches = 24;
    params.energyRegrow = 0;
    params.decay = 0;
    params.diffuse = 0;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    expect(await sim.openFieldGpu(), fieldGpu.lastError).toBe(true);
    sim.wantFieldReadback = true;
    // One body, pinned and full, so there is a frame to dispatch on at all
    // and nothing for it to graze. An empty pond never reaches the field.
    const a = sim.spawn('con', sim.w * 0.5, sim.h * 0.5, 0, params, true)!;
    a.pinned = true;

    const want = sim.energy.uniformMass;
    expect(want).toBeGreaterThan(0);
    // One frame to dispatch the queued seed; before it the device holds the
    // clear and nothing else, which is the timeline note in `pond/ground.ts`.
    for (let i = 0; i < 2; i++) await sim.stepAsync(1 / 60, params);
    const got = sim.energy.storedTotal();
    expect(got / want, 'the whole dish arrived, once').toBeGreaterThan(0.95);
    expect(got / want, 'the whole dish arrived, once').toBeLessThan(1.05);

    /*
     * And it is a landscape rather than a flat dish: most of the disk is bare.
     * Diffusion is off, so this is the seed as laid — but it is laid through
     * `scatter`, which spreads each add bilinearly over the four cells around
     * its point, so every blob arrives with a one-cell ring the host's direct
     * `addAt` does not draw. On a 71px blob in a 12.5px cell that is about a
     * third more area, which is the whole difference between the quarter the
     * host lays and the figure below.
     */
    const f = sim.fields;
    const cs = f.worldW / f.cols;
    let inDisk = 0;
    let fed = 0;
    for (let j = 0; j < f.rows; j++) {
      for (let i = 0; i < f.cols; i++) {
        if (!sim.energy.inBounds(f.originX + (i + 0.5) * cs, f.originY + (j + 0.5) * cs)) continue;
        inDisk++;
        if (f.data[(j * f.cols + i) * CHANNELS + CH.energy] > 0) fed++;
      }
    }
    expect(inDisk).toBeGreaterThan(0);
    expect(fed / inDisk, 'most of the disk is bare').toBeGreaterThan(0.2);
    expect(fed / inDisk, 'most of the disk is bare').toBeLessThan(0.45);
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

  it('lands a sampled mouthful in the gut the way the host does', async ({ skip }) => {
    if (!device) skip();
    // Catabolism is host-side on both paths now, so what the shader can still
    // get wrong is the mouthful: the plan's gut room and budget as packed, the
    // share ceiling, and `creditHarvest` landing the draw in the gut a frame
    // late. Digestion then runs on the host from the same gut either way, so
    // agreement on the totals is agreement on the crossing.
    const params = pondParams();
    params.energyRegrow = 0;
    params.ambientEnergy = 0.3;
    params.decay = 0;
    params.upkeep = 0;
    params.uptakeVmax = 6;

    const build = (): Sim => {
      const sim = new Sim(1600, 1200, 128);
      loadPreset(sim, 'soup', params);
      const cx = sim.w * 0.5;
      const cy = sim.h * 0.5;
      for (let i = 0; i < 8; i++) {
        const a = sim.spawn('con', cx + i * 12 - 48, cy, 0, params, true)!;
        a.pinned = true;
        a.extra = 0.5;
      }
      /*
       * No artificial substrate: the bodies excrete `CH.conP` themselves, and
       * that is the only seeding both paths can be given identically.
       *
       * `fillDisk` writes the host's mirror, which the shader does not read —
       * `openFieldGpu` clears the device and reseeds only `CH.energy`. And a
       * conserved add placed before that call is not deferred, so it lands in
       * the mirror too. There is exactly one crossing for a non-ground
       * species, the deferred adds, and it only exists once the device is
       * open. Letting excretion make the substrate sidesteps the whole
       * question and is what a real pond does anyway.
       */
      return sim;
    };

    const cpu = build();
    const gpu = build();
    expect(await gpu.openFieldGpu(), fieldGpu.lastError).toBe(true);
    gpu.wantFieldReadback = true;
    for (let i = 0; i < 90; i++) {
      cpu.step(1 / 60, params);
      await gpu.stepAsync(1 / 60, params);
    }
    /*
     * The tanks are the tight assertion, and they are what the gate decides:
     * if the shader multiplied the co-substrate factor in the wrong place, or
     * applied it to the ground's own row, these would part company.
     */
    const held = (s: Sim) => [...s.agents.values()].reduce((n, a) => n + a.extra, 0);
    expect(held(cpu)).toBeGreaterThan(0);
    expect(held(gpu)).toBeCloseTo(held(cpu), 1);
    /*
     * The standing `conP` loosely, because here it is the small difference
     * between what eight bodies excreted and what they ate back again — so one
     * frame of the GPU harvest's designed lag is a large share of it. The
     * quantity is a residue, not a stock.
     */
    const c = totalOf(cpu.fields, CH.conP);
    expect(c, 'nothing was excreted').toBeGreaterThan(0);
    expect(Math.abs(totalOf(gpu.fields, CH.conP) - c) / c).toBeLessThan(0.06);
  });

  it('runs the whole reaction table the way the host does', async ({ skip }) => {
    if (!device) skip();
    /*
     * The food loop, on the device: ground out of the field, into a gut,
     * digested, and the tanks on both sides agreeing about how much.
     *
     * It used to be four species out and four back in — `harvest` carried a
     * per-species drain and a row of four affinities, and excretion put four
     * back. A body eats the ground and nothing else now and excretes nothing,
     * so the whole table is one uptake row, the harvest stride is three floats
     * instead of six, and the signalling channels are minted rather than
     * traded. Nothing but a device says the two paths agree.
     */
    const params = pondParams();
    params.energyRegrow = 0;
    params.ambientEnergy = 0.4;
    params.decay = 0;
    params.upkeep = 0;
    params.uptakeVmax = 1.5;

    const build = (): Sim => {
      const sim = new Sim(1600, 1200, 128);
      loadPreset(sim, 'soup', params);
      const cx = sim.w * 0.5;
      const cy = sim.h * 0.5;
      for (let i = 0; i < 12; i++) {
        const a = sim.spawn('con', cx + (i % 4) * 12 - 24, cy + Math.floor(i / 4) * 12 - 12, 0, params, true)!;
        a.pinned = true;
        a.extra = 1;
      }
      return sim;
    };

    const cpu = build();
    const gpu = build();
    expect(await gpu.openFieldGpu(), fieldGpu.lastError).toBe(true);
    gpu.wantFieldReadback = true;

    for (let i = 0; i < 90; i++) {
      cpu.step(1 / 60, params);
      await gpu.stepAsync(1 / 60, params);
    }

    const held = (s: Sim) => [...s.agents.values()].reduce((n, a) => n + a.extra, 0);
    // Something was actually eaten, or this asserts about a still pond.
    expect(held(cpu), 'nothing was banked').toBeGreaterThan(12);
    // The tanks agree. The harvest's one-frame lag is why this is not exact.
    expect(held(gpu)).toBeCloseTo(held(cpu), 1);
    /*
     * And the ground they ate it out of, relative rather than absolute: the
     * GPU harvest is credited a frame late by design, so the two paths stand
     * one frame's uptake apart. A tolerance tighter than this asserts they
     * agree about *when*, which they deliberately do not; one looser would
     * wave through a real divergence.
     */
    const ground = totalOf(cpu.fields, CH.energy);
    expect(ground, 'the dish was stripped, so there is nothing to compare').toBeGreaterThan(0);
    expect(Math.abs(totalOf(gpu.fields, CH.energy) - ground) / ground, 'the ground differs').toBeLessThan(0.02);
    /*
     * The signalling channels are minted rather than traded, so they should
     * stand at the same level either way — within the same one-frame lag,
     * which is the deposit pass being a frame behind on the device.
     *
     * All three, and the rig is all Cons. That used to mean `conP` and `aux`,
     * and it means something else now. `conP` is still what a Con's principal
     * says. `aux` is the *ground's* voice — `Fields.grow` mints it in
     * proportion to what is standing in a cell, so there is no body in this
     * rig entitled to write it and it is there anyway. And `dupP` is on with
     * no Dup in the dish at all, because a free auxiliary port leaks both body
     * voices equally: that is what keeps "there is somewhere to attach here"
     * kind-independent now that aux belongs to the ground.
     */
    for (const ch of [CH.conP, CH.dupP, CH.aux]) {
      const c = totalOf(cpu.fields, ch);
      expect(c, `nothing on channel ${ch}`).toBeGreaterThan(0);
      expect(Math.abs(totalOf(gpu.fields, ch) - c) / c, `channel ${ch} differs`).toBeLessThan(0.02);
    }
  });

  it('marks a free principal on its own channel and a socket on both', () => {
    /*
     * Which marker is which, in a dish of nothing but Cons.
     *
     * A Con's free *principal* says so on `conP` — the one channel a Con is
     * deaf to, so it cannot follow its own trail or huddle with its own kind —
     * and its free *auxiliaries* say "somewhere to attach" on both. So `dupP`
     * in a pond with no Dup in it is the socket marker and nothing else, and
     * turning `auxLeak` off alone silences that channel while `conP` keeps
     * both the voice and the principal's mark.
     */
    const params = pondParams();
    params.energyRegrow = 0;
    params.decay = 0;
    params.upkeep = 0;
    const dish = (leak: number, aux: number): Sim => {
      params.portLeak = leak;
      params.auxLeak = aux;
      const sim = new Sim(1600, 1200, 128);
      loadPreset(sim, 'soup', params);
      for (let i = 0; i < 12; i++) {
        const a = sim.spawn('con', sim.w * 0.5 + i * 12, sim.h * 0.5, 0, params, true)!;
        a.pinned = true;
      }
      for (let i = 0; i < 90; i++) sim.step(1 / 60, params);
      return sim;
    };
    const both = dish(0.7, 0.2);
    expect(totalOf(both.fields, CH.conP), 'the Cons speak').toBeGreaterThan(0);
    expect(totalOf(both.fields, CH.dupP), 'and their sockets do').toBeGreaterThan(0);

    // Sockets silent, principals still marking: `dupP` has no other writer.
    const noAux = dish(0.7, 0);
    expect(totalOf(noAux.fields, CH.conP)).toBeGreaterThan(0);
    expect(totalOf(noAux.fields, CH.dupP), 'a Con marked a Dup channel').toBe(0);

    // And the principal's mark is on `conP`, so dropping it leaves the voice.
    const noMark = dish(0, 0);
    expect(totalOf(noMark.fields, CH.conP), 'the Cons still speak').toBeGreaterThan(0);
    expect(totalOf(noMark.fields, CH.conP)).toBeLessThan(totalOf(noAux.fields, CH.conP));
    expect(totalOf(noMark.fields, CH.dupP), 'and nothing else does').toBe(0);
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
