import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { PLASTIC_BASE, PLASTIC_LEN, HEAD_TABLE, W_IN, exploreAt } from '../chem-layout.ts';
import { defaultParams, type Params } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { fieldGpu } from './field-gpu.ts';
import { closeWebGpu, openWebGpu } from '../pond/webgpu-node.ts';

/*
 * The genome shader's *learning*, run.
 *
 * `genome-kernel.test.ts` next door compares the shader against a hand-written
 * JS mirror, and it does so at `fixedParams()` — which pins `learnRate` to
 * zero. So the whole learning block at the end of `genome.wgsl` has never been
 * executed by the suite against anything, and the head learning added on top
 * of it is a hand port of a hand port. That is the arrangement this project
 * has paid for before: two implementations agreeing because they are the same
 * mistake, or one of them never running at all.
 *
 * This runs the shipping shader against the shipping CPU pass through the
 * public path and compares what each body *learned*. Bodies are pinned and the
 * dish is uniform, so nothing about physics or foraging can drift the two
 * ponds apart — what is left is the arithmetic.
 *
 * Skips where Dawn is not installed, like the field tests, and for the same
 * reason: a skip says the check did not happen.
 */

let device = false;

beforeAll(async () => {
  const open = await openWebGpu();
  device = open.ok;
});

afterAll(() => {
  closeWebGpu(fieldGpu.gpuDevice);
});

function learnParams(): Params {
  const params = defaultParams();
  params.soupCount = 0;
  params.spawnInterval = 0;
  // A dish that cannot change under either pond, so `x4` and the sense inputs
  // are the same numbers on both sides of the comparison.
  params.groundPatches = 0;
  params.energyRegrow = 0;
  params.energyDiffuse = 0;
  params.decay = 0;
  params.diffuse = 0;
  params.deposit = 0;
  /*
   * Rich enough that thirty frames of grazing cannot dent it. The bodies below
   * are pinned in place but they still eat, and once the shipped
   * `ambientEnergy` halved they were pulling their own cells down inside the
   * run — so the two ponds' harvests, which are different kernels, started
   * from the same dish and ended on different ones, and what this compares is
   * the learning rule rather than the harvest.
   */
  params.ambientEnergy = 20;
  /*
   * And no rent, for the same reason: it draws the pinned tanks down over the
   * run, and `x4` is an input to the very rule under test — so the two copies
   * would be learning from a number that is itself drifting apart.
   */
  params.upkeep = 0;
  // The reactor swings every wire's rest length, which moves bodies and is not
  // what this is asking about.
  params.metabolicRate = 0;
  params.learnRate = 0.02;
  params.learnExplore = 0.05;
  return params;
}

function build(params: Params): Sim {
  const sim = new Sim(1600, 1200, 128);
  loadPreset(sim, 'soup', params);
  const cx = sim.w * 0.5;
  const cy = sim.h * 0.5;
  // A wired line, because `Wn` is the only term that reads outside a body and
  // a pass that dropped it would agree everywhere else.
  for (let i = 0; i < 10; i++) {
    const a = sim.spawn(i % 3 === 0 ? 'era' : i % 3 === 1 ? 'con' : 'dup', cx + i * 26 - 130, cy, 0, params, true)!;
    a.pinned = true;
    // Part full, so the teacher's level term has a gradient and is not the
    // same saturated zero for every body.
    a.extra = a.energyCap * (0.2 + 0.06 * i);
  }
  return sim;
}

describe('genome.wgsl on a device', () => {
  it('draws the same exploration the host does', () => {
    /*
     * The one number the two paths cannot derive from each other. The shader
     * has no RNG state and neither does the host: both hash (body, frame,
     * output) and must land on the same float, or the learner credits a
     * displacement the body never made. Transcribed by hand into WGSL, so it
     * is worth pinning a few values here in the open — a change to either side
     * that does not change the other shows up as these numbers.
     */
    expect(exploreAt(0, 0, 0)).toBeCloseTo(-0.183302, 5);
    expect(exploreAt(1, 0, 0)).toBeCloseTo(-0.654159, 5);
    expect(exploreAt(0, 1, 0)).toBeCloseTo(-0.390226, 5);
    expect(exploreAt(0, 0, 1)).toBeCloseTo(0.542518, 5);
    // Spread over [-1, 1) rather than clustered, which the hash is for.
    let lo = 1;
    let hi = -1;
    let sum = 0;
    const n = 20000;
    for (let k = 0; k < n; k++) {
      const v = exploreAt(k % 97, (k / 97) | 0, k % 15);
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
      sum += v;
    }
    expect(lo).toBeLessThan(-0.999);
    expect(hi).toBeGreaterThan(0.999);
    expect(Math.abs(sum / n), 'unbiased, or it is a drift rather than a search').toBeLessThan(0.02);
  });

  it('learns the heads the way the host does', async ({ skip }) => {
    if (!device) skip();
    const FRAMES = Number(process.env.PARITY_FRAMES ?? 30);
    const params = learnParams();
    const host = build(params);
    const dev = build(params);
    expect(await dev.openFieldGpu(), fieldGpu.lastError).toBe(true);

    for (let i = 0; i < FRAMES; i++) {
      host.step(1 / 60, params);
      await dev.stepAsync(1 / 60, params);
    }

    /*
     * The learning row lives on the device and is never written back by the
     * frame — `updateState` returns early with the genome on the GPU and only
     * unpacks the heads. Draining it is what makes it readable at all, and
     * forgetting to is how this comparison silently reads a row of zeros and
     * calls it a disagreement.
     */
    expect(await dev.syncLearningToHost()).toBe(true);

    const a = host.agentStore;
    const b = dev.agentStore;
    const bodies = [...host.agents.values()];
    let worst = 0;
    let moved = 0;
    for (const body of bodies) {
      const o = body.slot * PLASTIC_LEN;
      for (let k = 0; k < PLASTIC_LEN; k++) {
        worst = Math.max(worst, Math.abs(a.plasticAll[o + k] - b.plasticAll[o + k]));
        if (Math.abs(a.plasticAll[o + k]) > 1e-6) moved++;
      }
    }
    // The heads have to have actually moved, or this compares two zeros.
    const headFirst = HEAD_TABLE[0].at;
    let headMoved = 0;
    for (const body of bodies) {
      const o = body.slot * PLASTIC_LEN;
      for (let k = headFirst; k < PLASTIC_LEN; k++) if (Math.abs(a.plasticAll[o + k]) > 1e-9) headMoved++;
    }
    expect(moved, 'nothing learned, so nothing was compared').toBeGreaterThan(0);
    expect(headMoved, 'the heads learned nothing, so the new block was not exercised').toBeGreaterThan(0);
    // `f32` on the device against `f64` here, over thirty frames of a rule
    // that integrates. The bound is what the same comparison uses next door.
    expect(worst, 'host and device disagree about what a body learned').toBeLessThan(1e-4);
  });

  it('puts what it learned where the genome keeps that head', () => {
    /*
     * The bookkeeping the shader and the host both have to get right, checked
     * against the layout rather than against each other: the learned block's
     * head section starts where `E` does and ends where `g0` does, and every
     * head's offset inside the block is its genome offset less `W_IN`.
     */
    let rows = 0;
    for (const h of HEAD_TABLE) {
      rows += h.rows;
      expect(h.at + h.rows * 4).toBeLessThanOrEqual(PLASTIC_LEN);
      if (h.base >= 0) expect(h.base + h.rows).toBeLessThanOrEqual(PLASTIC_LEN);
    }
    expect(rows).toBe(15);
    expect(PLASTIC_BASE).toBe(W_IN);
  });
});
