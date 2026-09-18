import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { GAIT_ANCHOR_MAX, HEAD_RANGE, PLASTIC_BASE, PLASTIC_LEN, HEAD_TABLE, W_IN, exploreAt } from '../chem-layout.ts';
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
  // With the others: the ground mints aux every frame in proportion to what is
  // standing in a cell, and this rig wants a field that does not move.
  params.groundSmell = 0;
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

/**
 * The same dish, with every head the sliders can reach seeded past its clamp.
 *
 * `learnRate` is zero here on purpose: the test next door covers the learning
 * rule, and what this one is about is the forward pass and the six pairs of
 * bounds at the end of it. With the genome static and the bodies pinned on a
 * still dish, a head is the same number every frame on both paths, and any
 * difference between them is arithmetic.
 */
function saturatedParams(): Params {
  const params = learnParams();
  params.learnRate = 0;
  params.learnExplore = 0;
  // Each of these seeds a head base at `slider / HEAD_SCALE`, so a slider well
  // past `HEAD_RANGE` puts the head on its bound. Both ends are used: a bound
  // that was only ever approached from one side is half-checked.
  params.stepSpeed = 400;        // cruise -> 180
  params.turnRate = 40;          // turn   -> 8
  params.flockAlign = -50;       // align  -> -8, the one lower bound reachable
  params.flockSep = 400;         // sep    -> 120
  params.transportThrust = 5;    // thrust -> 1
  params.transportRecoil = 900;  // recoil -> 200
  return params;
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

  it('clamps the heads the way the host does, on the bounds themselves', async ({ skip }) => {
    if (!device) skip();
    /*
     * The gap `HEAD_RANGE` was pulled out of, closed from the other side.
     *
     * The bounds used to be bare literals in `sim.ts` and again in
     * `genome.wgsl`, and halving one in the shader alone passed the whole
     * suite. `genome-kernel.test.ts` compares the shader against a mirror
     * written in the test rather than against `sim.ts`, and never drove a head
     * hard enough to saturate; the test next door runs both shipping paths but
     * compares what they *learned*, not what they put out. So no check in the
     * program ever looked at a clamped head on both paths at once.
     *
     * This does. Every head the sliders reach is seeded past its bound, and
     * the anchor — which is seeded by kind and cannot be pushed from a
     * slider — rides along unsaturated, so this is not two clamp constants
     * being compared to each other.
     */
    const params = saturatedParams();
    const host = build(params);
    const dev = build(params);
    expect(await dev.openFieldGpu(), fieldGpu.lastError).toBe(true);

    // Few frames rather than many: the genome pass is deterministic here, so
    // one would do, and a short run keeps any physics difference from being
    // what this reads.
    for (let i = 0; i < 5; i++) {
      host.step(1 / 60, params);
      await dev.stepAsync(1 / 60, params);
    }

    const a = host.agentStore;
    const b = dev.agentStore;
    const HEADS = [
      { name: 'cruise', of: 'cruise', bound: HEAD_RANGE.cruise.max },
      { name: 'turn', of: 'turn', bound: HEAD_RANGE.turn.max },
      { name: 'align', of: 'flockAlign', bound: HEAD_RANGE.align.min },
      { name: 'sep', of: 'flockSep', bound: HEAD_RANGE.sep.max },
      { name: 'thrust', of: 'transportThrust', bound: HEAD_RANGE.thrust.max },
      { name: 'recoil', of: 'transportRecoil', bound: HEAD_RANGE.recoil.max },
      { name: 'anchor', of: 'gaitAnchor', bound: null },
    ] as const;

    const bodies = [...host.agents.values()];
    expect(bodies.length, 'nothing to compare').toBeGreaterThan(0);

    for (const h of HEADS) {
      const hv = a[h.of] as Float64Array;
      const dv = b[h.of] as Float64Array;
      for (const body of bodies) {
        const s = body.slot;
        // `f32` on the device against `f64` here. A saturated head is exact on
        // both — every bound in `HEAD_RANGE` is representable — so the
        // tolerance is only doing work for the anchor.
        const tol = Math.max(1e-5, Math.abs(hv[s]) * 1e-6);
        expect(dv[s], `${h.name} on body ${body.id}: host ${hv[s]} device ${dv[s]}`).toBeCloseTo(
          hv[s],
          Math.max(0, Math.ceil(-Math.log10(tol))),
        );
      }
      if (h.bound !== null) {
        // And the bound was actually reached, or this compared two unclamped
        // numbers and the clamps were never executed at all.
        const onBound = bodies.filter((body) => hv[body.slot] === h.bound).length;
        expect(onBound, `${h.name} never reached ${h.bound}, so its clamp did not run`).toBeGreaterThan(0);
      }
    }

    // The anchor is the control: if it too were pinned to a bound, every
    // comparison above would be a clamp against a clamp.
    const anchors = bodies.map((body) => a.gaitAnchor[body.slot]);
    expect(
      anchors.some((v) => v !== 0 && Math.abs(v) < GAIT_ANCHOR_MAX),
      'the anchor saturated too, so nothing unclamped was compared',
    ).toBe(true);
  });

  it('does not hand a new pond the last pond\'s learning', async ({ skip }) => {
    if (!device) skip();
    /*
     * The learning row is the one thing on the device that is meant to outlive
     * a frame, which is how it came to outlive a pond.
     *
     * It is indexed by slot and resident, and `syncLearn` pushes a row up only
     * for a slot the host has marked dirty — which a fresh `Sim` never does,
     * because its own `plasticAll` is already zero and it has nothing it
     * thinks needs saying. So the device kept the previous pond's rows and
     * handed them to whoever took those slots next.
     *
     * `genome-gpu.ts` carried the reason this was safe: `FieldGpu.init` had no
     * ready guard, so a second `Sim` got a new device and the whole class
     * rebuilt. That guard exists now, two ponds of the same grid share a
     * device, and the invariant quietly stopped holding. Found by the clamp
     * test above, which passed alone and failed after this file's learning
     * test — the anchor, the only head it leaves unclamped, came back 6% high.
     *
     * The CPU path was never affected: `plasticAll` is a fresh array per Sim.
     * So this is also a parity test, and the host side is what it asserts
     * against.
     */
    const on = learnParams();
    const first = build(on);
    expect(await first.openFieldGpu(), fieldGpu.lastError).toBe(true);
    for (let i = 0; i < 30; i++) await first.stepAsync(1 / 60, on);
    expect(await first.syncLearningToHost()).toBe(true);

    let taught = 0;
    for (const body of first.agents.values()) {
      const o = body.slot * PLASTIC_LEN;
      for (let k = 0; k < PLASTIC_LEN; k++) {
        if (Math.abs(first.agentStore.plasticAll[o + k]) > 1e-6) taught++;
      }
    }
    // Or the second pond has nothing to inherit and this proves nothing.
    expect(taught, 'the first pond learned nothing, so there was nothing to carry').toBeGreaterThan(0);

    const off = learnParams();
    off.learnRate = 0;
    off.learnExplore = 0;
    const second = build(off);
    expect(await second.openFieldGpu(), fieldGpu.lastError).toBe(true);
    for (let i = 0; i < 5; i++) await second.stepAsync(1 / 60, off);
    expect(await second.syncLearningToHost()).toBe(true);

    let worst = 0;
    for (const body of second.agents.values()) {
      const o = body.slot * PLASTIC_LEN;
      for (let k = 0; k < PLASTIC_LEN; k++) {
        worst = Math.max(worst, Math.abs(second.agentStore.plasticAll[o + k]));
      }
    }
    // Exactly zero, not nearly: with the rate at zero there is no arithmetic
    // that could round to something. Before the fix this read 4 — `maxWeight`,
    // every head of the previous pond saturated.
    expect(worst, 'a fresh pond came back holding the previous pond\'s learning').toBe(0);
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
