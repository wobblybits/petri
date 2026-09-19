import { describe, expect, it } from 'vitest';
import { CH, CHANNELS } from './fields.ts';
import { TASTE } from './agents.ts';
import { fixedParams } from './test-params.ts';
import { Sim } from './sim.ts';
import type { Params } from './params.ts';

/**
 * The nose-to-tail coordinate.
 *
 * A net can undulate and cannot aim: the wave runs along the wiring order at a
 * wavelength set by global constants, and nothing about where the food is
 * reaches it. `rho-hat` is what the inchworm bench aims with — depth behind
 * the nose as a fraction of nose-to-tail, computed from relayed scalars alone
 * so no body needs to know how big its net is.
 *
 * Three claims are worth pinning, and they are the ones the bench says carry
 * the mechanism: it runs nose to tail rather than scattering; **a head is a
 * body that claims itself**, which after `interiorTaste` makes every head a
 * leaf by construction; and it re-aims when the food moves, without anything
 * being relearned.
 */

const AT = { x: 2000, y: 2000 };
const N = 8;

function chain(tune?: (p: Params) => void): {
  sim: Sim;
  params: Params;
  ids: number[];
} {
  const params = fixedParams();
  params.spawnInterval = 0;
  params.soupCount = 0;
  params.deposit = 0;
  params.portLeak = 0;
  params.rewriteDuration = 0;
  /*
   * The smell is what a nose climbs — `groundSmell` mints aux in proportion to
   * the ground in a cell and aux travels at the full signal rate, where the
   * ground itself is held at 0.006 of it. This test paints that gradient by
   * hand rather than growing one: how the halo gets there is Phase 2's
   * question and diffusion, grazing and drift would all be in the answer.
   * What is under test is the relay over a reading that already exists.
   */
  params.ambientEnergy = 0;
  params.groundSmell = 0;
  params.groundDropEvery = 0;
  /*
   * Not pinned, and that is not laziness. The relay reads `store.trail` — the
   * body's own taste score at its own position — and the steer pass, which is
   * the only thing that writes it, skips a pinned body. So a pinned chain has
   * no readings at all and the relay has nothing to relay. Held still by
   * turning the swimming off instead.
   */
  params.stepSpeed = 0;
  params.swimNoise = 0;
  params.turnRate = 0;
  params.decay = 0;
  params.diffuse = 0;
  tune?.(params);
  const sim = new Sim(4000, 4000);
  const ids: number[] = [];
  for (let i = 0; i < N; i++) {
    // Mouth to ear down the chain, so nothing is principal-to-principal and
    // nothing is a redex: an Era at the head, which has only a principal to
    // give, and Cons after it, each taking the one before on its left ear.
    const kind = i === 0 ? 'era' : 'con';
    const a = sim.spawn(kind, AT.x + i * 60, AT.y, 0, params, true)!;
    // One thing to smell: the ground.
    for (let k = 0; k < 4; k++) a.chem[TASTE + k] = 0;
    a.chem[TASTE + CH.aux] = 1;
    ids.push(a.id);
  }
  for (let i = 0; i + 1 < N; i++) {
    sim.graph.connect(
      sim.agents,
      { id: ids[i], slot: 'p' },
      { id: ids[i + 1], slot: 'l' },
      sim.w,
      sim.h,
      params,
      sim.time,
    );
  }
  return { sim, params, ids };
}

/**
 * A smell that falls off with distance from `(x, y)`, painted straight onto
 * the field. With `diffuse` and `decay` at 0 and nothing depositing, it stays
 * exactly as painted for the length of the test.
 */
const SMELL_REACH = 1400;
function feedAt(sim: Sim, params: Params, x: number, y: number): void {
  const f = sim.fields;
  const d = f.data;
  const cell = f.worldW / f.cols;
  for (let j = 0; j < f.rows; j++) {
    const cy = f.originY + (j + 0.5) * cell;
    for (let i = 0; i < f.cols; i++) {
      const cx = f.originX + (i + 0.5) * cell;
      const r = Math.hypot(cx - x, cy - y);
      const k = (j * f.cols + i) * CHANNELS;
      d[k + CH.aux] = Math.max(0, 1 - r / SMELL_REACH);
    }
  }
  // The dirty box has to cover what was painted, or nothing samples it.
  f.touchWorld(f.originX + cell * 0.5, f.originY + cell * 0.5);
  f.touchWorld(f.originX + (f.cols - 0.5) * cell, f.originY + (f.rows - 0.5) * cell);
  // Long enough for a claim to cross eight bodies at one hop a frame.
  for (let fr = 0; fr < 60; fr++) sim.step(1 / 60, params);
}

const trailOf = (sim: Sim, ids: number[]): number[] =>
  ids.map((id) => sim.agentStore.trail[sim.agents.get(id)!.slot]);

const depthOf = (sim: Sim, ids: number[]): number[] =>
  ids.map((id) => sim.agentStore.depth[sim.agents.get(id)!.slot]);
const headsOf = (sim: Sim, ids: number[]): number[] =>
  ids.map((id) => sim.agentStore.depthHead[sim.agents.get(id)!.slot]);
const wiresOf = (sim: Sim, ids: number[]): number[] =>
  ids.map((id) => sim.agentStore.wires[sim.agents.get(id)!.slot]);

describe('the depth coordinate', () => {
  it('runs nose to tail, from whichever end is nearest the food', () => {
    const { sim, params, ids } = chain();
    feedAt(sim, params, AT.x - 260, AT.y);
    // The gradient has to exist, or everything below is about nothing.
    const smelt = trailOf(sim, ids);
    expect(smelt[0], `no gradient along the chain: ${smelt.map((v) => v.toFixed(3)).join(' ')}`)
      .toBeGreaterThan(smelt[N - 1] * 1.2);
    const near = depthOf(sim, ids);
    const dbg = ids.map((id) => {
      const sl = sim.agents.get(id)!.slot;
      const st = sim as unknown as { hDst: Float64Array; tDst: Float64Array; hSrc: Int32Array };
      return `${st.hDst[sl].toFixed(0)}/${st.tDst[sl].toFixed(0)}@${st.hSrc[sl]}`;
    }).join(' ');
    expect(near[0], `nose? ${near.map((v) => v.toFixed(2)).join(' ')} | ${dbg}`).toBeLessThan(0.1);
    expect(near[N - 1], `tail? ${near.map((v) => v.toFixed(2)).join(' ')} | ${dbg}`).toBeGreaterThan(0.9);
    for (let i = 1; i < N; i++) {
      expect(near[i], `depth fell between ${i - 1} and ${i}: ${near.join(' ')}`)
        .toBeGreaterThanOrEqual(near[i - 1] - 1e-9);
    }

    /*
     * And it turns round when the food does. Nothing is relearned and no
     * parameter moves: the frame is the field's, so the same profile over it
     * points the other way.
     */
    feedAt(sim, params, AT.x + (N - 1) * 60 + 260, AT.y);
    const far = depthOf(sim, ids);
    expect(far[0], 'the nose did not move to the other end').toBeGreaterThan(0.9);
    expect(far[N - 1]).toBeLessThan(0.1);
  });

  it('makes every head a leaf, by construction', () => {
    const { sim, params, ids } = chain();
    feedAt(sim, params, AT.x - 260, AT.y);
    const heads = headsOf(sim, ids);
    const wires = wiresOf(sim, ids);
    expect(wires, 'the rig is not the chain this test is about')
      .toEqual([1, 2, 2, 2, 2, 2, 2, 1]);
    for (let i = 0; i < N; i++) {
      if (wires[i] > 1) {
        expect(heads[i], `body ${i} has ${wires[i]} wires and claimed itself`).toBe(0);
      }
    }
    expect(heads.reduce((s, h) => s + h, 0), 'nothing claimed itself').toBeGreaterThan(0);
  });

  it('is off at a discount of zero, and costs nothing when it is', () => {
    const { sim, params, ids } = chain((p) => {
      p.depthCost = 0;
    });
    feedAt(sim, params, AT.x - 260, AT.y);
    expect(depthOf(sim, ids).every((d) => d === 0)).toBe(true);
    expect(headsOf(sim, ids).every((h) => h === 0)).toBe(true);
  });
});
