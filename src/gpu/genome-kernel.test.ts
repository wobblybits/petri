import { describe, expect, it } from 'vitest';
import shader from './genome.wgsl?raw';
import {
  B_STATE, CHEM_LEN, EMIT, E_OUT, F_BASE, F_OUT, HEAD_SCALE, IN_DIMS, L_BASE, L_OUT,
  LEARN_CRITIC, LEARN_PREV_V, LEARN_STRIDE, LEARN_TRACE,
  P_BASE, P_OUT, PLASTIC_LEN, SENSE_SCALE, STATE_DIMS, TASTE, T_OUT, W_IN, W_NET, W_SELF,
} from '../chem-layout.ts';
import { refreshReadsField } from '../agents.ts';
import type { AgentKind } from '../agents.ts';
import { WireAdjacency } from '../energy.ts';
import { CH } from '../fields.ts';
import { defaultParams } from '../params.ts';
import { Sim } from '../sim.ts';

/**
 * `genome.wgsl` against `chem-layout.ts` and against the pass it ports.
 *
 * There is no WebGPU under Node, so the shader cannot run here. Two things
 * can still be checked, and they are the two that have actually gone wrong in
 * this project.
 *
 * The layout constants are transcribed into the shader by hand, because WGSL
 * has no way to import them. That is exactly the duplication `chem-layout.ts`
 * was created to end — a hand-copied `CHEM_LEN` drifted twice in one session
 * and the second time shipped a `RangeError` from four frames inside a typed
 * array. So the first test parses them back out of the shader source and
 * compares. It cannot drift silently again.
 *
 * The second is arithmetic: a mirror of the kernel, checked against the CPU
 * `updateState` through a hand-built genome.
 */

/** Pull `const NAME: u32 = 123u;` out of the shader. */
function shaderConst(name: string): number {
  const m = new RegExp(`const\\s+${name}\\s*:\\s*u32\\s*=\\s*(\\d+)u\\s*;`).exec(shader);
  if (!m) throw new Error(`no const ${name} in genome.wgsl`);
  return Number(m[1]);
}

describe('the genome shader matches the genome layout', () => {
  it('carries the same offsets chem-layout.ts derives', () => {
    // If this fails, the shader is reading a different genome than the rest of
    // the program writes, and every number it produces will be plausible.
    const want: Record<string, number> = {
      STATE_DIMS, IN_DIMS, EMIT, TASTE, E_OUT, T_OUT, W_IN, W_SELF, W_NET,
      B_STATE, F_OUT, F_BASE, P_OUT, P_BASE, L_OUT, L_BASE,
      // The learning row is indexed by the same hand-copied constants and
      // carries the same hazard.
      PLASTIC_LEN, LEARN_TRACE, LEARN_CRITIC, LEARN_PREV_V, LEARN_STRIDE,
    };
    for (const [name, value] of Object.entries(want)) {
      expect(shaderConst(name), `genome.wgsl's ${name}`).toBe(value);
    }
  });

  it('writes the eighteen floats a body the host unpacks', () => {
    // h(4) + emit(4) + taste(4) + six heads. The host reads them back by this
    // stride, so a mismatch shifts every field by a body.
    expect(shaderConst('OUT_STRIDE')).toBe(4 + 4 + 4 + 6);
  });

  it('has no genome offset the layout does not derive', () => {
    // The other direction: a constant added to the shader alone would pass
    // every check above by never being looked at.
    const declared = [...shader.matchAll(/const\s+([A-Z_][A-Z0-9_]*)\s*:\s*u32/g)].map((m) => m[1]);
    const known = new Set([
      'STATE_DIMS', 'IN_DIMS', 'EMIT', 'TASTE', 'E_OUT', 'T_OUT', 'W_IN', 'W_SELF',
      'W_NET', 'B_STATE', 'F_OUT', 'F_BASE', 'P_OUT', 'P_BASE', 'L_OUT', 'L_BASE',
      'OUT_STRIDE', 'PLASTIC_LEN', 'LEARN_TRACE', 'LEARN_CRITIC', 'LEARN_PREV_V',
      'LEARN_STRIDE',
    ]);
    expect(declared.filter((d) => !known.has(d)), 'undocumented shader constant').toEqual([]);
  });

  it('takes its uniform in the order the host writes it', () => {
    /*
     * The host hand-packs this struct into an ArrayBuffer by index, so the
     * field order here is load-bearing in exactly the way the offsets above
     * are: a field inserted on one side and not the other reads as a
     * plausible number rather than an error. Every member is a scalar, so
     * index and slot are the same thing and nothing has to be aligned.
     */
    const body = /struct\s+GenomeParams\s*\{([^}]*)\}/.exec(shader);
    expect(body, 'no GenomeParams in genome.wgsl').not.toBeNull();
    const fields = [...body![1].matchAll(/^\s*([A-Za-z_]\w*)\s*:/gm)].map((m) => m[1]);
    expect(fields).toEqual([
      'n', 'chemLen', 'senseScale', 'groundScale',
      'sCruise', 'sTurn', 'sAlign', 'sSep', 'sThrust', 'sRecoil', 'energyCh',
      'learnRate', 'learnCritic', 'learnTrace', 'learnDiscount', 'maxWeight',
      'pad0', 'pad1', 'pad2', 'pad3',
    ]);
    // A uniform buffer's size has to be a whole number of sixteen-byte
    // blocks, which is what the pads are for.
    expect((fields.length * 4) % 16).toBe(0);
  });

  it('binds one buffer per thing it reads, inside the guaranteed eight', () => {
    /*
     * WebGPU guarantees only eight storage buffers per compute stage, and an
     * adapter offering more is the trap: taking it up works on the machine it
     * was written on and nowhere else. The field pass next door had to merge
     * two bindings to fit; this one has the learning row and no room left.
     */
    const storage = [...shader.matchAll(/@binding\(\d+\)\s*var<storage/g)].length;
    expect(storage, 'genome.wgsl is over the guaranteed storage-buffer limit').toBeLessThanOrEqual(8);
  });

  it('reads inside the genome it is given', () => {
    // The furthest the shader can reach is the last head weight. If the layout
    // ever grows a field after `L_BASE` without the shader knowing, this stays
    // true and the test above catches it; if the shader ever reaches past the
    // genome, this is what fails.
    expect(L_BASE + 2).toBe(CHEM_LEN);
  });
});

/**
 * `state`, line for line, in the same list order the shader uses.
 *
 * `facts` is [full, bound, demand, readsField] a body, `slots` the genome row,
 * and `samples` the raw four channels the field probe left under it — the same
 * four the shader reads out of `samples[i*2+1]`.
 */
function mirrorState(a: {
  chem: Float32Array;
  hPrev: Float32Array;
  facts: Float32Array;
  slots: Int32Array;
  samples: Float32Array;
  off: Uint32Array;
  nei: Uint32Array;
  n: number;
  groundScale: number;
  energyCh: number;
  /** The device's learning rows, `LEARN_STRIDE` a slot. Absent = nothing learned. */
  learn?: Float32Array;
}): Float32Array {
  const S = STATE_DIMS;
  const { chem, hPrev, facts, slots, samples, off, nei, n } = a;
  const learn = a.learn;
  const out = new Float32Array(n * 18);
  const phi = (v: number): number => v / (1 + Math.abs(v));
  const cl = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);
  for (let i = 0; i < n; i++) {
    const g = slots[i] * CHEM_LEN;
    const lb = slots[i] * LEARN_STRIDE;
    /** A state weight as the body has it: gene plus what it learned. */
    const sw = (k: number): number => chem[g + W_IN + k] + (learn ? learn[lb + k] : 0);
    // No sense gate, matching the shader: `gather` has taken the sample for
    // every body either way, so gating it saves nothing there, and the flag
    // is settled at birth while a learned sense weight is not.
    const s = [0, 0, 0, 0];
    for (let c = 0; c < 4; c++) s[c] = samples[i * 4 + c] / SENSE_SCALE;
    s[a.energyCh] = samples[i * 4 + a.energyCh] * a.groundScale;
    const x = [s[0], s[1], s[2], s[3], facts[i * 4], facts[i * 4 + 1], facts[i * 4 + 2]];
    const p = [hPrev[i * S], hPrev[i * S + 1], hPrev[i * S + 2], hPrev[i * S + 3]];
    const m = [0, 0, 0, 0];
    const lo = off[i];
    const hi = off[i + 1];
    const deg = hi - lo;
    for (let e = lo; e < hi; e++) {
      for (let k = 0; k < S; k++) m[k] += hPrev[nei[e] * S + k];
    }
    if (deg > 0) for (let k = 0; k < S; k++) m[k] /= deg;
    const h = [0, 0, 0, 0];
    for (let d = 0; d < S; d++) {
      const wi = d * IN_DIMS;
      const ws = W_SELF - W_IN + d * S;
      const wn = W_NET - W_IN + d * S;
      let v = sw(B_STATE - W_IN + d);
      for (let k = 0; k < IN_DIMS; k++) v += sw(wi + k) * x[k];
      for (let k = 0; k < S; k++) v += sw(ws + k) * p[k];
      for (let k = 0; k < S; k++) v += sw(wn + k) * m[k];
      h[d] = phi(v);
    }
    const dot = (base: number, row: number): number => {
      const o = g + base + row * S;
      let v = 0;
      for (let k = 0; k < S; k++) v += chem[o + k] * h[k];
      return v;
    };
    const emit = [0, 0, 0, 0];
    let sum = 0;
    for (let c = 0; c < 4; c++) {
      const w = Math.max(chem[g + EMIT + c] + dot(E_OUT, c), 0);
      emit[c] = w;
      sum += w;
    }
    if (sum > 1e-6) for (let c = 0; c < 4; c++) emit[c] /= sum;
    const o = i * 18;
    for (let d = 0; d < S; d++) out[o + d] = h[d];
    for (let c = 0; c < 4; c++) out[o + 4 + c] = emit[c];
    for (let c = 0; c < 4; c++) out[o + 8 + c] = chem[g + TASTE + c] + dot(T_OUT, c);
    const head = (mat: number, base: number, row: number, scale: number): number =>
      (chem[g + base + row] + dot(mat, row)) * scale;
    out[o + 12] = cl(head(L_OUT, L_BASE, 0, HEAD_SCALE.cruise), 0, 180);
    out[o + 13] = cl(head(L_OUT, L_BASE, 1, HEAD_SCALE.turn), 0, 8);
    out[o + 14] = cl(head(F_OUT, F_BASE, 0, HEAD_SCALE.align), -8, 16);
    out[o + 15] = cl(head(F_OUT, F_BASE, 1, HEAD_SCALE.sep), -60, 120);
    out[o + 16] = cl(head(P_OUT, P_BASE, 0, HEAD_SCALE.thrust), 0, 1);
    out[o + 17] = cl(head(P_OUT, P_BASE, 1, HEAD_SCALE.recoil), 0, 200);
  }
  return out;
}

describe('the genome shader computes what updateState computes', () => {
  it('agrees with the CPU pass on a wired pond, state, emit, taste and heads', () => {
    /*
     * The real reference, not a second hand-written one: build a pond, let the
     * CPU pass run, then feed the mirror the same inputs and compare what the
     * store holds. Wires matter — `Wn` is the only term that reads anything
     * outside a body, and a pass that ignored it would agree everywhere else.
     */
    /*
     * The inputs have to be *constant across a frame*, or this compares the
     * wrong moment: `updateState` runs inside `endFrame`, after the solve has
     * moved every body and after harvest and `pulseRequests` have rewritten
     * `extra` and `request`. A snapshot taken between frames is not what the
     * pass sees.
     *
     * So the pond is arranged to hold still in every way the inputs depend on.
     * The field is made uniform and nothing is allowed to change it — no
     * deposit, no diffusion, no decay, no growth, no reaction — which makes a
     * body's sense reading independent of where it has drifted to. Every body
     * starts full, so harvest skips it and `FULL` stays 1. With no rewrites
     * and no snapping, `BOUND` cannot move, and with nobody hungry and no
     * appetite, `DEMAND` stays 0.
     *
     * What still varies frame to frame is `h` itself, through `Wh` and `Wn`,
     * which is the part worth testing.
     */
    const params = defaultParams();
    params.spawnInterval = 0;
    params.rewriteDuration = 0;
    params.snapRadius = 0;
    params.upkeep = 0;
    params.deposit = 0;
    params.diffuse = 0;
    params.decay = 0;
    params.energyRegrow = 0;
    params.energyDiffuse = 0;
    params.farmRate = 0;
    params.swimCost = 0;
    params.forageAsk = 0;
    params.fertilise = 0;
    params.reactFeed = 0;
    params.reactKill = 0;
    params.ambientEnergy = 1;
    const sim = new Sim(20000, 20000);
    const kinds: AgentKind[] = ['con', 'dup', 'era'];
    const ids = [];
    for (let i = 0; i < 24; i++) {
      ids.push(sim.spawn(kinds[i % 3], 9000 + (i % 6) * 60, 9000 + ((i / 6) | 0) * 60, i * 0.7, params, true)!);
    }
    sim.step(1 / 60, params);
    for (let i = 0; i + 1 < ids.length; i += 2) {
      sim.wire(ids[i].id, 'p', ids[i + 1].id, 'p', params);
    }
    // Genomes that actually exercise every term: seeded ones have zero
    // matrices, so `h` stays at zero and every head is its base.
    let seed = 12345;
    const rnd = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return (seed / 0x7fffffff) * 2 - 1;
    };
    for (const a of sim.agents.values()) {
      for (let k = 0; k < CHEM_LEN; k++) a.chem[k] = rnd() * 1.5;
      refreshReadsField(a);
    }
    // A uniform field, so where a body has drifted to does not change what it
    // smells, and full tanks so nothing grazes.
    const uniform = [0.8, 1.3, sim.energy.cellCap, 0.45];
    for (let i = 0; i < sim.fields.data.length; i += 4) {
      for (let c = 0; c < 4; c++) sim.fields.data[i + c] = uniform[c];
    }
    for (const a of sim.agents.values()) a.extra = a.energyCap;
    for (let f = 0; f < 3; f++) sim.step(1 / 60, params);

    // Snapshot the inputs the way `gpuFieldStep` will, before the pass runs.
    const store = sim.agentStore;
    const list = [...sim.agents.values()];
    const n = list.length;
    const adj = new WireAdjacency();
    const index = new Map<number, number>();
    for (let i = 0; i < n; i++) index.set(list[i].id, i);
    adj.build(n, index, () => sim.graph.wires.values());
    const hPrev = new Float32Array(n * STATE_DIMS);
    const facts = new Float32Array(n * 4);
    const slots = new Int32Array(n);
    const samples = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const s = list[i].slot;
      slots[i] = s;
      for (let d = 0; d < STATE_DIMS; d++) hPrev[i * STATE_DIMS + d] = store.hAll[s * STATE_DIMS + d];
      const cap = store.energyCap[s];
      const full = cap > 0 ? store.extra[s] / cap : 0;
      facts[i * 4] = full <= 0 ? 0 : full >= 1 ? 1 : full;
      facts[i * 4 + 1] = store.bound[s];
      const r = store.request[s];
      facts[i * 4 + 2] = r <= 0 ? 0 : r >= 1 ? 1 : r;
      facts[i * 4 + 3] = store.readsField[s];
      for (let c = 0; c < 4; c++) samples[i * 4 + c] = uniform[c];
    }

    // One more CPU frame: that is the pass the mirror is predicting.
    sim.step(1 / 60, params);

    const got = mirrorState({
      chem: store.chemAll, hPrev, facts, slots, samples,
      off: Uint32Array.from(adj.off.subarray(0, n + 1)),
      nei: Uint32Array.from(adj.nei),
      n, groundScale: 1 / sim.energy.cellCap, energyCh: CH.energy,
    });

    let worstH = 0;
    let worstEmit = 0;
    let worstTaste = 0;
    let worstHead = 0;
    let moved = 0;
    for (let i = 0; i < n; i++) {
      const s = list[i].slot;
      const o = i * 18;
      for (let d = 0; d < STATE_DIMS; d++) {
        worstH = Math.max(worstH, Math.abs(got[o + d] - store.hAll[s * STATE_DIMS + d]));
        if (Math.abs(store.hAll[s * STATE_DIMS + d]) > 1e-3) moved++;
      }
      for (let c = 0; c < 4; c++) {
        worstEmit = Math.max(worstEmit, Math.abs(got[o + 4 + c] - store.emitAll[s * 4 + c]));
        worstTaste = Math.max(worstTaste, Math.abs(got[o + 8 + c] - store.tasteAll[s * 4 + c]));
      }
      worstHead = Math.max(
        worstHead,
        Math.abs(got[o + 12] - store.cruise[s]),
        Math.abs(got[o + 13] - store.turn[s]),
        Math.abs(got[o + 14] - store.flockAlign[s]),
        Math.abs(got[o + 15] - store.flockSep[s]),
        Math.abs(got[o + 16] - store.transportThrust[s]),
        Math.abs(got[o + 17] - store.transportRecoil[s]),
      );
    }
    expect(moved, 'every state stayed at zero, so this compared nothing').toBeGreaterThan(n);
    expect(worstH, 'the recurrent state diverged').toBeLessThan(1e-5);
    expect(worstEmit, 'the emit vector diverged').toBeLessThan(1e-5);
    expect(worstTaste, 'the taste vector diverged').toBeLessThan(1e-5);
    expect(worstHead, 'a head diverged').toBeLessThan(1e-3);
  });
});

