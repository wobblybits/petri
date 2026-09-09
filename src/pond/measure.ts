import { CHEM_LEN, EMIT, IN_DEMAND, STATE_DIMS, TASTE, T_OUT, W_IN, W_SELF } from '../chem-layout.ts';
import { CH, CHANNELS } from '../fields.ts';
import type { Sim } from '../sim.ts';

/*
 * Diversity and divergence, as numbers.
 *
 * The pond has always been able to say how *much* there is — bodies, wires,
 * energy — and how deep a lineage runs. It has never been able to say whether
 * what is there is becoming *different*, which is the only question an
 * open-ended system is really being asked. `matrixDrift` comes closest and
 * cannot answer it: it rises under drift and under selection alike, and a
 * whole pond walking together in one direction reads exactly like a pond
 * splitting into two kinds of thing.
 *
 * So the measures here are of *structure*, not magnitude:
 *
 *   - **Richness and evenness.** How many founder lines survive, and whether
 *     the population is spread across them or piled on one. A pond with 200
 *     lines of which one holds 95% is not diverse, and counting lines says it
 *     is. Shannon entropy and its exponential — the effective number of lines
 *     — are what distinguish those.
 *
 *   - **Differentiation between nets.** The one that matters, and the one
 *     nothing measured. Nets are the groups; the statistic is the share of
 *     genetic variance that lies *between* them rather than within, which is
 *     Wright's F_ST in its quantitative form. Near zero, every net is a
 *     random sample of one pond-wide gene pool and the wire graph is not
 *     structuring anything. Rising, nets are becoming distinguishable — which
 *     is what "the simulation builds organisms" would have to look like from
 *     the outside before it looks like anything else.
 *
 *   - **Standing variance.** How much raw material selection has to work on,
 *     split by where in the genome it sits. The seeded bases and the drifted
 *     matrices behave differently and averaging them hides it.
 *
 * Nothing here is a fitness term and nothing here is read by the simulation.
 * These exist to be plotted and compared across runs.
 */

/** Shannon entropy in nats over a count distribution, and its exponential. */
export function shannon(counts: Iterable<number>): { h: number; effective: number } {
  let n = 0;
  const list: number[] = [];
  for (const c of counts) {
    if (c > 0) {
      list.push(c);
      n += c;
    }
  }
  if (n === 0 || list.length === 0) return { h: 0, effective: 0 };
  let h = 0;
  for (const c of list) {
    const p = c / n;
    h -= p * Math.log(p);
  }
  // Hill number of order 1: the number of equally-common groups that would
  // give this entropy. In groups, not nats, which is the readable unit —
  // "effectively four lines" says something an entropy of 1.386 does not.
  return { h, effective: Math.exp(h) };
}

export interface Diversity {
  bodies: number;
  /** Distinct surviving founder lines. */
  lines: number;
  /** Effective number of lines: `exp(H)` over the line-size distribution. */
  linesEffective: number;
  /** The largest line's share of the population, 0 to 1. */
  lineDominance: number;
  /** Connected components with two or more bodies. */
  nets: number;
  /** Effective number of nets by size — one huge net and a hundred pairs is not a hundred nets. */
  netsEffective: number;
  /** Bodies in the largest net, as a share of the pond. */
  netDominance: number;
  /** Effective number of kinds, out of three. */
  kindsEffective: number;
  /**
   * Mean per-locus variance over the genome's drifted span — the matrices and
   * heads past the taste bases, which seed to zero. The raw material.
   */
  varianceDrifted: number;
  /** The same over the eight seeded bases, which start kind-specific. */
  varianceSeeded: number;
  /**
   * Share of genetic variance lying between nets rather than within them —
   * F_ST over the drifted span, in [0, 1]. Null when there is nothing to
   * compare: fewer than two nets, or no variance at all.
   *
   * This is the divergence number. Zero means every net is a random draw from
   * one pond-wide pool.
   */
  netFst: number | null;
  /** The same with founder lines as the groups, which is drift's own signature. */
  lineFst: number | null;
  /**
   * Commutes per latch — `docs/energy-chemistry-plan.md` §8's tripwire on
   * whether nets are doing internal work or merely re-acquiring structure.
   */
  commutesPerLatch: number | null;
  /**
   * Standing stock of the three signalling species across the whole field.
   *
   * The number that says whether anybody is saying anything. It matters most
   * when reading `excreteRate`, because that dial trades two things against
   * each other and reproduction only shows one of them: excretion is what puts
   * signal into the world *and* what empties a body's tank, and the minted
   * deposit is off whenever it is on. So a rate low enough to leave the pond
   * fertile can also be low enough to leave it silent, and a pond nobody can
   * hear is not a cheaper version of a signalling one — it is a different
   * simulation with the same parameters.
   *
   * The ground is excluded for the reason `maxSignal` excludes it: it is the
   * substance rather than something anyone is saying.
   */
  signalTotal: number;
  /**
   * How much more ground a body is standing on than a body placed at random
   * would be. 1 is indifference; above 1 is foraging.
   *
   * The precondition for every question about resource structure, and much
   * cheaper to answer than any of them: it is a spatial statistic on one
   * frame, not an evolutionary outcome over ten minutes, so its seed-to-seed
   * variance is a fraction of `bornMean`'s. If bodies do not find the patches,
   * nothing downstream of "does structure change how nets develop" can mean
   * anything, and there is no point buying seeds to resolve it.
   *
   * The denominator is the dish mean — total ground over the disk's cells,
   * counted from its area rather than by testing the mask — so a body dropped
   * uniformly scores 1 whatever the layout. Null when there is no ground to
   * stand on, or nobody to stand on it.
   */
  forageRatio: number | null;
  /**
   * Engagement gauges: was the mechanism a question is about switched on at
   * all in this pond? See `docs/experiments.md` §1.B.
   *
   * Four structure sweeps once returned a null because every body sat at 1.20
   * of a 1.25 tank — demand read 0.000, the seeded foraging pathway that is
   * gated on it never engaged, and the ponds were indifferent to where the
   * food was for a reason that had nothing to do with foraging. Nothing in
   * the timeline could have said so. These are the numbers that say so.
   *
   * `demandMean` is the mean of `DEMAND` as the genome reads it, clamped to
   * [0, 1]; `fullMean` is the mean tank fraction the same way. Both are
   * host-side arrays on every path.
   */
  demandMean: number;
  fullMean: number;
  /**
   * The p90 of the three signalling species summed at a body's own position,
   * raw — before `senseScale`. Multiplied by the run's `senseScale` it is
   * what a strong local signal reads as on the way into `x`, and that
   * product is what has to sit near one: it is how `SENSE_SCALE` was set,
   * and a sweep once ran an excreting arm at the minted scale, three orders
   * out, and called the bodies indifferent to structure when they were blind.
   */
  signalP90: number | null;
  /**
   * Population means of three named genes.
   *
   * The seeded foraging pathway is two weights — `Wx[0][IN_DEMAND]` carries
   * demand into `h[0]` and `T[energy][0]` reads it as a taste for ground —
   * and the hunger memory a lineage might acquire is `Wh[0][0]`, the
   * self-recurrence on that dimension. A claim that a lineage has learned to
   * act on hunger is a claim about these three numbers, not about an
   * aggregate drift; `matrixDrift` rises whether the walk is selected or
   * random. Read off `chem`, which is the host's to write on every path.
   */
  loci: { wDemandH0: number; wSelf00: number; tFoodH0: number };
}

/** Grouped variance decomposition over a set of loci. Returns F_ST, or null. */
function fst(
  chem: Float32Array,
  slotsAll: number[],
  groupOfAll: number[],
  loci: number[],
  keep?: (i: number) => boolean,
): number | null {
  const slots = keep ? slotsAll.filter((_, i) => keep(i)) : slotsAll;
  const groupOf = keep ? groupOfAll.filter((_, i) => keep(i)) : groupOfAll;
  const n = slots.length;
  if (n < 2) return null;
  const groups = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const g = groups.get(groupOf[i]);
    if (g) g.push(i);
    else groups.set(groupOf[i], [i]);
  }
  if (groups.size < 2) return null;
  /*
   * At least one group has to have two members in it.
   *
   * With every group a singleton the within-group variance is identically
   * zero, so `1 - vW/vT` is **1** by construction — and that is a fact about
   * the grouping, not about the data. A fresh soup is exactly that case: every
   * body is its own component and its own founder line, and the first sample
   * of every run was reporting perfect differentiation for a pond with no
   * structure in it at all. Maximally misleading, because 1 is the reading
   * that would mean the thing we are watching for.
   */
  let estimable = false;
  for (const members of groups.values()) {
    if (members.length >= 2) {
      estimable = true;
      break;
    }
  }
  if (!estimable) return null;

  let totalVar = 0;
  let withinVar = 0;
  for (const k of loci) {
    let sum = 0;
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const v = chem[slots[i] * CHEM_LEN + k];
      sum += v;
      sq += v * v;
    }
    const mean = sum / n;
    const vT = Math.max(0, sq / n - mean * mean);
    if (vT <= 0) continue;
    let vW = 0;
    for (const members of groups.values()) {
      let s = 0;
      let q = 0;
      for (const i of members) {
        const v = chem[slots[i] * CHEM_LEN + k];
        s += v;
        q += v * v;
      }
      const m = s / members.length;
      // Weighted by group size, which is what makes this the *within*
      // component of the same total rather than an average of averages.
      vW += (members.length / n) * Math.max(0, q / members.length - m * m);
    }
    totalVar += vT;
    withinVar += vW;
  }
  if (totalVar <= 0) return null;
  // Variance-weighted across loci rather than a mean of per-locus ratios: a
  // locus with almost no variance has an unstable ratio and would otherwise
  // count as much as one carrying the whole signal.
  return Math.min(1, Math.max(0, 1 - withinVar / totalVar));
}

/** Every measure above, off a live pond. Reads, never writes. */
export function measureDiversity(sim: Sim): Diversity {
  const store = sim.agentStore;
  const chem = store.chemAll;
  const comps = sim.graph.componentIds(sim.agents, sim.rosterVersion);

  const slots: number[] = [];
  const lineOf: number[] = [];
  const compOf: number[] = [];
  const lineCount = new Map<number, number>();
  const compCount = new Map<number, number>();
  const kindCount = new Map<string, number>();
  for (const a of sim.agents.values()) {
    slots.push(a.slot);
    lineOf.push(a.lineage);
    const root = comps.get(a.id) ?? a.id;
    compOf.push(root);
    lineCount.set(a.lineage, (lineCount.get(a.lineage) ?? 0) + 1);
    compCount.set(root, (compCount.get(root) ?? 0) + 1);
    kindCount.set(a.kind, (kindCount.get(a.kind) ?? 0) + 1);
  }
  const n = slots.length;

  const drifted: number[] = [];
  for (let k = TASTE + 4; k < CHEM_LEN; k++) drifted.push(k);
  const seeded: number[] = [];
  for (let k = EMIT; k < TASTE + 4; k++) seeded.push(k);

  const variance = (loci: number[]): number => {
    if (n === 0) return 0;
    let acc = 0;
    for (const k of loci) {
      let sum = 0;
      let sq = 0;
      for (const s of slots) {
        const v = chem[s * CHEM_LEN + k];
        sum += v;
        sq += v * v;
      }
      const mean = sum / n;
      acc += Math.max(0, sq / n - mean * mean);
    }
    return acc / loci.length;
  };

  const lineStats = shannon(lineCount.values());
  // Nets, not components: a lone unattached body is a component of one and is
  // not a net, and counting it as one makes a soup look richly structured.
  const netSizes = [...compCount.values()].filter((c) => c >= 2);
  const netStats = shannon(netSizes);
  const kindStats = shannon(kindCount.values());
  const maxLine = Math.max(0, ...lineCount.values());
  const maxNet = netSizes.length > 0 ? Math.max(...netSizes) : 0;

  /*
   * Sampled at each body's own position, the same bilinear read the genome
   * gets. On the GPU path `fields.data` is a stale mirror — the runner asks
   * for a readback on the frame before a sample precisely so this and `ground`
   * are true when they are read.
   */
  let atBodies = 0;
  const probe = new Float64Array(CHANNELS);
  const signalAt: number[] = [];
  let demandSum = 0;
  let fullSum = 0;
  let wDemandH0 = 0;
  let wSelf00 = 0;
  let tFoodH0 = 0;
  const REQUEST = store.request;
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  for (const a of sim.agents.values()) {
    sim.fields.sampleAll(a.x, a.y, probe, 0);
    atBodies += probe[CH.energy];
    signalAt.push(probe[CH.conP] + probe[CH.dupP] + probe[CH.aux]);
    // The same clamps `updateState` applies when it builds `x`, so these
    // are the inputs the genome saw and not the raw store.
    const r = REQUEST[a.slot];
    demandSum += r <= 0 ? 0 : r >= 1 ? 1 : r;
    const cap = CAP[a.slot];
    const full = cap > 0 ? EXTRA[a.slot] / cap : 0;
    fullSum += full <= 0 ? 0 : full >= 1 ? 1 : full;
    const g = a.slot * CHEM_LEN;
    wDemandH0 += chem[g + W_IN + IN_DEMAND];
    wSelf00 += chem[g + W_SELF];
    tFoodH0 += chem[g + T_OUT + CH.energy * STATE_DIMS];
  }
  let signalP90: number | null = null;
  if (signalAt.length > 0) {
    signalAt.sort((p, q) => p - q);
    signalP90 = signalAt[Math.min(signalAt.length - 1, Math.floor(0.9 * (signalAt.length - 1)))];
  }
  const f = sim.fields;
  const cell = f.worldW / f.cols;
  const diskCells = (Math.PI * sim.worldR * sim.worldR) / (cell * cell);
  let groundTotal = 0;
  for (let k = CH.energy; k < f.data.length; k += CHANNELS) groundTotal += f.data[k];
  const dishMean = diskCells > 0 ? groundTotal / diskCells : 0;
  const forageRatio = n > 0 && dishMean > 0 ? atBodies / n / dishMean : null;

  let signalTotal = 0;
  const field = sim.fields.data;
  for (let k = 0; k < field.length; k += CHANNELS) {
    for (let c = 0; c < CHANNELS; c++) if (c !== CH.energy) signalTotal += field[k + c];
  }

  return {
    bodies: n,
    lines: lineCount.size,
    linesEffective: lineStats.effective,
    lineDominance: n > 0 ? maxLine / n : 0,
    nets: netSizes.length,
    netsEffective: netStats.effective,
    netDominance: n > 0 ? maxNet / n : 0,
    kindsEffective: kindStats.effective,
    varianceDrifted: variance(drifted),
    varianceSeeded: variance(seeded),
    // Nets, not components: a lone unattached body is not a net, and grouping
    // by it would let the soup's singletons decide a statistic about nets.
    // Same definition `nets` above uses.
    netFst: fst(chem, slots, compOf, drifted, (i) => (compCount.get(compOf[i]) ?? 0) >= 2),
    lineFst: fst(chem, slots, lineOf, drifted),
    commutesPerLatch: sim.census().commutesPerLatch,
    signalTotal,
    forageRatio,
    demandMean: n > 0 ? demandSum / n : 0,
    fullMean: n > 0 ? fullSum / n : 0,
    signalP90,
    loci: {
      wDemandH0: n > 0 ? wDemandH0 / n : 0,
      wSelf00: n > 0 ? wSelf00 / n : 0,
      tFoodH0: n > 0 ? tFoodH0 / n : 0,
    },
  };
}
