import { CHEM_LEN, CHEM_SPECIES, EMIT, G_BASE, IN_DEMAND, ROW_EXCRETE, STATE_DIMS, TASTE, T_OUT, W_IN, W_SELF, X_BASE } from '../chem-layout.ts';
import { EXTRA_CAP } from '../energy.ts';
import { CH, CHANNELS } from '../fields.ts';
import type { Params } from '../params.ts';
import type { Sim } from '../sim.ts';

/*
 * Diversity and divergence measures: richness and evenness of founder lines,
 * differentiation between nets (Wright's F_ST over genetic variance), and
 * standing variance split by whether a locus is seeded or drifted.
 * Nothing here is a fitness term and nothing here is read by the simulation.
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
  // give this entropy.
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
  /** Mean per-locus variance over the loci that start identical on every body. */
  varianceDrifted: number;
  /** The same over the bases that start kind-specific: emit, taste, gait and what a kind makes. */
  varianceSeeded: number;
  /**
   * Share of genetic variance lying between nets rather than within them —
   * F_ST over the drifted span, in [0, 1]. Null when there is nothing to
   * compare: fewer than two nets, or no variance at all.
   */
  netFst: number | null;
  /** The same with founder lines as the groups, which is drift's own signature. */
  lineFst: number | null;
  /** Commutes per latch: whether nets do internal work or merely re-acquire structure. */
  commutesPerLatch: number | null;
  /**
   * Standing stock of the three signalling species across the whole field,
   * ground excluded. Read beside `excreteRate`: a rate low enough to leave the
   * pond fertile can also leave it silent.
   */
  signalTotal: number;
  /**
   * How much more ground a body is standing on than a body placed at random
   * would be. 1 is indifference; above 1 is foraging. The denominator is the
   * dish mean (total ground over the disk's cells, counted from its area), so
   * a body dropped uniformly scores 1 whatever the layout. Null when there is
   * no ground, or nobody to stand on it.
   */
  forageRatio: number | null;
  /**
   * Size-weighted mean speed of a net's centre of mass, in px/s, over
   * components of two or more. A snapshot, so it measures drift and jostling
   * alike and tracks how mobile the pond is, not how well it swims;
   * `netCoherence` tells them apart, and the control is `transportRecoil` at 0
   * (see the coupling table in `docs/experiments.md`). Null when there is no net.
   */
  netDrift: number | null;
  /**
   * `netDrift` over what it would be if the net's bodies moved independently:
   * the baseline is `sqrt(sum(m_i^2 |v_i|^2)) / sum(m_i)`, so independence
   * reads 1 whatever the net's size and a rigid net reads
   * `sum(m_i) / sqrt(sum(m_i^2))`. The ceiling therefore grows with net size,
   * and a net dragged bodily by the flock scores above 1 too: read it against
   * a control at the same `net_dominance`, never as an absolute level.
   */
  netCoherence: number | null;
  /**
   * Engagement gauges (`docs/experiments.md` §1.B): `demandMean` is the mean
   * of `DEMAND` as the genome reads it, clamped to [0, 1]; `fullMean` is the
   * mean tank fraction the same way. Both are host-side arrays on every path.
   */
  demandMean: number;
  fullMean: number;
  /**
   * The p90 of the three signalling species summed at a body's own position,
   * raw, before `senseScale`. Its product with the run's `senseScale` is what
   * a strong local signal reads as on the way into `x`, and should sit near one.
   */
  signalP90: number | null;
  /**
   * Population means of the seeded foraging pathway: `Wx[0][IN_DEMAND]`
   * carries demand into `h[0]`, `T[energy][0]` reads it as a taste for ground,
   * and `Wh[0][0]` is the self-recurrence that could hold a hunger memory.
   * Read off `chem`, which is the host's to write on every path.
   */
  loci: { wDemandH0: number; wSelf00: number; tFoodH0: number };
  /**
   * The larval window: how long a body is alone before it joins anything,
   * cumulative over the whole run. Read `latchP50` against `tankLife`.
   * `loneliness` is the share of arrivals that died having never latched.
   */
  latchP50: number | null;
  latchP90: number | null;
  loneliness: number | null;
  /** Seconds of upkeep a full tank buys, at this run's `upkeep`. */
  tankLife: number | null;
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
  // At least one group must have two members: with every group a singleton
  // the within-group variance is identically zero and F_ST reads 1 by
  // construction, which is a fresh soup's case.
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
      // Weighted by group size so this is the within component of the same total.
      vW += (members.length / n) * Math.max(0, q / members.length - m * m);
    }
    totalVar += vT;
    withinVar += vW;
  }
  if (totalVar <= 0) return null;
  // Variance-weighted across loci: a near-constant locus has an unstable ratio.
  return Math.min(1, Math.max(0, 1 - withinVar / totalVar));
}

/**
 * Net motion: speed of a net's centre of mass, and how much of that is the net
 * agreeing with itself. Both size-weighted over nets, as `netFst` is.
 * Components of one are skipped: a lone body's coherence is 1 by construction.
 */
function netMotion(
  sim: Sim,
  comps: Map<number, number>,
  compCount: Map<number, number>,
): { drift: number | null; coherence: number | null } {
  const store = sim.agentStore;
  const VX = store.vx;
  const VY = store.vy;
  const MASS = store.mass;
  // px, py: net momentum. m: net mass. sq: sum of (m_i |v_i|)^2, the
  // independent-motion baseline's numerator.
  const acc = new Map<number, { px: number; py: number; m: number; sq: number }>();
  for (const a of sim.agents.values()) {
    const root = comps.get(a.id) ?? a.id;
    if ((compCount.get(root) ?? 0) < 2) continue;
    const s = a.slot;
    const m = Math.max(0.08, MASS[s]);
    const vx = VX[s];
    const vy = VY[s];
    let e = acc.get(root);
    if (!e) acc.set(root, (e = { px: 0, py: 0, m: 0, sq: 0 }));
    e.px += m * vx;
    e.py += m * vy;
    e.m += m;
    e.sq += m * m * (vx * vx + vy * vy);
  }
  if (acc.size === 0) return { drift: null, coherence: null };
  let driftSum = 0;
  let cohSum = 0;
  let cohWeight = 0;
  let weight = 0;
  for (const [root, e] of acc) {
    const size = compCount.get(root) ?? 0;
    const speed = e.m > 0 ? Math.hypot(e.px, e.py) / e.m : 0;
    driftSum += size * speed;
    weight += size;
    // A net standing still has no direction to be coherent about; leave it out.
    const baseline = e.m > 0 ? Math.sqrt(e.sq) / e.m : 0;
    if (baseline > 0) {
      cohSum += size * (speed / baseline);
      cohWeight += size;
    }
  }
  return {
    drift: weight > 0 ? driftSum / weight : null,
    coherence: cohWeight > 0 ? cohSum / cohWeight : null,
  };
}

/**
 * Every measure above, off a live pond. Reads, never writes. `params` only
 * supplies `upkeep` for `tankLife`, which is not on the Sim.
 */
export function measureDiversity(sim: Sim, params?: Params): Diversity {
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

  /*
   * The split is by what a locus is at birth: `drifted` starts the same on
   * every body, `seeded` starts kind-specific (emit and taste bases, the gait
   * head bases from `seedGait`, the excretion bases from `seedProduction`).
   * `X`'s uptake bases and both head matrices seed identically on every kind
   * and stay with the drifted set.
   */
  const kindSeeded = new Set([G_BASE, G_BASE + 1]);
  for (let c = 0; c < CHEM_SPECIES; c++) kindSeeded.add(X_BASE + ROW_EXCRETE + c);
  const drifted: number[] = [];
  for (let k = TASTE + 4; k < CHEM_LEN; k++) if (!kindSeeded.has(k)) drifted.push(k);
  const seeded: number[] = [];
  for (let k = EMIT; k < TASTE + 4; k++) seeded.push(k);
  for (const k of kindSeeded) seeded.push(k);

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
  // Nets, not components: a lone body is a component of one and not a net.
  const netSizes = [...compCount.values()].filter((c) => c >= 2);
  const netStats = shannon(netSizes);
  const kindStats = shannon(kindCount.values());
  const maxLine = Math.max(0, ...lineCount.values());
  const maxNet = netSizes.length > 0 ? Math.max(...netSizes) : 0;

  // Sampled at each body's own position, the same bilinear read the genome
  // gets. On the GPU path `fields.data` is a mirror the runner refreshes with
  // a readback on the frame before a sample, so it is true when read here.
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
    // The same clamps `updateState` applies when it builds `x`.
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

  const motion = netMotion(sim, comps, compCount);

  let signalTotal = 0;
  const larval = sim.larval.read();
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
    // Nets, not components, the same definition `nets` above uses.
    netFst: fst(chem, slots, compOf, drifted, (i) => (compCount.get(compOf[i]) ?? 0) >= 2),
    lineFst: fst(chem, slots, lineOf, drifted),
    commutesPerLatch: sim.census().commutesPerLatch,
    signalTotal,
    forageRatio,
    netDrift: motion.drift,
    netCoherence: motion.coherence,
    demandMean: n > 0 ? demandSum / n : 0,
    fullMean: n > 0 ? fullSum / n : 0,
    signalP90,
    loci: {
      wDemandH0: n > 0 ? wDemandH0 / n : 0,
      wSelf00: n > 0 ? wSelf00 / n : 0,
      tFoodH0: n > 0 ? tFoodH0 / n : 0,
    },
    latchP50: larval.p50,
    latchP90: larval.p90,
    loneliness: larval.loneliness,
    tankLife: params && params.upkeep > 0 ? EXTRA_CAP / params.upkeep : null,
  };
}
