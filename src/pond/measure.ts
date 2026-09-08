import { CHEM_LEN, EMIT, TASTE } from '../chem-layout.ts';
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
  };
}
