import { CHEM_LEN, CRITIC_LEN, PLASTIC_LEN, STATE_DIMS, TASTE } from '../chem-layout.ts';
import { refreshReadsField } from '../agents.ts';
import { formatNet } from '../net-text.ts';
import type { Params } from '../params.ts';
import { TRAIT_KEYS } from '../rewrite.ts';
import type { Sim } from '../sim.ts';
import type { PortSlot } from '../agents.ts';
import type { NetBody, NetData, NetWire } from './net-blob.ts';

/*
 * A net, out of a live pond and back into one. A "net" is a connected
 * component of the wire graph (`componentIds`); `captureNets` skips
 * singletons by default.
 */

/** Summary numbers, so the database can be queried without opening a blob. */
export interface NetStats {
  bodies: number;
  wires: number;
  era: number;
  dup: number;
  con: number;
  /** Principal-to-principal wires, and the Con-Dup ones among them. */
  ppWires: number;
  conDupWires: number;
  /** Rewrite depth of the deepest body, and the mean. */
  bornMax: number;
  bornMean: number;
  /** Distinct founder lines present, and the largest one's share. */
  lines: number;
  dominant: number;
  dominantShare: number;
  /** Energy held in the bodies. */
  energy: number;
  /** Fraction of bodies that have learned anything at all. */
  learned: number;
  /** Mean absolute learned delta per weight; 0 for a net that has not learned. */
  plasticMean: number;
  /** Mean absolute weight over the part of the genome that seeds to zero; drift and selection alike. */
  matrixDrift: number;
}

export interface CapturedNet {
  /** The pond ids these bodies had, in blob-index order. */
  ids: number[];
  data: NetData;
  stats: NetStats;
  /** The HVM2 net IR for this component. Topology only; pose is discarded. */
  text: string;
}

export interface CaptureOptions {
  /** Components smaller than this are skipped. One body is not a net. */
  minBodies?: number;
  /** Keep only the largest `limit` components, by body count. */
  limit?: number;
}

/**
 * Every connected component of the pond, largest first. Genome and matrix
 * arrays are copied, not aliased: the store reallocates and recycles slots.
 */
export function captureNets(sim: Sim, opts: CaptureOptions = {}): CapturedNet[] {
  const minBodies = opts.minBodies ?? 2;
  const comps = sim.graph.componentIds(sim.agents, sim.rosterVersion);
  const groups = new Map<number, number[]>();
  for (const [id, root] of comps) {
    const g = groups.get(root);
    if (g) g.push(id);
    else groups.set(root, [id]);
  }

  // Wires bucketed by component: one pass over the graph, not one per component.
  const wiresBy = new Map<number, GraphWire[]>();
  for (const w of sim.graph.wires.values()) {
    const root = comps.get(w.a.id);
    if (root === undefined) continue;
    const list = wiresBy.get(root);
    const rec = { a: w.a.id, aSlot: w.a.slot, b: w.b.id, bSlot: w.b.slot };
    if (list) list.push(rec);
    else wiresBy.set(root, [rec]);
  }

  const out: CapturedNet[] = [];
  for (const [root, ids] of groups) {
    if (ids.length < minBodies) continue;
    // Sorted so a net captured twice from the same pond is byte-identical.
    ids.sort((a, b) => a - b);
    out.push(captureComponent(sim, ids, wiresBy.get(root) ?? []));
  }
  out.sort((a, b) => b.stats.bodies - a.stats.bodies || a.ids[0] - b.ids[0]);
  return opts.limit === undefined ? out : out.slice(0, opts.limit);
}

/** A graph wire flattened to ids, before body indices are known. */
interface GraphWire {
  a: number;
  aSlot: PortSlot;
  b: number;
  bSlot: PortSlot;
}

function captureComponent(sim: Sim, ids: number[], wires: GraphWire[]): CapturedNet {
  const store = sim.agentStore;
  const index = new Map(ids.map((id, i) => [id, i]));
  const bodies: NetBody[] = [];
  const lineCount = new Map<number, number>();
  let bornSum = 0;
  let bornMax = 0;
  let energy = 0;
  let learned = 0;
  let plasticSum = 0;
  let driftSum = 0;
  let era = 0;
  let dup = 0;
  let con = 0;

  for (const id of ids) {
    const a = sim.agents.get(id)!;
    const s = a.slot;
    const chem = store.chemAll.slice(s * CHEM_LEN, (s + 1) * CHEM_LEN);
    const plastic = store.plasticAll.slice(s * PLASTIC_LEN, (s + 1) * PLASTIC_LEN);
    bodies.push({
      kind: a.kind,
      x: a.x,
      y: a.y,
      heading: a.heading,
      extra: a.extra,
      requestDecay: a.requestDecay,
      energyCap: a.energyCap,
      debtCap: a.debtCap,
      rescueTo: a.rescueTo,
      assort: a.assort,
      adenylate: a.adenylate,
      born: a.born,
      lineage: a.lineage,
      chem,
      plastic,
      trace: store.traceAll.slice(s * PLASTIC_LEN, (s + 1) * PLASTIC_LEN),
      critic: store.criticAll.slice(s * CRITIC_LEN, (s + 1) * CRITIC_LEN),
      prevValue: store.prevValue[s],
      h: store.hAll.slice(s * STATE_DIMS, (s + 1) * STATE_DIMS),
    });
    if (a.kind === 'era') era++;
    else if (a.kind === 'dup') dup++;
    else con++;
    bornSum += a.born;
    if (a.born > bornMax) bornMax = a.born;
    energy += a.extra;
    lineCount.set(a.lineage, (lineCount.get(a.lineage) ?? 0) + 1);
    if (store.plasticOn[s]) learned++;
    let p = 0;
    for (let k = 0; k < PLASTIC_LEN; k++) p += Math.abs(plastic[k]);
    plasticSum += p / PLASTIC_LEN;
    // Past the taste bases is matrices and heads, zero at seed bar two entries.
    let d = 0;
    for (let k = TASTE + 4; k < CHEM_LEN; k++) d += Math.abs(chem[k]);
    driftSum += d / (CHEM_LEN - TASTE - 4);
  }

  let dominant = 0;
  let dominantCount = 0;
  for (const [line, count] of lineCount) {
    if (count > dominantCount || (count === dominantCount && line < dominant)) {
      dominant = line;
      dominantCount = count;
    }
  }

  let pp = 0;
  let conDup = 0;
  const netWires: NetWire[] = [];
  for (const w of wires) {
    const a = index.get(w.a);
    const b = index.get(w.b);
    if (a === undefined || b === undefined) continue;
    netWires.push({ a, aSlot: w.aSlot, b, bSlot: w.bSlot });
    if (w.aSlot !== 'p' || w.bSlot !== 'p') continue;
    pp++;
    const ka = bodies[a].kind;
    const kb = bodies[b].kind;
    if ((ka === 'con' && kb === 'dup') || (ka === 'dup' && kb === 'con')) conDup++;
  }

  const n = bodies.length;
  return {
    ids,
    data: { bodies, wires: netWires },
    text: formatNet(sim, ids),
    stats: {
      bodies: n,
      wires: netWires.length,
      era,
      dup,
      con,
      ppWires: pp,
      conDupWires: conDup,
      bornMax,
      bornMean: bornSum / n,
      lines: lineCount.size,
      dominant,
      dominantShare: dominantCount / n,
      energy,
      learned: learned / n,
      plasticMean: plasticSum / n,
      matrixDrift: driftSum / n,
    },
  };
}

export interface PlantOptions {
  /**
   * Overwrite every body's founder line with this: a stored net's `lineage`
   * values are ids from another pond. The runner passes a negative number per
   * planted net; ids are positive, so a negative line means "from the database".
   */
  lineage?: number;
  /** Rotate the whole net about its centroid before placing it. */
  heading?: number;
}

/**
 * Drop a stored net into a pond, centred on `(x, y)`. Returns the new ids in
 * blob-index order, or an empty array if the whole net does not fit under
 * `params.maxAgents`; a truncated net is a torn piece, not the net. Wires are
 * `silent` so the audio queue does not hear fifty latches.
 */
export function plantNet(
  sim: Sim,
  params: Params,
  net: NetData,
  x: number,
  y: number,
  opts: PlantOptions = {},
): number[] {
  const n = net.bodies.length;
  if (n === 0) return [];
  if (!sim.canSpawn(params, n)) return [];
  let cx = 0;
  let cy = 0;
  for (const b of net.bodies) {
    cx += b.x;
    cy += b.y;
  }
  cx /= n;
  cy /= n;
  const rot = opts.heading ?? 0;
  const cos = Math.cos(rot);
  const sin = Math.sin(rot);

  const store = sim.agentStore;
  const ids: number[] = [];
  for (const b of net.bodies) {
    const dx = b.x - cx;
    const dy = b.y - cy;
    // Forced, because the cap was already checked for the whole net above.
    const a = sim.spawn(b.kind, x + dx * cos - dy * sin, y + dx * sin + dy * cos, b.heading + rot, params, true);
    // If `spawn` refuses anyway, the net comes back out rather than going in half.
    if (!a) break;
    ids.push(a.id);
    a.vx = 0;
    a.vy = 0;
    a.omega = 0;
    a.prevX = a.x;
    a.prevY = a.y;
    a.prevHeading = a.heading;
    a.drive = 0;
    a.extra = b.extra;
    // The list itself, so a trait added to `TRAIT_KEYS` cannot be missed here.
    for (const k of TRAIT_KEYS) a[k] = b[k];
    a.born = b.born;
    a.lineage = opts.lineage ?? b.lineage;

    const s = a.slot;
    store.chemAll.set(b.chem, s * CHEM_LEN);
    store.plasticAll.set(b.plastic, s * PLASTIC_LEN);
    store.traceAll.set(b.trace, s * PLASTIC_LEN);
    store.criticAll.set(b.critic, s * CRITIC_LEN);
    store.prevValue[s] = b.prevValue;
    store.hAll.set(b.h, s * STATE_DIMS);
    // Set by hand: the learning pass only sets it when a weight first moves,
    // and a body arriving with a delta would otherwise run its bare genome.
    let on = 0;
    for (let k = 0; k < PLASTIC_LEN && !on; k++) if (b.plastic[k] !== 0) on = 1;
    store.plasticOn[s] = on;
    store.markLearn(s);
    // Every `chem` write passes through this: it refreshes the sense gate and stamps the GPU genome stale.
    refreshReadsField(a);
  }

  if (ids.length !== n) {
    for (const id of ids) sim.kill(id);
    return [];
  }

  for (const w of net.wires) {
    const a = { id: ids[w.a], slot: w.aSlot };
    const b = { id: ids[w.b], slot: w.bSlot };
    if (!sim.graph.isFreeAt(a.id, a.slot) || !sim.graph.isFreeAt(b.id, b.slot)) continue;
    sim.graph.connect(sim.agents, a, b, sim.w, sim.h, params, sim.time, { silent: true });
  }
  return ids;
}
