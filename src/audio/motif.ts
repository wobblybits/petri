/**
 * Which connected components are a commute-mesh, and which agents of those
 * stay as full waveguides (skin) versus collapsing into a tissue resonator.
 *
 * A Lafont commute always produces the same local graph: four children and a
 * K₂,₂ of leftover wires. Repeated, that is a regular Dup/Con mesh, not an
 * arbitrary net. The interior of a settled mesh is an LTI delay network of
 * huge order whose audible output is a handful of lattice modes — so it can
 * be one resonator, as long as the cell under the cursor (and one ring of
 * neighbours) stays a real string.
 *
 * Mixed soup, pairs, and anything with an Era stay explicit. No camera means
 * no reduction: tests and headless builds want the full net.
 */

/** Same threshold the shard assigner uses: a pair is not a machine. */
export const MESH_MIN = 8;

/**
 * Closed mixed machines (the oscillator and its cousins): small enough that
 * the orbit is finite, large enough that a pair is not one, and every stem
 * is occupied so it is not still foraging.
 */
export const CLOCK_MIN = 4;
export const CLOCK_MAX = 16;

/**
 * Component roots whose agents are all Dup/Con and at least `MESH_MIN` large.
 * `kindOf` is 0 Era, 1 Dup, 2 Con.
 */
export function meshRoots(
  roots: Map<number, number>,
  kindOf: (id: number) => number,
): Set<number> {
  const groups = new Map<number, { n: number; era: boolean }>();
  for (const [id, root] of roots) {
    let g = groups.get(root);
    if (!g) {
      g = { n: 0, era: false };
      groups.set(root, g);
    }
    g.n++;
    if ((kindOf(id) | 0) === 0) g.era = true;
  }
  const out = new Set<number>();
  for (const [root, g] of groups) {
    if (g.n >= MESH_MIN && !g.era) out.add(root);
  }
  return out;
}

/**
 * Component roots that are a closed clock: mixed (has Era), fully wired,
 * and in the size band where an orbit can stay finite. Looking at any cell
 * of one of these should voice the whole machine.
 */
export function clockRoots(
  roots: Map<number, number>,
  kindOf: (id: number) => number,
  openPorts: (id: number) => number,
): Set<number> {
  const groups = new Map<number, { n: number; era: boolean; open: number }>();
  for (const [id, root] of roots) {
    let g = groups.get(root);
    if (!g) {
      g = { n: 0, era: false, open: 0 };
      groups.set(root, g);
    }
    g.n++;
    g.open += Math.max(0, openPorts(id) | 0);
    if ((kindOf(id) | 0) === 0) g.era = true;
  }
  const out = new Set<number>();
  for (const [root, g] of groups) {
    if (g.n >= CLOCK_MIN && g.n <= CLOCK_MAX && g.era && g.open === 0) out.add(root);
  }
  return out;
}

/**
 * Skin set: every agent that is not in a mesh, plus the NEAR/forced cells of
 * a mesh and one hop of neighbours. One hop, not a flood — flooding would
 * promote the whole connected component and there would be no interior left
 * to collapse.
 */
export function skinAgents(
  meshes: Set<number>,
  roots: Map<number, number>,
  seed: Iterable<number>,
  wires: { agentA: number; agentB: number }[],
): Set<number> {
  const skin = new Set<number>();
  for (const [id, root] of roots) {
    if (!meshes.has(root)) skin.add(id);
  }
  for (const id of seed) {
    const root = roots.get(id);
    if (root !== undefined && meshes.has(root)) skin.add(id);
  }
  const extra: number[] = [];
  for (const w of wires) {
    const root = roots.get(w.agentA);
    if (root === undefined || !meshes.has(root)) continue;
    if (skin.has(w.agentA) || skin.has(w.agentB)) {
      extra.push(w.agentA, w.agentB);
    }
  }
  for (const id of extra) skin.add(id);
  return skin;
}

/** Median of a nonempty list; 0 if empty. */
export function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const a = xs.slice().sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}
