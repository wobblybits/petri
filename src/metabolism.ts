import { AgentStore } from './agent-store.ts';
import { type Agent, GAIT_ANCHOR_MAX } from './agents.ts';
import { EnergyGrid, payOut } from './energy.ts';
import { Graph } from './graph.ts';
import { type Params } from './params.ts';

/** Most either metabolite may pile up to. A pathway is not a warehouse. */
const REACT_CAP = 12;
/**
 * Largest reaction step taken at once. The pathway is stiff where its
 * activator spikes, and explicit Euler past about this rings at the step
 * frequency instead of oscillating — so `advanceGait` splits a frame into as
 * many of these as it needs, which makes the amplitude the same at every
 * `metabolicRate` and the period exactly proportional to it.
 */
const REACT_H = 0.04;

/** Scratch for one diffusion step: the pull on each body, and its degree. */
let gaitPull = new Float64Array(0);
let gaitDegree = new Float64Array(0);

/** Rate steps the drag table divides itself into. See `dampVelocities`. */
const GRIP_STEPS = 256;

/** Per-frame `exp(-rate * dt)` by total rate, rebuilt only when it varies. */
const gripKeep = new Float64Array(GRIP_STEPS + 1);

/** What the metabolism and drag passes read off the simulation. */
export interface MetabolismHost {
  agentStore: AgentStore;
  agents: Map<number, Agent>;
  energy: EnergyGrid;
  graph: Graph;
}

/**
 * One step of the body's metabolism, and what this frame's stroke comes to.
 *
 * Three reactions over two pools, and an adenylate pool that is conserved:
 *
 *     supply   extra   -> sub        rate  metabolicSupply * (1 - charge),
 *                                        priced at metabolicCost a unit
 *     burn     sub, ATP -> ADP       rate  sub * (base + adp^2)
 *     regen    ADP     -> ATP        rate  metabolicRegen * adp
 *     work     ATP     -> ADP        rate  metabolicWork * swell * |wave|
 *
 * `atp + adp` is the body's `adenylate`, and nothing here changes it — the
 * pool is currency and can only be cycled, never minted. What *is* spent is
 * `extra`, bought as substrate, and `supply` is where this pathway joins
 * the economy rather than shadowing it.
 *
 * The regulation is the real one. A cell does not pull harder on its fuel
 * because it is holding a lot; it pulls harder because it is *discharged*.
 * So supply runs on `1 - charge`, which means a net that is working draws
 * its tank down, a drawn-down tank is what `spreadRequests` carries, and
 * `flowCharges` answers it from wherever the net has surplus. Two wired
 * bodies are coupled through that whether or not `metabolicDiffuse` is set.
 *
 * `burn` is the autocatalysis, and it is the same one glycolysis uses:
 * phosphofructokinase is activated by the ADP it produces, so a discharged
 * body burns faster, discharges further, and burns faster still — until the
 * substrate runs out and `regen` catches up. That is the oscillation, and
 * it is a *consequence* of the regulation rather than a clock beside it.
 *
 * `base` is not decoration. Without it the burn is `sub * adp^2`, which is
 * zero at full charge, so a body that ever tops up stops metabolising for
 * good — zero is absorbing. That is exactly why Selkov's own equations
 * flatlined here at every fullness. Real phosphofructokinase turns over
 * without ADP too; the activation is a multiplier on an enzyme that is
 * already running.
 *
 * What leaves the tank lands on the ground, through the same `excreteRate`
 * path `tickUpkeepFast` already uses for rent. So a body that metabolises
 * hard fertilises the cell it stands in, nothing is destroyed, and the
 * conservation the economy is checked against still holds.
 *
 * The wave is `(atp - adp) / adenylate`, which is the adenylate energy
 * charge mapped to [-1, 1]: +1 fully charged, -1 fully spent, 0 half. No
 * arbitrary normalisation, and a body whose pathway has stalled sits at
 * whatever charge it stalled at and does not stroke.
 */
export function advanceMetabolism(host: MetabolismHost, params: Params, dt: number): void {
  const store = host.agentStore;
  const SUB = store.sub;
  const ATP = store.atp;
  const POOL = store.adenylate;
  const WAVE = store.gaitWave;
  const GA = store.gaitAnchor;
  const ANCHOR = store.anchor;
  const EXTRA = store.extra;
  const rate = params.metabolicRate;
  if (!(rate > 0)) {
    for (const agent of host.agents.values()) {
      ANCHOR[agent.slot] = 0;
      WAVE[agent.slot] = 0;
    }
    return;
  }

  const supply = params.metabolicSupply;
  const base = params.metabolicBase;
  const regen = params.metabolicRegen;
  const workRate = params.metabolicWork * params.gaitSwell;
  const price = params.metabolicCost;
  /*
   * `upkeepExcrete`, and not `excreteRate`, which is what this read for four
   * commits and is the whole of why `energy.test.ts`'s pond conservation
   * broke: the spend below is rent by another name, and rent's dial is
   * `upkeepExcrete`. `excreteRate` is the reaction table's, a rate in its
   * own units rather than a fraction — so at its default of zero the
   * pathway's spend was destroyed, and above one it would have minted.
   */
  const back = params.upkeepExcrete;
  const grid = host.energy;
  const h = rate * dt;
  const sub = Math.max(1, Math.ceil(h / REACT_H));
  const hs = h / sub;

  for (const agent of host.agents.values()) {
    const s = agent.slot;
    const pool = POOL[s];
    if (!(pool > 0)) {
      WAVE[s] = 0;
      ANCHOR[s] = 0;
      continue;
    }
    let a = ATP[s];
    let m = SUB[s];
    let spent = 0;
    for (let k = 0; k < sub; k++) {
      const d = pool - a;
      const charge = a / pool;
      // Bought, not conjured: a body with an empty tank cannot buy
      // substrate and its pathway winds down.
      // Priced, not free, and bounded by what the tank can actually pay for
      // — a body with nothing left buys nothing and its pathway winds down.
      const want = supply * (1 - charge) * hs;
      const have = EXTRA[s] - spent;
      const afford = price > 0 ? have / price : want;
      const buy = want <= 0 || afford <= 0 ? 0 : want > afford ? afford : want;
      spent += buy * price;
      const burn = m * (base + d * d);
      const back = regen * d;
      const work = workRate * (a >= d ? a - d : d - a) / pool;
      m = m + buy - burn * hs;
      a = a + (back - burn - work) * hs;
      if (m < 0) m = 0;
      else if (m > REACT_CAP) m = REACT_CAP;
      if (a < 0) a = 0;
      else if (a > pool) a = pool;
    }
    SUB[s] = m;
    ATP[s] = a;
    if (spent > 0) {
      EXTRA[s] -= spent;
      // The same road out as rent, and the same mix: what left the tank
      // lands on the dish as whatever this body makes, so metabolising is
      // fertilising and the pond's total is unchanged. This pass runs before
      // `refreshExpression`, so the rows are last frame's — one frame stale,
      // which is what the harvest plan lives with too.
      if (back > 0) payOut(grid, store.expressAll, s, store.x[s], store.y[s], spent * back);
    }
    const w = (2 * a - pool) / pool;
    WAVE[s] = w;
    ANCHOR[s] = GA[s] * w;
  }

  const spread = params.metabolicDiffuse;
  if (spread > 0) {
    if (gaitPull.length < store.capacity) {
      gaitPull = new Float64Array(store.capacity);
      gaitDegree = new Float64Array(store.capacity);
    }
    const pull = gaitPull;
    const count = gaitDegree;
    for (const agent of host.agents.values()) {
      pull[agent.slot] = 0;
      count[agent.slot] = 0;
    }
    // Accumulated before it is applied, so every wire sees the same field
    // and the answer does not depend on the order the wire map is in.
    for (const wire of host.graph.wires.values()) {
      const A = host.agents.get(wire.a.id);
      const B = host.agents.get(wire.b.id);
      if (!A || !B) continue;
      const d = SUB[B.slot] - SUB[A.slot];
      pull[A.slot] += d;
      pull[B.slot] -= d;
      count[A.slot] += 1;
      count[B.slot] += 1;
    }
    // Per wire, so a hub is not driven as many times as it has wires, and
    // capped at a half so an explicit step cannot overshoot and ring.
    const step = Math.min(0.5, spread * dt);
    for (const agent of host.agents.values()) {
      const s = agent.slot;
      const n = count[s];
      if (n <= 0) continue;
      const next = SUB[s] + step * (pull[s] / n);
      SUB[s] = next <= 0 ? 0 : next >= REACT_CAP ? REACT_CAP : next;
    }
  }
}

/**
 * One drag law for bodies; the rope is damped inside the substep loop.
 *
 * A body's rate is `drag + grip * fullness + anchor`. `grip` makes it
 * depend on the tank, which is what lets a net's internal pumping carry it
 * anywhere — see the parameter, which has the argument and the arithmetic.
 * `anchor` is this frame's point in the gait, and is what turns a standing
 * asymmetry into a stroke; see `advanceGait`.
 *
 * The exponential is read off a table rather than taken per body.
 * `Math.exp` is by an order of magnitude the most expensive thing in this
 * loop and `dt` is the same for everyone, so 256 of them serve fifty
 * thousand bodies. The table is over the *total rate* rather than over
 * fullness, which it used to be: with a gait the rate is no longer a
 * function of one bounded scalar, so there is nothing narrower to key on.
 * 256 steps rather than 64 because the span is now several times wider and
 * the resolution should not go backwards — at the shipping dials it is
 * 0.057/s a step against 0.031 before.
 *
 * Interpolated rather than snapped: two bodies a hair apart in rate should
 * stay a hair apart in drag, and a step function there would quietly sort
 * the pond into 256 kinds. With `grip` and `gaitRate` both at 0 the loop
 * below takes the flat path and never builds the table — a dial at its
 * off value should cost nothing.
 */
export function dampVelocities(host: MetabolismHost, params: Params, dt: number): void {
  const angKeep = Math.exp(-Math.max(0, params.angDrag) * dt);
  const store = host.agentStore;
  const LOCKED = store.locked;
  const PINNED = store.pinned;
  const VX = store.vx;
  const VY = store.vy;
  const OMEGA = store.omega;
  const base = Math.max(0, params.drag);
  const grip = params.grip;
  const gait = params.metabolicRate > 0;
  if (grip === 0 && !gait) {
    const linKeep = Math.exp(-base * dt);
    for (const agent of host.agents.values()) {
      const s = agent.slot;
      if (LOCKED[s] || PINNED[s]) continue;
      VX[s] *= linKeep;
      VY[s] *= linKeep;
      OMEGA[s] *= angKeep;
    }
    return;
  }
  const steps = GRIP_STEPS;
  const table = gripKeep;
  // The ceiling the table spans. Both terms are bounded — `grip` by its
  // slider and `anchor` by the head's own clamp — so this is the largest
  // rate any body can ask for, and anything past it saturates rather than
  // reading off the end.
  const hi = base + Math.max(0, grip) + (gait ? GAIT_ANCHOR_MAX : 0);
  const span = hi > 1e-9 ? hi : 1;
  for (let i = 0; i <= steps; i++) {
    // Built over a rate that is already non-negative: a negative total is
    // clamped at the lookup instead, because `exp` of one silently
    // amplifies rather than failing.
    table[i] = Math.exp(-((span * i) / steps) * dt);
  }
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  const ANCHOR = store.anchor;
  for (const agent of host.agents.values()) {
    const s = agent.slot;
    if (LOCKED[s] || PINNED[s]) continue;
    // The same clamp the genome's `IN_FULL` gets, so grip and the body's own
    // sense of how full it is never disagree.
    const cap = CAP[s];
    const raw = cap > 0 ? EXTRA[s] / cap : 0;
    const full = raw <= 0 ? 0 : raw >= 1 ? 1 : raw;
    const rate = base + grip * full + ANCHOR[s];
    const r = rate <= 0 ? 0 : rate >= span ? span : rate;
    const u = (r / span) * steps;
    const i = u < steps ? u | 0 : steps - 1;
    const keep = table[i] + (table[i + 1] - table[i]) * (u - i);
    VX[s] *= keep;
    VY[s] *= keep;
    OMEGA[s] *= angKeep;
  }
}
