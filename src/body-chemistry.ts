import { AgentStore } from './agent-store.ts';
import { type Agent, CHEM_LEN, CHEM_SPECIES, ROW_COUNT, ROW_EXCRETE, ROW_UPTAKE, STATE_DIMS, expressVector } from './agents.ts';
import { EXTRA_FULL_EPS, EnergyGrid, payOut } from './energy.ts';
import { CH, DIGEST_ORDER } from './fields.ts';
import { type Params } from './params.ts';

/** Four species' worth of excretion, per body, per frame. */
const excreteScratch = new Float64Array(CHEM_SPECIES);

/** What the chemistry passes read off the simulation. */
export interface ChemistryHost {
  agentStore: AgentStore;
  agents: Map<number, Agent>;
  energy: EnergyGrid;
}

/**
 * Materialise every body's expression vector for the frame.
 *
 * Its own pass because `runDigestion`, `runExcretion` and rent all read it
 * and are gated on different dials. Computed on the host on both field
 * paths: a pure function of `chem` and `h`, and `unpackGenome` brings `h`
 * back every frame.
 */
export function refreshExpression(host: ChemistryHost, params: Params, t: number): boolean {
  const expressed = params.excreteRate > 0 || params.uptakeVmax > 0 || params.upkeepExcrete > 0;
  if (!expressed) return false;
  const store = host.agentStore;
  const CHEM = store.chemAll;
  const H = store.hAll;
  const EX = store.expressAll;
  const perRow = params.rowCost * t;
  const back = params.upkeepExcrete;
  for (const a of host.agents.values()) {
    const s = a.slot;
    const o = s * ROW_COUNT;
    expressVector(CHEM, s * CHEM_LEN, H, s * STATE_DIMS, EX, o);
    if (!(perRow > 0) || a.locked) continue;
    /*
     * The fixed cost of running a reaction at all: two rows cost `2c`, one
     * costs `c`, so a specialist keeps what a generalist spends on breadth.
     * Counted on rows that are expressed, which relu makes exact; a body
     * whose `X` is all zero takes `expressVector`'s flat fallback and pays
     * for all eight. Excreted on the same terms as upkeep, so nothing is
     * destroyed.
     */
    let rows = 0;
    for (let r = 0; r < ROW_COUNT; r++) if (EX[o + r] > 0) rows++;
    if (rows === 0) continue;
    const was = a.extra;
    a.extra = Math.max(a.debtCap, was - perRow * rows);
    if (back > 0) {
      const paid = Math.max(0, was) - Math.max(0, a.extra);
      if (paid > 0) payOut(host.energy, EX, s, a.x, a.y, paid * back);
    }
  }
  return true;
}

/**
 * The gut: what a body swallowed becomes what a body has, or does not.
 *
 * The harvest is a sample of the water, so a body holds species it may not
 * be able to touch; waste is the gap between the sample and the recipe.
 *
 * - The ground (`CH.energy`) converts raw.
 * - The other three need the uptake row: `ROW_COUNT` in the factor so a
 *   row at an eighth, every seeded uptake row, converts at the global rate,
 *   clamped at one because a recipe is a capability, not an amplifier.
 * - Catabolism spends ground: `catCoSubstrate` units of `CH.energy` per
 *   unit of another species, drawn from the gut and banked with what it
 *   unlocked. A budget shared across the three rows, not a rate factor.
 *
 * Bounded by room in the tank rather than by discarding: a full body cannot
 * digest, so its gut fills, so it cannot eat. Nothing is destroyed.
 */
export function runDigestion(host: ChemistryHost, params: Params, t: number, gutLive: boolean): boolean {
  const rate = params.digestRate;
  // Nothing swallowed is every frame at `uptakeVmax` 0; guarded rather than
  // walked.
  if (!(rate > 0) || !gutLive) return gutLive;
  const store = host.agentStore;
  const GUT = store.gut;
  const EX = store.expressAll;
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  const co = params.catCoSubstrate;
  const full = 1 - Math.exp(-rate * t);
  let live = false;
  for (const a of host.agents.values()) {
    const s = a.slot;
    const go = s * CHEM_SPECIES;
    if (store.gutTotal(s) <= 0) continue;
    live = true;
    let room = CAP[s] - EXTRA[s];
    if (room <= 0) continue;
    const xo = s * ROW_COUNT + ROW_UPTAKE;
    /*
     * The ground is a reagent, not a catalyst: `co` units of it are spent
     * per unit of species converted, and `pair` is that budget shared across
     * the three rows. At `co` 0 it is unbounded and the ground is not drawn.
     */
    let pair = co > 0 ? GUT[go + CH.energy] : Infinity;
    let moved = 0;
    // The three signalling species, then the ground: see `DIGEST_ORDER` for
    // why the co-substrate has to be paired off before it is digested.
    for (let k = 0; k < DIGEST_ORDER.length && room > 0; k++) {
      const c = DIGEST_ORDER[k];
      const have = GUT[go + c];
      if (have <= 0) continue;
      const ground = c === CH.energy;
      let use = 1;
      if (!ground) {
        const row = ROW_COUNT * EX[xo + c];
        use = row > 1 ? 1 : row;
      }
      if (!(use > 0)) continue;
      /*
       * Mass action as an exponential, so a rate above one frame's worth
       * cannot take more than the body holds. Snapped to empty below
       * `EXTRA_FULL_EPS`, or mass action never reaches zero and `gutLive`
       * never clears; the crumb lands in the tank, nothing is destroyed.
       */
      let take = use >= 1 ? have * full : have * (1 - Math.exp(-rate * use * t));
      if (have - take < EXTRA_FULL_EPS) take = have;
      // What it costs in ground, and what is left to pay with.
      let spend = 0;
      if (!ground && co > 0) {
        if (take * co > pair) take = pair / co;
        spend = take * co;
      }
      if (take + spend > room) {
        const k2 = room / (take + spend);
        take *= k2;
        spend *= k2;
      }
      if (!(take > 0)) continue;
      GUT[go + c] = have - take;
      if (spend > 0) {
        GUT[go + CH.energy] -= spend;
        pair -= spend;
      }
      room -= take + spend;
      moved += take + spend;
    }
    if (moved > 0) EXTRA[s] += moved;
  }
  // Read before this pass moved anything, so a pond that has just digested
  // the last of it runs one more empty pass and then stops: one frame in
  // the safe direction.
  return live;
}

/**
 * The reaction table's excretion rows: gut and tank -> field, conserved.
 *
 * Mass action on everything the body holds, gut and tank together, split by
 * the excretion rows: the simplex bounds what a body can say relative to
 * what else it says, and conservation bounds it outright. Species `c` is
 * taken from the gut if the body holds any and the shortfall synthesised
 * out of the tank — matter is conserved, not species, since nothing in a
 * conserved dish creates `conP`. The only way out of the gut is here, so a
 * body must express the excretion row for what it cannot digest or clog.
 *
 * `ROW_COUNT` in the rate so a row at an eighth, the flat fallback, runs at
 * exactly `excreteRate`. Through `addSpeciesAt`, the conserving deposit,
 * not the scent path's density scatter; `scentMints` keeps the two from
 * running at once.
 */
export function runExcretion(host: ChemistryHost, params: Params, t: number): void {
  if (!(params.excreteRate > 0)) return;
  const store = host.agentStore;
  const EX = store.expressAll;
  const OUT = store.excreteAll;
  const GUT = store.gut;
  const EXTRA = store.extra;
  const LOCKED = store.locked;
  const X = store.x;
  const Y = store.y;
  const w = excreteScratch;
  const rate = params.excreteRate * t * ROW_COUNT;
  for (const a of host.agents.values()) {
    const s = a.slot;
    const eo = s * ROW_COUNT + ROW_EXCRETE;
    const oo = s * CHEM_SPECIES;
    const go = s * CHEM_SPECIES;
    // Mass action on everything the body is holding, digested or not: a body
    // with more to give gives more, and one with nothing gives nothing.
    const banked = EXTRA[s] > 0 ? EXTRA[s] : 0;
    const have = banked + store.gutTotal(s);
    if (LOCKED[s] || have <= 0) {
      OUT.fill(0, oo, oo + CHEM_SPECIES);
      continue;
    }
    let owed = 0;
    for (let c = 0; c < CHEM_SPECIES; c++) {
      const amt = rate * EX[eo + c] * have;
      w[c] = amt;
      // What the gut cannot cover has to be built out of stock.
      const gut = GUT[go + c];
      owed += amt > gut ? amt - gut : 0;
    }
    // A long frame, or a rate above one, can ask for more stock than the
    // body has: the synthesised part scales down and the gut part does not,
    // since clearing what is in the way is not something poverty prevents.
    const k = owed > banked ? (owed > 0 ? banked / owed : 0) : 1;
    let total = 0;
    let spent = 0;
    for (let c = 0; c < CHEM_SPECIES; c++) {
      const gut = GUT[go + c];
      const amt = w[c];
      const fromGut = amt > gut ? gut : amt;
      const made = (amt - fromGut) * k;
      const out = fromGut + made;
      w[c] = out;
      if (fromGut > 0) GUT[go + c] = gut - fromGut;
      spent += made;
      total += out;
    }
    if (spent > 0) EXTRA[s] -= spent;
    if (total <= 0) {
      OUT.fill(0, oo, oo + CHEM_SPECIES);
      continue;
    }
    for (let c = 0; c < CHEM_SPECIES; c++) OUT[oo + c] = w[c];
    host.energy.addSpeciesAt(X[s], Y[s], w);
  }
}

/**
 * Whether the scent path still mints. Either a body's voice is minted at
 * `params.deposit` and its tank untouched, or it is excreted from the tank
 * and conserved; running both would deposit the expression twice.
 */
export function scentMints(params: Params): boolean {
  return !(params.excreteRate > 0);
}
