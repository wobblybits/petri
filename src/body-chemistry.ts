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
 * Its own pass because four things read it and they are gated on three
 * different dials: `runDigestion`'s uptake rows (`uptakeVmax`),
 * `runExcretion`'s excretion rows (`excreteRate`), and rent's `payOut` and
 * `upkeepRateOf` (`upkeepExcrete`). It was computed inside `runExcretion`
 * at first, which meant a pond with metered uptake and no excretion read an
 * all-zero expression and could digest nothing it swallowed.
 *
 * Computed on the host on both field paths — it is a pure function of `chem`
 * and `h`, and `unpackGenome` brings `h` back every frame, so the genome
 * shader needs no new output slot and no new binding for any of the reaction
 * table. That pass is already at eight storage buffers of a guaranteed eight.
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
     * The fixed cost of running a reaction at all — §3's first way to buy
     * the superadditivity division of labour needs, and which a linear
     * budget cannot supply. Running two rows costs `2c` and running one
     * costs `c`, so a specialist keeps what a generalist spends on breadth.
     *
     * Counted on rows that are *expressed*, which relu makes an exact
     * question: a pre-activation at or below zero is a row switched off, and
     * driving one there is how a lineage specialises. A body whose `X` is
     * all zero takes `expressVector`'s flat fallback and pays for all eight
     * — it is expressing evenly, not expressing nothing, and charging it
     * nothing would make "say nothing, act as a generalist" free and
     * strictly best at any cost. The seed is not that body: `seedProduction`
     * gives it the rows it makes and the four it eats with, so a seeded Con
     * or Dup pays for six of eight and a seeded Era for five, which is the
     * breadth each of them actually runs.
     *
     * Excreted on the same terms as upkeep, and for the same reason: what
     * left a tank has to arrive somewhere or a dial nobody turned on is
     * quietly destroying matter.
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
 * The harvest is a *sample of the water* — a body cannot decline the part of
 * the mixture it has no use for — so a body is necessarily holding species
 * it may not be able to touch. This is where that is settled, and it is the
 * whole reason the gut exists: waste is the gap between the sample and the
 * recipe, and nothing has to nominate it.
 *
 * Three rules, and each is one line:
 *
 * - **The ground converts raw.** `CH.energy` needs no machinery and no
 *   permission, which is what makes it the ground rather than a signal.
 * - **The other three need the row.** A body's uptake row for a species is
 *   its recipe for that species — `ROW_COUNT` in the factor so a row at an
 *   eighth, which is what every seeded uptake row is, converts at exactly
 *   the global rate, and clamped at one because a recipe is a capability and
 *   not an amplifier. A body expressing nothing on a row converts none of
 *   that species and holds it until excretion takes it away.
 * - **Catabolism spends ground.** `catCoSubstrate` is how many units of
 *   `CH.energy` one unit of another species is converted *with*, drawn from
 *   the gut and banked with what it unlocked — spent as a licence, not
 *   destroyed. A reagent, not a catalyst: it is a budget shared
 *   across the three rows rather than a factor on their rate, so a body with
 *   a little ground has to choose what to spend it on and one unit cannot
 *   unlock everything. Continuous from zero, so a body with a little
 *   capability still does better than one with none and selection has a
 *   slope to climb rather than a cliff. Nothing is destroyed — the paired
 *   ground lands in the tank alongside what it unlocked — so it still buys
 *   access and never amplification.
 *
 * Bounded by room in the tank, which is what makes satiety three mechanisms
 * deep rather than a clamp: a full body cannot digest, so its gut fills, so
 * it cannot eat. And bounded that way rather than by discarding, because
 * matter that had nowhere to go would have to be destroyed to fit.
 */
export function runDigestion(host: ChemistryHost, params: Params, t: number, gutLive: boolean): boolean {
  const rate = params.digestRate;
  // Nothing has ever been swallowed, which is every frame of a pond running
  // at `uptakeVmax` 0 — the whole of the shipped default. Guarded rather
  // than walked, for the reason `HarvestPlan.build` guards on `meter`: a
  // per-body pass that always finds zero is still a per-body pass.
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
     * The ground is a *reagent*, not a catalyst.
     *
     * §6b says the other three are converted *with* `CH.energy`, and a
     * factor that only scales a rate says "in the presence of", which is a
     * catalyst — one unit of ground in the gut licensed unlimited scent, and
     * converting scent is how a body keeps a unit of ground. So the ground
     * is spent here: `co` units of it per unit of species converted, drawn
     * from what this body is holding and unable to license a second thing.
     *
     * `pair` is that budget and it is shared across the three rows, which is
     * the constraint the ratio could not express: a body with a little
     * ground must choose what to spend it on. At `co` 0 it is unbounded and
     * the ground is not drawn at all, which is what phase 3 shipped.
     *
     * Nothing is destroyed — the paired ground lands in the tank alongside
     * what it unlocked, exactly as it would have on its own row. What it
     * cannot do is unlock a second thing. Conservation is untouched and it
     * still buys access rather than amplification.
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
       * Mass action, as an exponential rather than a product, so a rate
       * above one frame's worth cannot take more than the body is holding.
       * `full` is the ground's factor and every row at or above the flat
       * seed's, hoisted; only a row with a partial recipe pays for an `exp`.
       *
       * Snapped to empty below `EXTRA_FULL_EPS`, because mass action never
       * reaches zero on its own and `gutLive` would then never clear: a gut
       * holding a crumb forever is a roster walk forever. The crumb lands in
       * the tank like the rest of it; nothing is destroyed.
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
  /*
   * Read before this pass moved anything, so a pond that has just finished
   * digesting the last of it runs one more empty pass and then stops. The
   * error is one frame in the safe direction; the other direction would
   * leave a gut that nothing ever drained. The snap above is what makes
   * "finished" a state a gut can actually reach.
   */
  return live;
}

/**
 * The reaction table's excretion rows: gut and tank -> field, conserved.
 *
 * Mass action on everything the body holds, gut and tank together, split by
 * the excretion rows — which is where "absolute honesty" comes from. A body
 * near empty excretes near nothing however loudly its genome would like to:
 * the simplex bounds what you can say relative to what else you say, and
 * conservation bounds it outright. Neither implies the other and they
 * compound.
 *
 * Two sources, one operation. A body wanting to put out species `c` takes it
 * from the gut if it is holding any — it is already that species, and it is
 * in the way — and synthesises the shortfall out of the tank, which is what
 * this has always done and is where the three signal species come from at
 * all. Mass is conserved either way; what the gut buys is that clearing
 * waste is *free*, where saying the same thing out of stock costs.
 *
 * That the tank half survives is not a compromise. Per-species conservation
 * is a stronger property than §5 asks for and it is one the pond cannot
 * afford: nothing in a conserved dish creates `conP`, so a pond whose only
 * input is ground would be permanently silent. Bodies conserve *matter*.
 * Turning matter into a different molecule is what a metabolism is.
 *
 * What the gut half adds is the necessity. What a body cannot convert
 * occupies the room that bounds its next mouthful, and the only way out of
 * the gut is here — so a body must express the excretion row for whatever it
 * cannot digest, or clog and starve holding food it cannot use. The rows
 * stop being purely a preference about what to broadcast and become, for
 * that species, a clearance rate. Nothing declares what a body's waste is;
 * the shape of what it is stuck with does.
 *
 * `ROW_COUNT` in the rate so that a row at an eighth — the flat fallback,
 * and what every seeded *uptake* row still is — runs at exactly
 * `excreteRate`; a seeded Era's ground row at `ERA_GROUND_SHARE` runs at
 * four times it. Every dial in this file reduces to a number you can say out
 * loud at the seed.
 *
 * Through `addSpeciesAt` and therefore through the *conserving* deposit, not
 * the scent path's density scatter. That is the whole difference: a signal
 * is a density that may be clipped at the rim, and this is matter that
 * cannot be. `scentMints` is what stops the two running at once.
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
    /*
     * A long frame, or a rate above one, can ask for more stock than the
     * body has. The synthesised part scales down and the gut part does not:
     * clearing what is already in the way is not something poverty should be
     * able to prevent, and it is the half a clogged body most needs.
     */
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
 * Whether the scent path still mints.
 *
 * The two ways matter reaches the field are mutually exclusive by
 * construction: either a body's voice is minted at `params.deposit` and its
 * tank is untouched, or it is excreted from the tank and conserved. Running
 * both would deposit the same expression twice and mint half of it.
 */
export function scentMints(params: Params): boolean {
  return !(params.excreteRate > 0);
}
