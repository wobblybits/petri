/**
 * Where a body's genome is laid out in memory.
 *
 * Imports nothing, so `AgentStore` (which allocates one body's `chem`) and
 * `agents.ts` (which reads it) can both depend on it without a cycle. Every
 * offset is derived from the one before it rather than written down, and
 * `genome.wgsl` transcribes the offsets; adding a dimension or a head moves
 * everything that has to move.
 */

/**
 * The width of `h`, the recurrent state.
 *
 * Its dimensions have no names. `phi` keeps every one inside (-1, 1), which
 * is what makes `CHEM_SLOPE_MAX` cap anything. `h[0]` is spoken for at seed:
 * it carries `IN_DEMAND` and the ground's taste weight reads it; a second
 * seeded pathway on dim 0 would collide silently. The gait's clock is not in
 * `h` (a rotation in `Wh` collapses under any constant input); `gaitPhase` is
 * a register on the body and `G_OUT` supplies its amplitudes.
 */
export const STATE_DIMS = 4;

/**
 * The input vector `x`: four raw channel readings at this body's position,
 * then three facts about the body itself. `IN_DEMAND` is the neighbourhood's
 * unmet need, spread along the wires and decayed per hop; `IN_FULL` is this
 * body's own tank; `IN_BOUND` is how much of it is attached. `IN_DEMAND` is
 * an input and not part of `h` so that a genome cannot lie to `flowCharges`
 * about what it is short of. All seven are scaled to roughly the same range
 * before they are read; see `SENSE_SCALE`.
 */
export const IN_SENSE = 0;
export const IN_FULL = 4;
export const IN_BOUND = 5;
export const IN_DEMAND = 6;
export const IN_DIMS = 7;

/**
 * What one unit of a signal channel reading is worth, on the way into `x`.
 * The other inputs are about [0,1]; this brings the signal channels onto the
 * same scale so one `CHEM_MUTATE` step means the same through every input
 * weight (the output-side counterpart is `HEAD_SCALE`).
 */
export const SENSE_SCALE = 4.3;

/**
 * `chem` layout.
 *
 * A body is a very small recurrent network over the wire graph:
 *
 *     x  = [ sense(4) , FULL , BOUND , DEMAND ]              in R^7
 *     h <- phi( Wx.x + Wh.h + Wn.mean(h of wired neighbours) + b )   in R^4
 *     emit  = normalise(relu( E.h + e0 ))                    in R^4
 *     taste = T.h + t0                                       in R^4
 *
 * `e0` and `t0` are the named, kind-seeded part; every matrix seeds to zero,
 * so a fresh body computes `h = phi(0) = 0` and behaves as its bases say.
 */
export const EMIT = 0;
export const TASTE = 4;
/** `E`, state -> emit. Row-major by channel: `E_OUT + c * STATE_DIMS + d`. */
export const E_OUT = 8;
/** `T`, state -> taste. Same shape. */
export const T_OUT = E_OUT + 4 * STATE_DIMS;
/** `Wx`, input -> state. Row-major by state dim: `W_IN + d * IN_DIMS + k`. */
export const W_IN = T_OUT + 4 * STATE_DIMS;
/** `Wh`, state -> state. This body's own memory. */
export const W_SELF = W_IN + STATE_DIMS * IN_DIMS;
/** `Wn`, mean neighbour state -> state. Its diagonal is a learned per-hop decay. */
export const W_NET = W_SELF + STATE_DIMS * STATE_DIMS;
/** `b`, the state bias. */
export const B_STATE = W_NET + STATE_DIMS * STATE_DIMS;
/**
 * `F`, state -> flocking, and its base. Two rows: align, then separate.
 * The bases seed from the sliders and the matrices from zero. The store's
 * field is the phenotype, rewritten each frame by `updateState`.
 */
export const F_OUT = B_STATE + STATE_DIMS;
export const F_BASE = F_OUT + 2 * STATE_DIMS;
/** `P`, state -> transport. One row: recoil. */
export const P_OUT = F_BASE + 2;
export const P_BASE = P_OUT + STATE_DIMS;
/** `L`, state -> locomotion. Two rows: cruise, then turn. Per-body variation over the global `stepSpeed` and `turnRate`. */
export const L_OUT = P_BASE + 1;
export const L_BASE = L_OUT + 2 * STATE_DIMS;

/**
 * The body reaction table's rows: `excrete_c` and `uptake_c` for each of the
 * four species. Derived from the species count rather than written down.
 */
export const CHEM_SPECIES = 4;
export const ROW_EXCRETE = 0;
export const ROW_UPTAKE = CHEM_SPECIES;
export const ROW_COUNT = 2 * CHEM_SPECIES;

/**
 * `X`, state -> expression: one rate multiplier per reaction row. Row-major
 * by row: `X_OUT + row * STATE_DIMS + d`. The matrix seeds to zero; the bases
 * are written by `seedProduction`. The rows are read through a unit simplex
 * like `emitVector`'s, so this is a budget, not eight independent dials.
 * Outside the plastic span on purpose: expression is inherited, not learned.
 */
export const X_OUT = L_BASE + 2;
export const X_BASE = X_OUT + ROW_COUNT * STATE_DIMS;

/**
 * `ks`, the half-saturation of each species' uptake, one dimensionless gene
 * per species. The effective constant is `params.uptakeKs * ks[c]`, and it
 * seeds to 1 so a fresh body uses the global. A heritable gene rather than a
 * head off `h`: affinity is a property of the transporter, not of mood.
 */
export const KS_BASE = X_BASE + ROW_COUNT;

/**
 * What `seedProduction` writes: the bias on a kind's production rows, and the
 * bias on every uptake row. Invariant: `SEED_UPTAKE / (SEED_PRODUCTION +
 * CHEM_SPECIES * SEED_UPTAKE)` equals `1 / ROW_COUNT`, so the uptake half
 * lands on the flat fallback's own eighth a row.
 */
export const SEED_PRODUCTION = 2;
export const SEED_UPTAKE = 0.5;

/**
 * What share of its whole chemical budget a seeded Era puts on making ground.
 * `chemistry.test.ts` pins the derivation to what `expressVector` produces.
 */
export const ERA_GROUND_SHARE = SEED_PRODUCTION / (SEED_PRODUCTION + CHEM_SPECIES * SEED_UPTAKE);

/**
 * `G`, state -> gait, and its base. One row: `anchor`.
 *
 * How hard this body holds still at its point in the stroke: added to the
 * drag rate on the same cosine that swings its wires' rest length. A wire
 * swinging its rest length does not move its centre; travel comes only from
 * the two ends damping at different rates, so the difference in anchor
 * between a wire's ends is the whole of the step. Serves "locomotion" in
 * `docs/concepts.md`. Row-major: `G_OUT + row * STATE_DIMS + d`.
 */
export const G_OUT = KS_BASE + CHEM_SPECIES;
export const G_BASE = G_OUT + STATE_DIMS;
export const CHEM_LEN = G_BASE + 1;

/**
 * The span of `chem` a body can change while it is alive: `Wx`, `Wh`, `Wn`
 * and `b`, which the layout above puts next to each other.
 *
 * Only the state matrices learn (they are the only weights with a local
 * gradient). Contiguous, so the learned block and the eligibility trace are
 * flat arrays indexed exactly as the genome indexes the same weights.
 */
export const PLASTIC_BASE = W_IN;
export const PLASTIC_LEN = F_OUT - W_IN;

/**
 * The critic: a linear readout of `h` that predicts the cost this body is
 * heading into, plus its bias. Its error is what gates learning.
 */
export const CRITIC_LEN = STATE_DIMS + 1;

/**
 * One body's learning state, as the GPU holds it: the learned deltas, then
 * the eligibility traces, then the critic, then last frame's value estimate.
 * The CPU keeps these as four separate store arrays; both sides derive the
 * offsets here.
 */
export const LEARN_PLASTIC = 0;
export const LEARN_TRACE = PLASTIC_LEN;
export const LEARN_CRITIC = 2 * PLASTIC_LEN;
export const LEARN_PREV_V = 2 * PLASTIC_LEN + CRITIC_LEN;
export const LEARN_STRIDE = LEARN_PREV_V + 1;

/**
 * Widest swing the gait's anchor may ask for, in drag-rate units.
 * `updateState` clamps the head to it, `dampVelocities` spans its rate table
 * to it, and `genome.wgsl` transcribes it; the three must agree.
 */
export const GAIT_ANCHOR_MAX = 8;

/**
 * What one unit of a head's output is worth, per row. Genes are dimensionless
 * so one `CHEM_MUTATE` step means the same thing everywhere in the genome; a
 * gene of 1 is one natural unit of the behaviour.
 */
export const HEAD_SCALE = {
  align: 8,
  sep: 60,
  recoil: 100,
  cruise: 40,
  turn: 2,
  /** Same natural unit as the `grip` slider: one gene is one 1/s of drag. */
  anchor: 2,
} as const;

/**
 * This body's half-saturation for species `c`, in the field's own units.
 * Here because `energy.ts` reads it and `agents.ts` imports `energy.ts`.
 */
export function uptakeKsOf(chem: Float32Array, g: number, c: number, globalKs: number): number {
  const gene = chem[g + KS_BASE + c];
  const ks = globalKs * (gene > 0 ? gene : 0);
  // Zero affinity is division by zero downstream; floored at a thousandth of
  // the global.
  return ks > globalKs * 1e-3 ? ks : globalKs * 1e-3;
}
