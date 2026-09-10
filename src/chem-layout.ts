/**
 * Where a body's genome is laid out in memory.
 *
 * Its own module, and that is the whole point. `AgentStore` has to know how
 * wide one body's `chem` is in order to allocate it, and `agents.ts` builds
 * its flyweight on top of the store — so the store cannot import back, and the
 * width used to be a hand-copied duplicate with a comment calling it "safe to
 * duplicate, a structural fact rather than a tunable". It was neither. Within
 * one session it drifted twice, and the second time shipped: the store
 * allocated 104 floats a body while `seedChem` wrote 124, and every spawn died
 * on `RangeError: offset is out of bounds` from inside a typed-array `set`,
 * four frames of callstack away from anything that named a genome.
 *
 * Nothing in here imports anything, so both sides can depend on it and the
 * dependency direction stays one-way. The numbers are derived from each other
 * rather than written down, so adding a dimension or a head moves everything
 * that has to move.
 */

/**
 * The width of `h`, the recurrent state.
 *
 * Its dimensions have no names and cannot: whatever a lineage makes them mean
 * is the point of having them. They are bounded — `phi` keeps every one inside
 * (-1, 1) — and that is load-bearing rather than tidiness. A weight multiplies
 * its input, so an unbounded state against a bounded weight is the same
 * unbounded product; bounding the state is what makes `CHEM_SLOPE_MAX` cap
 * anything and the mutation range mean something.
 *
 * One dimension is spoken for at seed. `h[0]` is wired to carry `IN_DEMAND`
 * and the ground's taste weight is wired to read `h[0]`, which is what makes a
 * fresh body seek food when its neighbourhood is hungry and ignore it
 * otherwise. A second seeded pathway that also picked dim 0 would collide with
 * that silently.
 *
 * The gait's clock is deliberately *not* built here. A rotation in `Wh` is an
 * oscillator on paper and a bad one in practice: `phi` saturates
 * componentwise, so a slow period forces a loop gain near 1, and at a gain
 * near 1 any constant input — `IN_FULL` is one, and is never zero for long —
 * pushes the state into the contractive region and it collapses onto a fixed
 * point. Measured over the map: at gain 1.3 and a 60-frame period a drive of
 * 0.05 is already enough to kill it, and adding an adaptation dimension only
 * widens that to about 0.2. So `gaitPhase` is a register on the body and this
 * head supplies its amplitudes. See `G_OUT`.
 */
export const STATE_DIMS = 4;

/**
 * The input vector `x`: four raw channel readings at this body's position,
 * then three facts about the body itself.
 *
 * These are the named, bounded, differently-sourced components — `h`'s are
 * none of those things. What matters is that the three body facts are about
 * different things, because two inputs that move together buy nothing a single
 * one did. `IN_DEMAND` is the neighbourhood's unmet need, spread along the
 * wires and decayed per hop. `IN_FULL` is this body's own tank and nobody
 * else's, so a body can tell "I am hungry" from "my net is hungry" — the
 * distinction every rule about when to forage and when to give away turns on.
 * `IN_BOUND` is how much of it is attached, which is the only one that says
 * anything about position in a net, and the one that lets a channel mean
 * something different once a port is matched.
 *
 * `IN_DEMAND` is an input and deliberately not part of `h`. It is a
 * max-relaxation of what bodies are actually short of, which is what makes it
 * a potential `flowCharges` can move energy down. Let a genome decide what to
 * put in it and selection drives "ask maximally" within a few generations, the
 * field goes flat, and a flat field moves nothing. A lineage can still evolve
 * to *broadcast* its hunger — `Wx` reading this, `E` putting it on a channel —
 * it simply cannot lie to the transport layer about it.
 *
 * All seven are scaled to roughly the same range before they are read; see
 * `SENSE_SCALE`.
 */
export const IN_SENSE = 0;
export const IN_FULL = 4;
export const IN_BOUND = 5;
export const IN_DEMAND = 6;
export const IN_DIMS = 7;

/**
 * What one unit of a signal channel reading is worth, on the way into `x`.
 *
 * The other four inputs are already about [0,1] — `IN_FULL`, `IN_BOUND` and
 * `IN_DEMAND` are clamped, and the ground is divided by `cellCap`. The three
 * signal channels were passed raw, and they are not on that scale at all.
 *
 * Two things followed, and both cost the sense genes their usefulness. One
 * `CHEM_MUTATE` step moved the pre-activation by about seven times as much
 * through a sense weight as through any other input weight, so those genes had
 * seven times the leverage per mutation of everything beside them. And `phi`
 * is half-saturated by a weight of 0.14 and effectively pinned past 0.6, so
 * roughly seven per cent of the gene's legal range carried information and the
 * rest was flat — deaf to pinned in a few steps, with nothing in between to
 * select over.
 *
 * That is the same defect `HEAD_SCALE` cures on the output side, and this is
 * the same cure. Measured over soups of 60, 400 and 2000 bodies at 600 frames,
 * the p90 reading at a body's own position is 4.16, 4.24 and 4.53 — flat
 * across a thirty-fold range of pond size, which is what makes it a sensible
 * anchor. So a strong local signal reads about 1, the median reads near zero,
 * and a genuinely loud spot can still saturate, which it should be able to.
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
 * `e0` and `t0` are the named, kind-seeded part — "a Con emits ch0 and seeks
 * ch1" lives there, and every matrix seeds to zero, so a fresh body computes
 * `h = phi(0) = 0` and behaves exactly as it did before any of this existed.
 * The matrices are what drift. That is deliberate: a body spawned out of the
 * soup should arrive with its scent interactions unevolved and pick them up by
 * breeding into a net.
 *
 * `h`'s four dimensions have no names, and cannot: whatever a lineage makes
 * them mean is the point. `e0[con] = 0.72` is still readable off a body, which
 * is why the bases stayed vectors instead of being folded into the matrices.
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
 *
 * The output heads are where a behaviour stops being a number a body carries
 * and becomes something it computes. Alignment and separation were heritable
 * scalars, fixed for life — so a lineage could be shoaling or not, but no body
 * could shoal while fed and scatter while starving, which is the obvious thing
 * for a forager to do and was not expressible at any genome.
 *
 * The bases seed from the sliders and the matrices from zero, so a fresh body
 * has exactly the constant it used to have. What the store now holds is the
 * *phenotype*, rewritten each frame by `updateState`; the genome is `F` and
 * `f0`. Everything downstream — the native flock packing, the JS pair force,
 * `state-hash` — keeps reading the same field and does not care.
 */
export const F_OUT = B_STATE + STATE_DIMS;
export const F_BASE = F_OUT + 2 * STATE_DIMS;
/** `P`, state -> transport. One row: recoil. */
export const P_OUT = F_BASE + 2;
export const P_BASE = P_OUT + STATE_DIMS;
/**
 * `L`, state -> locomotion. Two rows: cruise, then turn.
 *
 * Unlike `F` and `P`, this one does not replace a heritable scalar — there
 * never was one. `stepSpeed` and `turnRate` were global params, the same for
 * every body in the pond, so this *adds* per-body variation rather than
 * relocating it. A body could not previously swim differently because it was
 * hungry, however far its lineage had drifted.
 */
export const L_OUT = P_BASE + 1;
export const L_BASE = L_OUT + 2 * STATE_DIMS;

/**
 * The body reaction table's rows: `excrete_c` and `uptake_c` for each of the
 * four species. See `docs/energy-chemistry-plan.md` §3.
 *
 * Eight, not four. The plan's own arithmetic says "4 + 4 * STATE_DIMS = 20
 * floats" while its table lists eight rows and its prose says the simplex is
 * "normalised across all eight rows"; the table and the prose agree with each
 * other and the sizing does not, so the sizing is the slip. Derived here from
 * the species count rather than written down, which is the point of this
 * module.
 */
export const CHEM_SPECIES = 4;
export const ROW_EXCRETE = 0;
export const ROW_UPTAKE = CHEM_SPECIES;
export const ROW_COUNT = 2 * CHEM_SPECIES;

/**
 * `X`, state -> expression: one rate multiplier per reaction row.
 *
 * The head a body's chemistry is a phenotype through. An enzyme, in the sense
 * the plan commits to, is a reaction whose rate constant comes from here —
 * nothing more. Row-major by row: `X_OUT + row * STATE_DIMS + d`.
 *
 * The matrix seeds to zero; the bases are written by `seedProduction`, so a
 * fresh body already expresses its kind's production half and an even uptake
 * half (`SEED_PRODUCTION`, `SEED_UPTAKE`). Every reaction still runs at the
 * constant it ran at before this existed, because those constants carry
 * `ROW_COUNT` and the uptake rows land on the flat fallback's own eighth. The
 * rows are read through a unit simplex like `emitVector`'s, so this is a
 * budget rather than eight independent dials: a body cannot both shout and
 * eat without giving something up, which is the trade-off division of labour
 * needs and the reason the head is one head and not eight.
 *
 * Outside `[PLASTIC_BASE, PLASTIC_BASE + PLASTIC_LEN)` on purpose — see the
 * plan's §8. Expression is inherited and mutated, not learned, though it
 * still varies within a life because it reads `h`.
 */
export const X_OUT = L_BASE + 2;
export const X_BASE = X_OUT + ROW_COUNT * STATE_DIMS;

/**
 * `ks`, the half-saturation of each species' uptake, one dimensionless gene
 * per species. The effective constant is `params.uptakeKs * ks[c]`, the way
 * `HEAD_SCALE` converts every other dimensionless gene into its natural unit,
 * and it seeds to 1 so a fresh body uses the global.
 *
 * A plain heritable gene rather than a head off `h`, which is a departure
 * from the plan's §4 ("`vmax` and `Ks` read off `h` as heads") and a
 * deliberate one. `X` supplies `vmax`: how much transporter a body is
 * *expressing*, which is regulation and should depend on how hungry it is.
 * Affinity is a property of the transporter itself — which one you have, not
 * how much of it you made — so it has no business moving with mood, and
 * giving it a head would cost twenty floats to model something as a decision
 * that is not one.
 *
 * The pair is what matters either way: `(vmax, ks)` do not dominate each
 * other, so a fast grazer needing rich ground and a scavenger living on
 * scraps are both viable and neither wins everywhere. That is the plan's
 * stated payoff and it survives the change.
 */
export const KS_BASE = X_BASE + ROW_COUNT;

/**
 * What `seedProduction` writes: the bias on a kind's production rows, and the
 * bias on every uptake row.
 *
 * Chosen together so that the uptake half lands on exactly an eighth a row —
 * the flat fallback's own value, which is what a seeded genome expressed
 * before any kind produced anything in particular. Four uptake rows at
 * `SEED_UPTAKE` against production summing to `SEED_PRODUCTION` is a simplex
 * of `SEED_PRODUCTION + CHEM_SPECIES * SEED_UPTAKE`, and `SEED_UPTAKE` over
 * that has to be `1 / ROW_COUNT`. A kind with two production species splits
 * `SEED_PRODUCTION` between them.
 */
export const SEED_PRODUCTION = 2;
export const SEED_UPTAKE = 0.5;

/**
 * What share of its whole chemical budget a seeded Era puts on making ground:
 * its entire production half, after normalisation.
 *
 * Derived from the seed rather than restated beside it, because two places
 * read it as a *scale* — the producer's upkeep discount is "how far along is
 * this body toward what an Era expresses" — and a copy of the number would
 * drift the moment the seed moved. `chemistry.test.ts` pins the derivation to
 * what `expressVector` actually produces.
 */
export const ERA_GROUND_SHARE = SEED_PRODUCTION / (SEED_PRODUCTION + CHEM_SPECIES * SEED_UPTAKE);

/**
 * `G`, state -> gait, and its base. One row: `anchor`.
 *
 * How hard this body holds still at its point in the stroke. It adds to the
 * drag rate, on the same cosine that swings the rest length of its wires, so
 * a body grips while its wires pull and lets go while they lengthen.
 *
 * The difference between a wire's two ends is the whole of the travel. A wire
 * swinging its rest length moves both its bodies and not their centre — the
 * correction is shared by inverse mass, which is exactly the operation that
 * leaves a centre where it was. What is left over is the *velocity* that
 * correction induces, and that decays at each body's own rate. Equal rates,
 * nothing left; different rates, a step.
 *
 * `grip * fullness` supplies one such difference and this supplies the other,
 * and unlike fullness this one is in phase with the stroke by construction
 * rather than by luck.
 *
 * A head rather than a constant so a body grips on its own account and by how
 * hungry it is, and so `seedGait` can make an Era and a Con different kinds
 * of thing: an Era barely grips, which is what makes a leaf an oar, and a Con
 * grips hard, which is what makes an interior body a foot.
 *
 * Row-major, like every head here: `G_OUT + row * STATE_DIMS + d`.
 */
export const G_OUT = KS_BASE + CHEM_SPECIES;
export const G_BASE = G_OUT + STATE_DIMS;
export const CHEM_LEN = G_BASE + 1;

/**
 * The span of `chem` a body can change while it is alive: `Wx`, `Wh`, `Wn`
 * and `b`, which the layout above happens to put next to each other.
 *
 * Only the state matrices learn. They are the only weights with a local
 * gradient — `phi'` gives each one a defensible eligibility — while an
 * output head has no per-channel error signal to learn from, so teaching one
 * would need random feedback and that is a separate decision. Behaviour
 * still changes, because every head reads `h`.
 *
 * Contiguous, so the learned block and the eligibility trace are both flat
 * arrays indexed exactly as the genome indexes the same weights, and the
 * effective weight is one add. Derived rather than written down, like every
 * other number in this file; if a matrix moves, this moves with it.
 */
export const PLASTIC_BASE = W_IN;
export const PLASTIC_LEN = F_OUT - W_IN;

/**
 * The critic: a linear readout of `h` that predicts the cost this body is
 * heading into, plus its bias. Its error is what gates learning — the only
 * part of the rule that is a real gradient rather than a correlation.
 */
export const CRITIC_LEN = STATE_DIMS + 1;

/**
 * One body's learning state, as the GPU holds it: the learned deltas, then
 * the eligibility traces, then the critic, then last frame's value estimate.
 *
 * The CPU keeps these as four separate store arrays, which is the right
 * shape for a pass that walks one of them at a time. The device wants one
 * row a body so it is one binding, and a birth or a readback transcodes a
 * handful of rows between the two. Both sides derive the offsets from here.
 */
export const LEARN_PLASTIC = 0;
export const LEARN_TRACE = PLASTIC_LEN;
export const LEARN_CRITIC = 2 * PLASTIC_LEN;
export const LEARN_PREV_V = 2 * PLASTIC_LEN + CRITIC_LEN;
export const LEARN_STRIDE = LEARN_PREV_V + 1;

/**
 * What one unit of a head's output is worth, per row.
 *
 * The heads exist in the same genome as the emit and taste weights, which live
 * around 0 to 4 and are mutated by one `CHEM_MUTATE` for all of them. Flocking
 * separation sits near 48 and transport recoil near 100, so storing those raw
 * would give them a mutation step of 0.02 against a value of a hundred —
 * evolutionarily frozen, while a taste weight beside them moved freely.
 *
 * So the genes are dimensionless and this is the conversion. A gene of 1 means
 * "one natural unit of this behaviour", the mutation step means the same thing
 * everywhere in the genome, and the physical ranges stay where the forces need
 * them. Roughly the magnitude each slider ships at, rounded.
 */
/**
 * Widest swing the gait's anchor may ask for, in drag-rate units.
 *
 * `updateState` clamps the head to it, `dampVelocities` spans its rate table
 * to it, and `genome.wgsl` transcribes it — three places that must agree
 * about how far a stroke can reach, so the number lives here with the rest of
 * the layout rather than on `Sim`.
 */
export const GAIT_ANCHOR_MAX = 8;

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
 *
 * Here rather than in `agents.ts` for the reason this module exists at all:
 * `energy.ts` reads it inside the harvest, and `agents.ts` imports `energy.ts`
 * for `extraCapFor`, so the other direction would be a runtime cycle. Reading
 * a gene is layout knowledge, which is what lives here.
 */
export function uptakeKsOf(chem: Float32Array, g: number, c: number, globalKs: number): number {
  const gene = chem[g + KS_BASE + c];
  const ks = globalKs * (gene > 0 ? gene : 0);
  // Zero affinity is division by zero downstream, and a gene mutated to or
  // past zero is a body with an infinitely good transporter, which is not a
  // thing. Floored at a thousandth of the global, which is a very good one.
  return ks > globalKs * 1e-3 ? ks : globalKs * 1e-3;
}