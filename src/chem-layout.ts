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
/*
 * The recurrent core comes first and the heads follow it, so that everything
 * a body can learn is one contiguous run: `Wx` through `g0`. The order used to
 * be `E, T` then the core then the rest, which put two heads on the wrong side
 * of the learned block and was the only reason a head could not learn.
 */
/** `Wx`, input -> state. Row-major by state dim: `W_IN + d * IN_DIMS + k`. */
export const W_IN = 8;
/** `Wh`, state -> state. This body's own memory. */
export const W_SELF = W_IN + STATE_DIMS * IN_DIMS;
/** `Wn`, mean neighbour state -> state. Its diagonal is a learned per-hop decay. */
export const W_NET = W_SELF + STATE_DIMS * STATE_DIMS;
/** `b`, the state bias. */
export const B_STATE = W_NET + STATE_DIMS * STATE_DIMS;
/** `E`, state -> emit. Row-major by channel: `E_OUT + c * STATE_DIMS + d`. */
export const E_OUT = B_STATE + STATE_DIMS;
/** `T`, state -> taste. Same shape. */
export const T_OUT = E_OUT + 4 * STATE_DIMS;
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
export const F_OUT = T_OUT + 4 * STATE_DIMS;
export const F_BASE = F_OUT + 2 * STATE_DIMS;
/** `P`, state -> transport. Two rows: thrust, then recoil. */
export const P_OUT = F_BASE + 2;
export const P_BASE = P_OUT + 2 * STATE_DIMS;
/**
 * `L`, state -> locomotion. Two rows: cruise, then turn.
 *
 * Unlike `F` and `P`, this one does not replace a heritable scalar — there
 * never was one. `stepSpeed` and `turnRate` were global params, the same for
 * every body in the pond, so this *adds* per-body variation rather than
 * relocating it. A body could not previously swim differently because it was
 * hungry, however far its lineage had drifted.
 */
export const L_OUT = P_BASE + 2;
export const L_BASE = L_OUT + 2 * STATE_DIMS;

/**
 * The body reaction table's rows: `excrete_c` and `uptake_c` for each of the
 * four species.
 *
 * Eight, not four. The plan this came from had arithmetic saying "4 + 4 * STATE_DIMS = 20
 * floats" while its table lists eight rows and its prose says the simplex is
 * "normalised across all eight rows"; the table and the prose agree with each
 * other and the sizing does not, so the sizing is the slip. Derived here from
 * the species count rather than written down, which is the point of this
 * module.
 */
export const CHEM_SPECIES = 4;

/*
 * `X` and `x0` were here: a head from `h` to eight reaction rows, one unit
 * budget over four excretion reactions and four uptake ones, so a body could
 * not both shout and eat without giving something up.
 *
 * Both halves lost their subject. Nothing excretes, so the production rows
 * have nothing to produce; a body eats the ground and nothing else, so an
 * uptake row has one species to be a recipe for and every body can digest it
 * raw. Forty floats a body, recomputed every frame by `expressVector` and read
 * by nothing — `rowCost` shipped at 0 and `upkeep` at 0, which were the only
 * two things left that looked at them. Deleted rather than left as a door,
 * because the door led to a room that is no longer there.
 */





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
export const G_OUT = L_BASE + 2;
export const G_BASE = G_OUT + STATE_DIMS;

/**
 * `ks`, the half-saturation of a body's uptake: one dimensionless gene. The
 * effective constant is `params.uptakeKs * ks`, the way `HEAD_SCALE` converts
 * every other dimensionless gene into its natural unit, and it seeds to 1 so a
 * fresh body uses the global.
 *
 * One rather than four, because a body eats one species. A plain heritable
 * gene rather than a head off `h`: affinity is a property of the transporter
 * itself — which one you have, not how much of it you made — so it has no
 * business moving with mood — which is now also where it *sits*: the first
 * gene past `g0`, so the learnable run ends cleanly and no gene that must
 * not move is caught inside it.
 *
 * `(vmax, ks)` do not dominate each other, so a fast grazer needing rich
 * ground and a scavenger living on scraps are both viable and neither wins
 * everywhere. `vmax` is global (`uptakeVmax`) now that `X` is gone, so what a
 * lineage owns of the pair is the affinity.
 */
export const KS_BASE = G_BASE + 1;

/**
 * `Tx`, the transmission vector, and `Gx`, the gate vector: one gene per
 * species each, both plain heritable genes rather than heads off `h`.
 *
 * The doc's §4.1 transmission matrix and §4's threshold gate, which §8 lists
 * among the per-agent parameters. `Tx[c]` is how fast this body broadcasts
 * species `c` down its principal wire; `Gx[c]` is the concentration it has to
 * be holding before it broadcasts any. Seeded by kind and free to drift:
 * an Era pushes fuel and primer downstream, a Con broadcasts the catalyst, a
 * Dup broadcasts the inhibitor.
 *
 * Genes and not a head, which is a change of position and the reason is
 * arithmetic. Two rows off `h` cost ten floats; eight would cost forty, and
 * the doc wants all eight. `ks`, the uptake affinity, already makes the same
 * trade with the same argument: which transporter a body has is not a thing
 * that should move with its mood. What a body *broadcasts* does move with its
 * mood, because the rate is `Tx[c] * (x_c - Gx[c])` and the concentration is
 * the swinging part — the impulse is in the chemistry, not in the gene.
 */
export const TX_BASE = KS_BASE + 1;
/**
 * Slots in `Tx` and `Gx`, the doc's four: A is the gut, then the reactor's
 * three. `TX_A` moves swallowed food down a wire, which is §6.1's Era pushing
 * fuel into the net it hangs off.
 */
export const TX_A = 0;
export const TX_B = 1;
export const TX_C = 2;
export const TX_D = 3;
export const GX_BASE = TX_BASE + CHEM_SPECIES;

/**
 * `Sw` and `Gw`: what mixture of the reactor's three species each of the two
 * actuators reads. **This is where a lineage owns the phase of its own gait.**
 *
 * The reactor holds B, C and D at fixed angles to each other — the eigenvector
 * at the operating point puts B at +84 degrees, C at 0 and D at −51 to −65,
 * and those are `k3` and `d` rather than anything anybody dialled. Three
 * phasors spanning more than 180 degrees, so a weighted sum of them reaches
 * **any** phase and amplitude. A body that changes its weights changes when in
 * its own cycle it strokes, and when it grips.
 *
 * Both, and not one with the other as a reference, because the phase that
 * matters is not only the angle between the two actuators. The impulse a
 * stroke works against is the gut broadcast, which is timed by the chemistry
 * (`Sim.advanceGait`), so each actuator's angle to *that* is a real degree of
 * freedom. Fixing one would lose reachable gaits.
 *
 * Genes and not heads, for `Tx`'s reason: three floats each against eighteen,
 * and a phase that moved with a body's mood would be a body that could not
 * hold a gait. Seeded so a fresh pond is exactly what it was — `Sw` pure C,
 * which is the stroke the rest length has always read, and `Gw` pure D, which
 * is the grip swing's 53-degree lag. What is new is everything between.
 *
 * `gaitSwell` and `gripSwing` stay the global magnitudes: the weights choose a
 * direction in (B, C, D) and those choose how hard it is pulled.
 */
export const SPECIES_W = 3;
export const SW_BASE = GX_BASE + CHEM_SPECIES;
export const GW_BASE = SW_BASE + SPECIES_W;
/** Slots in `Sw` and `Gw`, in the reactor's own order. */
export const W_B = 0;
export const W_C = 1;
export const W_D = 2;
export const CHEM_LEN = GW_BASE + SPECIES_W;

/**
 * The genome as a list of named segments, in memory order.
 *
 * This is what lets a stored net outlive a layout change. A blob written by
 * an older build carries its own copy of this list, and `pond/net-blob.ts`
 * lines the two up by *name*: a segment that kept its length is carried
 * across wherever it moved to, one the old build never had is seeded for the
 * body's kind, one this build no longer has is dropped, and one that changed
 * length refuses the whole net — its numbers no longer mean what they meant,
 * and the header exists so that never happens silently.
 *
 * The list has to tile `[0, CHEM_LEN)` exactly and the learned block has to
 * be a run of whole segments; `layoutSelfCheck` in `net-blob.ts` says so, the
 * suite asserts it, and the pond CLI refuses to start without it. A head
 * added above without an entry here therefore fails before a run rather than
 * writing blobs the next layout change cannot read.
 */
export interface ChemSegment {
  name: string;
  at: number;
  len: number;
}

export const CHEM_SEGMENTS: readonly ChemSegment[] = [
  { name: 'emit', at: EMIT, len: 4 },
  { name: 'taste', at: TASTE, len: 4 },
  { name: 'Wx', at: W_IN, len: STATE_DIMS * IN_DIMS },
  { name: 'Wh', at: W_SELF, len: STATE_DIMS * STATE_DIMS },
  { name: 'Wn', at: W_NET, len: STATE_DIMS * STATE_DIMS },
  { name: 'b', at: B_STATE, len: STATE_DIMS },
  { name: 'E', at: E_OUT, len: 4 * STATE_DIMS },
  { name: 'T', at: T_OUT, len: 4 * STATE_DIMS },
  { name: 'F', at: F_OUT, len: 2 * STATE_DIMS },
  { name: 'f0', at: F_BASE, len: 2 },
  { name: 'P', at: P_OUT, len: 2 * STATE_DIMS },
  { name: 'p0', at: P_BASE, len: 2 },
  { name: 'L', at: L_OUT, len: 2 * STATE_DIMS },
  { name: 'l0', at: L_BASE, len: 2 },
  { name: 'G', at: G_OUT, len: STATE_DIMS },
  { name: 'g0', at: G_BASE, len: 1 },
  { name: 'ksg', at: KS_BASE, len: 1 },
  { name: 'Tx', at: TX_BASE, len: CHEM_SPECIES },
  { name: 'Gx', at: GX_BASE, len: CHEM_SPECIES },
  { name: 'Sw', at: SW_BASE, len: SPECIES_W },
  { name: 'Gw', at: GW_BASE, len: SPECIES_W },
];

/**
 * The span of `chem` a body can change while it is alive: the recurrent core
 * (`Wx`, `Wh`, `Wn`, `b`) **and every output head** (`E`, `T`, `F`, `P`, `L`,
 * `G` with their bases), which the layout above puts next to each other for
 * exactly this reason.
 *
 * It used to be the core alone, and the argument was that only the state
 * matrices have a local gradient — `phi'` gives each one a defensible
 * eligibility — while an output head has no per-channel error to learn from.
 * That argument was right about the gradient and wrong about the conclusion.
 * Measured on a grown pond, four of the six heads sat at the mutation floor
 * for their whole lives: the learner could reshape `h` all it liked against a
 * map from `h` to behaviour that was fixed at birth and only ever drifted.
 * Learning the state without learning the read-out is learning half a policy.
 *
 * What supplies the missing factor is `exploreAt`: each head output is
 * perturbed by its own reproducible noise, and the eligibility correlates
 * *that* noise with what the critic says happened next. See it there.
 *
 * `emit`'s and `taste`'s bases are the two things inside these heads that do
 * *not* learn, and they are outside the run on purpose. `emit` is normalised
 * to a simplex — a body's whole voice budget — so a learned delta on the base
 * would be renormalised away every frame; `taste`'s base is the lineage's
 * standing preference, which is what a learned delta is supposed to be
 * measured against. `ksg` sits just past the run for the same kind of reason,
 * written where it is defined.
 *
 * Contiguous, so the learned block and the eligibility trace are both flat
 * arrays indexed exactly as the genome indexes the same weights, and the
 * effective weight is one add. Derived rather than written down, like every
 * other number in this file; if a matrix moves, this moves with it.
 */
export const PLASTIC_BASE = W_IN;
export const PLASTIC_LEN = KS_BASE - W_IN;

/**
 * The heads, in the order the genome lays them, as offsets *within* the
 * learned block. `rows` is how many outputs each drives; `base` is where that
 * head's biases start, or −1 when its base lives outside the block.
 *
 * One table rather than seven call sites, because three things have to agree
 * about it exactly: which weight the state pass reads, which perturbation the
 * output gets, and which eligibility the learner credits. `genome.wgsl`
 * unrolls the same table and `genome-kernel.test.ts` checks that it did.
 */
export const HEAD_TABLE: readonly { at: number; rows: number; base: number }[] = [
  { at: E_OUT - W_IN, rows: 4, base: -1 },
  { at: T_OUT - W_IN, rows: 4, base: -1 },
  { at: F_OUT - W_IN, rows: 2, base: F_BASE - W_IN },
  { at: P_OUT - W_IN, rows: 2, base: P_BASE - W_IN },
  { at: L_OUT - W_IN, rows: 2, base: L_BASE - W_IN },
  { at: G_OUT - W_IN, rows: 1, base: G_BASE - W_IN },
];

/** How many head outputs one body drives, which is how many noises it needs. */
export const HEAD_ROWS = HEAD_TABLE.reduce((n, h) => n + h.rows, 0);

/**
 * The exploration noise one head output gets this frame, in [-1, 1).
 *
 * A pure function of (body, frame, output) and not a stream, which is the
 * only shape that can be identical on the CPU and inside `genome.wgsl`: there
 * is no RNG state to keep in step, no buffer to carry across a dispatch, and
 * a body that migrates between the two paths mid-run sees the same sequence.
 * The hash is the `lowbias32` finalizer, whose multiplies are exact in `u32`
 * on both sides — `Math.imul` here, native wrapping there — and the top 24
 * bits are taken so the result is exact in `f32` too.
 *
 * Why a head needs noise at all: node perturbation. The three-factor rule on
 * the recurrent core uses `phi'(v)` as its post-synaptic factor, which is the
 * gradient of `h` with respect to the weight. A head is linear, so its
 * equivalent factor is 1 — the same for every row — and every head in the body
 * would then move in lockstep on the critic's sign, never able to find that
 * cruise should rise while turn falls. Perturbing each output independently
 * and correlating *its own* perturbation with the reward that followed is what
 * gives each row its own credit, and it is the standard answer for a policy
 * with a scalar reward and no target vector (Fiete & Seung 2006; Werfel,
 * Xie & Seung 2005).
 */
export function exploreAt(slot: number, frame: number, row: number): number {
  // The `+ 1` matters: `lowbias32(0)` is 0, so without it body 0 at frame 0
  // would draw exactly -1 on row 0 every time a pond starts.
  let v = (Math.imul(slot, 0x9e3779b1) + Math.imul(frame, 0x85ebca6b) + Math.imul(row, 0xc2b2ae35) + 1) >>> 0;
  v = (v ^ (v >>> 16)) >>> 0;
  v = Math.imul(v, 0x7feb352d) >>> 0;
  v = (v ^ (v >>> 15)) >>> 0;
  v = Math.imul(v, 0x846ca68b) >>> 0;
  v = (v ^ (v >>> 16)) >>> 0;
  return (v >>> 8) * (2 / 16777216) - 1;
}

/**
 * Frames wrap here, so the noise key stays exact in an `f32`: the uniform
 * carries the count as a float and 2^24 is the last integer it can hold
 * without rounding. At 60fps that is 77 hours of simulated time before the
 * sequence repeats, against runs measured in minutes.
 */
export const FRAME_WRAP = 1 << 24;

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
/**
 * Last frame's `IN_FULL`, for the *rate* half of the reward. See
 * `params.learnReward`: the level signal says how short a body is and the rate
 * signal says whether it is gaining, and a body at its cap has a gradient
 * under the second where it has none under the first.
 */
export const LEARN_PREV_FULL = LEARN_PREV_V + 1;
export const LEARN_STRIDE = LEARN_PREV_FULL + 1;

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
  thrust: 1,
  recoil: 100,
  cruise: 40,
  turn: 2,
  /** Same natural unit as the `grip` slider: one gene is one 1/s of drag. */
  anchor: 2,
} as const;

/**
 * This body's half-saturation for the ground, in the field's own units.
 *
 * Here rather than in `agents.ts` for the reason this module exists at all:
 * `energy.ts` reads it inside the harvest, and `agents.ts` imports `energy.ts`
 * for `extraCapFor`, so the other direction would be a runtime cycle. Reading
 * a gene is layout knowledge, which is what lives here.
 */
export function uptakeKsOf(chem: Float32Array, g: number, globalKs: number): number {
  const gene = chem[g + KS_BASE];
  const ks = globalKs * (gene > 0 ? gene : 0);
  // Zero affinity is division by zero downstream, and a gene mutated to or
  // past zero is a body with an infinitely good transporter, which is not a
  // thing. Floored at a thousandth of the global, which is a very good one.
  return ks > globalKs * 1e-3 ? ks : globalKs * 1e-3;
}