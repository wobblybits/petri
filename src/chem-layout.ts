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
 * The inner state a body's chemistry is modulated by.
 *
 * It was one number — `request` — and everything a body could condition on
 * had to be expressible as "how badly does my neighbourhood need energy". So a
 * lineage could evolve to shout when its net was hungry and nothing else: not
 * to go quiet when it was itself full, not to hunt harder when it liked where
 * it was standing, not any rule with two clauses in it. One input is a gain
 * knob, not a controller.
 *
 * Three, now, and the three are chosen to be about different things — which
 * matters more than how many there are, because two inputs that move together
 * buy nothing a single one did not. `NEED` is the neighbourhood's, spread over
 * the wires. `FULL` is this body's own and nobody else's. `HERE` is about the
 * world rather than the body at all.
 *
 * Each is bounded, and that is load-bearing rather than tidiness. A slope
 * multiplies its input, so an unbounded state and a bounded weight is the same
 * unbounded product — one runaway body would emit a number the field cannot
 * hold. Bounding the state instead means `CHEM_SLOPE_MAX` actually caps what a
 * slope can do, which is what makes the mutation range mean something.
 */
export const STATE_DIMS = 4;
/**
 * The neighbourhood's unmet need, 0..1. Aggregated by `spreadRequests` over
 * the wire graph and decayed per hop, so this is emphatically *not* the body's
 * own hunger — it is what the net around it is short of. Emitting on it turns
 * a gradient that only travels along wires into one that travels through
 * space, so a starving net can call to a forager that is not attached to it.
 */
export const NEED = 0;
/**
 * How full this body's own tank is, 0 (at or below break-even) to 1 (at its
 * own `energyCap`). The private counterpart to `NEED`: a body can now tell the
 * difference between "I am hungry" and "my net is hungry", which is the
 * distinction every rule about when to forage and when to give away turns on.
 */
export const FULL = 1;
/**
 * How much this body likes where it is standing, squashed to (-1, 1).
 *
 * Its own `trail` — everything it can smell, already weighted by its own taste
 * — through `x / (1 + |x|)`. Signed, because a body can be somewhere it is
 * repelled by, and that is a different situation from being somewhere dull,
 * and the two should be able to drive different behaviour.
 *
 * The one dimension that is about the world. It is also the one that closes a
 * loop: taste feeds the trail, the trail feeds the state, and the state feeds
 * taste. A lineage can evolve a body whose sense of smell sharpens the more it
 * likes what it smells, or one that goes blind when overwhelmed.
 */
export const HERE = 2;
/**
 * How much of this body is attached: filled ports over total, 0 to 1.
 *
 * The dimension that lets a channel mean two different things in one lifetime.
 *
 * A port's scent is only doing latching work while that port is open. Once it
 * is matched the seeded meaning — Con emits ch0 and seeks ch1, which is what
 * makes a redex — has done its job, and the channel is free to carry anything
 * the lineage has drifted onto. When a neighbour dies and the socket reopens,
 * the latching meaning is wanted again, and at exactly the moment it becomes
 * useful: an open port is how two nets can fuse. Without a way to read its own
 * occupancy a body cannot tell those two regimes apart, so its channels have to
 * mean one thing forever and every signal competes with mate-finding.
 *
 * It is also the only dimension that says anything about *position in a net*.
 * `NEED` is the neighbourhood's, `FULL` and `HERE` are strictly local; none of
 * them distinguishes an interior body from one on the boundary. That
 * distinction is where differentiated tissue would have to start, and there is
 * no net-level reproduction to build organs any other way — only die-off
 * reopening sockets and nets fusing.
 *
 * An Era has one port, so its `BOUND` is 0 or 1 and nothing between; a Con or
 * Dup has three and can be anywhere on thirds.
 */
export const BOUND = 3;

/**
 * The input vector `x` the state is driven by: the four raw channel readings
 * at this body's position, then the three physical facts about it.
 *
 * `DEMAND` is in here as an *input* and deliberately not as part of `h`. It is
 * the energy-shortfall field — a max-relaxation of what bodies are actually
 * short of, which is what makes it a potential `flowCharges` can move energy
 * down. Let a genome decide what to put in it and selection drives "ask
 * maximally" within a few generations; the field goes flat, and a flat field
 * moves nothing. A lineage can still evolve to *broadcast* its hunger — `Wx`
 * picking this up and `E` putting it on a channel — it simply cannot lie to
 * the transport layer about it.
 */
export const IN_SENSE = 0;
export const IN_FULL = 4;
export const IN_BOUND = 5;
export const IN_DEMAND = 6;
export const IN_DIMS = 7;

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
export const CHEM_LEN = L_BASE + 2;

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
export const HEAD_SCALE = {
  align: 8,
  sep: 60,
  thrust: 1,
  recoil: 100,
  cruise: 40,
  turn: 2,
} as const;
