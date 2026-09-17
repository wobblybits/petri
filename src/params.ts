import { FIELD_CELL } from './fields.ts';
// Derived, not copied. These three are the values the pond has always run at,
// and they live where the economy that uses them lives; writing the numbers
// again here is exactly the duplication `chem-layout.ts` exists to warn about.
import { ERA_CAP_RATIO, ERA_UPKEEP_RATIO, REWRITE_SHARE } from './energy.ts';
import { SENSE_SCALE } from './chem-layout.ts';

export interface Params {
  deposit: number;
  diffuse: number;
  decay: number;
  /**
   * Global gain on how strongly anything smelled moves a body.
   *
   * Stays global, and stays. It looks redundant with the magnitude of a body's
   * own taste weights — two knobs for one quantity — but they are not the same
   * kind of knob. This is a property of the *world*, like `diffuse` or
   * `deposit`: how much chemotaxis drives anything at all here. Taste is a
   * property of a body: what it cares about, relative to what its neighbours
   * care about. Folding this into the seeds would make "smell matters less in
   * this pond" unexpressible without editing every genome in it.
   */
  sense: number;
  attractStrong: number;
  attractMedium: number;
  snapRadius: number;
  snapArc: number;
  snapWell: number;
  faceRadius: number;
  faceAttract: number;
  rewriteDuration: number;
  springK: number;
  springDamp: number;
  /** How far an aux port aims off its neighbour, toward its own side. 1 = 20 degrees. */
  auxSpread: number;
  /**
   * Strength of the soft inverse-square push a fully wired agent exerts on
   * agents from *other* nets. Flocking separation only ever applied within a
   * net, so before this nothing pushed separate nets apart at all.
   */
  /**
   * Personal space: a local force holding bodies off each other.
   *
   * Measured to cost reproduction, together with `flockAlign`, and neither
   * alone. Over 45s of a 250-body soup, counting how deep the average lineage
   * gets (`Sim.census().bornMean`):
   *
   *     declutter 1.4, flockAlign 5.5   ->  0.36    (as shipped)
   *     declutter 0,   flockAlign 5.5   ->  0.29
   *     declutter 1.4, flockAlign 0     ->  0.42
   *     declutter 0,   flockAlign 0     ->  6.54
   *
   * Eighteen times the generation depth with both off, and almost nothing from
   * turning off either one. Read as a comparison at equal age, which is what
   * it is — 45 s is mostly warm-up for a preset that drops its whole
   * population in at once as founders, so the absolute figures are far too
   * early to say what any of these settles at. See `Sim.census`. The mechanism is visible in the wire counts: at the
   * shipped values only 2 of 93 wires are principal-to-principal and none are
   * Con-Dup, while 98% of bodies can afford a rewrite and the ground is still
   * at capacity. Bodies are rich and idle — they are not meeting. Personal
   * space holds them apart and alignment turns a head-on approach into a shoal
   * swimming the same way, and a commute needs two principals nose to nose.
   *
   * Left as they are, because these are what make a settled net look settled
   * and a pond look like a pond, and that is a real thing to want. But it is a
   * direct trade against evolution and it should be a decision rather than a
   * surprise.
   */
  declutter: number;
  /**
   * How hard a wire crossing a principal connection reels its own ends
   * together. Off by default: measured over three minutes of soup it made
   * crossings, clumping and rewrite throughput all slightly worse once
   * `declutter` was in, which already removes the crossings a local force
   * can plausibly undo. Kept as a knob because the detection is the cheap part.
   */
  /**
   * How hard a rope pushes off other ropes and off bodies it is not attached
   * to. Segment-based and one-way — a rope never moves an agent — so this
   * cannot feed back into the joint solver.
   */
  wireClear: number;
  /** Port-axis stiffness multiplier. Higher = wires hug their port axis harder. */
  portStiff: number;
  /** Rest-length breathing amplitude, as a fraction. 0 = a settled net freezes. */
  wireBreathe: number;
  /**
   * Seconds a taut wire keeps its rest-shape constraint. 0 = never drop it.
   * Shape only matters while slack is degenerate; a taut rope does not need it.
   */
  wireShapeAge: number;
  /**
   * Seconds before a taut wire becomes span-only (joint, no rope nodes).
   * 0 = never. A leftover that goes slack again gets the full rope back.
   */
  wireSpanAge: number;
  /**
   * Live length / rest at or below which a wire counts as taut for aging.
   * A coarsened wire stays coarsened until it exceeds this by a small band.
   */
  wireTaut: number;
  /**
   * Live length / rest at which a wire tears loose. 0 = wires never break.
   *
   * Well above `wireTaut`, because the span constraint is compliant and a
   * loaded wire legitimately stretches — a threshold near the taut ratio
   * shreds a working net rather than relieving it. Measured over thirty
   * seconds of a 400-body soup, against snapping switched off entirely:
   *
   *   off   281 wires, 340 bodies
   *   4.0   283 wires, 344 bodies   never fires
   *   3.0   277 wires, 338 bodies   fires, costs nothing in aggregate
   *   2.2   205 wires, 260 bodies   a quarter of the pond gone
   *   1.5   126 wires, 220 bodies   shredded
   *
   * 3.0 is the setting that relieves a strained net without dismantling a
   * working one. Below about 2.5 this stops being a safety valve and becomes
   * a second death rate.
   *
   * It closes a loop that already existed with nothing at the end of it:
   * `wireShrink` pulls wired bodies together, and in a crowded net they cannot
   * close the distance, so tension builds and simply stays. Now the most
   * strained link lets go, the ports come free to latch elsewhere, and an
   * over-packed net reconfigures instead of straining forever.
   */
  wireSnap: number;
  /**
   * How far a wire carrying energy pulls its own ends together, 0 to 1.
   *
   * `concepts.md`'s Shape heading has asked for this since it was written —
   * *tension is held by energy flowing along a wire, so a net that moves no
   * energy is slack* — and the honest note under it has always been "not yet
   * true". This is the term that makes it true. One transport packet across a
   * wire this frame shortens it by half of this; a busy wire approaches all of
   * it; a wire that moves nothing sits at exactly its `restBase`.
   *
   * **It was built once before and withdrawn, and both objections are now
   * answered.** The first was that the pull and the grip were driven by the
   * same packet at the same instant, so the loop they traced in (length, grip)
   * closed on a line and its area — which is the net displacement — was
   * nothing. `gripSwing` is the grip now, driven by the reactor's D and 53
   * degrees off the stroke, so the loop has area. The second was that it spent
   * the wire's rest length, which is what decides whether two bodies ever meet:
   * turned up far enough to swim it held every pair too tight to rewrite. That
   * was true while `Sim.principalRedexReady` measured against one global
   * `wireMinRest`. It measures against `Wire.restBase` now, which is the
   * wire's own, so shortening a wire is a change of shape and not a change of
   * who breeds.
   *
   * **Measured on the worm bench, and it is the largest locomotion effect this
   * project has found:** at a matched grip swing the worm travels 0.66 px/s
   * with this off and 2.62 with it at 0.3 — and 4.22 with the grip swing off
   * as well. Three to four times, from a term that costs nothing because the
   * transport it reads was happening anyway.
   *
   * One consequence to know rather than discover. A tugged wire's ends sit
   * closer, and `principalRedexReady` compares the *actual* span against
   * `restBase * 1.3` — so a wire that is carrying energy reaches its rewrite
   * gate sooner. Energy flowing makes things react, which reads as more churn:
   * over three seeds at 90 s the pond stands at fewer bodies and fewer wires
   * with this on. Depth does not resolve at three seeds, and the seed spread
   * is larger than the effect, so that is a thing to look at rather than a
   * number to trust.
   */
  wireTug: number;
  wireMinRest: number;
  wireShrink: number;
  eraMass: number;
  nodeMass: number;
  turnRate: number;
  /**
   * How strongly a fresh body is drawn to full ground, against a reading of 1
   * for a cell at capacity.
   *
   * The seed only. Like `attractStrong` and `attractMedium` this sets where a
   * population starts and is never read again — breeding takes it from there,
   * and a lineage is free to drift to indifference or to outright avoidance,
   * which for food would be a strange thing to become but is not this
   * parameter's business to prevent.
   *
   * Seeds the taste *slope* against `request`, not the flat weight, so a fed
   * body ignores food and a hungry one turns toward it. A flat attraction
   * seems safe — untouched ground is uniform, and a uniform field steers
   * nothing — but a body eats a dip under itself within a frame or two and
   * then chases the dip. See `seedChem`, which has the measurement.
   */
  attractFood: number;
  sensorAngle: number;
  sensorDist: number;
  stepSpeed: number;
  /** Persistence time of self-propulsion, in seconds. Longer = smoother runs. */
  swimTau: number;
  /** Self-propulsion noise, as a fraction of cruise speed. 0 = a flat setpoint. */
  swimNoise: number;
  drag: number;
  angDrag: number;
  /**
   * How much a full tank drags: added to `drag`, in proportion to how full a
   * body is. 0 = one drag law for every body, which is the pond as it was.
   *
   * Locomotion had a hole in it. `drag` is one rate applied to every body
   * alike, and the rope shares its corrections by mass, so no arrangement of
   * internal forces can move a net's centre of mass. `applyTransportRecoil`
   * says as much outright: at `transportThrust` 0 the pair is equal and
   * opposite and the flock's centre never moves. A net therefore travelled
   * only because thrust *withheld* part of the receiver's kick — a
   * reactionless drive, minting momentum wherever energy flows.
   *
   * Friction that differs body to body is the way out, and the one every
   * crawler takes. Gray and Hancock's flagellum swims because a slender
   * segment drags about twice as hard sideways as lengthwise, and a snake's
   * belly does the same against sand; an earthworm and a crawling cell
   * instead *modulate* their grip, anchoring one end while the other slides.
   * The pond's stroke is longitudinal — a kick runs along a wire — so the
   * anisotropic version buys it nothing: a straight chain has no transverse
   * wave, and Purcell's scallop still cannot swim. Modulated grip is the one
   * that fits, and the pond already carries the state to modulate on.
   *
   * So a transfer does two things at once. It kicks the pair apart along the
   * wire, and it changes which of the two is anchored: the sender is lighter
   * on its feet by exactly what it just sent, the receiver heavier by the
   * same. An impulse `p` into a body of mass `m` at rate `k` carries it
   * `p / (m k)` before it stops, so the pair's centre ends up
   *
   *     |p| * (1/k_sender - 1/k_receiver) / (m_sender + m_receiver)
   *
   * along the sender's recoil. Mass cancels out of the *direction*: which way
   * a net goes is decided only by which end is grippier. The rope cannot undo
   * it either, because a mass-weighted position correction moves the two
   * bodies and not their centre.
   *
   * The sign is left open, because the pond should settle it rather than this
   * file. Transport runs down the demand gradient, so a sender is usually the
   * fuller of the two:
   *
   *   - **Positive** — full grips, empty slides. The receiver is the one that
   *     slides, so a net holding a gradient walks toward its hungry end,
   *     *with* the flow: it carries itself toward whatever it is feeding.
   *   - **Negative** — full slides, empty grips. The sender slides, so the net
   *     walks back toward its supply, *against* the flow, which is the
   *     direction `transportThrust` already drifts it.
   *
   * The sum is clamped at zero, so a rate can be cancelled but never
   * inverted: negative drag is an energy source and the soup comes apart in
   * seconds. Angular drag is left alone, so this does one thing.
   *
   * Negative is worth reading carefully, because it looks like more life and
   * is mostly less. Nothing about mass is involved — the rate multiplies
   * velocity directly, so a heavy body and a light one damp alike. What a
   * negative value does is make a *full* body slippery: at -2 against a
   * `drag` of 0.55 the sum reaches zero at 27% of a tank, and every body
   * fuller than that is frictionless and coasts until something hits it.
   * The pond fills with motion, and none of it is anybody's doing.
   *
   * It also quietly kills the gait. A stroke travels on the *difference*
   * between a wire's two ends, and once both ends are past the clamp they are
   * both at exactly zero — identical, however hard the wires pull. So
   * negative grip buys drift at the cost of the one term that made drift
   * directed.
   *
   * Two jobs at once even so, which `docs/concepts.md` asks to be visible. It
   * sets a net's stroke, and it sets how far a *loner* coasts — a hungry body
   * and a fed one no longer swim alike.
   *
   * Nothing happens in a pond that is topped up: a stroke needs a difference
   * in fullness across a wire, and `full_mean` at 1.0 says there is none.
   * Read `demand_mean` before reading `net_drift`.
   *
   * With `transportRecoil` this is now the whole of locomotion, and the two
   * together need no phase. A recoil is an impulse pair; grip makes the two
   * ends coast different distances from it, so the pair's centre ends up
   * `|p| * (1/k_sender - 1/k_receiver) / (m_sender + m_receiver)` along the
   * sender's recoil — first order in the impulse, and settled per transfer
   * rather than per cycle.
   *
   * A contracting wire was built beside it and withdrawn. `wireTug` shortened
   * the wire a transfer had just crossed and `wireTugTau` relaxed it again,
   * which reads like an inchworm and is not one: both the pull and the grip
   * were driven by the same packet at the same instant and then decayed, so
   * the loop they traced in (length, grip) closed on a line and its area was
   * only whatever the difference between two relaxation times left behind.
   * A gait needs two degrees of freedom with a phase between them that
   * something can *set*; a packet clock plus two exponentials is one degree
   * of freedom and a race. It also spent the wire's rest length, which is
   * what decides whether two bodies ever meet — turned up far enough to swim
   * it held every pair too tight to rewrite. See `docs/concepts.md`.
   *
   * Global for now, unlike `transportThrust` and `transportRecoil` beside it,
   * which are heritable. Making a lineage's own grip heritable is the obvious
   * next move and costs a genome head; it is worth spending once a sign is
   * known to carry a net at all.
   */
  grip: number;
  /**
   * How fast the pathway runs, as a plain multiple. 0 = no metabolism, so no
   * gait, and no pathway spend either — which is what it ships at.
   *
   * Multiplying every reaction rescales time and nothing else, so this moves
   * the period without moving where the pathway oscillates.
   *
   * The master switch for the whole gait, and the one dial that has to be
   * moved to see any of it: at zero `advanceGait` returns with `gaitWave` and
   * `anchor` at zero, so the stroke is exactly 1 and `grip` reads what it read
   * before the pathway existed. The pond is bit-for-bit yesterday's.
   *
   * It ships there because the pond at 6 is not yesterday's and the amount by
   * which is not small. The suite, which is only a change detector, still says
   * it plainly: a fresh era-era latch peaks at 239 rad/s of spin against a
   * bound of 20, and a body standing still loses a sixth of a full tank inside
   * a couple of seconds to substrate it buys and cannot help buying. Neither
   * is a gait; both are a stroke amplitude and a period picked to be seen
   * rather than measured, driving a span constraint stiff enough to answer
   * every one of them.
   *
   * Finding the value it should ship at is a `npm run pond` job — minutes of
   * pond, the seed budget its measure needs — and not a thing the suite can
   * answer. Until that sweep has run this is the honest place for it, and the
   * slider is right there for anyone who wants to watch it move.
   */
  metabolicRate: number;
  /**
   * What a unit of digested food is worth in the reactor, as `B` per unit of
   * tank energy. The doc's `r1` yield.
   *
   * `runDigestion` splits what it converts between the tank and the reactor,
   * and `intake` is the share each body routes to its own. This is the
   * conversion on that share, and it is large because a reactor turns its
   * pool over many times per unit of matter consumed — that is what a
   * *currency* is. It replaces `metabolicSupply` and `metabolicCost`, which
   * were the same conversion written as a purchase: the reactor used to buy
   * fuel out of the tank, so food ran grid -> gut -> tank -> reactor and the
   * tank sat in the middle of a round trip. Now the cost of running a
   * metabolism is the food it did not bank, which needs no price at all.
   */
  metabolicYield: number;
  /**
   * Seconds a body's primer may stay at absolute depletion before it dies.
   * 0 = nothing starves to death.
   *
   * The doc's §7.2, and it is what lets `upkeep` go. A standing charge was
   * the only thing that could push a tank past its floor, so rent was the
   * only thing in the pond that killed anything; the metabolism draws in
   * proportion to what a body is holding, so it empties a body and then
   * stops. This makes running out *be* the death.
   *
   * Long against a burst and short against a life: a fed body's primer never
   * comes near the threshold between bursts, and a body with nothing to eat
   * reaches it within a second or two and then has this long to find food.
   */
  starveTime: number;
  /*
   * The reactor's rate constants, in the doc's own order. `docs/scratch.txt`
   * §3 gives four reactions over a four-chemical state vector, and §8 says
   * they are global: the stoichiometry and the pace are laws of the world,
   * and a lineage that changed them locally would change what a wavelength
   * means across its own net. What a body owns is its intake — see `intake`,
   * which is §8's per-agent `J`.
   *
   *     r1  A -> B            digestion, at the recipe's own rate
   *     r2  -> C              metabolicCat    * B * (base + C) / (1 + sigma*C)
   *     r3  C -> D            metabolicReset  * C
   *     r4  B + D ->          metabolicQuench * B * D
   *
   * plus a uniform outflow `metabolicDecay` on all four, which the doc's
   * diagonal decay matrix supplies and which is structurally required: `D`
   * has no other sink, so at zero decay it grows without bound.
   *
   * **Whether this oscillates at all is a narrow question, and the doc's own
   * constants get it wrong.** A's row decouples — it is fed from outside and
   * only feeds B — so the dynamics live in the 3x3 over (B, C, D), which is a
   * negative feedback loop B -> C -> D -| B carrying a positive self-loop on
   * C from the autocatalysis. Writing `p` for B's removal, `q` for C's *net*
   * removal once the autocatalysis is subtracted, `r` for D's and `L` for the
   * loop gain, the characteristic polynomial is `(x+p)(x+q)(x+r) + L`, so a
   * complex pair crosses into the right half plane when
   *
   *     L > (p+q+r)(pq+pr+qr) - pqr
   *
   * With three equal stages that is the familiar factor of eight. Here they
   * are nothing like equal, and the useful reading is the limit where the
   * autocatalysis nearly cancels C's own removal: as `q -> 0` the condition
   * collapses to `metabolicReset > (2 + 2*sqrt(2)) * metabolicDecay`, about
   * 4.83 times. The python in `docs/scratch.py` has 0.4 against 0.483 — just
   * the wrong side of it, which is why its reactor settles at every setting,
   * and it never notices because it only ever prints its gates.
   *
   * The other half is that `q` can only be driven to zero when the saturation
   * is weak. At the fixed point, C's self-gain has no free parameters left in
   * it at all:
   *
   *     f = (k3 + d) * C* (1 - sigma*base) / ((base + C*)(1 + sigma*C*))
   *
   * — independent of the fuel, of B, and of `metabolicCat`. So turning the
   * autocatalysis up does nothing: it moves C* and B* together and `f` is
   * flat. Its maximum over C* sits at `C* = sqrt(base/sigma)`, which at the
   * python's `base` 0.02 and `sigma` 0.5 leaves `q` stuck near a fifth of
   * `k3 + d`. That is why both ship small.
   */
  metabolicCat: number;
  metabolicReset: number;
  metabolicQuench: number;
  /**
   * The saturation that caps the autocatalysis, the doc's §3.1.4 sigma.
   *
   * The doc wants it above zero to stop runaway. It is above zero, and small,
   * because it is also what holds the reactor *stable* — see the rate block
   * above. The excursion is bounded by the loop instead: C makes D, and D
   * quenches the B that C is made from.
   */
  metabolicSigma: number;
  /**
   * Uniform outflow on all four species, per unit of reaction time.
   *
   * Not the decay the standing rules forbid. That one is about the pond —
   * lineages, learned weights, memory — where forgetting erases a difference
   * something paid for. This is a well-stirred vat's outflow, it is what
   * closes the loop, and `D` has no other sink at all: at zero it grows
   * without bound and there is no fixed point to oscillate around.
   */
  metabolicDecay: number;
  /**
   * The basal term in the autocatalysis, `base + C` where the doc has `C`.
   *
   * Not in the doc, and required by it. The doc's own `r2` carries a factor
   * of C, so C = 0 is absorbing — and the decay above drives C to zero, so a
   * reactor that ever empties stays empty. That is the same flatline Selkov's
   * equations gave this pond, arriving by the same door. Small enough not to
   * move the fixed point, large enough that nothing is ever permanently dead.
   */
  metabolicBase: number;
  /**
   * Half-saturation of the stroke on the catalyst: the wave is
   * `2C/(metabolicWave + C) - 1`.
   *
   * The doc's §5.2 drives contraction by C directly and unbounded. A rest
   * length needs a bounded, signed multiplier, and this is the same
   * Michaelis form the doc uses elsewhere, so a body empty of catalyst reads
   * -1, one holding `metabolicWave` reads 0, and a saturated one approaches
   * +1 without ever asking a span constraint for more than it can give.
   */
  metabolicWave: number;
  /**
   * Fuel burned per unit of stroke, per unit of reaction time.
   *
   * What makes moving cost something: the stroke draws A down in proportion
   * to how far it is actually swinging a wire, so a net that undulates hard
   * runs its fuel down, pulls harder on its tank, and has to eat.
   *
   * It no longer has a second job. Under the two-pool pathway the stroke's
   * own cost was also what made the steady state non-zero, so taking it to
   * zero stopped the clock — measured, and it cost an afternoon. The
   * four-species reactor oscillates on its own loop, so this is a price and
   * nothing else.
   */
  metabolicWork: number;
  /**
   * How fast a body broadcasts down its principal wire, per second per unit
   * of what it is holding above the gate. 0 = every body's reactor is its
   * own.
   *
   * The doc's §4 transmission rate. *Which* species a body broadcasts is its
   * `Tx` gene, seeded by kind — a Con sends the catalyst, a Dup the
   * inhibitor, an Era the fuel and the primer it ate — and this is the one
   * global scale on all of them. Mass action on the concentration, so the
   * impulse is in the chemistry: a body sends most at its catalyst's peak
   * and nothing at its trough, with no clock and no threshold needed to make
   * it a pulse.
   *
   * **There are three regimes and the middle one is the wave.** Measured on
   * six Cons wired principal to auxiliary, grazing regrowing ground rather
   * than fed by hand, against a period of about 161 frames, reading the
   * signed phase difference per interior wire and its spread:
   *
   *     0     26, 27, -21   spread 48 — no lock; they never converge
   *     0.1   14, 20, -3    spread 23 — partial
   *     0.4   8, 11, 1      spread 10 — partial
   *     1     3, 7, 4       spread  4 — a wave, a few per cent of a cycle a wire
   *     3     -3, 2, 1      spread  5 — synchrony
   *     8     -12, -2, -2   spread 10 — the senders start to run down
   *
   * It ships at 1, the cleanest lock that still carries a phase. The upper
   * edge is the reactor's: broadcasting spends the very catalyst the sender's
   * loop runs on, so a chain shouted through loudly enough stops travelling
   * and then stops oscillating.
   *
   * The band moved when the reactor started being fed by eating rather than
   * by a hand-forced tank, and the old figures are not comparable: a
   * force-fed body runs its reactor hotter than one grazing for itself.
   *
   * A body at the end of a chain whose principal is free only ever *receives*
   * and can be driven into the inhibited state: the catalyst it is fed runs
   * to D, D quenches the primer, and its own loop stalls. That is the doc's
   * stoichiometry rather than a tuning failure, and in a grown net it is a
   * body waiting to latch rather than a resting state.
   *
   * The slow coupling is already there and costs nothing: two wired bodies
   * eat from ground that `flowCharges` and the harvest both reach, so their
   * reactors are coupled through the economy whether this is set or not.
   */
  metabolicDiffuse: number;
  /**
   * How much of a species a fresh body has to be holding before it broadcasts
   * any of it, **as a fraction of that species' own resting level**. Seeds all
   * four of `Gx` and is heritable per species from there.
   *
   * The doc's §4 `H(x_j - G_j)`, rectified rather than a step. It shipped at
   * zero and gated nothing, and the reason was the units: one flat number
   * across four pools whose scales differ by orders — the catalyst sits near
   * 1, the inhibitor at `k3/d` times that, the primer at `(k3+d)/k2` — is
   * either inert or shut, never a gate. `seedGait` scales it per species now,
   * so this means one thing: *stay quiet below this much of your own resting
   * level*.
   *
   * The broadcast was already a pulse — mass action sends most at a peak and
   * least at a trough. This gives the pulse a **floor**, so a body at its
   * trough says nothing at all rather than a little, and that is what keeps a
   * front from filling the gap behind it.
   *
   * **It is what stops a loud chain collapsing into synchrony.** Ungated,
   * turning the broadcast up locks a chain harder until the phase difference
   * goes to nothing — the pond-wide pulse this branch began by removing, and
   * the reason the broadcast rate sat near the bottom of its range. At 0.1 the
   * same chain holds 2.33 frames a wire.
   *
   * **The window is narrow and the far edge is measured.** 0.05 through 0.2
   * all travel; **0.3 flatlines the chain**, because a gate breaks the
   * property mass action was chosen for — a body sending in proportion to what
   * it *has* cannot send itself empty, and one sending in proportion to
   * `x - gate` can drive itself down to the gate. That is the same failure the
   * old wave-gated broadcast had. It ships at a third of where it breaks.
   */
  metabolicGate: number;
  /**
   * How fast a body feeds its reactor, in matter per second. The one thing
   * about its own clock a lineage owns, and heritable like every other trait.
   *
   * A *rate*, not a share of what it digested. A share makes the period a
   * property of the dish — a body on rich ground routes more and saturates,
   * one on poor ground starves — and it only looked right while the tank had
   * a continuous outflow keeping throughput small. A rate makes the period a
   * property of the body, which is what `docs/metabolism-spec.md` §1 asks for
   * first.
   *
   * The influx the reactor sees is `intake * metabolicYield / metabolicRate`,
   * and the fuel window in those units is 0.55 to 3.05 — so this oscillates
   * between about **0.017 and 0.095**, and outside it a body is quiet below
   * and saturated above. Measured on a lone pinned Con: 0.10 gives a 1.54 s
   * period, 0.06 gives 2.13 s and the widest swing, 0.04 gives 2.43 s, and
   * 0.25 is glutted and still. It ships at 0.06, mid-window.
   *
   * Still fed by eating: a body can only route what it has digested, so a full
   * tank and an empty gut is no clock at all however high this is set.
   */
  intake: number;
  /**
   * How far the reset inhibitor **D** swings a body's grip, either way.
   *
   * `grip` makes a body's drag depend on how full it is, and that is the only
   * thing that turns a stroke into travel: a transport kick is equal and
   * opposite, so it cancels at the centre of mass unless the two ends damp
   * differently. This modulates that difference with the gait, so a body grips
   * hardest at one point in its cycle and slips at another — which is what an
   * inchworm does, and what a standing difference in fullness cannot do on its
   * own.
   *
   * **Driven by D and not by C, and that is the whole point.** The rest length
   * and the anchor both swing on `wave(C)`, so they are one degree of freedom:
   * a cycle that deforms and undeforms through the same shapes is reciprocal,
   * and a reciprocal cycle nets zero displacement — Purcell's scallop theorem,
   * and the reason `wireTug` was withdrawn ("one degree of freedom and a
   * race"). A second actuator has to be *out of phase* with the first, and the
   * reactor already supplies that: `dD/dt = k₃C − dD` is a first-order lag, so
   * D trails C by `atan(ω/d)` — **51° at the bottom of the fuel window and 65°
   * at the top**, set by the chemistry and by nothing anybody dialled.
   *
   * D is read in catalyst units, `D · d / k₃`, which is what D would be if the
   * loop stopped — so it shares `metabolicWave` as its half-point and needs no
   * constant of its own.
   *
   * At 0.5 a full body's grip runs between half and one and a half times
   * `grip` over a cycle. At 1 it falls to nothing at the trough.
   *
   * **It only pays once something times the kicks.** A cyclic grip has to be
   * phase-locked to whatever delivers the impulse, and until the gait's own
   * broadcast recoiled (`Sim.advanceGait`) the only impulse in the pond came
   * from the demand gradient — the economy's clock, not the chemistry's. An
   * uncorrelated zero-mean modulation on a steady asymmetry is a *loss*,
   * because travel goes as `1/k` and that is convex. Measured on the worm
   * bench, which is deterministic so these are exact: with the broadcast
   * inert, 4.22 px/s at swing 0 falling to 1.88 at 1. With it kicking, 3.97 at
   * 0 and **5.09 at 0.75** — a 28% gain, with an optimum rather than a slope.
   * It ships at 0.5, one notch conservative of that.
   */
  gripSwing: number;
  /**
   * How far the gait swings a wire's rest length, as a fraction of it.
   * 0 = the wire ignores the clock, which is the pond before this.
   *
   * The visible half of the stroke, and it is a separate mechanism from the
   * impulse on purpose because this engine will only give one thing each.
   * `Sim.strokeWires` is what moves a net: an impulse is the only actuator a
   * centre of mass responds to, since `grip` works on the coast afterwards.
   * But a wire's span constraint is near-rigid, so it puts the two ends back
   * where `rest` says inside the same frame — the net walks and its shape
   * never changes, which is why the gait has been invisible.
   *
   * Driving `rest` is the exact reverse: the constraint serves it, so it
   * reads at once, and it is worth no travel at all. So both, off one phase
   * and agreeing — the ends pull together as the muscle pulls, and a chain
   * with a phase lag along it shows a wave running down its length.
   *
   * Sized to be seen rather than to be safe: 0.3 swings a 48 px wire between
   * about 34 and 62. `wireBreathe`, which this rides on top of, is 0.04 —
   * two pixels, and never meant as a gait.
   */
  gaitSwell: number;
  /** Shoaling. See `declutter` for what this costs reproduction, and why the
   *  two only matter together. */
  flockAlign: number;
  flockSep: number;
  maxAgents: number;
  soupCount: number;
  /** Seconds between automatic free-agent spawns (0 = off). */
  spawnInterval: number;
  /**
   * World-space size of one energy cell. Kept a whole multiple of the scent
   * field's cell (10 units) so the two grids line up — an energy cell is a
   * 4x4 block of scent cells rather than a lattice at an unrelated pitch.
   */
  energyCell: number;
  /**
   * Free energy in an unvisited cell. A cell holds a whole extra, so an agent
   * arriving on untouched ground fills in one step and the grid, not the
   * charging rate, is what the net is competing over.
   *
   * **Halved, with `energyRegrow` and `energyDiffuse`, so that the patches
   * stay visible.** The three of them together decide whether the dish the
   * seed lays down survives being lived on, and at the old settings it did not:
   * measured over three seeds, a dish that starts 70% bare with a spatial
   * coefficient of variation of 1.64 was flat inside thirty seconds and stayed
   * that way — 15% bare and cv 1.50 at two minutes, *below* what it was seeded
   * with. Diffusion put a crumb in every cell and regrowth, which is
   * proportional to what is already there, inflated every crumb to capacity.
   * A patchy seed and a uniform pond.
   *
   * See `energyDiffuse` for the three settings measured and what each bought.
   */
  ambientEnergy: number;
  /**
   * How fast the ground spreads, as a multiple of the scent `diffuse` slider.
   *
   * Small, because energy is not a smell. A signal wants to reach across the
   * dish inside a second — that is what makes a trail worth following. Ground
   * that did the same would be a single shared pool with no local scarcity in
   * it, and nothing to forage toward. This is the number that decides how far
   * a grazed patch can draw on its neighbours, and so how big a dead zone a
   * net can make before it has to move.
   *
   * **This, `energyRegrow` and `ambientEnergy` are one setting in three
   * numbers**, and they were set when the dish was uniform and nothing about
   * them showed. On a patchy dish they decide whether the patches survive
   * being lived on. Three seeds, two minutes, 400 founders — `bare` is the
   * share of the disk under a twentieth of capacity, `cv` the spatial
   * coefficient of variation of the ground, `forage` whether bodies stand on
   * better-than-average ground:
   *
   *     ambient/regrow/diffuse   bare    cv   bodies  lines  forage
   *     1.0  / .04 / .05         0.15  1.50     623    192    1.19
   *     0.5  / .02 / .006        0.28  2.94     425    172    1.65
   *     0.4  / .01 / .002        0.37  3.50     386    142    1.09
   *
   * The middle row ships. The seeded dish is cv 1.64, so the old settings end
   * *below* what they started with — washed flat — while these end well above
   * it: the structure at two minutes is not the seed surviving, it is grazing
   * carving new holes faster than the ground closes them.
   *
   * The bottom row is not simply more of a good thing, which is why it is here
   * and not shipped. It has the barest dish and the highest contrast and the
   * *worst* foraging — under 1, so bodies are on worse ground than average —
   * and it loses a quarter of the founder lines. Past some point there is not
   * enough food to be worth finding, the clusters are evident and nothing is
   * standing on them. The middle row is where the contrast doubles and the
   * bodies are most clearly *on* it.
   */
  energyDiffuse: number;
  /**
   * Logistic regrowth rate, per second, toward `ambientEnergy` per cell.
   *
   * Not a refill timer. Growth is proportional to what is already in the cell,
   * so a cell taken to exactly zero never comes back on its own and has to be
   * recolonised from a neighbour — grazing to the floor makes a scar that
   * heals from its rim at the speed `energyDiffuse` sets. 0 turns the ground
   * back into the seam of ore it used to be.
   *
   * Halved with the other two; the table on `energyDiffuse` is the measurement
   * for all three. This is the one that does the filling: diffusion only has
   * to put a crumb in a cell, and logistic growth — proportional to what is
   * already there — takes the crumb the rest of the way to capacity. Lowering
   * the spread alone leaves every cell reached and then filled.
   */
  energyRegrow: number;
  /**
   * A standing rent on the tank, per second. **Back on at 0.01**, for the one
   * job of its four that nothing else took over.
   *
   * It was one dial doing three jobs. *The standing cost* is the reactor's own
   * outflow: a body must keep eating to hold its pools against `metabolicDecay`
   * and one that stops dies on the starvation window, which on a bare dish
   * fired first anyway (28.8 s against the rent's 69.0 s). *The Era's income*
   * was this rent paid negative, which mattered when a producer had no other
   * income; now that the reactor is fed by eating, an Era's income is its
   * uptake, and measured across three seeds the Era share does not move when
   * the rent goes (28.9% to 30.8%). *Rationing rewrites* belongs to the
   * rewrite's own price, and that is `bodyValue`.
   *
   * What the rent was also doing, unbilled, was **holding nets shallow**, and
   * that is the job it is back for. It charges every body in a net whether or
   * not that body is doing anything, so depth costs and a lineage that grows
   * one bleeds. With it off, dense fast-growing nets arose and then would not
   * die: nothing charges a body for merely existing, so a net that stops
   * growing is free, and `upkeepExcrete` at 1 returns its reactor's food to
   * the cell it is standing in, so it restocks its own pasture and no body in
   * it is ever hungry enough to start the §7.2 clock. Measured, a 40-body knot
   * plateaued at 84 and sat there for four minutes with the clock reading zero
   * the whole time.
   *
   * Rent is what makes that cost something, and it is density-dependent for
   * free: at `upkeepExcrete` 1 the rent lands back on the ground under the
   * body, so a lone body re-eats its own rent while a crowded one recovers
   * only its share of it — the harvest is rate-limited and shared. Three
   * seeds, 120 s, 400 founders, `biggestNet` being the largest connected
   * component:
   *
   *     upkeep   bodies  lines  wires/body  biggestNet  tank  forage
   *     0           424    171        0.98         146  0.46    1.65
   *     0.005       312    167        0.91          88  0.59    1.44
   *     0.01        277    189        0.89          82  0.65    1.40
   *     0.02        218    158        0.84          31  0.74    1.27
   *
   * `biggestNet` is monotone in it and nothing else here is, which is what
   * says this dial is the one that governs runaway size. 0.01 nearly halves it
   * and has the *most* founder lines of the four — capping the runaway net is
   * what leaves room for the others. 0.02 crushes it to 31 and costs lines.
   *
   * At 0.01 a full body has 200 s of arrears before `debtCap` kills it, which
   * sits deliberately outside the 25 s starvation window: rent is the slow
   * pressure and going hungry is the fast one, and a body can be killed by
   * either.
   *
   * Hitting `debtCap` still kills; nothing else charges rent to do it.
   */
  upkeep: number;
  /**
   * Extra per second per unit of speed, charged for moving.
   *
   * What ties a net's energy to its locomotion, and so what lets a net have a
   * motor at all. Swimming was free, which made thrust a property of a body
   * rather than of the net that feeds it: a starving swimmer swam exactly as
   * hard as a full one, and there was nothing for the transport machinery to
   * be *for* beyond keeping redexes alive.
   *
   * With a price on it the wire network becomes a fuel line. A sub-net that
   * swims runs itself down and asks; `spreadRequests` carries the ask inward
   * and `flowCharges` sends stock back out; `applyTransportRecoil` already
   * kicks the pair as it goes. Which bodies a net chooses to feed is which
   * way it goes — and because `transportThrust`, `transportRecoil` and
   * `requestDecay` are all heritable, what a lineage does with that is
   * something it can evolve rather than something set here.
   *
   * Off by default. Turning it on is a real change to the economy: at a
   * cruise of 38 and an upkeep of 0.015, a cost of 0.0004 roughly doubles
   * what a moving body pays to exist.
   */
  swimCost: number;
  /**
   * Fraction of this body's own tank a rescue fills, from `debtCap` at 0 to
   * `energyCap` at 1. The absolute extra it asks up to is
   * `debtCap + rescueTo * (energyCap - debtCap)`, so it cannot land outside
   * the tank.
   *
   * 0 is the old ambulance that stops at break-even once extra is no longer
   * negative. 1 strips the neighbourhood to top the patient off. The seed is
   * a little under a full tank, so a default Con comes out of rescue able to
   * pay a rewrite share without emptying every reservoir it can reach.
   *
   * Heritable: this slider only seeds a fresh body. A commute recombines the
   * two parents' fills; an erase clones the Era's fill with a mutation nudge.
   */
  rescueTo: number;
  /**
   * What a fresh body's `assort` starts at: how likely each gene of its
   * children is copied whole from one parent rather than blended.
   *
   * 0.5 and not by accident. `assortChance` offsets by the child's kind, so at
   * 0.5 a Con child blends every gene and a Dup child assorts every gene —
   * exactly the absolute rule this replaces. Nothing changes until the trait
   * drifts, and then both kinds move together.
   *
   * A seed like the other heritable traits: read once by `createAgent`, never
   * again.
   */
  assortBias: number;
  /**
   * Extra at which a fresh body dies. Always negative — a debt depth, never
   * a second positive cap. The live value lives on the body as `debtCap` and
   * drifts by breeding; this is only the seed.
   *
   * Deeper debt is more time for a rescue and a louder hunger, and a corpse
   * that can leave nothing. Shallower debt dies sooner and leaves more on
   * the ground.
   */
  debtCap: number;
  /**
   * How much of a body's demand its neighbour hears, per wire. 1 = no decay.
   *
   * Sets how far a shortage is audible — against the field's floor, a whole
   * unit of need carries 20 hops at 0.8 and 43 at 0.9 — and how much crosses,
   * since a transfer is capped by the field at the receiving end. Turn it down
   * for a net of local pools that each look after their own; turn it up for one
   * that answers a shortage anywhere on it.
   *
   * The slider stops short of 1 because an undecayed field is a flat one: every
   * body holds the same need, no neighbour is strictly needier, and transport
   * stops dead.
   *
   * Heritable: this only seeds a fresh body's own `requestDecay`. Once alive,
   * a body relays demand at its own rate, and a Con+Dup commute recombines
   * the two parents' rates into each child (blended for a Con child, one
   * whole parent's rate for a Dup child) — this slider just sets where a new
   * population starts and what a mutation is centred near.
   */
  requestDecay: number;
  /**
   * Momentum a body recoils with per unit of energy it pumps to a neighbour.
   * 0 = off.
   *
   * Stable well past the slider's range — a seeded soup is still calm at 800
   * and only comes apart near 3000. The ceiling is low because the visible
   * events are one-off transfers of most of a unit, and 20 makes one of those
   * a ~56 px/s nudge on an Era against settled speeds around 50.
   *
   * Heritable, like `requestDecay` above: a pump's actual kick is its own
   * `transportRecoil`, seeded from this slider and free to drift by breeding.
   */
  transportRecoil: number;
  /**
   * Whole units a transfer moves along a wire. 0 = the continuous law, which
   * is the pond as it was.
   *
   * Transport has always been a trickle. A body gives whatever the demand
   * gradient asks for that frame, and at steady state that is about 4e-4 —
   * so `transportRecoil` at its default of 100 delivers an impulse of 0.04
   * against settled speeds of 20 to 60 px/s. Two thirty-trial sweeps found
   * neither `transportRecoil` nor `transportThrust` registering a resolved
   * effect on anything at all, and `grip` turned out to be damping the dish
   * rather than swimming it: its effect on `net_drift` was the same size with
   * the kicks switched off. The momentum machinery is not wrong, it is three
   * orders under its own noise.
   *
   * A packet is what puts it above. Above zero a transfer is one whole unit
   * or nothing, and only a body already holding a whole unit can send one, so
   * a donor accumulates, fires, and accumulates again. At 0.5 against a tank
   * of 1.25 that is a kick of 50 rather than 0.04.
   *
   * It also gives `grip` the clock it was missing. A net under a standing
   * gradient has a nearly static fullness pattern, so grip anchors one end
   * and the other pivots — which is what a pond at grip 100 visibly does.
   * Packets make the pattern *travel*: the sender drops a whole unit, the
   * receiver gains one, and the packet hops along the chain. A moving anchor
   * is what a crawler has and a static one does not.
   *
   * Neither the receiver's demand nor its room bounds the amount any more.
   * Demand still decides *whether* to send — a transfer needs a neighbour
   * strictly needier than the donor — but a whole packet crosses either way,
   * and what will not fit is deposited on the ground under the receiver, the
   * same place a rewrite's leftovers go. This is the second path that can
   * overfill a body; `payUpkeep` was the first.
   *
   * Heritable, like `transportRecoil` and `transportThrust` beside it: this
   * only seeds a fresh body's own quantum, and the flow law reads whichever
   * body is *sending*. Moving the slider does nothing to anything already
   * alive.
   *
   * That is the interesting half. A body accumulates until it holds a packet
   * and then fires, which makes it a relaxation oscillator whose period is its
   * quantum over its income — so bodies with *different* quanta are coupled
   * oscillators at different natural frequencies, which is the standard
   * account of how gut peristalsis comes to travel one way rather than pulse
   * in place. Measured on the worm bench with a source at one end and a sink
   * at the other and nothing else imposed: uniform quanta swim at 0.115 px/s,
   * which is the bench's noise floor, and quanta varying along the body swim
   * at 0.93 to 1.27. The wave does not have to be tidy — a scattered mix with
   * barely a consistent direction still swims — but it does have to vary
   * *along* the line of travel. Varying it across the body buys nothing.
   *
   * Two things to watch. A chain of bodies all holding less than a packet
   * cannot feed each other at all, where the continuous law would have shared
   * out thin — so read `died` and `can_pay`, and keep this well under
   * `REWRITE_SHARE`. And the throughput the bench needed for those speeds was
   * tens of thousands of units in thirty seconds, which is a pair swapping a
   * packet back and forth near frame rate rather than anything a pond's
   * income could pay for.
   */
  transportQuantum: number;
  /**
   * How much of the receiver's kick is withheld, 0–1, and therefore how much
   * of a pump's recoil survives as motion of the pair.
   *
   * At 0 the pump is equal and opposite and a net can never shift itself by
   * moving energy around inside itself. At 1 only the sender is kicked, so the
   * pair — and any net holding a standing gradient — drifts back along the
   * wire, against the direction the energy is flowing. That is the swimming
   * stroke: pushing charge toward the hungry end pushes the body the other way.
   *
   * Drag bounds it, so this is a cruising speed rather than an acceleration.
   * One whole unit pumped between an Era and a Con at the default recoil
   * leaves the pair drifting `20 * thrust / (0.45 + 1)` px/s — ~7 at the
   * default, ~14 at 1 — against settled speeds around 40. In a live soup that
   * is invisible for the same reason the recoil gain is: mean speed over a
   * seeded 30 s soup at 0 / 0.25 / 0.5 / 1 is 40 / 60 / 34 / 47 px/s, which is
   * seed noise. It reads on the events, and on a net actually holding a
   * gradient — drive one end full and the other hungry and the chain visibly
   * runs away from its own supply.
   *
   * Heritable, like the two above. A transfer reads the *sender's* own
   * `transportRecoil` and the *receiver's* own `transportThrust`, so breeding
   * a strong low-thrust pump against a high-thrust receiver can drift a
   * net's stroke somewhere neither parent line swims alone.
   *
   * **Stays at 0, and the measurement is the reason rather than an oversight —
   * but it is a weak reason.** Three seeds at 120 s on the conservative patchy
   * dish:
   *
   *     thrust    0        0.25            0.5
   *     bodies  621        332             490
   *      seeds  603/613/647  291/239/467   287/297/887
   *      lines  192        193             209
   *
   * Nothing is resolved there. The spread *within* an arm is wider than the
   * gap between arms — 287 to 887 inside one of them — which is this pond's
   * standing lesson about three seeds, and the founder lines, which are
   * steadier, barely move at all. So this is not "0 is better"; it is "0 is
   * where it was, and nothing here is evidence for moving it".
   *
   * What makes 0 a *seed* rather than a ban is that `P` is inside the learned
   * block now. The base is what a fresh body has; the row off `h` moves within
   * a life, so a lineage in a pond where running away from its own supply pays
   * can find its own thrust without every other body paying for it, and
   * without a sweep having to resolve a global first.
   */
  transportThrust: number;
  /**
   * How fast a body's state matrices change while it is alive. 0 = off, and
   * off is exactly the simulation as it was.
   *
   * A body is a small recurrent network whose weights have until now been
   * fixed from birth: everything it knows, it inherited. This is the other
   * way information can get into one. Each weight keeps an eligibility trace
   * of what it was lately doing, and a temporal-difference error from the
   * body's own critic says whether that was better or worse than expected;
   * the product is the update. Three factors, all local to the body, no
   * error signal handed down from anywhere.
   *
   * What is learned is kept for life and goes with the body when it detaches
   * and latches somewhere else. That is the point rather than a side effect:
   * it makes an agent that has been through a net different from one that
   * has not, and it is how one net's experience reaches another.
   */
  learnRate: number;
  /** How fast the critic itself learns to predict. Its own delta rule. */
  learnCritic: number;
  /**
   * How hard a body jitters its own head outputs so that it can learn them,
   * in the heads' own dimensionless units. 0 freezes every head at whatever
   * the genome says and learning goes back to the recurrent core alone.
   *
   * This is the exploration half of node perturbation, and it is not optional
   * noise: it is the *signal* the head learning rule correlates against. Each
   * of a body's fifteen outputs is displaced by its own reproducible draw
   * (`exploreAt`), the eligibility trace remembers that draw against the state
   * that produced it, and the critic's error says whether the displacement was
   * worth keeping. Without it every head row would move on the same sign and
   * no body could discover that it should swim faster while turning less.
   *
   * Small on purpose, and smaller than it started. The heads are dimensionless
   * and `HEAD_SCALE` converts them, so 0.02 is about 0.8 px/s on a 38 px/s
   * cruise and about 0.16 on an alignment gain of 5.5.
   *
   * It shipped at 0.05 for one afternoon, where the pond was indifferent —
   * population and founder lines level with a no-learning arm over three seeds
   * — but `scent-steering.perf.test.ts` was not. Twenty-four bodies climbing a
   * gradient on a barren dish over ten seconds closed, against a bar of 4%:
   *
   *     explore   0     0.01   0.02   0.03   0.05
   *     closed   pass    3%     1%     3%    -2%
   *
   * Above 0.03 the jitter is enough to walk a body off a gradient it was
   * following, and the rig is the worst case for the rule by construction:
   * with `ambientEnergy` at 0 nothing a body does improves its tank, so the
   * critic's error is a standing negative and the perturbation anti-reinforces
   * whatever the body was doing — including the taste it was seeded with.
   * Turned down rather than argued with, which is what this project does with
   * a mechanism that costs a behaviour it already had.
   */
  learnExplore: number;
  /**
   * What a body is taught by: the level of its tank, or the rate it is filling
   * at. 0 is all level, 1 is all rate.
   *
   * **This is the one designed objective in the pond, and it should be said
   * out loud.** Everything else here is emergent — there is no fitness term
   * and selection is what survives — but a learner needs a teacher, and this
   * is it. What keeps it honest is that it is not arbitrary: `IN_FULL` is
   * distance from `debtCap`, which is the system's own death condition, so
   * the reward is "how close am I to dying" rather than a goal anybody chose.
   *
   * The level signal is `IN_FULL - 1`, in `[-1, 0]`. It can say *you are
   * short* and never *you are doing well*, and it saturates: measured in a
   * live soup at 60 s, the median body reads 0.342 but **31.7% sit at exactly
   * 1**, so a third of the pond has a reward of zero and no gradient at all.
   * Learning stops exactly where behaviour is succeeding.
   *
   * The rate signal is the change in `IN_FULL` per second, clipped to the same
   * `[-1, 1]`. A body at its cap that is spending still reads negative, and
   * one that is gaining reads positive, so the dead zone goes. It is also very
   * nearly the potential-based shaping of the level signal — with a discount
   * near one, `gamma*PHI' - PHI` is the difference — which is the classical
   * result that says shaping this way cannot change which policy is best,
   * only how fast it is found.
   */
  learnReward: number;
  /**
   * Eligibility trace decay, per frame. Not weight decay — nothing here
   * forgets. This is the credit window: how far back a weight is still
   * held responsible for what the body's tank is doing now. 0.95 is about
   * twenty frames, which is roughly how long a transfer takes to land.
   */
  learnTrace: number;
  /** Discount on the critic's own prediction, per frame. */
  learnDiscount: number;
  /**
   * How unevenly a rewrite's children divide the reactor their parents were
   * running, 0 to 1.
   *
   * **A rewrite used to throw the clock away.** Children are built by
   * `createAgent`, which takes a fresh slot with `B`, `C` and `D` at zero, and
   * nothing carried the parents' across — so every body ever made by a rewrite
   * was born with a dead reactor reading `wave` −1, and had to eat its way
   * back into the fuel window before it could stroke at all. The genome was
   * inherited and the gut was spilled back onto the ground; the one thing that
   * was simply lost was the phase its parents were at.
   *
   * The pools are outside the pond's books, so dividing them is free. What it
   * buys is two things. Children are born **running**, at their parents'
   * point in the cycle rather than from a standstill. And at anything above
   * zero they are born *different*: the split is drawn per species, so two
   * children get different `B : C : D` ratios, and the phase of this reactor
   * is set by those ratios rather than by the size of the pools. An even split
   * would hand every child the same phase.
   *
   * That matters because symmetry has to break somewhere. Identical bodies do
   * develop a phase gradient under the broadcast — measured on the worm bench,
   * 6.6 frames a segment with two thirds of pairs agreeing — but they develop
   * it from whatever difference the coupling can amplify, and a rewrite that
   * hands its children identical states gives it nothing to work on.
   *
   * At 0 an even split. At 1 a child's share of a species runs from nothing to
   * twice its even share.
   */
  reactorSplit: number;
  /**
   * How much of what a parent learned is consolidated into its children's
   * genome, 0 to 1.
   *
   * At 1 a lineage compounds what it learned: a commute's children start
   * from `chem + plastic`, so the worm keeps the experience of the bodies it
   * grew from. At 0 learning is somatic and dies with the body. The children
   * always start with an empty slate of their own; what this scales is how
   * much of their parents' arrived already written into the genome.
   */
  inheritLearned: number;

  /*
   * Chemistry. Every one of these ships
   * at the value that reduces to the behaviour before it existed.
   */

  /**
   * The whole mouthful a body may swallow per second, at saturation: one
   * budget shared across all four species in proportion to what is standing
   * in the cell (see `UptakeKinetics`), landing in the gut, where
   * `digestRate` and the uptake rows decide what any of it is worth.
   *
   * **Zero is the old path**, and that is not the same as "no uptake": at
   * zero, `runHarvestPlan` takes what fits in the tank instantaneously, as it
   * always has, ground alone and straight into the tank. Anything above zero
   * makes uptake a *rate* and a *sample*, which is what gives it a phenotype
   * for selection to grip and what dissolves the id-order artifact — older
   * bodies systematically eating first in a contested cell, a fitness
   * gradient on age that nobody chose.
   */
  uptakeVmax: number;
  /**
   * Half-saturation constant: the ground density at which uptake runs at half
   * `uptakeVmax`.
   *
   * With `uptakeVmax` this is the non-dominating trade-off the plan is after.
   * High/high is a fast grazer that needs rich ground; low/low is a scavenger
   * living on scraps; neither wins everywhere, which is the precondition for
   * coexistence rather than takeover.
   */
  uptakeKs: number;
  /**
   * Hill coefficient on uptake. The other superadditivity dial.
   *
   * At 1 this is plain Monod. Above 1 the response is convex at low
   * expression, which is the other way to make specialising beat splitting.
   */
  hillN: number;
  /**
   * Yield on energy a body takes up directly from the ground, 0 to 1.
   *
   * 1 is today. Moving it toward 0 makes bodies obligately dependent on what
   * a net's Eras bring in — but check the larval window first: a fresh spawn
   * has about `EXTRA_CAP / upkeep` seconds of tank, and if mean
   * time-to-encounter with a net is not well under that, obligate dependency
   * kills the soup rather than structuring it.
   */
  yDirect: number;
  /**
   * Yield on energy an Era takes up, 0 to 1 and normally above `yDirect`.
   *
   * An Era is exactly a terminated port, so the count of them is a net's
   * boundary size while upkeep is charged per body: income scales with the
   * boundary and cost with the volume. Surface-to-volume becomes a real
   * constraint on net size and the only way to get bigger is to get
   * branchier — a morphological pressure the simulation has no other source
   * of. 1 is today, where an Era is no better a grazer than anything else.
   */
  yEra: number;
  /**
   * How much of what crosses out of the pond's books is put back onto the
   * ground rather than destroyed, 0 to 1.
   *
   * Two things cross out: the rent `upkeep` charges, and the food `intake`
   * routes to the reactor — the reactor's pools are in their own units and
   * outside the books, so that food leaves them. At 1 both land back on the
   * ground the body is standing on, which is also what makes the rent
   * density-dependent: see `upkeep`.
   *
   * **1, so a body's *metabolism* neither creates nor destroys matter**: what
   * `intake` routes to the reactor lands back on the ground rather than
   * leaving the books. That is the invariant `docs/metabolism-spec.md` §3.2
   * asks for and the dial used to deny.
   *
   * It is not the same as a conserved *pond*, and an earlier draft of this
   * note overstated it. Measured with every inflow off — no regrowth, no
   * immigration — and counting each living body's own `bodyValue` as well as
   * its tank, a pond loses 8% of its matter in two minutes at `upkeep` 0 and
   * 12% at 0.01. `deathYield` is `max(0, bodyValue + extra)`, so a body dying
   * in debt destroys what it owed; that is the largest sink and it is
   * deliberate — a corpse that starved leaves less than one that did not.
   * A negative `eraUpkeepRatio` is a small source pointing the other way: it
   * adds to a producer's tank without taking from the ground.
   *
   * This shipped at 0 — the reactor a sink, the pond quietly bleeding matter —
   * because turning it on used to lay the food down as the body's own
   * *excretion mix*, which for a Con is signal rather than food. Nothing
   * excretes now and it returns as ground.
   *
   * Measured three seeds at 120 s on the patchy dish, against 0:
   *
   *     bodies 327 -> 621    lines 187 -> 192
   *     forage 0.65 -> 1.19  wires/body 0.86 -> 1.03
   *
   * Every seed above every seed of the other arm, and `forage_ratio` crossing
   * 1 for the first time in anything measured here — bodies standing on
   * better-than-average ground. Read that last one carefully: a body that puts
   * its reactor's food back under itself makes its own cell rich, so some of
   * the crossing is the deposit rather than the finding. The population and
   * the lineage count are not open to that reading.
   */
  upkeepExcrete: number;
  /**
   * What a body's existence is worth when it dies, `EXTRA_CAP` today.
   *
   * **Ships at `REWRITE_SHARE` (1), so the commute-then-annihilate cycle no
   * longer mints.** At `BODY_VALUE` (`EXTRA_CAP`, 1.25) it made
   * `2 * (EXTRA_CAP - REWRITE_SHARE)` = 0.5 out of nothing every cycle, which
   * was the metabolism rather than a slip: a standing rent drained
   * continuously, so a net that kept rewriting fed itself and a net that sat
   * still starved. The rent it balanced was taken to 0, which left the mint
   * making free energy against nothing; `upkeep` is back at 0.01 but for a
   * different job — pricing size — and a mint that pays for rewriting is
   * still not what should offset it. Conserved, so the cycle is neutral and
   * the price of rewriting is stated below rather than fallen into.
   *
   * Below 1 the cycle *costs* `2 * (1 - bodyValue)`, which is where a brake on
   * rewriting belongs — on the rewrite's own price, rather than on a standing
   * charge against every body whether it rewrites or not. Nothing needs one
   * today; the dial is here for when something does.
   */
  bodyValue: number;
  /**
   * How much more an Era holds than a Con or a Dup. 2 today; 1 drops the rule.
   *
   * `energyCap` is already heritable and recombined across a commute's
   * children, so storage is an evolvable axis available to every kind. The
   * kind rule duplicates it with a wall instead of a gradient — and while it
   * is in place nobody can learn whether big tanks actually belong on the
   * boundary.
   */
  eraCapRatio: number;
  /**
   * A *producer's* upkeep as a multiple of everyone else's. Negative today,
   * which means a producer earns rather than pays.
   *
   * No longer keyed on the glyph: `upkeepRateOf` interpolates it on how much
   * of a body's chemical budget goes on the ground's excretion row, against
   * what a seeded Era expresses. A seeded Era still lands on this value
   * exactly and a seeded Con on 1 exactly, so nothing about a fresh pond
   * moved; what is new is that a Con breeding toward making ground earns the
   * discount and an Era abandoning it loses one.
   *
   * At 1 nobody is discounted and income has to come from the ground under a
   * body — which is what makes `#Eras` a net's boundary size against an upkeep
   * charged per body, and surface-to-volume a real constraint on how big a net
   * can get.
   */
  eraUpkeepRatio: number;
  /**
   * What one unit of a signal reading is worth on the way into `x`.
   *
   * A `Params` field rather than the `SENSE_SCALE` constant it defaults to,
   * because the number is a property of *how signal reaches the field*.
   * `SENSE_SCALE` was measured as the p90 reading at a body's own position —
   * 4.16, 4.24 and 4.53 over soups of 60, 400 and 2000 — and remeasured on
   * this build it still reads 4.26, 4.60 and 5.19.
   *
   * It was wrong for a while and is right again without being touched. A body
   * used to be able to pay for its voice out of its tank, and that stopped the
   * mint: the field then held about 0.002 at the same three sizes, three
   * orders down, because a minted deposit is unbounded in time while an
   * excreted one is bounded by what the bodies hold, and because the scent
   * path lays a *density* where a conserved add lays a *quantity*. Nothing
   * excretes now, so a voice is always minted, and 4.3 is the value for the
   * pond that ships.
   */
  senseScale: number;
  /**
   * How fast the gut turns into the tank, per second, per species.
   *
   * Mass action on what a body is holding, so a gut empties on an exponential
   * and never overshoots; the last crumb snaps to zero so that it does empty.
   * One species, flat: a body eats the ground and the ground is the thing
   * everyone can use raw, which is what makes it the ground. There used to be
   * a per-species recipe and a co-substrate to pair it with, and both went
   * when a mouthful stopped being a sample of the water.
   *
   * Bounded by room in the tank, because a full body has nowhere to put what
   * it digests, and matter that had nowhere to go would have to be destroyed.
   * So a fed body stops digesting, its gut fills, and it stops eating — which
   * is what satiety is here, and it is three mechanisms deep rather than a
   * clamp.
   *
   * Inert until something fills a gut, and only the metered harvest does, so
   * this dial does nothing at `uptakeVmax` 0.
   */
  digestRate: number;
  /**
   * How much a body can hold undigested, as a multiple of its `energyCap`.
   *
   * Not a trait of its own: `energyCap` is already heritable and already means
   * "how much can this body hold", so a lineage that breeds a bigger tank
   * breeds a bigger gut with it and there is one number to select on rather
   * than two that must be selected together.
   *
   * This is the number that makes a filthy cell expensive twice over. The
   * first cost is the sample — ground that is a quarter of what is standing in
   * a cell is a quarter of the mouthful. The second is that the other three
   * quarters have to *go* somewhere, and if the body cannot convert them they
   * sit here, and the next mouthful is smaller for it.
   */
  gutSize: number;
  /**
   * How many patches the ground is laid down in, at the same total mass.
   * 0 spreads it over the whole disk, which is what every preset used to do.
   *
   * A `Params` field rather than a runner flag so that it can be a *sweep
   * axis*: "does the pond develop differently when food is somewhere rather
   * than everywhere" is a question about a grid point, not about an
   * invocation. `soupCount` is setup-time and lives here for the same reason.
   *
   * The mass is held constant across every setting on purpose. A patchy dish
   * against a thinner one compares how much food there is, which is not the
   * question; against a uniform one at the same total it compares structure,
   * which is. For less food, move `ambientEnergy`.
   *
   * Read by `Energy.configure` every frame and acted on when it changes, so
   * moving the slider re-lays the dish under whatever is standing on it. It
   * used to be read once at setup by the headless runner alone, which is why
   * the page had never seen a patch and why a half-saturation constant like
   * `ksg` was inert: on a uniform dish every body faces the same density
   * everywhere, and a scavenger and a grazer are the same animal.
   *
   * Note what the pond already does with the structure: `Fields.grow` skips a
   * cell at zero, so a patch grazed bare only comes back by diffusion from a
   * living neighbour, and a region cleared outright stays dead. That is
   * regeneration with a history rather than a refill timer, which is most of
   * what §6 wants from a reaction-diffusion ground, for free.
   */
  groundPatches: number;
}

export function defaultParams(): Params {
  return {
    deposit: 5,
    diffuse: 0.6,
    decay: 0.01,
    sense: 520,
    attractStrong: 1.45,
    attractMedium: 0.72,
    attractFood: 0.9,
    snapRadius: 22,
    snapArc: 0.3,
    snapWell: 48,
    faceRadius: 90,
    faceAttract: 32,
    rewriteDuration: 0.7,
    springK: 12,
    springDamp: 45,
    auxSpread: 1.7,
    declutter: 1.4,
    wireClear: 1,
    portStiff: 2,
    wireBreathe: 0.04,
    wireShapeAge: 2,
    wireSpanAge: 10,
    wireTaut: 1.08,
    wireSnap: 3,
    wireTug: 0.3,
    wireMinRest: 48,
    wireShrink: 0.2,
    eraMass: 0.45,
    nodeMass: 1,
    turnRate: 1.6,
    sensorAngle: 0.48,
    sensorDist: 24,
    stepSpeed: 38,
    swimTau: 1.1,
    swimNoise: 0.35,
    drag: 0.55,
    angDrag: 2.4,
    grip: 2,
    metabolicRate: 15,
    metabolicYield: 480,
    starveTime: 25,
    metabolicCat: 0.266,
    metabolicReset: 1,
    metabolicQuench: 0.0283,
    metabolicSigma: 0.005,
    metabolicDecay: 0.1,
    metabolicBase: 0.001,
    metabolicWave: 1.2,
    metabolicWork: 0.05,
    metabolicDiffuse: 1,
    metabolicGate: 0.1,
    intake: 0.06,
    gripSwing: 0.5,
    gaitSwell: 0.04,
    flockAlign: 5.5,
    flockSep: 48,
    maxAgents: 100000,
    soupCount: 10000,
    spawnInterval: 0.5,
    energyCell: 40,
    ambientEnergy: 0.5,
    energyDiffuse: 0.006,
    energyRegrow: 0.02,
    upkeep: 0.01,
    swimCost: 0,
    rescueTo: 0.9,
    assortBias: 0.5,
    debtCap: -1,
    requestDecay: 0.95,
    transportRecoil: 100,
    transportQuantum: 0.5,
    transportThrust: 0,
    learnRate: 0.02,
    learnCritic: 0.2,
    learnExplore: 0.02,
    learnReward: 0.5,
    learnTrace: 0.99,
    learnDiscount: 0.99,
    reactorSplit: 0.7,
    inheritLearned: 1,
    uptakeVmax: 2,
    uptakeKs: 0.25,
    hillN: 1,
    yDirect: 1,
    yEra: 1,
    upkeepExcrete: 1,
    bodyValue: REWRITE_SHARE,
    eraCapRatio: ERA_CAP_RATIO,
    eraUpkeepRatio: ERA_UPKEEP_RATIO,
    senseScale: SENSE_SCALE,
    digestRate: 12,
    gutSize: 1,
    groundPatches: 24,
  };
}

export interface SliderSpec {
  key: keyof Params;
  label: string;
  min: number;
  max: number;
  step: number;
}

export const SLIDERS: SliderSpec[] = [
  { key: 'deposit', label: 'Deposit', min: 0, max: 6, step: 0.05 },
  { key: 'diffuse', label: 'Diffuse', min: 0, max: 1, step: 0.01 },
  { key: 'decay', label: 'Decay', min: 0, max: 0.08, step: 0.001 },
  { key: 'sense', label: 'Sense', min: 0, max: 1200, step: 10 },
  // Seeds, not settings. These are read once by `seedChem` and never
  // again — they decide what a *newly spawned* body starts as, and moving them
  // does nothing to anything already alive. Everything above and below changes
  // the world continuously.
  { key: 'attractStrong', label: 'Strong attract (seed)', min: 0, max: 3, step: 0.05 },
  { key: 'attractMedium', label: 'Medium attract (seed)', min: 0, max: 2, step: 0.05 },
  { key: 'attractFood', label: 'Food attract (seed)', min: -2, max: 4, step: 0.05 },
  { key: 'sensorAngle', label: 'Sensor arc', min: 0.1, max: 1.2, step: 0.02 },
  { key: 'stepSpeed', label: 'Step speed', min: 10, max: 180, step: 1 },
  { key: 'swimTau', label: 'Swim persistence', min: 0.1, max: 4, step: 0.05 },
  { key: 'swimNoise', label: 'Swim noise', min: 0, max: 2, step: 0.05 },
  { key: 'drag', label: 'Fluid drag', min: 0, max: 4, step: 0.01 },
  { key: 'angDrag', label: 'Spin damp', min: 0, max: 8, step: 0.05 },
  { key: 'grip', label: 'Grip (tank)', min: -4, max: 12, step: 0.05 },
  { key: 'metabolicRate', label: 'Metabolic rate', min: 0, max: 20, step: 0.1 },
  { key: 'metabolicYield', label: 'Fuel yield', min: 0, max: 1200, step: 10 },
  { key: 'starveTime', label: 'Starve window (s)', min: 0, max: 120, step: 1 },
  { key: 'metabolicCat', label: 'Autocatalysis (k2)', min: 0.01, max: 4, step: 0.002 },
  { key: 'metabolicReset', label: 'C to D (k3)', min: 0.05, max: 4, step: 0.01 },
  { key: 'metabolicQuench', label: 'Quench (k4)', min: 0, max: 0.4, step: 0.0005 },
  { key: 'metabolicSigma', label: 'Saturation', min: 0, max: 1, step: 0.001 },
  { key: 'metabolicDecay', label: 'Outflow', min: 0.005, max: 1, step: 0.005 },
  { key: 'metabolicBase', label: 'Basal enzyme', min: 0, max: 0.2, step: 0.0005 },
  { key: 'metabolicWave', label: 'Stroke half-point', min: 0.1, max: 12, step: 0.1 },
  { key: 'metabolicWork', label: 'Stroke cost', min: 0, max: 1, step: 0.005 },
  { key: 'intake', label: 'Fuel intake rate (seed)', min: 0, max: 0.2, step: 0.002 },
  { key: 'metabolicDiffuse', label: 'Broadcast', min: 0, max: 8, step: 0.02 },
  { key: 'metabolicGate', label: 'Speak above (seed)', min: 0, max: 6, step: 0.05 },
  { key: 'gripSwing', label: 'Grip swing (D)', min: 0, max: 1, step: 0.02 },
  { key: 'gaitSwell', label: 'Gait swell', min: 0, max: 0.8, step: 0.01 },
  { key: 'flockAlign', label: 'Flock align (seed)', min: 0, max: 16, step: 0.1 },
  { key: 'flockSep', label: 'Flock separate (seed)', min: 0, max: 120, step: 1 },
  { key: 'snapRadius', label: 'Snap reach', min: 4, max: 48, step: 1 },
  { key: 'snapArc', label: 'Snap arc', min: 0.08, max: 1.2, step: 0.02 },
  { key: 'wireShrink', label: 'Wire shrink', min: 0.1, max: 3, step: 0.05 },
  { key: 'wireTug', label: 'Wire tug (flux)', min: 0, max: 1, step: 0.02 },
  { key: 'wireMinRest', label: 'Wire min length', min: 8, max: 48, step: 1 },
  { key: 'springK', label: 'Spring stiffness', min: 0, max: 80, step: 0.5 },
  { key: 'springDamp', label: 'Rope damp', min: 0, max: 120, step: 1 },
  { key: 'portStiff', label: 'Port stiffness', min: 0.1, max: 4, step: 0.05 },
  { key: 'auxSpread', label: 'Aux spread', min: 0, max: 3, step: 0.05 },
  { key: 'declutter', label: 'Personal space', min: 0, max: 4, step: 0.05 },
  { key: 'wireClear', label: 'Wire clearance', min: 0, max: 4, step: 0.05 },
  { key: 'wireBreathe', label: 'Wire breathe', min: 0, max: 0.15, step: 0.005 },
  { key: 'wireShapeAge', label: 'Shape drop (s)', min: 0, max: 30, step: 0.1 },
  { key: 'wireSpanAge', label: 'Span-only (s)', min: 0, max: 60, step: 0.5 },
  { key: 'wireTaut', label: 'Taut ratio', min: 1, max: 1.5, step: 0.01 },
  { key: 'wireSnap', label: 'Wire snap ratio', min: 0, max: 6, step: 0.1 },
  { key: 'eraMass', label: 'Era mass', min: 0.15, max: 2, step: 0.05 },
  { key: 'nodeMass', label: 'Con/Dup mass', min: 0.3, max: 4, step: 0.05 },
  { key: 'maxAgents', label: 'Max agents', min: 8, max: 8000, step: 1 },
  { key: 'spawnInterval', label: 'Auto spawn (s)', min: 0, max: 30, step: 0.5 },
  // Step is a whole FIELD_CELL: an energy cell has to stay a multiple of the
  // scent field's cell for the two grids to line up (see EnergyGrid).
  { key: 'energyCell', label: 'Energy cell', min: FIELD_CELL, max: 160, step: FIELD_CELL },
  { key: 'ambientEnergy', label: 'Ambient energy', min: 0, max: 2, step: 0.05 },
  { key: 'energyDiffuse', label: 'Ground spread', min: 0, max: 0.5, step: 0.005 },
  { key: 'energyRegrow', label: 'Ground regrow', min: 0, max: 0.4, step: 0.005 },
  { key: 'upkeep', label: 'Upkeep', min: 0, max: 0.2, step: 0.005 },
  { key: 'swimCost', label: 'Swim cost', min: 0, max: 0.002, step: 0.00005 },
  { key: 'rescueTo', label: 'Rescue fill', min: 0, max: 1, step: 0.05 },
  { key: 'assortBias', label: 'Assortment (seed)', min: 0, max: 1, step: 0.05 },
  { key: 'debtCap', label: 'Debt cap', min: -2.5, max: -0.05, step: 0.05 },
  { key: 'requestDecay', label: 'Demand decay', min: 0.5, max: 0.98, step: 0.01 },
  { key: 'transportRecoil', label: 'Pump recoil (seed)', min: 0, max: 200, step: 5 },
  { key: 'transportThrust', label: 'Pump thrust (seed)', min: 0, max: 1, step: 0.05 },
  { key: 'transportQuantum', label: 'Transport quantum', min: 0, max: 1, step: 0.05 },
  { key: 'learnRate', label: 'Learn rate', min: 0, max: 0.02, step: 0.0005 },
  { key: 'learnCritic', label: 'Learn critic', min: 0, max: 0.2, step: 0.005 },
  { key: 'learnExplore', label: 'Learn explore', min: 0, max: 0.4, step: 0.005 },
  { key: 'learnReward', label: 'Teacher: level -> rate', min: 0, max: 1, step: 0.05 },
  { key: 'learnTrace', label: 'Learn trace decay', min: 0.5, max: 0.995, step: 0.005 },
  { key: 'learnDiscount', label: 'Learn discount', min: 0.5, max: 0.995, step: 0.005 },
  { key: 'reactorSplit', label: 'Reactor split at rewrite', min: 0, max: 1, step: 0.05 },
  { key: 'inheritLearned', label: 'Inherit learned', min: 0, max: 1, step: 0.05 },
  { key: 'uptakeVmax', label: 'Uptake rate', min: 0, max: 4, step: 0.05 },
  { key: 'uptakeKs', label: 'Uptake half-sat', min: 0.01, max: 2, step: 0.01 },
  { key: 'hillN', label: 'Hill coefficient', min: 1, max: 4, step: 0.1 },
  { key: 'yDirect', label: 'Direct uptake yield', min: 0, max: 1, step: 0.05 },
  { key: 'yEra', label: 'Era uptake yield', min: 0, max: 4, step: 0.05 },
  { key: 'upkeepExcrete', label: 'Upkeep excretes', min: 0, max: 1, step: 0.05 },
  { key: 'bodyValue', label: 'Body value', min: 0.5, max: 2, step: 0.05 },
  { key: 'eraCapRatio', label: 'Era tank ratio', min: 1, max: 4, step: 0.1 },
  { key: 'eraUpkeepRatio', label: 'Era upkeep ratio', min: -1, max: 2, step: 0.05 },
  { key: 'senseScale', label: 'Sense scale', min: 0.001, max: 8, step: 0.001 },
  { key: 'digestRate', label: 'Digest rate', min: 0, max: 40, step: 0.5 },
  { key: 'gutSize', label: 'Gut size', min: 0.1, max: 4, step: 0.1 },
  { key: 'groundPatches', label: 'Ground patches', min: 0, max: 128, step: 1 },
];
