import { FIELD_CELL } from './fields.ts';
// Derived, not copied. These three are the values the pond has always run at,
// and they live where the economy that uses them lives; writing the numbers
// again here is exactly the duplication `chem-layout.ts` exists to warn about.
import { BODY_VALUE, ERA_CAP_RATIO, ERA_UPKEEP_RATIO } from './energy.ts';
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
   * Activity LOD budget. Settled taut islands run the cheap disc+span path
   * even when they are on screen and close up; 0 turns the whole thing off.
   * Live ropes, loners, grabs, rewrites and fresh latches stay NEAR whatever
   * this says — the budget only caps how far a contact can propagate a wake.
   *
   * Off by default, because measured on this branch it costs more than it
   * saves. It does exactly what it claims — a settled 1518-body mesh filling
   * the viewport goes from 95% detailed to 2% — but the frame goes 22.8ms to
   * 25.5ms, and the profiler puts the whole difference in `solve`. The reason
   * is that the two mechanisms overlap almost perfectly: `ropeIsLive` already
   * returns false for any wire whose `ropePath` is 'span', whatever the detail
   * flag says, and a wire turns 'span' under the same aged-and-taut condition
   * that makes its net sleepable. So the expensive half of NEAR is already off
   * before sleep gets a vote, and all sleep can still remove is SAT on settled
   * bodies — which is cheaper than the FAR path it moves them onto.
   *
   * Kept because the mechanism is sound and cheap when off, and because the
   * branch it came from carried per-body audio work that NEAR paid for and
   * this one no longer has. If NEAR ever grows an expensive per-body pass
   * again, this is already here and already tested.
   */
  nearBudget: number;
  /**
   * How hard a wire crossing a principal connection reels its own ends
   * together. Off by default: measured over three minutes of soup it made
   * crossings, clumping and rewrite throughput all slightly worse once
   * `declutter` was in, which already removes the crossings a local force
   * can plausibly undo. Kept as a knob because the detection is the cheap part.
   */
  uncross: number;
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
   * Energy per second of penetration depth, charged to both bodies in a
   * contact. 0 = collisions are free.
   *
   * Damage rather than death. One rule kills — `extra` reaching the floor —
   * and everything lethal works through the economy, so a bad knock is
   * survivable and a body can recover from it. A separate physics death path
   * would also turn any numerical fault into an extinction: the pond flew
   * apart once already this week from a constant that disagreed across the
   * wasm wall, and that should stay a thing you can watch and diagnose.
   *
   * It is also what gives `flockSep` something to select on. Keeping your
   * distance is now heritable and nothing rewarded it; a cost for crowding
   * lets a lineage work out whether to space out or tolerate the scrum.
   */
  contactCost: number;
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
   * any of it, in that species' own units. Seeds all four of `Gx` and is
   * heritable per species from there.
   *
   * The doc's §4 `H(x_j - G_j)`, rectified rather than a step. It seeds at
   * zero, so a fresh body broadcasts in proportion to what it holds and the
   * pulse comes from the concentration's own swing; a threshold is something
   * a lineage evolves toward when being quiet at the trough pays.
   */
  metabolicGate: number;
  /**
   * What share of the food it digests a fresh body routes to its reactor
   * rather than banking in its tank. Heritable from there.
   *
   * The doc's §8 per-agent `J`, and the one thing about the reactor a lineage
   * owns. It decides where in the fuel window a body sits, and the window has
   * both edges: starved, the reactor sits empty and still; fed, it runs a
   * limit cycle whose period shortens as the fuel rises; glutted, it sits
   * saturated and still again.
   */
  intake: number;
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
   */
  /**
   * Gray-Scott feed and kill, between the two signal channels. Both 0 = off.
   *
   * `u + 2v -> 3v` run on `CH.conP` (substrate) and `CH.dupP` (activator).
   * Without it every channel is a decaying hill around whoever is emitting, so
   * what a body smells is always *who is there* — the field carries information
   * but holds none of its own. A reaction puts maxima where nobody is standing,
   * fronts that travel, and regions just used up and briefly unusable, so
   * "over there" can start to mean something no emitter is saying.
   *
   * The ground is deliberately not one of the two. `CH.energy` is a conserved
   * quantity the economy balances, and a reaction that converts it would create
   * and destroy food as a side effect of signalling.
   *
   * Two things have to be arranged before this does anything but wash flat, and
   * both are yours. The species must diffuse at different rates — Gray-Scott
   * wants the substrate at roughly twice the activator, and `diffuseRate` is
   * where that lives; equal rates have no instability to find. And the pair
   * sits in a thin sliver of its own plane, roughly F in [0.01, 0.09] against
   * k in [0.045, 0.07], with anything worth looking at inside a fraction of
   * that. A default picked without a screen in front of it would be a uniform
   * wash that looked like the code not working, which is why both ship at zero.
   *
   * Note `decay` still acts on both channels, so the effective kill is
   * `decay + reactKill` rather than `reactKill` alone.
   */
  reactFeed: number;
  reactKill: number;
  energyDiffuse: number;
  /**
   * Logistic regrowth rate, per second, toward `ambientEnergy` per cell.
   *
   * Not a refill timer. Growth is proportional to what is already in the cell,
   * so a cell taken to exactly zero never comes back on its own and has to be
   * recolonised from a neighbour — grazing to the floor makes a scar that
   * heals from its rim at the speed `energyDiffuse` sets. 0 turns the ground
   * back into the seam of ore it used to be.
   */
  energyRegrow: number;
  /**
   * How much the fertiliser channel accelerates regrowth, per unit of it.
   *
   * Growth becomes `r * (1 + fertilise * C) * E * (1 - E/K)`, where `C` is
   * `FERTILISE_CH`. What this buys that the ground's excretion row does not
   * is a *reason for
   * two lineages to need each other*.
   *
   * Farming moves stock from a tank onto the dish: one body's investment, and
   * the returner is the same body. This is catalysis — a lineage that emits on
   * the fertiliser channel creates no energy at all, it makes the ground
   * recover faster wherever it happens to be. It cannot feed itself that way
   * any more than it already could, because the growth it accelerates is
   * bounded by the same carrying capacity. What it can do is make the patch it
   * stands in worth more to *somebody else*, and a grazer that stays near a
   * fertiliser does better than one that does not. That is mutualism out of
   * two genes and no new machinery.
   *
   * Negative is an inhibitor, which a lineage should be able to become — a
   * body that poisons the ground around it denies a competitor more than it
   * costs itself. The effective rate is clamped at zero, so an inhibitor can
   * stall regrowth but never run it backwards; ground destroyed by being
   * smelled at would be a hole in the conservation the economy depends on.
   *
   * Off by default, like every dial that changes what energy does. This one
   * needs `energyRegrow` non-zero to mean anything at all — it scales a rate,
   * and scaling zero is zero.
   */
  fertilise: number;
  /** Extra drained per second. 0 = off. Hitting −1 kills the agent. */
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
   * How loudly a body asks for energy on account of liking where it is.
   *
   * The other half of the motor. A price on swimming alone gives a net a fuel
   * bill; this is what makes the bill *directional*. A body with a free
   * principal reads its own `trail` — everything it can smell, through its own
   * taste weights — and asks in proportion, so the bodies standing where the
   * net most wants to be are the ones that get fed and thrust hardest.
   *
   * Off by default, like the other two dials that change what energy is spent
   * on. Not caution for its own sake: the field takes the largest claim it can
   * see, so an appetite competes directly with `rescueNeed` and `redexNeed` —
   * somebody about to die, and somebody about to reproduce. At 0.05 against
   * the scent a pond makes of itself it already outbid a rescue, topping a
   * dying body past its own `rescueTo` to a full tank while its donors went
   * without. Wanting to go somewhere nice should lose to both of those, and
   * where the crossover sits depends on how loud the pond is, which is four
   * other sliders. Worth tuning by eye rather than guessing a default.
   */
  forageAsk: number;
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
   * Hops the need field carries in one frame. 0 = as far as it goes, which is
   * the pond as it was.
   *
   * The field is a potential: every body holds the largest need it can see,
   * attenuated per hop by whoever relays it. At 0 that potential is solved to
   * a fixpoint every frame, so a shortage anywhere is *known* everywhere on
   * the frame it arises — demand has no front, nothing propagates, and the
   * whole net answers at once. That is most of why a net pulses rather than
   * walks, and it is why nothing in here can carry a wave.
   *
   * Above 0 the field advances that many hops a frame off the previous
   * frame's values, so a need takes a frame per wire to travel and drains
   * behind itself when it stops. The fixpoint is unchanged — `request_i =
   * max(claim_i, max_j request_j * keep_j)` is the same equation either way —
   * so a settled field settles where it always did. What changes is that
   * getting there takes time, and that time is what a travelling wave is made
   * of.
   *
   * It also changes what the economy *is*, which is why it ships off. A
   * global relaxation is global triage: every donor compares its neighbour
   * against the worst case anywhere on the net, so the dying body outranks
   * the merely empty one however far away it is. One hop a frame is local
   * equalisation: a body can only weigh what it can see, and a corridor of
   * empty bodies absorbs a reservoir on its way past rather than relaying it.
   * Measured on `energy.test.ts`'s corridor — twelve bodies, a reservoir at
   * one end and a body in debt at the other — at 1 the reservoir is empty
   * inside twenty frames and the patient ends on exactly zero, having been
   * filled and then drained back into the corridor. At 0 it is fed to its
   * rescue target and stays there.
   *
   * So this is not a free improvement, it is a trade, and the thing that
   * would pay for it is a need field that tells dying from empty by more than
   * the difference between `rescueNeed`'s two branches. That is the next
   * question and it is not settled here.
   */
  requestReach: number;
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
   *
   * See `docs/plasticity-plan.md`.
   */
  learnRate: number;
  /** How fast the critic itself learns to predict. Its own delta rule. */
  learnCritic: number;
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
   * Chemistry. See `docs/energy-chemistry-plan.md`; every one of these ships
   * at the value that reduces to the behaviour before it existed, which is
   * the same discipline `fertilise` and `reactFeed` follow.
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
   * Fixed cost, per second, of expressing a reaction row at all.
   *
   * One of the two dials that decide whether nets differentiate. A linear
   * budget with concave payoffs — and Monod saturation is concave — puts the
   * optimum in the interior and makes everyone a generalist; specialisation
   * needs `f(1) > 2 f(1/2)`, which concavity forbids. A fixed cost per row
   * supplies the superadditivity: two rows cost `2c`, one costs `c`.
   */
  rowCost: number;
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
   * How much of ordinary upkeep is put back into the field rather than
   * destroyed, 0 to 1 — as the body's own excretion mix, through `payOut`,
   * so a Con pays its rent in `conP` and `aux` and only an Era pays in ground.
   * The gait's pathway and the row cost leave by the same road.
   *
   * 0 is today: rent vanishes. 1 makes bodies conservative — no reaction a
   * body runs creates or destroys matter — which is the invariant that makes
   * selection honest. It is a dial rather than a constant because turning it
   * on changes the pond's standing stock, and the plan's discipline is that
   * nothing changes behaviour until somebody has looked. Above zero it also
   * turns `refreshExpression` on, because the mix is read off the rows.
   */
  upkeepExcrete: number;
  /**
   * What a body's existence is worth when it dies, `EXTRA_CAP` today.
   *
   * At `REWRITE_SHARE` (1) the commute-then-annihilate cycle stops minting:
   * today it makes `2 * (EXTRA_CAP - REWRITE_SHARE)` = 0.5 out of nothing,
   * which the comment on `BODY_VALUE` has always described as the metabolism
   * rather than a slip. With uptake rate-limited, a net's income no longer
   * has to be proportional to its rewrite rate, which is the argument for
   * taking it — but taking it moves the pond, so it is a dial and not an edit.
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
   * Rate at which a body puts out its excretion rows, per second per unit it
   * holds in gut and tank together, for a row at an eighth. What the gut holds
   * of a species leaves first and is free; the shortfall is synthesised from
   * the tank. This is also the farming dial: an Era's whole production half
   * is the ground row, so at this rate it lays ground, which is what
   * `farmRate` used to be.
   *
   * **Zero is today, and turning it on is a switch rather than a slider.**
   * Today a body's voice is *minted*: `effEmit` is multiplied by
   * `params.deposit`, which is five, and nothing is taken out of the tank —
   * with `CH.energy` firewalled out of that path precisely because five units
   * of food a frame out of nothing would be absurd. Above zero, all four
   * species leave the body conserved and the minted deposit stops: the
   * firewall becomes a stoichiometry rather than a special case, and a poor
   * body physically cannot shout.
   *
   * `docs/energy-chemistry-plan.md` §3 and §8. Expect the measured signalling
   * constants to move with it — signal amplitude drops by about the deposit
   * multiplier, so `SENSE_SCALE` and the steering dead zone were measured
   * against a world that no longer exists. Remeasure rather than rescale.
   */
  excreteRate: number;
  /**
   * What one unit of a signal reading is worth on the way into `x`.
   *
   * A `Params` field rather than the `SENSE_SCALE` constant it defaults to,
   * because the number is a property of *how signal reaches the field*, and
   * `excreteRate` changes that completely. `SENSE_SCALE` was measured as the
   * p90 reading at a body's own position — 4.16, 4.24 and 4.53 over soups of
   * 60, 400 and 2000 — and remeasured on this build it still reads 4.26, 4.60
   * and 5.19, so the constant is right for the pond it was measured in.
   *
   * **Under conserved excretion it reads about 0.002**, measured the same way
   * at the same three sizes: 0.0016, 0.0017, 0.0024. Not the ~5x the plan's §8
   * predicted, and the extra three orders are worth understanding rather than
   * absorbing. Two things compound. The minted deposit is unbounded in time —
   * nothing is taken out of a tank to pay for it — so the field accumulates to
   * whatever decay allows, while excretion is bounded by what the bodies
   * actually hold. And the scent path lays a *density*, scaled by cell area,
   * where a conserved add lays a *quantity*; a unit of matter spread over a
   * million-cell dish simply does not read like a unit of shouting.
   *
   * Left at the minted value, because moving it would move the pond that
   * ships. A run with `excreteRate` on wants this near 0.002 or its sense
   * genes are reading a signal three orders below the range `phi` can resolve.
   */
  senseScale: number;
  /**
   * How many units of ground one unit of a signalling species is converted
   * *with*, spent out of the body's own gut. See `docs/energy-chemistry-plan.md`
   * §6b and `Sim.runDigestion`.
   *
   * 0 is what phase 3 shipped: an uptake row eats its species raw, which is
   * "eating scent" and is the thing §6b calls wrong. Above it the ground is a
   * reagent, not a catalyst — a body converts species 0, 1 and 3 into energy
   * only by pairing them with ground it has swallowed, one budget across the
   * three, so a body with a little ground must choose what to spend it on and
   * one unit cannot unlock everything. Energy is the co-substrate everyone
   * can already use, and the others are mass nobody can touch without it.
   *
   * A dial and continuous, because that is what keeps the gradient: every
   * unit of ground a body swallows unlocks a proportional unit of something
   * else, so a body with a little capability for a species does a little
   * better than one with none and selection has a slope to climb. Forcing a
   * hard requirement is what makes machinery worthless until complete, which
   * is the trap §6b is written around.
   *
   * It buys access, never amplification — conservation still holds, the
   * ground spent lands in the tank alongside what it unlocked. What a
   * catabolist gains is a pool its competitors cannot reach, and the pool is
   * largest exactly where other bodies are dense and the ground is grazed out.
   * The slider stops at 1, which is a choice about how expensive scent should
   * be and not a bound in the mechanism.
   */
  catCoSubstrate: number;
  /**
   * How fast the gut turns into the tank, per second, per species.
   *
   * Mass action on what a body is holding, so a gut empties on an exponential
   * and never overshoots; the last crumb snaps to zero so that it does empty.
   * The ground converts at this rate flat — it is the thing everyone can use
   * raw, which is what makes it the ground — and the other three convert at
   * this rate scaled by the body's uptake row for that species, and only as
   * far as the ground in its gut will pair with them (`catCoSubstrate`). A
   * body whose recipe cannot touch a species converts none of it, and it
   * stays in the gut occupying the room that bounds the next mouthful until
   * excretion clears it.
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
   * 0 spreads it over the whole disk, which is what every preset does.
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
   * Read once, at setup, by `pond/ground.ts` — and note what the pond already
   * does with it: `Fields.grow` skips a cell at zero, so a patch grazed bare
   * only comes back by diffusion from a living neighbour, and a region cleared
   * outright stays dead. That is regeneration with a history rather than a
   * refill timer, which is most of what §6 wants from a reaction-diffusion
   * ground, for free.
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
    nearBudget: 0,
    uncross: 0,
    wireClear: 1,
    portStiff: 2,
    wireBreathe: 0.04,
    wireShapeAge: 2,
    wireSpanAge: 10,
    wireTaut: 1.08,
    wireSnap: 3,
    contactCost: 0,
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
    metabolicGate: 0,
    intake: 0.25,
    gaitSwell: 0.04,
    flockAlign: 5.5,
    flockSep: 48,
    maxAgents: 100000,
    soupCount: 10000,
    spawnInterval: 0.5,
    energyCell: 40,
    ambientEnergy: 1.0,
    reactFeed: 0,
    reactKill: 0,
    energyDiffuse: 0.05,
    energyRegrow: 0.04,
    fertilise: 0,
    upkeep: 0.015,
    swimCost: 0,
    forageAsk: 0,
    rescueTo: 0.9,
    assortBias: 0.5,
    debtCap: -1,
    requestDecay: 0.95,
    requestReach: 0,
    transportRecoil: 100,
    transportQuantum: 0.5,
    transportThrust: 0,
    learnRate: 0.02,
    learnCritic: 0.2,
    learnTrace: 0.99,
    learnDiscount: 0.99,
    inheritLearned: 1,
    uptakeVmax: 2,
    uptakeKs: 0.25,
    rowCost: 0,
    hillN: 1,
    yDirect: 1,
    yEra: 1,
    upkeepExcrete: 0,
    bodyValue: BODY_VALUE,
    eraCapRatio: ERA_CAP_RATIO,
    eraUpkeepRatio: ERA_UPKEEP_RATIO,
    excreteRate: 1,
    senseScale: SENSE_SCALE,
    catCoSubstrate: 0,
    digestRate: 12,
    gutSize: 1,
    groundPatches: 0,
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
  { key: 'intake', label: 'Fuel intake (seed)', min: 0.2, max: 6, step: 0.1 },
  { key: 'metabolicDiffuse', label: 'Broadcast', min: 0, max: 8, step: 0.02 },
  { key: 'metabolicGate', label: 'Speak above (seed)', min: 0, max: 6, step: 0.05 },
  { key: 'gaitSwell', label: 'Gait swell', min: 0, max: 0.8, step: 0.01 },
  { key: 'flockAlign', label: 'Flock align (seed)', min: 0, max: 16, step: 0.1 },
  { key: 'flockSep', label: 'Flock separate (seed)', min: 0, max: 120, step: 1 },
  { key: 'snapRadius', label: 'Snap reach', min: 4, max: 48, step: 1 },
  { key: 'snapArc', label: 'Snap arc', min: 0.08, max: 1.2, step: 0.02 },
  { key: 'wireShrink', label: 'Wire shrink', min: 0.1, max: 3, step: 0.05 },
  { key: 'wireMinRest', label: 'Wire min length', min: 8, max: 48, step: 1 },
  { key: 'springK', label: 'Spring stiffness', min: 0, max: 80, step: 0.5 },
  { key: 'springDamp', label: 'Rope damp', min: 0, max: 120, step: 1 },
  { key: 'portStiff', label: 'Port stiffness', min: 0.1, max: 4, step: 0.05 },
  { key: 'auxSpread', label: 'Aux spread', min: 0, max: 3, step: 0.05 },
  { key: 'declutter', label: 'Personal space', min: 0, max: 4, step: 0.05 },
  { key: 'nearBudget', label: 'NEAR budget', min: 0, max: 2000, step: 25 },
  { key: 'uncross', label: 'Uncross', min: 0, max: 4, step: 0.05 },
  { key: 'wireClear', label: 'Wire clearance', min: 0, max: 4, step: 0.05 },
  { key: 'wireBreathe', label: 'Wire breathe', min: 0, max: 0.15, step: 0.005 },
  { key: 'wireShapeAge', label: 'Shape drop (s)', min: 0, max: 30, step: 0.1 },
  { key: 'wireSpanAge', label: 'Span-only (s)', min: 0, max: 60, step: 0.5 },
  { key: 'wireTaut', label: 'Taut ratio', min: 1, max: 1.5, step: 0.01 },
  { key: 'wireSnap', label: 'Wire snap ratio', min: 0, max: 6, step: 0.1 },
  { key: 'contactCost', label: 'Impact cost', min: 0, max: 0.2, step: 0.005 },
  { key: 'eraMass', label: 'Era mass', min: 0.15, max: 2, step: 0.05 },
  { key: 'nodeMass', label: 'Con/Dup mass', min: 0.3, max: 4, step: 0.05 },
  { key: 'maxAgents', label: 'Max agents', min: 8, max: 8000, step: 1 },
  { key: 'spawnInterval', label: 'Auto spawn (s)', min: 0, max: 30, step: 0.5 },
  // Step is a whole FIELD_CELL: an energy cell has to stay a multiple of the
  // scent field's cell for the two grids to line up (see EnergyGrid).
  { key: 'energyCell', label: 'Energy cell', min: FIELD_CELL, max: 160, step: FIELD_CELL },
  { key: 'ambientEnergy', label: 'Ambient energy', min: 0, max: 2, step: 0.05 },
  { key: 'reactFeed', label: 'React feed', min: 0, max: 0.1, step: 0.001 },
  { key: 'reactKill', label: 'React kill', min: 0, max: 0.08, step: 0.001 },
  { key: 'energyDiffuse', label: 'Ground spread', min: 0, max: 0.5, step: 0.005 },
  { key: 'energyRegrow', label: 'Ground regrow', min: 0, max: 0.4, step: 0.005 },
  { key: 'fertilise', label: 'Fertilise', min: -2, max: 8, step: 0.1 },
  { key: 'upkeep', label: 'Upkeep', min: 0, max: 0.2, step: 0.005 },
  { key: 'swimCost', label: 'Swim cost', min: 0, max: 0.002, step: 0.00005 },
  { key: 'forageAsk', label: 'Forage ask', min: 0, max: 0.5, step: 0.01 },
  { key: 'rescueTo', label: 'Rescue fill', min: 0, max: 1, step: 0.05 },
  { key: 'assortBias', label: 'Assortment (seed)', min: 0, max: 1, step: 0.05 },
  { key: 'debtCap', label: 'Debt cap', min: -2.5, max: -0.05, step: 0.05 },
  { key: 'requestDecay', label: 'Demand decay', min: 0.5, max: 0.98, step: 0.01 },
  { key: 'requestReach', label: 'Demand hops/frame', min: 0, max: 12, step: 1 },
  { key: 'transportRecoil', label: 'Pump recoil (seed)', min: 0, max: 200, step: 5 },
  { key: 'transportThrust', label: 'Pump thrust (seed)', min: 0, max: 1, step: 0.05 },
  { key: 'transportQuantum', label: 'Transport quantum', min: 0, max: 1, step: 0.05 },
  { key: 'learnRate', label: 'Learn rate', min: 0, max: 0.02, step: 0.0005 },
  { key: 'learnCritic', label: 'Learn critic', min: 0, max: 0.2, step: 0.005 },
  { key: 'learnTrace', label: 'Learn trace decay', min: 0.5, max: 0.995, step: 0.005 },
  { key: 'learnDiscount', label: 'Learn discount', min: 0.5, max: 0.995, step: 0.005 },
  { key: 'inheritLearned', label: 'Inherit learned', min: 0, max: 1, step: 0.05 },
  { key: 'uptakeVmax', label: 'Uptake rate', min: 0, max: 4, step: 0.05 },
  { key: 'uptakeKs', label: 'Uptake half-sat', min: 0.01, max: 2, step: 0.01 },
  { key: 'rowCost', label: 'Expression row cost', min: 0, max: 0.02, step: 0.0005 },
  { key: 'hillN', label: 'Hill coefficient', min: 1, max: 4, step: 0.1 },
  { key: 'yDirect', label: 'Direct uptake yield', min: 0, max: 1, step: 0.05 },
  { key: 'yEra', label: 'Era uptake yield', min: 0, max: 4, step: 0.05 },
  { key: 'upkeepExcrete', label: 'Upkeep excretes', min: 0, max: 1, step: 0.05 },
  { key: 'bodyValue', label: 'Body value', min: 0.5, max: 2, step: 0.05 },
  { key: 'eraCapRatio', label: 'Era tank ratio', min: 1, max: 4, step: 0.1 },
  { key: 'eraUpkeepRatio', label: 'Era upkeep ratio', min: -1, max: 2, step: 0.05 },
  { key: 'excreteRate', label: 'Excrete rate', min: 0, max: 2, step: 0.02 },
  { key: 'senseScale', label: 'Sense scale', min: 0.001, max: 8, step: 0.001 },
  { key: 'catCoSubstrate', label: 'Catabolism needs ground', min: 0, max: 1, step: 0.05 },
  { key: 'digestRate', label: 'Digest rate', min: 0, max: 40, step: 0.5 },
  { key: 'gutSize', label: 'Gut size', min: 0.1, max: 4, step: 0.1 },
  { key: 'groundPatches', label: 'Ground patches', min: 0, max: 128, step: 1 },
];
