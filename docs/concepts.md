# Concepts

The design bets on a few well-tuned mechanics that leave room for emergence.
It does not add mechanics, fitness measures, rule systems or workarounds where
an existing mechanic can be tuned or a condition changed. This document is how
that is kept true: every research-level concept below must be tied to one
implemented mechanic, so that "what gives us X" has a one-line answer naming a
dial, and so that a mechanic being asked to do two things at once is visible.

A proposal names the heading it serves. If the heading already has a mechanic,
the proposal is a tuning, a measurement or a bug fix. If the heading has no
mechanic, the proposal says so, and says why no existing one can be tuned to
cover it.

The list below is the headings only. Each is to be filled with one
prescriptive statement: the mechanic it lives in and what that mechanic must
not also be asked to do. Nothing else belongs here; measurements go in
`experiments.md`, arguments in the plans.

## Agent

- Sensing
- Movement and steering
- Signalling
- Metabolism: eating. **Not filled, but three of the things it had to settle
  are settled.** Upkeep is gone — it was one dial doing three jobs and
  `docs/metabolism-spec.md` §7 has where each went. Excretion is gone with it:
  a body ate a sample of all four channels and so had to be able to clear what
  it could not convert, and a body that eats only the ground has nothing in its
  gut it cannot digest. What is left to settle is that the reactor which *is* a
  body's metabolism is written up under Net > Locomotion, because it is also
  the gait's clock — one mechanic serving two headings, which is the thing this
  document exists to make visible.

  **Matter is the ground; signal is the other three channels; nothing crosses.**
  A body eats ground, banks it, and feeds its reactor out of it at `intake`, a
  rate it owns. It excretes nothing. A voice is minted and smelled and is not
  matter, so the conservation books are one column. That is the whole of what
  the four channels are for, and it is why there is no longer a fork between a
  minted pond and a conserving one.
- Memory
- Learning. Lives in two rules sharing one TD error from a per-body critic.
  On the state matrices it is the three-factor rule: `delta` times an
  eligibility trace on `phi'(v) * pre`. On the output heads it is node
  perturbation: each of a body's fifteen head outputs is displaced by its own
  reproducible draw, and the trace credits *that* draw. Everything from `Wx`
  to `g0` moves within a life — the recurrent core and every head — and every
  other gene moves by breeding alone.

  **The teacher is the one designed objective in this pond, and it is said out
  loud here because everything else is emergent.** There is no fitness term and
  selection is what survives — but a learner needs something to be taught by,
  and this is it. What keeps it honest is that it is not a goal anybody chose:
  `IN_FULL` is distance from `debtCap`, which is the system's own death
  condition, so the reward is "how close am I to dying". `learnReward` blends
  the *level* of that (`IN_FULL - 1`, which saturates — 31.7% of a live soup
  sits at exactly zero reward) with its *rate* (the change per second, which
  still reads for a body at its cap). It ships at 0.5, both.

  **What that teacher can reach was one scalar wide, and the heads are how it
  widened.** `h` only becomes behaviour through the six head matrices, which
  seed to zero but for two entries, so a fresh body's whole state-dependent
  behaviour is `IN_DEMAND -> h[0] -> taste(ground)` and
  `IN_FULL -> h[1] -> flock alignment`. The rest used to be unreachable:
  measured on a grown pond, four of the six heads sat at the mutation floor for
  their whole lives, because an undirected walk around zero is unbiased and
  half the population has the wrong sign at all times.

  Teaching a head needs a per-channel error the scalar reward does not supply,
  so what supplies it is the perturbation: the body swam at `head + xi` this
  frame, and if things then went better than the critic predicted, the weights
  that produced that displacement are the ones to keep. That is a
  policy-gradient estimator rather than an extension of the Hebbian rule, and
  it is why there are two rules here rather than one. `learnExplore` is how
  hard a body jitters itself, and at zero the heads freeze and learning goes
  back to the core alone.

  **What is still open under this heading is whether any of it helps.** The
  heads demonstrably move. Whether a pond that learns forages better than one
  that does not is not resolvable at three seeds — `forage_ratio` varies more
  between seeds of the same arm than between arms — and it sits below 1 in
  every arm, which says no pond here forages yet.

  `inheritLearned` ships at 1, so what a body learns is written into its
  children's genome: this pond is Lamarckian, deliberately.

- Regulation: behaviour that follows state without learning
- Death

## Net

- Identity: what counts as one organism
- Shape. Lives in wire physics, and **it is true now**. Tension is held by
  energy flowing along a wire, so a net that moves no energy is slack:
  `Wire.carried` is what the transport pass moved across a wire this frame and
  `wireTug` pulls the wire in by up to that much of its own resting length. A
  wire that moves nothing sits at exactly its rest.
  **And the rest length is the wire's own** (`Wire.restBase`), not one global
  every wire drifts to, so a net can have long and short wires — which is the
  only morphology this heading ever had. That is also what makes the tension
  term safe: `Sim.principalRedexReady` asks whether two principals are within
  1.3 of *that wire's* rest, a ratio rather than a distance in pixels, so
  shortening a wire is a change of shape and not a change of who breeds. It
  was the absolute gate that retired `wireTug` the first time.
  One consequence rather than a surprise: a carrying wire's ends sit closer, so
  it reaches its gate sooner. Energy flowing makes things react.
- Locomotion. Lives in the drag law, whose rate is `drag + grip * fullness`.
  A net moves itself by moving energy through itself: a transport kick is
  equal and opposite, and under one rate for every body it cancels at the
  centre of mass, so `grip` is the only thing that lets a stroke become
  travel. The pair's centre keeps
  `|p| * (1/k_sender - 1/k_receiver) / (m_sender + m_receiver)` of the kick,
  which is first order in the impulse and needs no cycle and no phase — mass
  cancels from the direction, so which way a net goes is decided only by which
  end is grippier. Nothing else may create motion a net did not pay for;
  `transportThrust`, which does, is a reactionless drive and is retired at 0.
  Two mechanics were built beside it and withdrawn. A speed-dependent term
  swam (1.63 px/s on the worm bench against grip's 2.31, and the two did not
  add), but one heading gets one mechanic, and grip is the one whose stroke
  depends on the energy pattern a net controls rather than on where its heavy
  bodies happen to be. A contracting wire (`wireTug`, `wireTugTau`) read like
  an inchworm and was not one: the pull and the grip were driven by the same
  packet at the same instant and then decayed, so the loop they traced in
  (length, grip) closed on a line, and its area was only what the difference
  between two relaxation times left over. A gait wants two degrees of freedom
  with a phase something can *set*; a packet clock and two exponentials are
  one degree of freedom and a race. It also spent the wire's rest length,
  which is what decides whether two bodies ever meet, so a tug hard enough to
  swim held every pair too tight to rewrite. Not yet true: grip is a free
  global constant where the stroke should be the net's own; and turned up it
  anchors a net rather than walking it, because a crawler needs its grip to
  travel and only `transportQuantum` makes the fullness pattern move. A real
  gait wants a clock that is not the packet. `gaitRate` is one: every body
  carries a phase, and `G`'s two heads scale its cosine into the body's own
  drag rate and into an impulse along each of its wires. That is the same
  first-order mechanism — an impulse, then two ends coasting different
  distances from it — put on a clock the net owns rather than on whenever a
  packet crossed, and it is heritable in the way that matters: the amplitudes
  are heads off `h`, their product is the speed, and their relative sign is
  the direction.

  Two things were learned building it and are worth not re-learning. The
  clock cannot come out of `h`: `phi` saturates each dimension separately, so
  a rotation in `Wh` slow enough to be a gait has a loop gain barely above 1,
  and there any steady input — `IN_FULL` is one — collapses it onto a fixed
  point. And the stroke cannot be a length: a wire's rest length is served by
  an XPBD span constraint, and a position correction split by inverse mass
  moves the two bodies and not their centre, at any phase, which is the
  deeper reason `wireTug` never swam. Measured, a wired pair driven that way
  travels 0.000 px in twenty seconds and the same pair driven by an impulse
  travels 5.2.

  What a wire's two ends differ in is the whole of the stroke, and there are
  three places that difference comes from. Fullness is the economic one, and
  makes a net walk only while it holds a gradient. Phase is an accident:
  `createAgent` scatters it, so two bodies sit at different points of the same
  cosine. Kind is the structural one, and is what makes an Era a limb — it has
  one port so it is always a leaf, one wire so its stroke is uncancelled where
  an interior body's three partly fight, and it is light. `seedGait` gives it
  a large stroke and almost no anchor, and gives a Con or a Dup the reverse.
  Measured on a fed pair held at one phase, twenty seconds: 1.281 px from a
  fullness gradient alone, 0.756 from a phase difference alone, 1.205 from an
  Era on the end alone, and 0.000 between two Cons with none of the three.

  A leaf is a limb only on an *auxiliary* port. A redex needs principals at
  both ends and an Era has nothing but a principal, so on a Con's `l` or `r`
  it is an appendage and on a Con's `p` it is an erase. Which one a lineage
  gets is about where it latches, which no gene here reaches.

  The stroke moves a wire's rest length, and that is the only actuator it
  has. `rest` is what this engine already moves things with — `wireShrink`
  reels a latch in through it, `Wire.collapse` hauls a rewrite's ends together
  with it, `wireBreathe` makes tissue move with it — because the span
  constraint serves it rather than fighting it. The gait is the fourth thing
  that writes it, which is why it is `wireTug` with a better clock rather than
  a new mechanism: the tug was driven by whichever packet last crossed, and
  this is driven by a phase the net owns.

  A correction shared by inverse mass moves both bodies and not their centre,
  so the swing itself carries nothing. What carries is the velocity it induces
  and `grip` then spends: a wire whose two ends damp differently keeps a step
  out of every cycle, and one whose ends match keeps nothing. `grip *
  fullness` is one such difference and `G`'s `anchor` is the other, riding the
  same cosine as the swing so a body grips exactly while its wires pull.

  `gaitCouple` and `gaitLag` are what make it a pattern rather than n
  twitches. A body's share of its own stroke is divided by the whole net's
  mass, so at unrelated phases the shares sum as sqrt(n) against a mass of n
  and a longer net moves itself *less*. Coupled with a lag, a chain settles at
  a fixed phase difference per wire, the shares add, and a fixed difference
  per wire is a travelling wave — which along a body is peristalsis. At lag 0
  it is plain synchrony, which is the pond-wide pulse this branch began by
  removing.

  The clock is a chemistry, and it is in the economy rather than beside it.
  Every body runs the four-chemical reactor of `docs/scratch.txt` §3 — A
  primary fuel, B active primer, C saturated catalyst, D reset inhibitor —
  over four reactions and a uniform outflow:

      r1  A -> B      r2  -> C (autocatalytic in C, saturating)
      r3  C -> D      r4  B + D ->

  **A is the gut** — what the body has swallowed off the grid and not yet
  used — so the reactor is fed by *eating* rather than by buying. `intake` is
  the share of digested food a body routes to it. A body with a full tank and
  an empty gut therefore has no clock, and running a metabolism needs no price
  of its own because it is food the body did not bank. What crosses out of the
  books should land on the ground through the same road rent uses, which is
  `upkeepExcrete` and ships at 0; until it is on, the reactor is a sink.

  **A is steady.** Its row decouples — it is fed from outside and only feeds
  B — so it settles to `J/(k1 + decay)` whatever it started at and the
  oscillation is the loop `B -> C -> D -| B`, a three-stage negative feedback
  carrying a positive self-loop on C. That is what makes the stability
  question answerable in closed form, and the answer is that the doc's own
  constants do not oscillate: writing `q` for C's net removal after the
  autocatalysis is subtracted, a complex pair crosses into the right half
  plane only when `L > (p+q+r)(pq+pr+qr) - pqr`, which in the limit `q -> 0`
  is `metabolicReset > (2 + 2*sqrt 2) * metabolicDecay`. The python has 0.4
  against 0.483. And `q` can only be driven toward zero when the saturation
  is weak, because C's self-gain at the fixed point depends on nothing but
  `base` and `sigma * C*` — not on the fuel, not on the autocatalytic rate.
  `src/pond/spectrum.ts` carries the derivation and prints it;
  `docs/metabolism-spec.md` §4 is the argument.

  The fuel window has both edges and that is the gate: starved, the reactor
  sits empty and still; fed, it runs a limit cycle whose period shortens as
  the fuel rises, so a full body strokes faster than a lean one; glutted, it
  sits saturated and still again. Nothing had to be added to get "a starving
  body does not undulate" — it is where the Hopf boundary is.

  The coupling is the doc's §4 transmission and it lives on the wire. A body
  broadcasts out of its principal port only, so it has one mouth and up to two
  ears and an Era is a leaf by construction. **One kind, one species**: `Tx`
  says which of the four a body speaks, seeded A on an Era, C on a Con, D on a
  Dup, and `Gx` how much it must hold first — the doc's own `H(x - G)`,
  rectified rather than stepped, on the *concentration* and not on the wave,
  so the impulse is in the chemistry. An Era sends the food it ate and the
  interior makes its own primer from it; sending the primer instead ships the
  fast currency into one neighbour and makes that body the instability the net
  is slaved to. The wire carries each species as one signed number both ends
  read (`Wire.fluxGut`, `fluxC`, `fluxD`), which is what makes a transfer
  antisymmetric by construction.

  Measured on six pinned Cons wired principal to auxiliary, against a period
  of about 100 frames: uncoupled the interior offsets are 18, 0 and 0 frames
  and never converge, which is no lock; at the shipped rate they are 10, 9
  and 9, which is a tenth of a cycle a wire and is a travelling wave. The
  band is narrow and both edges are the reactor's — broadcasting spends the
  very catalyst the sender's loop runs on, so at five times the shipped rate
  the senders run down and at eight times every reactor in the chain
  flatlines.

  There is no conduction delay, and taking it out was measured. Under the
  two-pool pathway a wire that relaxed toward the ask over `rest / speed` was
  the only thing that turned a one-frame cascade into a travelling wave. This
  reactor supplies its own delay: a body driven with catalyst takes about a
  tenth of a cycle to answer, because the catalyst has to run through D and
  quench the primer before the loop responds. Across the whole range of
  conduction speeds the interior lag sat at 10 to 12 frames either way and
  the wire's own lag only made it less uniform, so the dial is gone.

  **Two actuators, and the angle between them is the reactor's.** The rest
  length and the anchor both swing on `wave(C)`, which is one degree of
  freedom: a cycle that deforms and undeforms through the same shapes is
  reciprocal and nets zero displacement, which is Purcell's scallop theorem and
  the same objection that retired `wireTug`. So `grip` — the term that actually
  turns a transport kick into travel — is swung by **D** instead, at
  `gripSwing`. `dD/dt = k3*C - d*D` is a first-order lag, so D trails C by
  `atan(w/d)`: 51 degrees at the bottom of the fuel window and 65 at the top,
  measured at 53.5 on a lone fed body. Nobody chose those numbers. B leads C by
  84 degrees, near quadrature, and is not used by anything yet.

  **And the phase is a gene now.** `Sw` and `Gw` are what mixture of B, C and D
  the stroke and the grip each read. The three sit at fixed angles to one
  another, so a weighted sum of them reaches any phase — a lineage owns when in
  its own cycle it strokes and when it grips, which is what this heading has
  meant by "a phase something can *set*" since it was written. Seeded pure C
  and pure D, which is what they were welded to; the globals `gaitSwell` and
  `gripSwing` stay the magnitudes.

  It ships **on**: rate 15, broadcast 1, stroke 0.04, grip swing 0.5. The stroke is a
  quarter of what the two-pool pathway could drive, and that is the honest
  cost of a relaxation oscillator: its transitions are fast, so it slews a
  rest length hard, and above 0.04 a freshly latched pair spikes past the
  spin suite's bound. Locomotion comes mostly through the anchor, which
  modulates drag and is not limited by a span constraint.

  Not yet true, and this is the list. The rates are global constants and
  should stay so — they are the doc's §8 laws of the world, and a lineage
  that changed them locally would change what a wavelength means across its
  own net. But that leaves the *wavelength* global too, and nothing about a
  net's own shape reaches it: the conduction delay used to be the one thing
  that did, and it was measured not to work. What a body owns is `intake`,
  which species it speaks and how much it must hold to speak. The reactor is
  no longer private: its A *is* the four field species, so a body's clock
  comes out of the grid it is standing on, and an Era pushing food downstream
  is what makes a leaf its net's feeder. At the shipping dials that push moves
  a neighbour's influx by only about 8%, because digestion is twelve times
  faster than the transport feeding it — so structure is a gradient here and
  not yet a strong one. A body at the end of a chain whose principal is
  free only ever receives, and the catalyst it is fed runs to D, quenches its
  primer and stalls its loop — the doc's stoichiometry, and in a grown net it
  is a body waiting to latch. And an Era is
  both the leaf *and*, by `ERA_UPKEEP_RATIO` and its
  larger store, the fullest body on its wire — so the seeded head makes it
  the oar while the economy makes it the anchor, and those pull opposite
  ways. On the bench they are separate rigs and both carry a pair; which of
  them wins in a net where both act at once is a pond question and has not
  been asked.
- Communication within a net
- Transport of energy within a net. Lives in the demand gradient: a body gives
  to whichever neighbour is strictly needier, and `transportQuantum` sets
  whether that is a trickle or a packet. Demand decides whether to send;
  above zero the quantum decides how much, a body must hold a whole packet to
  send one, and what will not fit at the far end goes to the ground. The
  packet is also the pond's only clock — so this mechanic is asked to do two
  things and that is known. It ships at 0.5, the packet; at 0, the trickle,
  the momentum it carries is three orders below the pond's own noise.

  Demand crosses a net in the frame it appears, and that is the reason
  nothing here can carry a wave: the need field is solved to a fixpoint every
  frame, so a shortage anywhere is known everywhere at once and there is
  never anything left to propagate.

  There was a dial for the other behaviour — `requestReach`, which advanced
  the field a fixed number of hops a frame so that a need had a front. It
  shipped at 0 for its whole life and has been removed, but what it measured
  is why, and that stands. A global relaxation is global triage: every donor
  weighs its neighbour against the worst case anywhere on the net, so a dying
  body outranks a merely empty one however far away. One hop a frame is local
  equalisation, and a corridor of empty bodies absorbs a reservoir on its way
  past instead of relaying it — measured on twelve bodies with a reservoir at
  one end and a body in debt at the other, the reservoir was empty inside
  twenty frames and the patient ended on exactly zero, filled and then
  drained back into the corridor.

  So locality is not an improvement waiting to be switched on. What would pay
  for it is a need field that tells dying from empty by more than the
  difference between `rescueNeed`'s two branches, and until that exists,
  locality costs the economy more than it buys locomotion. That is the
  question, and a dial nobody moved was not an answer to it.
- Growth
- Breeding: nets connecting at free terminals
- Freeing terminals: death and detachment within a net
- Specialisation within a net

## Lineage

- Reproduction of agents
- Inheritance
- Mutation and other variation
- Selection
- Drift
- Speciation: nets becoming different from each other
- Development: from immigrant to member
- Immigration

## Ecology

- The energy economy and conservation
- The world: ground, its structure and regeneration
- Competition
- Cooperation and mutualism
- Niches and coexistence

## Principles

- Few well-tuned mechanics; room for emergence
- Selection is emergent; no fitness term
- Nothing true of a kind beyond what the calculus requires
- Persistence and difference are the point; nothing decays, averages or resets
- Passive by default: control over anything is acquired, not given
- Maintained order is paid for: anything held against the physics is work
  drawn from the one energy scalar
- Every mechanic ships at the setting that reproduces the pond before it
