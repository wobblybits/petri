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
- Metabolism: eating, upkeep, excretion
- Memory
- Learning
- Regulation: behaviour that follows state without learning
- Death

## Net

- Identity: what counts as one organism
- Shape. Lives in wire physics. Tension is held by energy flowing along a
  wire, so a net that moves no energy is slack. Not yet true: stiffness is a
  free global constant.
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

  The clock is a metabolism. Every body runs a two-species autocatalytic
  pathway — the Brusselator, the standard minimal chemical oscillator and the
  usual toy for Belousov-Zhabotinsky — fed at a rate set by its own fullness,
  and the activator is the wave. It runs exactly when
  `influx * fullness > 1 + base^2`, which at the shipped constants is half a
  tank: a body over half full undulates and one under it is perfectly still.
  So the gait is something the pond produces out of what it did with its
  energy, rather than a metronome bolted beside it, and the coupling is the
  activator diffusing along wires — a reaction that diffuses carries a front,
  and a front running down a chain is peristalsis.

  Selkov's own equations came first and could not do it: `adp * (atp * adp -
  k)` makes zero a stable fixed point, so the pathway collapses the moment
  its product dips and never restarts, and its window `v < k^(3/2)` is the
  wrong shape — a well-fed body falls out of the top of it and goes still.

  Not yet true, and this is the list. Nothing about the pathway is heritable:
  `metabolicInflux` and `metabolicBase` are global constants, where which
  side of `1 + base^2` a lineage chooses to live on is exactly the sort of
  thing it should own — that wants two more rows on `G`, or a head off `X`,
  which is what the expression head is for. And an
  Era is both the leaf *and*, by `ERA_UPKEEP_RATIO` and its larger store, the
  fullest body on its wire — so the seeded head makes it the oar while the
  economy makes it the anchor, and those pull opposite ways. On the bench they
  are separate rigs and both carry a pair; which of them wins in a net where
  both act at once is a pond question and has not been asked.
- Communication within a net
- Transport of energy within a net. Lives in the demand gradient: a body gives
  to whichever neighbour is strictly needier, and `transportQuantum` sets
  whether that is a trickle or a packet. Demand decides whether to send;
  above zero the quantum decides how much, a body must hold a whole packet to
  send one, and what will not fit at the far end goes to the ground. The
  packet is also the pond's only clock — so this mechanic is asked to do two
  things and that is known. It ships at 0.5, the packet; at 0, the trickle,
  the momentum it carries is three orders below the pond's own noise.

  How far demand travels in a frame is `requestReach`, and it is the reason
  nothing here can carry a wave. At its shipped 0 the need field is solved to
  a fixpoint every frame, so a shortage anywhere is known everywhere at once
  and there is never anything left to propagate. Above 0 it advances that
  many hops a frame and a need has a front. The fixpoint is the same either
  way; only the time to reach it changes.

  That is a trade and not an improvement, which is why it ships off. A global
  relaxation is global triage — every donor weighs its neighbour against the
  worst case anywhere on the net, so a dying body outranks a merely empty one
  however far away. One hop a frame is local equalisation, and a corridor of
  empty bodies absorbs a reservoir on its way past instead of relaying it:
  measured on twelve bodies with a reservoir at one end and a body in debt at
  the other, the reservoir is empty inside twenty frames and the patient ends
  on exactly zero, filled and then drained back into the corridor. Not yet
  true: the field cannot tell dying from empty by enough to survive being
  local, and until it can, locality costs the economy more than it buys
  locomotion.
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
