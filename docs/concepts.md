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

  Not yet true: the clock's *rate* is a free global constant, so bodies
  cannot drift out of lock and back — there is no coupling term, and a
  per-body rate wants one first. `Wn` already carries the mean of a body's
  wired neighbours and is the obvious place for it.
- Communication within a net
- Transport of energy within a net. Lives in the demand gradient: a body gives
  to whichever neighbour is strictly needier, and `transportQuantum` sets
  whether that is a trickle or a packet. Demand decides whether to send;
  above zero the quantum decides how much, a body must hold a whole packet to
  send one, and what will not fit at the far end goes to the ground. The
  packet is also the pond's only clock, which is the whole reason locomotion
  cannot yet have a gait — so this mechanic is asked to do two things and that
  is known. It ships at 0.5, the packet; at 0, the trickle, the momentum it
  carries is three orders below the pond's own noise.
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
