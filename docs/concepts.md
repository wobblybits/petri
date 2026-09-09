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
- Locomotion
- Communication within a net
- Transport of energy within a net
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
