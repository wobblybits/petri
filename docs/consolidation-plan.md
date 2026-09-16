# Plan: one clock that nobody sets, and a net that has to have leaves

> **Superseded by `docs/metabolism-spec.md`**, which states the same design
> formally: the state, the reaction table, the two inequalities that decide
> which bodies oscillate and which relay, the coupling operator and its
> spectrum, the surviving parameters, and the phases with what would show
> each one worked. This document is kept for the analysis it carries — §0's
> spectrum, §3's timescale argument and §5's separation of `upkeep`'s three
> jobs — which the specification cites rather than repeats.

Written 2026-09-16, after building `docs/scratch.txt`'s reactor and auditing
what the pond ended up with. The reactor works and the pond is more
complicated than it needs to be. This is what to take out, what to move, and
the analysis that decides each one.

Three things the author asked for, and all three fall out of the same result:
no external clock, use the Laplacian to decide what will work, and give only
Eras an intake.

---

## 0. The finding, which is a spectrum

Under the coupling as built — a body broadcasts out of its **principal** port
only — the signalling graph is *functional*: every body has out-degree at most
one. That makes its spectrum exactly computable instead of something to sweep
for, and computing it on the two nets in `nets/` says:

|  | bodies | out-degree > 0 | cycles | cycle lengths |
|---|---:|---:|---:|---|
| `deep-87` | 87 | 86 | 2 | 2, 2 |
| `mixed-308` | 308 | 308 | 9 | all 2 |

Every cycle is length two, and a length-two cycle is two principals facing
each other — which is a **redex**, the thing the rewriter exists to consume.

For a functional graph the adjacency `A` is a permutation on its cycles and
nilpotent everywhere else, so its eigenvalues are the roots of unity around
each cycle plus zeros. With every cycle of length two those roots are ±1, all
real, and the directed Laplacian `L = D_out − A` has eigenvalues in {0, 1, 2}.
**Not one of them has an imaginary part.**

A travelling wave is an eigenmode with a complex eigenvalue. There are none
here and there structurally cannot be: a principal cycle longer than two is a
shape interaction combinators rewrite away. So **principal-out coupling can
never sustain a travelling wave in this pond**, and every phase gradient I
measured was a driven transient down a nilpotent tree rather than a mode.

That explains the measurements it took a day to gather. The chain's lag was
small and uneven and decayed along its length; its tail died because nothing
was downstream of it; and turning the broadcast up gave synchrony — the λ = 0
mode — rather than a sharper wave.

**The conclusion is not that the coupling is wrong.** A driven feed-forward
tree is a perfectly good peristaltic medium, and it is what a net shaped like
these is. What it needs is a *pacemaker at a leaf* and relays that amplify
rather than attenuate. Which is the next two sections, and they are the same
change.

---

## 1. Only Eras eat for the reactor

**Concepts heading: Metabolism, and Net > Specialisation.**

`intake` — the share of digested food a body routes to its reactor — becomes
a thing only an Era seeds above zero, exactly as `Tx` already seeds what a
kind broadcasts. A Con or a Dup that breeds an intake can still have one;
what changes is where a fresh pond starts.

Three things follow, and the third is the one worth having.

**It is the pacemaker.** An Era with an intake sits above the Hopf boundary
and runs a limit cycle of its own. A Con with none sits below it: its primer
arrives from a neighbour rather than from its gut, so it does not
self-oscillate, it *responds*. That is an excitable medium with pacemakers,
which is the only arrangement that carries a travelling wave on a tree.

**It gives the doc's §6.1 a job.** An Era already broadcasts A and B — the
food it ate and the primer it made. If Cons and Dups have no intake, that
broadcast is the *only* way a net's interior gets fuel. A leaf stops being
decoration and becomes the organ the rest of the net eats through.

**It is the missing ingredient `energy-chemistry-plan.md` §0 names.** That
document's central complaint is that "a net buys it transport and rewrite
partners, never *access to energy it could not otherwise reach*" — and so
complexity has no reason to appear. Era-only intake is exactly that access: a
Con's clock requires a net with a leaf in it. Nothing else in this simulation
has ever made structure load-bearing for energy.

**What to check before believing it.** A Con with no intake must be
*excitable* and not merely dead. With B near zero the autocatalytic step
`k2 * B * (base + C)` cannot amplify anything, so a Con starved of primer is
a passive cable that attenuates. The relay only works if the Era's broadcast
of B is enough to hold its neighbours above the amplification threshold and
below the Hopf boundary. That is a two-sided condition on one quantity and it
is exactly what the Jacobian answers — see §4.

---

## 2. Delete the clock

**Concepts heading: Locomotion.**

`metabolicRate` multiplies every reaction rate. It exists for one reason: to
set the period. That is an external clock with a slider on it, and it should
go.

It is a pure reparameterisation, so deleting it costs nothing: state the rate
constants in per-second units and the dial is redundant. What is left is that
the *period* is then a consequence rather than a setting — of the pacemaker's
income, its pool sizes, and the rate constants, in that order of how much a
lineage controls them.

The Hopf condition is scale-free — `metabolicReset > (2 + 2√2) *
metabolicDecay` is a ratio — so the absolute timescale is free to choose, and
choosing it in seconds rather than in "reaction units times a dial" is the
whole change.

**The one thing to get right.** Period and starvation buffer are the same
number in a relaxation oscillator: a cycle of T seconds implies a pool that
holds about T seconds of throughput. A two-second gait therefore buffers two
seconds, and a body must survive minutes. That is what the tank is for and it
is why the next section does *not* delete it.

---

## 3. Two energies, and why that is not the excess

The pond has `extra`, the tank, and B, the primer. They are made by one
reaction from one A and part at one number. The obvious simplification is to
merge them, and it is wrong.

**The catalyst cannot be a store.** Measured: C's trough is zero by
construction — that is what a relaxation cycle is — so a standing charge
against it drives it to zero and the loop never restarts. At a rent of 0.002
against a peak of 1.2 the reactor was dead in every arm.

**The primer can be**, but it does not want to be. Measured in the standalone
integration, B survives a standing drain up to 0.2 per reaction-unit with its
trough well clear of zero, which is a couple of hundred times the rent this
pond charges. So the blocker is not dynamics.

**The blocker is timescale, and it is real structure rather than debt.** The
reactor turns its pool over many times per unit of matter — that is what a
currency is — so B measures seconds and the tank measures minutes. One pool
cannot do both: sized for the cycle it starves the body, and sized for the
body it cannot cycle. The conversion between them is one number,
`metabolicYield`, and one number is the right price for two timescales.

So: **keep both, and say why in `concepts.md`.** What is excess is
everything in §5.

---

## 4. The Laplacian, and how to stop sweeping

Two analyses, and between them they replace the parameter sweeps that have
eaten most of two sessions.

**The Jacobian decides what one body does.** Linearise the reactor about its
fixed point; A's row decouples; the 3×3 over (B, C, D) is a three-stage
negative feedback with a positive self-loop, characteristic polynomial
`(x+p)(x+q)(x+r) + L`, and the Hopf boundary is `L > (p+q+r)(pq+pr+qr) −
pqr`. This is already written down in `metabolicFuel` and it is how the
shipped constants were chosen after a forty-thousand-draw random sweep found
nothing.

For §1 it answers the two-sided question directly: find the primer influx at
which a body crosses the Hopf boundary. **Eras are seeded above it and
Cons below it**, and the gap between "amplifies" and "self-oscillates" is the
band the broadcast has to land a neighbour in.

**The Laplacian decides what a net does.** For reactors coupled through
species σ with strength ε on a graph with Laplacian `L`, the variational
equation about the synchronous state block-diagonalises into one mode per
Laplacian eigenvalue:

    δ̇ₖ = [ J(t) − ε λₖ P ] δₖ

so the stability of each spatial mode is a single scalar function of `ε λₖ`.
Synchrony is stable exactly when that is negative for every λₖ ≠ 0. Where it
is positive the pond breaks into the corresponding mode, and a complex λ is
what makes that mode *travel*.

That is the tool for picking `metabolicDiffuse`, and it is the tool that
would have said in ten minutes what §0 took a day of sweeping to find.

**What it says to build.** If travelling eigenmodes are wanted rather than
driven transients, the coupling graph needs complex spectrum, and
principal-out cannot give it. The options, in order of how much they cost:

1. **Accept driven waves.** Pacemaker at a leaf, relays inward, wave speed
   set by the per-hop response. This is §1 and it needs no new mechanism.
2. **Couple on all three ports** rather than the principal only. The graph
   becomes the net's own topology — symmetric, so real spectrum, so standing
   patterns rather than travelling ones. Turing, not peristalsis.
3. **Asymmetric on all three ports**: send forward at one rate and back at
   another. Non-normal with complex spectrum on any cycle, and it makes the
   principal/auxiliary distinction a *gradient* rather than a gate.

Option 1 first, because it is free and because §1 wants it anyway. Option 3
is the interesting one and the analysis above is how to decide it without
building it twice.

---

## 5. What comes out

Counted, with what each is doing now.

| out | why |
|---|---|
| `metabolicRate` | the external clock; a reparameterisation (§2) |
| `upkeep` | three jobs, none of them its own (below) |
| `intake` on Cons and Dups | Era-only seeding (§1) |
| `metabolicWave` | derivable from the pool it half-saturates rather than dialled |
| `Gx` as shipped | seeds to zero, so it gates nothing; either seed it or drop it |

**`upkeep`'s three jobs, separated.** It is the clearest case of one mechanic
doing several and it is measured.

- *The standing cost* is `metabolicDecay`, which is the doc's own diagonal
  decay matrix. Duplicate; delete the rent half.
- *Death* is the starvation window, already built and already the one that
  fires first: on a bare dish, rent and timer kill at 30.1s, timer alone at
  28.8s, rent alone at 69.0s, neither never.
- *The Era's income* is the **negative** rent `upkeepRateOf` pays a ground
  producer, so zeroing rent zeroes an Era's whole income. It belongs on
  `yEra`, which the function's own comment already says, and raising `yEra`
  restores Era numbers (60 → 78 over a minute). With §1 this matters more,
  not less: the Era is the organ the net eats through.
- *Rationing rewrites* is the job with no other home. Without rent,
  annihilations go 112 → 160 over a minute and the population settles 35%
  lower, because bodies that can afford to rewrite do. This belongs to the
  **rewrite's own price**, not to a standing charge on everyone —
  `rewriteShareOf` and `bodyValue` are where it should live.

---

## 6. Order of work

1. **Era-only intake**, and the Jacobian check that a fuelled Con amplifies
   without self-oscillating. Nothing else moves. Look at a planted net: the
   wave should start at leaves.
2. **Delete `metabolicRate`**, restating the constants in seconds. Pure
   reparameterisation; the pond should not move at all, which
   `state-hash` can check.
3. **Move `upkeep`'s jobs out** — `yEra` for the income, the rewrite price
   for the rationing — then set it to zero and watch the population and the
   commute rate, which are the two numbers it was holding up.
4. **The master stability function**, on a real net's Laplacian, to pick the
   broadcast rate instead of sweeping it.
5. **Then, and only then**, option 3 above if driven waves are not enough.

---

## 7. What this deliberately does not do

- It does not merge the tank and the primer. Two timescales, one conversion,
  and §3 has the measurement.
- It does not add a channel. Making the catalyst a field species would unify
  signalling with metabolism, and it is the most interesting unbuilt idea in
  the audit, but the vocabulary is deliberately four and the reactor's
  throughput would saturate a cell in a frame without a conversion.
- It does not fix `SENSE_SCALE`, which conserved excretion invalidated and
  which is measured, not guessed. That is its own job and it is next after
  step 1.
