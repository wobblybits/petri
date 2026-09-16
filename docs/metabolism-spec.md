# Specification: the pond's metabolism

Status: **specification, not yet built.** Supersedes the direction in
`docs/consolidation-plan.md`, which is kept for the analysis behind §5.2 and
§7. `docs/scratch.txt` is the source document this implements;
`docs/mka-plan.md` records which of its clauses were taken and which were
refused, with the measurement for each.

This document states the target: the state a body holds, the reactions over
it, the conditions that decide which bodies oscillate and which relay, the
operator that couples them, and the parameters that survive. It is written to
be checkable — every design choice below is either a stated equation, a
stated inequality, or a measurement with a number.

---

## 1. Intent

One metabolism, doing the work that is currently split across four systems:
the field's four species, the body's energy tank, the gut with its expression
table, and the reactor. It must supply, in this order of priority:

1. **A clock nobody sets.** A body's period is a consequence of what it eats,
   its pool sizes and the rate constants — in that order of how much a lineage
   controls them. There is no global time-scale dial.
2. **A reason for a net to exist.** A body's access to fuel depends on the
   structure it is part of, not only on where it is standing.
3. **A gait that travels.** A phase difference per wire that is consistent
   along a body, so that a net undulates rather than pulsing.
4. **Death as a consequence.** Running out is the death; nothing else needs
   to charge rent to kill.

---

## 2. State

### 2.1 Per body

| symbol | name | width | units | in the pond's books |
|---|---|---:|---|---|
| `gut` | **A**, primary fuel | 4 | tank | yes |
| `extra` | the tank | 1 | tank | yes |
| `react[0]` | **B**, active primer | 1 | reactor | no |
| `react[1]` | **C**, saturated catalyst | 1 | reactor | no |
| `react[2]` | **D**, reset inhibitor | 1 | reactor | no |
| `starve` | seconds at depletion | 1 | s | — |

`gut` is indexed by field channel, so the doc's single primary fuel is this
pond's four species: what a body swallowed, per channel. `extra`, `B`, `C`
and `D` are scalars.

### 2.2 Per wire

`fluxGut` (4) and `fluxB`, `fluxC`, `fluxD` (1 each): what crossed this
frame, signed from the wire's `a` end toward its `b` end. Written once per
wire by the wire pass and read by both of its bodies, so a transfer is
antisymmetric by construction and no accumulation order can matter.

### 2.3 Units, and why there are two

`extra` and `B` are both usable energy and they are **not** merged. The
reason is timescale, and it is structural rather than historical:

- A relaxation oscillator's pool holds about one period's worth of
  throughput. A two-second gait therefore buffers two seconds.
- A body must survive minutes without food.

One pool cannot do both: sized for the cycle it starves the body, and sized
for the body it cannot cycle. `metabolicYield` is the conversion, one number,
and it is the price of two timescales.

`C` is ruled out as a store outright and by measurement: its trough is zero —
that is what a relaxation cycle *is* — so a standing charge against it drives
it to zero and the loop never restarts. At a rent of 0.002 against a peak of
1.2 the reactor was dead in every arm tried.

---

## 3. Reactions

### 3.1 The table

Rates are per **second**. There is no time-scale multiplier; see §6.3.

| # | reaction | rate |
|---|---|---|
| h | field → `gut` | the sampled mouthful: `uptakeVmax`, `ks`, one budget shared by what is standing in the cell |
| r1 | `gut` → `extra` and `B` | digestion, at the recipe's rate; split by `intake`, converted to reactor units by `metabolicYield` |
| r2 | → `C` | `k₂ · B · (base + C) / (1 + σC)` |
| r3 | `C` → `D` | `k₃ · C` |
| r4 | `B + D` → | `k₄ · B · D` |
| w | `B` → | `metabolicWork · gaitSwell · \|wave\|` |
| δ | `B`, `C`, `D` → | `d` on each |
| x | `extra` → field | the excretion rows, mass action on the tank |
| t | `extra` ↔ neighbour | the demand gradient, in packets |
| b | species → neighbour | §5 |

`base` is not in the source document and is required by it: `r2` carries a
factor of `C`, so `C = 0` is absorbing and `δ` drives `C` there. Without it
any reactor that empties stays empty.

The wave is `2C / (K_w + C) − 1`: bounded, signed, no arbitrary
normalisation. It drives the rest length and the anchor.

### 3.2 Invariants

1. **`gut` and `extra` are counted; `B`, `C` and `D` are not.** The reactor
   turns its pool over many times per unit of matter, which is what a
   currency is, so its pools are in their own units and outside the books.
2. **What crosses that boundary must leave.** Food routed to the reactor by
   r1 is paid onto the ground through the same road rent uses.
3. **The dish is driven; the bodies are conservative.** Logistic regrowth
   mints and is meant to; every body-side reaction moves matter or converts
   it, and `energy.test.ts` asserts this with the drive off.
4. **δ is not the decay the standing rules forbid.** Those protect the pond —
   lineages, learned weights, memory — where forgetting erases a difference
   something paid for. δ is a well-stirred vat's outflow, it is what closes
   the loop, and `D` has no other sink at all.

---

## 4. Roles: who oscillates and who relays

### 4.1 The two conditions

Both are conditions on `v`, a body's **influx of B**, and both are computed
from the same linearisation. `v` reaches an Era from its own gut and a Con or
Dup from a neighbour's broadcast, so one analysis covers both.

**Excitability.** Linearising `C` about zero, `dC/dt ≈ (k₂B − k₃ − d)·C +
k₂B·base`, so an arriving pulse of catalyst grows rather than decaying when

        k₂ · B  >  k₃ + d                                      (E)

Below (E) a body is a passive cable: it attenuates what it is sent and the
wave dies within a hop or two.

**Self-oscillation.** `A`'s row decouples, so the dynamics are the 3×3 over
(B, C, D): a three-stage negative feedback `B → C → D ⊣ B` carrying a
positive self-loop on `C`. With `p = k₄D* + d` for B's removal, `q = (k₃+d) −
∂r₂/∂C` for C's *net* removal, `r = d` for D's, and `L = k₃ · ∂r₂/∂B · k₄B*`
for the loop gain, the characteristic polynomial is `(x+p)(x+q)(x+r) + L` and
a complex pair crosses into the right half plane when

        L  >  (p+q+r)(pq+pr+qr) − pqr                          (H)

In the limit `q → 0` this collapses to `k₃ > (2 + 2√2)·d`, about 4.83, which
is why the source document's own constants never oscillate: its python has
0.4 against 0.483.

**The design requirement.** A pacemaker is above (H). A relay is above (E)
and below (H), and close enough to (H) that an arriving pulse makes a large
excursion before returning — the standard excitable regime.

        relay:      (E) satisfied,  (H) not,  and near the boundary
        pacemaker:  (H) satisfied

### 4.2 Seeding by kind

Not rules keyed on kind — seeds a lineage drifts from, the way
`seedProduction` and `Tx` already work.

| kind | `intake` | `Tx` broadcasts | role |
|---|---|---|---|
| Era | above zero | A and B | pacemaker and the net's feeder |
| Con | zero | C | relay, excitatory |
| Dup | zero | D | relay, inhibitory; resets the front |

**Only an Era eats for its reactor.** A Con's or a Dup's primer arrives from
a neighbour, so its clock requires a net with a leaf in it. This is the
missing ingredient `energy-chemistry-plan.md` §0 names: until now a net
bought its members transport and rewrite partners but never *access to energy
they could not otherwise reach*, and so complexity had no reason to appear.

---

## 5. Coupling

### 5.1 The operator

A body broadcasts out of its **principal port only**, at `Tx[σ]` times
whatever it holds above `Gx[σ]`, rectified:

        flux(σ) = metabolicDiffuse · Tx[σ] · max(0, x_σ − Gx[σ])

bounded to a quarter of the source's stock and a quarter of the sink's room,
so a body's species cannot be overdrawn by its own broadcast plus the far end
of each of its three wires.

Rectified and not a step: a Heaviside rings under explicit Euler and gives a
gate nothing to move along. Mass action on the concentration, so the impulse
is in the chemistry — a body sends most at its catalyst's peak and nothing at
its trough, with no clock and no threshold needed to make it a pulse.

### 5.2 The spectral condition

Principal-out makes the signalling graph **functional**: out-degree at most
one. Its adjacency is therefore a permutation on its cycles and nilpotent
everywhere else, and its eigenvalues are exactly the roots of unity around
each cycle plus zeros.

Measured on the library:

| net | bodies | out-degree > 0 | cycles | lengths |
|---|---:|---:|---:|---|
| `deep-87` | 87 | 86 | 2 | 2, 2 |
| `mixed-308` | 308 | 308 | 9 | all 2 |

Every cycle has length two, and a two-cycle is two principals facing each
other — a **redex**, which is the shape the rewriter consumes. So the
directed Laplacian's eigenvalues lie in {0, 1, 2}, all real, and:

> **There is no travelling eigenmode under principal-out coupling, and there
> structurally cannot be one.**

Every phase gradient measured before this was a driven transient down a
nilpotent tree. That is why a chain's lag was uneven and decayed along its
length, why its tail died, and why turning the broadcast up produced
synchrony — the zero mode — rather than a sharper wave.

**This specification therefore builds a driven wave, not a mode**: a
pacemaker at a leaf, relays inward, and a speed set by the per-hop response
time. §4 is what makes that work.

For reference, the condition that would give a genuine travelling eigenmode.
For reactors coupled through species σ with strength ε on a graph with
Laplacian `L`, the variational equation about synchrony block-diagonalises
into one mode per eigenvalue `λₖ`:

        δ̇ₖ = [ J(t) − ε λₖ P ] δₖ

Synchrony is stable exactly when the resulting exponent is negative for every
`λₖ ≠ 0`; where it is positive the pond breaks into that mode, and a
**complex** `λₖ` is what makes the mode travel. Reaching one needs either a
principal cycle longer than two — which the rewriter forbids — or coupling on
all three ports with different forward and backward rates, which is
non-normal with complex spectrum on any cycle. That is the door, and this
specification does not open it.

---

## 6. Parameters

### 6.1 Global constants

The source document's §8 is right that these are laws of the world: a lineage
that changed them locally would change what a wavelength means across its own
net.

`k₂` (`metabolicCat`), `k₃` (`metabolicReset`), `k₄` (`metabolicQuench`),
`σ` (`metabolicSigma`), `d` (`metabolicDecay`), `base` (`metabolicBase`),
`metabolicYield`, `metabolicDiffuse`, `metabolicWork`, `gaitSwell`,
`starveTime`.

### 6.2 Per-agent genes

The source document's §8 per-agent set, less the ones it lists that this pond
has no use for.

| gene | width | what it decides |
|---|---:|---|
| `Tx` | 4 | which species this body broadcasts |
| `Gx` | 4 | how much it must hold before it broadcasts any |
| `intake` | 1 | the share of digested food routed to its reactor |
| `ks` | 4 | uptake affinity per species |
| `X` | 8 rows | what it produces and what it can digest |
| `G`, `g0` | 5 | the gait's anchor |

### 6.3 Deleted

| out | why |
|---|---|
| `metabolicRate` | the external clock. It multiplies every rate and exists only to set the period, so it is a reparameterisation: state the constants in seconds and it is redundant. The Hopf condition is a ratio, so the absolute timescale was always free. |
| `upkeep` | three jobs, none of them its own. §7. |
| `metabolicWave` | derivable from the pool it half-saturates rather than dialled. |
| `Gx` seeded at zero | it gates nothing at the seed; either seed it inside the band §4.1 names, or drop the vector. |

---

## 7. Death, cost, and rationing

`upkeep` is one mechanic doing three jobs. Each has a home and the
measurements are in hand.

**The standing cost is `δ`**, the source document's own diagonal decay
matrix. A body must keep eating to hold its pools against it. Rent duplicates
this.

**Death is the starvation window.** A body whose primer stays below `ε` for
longer than `starveTime` dies. Measured on a bare dish: rent and window kill
at 30.1 s, window alone at 28.8 s, rent alone at 69.0 s, neither never. A fed
body's clock never starts. The window already fires first.

**The Era's income is `yEra`.** `upkeepRateOf` currently pays a ground
producer *negative* rent, so an Era's whole income is the rent and zeroing
rent zeroes it — which is why a no-rent pond collapsed. The function's own
comment already says this belongs on the uptake yield, and raising `yEra`
restores Era numbers (60 → 78 over a minute). Under §4.2 this matters more,
not less: the Era is the organ the net eats through.

**Rationing rewrites belongs to the rewrite's price.** This is the job with
no other home today. Without rent, annihilations go 112 → 160 over a minute
and the population settles 35% lower, because bodies that can afford to
rewrite do. `intake` can make bodies equally poor but only by starving the
tank of income, which took commutes from 21 to 1. The price of a rewrite is
`rewriteShareOf` and `bodyValue`, and that is where a brake on rewriting
belongs — not on a standing charge against everyone.

---

## 8. Phases

Each phase states what it changes and what would show it worked. No phase
ships at a neutral value; the discipline is `CLAUDE.md`'s — on, visible,
behind a slider, and the author looks.

**Phase 1 — Era-only intake.** `seedTraits` gives an Era an `intake` and a
Con or Dup zero. Compute (E) and (H) from §4.1 for both, and choose
`metabolicYield` and `metabolicDiffuse` so that an Era clears (H) and a Con
fed by one lands between (E) and (H).
*Acceptance:* a planted net's wave starts at leaves and runs inward; a Con
with no Era neighbour has no clock; the pond's population and commute rate
do not fall.

**Phase 2 — delete `metabolicRate`.** Restate the constants in per-second
units. A pure reparameterisation.
*Acceptance:* `state-hash` unchanged, or changed only by float ordering.

**Phase 3 — move `upkeep`'s jobs out.** `yEra` for the income, the rewrite
price for the rationing, then set `upkeep` to zero.
*Acceptance:* population and commutes hold within the noise of the shipped
pond; starvation deaths still occur on a bare dish.

**Phase 4 — the master stability function.** Build the Laplacian of a real
net, evaluate §5.2's exponent across `metabolicDiffuse`, and pick the rate
from it rather than by sweeping.
*Acceptance:* the predicted synchrony boundary matches the measured one on a
chain within a factor of two.

**Phase 5 — `SENSE_SCALE`.** Conserved excretion stopped the scent mint, so
signal amplitude fell about fivefold and the constant measured against the
minted field is wrong; with the shipping parameters every body in a foraging
soup reads a negative trail. Remeasure the p90 rather than rescaling by hand,
which is what made the constant trustworthy the first time.
*Acceptance:* a foraging soup reads a positive trail and forages above chance.

---

## 9. What this does not do

- **It does not merge the tank and the primer.** Two timescales, one
  conversion; §2.3 has the measurement.
- **It does not add a field channel.** Making the catalyst a field species
  would unify signalling with metabolism and is the most interesting unbuilt
  idea in the audit, but the vocabulary is deliberately four and the
  reactor's throughput would saturate a cell in a frame without a conversion.
- **It does not open the travelling-eigenmode door.** §5.2 states the
  condition and what it would cost.
- **It adds no learning rule.** The source document's §7.1 trophic gate is
  refused for the reason `mka-plan.md` §3 gives: the pond has one learning
  rule and a second would be a second.
- **It keeps no rule keyed on kind.** Everything in §4.2 is a seed.
