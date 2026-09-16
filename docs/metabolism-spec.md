# Specification: the pond's metabolism

`docs/scratch.txt` is the source document this implements. Where a clause of
it was refused, the refusal and its measurement are stated here — there is no
second document to consult.

This document states the target and what stands between here and it: the state
a body holds, the reactions over it, the conditions that decide which bodies
pace and which relay, the operator that couples them, and the parameters that
survive. Every design choice below is a stated equation, a stated inequality, or
a measurement with a number.

**Status.** §2–§7 are built. §8 is what is left and in what order.

---

## 1. Intent

The metabolism must supply, in this order of priority:

1. **A clock nobody sets.** A body's period is a consequence of what it eats,
   its pool sizes and the rate constants — in that order of how much a lineage
   controls them. There is no global time-scale dial.
2. **A reason for a net to exist.** A body's access to fuel depends on the
   structure it is part of, not only on where it is standing.
3. **A gait that travels.** A phase difference per wire that is consistent
   along a body, so that a net undulates rather than pulsing.
4. **Death as a consequence.** Running out is the death; nothing else needs to
   charge rent to kill.

**There were four systems and there are two.** The field's species in the gut,
the tank, the reactor's pools, and the expression rows. The rows are gone — they
said what a body produced and what it could digest, and nothing excretes and
there is one thing to digest. The gut and the tank are one substance on two
sides of a membrane. What is left that is genuinely separate is **matter**, in
the gut and the tank and the ground, and the **reactor**, whose pools are in
their own units and outside the books; §2.3 is why those two do not merge.
Signal is a third thing and is not matter at all.

---

## 2. State

### 2.1 Per body

| symbol | name | width | units | in the pond's books |
|---|---|---:|---|---|
| `gut` | **A**, primary fuel — the ground, and only the ground | 1 | tank | yes |
| `extra` | the tank | 1 | tank | yes |
| `react[0]` | **B**, active primer | 1 | reactor | no |
| `react[1]` | **C**, saturated catalyst | 1 | reactor | no |
| `react[2]` | **D**, reset inhibitor | 1 | reactor | no |
| `starve` | seconds at depletion | 1 | s | — |

**A body eats the ground and nothing else.** It used to swallow a sample of
all four channels, so a scent was a meal as well as a message: `deposit` minted
food by shouting, a body could eat its own signal back, and the conservation
books had to carry four columns. Matter is the ground and signal is the other
three, and nothing crosses. The array is still four wide; three of its entries
are always zero and are the next thing to go (§8).

### 2.2 Per wire

`fluxGut` (4) and `fluxB`, `fluxC`, `fluxD` (1 each): what crossed this frame,
signed from the wire's `a` end toward its `b` end. Written once per wire by the
wire pass and read by both bodies, so a transfer is antisymmetric by
construction and no accumulation order can matter.

### 2.3 Two energies, and why that is not the excess

`extra` and `B` are both usable energy and they are **not** merged:

- A relaxation oscillator's pool holds about one period's worth of throughput.
  A two-second gait therefore buffers two seconds.
- A body must survive minutes without food.

One pool cannot do both: sized for the cycle it starves the body, sized for the
body it cannot cycle. `metabolicYield` is the conversion, one number, and it is
the price of two timescales.

`C` is ruled out as a store by measurement: its trough is zero — that is what a
relaxation cycle *is* — so a standing charge against it drives it to zero and
the loop never restarts. At a rent of 0.002 against a peak of 1.2 the reactor
was dead in every arm tried.

**This argument decides §5.2 as well.** If the gut is the slow currency and the
primer the fast one, then what travels between bodies is the gut.

---

## 3. Reactions

### 3.1 The table

Rates are per **second**; §6.4 is why there is still a multiplier in the code.

| # | reaction | rate |
|---|---|---|
| h | field → `gut` | the sampled mouthful: `uptakeVmax`, `ks`, one budget shared by what is standing in the cell |
| r1 | `gut` → `B` at `intake` per second, and `extra` with the rest | digestion; `metabolicYield` converts what the reactor takes |
| r2 | → `C` | `k₂ · B · (base + C) / (1 + σC)` |
| r3 | `C` → `D` | `k₃ · C` |
| r4 | `B + D` → | `k₄ · B · D` |
| w | `B` → | `metabolicWork · gaitSwell · \|wave\|` |
| g | — | the grip swing, `gripSwing · wave(D·d/k₃)`, on the drag law |
| δ | `B`, `C`, `D` → | `d` on each |
| t | `extra` ↔ neighbour | the demand gradient, in packets |
| b | species → neighbour | §5 |

`base` is not in the source document and is required by it: `r2` carries a
factor of `C`, so `C = 0` is absorbing and `δ` drives `C` there.

The wave is `2C / (K_w + C) − 1`: bounded, signed, no arbitrary normalisation.
It drives the rest length and the anchor.

**A second wave, from D, drives the grip.** `grip` makes a body's drag depend
on how full it is, and that is the only thing that turns a stroke into travel —
a transport kick is equal and opposite, so it cancels at the centre of mass
unless the two ends damp differently. `gripSwing` modulates that difference with
the cycle, so a body grips hardest at one point and slips at another.

**It is driven by D and not by C, and that is the whole of it.** The rest
length and the anchor are both `wave(C)`, so they are one degree of freedom: a
cycle that deforms and undeforms through the same shapes is reciprocal, and a
reciprocal cycle nets zero displacement — Purcell's scallop theorem, and the
same objection that retired `wireTug` ("one degree of freedom and a race"). A
second actuator has to be out of phase with the first, and **the reactor
already supplies the angle**: `dD/dt = k₃C − dD` is a first-order lag, so D
trails C by `atan(ω/d)`.

| | phase relative to C |
|---|---|
| B | **+84°** — near quadrature, and unused |
| C | 0° — rest length and anchor |
| D | **−51°** at the bottom of the fuel window, **−65°** at the top — the grip |

Measured on a lone fed Con at the shipping dials: **53.5°**, against 51–65°
from the linearisation. Nobody chose those numbers; they are `k₃` and `d`.

**And now a lineage can.** `Sw` and `Gw` are what mixture of the three each
actuator reads. Three phasors spanning more than 180° means a weighted sum
reaches any phase and amplitude, so a body owns *when in its own cycle* it
strokes and when it grips — which is what the Locomotion heading means by "a
phase something can set", and what was previously welded to `k₃` and `d` for
every body in the pond forever.

Both, and not one against the other as a reference: the impulse a stroke works
against is the gut broadcast, which is timed by the chemistry, so each
actuator's angle to *that* is its own degree of freedom. `Sw` seeds pure C and
`Gw` pure D, which is exactly what they were welded to, so a fresh pond is
unchanged. Each species enters in units its own balance makes natural —
`B·k₂/(k₃+d)`, `C`, `D·d/k₃`, all near one at a fixed point — so one
`metabolicWave` serves all three.

D is read in catalyst units, `D · d / k₃` — what D would be if the loop stopped
— so it shares `metabolicWave` as its half-point and costs no constant.

### 3.2 Invariants

1. **`gut` and `extra` are counted; `B`, `C` and `D` are not.** The reactor
   turns its pool over many times per unit of matter, which is what a currency
   is, so its pools are in their own units and outside the books.
2. **What crosses that boundary should leave — and today it does not.** Food
   routed to the reactor by r1 would go back onto the ground at `upkeepExcrete`
   above zero. It **ships at 0**, so that food is destroyed and the reactor is
   a sink, which contradicts invariant 3. It used to cost the pond to turn on,
   because what came back was the body's *excretion mix* — signal, not ground.
   The excretion rows are gone, so it now returns as ground, as food. Measured
   over three seeds at 90 s it is no longer a cost and no longer a clear gain
   either: bodies 678 → 804, depth 8.8 → 11.7, and the seed spread is wider
   than the effect. **Still a decision rather than a bug**, but a cheaper one
   than it was.
3. **The dish is driven; the bodies are conservative.** Logistic regrowth mints
   and is meant to; `energy.test.ts` asserts the rest with the drive off. **The
   books are one column.** A signal is not matter — nothing eats it, nothing
   excretes it, and `deposit` mints it out of nothing on purpose — so
   `pondMatter` counts the ground channel alone.
4. **δ is not the decay the standing rules forbid.** Those protect lineages,
   learned weights and memory, where forgetting erases a difference something
   paid for. δ is a well-stirred vat's outflow, it is what closes the loop, and
   `D` has no other sink at all.

---

## 4. Roles: who paces and who relays

### 4.1 The instrument

Both questions below are answered in closed form, and the algebra is checked in
as `src/pond/spectrum.ts` — `npm run pond -- spectrum` prints all of it. This is
the second time a sweep was the wrong tool for the first question, which is why
it is code rather than a note.

**Does one body pace?** `A`'s row decouples, so the dynamics are the 3×3 over
(B, C, D): a three-stage negative feedback `B → C → D ⊣ B` carrying a positive
self-loop on `C`. Parameterise the fixed point by its own catalyst level and the
solve disappears:

    B* = (k₃+d)·C*(1+σC*) / (k₂(base+C*))    D* = k₃C*/d    j = k₄B*D* + dB*

With `p = k₄D* + d`, `q = (k₃+d) − ∂r₂/∂C`, `r = d` and `L = k₃ · ∂r₂/∂B · k₄B*`
the characteristic polynomial is `(x+p)(x+q)(x+r) + L`, and Routh–Hurwitz puts
the Hopf boundary at

        L  >  (p+q+r)(pq+pr+qr) − pqr                          (H)

In the limit `q → 0` that collapses to `k₃ > (2 + 2√2)·d`, about 4.83, which is
why the source document's own constants never oscillate: its python has 0.4
against 0.483.

**Does a chain carry a wave?** Not an impulse question — a relay is driven
continuously at its pacemaker's frequency, and the impulse response of this
reactor peaks at `t = 0` and decays at *every* stable fixed point, because C's
only positive feedback is its own diagonal entry and the loop leaving it,
`C → D ⊣ B → C`, is negative. Eliminating B and D from `(sI − J)` leaves the
catalyst channel's transfer function — the same polynomial with the C cofactor
on top:

        G(s) = (s+p)(s+r) / [ (s+p)(s+q)(s+r) + L ]

and one hop multiplies the catalyst by

        ε · |G(jω)|  ≥  1                                      (R)

with `ω = √(pq+pr+qr)`. A body just under its Hopf boundary is a **high-Q
resonator**: `|G(jω)|` is 12.4 at `j = 0.45` and 26.3 at `j = 0.50`.

### 4.2 The bands, at the shipping constants

| | influx `j`, per reactor unit |
|---|---|
| quiet | below 0.4663 |
| **relay** — (R) holds, (H) does not | 0.4663 … 0.5474 |
| **pacemaker** — (H) holds | 0.5474 … 3.0455 |
| saturated, still | above 3.0455 |

Period 3.35 s wall at the lower boundary, 1.45 s at the upper, so a fuller body
strokes faster.

**The source document's excitability line `k₂·B > k₃+d` is not the lower edge**
and is not used. It sits at `j = 0.4135`, where the per-hop gain is 0.578 — a
body satisfying it attenuates. It is also written in the wrong variable: `B*` is
pinned between 3.8 and 5.0 across a 145× change in influx, so a condition on `B`
barely varies over the whole operating range. (R) replaces it.

### 4.3 Seeding by kind

Not rules keyed on kind — seeds a lineage drifts from, the way `seedProduction`
already works. **One kind, one species**, which is `scratch.txt` §4.1 exactly:

| kind | broadcasts | role |
|---|---|---|
| Era | **A**, the food it ate | the net's feeder |
| Con | **C**, the catalyst | relay, excitatory |
| Dup | **D**, the inhibitor | relay, resets the front |

An Era used to send the primer as well, which was §6.1's reading. §2.3 is the
argument against it: broadcasting B ships the *fast* currency into one
neighbour, drives that body far up its own operating curve and makes it the
instability the net is then slaved to. Broadcasting the gut ships the slow one,
and the neighbour makes its own primer from it at its own `intake`. Computed
over the library and 38 grown nets, dropping the primer broadcast gives **half
again as many independently oscillating modes (278 → 418) and a quarter lower
leading growth rate (+0.0285 → +0.0213)**.

**A is the gut, so it does not couple the reactor — it moves the operating
point.** It reaches the reactor only through digestion, which makes the A block
linear and block-triangular: A cannot oscillate. `j = intake · yield · ρ · A`.

---

## 5. Coupling

### 5.1 The operator

A body broadcasts out of its **principal port only**, at `Tx[σ]` times whatever
it holds above `Gx[σ]`, rectified:

        flux(σ) = metabolicDiffuse · Tx[σ] · max(0, x_σ − Gx[σ])

bounded to a quarter of the source's stock and a quarter of the sink's room.
Rectified and not a step: a Heaviside rings under explicit Euler and gives a
gate nothing to move along. Mass action on the concentration, so the impulse is
in the chemistry — a body sends most at its catalyst's peak and nothing at its
trough, with no clock and no threshold needed.

`Gx` seeds at zero, so it gates nothing in a fresh pond. It is mutated and
clamped like every other gene, so it is a door evolution can open rather than
dead weight; `metabolicGate` is the dial that would seed it shut.

Two notes the code needs and the algebra above does not:

- **The quarter bound is about four times tighter than it needs to be.** Its
  argument is a body's own broadcast plus the far end of each of three wires,
  but a body has one principal, so at most one wire draws on it in a frame. It
  binds, rather than `metabolicDiffuse`, above a broadcast rate of 15.
- **There are two clocks.** The reactor advances by `h = metabolicRate · dt`
  and the coupling moves by `k = metabolicDiffuse · dt`, so the coupling the
  chemistry feels is `ε = metabolicDiffuse / metabolicRate` = 1/15. The two
  dials cannot be changed independently; see §6.4.

### 5.2 What the spectrum says

Principal-out makes the signalling graph **functional**: out-degree at most one,
because a body has one principal. Its adjacency is a permutation on its cycles
and nilpotent everywhere else, so the directed Laplacian's eigenvalues are
exactly the roots of unity around each cycle plus zeros. **Finding the cycles is
the whole calculation** — no eigensolver. Measured on the library:

| net | bodies | wired principal | cycles | lengths |
|---|---:|---:|---:|---|
| `deep-87` | 87 | 86 | 2 | 2, 2 |
| `mixed-308` | 308 | 308 | 9 | all 2 |

Every cycle has length two and is a p–p redex, so both spectra are real and
**neither net has a travelling eigenmode**. What it measured before this was a
driven transient down a nilpotent tree.

**That is a fact about these two nets, not a structural impossibility.** A
directed cycle needs each body's principal wired to the *next body's auxiliary*,
and p–aux wires are ordinary — 82 of 88 and 290 of 407. Such a cycle is not a
redex and the rewriter does not consume it; its eigenvalues are roots of unity
of order ≥ 3, which are complex. `deep-87` is a near-tree (cyclomatic 2,
diameter ≥ 44) and should not be read as evidence about shape at all.

**And the leading mode is localised**, in every net and regime tried: the top
five bodies hold over 99% of the eigenvector. The leading instability is a few
bodies going unstable on their own, because in-degree varies and with it the
operating point. The interesting modes are the sub-leading ones.

### 5.3 The doors this does not open, and what they cost

- **All ports, symmetric.** The Laplacian stays real; the leading mode goes
  real and its growth rate jumps five to seven fold. Turing, not peristalsis.
- **All ports, asymmetric** — forward at one rate, back at another. This
  *does* give complex spectrum, and generically: the condition is a net
  imbalance of principal orientations around a cycle, and **88% of cycles in
  random valid nets and 11 of 14 in grown ones carry one**. `mixed-308`, whose
  100 cycles are all balanced, is the outlier.
- **For any asymmetric scheme the spectrum is not the instrument.** The
  operator is enormously non-normal — the similarity that symmetrizes
  `mixed-308` has condition number `e^17` at β = 0.5 and `e^73` at 0.05 — and
  the symmetric part of `−L` has a positive eigenvalue where the spectrum has
  none, which guarantees transient growth: a disturbance amplified as it is
  carried, which is convective instability and is the travelling behaviour
  wanted. Use the **numerical abscissa and the pseudospectrum**. A float
  eigensolver on that matrix also reports complex eigenvalues that are rounding.
- **The master stability function does not apply**, which is why it is not
  built. The coupling is not diffusive — it does not vanish at synchrony, and
  `indeg − outdeg ≠ 0` for 97% of `deep-87` — and `Tx` is kind-dependent, so it
  does not factor as `ε·L⊗P`. Since `Gx` ships at 0 and the quarter clamp is
  slack, the coupling is *exactly linear* and the whole net is one constant
  matrix whose spectrum answers the question directly.

---

## 6. Parameters

### 6.1 What a body owns, against what the world sets

`docs/scratch.txt` §8 divides its parameters into "the physics of the
universe" and "the learnable genetics". This pond's split, against that one:

| scratch.txt §8 | there | here | ours |
|---|---|---|---|
| Threshold gates `G_i` | per-agent | `Gx`, 4 | **per-agent** |
| Transmission matrix `T_i` | per-agent | `Tx`, 4 | **per-agent** |
| Inflow `J_i` | per-agent | `intake` | **per-agent** |
| Transmission rates `α_c`, `β_d` | per-agent | `metabolicDiffuse` | **global** |
| Learning rate `μ_i` | per-agent | `learnRate` | **global** |
| Saturation `σ`, stoichiometry `S`, decays `d` | global | `metabolicSigma`, the reaction table, `metabolicDecay` | global |
| Reward window `τ` | global | `learnTrace`, `learnDiscount` | global |
| Traction `γ`, springs `K`, `L` | global | `gaitSwell`, `metabolicWork`, `springK` | global |

**Two divergences, and both are deliberate.**

*The transmission rate is global.* A body owns *which* species it speaks and
*how much it must hold* to speak — the two vectors — but not the rate, because
the rate is `ε` in §4.1's per-hop gain and a lineage that changed it locally
would change what a wavelength means across its own net. This is the source
document's own §8 argument for `σ`, applied one dial further than it applied
it.

*The learning rate is global.* `learnRate`, `learnTrace`, `learnDiscount` and
`learnCritic` are all global, and reach the device as uniforms in `genome.wgsl`'s
`G` struct. §8 offers a per-agent `μ_i` "if you decide to go the meta-learning
route, modulated by cell boredom", and boredom is refused — it inflates the rate
where variance is low, which erases what a quiet region learned, and the pond's
standing rule is that nothing decays but the eligibility trace. Without boredom
a per-agent rate has nothing to move it but drift, and a lineage that drifts its
own learning rate to zero is indistinguishable from one that has learned. So it
stays a law of the world.

### 6.2 The genome, in full

148 floats, 19 segments. Everything here is inherited, mutated and clamped;
`Wx`, `Wh`, `Wn` and `b` are also **learned within a life** — 64 floats at
offset 40, and the only span that moves without breeding.

| segment | floats | what it decides |
|---|---:|---|
| `emit`, `E` | 4 + 16 | what a body says, and how its state shapes that |
| `taste`, `T` | 4 + 16 | what it seeks, and how its state shapes that |
| `Wx`, `Wh`, `Wn`, `b` | 64 | the recurrent state `h`. **The learned block.** |
| `F`, `f0`, `P`, `p0`, `L`, `l0` | 30 | the locomotion heads |
| `ksg` | 1 | uptake affinity — the whole of what a lineage owns about eating |
| `G`, `g0` | 5 | the gait's anchor |
| `Tx` | 4 | which species this body broadcasts |
| `Gx` | 4 | how much it must hold before it broadcasts any |
| `Sw`, `Gw` | 3 + 3 | **the phase of each actuator** — what mixture of B, C and D the stroke and the grip read |

Plus seven heritable scalars outside `chem`: `extra`, `requestDecay`,
`energyCap`, `debtCap`, `rescueTo`, `assort`, `intake`.

**Every segment has a reader** — checked one at a time after this session's
cuts, the same way the parameter surface was.

**But the learned block reaches behaviour through one scalar.** `Wx`, `Wh`,
`Wn` and `b` are the only 64 floats that move within a life, and `h` only
becomes behaviour through the six head matrices `E`, `T`, `F`, `P`, `L` and
`G` — 60 floats which seed to **zero**, with exactly one exception. A seeded
body's whole state-dependent behaviour is:

        IN_DEMAND  --Wx[0][6]=1-->  h[0]  --T[energy][0]=1.8-->  taste(ground)

The neighbourhood's unmet need makes a body hungrier for ground, and that is
the single channel lifetime learning has to act through until a lineage mutates
another head entry off zero. A fresh pond learns into a bottleneck one scalar
wide. That is a door rather than a bug — the bases seed from the sliders so a
fresh body is exactly the constant it used to be — but it is worth knowing that
the door is this narrow, because it bounds what any learning rule here can do.

**Nothing in the genome is per-species any more except `Tx` and `Gx`.** `ks`
was four affinities and is one; `X` and `x0` were eight reaction rows and are
gone. Both were four-wide because a body ate four species, and it eats one.

### 6.3 Global constants

Laws of the world: a lineage that changed them locally would change what a
wavelength means across its own net.

`k₂` (`metabolicCat`), `k₃` (`metabolicReset`), `k₄` (`metabolicQuench`), `σ`
(`metabolicSigma`), `d` (`metabolicDecay`), `base` (`metabolicBase`),
`metabolicYield`, `metabolicDiffuse`, `metabolicWork`, `gaitSwell`,
`starveTime`, `uptakeVmax`, `uptakeKs`, `digestRate`, `gutSize`.

### 6.4 To delete

Gone already: `upkeep` (§7), `excreteRate` and `catCoSubstrate` (nothing
excretes, and there is no scent to pair ground with), and `bodyValue` moved to
`REWRITE_SHARE`. With them went `runExcretion`, `payOut`, `scentMints`, the
minted-versus-conserved fork and the three couplings and one protocol built on
it.

Still to go:

| out | why |
|---|---|
| `X`, `x0`, and three of `ks` | 48 floats a body that nothing reads. A genome-layout change, so it needs the segment migration and the GPU genome pass. |
| `metabolicRate` | the external clock. It multiplies every rate and exists only to set the period, so it is a reparameterisation — but not a free one, and §8 has the three things that go with it. |
| `metabolicWave` | derivable from the pool it half-saturates rather than dialled. |

---

## 7. Death, cost, and rationing

`upkeep` was one mechanic doing three jobs. **Done:** it ships at 0 and
`bodyValue` at `REWRITE_SHARE`. Where each job went:

**The standing cost is `δ`**, the source document's own diagonal decay matrix. A
body must keep eating to hold its pools against it, and a body that stops dies
on the window below. Rent duplicated this and charged it to bodies that were not
even running a clock.

**Death is the starvation window.** A body whose primer stays below `ε` for
longer than `starveTime` dies. Measured on a bare dish: rent and window kill at
30.1 s, window alone at 28.8 s, rent alone at 69.0 s, neither never. The window
already fired first, so nothing about death changed.

**The Era's income is its uptake**, and needed no replacement. The negative rent
`upkeepRateOf` paid a ground producer *was* an Era's income back when a producer
had no other; now that the reactor is fed by eating, an Era eats like everything
else. Measured over three seeds at 90 s, the Era share does not move when the
rent goes: 28.9% → 30.8%. `yEra` was not touched.

**Rationing belongs to the rewrite's price, and is currently off.** A rent
rations rewriting only sideways, by making everyone poor. The direct lever is
`bodyValue`: at `BODY_VALUE` (1.25) each commute-then-annihilate cycle *minted*
0.5, which balanced the rent and paid for rewriting; below `REWRITE_SHARE` it
costs `2 · (1 − bodyValue)`, which is a brake on the thing being braked. It
ships at exactly `REWRITE_SHARE` — conserved, no mint and no brake — because
nothing needs braking once the mint is gone.

**What the rent was also doing, unbilled: holding nets shallow.** It charges
every body in a net whether or not that body is doing anything, so depth costs
and a lineage that grows one bleeds. Measured over three seeds at 90 s, rent off
and `bodyValue` conserved:

| | depth | drift | lines | bodies |
|---|---:|---:|---:|---:|
| with rent | 0.49 | 0.172 | 364 | 429 |
| without | **1.11** | **0.257** | 309 | 450 |

Fewer lineages, twice as deep, and further apart. The depth ranges do not
overlap across the three seeds.

---

## 8. What is left

Each states what it changes and what would show it worked. No phase ships at a
neutral value; the discipline is `CLAUDE.md`'s — on, visible, behind a slider,
and the author looks.

**Done, and what the pond did.** One kind one species; the rent and the mint
(§7); a body eats the ground and nothing else; nothing excretes; `intake` as a
rate. Measured over three seeds at 90 s against the pond before any of it:

| | bodies | wires | lines | depth | drift |
|---|---:|---:|---:|---:|---:|
| before | 429 | 200 | 364 | 0.49 | 0.17 |
| after | 449–987 | 310–995 | 200–255 | **3.9–13.5** | **0.42–0.67** |

Nets an order of magnitude deeper, many more wires, fewer and more
differentiated lineages. The seed spread is wide — that is the thing to look at
before anything else is built on top of it, because a pond this much livelier
may also be a pond closer to running away.

**Done — the dead genome rows and the gut's empty channels.** `X` and `x0`
were 48 floats a body that nothing read, recomputed every frame by
`expressVector`; three of `ks`'s four entries were never looked at. Both halves
of `X` had lost their subject — nothing excretes and there is one thing to
digest — and the only two things left reading them, `rowCost` and `upkeep`, both
shipped at 0. **Genome 191 → 148 floats.** The gut, `fluxGut`, `gutTotal` and
`spillGut` went from four wide to one with them, and `excreteAll` and `rowCost`
went entirely. `ks` is `ksg` in the segment map, because a gene that went from
four affinities to one is a different gene and the migration refuses a length
change by name.

**Then — delete `metabolicRate`.** Restate the constants in per-second units.
Three things go with it and none is optional: `REACT_H` must be rescaled to
`0.04/15` or the sub-step count drops from 7 to 1; `metabolicDiffuse` must be
divided by 15 or the broadcast becomes 15× stronger relative to the chemistry,
past the measured flatline; and `fixedParams()` uses it to switch the whole
metabolism off for the suite, so the suite needs another answer.
*Acceptance:* `state-hash` unchanged, or changed only by float ordering.

**Resolved without being done: `SENSE_SCALE`.** It was wrong because conserved
excretion had stopped the scent mint, so the constant measured against a minted
field no longer matched the pond. Nothing excretes now, so a voice is always
minted, and `SENSE_SCALE`'s 4.3 is the value it was measured at. No remeasure.

## 9. What this does not do

- **It does not merge the tank and the primer.** Two timescales, one
  conversion; §2.3 has the measurement.
- **It does not add a field channel.** Making the catalyst a field species
  would unify signalling with metabolism and is the most interesting unbuilt
  idea in the audit, but the vocabulary is deliberately four and the reactor's
  throughput would saturate a cell in a frame without a conversion.
- **It does not open the travelling-mode door.** §5.3 states what it costs and
  what instrument it would need.
- **It adds no learning rule.** The source document's §7.1 trophic gate is
  refused because the pond has one learning rule — three-factor Hebbian with a
  TD critic on the state matrices — and a second would be a second. What §7.1
  reaches for is already reachable: put the gate on a head off `h` and the
  existing rule moves it.
- **It keeps no rule keyed on kind.** Everything in §4.3 is a seed.

---

## 10. How the numbers here were got

Nothing above was measured by running a pond, except where it says so. The
fixed points are Newton or a closed-form inversion, the bands are Routh–Hurwitz,
the per-hop gain is a transfer function, and the net spectra are the cycle
structure of a functional graph. `src/pond/spectrum.ts` is the code and
`npm run pond -- spectrum` is the command.

Two rules this cost enough to be worth stating:

- **Use the linear quantity the question is about.** An impulse response and a
  frequency response answer different questions, and a chain asks the second.
  Eigenvalues answer neither for a strongly non-normal operator.
- **A claim about shape has to be sampled over shapes.** The two nets in
  `nets/` are not a sample — one of them is a chain. Random port pairings and a
  short `npm run pond` both cost minutes.
