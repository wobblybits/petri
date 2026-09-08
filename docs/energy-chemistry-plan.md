# Plan: chemistry, and what makes energy usable

Agreed 2026-09-08 with the author, after reading `energy.ts`, `fields.ts`,
`chem-layout.ts` and the seeding in `agents.ts`. Same shape as
`plasticity-plan.md`: the rule, what changes, the order of work, and what to
watch.

Nothing here adds a fitness term. Everything here ships at zero and reduces
to today's behaviour, which is the discipline `farmRate`, `fertilise` and
`reactFeed` already follow.

---

## 0. The finding this is all downstream of

**A single Con with a full tank standing on ambient ground is a complete,
self-sufficient organism.** It eats, it pays rent, it can find a partner and
commute. There is nothing it can do inside a net that it needs the net for —
a net buys it transport and rewrite partners, never *access to energy it
could not otherwise reach*.

That is why the simulation does not build organisms. It is not a genome
problem, not a plasticity problem, not a field problem. Complexity appears
where a resource gradient must be crossed by machinery no single body
embodies, and there is currently no such gradient.

The second finding is already written down in the source. `Fields.maxSignal`
excludes `CH.energy`, and says why: *"the ground sits at capacity across the
whole dish, so including it would make this a constant a bit under `cellCap`
no matter what the pond was doing."* The environment is spatially uniform,
temporally stationary, and decoupled from what bodies do. That is a puddle,
diagnosed in our own comment.

---

## 1. The vocabulary: four species, no roles

`conP`, `dupP`, `energy`, `aux` become **species 0, 1, 2, 3**. Nothing else
changes: `e0` and `t0` are already only *seeds*, the matrices already drift,
and "Con's smell" is already just "the species a Con is seeded emitting".
The names are historical and the semantic layer is fiction we maintain by
hand.

Four is the whole vocabulary, and the smallness is the point. With four
species no lineage can have a private channel — everything anyone excretes
is something someone else can eat, be poisoned by, or misread. Interaction
is *forced*. At twenty channels lineages partition into non-interacting
niches and nothing ever conflicts.

What makes four non-degenerate is that species must be distinguishable **by
their physics, not their names**. The hook already exists: `diffuseRate` and
`decayRate` are per-channel `Float64Array(4)`, currently all ones. Give each
species its own constants and the four stop being interchangeable.

### Locality is a diffusion constant, not a guard rail

The one real risk of letting chemistry touch species 0 and 1 is that
polluting them degrades Con–Dup encounter. **We are not guarding them.** A
guarded species cannot be part of the chemistry, which would leave a
two-species network, one of which is food — not a network at all.

Instead, locality is made structural. Pollution is punished by selection
only when the harm lands on the polluter's own lineage, which is a length
ratio:

    diffusion length over a lifetime      sqrt(D * tau),  tau ~ 1.25/0.015 ~ 83 s
    versus lineage dispersal              l_lineage
    versus the dish                       R_pond

Want `l_D << l_lineage << R_pond`. If `l_D` approaches the pond radius the
field is well mixed, every lineage's encounter rate falls together, there is
no differential fitness, and the pond does not punish pollution — it quietly
slows down, with the throttled quantity being reproduction, which is the
mechanism the punishment would have to arrive through. That failure is
circular, not merely bad.

So: chemically active species get **short diffusion lengths**, by setting
`diffuseRate`. One number per species, no special cases.

---

## 2. Three words that are doing different jobs

- **Substrate** — a field species a reaction consumes.
- **Catalyst** — a species that scales a rate without being consumed.
  *Already implemented*: `grow(ch, r, cap, catCh, gamma)`, clamped at zero so
  an inhibitor can stall growth but never reverse it.
- **Enzyme** — a *body* catalysing a reaction at a rate set by expression.
  Does not exist, and is the only one of the three with a genome hook.

The definition to commit to:

    v_i(x) = k_i * e_i(h) * prod_j [S_j(x)] ^ a_ij

`e_i(h)` is the expression head off the recurrent state, `a_ij` the
stoichiometric order, `x` the body's cell. **An enzyme is a reaction whose
rate constant is a phenotype.** Nothing more.

Note what this does *not* need: catalysis requires no enzyme row. Catalysis
is a property of the species — a rate constant in the field's own table —
and a body participates by excreting that species. `fertilise` already works
exactly this way, and is right.

---

## 3. The body reaction table

Eight rows. For each species `c`:

| row | reaction | what it is today |
|---|---|---|
| `excrete_c` | tank -> S_c | `effEmit` for c in {0,1,3}; `farmRate` for c = 2 |
| `uptake_c`  | S_c -> tank | `harvestSlots`, for c = 2 only, unmetered |

Four mechanisms collapse into one operator over one table. `fertilise` is
not a row — it is the field's catalysis of species 1's reaction by species
0, downstream of `excrete_0`.

The firewall in `effEmit` — *"returns 0 for `CH.energy`, because the deposit
path multiplies by `params.deposit`, which is five"* — stops being a special
case and becomes a stoichiometry.

### One unit simplex over all eight

Expression is a new head `X`, normalised across all eight rows exactly as
`emitVector` normalises across four. Cost: `4 + 4 * STATE_DIMS = 20` floats,
`CHEM_LEN` 124 -> 144, +16%.

This is the decision with the most consequence in the document, so the
argument in full:

**Relative honesty** (the existing argument): a body cannot say two things at
once, so spending voice on farming trades against being heard. Extending the
simplex to uptake means it also trades against *eating*, which is the
strongest form of the trade-off.

**Absolute honesty** (new, and it comes from conservation): if excreting is
moving conserved matter out of the tank, a poor body physically cannot shout.
The simplex bounds what you can say relative to what else you say;
conservation bounds it in absolute terms. These compound and neither implies
the other.

### The specialist/generalist condition, stated precisely

Division of labour requires a trade-off, but a linear budget is not
sufficient. With a linear constraint and **concave** payoffs — and Monod
saturation *is* concave — the optimum is interior and everyone becomes a
generalist. Pooling across wires does not fix this: specialisation beats
splitting only when

    f(1) > 2 * f(1/2)

which concavity forbids. Superadditivity has to come from somewhere. Two
cheap sources:

1. a **fixed cost per expressed row** — pay `c` to run reaction `i` at all,
   so running two costs `2c` and running one costs `c`; or
2. **Hill kinetics**, `v = vmax * S^n / (K^n + S^n)` with `n > 1`, one
   exponent, convex at low expression.

Either one *plus* the wire graph — which is what makes a specialist viable at
all, since a specialist alone starves — gives division of labour. Neither
alone does. Ship both at the neutral value (`c = 0`, `n = 1`) and treat them
as the two dials that decide whether nets differentiate.

---

## 4. Uptake becomes a rate

Today, in `runHarvestPlan`:

    got = grid.take(key, cap - extra)

Instantaneous, to saturation, in **id order**. Two consequences: uptake has
no phenotype at all, so there is nothing for selection to grip; and ids are
monotone, so lower id means older, so **older bodies systematically eat first
in a contested cell** — a fitness gradient on age that nobody chose.

Replace with Monod:

    v   = vmax * S / (Ks + S)
    got = min(v * dt, cap - extra)

`vmax` and `Ks` read off `h` as heads, so uptake is state-dependent and a
body can upregulate when hungry. That is *regulation*, which is the one thing
minimal cells reliably do have.

The payoff is that `(vmax, Ks)` is a **non-dominating** trade-off: high/high
is a fast grazer needing rich ground, low/low is a scavenger living on
scraps, and neither wins everywhere. That is the precondition for
coexistence rather than takeover, and it is the most reliable diversity
generator in microbial ecology. Two numbers.

The id-order artifact dissolves: everyone in a block draws concurrently and
order stops mattering except at exhaustion.

**Migration is a dial, not a seed.** At `uptakeVmax = 0` the old
take-what-fits path runs unchanged; seeding the expression rows to zero is
not sufficient, because zero expression would mean zero uptake, which is not
today's behaviour.

---

## 5. What conservation should mean

Not a closed pond. The correct split:

> **The dish is driven** — feed in, kill out, patterned.
> **The bodies are conservative** — no reaction a body runs creates or
> destroys matter.

The second is the invariant that makes selection honest, and it is testable
as one assertion over the body reaction table. A driven medium is physically
right and is the precondition for the ground having patterns at all.

Today there are three sources and one sink, all disconnected:

| | |
|---|---|
| harvest | cell -> tank (conserved) |
| upkeep | tank -> **nothing** |
| Era upkeep (`ERA_UPKEEP_RATIO` -0.2) | **nothing** -> tank, ~0.003/s |
| commute + annihilate | mints `2*(EXTRA_CAP - REWRITE_SHARE)` = 0.5 |
| `grow` | **nothing** -> cell, up to K |

Three changes, each one line:

- **Upkeep excretes.** `tickUpkeep` already holds the grid handle for the
  overfill spill; send the ordinary spend to `grid.addAt(a.x, a.y, r*dt)` too.
- **`BODY_VALUE = REWRITE_SHARE`.** The comment already offers this. The
  reason to take it now is that with real rate-limited uptake, a net's income
  no longer *needs* to be proportional to its rewrite rate.
- **Drop `ERA_CAP_RATIO`.** `energyCap` is already heritable and recombined
  across a commute's children, so storage is already an evolvable axis
  available to every kind. The kind rule duplicates it with a wall instead of
  a gradient — and with it in place we can never learn whether big tanks
  actually belong on the boundary.

### Eras are metabolism, not storage

`ERA_UPKEEP_RATIO` is a mint keyed on a glyph, conditional on nothing. It
converts nothing; it just appears. Replace it with **a high uptake yield on
Eras** and a normal upkeep, so an Era's income is bounded by the ground under
it and an Era on grazed ground starves.

The reason is structural rather than biological. In interaction combinators
an Era is exactly a terminated port, so **#Eras is a net's boundary size**
while upkeep is charged per body:

    income  ~  d(net)          cost  ~  |net|

Surface-to-volume becomes a hard constraint on net size and the only way to
get bigger is to get *branchier*. That is a morphological selection pressure
we currently have no source of, and it costs one rule.

**Facultative to obligate is one dial.** A body's uptake yield is
`y_direct`; an Era's is `y_era`. Today is `y_direct = y_era`; obligate
trophic dependency is `y_direct = 0`. Before moving `y_direct` far, check the
larval window: a fresh spawn has `1.25 / 0.015 ~ 83 s` of tank, and if mean
time-to-encounter with a net is not well under that, obligate dependency
kills the soup rather than structuring it. `economy.exp.ts` can answer this
before anything is committed.

---

## 6. The ground: the reaction is already written

    r * E * (1 - E/K)

is **exactly mass-action autocatalysis** `E + W -> 2E` with the complement
`W = K - E` held implicit: `k*E*(K-E) = kK * E * (1 - E/K)`, so `r = kK`.
`grow` is already a reaction. A degenerate one, because `W` cannot move,
cannot be depleted independently, and has no diffusion constant of its own.

Make `W` explicit and it becomes activator–substrate depletion — the
Schnakenberg / Gray-Scott family. **Which is `Fields.react`, sitting at
`reactFeed = reactKill = 0`.** Our food channel's regrowth is a spatially
structureless special case of a reaction we already implemented and never
turned on.

Promote food onto it: food as the autocatalytic activator `V`, its
complement as substrate `U`, with `D_U ~ 2 * D_V` (our own comment: *"the
substrate at roughly twice the activator"*). What that buys:

- Standing crop that forms patches, fronts, splits and dies **on its own**,
  with no carrying-capacity constant. The equilibrium is the reaction's fixed
  point, not a fiat `K`. `energyRegrow` and `cellCap` stop being fiat.
- Grazing perturbs a pattern instead of punching a static hole. A grazed spot
  can nucleate a new front or be swallowed — regeneration with history rather
  than a refill timer.
- The feed and kill terms, `F*(1-u)` and `-(F+k)*v`, are the drive and the
  dissipation. A closed conservative cycle has no direction; this one does.
  `feed` is the sun and `kill` is the grave, and that is what licenses
  section 5's "driven dish, conservative bodies".

### Scale check, to run before committing

Gray-Scott spot spacing runs roughly 5–15 cells depending on parameters. At
`FIELD_CELL = 10` that is 50–150 world units, against `sensorDist = 24` and
`stepSpeed = 38`. A body resolves a gradient across a patch (its sensor
baseline is a fifth to a half of the wavelength) and crosses one in 1.5–4 s.
In range — navigable rather than invisible or overwhelming. If it lands
outside, `F` and `k` move it.

### The honest cost

Gray-Scott's interesting region is thin: `F` in [0.01, 0.09] against `k` in
[0.045, 0.07], with the good part a fraction of that. This trades a robust
boring mechanism for a fragile interesting one. Outside the window you get a
uniform wash (a puddle again) or extinction that never heals (a strip mine).

Mitigation is our own discipline: **keep `grow` behind a dial**, ship the
reaction at zero, and sweep `F, k` with the existing harness before switching
the default.

---

## 7. Order of work

**Phase 0. Layout and parameters.** *Done, 2026-09-08.* The expression head
`X` is derived in `chem-layout.ts` and `CHEM_LEN` is 134 -> 174. **Sized for
eight rows, not four**: this document's "4 + 4 * STATE_DIMS = 20 floats"
disagrees with its own §3 table and with its own prose ("normalised across all
eight rows"), and the table and the prose agree with each other. The `124 ->
144` was also written against a `CHEM_LEN` that had already moved. Nothing
reads `X`; phase 3 wires it. `genome-kernel.test.ts`'s layout tripwire now
asserts what it always meant — the shader stops at `X_OUT` and everything past
it is the part it does not know about yet.

`uptakeVmax`, `uptakeKs`, `rowCost`, `hillN`, `yDirect`, `yEra` are in
`params.ts` at the neutral value, plus `upkeepExcrete`, `bodyValue`,
`eraCapRatio` and `eraUpkeepRatio` for phase 2.

**Phase 1. Monod uptake.** *Done, 2026-09-08.* `runHarvestPlan`,
`harvestSlots` and `field.wgsl`'s `harvest` all take a rate; the two uptake
dials went into the two slots `FieldParams` was padding, so no binding was
added and the field pass stays inside its eight. The rate is read off the
block's mean density **before anybody eats**, which is what dissolves the
id-order artifact: everyone in a block draws concurrently and order stops
mattering except at exhaustion. At `uptakeVmax = 0` both paths are
bit-identical to what they were, which `energy.test.ts` pins.

Kept as a *global* rate rather than a head off `h`. §4 wants `(vmax, Ks)` read
off the state, and the head to read them from is the phase-0 `X` that phase 3
wires — so per-body uptake arrives with the rest of the reaction table rather
than as a fifth thing to keep in step.

**Stop here and look.** The instrument is `npm run pond -- sweep` and
`analyze`, and `measure.ts` is what it measures with — richness against
evenness, and `netFst`, the share of genetic variance lying between nets
rather than within them. See `src/pond/README.md`.

**Phase 2. Body-side accounting.** *Done, 2026-09-08, as dials rather than
edits.* All three are implemented and all three ship at today's value:
`upkeepExcrete` 0, `bodyValue` `EXTRA_CAP`, `eraCapRatio` 2 (plus
`eraUpkeepRatio` at `ERA_UPKEEP_RATIO`, which §5 needs and phase 2 did not
name). The document's opening discipline is that everything reduces to today's
behaviour; taking these as constants would have moved the pond, and moving it
is a decision to make while looking at a sweep rather than while reading a
diff.

The conservation assertion is in `energy.test.ts`, and it holds: with the
dials at their conservative settings and the dish's drive off, a 900-frame
pond that latched, commuted, erased and annihilated throughout balances to
better than a part in ten thousand. Two things had to be got right first.

- **Upkeep excretes what left the tank, not what was billed.** A body billed
  past empty runs a debt; excreting the billed amount would mint whatever the
  pond's starving bodies owed.
- **The dish wall was eating the substance.** `Fields.diffuse` treats a
  neighbour outside the disk as zero — an absorbing rim, which is right for a
  signal and wrong for `CH.energy`, the one channel `decayRate` is zeroed for
  precisely so that nothing can destroy it. With no bodies in the dish and
  decay off, the ground lost **10.587% of itself over 900 frames** at 128
  cells a side, 2.658% at 512, and **1.330% at the production 1024** — it goes
  as perimeter over area. What hid it is `grow`: regrowth kept topping the
  dish up, so the ground sat at an equilibrium between regrowth and a wall
  nobody knew was there rather than at `cellCap`. The conserved channel now
  reflects and the three signal channels still absorb; fixed on both paths,
  mirrored by `field-kernel.test.ts`, and confirmed exact on a real device.

  Worth reading against §0 and §6. "The environment is spatially uniform,
  temporally stationary, and decoupled from what bodies do" was true, and part
  of why `energyRegrow` and `cellCap` felt like fiat was that they were not
  actually setting the standing crop — the wall was.

`census()` reports `commutesPerLatch`, §8's tripwire.

**Phase 3. The body reaction table.** *Done, 2026-09-08.* Eight rows, one
simplex, on both field paths. `expressVector` divides one unit of chemical
effort across the table — relu then normalised, flat at the seed, the same
silence rule `emitVector` follows.

- **Excretion**, above `excreteRate` zero: all four species leave the tank
  conserved and the minted scent deposit stops. One or the other, never both;
  `scentMints` enforces it at the three places a deposit is laid. The rate is
  mass action on the tank, so a poor body physically cannot shout.
- **Uptake**, four species with per-body rates and affinities. `EnergyGrid`
  gained a species dimension; the harvest's flow buffer gained a stride.
- **`KS_BASE`**, a per-species affinity gene, is a plain heritable gene rather
  than the head §4 asks for. `X` supplies `vmax` — how much transporter a body
  expresses, which is regulation. Affinity is *which* transporter it has, so
  it has no business moving with mood, and a head would spend twenty floats
  modelling a decision that is not one. The non-dominating `(vmax, ks)` pair,
  which is what §4's argument rests on, survives.

**Phase 6 turned out not to be a phase.** The genome shader was never
involved: expression is a pure function of `chem` and `h`, and `unpackGenome`
brings `h` back to the host every frame, so the whole table is computed here
on both paths. That pass binds eight storage buffers of a guaranteed eight and
had nothing to spend; it did not need to. `field.wgsl`'s `harvest` did have to
learn four species, and it did so by widening a stride rather than adding a
binding, for the same reason.

**The coupling worth knowing about.** Metering uptake across all four species
while the scent path still *mints* is a matter fountain: a body deposits five
times its voice into three channels out of nothing and eats it back. Measured,
that ran at twice the rate cap and filled every tank. So the species rows
follow `excreteRate` and not `uptakeVmax`.

**Phase 4. Trophic dependency.** *Done as dials, 2026-09-08.* `yDirect` and
`yEra` scale the uptake *rate*, which makes obligate dependency conservative
without a second grid write per body per species: at `yDirect` 0 a body draws
nothing and the ground under it is untouched, so a starving body is not also a
wasteful one. Both ship at 1. **The larval-window measurement §5 asks for has
not been done**, and should be before `yDirect` moves far.

**Phase 5. The ground.** *Not started, and it has a structural question the
plan does not answer.*

§6 wants food promoted onto `react` as the autocatalytic activator with "its
complement as substrate". But §1 fixes the vocabulary at four species, and
after phase 3 all four are body reaction rows — there is no spare channel for
an explicit complement. The complement would have to *be* one of `conP`,
`dupP` or `aux`, which is either the best idea in the document or a
significant change to what those species are:

> everything anyone excretes is something someone else can eat, be poisoned by,
> or misread

reads very well if the substrate food grows from is a species bodies excrete —
fertilising becomes a real trophic act rather than `farmRate`'s special case,
and §2's `fertilise` catalysis is already a rehearsal for it. But it makes one
of the three signalling channels structurally special, which is exactly what
§9 says not to do, and choosing *which* is a decision with a large blast
radius. It wants the author, not an overnight guess.

The rest of phase 5 is ready: `Fields.react` exists and ships at zero, and
`npm run pond -- sweep --axis reactFeed=... --axis reactKill=...` already
parses — the `F, k` grid can be queued the moment the complement is chosen.

---

## 8. Consequences to watch

- **The measured signalling constants break in phase 3.** `params.deposit = 5`
  exists because signal is currently minted; conserved excretion removes it,
  signal amplitude drops ~5x, and `SENSE_SCALE = 4.3` — measured, and flat
  across a thirtyfold range of pond size — stops being right. Same for the
  steering dead zone. Remeasure the p90 rather than rescaling by hand; that
  is what made the constant trustworthy the first time.

- **`commutes / latches` is the tripwire for the whole no-guard decision.**
  Latching is proximity plus an arc test; scent never gates it. And net
  reduction is confluent, so a net is a *fixed budget of evolutionary events,
  spent down* — scent cannot touch that budget, only how nets acquire new
  structure. If `commutes/latches >> 1`, nets are doing real internal
  computation and chemistry polluting species 0 and 1 is cheap. If it is
  around 1 or below, every commute is roughly paid for by a fresh latch and
  encounter rate is load-bearing after all. Both counters already exist in
  `Sim.tally`; put the ratio in `census()`.

- **There is very little headroom.** The audit calls births rare and cites
  `bornMean` 0.36 over 45 s at ~293 bodies; if that is births per second, a
  commute producing four bodies puts the pond near **0.09 commutes/s** — one
  every eleven seconds, for 293 bodies with an 83 s tank. Most bodies never
  rewrite. Genetic turnover is already the scarcest thing in the simulation,
  and anything that halves the latch rate is halving something near the floor.

- **`spawnInterval: 0.5` is both the safety net and the ceiling.** Constant
  immigration of freshly seeded bodies means the pond cannot go permanently
  extinct, which is what makes "let die-off punish it" recoverable by
  construction. It is also migration load: unevolved immigrants dilute any
  evolved lineage. The same dial does both.

- **F5 dissolves.** The ground emit slot being genetic load at `farmRate = 0`
  stops being true the moment `excrete_2` is a row that does something.

- **Interaction with plasticity.** The learned span is `[W_IN, F_OUT)` and the
  new head sits outside it, so `PLASTIC_LEN` is unchanged and the expression
  head is inherited-and-mutated only, not learned. Behaviour still changes
  within a life because `X` reads `h`. Whether expression should learn is a
  later door, and it is the same door the output heads are behind.

---

## 9. What this deliberately does not do

- No new fitness term. Every teacher and every cost is a quantity the
  simulation already computes for its own reasons.
- No rule keyed on kind. Eras are seeded toward uptake; whether they stay
  there is selection's business. The codebase has already learned this lesson
  twice, explicitly.
- No guarded channels. Locality comes from `diffuseRate`, not from exempting
  anyone.
- No fully-emergent regeneration. A pond whose only producers are biotic
  cannot recover once they die, and evolution cannot act on a dead pond. The
  abiotic drive stays; biota modulate a rate they did not create.
