# Asking the pond a question

How to run an experiment on this simulation without fooling yourself. The
rules here are not general good practice; every one of them was paid for in a
recent session, and the section that introduces each says what it cost.

[`concepts.md`](concepts.md) is the companion: which idea is tied to which
mechanic, so that a question is asked of the right dial and a result is read
against the right mechanism.

The instrument is `npm run pond` (`src/pond/README.md`). The part of it this
document adds is the **protocol**: an experiment written down before it runs —
question, prediction, arms, held constants, engagement gauges, outcome
measure, seed budget — checked by the machine for the mistakes below, and read
back by the same machine in the same terms.

```bash
npm run pond -- protocols                       # what is written down
npm run pond -- protocol forage-engages --smoke # plumbing only, numbers discarded
npm run pond -- protocol forage-engages         # the real thing, at its seed budget
npm run pond -- analyze --protocol forage-engages
```

A protocol answers a question that is already sharp. When it is not — when
what you have is nineteen dials and no idea which three matter — sample them
instead and let the library say (§7):

```bash
npm run pond -- import ponds/*.db               # one library, not thirty-eight
npm run pond -- sweep --name survey1 --random 120 --sample 'excreteRate=0..0.06,uptakeVmax=0..8,…'
npm run pond -- explore                         # what moved together, and where it was conditional
```

---

## 1. What went wrong, and the rule each failure became

Two sessions, 2026-09-07 to 2026-09-09, running the first sweeps through the
pond library. The findings were mostly right in the end; the route to them
was not, and the same shapes will recur.

**A. A constant held fixed across a sweep did not mean the same thing at
each end of it.** `senseScale` was held at its minted value (4.3) across an
`excreteRate` axis where the conserved arm reads three orders lower — one arm
was blind, in a sweep about whether bodies navigate. `energyRegrow` was held
across a `groundPatches` axis where logistic growth is zero at cap: the
uniform dish produced nothing and the patchy ones manufactured ground as they
spread, so layouts seeded at equal mass ended 50% apart. Four sweeps, two
confounds, both invisible to the harness because the assumption is about the
model, not the grid.
→ **Rule: before sweeping an axis, ask what every held constant means at each
end of it.** The known cases are a table in code, `src/pond/couplings.ts`;
`sweep` and `protocol` warn when an axis is crossed with a coupled constant
that does not move with it. Run coupled regimes as **arms**, each with its own
constants, not as points on one grid. A protocol that knowingly holds a
coupled constant says so in `accepts`, with the reason.

**B. The mechanism under test was never engaged.** Structure sweeps ran with
`uptakeVmax = 6`, which kept every body at 1.20 of a 1.25 tank. Demand read
0.000, the seeded foraging pathway (gated on demand) never switched on, and
the ponds were indifferent to where the food was — not because they could not
forage but because nothing was ever hungry enough to look. Separately, every
learning claim was made about a pond with `learnRate` at its default of 0, and
the critic's horizon (0.33 s) could not see a five-second trip.
→ **Rule: a protocol names its engagement gauges and the run checks them.**
`demand_mean`, `full_mean`, `sense_read_p90`, `can_pay`, `died`, `forage_ratio`
are measures now; a precondition is `metric@summary` with a bound, evaluated
per trial, and outcomes are read only against it. A null on an unengaged
mechanism is not a null.

**C. A transient was read at its last sample.** Foraging on a patchy dish
peaks at 1.6–1.9× chance ground in the first minute, then bodies fill up and
stop; the closing sample reads parity. `analyze` read the last sample and said
"no foraging" from data that plainly foraged.
→ **Rule: every metric has a summary mode, and the default is the honest one
for its shape.** `forage_ratio` is `peak`; cumulative counters are `window`
(a rate over the last interval); most standing measures are `last`. Ask for
another with `metric@peak`, `@mean`, `@slope`, `@trough`, `@last`.

**D. Numbers were quoted before the seeds were in.** A 25-second smoke run
reported η² 0.91 for a dial whose 180-second, three-seed version read 0.21. A
2.4× foraging effect from one seed was 1.6× at five. A "the ratio is what
matters" hypothesis was stated before the crossed grid that refuted it.
→ **Rule: a smoke run is plumbing.** `--smoke` tags its runs separately and
prints a banner; its numbers are not findings. The effect table marks a result
`thin` when there are fewer than three seeds a level whatever η² says, and
`unresolved` with an estimated seed budget when η² is under 0.3.

**E. The outcome measure was the noisiest one available.** `bornMean`
(lineage depth) has a seed-to-seed CV of 0.73 — a long-tailed mean that a
handful of deep lineages carry. Resolving a 30% effect on it wants ~25 seeds;
`netFst` (CV 0.26) wants about five; `forage_ratio` and `ground` fewer.
→ **Rule: pick the outcome by its noise, then budget seeds for it.** §3 is
the catalogue. Excretion moved depth ninefold and resolved at five seeds;
ground structure moves it by tens of per cent and did not.

**F. Young-pond numbers were read as steady-state numbers.** A preset drops
its whole population in as founders, so `bornMean` starts at exactly zero,
`lines / bodies` at one, `netFst` at a construction artefact, and the first
half-minute is a latch storm that drags every cumulative ratio down for
minutes.
→ **Rule: never quote a pond younger than a couple of simulated minutes;
window cumulative counters; treat the warm-up as the seeding.** `mean` and
`slope` summaries skip the first 60 s by default (`--warmup`).

**G. A dial read inert because another dial was saturating what it
controlled.** `uptakeVmax` scored η² 0.02 while excretion drained tanks faster
than any uptake could refill them; three sweeps later it was most of the cost.
→ **Rule: an inert dial is a claim about a regime.** Before recording one,
check the quantity it controls is not pinned (§B's gauges again).

**H. A mechanism was proposed for a concept that already had one.** A
longer-memory hunger signal, when `Wh` is the memory. Immigrants spawned from
the population, when they are meant to be larval. A swim cost, when nets move
by transport. Weight decay, when persistence is the design.
→ **Rule: `concepts.md` before any mechanism.** Name the heading the proposal
serves and the mechanic that heading already has.

**I. The population's floor was hunted as if it were the result.** A falling
population was read as failure and chased to its asymptote at three sizes. It
was the opening cull, and the question was frame time.
→ **Rule: an early die-off is the design working.** Measure what the question
asked.

**J. The wrong pipeline was measured.** Every early profile was of the sync
CPU twin; a headless pond ran the JS solver because `nativeSolver.init()` was
never called; the GPU host mirrors of the learning table and the field read as
zeros that looked like answers.
→ **Rule: the runner uses the shipping path and drains the mirrors before
reading them** (`wantFieldReadback` the frame before a sample,
`syncLearningToHost` before a harvest). A new measure that reads a store array
must say which path it is fresh on.

**K. Trials were not independent.** `fieldGpu` is a singleton whose buffers
outlived the `Sim` that used them; every sweep trial after the first stood in
the previous pond's dish until the device tests caught it.
→ **Rule: a sweep's first sanity check is that the same seed run twice gives
the same closing census.** Cheap, and it catches the whole class.

**L. Absence was reported as zero.** `netFst` for a pond with one net; a
ratio over no latches.
→ **Rule: null means nothing to compare and is dropped from rankings.** Zero
is a finding. Say "unresolved at n" rather than reporting a rank order.

---

## 2. The shape of a protocol

A protocol is a value in `src/pond/protocol.ts`. The fields are the checklist.

| field | what it forces you to decide |
|---|---|
| `question` | one sentence, about the pond, not about a dial |
| `prediction` | directional and falsifiable: which measure moves which way along which axis |
| `nullReading` | written **before** the run: what an unresolved or flat result means, so a story is not picked afterwards |
| `arms` | regimes that need different constants (§1.A). Each arm is its own sweep, `<protocol>/<arm>`, and the arm is an axis in the analysis |
| `axes` | the grid crossed within every arm |
| `base` | held constants common to all arms — every one of them has to mean the same thing at every point |
| `accepts` | couplings the protocol knowingly holds, with the reason. Unaccepted ones fail preflight |
| `preconditions` | engagement gauges (§1.B): `metric`, `summary`, `min`/`max`, optionally the arms or grid points they apply to, and why |
| `outcomes` | the measures the prediction is about, each with its summary and expected direction |
| `seeds`, `seconds`, `soupCount` | the budget, from §3. The CLI can raise seeds; lowering them is a smoke run |

Preflight checks parameter names, metric names, seed and duration minima,
and couplings; the report evaluates preconditions per trial, prints the point
table and effect table with the arm as an axis, and then reads each outcome
against its prediction as *resolved / unresolved / thin* and *matches / does
not match*. The null reading is printed whenever an outcome is unresolved.

Adding one: copy an entry, run `npm test -- protocol` (the registry is
checked against its own preflight), then `--smoke` before the real run.

---

## 3. Measures

Everything `analyze` can rank, what it says, its default summary, and what
it costs in seeds. CVs are seed-to-seed at the close of ten-minute,
500-body ponds, five seeds, measured 2026-09-08; treat them as the order of
magnitude, not the value.

| key | says | default | CV | caveat |
|---|---|---|---|---|
| `bodies` | population | last | 0.31 | rewrite arithmetic, not starvation (see `died`) |
| `born_mean` (depth) | mean rewrite depth — is anything breeding | last | **0.73** | the noisiest measure here; ~25 seeds for a 30% effect. Starts at 0 by construction |
| `born_max` | deepest lineage | last | — | a single-lineage number |
| `commutes`, `latches`, `died`, `born`, `spawned` | event rates | **window** (per second over the last interval) | 0.5 (commutes) | cumulative in the table; `@last` gives the total |
| `commutes_per_latch` | internal computation vs re-acquiring structure | window | — | opening latch storm drags the cumulative below 1 for minutes |
| `commute_edge` | latching chosen vs happened | last | — | negative while too crowded to steer |
| `can_pay` | share that could fund a rewrite | last | — | economy problem (low) vs meeting problem (high with few commutes) |
| `free`, `ground`, `mean_extra` | where the energy is | last | 0.15 (ground) | `ground` on the GPU path is fresh only on readback frames — the runner arranges it |
| `lines`, `lines_effective`, `line_dominance` | richness, evenness, takeover | last | 0.45 | with immigration on, `lines` counts arrivals |
| `nets`, `nets_effective`, `net_dominance` | structure count and takeover | last | 0.38 | components of two or more |
| **`net_fst`** | **the divergence number**: variance between nets over total | last | **0.26** | null with under two nets; ~5 seeds for a 30% effect |
| `line_fst` | drift's own signature | last | — | |
| `matrix_drift`, `var_drifted` | how far the unseeded genome has walked | last | — | drift and selection alike; **not comparable across `CHEM_LEN`** |
| `signal_total` | is anyone saying anything | last | — | the pair to `born_mean` when reading `excreteRate` |
| `forage_ratio` | ground under bodies over dish mean; 1 is indifference | **peak** | 0.18 | a transient: peaks in the first minute, decays to parity |
| `demand_mean` | mean `DEMAND` over bodies — was anyone hungry | peak | — | engagement gauge for every foraging or transport question |
| `net_drift` | how fast nets travel: size-weighted mean speed of a net's centre, px/s | last | — | **tracks the pond's mobility, not its swimming**: it followed the population step for step across `grip`. Read against a `transportRecoil` 0 control |
| `net_coherence` | that speed over what independent bodies would give; 1 is indifference | last | — | removes the pond's speed but not its structure: the ceiling is `sqrt(N)`, so a dial that grows nets raises it. Read beside `net_dominance` |
| `full_mean` | mean tank fraction | last | — | 1.0 throughout means the economy is not in play |
| `signal_p90`, `sense_read_p90` | p90 signal at bodies, raw and × `senseScale` | last | — | `sense_read_p90` far from ~1 means sense genes are out of `phi`'s range (§1.A) |
| `locus_demand_h0`, `locus_self_00`, `locus_food_h0` | population means of three named genes: `Wx[0][DEMAND]`, `Wh[0][0]`, `T[food][0]` — the seeded foraging pathway and hunger memory | last | — | the Baldwin question is a claim about these, not about an aggregate |
| **`latch_p50`**, `latch_p90` | **the larval window**: seconds from a body's arrival to its first latch | last | — | cumulative over the run, so `@last` is the whole population. Read against `tank_life`, never alone |
| `loneliness` | share of arrivals that died having never latched | last | — | a rewrite consumes wired bodies, so this is starvation and collision |
| `tank_life` | `EXTRA_CAP / upkeep`, the seconds a full tank buys | last | — | the scale `latch_p50` is compared to; null when `upkeep` is 0 |

Summary modes: `last`, `peak`, `trough` (over `t > 0`), `mean` and `slope`
(per simulated minute, over `t ≥ warmup`), `window` (change per second over
the last sample interval). Any metric takes any mode: `net_fst@slope`,
`bodies@trough`.

**The larval window, and what it decides.** `yDirect = 0` is obligate trophic
dependency: a body draws nothing from the ground and lives on what a net
sends it. Whether that structures the soup or kills it is one comparison —
`latch_p50` against `tank_life`. Well under, and a body reliably reaches a
net before its tank runs out, so dependency is a pressure. Near or over, and
dependency is a cull. `loneliness` is the same question asked of the tail.
`docs/energy-chemistry-plan.md` §5 asks for this before `yDirect` moves far.

Still not measured and wanted: the learned-weight norm on the GPU path at
sample time (the host mirror is stale; only harvests drain it).

---

## 4. Couplings

Pairs where the setting of one dial changes what another *means*, so the
second has to be chosen per level of the first. The code is the source of
truth (`src/pond/couplings.ts`); this is the reading of it.

| when you vary | re-choose | because |
|---|---|---|
| `excreteRate` | `senseScale` | minted signal reads p90 ≈ 4.3, conserved ≈ 0.002; one scale leaves one arm blind |
| `excreteRate` | `uptakeVmax` | the species uptake rows follow `excreteRate`; at 0 `uptakeVmax` meters the ground only, so it is a different mechanism in each arm |
| `excreteRate` | `deposit` | inert above zero: the mint is off whenever excretion is on |
| `groundPatches` | `energyRegrow` | growth is zero at cap; a uniform dish seeded at cap produces nothing, patches manufacture ground. `energyRegrow = 0` is the clean layout control |
| `groundPatches` | `learnDiscount`, `learnTrace` | the critic's horizon has to cover the trip to the reward, and patch spacing sets the trip (5–12 s against a default 0.33 s horizon) |
| `ambientEnergy` | `uptakeKs` | a half-saturation constant only means something relative to the density it is measured against |
| `soupCount` | `declutter`, `flockAlign` | the spacing forces gate the opening scramble, whose severity is density; the 18× was measured at 250 bodies |
| `energyRegrow` | `fertilise` | scales a rate; scaling zero is zero |
| `eraUpkeepRatio` | `farmRate` | a seeded Era earns ~0.003/s from the mint; a farm rate above that starves it |

Measures with the same problem: `lines` against `spawnInterval` (immigration
manufactures lines), `matrix_drift` against the commit (`CHEM_LEN` is its
denominator), every cumulative counter against run length, `born_mean`
against `soupCount` (founders are depth zero).

---

## 5. Seeds and time

The pond is chaotic and a net is a fixed budget of rewrite events spent down
stochastically, so **the seeds are not a nuisance term; they are most of the
variance.** Rules of thumb, from the CV table:

- Seeds to resolve a relative difference `d` on a measure with CV `c`, at two
  standard errors between two levels: roughly `8 · (c / d)²` per level. Depth
  at 30%: about 48. `netFst` at 30%: 6. `forage_ratio` at 30%: 3. The effect table
  prints this estimate for every unresolved row, from the spread it observed.
- **Three seeds cannot resolve η² below about 0.3.** Say unresolved.
- **Under a couple of simulated minutes is warm-up.** The founders are all
  depth zero, the latch storm is running, and differentiation is founder
  mixing. Ten minutes is the working length for anything evolutionary;
  three for a spatial transient like foraging.
- A smoke run answers "does it throw" and nothing else.

---

## 6. Reading a result

- **Direction before magnitude.** "Depth fell with excretion" is supportable
  at five seeds; "0.015 rather than 0.01" is not. Say which kind of claim you
  are making.
- **Resolved / unresolved / thin**, as the effect table prints them. An
  unresolved outcome gets the protocol's null reading, verbatim, not a new
  story.
- **Check the gauges first.** If a precondition is unmet in an arm, the
  outcome in that arm is not read; say the mechanism was not engaged.
- **Two measures agreeing is worth more than one moving.** `netFst` rising
  while `linesEffective` falls is the drift-versus-selection signature; it was
  the reason to trust the excretion result.
- **A null is not a zero.** `netFst` is null with one net; a ratio over no
  latches is null. They are dropped from the ranking, not scored.
- **Name what you held.** Every claim carries its regime: the arm's
  constants, the ground layout, the duration, the seed count.

---

## 7. Sampling instead of gridding

A grid over two axes at five seeds spends thirty runs learning about two
dimensions, and what it learns informs that grid and nothing else. Thirty
draws over nineteen axes learn about nineteen — Bergstra and Bengio's
argument, whose precondition is that only a few dimensions matter, which is
what this pond keeps demonstrating: `excreteRate` explained 68 to 93 per cent
of the variance in a sweep where `uptakeVmax` explained two to eight and was
unresolved.

The decisive practical difference is that draws **compose**. A grid point
informs its own grid; a random draw joins every regression ever run over the
library. Which is the whole reason the runs belong in one database — and
`pond import a.db b.db …` is how the ones that did not get there.

```
pond sweep --name survey1 --random 120 --sample 'excreteRate=0..0.06,uptakeVmax=0..8,uptakeKs,…'
pond explore                      # the whole library
pond explore --name survey1       # one sweep of it
```

`--sample` takes bare keys, whose ranges come from `SLIDERS`, or `key=lo..hi`
to override — and overriding is often right, because **a slider range is an
assumption**. `learnDiscount` stops at 0.995, a 3.3-second horizon, when
travel to a patch takes 5 to 12; a search bounded by the slider inherits the
belief that the answer is inside it. Draws are a Latin hypercube (each axis
cut into `count` bins, one draw per bin, bins shuffled independently), ranges
spanning 10× with a floor above zero are drawn log-uniform, and an axis whose
slider steps by 1 or more is a count and snaps — the sim floors
`groundPatches`, so an unsnapped 5.2 would store a value that never ran.

### What `explore` does, and what it cannot

Four passes, none of them causal. This is a map of a library that was not
designed as an experiment; its job is to say which experiment to run.

| pass | question | blind to |
|---|---|---|
| joint PCA | what moves together at all, parameters and outcomes in one column space | direction; a component is a correlation, not a cause |
| PLS | which directions link parameters **to** outcomes | the same |
| k-means | what kinds of pond this makes, read back to what made them | anything the chosen outcomes do not measure |
| conditional | where one dial changes what another means | main effects — that is the point |

Only the last sees an interaction, and interactions are where this simulation
keeps hiding things (§4). It splits the library at each parameter's median
and correlates driver with outcome in each half; the `swing` is how much the
correlation moved.

Three behaviours worth knowing, each of which the first real library forced:

- **A metric most of the library predates is dropped, loudly, rather than
  dropping the library.** `fullMean` was added to `measureDiversity` after
  those runs were made, and requiring it vetoed all thirty.
- **Parameters that never moved apart fold into one named column**, printed
  as `excreteRate~-senseScale~catCoSubstrate` (a `-` marks one that ran
  backwards). A sweep that sets three dials together makes them one dial;
  before this, every finding printed three times with the loading split three
  ways.
- **The setup goes in beside the parameters** — duration, founders, dish
  cells. Runs in a library differ in all three, and a regression blind to
  that credits whichever parameter moved alongside them.

Read the noise floor it prints. With *n* runs, a loading below about
1/√n is a coin; at 30 runs that is 0.18, which is most of what a small
library will show you.

---

## 8. Pipeline hazards

Things that produce a plausible wrong number rather than an error.

- `nativeSolver.init()` — without it the JS twin runs, three times slower
  and identical in arithmetic. The CLI does it; a new entry point must.
- `fields.data` on the GPU path is a stale mirror; the runner sets
  `wantFieldReadback` the frame before a sample. A new measure reading the
  field at another time reads the wrong pond.
- `plasticAll` on the GPU path is fresh only for rewrite parents;
  `syncLearningToHost` before a harvest. Learned-weight measures at sample
  time are not yet possible on the device.
- `fieldGpu` is a singleton; trial independence was broken once. Same seed
  twice, same census, is the check.
- A patchy run's first sample reads zero ground on the GPU (adds are queued
  until the first field step). Expected, not a bug.
- `matrix_drift` is not comparable across a genome-width change; the run row
  records `chem_len`.
- A sweep applies `--set` uniformly. That is the mechanism behind §1.A.
- **A sweep inherits its held constants from the working tree.** Whatever
  `defaultParams()` says at the moment the process starts is what every trial
  holds, so an edit someone is in the middle of becomes a constant across 120
  runs. Pin anything a finding depends on with `--set` rather than trusting
  the file, and read the run row's commit: `9a3f21c+dirty` means the hash does
  not describe what ran.
- **Turning a mechanism off can take more dials than it has a name.**
  `grip = 0` is not locomotion off: `transportQuantum` at 0.5 changes the pond
  on its own with grip at zero — measured, 183 bodies against 481 at 60
  simulated seconds. Locomotion off is `grip=0 wireTug=0 transportQuantum=0`,
  and with those and the pre-learning defaults HEAD reproduces `7a4086e`
  bit-for-bit. Before using a dial as a control, check that zeroing it
  actually restores the old pond.

---

## 9. What is in the library, and what pools with what

`ponds/pond.db` holds every run. Pooling them is only valid where the
simulation did not change underneath, and it changed twice in one day.

| sweep | n | design | pond |
|---|---|---|---|
| `survey1` | 120 | 19 chemistry axes, sampled | learning off, no locomotion |
| `survey2` | 120 | the same 19 axes | the same, plus the larval instrumentation |
| `phys1` | 120 | 18 physics axes, sampled | the same, learning pinned off |
| `chem3` | 120 | the 19 chemistry axes again | **learning on, locomotion on** |
| `combo-old` / `combo-new` | 32 each | `wireShrink` × `spawnInterval`, 8 seeds | the two ponds, side by side |

- **`survey1 + survey2` pool.** The commit between them added the larval
  window, which is instrumentation: same seed, same pond, every column of
  every sample bit-identical. Pass `--drop maxAgents` — `survey2` set a fuse
  at 8000 that the peak population of 3788 never reached, so it is a label for
  which sweep a run came from and not a parameter.
- **`chem3` does not pool with them.** `learnCritic` 0.02 → 0.2, `learnTrace`
  and `learnDiscount` 0.95 → 0.99, and locomotion shipped on. Read it as its
  own 120-run sample of the newer pond.
- **`explore` says so itself.** A parameter constant within every sweep and
  different between them is named as a label rather than reported as a
  finding; that is what catches this class.
