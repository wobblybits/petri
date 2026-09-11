# Asking the pond a question

How to run an experiment on this simulation without fooling yourself. The
instrument is `npm run pond` (`src/pond/README.md`). This document is the
protocol format, the measure catalogue, the coupling table, and the rules.
`concepts.md` says which mechanic and which dial a question is about, and
which gauge says whether the mechanic was engaged.

Nothing here quotes a number a command cannot reproduce. Numbers live in the
pond library (`ponds/pond.db`) and in the run rows that produced them.

```bash
npm run pond -- protocols                       # what is written down
npm run pond -- protocol <name> --smoke         # plumbing only, numbers discarded
npm run pond -- protocol <name>                 # the real thing, at its seed budget
npm run pond -- analyze --protocol <name>
npm run pond -- import ponds/*.db               # one library
npm run pond -- sweep --name s1 --random 120 --sample 'excreteRate=0..0.06,uptakeVmax=0..8'
npm run pond -- explore                         # what moved together, and where it was conditional
```

---

## 1. Rules

Each of these was paid for. The code enforces the ones it can.

1. **Before sweeping an axis, ask what every held constant means at each end
   of it.** The known cases are `src/pond/couplings.ts`; `sweep` and
   `protocol` warn when an axis crosses a coupled constant that does not move
   with it. Coupled regimes are arms with their own constants, not points on
   one grid.
2. **A protocol names its engagement gauges and the run checks them.** A
   precondition is `metric@summary` with a bound, evaluated per trial. A null
   on an unengaged mechanism is not a null. `full_mean` and `demand_mean` go
   either way at the defaults depending on the seed; check them, never assume
   them.
3. **Every metric has a summary mode, and the default is the honest one for
   its shape.** Transients are `peak`; cumulative counters are `window`;
   standing measures are `last`.
4. **A smoke run is plumbing.** `--smoke` tags its runs and prints a banner.
   The effect table marks a result `thin` under three seeds a level and
   `unresolved` with a seed estimate when eta-squared is under 0.3.
5. **Pick the outcome by its noise, then budget seeds for it.** The effect
   table prints the seed estimate from the spread it observed.
6. **Never quote a pond younger than a couple of simulated minutes.** Founders
   are depth zero and the opening latch storm drags every cumulative ratio
   for minutes. `mean` and `slope` skip the first 60 s (`--warmup`).
7. **An inert dial is a claim about a regime.** Before recording one, check
   the quantity it controls is not pinned by another dial.
8. **`concepts.md` before any mechanism.** Name the heading and the mechanic
   the heading already has.
9. **An early die-off is the design working.** Measure what the question
   asked.
10. **Measure the pipeline that ships**: wasm solver, GPU field and genome,
    `stepAsync`. The runner drains the device mirrors before reading them.
11. **The same seed run twice gives the same closing census.** That is a
    sweep's first sanity check and it catches every shared-state bug.
12. **Null means nothing to compare and is dropped from rankings.** Zero is a
    finding.

---

## 2. The shape of a protocol

A protocol is a value in `src/pond/protocol.ts`. The fields are the checklist.

| field | what it forces you to decide |
|---|---|
| `question` | one sentence, about the pond, not about a dial |
| `prediction` | directional and falsifiable: which measure moves which way along which axis |
| `nullReading` | written before the run: what an unresolved or flat result means |
| `arms` | regimes that need different constants. Each arm is its own sweep and an axis in the analysis |
| `axes` | the grid crossed within every arm |
| `base` | held constants common to all arms; every one has to mean the same thing at every point |
| `accepts` | couplings the protocol knowingly holds, with the reason. Unaccepted ones fail preflight |
| `preconditions` | engagement gauges: `metric`, `summary`, `min`/`max`, optionally where they apply, and why |
| `outcomes` | the measures the prediction is about, each with its summary and expected direction |
| `seeds`, `seconds`, `soupCount` | the budget. The CLI can raise seeds; lowering them is a smoke run |

Preflight checks parameter names, metric names, seed and duration minima and
couplings. The report evaluates preconditions per trial, prints the point and
effect tables with the arm as an axis, and reads each outcome as
*resolved / unresolved / thin* and *matches / does not match*. Adding one:
copy an entry, run `npm test -- protocol`, then `--smoke` before the real run.

---

## 3. Measures

Everything `analyze` can rank (`src/pond/analyze.ts`, `METRICS`), what it
says, and its default summary. Seed-to-seed spread is printed by the effect
table for the runs at hand rather than quoted here.

| key | says | default | caveat |
|---|---|---|---|
| `bodies` | population | last | rewrite arithmetic, not starvation; see `died` |
| `born_mean` | mean rewrite depth: is anything breeding | last | the noisiest measure; starts at 0 by construction |
| `born_max` | deepest lineage | last | a single-lineage number |
| `commutes`, `latches`, `died`, `born`, `spawned` | event rates | window | cumulative in the table; `@last` is the total |
| `commutes_per_latch` | internal computation against re-acquiring structure | window | the opening latch storm drags it below 1 for minutes |
| `commute_edge` | latching chosen against happened | last | negative while too crowded to steer |
| `can_pay` | share that could fund a rewrite | last | low is an economy problem; high with few commutes is a meeting problem |
| `free`, `ground`, `mean_extra` | where the energy is | last | `ground` on the GPU path is fresh only on readback frames; the runner arranges it |
| `lines`, `lines_effective`, `line_dominance` | richness, evenness, takeover | last | with immigration on, `lines` counts arrivals |
| `nets`, `nets_effective`, `net_dominance` | structure count and takeover | last | components of two or more |
| `net_fst` | variance between nets over total: the divergence number | last | null under two nets |
| `line_fst` | drift's own signature | last | |
| `matrix_drift`, `var_drifted` | how far the unseeded genome has walked | last | drift and selection alike; not comparable across `CHEM_LEN` |
| `signal_total` | is anyone saying anything | last | the pair to `born_mean` when reading `excreteRate` |
| `forage_ratio` | ground under bodies over dish mean; 1 is indifference | peak | a transient |
| `demand_mean` | mean `DEMAND` over bodies: was anyone hungry | peak | engagement gauge for every foraging, transport and learning question |
| `net_drift` | size-weighted mean speed of a net's centre, px/s | last | tracks the pond's mobility, not its swimming; read against a `transportRecoil` 0 control |
| `net_coherence` | that speed over what independent bodies would give | last | ceiling is sqrt(N), so a dial that grows nets raises it |
| `full_mean` | mean tank fraction | last | 1.0 throughout means the economy is not in play; the learning teacher is zero there |
| `signal_p90`, `sense_read_p90` | p90 signal at bodies, raw and times `senseScale` | last | `sense_read_p90` far from 1 means sense genes are out of `phi`'s range |
| `locus_demand_h0`, `locus_self_00`, `locus_food_h0` | population means of three named genes | last | a Baldwin claim is about these, not an aggregate |
| `latch_p50`, `latch_p90` | seconds from a body's arrival to its first latch | last | cumulative; read against `tank_life` |
| `loneliness` | share of arrivals that died having never latched | last | starvation and collision |
| `tank_life` | `EXTRA_CAP / upkeep`, the seconds a full tank buys | last | null when `upkeep` is 0 |

Summary modes: `last`, `peak`, `trough`, `mean` and `slope` (per simulated
minute, over `t >= warmup`), `window` (change per second over the last
interval). Any metric takes any mode: `net_fst@slope`, `bodies@trough`.

Not yet measurable: the learned-weight norm at sample time on the GPU path.
The host mirror is fresh only for rewrite parents and harvests.

---

## 4. Couplings

Pairs where the setting of one dial changes what another means, so the second
has to be chosen per level of the first. `src/pond/couplings.ts` is the
source; this is the reading of it.

| when you vary | re-choose | because |
|---|---|---|
| `excreteRate` | `senseScale` | the minted and the conserved signal differ by orders of magnitude |
| `excreteRate` | `deposit` | the mint is off whenever excretion is on |
| `uptakeVmax` | `digestRate`, `gutSize` | nothing fills a gut at 0 |
| `groundPatches` | `energyRegrow` | growth is zero at cap; patches manufacture ground |
| `groundPatches` | `learnDiscount`, `learnTrace` | the critic's horizon has to cover the trip to the reward |
| `ambientEnergy` | `uptakeKs` | a half-saturation constant is relative to the density it is measured against |
| `soupCount` | `declutter`, `flockAlign` | the spacing forces gate the opening scramble, whose severity is density |
| `energyRegrow` | `fertilise` | scales a rate; scaling zero is zero |
| `eraUpkeepRatio` | `excreteRate` | a producer's discount is worth what it earns |

Measures with the same problem: `lines` against `spawnInterval`,
`matrix_drift` against `CHEM_LEN`, every cumulative counter against run
length, `born_mean` against `soupCount`.

---

## 5. Seeds and time

The pond is chaotic and a net is a fixed budget of rewrite events spent down
stochastically, so the seeds are most of the variance.

- Seeds to resolve a relative difference `d` on a measure with seed-to-seed
  CV `c`, at two standard errors between two levels: roughly `8 (c / d)^2`
  per level. The effect table prints this from the spread it observed.
- Three seeds cannot resolve eta-squared below about 0.3. Say unresolved.
- Under a couple of simulated minutes is warm-up. Ten minutes is the working
  length for anything evolutionary; three for a spatial transient.
- A smoke run answers "does it throw" and nothing else.

---

## 6. Reading a result

- Direction before magnitude. Say which kind of claim you are making.
- Resolved / unresolved / thin, as the effect table prints them. An
  unresolved outcome gets the protocol's null reading, verbatim.
- Check the gauges first. If a precondition is unmet in an arm, the outcome
  in that arm is not read.
- Two measures agreeing is worth more than one moving.
- A null is not a zero.
- Name what you held: the arm's constants, the ground layout, the duration,
  the seed count.

---

## 7. Sampling instead of gridding

A grid over two axes informs that grid. Random draws over many axes compose:
every draw joins every regression ever run over the library, which is why the
runs belong in one database.

`--sample` takes bare keys, whose ranges come from `SLIDERS`, or `key=lo..hi`.
A slider range is an assumption. Draws are a Latin hypercube; ranges spanning
10x with a floor above zero are drawn log-uniform; an axis whose slider steps
by 1 or more snaps to a count.

`explore` runs four passes, none of them causal: joint PCA, PLS, k-means, and
a conditional pass that splits the library at each parameter's median and
reports where a correlation moved. Only the last sees an interaction. It
drops a metric most of the library predates, folds parameters that never
moved apart into one named column, and puts the setup (duration, founders,
dish cells) beside the parameters. Read the noise floor it prints: with `n`
runs a loading below about `1/sqrt(n)` is a coin.

---

## 8. Pipeline hazards

Things that produce a plausible wrong number rather than an error.

- `nativeSolver.init()`: without it the JS twin runs, slower and identical
  in arithmetic. The CLI does it; a new entry point must.
- `fields.data` on the GPU path is a stale mirror; the runner sets
  `wantFieldReadback` the frame before a sample.
- `plasticAll` on the GPU path is fresh only for rewrite parents;
  `syncLearningToHost` before a harvest.
- `fieldGpu` is a singleton. Same seed twice, same census, is the check.
- A patchy run's first sample reads zero ground on the GPU. Expected.
- `matrix_drift` is not comparable across a genome-width change; the run
  row records `chem_len`.
- A sweep applies `--set` uniformly, and inherits its held constants from the
  working tree. Pin anything a finding depends on with `--set`; a run row
  whose commit ends in `+dirty` was not made by the hash it names.
- Turning a mechanism off can take more dials than it has a name. Locomotion
  off is `grip=0 metabolicRate=0 transportQuantum=0`. Before using a dial as
  a control, check that zeroing it restores the old pond.
- The learning horizons `learnTrace` and `learnDiscount` are per frame. The
  browser's frame is not fixed; the runner's is.

---

## 9. The library

`ponds/pond.db` holds every run. Pooling is valid only where the simulation
did not change underneath: `explore` names a parameter constant within every
sweep and different between them as a label rather than a finding, and the
run row's commit says which pond made it. Which sweeps pool with which, at
each date, is a dated fact and lives in `docs/history/`.
