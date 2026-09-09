# The pond library

A pond you can leave running, and a file that remembers what it grew.

```bash
npm run pond -- run --seconds 600 --bodies 400 --seed 7
npm run pond -- runs
npm run pond -- nets --run 1 --order depth
npm run pond -- show 41
npm run pond -- run --from-run 1 --top 8 --seconds 600     # keep going
npm run pond -- run --seconds 600 --gpu off               # CPU field

npm run pond -- sweep --name uptake --axis uptakeVmax=0,0.75,2 --seeds 1,2,3
npm run pond -- analyze --name uptake
npm run pond -- --help
```

The database lands at `ponds/pond.db` unless `--db` says otherwise, and
`ponds/` is gitignored — these files are megabytes of evolved genome and
belong to the machine that grew them.

## Why this exists

The test suite asks "did I break it" and the experiment harness asks "what
does this dial do", and both throw the pond away when they are done. Nothing
kept a net. A lineage twenty rewrites deep with a genome that has walked a
long way from its seed took ten minutes of simulated time to make and
survived exactly as long as the process did.

So: a run writes its nets to disk with the parameters and the commit that
produced them, and a later run can plant them and carry on. Evolution across
sessions instead of within one.

## What runs

`Sim.step` with the wasm solver on. The GPU path is the same simulation bar
one frame of genome latency, so a lineage grown here is one the page can open.
The measurement at each sample point is `sampleSim` from
`src/experiments/harness.ts`, not a second one written here: a sweep's JSON
and a run's timeline should be the same numbers or one of them is lying.

**`nativeSolver.init()` is not optional.** `Sim.nativeForces` defaults to
true, but `nativeSolver.ready` does not — the wasm is instantiated by an
explicit call, which the three pages each make at startup. Without it every
guarded call falls through to the JavaScript twin, silently and correctly,
and the pond runs the reference implementation of a solver that exists in C.
Measured back to back at 1,800 bodies:

| phase       | JS twin | wasm    |
| ----------- | ------- | ------- |
| whole frame | 57.0 ms | 16.9 ms |
| `solve`     | 37.6 ms | 4.8 ms  |

## The GPU

`field.wgsl` and `genome.wgsl` ship in the browser and are checked against
their CPU twins by `field-kernel.test.ts` and `genome-kernel.test.ts`. The
only thing stopping a headless run using them was that Node has no
`navigator.gpu`. [`webgpu-node.ts`](webgpu-node.ts) supplies one out of Dawn —
the implementation Chrome's WebGPU is built on — through the optional `webgpu`
package, and nothing in `src/gpu/` knows it is not a browser.

`--gpu auto` (the default) uses a device when Dawn can open one, `on` fails
the run rather than quietly costing twice the wall clock, `off` never asks.
Measured back to back, 120 simulated seconds from a 600-body soup, 512-cell
field, same seed:

| | ms/frame | wall | bodies at close |
| --- | --- | --- | --- |
| `--gpu off` | 9.39 | 67.6 s | 805 |
| `--gpu auto` | **4.33** | **31.2 s** | 928 |

The two ponds diverge — different implementations of the same arithmetic, in
a system that amplifies a last-bit difference — but they land in the same
regime: depth 12.84 against 13.44, matrix drift 0.274 against 0.276.

Dawn is about 21 MB per platform and is an **optional** dependency, so a
machine that cannot install it still runs ponds on the CPU field.

### Two things the device owns that the host has to ask for

Both are cases where the host's copy of something is stale on the GPU path,
plausible-looking, and wrong.

**The learning.** `plastic`, `trace` and `critic` are resident on the device;
`syncLearn` fetches only the two parents of each pending rewrite, because a
handful a frame is all the simulation needs. Everything else in
`AgentStore.plasticAll` is whatever it was when the pond moved to the device,
which is zero. A harvest therefore calls `Sim.syncLearningToHost()` first,
which drains the whole table through `GenomeGpu.drainLearn`. Without it every
stored net would record a genome and no learned matrices — the database
quietly wrong rather than loudly empty. `docs/plasticity-plan.md` phase 5
calls for this reader; the headless harvest is its first user.

**The ground.** `fields.data` is a stale copy on the GPU path, and
`sampleSim` reads it for its `ground` column through `energy.storedTotal()`.
So the runner sets `wantFieldReadback` on the frame *before* a sample — the
readback happens at the end of the frame that asks. It is a 16 MB copy paid
once per sample, one frame in six hundred at the default schedule.

`tools/node-hooks.mjs` is what lets `node --experimental-strip-types` load
`src/` directly, by doing for `?raw` and `?url` what Vite does for the page.
No build step stands between an edit and a pond.

## Tables

| table    | one row per                                            |
| -------- | ------------------------------------------------------ |
| `run`    | invocation: when, what seed, the whole `Params`, the commit |
| `net`    | connected component stored out of a run, with its blob  |
| `plant`  | stored net planted into a run, and the founder line it was given |
| `sample` | timeline point, mirroring the experiment harness's `Sample` |

Everything a query might sort or filter on is a column; the genomes are one
BLOB per net beside them. That split is what lets a listing of a hundred nets
avoid reading a hundred genomes.

```sql
-- the deepest nets anywhere in the library
SELECT id, run_id, bodies, born_max, matrix_drift FROM net
ORDER BY born_max DESC LIMIT 20;

-- how a lineage moved across runs
SELECT n.id, n.run_id, n.bodies, n.born_max, n.parent_net
FROM net n WHERE n.parent_net IS NOT NULL ORDER BY n.id;
```

## Ancestry across runs

Agent ids mean nothing outside the pond that issued them, so they are not
stored and not carried. What is carried is a founder line: `plantNet` writes
a **negative** `lineage` onto every body of a planted net — ids are always
positive, so a negative line is unambiguously "arrived from the database" —
and the `plant` table says which stored net each negative line names.

A harvested net's `dominant` column is its most common founder line. If it is
negative, `parent_net` is the row it descends from; if positive, the net grew
here and has no parent. A net that has drifted onto a different founder is
therefore not falsely attributed to one, which is the property that matters:
this is a measurement, not a label.

`census().lines` keeps counting something true throughout, because a negative
line is still one line.

## The blob

`net-blob.ts`: a JSON header describing a binary payload, in one buffer.

```
u32  header length, little-endian
     header JSON, UTF-8
     padding to an 8-byte boundary
     sections, back to back
```

The header carries the genome's dimensions and, for every section, its name,
element type, byte offset, total count and per-body stride — plus field names
where the stride is a named tuple, and the value names where it is an
enumeration. So a reader does not have to be this module: walk `sections`
with a `DataView` and you can render `energyCap` by name without importing
anything from the simulation.

| section     | type | stride        | what                                   |
| ----------- | ---- | ------------- | -------------------------------------- |
| `pose`      | f64  | 3             | x, y, heading                          |
| `scalar`    | f64  | 6             | extra, then the heritable traits        |
| `critic`    | f64  | `CRITIC_LEN`  | the value readout's weights and bias    |
| `prevValue` | f64  | 1             | last frame's value estimate             |
| `h`         | f64  | `STATE_DIMS`  | the recurrent state                     |
| `chem`      | f32  | `CHEM_LEN`    | the genome                              |
| `plastic`   | f32  | `PLASTIC_LEN` | what this body learned while alive      |
| `trace`     | f32  | `PLASTIC_LEN` | the eligibility trace                   |
| `ancestry`  | i32  | 2             | born, lineage                           |
| `wire`      | i32  | 4             | body index, slot, body index, slot      |
| `kind`      | u8   | 1             | 0 era, 1 dup, 2 con                     |

Sections are laid out widest-alignment-first, so each lands on its natural
boundary and a decode is views onto the buffer rather than a copy.

Wires address bodies by **index into this blob**, not by agent id. Slots are
`0 = p, 1 = l, 2 = r`.

### The header is a safety check, not decoration

`chem-layout.ts` exists because a hand-copied `CHEM_LEN` drifted twice in one
session and the second time shipped. A blob written before such a drift and
read after it would not crash — it would cut every body's genome at the wrong
offsets and hand a lineage's evolved behaviour back as garbage that still
runs. So `decodeNet` compares the header's dimensions against the ones the
running build compiled with and refuses the blob, naming both. `run.chem_len`
records the same thing at the run level, so a database written across a layout
change says where the boundary is.

### What is deliberately not stored

Agent ids, mass, scale and alpha (derived from kind and params at birth),
velocity, drive, stun, trail, and the rope node positions. A planted net is a
net dropped into different water; none of that survives the move in any
meaningful sense, and `plantNet` zeroes it.

`trace` and `h` *are* stored even though neither is inherited and both are
gone within a second of simulated time. They cost 68 floats a body and they
are the difference between reloading a net and reloading the net that was
actually running.

## Sweeps, and what they are for

```bash
npm run pond -- sweep --name uptake --axis uptakeVmax=0,0.75,2 --axis uptakeKs=0.1,0.5 \
                --seeds 1,2,3 --seconds 180 --bodies 500
npm run pond -- analyze                      # what sweeps are in the library
npm run pond -- analyze --name uptake
```

`src/experiments/` already sweeps: a grid crossed with seeds, a table on
stdout, one JSON file per invocation. What it cannot do is **remember**. Every
sweep is a fresh file, nothing relates one to the next, and "what have we
learned about `declutter`" is answered by opening whichever files you still
have and reading them by eye.

A sweep here writes into the library instead. A trial is a `run` row with its
whole `Params`, its commit, its timeline and optionally its nets; the sweep's
name and grid point go on the row. Which makes the analysis a query over
everything ever run rather than over this afternoon, and makes a sweep
resumable, comparable across commits, and joinable against the nets it made.

### The effect table is the output

A per-point table is what a sweep *produced*. The effect table is what it
**found**: each axis ranked by how much of the variation in a metric it
explains, against the seed-to-seed noise it has to beat.

```
axis        metric     eta2    noise   low            ->  high          n
uptakeVmax  netFst     0.9078  0.0353  0: 0.4641      ->  1.50: 0.6871  4
uptakeVmax  free       0.8397  6.98    1.50: 460      ->  0: 499        4
```

The statistic is eta-squared — the share of a metric's total variance lying
between an axis's levels rather than within them. Bounded in [0, 1], no
distributional assumption, and it reads the way you want: 0.02 means the dial
did nothing you could see over three seeds, 0.9 means it is most of what
happened. It **ranks; it does not test.** Over a small grid with everything
else held, this describes the sweep in front of you and estimates nothing.

### How many seeds, and how long

**Three seeds is not enough for this pond, and the first sweep run through
this harness is the evidence.** `uptakeVmax` x `uptakeKs`, three levels by
two, three seeds, 180 simulated seconds from a 500-body soup — eighteen
trials, and the largest effect it found was eta-squared 0.21 against a
within-level spread of 0.11. Everything else was below 0.18. The per-point
standing deviations were the story: bodies 850 ± 822, depth 14.6 ± 17.9,
commutes 1301 ± 535. One seed in three routinely does something the other two
do not.

So: **treat eta-squared below about 0.3 at three seeds as unresolved**, and
raise the seed count rather than the grid when a result matters. A pond is a
chaotic system with a rewrite budget spent down stochastically; the seeds are
not a nuisance term, they are most of the variance.

The corollary is worse and worth stating plainly. A 25-second smoke run of the
same axis gave eta-squared **0.91** for `uptakeVmax` on `netFst`; the
180-second version gave **0.21**. Short trials do not measure a weak version of
the same thing — a young pond is still resolving its preset, and its
differentiation is founder mixing rather than anything a dial did.

### Which measure to put on the outcome

Not all of them cost the same number of seeds. Measured over thirty trials at
five seeds, the seed-to-seed coefficient of variation (standard deviation over
mean, at the last sample):

| measure | CV |
| --- | --- |
| `ground` | 0.151 |
| `netFst` | 0.257 |
| `bodies` | 0.313 |
| `netsEffective` | 0.381 |
| `linesEffective` | 0.451 |
| `commutes` | 0.504 |
| **`bornMean` (depth)** | **0.728** |

**Depth is the noisiest thing here**, because it is a mean over a distribution
with a long tail — a handful of deep lineages carry it, and whether a pond
grows one is close to a coin flip. Resolving a 30% difference in depth wants
about **25 seeds**; `netFst` wants about five for the same relative effect.

So: put `netFst` or `ground` on the outcome where the question allows it, and
budget seeds properly when it has to be depth. This is why the `excreteRate`
sweeps resolved at five seeds and the ground-structure ones did not — excretion
moved depth ninefold, far past the noise, while structure moves it by tens of
per cent.

### Before sweeping, ask what the constants mean at each end

The harness assumes every grid point is comparable at a fixed `--set`. That is
false whenever one dial changes what another *means*, and it cannot detect it —
the assumption is about the model, not the grid. Two instances found the hard
way in one afternoon:

- **`senseScale` against `excreteRate`.** Conserved excretion drops signal
  amplitude by three orders, so one sense scale cannot serve both regimes. A
  crossed sweep left the excreting arm effectively blind, in a sweep about
  whether bodies navigate to structure. Run them as two sweeps.
- **`energyRegrow` against `groundPatches`.** Logistic growth is zero at both
  ends and `Fields.grow` skips a cell at `e <= 0` and at `e >= cap`. A uniform
  dish is seeded at exactly `cellCap`, so it produces *nothing* except where
  something has grazed; patches drag their edges through the productive band
  and manufacture ground as they spread. Layouts seeded at equal mass ended 50%
  apart. The clean control for a layout question is `energyRegrow = 0`.

## Diversity and divergence

`measure.ts`. The pond could always say how *much* there was — bodies, wires,
energy — and how deep a lineage ran. It could not say whether what was there
was becoming **different**, which is the only question an open-ended system is
really being asked. `matrixDrift` comes closest and cannot answer it: it rises
under drift and under selection alike, so a whole pond walking together in one
direction reads exactly like a pond splitting in two.

| measure | what it says |
| --- | --- |
| `lines`, `linesEffective` | richness, and evenness. A hundred lines of which one holds 95% is not a hundred lines, and a count says it is. `exp(H)` says it is about one. |
| `lineDominance`, `netDominance` | the largest line's and largest net's share |
| `nets`, `netsEffective` | components of two or more, by count and by size evenness |
| `varianceDrifted`, `varianceSeeded` | standing genetic variance, split by where in the genome it sits |
| **`netFst`** | **the divergence number.** Share of genetic variance lying *between* nets rather than within — Wright's F_ST in its quantitative form. Near zero, every net is a random draw from one pond-wide gene pool and the wire graph structures nothing. Rising, nets are becoming distinguishable. |
| `lineFst` | the same with founder lines as the groups, which is drift's own signature |
| `commutesPerLatch` | `docs/energy-chemistry-plan.md` §8's tripwire |

`netFst` is **null**, not zero, when there is nothing to compare — fewer than
two nets, or no variance at all. Zero would read as a finding where there is
an absence, and the effect ranking drops nulls rather than scoring them.

Both F_ST calibrations are pinned by tests: genomes shuffled at random between
nets must read near the floor (or the statistic is measuring group *sizes*),
and one constant genome per net must saturate.

## The chemistry dials

`docs/energy-chemistry-plan.md` phases 0 to 4 are in, and every one of them
ships at the value that reproduces the pond before it. Turning them on is a
sweep, not a default.

| dial | 0 / today | what it does |
| --- | --- | --- |
| `uptakeVmax` | 0 | uptake becomes a *rate* against the ground's density instead of take-what-fits |
| `uptakeKs` | 0.25 | half-saturation; with `uptakeVmax` it is the non-dominating grazer/scavenger pair |
| `excreteRate` | 0 | **the switch.** All four species leave the tank conserved, and the minted scent deposit stops |
| `senseScale` | 4.3 | what one unit of signal is worth on the way in. See below |
| `yDirect`, `yEra` | 1, 1 | uptake yield; `yDirect` 0 is obligate trophic dependency, `yEra` above 1 makes an Era the net's mouth |
| `upkeepExcrete` | 0 | rent returns to the ground rather than vanishing |
| `bodyValue` | `EXTRA_CAP` | at `REWRITE_SHARE` a commute stops minting |
| `eraCapRatio`, `eraUpkeepRatio` | 2, −0.2 | at 1, 1 the two rules keyed on an Era's glyph are gone |
| `rowCost`, `hillN` | 0, 1 | the two superadditivity dials §3 argues division of labour needs: a fixed cost per expressed row, and a Hill exponent on uptake |

**`senseScale` has to move with `excreteRate`.** Measured as the p90 signal
reading at a body's own position over soups of 60, 400 and 2000 — the same
measurement the constant was set by — a minted pond reads 4.26 / 4.60 / 5.19
and an excreting one reads **0.0016 / 0.0017 / 0.0024**. Left at 4.3, a run
with `excreteRate` on has sense genes reading three orders below anything
`phi` can resolve. A useful starting point:

```bash
npm run pond -- run --seconds 600 \
  --set excreteRate=0.015 --set uptakeVmax=6 --set senseScale=0.002
```

### Why those numbers

Measured over four sweeps, 600 simulated seconds from a 500-body soup, five
seeds a point, against a control with the chemistry off (lineage depth 68.7).

`excreteRate` is the dial that matters — it explained 68–93% of the variance
in reproduction, energy and effective lines, while `uptakeVmax` explained
2–8% and was unresolved. At 0.5 the pond is **sterile**: excretion is mass
action on the tank, so bodies shed faster than they can refill and never hold
the share a commute costs (depth 1.09, free 0.11 a body against a share of 1).
Lowering it recovers reproduction, and then **stops helping** — 0.015 and
0.005 are indistinguishable at depth ~13, while 0.005 throws away two-thirds
of the standing signal for nothing. 0.015 is where the curve turns over.

The remaining cost turned out not to be excretion at all. Phase 1's metered
uptake was on in every chemistry run and off in the control, and it was most
of the gap: at `excreteRate` 0.015, depth runs 13.2 metered at 1.5, 25.3 at 6,
and 39.0 unmetered. Unmetered is the cheapest by reproduction and the most
differentiated pond measured (`netFst` 0.418) — but it takes the
single-species harvest path, so bodies would excrete four species and eat one,
which is the loop phase 3c exists to close. 6 keeps the table whole at 2.7x
the control's depth, and it was the top of the range with the trend still
rising, so there is probably more there.

**Read these as a regime, not a tuning.** The seed spread is wide — depth
25.3 ± 13.8 at the recommended point — so the large steps are real and the
fine ranking is not. "0.015, where the curve turns over" is a claim the data
supports; "0.015 rather than 0.01" is not.

Two couplings the code enforces rather than trusting you to remember. Uptake
covers all four species only when `excreteRate` is on — metered uptake with
the deposit still minting is a matter fountain, since a body would eat back
five times the voice it never paid for. And excretion and the minted deposit
are mutually exclusive at all three places a deposit is laid.

## Reading it from the lab page

Nothing in `db.ts` is importable from a page — `node:sqlite` is not. The
format is split so it does not have to be. `net-blob.ts` imports only
`chem-layout.ts` and the kind codes, so it runs in the browser unchanged, and
whichever way the rows arrive — a dev-server endpoint that runs `db.ts`, or
the `.db` file fetched and opened with a wasm SQLite build — the blob the page
decodes is the same blob with the same reader.
