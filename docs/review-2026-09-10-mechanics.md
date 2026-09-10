# Review: metabolism, locomotion, learning

2026-09-10, revised the same day after vetting. A reading of three mechanics
at HEAD (`aaae4ed`) against `concepts.md` and the code. Each finding names
the concept heading it is about and the mechanic that heading already has.
Nothing here adds a fitness term, a decay, or a mechanic.

**How each claim was checked.** Every claim below is tagged:

- **[code]** derived from reading the source, not comments, tests or docs.
- **[sim]** measured by driving the real `Sim` headlessly (CPU field, wasm
  solver where the run says so), with the scripts' assumptions stated.
- **[unresolved]** measured and inconclusive at the seeds run.

Three claims in the first draft of this review were wrong and are retracted
in §6. Nothing in this document rests on a number quoted from
`experiments.md`, the test suite, or a comment.

---

## 0. What ships, and what is engaged

[code] At `defaultParams()`:

| mechanic | heading | shipped | consequence |
|---|---|---|---|
| the pathway, `advanceGait` | Net: Locomotion (clock); Agent: Metabolism | `metabolicRate 0` | returns before the loop; wave and anchor 0; `strokeOf` is exactly 1 |
| metered uptake and the gut | Agent: Metabolism (eating) | `uptakeVmax 0` | the take-what-fits path |
| `grip` and the packet | Net: Locomotion | `grip 2`, `transportQuantum 0.5` | on |
| three-factor learning | Agent: Learning | `learnRate 0.02` | on |

[sim] Whether the learning teacher and the grip stroke have anything to
work on depends on the pond's own population, which varies more between
seeds than between arms. Eight 120-second ponds, 400 to 500 founders, no
patches, both packet settings:

| run | bodies at 120 s | `full_mean` | `demand_mean` |
|---|---:|---:|---:|
| seed 1, 2400², quantum 0 | 514 | 0.64 | 0.44 |
| seed 1, 2400², quantum 0.5 | 861 | 0.27 | 0.65 |
| seed 2, 2400², quantum 0 | 281 | 0.88 | 0.11 |
| seed 2, 2400², quantum 0.5 | 315 | 0.95 | 0.003 |
| seed 3, 2400², quantum 0 | 487 | 0.94 | 0.07 |
| seed 3, 2400², quantum 0.5 | 996 | 0.18 | 0.64 |
| seed 1, 1600×1200, quantum 0 | 615 | 0.86 | 0.07 |
| seed 1, 1600×1200, quantum 0.5 | 893 | 0.58 | 0.007 |

The same arm runs from pinned full (0.95, demand 0.003) to hungry (0.18,
demand 0.64) depending on the seed, because the pond decides its own
density. So "is the economy in play" is not a property of the defaults; it
is a per-trial gauge, and any claim about learning or the stroke has to be
read against it. The protocol machinery's `preconditions` is the right
tool; `full_mean` and `demand_mean` are the gauges.

---

## 1. Metabolism: the pathway

Heading: Agent, Metabolism (upkeep) and Net, Locomotion (the clock).
Mechanic: three reactions over a conserved adenylate pool in
`Sim.advanceGait`, spending through `payOut`.

### 1a. The oscillation band is pool-relative, and the pool is the one heritable gene

[code] The influx is `supply * (1 - charge)` with `charge = atp / pool`,
while every other rate is per unit of pool. Summing the two rows leaves a
steady-state balance `u* (regen - supply / pool) = work(u*)`, so whether a
steady state exists at all depends on `pool`, and `pool` is `adenylate`,
heritable with range 0.2 to 6 and a mutation step of 0.15.

[sim] One fed Con, no wires, no rent, no ground, `metabolicRate 6`,
sixty seconds, swing of `gaitWave` over the second half and tank spend per
second (the tank is refilled every frame and the shortfall counted):

| `adenylate` | swing | spend/s |
|---:|---:|---:|
| 0.8 | 0.00 | 0.171 |
| 1.0 | 0.00 | 0.169 |
| 1.3 | 0.00 | 0.164 |
| **1.5 (seed)** | **1.86** | **0.031** |
| 2.0 | 0.00 | 0.016 |
| 3.0 | 0.00 | 0.008 |

Only the seeded pool oscillates. Below it the body sits at the discharged
clamp buying substrate at eleven times its rent for a clock that never
ticks; above it the body sits charged and still. One mutation step in
either direction leaves the band.

Scale the influx with the pool, `supply * pool * (1 - charge)`, and ship
`metabolicSupply` at 2 (`2 * 1.5` is exactly 3 in floating point, so the
seeded body is bit-identical). [sim] Emulated through the global by setting
`supply = 2 * pool` for each pool:

| `adenylate` | swing | spend/s |
|---:|---:|---:|
| 0.8 | 1.75 | 0.027 |
| 1.0 | 1.80 | 0.028 |
| 1.5 | 1.86 | 0.031 |
| 2.0 | 1.90 | 0.033 |
| 3.0 | 1.93 | 0.040 |

Every pool oscillates. The pool then sets depth and period rather than
whether there is a clock. The band in `regen` stays narrow; it stops being
a function of a gene.

### 1b. Work is charged on holding, not on moving, and on bodies with nothing to move

[code] The load term is `workRate * |atp - adp| / pool`, which is `|wave|`,
the distance from mid-charge and not its rate of change. It is charged per
body with no reference to wires.

[sim] Same rig as §1a at the seeded pool: a loner spends 0.031/s with
`metabolicWork` at its shipped 0.6, and 0.0012/s with it at 0, where the
pathway parks charged. Rent is 0.015/s. So a loner pays twice its rent to
run a clock nothing reads. Gating the work term on wired degree
(`store.bound` is there) gives a loner the second row and a wired body the
first, for one multiply.

Whether the load should be isometric (`|wave|`) or isotonic (`|dwave/dt|`)
is a smaller question; either is defensible, and the code should say which
it is.

### 1c. The spend is destroyed at the shipped defaults

[code] The pathway's spend reaches the ground only through `payOut`, and
only when `upkeepExcrete > 0`. It ships at 0. That matches rent and is a
regime, not a bug; `concepts.md` states conservation as a property of the
mechanism and should state the condition.

### 1d. Diffusive coupling gives synchrony, and synchrony does not walk

[sim] Twelve Cons in a chain wired principal to left aux, pitch 72,
`wireMinRest 72`, fed every frame, `metabolicRate 6`, `gaitSwell 0.3`,
`grip 2`, no rent, no ground, no steering, no rewriting. The spread is the
range of `gaitWave` across the twelve at an instant, averaged over a window;
a full swing is 1.86. Three starts: the birth hash as coded, the birth hash
as intended (see §1e), and ATP and substrate drawn uniformly.

| start | `metabolicDiffuse` | spread 5–15 s | spread last 10 s | centre travel, 35 s |
|---|---:|---:|---:|---:|
| as coded | 0 | 0.27 | 0.31 | 0.2 px |
| as coded | 2 | 0.04 | 0.01 | 0.5 px |
| as intended | 0 | 0.22 | 0.34 | −2.5 px |
| as intended | 2 | 0.01 | 0.00 | 0.1 px |
| uniform | 0 | 0.72 | 0.59 | −4.4 px |
| uniform | 2 | 0.07 | 0.01 | 0.1 px |

With the substrate diffusing along wires the chain is in phase inside ten
seconds from every start, and an in-phase chain of one kind does not move.
An unsynchronised chain twitches a few pixels in half a minute, which is
the sqrt(n)-of-uncorrelated-strokes case; a loner's `stepSpeed` is 38 px/s.
Adding an Era leaf at every Con (seeded anchor 0.02 against 0.25) moved the
centre about 1 px in the same time. Nothing here is travel.

A travelling wave along a chain of relaxation oscillators comes from a
frequency gradient, not from diffusion. After §1a a graded `adenylate`
along a body is a graded period that stays in the band. That is the
measurement to make, and it wants no new mechanic. `worm.exp.ts` has no
pathway rig today [code: it never sets `metabolicRate`].

### 1e. The birth-phase hash is signed, and the intended scatter is not applied

[code] `createAgent` ends its hash with `mix ^= mix >>> 16`. Every earlier
step is forced unsigned with `>>> 0`; this one is not, and `^=` in
JavaScript yields a signed 32-bit integer. `mix / 4294967296` therefore lies
in [−0.5, 0.5) rather than [0, 1), and `atp = pool * (0.25 + u * 0.5)` lands
in [0, 0.5) of the pool rather than the intended [0.25, 0.75).

[sim] Read straight off `store.atp` after twelve spawns at pool 1.5: 0.007,
0.097, 0.144, 0.19 … 0.744. Ids 2, 3 and 4 of the first five hash negative.
The fix is `>>> 0` on the last line. It moves `state-hash` for any pond
with the pathway on and nothing else.

Its consequence is smaller than it looks. [sim] Forty unwired bodies born
on the same frame settle within a tenth of a period of each other whether
the scatter is as coded or as intended, because the relaxation oscillator
collapses initial conditions onto its slow branch. In a pond, birth *time*
scatters phase far more than the hash does. What the hash does decide is
that a commute's four children, born on one frame with consecutive ids,
start in phase with each other.

---

## 2. Locomotion: the drag law

Heading: Net, Locomotion. Mechanic: `dampVelocities`, rate `drag + grip *
fullness + anchor`.

### 2a. The packet did not cost population in two minutes; what it costs is unresolved

[sim] The table in §0. At 120 s the packet arm had more bodies than the
trickle arm in all four seed-matched pairs (861 vs 514, 315 vs 281, 996 vs
487, 893 vs 615), and `died` was 0 to 3 in every run. Nobody starved. In
the two pairs where the packet pond grew large it also ran hungry
(`full_mean` 0.18 to 0.27), which is density, not the packet.

[unresolved] Four pairs with a consistent direction supports "the packet
does not reduce population inside two minutes" and nothing finer; the
seed-to-seed spread within an arm (281 to 514) is larger than the arm
difference in two of the four. What the packet buys in travel was not
measured here: `net_drift` against a `transportRecoil 0` control is the
protocol (§5), and it is the same sweep that decides whether the pathway
can be the clock instead.

### 2b. A global constant cannot have its sign settled by the pond

[code] `grip` is read only from `params`; no store array, gene, or trait
carries it. The drag table in `dampVelocities` spans `drag + max(0, grip) +
GAIT_ANCHOR_MAX` off the same global. Making it a scalar trait in
`TRAIT_KEYS`, seeded from the slider with the slider's range as its clamp,
is the path `adenylate`, `requestDecay` and `transportQuantum` already take:
no head and no change to `CHEM_LEN`. The table's span then has to be built
over the trait's maximum rather than the global.

This makes both drag modulations under this heading, `grip * fullness` and
`anchor * wave`, heritable, so which one a lineage walks on is selection's
business.

### 2c. Stale references

[code] `concepts.md`'s Locomotion entry introduces `gaitRate`,
`gaitCouple` and `gaitLag` in the present tense; none exists in `params.ts`.
`dampVelocities`'s own comment still says "with `grip` and `gaitRate` both
at 0". The entry wants rewriting around what exists: the actuator (rest
length), the two sources of difference between a wire's ends (fullness,
anchor), the clock (the pathway), and the not-yet-true list.

---

## 3. Learning: the three-factor rule

Heading: Agent, Learning. Mechanic: eligibility trace times TD error from a
linear critic, in `updateState` and `genome.wgsl`, on `Wx`, `Wh`, `Wn`, `b`.

### 3a. The rule has no direction per state dimension

[code] For every weight `w_dj` feeding state dimension `d` from input `j`:

```
e_dj  = lam * e_dj + phi'(v_d) * pre_j       phi' > 0
w_dj += eta * dlt * e_dj                     dlt one scalar per body
```

The sign of the update is `sign(dlt) * sign(e_dj)`, and `e_dj` is a
positively weighted sum of the input's recent values. So for an input that
does not change sign (`IN_FULL`, `IN_BOUND`, `IN_DEMAND`, the bias, the
sense readings), all four state dimensions move the same way on every
frame. Nothing in the rule relates the direction of `h_d` to the outcome.
The GPU rule is line-for-line the same [code: `genome.wgsl` learning
block].

[sim] Two single-body arms, the tank written each frame from the body's own
`h_0`, learning at the shipped rate, 50 seconds. In the arm where the tank
fills when `h_0` is *high*, the rule drove `h_0` to −0.93 and ran the
`IN_FULL` column and the bias of all four rows to the negative clamp
(−3.64, −2.66, identical across rows). The arm where the tank fills when
`h_0` is low was uninformative (the tank sat full and `dlt` near zero). A
separate run with a random-walk tank left every `IN_FULL` weight of every
row at the same clamped value, −3.857.

So the rule does not climb reward; in the informative arm it ran away from
it, because a shortfall (`dlt < 0`) times a positive trace lowers every
weight whatever would have helped. `plasticity.test.ts` checks that
weights move, persist and inherit, not which way; the plan's own
"direction" test was never written, and it fails.

The change inside the mechanic is node perturbation: add a hashed
perturbation `xi_d` to each `v_d` (hashed from slot and frame, so hashes
stay deterministic) and put it in the trace in place of `phi'`:

```
e_dj  = lam * e_dj + xi_d * pre_j
```

With `dlt` as the modulator this is the Fiete and Seung estimator of the
reward gradient through the heads, which need not be differentiable or
learn. Four hash draws a body a frame, no new state, and the perturbation
is the exploration the body otherwise lacks (`swimNoise` is downstream of
`h`). Write the direction test first; it is the acceptance test.

### 3b. Learning is engaged, and it is large

[sim] In every pond of §0, every body had non-zero learned weights inside
30 s. At 120 s the mean absolute learned delta per weight was 0.2 to 0.7
against a clamp of 4, with 1 to 3 % of weights at the clamp; the sense
columns were as large as any other, so every body's `readsField` gate is on
and the CPU sampling shortcut saves nothing. Given §3a, what has been
learned is a state shaped by surprise, and consolidation (`inheritLearned`
1) writes that into every child. Fix §3a before reading any inherited
locus.

The teacher question the plan left open (`IN_FULL` against `IN_DEMAND`) is
still open and is an arm in §5, not a decision here.

### 3c. The horizon is per frame, and the frame is not fixed

[code] `learnTrace` and `learnDiscount` multiply per call with no `dt`.
`main.ts` steps at `min(0.05, elapsed)`; the runner steps at a fixed
`spec.dt`. A browser pond at 30 fps has half the headless horizon at the
same setting, while `advanceGait` and every drag rate take `dt`. Parametrise
in seconds: `lam = exp(-dt / tau)`; `tau = 1.66 s` reproduces 0.99 at
1/60. Not bit-identical; pin it with the change detector.

---

## 4. Documentation fixes, all small

- `concepts.md`, Locomotion: rewrite per §2c; conservation condition per §1c.
- `params.ts`, `learnTrace`: the note still describes 0.95 as the default.
- `sim.ts`, `dampVelocities`: the `gaitRate` reference.
- `advanceGait`'s comment: say what the work term charges (§1b).

---

## 5. What to measure, in order

1. **The direction test** (§3a). A test, not a pond. Cheapest and decisive.
2. **Pool-relative supply** (§1a), **work gated on wires** (§1b), **the
   hash sign** (§1e). All bit-identical to today at `metabolicRate 0`.
3. **The worm with a pathway** (§1d). A rig in `worm.exp.ts`: pathway on,
   `adenylate` uniform against graded, with and without Eras.
   Centre-of-mass speed, twenty seconds, the bench's own noise floor.
4. **A protocol, `locomotion-clock`.** Arms: `transportQuantum` 0 and 0.5.
   Axis: `metabolicRate` 0 and the bench's value from step 3. Held: `grip`
   2. Control: `transportRecoil 0`. Preconditions: `full_mean` under 0.9
   and `demand_mean` above 0 per trial, because §0 shows both go either
   way at the defaults. Outcomes: `net_drift` against the control,
   `born_mean` for the cost. Ten minutes, five seeds.
5. **`baldwin-hunger`** after §3a, with a second arm for the teacher
   (`FULL` against `DEMAND`) and `full_mean` as a precondition.
6. **`grip` heritable** (§2b) after step 4 says a sign carries a net.

---

## 6. Retracted from the first draft

- **"The teacher is pinned in the shipped regime."** Wrong as stated. §0:
  it is pinned in some seeds and fully engaged in others, at the same
  defaults. The correct statement is that engagement is a per-trial gauge.
- **"The packet is paid for in starvation" (183 vs 481 bodies).** Taken
  from `experiments.md` and not reproduced. §2a: in four seed-matched pairs
  the packet arm ended with more bodies and nobody starved.
- **"Gating work on wires is most of the tank loss `b6f6b80` cites."**
  Wrong attribution. That loss (a sixth of a tank in seconds) is the
  stuck-discharged regime of §1a at the old `regen` of 1, which spends
  0.17/s; the work term at the shipped band costs a loner 0.031/s.
- **"The drag table needs no change for a heritable grip."** It spans the
  global; it would have to span the trait's maximum.
