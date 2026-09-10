# Review: metabolism, locomotion, learning

2026-09-10. A reading of three mechanics against `concepts.md`, the plans,
and the code as it ships at HEAD (`aaae4ed`). Each finding names the concept
heading it is about and the mechanic that heading already has, and every
proposal here is a tuning, a measurement, a bug fix or a documentation fix.
Nothing adds a fitness term, a decay, or a mechanic.

Where a number is quoted it was either taken from the docs (cited) or from
integrating `advanceGait`'s reaction loop on its own, outside a pond, which
is the same thing the pathway's own commits did. None of it is a pond
measurement; the protocols in §5 are the pond measurements this review asks
for.

---

## 0. The finding the rest sits under

At the shipped defaults, none of the three mechanics is doing the job its
heading names, in the pipeline that ships:

| mechanic | heading | shipped state | why it is not engaged |
|---|---|---|---|
| the pathway (`advanceGait`) | Net: Locomotion, its clock; Agent: Metabolism | `metabolicRate 0` | switched off, pending a sweep (`b6f6b80`) |
| metered uptake, the gut | Agent: Metabolism, eating | `uptakeVmax 0` | the take-what-fits path, instantaneous, in id order |
| grip and the packet | Net: Locomotion | `grip 2`, `transportQuantum 0.5` | on; but a stroke needs a fullness difference and `full_mean` sits near 1 |
| three-factor learning | Agent: Learning | `learnRate 0.02` | on; the teacher is `FULL - 1`, which is 0 for a full body |

So locomotion ships as grip plus packets with the gait off, and learning
ships on with a teacher that reads zero wherever the economy is not in play.
`experiments.md` §1.B found exactly this shape once already (demand read
0.000 under `uptakeVmax 6`). The improvement is not a mechanism. It is one
protocol that engages all three at once (§5) and reads them against their
gauges, before any of the tunings below are shipped.

---

## 1. Metabolism: the pathway

Heading: Agent, Metabolism (upkeep) and Net, Locomotion (the clock).
Mechanic: the three-reaction pathway over a conserved adenylate pool in
`Sim.advanceGait`, spending through `payOut`.

### 1a. The oscillation band is pool-relative, and the pool is the one heritable thing

`metabolicRegen`'s note derives the balance `u* (regen - supply/pool) =
work(u*)` and says the band is per body because `adenylate` is heritable.
That is understated. Integrating the loop as written, one fed body, sixty
seconds, everything else at the shipped values:

| `adenylate` | swing of the wave | spend, tank units/s |
|---:|---:|---:|
| 0.8 | 0.00 | 0.171 |
| 1.0 | 0.00 | 0.168 |
| 1.3 | 0.00 | 0.163 |
| **1.5 (seed)** | **1.86** | **0.031** |
| 2.0 | 0.00 | 0.016 |
| 3.0 | 0.00 | 0.007 |

Only the seeded value oscillates. One mutation step is 0.15 and the range
is 0.2 to 6, so a child lands outside the band more often than in it. Below
the band the body sits at the discharged clamp and pulls substrate at full
rate: eleven times its rent (`upkeep` 0.015), for a clock that never ticks.
Above it the body sits charged and still. The gene the concept doc calls
"the first thing about a lineage's metabolism that selection can reach" is
a trap on both sides of its seed.

The cause is that `supply` is in absolute reaction units while `1 - charge`
is a fraction of the pool, so the influx per unit of discharge does not
scale with the pool. Make it scale:

```
want = supply * pool * (1 - charge) * hs      // supply per unit of pool
```

and ship `metabolicSupply` at 2 (today's 3 over the seeded 1.5, and `2 *
1.5` is exactly 3 in floating point, so the seeded body is bit-identical).
The balance becomes `u* (regen - supply) = work(u*)`, independent of the
pool. Same integration:

| `adenylate` | swing | spend/s |
|---:|---:|---:|
| 0.8 | 1.75 | 0.027 |
| 1.0 | 1.80 | 0.028 |
| 1.5 | 1.86 | 0.031 |
| 2.0 | 1.90 | 0.032 |
| 3.0 | 1.93 | 0.040 |
| 4.0 | 1.95 | 0.051 |

Every pool oscillates, and the pool now does what its gene comment says it
does: sets the depth and period of the stroke, not whether there is one.
The narrow band the concept doc calls "a real property of a two-pool
network" is still narrow in `regen`; it is just no longer a function of a
gene. One line, a default change, and `gait.test.ts`'s oscillation criterion
run at three pool sizes rather than one.

### 1b. Work is charged on holding, not on moving, and on bodies with nothing to move

The comment says the stroke "discharges the pool in proportion to how far it
is actually swinging a wire". The code charges `workRate * |atp - adp| /
pool`, which is `|wave|`: the distance from mid-charge, not the rate of
change. A body holding its wires at full extension pays the most; one
swinging through the midpoint pays nothing. And it is charged per body
whether or not the body has a wire, so a loner runs the clock at the full
load.

At the shipped band a fed loner spends 0.031/s, twice its rent, on a clock
nothing reads. Gate the work term on wired degree (the store already has
`bound`), and a loner's balance forces `u* = 0`: its pathway sits charged,
buys almost nothing (0.0009/s in the same integration), and starts ticking
when it latches. That is the right behaviour and it costs one multiply. It
is also most of the "sixth of a tank inside a couple of seconds" that
`b6f6b80` cites as the reason the pathway ships off; the rest is `gaitSwell`
"sized to be seen rather than to be safe".

Whether the load should be isometric (`|wave|`, as now) or isotonic
(`|dwave/dt|`) is a smaller question and either is defensible; the comment
should say which.

### 1c. "Nothing is destroyed" is conditional on a dial that ships at 0

`concepts.md` and `advanceGait`'s comment both say the pathway's spend lands
on the ground through the rent path. It does so only when `upkeepExcrete >
0`, and it ships at 0, where the spend is destroyed exactly as rent is.
That is consistent with rent and fine as a regime; the docs should say
"when `upkeepExcrete` is on" rather than stating conservation as a property
of the mechanism.

### 1d. Diffusive coupling of identical clocks is synchrony, and synchrony does not walk

`82c954f` removed `gaitLag` on the argument that "a reaction that diffuses
carries a front". A front is what you get from one excited cell in a
resting medium. A chain of identical limit-cycle oscillators coupled
diffusively (`metabolicDiffuse` on `sub`) tends to in-phase synchrony, and
in synchrony every wire's two ends are at the same point of the same
cosine, which the concept doc's own bench numbers put at 0.000 px for two
Cons. What is left is the kind difference (Era 0.02 against node 0.25),
which is a ratchet on each Con-Era pair and not a wave along the body.

The packet bench already learned the lesson that applies: "uniform quanta
swim at the noise floor; quanta varying along the body swim". A travelling
wave in a chain of relaxation oscillators comes from a frequency gradient.
The frequency here is set by the pool, and after §1a a graded pool is a
graded period without leaving the band. So the measurement is on the worm
bench, which has no pathway rig today (`worm.exp.ts` never sets
`metabolicRate`): pathway on, `adenylate` uniform against `adenylate`
graded head to tail, centre-of-mass speed. If graded swims and uniform does
not, the concept doc gets its "phase something can set" from a gene that
already exists, and the Era-as-oar question in §1e has a control.

### 1e. Known and not repeated

The Era being both the oar (seeded anchor) and the fullest body on its wire
(larger store, producer discount) is in `concepts.md` already, as a pond
question. It stays one. §1d's bench is the cheapest way to ask it: run the
graded worm with and without Eras.

---

## 2. Locomotion: the drag law

Heading: Net, Locomotion. Mechanic: `dampVelocities`, rate `drag + grip *
fullness + anchor`.

### 2a. The shipped stroke is paid for in starvation

With the pathway off, the shipped locomotion is `grip 2` and the packet.
`experiments.md` §8 records what the packet costs on its own: 183 bodies
against 481 at 60 simulated seconds with grip at zero, because a chain of
bodies each holding less than a packet cannot feed each other. That is the
"mechanic asked to do two things" the concept doc admits, and it is the
expensive half.

The pathway was built so the clock need not be the packet. If the sweep in
§5 finds the pathway carries a net, `transportQuantum` can return to 0, the
trickle, and transport stops paying for locomotion. If it does not, the
packet's cost should at least be put against its travel in one table:
`net_drift` against a `transportRecoil 0` control, with `died` and
`can_pay` beside it.

### 2b. A global constant cannot have its sign settled by the pond

`grip`'s note says "the sign is left open, because the pond should settle it
rather than this file". `grip` is a global. Nothing in the pond can move it.
The doc then says making it heritable "costs a genome head", but the cheap
path is already taken by `adenylate`, `requestDecay` and `transportQuantum`:
a scalar trait in `TRAIT_KEYS`, seeded from the slider, with the slider's
range as its clamp (`-4` to `12`). No head, no change to `CHEM_LEN`, so
`matrix_drift` stays comparable across the library, and the sign becomes
the pond's to settle in the way the note asks for. The drag table already
spans the slider's maximum, so `dampVelocities` needs no change.

This also makes the two drag modulations under this heading, `grip *
fullness` and `anchor * wave`, both heritable, so which one a lineage walks
on is selection's business rather than a default's.

### 2c. The concept entry describes a mechanism that no longer exists

`concepts.md`'s Locomotion entry introduces `gaitRate`, `gaitCouple` and
`gaitLag` in the present tense, then says the clock is a metabolism.
`82c954f` removed all three. A reader with no history, which is who the
document is for, cannot tell that the cosine clock is gone and that its
lessons (the clock cannot come out of `h`; the stroke cannot be a length)
are what survive. The entry wants rewriting as: the actuator (rest length),
the two things that make a difference between a wire's ends (fullness,
anchor), the clock (the pathway), and the not-yet-true list. The history
belongs in the commit log, which has it.

---

## 3. Learning: the three-factor rule

Heading: Agent, Learning. Mechanic: eligibility trace times TD error from a
linear critic, in `updateState` and `genome.wgsl`, on `Wx`, `Wh`, `Wn`, `b`.

### 3a. The rule has no direction per state dimension

The update is

```
e_dj  = lam * e_dj + phi'(v_d) * pre_j
w_dj += eta * dlt * e_dj
```

`phi'` is positive, `pre_j` is whatever the input was, and `dlt` is one
scalar for the body. So a positive surprise pushes every one of the four
pre-activations up along its recent inputs, and a negative surprise pushes
all four down. There is nothing in the rule that says whether `h_0` going
up was what made the tank fill. The plan calls this "the RFLO / e-prop
form", but e-prop's learning signal is per unit (`sum_k B_dk * error_k`),
and here it is one sign broadcast to all units. What this optimises is
`dlt * sum_d v_d`: the state is shaped to correlate with prediction error,
and behaviour changes only through whatever sign the fixed output heads
happen to put on each dimension.

The plan's Phase 3 asked for "a rigged two-body net where need falls and
the learned weights move in the direction that made it fall". That test was
never written; `plasticity.test.ts` checks that weights move, that they
persist, and that they are inherited, not which way they go. Write it
first. It will fail against the rule above, and that failure is the
measurement that licenses the change.

The change, within the mechanic: node perturbation. Add a small hashed
perturbation `xi_d` to each `v_d` (hashed from slot and frame, so
`state-hash` stays deterministic), and put the perturbation in the trace in
place of the unsigned `phi'`:

```
e_dj  = lam * e_dj + xi_d * pre_j
```

With `dlt` as the modulator this is the Fiete and Seung rule, an unbiased
estimate of the reward gradient through the heads without the heads having
to be differentiable or to learn. It costs four hash draws a body a frame
and no new state. The perturbation doubles as the exploration the body
otherwise does not have: `swimNoise` is downstream of `h` and cannot serve.

The cheaper alternative, which keeps the code shape and adds no noise, is
to use the critic's own weights as the per-dimension sign: `e_dj += c_d *
phi'(v_d) * pre_j`. That climbs the critic's estimate rather than the
reward and is weaker, but it at least gives each dimension a sign the body
learned. Perturbation is the recommendation.

### 3b. The teacher is pinned in the shipped regime

`r = FULL - 1` is zero at a full tank, and in a pond on ambient ground with
uptake unmetered most tanks are full most of the time (`experiments.md`
§1.B). Learning is on and idle. `full_mean` is its engagement gauge and no
protocol names it as one; `baldwin-hunger` should.

The plan left `IN_DEMAND` as the alternative teacher, "one line away".
Demand is nonzero for a full body whose net has a hungry member, so it
engages far more often, and it is the organism-level signal the audit
originally asked for. Run the two as arms of `baldwin-hunger` rather than
deciding it here.

### 3c. The horizon is per frame, and the frame is not fixed

`learnTrace` and `learnDiscount` are per-frame decays. The runner steps at
`spec.dt`; the browser steps at whatever the clamp (`0.05`) leaves, so a
browser pond at 30 fps has half the headless pond's horizon at the same
setting. `advanceGait` and every drag rate are in per-second units and take
`dt`; the learning rule should too: `lam = exp(-dt / tau)` with `tau` in
seconds. At `dt = 1/60`, `tau = 1.66 s` reproduces 0.99.

While there: the shipped 0.99 is 1.7 s against a foraging trip of 5 to 12 s
(`experiments.md` §7), the slider stops at 0.995, and `learnTrace`'s note
still reads "0.95 is about twenty frames", which was true before `172dbff`
moved the default.

### 3d. Inheritance compounds whatever 3a produces

`inheritLearned` at 1 writes a parent's learned delta into its children's
genome. That is the design. Until §3a is fixed, what is consolidated is a
state shaped to correlate with surprise, not a behaviour that reduced it,
and `baldwin-hunger`'s loci would be reading that. Fix the rule, then run
the protocol.

---

## 4. Documentation fixes, all small

- `concepts.md`, Locomotion: rewrite per §2c; conservation caveat per §1c.
- `params.ts`, `learnTrace`: the stale "0.95 is about twenty frames".
- `params.ts`, `grip`: the sign note, once §2b lands.
- `advanceGait`'s comment: say what the work term charges (§1b).

---

## 5. What to measure, in order

1. **The direction test** (§3a). A test, not a pond. Cheapest and decisive.
2. **Pool-relative supply** (§1a) and **work gated on wires** (§1b). Both
   bit-identical to today at `metabolicRate 0`, so they ship at once; the
   pathway still ships off.
3. **The worm with a pathway** (§1d). `worm.exp.ts` gains a rig: pathway
   on, `adenylate` uniform against graded, with and without Eras.
   Centre-of-mass speed, twenty seconds, the bench's own noise floor.
4. **A protocol, `locomotion-clock`.** Arms: `transportQuantum` 0 and 0.5.
   Axis: `metabolicRate` 0 and the bench's value from step 3. Held: `grip`
   2. Control: `transportRecoil 0`. Gauges: `full_mean < 0.9`,
   `demand_mean > 0`, `died`, `can_pay`. Outcome: `net_drift`, read
   against the control, and `born_mean` for what it costs. Ten minutes,
   five seeds, per §3 of `experiments.md`. This is the sweep `b6f6b80`
   says has to run before the pathway can ship on, and the same run says
   whether the packet can go back to 0.
5. **`baldwin-hunger`, again**, after §3a, with a second arm for the
   teacher (`FULL` against `DEMAND`) and `full_mean` as a precondition.
6. **`grip` heritable** (§2b) after step 4 says a sign carries a net at
   all, which is the order `grip`'s own note asks for.
