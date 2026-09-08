# A worm that swims on its own metabolism

Measured 2026-09-08 on `claude/evolutionary-sim-experiments-yfvs9l`, with
`src/experiments/worm.exp.ts` at `EXP_SECONDS=30`, eight Con segments, five
seeds a condition.

The question: can `applyTransportRecoil` be a worm's only motor? Its own
comment says the effect has only ever been seen on events — a latch, a rescue,
a refill — because a topped-up soup moves ~4e-4 a frame, and names the untested
case as "a net held under a real gradient". This is that case.

## The body plan

A chain of Cons wired principal-into-aux, `p(i) -> l(i+1)`. That is the one
pairing whose two port torques agree on a straight line: a principal's axis
points along its body's heading and an aux's points against it, so each end
wants the other exactly where a straight chain puts it. Wiring aux to aux
instead makes both ends want the other *behind* them, which no chain of three
can satisfy — it buckles into a rosette. Worth knowing rather than avoiding;
it is the obvious way to build a spring, but it is not a worm.

The chain has one free principal, at the nose. Every other segment is cargo, so
none of them can swim under its own port even with swimming switched on.

## The result

Every other motor is off (`motorsOff`): no cruise, no steering, no flocking, no
breathing wires, no declutter, no latching, no immigration. What is left that
can move a body is the constraints (internal), the port torques (internal), and
the pump.

| condition | px/s | headward | straight | moved | hops | intact |
|---|---:|---:|---:|---:|---:|---|
| pump | 34.7 | 0.98 | 0.44 | 263.6 | 11186 | y |
| thrust 0 | 0.8 | 1.00 | 1.00 | **263.6** | **11186** | y |
| recoil 0 | 0.8 | 1.00 | 1.00 | **263.6** | 1 | y |
| no feeder | 0.0 | — | 0.00 | 25.1 | 70 | **n** |
| ground on | 0.0 | — | 0.51 | **0.0** | **0** | y |

**The `thrust 0` row is the whole argument.** It moves the same 263.6 units of
energy over the same 11,186 transfers and goes nowhere, because the receiver
cancels the sender exactly. The displacement is the injected momentum and not a
side effect of transfers disturbing the solver. The residual 0.8 px/s in both
control rows is the build jitter relaxing straight; it is identical across
seeds, which is the other half of the same point.

`headward` — the path-weighted cosine between the centre of mass's travel and
the worm's own tail-to-head axis — is **0.90 to 1.00 in every pumping
condition**. The worm never swims backwards or sideways. At thrust 1 the
receiver catches nothing, so each transfer kicks only the donor, and it kicks
it away from where the charge went; every segment donates in turn as charge
passes through it, so the kick travels down the body with the charge. At 6.2
transfers a frame across 7 wires, every wire is moving charge nearly every
frame. This is a peristaltic conveyor, not a jet at one end.

## What varies is straightness, not direction

| gain | px/s ± sd | headward | straight | bend swing | cot |
|---|---:|---:|---:|---:|---:|
| recoil 25 | 25.0 ± 1.7 | 1.00 | **0.93** | 1.31 | 0.354 |
| recoil 50 | 10.2 ± 5.9 | 0.99 | 0.22 | 5.81 | 1.804 |
| recoil 100 | 34.7 ± 6.7 | 0.98 | 0.44 | 5.17 | 0.265 |
| recoil 200 | 38.6 ± 5.1 | 0.90 | 0.28 | 5.50 | 0.233 |

Speed is nearly flat across an eightfold gain range while bend swing goes from
1.3 rad to ~5.5. The extra impulse does not become distance; it becomes
coiling. Same story with length: four segments hold `straight` at 0.86, eight
at 0.44, sixteen at 0.30, and speed barely moves (37.2 / 34.7 / 32.9).

**The gain is not paid for.** `transportRecoil` multiplies the impulse without
consuming anything extra — the energy budget is set by upkeep, not by the
gain — so cost of transport *improves* monotonically with gain (0.354 → 0.233)
while the worm gets steadily worse at going anywhere in particular. Any
selection scheme scored on distance travelled will pin `transportRecoil` at its
clamp and breed a thrasher. **Score arriving somewhere, not moving.**

## The ground is not a sink

The first design was ecological: give the tail the farm gene so it dumps stock
onto the dish, let the head graze, and the worm eats at one end and excretes at
the other. It moved *nothing* — `moved` and `hops` flat zero in every
condition, which is the `ground on` row above.

Farming deposits at `a.x, a.y` and harvesting takes from the cell a body is
standing in, so a farming body is standing in its own excretion and grazes it
back the next frame. Worse, every segment is on ground of its own and tops
itself up to cap for free, so no segment can be in deficit at all.
`flowCharges` treats a flat field as a stall by design, and a pond where
everyone is standing on food is a flat field.

**A metabolic gradient needs the sink somewhere the source is not, and the
ground is everywhere.** Options, none tried yet: make the farm deposit
downstream of the body rather than under it; make the head the only segment
that can harvest; or move the source off the ground entirely.

The bench therefore uses declared puppetry — no ground, a feeder pinning the
head at its cap, upkeep as the sink — so the measurement is about the motor and
not about the economy. The `no feeder` row shows the economy still bites: with
no supply the worm starves from the tail and loses segments (`intact` false).

## The ground can drive it, once uptake is a rate

Re-measured on `pond-and-chemistry`. Everything above reproduces: the wasm
solver is now on in the experiments project (it was silently falling through to
the JS twin), which moves the pump from 34.7 to 36.2 px/s and leaves every
conclusion where it was. `thrust 0` still moves the same 263.6 over the same
11,186 hops and still goes nowhere.

The flat-field finding was downstream of one line — the old harvest took
whatever fitted in the tank, instantly. Phase 1's Monod uptake makes income a
*rate*, and the question becomes askable. Three ways to ask it, three answers,
all on real ground with no puppet feeder:

| condition | px/s | headward | straight | moved | hops |
|---|---:|---:|---:|---:|---:|
| still, uniform, vmax 0.25 | 2.4 | 0.32 | 0.40 | 85.6 | 3129 |
| still, uniform, vmax 4 | 8.2 | 0.97 | 0.99 | 34.0 | 122 |
| kicked, vmax 4 | 13.0 | 1.00 | 1.00 | 27.3 | 145 |
| **one mouth, vmax 4** | **28.7** | **0.98** | **0.98** | 246.4 | 5124 |

It self-starts. And the coherence runs *backwards* to the flow: `vmax` 0.25
moves nearly three times the energy over twenty-five times the transfers and
manages a third of the speed at `headward` 0.32, because a worm where every
segment is hungry pumps in every direction at once. A worm where only the
trailing segment occasionally falls behind pumps one way, rarely, and gets
somewhere. **Coherence comes from the scarcity of transfers, not their
abundance.**

The mouth row is the one that matters. One segment expressing the ground's
uptake row — which by the simplex silences the other seven on everybody else —
is §5's obligate trophic dependency, built rather than dialled, and it reaches
four fifths of what the puppet feeder bought from a worm that feeds itself.

## A clock the net keeps, and the wave that would not travel

The pump above is still the net *conducting* a gradient something else made. A
muscle is the other thing: the net decides when to spend, on its own schedule.

The machinery for that is the expression head. `Wh` as a rotation with gain
limit-cycles under `phi` (see `oscillator`), and `X` reads `h`, so a segment
can shift effort between taking energy in and putting `aux` back out on a clock
of its own. A phase offset per segment makes that a wave along the body.

Three things had to line up, and each was found by its failure:

- **A single oscillating row does nothing.** `expressVector` normalises across
  the eight, so one row alone keeps a share of 1 however hard it is driven. A
  rhythm has to move share *between* rows.
- **The body cannot be allowed to eat.** With every segment on ambient ground
  the clock ran perfectly and changed nothing: `h[2]` swung ±0.26 and the
  shares swung 0.30/0.70, while the tank sat pinned at cap, because uptake at
  `vmax * ROW_COUNT * share` is ~9.6/s against an excretion of 0.35/s. That is
  §0's finding — a body on ambient ground is a complete self-sufficient
  organism — reappearing inside the muscle. Fixed with a deliberately terrible
  affinity gene on every segment but the mouth.
- **The tank has to be brought to the clock.** Excretion is mass action on the
  tank, so it asymptotes at zero and can never by itself put a body into debt —
  and `rescueNeed`, which is what the demand field listens to, only latches
  *below* zero. A full tank drains toward zero and then waits on upkeep for a
  minute. Small tanks and a deeper debt cap put the metabolic cycle on the
  clock's timescale.

With all three, the clocked worm swims: **20.1 px/s, straightness 0.93,
`headward` 0.98**, on real ground, no puppet, no shove, its spending scheduled
by a rhythm the net generates. That is the internally-driven version.

**But the phase of the wave does nothing at all.**

| mouth at the nose, recoil 25 | px/s | along | straight | lag |
|---|---:|---:|---:|---:|
| phase 0 (every segment in sync) | 19.5 | 1068 | 0.93 | 0.14 |
| phase +pi/2 | 20.1 | 1000 | 0.88 | 1.09 |
| phase -pi/2 | 20.1 | 1112 | 0.93 | -1.14 |

Reversing the wave was supposed to reverse the swim. It does not move it. And
the control that isolates the question — the mouth moved amidships, so the
standing gradient is symmetric and cancels and the wave is the only thing left
that could decide a direction — goes nowhere in every phase condition:
`headward` 0.02 to 0.05, `along` and `perp` identical across all three.

### Why, and it is structural

`spreadRequests` is a **max-relaxation**: every body ends up holding the
largest need it can see, attenuated per hop. That is a potential with a single
global maximum, and a potential cannot carry a phase. `flowCharges` then moves
energy one hop up it per frame, toward whoever is neediest right now. So a
travelling wave of demand is flattened, on the frame it is created, into
whichever segment happens to be neediest — and the measured `lag` between
adjacent tanks is noise (0.14, 1.09, -1.14) even though the clocks driving them
are exactly phase-locked.

**The transport layer is a diffusive rescue network, not an actuator bus.** It
answers "who is worst off and how far away", which is the right question for
keeping a net alive and the wrong one for timing a stroke. Direction of travel
is therefore set by *where the source is*, not by anything the net computes: a
mouth at the nose swims nose-first, a mouth amidships swims nowhere.

A peristaltic muscle whose direction is a phenotype needs directed transfer —
some term that moves energy along a chosen wire because a body decided to,
rather than because the far end is hungrier. That does not exist, and it is a
real addition rather than a dial. Worth weighing against what it buys: the
rescue semantics are load-bearing everywhere else.

## What this does not yet show

- **A steerable worm.** `headward` is high because the body only ever goes
  where its nose points, but nothing chooses where the nose points.
- **Robustness.** Several of the strong clocked conditions lose segments
  (`intact` false): one mouth cannot supply a long obligate chain, and a
  segment that starves takes its wire with it.
- **Anything evolutionary.** No reproduction: a Con chain with no active pair
  cannot rewrite, which is what makes it a stable body and also a sterile one.
  And the plan's own §3 argument says the polarity above is not reachable by
  selection at `rowCost = 0, hillN = 1`: with a linear budget and concave
  payoffs the optimum is interior and every segment becomes a generalist. Those
  two dials decide whether any of this is evolvable rather than only buildable.

## Reproducing

```bash
npm run experiment -- worm
EXP_SECONDS=120 EXP_SEGMENTS=12 npm run experiment -- worm
```

`Sim.tally` gained `moved`, `hops` and `pumpImpulse` for this — energy that
crossed a wire, the number of crossings, and `sum(gain * amount)`, which is
what `applyTransportRecoil` hands out. Note `moved` is unit-*hops*: a charge
that travels five wires counts five times, which is the right currency for a
pump and the wrong one for consumption.
