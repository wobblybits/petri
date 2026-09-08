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

## What this does not yet show

- **A rhythm.** The stroke is continuous, not cyclic. The demand field is a
  standing gradient here; an oscillating one needs `h` driving taste or the
  farm gene, which means writing the state matrices rather than only the bases.
- **A steerable worm.** `headward` is high because the body only ever goes
  where its nose points, but nothing chooses where the nose points.
- **Anything evolutionary.** No reproduction: a Con chain with no active pair
  cannot rewrite, which is what makes it a stable body and also a sterile one.

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
