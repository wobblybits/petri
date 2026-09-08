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
rather than because the far end is hungrier. That did not exist. It does now;
see below.

## Two operators, and what each one fixed

`params.transportSpeed` and `params.pushRate`, both shipping at zero.

**Conduction speed.** The diagnosis above was half right. Energy was never
instantaneous — `flowCharges` moves one hop a frame — but the *signal* was:
`spreadRequests` relaxes to convergence every frame from a field that has just
been overwritten, so it has no memory and no delay. `requestDecay` was already
the space constant of that cable; what was missing was the time constant. With
a speed, the relayed part of the field lags toward its target at `conductSpeed`
hops a second, so need takes `hops / speed` to arrive, path length decides
timing, and — the half that matters — it *recedes* at the same rate, so a pulse
has a falling edge. A body's own claim stays instant, because a cell knows its
own state now and hears about its neighbours' late. `conductSpeed` is a
heritable trait beside `requestDecay`, not a head: conduction velocity is a
property of tissue, not a mood. Fast-and-far against slow-and-local is then an
axis a net can differentiate along.

**Directed push.** A `PUSH` head with one row per port, so a body pumps matter
out of a port it chose, into a neighbour that may be perfectly comfortable.
Per *port* rather than per neighbour is the trick: a body cannot name who it is
wired to, but it does have a fixed anatomy — a principal points along the
heading, an aux against it — so the same gene is opposite thrusts in the body's
own frame. Bounded by the donor's spare and the receiver's room, so it is
conserved and cannot force-feed. The recoil is the one every transfer already
earned.

Measured with the economy taken out from under it — every tank pinned at half
cap, so nobody claims, `flowCharges` has nothing to do, and every transfer in
the run is a push:

| condition | along | headward | straight | moved | hops |
|---|---:|---:|---:|---:|---:|
| push from an aux | **+212** | **+1.00** | 1.00 | 40.5 | 16194 |
| push from the nose | **-245** | **-1.00** | 1.00 | 47.9 | 18893 |
| push off | +63 | 0.75 | 0.97 | **0.0** | **0** |

**One gene reverses the swim.** `headward` goes from +1.00 to -1.00 on nothing
but which port the segment pumps out of, at straightness 1.00 over five seeds.
Direction is a phenotype now, and it is anatomy times expression rather than a
consequence of where the food happens to be.

Two bounds worth knowing, both found by tripping over them. Push is capped by
the receiver's room, so a pond of *full* bodies cannot push at all — the first
version of this measurement read zero transfers for exactly that reason. And it
is capped by the donor's spare, so a starving segment cannot pump: in the
obligate mid-mouth configuration the worm dies before the actuator can show
anything (`intact` false, impulse 4k against 26k in the fed runs). A muscle
needs a full tank and somewhere to put what it moves.

## And phase still does not matter — for a better reason

With the actuator working, the phase sweep is flat again: 210.9 / 213.3 / 212.4
px/s for phase 0, +pi/2 and -pi/2, and a measured `lag` of 0.00.

This time the reason is not the demand field. It is the medium.
`dampVelocities` is isotropic and linear in the world frame, so a thruster's
contribution to displacement depends on how hard it pushed and which way, and
not at all on *when*. A set of co-directed thrusters therefore sums to the same
displacement however their firing is staggered — a phase gradient redistributes
the impulse in time and time-averages away.

For phase to buy anything, one of two things has to be true. Either thrust
direction varies along the body, so that a wave is a wave of *direction* rather
than of timing — reachable today, by giving alternate segments different `PUSH`
rows. Or the medium has to care when: anisotropic drag, tangential against
normal, which is what makes a real body wave propel and which this sim does not
have. That was flagged at the very start of this work as the reason sine-wave
swimming was out of reach, and it has now been arrived at from the opposite
direction: with a perfect internal clock, a working directed actuator and a
phase gradient, phase is *still* inert, because the water does not notice.

That makes anisotropic drag the single remaining ingredient for gait, and it is
a few lines in `dampVelocities` and its wasm twin — project the velocity onto
the heading and damp the two components differently.

## Anisotropic drag, a recoil that turns the body, and still no gait

Three more things, all shipping neutral, all tested.

**`dragAniso`** is normal drag over tangential, 1 being the isotropic medium
this sim has always had. `drag` stays the tangential coefficient, so the
neutral value is the old behaviour exactly; the isotropic path is kept as its
own loop rather than folded in at a ratio of one.

**`recoilLever`** fixes something that was simply wrong. Every other impulse in
the sim lands where it acts — `chain.ts`'s `applyImpulse` takes an attachment
and adds the angular term, which is what makes a rope on an off-centre port
torque its body rather than only drag it. Transport recoil did not, so pumping
matter out of a port shoved the body and never turned it, whatever the port's
offset. An aux stem sits about nine units off the centreline and a principal
sits on it, so at lever 1 pushing out of an aux thrusts *and* bends, and
pushing out of the principal only thrusts. Applied to pushes only: they are the
transfers that name a port, while `flowCharges` works over an adjacency that
does not carry which ports a wire joined.

**An alternating spine.** `p(i) -> l(i+1)` on even segments and `-> r(i+1)` on
odd, so consecutive backward joints sit on opposite sides of the body. One push
gene is then two opposite bends — a muscle running down alternate sides, and
the only body plan here on which a travelling phase could become a travelling
bend.

With all three, and a stiffness sweep on the backbone besides, **phase still
does nothing**:

| portStiff | phase 0 | +pi/2 | -pi/2 | straight | bend |
|---|---:|---:|---:|---:|---:|
| 2 | 7.00 | 4.64 | 5.34 | 0.26-0.36 | 6.2 |
| 0.25 | 6.52 | 6.85 | 9.54 | 0.38-0.52 | 6.2 |
| 0.05 | 8.64 | 5.54 | 6.88 | 0.37-0.51 | 6.0 |

The ordering is different at every stiffness, which is what noise looks like.
And at the gentler drive that keeps the worm straight (`pushRate` 1.5, recoil
25) the bend swing is 2.3 to 2.9 rad, straightness 0.98, and the phase rows are
within 4% of each other — with the lever off giving 3.74 px/s against 3.73 with
it on, so the bend actuator was not reaching the body at all.

### The diagnosis: the chain has no bending stiffness

There is no drive at which a controlled curvature wave exists. Gentle, and the
body stays straight and the actuator is swallowed. Hard, and it coils — bend
swing 6.2 rad is gross reconfiguration, not a stroke. Nothing in between.

The reason is that **a joint between two bodies has no rest angle and no
bending stiffness**. What holds the chain's shape is `portTorques`, a
critically-damped servo at `portStiff * 320` aiming each port at its
neighbour's stem. That is an aiming controller, not a beam: it has no preferred
*relative* angle between two bodies and nothing that resists a joint hinging,
only something that rotates each body to face the other. `COMPLIANCE.bend`
exists but it is on the rope's own nodes — the shape of the wire between two
bodies — not on the pair.

So the servo is either stiff enough to reject the actuator or soft enough that
the chain has no shape, and there is no compliance regime between the two. A
real undulatory swimmer has a passive elastic backbone with a straight rest
configuration and a defined bending stiffness, which is exactly what turns a
periodic actuator into bounded curvature.

**The next ingredient is a joint with a rest angle**, not more actuator: an
angular spring between consecutive bodies preferring some relative angle, with
a stiffness. And once that exists the obvious bending actuator is to drive the
*rest angle* off `h` rather than to push energy around — which is what a muscle
actually is, a thing that changes a rest length, and would make the whole
`PUSH`-as-muscle route a jet rather than a muscle after all.

## Where bending stiffness can come from: topology, up to a hard limit

A ladder instead of a new primitive — two rails cross-linked rung by rung. The
port budget works out exactly, which is the first sign it is the right shape: a
Con has three ports, a rail uses `p` forward and `l` backward, and that leaves
`r`, one per body, exactly one rung's worth.

Cantilever test in `beam.exp.ts` — pin the rear, settle, push the free end
sideways with a fixed total force, read how far it goes:

| shape | deflect @25 | @100 | @400 | recovered |
|---|---:|---:|---:|---:|
| chain | 51.2 | 184.5 | 448.9 | negative |
| chain + Era fins | — | — | 365.0 | negative |
| ladder, gap 48 | — | — | 291.7 | negative |
| ladder, gap 96 | — | — | 262.9 | negative |
| ladder, gap 144 | 14.2 | **74.9** | 218.8 | negative |
| 3 rails, gap 48 | — | — | 225.6 | negative |

**The ladder is about two and a half times stiffer than the chain** at a load
where both are still roughly linear (184.5 against 74.9, tight over three
seeds), and stiffness rises monotonically with rail separation — 291.7, 262.9,
218.8 as the gap goes 48, 96, 144. Topology pays for stiffness, no new force
required.

### But it is resistance, not elasticity, and that is a theorem

`recovered` is negative for every shape at every load: released, the structure
keeps drifting rather than springing back. It resists while loaded and then
stays where it was put.

The reason is that a wire between two ports is a *distance* constraint, so as a
pin-jointed structure a ladder cell is a quadrilateral with four pin joints —
a mechanism with one degree of freedom, which shears for free. It bends without
stretching anything, so nothing stores strain energy and nothing pushes back.

Bracing cannot fix it, and this is the part worth writing down. Generic
rigidity in the plane needs `|E| >= 2|V| - 3` (Laman). A Con or Dup has three
ports and an Era one, so a net has maximum degree three and therefore at most
`3|V|/2` wires. Those meet only when `3|V|/2 >= 2|V| - 3`, i.e. **`|V| <= 6`**.
Past six bodies, *no* net of interaction combinators can be a rigid
pin-jointed truss. The alphabet forbids it. Eras make it strictly worse, not
better: each adds one node and one wire, and the deficit grows by one — which
is why the Era fins bought 449 to 365 while the ladder bought 449 to 219. The
fins are mass and drag surface at a lever arm, which is a paddle and a useful
thing, but they are not stiffness.

So the ladder's advantage is not truss action. It is that `portTorques` — the
angular servo that is the only *angular* constraint in the system — gets a
longer moment arm to work against. Wider rails, more leverage, which is exactly
the monotone trend measured.

**Which settles the earlier question.** The restoring force has to be angular,
because the angular side is the only side that can supply it. A joint with a
rest angle is not one option among several; given a three-port alphabet in two
dimensions it is the only one. That is worth knowing before building it.

## Large nets have no free ports at all

Measured before building anything on them, in `ports.exp.ts`: a 400-body soup
run 90 s, components counted, free ports as `sum(ports) - 2 * wires`.

| net size | nets | mean free | free/body | wires/body |
|---:|---:|---:|---:|---:|
| 1 | 82 | 1.4 | 1.366 | 0.000 |
| 2-3 | 30 | 2.0 | 0.859 | 0.592 |
| 4-7 | 14 | 3.5 | 0.690 | 0.873 |
| 8-15 | 7 | 4.6 | 0.400 | 1.000 |
| 16-31 | 1 | 5.0 | 0.263 | 1.053 |
| 32-63 | 1 | 4.0 | 0.111 | 1.194 |
| 64+ | 2 | **1.0** | **0.007** | 1.417 |

The largest net in the pond is **186 bodies, 279 wires, zero free ports**:
`279 * 2 = 558 = 3 * 186`, every port consumed. Free ports do not scale
sublinearly with size, they scale to *nothing*, because a grown net is port
saturated.

Three consequences, only the first of which was the reason for measuring:

- **A boundary cannot be a motor.** Drag here is charged per body, so a net's
  drag goes as its size; thrust from free ports would go as an absence that
  vanishes. A 186-body organism would pay drag on 186 bodies and have nowhere
  to push from. An Era-terminated port is a different thing entirely — a
  *place*, paid for with a body, and a net grows however many of them it
  invests in. That is the plan's §5 boundary, and it is the one that scales.
- **A large net cannot latch.** Latching needs a free port at both ends, so
  past some size a net can only change by rewriting what it already has. Its
  budget of structural events is closed.
- **`IN_BOUND` stops carrying information.** It is the fraction of a body that
  is attached, and in a saturated net it is 1 for everybody. The genome input
  that was meant to let a channel mean something different once a port is
  matched is constant in exactly the organisms complex enough to need it.

### And the momentum question this forces

`pushCharges` works on *wired* ports, so unlike a free-port scheme it scales in
a saturated net. But then the honest question cannot be dodged: an internal
transfer must not create momentum. From `applyTransportRecoil`, A's momentum
changes by `-n*p` and B's by `+n*p*catches`, so the pair's total is
`n*p*(catches - 1)` — zero **iff** `transportThrust` is 0.

So the 36 px/s worm at the top of this document was swimming on `thrust = 1`,
which is a momentum pump, and the `thrust 0` control that reads as a clean
falsification is in fact the physically correct configuration going nowhere.
The measurements stand; what they measured was a non-conservative force.

That leaves two honest thrusters. **Ejection into the field** — excretion
recoil, through Eras as nozzles, already costed and already inside the
expression simplex. And **shape change against anisotropic drag** — internal
forces cannot move a centre of mass but they can change a shape, and an
anisotropic medium turns shape change into displacement, which is how real
undulatory swimming works. Under that split `PUSH` is not a thruster at all but
an internal bending actuator, which makes `recoilLever` load-bearing and
`dragAniso` the mechanism rather than a prerequisite.

### One caveat on the numbers above

`lag` reads 0.00 throughout this section and means nothing there: it is
measured off the tank traces, and the `feedAll` conditions pin every tank each
frame so there is no trace to correlate. It is live in the metabolic
conditions, where it was noise for the separate reason recorded above.

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
