# Plan: what to take from the MKA sketch

Read 2026-09-15 against `docs/scratch.txt`, `docs/scratch.py`, `concepts.md`,
`energy-chemistry-plan.md`, `plasticity-plan.md`, `Sim.advanceGait`,
`Graph.strokeOf`, `Sim.updateState` and `seedChem`. Same shape as the other
plans: the rule, what changes, the order of work, and what to look for.

The sketch is five layers: a random directed graph of three node kinds, a
four-chemical reactor per node, threshold-gated one-way transmission along
edges keyed by kind, edge stiffness driven by the activator, and a gate that
learns from the fuel that arrives after it fires. Four of the five the pond
already has in a better form. The one it does not have is the one that
matters: **the coupling between bodies' pathways is directed and signed, and
that is what turns a chain of oscillators into a wave.**

---

## 0. The finding

`advanceGait` couples pathways by diffusing substrate symmetrically along
every wire (`metabolicDiffuse`), and the commit that shipped it says "a
reaction that diffuses carries a front" and "nothing was measured in a pond".
The first claim is true of an excitable medium with a refractory tail. It is
not true of identical relaxation oscillators under symmetric diffusive
coupling, whose attractor is in-phase synchrony — the pond-wide pulse
`concepts.md` says the locomotion branch began by removing. The old
`gaitCouple` plus `gaitLag` produced a travelling wave because the lag was
*set*; the metabolism removed the lag and put nothing in its place.

The sketch's answer is the right one and it costs no new species: the
coupling is one-way and has a sign. A Con passes activator forward, a Dup
passes inhibitor, an Era only injects. With symmetric coupling every wire is
the same wire; with a sign per body a chain is a sequence of pushes and
brakes, and a sequence of pushes and brakes has a direction.

Everything else in the sketch is either already here, already tried and
withdrawn with a measurement, or against a standing rule.

---

## 1. What the sketch has, heading by heading

Per `experiments.md` rule H: the heading, the mechanic it already has, and
whether the sketch changes anything.

| sketch | heading | mechanic here | verdict |
|---|---|---|---|
| §2 Era deg 1, Con/Dup ≤3, directed edges | Net: Identity, Shape | interaction combinators; ports `p`,`l`,`r` | **have it, and better**: direction is structural (principal vs auxiliary), not random |
| §3 four-species reactor, A→B→C→D | Locomotion: the pathway | `sub`, `atp`/`adp`, `advanceGait` | **have it**; the inhibitor `D` is a sign on the coupling, not a fourth pool (§2 below) |
| §3 Michaelis-Menten on the autocatalysis | Locomotion | `sub * (base + adp²)` | **reject, measured**: `params.ts` `metabolicRegen`: "saturating the burn widens nothing, it removes the oscillation entirely" |
| §3 decay on all four species | Principles | nothing decays | **reject** |
| §4 one-way, gated, kind-keyed transmission | Locomotion: coupling | `metabolicDiffuse`, symmetric | **adopt** — §2 |
| §4 Heaviside gate | Locomotion: coupling | — | adopt the *rectifier* the python actually uses, `max(0, x − gate)`; a step rings under Euler and has nothing for learning to move along |
| §5 stiffness ∝ activator | Locomotion: the stroke | rest length ∝ wave, `anchor` ∝ wave | **reject, measured**: a span correction split by inverse mass moves both bodies and not their centre (0.000 px in twenty seconds); stiffness would move even less. The python's own spring sum is exactly zero over the organism, so it cannot translate at all |
| §5 `max(C_i, C_j)` per edge | Locomotion: the stroke | `strokeOf` mean of ends | keep the mean; chosen so a phase difference shortens the stroke instead of tearing it |
| §6 Era near food injects fuel | Metabolism; Immigration | every body samples the water; `yDirect`, `yEra` at 1; Era seeded to make ground | **have the dial**; obligate dependency is `yDirect = 0`, and the larval-window check (`energy-chemistry-plan.md` §5) comes before moving it |
| §6 Era cannot receive | Locomotion: coupling | Era has one port | falls out of §2 for free: a leaf that transmits out of its principal is an injector by construction |
| §7.1 gate learns from `C · ΔA` | Learning | three-factor Hebbian with TD critic on `Wx`,`Wh`,`Wn`,`b`; `learnRate` 0.02 | **adopt the target, not the rule**: put the gate on a head off `h` and the existing rule reaches it (§3 below) |
| §7.1 `η (G₀ − G)` decay to baseline | Learning | nothing decays; clamps bound the genes | **reject** |
| §7.1 boredom: rate inflates when variance is low | Learning | per-body rate genes reserved, seeded 0 (`plasticity-plan.md` §6) | **reject**: a second clock, a window statistic, and a reset in disguise — it erases what a quiet region learned. The reserved genes are the door evolution opens if a faster rate pays |
| §7.2 transmission ∝ `A/(1+A)` | Metabolism | supply is bought with `extra` at `metabolicCost`; a poor body cannot buy | **have it** |
| §7.2 starvation → `T → 0` | Death | debt floor; death frees terminals | **have it** |
| §8 global vs per-agent split | Principles | "every rate is still a global constant" (concepts, Locomotion) | **adopt the split** as the answer to that open item — §4 |

---

## 2. The coupling: directed, signed, gated, and it lives on the wire

Heading: **Locomotion**. Mechanic: the pathway's coupling along wires, which
is `metabolicDiffuse` today. This *replaces* the symmetric loop in
`advanceGait`; it does not sit beside it.

Per body, two numbers:

    send    signed: what this body does to the far end while it fires
    gate    the wave level it has to be below to count as firing

Per wire, with `w` the sender's wave:

    drive  = max(0, gate_sender − w)
    flux   = metabolicDiffuse · send_sender · drive · dt      (in ATP)

    send > 0   ATP moves receiver → sender  (excite: a firing Con draws its
               neighbour's charge into its burst, and the neighbour fires)
    send < 0   ATP moves sender → receiver  (inhibit: a firing Dup gives its
               neighbour charge, and the neighbour has nothing to fire with)

**The currency is charge, not fuel, and it moves while the sender fires.**
The sketch transmits the *catalyst*, and in this pathway the autocatalyst is
ADP — the burn runs on `adp²` — so what sets a neighbour off is discharge.
The first form built here moved substrate out of *charged* bodies, and
looked at on a chain of thirty Cons it kept every interior body charged and
silent while the fuel drained to the end: a pipe, not a wave, because a
charged body is exactly one that has not burned and giving its fuel away
keeps it that way. Both directions conserve ATP between pools; each body's
`atp + adp` is still its own `adenylate`, and nothing is minted. No fourth
pool, no decay.

**The wire is the bucket.** The pass is bipartite: a wire pass computes each
wire's flux once and writes it on the wire, and a body pass reads the three
wires on its ports and applies them. Two things follow that the body-side
form cannot give. The two ends of a wire read *one* number, so a transfer is
antisymmetric by construction rather than by two calculations agreeing to
the bit — which is what `far.wgsl`'s span pass has to hope for, computing
every constraint twice. And it is order-independent without an accumulate
buffer, because nothing is written to a body until every wire has spoken.
The GPU gains nothing from it *yet* — ports cap degree at three, so the
gather is already bounded — but the shape is the one a device pass wants.

A wire is a rope with a rest length, so a bucket with capacity would give a
lag per hop that is the net's own *geometry*, which is what `gaitLag` set by
fiat and the metabolism removed without replacing. `concepts.md`'s Shape
heading already asks for the same float from the other side: "tension is
held by energy flowing along a wire, so a net that moves no energy is slack.
Not yet true." The staging value is the first step toward it; the reservoir
is the second, and it waits on step 0's look.

Each wire's flux is bounded by a quarter of what the source holds and a
quarter of the room at the sink: a body's ATP can be drawn on by at most four
transfers a frame (its own push, and a pull from the far end of each of its
three wires), so nothing can overdraw it and the pass conserves ATP exactly,
with no clamp at the apply.

**Which end sends.** The wire already knows: each end is a port, and the
port is `p` or auxiliary. A body transmits out of its **principal** port
only. Then every body has out-degree exactly one, a Con or Dup receives on up
to two auxiliaries, and an Era, which has nothing but a principal, is the
sketch's §6.1 injector without a rule saying so. A wire between two
principals — a redex — carries a transfer each way, each under its own
sender's sign and gate. This is the one decision in the plan the author
should look at, because it fixes which way a wave runs relative to a net's
tree — principal-out runs *toward the leaves* — and which of those is a
worm's head is a question about what a grown chain looks like.

**Seeding, not a kind rule.** `send` seeds +1 on a Con, −1 on a Dup, +1 on
an Era — so a wave starts at leaves and runs toward the root; `gate` seeds at
`metabolicGate` for all, 0 being the discharged half of the cycle. Heritable, mutated and
clamped like every other gene, so "Con excites, Dup inhibits" is where a
fresh pond starts and not what it is held to — the same discipline
`seedProduction` follows for what a kind produces.

**Where the genes live.** `concepts.md` already says "that wants two more
rows on `G`". They go in as *new segments*, `Gc` and `gc0`, rather than by
widening `G`: `net-blob.ts` refuses a net whose segment changed length, and
`nets/deep-87.petrinet` and `nets/mixed-308.petrinet` keep loading. A new
segment is seeded for the body's kind on load, which is the migration path
the header exists for.

**Dials.** `metabolicDiffuse` keeps its name and becomes the transmission
rate. One new global, `metabolicGate`, is the seed value of `gate`. Nothing
else.

---

## 3. The gate learns, through the door that is already open

Heading: **Learning**. Mechanic: the three-factor rule on the state matrices.
No second rule.

The sketch's update is `dG/dt = −μ · C · ΔA`: eligibility is activity,
teacher is the fuel that arrived afterwards. The rule here is the same shape
with a critic where the sketch has a raw difference — `r = FULL − 1`, TD
error, trace on `phi'(v) · pre`. What differs is only *what* it reaches. The
sketch learns the gate directly; here the output heads are not learned
("a later door", `plasticity-plan.md` §2). The door does not need opening:
if `gate` and `send` are rows off `h`, then every learned change to `Wx`,
`Wh`, `Wn` moves them, with a credit assignment that is already defensible.
The sketch's `C` is `h`'s activity and the sketch's `ΔA` is the critic's
surprise.

So §2's rows are a head, `Gc`, read exactly as `G` is read — `gc0 + Gc · h` —
and step 2 below is what makes the gate plastic: nothing but the layout.

What this does *not* give: the sketch's gate learns which *wire* paid, and a
head off `h` learns which *state* paid. That is coarser. If it turns out too
coarse the honest next step is per-head learning with random feedback, which
is the door the plan names, and it stays shut until looking says otherwise.

---

## 4. The global/per-agent split, taken as the answer

`concepts.md`, Locomotion: "every rate is still a global constant, where a
lineage should own them — that is what `X` exists for and is read by
nothing." The sketch's §1/§2 is an argument for *not* doing that, and it is
right: `metabolicRate`, `metabolicBase`, `metabolicRegen`, `metabolicSupply`
set the time base of the wave, and a lineage that changes them locally
changes what a wavelength means across a net. What a body should own is what
it *does* with the shared clock — its pool (`adenylate`, already heritable),
its sign and its threshold (§2). Take that as closing the item rather than
opening `X` for every rate.

---

## 5. Order of work, and what to look for

Per `CLAUDE.md`: on, visible, behind a slider, and the author looks. The
pathway ships at `metabolicRate` 0, so everything under it is free to change
before anything is seen; but this plan is not done until the rate is *on* at
a value the eye can see, because that is the rule.

**Step 0. Look first, no code.** Load `nets/deep-87.petrinet`, set
`metabolicRate` to about 15 and `metabolicDiffuse` at its 2. Watch the
chain for a minute. If it runs a wave down its length the finding in §0 is
wrong and steps 1–2 are still worth having for the sign, but the urgency is
gone. If it pulses in unison, or goes still, §0 stands. Either way this is
the observation the shipping commit did not make.

**Step 1. The coupling.** `advanceGait`: replace the symmetric accumulate
with the wire pass and body pass of §2. `Wire` gains `flux`; the store gains
`gaitSend` and `gaitGate`. Two dials. The GPU is not involved: this pass is
host-side and walks the wire map already. Look for: on the same chain, a
wave that runs one way; a Dup between two Cons as the place the wave does
not cross back; an Era at the end lit first or last, never mid-chain.

**Step 2. The head.** `Gc`/`gc0` segments in `chem-layout.ts`, seeded in
`seedGait`, read where `gaitAnchor` is read in `updateState` and in
`genome.wgsl` (one more head, no new binding). `layoutSelfCheck` and the
layout tripwire in `genome-kernel.test.ts` follow. Both stored nets must
load. Look for: with `learnRate` at its 0.02 and a fed chain on patchy
ground, drain `plasticAll` and see whether the gate row's effective value
moves at all over a few minutes. If it does not, the head is too far from
the wire and §3's last paragraph is the next question.

**Step 3. The pond question.** Only now `npm run pond`: worm travel with the
pathway on against the grip-only control, engagement gauges `full_mean`
inside the oscillation window and wave amplitude non-zero, seeds per
`experiments.md` §3. This is the measurement `concepts.md` says has not
been made — which of the oar and the anchor wins on an Era when both act at
once — and it needs a wave to exist before it can be asked.

---

## 5b. What was looked at, 2026-09-15

Steps 1 and 2 are built: the wire pass, `Wire.flux`, the `Gc` head on both
genome paths, the two dials, and the tests. The look was done in the browser
pane on `?soup=0&net=deep-87`, driving frames by hand, reading each body's
charge along the longest chain (31 bodies, Era–Dup–Dup–Con×27–Era, every
link principal-to-left) as one digit per body per tenth of a second.

- **The pathway does not oscillate in the stored net as it is.** Most planted
  bodies sit pinned charged or discharged and never move, because
  `nets/*.petrinet` carry NaN and negative `adenylate` (13 NaN and 12 negative
  of 87 in deep-87; 44 and 44 of 308 in mixed-308) and a lineage's pool sets
  which side of the window it lives on. With every pool set to the seed value
  and every body fed, all bodies oscillate — in unison, because regen erases
  any charge scatter in a fifth of a second and the burst is timed by
  substrate accumulation, which is identical for identical bodies. This
  oscillator holds no phase.
- **Substrate coupling made a pipe.** See §2. Replaced with charge.
- **Charge coupling works and is visible, and what it does is entrain.** On
  a chain with substrate staggered along it, uncoupled bodies fire on their
  own schedules and keep the stagger every cycle. Coupled at 8, the first
  body to fire pulls its downstream neighbour over within a frame or two,
  the cascade runs the whole chain in about 0.4 s, and from the next cycle
  on the chain fires as one. The propagation speed is the pass's one-frame
  staging delay, thirty hops in under half a second against a period of
  three seconds. That is a wave, and it is far too fast to be a gait.
- **So the reservoir is the next step, not an option.** A wire that holds
  charge in transit for a time set by its rest length would make the cascade
  cross a hop in a time the net's own geometry sets, which is a travelling
  wave with a wavelength — the thing `gaitLag` used to set by fiat. The
  refractory tail already exists: a body that has just fired has burned its
  substrate and cannot fire again until supply refills it, so a wave cannot
  run backward through it.
- **Not this change, but found by it.** In the pane the GPU field pass
  errors (`used in submit while pending map`) on this machine and leaves
  57 of 82 planted bodies with NaN state before the sim drops to the CPU;
  NaN `h` never heals and pins every head at its clamp floor. Confirmed on
  the untouched tree; `field-device.test.ts` fails two tests there the same
  day. Headless on the CPU path the same net keeps finite state. Every
  observation above was made on the CPU path.

What ships: `metabolicRate` is still 0. The plan says it is not done until
the rate is on at a value the eye can see, and the honest state is that with
the rate on, a planted net today mostly does not oscillate for reasons
upstream of this change — the pools in the files and the window — and a
chain that does oscillate entrains rather than walks. Turning it on is the
author's call once the reservoir exists; the sliders are there.

## 6. What the python gets wrong

Listed because the author said it was not quite right, and because two of
these are the project's own findings arriving by a different door.

- **It cannot move.** Every spring force is equal and opposite and applied
  with the same mobility, so the sum over the organism is identically zero
  and the centre of mass never translates. This is exactly why `grip` and
  `anchor` exist here: travel comes from the two ends of a wire damping
  differently, and the sketch has no such term.
- **Transmission mints.** Flux is added to the receiver and never subtracted
  from the sender, for all three kinds. The Era injection is unconditional
  and unbounded.
- **The stoichiometry is not what the prose says.** `S.T @ rates` sums by
  reaction, not by species; it should be `S @ rates`. And as written, `r2`
  makes `C` from nothing and `r4` destroys `B` into nothing, so the
  "conserved mass-action" reactor conserves nothing.
- **Dup never learns.** The update touches gate index 2 (`C`) for both
  kinds, but a Dup gates on index 3 (`D`).
- **Boredom is a reset.** Learning rate up to 0.5 per step on a gate clamped
  to [0.05, 0.95] randomises the gate.
- No selection, no death, one organism, fixed N — none of which it claims,
  but the framework text does.

---

## 7. What this deliberately does not do

- No fourth chemical. The inhibitor is a sign.
- No reservoir on the wire yet. The bucket is a staging value until step 0
  says a geometric lag is what is missing.
- No second learning rule, no decay to baseline, no meta-learning rate.
- No per-lineage reaction constants; the time base stays global (§4).
- No Heaviside; a rectifier above a gate.
- No change to the stroke: rest length and anchor stay the actuators.
- No move on `yDirect` until the larval window is measured.
