> **Not maintained.** A dated record of what was argued and measured at the time. The code and `docs/concepts.md` are the source of truth; where this disagrees with them, this is the one that is stale.

# Audit: evolution, internal state, and signalling

Audited 2026-09-06 at `e0d2dd0` on `alife-energy`, covering the architecture
built in `c862444`..`e0d2dd0` — the recurrent `h`, the output heads, and the
emit/taste genome.

This file exists so the findings can be acted on without the session that
produced them. Every claim below was measured; the repro is given with each so
you can re-check rather than trust it. Findings are ranked by consequence, not
by effort.

> ## Status, added 2026-09-06 after acting on it
>
> Worked through in `acba212`..`3ed9894`. The findings below are left as
> written — they were accurate — but most describe code that has changed, so
> read this first.
>
> | | | |
> |---|---|---|
> | **F1** emit budget unenforced | **fixed** | `947b15f`. Verified the repro first: realised sum 17 against a budget of 1. `emitVector` normalises the realised vector; the test now asserts the output sum, not the genome's. |
> | **F2** two architectures documented | **fixed** | `6f759d3`. `chemState`, `NEED`, `HERE` deleted; the block now describes the input vector. The taste→trail→state loop stays gone — noted as an eighth input if wanted back. |
> | **F3** "every matrix seeds to zero" | **fixed** | `6f759d3`. Corrected in all three places; `h[0]` documented as spoken for. |
> | **F4** input vector ill-conditioned | **fixed** | `6f759d3`. `SENSE_SCALE = 4.3`, from a measured p90 that is flat (4.16/4.24/4.53) across 60/400/2000 bodies. |
> | **F5** ground emit is genetic load | **open, by choice** | Real, and the fix is a balance decision — it stops being load the moment `farmRate` is non-zero. |
> | **F6** emit/taste frame asymmetry | **resolved incidentally** | `947b15f`. Both now read this frame's `h`. Moves `state-hash`. |
> | **F7** `cloneAgent` drops `h`/`sense` | **fixed** | `6f759d3`. |
> | **F8** dead attract sparams | **fixed** | `1f2e6cc`. Slots reserved, not renumbered. |
> | **F9** stale comments | **fixed** | `6f759d3`, `1f2e6cc`. |
> | **P2.1** cache the sense bit | **done** | `6f759d3`, 4.1 ms as estimated. The cache has a sharp edge: anything writing `chem` outside birth must call `refreshReadsField`. |
> | **P2.2** evaluate heads once | **done, worth less than estimated** | `947b15f`. Consumers −14.5 ms, `state` +11 — the work relocated. `effEmit(a, c)` only ever computed row `c`, so four calls were four rows, not four walks; the real redundancy was `scentAt` rebuilding taste for each of three samples. |
> | **P1/P2.3** move the sweep to wasm | **superseded** | `1cdf711`. The diagnosis (compute-bound) was right, the cause was not: it was **loop overhead**, not arithmetic. `STATE_DIMS`=4 and `IN_DIMS`=7 are compile-time constants. Unrolled: `state` 41.4 → 15.6 ms, frame 168.5 → 128.4. That is 25.8 ms against P2.3's ~25 ms estimate, with no second implementation. |
> | **P3** GPU | **still blocked, list extended** | The fourth tripwire item this audit found is now recorded at `Sim.openFieldGpu` as asked. |
>
> **Two things the audit could not have known, both measured since:**
>
> The genome would not have fitted. Static wasm use is already **64.7 MB of an
> 80 MB fixed allocation**, and `chemAll` plus `h` at `MAX_BODIES` is another
> 17.8 MB — so P2.3 meant raising the allocation for every tab, or capping the
> fast path by body count and keeping the JS version live as a second
> implementation of the architecture's central pass.
>
> And the ranking is now stale. At 20k the ledger reads `solve` 37.1, `fields`
> 21.7, `state` 15.9. `fields` is **memory-bound**, unlike `state`: hoisting a
> loop-invariant out of its inner loop moved 9.17 ms to 9.18, because it is
> already streaming ~96 MB a pass at ~10.4 GB/s. Those two passes look alike
> and are not, which is written down at `Fields.diffuse` so the next person does
> not spend an afternoon rediscovering it.

**Do not treat this file as a to-do list to work through in order.** F1 changes
what the simulation selects for and needs a judgement call about balance; the
performance items are independent of it and of each other.

---

## F1 — The emit budget is enforced on the genome, not on behaviour

**Severity: high. This one is actively selecting for the wrong thing.**

The architecture is documented in three places as

    emit = normalise(relu(E·h + e0))

`src/agents.ts:787` (`effEmit`) and `src/agents.ts:808` (`emitEnergy`) do the
relu and no normalise. The only normalisation is in `inheritChem`
(`src/rewrite.ts:1032`), which projects `e0` onto the unit simplex **at birth**.
`E` is then free to add up to `CHEM_SLOPE_MAX` per state dim on top.

A genome satisfying every invariant `inheritChem` enforces — `e0` non-negative
and summing to 1, every `E` entry within `CHEM_SLOPE_MAX` — realises:

    ch0 4.25   ch1 4.25   ground 4.25   total 17.0

against a documented budget of 1.

### Why it matters

- **"Louder is strictly better" is reachable.** Preventing that is the entire
  reason the budget exists. Selection has a free 5x per channel available
  through `E`, and nothing downstream caps it.
- **The honesty mechanism that justified deleting `emitCost` does not hold.**
  The argument was that spending voice on farming trades against being heard.
  Measured, ground emit and both signal channels rise *together* off one state
  dim: at `h = 0` each is 0.25, at `h = [1,0,0,0]` each is 1.25. No trade-off.
- **`CHEM_SLOPE_MAX`'s stated justification is the `STATE_DIMS = 1` argument.**
  Its comment says a slope of one "can silence a body or double it at full
  need, no more". True with one state dim. With four the per-channel ceiling is
  `1 + 4·1 = 5`.
- **The tests pass because they assert the invariant where it holds.**
  `src/chem-evolution.test.ts:161` and `:74` both check `chem[EMIT..EMIT+4]`,
  the genome bases. Nothing asserts anything about `effEmit`'s output.

### Repro

```ts
// A genome satisfying exactly what inheritChem guarantees.
const c = new Float32Array(CHEM_LEN);
for (let k = 0; k < 4; k++) c[EMIT + k] = 0.25;            // unit sum
for (let k = E_OUT; k < E_OUT + 4 * STATE_DIMS; k++) c[k] = CHEM_SLOPE_MAX;
const a = bareBody(c, { h: [1, 1, 1, 1] });                // phi permits this
effEmit(a, 0) + effEmit(a, 1) + effEmit(a, 3) + emitEnergy(a);  // => 17
```

### Options

1. **Normalise at read time.** Needs all four channels at once, so it means
   replacing the four independent `effEmit` calls with one
   `emitVector(a, out: Float64Array)`. That is worth doing on performance
   grounds anyway — see P2 — because the four calls walk the genome four times.
   Note `effEmit` returns 0 for `CH.energy` while `emitEnergy` returns it, so
   the normaliser must see the raw four and the *callers* keep the split.
2. **Scale the slope bound**: `CHEM_SLOPE_MAX / STATE_DIMS`. One line, keeps
   the read path as it is, but bounds the worst case at 2x rather than pinning
   the budget at 1 — a body can still get louder by saturating, just less.

(1) restores the documented invariant. (2) is a mitigation. Either way, add a
test that asserts the realised sum, not the base sum — the absence of that test
is what let this ship.

---

## F2 — `chem-layout.ts` documents two incompatible architectures

**Severity: medium. It is the design record, and it contradicts itself.**

`src/chem-layout.ts:20-96` names the four state dims (`NEED`, `FULL`, `HERE`,
`BOUND`), and argues at length that each is bounded and about a different
thing. `src/chem-layout.ts:134-136`, in the same file, says "`h`'s four
dimensions have no names, and cannot".

The named version is the pre-`00f4ea5` architecture. The function implementing
it, `chemState` (`src/agents.ts:710`), **has zero callers** — production or
test. `NEED` and `HERE` are referenced by nothing else at all.

`HERE` is the only casualty with substance. It was `soft(a.trail)`, and it
closed a loop the comment is right to call out: taste fed the trail, the trail
fed the state, the state fed taste. That loop no longer exists. `h` reads *raw*
sense (`IN_SENSE`) instead, so a body can no longer condition on its own
taste-weighted reading of where it is standing. That may well be the better
design — raw is strictly more information — but it was not a decision recorded
anywhere, it is a side effect of the input vector changing shape.

### Action

Delete `chemState`, `NEED`, `HERE`, and the parts of `bareBody`/`StateBody`
that exist only to feed it (`request`, `extra`, `energyCap`, `trail` — check
each; `bareBody`'s `h` and `chem` are still used by `chem-evolution.test.ts`).
Rewrite `src/chem-layout.ts:20-96` to describe the input vector, which is what
actually has named, bounded, differently-sourced components now. Keep `FULL`
and `BOUND` only if something still reads them as state indices — at the time
of writing nothing does; the live constants are `IN_FULL` and `IN_BOUND`.

If the taste→trail→state loop is wanted back, that is a separate decision:
it means adding `soft(trail)` as an eighth input, not resurrecting `chemState`.

---

## F3 — "Every matrix seeds to zero" is false, and stated three times

**Severity: medium. The claim is the foundation of the seeding strategy.**

`seedChem` seeds two matrix entries away from zero:

- `src/agents.ts:898` — `Wx[0][IN_DEMAND] = 1`
- `src/agents.ts:899` — `T[CH.energy][0] = params.attractFood * 2` (default
  `attractFood` is 0.9, so this is live)

Contradicted by:

- `src/chem-layout.ts:129` — "every matrix seeds to zero, so a fresh body
  computes `h = phi(0) = 0` and behaves exactly as it did before any of this
  existed"
- `src/agents.ts:858` — "Slopes start at zero ... the whole modulation is inert
  until breeding moves it"
- `src/agents.ts:888` — "The one place a matrix is seeded away from zero" —
  itself off by one

A fresh hungry body has `h[0] = phi(DEMAND) != 0`. Nothing downstream breaks,
because `E`, `F` and `P` really are zero at seed, so the only expressed effect
is the intended ground taste. But the guarantee that a newborn behaves exactly
as it did before the recurrent state existed is not true as written, and that
guarantee is what the whole "matrices drift, bases are seeded" strategy rests
on.

### Action

Correct all three comments to say what is actually true: the *output* matrices
(`E`, `T` apart from the ground column, `F`, `P`) seed to zero, and there is
one seeded two-hop pathway through `h[0]`. Note in `chem-layout.ts` that `h[0]`
is spoken for at seed, since a second seeded pathway that also picked dim 0
would silently collide with it.

---

## F4 — The input vector is badly conditioned for evolution

**Severity: medium. It costs the sense genes most of their useful range.**

`x = [ sense(4), FULL, BOUND, DEMAND ]`. Four of those seven are deliberately
scaled to roughly [0,1]: `FULL`, `BOUND` and `DEMAND` are clamped, and the
ground channel is divided by `cellCap` — `src/state-net.test.ts:58` documents
exactly why that scaling was necessary.

The three *signal* channels are passed raw. Measured in a live pond (60 bodies,
240 frames, defaults): **max raw signal reading 7.24**, max raw ground 0.0624
(so ~1.0 after `groundScale`).

Consequences:

- One `CHEM_MUTATE` step (0.0216 at `CHEM_LEN` 124) moves `v` by ~0.16 through
  a sense weight, and by 0.0216 through the `DEMAND` weight. Sense genes have
  ~7x the evolutionary leverage per step of every other input gene.
- `soft` is already at 0.5 by `Wx ~= 0.14` and effectively pinned past ~0.6, so
  roughly 7% of the gene's legal +/-4 range is informative and the rest is
  saturated. The gene goes from "deaf" to "pinned" in a few mutation steps.

This is the same defect `HEAD_SCALE` was introduced to cure on the *output*
side (see its comment in `src/chem-layout.ts:174-193`), left untreated on the
input side.

### Action

Scale the three signal channels into the same range the other four inputs
already occupy, in `updateState` where `groundScale` is already applied. Pick
the divisor the way `groundScale` was picked — from a measured typical peak,
not a guess — and state it in `chem-layout.ts` next to `IN_SENSE`. Existing
genomes' sense weights change meaning, which at present costs nothing because
`seedChem` leaves them all at zero (verified for all three kinds).

---

## F5 — The ground emit gene is genetic load at default settings

**Severity: low, but it scales with how long a pond runs.**

`inheritChem` includes `CH.energy` in the unit-sum budget, so a lineage can
park voice there. `effEmit` returns 0 for that channel and `farmRate` ships at
0, so budget parked there does nothing at all. Under drift, a quarter of the
emit budget ends up on an inert channel, diluting signalling by that much.

A seeded Era is the extreme case: `src/agents.ts:945` puts its *whole* unit
there, so a fresh Era is completely mute and completely inert until `farmRate`
is turned on.

This is a known trade — the comments say "inert until `farmRate` is turned on,
like every other economy dial" — but it is worth being explicit that the cost
is not zero while it is off.

---

## P1 — The genome costs ~26% of the frame, and it is compute-bound

**Severity: high, and the obvious fix is the wrong one.**

Measured with `src/native/frame-profile.perf.test.ts` at 20,000 bodies and
29,348 wires.

**Absolute frame numbers drift several percent between runs on this machine
(thermal). Every delta below is from an A/B pair run back to back; the totals
are from a single run and should be read as "about this".**

| phase | ms/frame | share |
|---|---:|---:|
| `solve` | 40.8 | 21.6% |
| **`state`** | **34.8** | **18.5%** |
| `fields` | 22.5 | 11.9% |
| whole frame | 188.6 | |

`state` is second only to the solve, and larger than the entire field pipeline.
The design note that specified this architecture estimated 2-3 ms.

That is not the whole cost. The `E` and `T` heads are evaluated *again*,
outside `state`, by the passes that consume them — each walking the genome per
body per frame:

| A/B (back to back) | with | without | delta |
|---|---:|---:|---:|
| `effEmit` in `scent:bodyPack` | 9.83 | 4.27 | **5.56** |
| `tasteOf` in `steer` | 16.32 | 8.10 | **8.22** |

**Total genome evaluation: ~48.6 ms/frame, ~26% of the frame.**

### Where `state` goes

Ablation, one back-to-back series (30.18 ms total on that pass):

| section | ms | share |
|---|---:|---:|
| `h` update (`Wx`/`Wh`/`Wn`/`b` + phi) | 17.5 | 58% |
| output heads (`F`, `P`) | 4.8 | 16% |
| sense gate | 4.1 | 14% |
| prologue (slot gather, `hPrev` copy, store reads) | 2.9 | 10% |
| neighbour mean | 0.8 | 3% |

### It is compute-bound, not memory-bound

This matters because the reflex in `sim.ts` — and the right answer everywhere
else in it — is to pack the data tighter. **That will do nothing here.**
Isolating the `h` kernel over synthetic arrays at n=20,000:

| variant | ms |
|---|---:|
| f32 genome, 20k distinct slots (9.9 MB) | 9.80 |
| f32 genome, every body reads slot 0 (fits L1) | 9.75 |
| f64 genome, 20k distinct slots (19.8 MB) | 9.95 |
| f32, only this pass's 64 weights packed contiguously | 9.90 |

Identical from L1 and from 9.9 MB. Shuffling slot order to fragment access
changed 14.42 ms to 14.77 ms — noise. 1.2M multiply-adds in 9.8 ms is about
122 Mflop/s: this is JS scalar float arithmetic with bounds-checked typed-array
loads, roughly two orders of magnitude off what the machine can do.

It is the only dense-linear-algebra pass in the simulation still outside wasm.

---

## P2 — Ranked performance actions

### P2.1 Cache the sense-read bit (4.1 ms, trivial, no behaviour change)

`src/sim.ts:4237` re-derives, every body every frame, whether that body's
genome reads the scent field — 16 `Float32` reads to answer it. **A genome is
fixed for a body's life, so the answer cannot change after birth.** Verified:
all three kinds seed with zero non-zero `Wx` sense entries, so for a seeded
pond the answer is always "no" and the gate is pure overhead.

Compute it once in `createAgent` and in `commitRewrite` (after `inheritChem`),
store it as a `Uint8Array` in `AgentStore`. ~14% of the phase for free.

Careful: anything that writes `chem` after birth must invalidate it. Today that
is only tests and `cloneAgent`; make the flag a function of the setter if you
want it to be safe against future callers.

### P2.2 Evaluate each head once per frame, not once per consumer

`effEmit` is called 4x per body in the scent pass (5.56 ms) and `tasteOf` 4x
per body in the steer pass (8.22 ms), each re-walking the genome. Both are pure
functions of `h`, which `updateState` has already computed this frame.

Write both vectors into store arrays inside `updateState`, next to where `F`
and `P` already are, and have the consumers read the array. This folds into F1
naturally: the normaliser needs all four emit channels at once anyway. Expect
most of ~13.8 ms, at the cost of 8 floats per body of store.

Watch the frame ordering while doing this — see F6 below. `effEmit` currently
reads *this* frame's `h` and `tasteOf` reads the previous frame's, and
materialising both in `updateState` silently makes taste one frame fresher.
That is probably an improvement, but it is a behaviour change and will move
`state-hash`.

### P2.3 Move the sweep to wasm (~25 ms)

The remaining `h` update, heads and prologue. The project's own history is the
argument: every other per-body pass moved for this reason, and `steer` does
20k bodies in ~8 ms (excluding the taste packing) including field sampling.

What makes this easier than the passes that came before it:

- **The genome is static.** Unlike `steer`'s per-frame taste packing, `chem`
  changes only at birth. Upload once, patch 496 bytes on birth. It does not
  need to cross per frame.
- **The outputs are already consumed in wasm.** `flockAlign`/`flockSep` feed
  the native flock pass (`src/sim.ts:3676`), and `transportThrust`/`Recoil`
  feed transport. If the heads are computed in wasm they never cross at all.
- **The adjacency already exists** as a CSR (`WireAdjacency`), and
  `nativeSolver` already carries per-body topology arrays.

What has to cross per frame: `extra`, `energyCap`, `request`, `bound` (x/y are
already packed). Four f64 per body.

---

## P3 — Can this go on the GPU with the field work?

**The premise needs correcting first: the field is not on the GPU.**
`Sim.openFieldGpu` (`src/sim.ts:2544`) unconditionally `return false`. It is a
deliberate tripwire, and its comment explains why — `grow` and `tuneChannels`
sit inside `if (!this.fieldOnGpu)`, and `EnergyGrid.take`/`addAt` read and
write `fields.data` directly, so turning it on as it stands silently stops the
ground regrowing and lets the pond mine a CPU array the GPU is not looking at.

**This audit adds a fourth item to that tripwire's list, which its comment does
not mention:** `updateState` samples the field on the CPU
(`src/sim.ts:4249` -> `Fields.sampleAll` -> `this.data`, `src/fields.ts:892`).
With the field on the GPU, `fields.data` is a stale copy, so any body that has
evolved a non-zero `Wx` sense weight reads garbage — silently, and only on
machines with a GPU, and only once evolution has moved a gene off its seed.
**Add this to the comment at `src/sim.ts:2528` whether or not anyone acts on
the rest of this section.** It also means the coupling runs both ways: if the
field moves to the GPU, the state pass has to follow it or lose its sense
inputs.

### Once the field is genuinely there, the fit is very good

Better than wasm, for this pass specifically:

- It is 20,000 independent 4x15 matvecs — as data-parallel as work gets.
- The neighbour mean is a CSR gather, the standard SpMV pattern.
- **The sense inputs are already resident.** Today `updateState` does a
  bilinear sample per reading body against a 16 MB CPU array. On the GPU the
  field is the thing the shader is already holding.
- **`emit` and `taste` would stop crossing the bus entirely.** Look at
  `gpuFieldStep` (`src/sim.ts:2562`): the host currently computes `effEmit` per
  body and writes it into the deposit list, and `tasteOf` per body into the
  probe list. Those are exactly the `E` and `T` heads. Computed on the GPU they
  feed the scatter and the gather in place. That is the 13.8 ms from P2.2
  disappearing rather than being reduced.
- **The genome upload is a non-issue.** 9.9 MB at 20k, but static — upload
  once, patch on birth. Births are rare (see the census note: `bornMean` 0.36
  over 45 s for ~293 bodies).
- **The readback is nearly free, and the round trip already exists.**
  `gpuFieldStep` already awaits a readback of 3 floats per body. Adding the 4
  head values is 320 KB at 20k, in the same round trip. `h` itself stays
  resident and never comes back.

### What makes it hard

- **It is blocked behind the tripwire.** Doing this means first landing `grow`
  in `field.wgsl`, getting the per-channel decay and diffuse rates into the
  shader, and deciding whether the ground reads back or moves off the shared
  field. That is real work and it is the actual prerequisite.
- **The heads must come back every frame** because the flock and transport
  forces run in wasm on the CPU. That is fine — it is one small readback in an
  existing round trip — but it does mean the state pass cannot be fully
  fire-and-forget.
- **It splits the genome across two runtimes.** Mutation and inheritance are in
  `rewrite.ts` on the CPU and must stay there; the GPU copy becomes a cache
  that birth invalidates. That is a correctness surface worth a test.
- **`Sim` would gain a second all-or-nothing GPU mode**, with the CPU path kept
  for machines without a device — so the sweep gets written twice, once in WGSL
  and once in JS or wasm, and the two have to agree. This file's F3 exists
  because a value written down twice disagreed; the same risk applies here, and
  `far-span-jacobi.test.ts` is the precedent for how to test it.

### Recommendation

Do them in this order, and stop whenever the frame is fast enough:

1. **P2.1** — 4.1 ms, an afternoon, no behaviour change.
2. **P2.2** — ~13.8 ms, and it is the natural shape for fixing F1.
3. **P2.3 (wasm)** — ~25 ms, well-trodden path in this repo.
4. **GPU** — only as part of finishing the field GPU work, not before it, and
   not as a way of avoiding (3). If the field goes to the GPU, the state pass
   must go with it; if it does not, wasm gets most of the win for a fraction of
   the risk.

---

## Correctness

Checked specifically and found sound: `hPrev` double-buffering and its
list-index vs slot-index discipline; slot recycling zeroes `hAll` and
`senseAll` so no state leaks between occupants (`AgentStore.clearSlot`);
`flockGain` is applied at all four read sites, so a negative head cannot
produce the negative damping its comment warns about; the JS `deposit` and
`solver_deposit` agree on the aux gate and on the principal laying all four
channels; `updateState` covers the whole roster; `refreshBound` is keyed on
graph and roster version and cannot be skipped by an absent native solver.

### F6 — Emit reads this frame's `h`, taste reads last frame's (not a bug)

`updateState` runs in `endFrame`. `scentWrite` (`effEmit`) and farming
(`emitEnergy`) run after it in the same frame, so they see this frame's `h`.
`steer` (`effTaste`) and `flock` run in the *next* frame's `beginFrame`, so
they see the previous. A frame of latency in smell is documented; this
asymmetry between a body's voice and its taste is not, and it matters to anyone
reasoning about a feedback loop between them. Write it down at `updateState`.
See the warning in P2.2 — the obvious refactor changes it.

### F7 — `cloneAgent` silently drops `h` and `sense`

`src/agents.ts:498` copies every other store field, including `bound` and
`trail`. Only native-parity tests call it and they do not touch `h`, so this is
latent rather than live — but it is a clone that does not clone, and the next
caller (a designer ghost, a rollback) gets a zeroed internal state with no
error. Two lines to fix.

### F8 — Dead cross-wall wiring for the attract seeds

`src/sim.ts:3428-3429` packs `params.attractStrong`/`attractMedium` into
`sp[12]`/`sp[13]` every frame. `native/solver.c:1444-1445` `#define`s
`SP_ATTRACT_STRONG` and `SP_ATTRACT_MEDIUM` and reads neither. Harmless, but it
contradicts "read once by `seedChem` and never again" at a glance.

### F9 — Stale comments over live code

- `src/agents.ts:863-885` — a block describing the food seed as "on the slope
  against `request`, not on the base" sits directly above the block that
  supersedes it. There is no such slope; the pathway is `Wx` then `T`.
- `src/agents.ts:915-932` — "An Era says nothing at seed" sits directly above
  "An Era's one unit of voice goes into the ground", over
  `c[EMIT + CH.energy] = 1`.
- `src/agent-store.ts:64` — "slot `i`'s 32 floats", "16 floats per agent".
  It is 124 (`CHEM_LEN`).
- `src/agent-store.ts:104` — `bound` is cached "because `chemState` reads it
  per channel per body". `chemState` is dead (F2) and `updateState` reads it
  once per body.
- `src/agents.ts:314` and `src/agent-store.ts:89` — "thirty-two genes". The
  `CHEM_MUTATE` formula is right; the prose is two architectures out of date.
- `src/params.ts:463` — "These four are read once by `seedChem`". Three follow.

In a codebase where the comments are the design record, a contradicting pair is
worse than no comment.

---

## How to re-measure

```bash
npx vitest run --project bench --disableConsoleIntercept src/native/frame-profile.perf.test.ts
```

Prints the per-phase ledger for a 20k-body frame. Two things that cost real
time if you do not know them:

- `console.log` in a test is swallowed by the default reporter;
  `--disableConsoleIntercept` is what shows it.
- The `bench` project sets `fileParallelism: false` because contention wrecks
  timings. A background suite run roughly doubles every measured frame time —
  kill other vitest processes before trusting a number.

Quote the *phase* number, which is stable. The whole-frame number drifts by
20% run to run on this machine, so any before/after must be run back to back in
one command.
