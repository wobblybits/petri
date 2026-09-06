# Plan: weights that learn within a lifetime

Agreed 2026-09-06 with the author, after the audit. The goal is the one
stated in `docs/audit-2026-09-06.md` §7.4: a body's state matrices change
while it lives, taught by the net's own need signal, so that an agent which
has been through a net carries something an immigrant does not.

Nothing here adds a fitness term. The teacher is a quantity the simulation
already computes for its own reasons and no genome can fake.

---

## 1. The rule

Three-factor Hebbian, in the RFLO / e-prop form: a local eligibility trace
per weight, gated by a global-to-the-body modulator that is a temporal
difference error from a linear critic.

```
                                              per body, per frame
x, h, mean(h_j)   as today
v_d               pre-activation of state dim d
h_d = phi(v_d)    phi(v) = v / (1 + |v|),  phi'(v) = 1 / (1 + |v|)^2

reward   r    = FULL - 1              own tank shortfall. Need is the cost.
value    V    = c . h + c0            one linear readout, 5 floats
TD error dlt  = r + gamma * V - Vprev
critic   c   += etaC * dlt * hPrev    Vprev was c . hPrev, so hPrev is the gradient
trace    e   += lambda * e + phi'(v_d) * pre_j
weight   p   += etaM * dlt * e        clamped so gene + learned stays in range
```

The cost is the body's **own** shortfall: `IN_FULL` is its tank clamped to
[0,1], so `FULL - 1` is a dense cost that is zero when full and -1 when
empty. Local, and already in the input vector, so it is free to read.

The alternative is `IN_DEMAND`, which `spreadRequests` has already relaxed
over the wire graph with per-hop decay and which is therefore the *net's*
unmet need as seen from this body. That would make the teacher an
organism-level signal with no extra aggregation. It is one line away and
both are in `x` already, so the two can be compared without restructuring
anything. Starting local, by decision.

**No weight decay, ever.** Learned weights persist for the life of the body
and travel with it when it detaches and latches elsewhere. That transfer is
the mechanism net-level specialisation depends on. The stability job decay
would have done is covered by the clamps that already bound every gene; if
saturation shows up, the fix is Oja-style normalisation, which bounds the
norm of a weight vector while preserving its direction.

The trace decay `lambda` is a different thing and stays. It is the credit
window, about twenty frames, which is roughly how long a transfer takes to
show in the need field.

## 2. What learns

The state matrices only: `Wx`, `Wh`, `Wn` and `b`. That is the author's own
framing ("the weights of an agent's h matrices") and it is the principled
choice: they are the only weights with a local gradient, since `phi'` gives
each one a defensible eligibility. The output heads have no per-channel
error signal, so learning them would need random feedback, which is a
separate decision and a later door.

Behaviour still changes, because every head reads `h`.

The four are **contiguous in `chem`**, which makes the whole design cheap:

| | offset | floats |
|---|---:|---:|
| `Wx` | `W_IN` | 28 |
| `Wh` | `W_SELF` | 16 |
| `Wn` | `W_NET` | 16 |
| `b` | `B_STATE` | 4 |
| span `[W_IN, F_OUT)` | | **64** |

So the learned block and the trace block are 64 floats each, indexed
exactly as the genome indexes the same weights.

## 3. Where it lives

Three new per-body arrays in `AgentStore`:

- `plasticAll`, 64 f32. The learned *delta*. Effective weight is
  `chem[k] + plastic[k]`.
- `traceAll`, 64 f32. The eligibility trace.
- `criticAll`, 5 f64, and `prevValue`, 1 f64.

Kept separate from `chem` rather than written into it for two reasons. The
lab page can then show learned drift and inherited drift as different
curves, which `matrixDrift` cannot today. And it gives the inheritance
slider something to scale.

`plasticOn`, one byte a body, is set the first time a body writes a
non-zero learned weight and never cleared, because nothing decays. A pond
that never learns therefore pays one branch a body and reads its genome
exactly as it does now.

Storage at fifty thousand bodies:

| | bytes/body | at 50k |
|---|---:|---:|
| plastic + trace | 512 | 26 MB |
| critic + prevValue | 40 | 2 MB |

Comparable to the genome table already on the device.

## 4. Inheritance

At birth, learning is **consolidated into the genome**: a parent's
contribution to a child is `chem[k] + kappa * plastic[k]` over the learned
span, combined by the existing blend-or-assort path and then mutated and
clamped as now. The child starts with `plastic` and `trace` at zero.

`kappa` is `params.inheritLearned`, a slider, default 1. At 1 a lineage
compounds what it learned down the worm; at 0 learning is somatic only and
dies with the body. The critic is blended the same way, since a child born
into its parents' neighbourhood inherits a useful prior about it.

Immigrants from `autoSpawn` get `seedChem` and zeros, which is the larval
stage as designed.

## 5. Order of work

**Phase 0. Layout, storage, parameters.** Constants derived in
`chem-layout.ts`; arrays in `AgentStore`; five parameters, all defaulting to
zero except `inheritLearned`. Nothing reads them yet.

**Phase 1. CPU reference in `updateState`.** The effective-weight read, the
critic, the traces, the update. Behind `params.learnRate`; at zero the
state pass must be bit-identical to today, which `state-hash` checks.

**Phase 2. Inheritance in `rewrite.ts`.** Consolidation at commute and at
erase.

**Phase 3. Tests.** Parity at zero. A rigged two-body net where need falls
and the learned weights move in the direction that made it fall. Trace
decay. Clamps against a saturating modulator. Persistence across a detach
and re-latch, which is the property the whole design is for.

**Phase 4. GPU port.** `genome.wgsl` gains the learned block. This needs a
binding merge first: the pass already uses eight storage buffers against a
per-stage guarantee of eight, so `hPrev` and `inputs` merge into one
per-body buffer to make room for one read-write learning buffer. Readback
is needed only for rewrite parents, and `beginRewrite` gives about forty
frames of notice, so the pair's rows can ride the next field round trip.

**Phase 5. Instrumentation.** Learned-weight norm and critic error into
`census()` and the experiment sample, ready for the lab page.

## 6. Consequences to watch

- **`readsField` stops being fixed for life.** A body can learn a non-zero
  sense column. The flag becomes monotone: set it the moment a learned sense
  weight becomes non-zero, never clear it. No per-frame rescan.
- **The state pass roughly doubles when learning is on**: 64 trace updates
  and 64 weight updates a body against about 60 multiply-adds for `h`.
  It is off by default and the cost is proportional to what is switched on.
- **`Wh` and `Wn` are the risk.** Online learning inside a recurrence can
  destabilise dynamics that a feedforward path cannot. They get their own
  rate genes seeded at zero, so the pathway exists and evolution opens it
  only if it pays.
