# Inchworm — implementation spec

A 2D spring-graph body that learns to crawl toward a point food source. Degree ≤ 3 nodes;
each node modulates its own ground drag ("grip"), each edge its rest length ("tug"). Learning
is continuous and distributed — **no episodes, no resets, no environment rewind**.

This document is a complete reimplementation spec plus the measurements that calibrated it.
Numbers marked **measured** came from running the thing; treat unmarked reasoning as design
intent, not evidence.

---

## 1. Why the design is shaped this way

Three facts drive every choice. They are worth internalising before reading the code.

**Locomotion requires a two-parameter loop.** In the quasi-static limit, net displacement over a
closed loop in control space is a geometric phase. A single control parameter traces a path that
encloses zero area and produces zero net motion (Purcell's scallop theorem, which transfers
intact to dry crawling). Grip alone: nothing. Rest length alone: nothing. Confirmed numerically
in §8.

**The signal lives in phase relationships, not in values.** A memoryless map from sensor to
output retraces its own path and cannot locomote. The controller therefore needs an oscillator,
and the learned quantity must be a *phase offset*.

**The area rule holds only for viscous grip.** With Coulomb friction and genuine sticking you get
Gidoni's rate-independent sweeping process: stasis domains where a small control loop produces
*exactly zero*, and displacement linear in loop amplitude minus a stiffness threshold once a slip
surface is crossed — not area × curvature. This implementation uses **viscous grip** (force linear
in velocity), where the clean geometry does hold and everything stays differentiable.

---

## 2. Units and constants

Length unit ≈ one edge rest length. Time unit ≈ one spring relaxation time (c/k).

```
DT        = 0.01     physics + control step
K_SPR     = 1.0      spring stiffness
K_REP     = 2.0      soft repulsion, non-adjacent pairs only
R_REP     = 1.0      repulsion range
C0        = 1.0      baseline drag
GAMMA     = 2.0      grip -> drag exponent
STRAIN    = 0.28     strain limit, fraction of current rest length
CAPTURE   = 1.4      capture radius
FOOD_D    = 12       respawn distance from centroid
BLIND_BIG = 1e9      sentinel for "this node has no reading"

W_REWARD  = 0.15     low-pass on raw reward   (must sit well below omega)
W_HIGH    = 0.004    high-pass = the baseline (must sit well below dither freq)
W_DEMOD   = 0.004    demodulator low-pass
W_RMS     = 0.004    reward-scale tracker
```

Tunable parameters and their calibrated defaults:

```
n=24  omega=1.5  w=1.5  rg=0.45  xg=0.50  a=0.22  kappa=1.0  kappaT=0
eta=0.01  amp=0.35  holdPeriods=8  lamG=0.004  lamL=0.004
gamma=0.35  convexP=1.0  d0=4.0  mu=0.10  kbend=0.5  cycles=3
```

---

## 3. Bodies

All topologies keep max degree 3. **No degree-≤3 graph with n > 6 is ever generically rigid in
2D** — 3n/2 available edges against the 2n−3 required — so angular springs (§6) are the entire
shape-holding mechanism at every size. This is not optional.

| name | construction |
|---|---|
| `chain` | path graph |
| `ladder` | two rails of m = n/2 plus rungs; every node ends at degree ≤ 3 |
| `geometric` | points in a 4:1 strip; **spanning path along x first** (connectivity by construction), then greedy shortest remaining pairs up to degree 3 |
| `unet` | ladder edges with an explicit U embedding (§3.1) |
| `limbed` | degrees 1 and 3 only (§3.2) |
| `cubic` | random 3-regular via the pairing model, rejecting non-simple or disconnected draws |

Geometry: `unet` uses its explicit embedding; every other body is placed by relaxing springs
(rest length 1) plus repulsion from a random seed for 900 iterations at h = 0.04. Edge rest
lengths `ell0[k]` are then read off the relaxed positions, so the body starts unstressed.

> **Trap.** An earlier `geometric` generator did greedy-shortest-first with no spanning path and
> produced *disconnected* bodies (a stray component orbiting the main one). Always assert
> `components == 1` after generating.

### 3.1 U-net embedding

Two rails bent around a semicircle: two straight legs at x = ±R joined by a 180° arc of radius
R = 2.0, sampled at unit arclength, with rails offset ±W/2 = ±0.475 along the path normal. Rail A
is indices 0..m−1, rail B is m..2m−1, where m = n/2. Tips are the rungs at path positions 0 and
m−1 — spatially close (gap ≈ 2R) but maximally far apart around the graph, which is the whole
point (§7).

### 3.2 Limbed body — degrees 1 and 3 only

Counting edge-ends with cyclomatic number c = m − n + 1 forces, **exactly**:

```
n1 = n3 + 2 - 2c
```

Every independent cycle costs two sensory nodes. At c = n3/2 + 1 the body is completely blind.
Cycles and sensing are in direct, quantified competition.

Construction for target (n, c): set `n3 = (n - 2 + 2c)/2`, build a backbone path over
0..n3−1, give each interior backbone node 1 spare stub and each end 2 (total n3+2 stubs), spend
2c of them on chords between backbone positions `gap` apart (try gap = 3..8), and attach the
remaining n1 stubs to leaf nodes. n must be even.

---

## 4. Per-node state

**Frozen at construction**

| field | notes |
|---|---|
| `adj[i]` | neighbour list, ≤ 3 |
| `rho_id[i]` | 4-dim random vector, one draw, kept forever |
| `blind[i]` | 1 iff degree > 1 |
| `centroid` | Jordan centre (min-eccentricity node), computed once |

`rho_id` exists because of **1-WL**: with identical initial features on a regular graph, mean or
max aggregation returns the same representation at every node forever, at any depth. Frozen
random IDs give universal approximation (Abboud et al.) and, since there is only one graph,
avoid the cross-graph generalization problem that per-episode resampling causes.

**Learned** (written continuously by the outer loop, §7)

Free mode: `theta[i]`, one phase offset per node, gauge-fixed by subtracting the mean each step
(a uniform shift of θ is unobservable). Profile mode: three global coefficients `c1,c2,c3`.
Optionally also `rg, xg, a, kappa`.

**Fast** (every step)

```
p[i]        position (no velocity — overdamped)
phi[i]      detrended phase
thEff[i]    phase offset actually applied this step
s[i]        raw sensor  = |p_i - food|
sLP[i]      low-pass of s[i]   (alpha = 1 - exp(-DT*0.6))
g[i]        grip in [0,1]
hVal/hDst/hSrc   head-field claim: source reading, graph distance, source index
tVal/tDst/tSrc   tail-field claim (same machinery on negated readings)
rho[i], rhoHat[i]
```

Per edge: `ell0[k]` (natural length), `ell[k]` (commanded rest length this step).

All inter-node reads are **double-buffered**. If a node reads a neighbour value already updated
this step, you have silently built a sequential algorithm whose behaviour depends on node
ordering.

---

## 5. The head field

Each node relays a **claim**: the source's live reading, the graph distance to it, and the source's
index. It keeps the cheapest of its own reading (distance 0, source itself) and each neighbour's
claim carried one more hop.

```
cost(d) = gamma * d                              if convexP == 1
        = gamma * d0 * (d/d0)^convexP            otherwise

for each i:
    own = blind[i] ? BLIND_BIG : sgn * s[i]
    best = own - (heldSource[i] == i ? mu : 0)
    bV, bD, bS = own, 0, i
    for j in adj[i]:
        D  = dOut[j] + |p_j - p_i|
        sc = vOut[j] + cost(D) - (srcOut[j] == heldSource[i] ? mu : 0)
        if sc < best: best, bV, bD, bS = sc, vOut[j], D, srcOut[j]
    store (bV, bD, bS)
    f = bV + cost(bD);  if f > own: f, claim = own, (own, 0, i)
    field[i] = sgn * f
```

Run once with `sgn = +1` for the head field `m`, once with `sgn = -1` for the tail field `M`.

**A head is a node that claims itself.** Exact, not a threshold.

Three properties that matter:

- **At convexP = 1 and gamma ≥ 1 the relay is exactly the identity.** s is 1-Lipschitz w.r.t. the
  graph metric — the triangle inequality gives `s_i ≤ s_j + ell_ij` on every edge — so a node's own
  term always wins. **Measured:** ρ_max = 0.0000 at γ = 1.0 and γ = 1.2.
- **The criterion is local, not global.** i is beaten by j iff `s_i - s_j > gamma*ell`, and since
  `s_i - s_j = ell*cos(alpha)`, that is just `cos(alpha) > gamma`. **γ is a cosine threshold on
  edge alignment** — a node is a head unless one of its edges points within arccos(γ) of the food
  (70° at γ = 0.35). Curvature, not topology, produces most heads at p = 1.
- **It cannot latch.** Relaying the *source* rather than a combined value means every source
  re-asserts its live reading each step, so a claim is only as stale as its hop count. Stale cycles
  self-destruct because distance accumulates around a loop until the node's own reading wins.
  A scalar min-plus relay *does* latch (min propagation is monotone decreasing) and needs an
  explicit leak; the claim form does not.

### Depth coordinate

```
blind mode:    rho = cost(hDst)                     ; rhoHat = rho / (rho + cost(tDst))
sighted mode:  rho = s - m                          ; rhoHat = (s - m) / (M - m)
```

Both are in [0,1] and computed **entirely from relayed scalars** — no node needs the body's size.
An earlier version divided by a hardcoded `REF_RHO = 3.0`, which smuggled a global body-scale
assumption into a supposedly local rule. The tail field removes it and makes the learned profile
coefficients transfer across body sizes.

---

## 6. One update step

Order matters. Steps 1–4 are fully parallel across nodes; step 5 is the only global operation.

**0 · Dithers.** Every `holdPeriods * 2π/omega` of sim time, resample an independent Rademacher
sign `d_k ∈ {−1,+1}` for each learned parameter.

**1 · Sense.** `s[i] = |p_i − food|`; `sLP[i] += alpha*(s[i] − sLP[i])`.

**1b · Relay.** Head field, tail field, then ρ, ρ̂, head count, centre owner (§5).

**2 · Exchange.** Snapshot `phi` and `s` into outboxes.

**3 · Phase.**

```
phi_i += DT * ( omega
              + w * mean_{j in couple(i)} sin(phi_j - phi_i)
              + kappa   * (s_i - mean_{j in adj(i)} s_j)
              + kappaT  * (s_i - sLP_i) )
```

Coupling is **normalised by neighbour count** so the star/line/body ablation stays comparable.
Both steering terms advance the phase of a node that is *worse off* — farther than its neighbours,
or receding. **The sign is not free: it is coupled to the gait's chirality** (§8).

`couple(i)` is the ablation switch: the body graph, a star (node 0 to all), a line through index
order, or nothing.

**4 · Actuation.**

```
free mode:     thEff_i = theta_i + amp * d_i
profile mode:  thEff_i = C1*u + C2*u^2 + C3*u^3,  u = rhoHat_i   (no constant term: pure gauge)

g_i     = clip(xg + rg*cos(phi_i + thEff_i), 0, 1)
drag_i  = C0 * exp(GAMMA * (2*g_i - 1))
ell_ij  = ell0_ij * (1 + 0.5*a*(cos(phi_i + thEff_i) + cos(phi_j + thEff_j)))
```

The edge uses the **sum** of endpoint oscillations, not the average of their phases — symmetric in
(i,j) by construction and it avoids circular-mean entirely.

**5 · Physics.** Overdamped, no inertia.

```
F_i  = sum_{j in adj} K_SPR*(|p_j - p_i| - ell_ij) * unit(p_j - p_i)
     + sum_{j non-adjacent, |p_j-p_i| < R_REP} -K_REP*(R_REP - |p_j-p_i|) * unit(...)
     + bending (below)
p_i += DT * F_i / drag_i
```

Repulsion over non-adjacent pairs is **not optional** — with attraction only, the body folds onto
itself and there is nothing left to crawl with. O(n²) is free at n = 24.

*Bending*, for each node and each pair of its edges, with rest angles captured at build time:

```
u = unit(p_j - p_i), v = unit(p_k - p_i), a = |p_j - p_i|, b = |p_k - p_i|
th = atan2(u x v, u . v);  delta = wrapPi(th - restAngle)
F_j = +kbend*delta/a * perp(u)        perp(x,y) = (-y, x)
F_k = -kbend*delta/b * perp(v)
F_i = -(F_j + F_k)
```

*Strain limiting* after integration: clamp each edge length into `ell*(1 ± STRAIN)` by repositioning
endpoints, weighted by mobility `1/drag` so an anchored node barely moves. This is a projection,
not a force, and it is what keeps a penalty-contact spring net from exploding.

**6 · Reward and learning** (§7). **7 · Capture.** **8 · Diagnostics.**

---

## 7. Continuous learning — extremum seeking

No episodes. The episodic machinery is replaced piece by piece:

| episodic | continuous replacement |
|---|---|
| return to a terminal boundary | reward **rate**: low-passed approach speed |
| perturbation fixed for an episode | perturbation **dithered slowly** relative to the gait |
| baseline reset per episode | **high-pass filter** on R(t) |
| reset on reaching the food | the **food relocates** |

Three timescales, strictly separated: `learning << dither << gait`. Violate the right-hand
inequality and the demodulator picks up gait harmonics and walks off confidently in a wrong
direction. Check for a clean trough in the reward power spectrum between the two.

```
R_raw = (prevDist - dist)/DT  -  lamG*meanGrip  -  lamL*(sum|d ell|/DT)/m
R_lp += DT*W_REWARD*(R_raw - R_lp)
chi  += DT*W_HIGH  *(R_lp  - chi)
Rt    = R_lp - chi                                   # baseline by filter, not by variable
rms  += DT*W_RMS   *(Rt*Rt - rms)
Rn    = Rt / (sqrt(max(rms,1e-12)) + 1e-9)           # scale-free, so eta is portable

for each learned parameter k:
    G_k     += DT * W_DEMOD * (Rn * d_k - G_k)
    theta_k += DT * eta * scale_k * G_k
```

Each node demodulates the **shared broadcast reward** against **its own dither sign**. Correlated
components survive the low-pass; everything else averages away. One scalar broadcast channel,
frequency-division multiplexed. This dissolves the anchor-node credit problem: an anchor's own
Δs is negative while it does the most useful work, but its dither correlation with the global
reward is positive.

Per-parameter dither and update scales: `theta, c1..c3 → 1.0`, `rg, xg → 0.15`, `a → 0.10`,
`kappa → 0.30`. Clamps: `rg∈[0,0.5]`, `xg∈[0.1,0.9]`, `a∈[0,0.30]`, `kappa∈[-2,2]`, `c∈[-8,8]`.

**Persistent excitation never stops.** The system always jitters; that is correct, and it is what
lets it keep tracking. Annealing `amp` as |G| shrinks is fine; driving it to zero kills the
gradient estimate.

**Food relocation** on capture (any node within CAPTURE): teleport to a random bearing at
FOOD_D from the centroid, and **skip the reward for that step** — otherwise the distance
discontinuity injects a huge spike straight into the gradient.

**Dither amplitude is two-sided.** Too large detunes the gait permanently; too small and
`Rt*d_k` drowns in reward noise over the demodulation window. Set it by measuring reward σ
over 1/W_DEMOD with the dither off, then choose amp so the induced swing is a few times that.

---

## 8. Measurements

### Physics validates against theory

Chain-24, hand-set traveling wave, displacement per gait cycle:

| case | disp/cycle |
|---|---|
| traveling wave | 0.022 – 0.116 |
| **θ = 0** (no phase gradient) | **0.000** |
| **r_g = 0** (uniform grip) | **0.000** |
| **a = 0** (no shape change) | **0.000** |

Three exact zeros. The scallop theorem and the uniform-grip result reproducing themselves.

### The steering sign is not free

Chain-24, hand-set wave, no learning, mean approach speed:

| κ | −2 | −1 | −0.6 | 0 | +0.6 | +1 | +2 |
|---|---|---|---|---|---|---|---|
| speed | 0.026 | 0.027 | 0.015 | 0.003 | −0.018 | −0.028 | −0.027 |
| captures | 19 | 19 | 11 | 3 | 0 | 0 | 0 |

(Under the *old* sign convention; the current code uses `s_i − ŝ_i`, so positive κ is correct.)
Wrong sign ⇒ reliably marches away and never once reaches the food. κ and θ must be learned
jointly.

### Learning rate

Chain-24, learning from random θ, mean approach speed over six 1500-time windows:

| η | 0.004 | 0.01 | 0.025 | 0.06 | 0.15 | 0.5 |
|---|---|---|---|---|---|---|
| captures | 5 | **7** | 4 | 2 | 1 | 1 |

η ≈ 0.01. Above ~0.06 it chases noise and does worse than not learning.

### Profile parameterization beats free offsets by ~6×

Learning from a random start, 9000 sim-time:

| body | θ | captures |
|---|---|---|
| chain | free (28 params) | 3 |
| chain | **profile (3 params)** | **35** |
| geometric | free | 9 |
| geometric | **profile** | **54** |

Profile windows are also *flat* (0.035–0.049 sustained) where free-θ windows scatter and go
negative — the food-aligned frame means the wave re-aims itself instead of being relearned.
Three coefficients beat a hand-tuned gait (~0.027).

### Head counts

Mean heads, p = 1, γ = 0.35: chain 5.42, cubic 4.08, geometric 3.46, U-net 2.49. Multiple heads
are the *default*, and most of them are body curvature rather than real lobes.

Convexity suppresses the spurious ones (expander): p=1 → 3.40 heads, p=1.5 → 1.97,
**p=2.0 → 1.31**. On the U it converts a bearing-dependent scatter of 1–4 heads into a **stable 2
at nearly every bearing**. p ≈ 2 is what separates a lobe from a wiggle.

U-net shape retention over 600 sim-time: k_b = 0 collapses tip gap 3.05 → 2.04; k_b = 1.5 holds
3.05 → 3.05.

### Degree-1/3 body

Identity `n1 = n3 + 2 − 2c` verified exactly at c = 0, 2, 4, 6, 9. Across every run, **no blind node
ever became a head** — the degree sequence enforces structurally what convexity only
approximates. Mean heads tracks the leaf count: 5.83 → 4.56 → 3.53 → 2.04 as c goes 0 → 6.

Blinding 14 of 24 nodes costs nothing measurable — 33 captures vs 35 sighted, approach speeds
within noise. Interior readings are redundant; the extremities carry the directional information.

### One hypothesis falsified

"The centre commits to a limb before the body turns." Over 462 turning events the Jordan centre's
basin flipped 13 times, with 236 flips preceding a turn and 226 following — chance, and two
orders of magnitude too infrequent to be driving anything. The centre's basin is a genuinely
*stable* slow variable (topological hysteresis working as intended), but it is not the steering
mechanism.

---

## 9. Known traps

1. **Single-buffered messaging** silently makes the algorithm order-dependent.
2. **Disconnected bodies** from greedy graph generation. Assert connectivity.
3. **Damping fighting translation.** Subtract rigid-body motion before applying any velocity
   damping, or net locomotion is quietly suppressed.
4. **Reward spike on food relocation.** Skip that step.
5. **Timescale collapse.** If dither frequency creeps toward gait frequency, the demodulator
   estimates harmonics and the parameters drift confidently in a wrong direction. Silent failure.
6. **Node perturbation instead of weight perturbation.** For a temporally extended task NP's
   error floor scales with episode length T, WP's with the parameter count. Perturb once per
   dither hold, not every step.
7. **Wrong baseline is a bias, not just noise** — either centre the trace or use the exact mean
   reward.
8. **Zero-init the output layer** if you replace the scalar controller with a network, so the
   initial behaviour is do-nothing and perturbation starts from a stable fixed point.

## 10. Open / unresolved

- Per-node *independent learning* (separate weights per node) is untested here and has no
  published recipe at n = 30; the MARL literature predicts failure from non-stationarity under
  simultaneous updates. Everything here shares one parameter vector.
- The Kurin ablation (true graph vs star vs line coupling) is wired in but has not been run. If
  the star matches the body graph, the message-passing layer is not earning its keep.
- Coulomb grip with genuine sticking is not implemented. Expect stasis domains and
  threshold-linear displacement, not the area rule.
- Specialization here is **imposed** by the degree sequence, not self-organized. Whether
  identical nodes can differentiate into these roles on their own is the original question and
  remains open.
- `kappaT` (bacterial temporal-difference steering) defaults to 0 and is unmeasured.
