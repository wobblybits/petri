# Can division of labour pay?

Measured 2026-09-09 with `src/experiments/labour.exp.ts`, eight Con segments,
three seeds, sixty simulated seconds.

Everything the worm work has built — a mouth feeding a chain through its wires,
a polarised body plan, obligate trophic dependency — is *buildable*. Whether it
is **reachable by selection** is a different question, and
`docs/energy-chemistry-plan.md` §3 already answers it in the abstract:
specialisation beats splitting only when `f(1) > 2·f(1/2)`, and a linear budget
against a concave payoff puts the optimum in the interior. The expression
simplex is that linear budget and Monod saturation is that concave payoff. So
the prediction is that every body becomes a generalist unless something buys
the superadditivity back, and the plan ships two candidates at neutral:
`rowCost`, a fixed price per expressed row, and `hillN`, which makes uptake
convex at low density.

## The two plans

**Generalist** — every segment at the seed, which `expressVector` turns into a
flat eighth on all eight rows. Every segment eats from its own cell and pays
`rowCost` eight times over.

**Specialist** — one mouth expressing the ground's uptake row alone, so it draws
at eight times the rate; every other segment expresses a single inert row and
must be fed through the wires. One row a body.

Fitness proxy is the mean tank across surviving bodies, with the body count
beside it because a plan that cannot feed itself loses its far end rather than
merely thinning.

## Result

| | rowCost 0 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|
| **hillN 1** generalist | **0.400** (8) | **0.399** (8) | **0.398** (8) | −0.085 (5.0) |
| **hillN 1** specialist | 0.295 (8) | 0.285 (8) | 0.241 (8) | **−0.038** (6.3) |
| **hillN 2** generalist | **0.400** (8) | **0.397** (8) | −0.133 (0.7) | −0.200 (6.3) |
| **hillN 2** specialist | 0.244 (8) | 0.138 (8) | **−0.018** (6.0) | **+0.029** (5.7) |

**At the shipped dials the generalist wins outright**, at every seed, with full
tanks against the specialist's 0.295. The plan's argument holds: division of
labour is not reachable at `rowCost = 0, hillN = 1`, so a hand-built polarised
worm is a thing this simulation can be shown and not a thing it can find.

Both dials buy the crossover and they compound — `hillN` 2 halves the
`rowCost` needed, from 0.02 to 0.01. So the mechanism the plan proposed does
work.

## The caveat that matters

**The crossover is at the edge of the economy, not inside it.** `rowCost` 0.02
is the top of its slider, and at the settings where the specialist wins both
plans are in collapse — negative tanks, bodies dying. The specialist mostly
wins by dying less. There is exactly one cell in the table where it is actually
solvent while the generalist is not: `hillN` 2 with `rowCost` 0.02, at +0.029
against −0.200.

So division of labour is reachable, but only in a narrow band near the point
where the pond stops working, and the defaults sit firmly in the generalist
regime. Widening that band — a cheaper row cost that still buys
superadditivity, or a third source of it — is a balance question the plan does
not answer.

## What this does not show

A hand-built comparison of two fixed plans says which the economy *favours*,
not that selection would *find* it: there is no reproduction here and no
mutation between the two. The specialist design is also crude — one mouth and
seven inert segments — so it is a lower bound on what a polarised body could
manage rather than a fair representative of one. And in the crossover region
the comparison is between two failing designs, which is a weaker claim than a
crossover between two working ones.

## Reproducing

```bash
npm run experiment -- labour
EXP_SECONDS=120 EXP_SEGMENTS=12 npm run experiment -- labour
```

Note the tank sizing. At the shipped cap of 1.25 against an upkeep of 0.015 a
body takes 83 s just to reach break-even, and a first pass at 60 s read `moved`
0.00 for both plans — the specialist's segments had not yet gone short, so
nothing had asked and no transport had happened. The trial caps tanks at 0.4 so
the run measures a steady state rather than the drain down to one.
