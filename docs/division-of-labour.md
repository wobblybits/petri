# Can division of labour pay?

Measured 2026-09-09 with `src/experiments/labour.exp.ts`. Eight Con segments in
a chain, three seeds, sixty simulated seconds.

Everything the worm work has built — a mouth feeding a chain through its wires,
a polarised body plan, obligate trophic dependency — is *buildable*. Whether it
is **reachable by selection** is a different question, and
`docs/energy-chemistry-plan.md` §3 answers it in the abstract: specialisation
beats splitting only when `f(1) > 2·f(1/2)`, and a linear budget against a
concave payoff puts the optimum in the interior. The expression simplex is that
budget; Monod saturation is that payoff. The plan ships two candidate sources
of the missing superadditivity at neutral — `rowCost`, a fixed price per
expressed row, and `hillN`, which makes uptake convex at low density.

## The answer

**Division of labour does not pay here, at any setting either dial can take.**

The generalist wins at every `rowCost` from 0 to 0.02 and at both `hillN` 1 and
2 — and `hillN` 2 makes the specialist *worse*, not better, which is the
opposite of what it was added for.

| hillN 1 | rowCost 0 | 0.005 | 0.01 | 0.0125 | 0.015 | 0.02 |
|---|---|---|---|---|---|---|
| generalist | **0.398** | **0.398** | **0.394** | **0.390** | **0.385** | **0.364** |
| specialist | 0.367 | 0.367 | 0.364 | 0.358 | 0.348 | 0.317 |

It still wins when the second job is made enormously valuable: at `fertilise`
5000 the generalist reaches 0.345 against the specialist's 0.122.

The reason is in the `moved` column, which is 0.00 for every generalist run and
33 to 1,900 for the specialists. **A generalist net never has to move anything.**
Splitting jobs across bodies forces energy through the wires, and transport is
lossy in the way that counts: a body waiting for a delivery is a body that is
short in the meantime. That cost is structural and no row-count saving covers
it — the biggest saving on offer is 2 rows against 1.

## Two earlier versions of this were wrong, and how

**The first crossover was an artifact of an unfair champion.** With
`excreteRate` at zero there is exactly *one* economically live row in the whole
reaction table — the ground's uptake — so there is no labour to divide, and the
only "specialist" a net can form is one body doing the job while seven
freeload. That is a passenger list. Against a generalist paying for all eight
rows it is a 8× row-cost saving, which is not what division of labour costs,
and it produced an apparent crossover at `rowCost` 0.02. A real two-job split
pays 1 row against 2, and there is no crossover at all.

**The second version measured a job worth nothing.** Turning on `excreteRate`
and `fertilise` gives a genuine second job — a body excreting the fertiliser
channel makes the ground it stands on regrow faster. But at the shipped
ambient of 1 a cell holds two and a half tanks and every body sits pinned at
cap (`meanExtra` 0.398 against a cap of 0.4), so the ground is not the binding
constraint and nothing that increases the supply of ground can matter.
Measured, `fertilise` at 0, 3 and 12 gave byte-identical ponds. Half the
specialist's bodies were doing a job worth exactly zero.

The trial now runs food-limited, at a quarter of a unit a cell against a body
needing 0.9 over the trial.

## A calibration finding worth having on its own

**`fertilise` does nothing across its entire slider range.** Its useful values
are about three orders of magnitude above where it ships:

| fertilise | 0 | 12 | 200 | 5000 |
|---|---|---|---|---|
| generalist | 0.127 | 0.127 | 0.132 | **0.345** |

The slider runs to 8. The mechanism is connected and correct — it is the
scaling that is off, because a body's excreted density is tiny against a
multiplier applied to an already-small growth rate. Anyone turning `fertilise`
up to look for mutualism will see nothing and conclude the idea does not work.

## What this does not show

It is a hand-built comparison of fixed plans: it says which the economy
*favours*, not that selection would *find* it. There is no reproduction and no
mutation between the two. It is one topology (a chain), one length, and one
pair of jobs. A body plan where the two jobs are spatially interleaved more
tightly than a 48-unit segment spacing allows, or where the second job's
product does not have to diffuse to be useful, might change the transport
arithmetic that decides it.

What would settle it properly is reproduction: let the pond breed with
`rowCost` and `hillN` non-zero and see what expression profile the survivors
actually carry. That needs commutes, which needs active pairs, which a stable
worm by construction does not have.

## Reproducing

```bash
npm run experiment -- labour
```

The null control runs first and on purpose. Four times in this work a "nothing
happened" result has been a disconnected instrument rather than a fact about
the pond, and each time the tell was a control that should have differed and
did not. Here it caught two: the tank sizing that had both plans coasting on
the energy they were built with, and the fertiliser job that was worth nothing.
