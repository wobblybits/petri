# Experiments

Sweeps that study the pond instead of asserting about it. A trial is a
seeded soup run for a fixed number of simulated seconds and sampled on a
schedule; a sweep is a parameter grid crossed with seeds. Nothing here passes
or fails — the output is a table on stdout and a JSON file under
`experiments/out/` you can plot.

```bash
npm run experiment -- breeding          # declutter x flockAlign x spawnInterval
npm run experiment -- economy           # the dials that ship at zero, one at a time
npm run experiment -- worm              # one hand-built organism, not a pond
npm run experiment -- beam              # where a body plan's bending stiffness comes from
EXP_SECONDS=180 EXP_BODIES=600 npm run experiment -- breeding
```

`--disableConsoleIntercept` is already in the script; without it vitest
swallows the table.

## What to read

- `bornMean` and `lines` together. Selection loses lines while depth climbs;
  drift with immigration keeps as many lines as arrived.
- `commutes` is the reproduction rate. `conDupWires` is how many are waiting
  to happen, so a low rate with a high count is an economy problem and a low
  count is a meeting problem.
- `matrixDrift` is how far the unseeded part of the genome has moved. It
  rises under drift and under selection alike; it only says the genome is
  being touched at all. If you want to know whether the pond is *diverging*
  rather than merely moving, that is `netFst` — see `src/pond/README.md`.

  **Not comparable across `CHEM_LEN`.** It is a mean over the span past the
  taste bases, so adding a head widens the denominator: the expression head
  `X` took `CHEM_LEN` from 134 to 174 and nothing reads it yet, which scales
  every `matrixDrift` by about 0.75 against runs from before. Within a sweep
  the factor cancels; across commits it does not. The `run` row in a pond
  library records `chem_len` for exactly this reason.

- The sweeps here write one JSON file per invocation and relate to nothing.
  `npm run pond -- sweep` writes the same trials into the pond library
  instead, where they accumulate and can be queried across sessions and
  commits. Prefer it for anything you want to still know next week.
- `canPay` is the fraction of bodies that could fund a rewrite this frame.

## One organism, not a pond

`organism.ts` is the other bench. A sweep over a soup wants population
statistics and every number in `Sample` is an average over thousands of
bodies; none of that says whether *this* net does *that* thing. A worm is one
organism with an inside — segments in an order, a gradient along it, a stroke
with a phase — and averaging it away is the wrong move.

So a trial there is a single net built to a spec, run with the rest of the pond
switched off, and sampled per segment. `buildWorm` lays a chain,
`dress` hand-writes genomes, `motorsOff` returns the overrides that leave one
motor running, and `gaitOf` reduces a run to a gait: speed, `headward`
(is it going where its nose points), straightness, and cost of transport.

Read `headward` and `straightness` together, and never `along` alone — a worm
that turns and then swims well scores badly on its build axis. Run several
seeds: `jitter` is what makes a seed mean anything, since nothing in a
motors-off run consumes `Math.random`, and the difference between swimming and
tumbling is in the spread rather than the mean.

See `docs/transport-worm.md` for what the first one measured.

## Adding one

Copy `breeding.exp.ts`. A sweep is a `grid` of `Params` keys to value lists,
a list of seeds, and the trial shape. The harness runs the CPU path
(`Sim.step`), which Node can run and which is the same simulation as the GPU
path bar one frame of genome latency.

Warm-up is real: a preset drops its whole population in as founders, so the
first thirty seconds measure the seeding. Sample every ten seconds and read
the slope, not the first row.
