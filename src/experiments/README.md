# Experiments

Sweeps that study the pond instead of asserting about it. A trial is a
seeded soup run for a fixed number of simulated seconds and sampled on a
schedule; a sweep is a parameter grid crossed with seeds. Nothing here passes
or fails — the output is a table on stdout and a JSON file under
`experiments/out/` you can plot.

```bash
npm run experiment -- breeding          # declutter x flockAlign x spawnInterval
npm run experiment -- economy           # the dials that ship at zero, one at a time
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

## Adding one

Copy `breeding.exp.ts`. A sweep is a `grid` of `Params` keys to value lists,
a list of seeds, and the trial shape. The harness runs the CPU path
(`Sim.step`), which Node can run and which is the same simulation as the GPU
path bar one frame of genome latency.

Warm-up is real: a preset drops its whole population in as founders, so the
first thirty seconds measure the seeding. Sample every ten seconds and read
the slope, not the first row.
