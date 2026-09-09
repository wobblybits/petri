# Petri

An artificial-life pond: interaction-net bodies (Con, Dup, Era) that latch,
rewrite, eat, signal and learn. `src/` is the simulation and pages; `src/pond/`
is the headless runner and the library it writes; `docs/` holds the plans and
audits.

Two documents are the ones to read before proposing anything:

- **`docs/concepts.md`** — the research-level concepts (memory, learning,
  locomotion, shape, breeding, inheritance, selection, and the rest) and the
  one mechanic each is tied to, so that a proposal names the heading it serves
  and a mechanic asked to do two things is visible. Prescriptive, and written
  for a reader with no history of the project; it is being filled in heading
  by heading with the author.
- **`docs/experiments.md`** — how to ask the pond a question: the failures
  the last sweeps paid for and the rule each became, the measure catalogue with
  seed budgets, the coupling table, and the protocol format
  (`npm run pond -- protocols`).

Standing rules, each argued in those files: a few well-tuned mechanics rather
than new ones, and every proposal names the concept heading it serves and the
mechanic that heading already has; no fitness term, selection is emergent; nothing decays but the eligibility trace; every mechanism ships at
the value that reproduces yesterday's pond; immigrants are larval and never
spawned from the population; measure the pipeline that ships (wasm solver,
GPU field and genome, `stepAsync`); never quote a pond younger than a couple
of simulated minutes or a sweep with fewer seeds than its measure needs.

`npm test` is a change detector for mechanical rewrites, not a measure of
behaviour. Behaviour is measured with `npm run pond` on runs of minutes.
