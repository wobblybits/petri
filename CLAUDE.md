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

Standing design rules: a few well-tuned mechanics rather than new ones, and
every proposal names the concept heading it serves and the mechanic that
heading already has; no fitness term, selection is emergent; nothing decays
but the eligibility trace; immigrants are larval and never spawned from the
population. The pipeline that ships is the wasm solver, the GPU field and
genome, and `stepAsync`.

How features are built and judged: from first principles and from working
code and papers elsewhere, named in the commit. A mechanism ships on, at a
setting the eye can see, behind a slider. Done means the author opened the
page and saw the effect, and the commit message says what to look for. If it
does not look right it is turned down or taken out. No bench, sweep or
protocol gates a feature, and nothing ships at zero to preserve the previous
pond.

`npm test` is a change detector; when a feature changes the pond, the tests
follow the pond. `npm run pond` is for questions the eye cannot reach over
ten minutes, and is not a gate. Grown nets for tests and for the page live
under `nets/` (`src/pond/README.md`, "Nets as files").
