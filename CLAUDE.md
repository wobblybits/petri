# Petri

An artificial-life pond: interaction-net bodies (Con, Dup, Era) that latch,
rewrite, eat, signal and learn. `src/` is the simulation and pages; `src/pond/`
is the headless runner and the library it writes; `docs/` holds the two living
documents and, under `docs/history/`, dated records that are not maintained.

Read these two before proposing anything:

- **`docs/concepts.md`**: one table per group, each research-level concept
  tied to the mechanic that implements it, the dials, the default, the gauge
  that says whether it was engaged, and what is not yet true.
  `src/concepts-dials.test.ts` fails if the table names a dial that does not
  exist or a slider no heading claims.
- **`docs/experiments.md`**: how to ask the pond a question with
  `npm run pond`: the rules, the protocol format, the measure catalogue, the
  coupling table, and the pipeline hazards.

Where the frame lives: `src/sim.ts` owns the frame and the passes that read
most of its state; `src/state.ts` is the state and learning pass,
`src/metabolism.ts` the pathway and the drag law, `src/body-chemistry.ts`
expression, digestion and excretion, `src/energy.ts` the economy,
`src/rewrite.ts` the interaction-net rules and inheritance, `src/gpu/` the
field and genome shaders and their hosts.

Standing rules: a few well-tuned mechanics rather than new ones, and every
proposal names the concept heading it serves and the mechanic that heading
already has; no fitness term, selection is emergent; nothing decays but the
eligibility trace; every mechanism ships at the value that reproduces
yesterday's pond; immigrants are larval and never spawned from the
population; measure the pipeline that ships (wasm solver, GPU field and
genome, `stepAsync`); never quote a pond younger than a couple of simulated
minutes or a sweep with fewer seeds than its measure needs.

Comments say what a thing does and the invariant it protects. History goes in
the commit message. A number in a comment names the command or test that
produced it or is not there.

`npm test` is a change detector for mechanical rewrites, not a measure of
behaviour. Behaviour is measured with `npm run pond` on runs of minutes.
