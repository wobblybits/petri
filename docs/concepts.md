# Concepts

The design bets on a few well-tuned mechanics that leave room for emergence.
It does not add mechanics, fitness measures, rule systems or workarounds where
an existing mechanic can be tuned or a condition changed. This document keeps
that true by tying every research-level concept to one implemented mechanic,
so that "what gives us X" has a one-line answer naming a dial, and a mechanic
asked to do two things at once is visible.

A proposal names the heading it serves. If the heading has a mechanic, the
proposal is a tuning, a measurement or a bug fix. If it has none, the proposal
says so and says why no existing mechanic can be tuned to cover it.

How to read the table:

- **mechanic** is the function that implements the heading, in `src/`.
- **dials** are `Params` keys. `src/concepts-dials.test.ts` checks every one
  named here exists, so a dial that is deleted has to be deleted here too.
- **default** is what `defaultParams()` ships. *off* means the mechanic is
  present but inert at the default.
- **gauge** is the `npm run pond` measure that says whether the mechanic was
  engaged in a run. A result read without its gauge is not a result.
- **not yet true** is the gap between the heading and the mechanic, stated
  once and without history. History is in `git log` and `docs/history/`.

## Agent

| heading | mechanic | dials | default | gauge | not yet true |
|---|---|---|---|---|---|
| Sensing | `Sim.steer`: a body samples the four field species around its principal and steers by its taste head | `sense`, `sensorDist`, `sensorAngle`, `senseScale`; the taste seeds `attractStrong`, `attractMedium`, `attractFood` | on | `sense_read_p90` | `senseScale` is measured for the minted signal; a conserved signal reads three orders lower and needs its own value |
| Movement and steering | cruise and turn heads on a free principal, flocking, the drag law | `stepSpeed`, `turnRate`, `swimTau`, `swimNoise`, `flockAlign`, `flockSep`, `declutter`, `drag`, `angDrag`, `eraMass`, `nodeMass` | on | `net_drift` | a wired body does not steer; only the net moves it |
| Signalling | emit head laid into the field: minted (`deposit`) or conserved excretion rows (`excreteRate`) | `deposit`, `diffuse`, `decay`, `excreteRate` | minted | `signal_total`, `signal_p90` | the three signal species have identical physics, so no channel can mean anything different from another |
| Metabolism: eating | harvest into the tank (take-what-fits) or, metered, into the gut and then digested | `ambientEnergy`, `uptakeVmax`, `uptakeKs`, `digestRate`, `gutSize`, `catCoSubstrate`, `yDirect`, `yEra` | take-what-fits | `full_mean`, `demand_mean` | at `uptakeVmax` 0 uptake is instantaneous and in id order, so older bodies eat first in a contested cell |
| Metabolism: upkeep | rent per second off the tank; the pathway in `advanceMetabolism` buys substrate off the tank when discharged | `upkeep`, `upkeepExcrete`, `eraUpkeepRatio`, `rowCost`, `metabolicRate`, `metabolicSupply`, `metabolicBase`, `metabolicRegen`, `metabolicWork`, `metabolicCost`, `metabolicDiffuse`, `adenylate` | pathway off | `tank_life`, `can_pay` | the pathway oscillates only near the seeded pool size because its influx is not scaled by the pool; its load is charged on charge held, not on movement, and on bodies with no wire |
| Metabolism: excretion | `runExcretion`: the gut first, the tank for the shortfall, as the body's expression rows | `excreteRate`, `upkeepExcrete` | off | `ground`, `free` | at the defaults rent and the pathway's spend are destroyed rather than laid down |
| Memory | the recurrent state `h` through `Wh`, four dimensions | none | on | `locus_self_00` | |
| Learning | three-factor rule in `updateState` and `genome.wgsl`: eligibility trace per state weight, gated by a TD error from a linear critic on the body's own tank | `learnRate`, `learnCritic`, `learnTrace`, `learnDiscount`, `inheritLearned` | on | `full_mean` (the teacher is zero at a full tank) | the eligibility carries no sign per state dimension, so the update does not climb reward; the horizons are per frame at a frame the browser varies |
| Regulation | every output head reads `h`: emit, taste, flock, recoil, anchor, expression | none | on | | |
| Death | `tickUpkeepFast` kills at the body's own debt floor; the corpse's worth returns to the ground | `debtCap`, `bodyValue` | on | `died`, `loneliness` | |

## Net

| heading | mechanic | dials | default | gauge | not yet true |
|---|---|---|---|---|---|
| Identity | a connected component of wires | none | | `nets`, `nets_effective` | |
| Shape | wire physics: a span constraint on the rest length, port torques, ropes | `wireMinRest`, `wireShrink`, `springK`, `springDamp`, `portStiff`, `auxSpread`, `wireBreathe`, `wireSnap`, `wireTaut`, `wireClear`; rope detail ages out by `wireShapeAge`, `wireSpanAge` | on | `pp_wires`, `con_dup_wires` | stiffness is a global constant; a net that moves no energy is not slack |
| Locomotion | the drag law in `dampVelocities`: `drag + grip * fullness + anchor`. A transport kick is equal and opposite, so only a difference in damping between a wire's two ends turns it into travel. `anchor` rides the pathway's wave and the stroke swings the rest length by `gaitSwell` | `grip`, `gaitSwell`, `metabolicRate`, `transportRecoil`, `transportQuantum` | grip and packets on, pathway off | `net_drift` against `transportRecoil` 0, `demand_mean` | `grip` is global, so its sign cannot be settled by selection; substrate diffusion synchronises a chain, and a synchronised chain does not travel |
| Communication within a net | `Wn`: the mean of wired neighbours' state; the demand field | `requestDecay`, `requestReach` | on | `demand_mean` | |
| Transport of energy | `flowChargesFast`: a body gives to a strictly needier neighbour; a packet or a trickle | `transportQuantum`, `requestDecay`, `requestReach`, `rescueTo` | packet | `demand_mean`, `can_pay` | the packet is also the only clock the drag law has while the pathway ships off |
| Growth | rewrites: a commute turns two bodies into four | `rewriteDuration` | on | `born`, `commutes` | |
| Breeding: nets connecting | latching at free ports | `snapRadius`, `snapArc`, `snapWell`, `faceRadius`, `faceAttract` | on | `latches`, `latch_p50` | |
| Freeing terminals | death, `wireSnap`, and the erase rules | `wireSnap` | on | `died`, `commutes_per_latch` | |
| Specialisation | the expression simplex `X`, priced per row | `rowCost`, `hillN` | off | `net_fst` | with a linear budget and concave payoff the optimum is a generalist; `rowCost` or `hillN` above neutral is what makes a specialist pay |

## Lineage

| heading | mechanic | dials | default | gauge | not yet true |
|---|---|---|---|---|---|
| Reproduction | a commute's four children | none | | `born_mean` | |
| Inheritance | `inheritChem`: blend or assort per gene, plus the scalar traits; learned weights consolidated by `inheritLearned` | `assortBias`, `inheritLearned` | on | `locus_*` | |
| Mutation | one `CHEM_MUTATE` step per gene per birth, scaled to the genome length; `TRAIT_RANGE` per trait | none | on | `matrix_drift`, `var_drifted` | |
| Selection | none: no fitness term. Whatever persists, persists | none | | `line_dominance`, `net_fst` | |
| Drift | mutation on genes nothing reads | none | | `line_fst`, `matrix_drift` | most of the genome is inert for a body in a given position, so it walks |
| Speciation | nets diverging | none | | `net_fst`, `nets_effective` | |
| Development | an immigrant carries `seedChem` and nothing learned; its first latch is its metamorphosis | none | | `latch_p50` against `tank_life` | |
| Immigration | `autoSpawn` near the pond's centre of mass | `spawnInterval`, `soupCount`, `maxAgents` | on | `spawned`, `lines` | immigrants are never spawned from the population; they are larval by design |

## Ecology

| heading | mechanic | dials | default | gauge | not yet true |
|---|---|---|---|---|---|
| The energy economy | one scalar; bodies conserve it when `upkeepExcrete` and `excreteRate` are on; the dish is driven | `bodyValue`, `upkeepExcrete`, `excreteRate`, `eraCapRatio` | bodies destroy rent | `free`, `ground`, `mean_extra` | |
| The world | the ground: ambient, logistic regrowth, optional patches or a reaction | `energyCell`, `ambientEnergy`, `energyRegrow`, `energyDiffuse`, `groundPatches`, `reactFeed`, `reactKill`, `fertilise` | uniform, regrowing | `ground`, `forage_ratio` | growth is zero at cap, so a uniform dish at cap produces nothing |
| Competition | none; contested cells share by rate above `uptakeVmax` 0 | | | `forage_ratio` | |
| Cooperation and mutualism | none; a net's transport is the only sharing | | | `demand_mean` | |
| Niches and coexistence | none; the `(uptakeVmax, uptakeKs)` pair is the intended non-dominating trade-off | | | `lines_effective` | |

## Principles

- Few well-tuned mechanics; room for emergence.
- Selection is emergent; no fitness term.
- Nothing true of a kind beyond what the calculus requires.
- Persistence and difference are the point; nothing decays, averages or
  resets, except the eligibility trace, which is a credit window.
- Passive by default: control over anything is acquired, not given.
- Maintained order is paid for: anything held against the physics is work
  drawn from the one energy scalar.
- Every mechanic ships at the setting that reproduces the pond before it.
- A claim about the pond is a run of minutes at the seed budget its measure
  needs, on the pipeline that ships. See `experiments.md`.
