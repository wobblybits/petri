import { FIELD_CELL } from './fields.ts';
import { BODY_VALUE, ERA_CAP_RATIO, ERA_UPKEEP_RATIO } from './energy.ts';
import { SENSE_SCALE } from './chem-layout.ts';

export interface Params {
  deposit: number;
  diffuse: number;
  decay: number;
  /** Global gain on how strongly anything smelled moves a body. A property of
   *  the world, like `diffuse`; a body's taste weights say what it cares about. */
  sense: number;
  attractStrong: number;
  attractMedium: number;
  snapRadius: number;
  snapArc: number;
  snapWell: number;
  faceRadius: number;
  faceAttract: number;
  rewriteDuration: number;
  springK: number;
  springDamp: number;
  /** How far an aux port aims off its neighbour, toward its own side. 1 = 20 degrees. */
  auxSpread: number;
  /** Personal space: a local force holding bodies off each other. With
   *  `flockAlign` it keeps principals apart and costs reproduction
   *  (`Sim.census().bornMean`); neither alone does. */
  declutter: number;
  /** How hard a rope pushes off other ropes and bodies it is not attached to.
   *  One-way — a rope never moves an agent — so it cannot feed back into the
   *  joint solver. */
  wireClear: number;
  /** Port-axis stiffness multiplier. Higher = wires hug their port axis harder. */
  portStiff: number;
  /** Rest-length breathing amplitude, as a fraction. 0 = a settled net freezes. */
  wireBreathe: number;
  /** Seconds a taut wire keeps its rest-shape constraint. 0 = never drop it. */
  wireShapeAge: number;
  /** Seconds before a taut wire becomes span-only (joint, no rope nodes). 0 = never. */
  wireSpanAge: number;
  /** Live length / rest at or below which a wire counts as taut for aging. A
   *  coarsened wire stays coarsened until it exceeds this by a small band. */
  wireTaut: number;
  /** Live length / rest at which a wire tears loose. 0 = wires never break.
   *  Keep it well above `wireTaut`: a loaded wire legitimately stretches, and
   *  a threshold near the taut ratio shreds a working net. */
  wireSnap: number;
  wireMinRest: number;
  wireShrink: number;
  eraMass: number;
  nodeMass: number;
  turnRate: number;
  /** How strongly a fresh body is drawn to full ground, against a reading of 1
   *  for a cell at capacity. Seed only, read once by `seedChem`. Seeds the
   *  taste slope against `request`, so a fed body ignores food and a hungry
   *  one turns toward it. */
  attractFood: number;
  sensorAngle: number;
  sensorDist: number;
  stepSpeed: number;
  /** Persistence time of self-propulsion, in seconds. Longer = smoother runs. */
  swimTau: number;
  /** Self-propulsion noise, as a fraction of cruise speed. 0 = a flat setpoint. */
  swimNoise: number;
  drag: number;
  angDrag: number;
  /** Drag added to `drag` in proportion to how full a body is, per second.
   *  0 = one drag law for every body. Positive: full grips, empty slides, so a
   *  net walks toward its hungry end; negative: the reverse. The sum is clamped
   *  at zero, so a rate can be cancelled but never inverted. A stroke needs a
   *  difference in fullness across a wire, so a topped-up pond does not move. */
  grip: number;
  /** How fast the pathway runs, as a plain multiple. 0 = no metabolism, no
   *  gait, no pathway spend (`advanceGait` returns with `gaitWave` and `anchor`
   *  at zero). Rescales time only: moves the period, not where it oscillates. */
  metabolicRate: number;
  /** Substrate a discharged body pulls out of its own tank, per second at full
   *  discharge. It lands on the ground through the `excreteRate` path. */
  metabolicSupply: number;
  /** The pathway's basal rate, with no ADP to activate it. Must be non-zero:
   *  the autocatalytic step is `sub * adp^2`, so a fully charged body would
   *  otherwise stop metabolising forever. */
  metabolicBase: number;
  /** How fast ADP is recharged to ATP, per second. Oscillates only in a narrow
   *  band just above `metabolicSupply / adenylate`: below it every body sits at
   *  the discharged clamp, far above it the pool sits full and still. Needs
   *  `metabolicWork` non-zero, and `adenylate` is heritable, so it is per body. */
  metabolicRegen: number;
  /** ATP spent per unit of stroke, per second. What makes moving cost something. */
  metabolicWork: number;
  /** Energy debited from the tank per unit of substrate the pathway buys.
   *  0 = metabolism is free. Converts reaction units to tank units. */
  metabolicCost: number;
  /** How fast the upstream metabolite crosses a wire, per second. 0 = every
   *  body's pathway is its own. */
  metabolicDiffuse: number;
  /** What a fresh body's adenylate pool starts at. Heritable from there. */
  adenylate: number;
  /** How far the gait swings a wire's rest length, as a fraction, on top of
   *  `wireBreathe`. 0 = the wire ignores the clock. Shape only: the impulse in
   *  `Sim.strokeWires` is what moves a net's centre of mass. */
  gaitSwell: number;
  /** Shoaling. See `declutter` for what the two together cost reproduction. */
  flockAlign: number;
  flockSep: number;
  maxAgents: number;
  soupCount: number;
  /** Seconds between automatic free-agent spawns (0 = off). */
  spawnInterval: number;
  /** World-space size of one energy cell. Must stay a whole multiple of the
   *  scent field's cell (`FIELD_CELL`) so the two grids line up. */
  energyCell: number;
  /** Free energy in an unvisited cell. A cell holds a whole extra. */
  ambientEnergy: number;
  /** Gray-Scott feed and kill, `u + 2v -> 3v` on `CH.conP` (substrate) and
   *  `CH.dupP` (activator). Both 0 = off. `CH.energy` is never a reactant.
   *  Washes flat unless the substrate diffuses at about twice the activator's
   *  rate (`diffuseRate`) and the pair sits roughly in F [0.01, 0.09] against
   *  k [0.045, 0.07]. `decay` still acts, so effective kill is `decay + reactKill`. */
  reactFeed: number;
  reactKill: number;
  /** How fast the ground spreads, as a multiple of the scent `diffuse` slider.
   *  Small: ground that reached across the dish would have no local scarcity. */
  energyDiffuse: number;
  /** Logistic regrowth rate, per second, toward `ambientEnergy` per cell.
   *  Proportional to what is there, so a cell at exactly zero never comes back
   *  on its own and must be recolonised via `energyDiffuse`. 0 = no regrowth. */
  energyRegrow: number;
  /** How much the fertiliser channel (`FERTILISE_CH`, `C`) accelerates regrowth,
   *  per unit: growth is `r * (1 + fertilise * C) * E * (1 - E/K)`. Negative
   *  inhibits; the rate is clamped at zero, so ground is never destroyed.
   *  0 = off, and it scales `energyRegrow`, so does nothing while that is 0. */
  fertilise: number;
  /** Extra drained per second. 0 = off. Hitting −1 kills the agent. */
  upkeep: number;
  /** Fraction of this body's own tank a rescue fills, from `debtCap` at 0 to
   *  `energyCap` at 1, so it cannot land outside the tank. Heritable; seed only. */
  rescueTo: number;
  /** What a fresh body's `assort` starts at: how likely each gene of its
   *  children is copied whole from one parent rather than blended.
   *  `assortChance` offsets by kind, so at 0.5 a Con child blends every gene
   *  and a Dup child assorts every gene. Seed only, read once by `createAgent`. */
  assortBias: number;
  /** Extra at which a fresh body dies. Always negative — a debt depth, never a
   *  second positive cap. Seed only; the live `debtCap` drifts by breeding. */
  debtCap: number;
  /** How much of a body's demand its neighbour hears, per wire. Must stay
   *  below 1: an undecayed field is flat, no neighbour is strictly needier, and
   *  transport stops dead. Heritable; seeds a fresh body's own `requestDecay`. */
  requestDecay: number;
  /** Hops the need field advances per frame. 0 = solved to a fixpoint every
   *  frame, so a shortage is known everywhere the frame it arises. Above 0 a
   *  need takes a frame per wire. Same fixpoint, but triage turns local: a
   *  corridor of empty bodies absorbs a reservoir before it reaches the dying
   *  body (see the corridor in `energy.test.ts`). */
  requestReach: number;
  /** Momentum a body recoils with per unit of energy it pumps to a neighbour.
   *  0 = off. Heritable; seed only. */
  transportRecoil: number;
  /** Whole units a transfer moves along a wire. 0 = the continuous law. Above
   *  zero a transfer is a whole unit or nothing, only a body holding one can
   *  send, and what will not fit in the receiver lands on the ground under it
   *  (the second path that can overfill a body, after `payUpkeep`). Heritable;
   *  seed only, read off the sender. Keep it well under `REWRITE_SHARE`: bodies
   *  all holding less than a packet cannot feed each other at all. */
  transportQuantum: number;
  /** How fast a body's state matrices change while it is alive. 0 = off.
   *  Three-factor: eligibility trace times the body's own critic's TD error,
   *  all local. What is learned is kept for life and travels with the body. */
  learnRate: number;
  /** How fast the critic itself learns to predict. Its own delta rule. */
  learnCritic: number;
  /** Eligibility trace decay, per frame. Not weight decay — nothing here forgets.
   *  The credit window: how far back a weight is held responsible for the tank. */
  learnTrace: number;
  /** Discount on the critic's own prediction, per frame. */
  learnDiscount: number;
  /** How much of what a parent learned is consolidated into its children's
   *  genome, 0 to 1. At 1 a commute's children start from `chem + plastic`; at
   *  0 learning dies with the body. Children always start with an empty slate. */
  inheritLearned: number;

  /* Chemistry. */

  /** The whole mouthful a body may swallow per second, at saturation: one
   *  budget across all four species in proportion to what stands in the cell
   *  (`UptakeKinetics`), landing in the gut. 0 is a different path, not "no
   *  uptake": `runHarvestPlan` takes what fits, ground alone, straight into the tank. */
  uptakeVmax: number;
  /** Half-saturation constant: the ground density at which uptake runs at half `uptakeVmax`. */
  uptakeKs: number;
  /** Fixed cost, per second, of expressing a reaction row at all. Two rows cost
   *  `2c`: the superadditivity that lets specialising beat splitting under Monod. */
  rowCost: number;
  /** Hill coefficient on uptake. 1 = plain Monod; above 1 the response is
   *  convex at low expression, the other superadditivity dial. */
  hillN: number;
  /** Yield on energy a body takes up directly from the ground, 0 to 1. Toward
   *  0 a fresh spawn has about `EXTRA_CAP / upkeep` seconds to meet a net, or
   *  obligate dependency kills the soup rather than structuring it. */
  yDirect: number;
  /** Yield on energy an Era takes up, 0 to 1, normally above `yDirect`. Eras
   *  are a net's boundary and upkeep is per body: surface-to-volume. */
  yEra: number;
  /** Fraction of ordinary upkeep put back into the field rather than destroyed,
   *  0 to 1, as the body's own excretion mix through `payOut`; the pathway and
   *  the row cost leave the same way. At 1 no reaction creates or destroys
   *  matter. Above zero it also turns `refreshExpression` on. */
  upkeepExcrete: number;
  /** What a body's existence is worth when it dies. At `REWRITE_SHARE` the
   *  commute-then-annihilate cycle stops minting `2 * (bodyValue - REWRITE_SHARE)`. */
  bodyValue: number;
  /** How much more an Era holds than a Con or a Dup. 1 drops the rule;
   *  `energyCap` is heritable, so storage is evolvable without it. */
  eraCapRatio: number;
  /** A producer's upkeep as a multiple of everyone else's; negative means it
   *  earns. `upkeepRateOf` interpolates on how much of a body's chemical budget
   *  goes on the ground row, so a seeded Era lands here and a seeded Con on 1. */
  eraUpkeepRatio: number;
  /** Rate at which a body puts out its excretion rows, per second per unit held
   *  in gut and tank, for a row at an eighth; gut contents leave first and free,
   *  the shortfall is synthesised from the tank. An Era's ground row makes this
   *  the farming dial too. 0 = the minted path: `effEmit` times `params.deposit`,
   *  nothing out of the tank, `CH.energy` firewalled out. Above zero the mint
   *  stops and all four species leave conserved; signal amplitude drops by about
   *  the deposit multiplier, so `senseScale` must be remeasured, not rescaled. */
  excreteRate: number;
  /** What one unit of a signal reading is worth on the way into `x`. Defaults
   *  to `SENSE_SCALE`, the p90 field reading at a body's own position in the
   *  minted-deposit pond; under `excreteRate` > 0 that reading is orders of
   *  magnitude smaller and the sense genes are below what `phi` can resolve. */
  senseScale: number;
  /** Units of ground one unit of a signalling species is converted with, out of
   *  the body's own gut (`Sim.runDigestion`). 0 = an uptake row eats its species
   *  raw. Above it species 0, 1 and 3 convert only paired with swallowed ground,
   *  one budget across the three. Access, never amplification: the ground spent
   *  lands in the tank alongside what it unlocked. */
  catCoSubstrate: number;
  /** How fast the gut turns into the tank, per second, per species. Ground
   *  converts flat; the other three scaled by the body's uptake row and paired
   *  via `catCoSubstrate`. Bounded by room in the tank, so a fed body stops
   *  digesting, its gut fills and it stops eating. Inert at `uptakeVmax` 0. */
  digestRate: number;
  /** How much a body can hold undigested, as a multiple of its `energyCap`.
   *  What a body cannot convert sits here and shrinks the next mouthful. */
  gutSize: number;
  /** How many patches the ground is laid down in, at the same total mass.
   *  0 = the whole disk. Read once at setup by `pond/ground.ts`; a `Params`
   *  field so it can be a sweep axis. For less food, move `ambientEnergy`. */
  groundPatches: number;
}

export function defaultParams(): Params {
  return {
    deposit: 5,
    diffuse: 0.6,
    decay: 0.01,
    sense: 520,
    attractStrong: 1.45,
    attractMedium: 0.72,
    attractFood: 0.9,
    snapRadius: 22,
    snapArc: 0.3,
    snapWell: 48,
    faceRadius: 90,
    faceAttract: 32,
    rewriteDuration: 0.7,
    springK: 12,
    springDamp: 45,
    auxSpread: 1.7,
    declutter: 1.4,
    wireClear: 1,
    portStiff: 2,
    wireBreathe: 0.04,
    wireShapeAge: 2,
    wireSpanAge: 10,
    wireTaut: 1.08,
    wireSnap: 3,
    wireMinRest: 48,
    wireShrink: 0.2,
    eraMass: 0.45,
    nodeMass: 1,
    turnRate: 1.6,
    sensorAngle: 0.48,
    sensorDist: 24,
    stepSpeed: 38,
    swimTau: 1.1,
    swimNoise: 0.35,
    drag: 0.55,
    angDrag: 2.4,
    grip: 2,
    metabolicRate: 0,
    metabolicSupply: 3,
    metabolicBase: 0.02,
    metabolicRegen: 2.3,
    metabolicWork: 0.6,
    metabolicCost: 0.01,
    metabolicDiffuse: 2,
    adenylate: 1.5,
    gaitSwell: 0.3,
    flockAlign: 5.5,
    flockSep: 48,
    maxAgents: 100000,
    soupCount: 10000,
    spawnInterval: 0.5,
    energyCell: 40,
    ambientEnergy: 1.0,
    reactFeed: 0,
    reactKill: 0,
    energyDiffuse: 0.05,
    energyRegrow: 0.04,
    fertilise: 0,
    upkeep: 0.015,
    rescueTo: 0.9,
    assortBias: 0.5,
    debtCap: -1,
    requestDecay: 0.95,
    requestReach: 0,
    transportRecoil: 100,
    transportQuantum: 0.5,
    learnRate: 0.02,
    learnCritic: 0.2,
    learnTrace: 0.99,
    learnDiscount: 0.99,
    inheritLearned: 1,
    uptakeVmax: 0,
    uptakeKs: 0.25,
    rowCost: 0,
    hillN: 1,
    yDirect: 1,
    yEra: 1,
    upkeepExcrete: 0,
    bodyValue: BODY_VALUE,
    eraCapRatio: ERA_CAP_RATIO,
    eraUpkeepRatio: ERA_UPKEEP_RATIO,
    excreteRate: 0,
    senseScale: SENSE_SCALE,
    catCoSubstrate: 0,
    digestRate: 12,
    gutSize: 1,
    groundPatches: 0,
  };
}

export interface SliderSpec {
  key: keyof Params;
  label: string;
  min: number;
  max: number;
  step: number;
}

export const SLIDERS: SliderSpec[] = [
  { key: 'deposit', label: 'Deposit', min: 0, max: 6, step: 0.05 },
  { key: 'diffuse', label: 'Diffuse', min: 0, max: 1, step: 0.01 },
  { key: 'decay', label: 'Decay', min: 0, max: 0.08, step: 0.001 },
  { key: 'sense', label: 'Sense', min: 0, max: 1200, step: 10 },
  // Seeds, not settings: read once by `seedChem` for a newly spawned body;
  // moving them does nothing to anything already alive.
  { key: 'attractStrong', label: 'Strong attract (seed)', min: 0, max: 3, step: 0.05 },
  { key: 'attractMedium', label: 'Medium attract (seed)', min: 0, max: 2, step: 0.05 },
  { key: 'attractFood', label: 'Food attract (seed)', min: -2, max: 4, step: 0.05 },
  { key: 'sensorAngle', label: 'Sensor arc', min: 0.1, max: 1.2, step: 0.02 },
  { key: 'stepSpeed', label: 'Step speed', min: 10, max: 180, step: 1 },
  { key: 'swimTau', label: 'Swim persistence', min: 0.1, max: 4, step: 0.05 },
  { key: 'swimNoise', label: 'Swim noise', min: 0, max: 2, step: 0.05 },
  { key: 'drag', label: 'Fluid drag', min: 0, max: 4, step: 0.01 },
  { key: 'angDrag', label: 'Spin damp', min: 0, max: 8, step: 0.05 },
  { key: 'grip', label: 'Grip (tank)', min: -4, max: 12, step: 0.05 },
  { key: 'metabolicRate', label: 'Metabolic rate', min: 0, max: 20, step: 0.1 },
  { key: 'metabolicSupply', label: 'Substrate pull', min: 0, max: 6, step: 0.05 },
  { key: 'metabolicBase', label: 'Basal enzyme', min: 0, max: 2, step: 0.01 },
  { key: 'metabolicRegen', label: 'Recharge rate', min: 0.05, max: 6, step: 0.05 },
  { key: 'metabolicWork', label: 'Stroke cost', min: 0, max: 4, step: 0.02 },
  { key: 'metabolicCost', label: 'Substrate price', min: 0, max: 0.1, step: 0.002 },
  { key: 'adenylate', label: 'Adenylate pool (seed)', min: 0.2, max: 6, step: 0.1 },
  { key: 'metabolicDiffuse', label: 'Activator spread', min: 0, max: 20, step: 0.1 },
  { key: 'gaitSwell', label: 'Gait swell', min: 0, max: 0.8, step: 0.01 },
  { key: 'flockAlign', label: 'Flock align (seed)', min: 0, max: 16, step: 0.1 },
  { key: 'flockSep', label: 'Flock separate (seed)', min: 0, max: 120, step: 1 },
  { key: 'snapRadius', label: 'Snap reach', min: 4, max: 48, step: 1 },
  { key: 'snapArc', label: 'Snap arc', min: 0.08, max: 1.2, step: 0.02 },
  { key: 'wireShrink', label: 'Wire shrink', min: 0.1, max: 3, step: 0.05 },
  { key: 'wireMinRest', label: 'Wire min length', min: 8, max: 48, step: 1 },
  { key: 'springK', label: 'Spring stiffness', min: 0, max: 80, step: 0.5 },
  { key: 'springDamp', label: 'Rope damp', min: 0, max: 120, step: 1 },
  { key: 'portStiff', label: 'Port stiffness', min: 0.1, max: 4, step: 0.05 },
  { key: 'auxSpread', label: 'Aux spread', min: 0, max: 3, step: 0.05 },
  { key: 'declutter', label: 'Personal space', min: 0, max: 4, step: 0.05 },
  { key: 'wireClear', label: 'Wire clearance', min: 0, max: 4, step: 0.05 },
  { key: 'wireBreathe', label: 'Wire breathe', min: 0, max: 0.15, step: 0.005 },
  { key: 'wireShapeAge', label: 'Shape drop (s)', min: 0, max: 30, step: 0.1 },
  { key: 'wireSpanAge', label: 'Span-only (s)', min: 0, max: 60, step: 0.5 },
  { key: 'wireTaut', label: 'Taut ratio', min: 1, max: 1.5, step: 0.01 },
  { key: 'wireSnap', label: 'Wire snap ratio', min: 0, max: 6, step: 0.1 },
  { key: 'eraMass', label: 'Era mass', min: 0.15, max: 2, step: 0.05 },
  { key: 'nodeMass', label: 'Con/Dup mass', min: 0.3, max: 4, step: 0.05 },
  { key: 'maxAgents', label: 'Max agents', min: 8, max: 8000, step: 1 },
  { key: 'spawnInterval', label: 'Auto spawn (s)', min: 0, max: 30, step: 0.5 },
  // Step is a whole FIELD_CELL: an energy cell must stay a multiple of the
  // scent field's cell for the two grids to line up (see EnergyGrid).
  { key: 'energyCell', label: 'Energy cell', min: FIELD_CELL, max: 160, step: FIELD_CELL },
  { key: 'ambientEnergy', label: 'Ambient energy', min: 0, max: 2, step: 0.05 },
  { key: 'reactFeed', label: 'React feed', min: 0, max: 0.1, step: 0.001 },
  { key: 'reactKill', label: 'React kill', min: 0, max: 0.08, step: 0.001 },
  { key: 'energyDiffuse', label: 'Ground spread', min: 0, max: 0.5, step: 0.005 },
  { key: 'energyRegrow', label: 'Ground regrow', min: 0, max: 0.4, step: 0.005 },
  { key: 'fertilise', label: 'Fertilise', min: -2, max: 8, step: 0.1 },
  { key: 'upkeep', label: 'Upkeep', min: 0, max: 0.2, step: 0.005 },
  { key: 'rescueTo', label: 'Rescue fill', min: 0, max: 1, step: 0.05 },
  { key: 'assortBias', label: 'Assortment (seed)', min: 0, max: 1, step: 0.05 },
  { key: 'debtCap', label: 'Debt cap', min: -2.5, max: -0.05, step: 0.05 },
  { key: 'requestDecay', label: 'Demand decay', min: 0.5, max: 0.98, step: 0.01 },
  { key: 'requestReach', label: 'Demand hops/frame', min: 0, max: 12, step: 1 },
  { key: 'transportRecoil', label: 'Pump recoil (seed)', min: 0, max: 200, step: 5 },
  { key: 'transportQuantum', label: 'Transport quantum', min: 0, max: 1, step: 0.05 },
  { key: 'learnRate', label: 'Learn rate', min: 0, max: 0.02, step: 0.0005 },
  { key: 'learnCritic', label: 'Learn critic', min: 0, max: 0.2, step: 0.005 },
  { key: 'learnTrace', label: 'Learn trace decay', min: 0.5, max: 0.995, step: 0.005 },
  { key: 'learnDiscount', label: 'Learn discount', min: 0.5, max: 0.995, step: 0.005 },
  { key: 'inheritLearned', label: 'Inherit learned', min: 0, max: 1, step: 0.05 },
  { key: 'uptakeVmax', label: 'Uptake rate', min: 0, max: 4, step: 0.05 },
  { key: 'uptakeKs', label: 'Uptake half-sat', min: 0.01, max: 2, step: 0.01 },
  { key: 'rowCost', label: 'Expression row cost', min: 0, max: 0.02, step: 0.0005 },
  { key: 'hillN', label: 'Hill coefficient', min: 1, max: 4, step: 0.1 },
  { key: 'yDirect', label: 'Direct uptake yield', min: 0, max: 1, step: 0.05 },
  { key: 'yEra', label: 'Era uptake yield', min: 0, max: 4, step: 0.05 },
  { key: 'upkeepExcrete', label: 'Upkeep excretes', min: 0, max: 1, step: 0.05 },
  { key: 'bodyValue', label: 'Body value', min: 0.5, max: 2, step: 0.05 },
  { key: 'eraCapRatio', label: 'Era tank ratio', min: 1, max: 4, step: 0.1 },
  { key: 'eraUpkeepRatio', label: 'Era upkeep ratio', min: -1, max: 2, step: 0.05 },
  { key: 'excreteRate', label: 'Excrete rate', min: 0, max: 2, step: 0.02 },
  { key: 'senseScale', label: 'Sense scale', min: 0.001, max: 8, step: 0.001 },
  { key: 'catCoSubstrate', label: 'Catabolism needs ground', min: 0, max: 1, step: 0.05 },
  { key: 'digestRate', label: 'Digest rate', min: 0, max: 40, step: 0.5 },
  { key: 'gutSize', label: 'Gut size', min: 0.1, max: 4, step: 0.1 },
  { key: 'groundPatches', label: 'Ground patches', min: 0, max: 128, step: 1 },
];
