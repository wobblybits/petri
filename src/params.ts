import { FIELD_CELL } from './fields.ts';

export interface Params {
  deposit: number;
  diffuse: number;
  decay: number;
  /**
   * Global gain on how strongly anything smelled moves a body.
   *
   * Stays global, and stays. It looks redundant with the magnitude of a body's
   * own taste weights — two knobs for one quantity — but they are not the same
   * kind of knob. This is a property of the *world*, like `diffuse` or
   * `deposit`: how much chemotaxis drives anything at all here. Taste is a
   * property of a body: what it cares about, relative to what its neighbours
   * care about. Folding this into the seeds would make "smell matters less in
   * this pond" unexpressible without editing every genome in it.
   */
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
  /**
   * Strength of the soft inverse-square push a fully wired agent exerts on
   * agents from *other* nets. Flocking separation only ever applied within a
   * net, so before this nothing pushed separate nets apart at all.
   */
  /**
   * Personal space: a local force holding bodies off each other.
   *
   * Measured to cost reproduction, together with `flockAlign`, and neither
   * alone. Over 45s of a 250-body soup, counting how deep the average lineage
   * gets (`Sim.census().bornMean`):
   *
   *     declutter 1.4, flockAlign 5.5   ->  0.36    (as shipped)
   *     declutter 0,   flockAlign 5.5   ->  0.29
   *     declutter 1.4, flockAlign 0     ->  0.42
   *     declutter 0,   flockAlign 0     ->  6.54
   *
   * Eighteen times the generation depth with both off, and almost nothing from
   * turning off either one. Read as a comparison at equal age, which is what
   * it is — 45 s is mostly warm-up for a preset that drops its whole
   * population in at once as founders, so the absolute figures are far too
   * early to say what any of these settles at. See `Sim.census`. The mechanism is visible in the wire counts: at the
   * shipped values only 2 of 93 wires are principal-to-principal and none are
   * Con-Dup, while 98% of bodies can afford a rewrite and the ground is still
   * at capacity. Bodies are rich and idle — they are not meeting. Personal
   * space holds them apart and alignment turns a head-on approach into a shoal
   * swimming the same way, and a commute needs two principals nose to nose.
   *
   * Left as they are, because these are what make a settled net look settled
   * and a pond look like a pond, and that is a real thing to want. But it is a
   * direct trade against evolution and it should be a decision rather than a
   * surprise.
   */
  declutter: number;
  /**
   * Activity LOD budget. Settled taut islands run the cheap disc+span path
   * even when they are on screen and close up; 0 turns the whole thing off.
   * Live ropes, loners, grabs, rewrites and fresh latches stay NEAR whatever
   * this says — the budget only caps how far a contact can propagate a wake.
   *
   * Off by default, because measured on this branch it costs more than it
   * saves. It does exactly what it claims — a settled 1518-body mesh filling
   * the viewport goes from 95% detailed to 2% — but the frame goes 22.8ms to
   * 25.5ms, and the profiler puts the whole difference in `solve`. The reason
   * is that the two mechanisms overlap almost perfectly: `ropeIsLive` already
   * returns false for any wire whose `ropePath` is 'span', whatever the detail
   * flag says, and a wire turns 'span' under the same aged-and-taut condition
   * that makes its net sleepable. So the expensive half of NEAR is already off
   * before sleep gets a vote, and all sleep can still remove is SAT on settled
   * bodies — which is cheaper than the FAR path it moves them onto.
   *
   * Kept because the mechanism is sound and cheap when off, and because the
   * branch it came from carried per-body audio work that NEAR paid for and
   * this one no longer has. If NEAR ever grows an expensive per-body pass
   * again, this is already here and already tested.
   */
  nearBudget: number;
  /**
   * How hard a wire crossing a principal connection reels its own ends
   * together. Off by default: measured over three minutes of soup it made
   * crossings, clumping and rewrite throughput all slightly worse once
   * `declutter` was in, which already removes the crossings a local force
   * can plausibly undo. Kept as a knob because the detection is the cheap part.
   */
  uncross: number;
  /**
   * How hard a rope pushes off other ropes and off bodies it is not attached
   * to. Segment-based and one-way — a rope never moves an agent — so this
   * cannot feed back into the joint solver.
   */
  wireClear: number;
  /** Port-axis stiffness multiplier. Higher = wires hug their port axis harder. */
  portStiff: number;
  /** Rest-length breathing amplitude, as a fraction. 0 = a settled net freezes. */
  wireBreathe: number;
  /**
   * Seconds a taut wire keeps its rest-shape constraint. 0 = never drop it.
   * Shape only matters while slack is degenerate; a taut rope does not need it.
   */
  wireShapeAge: number;
  /**
   * Seconds before a taut wire becomes span-only (joint, no rope nodes).
   * 0 = never. A leftover that goes slack again gets the full rope back.
   */
  wireSpanAge: number;
  /**
   * Live length / rest at or below which a wire counts as taut for aging.
   * A coarsened wire stays coarsened until it exceeds this by a small band.
   */
  wireTaut: number;
  /**
   * Live length / rest at which a wire tears loose. 0 = wires never break.
   *
   * Well above `wireTaut`, because the span constraint is compliant and a
   * loaded wire legitimately stretches — a threshold near the taut ratio
   * shreds a working net rather than relieving it. Measured over thirty
   * seconds of a 400-body soup, against snapping switched off entirely:
   *
   *   off   281 wires, 340 bodies
   *   4.0   283 wires, 344 bodies   never fires
   *   3.0   277 wires, 338 bodies   fires, costs nothing in aggregate
   *   2.2   205 wires, 260 bodies   a quarter of the pond gone
   *   1.5   126 wires, 220 bodies   shredded
   *
   * 3.0 is the setting that relieves a strained net without dismantling a
   * working one. Below about 2.5 this stops being a safety valve and becomes
   * a second death rate.
   *
   * It closes a loop that already existed with nothing at the end of it:
   * `wireShrink` pulls wired bodies together, and in a crowded net they cannot
   * close the distance, so tension builds and simply stays. Now the most
   * strained link lets go, the ports come free to latch elsewhere, and an
   * over-packed net reconfigures instead of straining forever.
   */
  wireSnap: number;
  /**
   * Energy per second of penetration depth, charged to both bodies in a
   * contact. 0 = collisions are free.
   *
   * Damage rather than death. One rule kills — `extra` reaching the floor —
   * and everything lethal works through the economy, so a bad knock is
   * survivable and a body can recover from it. A separate physics death path
   * would also turn any numerical fault into an extinction: the pond flew
   * apart once already this week from a constant that disagreed across the
   * wasm wall, and that should stay a thing you can watch and diagnose.
   *
   * It is also what gives `flockSep` something to select on. Keeping your
   * distance is now heritable and nothing rewarded it; a cost for crowding
   * lets a lineage work out whether to space out or tolerate the scrum.
   */
  contactCost: number;
  wireMinRest: number;
  wireShrink: number;
  eraMass: number;
  nodeMass: number;
  turnRate: number;
  /**
   * How strongly a fresh body is drawn to full ground, against a reading of 1
   * for a cell at capacity.
   *
   * The seed only. Like `attractStrong` and `attractMedium` this sets where a
   * population starts and is never read again — breeding takes it from there,
   * and a lineage is free to drift to indifference or to outright avoidance,
   * which for food would be a strange thing to become but is not this
   * parameter's business to prevent.
   *
   * Seeds the taste *slope* against `request`, not the flat weight, so a fed
   * body ignores food and a hungry one turns toward it. A flat attraction
   * seems safe — untouched ground is uniform, and a uniform field steers
   * nothing — but a body eats a dip under itself within a frame or two and
   * then chases the dip. See `seedChem`, which has the measurement.
   */
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
  /** Shoaling. See `declutter` for what this costs reproduction, and why the
   *  two only matter together. */
  flockAlign: number;
  flockSep: number;
  maxAgents: number;
  soupCount: number;
  /** Seconds between automatic free-agent spawns (0 = off). */
  spawnInterval: number;
  /**
   * World-space size of one energy cell. Kept a whole multiple of the scent
   * field's cell (10 units) so the two grids line up — an energy cell is a
   * 4x4 block of scent cells rather than a lattice at an unrelated pitch.
   */
  energyCell: number;
  /**
   * Free energy in an unvisited cell. A cell holds a whole extra, so an agent
   * arriving on untouched ground fills in one step and the grid, not the
   * charging rate, is what the net is competing over.
   */
  ambientEnergy: number;
  /**
   * How fast the ground spreads, as a multiple of the scent `diffuse` slider.
   *
   * Small, because energy is not a smell. A signal wants to reach across the
   * dish inside a second — that is what makes a trail worth following. Ground
   * that did the same would be a single shared pool with no local scarcity in
   * it, and nothing to forage toward. This is the number that decides how far
   * a grazed patch can draw on its neighbours, and so how big a dead zone a
   * net can make before it has to move.
   */
  energyDiffuse: number;
  /**
   * Logistic regrowth rate, per second, toward `ambientEnergy` per cell.
   *
   * Not a refill timer. Growth is proportional to what is already in the cell,
   * so a cell taken to exactly zero never comes back on its own and has to be
   * recolonised from a neighbour — grazing to the floor makes a scar that
   * heals from its rim at the speed `energyDiffuse` sets. 0 turns the ground
   * back into the seam of ore it used to be.
   */
  energyRegrow: number;
  /** Extra drained per second. 0 = off. Hitting −1 kills the agent. */
  upkeep: number;
  /**
   * Extra per second per unit of speed, charged for moving.
   *
   * What ties a net's energy to its locomotion, and so what lets a net have a
   * motor at all. Swimming was free, which made thrust a property of a body
   * rather than of the net that feeds it: a starving swimmer swam exactly as
   * hard as a full one, and there was nothing for the transport machinery to
   * be *for* beyond keeping redexes alive.
   *
   * With a price on it the wire network becomes a fuel line. A sub-net that
   * swims runs itself down and asks; `spreadRequests` carries the ask inward
   * and `flowCharges` sends stock back out; `applyTransportRecoil` already
   * kicks the pair as it goes. Which bodies a net chooses to feed is which
   * way it goes — and because `transportThrust`, `transportRecoil` and
   * `requestDecay` are all heritable, what a lineage does with that is
   * something it can evolve rather than something set here.
   *
   * Off by default. Turning it on is a real change to the economy: at a
   * cruise of 38 and an upkeep of 0.015, a cost of 0.0004 roughly doubles
   * what a moving body pays to exist.
   */
  swimCost: number;
  /**
   * Extra per second a body converts into ground, per unit of voice it spends
   * on `CH.energy`. 0 = off.
   *
   * Farming. One for one at the point of transfer — nothing is created here,
   * stock simply moves from a tank onto the dish. What makes it worth doing is
   * what happens next: `grow` is logistic, so growth is proportional to what is
   * already in a cell and **zero is a fixed point**. A cell grazed to the floor
   * can never recover on its own; seeding it with anything at all restarts the
   * growth, and the ground carries it back toward capacity. Investing a little
   * in a scar returns much more than it cost, bounded by the dish's own
   * capacity so it is production rather than a mint.
   *
   * That is what an Era is for under this economy — one port, cannot commute,
   * pays no rent — and why `seedChem` puts an Era's whole unit of voice here.
   * It also means production is now a *phenotype* competing for the same
   * budget as being heard, rather than a rule keyed on kind.
   *
   * Off by default, like every other dial that changes what energy is spent
   * on, and this one wants care: an Era's income from `ERA_UPKEEP_RATIO` is
   * about 0.003/s at the default upkeep, so a rate much above that makes a
   * seeded Era spend faster than it earns and starve.
   */
  farmRate: number;
  /**
   * How loudly a body asks for energy on account of liking where it is.
   *
   * The other half of the motor. A price on swimming alone gives a net a fuel
   * bill; this is what makes the bill *directional*. A body with a free
   * principal reads its own `trail` — everything it can smell, through its own
   * taste weights — and asks in proportion, so the bodies standing where the
   * net most wants to be are the ones that get fed and thrust hardest.
   *
   * Off by default, like the other two dials that change what energy is spent
   * on. Not caution for its own sake: the field takes the largest claim it can
   * see, so an appetite competes directly with `rescueNeed` and `redexNeed` —
   * somebody about to die, and somebody about to reproduce. At 0.05 against
   * the scent a pond makes of itself it already outbid a rescue, topping a
   * dying body past its own `rescueTo` to a full tank while its donors went
   * without. Wanting to go somewhere nice should lose to both of those, and
   * where the crossover sits depends on how loud the pond is, which is four
   * other sliders. Worth tuning by eye rather than guessing a default.
   */
  forageAsk: number;
  /**
   * Fraction of this body's own tank a rescue fills, from `debtCap` at 0 to
   * `energyCap` at 1. The absolute extra it asks up to is
   * `debtCap + rescueTo * (energyCap - debtCap)`, so it cannot land outside
   * the tank.
   *
   * 0 is the old ambulance that stops at break-even once extra is no longer
   * negative. 1 strips the neighbourhood to top the patient off. The seed is
   * a little under a full tank, so a default Con comes out of rescue able to
   * pay a rewrite share without emptying every reservoir it can reach.
   *
   * Heritable: this slider only seeds a fresh body. A commute recombines the
   * two parents' fills; an erase clones the Era's fill with a mutation nudge.
   */
  rescueTo: number;
  /**
   * Extra at which a fresh body dies. Always negative — a debt depth, never
   * a second positive cap. The live value lives on the body as `debtCap` and
   * drifts by breeding; this is only the seed.
   *
   * Deeper debt is more time for a rescue and a louder hunger, and a corpse
   * that can leave nothing. Shallower debt dies sooner and leaves more on
   * the ground.
   */
  debtCap: number;
  /**
   * How much of a body's demand its neighbour hears, per wire. 1 = no decay.
   *
   * Sets how far a shortage is audible — against the field's floor, a whole
   * unit of need carries 20 hops at 0.8 and 43 at 0.9 — and how much crosses,
   * since a transfer is capped by the field at the receiving end. Turn it down
   * for a net of local pools that each look after their own; turn it up for one
   * that answers a shortage anywhere on it.
   *
   * The slider stops short of 1 because an undecayed field is a flat one: every
   * body holds the same need, no neighbour is strictly needier, and transport
   * stops dead.
   *
   * Heritable: this only seeds a fresh body's own `requestDecay`. Once alive,
   * a body relays demand at its own rate, and a Con+Dup commute recombines
   * the two parents' rates into each child (blended for a Con child, one
   * whole parent's rate for a Dup child) — this slider just sets where a new
   * population starts and what a mutation is centred near.
   */
  requestDecay: number;
  /**
   * Momentum a body recoils with per unit of energy it pumps to a neighbour.
   * 0 = off.
   *
   * Stable well past the slider's range — a seeded soup is still calm at 800
   * and only comes apart near 3000. The ceiling is low because the visible
   * events are one-off transfers of most of a unit, and 20 makes one of those
   * a ~56 px/s nudge on an Era against settled speeds around 50.
   *
   * Heritable, like `requestDecay` above: a pump's actual kick is its own
   * `transportRecoil`, seeded from this slider and free to drift by breeding.
   */
  transportRecoil: number;
  /**
   * How much of the receiver's kick is withheld, 0–1, and therefore how much
   * of a pump's recoil survives as motion of the pair.
   *
   * At 0 the pump is equal and opposite and a net can never shift itself by
   * moving energy around inside itself. At 1 only the sender is kicked, so the
   * pair — and any net holding a standing gradient — drifts back along the
   * wire, against the direction the energy is flowing. That is the swimming
   * stroke: pushing charge toward the hungry end pushes the body the other way.
   *
   * Drag bounds it, so this is a cruising speed rather than an acceleration.
   * One whole unit pumped between an Era and a Con at the default recoil
   * leaves the pair drifting `20 * thrust / (0.45 + 1)` px/s — ~7 at the
   * default, ~14 at 1 — against settled speeds around 40. In a live soup that
   * is invisible for the same reason the recoil gain is: mean speed over a
   * seeded 30 s soup at 0 / 0.25 / 0.5 / 1 is 40 / 60 / 34 / 47 px/s, which is
   * seed noise. It reads on the events, and on a net actually holding a
   * gradient — drive one end full and the other hungry and the chain visibly
   * runs away from its own supply.
   *
   * Heritable, like the two above. A transfer reads the *sender's* own
   * `transportRecoil` and the *receiver's* own `transportThrust`, so breeding
   * a strong low-thrust pump against a high-thrust receiver can drift a
   * net's stroke somewhere neither parent line swims alone.
   */
  transportThrust: number;
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
    nearBudget: 0,
    uncross: 0,
    wireClear: 1,
    portStiff: 2,
    wireBreathe: 0.04,
    wireShapeAge: 2,
    wireSpanAge: 10,
    wireTaut: 1.08,
    wireSnap: 3,
    contactCost: 0,
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
    flockAlign: 5.5,
    flockSep: 48,
    maxAgents: 100000,
    soupCount: 10000,
    spawnInterval: 0.5,
    energyCell: 40,
    ambientEnergy: 1.0,
    energyDiffuse: 0.05,
    energyRegrow: 0.04,
    upkeep: 0.015,
    swimCost: 0,
    farmRate: 0,
    forageAsk: 0,
    rescueTo: 0.9,
    debtCap: -1,
    requestDecay: 0.95,
    transportRecoil: 100,
    transportThrust: 1.0,
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
  // Seeds, not settings. These four are read once by `seedChem` and never
  // again — they decide what a *newly spawned* body starts as, and moving them
  // does nothing to anything already alive. Everything above and below changes
  // the world continuously.
  { key: 'attractStrong', label: 'Strong attract (seed)', min: 0, max: 3, step: 0.05 },
  { key: 'attractMedium', label: 'Medium attract (seed)', min: 0, max: 2, step: 0.05 },
  { key: 'attractFood', label: 'Food attract (seed)', min: -2, max: 4, step: 0.05 },
  { key: 'sensorAngle', label: 'Sensor arc', min: 0.1, max: 1.2, step: 0.02 },
  { key: 'stepSpeed', label: 'Step speed', min: 10, max: 180, step: 1 },
  { key: 'swimTau', label: 'Swim persistence', min: 0.1, max: 4, step: 0.05 },
  { key: 'swimNoise', label: 'Swim noise', min: 0, max: 2, step: 0.05 },
  { key: 'drag', label: 'Fluid drag', min: 0, max: 4, step: 0.05 },
  { key: 'angDrag', label: 'Spin damp', min: 0, max: 8, step: 0.05 },
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
  { key: 'nearBudget', label: 'NEAR budget', min: 0, max: 2000, step: 25 },
  { key: 'uncross', label: 'Uncross', min: 0, max: 4, step: 0.05 },
  { key: 'wireClear', label: 'Wire clearance', min: 0, max: 4, step: 0.05 },
  { key: 'wireBreathe', label: 'Wire breathe', min: 0, max: 0.15, step: 0.005 },
  { key: 'wireShapeAge', label: 'Shape drop (s)', min: 0, max: 30, step: 0.1 },
  { key: 'wireSpanAge', label: 'Span-only (s)', min: 0, max: 60, step: 0.5 },
  { key: 'wireTaut', label: 'Taut ratio', min: 1, max: 1.5, step: 0.01 },
  { key: 'wireSnap', label: 'Wire snap ratio', min: 0, max: 6, step: 0.1 },
  { key: 'contactCost', label: 'Impact cost', min: 0, max: 0.2, step: 0.005 },
  { key: 'eraMass', label: 'Era mass', min: 0.15, max: 2, step: 0.05 },
  { key: 'nodeMass', label: 'Con/Dup mass', min: 0.3, max: 4, step: 0.05 },
  { key: 'maxAgents', label: 'Max agents', min: 8, max: 8000, step: 1 },
  { key: 'spawnInterval', label: 'Auto spawn (s)', min: 0, max: 30, step: 0.5 },
  // Step is a whole FIELD_CELL: an energy cell has to stay a multiple of the
  // scent field's cell for the two grids to line up (see EnergyGrid).
  { key: 'energyCell', label: 'Energy cell', min: FIELD_CELL, max: 160, step: FIELD_CELL },
  { key: 'ambientEnergy', label: 'Ambient energy', min: 0, max: 2, step: 0.05 },
  { key: 'energyDiffuse', label: 'Ground spread', min: 0, max: 0.5, step: 0.005 },
  { key: 'energyRegrow', label: 'Ground regrow', min: 0, max: 0.4, step: 0.005 },
  { key: 'upkeep', label: 'Upkeep', min: 0, max: 0.2, step: 0.005 },
  { key: 'swimCost', label: 'Swim cost', min: 0, max: 0.002, step: 0.00005 },
  { key: 'farmRate', label: 'Farm rate', min: 0, max: 0.02, step: 0.0005 },
  { key: 'forageAsk', label: 'Forage ask', min: 0, max: 0.5, step: 0.01 },
  { key: 'rescueTo', label: 'Rescue fill', min: 0, max: 1, step: 0.05 },
  { key: 'debtCap', label: 'Debt cap', min: -2.5, max: -0.05, step: 0.05 },
  { key: 'requestDecay', label: 'Demand decay', min: 0.5, max: 0.98, step: 0.01 },
  { key: 'transportRecoil', label: 'Pump recoil (seed)', min: 0, max: 200, step: 5 },
  { key: 'transportThrust', label: 'Pump thrust (seed)', min: 0, max: 1, step: 0.05 },
];
