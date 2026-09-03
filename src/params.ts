export interface Params {
  deposit: number;
  diffuse: number;
  decay: number;
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
  declutter: number;
  /**
   * How hard a wire crossing a principal connection reels its own ends
   * together. Off by default: measured over three minutes of soup it made
   * crossings, clumping and rewrite throughput all slightly worse once
   * `declutter` was in, which already removes the crossings a local force
   * can plausibly undo. Kept as a knob because the detection is the cheap part.
   */
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
   * Pull back toward home for a body outside the world bound, per world unit
   * of overshoot per second. 0 = off, and the pond is unbounded again.
   */
  edgePull: number;
  wireMinRest: number;
  wireShrink: number;
  eraMass: number;
  nodeMass: number;
  turnRate: number;
  sensorAngle: number;
  sensorDist: number;
  stepSpeed: number;
  /** Persistence time of self-propulsion, in seconds. Longer = smoother runs. */
  swimTau: number;
  /** Self-propulsion noise, as a fraction of cruise speed. 0 = a flat setpoint. */
  swimNoise: number;
  drag: number;
  angDrag: number;
  gravity: number;
  flockAlign: number;
  flockSep: number;
  maxAgents: number;
  soupCount: number;
  /** Seconds between automatic free-agent spawns (0 = off). */
  spawnInterval: number;
  /** World-space size of one energy cell. */
  energyCell: number;
  /**
   * Free energy in an unvisited cell. A cell holds a whole extra, so an agent
   * arriving on untouched ground fills in one step and the grid, not the
   * charging rate, is what the net is competing over.
   */
  ambientEnergy: number;
  /** Extra drained per second. 0 = off. Hitting −1 kills the agent. */
  upkeep: number;
  /**
   * How full a body that has been in debt is fed back up to before it stops
   * asking. 0 restores the old behaviour, where a rescue stopped at break-even.
   *
   * At the default — one whole share — a rescue is a refill: the body comes out
   * of it able to pay for a rewrite, which is what makes a surplus at one end
   * of a net actually drain toward a starving end instead of trickling out the
   * few hundredths needed to keep it at exactly zero. Above `REWRITE_SHARE` it
   * eats into the headroom too and a starving neighbourhood will strip a
   * reservoir bare; below it the patient is discharged unable to act.
   */
  rescueTo: number;
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
    deposit: 2.4,
    diffuse: 0.28,
    decay: 0.018,
    sense: 520,
    attractStrong: 1.45,
    attractMedium: 0.72,
    snapRadius: 22,
    snapArc: 0.3,
    snapWell: 48,
    faceRadius: 90,
    faceAttract: 32,
    rewriteDuration: 0.7,
    springK: 12,
    springDamp: 45,
    auxSpread: 1.7,
    declutter: 1,
    nearBudget: 0,
    uncross: 0,
    wireClear: 1,
    portStiff: 2,
    wireBreathe: 0.04,
    wireShapeAge: 2,
    wireSpanAge: 10,
    wireTaut: 1.08,
    edgePull: 0.35,
    wireMinRest: 40,
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
    gravity: 0,
    flockAlign: 5.5,
    flockSep: 36,
    maxAgents: 10000,
    soupCount: 2500,
    spawnInterval: 0.5,
    energyCell: 48,
    ambientEnergy: 2,
    upkeep: 0.015,
    rescueTo: 1,
    requestDecay: 0.95,
    transportRecoil: 50,
    transportThrust: 0.5,
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
  { key: 'attractStrong', label: 'Strong attract', min: 0, max: 3, step: 0.05 },
  { key: 'attractMedium', label: 'Medium attract', min: 0, max: 2, step: 0.05 },
  { key: 'sensorAngle', label: 'Sensor arc', min: 0.1, max: 1.2, step: 0.02 },
  { key: 'stepSpeed', label: 'Step speed', min: 10, max: 180, step: 1 },
  { key: 'swimTau', label: 'Swim persistence', min: 0.1, max: 4, step: 0.05 },
  { key: 'swimNoise', label: 'Swim noise', min: 0, max: 2, step: 0.05 },
  { key: 'gravity', label: 'Gravity', min: 0, max: 0.8, step: 0.01 },
  { key: 'drag', label: 'Fluid drag', min: 0, max: 4, step: 0.05 },
  { key: 'angDrag', label: 'Spin damp', min: 0, max: 8, step: 0.05 },
  { key: 'flockAlign', label: 'Flock align', min: 0, max: 16, step: 0.1 },
  { key: 'flockSep', label: 'Flock separate', min: 0, max: 120, step: 1 },
  { key: 'snapRadius', label: 'Snap reach', min: 4, max: 48, step: 1 },
  { key: 'snapArc', label: 'Snap arc', min: 0.08, max: 1.2, step: 0.02 },
  { key: 'wireShrink', label: 'Wire shrink', min: 0.1, max: 3, step: 0.05 },
  { key: 'edgePull', label: 'Edge pull', min: 0, max: 2, step: 0.05 },
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
  { key: 'eraMass', label: 'Era mass', min: 0.15, max: 2, step: 0.05 },
  { key: 'nodeMass', label: 'Con/Dup mass', min: 0.3, max: 4, step: 0.05 },
  { key: 'maxAgents', label: 'Max agents', min: 8, max: 8000, step: 1 },
  { key: 'spawnInterval', label: 'Auto spawn (s)', min: 0, max: 30, step: 0.5 },
  { key: 'energyCell', label: 'Energy cell', min: 16, max: 160, step: 1 },
  { key: 'ambientEnergy', label: 'Ambient energy', min: 0, max: 2, step: 0.05 },
  { key: 'upkeep', label: 'Upkeep', min: 0, max: 0.2, step: 0.005 },
  { key: 'rescueTo', label: 'Rescue to', min: 0, max: 1.25, step: 0.05 },
  { key: 'requestDecay', label: 'Demand decay', min: 0.5, max: 0.98, step: 0.01 },
  { key: 'transportRecoil', label: 'Pump recoil', min: 0, max: 200, step: 5 },
  { key: 'transportThrust', label: 'Pump thrust', min: 0, max: 1, step: 0.05 },
];
