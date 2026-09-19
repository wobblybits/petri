import { GW_BASE, GX_BASE, SW_BASE, TX_A, TX_B, TX_BASE, TX_C, TX_D, exploreKey } from './chem-layout.ts';
import { REACT_B, REACT_C, REACT_D, REACT_SPECIES } from './agent-store.ts';
import {
  boundRadius,
  discRadius,
  createAgent,
  ERA_RADIUS,
  inSnapArc,
  momentOfInertia,
  momentOfInertiaAt,
  poseHeld,
  poseHeldAt,
  syncHeadingCosSin,
  ERA_SLOTS,
  NODE_SLOTS,
  portWorld,
  portWorldInto,
  stemRoot,
  stemOffset,
  B_STATE,
  CHEM_LEN,
  F_BASE,
  F_OUT,
  GAIT_ANCHOR_MAX,
  G_BASE,
  G_OUT,
  HEAD_RANGE,
  HEAD_SCALE,
  L_BASE,
  L_OUT,
  P_BASE,
  P_OUT,
  PLASTIC_LEN,
  HEAD_TABLE,
  HEAD_ROWS,
  FRAME_WRAP,
  exploreAt,
  CRITIC_LEN,
  LEARN_CRITIC,
  LEARN_PLASTIC,
  LEARN_PREV_V,
  LEARN_TRACE,
  IN_BOUND,
  IN_DEMAND,
  IN_DIMS,
  IN_FULL,
  IN_SENSE,
  STATE_DIMS,
  W_IN,
  W_NET,
  W_SELF,
  emitVector,
  tasteVector,
  effTaste,
  flockGain,
  stemOffsetInto,
  stemWorld,
  stemWorldInto,
  type Agent,
  type AgentKind,
  type PortSlot,
} from './agents.ts';
import { AgentStore, CODE_KIND } from './agent-store.ts';
import { queryHit, queryDiscHit, SLOP, type Hit } from './collide.ts';
import { closestTOnSegment, WIRE_RADIUS, wireBowBudget, bounceOffDisk } from './geom.ts';
import { PairGrid } from './grid.ts';
import { CHAIN_MASS, contactMechanics, portExitAngle, solveContact } from './chain.ts';
import { CH, FIELD_CELL, FIELD_CELLS, Fields, worldBoundRadius } from './fields.ts';
import { Graph, ropeIsLive, wrapPos, type Wire } from './graph.ts';
import { LarvalWindow } from './larval.ts';
import type { Params } from './params.ts';
import {
  advanceRewrite,
  beginRewrite,
  commitRewrite,
  detectRule,
  PULL_END,
  rewriteHandoffStems,
  CHEM_TASTE_MAX,
  TRAIT_KEYS,
  type Rewrite,
} from './rewrite.ts';
import {
  EXTRA_FULL_EPS,
  agentValue,
  BODY_VALUE,
  deathYield,
  EnergyGrid,
  flowChargesFast,
  harvestSlotsFast,
  HarvestPlan,
  payToward,
  rescueNeed,
  redexNeed,
  rewriteCost,
  rewriteShareOf,
  PENDING_STRIDE,
  HARVEST_GOT,
  HARVEST_STRIDE,
  rewriteYield,
  seedRequest,
  settlePool,
  stakeMet,
  relaxRequestsFast,
  tickUpkeepFast,
  WireAdjacency,
} from './energy.ts';
import { audio } from './audio/engine.ts';
import type { CollisionEvent, LiveContact, PanView, RewriteEvent } from './audio/types.ts';
import { AGENT_BAND, LOD_FAR, LodSelector, agentKey, apparentPx, onScreen, wiresDrawable } from './audio/lod.ts';
import { farGpu } from './gpu/far-gpu.ts';
import { fieldGpu } from './gpu/field-gpu.ts';
import { genomeGpu } from './gpu/genome-gpu.ts';
import { FAR, FAR_STRIDE, packFarWire } from './gpu/far-kernel.ts';
import {
  KIND_CON,
  KIND_DUP,
  KIND_ERA,
  nativeSolver,
  ND,
  NODE_STRIDE,
  HIT,
  HIT_STRIDE,
  STEER_FLAG,
  STEER_PARAM,
  WF_FULL,
  WF_HOLD,
  WF_SHAPE,
  WF_SKIP,
  WF_TETHER,
  WIRE_NEAR_STRIDE,
  WN,
} from './native/solver.ts';
import { angleDelta, clamp, wrapAngle, wrapDeltaVec } from './wrap.ts';

/** Attraction-only chemotaxis. Own principal trails are a different channel and are ignored. */
/**
 * Normalised sensor asymmetry that earns a full-arc turn.
 *
 * Steering compares the two sensors as `(right - left) / (|right| + |left|)`,
 * which is dimensionless and lives in [-1, 1]. This is where that saturates:
 * a quarter of the signal's own magnitude between the two sensors is a strong
 * gradient and turns as hard as the body can. Below it the turn ramps, so the
 * size of the difference means something and not only its sign.
 *
 * Passed to the solver through `sparams` rather than written down there too. A
 * constant that has to agree across the wasm wall and is stated twice
 * eventually disagrees — the deposit normalisation was 20 on one side and 10
 * on the other, and quadrupled every scent reading on the path the app runs.
 */
export const SENSE_SPAN = 0.1;

/**
 * What a body smells at a point: its taste weights against the four channels.
 *
 * This used to be a switch on kind returning one of three fixed weight rows.
 * Those rows are now the seed of a per-body genome (`seedChem`), so the switch
 * is a dot product and the weights can drift.
 *
 * `groundScale` is what makes that dot product legitimate now that one of the
 * four channels is the ground rather than a smell. The three signal channels
 * hold accumulated deposits — five units a port a frame against a decay of a
 * percent, so they run to peaks around ten. Energy holds a quantity per cell,
 * and a cell at full capacity holds `ambientEnergy / 16`, about a sixteenth
 * of one unit. Summing those raw would need a taste weight two orders of
 * magnitude past `CHEM_TASTE_MAX` before the ground could shift a decision,
 * so a lineage could never evolve into caring about food however much it
 * wanted to.
 *
 * The scale is `1 / cellCap`, which turns the ground into "how full is it
 * here", 0 to 1. That is the reading a taste weight can be sensible about,
 * and it stays sensible when `ambientEnergy` or the cell size moves — the
 * gene means the same thing at every setting of the sliders.
 */
export function mixScent(
  a: Agent,
  s0: number,
  s1: number,
  s2: number,
  s3: number,
  groundScale = 1,
): number {
  return (
    effTaste(a, 0) * s0 +
    effTaste(a, 1) * s1 +
    effTaste(a, 2) * s2 * groundScale +
    effTaste(a, 3) * s3
  );
}

/** Cruise multiplier from local trail strength (1 = clear, →0 in dense scent). */
export function scentSlowFactor(trail: number): number {
  return 1 / (1 + trail / 28);
}

/**
 * Turn authority rises as cruise falls — roughly constant locomotion budget,
 * shifting kinetic emphasis from translation to rotation in stronger scent.
 */
export function scentTurnBoost(trail: number): number {
  const slow = scentSlowFactor(trail);
  const linear = slow * slow;
  const spin = 1 - linear;
  return clamp(1 + spin / Math.max(0.1, linear), 1, 8);
}

/** Monotonic, process-wide. Only ever compared for equality. */
let nextSimId = 1;

/**
 * A contact pair as one number rather than a `"lo:hi"` string, which was
 * built and hashed once per SAT contact per substep. Ids below 2^26 pack
 * exactly, and the product stays inside the safe-integer range.
 */
const CONTACT_KEY_WIDTH = 67108864;

function contactKey(a: number, b: number): number {
  return a < b ? a * CONTACT_KEY_WIDTH + b : b * CONTACT_KEY_WIDTH + a;
}

/**
 * Energy two ends of a ready redex have put up toward their commute, held
 * against the wire that joins them. See `Sim.accrueRedexes`.
 *
 * `a`/`b` are the agent ids the stakes came from, so a refund can find them
 * without the wire; `x`/`y` are where the redex last stood, for the case where
 * it cannot.
 */
type RedexEscrow = {
  a: number;
  b: number;
  paidA: number;
  paidB: number;
  x: number;
  y: number;
};

/** Most either metabolite may pile up to. A pathway is not a warehouse. */
const REACT_CAP = 256;

/**
 * Primer below this is "absolute depletion" for the doc's §7.2 death clock.
 *
 * Well under a fed body's trough, which measures about 1.8 at the shipped
 * constants, so a body that is merely between bursts is never counted as
 * starving — the clock only runs for one that has genuinely run out.
 */
const STARVE_EPS = 0.05;
/**
 * How fast the §7.2 clock unwinds while a body is fed, per second, against
 * one per second while it is empty.
 *
 * 1 is symmetric and is the whole statement of the rule: a second spent with
 * nothing costs a second spent fed. Below 1 a body has to be fed most of the
 * time to hold steady; above it, only a sustained famine is lethal.
 *
 * A constant rather than a slider because `starveTime` is already the dial
 * this mechanic ships behind — that says how much arrears a body may run up,
 * and this says the exchange rate. Two numbers for one idea is how a
 * parameter surface sprawls.
 */
const STARVE_RECOVER = 1;
/**
 * Largest reaction step taken at once. The pathway is stiff where its
 * activator spikes, and explicit Euler past about this rings at the step
 * frequency instead of oscillating — so `advanceGait` splits a frame into as
 * many of these as it needs, which makes the amplitude the same at every
 * `metabolicRate` and the period exactly proportional to it.
 */
const REACT_H = 0.04;

/**
 * The Hill coefficient as both field paths use it: plain Monod for anything
 * not above zero. Resolved here, once, and uploaded resolved, so the shader's
 * `!= 1.0` test and the host's fallback in `runHarvestPlan` see the same
 * number — a protocol can set `hillN` to 0, and `pow(density, 0)` is not
 * Monod.
 */
function hillOf(params: Params): number {
  return params.hillN > 0 ? params.hillN : 1;
}

/**
 * One body's four taste weights, laid out for a consumer of the field.
 *
 * The ground scale has to be folded in in one place rather than left to each
 * consumer, because there are three of them — the JS dot product above, the
 * packed vector the wasm solver reads, and the GPU's probe buffer — and a
 * steering difference between them is the kind of bug that only shows up on
 * one machine. This is that place; `at` is where the four go, because the two
 * packs put them at different offsets in rows of different widths.
 *
 * It replaced a `tasteOf(agent, channel)` method called once per channel per
 * body. That was four calls and four re-reads of `a.slot` to index the array
 * this takes directly, and a comparison per channel against `CH.energy`,
 * which is a compile-time constant and so never had an answer that varied.
 */
/**
 * The channel a kind speaks on, and is itself deaf to. −1 for an Era.
 *
 * `seedChem`'s own arithmetic: a Con emits ch0 and tastes ch1 and ch3, a Dup
 * emits ch1 and tastes ch0 and ch3. Neither has a taste for what it says. So a
 * body advertising a free principal on this channel is inaudible to itself and
 * to its own kind — no trail to follow, no huddle to form — and loud to the
 * two kinds a redex with it would actually consume: the other node kind, and
 * an Era, which tastes both at `attractStrong`.
 *
 * An Era has none. It says nothing at seed and it is the kind that goes
 * looking rather than the kind that is looked for.
 */
function ownChannel(kind: AgentKind): number {
  return kind === 'con' ? CH.conP : kind === 'dup' ? CH.dupP : -1;
}

function packTaste(
  out: Float32Array | Float64Array,
  at: number,
  tasteAll: Float64Array,
  slot: number,
  groundScale: number,
  gain: number,
): void {
  const from = slot * 4;
  out[at] = tasteAll[from] * gain;
  out[at + 1] = tasteAll[from + 1] * gain;
  out[at + 2] = tasteAll[from + 2] * gain;
  out[at + 3] = tasteAll[from + 3] * gain;
  out[at + CH.energy] = tasteAll[from + CH.energy] * groundScale * gain;
}

export class Sim {
  /** Substeps per frame. One constraint iteration each. */
  private static readonly SUBSTEPS = 8;

  /** Hop radius for flocking. Farther pairs are 1/hops anyway. */
  private static readonly FLOCK_HOPS = 6;
  /** Neighbourhood lifted onto the detailed physics path around a NEAR/MID body. */
  private static readonly PHYS_HOPS = 2;

  /** Auto-spawn stays near the flock even when the camera cover is huge. */
  private static readonly SPAWN_REACH_CAP = 480;

  /** Repulsion in px/s² at exactly one wire's length; inverse-square inside that. */
  static DECLUTTER_FORCE = 200;

  /**
   * Run the per-body force passes in WASM when it is available. Off is the
   * reference implementation in JS — the two agree to a few parts in 10^4,
   * not bit-for-bit, since the C side is f32 and uses its own libm. Tests
   * flip this to check the pair against each other.
   */
  static nativeForces = true;

  /**
   * Check, at the solver, that the packed bodies still describe the agents.
   *
   * The force block hands its packed copy to whichever solver runs, on the
   * grounds that nothing between the force passes and the solve moves a body:
   * the passes write velocity only, the LOD pass reads positions, the rope
   * passes write wires. If that ever stops being true the packed pose goes
   * stale and `syncForces` papers over it by unpacking, which is silent. On,
   * this makes it loud instead. Off in normal running — it is a debug aid, and
   * the test below is what keeps the invariant honest.
   */
  static auditForceBlock = false;

  /** Range of the repulsion, in wire lengths. */
  private static readonly DECLUTTER_CUTOFF = 2;

  /** Closest distance the inverse-square law is evaluated at, in wire lengths. */
  private static readonly DECLUTTER_FLOOR = 0.4;

  /** Most a rope node may be pushed clear in one substep, in px. */
  static WIRE_CLEAR_STEP = 1.5;

  /** Compliance of the pointer hold. Firm enough to follow, soft enough to lag. */
  private static readonly GRAB_COMPLIANCE = 1.0e-6;

  /** Most the pointer may move a held agent in one substep, in px. */
  private static readonly GRAB_STEP = 1.5;

  /**
   * Speed cap on a held agent. The hold is a positional constraint, so its
   * correction reappears as velocity at 1/h — 1.5 px a substep is 720 px/s, and
   * without this the agent keeps all of it and rockets away on release.
   */
  private static readonly GRAB_MAX_SPEED = 160;

  w: number;
  h: number;
  coverW: number;
  coverH: number;
  time = 0;
  nextId = 1;
  /**
   * Bumped whenever the agent roster changes. Passes that cache anything keyed
   * on *index* — flocking's neighbourhood cache is the only one so far — need
   * this as well as `graph.version`, because `agents` is a Map iterated in
   * insertion order: deleting one body renumbers every body after it without
   * touching a single wire.
   */
  rosterVersion = 0;
  /**
   * What a body is worth dead, latched from `params.bodyValue` each frame.
   *
   * `kill` has no `Params` to read — see the note where this is set.
   */
  bodyValue = BODY_VALUE;

  /**
   * Register a roster change made without going through `spawn` or `kill`.
   *
   * `commitRewrite` and the lambda loader write into the agents Map directly.
   * Every cache keyed on the roster is wrong from the moment one of those runs
   * until this is called, and the failure is silent: a stale body list indexes
   * the wrong agents rather than throwing. The determinism hashes are what
   * catch a missing call, which is how the one in `commitRewrite` was found.
   */
  noteRosterChange(): void {
    this.rosterVersion++;
  }
  /** Identifies this Sim to the shared wasm solver's flocking cache. */
  private readonly simId = nextSimId++;

  /**
   * Per-phase frame timings, off by default.
   *
   * Set to a Map to collect; phases accumulate into it until it is cleared.
   * Off, `phase()` is a single null check and allocates nothing — which is the
   * point of taking the timestamp inside the helper rather than wrapping calls
   * in closures at each site.
   *
   * This exists because two rounds of guessing where the frame went were both
   * wrong, and because the alternative — measuring a pass by turning it off
   * and differencing whole frames — cannot resolve anything smaller than the
   * machine's own drift, several milliseconds on a frame this size.
   */
  /**
   * Which FAR solver to prefer when both can take a frame. 'auto' picks on body
   * count; 'on' and 'off' force it either way, which is how the kernel gets
   * exercised deliberately rather than only at the one zoom that happens to put
   * every body on the FAR tier.
   *
   * It stayed off for a long time and needed to: preferring the GPU ran
   * `solveFarGpu` in production for the first time and sent wires to infinite
   * length. No test caught it and none can -- there is no WebGPU under Node, so
   * `farGpu.ready` is false and the path is skipped in the whole suite. It is
   * verified in a browser now, against the twin, which is what earns 'auto'.
   */
  static farGpuMode: 'auto' | 'on' | 'off' = 'auto';

  /**
   * How far under wasm's capacity 'auto' has to fall before it hands the FAR
   * solve back. Only a hysteresis band: the switch up is the capacity itself.
   *
   * The two edges differ on purpose. A pond sitting exactly on the cap would
   * change solver every frame, and the two do not agree to the digit -- span is
   * Jacobi on the GPU and Gauss-Seidel in the twin -- so flapping would show as
   * a shimmer.
   */
  static FAR_GPU_RELEASE = 0.95;

  /** Which side of the hysteresis band 'auto' is currently latched to. */
  private farGpuLatched = false;

  static profile: Map<string, number> | null = null;
  private static profileMark = 0;

  /** Charge everything since the last call to `name`. */
  private static phase(name: string): void {
    const p = Sim.profile;
    if (!p) return;
    const now = performance.now();
    p.set(name, (p.get(name) ?? 0) + (now - Sim.profileMark));
    Sim.profileMark = now;
  }

  /** Start the clock. Call at the top of a frame, before the first phase. */
  private static phaseStart(): void {
    if (Sim.profile) Sim.profileMark = performance.now();
  }
  spawnAcc = 0;
  agents = new Map<number, Agent>();
  agentStore = new AgentStore();
  graph: Graph;
  fields: Fields;
  rewrites: Rewrite[] = [];
  energy = new EnergyGrid(48, 0.1);
  /**
   * Commute and erase copy parent traits onto children. The designer turns
   * this off so Play rewrites the net without breeding a new chemistry.
   */
  breed = true;
  /**
   * Last frame's need field, in `forceList` order, for `spreadRequests` to
   * step off. The field is state now, so a step has to read where it *was*
   * — reading it live would let one body's need race several hops in a pass,
   * in whatever order the roster happens to be in.
   */

  /** Scratch for one diffusion step: the pull on each body, and its degree. */

  /** Per-agent unmet need this frame, rebuilt by `pulseRequests`. */
  private readonly wireAdj = new WireAdjacency();
  /**
   * Which connected component each body is in, as a position in
   * `forceList()`, keyed on the graph and roster versions.
   *
   * Union-find over an `Int32Array` rather than the `Map`-of-ids `Graph`
   * hands out. Every consumer here — the force scratch the wasm passes read,
   * and the JS declutter fallback — indexes by list position and only ever
   * compares two roots for equality, so an index is as good a name for a
   * component as an id and costs a fraction to produce. `Graph.componentIds`
   * stays for the audio shards, which do want ids.
   */
  private componentOf = new Int32Array(0);
  private compVersion = -1;
  private compRoster = -1;
  /** Broad-phase results for wire clearance, flattened pairs, rebuilt per frame. */
  private clearBodyPairs: unknown[] = [];
  /** Broad-phase grids and their scratch coordinate arrays. */
  private bodyGrid = new PairGrid();
  private gx: number[] = [];
  private gy: number[] = [];
  private wx: number[] = [];
  private wy: number[] = [];
  private agentList: Agent[] = [];
  private wirePack: Wire[] = [];
  private clearWireList: Wire[] = [];
  private flockDist = new Int32Array(0);
  private flockQ = new Int32Array(0);
  private flockSeen: number[] = [];
  private flockSwim = new Uint8Array(0);
  private flockSlot = new Int32Array(0);
  private tmpStemA = { x: 0, y: 0 };
  private tmpStemB = { x: 0, y: 0 };
  private satBuf = new Uint8Array(0);
  private compBuf = new Int32Array(0);
  private slotBuf = new Int32Array(0);
  /** Pairs in contact last frame — a strike fires on onset, contact continues. */
  private contactAudioPrev = new Set<number>();
  private contactAudioNow = new Set<number>();
  /** Wire/body bows this frame, keyed by `wireId:agentId`. */
  /** Bodies currently overlapping, keyed by canonical `lo:hi` id pair. */
  contacts = new Map<number, LiveContact>();
  /** Crossing / overlapping ropes this frame. Geometry is not displaced. */
  /** Momentum lost to sound this frame, applied once after the substeps. */
  private radiated = new Map<number, { x: number; y: number }>();
  /**
   * Agent held by the pointer, and where it is being held. Solved as a
   * constraint inside the substep loop rather than by assigning a position:
   * writing a pose directly is the kinematic teleport that destabilised every
   * early version of this solver, and it would drag whole nets through their
   * joints at 1/h velocity.
   */
  grabbed: { id: number; x: number; y: number } | null = null;
  /** Agents in an active rewrite — their incident ropes are kinematic. */
  private rewriteFrozen = new Set<number>();
  /**
   * Energy banked against a ready redex, keyed by wire id. See `accrueRedexes`.
   *
   * This is real energy that has left its bodies and is not yet in the ground
   * or in a new body, so anything totalling the pond has to count it — that is
   * what `escrowTotal` is for.
   */
  private readonly escrow = new Map<number, RedexEscrow>();
  /**
   * Who grazes which block this frame. Reused rather than rebuilt, and shared
   * by both field paths so the binning has one implementation.
   *
   * On the GPU path it is built at the end of a frame and spent at the top of
   * the next, so it is a frame older than it looks — see `creditHarvest`.
   */
  private readonly harvestPlan = new HarvestPlan();
  /** Reused by the GPU deposit pack; see `portWorldInto`. */
  private readonly portScratch = { x: 0, y: 0 };
  /** True once a GPU harvest has been dispatched and not yet credited. */
  private harvestPending = false;
  /**
   * Ask for the GPU field to be copied back into `fields.data` each frame.
   *
   * Off by default and deliberately opt-in: it is a sixteen-megabyte copy, and
   * the only things that want it are the two debug overlays, which paint from
   * the CPU array. `render.ts` sets it from its own options, so the cost is
   * paid exactly while somebody is looking. Ignored when the field is here
   * anyway.
   */
  wantFieldReadback = false;
  /**
   * True once the genome pass has moved to the GPU as well.
   *
   * Separate from `fieldOnGpu` and strictly downstream of it: the pass reads
   * the field probe's own output buffer for its sense inputs, so there is
   * nothing for it to read until the field is there. When it is on,
   * `updateState` stops computing and starts unpacking.
   */
  private genomeOnGpu = false;
  /** True once a genome pass has been dispatched and not yet unpacked. */
  private genomePending = false;
  private genomeSlotBuf = new Int32Array(0);
  private genomeCount = 0;
  /** `AgentStore.chemVersion` the GPU's genome table was last synced to. */
  private genomeChemVersion = -1;
  /** `AgentStore.learnVersion` the device's learning rows were last synced to. */
  private genomeLearnVersion = -1;
  /**
   * Slots whose learning the CPU needs back from the device.
   *
   * On the GPU path the device owns what every body has learned, and the CPU
   * needs it in exactly one place: a rewrite's two parents, whose learning is
   * consolidated into their children's genome by `inheritChem`. `startRewrites`
   * marks the pair, and a rewrite takes about forty frames to commit, so the
   * rows are back long before anything reads them.
   */
  private readonly learnWanted: number[] = [];
  private readonly learnAsked: number[] = [];
  /**
   * Whether the solver steers from readings handed to it — the GPU probe's —
   * rather than sampling a field it holds. Mirrors `nativeSolver.useSamples`
   * for this Sim, since that flag is a singleton the whole process shares.
   */
  private steerFromSamples = false;
  /**
   * Which topology and roster `wireAdj` describes. See `wireAdjacency`.
   */
  private adjGraphVersion = -1;
  private adjRosterVersion = -1;
  /** Principal pairs ready to rewrite this frame. See `collectReadyRedexes`. */
  private readonly readyRedexes: Wire[] = [];
  /**
   * Running counts of the events selection acts through, since `clear()`.
   *
   * `census()` can say how deep the population is; it cannot say how fast it
   * turns over, and turnover is the number that decides whether anything
   * about a pond is evolving or merely wandering. Integers, bumped where the
   * events happen, read by the experiment harness and by nothing per frame.
   */
  readonly tally = {
    spawned: 0,
    born: 0,
    died: 0,
    commutes: 0,
    erases: 0,
    annihilations: 0,
    latches: 0,
    snaps: 0,
  };
  /**
   * Physics detail. When a view is passed, FAR agents keep disc contacts and a
   * chord constraint but skip SAT, rope XPBD, wire clearance, and Hertzian.
   * Missing view = everyone NEAR, which is what tests and a paused layout want.
   */
  private lodActive = false;
  /** False when zoom has shrunk the wire stroke below a visible hairline. */
  private ropesDrawable = true;
  private readonly physLod = new LodSelector();
  private readonly detailedAgents = new Set<number>();

  /**
   * Whichever physics LOD tier this agent last settled into. A pure read —
   * `LodSelector.peek` does not re-tier, unlike `tier()` itself, which also
   * feeds the hysteresis that keeps a body on a band edge from flapping.
   * Calling `tier()` again from here with different inputs would corrupt
   * that. Undefined (never tiered — LOD inactive, or the agent is new this
   * frame) reads as NEAR: full detail is the safe default, not FAR.
   */
  isFarTier(agentId: number): boolean {
    return this.physLod.peek(agentKey(agentId)) === LOD_FAR;
  }

  /** Eased centre of mass. Rewrites delete agents, which jumps the true COM. */
  private home: { x: number; y: number } | null = null;

  /**
   * Cells a side of the field a new Sim gets, when the constructor is not
   * told. The app leaves it at the world's full size; the test suite sets it
   * to a quarter, which is a sixteenth of the diffusion work — the CPU
   * field over a million cells was twenty milliseconds a frame in Node
   * whatever the body count, and most of what the suite's half hour was
   * spent on. The cell stays ten units either way, so it is a smaller dish
   * and not a coarser one, and nothing about steering changes.
   */
  static defaultFieldCells = FIELD_CELLS;

  /**
   * Time from arrival to first latch, and how many never got there.
   *
   * Cumulative over the run rather than per interval: a body that arrives
   * late in a sample window and latches in the next one belongs to neither,
   * and the question is about the population, not the minute.
   */
  readonly larval = new LarvalWindow();

  /**
   * A body's first wire, in the terms `larval` counts.
   *
   * `arrivedAt` doubles as the flag: `-1` means "has latched", so a body that
   * detaches and re-latches is not counted twice. Only the first latch
   * answers the larval question — after that the body has been fed by a net
   * and its tank is no longer the clock.
   */
  private noteFirstLatch(id: number): void {
    const slot = this.agentStore.slotFor(id);
    if (slot === undefined) return;
    const arrived = this.agentStore.arrivedAt[slot];
    if (arrived < 0) return;
    this.agentStore.arrivedAt[slot] = -1;
    this.larval.latchedAfter(this.time - arrived);
  }

  constructor(w: number, h: number, fieldCells = Sim.defaultFieldCells) {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    this.coverW = this.w;
    this.coverH = this.h;
    this.fields = new Fields(fieldCells, fieldCells * FIELD_CELL);
    this.graph = new Graph(this.agentStore);
    this.graph.onLatch = (ev) => {
      this.tally.latches++;
      this.noteFirstLatch(ev.agentA);
      this.noteFirstLatch(ev.agentB);
      audio.push(ev, this.graph, this.agents);
    };
    audio.contacts = this.contacts;
  }

  /** Viewport / spawn-box size. World coordinates are not scaled. */
  resize(w: number, h: number): void {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
  }

  /**
   * Size of the visible area, in world units. Nothing to do with the scent
   * field any more — that is world-fixed — but auto-spawn still places new
   * bodies within a fraction of what you can see.
   */
  setViewExtent(w: number, h: number): void {
    this.coverW = Math.max(32, w);
    this.coverH = Math.max(32, h);
  }

  centerOfMass(): { x: number; y: number } | null {
    let m = 0;
    let x = 0;
    let y = 0;
    for (const a of this.agents.values()) {
      x += a.x * a.mass;
      y += a.y * a.mass;
      m += a.mass;
    }
    if (m < 1e-6) return null;
    return { x: x / m, y: y / m };
  }

  clear(): void {
    this.agents.clear();
    this.agentStore = new AgentStore();
    // Port occupancy lives in the store, so the graph follows it. Ordered
    // before `clear`, which blanks occupancy on whichever store it holds.
    this.graph.useStore(this.agentStore);
    this.graph.clear();
    this.rewrites = [];
    // Not `refundEscrows`: the bodies and the ground are both being thrown
    // away on the next two lines, so there is nowhere for a stake to go back to.
    this.escrow.clear();
    this.fields.clear();
    this.energy.clear();
    this.time = 0;
    // A fresh store starts every slot at `arrivedAt = 0`, so the histogram
    // has to go with it or the next pond's first latches are measured from
    // the last pond's clock.
    this.larval.reset();
    this.nextId = 1;
    this.rosterVersion++;
    this.worldPinned = false;
    this.worldR = 0;
    this.spawnAcc = 0;
    this.home = null;
    this.contactAudioPrev.clear();
    this.contactAudioNow.clear();
    this.contacts.clear();
    this.physLod.clear();
    this.detailedAgents.clear();
    this.ropesDrawable = true;
    this.lodActive = false;
    /*
     * Everything the GPU path carries across a frame boundary. A harvest plan
     * or a genome dispatched against the old roster would otherwise be paid
     * out to whichever bodies of the new pond landed in the same slots — and
     * with ids restarting at one and slots at zero, that is nearly all of
     * them; the id check in `creditHarvest` and `unpackGenome` cannot tell a
     * new body 1 from an old one. The field is cleared on the device for the
     * same reason: `fields.clear()` above only zeroes the CPU mirror, and the
     * old pond's scent would still be in the buffers when the new one pinned
     * its world.
     */
    this.harvestPending = false;
    this.genomePending = false;
    this.genomeCount = 0;
    this.genomeChemVersion = -1;
    this.genomeLearnVersion = -1;
    this.learnWanted.length = 0;
    this.learnAsked.length = 0;
    this.steerFromSamples = false;
    this.adjGraphVersion = -1;
    this.adjRosterVersion = -1;
    this.readyRedexes.length = 0;
    if (this.fieldOnGpu) fieldGpu.clear();
    for (const k of Object.keys(this.tally) as (keyof Sim['tally'])[]) this.tally[k] = 0;
    audio.invalidateTopology();
  }

  canSpawn(params: Params, n = 1): boolean {
    return this.agents.size + n <= params.maxAgents;
  }

  spawn(
    kind: AgentKind,
    x: number,
    y: number,
    heading: number,
    params: Params,
    force = false,
  ): Agent | null {
    if (!force && !this.canSpawn(params)) return null;
    const a = createAgent(
      this.nextId++,
      kind,
      x,
      y,
      heading,
      params,
      this.agentStore,
    );
    this.agents.set(a.id, a);
    this.rosterVersion++;
    this.tally.spawned++;
    // Queued, so it lands after the topology that first contains this agent.
    audio.push({ type: 'spawn', agent: a.id, kind: a.kind }, this.graph, this.agents);
    return a;
  }

  /**
   * A body that has run a whole unit into debt is gone, not merely cut loose.
   * Leaving starved agents drifting as inert singletons filled the pond with
   * bodies that could never latch again, and existence is the thing upkeep is
   * charging for — stop paying and you stop existing.
   *
   * What it was made of goes back to the ground it died on: a full body's
   * worth plus whatever it still held, so a body that starved all the way to
   * −1 leaves `EXTRA_CAP − 1` behind. Onto the grid rather than into the net,
   * because a body only starves when the net around it had nothing to send.
   */
  kill(id: number): void {
    const agent = this.agents.get(id);
    if (!agent) return;
    const slot = this.agentStore.slotFor(id);
    if (slot !== undefined && this.agentStore.arrivedAt[slot] >= 0) this.larval.diedAlone();
    this.energy.addAt(agent.x, agent.y, deathYield(agent, this.bodyValue));
    // And whatever it had swallowed and not yet turned into anything, as
    // itself. A corpse that kept its gut would be a leak the size of the
    // pond's whole appetite, and a body dying with a gut full of what it could
    // not digest is exactly the death this mechanism makes possible.
    if (slot !== undefined) this.spillGut(slot, agent.x, agent.y);
    this.graph.detachAgent(id);
    this.agents.delete(id);
    this.agentStore.release(id);
    this.rosterVersion++;
    this.tally.died++;
    if (this.grabbed?.id === id) this.grabbed = null;
  }

  /** Its own, not `excreteScratch`: a death can land inside any pass. */

  /** Put one body's undigested holdings back on the ground it is standing on. */
  private spillGut(slot: number, x: number, y: number): void {
    const GUT = this.agentStore.gut;
    const held = GUT[slot];
    if (!(held > 0)) return;
    GUT[slot] = 0;
    this.energy.addAt(x, y, held);
  }

  wire(aId: number, aSlot: 'p' | 'l' | 'r', bId: number, bSlot: 'p' | 'l' | 'r', params: Params): void {
    this.graph.connect(
      this.agents,
      { id: aId, slot: aSlot },
      { id: bId, slot: bSlot },
      this.w,
      this.h,
      params,
      this.time,
    );
  }

  /**
   * Everything before the solve, shared by the two twins below. Returns the
   * clamped frame step.
   *
   * One body rather than two copies, because two copies drifted: the async
   * twin — the one the app runs — went a long time with no phase markers at
   * all, so the clock was never reset at the top of a frame and the first
   * marker to fire charged itself everything since the last one, and every
   * profile anyone took was of the synchronous twin, which cannot use the
   * GPU for either the field or the FAR solve.
   */
  private openFrame(dt: number, params: Params, view: PanView | null | undefined): number {
    Sim.phaseStart();
    /*
     * One counter for both paths, advanced where every frame begins. Only
     * `exploreAt` reads it, and it needs a number the host and the device
     * agree on: wrapped at `FRAME_WRAP` so it stays exact in the `f32` the
     * genome uniform carries it in.
     */
    this.frames = (this.frames + 1) % FRAME_WRAP;
    const t = this.beginFrame(dt, params);
    this.collectRewriteFrozen(params);
    Sim.phase('collectRewriteFrozen');
    this.assignPhysicsLod(view);
    Sim.phase('assignPhysicsLod');
    this.graph.syncRest(this.time, params, this.agents, this.agentStore.gaitWave, this.wireDetailed);
    Sim.phase('syncRest');
    this.graph.applyRopePaths(this.agents, this.w, this.h, this.time, params);
    Sim.phase('applyRopePaths');
    this.graph.syncRopeShape(this.agents, this.w, this.h, this.wireDetailed);
    Sim.phase('syncRopeShape');
    return t;
  }

  /**
   * The synchronous frame: wasm or the JS twin, never the GPU. What the test
   * suite and the experiment harness drive; the app runs `stepAsync`.
   */
  step(dt: number, params: Params, view?: PanView | null): void {
    this.frameDt = dt;
    const t = this.openFrame(dt, params, view);
    if (this.solveFarNative(params, t)) {
      this.lastFarPath = 'wasm';
      this.finishIntegrate(t);
    } else {
      this.lastFarPath = 'js';
      this.solve(params, t);
    }
    Sim.phase('solve');
    this.endFrame(params, t);
  }

  /** The frame that ships: the same as `step`, plus the GPU where it has one. */
  async stepAsync(dt: number, params: Params, view?: PanView | null): Promise<void> {
    const t = this.openFrame(dt, params, view);
    /*
     * The GPU when `wantFarGpu` says it is worth the round trip, then wasm,
     * then the GPU again for the frames wasm turned down, then the TS twin.
     *
     * The third branch is not redundant with the first: wasm declines any pack
     * over its body cap (MAX_BODIES, 32768), and above that the GPU is the only
     * real solver left -- the twin below it is brute force and would take
     * minutes. That branch used to be reachable only on a machine with no wasm
     * module at all, which is why its kernel went unexercised for so long.
     */
    if (this.wantFarGpu() && (await this.solveFarGpu(params, t))) {
      this.finishIntegrate(t);
    } else if (this.solveFarNative(params, t)) {
      this.lastFarPath = 'wasm';
      this.finishIntegrate(t);
    } else if (this.canFarGpu() && (await this.solveFarGpu(params, t))) {
      this.finishIntegrate(t);
    } else {
      this.lastFarPath = 'js';
      this.solve(params, t);
    }
    Sim.phase('solve');
    this.endFrame(params, t);
    if (this.fieldOnGpu) {
      await this.gpuFieldStep(params, t);
      Sim.phase('fieldGpu');
    }
  }

  private beginFrame(dt: number, params: Params): number {
    const t = clamp(dt, 0, 0.05);
    this.time += t;
    // The store stamps a new slot with this, so every creation path records
    // an arrival without any of them having to be found. See `larval.ts`.
    this.agentStore.now = this.time;
    this.trackHome(t);
    /*
     * One centre for the whole world grid, pinned once and never moved.
     *
     * It used to follow `home`, which is a habit left over from the field
     * being a camera window. Two things were wrong with it. The field scrolled
     * about a hundred times a minute, and on the GPU path only the CPU copy
     * scrolled — the origin in the uniform moved while the buffer's contents
     * did not, so the whole field slid through world space against home's
     * drift and banked up along an edge. Eras found it first, being the only
     * kind with `attractStrong`, and ended up clumped in the corners.
     *
     * The second is worse and applied to both paths: a bound that follows the
     * pond can be towed by whatever is escaping it. Anchoring the bound means
     * the wall has something to bounce *off*.
     *
     * Pinned at the first centre of mass unless a preset already pinned, so a
     * preset still decides where its world is rather than inheriting an
     * arbitrary origin.
     */
    /*
     * Before the pin, not after: `pinWorld` lays the ground down at whatever
     * capacity these say, and it can fire on this very frame. Read out of
     * `params` further down the frame — where `configure` also happens — and
     * the first pond in a run gets seeded from the defaults instead of from
     * its own settings, which is invisible in a soup and is the difference
     * between barren and fed in a test that asked for barren.
     */
    this.energyCell = params.energyCell;
    this.energyAmbient = params.ambientEnergy;
    this.energyPatches = params.groundPatches;
    /*
     * Latched here for the same reason the two above it are: `kill` is public
     * and takes no `Params` — the designer and the pond library both call it —
     * so what a body is worth dead has to be a fact about the Sim by the time
     * anything can die. Read out of `params` at the death site instead and a
     * pond killed from outside a frame would price its corpses from the
     * defaults. See `params.bodyValue`.
     */
    this.bodyValue = params.bodyValue;
    const h = this.home;
    if (h && !this.worldPinned) this.pinWorld(h.x, h.y, params);
    this.contactAudioNow.clear();
    this.contacts.clear();
    this.radiated.clear();
    Sim.phase('beginFrame:setup');

    // The whole force phase now lives in WASM, so it shares one copy of the
    // bodies instead of each pass making its own — which was the entire cost
    // of moving them over: measured at 9600 agents, a pass that dropped from
    // 13 ms to 1.6 ms of compute still cost 7 ms because it packed 6 ms of
    // bodies to get there.
    const block = this.openForceBlock();
    this.refreshForceScratch(this.forceList());
    this.refreshBound();
    Sim.phase('openForceBlock');
    this.steer(params, t);
    Sim.phase('steer');
    this.portTorques(params, t);
    Sim.phase('portTorques');
    this.declutter(params, t);
    Sim.phase('declutter');
    this.flock(params, t);
    Sim.phase('flock');
    // The hard rim lives inside the integrator (JS solve / native step /
    // FAR GPU), not as a background failsafe. Left open on purpose. Nothing
    // between here and the solver moves a body — the LOD pass and the rope
    // passes read positions and write wires — so the solver can inherit the
    // packed bodies instead of copying them in again. Whoever consumes them
    // closes it; `syncForces` is the backstop.
    void block;
    return t;
  }

  private endFrame(params: Params, t: number): void {
    // Backstop: every path that did not end in a native solve still owes the
    // agents their velocities.
    this.syncForces();
    Sim.phase('syncForces');
    this.applyRadiationLoss();
    this.advanceGait(params, t);
    this.dampVelocities(params, t);
    Sim.phase('damp');

    this.graph.refreshLengths(this.agents, this.w, this.h, this.rewriteFrozen, this.wireDetailed);
    this.snapTautWires(params);
    Sim.phase('refreshLengths');
    // Earn, distribute, spend, then pay rent. Energy that arrives to complete
    // a redex is spent in the same frame it lands, and a body that has just
    // paid its share is not billed into debt on top of it.
    //
    // Measured, this order is worth little by itself: what actually unblocked
    // rewrites was giving `EXTRA_CAP` headroom over `REWRITE_SHARE`. Commutes
    // over 30 s of oscillator, at ambient 0.25 / 0.5 / 1 —
    //   cap 1.00, rent first: 8 / 35 / 42      cap 1.00, rent last: 8 / 12 / 29
    //   cap 1.25, rent first: 20 / 38 / 46     cap 1.25, rent last: 20 / 39 / 46
    // Rent-last is only safe *because* of the headroom: at cap == share it
    // leaves every body a hair in debt the moment it commutes.
    this.energy.configure(params.energyCell, params.ambientEnergy, params.groundPatches);
    this.dropGround(params, t);
    /*
     * On the GPU path the grazing itself happened at the end of the last
     * frame, in the shader, against the field the shader owns. All that is
     * left here is to pay it out — and this is the right place for that, not
     * the top of the tick: `tickUpkeepFast` kills at `debtCap`, so crediting
     * after upkeep would let a body die owing itself a meal it had already
     * been served.
     */
    if (this.fieldOnGpu) this.creditHarvest();
    else {
      // `uptakeVmax` at zero is the take-what-fits path this has always run;
      // above it, it is one mouthful a frame, shared across the four species
      // by what is standing in the cell. The GPU twin applies the same limit
      // inside the shader — see `field.wgsl`'s `harvest`, `energy.ts`'s
      // `uptakeRate` and the note on `UptakeKinetics`.
      harvestSlotsFast(this.agents.values(), this.agentStore, this.energy, this.harvestPlan, {
        // `t` is the frame's clamped dt — see `beginFrame`.
        cap: params.uptakeVmax * t,
        ks: params.uptakeKs,
        yDirect: params.yDirect,
        yEra: params.yEra,
        hillN: hillOf(params),
        gutSize: params.gutSize,
      });
      // A metered mouthful lands in a gut, so something now has to digest it.
      // Set from the plan rather than from the draw, because the draw is in
      // `energy.ts` and this flag is the Sim's; over-setting costs one pass.
      if (this.harvestPlan.metered && this.harvestPlan.nEntries > 0) this.gutLive = true;
    }
    Sim.phase('harvestSlots');
    this.latchPass(params);
    Sim.phase('snap');
    // `pulseRequests` charges itself in four parts — seeding the need field,
    // relaxing it, moving energy down it, and the state update — because at
    // one marker it was the second-largest phase in the frame and there was no
    // way to tell which quarter of it was the cost.
    // Which principal pairs are ready is asked once, here, and read by the
    // three passes after it. Each used to walk every wire in the pond to
    // answer it for itself — three full passes, each resolving both endpoints
    // and measuring the rope, for a set that cannot change between them.
    this.collectReadyRedexes(params);
    this.pulseRequests(params);
    this.accrueRedexes(params);
    this.startRewrites(params);
    this.tickRewrites(params, t);
    Sim.phase('rewrites');
    /*
     * Rent on moving.
     *
     * Per unit of speed rather than per unit of distance, which is the same
     * thing over a frame and reads better against the other per-second costs.
     * Only bodies that can actually swim pay: a wired-in body is cargo, moved
     * by the constraint rather than by its own port, and billing it for the
     * net's motion would make being carried expensive.
     */
    if (params.swimCost > 0) {
      const rent = params.swimCost * t;
      for (const a of this.agents.values()) {
        if (a.locked || !this.graph.isFreeAtSlot(a.slot, 0)) continue;
        const speed = Math.hypot(a.vx, a.vy);
        if (speed > 0) a.extra -= rent * speed;
      }
    }
    /*
     * The body reaction table, in two passes: express, then digest.
     *
     * There used to be a third, `runExcretion`, which laid a body's excretion
     * rows onto the field out of its tank. It existed because a body swallowed
     * a sample of all four channels and had to be able to clear what it could
     * not convert or clog and starve. A body eats the ground and nothing else
     * now, so there is nothing in a gut that its owner cannot digest, and the
     * necessity went with it. **Matter in is ground and matter out is ground;
     * a signal is emitted and smelled, never eaten and never excreted.**
     */
    this.runDigestion(params, t);
    Sim.phase('digest');
    for (const id of tickUpkeepFast(this.agents.values(), this.agentStore, t, params.upkeep, this.energy, {
      rentBack: params.upkeepExcrete,
      eraRatio: params.eraUpkeepRatio,
    })) {
      this.kill(id);
    }
    /*
     * The doc's §7.2: a body whose primer has been empty for longer than
     * `starveTime` dies. The same `kill` rent uses, because it is the same
     * death — what changed is what counts as having run out. Rent could push
     * a tank past a floor and so was the only thing in the pond that killed
     * anything; the metabolism draws in proportion to what a body holds, so
     * it empties a body and then stops, and without this nothing would ever
     * finish one off.
     */
    const window = params.starveTime;
    if (window > 0) {
      const STARVE = this.agentStore.starve;
      const doomed = this.starving;
      doomed.length = 0;
      for (const agent of this.agents.values()) {
        if (STARVE[agent.slot] > window) doomed.push(agent.id);
      }
      for (let i = 0; i < doomed.length; i++) this.kill(doomed[i]);
    }
    Sim.phase('upkeep');
    /*
     * The GPU owns the field when it is available, and then none of this runs:
     * the deposit is a list of world positions handed over rather than a
     * scatter done here, and the two diffusions and the decay — 13.7ms a frame
     * at a million cells, the second largest fixed cost after the solve —
     * happen there. `gpuFieldStep` does it at the end of `stepAsync`, because
     * it has to await and this does not.
     */
    /*
     * Outside the branch, because both paths need it. `field-gpu.ts` resolves
     * the shader's per-channel rates by reading `fields.diffuseRate` and
     * `fields.decayRate` — the arrays this writes — so leaving it in here gave
     * the GPU path a `Fields` still holding its constructed defaults of all
     * ones. The ground would have decayed at the full scent rate instead of
     * not at all, and diffused at 1 rather than at `energyDiffuse`, which is
     * the exact failure the note on `openFieldGpu` warns about: the shader
     * gaining the capacity for per-channel rates is not the same as anything
     * feeding it.
     */
    this.tuneChannels(params);
    // Cached here, beside the channel rates, for the same reason: a slider
    // read once a frame rather than once a body per sensor per frame.
    this.interiorTaste = params.interiorTaste;
    if (!this.fieldOnGpu) {
      if (!this.scentWriteNative(params)) this.deposit(params);
      Sim.phase('scentWrite');
      this.fields.diffuse(params.diffuse);
      Sim.phase('field:diffuse1');
      this.fields.diffuse(params.diffuse * 0.65);
      Sim.phase('field:diffuse2');
      this.fields.decay(params.decay);
      Sim.phase('field:decay');
      // After the passes that move it, so a cell grows from what it kept
      // rather than from what it was about to lose.
      this.fields.grow(
        CH.energy,
        params.energyRegrow * t,
        this.energy.cellCap,
        params.groundSmell * t,
      );
      Sim.phase('field:grow');
    }
    this.autoSpawn(params, t);
    Sim.phase('autoSpawn');

    const prev = this.contactAudioPrev;
    this.contactAudioPrev = this.contactAudioNow;
    this.contactAudioNow = prev;
  }

  /**
   * Each wired port pulls its body toward pointing along its own wire. This is
   * an actuator, not a material constraint, so it acts as a torque in the force
   * phase rather than as a position correction inside the solve — a
   * position-level version is inertia-dependent (a light Era snaps 60% of the
   * error per substep where a Con moves 14%) and pumps angular velocity.
   *
   * As torques they simply add, so a fully wired agent comes to rest where its
   * ports' demands cancel. With Lafont's parallel aux axes that equilibrium is
   * mostly set by the principal, and the aux wires make the body nod and sway
   * as their neighbours drift.
   *
   * The target is the neighbour's stem, never the wire's own first rope node.
   * The rope is the least constrained thing in the system; aiming at it makes
   * body and rope chase each other into a runaway.
   */
  /**
   * Pack just the pose and inertia every force pass needs, plus the wire
   * endpoints. Deliberately not the full NEAR meta pack: the force passes run
   * in `beginFrame`, before the solver has decided anything, and they only
   * read where bodies are and which ports a wire joins.
   *
   * Returns the packed count, or -1 when the scene will not fit and the caller
   * should stay on the JS path.
   */
  private packForces(list: Agent[], wireList: Wire[]): number {
    const n = list.length;
    if (
      !nativeSolver.ready ||
      !nativeSolver.bodies ||
      !nativeSolver.wiresNear ||
      !nativeSolver.invInertia ||
      !nativeSolver.kind ||
      !nativeSolver.scale
    ) {
      return -1;
    }
    if (!nativeSolver.canNear(n, wireList.length, 0)) return -1;
    const bodies = nativeSolver.bodies;
    const invI = nativeSolver.invInertia;
    const kinds = nativeSolver.kind;
    const sc = nativeSolver.scale;
    const shared = this.forceBlock;
    // Everything but the wires is already there when a block owns the pack;
    // writing it again would discard what the earlier passes accumulated.
    if (!shared) {
      for (let i = 0; i < n; i++) {
        const a = list[i];
        const o = i * FAR_STRIDE;
        bodies[o + FAR.x] = a.x;
        bodies[o + FAR.y] = a.y;
        bodies[o + FAR.vx] = a.vx;
        bodies[o + FAR.vy] = a.vy;
        bodies[o + FAR.heading] = a.heading;
        bodies[o + FAR.omega] = a.omega;
        // A *force* pass leaves a body mid-rewrite alone, whatever is closing
        // it. Aiming a port at a neighbour half a pixel away is meaningless
        // work and the torque it produces is not: on the spin bench it read
        // peak omega 48 against a bound of 20. The span solve is the one pass
        // that must still move the pair, and it packs its own meta.
        const held = poseHeld(a) || a.locked;
        bodies[o + FAR.locked] = held ? 1 : 0;
        bodies[o + FAR.invMass] = held ? 0 : 1 / Math.max(0.08, a.mass);
        invI[i] = held ? 0 : 1 / Math.max(1e-4, momentOfInertia(a));
        kinds[i] = this.kindCode(a.kind);
        sc[i] = a.scale;
      }
    }
    /*
     * Not cacheable, though it looks it. Endpoints and slots are pure
     * topology, but `wiresNear` and `wires` are two views over the same wasm
     * buffer and the FAR and NEAR solves write their own layout through it
     * every frame — so the table has to be re-laid each time even when nothing
     * about the graph moved. The JS-side wire list above does survive.
     */
    const wires = nativeSolver.wiresNear;
    const ai = this.wireAI;
    const bi = this.wireBI;
    let k = 0;
    for (let e = 0; e < wireList.length; e++) {
      const w = wireList[e];
      const ia = ai[e];
      const ib = bi[e];
      if (ia < 0 || ib < 0) continue;
      const o = k * WIRE_NEAR_STRIDE;
      wires[o + WN.a] = ia;
      wires[o + WN.b] = ib;
      wires[o + WN.aSlot] = this.slotCode(w.a.slot);
      wires[o + WN.bSlot] = this.slotCode(w.b.slot);
      k++;
    }
    this.packedWires = k;
    return n;
  }

  private packedWires = 0;

  /** Read back only what a force pass writes. */
  private unpackSpin(list: Agent[]): void {
    const bodies = nativeSolver.bodies!;
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      if (poseHeld(a)) continue;
      a.omega = bodies[i * FAR_STRIDE + FAR.omega];
    }
  }

  private portTorques(params: Params, dt: number): void {
    const gain = params.portStiff * 320;
    if (gain <= 0 || dt <= 0) return;
    const splay = params.auxSpread * 0.35;
    if (this.portTorquesNative(gain, splay, dt)) return;
    const aim = (agent: Agent, slot: PortSlot, target: { x: number; y: number }): void => {
      // As the packs: a force pass leaves a body mid-rewrite alone.
      if (poseHeld(agent) || agent.locked) return;
      const I = momentOfInertia(agent);
      // Critically damped: a bare proportional torque windmills, and a port
      // that latches half a turn out is exactly the case that sets it going.
      const damp = 1.8 * Math.sqrt(gain * I);
      // Aux ports aim slightly off their neighbour, toward their own side of
      // the body. Aiming straight at it is side-blind: once a left neighbour
      // drifts across the centreline the torque simply turns the body to follow
      // it and holds the crossed pose. Offsetting the setpoint makes the
      // uncrossed pose the stable one, and leaves the drawn axes parallel.
      const root = stemRoot(agent.kind, slot);
      const want = slot === 'p' ? 0 : (root.y < 0 ? 1 : -1) * splay;
      const err = wrapAngle(portExitAngle(agent, slot, target) - want);
      agent.omega += (gain * err - damp * agent.omega) * dt / I;
    };
    for (const wire of this.graph.wires.values()) {
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      aim(A, wire.a.slot, stemWorld(B, wire.b.slot, this.w, this.h));
      aim(B, wire.b.slot, stemWorld(A, wire.a.slot, this.w, this.h));
    }
  }

  /**
   * True while the packed bodies, not the Agent objects, are the live copy.
   * Each native pass then reads and writes them in place and skips its own
   * copy in and out.
   */
  private forceBlock = false;

  /**
   * Open a shared copy for the run of WASM force passes.
   *
   * Only when every pass in that run is native, which is now the only kind
   * there is. It used to refuse whenever `params.uncross` was non-zero,
   * because `uncrossPrincipals` was a JS pass in the middle of the run and
   * would have read stale velocities and then had its own writes overwritten
   * on unpack. That pass is gone — it shipped at 0 and never ran — and with it
   * the one case where this had to decline and let every pass copy for itself.
   */
  private openForceBlock(): boolean {
    this.forceBlock = false;
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    const list = this.forceList();
    if (list.length === 0) return false;
    if (!this.packPose(list)) return false;
    this.forceBlock = true;
    return true;
  }

  /**
   * Positions and headings in the pack must still match the agents. Velocity
   * legitimately differs: moving it is what the force passes are for.
   */
  private auditPack(list: Agent[], data: Float32Array, where: string): void {
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      const fields: [string, number, number][] = [
        ['x', data[o + FAR.x], Math.fround(a.x)],
        ['y', data[o + FAR.y], Math.fround(a.y)],
        ['heading', data[o + FAR.heading], Math.fround(a.heading)],
      ];
      for (const [name, packed, live] of fields) {
        if (packed !== live) {
          throw new Error(
            `${where}: agent ${a.id} ${name} moved after the force block ` +
              `(packed ${packed}, live ${live}). Something between beginFrame ` +
              `and the solve is writing body positions.`,
          );
        }
      }
    }
  }

  /** Copy the packed velocities back, if the block still owns them. */
  private syncForces(): void {
    if (!this.forceBlock) return;
    this.unpackDrift(this.agentList);
    this.forceBlock = false;
  }

  /**
   * Hand ownership back without copying: a native solve has already written
   * the full pose to the agents, so the packed copy has nothing left to give.
   */
  private releaseForceBlock(): void {
    this.forceBlock = false;
  }

  /**
   * Pose, mass and locked flag for every body, in the order `agentList` holds
   * them. The force passes all want the same thing, so the pack is one
   * routine; what differs is the per-pass metadata each one adds.
   */
  private packPose(list: Agent[]): boolean {
    const bodies = nativeSolver.bodies;
    const bm = nativeSolver.bodyMass;
    const invI = nativeSolver.invInertia;
    const kinds = nativeSolver.kind;
    const sc = nativeSolver.scale;
    if (!bodies || !bm || !invI || !kinds || !sc) return false;
    if (!nativeSolver.canNear(list.length, 0, 0)) return false;
    /*
     * Straight out of the store, not through the flyweight.
     *
     * This is the largest JavaScript phase in the frame at fifty thousand
     * bodies, and it was fourteen accessor calls a body: every one of them
     * reaches through `store` and `slot` to arrive at exactly these arrays.
     * The store is the sim's one store, so it is hoisted; the slot is the
     * only thing that varies. Two property loads a body instead of
     * twenty-eight, and the values written are the same values.
     *
     * `kindCode` in particular was a round trip through a string: the store
     * holds the kind as the same small integer the solver wants, the
     * flyweight turned it into `'era'`/`'dup'`/`'con'`, and `this.kindCode`
     * turned it back.
     */
    const st = this.agentStore;
    const X = st.x;
    const Y = st.y;
    const VX = st.vx;
    const VY = st.vy;
    const HD = st.heading;
    const OM = st.omega;
    const LK = st.locked;
    const MS = st.mass;
    const SC = st.scale;
    const KC = st.kindCode;
    for (let i = 0; i < list.length; i++) {
      const sl = list[i].slot;
      const o = i * FAR_STRIDE;
      bodies[o + FAR.x] = X[sl];
      bodies[o + FAR.y] = Y[sl];
      bodies[o + FAR.vx] = VX[sl];
      bodies[o + FAR.vy] = VY[sl];
      bodies[o + FAR.heading] = HD[sl];
      bodies[o + FAR.omega] = OM[sl];
      const mass = MS[sl];
      const scale = SC[sl];
      const kc = KC[sl];
      // As `packForces`: a force pass leaves a body mid-rewrite alone.
      const held = poseHeldAt(st, sl) || LK[sl] !== 0;
      bodies[o + FAR.invMass] = held ? 0 : 1 / Math.max(0.08, mass);
      bodies[o + FAR.locked] = held ? 1 : 0;
      // Radius is deliberately absent: no force pass reads it. The broad
      // phases here take their cell size as an argument.
      bm[i] = mass;
      invI[i] = held ? 0 : 1 / Math.max(1e-4, momentOfInertiaAt(kc, mass, scale));
      kinds[i] = kc;
      sc[i] = scale;
    }
    return true;
  }

  /** Read back what a force pass writes: linear and angular velocity. */
  private unpackDrift(list: Agent[]): void {
    const bodies = nativeSolver.bodies!;
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      if (poseHeld(a)) continue;
      const o = i * FAR_STRIDE;
      a.vx = bodies[o + FAR.vx];
      a.vy = bodies[o + FAR.vy];
      a.omega = bodies[o + FAR.omega];
    }
  }

  /**
   * The bodies, in the order every pass indexes them by: the roster's own,
   * which is insertion order and so ascending id.
   *
   * The one list. Rebuilt when the roster version moves and not otherwise,
   * and every pass that wants a dense array of bodies reads it — the solve,
   * the packs, the grids, flocking, the wire adjacency, the GPU unpack. It
   * used to be rebuilt by six different methods from the same Map in the
   * same order, and they agreed only because nothing enforced anything else;
   * `wireAdjacency` and `unpackGenome` both assume this order, so one pass
   * sorting or filtering the shared array would have corrupted both silently.
   * Every writer of the roster bumps `rosterVersion` (`spawn`, `kill`,
   * `clear`, and `noteRosterChange` for the two modules that write the Map
   * directly), which is what makes the cache safe to share.
   */
  private forceList(): Agent[] {
    const list = this.agentList;
    if (this.listRoster === this.rosterVersion) return list;
    list.length = 0;
    for (const a of this.agents.values()) list.push(a);
    const cap = this.agentStore.capacity;
    if (this.slotIndex.length < cap) this.slotIndex = new Int32Array(cap * 2);
    const at = this.slotIndex;
    at.fill(-1);
    for (let i = 0; i < list.length; i++) at[list[i].slot] = i;
    this.listRoster = this.rosterVersion;
    return list;
  }

  /**
   * The latch pass, handed the resolved wire endpoints it needs.
   *
   * It has to know which two bodies each standing wire joins, so it can refuse
   * a latch whose chord would cross one. `wireListResolved` already answers
   * that and caches the answer on the graph and roster versions, which are
   * exactly the two things that can invalidate it. Tests reach for this rather
   * than `graph.snap` so there is one path, and it is the one the frame runs.
   */
  latchPass(params: Params): void {
    this.wireListResolved();
    this.graph.snap(
      this.agents,
      this.w,
      this.h,
      params,
      this.time,
      this.wirePack,
      this.wireEndA,
      this.wireEndB,
    );
  }

  private listRoster = -1;
  /**
   * Where each store slot sits in `forceList()`, or -1 for a slot nobody
   * lives in. Rebuilt with the list, on the same key.
   *
   * The one index. Six `Map<id, index>` used to answer this question, one
   * per pass, each rebuilt every frame — a hundred thousand map writes a
   * frame at fifty thousand bodies to describe an order that only changes
   * when the roster does. A body's slot is already in hand wherever this is
   * asked, so the answer is an array read.
   */
  private slotIndex = new Int32Array(0);
  /** Scratch owned by `packFar`; see the note there. */
  private readonly farWirePack: Wire[] = [];
  private farA = new Int32Array(0);
  private farB = new Int32Array(0);

  private wireListVersion = -1;
  private wireListRoster = -1;
  private readonly wireEndA: (Agent | undefined)[] = [];
  private readonly wireEndB: (Agent | undefined)[] = [];
  /**
   * The same two endpoints as positions in `forceList()`, or -1.
   *
   * Every pass that walks wires wants this and every one of them used to
   * ask a `Map` for it, twice a wire: the flocking adjacency, the body
   * graph, the need field, and all four packs. At thirty thousand wires
   * that is a few hundred thousand lookups a frame for a table that changes
   * only when the graph or the roster does, which is the same key this list
   * is already cached on.
   */
  private wireAI = new Int32Array(0);
  private wireBI = new Int32Array(0);

  /**
   * The wires in Map order, with both endpoints already resolved to agents.
   *
   * Both halves are pure topology. Resolving endpoints was two Map lookups per
   * wire in each of a dozen passes — 1.9ms a frame in the wall-mask pass alone
   * at 29,600 wires, which is more than the pass spends on arithmetic.
   *
   * Keyed on the roster as well as the graph version, because an agent that
   * dies takes its wires with it but the array of resolved references would
   * otherwise keep them alive and hand out a body that is no longer in the sim.
   */
  private wireListResolved(): Wire[] {
    const list = this.wirePack;
    if (
      this.wireListVersion === this.graph.version &&
      this.wireListRoster === this.rosterVersion
    ) {
      return list;
    }
    // Before the loop: it is what turns an endpoint into an index, and it
    // stamps the roster this list is being cut against.
    const at = (this.forceList(), this.slotIndex);
    list.length = 0;
    const eA = this.wireEndA;
    const eB = this.wireEndB;
    eA.length = 0;
    eB.length = 0;
    const m = this.graph.wires.size;
    if (this.wireAI.length < m) {
      this.wireAI = new Int32Array(m * 2);
      this.wireBI = new Int32Array(m * 2);
    }
    const ai = this.wireAI;
    const bi = this.wireBI;
    let k = 0;
    for (const w of this.graph.wires.values()) {
      list.push(w);
      const A = this.agents.get(w.a.id);
      const B = this.agents.get(w.b.id);
      eA.push(A);
      eB.push(B);
      ai[k] = A ? at[A.slot] : -1;
      bi[k] = B ? at[B.slot] : -1;
      k++;
    }
    this.wireListVersion = this.graph.version;
    this.wireListRoster = this.rosterVersion;
    return list;
  }
  private scratchFresh = false;

  /**
   * The per-body scratch the force passes read out of wasm memory: which
   * principal ports are free, what each principal wire's far end is, each
   * body's component, and whether every port is filled.
   *
   * All four are functions of the topology and the roster and none of them of
   * the pose, so like the flocking pair list they only change when the graph
   * does — and rebuilding them every frame was most of what the force passes
   * cost. Measured at 20k bodies, steer, portTorques and declutter spent
   * 14.5ms a frame between them handing over data for passes that take 1-4.5ms
   * to run.
   *
   * The arrays live in wasm memory, which every Sim in the process shares, so
   * ownership is tracked in the binding exactly as the flocking cache is.
   */
  /** Topology this body's `bound` was last computed for; -1 forces a rebuild. */
  private boundVersion = -1;
  private boundRoster = -1;

  /**
   * Fill every body's `bound` — the fraction of its ports that are attached.
   *
   * Its own pass rather than a line inside `refreshForceScratch`, which is
   * where the free-port bitmask is already built, because that one gives up
   * early whenever the native solver is absent or the pond has outgrown its
   * buffers. `BOUND` is a term in the chemistry now, so a body reading zero
   * because a solver did not initialise would not be a missing optimisation,
   * it would be a different genome expressing itself.
   *
   * Keyed on the graph and roster versions, which is exactly what port
   * occupancy depends on — so this is free on the frames when nothing latched,
   * detached or died, which is nearly all of them.
   */
  private refreshComponents(): Int32Array {
    if (this.compVersion === this.graph.version && this.compRoster === this.rosterVersion) {
      return this.componentOf;
    }
    const list = this.forceList();
    const n = list.length;
    const wires = this.wireListResolved();
    const ai = this.wireAI;
    const bi = this.wireBI;
    if (this.componentOf.length < n) this.componentOf = new Int32Array(Math.max(16, n * 2));
    const parent = this.componentOf;
    for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x: number): number => {
      let r = x;
      while (parent[r] !== r) r = parent[r];
      let cur = x;
      while (parent[cur] !== r) {
        const nx = parent[cur];
        parent[cur] = r;
        cur = nx;
      }
      return r;
    };
    for (let k = 0; k < wires.length; k++) {
      const a = ai[k];
      const b = bi[k];
      if (a < 0 || b < 0) continue;
      const ra = find(a);
      const rb = find(b);
      if (ra !== rb) parent[ra] = rb;
    }
    // Flattened, so every later read is one array access rather than a walk.
    for (let i = 0; i < n; i++) parent[i] = find(i);
    this.compVersion = this.graph.version;
    this.compRoster = this.rosterVersion;
    return parent;
  }

  /** `params.interiorTaste`, cached per frame for `tasteGain`. */
  private interiorTaste = 1;

  /**
   * How much of the field a body is allowed to smell: all of it at a leaf,
   * `interiorTaste` anywhere else.
   *
   * **A body with more than one wire stops tasting the ground.** It still
   * emits, still relays, still runs its reactor; it just has no reading of its
   * own. The argument is the inchworm bench's, where it is not decoration but
   * the thing that makes a relayed field mean anything: with every node
   * sighted a chain of 24 shows 5.42 "heads", almost all of them body
   * curvature rather than a real lobe, and blinding the interior makes every
   * head a leaf *by construction* — across every run there, no blind node ever
   * became one. It cost nothing to do: 14 of 24 blinded, 33 captures against
   * 35 sighted, approach speeds within noise. The extremities carry the
   * directional information and the interior readings are redundant.
   *
   * On wire count and not on kind, which is what the calculus allows to be
   * said about a body. In today's grown nets the two coincide — every leaf in
   * `nets/deep-87` and `nets/mixed-308` is an Era, because an Era has one port
   * and a Con or a Dup in a grown net is saturated — so this stacks a fifth
   * job on that one port, beside leaf, limb, feeder and anchor. The
   * coincidence is the net's; the rule is the graph's.
   *
   * A loner has no wires at all, so it keeps everything it had: it senses, it
   * steers, it forages. Only nets change.
   *
   * Applied as a gain on the *taste* row rather than as a branch, which is
   * what makes it one number in `packTaste` and no new kernel, no new uniform
   * and no new packed field on either device path: a zero taste row dots to
   * zero against any field, and `scentSlowFactor` and `scentTurnBoost` fall to
   * neutral on their own.
   */
  private tasteGain(slot: number): number {
    return this.agentStore.wires[slot] > 1 ? this.interiorTaste : 1;
  }

  private refreshBound(): void {
    if (this.boundVersion === this.graph.version && this.boundRoster === this.rosterVersion) {
      return;
    }
    this.boundVersion = this.graph.version;
    this.boundRoster = this.rosterVersion;
    const g = this.graph;
    const BOUND_OF = this.agentStore.bound;
    const WIRES_OF = this.agentStore.wires;
    for (const a of this.agents.values()) {
      const n = a.kind === 'era' ? 1 : 3;
      const sl = a.slot;
      let filled = 0;
      for (let k = 0; k < n; k++) if (!g.isFreeAtSlot(sl, k)) filled++;
      BOUND_OF[sl] = filled / n;
      // The count as well as the fraction. `bound` is saturation — a wired Era
      // and a saturated Con both read 1.0 — and saturation cannot tell a leaf
      // from a hub. `interiorTaste` needs the degree.
      WIRES_OF[sl] = filled;
    }
  }

  /**
   * Bit per unattached port: principal 1, left 2, right 4.
   *
   * By slot, not by id: this is asked of every body by both GPU packs, and
   * the caller is holding the body when it asks.
   */
  private freePortMask(a: Agent): number {
    const g = this.graph;
    const sl = a.slot;
    const p = g.isFreeAtSlot(sl, 0) ? 1 : 0;
    if (a.kind === 'era') return p;
    return p | (g.isFreeAtSlot(sl, 1) ? 2 : 0) | (g.isFreeAtSlot(sl, 2) ? 4 : 0);
  }

  /**
   * The wire graph as flat neighbour lists over `forceList()` order.
   *
   * Rebuilt only when the topology or the roster has moved, like every other
   * thing derived from them. It used to be built twice a frame on the GPU
   * path — once for the need field and once for the genome, each with its
   * own id-to-index map — on the grounds that rewrites and deaths happen
   * between the two. They do, and they bump the versions this is keyed on,
   * so on the frames they happen the second build still runs; on the frames
   * they do not, which is most of them, it is free.
   */
  private wireAdjacency(): WireAdjacency {
    if (this.adjGraphVersion === this.graph.version && this.adjRosterVersion === this.rosterVersion) {
      return this.wireAdj;
    }
    /*
     * The body adjacency CSR again, aliased rather than copied.
     *
     * Four passes wanted the wire graph as neighbour lists — the two LOD
     * passes, flocking, and this one for the need field and the genome — and
     * each built its own from the same wires in the same order on the same
     * key. `refreshBodyAdjacency` is the one that builds it; this hands the same
     * two arrays to the energy passes, which only read them. The queue
     * scratch stays this object's own, since the relaxation does write that.
     */
    this.refreshBodyAdjacency();
    this.wireAdj.off = this.adjOff;
    this.wireAdj.nei = this.adjNei;
    this.adjGraphVersion = this.graph.version;
    this.adjRosterVersion = this.rosterVersion;
    return this.wireAdj;
  }

  /**
   * Every wire whose two principals are ready to rewrite, with neither end
   * already in a rewrite. `pulseRequests`, `accrueRedexes` and
   * `startRewrites` all read this rather than each walking the wire map.
   *
   * `rewriteFrozen` is the busy set: it is collected from `rewrites` at the
   * top of the frame and nothing adds to `rewrites` before `startRewrites`,
   * which is the last of the three. Within that pass no two ready wires can
   * share an end — a principal has one wire — so nothing has to be added to
   * it as rewrites begin.
   */
  private collectReadyRedexes(params: Params): void {
    const out = this.readyRedexes;
    out.length = 0;
    if (params.rewriteDuration <= 0) return;
    const busy = this.rewriteFrozen;
    for (const wire of this.graph.wires.values()) {
      if (wire.a.slot !== 'p' || wire.b.slot !== 'p') continue;
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      if (busy.has(A.id) || busy.has(B.id)) continue;
      if (!this.principalRedexReady(wire, A, B, params)) continue;
      out.push(wire);
    }
  }

  private refreshForceScratch(list: Agent[]): void {
    const n = list.length;
    this.scratchFresh = nativeSolver.scratchHolds(
      this.simId,
      this.graph.version,
      this.rosterVersion,
      n,
    );
    if (this.scratchFresh || n === 0) return;
    const flags = nativeSolver.steerFlags;
    const pwire = nativeSolver.steerPwire;
    const comp = nativeSolver.declComp;
    const sat = nativeSolver.declSat;
    const free = nativeSolver.portFree;
    if (!flags || !pwire || !comp || !sat || !free) return;
    if (!nativeSolver.ready || n > nativeSolver.bodyCap) return;

    const comps = this.refreshComponents();
    const at = this.slotIndex;
    const g = this.graph;
    for (let i = 0; i < n; i++) {
      const a = list[i];
      // Bit 0 is "principal port free". Bit 2 is stun, which is per-frame and
      // written by the steer pass on top of this.
      const pFree = g.isFreeAtSlot(a.slot, 0);
      flags[i] = pFree ? STEER_FLAG.pFree : 0;
      sat[i] = g.portsFilledAt(a) ? 1 : 0;
      // Bitmask of free ports, for the scent deposit in endFrame. Built here
      // rather than there so it is walked once per topology instead of once
      // per body per frame.
      free[i] = this.freePortMask(a);
      comp[i] = comps[i];
      const pw = g.wireAtSlot(a.id, 'p');
      if (pw) {
        const other = pw.a.id === a.id ? pw.b : pw.a;
        const ob = this.agents.get(other.id);
        pwire[i * 2] = ob ? at[ob.slot] : -1;
        pwire[i * 2 + 1] = this.slotCode(other.slot);
      } else {
        pwire[i * 2] = -1;
        pwire[i * 2 + 1] = 0;
      }
    }
    nativeSolver.claimScratch(this.simId, this.graph.version, this.rosterVersion, n);
    this.scratchFresh = true;
  }

  /** Declutter in WASM. False when the scene will not pack. */
  private declutterNative(reach: number, atReach: number, cutoff: number, dt: number): boolean {
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    const comp = nativeSolver.declComp;
    const sat = nativeSolver.declSat;
    if (!comp || !sat) return false;
    const list = this.forceList();
    const n = list.length;
    if (n === 0) return true;
    if (!this.forceBlock && !this.packPose(list)) return false;
    if (!this.scratchFresh) {
      // Only when the topology cache could not be claimed — a scene too big
      // for the solver, or another Sim holding the arrays.
      const comps = this.refreshComponents();
      for (let i = 0; i < n; i++) {
        sat[i] = this.graph.portsFilledAt(list[i]) ? 1 : 0;
        comp[i] = comps[i];
      }
    }
    nativeSolver.declutter(n, reach, atReach, cutoff, Sim.DECLUTTER_FLOOR, dt);
    if (!this.forceBlock) this.unpackDrift(list);
    return true;
  }

  /** The same pass in WASM. False when the scene will not pack. */
  private portTorquesNative(gain: number, splay: number, dt: number): boolean {
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    const list = this.forceList();
    if (list.length === 0) return true;
    const wireList = this.wireListResolved();
    if (wireList.length === 0) return true;
    if (this.packForces(list, wireList) < 0) return false;
    nativeSolver.portTorques(list.length, this.packedWires, gain, splay, dt);
    if (!this.forceBlock) this.unpackSpin(list);
    return true;
  }

  private collectRewriteFrozen(_params: Params): void {
    this.rewriteFrozen.clear();
    this.rewriteWires.clear();
    for (const rw of this.rewrites) {
      this.rewriteFrozen.add(rw.a);
      this.rewriteFrozen.add(rw.b);
      if (rw.wireId >= 0) this.rewriteWires.add(rw.wireId);
    }
  }

  /**
   * The principal wires the live rewrites are consuming.
   *
   * **The one wire a rewrite is allowed to pull on.** A body mid-rewrite is
   * frozen out of every force pass and every wire touching it is skipped by
   * the solver, which is right for its *leftovers* — a rewrite that hauled its
   * neighbours in would make every commute a contraction pump, and the net's
   * shape would be whatever the last one left behind. But the wire being
   * consumed was skipped with them, so `Wire.collapse` spent the whole
   * duration hauling on a rest length nothing read, and the pair was closed
   * instead by `advanceRewrite` assigning positions to two bodies that physics
   * was not touching.
   *
   * Now that one wire is solved. The pair closes because its own wire is
   * shortening, shared by inverse mass, at whatever rate the tension allows —
   * and it still cannot move anything else, because everything else attached
   * to it is still skipped.
   */
  private readonly rewriteWires = new Set<number>();

  private agentDetailed(id: number): boolean {
    return !this.lodActive || this.detailedAgents.has(id);
  }

  private adjVersion = -1;
  private adjRoster = -1;
  private readonly adjList: Agent[] = [];
  private adjOff = new Int32Array(1);
  private adjNei = new Int32Array(0);
  private adjCursor = new Int32Array(0);

  /**
   * Bodies and their wire adjacency in CSR, rebuilt only when topology moves.
   *
   * The version this was ported from rebuilt an array-of-arrays every frame —
   * the same per-frame rebuild of topology-derived scratch that flocking and
   * the force passes were each doing, and which cost more than the work it fed.
   *
   * It was `refreshWakeGraph` until the activity LOD went, because a wake
   * front spreading two hops along the wires was the first thing that wanted
   * it. That reader is gone and four remain — `wireAdjacency`,
   * `assignPhysicsLod`, `flock` and `flockNative` — so it is named for what it
   * is rather than for whoever asked first.
   */
  private refreshBodyAdjacency(): number {
    const list = this.adjList;
    if (
      this.adjVersion === this.graph.version &&
      this.adjRoster === this.rosterVersion
    ) {
      return list.length;
    }
    // The shared list, so the adjacency is cut against exactly the order
    // every other pass indexes by.
    const src = this.forceList();
    list.length = 0;
    for (let i = 0; i < src.length; i++) list.push(src[i]);
    const n = list.length;
    if (this.adjOff.length < n + 2) this.adjOff = new Int32Array(n * 2 + 4);
    if (this.adjCursor.length < n) this.adjCursor = new Int32Array(n * 2);
    const off = this.adjOff;
    off.fill(0, 0, n + 2);
    // Counting sort into CSR: one pass to count degrees, one to place.
    const wires = this.wireListResolved();
    const wai = this.wireAI;
    const wbi = this.wireBI;
    let edges = 0;
    for (let k = 0; k < wires.length; k++) {
      const ia = wai[k];
      const ib = wbi[k];
      if (ia < 0 || ib < 0 || ia === ib) continue;
      off[ia + 1]++;
      off[ib + 1]++;
      edges += 2;
    }
    for (let i = 0; i < n; i++) off[i + 1] += off[i];
    if (this.adjNei.length < edges) this.adjNei = new Int32Array(edges * 2);
    const nei = this.adjNei;
    const cursor = this.adjCursor;
    for (let i = 0; i < n; i++) cursor[i] = off[i];
    for (let k = 0; k < wires.length; k++) {
      const ia = wai[k];
      const ib = wbi[k];
      if (ia < 0 || ib < 0 || ia === ib) continue;
      nei[cursor[ia]++] = ib;
      nei[cursor[ib]++] = ia;
    }
    this.adjVersion = this.graph.version;
    this.adjRoster = this.rosterVersion;
    return n;
  }
  /** True when this body is on the SAT / XPBD rope path. */
  isPhysicsDetailed(id: number): boolean {
    return this.agentDetailed(id);
  }

  /**
   * A wire keeps its live XPBD rope when either end is detailed *and* the
   * stroke is still wide enough to see the rope in. Sub-pixel is where the
   * chord and the rope draw the same streak, so the nodes buy nothing.
   *
   * This is a wire test, not a body test. Reading a stroke width as a verdict
   * on body physics is what put a hard cliff in the middle of the zoom range:
   * every agent went from SAT to packed-FAR in one wheel notch at zoom 0.407,
   * which is nowhere near either agent band edge.
   */
  wireDetailed = (wire: Wire): boolean =>
    this.ropesDrawable && (this.agentDetailed(wire.a.id) || this.agentDetailed(wire.b.id));

  /** SAT neighbourhood, and the rope is still a live XPBD chain. */
  wireSimulatesRope = (wire: Wire): boolean => ropeIsLive(wire, this.wireDetailed);

  private kindCode(kind: AgentKind): number {
    return kind === 'era' ? KIND_ERA : kind === 'dup' ? KIND_DUP : KIND_CON;
  }

  private slotCode(slot: PortSlot): number {
    return slot === 'p' ? 0 : slot === 'l' ? 1 : 2;
  }

  /**
   * Size on screen, then lift a small neighbourhood so a machine under the
   * cursor does not have half its ropes on the cheap path. The whole
   * connected component used to come along, which at a few hundred latched
   * agents meant one on-screen body put the entire soup on SAT.
   */
  private assignPhysicsLod(view: PanView | null | undefined): void {
    this.detailedAgents.clear();
    this.lodActive = !!(view && view.zoom > 0 && view.viewW > 0 && view.viewH > 0);
    if (!this.lodActive) {
      this.ropesDrawable = true;
      this.physLod.sweep();
      return;
    }
    this.ropesDrawable = wiresDrawable(view!.zoom);
    const seeds: number[] = [];
    // Apparent size decides this, and nothing else. Every body is tiered every
    // frame even when the answer is FAR, so the hysteresis in the selector has
    // the history it needs and a body sitting on a band edge does not flip
    // representation each time the wheel moves a notch.
    //
    // Every agent, every frame, unconditionally — the one loop here that
    // can't be skipped even when the whole pond is FAR. Reads the store
    // directly rather than through Agent's accessors: boundRadius's own
    // formula is inlined against kindCode/scale for the same reason.
    const store = this.agentStore;
    const ID = store.id;
    const KIND_CODE = store.kindCode;
    const SCALE = store.scale;
    const X = store.x;
    const Y = store.y;
    for (const a of this.agents.values()) {
      const s = a.slot;
      // boundRadius's own formula, against kindCode instead of the string
      // kind — era is 9px (agentSize), everything else 16px.
      const size =
        (KIND_CODE[s] === KIND_ERA ? ERA_RADIUS + 1.2 : 16 * 1.12) * SCALE[s] * 2;
      const px = apparentPx(size, view);
      const vis = onScreen(X[s], Y[s], size, view);
      if (this.physLod.tier(agentKey(ID[s]), px, vis, AGENT_BAND) !== LOD_FAR) seeds.push(ID[s]);
    }
    // A grab needs a neighbourhood so a pointer drag does not punch through
    // the cheap path. In-flight rewrites deliberately do not get one: seeding
    // them lifts PHYS_HOPS of leftover strings onto XPBD while the kinematic
    // pull is driving them, and that whips the ropes into knots. The pair
    // stays on the chord and tickRewrites owns the motion.
    if (this.grabbed) seeds.push(this.grabbed.id);
    if (seeds.length === 0) {
      this.physLod.sweep();
      return;
    }

    /*
     * The body adjacency CSR rather than an adjacency of its own.
     *
     * This used to build an array-of-arrays every frame — a push per wire
     * end into one of ten thousand arrays — over the same wires, in the same
     * order, that `refreshBodyAdjacency` already keeps as a flat CSR keyed on the
     * graph and the roster. Worse, it built it into `flockAdj`, which
     * `flock` caches on exactly that key and would happily have gone on
     * using: the two agreed only because they were building the same thing.
     */
    const n = this.refreshBodyAdjacency();
    const list = this.adjList;
    const at = this.slotIndex;
    const off = this.adjOff;
    const nei = this.adjNei;
    if (this.flockDist.length < n) {
      const cap = Math.max(n * 2, 16);
      this.flockDist = new Int32Array(cap);
      this.flockQ = new Int32Array(cap);
    }
    const hop = this.flockDist;
    const q = this.flockQ;
    hop.fill(-1, 0, n);
    let qt = 0;
    for (const id of seeds) {
      const a = this.agents.get(id);
      if (!a) continue;
      const i = at[a.slot];
      if (i < 0 || hop[i] >= 0) continue;
      hop[i] = 0;
      q[qt++] = i;
      this.detailedAgents.add(id);
    }
    let qh = 0;
    while (qh < qt) {
      const u = q[qh++];
      const du = hop[u];
      if (du >= Sim.PHYS_HOPS) continue;
      const a0 = off[u];
      const a1 = off[u + 1];
      for (let k = a0; k < a1; k++) {
        const v = nei[k];
        if (hop[v] >= 0) continue;
        hop[v] = du + 1;
        q[qt++] = v;
        this.detailedAgents.add(list[v].id);
      }
    }
    this.physLod.sweep();
  }

  /**
   * Ropes drape off bodies they are not attached to. Wire–wire pairs are
   * detected for friction audio but not displaced — two strings scrape, they
   * do not shove each other off the chord.
   *
   * Segment vs the body's bounding circle, still one-way: a rope never moves
   * an agent. That is what makes it safe — letting a wire shove its own anchors
   * is exactly the coupling that made the early drafts of this solver explode.
   * Node-only tests let a chord cut through a body between two nodes that each
   * sat just outside it.
   *
   * Solved inside the substep loop rather than after the frame, so the link,
   * bend and shape constraints get to re-settle the rope around the push
   * instead of the rope ending each frame off its own manifold. It is rate
   * limited for the same reason everything else here is: a displacement
   * resolved in one substep becomes that displacement times 1/h in velocity.
   */
  private clearWires(params: Params): void {
    const gain = params.wireClear;
    if (gain <= 0) return;
    const cap = Sim.WIRE_CLEAR_STEP;
    const frozen = this.rewriteFrozen;

    for (let k = 0; k < this.clearBodyPairs.length; k += 2) {
      const wire = this.clearBodyPairs[k] as Wire;
      const agent = this.clearBodyPairs[k + 1] as Agent;
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      if (frozen.has(A.id) || frozen.has(B.id)) continue;
      const sA = stemWorldInto(A, wire.a.slot, this.w, this.h, this.tmpStemA);
      const sB = stemWorldInto(B, wire.b.slot, this.w, this.h, this.tmpStemB);
      const keep = boundRadius(agent) + WIRE_RADIUS;
      // Bounds inline rather than through ropeAabb: this runs once per body
      // pair per substep, eight times a frame, and the returned box was the
      // single largest source of garbage in the step.
      let minX = sA.x < sB.x ? sA.x : sB.x;
      let maxX = sA.x > sB.x ? sA.x : sB.x;
      let minY = sA.y < sB.y ? sA.y : sB.y;
      let maxY = sA.y > sB.y ? sA.y : sB.y;
      for (let i = 0; i < wire.nodes.length; i++) {
        const nd = wire.nodes[i];
        if (nd.x < minX) minX = nd.x;
        else if (nd.x > maxX) maxX = nd.x;
        if (nd.y < minY) minY = nd.y;
        else if (nd.y > maxY) maxY = nd.y;
      }
      if (
        maxX < agent.x - keep ||
        minX > agent.x + keep ||
        maxY < agent.y - keep ||
        minY > agent.y + keep
      ) {
        continue;
      }
      const n = wire.nodes.length;
      for (let i = 0; i <= n; i++) {
        const p0 = i === 0 ? sA : wire.nodes[i - 1];
        const p1 = i === n ? sB : wire.nodes[i];
        if (
          Math.max(p0.x, p1.x) < agent.x - keep ||
          Math.min(p0.x, p1.x) > agent.x + keep ||
          Math.max(p0.y, p1.y) < agent.y - keep ||
          Math.min(p0.y, p1.y) > agent.y + keep
        ) {
          continue;
        }
        const t = closestTOnSegment(agent.x, agent.y, p0.x, p0.y, p1.x, p1.y);
        const qx = p0.x + (p1.x - p0.x) * t;
        const qy = p0.y + (p1.y - p0.y) * t;
        let dx = qx - agent.x;
        let dy = qy - agent.y;
        let d = Math.hypot(dx, dy);
        if (d >= keep) continue;
        if (d < 1e-8) {
          const ex = p1.x - p0.x;
          const ey = p1.y - p0.y;
          const len = Math.hypot(ex, ey);
          if (len < 1e-8) continue;
          dx = -ey / len;
          dy = ex / len;
          d = 0;
        } else {
          dx /= d;
          dy /= d;
        }
        const step = Math.min((keep - d) * gain, cap);
        moveRopeClosest(wire, i, t, dx * step, dy * step);
      }
    }

    this.leashRopes();
  }

  /**
   * A stacked clearance step can throw a node far off the chord. Pull it back
   * onto the same tube the renderer will stroke, before Catmull handles turn
   * that into an off-screen loop.
   */
  private leashRopes(): void {
    for (const wire of this.graph.wires.values()) {
      if (wire.nodes.length === 0 || wire.ropePath === 'span') continue;
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      const sA = stemWorldInto(A, wire.a.slot, this.w, this.h, this.tmpStemA);
      const sB = stemWorldInto(B, wire.b.slot, this.w, this.h, this.tmpStemB);
      const span = Math.hypot(sB.x - sA.x, sB.y - sA.y);
      const limit = wireBowBudget(span, wire.rest);
      // The chord is the same for every node, so its projection basis is
      // hoisted and the closest point written out longhand — closestPointOnSegment
      // returned a fresh vector per node per substep.
      const ex = sB.x - sA.x;
      const ey = sB.y - sA.y;
      const eLen2 = ex * ex + ey * ey;
      for (const node of wire.nodes) {
        let tt = eLen2 < 1e-12 ? 0 : ((node.x - sA.x) * ex + (node.y - sA.y) * ey) / eLen2;
        tt = tt < 0 ? 0 : tt > 1 ? 1 : tt;
        const qx = sA.x + ex * tt;
        const qy = sA.y + ey * tt;
        const dx = node.x - qx;
        const dy = node.y - qy;
        const d = Math.hypot(dx, dy);
        if (d <= limit) continue;
        const k = limit / d;
        node.x = qx + dx * k;
        node.y = qy + dy * k;
      }
    }
  }

  /**
   * Wire/body pairs close enough to be worth testing, rebuilt once per frame.
   * The narrow phase runs every substep, so pairing them up each substep costs
   * eight times what it needs to; the margins here are generous enough that a
   * frame of drift cannot smuggle a pair past it.
   *
   * There used to be a wire/wire list beside this one, kept even when
   * clearance was off because it fed slip-slide audio. Ropes never displaced
   * each other, so with that audio gone the whole broad phase went with it.
   */
  private buildClearPairs(params: Params): void {
    this.clearBodyPairs.length = 0;
    const slack = params.wireMinRest;

    const wires = this.clearWireList;
    wires.length = 0;
    let maxRope = 0;
    for (const wire of this.graph.wires.values()) {
      if (wire.nodes.length === 0 || wire.ropePath === 'span') continue;
      wires.push(wire);
      if (wire.ropeLen > maxRope) maxRope = wire.ropeLen;
    }
    const m = wires.length;
    if (m === 0) return;

    if (this.wx.length < m) {
      this.wx = new Array(m * 2);
      this.wy = new Array(m * 2);
    }
    for (let i = 0; i < m; i++) {
      const mid = wires[i].nodes[wires[i].nodes.length >> 1];
      this.wx[i] = mid.x;
      this.wy[i] = mid.y;
    }
    let maxBody = 0;
    for (const agent of this.agents.values()) maxBody = Math.max(maxBody, boundRadius(agent));
    const list = this.rebuildBodyGrid(maxRope * 0.5 + maxBody + WIRE_RADIUS + slack);
    for (let i = 0; i < m; i++) {
      const P = wires[i];
      const px = this.wx[i];
      const py = this.wy[i];
      const reach = P.ropeLen * 0.5 + maxBody + WIRE_RADIUS + slack;
      this.bodyGrid.forEachNear(px, py, reach, (k) => {
        const agent = list[k];
        if (agent.id === P.a.id || agent.id === P.b.id) return;
        if (!this.wireDetailed(P) && !this.agentDetailed(agent.id)) return;
        const span = P.ropeLen * 0.5 + boundRadius(agent) + WIRE_RADIUS + slack;
        const dx = agent.x - px;
        const dy = agent.y - py;
        if (dx * dx + dy * dy > span * span) return;
        this.clearBodyPairs.push(P, agent);
      });
    }
  }


  /**
   * Constraint relaxation with no forces and no integration, so the pointer can
   * still arrange a net while the sim is paused — which is exactly when you
   * would want to lay one out by hand. Velocities are cleared afterwards so
   * unpausing does not release stored-up correction as a kick.
   */
  dragStep(params: Params, dt: number, view?: PanView | null): void {
    if (!this.grabbed || dt <= 0) return;
    const h = dt / Sim.SUBSTEPS;
    this.collectRewriteFrozen(params);
    this.assignPhysicsLod(view);
    this.graph.syncRest(this.time, params, this.agents, this.agentStore.gaitWave, this.wireDetailed);
    this.graph.applyRopePaths(this.agents, this.w, this.h, this.time, params);
    this.graph.syncRopeShape(this.agents, this.w, this.h, this.wireDetailed);
    this.buildClearPairs(params);
    for (let sub = 0; sub < Sim.SUBSTEPS; sub++) {
      this.graph.solveWires(this.agents, params, h, this.time, this.rewriteFrozen, this.wireDetailed, this.rewriteWires);
      this.clearWires(params);
      this.solveGrab(h);
      this.solveContacts(h);
    }
    for (const a of this.agents.values()) {
      a.vx = 0;
      a.vy = 0;
      a.omega = 0;
    }
  }

  /**
   * Pull a held agent toward the pointer. Rate limited like every other
   * constraint here, so grabbing something across the screen reels it in rather
   * than launching it and whatever net it belongs to.
   */
  private solveGrab(h: number): void {
    const held = this.grabbed;
    if (!held) return;
    const agent = this.agents.get(held.id);
    if (!agent || poseHeld(agent)) return;
    const dx = held.x - agent.x;
    const dy = held.y - agent.y;
    const dist = Math.hypot(dx, dy);
    if (dist < 1e-6) return;
    const w = 1 / Math.max(0.08, agent.mass);
    const alphaTilde = Sim.GRAB_COMPLIANCE / Math.max(1e-12, h * h);
    const step = Math.min((dist * w) / (w + alphaTilde), Sim.GRAB_STEP);
    agent.x += (dx / dist) * step;
    agent.y += (dy / dist) * step;
  }

  /**
   * Personal space around a fully wired agent: a soft inverse-square push
   * against agents from *other* nets.
   *
   * A saturated agent has nothing left to join, so a stranger drifting close is
   * pure crowding, and crowding is what tangles nets. Nothing else does this —
   * flocking separation only walks same-net neighbours (and then only a few
   * hops out), so those pairs are skipped entirely and separate nets have never
   * repelled at all.
   *
   * Inverse-square rather than linear, which matters: at a wire's length the
   * push is gentle enough to be ignored, but it climbs steeply as the gap
   * closes, so it still wins where it needs to. A linear falloff strong enough
   * to hold the distance is a wall you can feel; this is a bubble you can lean
   * into. Equal and opposite, so it never moves the flock's centre of mass.
   *
   * Same-net crowding is left to flocking separation, which already handles it
   * with hop-weighted spacing.
   */
  private declutter(params: Params, dt: number): void {
    const gain = params.declutter;
    if (gain <= 0 || dt <= 0) return;
    const reach = params.wireMinRest;
    const cutoff = reach * Sim.DECLUTTER_CUTOFF;
    // Force at exactly one wire's length; it grows as (reach / d)^2 inside that.
    const atReach = gain * Sim.DECLUTTER_FORCE;
    const floor = reach * Sim.DECLUTTER_FLOOR;
    if (this.declutterNative(reach, atReach, cutoff, dt)) return;
    const list = this.rebuildBodyGrid(cutoff);
    const n = list.length;
    // Hoisted out of the inner loop: both were map lookups per pair.
    if (this.satBuf.length < n) {
      this.satBuf = new Uint8Array(n * 2);
      this.compBuf = new Int32Array(n * 2);
      this.slotBuf = new Int32Array(n * 2);
    }
    const sat = this.satBuf;
    const comp = this.compBuf;
    const slot = this.slotBuf;
    const comps = this.refreshComponents();
    for (let i = 0; i < n; i++) {
      sat[i] = this.graph.portsFilledAt(list[i]) ? 1 : 0;
      comp[i] = comps[i];
      slot[i] = list[i].slot;
    }
    const store = this.agentStore;
    const X = store.x;
    const Y = store.y;
    const VX = store.vx;
    const VY = store.vy;
    const MASS = store.mass;
    const LOCKED = store.locked;
    const PINNED = store.pinned;
    this.bodyGrid.forEachPair((i, j) => {
      {
        if (!sat[i] && !sat[j]) return;
        if (comp[i] === comp[j]) return;
        const a = slot[i];
        const b = slot[j];
        const dx = X[b] - X[a];
        const dy = Y[b] - Y[a];
        const dist = Math.hypot(dx, dy);
        if (dist > cutoff || dist < 1e-6) return;
        // Floored so the law cannot run away at touching distance; contacts own
        // that range anyway.
        const ratio = reach / Math.max(dist, floor);
        const force = atReach * ratio * ratio;
        const nx = dx / dist;
        const ny = dy / dist;
        if (!LOCKED[a] && !PINNED[a]) {
          const invM = 1 / Math.max(0.08, MASS[a]);
          VX[a] -= nx * force * invM * dt;
          VY[a] -= ny * force * invM * dt;
        }
        if (!LOCKED[b] && !PINNED[b]) {
          const invM = 1 / Math.max(0.08, MASS[b]);
          VX[b] += nx * force * invM * dt;
          VY[b] += ny * force * invM * dt;
        }
      }
    });
  }
  /**
   * Constrained integration. Wires, port axes and contacts are all compliant
   * constraints solved inside this one loop; nothing outside it writes a pose,
   * and velocity is derived from the result rather than repaired afterwards.
   *
   * Many substeps with a single iteration each converge far better than the
   * reverse at equal cost — Macklin et al., "Small Steps in Physics Simulation".
   */
  private solve(params: Params, dt: number): void {
    if (dt <= 0) return;
    // buildClearPairs reads positions, which the force passes never move, so
    // the block can stay open through the native attempt and be inherited.
    this.buildClearPairs(params);
    if (this.solveNearNative(params, dt)) return;
    // The JS fallback reads the agents outright, so they get their velocities
    // back here — including after a native attempt that packed and bailed.
    this.syncForces();
    const frozen = this.rewriteFrozen;
    const h = dt / Sim.SUBSTEPS;
    const invH = 1 / h;
    const held = this.grabbed?.id ?? -1;
    // Rope velocity is re-derived every substep, so a nudge of e px becomes
    // e/h — damping it once per frame is far too late to keep a slack rope calm.
    const ropeKeep = Math.exp(-Math.max(0, params.springDamp) * h);
    const list = this.forceList();
    const n = list.length;

    // Hoisted once: every agent shares this one store, so a hot loop over
    // `list` can index its typed arrays directly by slot instead of going
    // through Agent's getter/setter accessors per field per agent. See
    // agent-store.ts.
    const store = this.agentStore;
    const X = store.x;
    const Y = store.y;
    const VX = store.vx;
    const VY = store.vy;
    const HEADING = store.heading;
    const OMEGA = store.omega;
    const PREV_X = store.prevX;
    const PREV_Y = store.prevY;
    const PREV_HEADING = store.prevHeading;
    // The *pose* hold, not `locked`: a body mid-rewrite still has mass and
    // is still moved by its wires unless `rewritePull` says otherwise.
    const LOCKED = store.poseLock;
    const PINNED = store.pinned;
    const ID = store.id;

    for (let sub = 0; sub < Sim.SUBSTEPS; sub++) {
      for (let i = 0; i < n; i++) {
        const s = list[i].slot;
        PREV_X[s] = X[s];
        PREV_Y[s] = Y[s];
        PREV_HEADING[s] = HEADING[s];
        if (LOCKED[s] || PINNED[s]) continue;
        X[s] += VX[s] * h;
        Y[s] += VY[s] * h;
        HEADING[s] = wrapAngle(HEADING[s] + OMEGA[s] * h);
      }
      for (const wire of this.graph.wires.values()) {
        if (!this.wireSimulatesRope(wire)) continue;
        const hold = frozen.has(wire.a.id) || frozen.has(wire.b.id);
        for (const node of wire.nodes) {
          node.prevX = node.x;
          node.prevY = node.y;
          if (hold) continue;
          node.x += node.vx * h;
          node.y += node.vy * h;
        }
      }

      this.graph.solveWires(this.agents, params, h, this.time, frozen, this.wireDetailed, this.rewriteWires);
      this.solveGrab(h);
      this.clearWires(params);
      this.solveContacts(h);

      for (let i = 0; i < n; i++) {
        const s = list[i].slot;
        if (LOCKED[s] || PINNED[s]) {
          VX[s] = 0;
          VY[s] = 0;
          OMEGA[s] = 0;
          continue;
        }
        VX[s] = (X[s] - PREV_X[s]) * invH;
        VY[s] = (Y[s] - PREV_Y[s]) * invH;
        OMEGA[s] = wrapAngle(HEADING[s] - PREV_HEADING[s]) * invH;
        if (held === ID[s]) {
          const speed = Math.hypot(VX[s], VY[s]);
          if (speed > Sim.GRAB_MAX_SPEED) {
            const k = Sim.GRAB_MAX_SPEED / speed;
            VX[s] *= k;
            VY[s] *= k;
          }
        }
      }
      if (this.worldR > 0) {
        const cx = this.worldX;
        const cy = this.worldY;
        const R = this.worldR;
        for (let i = 0; i < n; i++) {
          const a = list[i];
          const s = a.slot;
          if (LOCKED[s] || PINNED[s]) continue;
          let maxr = R - discRadius(a);
          if (maxr < 0) maxr = 0;
          const hit = bounceOffDisk(X[s], Y[s], VX[s], VY[s], cx, cy, maxr);
          X[s] = hit.x;
          Y[s] = hit.y;
          VX[s] = hit.vx;
          VY[s] = hit.vy;
        }
      }
      for (const wire of this.graph.wires.values()) {
        if (!this.wireSimulatesRope(wire)) continue;
        for (const node of wire.nodes) {
          node.vx = (node.x - node.prevX) * invH * ropeKeep;
          node.vy = (node.y - node.prevY) * invH * ropeKeep;
          if (this.worldR > 0) {
            const hit = bounceOffDisk(
              node.x, node.y, node.vx, node.vy,
              this.worldX, this.worldY, this.worldR,
            );
            node.x = hit.x;
            node.y = hit.y;
            node.vx = hit.vx;
            node.vy = hit.vy;
          }
        }
      }
    }

    const STUN = store.stun;
    for (let i = 0; i < n; i++) {
      const a = list[i];
      STUN[a.slot] = Math.max(0, STUN[a.slot] - dt);
      wrapPos(a, this.w, this.h);
    }
    this.sanitizePoses();
  }

  private finishIntegrate(dt: number): void {
    this.syncForces();
    for (const a of this.agents.values()) {
      a.stun = Math.max(0, a.stun - dt);
      wrapPos(a, this.w, this.h);
    }
    this.sanitizePoses();
  }

  /** Recover from a blown-up solve so the next frame is still a sim. */
  private sanitizePoses(): void {
    const com = this.home ?? this.centerOfMass();
    const cx = com?.x ?? this.w * 0.5;
    const cy = com?.y ?? this.h * 0.5;
    const lim = 1e6;
    for (const a of this.agents.values()) {
      if (
        !Number.isFinite(a.x) ||
        !Number.isFinite(a.y) ||
        Math.abs(a.x) > lim ||
        Math.abs(a.y) > lim
      ) {
        a.x = cx;
        a.y = cy;
        a.vx = 0;
        a.vy = 0;
      }
      if (!Number.isFinite(a.vx) || !Number.isFinite(a.vy)) {
        a.vx = 0;
        a.vy = 0;
      }
      if (!Number.isFinite(a.heading) || !Number.isFinite(a.omega)) {
        a.heading = 0;
        a.omega = 0;
      }
    }
    for (const wire of this.graph.wires.values()) {
      if (!Number.isFinite(wire.rest) || wire.rest < 0 || wire.rest > 1e6) {
        wire.rest = 40;
      }
      for (const n of wire.nodes) {
        if (!Number.isFinite(n.x) || !Number.isFinite(n.y)) {
          n.x = cx;
          n.y = cy;
          n.vx = 0;
          n.vy = 0;
        }
      }
    }
  }

  private canFarPacked(): boolean {
    return this.lodActive && this.detailedAgents.size === 0 && !this.grabbed;
  }

  private canFarGpu(): boolean {
    return farGpu.ready && this.canFarPacked() && this.agents.size >= 8;
  }

  /**
   * Whether to put the GPU in front of wasm this frame. Capability first: the
   * kernel is the FAR solve only, so a frame needing the NEAR tier is declined
   * whatever the mode says -- that is wrong physics, not slower physics.
   *
   * 'auto' means the GPU only where wasm will not go, which is the pack
   * outgrowing MAX_BODIES or MAX_WIRES. It used to switch at a body count
   * measured to be the crossing point, and that measurement was wrong twice
   * over: it timed `stepFar`, which copies the scene in and back out again,
   * where the sim calls `stepFarInPlace` and pays neither copy; and it ran a
   * uniform scene where a real pond clumps, which costs the GPU's grid far more
   * than it costs a spatial hash. On a mature pond of ~6,000 bodies wasm takes
   * the solve in 7.1ms against the GPU's 13.6, and the whole tick is 13%
   * shorter for it. The crossing, if there is one below the cap, is nowhere
   * near where that number claimed.
   *
   * Reading the caps off the module rather than restating 32768 means this
   * follows if the C ever grows. A missing wasm module reports zero capacity,
   * which lands here as "wasm cannot take it" -- correct, and the reason the
   * check is capacity rather than a constant.
   */
  private wantFarGpu(): boolean {
    if (!this.canFarGpu()) {
      this.farGpuLatched = false;
      return false;
    }
    if (Sim.farGpuMode !== 'auto') {
      this.farGpuLatched = Sim.farGpuMode === 'on';
      return this.farGpuLatched;
    }
    const bodies = this.agents.size;
    const wires = this.graph.wires.size;
    const fits =
      nativeSolver.ready && bodies <= nativeSolver.bodyCap && wires <= nativeSolver.wireCap;
    if (this.farGpuLatched) {
      const release = Sim.FAR_GPU_RELEASE;
      if (
        fits &&
        bodies <= nativeSolver.bodyCap * release &&
        wires <= nativeSolver.wireCap * release
      ) {
        this.farGpuLatched = false;
      }
    } else if (!fits) {
      this.farGpuLatched = true;
    }
    return this.farGpuLatched;
  }

  private packFar(params: Params): {
    list: Agent[];
    data: Float32Array;
    wires: Float32Array;
    nWires: number;
  } | null {
    const list = this.forceList();
    const n = list.length;
    if (n === 0) return null;
    /*
     * Its own wire array, not the shared `wirePack`.
     *
     * `wireListResolved` caches `wirePack` and stamps it valid against the
     * graph and roster versions, with `wireEndA`/`wireEndB` and the endpoint
     * indices resolved to match it position by position. This pass builds a
     * *filtered* list — self-wires dropped, since a body wired to two of its
     * own ports has no span to solve — so borrowing that array left the cache
     * holding a different list under a stamp that still claimed to be
     * current, and every later reader paired wire k with the endpoints of
     * some other wire.
     */
    const all = this.wireListResolved();
    const ai = this.wireAI;
    const bi = this.wireBI;
    const wireList = this.farWirePack;
    if (this.farA.length < all.length) {
      this.farA = new Int32Array(all.length * 2);
      this.farB = new Int32Array(all.length * 2);
    }
    wireList.length = 0;
    let nw = 0;
    for (let k = 0; k < all.length; k++) {
      const a = ai[k];
      const b = bi[k];
      if (a < 0 || b < 0 || a === b) continue;
      this.farA[nw] = a;
      this.farB[nw] = b;
      wireList.push(all[k]);
      nw++;
    }
    const { data, wires } = farGpu.packTarget(n, wireList.length);
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      const locked = poseHeld(a) || this.rewriteFrozen.has(a.id);
      data[o + FAR.x] = a.x;
      data[o + FAR.y] = a.y;
      data[o + FAR.vx] = a.vx;
      data[o + FAR.vy] = a.vy;
      data[o + FAR.heading] = a.heading;
      data[o + FAR.omega] = a.omega;
      data[o + FAR.invMass] = locked ? 0 : 1 / Math.max(0.08, a.mass);
      // FAR never runs SAT, so the contact radius is the glyph-area disc and
      // not the fatter bound the SAT broad phase needs.
      data[o + FAR.radius] = discRadius(a);
      data[o + FAR.locked] = locked ? 1 : 0;
    }
    for (let k = 0; k < wireList.length; k++) {
      const w = wireList[k];
      const A = this.agents.get(w.a.id)!;
      const B = this.agents.get(w.b.id)!;
      const oa = stemOffset(A, w.a.slot);
      const ob = stemOffset(B, w.b.slot);
      const stiff = this.graph.stiffnessOf(w, this.time, params);
      packFarWire(
        wires,
        k,
        this.farA[k],
        this.farB[k],
        w.rest,
        oa.x,
        oa.y,
        ob.x,
        ob.y,
        stiff.scale * stiff.slack,
      );
    }
    return { list, data, wires, nWires: wireList.length };
  }

  private unpackFar(list: Agent[], data: Float32Array): void {
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      if (poseHeld(a) || this.rewriteFrozen.has(a.id)) continue;
      const o = i * FAR_STRIDE;
      a.x = data[o + FAR.x];
      a.y = data[o + FAR.y];
      a.vx = data[o + FAR.vx];
      a.vy = data[o + FAR.vy];
      a.heading = data[o + FAR.heading];
      a.omega = data[o + FAR.omega];
    }
  }

  /**
   * FAR straight against the solver's own buffers.
   *
   * `packFar` writes into a scratch array that `stepFar` then copies into WASM
   * and back out — the whole scene, twice each way. Writing where the solver
   * already reads removes both copies, and when the force block is still open
   * the pose is in there already and only the per-frame metadata is written.
   */
  private solveFarNative(params: Params, dt: number): boolean {
    if (!nativeSolver.ready || !this.canFarPacked()) return false;
    const data = nativeSolver.bodies;
    const wires = nativeSolver.wires;
    if (!data || !wires) return false;

    const list = this.forceList();
    const n = list.length;
    if (n === 0) return true;
    /*
     * Its own array, not the cached `wirePack`: this list drops self-wires,
     * which a body wired to two of its own ports genuinely is, and writing a
     * filtered list into the shared one leaves it stamped current while
     * every later reader pairs wire k with the endpoints of another.
     */
    const all = this.wireListResolved();
    const aiAll = this.wireAI;
    const biAll = this.wireBI;
    const wireList = this.farWirePack;
    if (this.farA.length < all.length) {
      this.farA = new Int32Array(all.length * 2);
      this.farB = new Int32Array(all.length * 2);
    }
    wireList.length = 0;
    let nw = 0;
    for (let k = 0; k < all.length; k++) {
      const a = aiAll[k];
      const b = biAll[k];
      if (a < 0 || b < 0 || a === b) continue;
      this.farA[nw] = a;
      this.farB[nw] = b;
      wireList.push(all[k]);
      nw++;
    }
    if (n > nativeSolver.bodyCap || wireList.length > nativeSolver.wireCap) return false;

    const inherited = this.forceBlock;
    if (inherited && Sim.auditForceBlock) this.auditPack(list, data, 'FAR');
    const scale = 12 / Math.max(1, params.springK);
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      const locked = poseHeld(a) || this.rewriteFrozen.has(a.id);
      if (!inherited) {
        data[o + FAR.x] = a.x;
        data[o + FAR.y] = a.y;
        data[o + FAR.vx] = a.vx;
        data[o + FAR.vy] = a.vy;
        data[o + FAR.heading] = a.heading;
        data[o + FAR.omega] = a.omega;
      }
      data[o + FAR.invMass] = locked ? 0 : 1 / Math.max(0.08, a.mass);
      data[o + FAR.radius] = discRadius(a);
      data[o + FAR.locked] = locked ? 1 : 0;
    }
    for (let k = 0; k < wireList.length; k++) {
      const w = wireList[k];
      const A = this.agents.get(w.a.id)!;
      const B = this.agents.get(w.b.id)!;
      stemOffsetInto(A, w.a.slot, this.tmpStemA);
      stemOffsetInto(B, w.b.slot, this.tmpStemB);
      // Inlined rather than through stiffnessOf, which allocates a record per
      // wire for two numbers, one of which does not vary across the pack.
      const soft = scale * (1 + 6 * Math.exp(-Math.max(0, this.time - w.born) / 0.8));
      packFarWire(
        wires,
        k,
        this.farA[k],
        this.farB[k],
        w.rest,
        this.tmpStemA.x,
        this.tmpStemA.y,
        this.tmpStemB.x,
        this.tmpStemB.y,
        soft,
      );
    }
    if (!nativeSolver.stepFarInPlace(n, wireList.length, dt, undefined, this.worldX, this.worldY, this.worldR)) return false;
    this.unpackFar(list, data);
    this.releaseForceBlock();
    return true;
  }

  /** Packed FAR pass on the GPU. True when the kernel ran. */
  /** Where the world grid is anchored. Set once, from the first centre of
   *  mass (or a preset), and fixed for the life of the sim. */
  private worldPinned = false;
  worldX = 0;
  worldY = 0;
  /** Live-disk radius. `0` until `pinWorld`. Shared by the wall, scent, energy, spawn. */
  worldR = 0;

  /**
   * Anchor the field, energy grid, scent mask, and hard rim on `(cx, cy)`.
   * Cover snaps the field origin to a whole cell; the radius is the largest
   * disk that then sits inside that window.
   */
  pinWorld(cx: number, cy: number, params?: Params): void {
    if (this.worldPinned) return;
    /*
     * `params` because pinning is when the ground gets laid down, and how much
     * ground there is is a slider.
     *
     * `beginFrame` reads the sliders into `energyCell`/`energyAmbient` before
     * it pins, so its own pin is covered. The other two callers — `loadPreset`
     * and the designer — pin from outside a frame, where those fields still
     * hold their construction fallbacks. That seeded the shipped soup at 6.4%
     * of the stock it was supposed to have, and the fallback cell size of 48
     * is not a whole multiple of `FIELD_CELL` either, so `index` and `block`
     * disagreed about which cells they were addressing until the first frame
     * corrected both.
     */
    if (params) {
      this.energyCell = params.energyCell;
      this.energyAmbient = params.ambientEnergy;
      this.energyPatches = params.groundPatches;
    }
    this.worldPinned = true;
    this.worldX = cx;
    this.worldY = cy;
    this.home = { x: cx, y: cy };
    this.fields.cover(cx, cy);
    this.worldR = worldBoundRadius(cx, cy, this.fields.originX, this.fields.originY, this.fields.worldW);
    this.fields.setWorldBound(cx, cy, this.worldR);
    this.energy.setBounds(cx, cy, this.worldR, this.fields.originX, this.fields.originY);
    /*
     * The ground moves onto the field, and gets laid down.
     *
     * Here rather than in the constructor because a disk is what makes the
     * ground finite, and there is no disk until the world is pinned. Before
     * this the grid answers out of its sparse map exactly as it always did,
     * which lasts the one frame it takes a pond to acquire a home — and only
     * ever that once. `bind` is not undone by `clear`, so a cleared sim keeps
     * reading a zeroed field rather than falling back to the implicit ambient.
     * That is the right behaviour (a cleared world has no ground until it is
     * seeded again) but it does mean "before the first pin" is the only window
     * in which the sparse path runs at all.
     */
    this.energy.bind(this.fields);
    this.energy.configure(this.energyCell, this.energyAmbient, this.energyPatches);
    this.energy.seedGround();
  }

  /*
   * Last configure, kept so `pinWorld` can seed the ground at the size and
   * capacity the sliders are actually set to. `step` writes these every frame
   * before it harvests; `pinWorld` can run before the first of those.
   */
  private energyCell = 40;
  private energyAmbient = 1;
  private energyPatches = 0;

  /**
   * Tell the field what each channel is, from the sliders, every frame.
   *
   * Cheap — four numbers — and it has to be per frame because two of them are
   * live sliders. Energy is the odd one: it never decays, because it is a
   * quantity and the economy is supposed to conserve it, and it spreads at a
   * fraction of the signal rate so that local scarcity survives long enough
   * to forage against.
   *
   * Aux is the other end of that same trade. It carries the *smell* of the
   * ground rather than the ground, so nothing is lost when it fades and
   * nothing is destroyed when it drifts — which is exactly what lets it run
   * fast where the substance has to run slow. It keeps the ordinary signal
   * decay: a smell that did not fade would still be announcing a patch that
   * was eaten a minute ago.
   */
  private tuneChannels(params: Params): void {
    const f = this.fields;
    f.decayRate[CH.energy] = 0;
    f.diffuseRate[CH.energy] = Math.max(0, params.energyDiffuse);
    f.diffuseRate[CH.aux] = Math.max(0, params.groundSmellSpread);
  }

  /** True once a device exists and the field has moved there for good. */
  private fieldOnGpu = false;

  /**
   * Move the field to the GPU if there is one. Call once, at startup.
   *
   * All or nothing for the session: a field that lived on the GPU on some
   * frames and here on others would have to be copied between them, and the
   * copy is 16 MB each way — several times what running it here costs in the
   * first place.
   */
  /**
   * Refuses while the ground lives on the field, which is always, so this
   * currently cannot succeed.
   *
   * Not an oversight left in — a tripwire in front of one. The GPU holds the
   * live field in its own buffers and only ever ships deposits and probes
   * across; `fields.data` is a CPU copy nobody syncs back. Anything reading or
   * writing that array is on the wrong side of the line.
   *
   * Two of the four hazards this note used to list are closed, and saying so
   * matters: a stale list of blockers reads as a list of *reasons*, and two
   * struck through make the rest look struck through too.
   *
   *   CLOSED (323e3c9) — the ground stopped regrowing, because `grow` had no
   *   shader. `field.wgsl` has one now, fertiliser catalyst and capacity clamp
   *   included, and `gpuFieldStep` feeds it.
   *
   *   CLOSED — channel 2 decayed at the scent rate, because per-channel rates
   *   never reached the shader. The uniform carries `mix`/`mix2`/`keep` as
   *   `vec4f` and `field-gpu.ts` resolves them the way `Fields` does. Note
   *   what this took beyond the shader: `tuneChannels` writes those rates and
   *   was itself inside `if (!this.fieldOnGpu)`, so the capacity existed while
   *   nothing fed it. It is hoisted out of the branch now. A shader that can
   *   express something is not a path that does.
   *
   *   OPEN — `EnergyGrid`'s `take` and `addAt` read and write `fields.data`
   *   directly. This is the load-bearing one. Turn this on as it stands and
   *   the pond mines a CPU array the GPU is not looking at down to nothing,
   *   while farming, death yield and rewrite leftovers land in an array
   *   nothing reads: the whole economy quietly detaches from the field it is
   *   supposed to be an economy of. Harvest has to move to the shader with it.
   *
   *   CLOSED — `updateState` sampled the field on the CPU, through
   *   `Fields.sampleAll` and so out of `fields.data`, which would have been a
   *   stale copy: any body whose genome had evolved a non-zero `Wx` sense
   *   weight read garbage, silently, only on machines with a device, and only
   *   once evolution had moved a gene off its seed. `gather` now returns the
   *   raw four channels under each body alongside the three taste-collapsed
   *   scalars, and `gpuFieldStep` writes them into the store, so the state
   *   pass takes its sense from wherever the field actually lives.
   *
   *   CLOSED — `EnergyGrid` wrote `fields.data` three ways and all three now
   *   go through the shader. `take` is the `harvest` kernel, a faithful port
   *   of the block drain rather than the proportional share a parallel
   *   rewrite would reach for; `addAt` rides the existing scatter with a
   *   conserve flag, because energy is a count and has to survive the rim
   *   where a scent density need not; `seedGround` is the `fill` pass, without
   *   which the pond would have started barren on exactly the machines this
   *   is for.
   *
   * So it opens now. Two things about that are worth knowing rather than
   * rediscovering.
   *
   * It is one way. There is no path back to the CPU field except a device
   * loss, which drops to it for good — a field that lived on the GPU on some
   * frames and here on others would have to be copied between them, and the
   * copy is sixteen megabytes each way, several times what running it here
   * costs in the first place.
   *
   * And `fields.data` is still the array the two debug overlays paint from.
   * They get `wantFieldReadback`, which `render.ts` sets from its own options,
   * so the copy is paid while somebody is looking and not otherwise. Anything
   * else that comes to read that array on this path is reading a stale copy,
   * and this is the fourth time that has been the bug.
   *
   * Nothing calls this today, which is why the whole thing was invisible. It
   * wants either the growth pass in `field.wgsl` and the ground read back, or
   * the ground moved off the shared field, before it is worth having.
   */
  async openFieldGpu(): Promise<boolean> {
    if (this.fieldOnGpu) return true;
    if (!(await fieldGpu.init(this.fields.cols))) return false;
    /*
     * Take the device's field as well as its arithmetic.
     *
     * `fieldGpu` is a module singleton and the buffers outlive whichever `Sim`
     * last used them. `clear()` runs from `Sim.clear`, which only fires for a
     * pond that was *already* on the device — so a second pond opening the
     * device in the same process inherits the first one's scent, ground and
     * accumulator, and starts life standing in somebody else's dish.
     *
     * The page never noticed because it has one `Sim` for the life of the tab.
     * A sweep has one per trial: `pond/sweep.ts` runs a hundred ponds in a
     * process, and every one after the first was reading the last one's field
     * until this line. Found by a device parity test whose second pond
     * excreted onto a channel the first had already filled.
     */
    fieldGpu.clear();
    /*
     * Everything that writes the ground has to be told before the first frame,
     * not on the frame it first tries: `seedGround` may already have run for
     * this world, and a queued seed is only picked up by `gpuFieldStep`.
     */
    this.energy.deferAdds(true);
    this.energy.seedGround();
    this.fieldOnGpu = true;
    /*
     * And the genome, which is strictly downstream: it reads the probe's own
     * output buffer for its sense inputs, so there is nothing for it to read
     * until the field is here. If it declines, the field still runs and
     * `updateState` keeps doing the arithmetic — the two are independent in
     * that direction.
     */
    const dev = fieldGpu.gpuDevice;
    if (dev && (await genomeGpu.init(dev))) {
      this.genomeOnGpu = true;
      /*
       * And its learning, for the reason `fieldGpu.clear()` above is called:
       * the resident learning row is indexed by slot and outlives whichever
       * `Sim` last used it, so this pond's body in slot 0 would start with
       * what the last pond's body in slot 0 had learned. `syncLearn` cannot
       * catch it — it pushes a row only when the host marks that slot dirty,
       * and a fresh `Sim` has nothing dirty to say.
       *
       * Only here, because `openFieldGpu` returns early for a pond that
       * already holds the device: a Sim clears the row when it takes the
       * device and never again, so nothing live is wiped.
       */
      genomeGpu.clearLearning();
    }
    return true;
  }

  /**
   * Hand the field a frame's worth of work and take back what steering needs.
   *
   * Everything geometric happens here rather than in the shader: the port
   * positions to deposit at and the sensor positions to sample come from the
   * same helpers the CPU path uses, so there is one place that knows where a
   * port is rather than three. `field.wgsl` sees bare world coordinates.
   *
   * The samples that come back describe the pose this frame *started* with,
   * and steering reads them at the top of the next one. A frame of latency in
   * smell is invisible at 60fps and much cheaper than stalling the pipeline to
   * map a buffer mid-frame.
   */
  private async gpuFieldStep(params: Params, dt: number): Promise<void> {
    const list = this.forceList();
    const n = list.length;
    // Reserve before taking references, not after: `reserve` reallocates the
    // staging arrays when it grows them, so a reference captured first points
    // at the array they replaced. Done the wrong way round this writes every
    // deposit and probe into a discarded buffer and the GPU reads zeros —
    // silently, because nothing about it is an error.
    /*
     * Built here, at the end of the frame, and spent at the top of the next.
     *
     * The binning is a walk over bodies rather than over the field, so it is
     * CPU work whichever side the field lives on; what crosses is the answer.
     */
    const plan = this.harvestPlan;
    plan.build(this.agents.values(), this.agentStore, this.energy, {
      cap: params.uptakeVmax * dt,
      ks: params.uptakeKs,
      yDirect: params.yDirect,
      yEra: params.yEra,
      hillN: hillOf(params),
      gutSize: params.gutSize,
    });
    Sim.phase('gpu:plan');
    // Three ports a body, plus whatever died, excreted, spilled or was refunded
    // this frame and had nowhere to put it.
    const nAdds = this.energy.pendingAdds;
    fieldGpu.reserve(n * 3 + nAdds, n, plan.nBlocks, plan.nEntries);
    const dep = fieldGpu.depositData;
    const pro = fieldGpu.probeData;
    const dStride = fieldGpu.depositStride;
    const pStride = fieldGpu.probeStride;

    const EMITS = this.agentStore.emitAll;
    const scale = this.fields.depositScale;
    const amt = params.deposit * scale;
    const leak = amt * params.portLeak;
    const auxLeak = amt * params.auxLeak;
    let nDep = 0;
    /*
     * Two shared constants rather than `slotsFor(a.kind)`, which builds a
     * fresh array per body and then a fresh iterator to walk it. Five thousand
     * bodies is ten thousand objects a frame for a value that has exactly two
     * possible answers.
     */
    const scratch = this.portScratch;
    // A voice is minted. It is not matter and it never was — nothing eats it,
    // nothing excretes it, and the conservation books do not count it.
    for (let i = 0; i < n; i++) {
      const a = list[i];
      if (a.locked) continue;
      const ports = a.kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
      for (let pi = 0; pi < ports.length; pi++) {
        const slot = ports[pi];
        // See `deposit`: a voice carries whether or not the port is attached;
        // a free *principal* adds its own kind's channel and a free auxiliary
        // adds a little of both.
        const free = this.graph.isFreeAt(a.id, slot);
        if (slot !== 'p' && !free) continue;
        const w = portWorldInto(a, slot, this.w, this.h, scratch);
        const o = nDep * dStride;
        dep[o] = w.x;
        dep[o + 1] = w.y;
        dep[o + 2] = 0; // a density; `depositScale` is already in `amt`
        // Neither the ground nor its smell is a body's to lay. See `effEmit`.
        dep[o + 6] = 0;
        dep[o + 7] = 0;
        if (slot === 'p') {
          const eo = a.slot * 4;
          const own = ownChannel(a.kind);
          const mark = free && own >= 0 ? leak : 0;
          dep[o + 4] = amt * EMITS[eo] + (own === CH.conP ? mark : 0);
          dep[o + 5] = amt * EMITS[eo + 1] + (own === CH.dupP ? mark : 0);
        } else {
          dep[o + 4] = auxLeak;
          dep[o + 5] = auxLeak;
        }
        nDep++;
      }
    }

    Sim.phase('gpu:packDeposit');

    /*
     * The frame's queued energy adds, on the same scatter as the voices.
     *
     * `conserve` is what separates them. A voice is a density and the host has
     * already multiplied in `depositScale`; these are counts, handed over raw,
     * and the shader spreads each over whichever of its four cells the disk
     * will take rather than dropping the share that falls outside. That rim
     * behaviour is the whole difference between `Fields.deposit` and
     * `Fields.addAt`, and it is the one a conserved channel cannot do without.
     */
    const seed = this.energy.pendingSeed;
    this.energy.pendingSeed = null;
    const pend = this.energy.pendingData;
    for (let k = 0; k < nAdds; k++) {
      const o = nDep * dStride;
      const p = k * PENDING_STRIDE;
      dep[o] = pend[p];
      dep[o + 1] = pend[p + 1];
      dep[o + 2] = 1;
      // Every species, not just the ground: excretion moves all four out of a
      // tank and each has to survive the rim the way a quantity does. The
      // shader's `Deposit.w` has been a `vec4f` all along.
      dep[o + 4] = pend[p + 2];
      dep[o + 5] = pend[p + 3];
      dep[o + 6] = pend[p + 4];
      dep[o + 7] = pend[p + 5];
      nDep++;
    }
    this.energy.clearPending();

    Sim.phase('gpu:packAdds');

    const arc = params.sensorAngle;
    const sd = params.sensorDist;
    /*
     * Out of the store, as `packPose` and the steer pack are.
     *
     * The two sensor directions come from the body's heading by the angle-sum
     * identity rather than from four `Math.sin`/`Math.cos` a body, which was
     * three of this pass's four and a half milliseconds at fifty thousand.
     * The cosine and sine of the heading itself come from the store's memo,
     * which the latch pass has usually already filled this frame for the same
     * body at the same heading.
     *
     * This is **not** bit-identical, and it is the one change in this file
     * that is not. Measured over four sensor angles and two million headings,
     * the worst disagreement is 5.1e-16 in the unit vector — a couple of ulps
     * — which is 1.3e-14 px of probe position against a field cell about
     * forty px across. Physically nothing; the sampled cell is the same one.
     * But the sim is chaotic, so the printed determinism hashes move, and
     * that was a deliberate call rather than an oversight.
     */
    const ca = Math.cos(arc);
    const sa = Math.sin(arc);
    const PX = this.agentStore.x;
    const PY = this.agentStore.y;
    const PH = this.agentStore.heading;
    const CSH = this.agentStore.csHeading;
    const CSC = this.agentStore.csCos;
    const CSS = this.agentStore.csSin;
    const TASTE_OF = this.agentStore.tasteAll;
    const groundScale = this.groundScale;
    for (let i = 0; i < n; i++) {
      const sl = list[i].slot;
      const o = i * pStride;
      const h = PH[sl];
      const x = PX[sl];
      const y = PY[sl];
      syncHeadingCosSin(CSH, CSC, CSS, sl, h);
      const c = CSC[sl];
      const sn = CSS[sl];
      // cos(h -/+ arc) and sin(h -/+ arc), from cos h and sin h.
      const lc = c * ca + sn * sa;
      const ls = sn * ca - c * sa;
      const rc = c * ca - sn * sa;
      const rs = sn * ca + c * sa;
      pro[o] = x + lc * sd;
      pro[o + 1] = y + ls * sd;
      pro[o + 2] = x + rc * sd;
      pro[o + 3] = y + rs * sd;
      pro[o + 4] = x;
      pro[o + 5] = y;
      packTaste(pro, o + 8, TASTE_OF, sl, groundScale, this.tasteGain(sl));
    }

    Sim.phase('gpu:packProbe');

    /*
     * Six per block here, eight per block there: the shader's `HarvestBlock`
     * carries two words of padding so a block is a round 32 bytes, which is
     * the only reason this is a transcode rather than a `set`.
     */
    if (plan.nBlocks > 0) {
      const blk = fieldGpu.blockData;
      const bStride = fieldGpu.blockStride;
      const src = plan.blocks;
      for (let b = 0; b < plan.nBlocks; b++) {
        const so = b * 6;
        const bo = b * bStride;
        blk[bo] = src[so];
        blk[bo + 1] = src[so + 1];
        blk[bo + 2] = src[so + 2];
        blk[bo + 3] = src[so + 3];
        blk[bo + 4] = src[so + 4];
        blk[bo + 5] = src[so + 5];
      }
      // `HARVEST_STRIDE` floats an entry now, not one: the gut room, the
      // mouthful and the four affinities are per body and cannot be uniforms.
      const rooms = fieldGpu.roomData;
      const span = plan.nEntries * HARVEST_STRIDE;
      for (let e = 0; e < span; e++) rooms[e] = plan.rooms[e];
      this.harvestPending = true;
    }

    /*
     * The same five passes `Sim.step` runs on the CPU, in the same order, with
     * the same numbers. `decay` is handed the *rate* rather than the keep
     * factor, because the shader resolves per-channel rates itself — one place
     * that knows how a slider becomes four numbers, not two.
     */
    Sim.phase('gpu:packBlocks');

    const submitted = fieldGpu.submit(
      this.fields,
      nDep,
      n,
      params.diffuse,
      params.diffuse * 0.65,
      params.decay,
      {
        ch: CH.energy,
        r: params.energyRegrow * dt,
        cap: this.energy.cellCap,
        smell: params.groundSmell * dt,
      },
      {
        ch: CH.energy,
        blocks: plan.nBlocks,
        entries: plan.nEntries,
        uptakeCap: params.uptakeVmax * dt,
        uptakeKs: params.uptakeKs,
        hillN: hillOf(params),
      },
      seed === null ? null : { ch: CH.energy, value: seed },
    );
    if (!submitted) {
      this.dropFieldGpu();
      return;
    }
    Sim.phase('gpu:dispatch');

    /*
     * The genome's dispatch goes on the queue now, behind the field's, before
     * either readback is waited on. Its inputs are the store and the wire
     * graph, none of which the field's readback touches; its one dependency —
     * the probe's output buffer — is on the device, where submission order is
     * the ordering. Waiting on the field first and dispatching after was two
     * full round trips a frame where one will do.
     */
    const genomeSubmitted = this.genomeOnGpu && this.submitGenome(list, n, params);

    if (!(await fieldGpu.collect())) {
      this.dropFieldGpu();
      return;
    }

    // Before the sample unpack rather than after, so a frame that is being
    // watched shows the field the same passes just produced.
    if (this.wantFieldReadback) await fieldGpu.readInto(this.fields);

    const got = fieldGpu.sampleData;
    const sStride = fieldGpu.sampleStride;
    /*
     * Two consumers, one readback.
     *
     * The first three floats are the taste-collapsed sensor readings the
     * solver steers on. The next four are the raw channels under the body,
     * which is what `updateState`'s `Wx` sense columns want — it used to get
     * them from `Fields.sampleAll`, out of a CPU array the GPU never writes,
     * and that was the quietest of the reasons `openFieldGpu` refuses.
     *
     * Both are written straight into the store by slot rather than kept in
     * list order, because that is how everything downstream indexes and
     * because the list can move before they are read — see `steerAll`.
     *
     * A frame late, like the steering samples and for the same reason: this
     * runs after `endFrame`, so what lands here is read at the top of the next
     * frame. A body born in between finds its slot holding whatever the last
     * occupant left, which is why `createAgent` zeroes it — one frame of no
     * smell for a newborn, against a stall every frame for everyone.
     */
    const SENSE = this.agentStore.senseAll;
    const STEER = this.agentStore.steerAll;
    // Scaled on the way in, the same way `updateState` scales its own
    // sample, so `senseAll` means one thing on both paths. Channel 2 is the
    // ground and reads against a full cell; the others against a strong
    // local signal.
    const sScale = 1 / params.senseScale;
    const gScale = this.groundScale;
    for (let i = 0; i < n; i++) {
      const o = i * sStride;
      const slot = list[i].slot;
      const to = slot * 3;
      STEER[to] = got[o];
      STEER[to + 1] = got[o + 1];
      STEER[to + 2] = got[o + 2];
      const so = slot * 4;
      SENSE[so] = got[o + 4] * sScale;
      SENSE[so + 1] = got[o + 5] * sScale;
      SENSE[so + 2] = got[o + 6] * gScale;
      SENSE[so + 3] = got[o + 7] * sScale;
    }
    this.steerFromSamples = true;
    Sim.phase('gpu:unpack');

    if (genomeSubmitted) {
      if (await genomeGpu.collect()) {
        this.scatterLearn();
        this.genomePending = true;
      } else {
        // Back to the CPU pass for good; nothing is owed, since `updateState`
        // recomputes from the store either way.
        this.genomeOnGpu = false;
        this.genomePending = false;
      }
      Sim.phase('gpu:genome');
    }
  }

  /**
   * The device went away mid-session. Fall back for good rather than leaving
   * the field frozen on whatever the GPU last held. Nothing grazed, so
   * nothing is owed: drop the pending credit or the next frame pays out of a
   * buffer the GPU never wrote. The genome goes with it, since its sense
   * inputs were the probe's.
   *
   * **And the ground's queue comes back with it.** `openFieldGpu` puts
   * `EnergyGrid` into deferring mode, where an `addAt` is queued for the
   * shader's `scatter` instead of written, and nothing turned that off again:
   * after a fallback every add — a corpse, a rewrite's leftovers, rent, a
   * refund, a food drop — went into a queue that only `gpuFieldStep` empties,
   * and `gpuFieldStep` no longer runs. The dish drained to zero and stayed
   * there, silently, with no error anywhere. Seen on the page by resetting a
   * ten-thousand-body pond down to two hundred: the field falls back, and the
   * ground never comes back.
   *
   * The frame's queued records go with the flag, which is `deferAdds`'s own
   * contract. They were addressed to a device that is gone, and the comment
   * above already accepts losing one frame of credit for the same reason.
   */
  private dropFieldGpu(): void {
    this.harvestPending = false;
    this.genomePending = false;
    this.fieldOnGpu = false;
    this.genomeOnGpu = false;
    this.steerFromSamples = false;
    this.energy.deferAdds(false);
    this.learnWanted.length = 0;
    this.learnAsked.length = 0;
    nativeSolver.useSamples(false);
  }

  /**
   * Hand the genome pass a frame, in the same list order everything else uses.
   *
   * Queued after the field's dispatch, and has to be: its sense inputs are
   * the raw channel readings `gather` writes, in a buffer it borrows rather
   * than a readback it waits for. The four numbers that would have been the
   * most expensive thing to ship never cross the bus.
   *
   * Like the field's samples, what comes back is spent at the top of the next
   * frame — `updateState` unpacks it where it used to compute it. So `h`
   * advances exactly one step per frame either way, and the inputs are a
   * frame older than the CPU pass would have read, which is the same bargain
   * steering and smell already take.
   *
   * True when the dispatch was queued; `gpuFieldStep` collects it.
   */
  private submitGenome(list: Agent[], n: number, params: Params): boolean {
    const samples = fieldGpu.sampleBuffer;
    if (!samples || n === 0) return false;
    const store = this.agentStore;
    const adj = this.wireAdjacency();
    const nNei = adj.off[n];
    let maxSlot = 0;
    for (let i = 0; i < n; i++) {
      if (list[i].slot > maxSlot) maxSlot = list[i].slot;
    }

    const chemFloats = (maxSlot + 1) * CHEM_LEN;
    /*
     * The genome table crosses only when it changed. A genome is written at
     * birth and nowhere else, and every path that writes one has to call
     * `refreshReadsField` or the sense gate goes stale — so that is the one
     * place the store learns a slot is dirty, and this is what spends the
     * mark. Uploading everything every frame was 2.7 MB at five thousand
     * bodies and would have been 27 MB at fifty; a frame with one birth now
     * pushes one genome. A moved buffer starts over from nothing.
     */
    const chemMoved = genomeGpu.reserve(n, nNei, chemFloats, maxSlot + 1);
    this.syncLearn(store, maxSlot + 1);
    if (chemMoved || this.genomeChemVersion !== store.chemVersion) {
      const lo = store.chemDirtyLo * CHEM_LEN;
      const hi = Math.min(store.chemDirtyHi * CHEM_LEN, chemFloats);
      if (chemMoved || hi <= lo) genomeGpu.uploadChem(store.chemAll, chemFloats);
      else genomeGpu.uploadChemRange(store.chemAll, lo, hi);
      this.genomeChemVersion = store.chemVersion;
      store.clearChemDirty();
    }

    const H = store.hAll;
    const hData = genomeGpu.hData;
    const inputData = genomeGpu.inputData;
    const EXTRA = store.extra;
    const CAP = store.energyCap;
    const REQUEST = store.request;
    const BOUND_OF = store.bound;
    const READS = store.readsField;
    for (let i = 0; i < n; i++) {
      const slot = list[i].slot;
      const ho = slot * STATE_DIMS;
      const po = i * STATE_DIMS;
      hData[po] = H[ho];
      hData[po + 1] = H[ho + 1];
      hData[po + 2] = H[ho + 2];
      hData[po + 3] = H[ho + 3];
      const o = i * 8;
      const cap = CAP[slot];
      const full = cap > 0 ? EXTRA[slot] / cap : 0;
      inputData[o] = full <= 0 ? 0 : full >= 1 ? 1 : full;
      inputData[o + 1] = BOUND_OF[slot];
      const r = REQUEST[slot];
      inputData[o + 2] = r <= 0 ? 0 : r >= 1 ? 1 : r;
      inputData[o + 3] = READS[slot];
      inputData[o + 4] = slot;
    }
    const offData = genomeGpu.offData;
    for (let i = 0; i <= n; i++) offData[i] = adj.off[i];
    const neiData = genomeGpu.neiData;
    for (let e = 0; e < nNei; e++) neiData[e] = adj.nei[e];
    Sim.phase('gpu:packGenome');

    if (
      !genomeGpu.submit(samples, n, nNei, this.groundScale, CH.energy, params.senseScale, {
        rate: params.learnRate,
        critic: params.learnCritic,
        trace: params.learnTrace,
        discount: params.learnDiscount,
        reward: params.learnReward,
        dtInv: this.frameDt > 0 ? 1 / this.frameDt : 0,
        maxWeight: CHEM_TASTE_MAX,
        explore: params.learnExplore,
        frame: exploreKey(this.frames, params.learnHold),
      })
    ) {
      this.genomeOnGpu = false;
      this.genomePending = false;
      return false;
    }
    this.genomeSlots(list, n);
    return true;
  }

  /**
   * Hand the device the learning rows the host has changed, and ask for the
   * ones the host is about to need.
   *
   * The host writes this state in one place only — `AgentStore.clearSlot`,
   * zeroing a slot that has been recycled — and that write has to land or a
   * newborn inherits the last occupant's experience through a buffer nobody
   * cleared. The three CPU arrays are transcoded into the one row the shader
   * indexes; a frame with one birth moves one row.
   */
  private syncLearn(store: AgentStore, learnSlots: number): void {
    if (this.genomeLearnVersion !== store.learnVersion) {
      const up = genomeGpu.learnUpData;
      const stride = genomeGpu.learnStride;
      const P = store.plasticAll;
      const T = store.traceAll;
      const C = store.criticAll;
      const V = store.prevValue;
      /** One slot's row into `learnUpData` at row `row`. */
      const fill = (slot: number, row: number): void => {
        const o = row * stride;
        const ps = slot * PLASTIC_LEN;
        for (let k = 0; k < PLASTIC_LEN; k++) {
          up[o + LEARN_PLASTIC + k] = P[ps + k];
          up[o + LEARN_TRACE + k] = T[ps + k];
        }
        const cs = slot * CRITIC_LEN;
        for (let k = 0; k < CRITIC_LEN; k++) up[o + LEARN_CRITIC + k] = C[cs + k];
        up[o + LEARN_PREV_V] = V[slot];
      };
      if (store.learnDirtyAll) {
        // The list overran, so all that is left is the span.
        const lo = store.learnDirtyLo;
        const hi = Math.min(store.learnDirtyHi, learnSlots);
        for (let s = lo; s < hi; s++) fill(s, s - lo);
        if (hi > lo) genomeGpu.pushLearn(lo, hi - lo);
      } else {
        /*
         * A row per dirty slot. The host only writes these to zero a recycled
         * slot, so they are scattered wherever the free list handed out: on a
         * grown pond, 62 slots a frame spanning 24,000. Carrying the span cost
         * 11.9 ms; carrying the slots costs one small write each.
         */
        const slots = store.learnDirtySlots;
        const count = store.learnDirtyCount;
        for (let i = 0; i < count; i++) {
          const slot = slots[i];
          if (slot >= learnSlots) continue;
          fill(slot, 0);
          genomeGpu.pushLearn(slot, 1);
        }
      }
      this.genomeLearnVersion = store.learnVersion;
      store.clearLearnDirty();
    }
    const want = this.learnWanted;
    this.learnAsked.length = 0;
    if (want.length === 0) return;
    const took = genomeGpu.readLearn(want, want.length);
    for (let i = 0; i < took; i++) this.learnAsked.push(want[i]);
    want.splice(0, took);
  }

  /**
   * Bring every body's learning back from the device.
   *
   * On the GPU path the learning row is device-resident and the host's copy
   * is fresh only for rewrite parents — `syncLearn` reads back a handful a
   * frame because a handful a frame is all the simulation needs. Anything
   * that *measures* what the pond has learned has to ask for the rest.
   *
   * This is that ask. The headless harvest calls it before storing a net,
   * because `plasticAll` and `criticAll` are two of the things a stored net
   * is for, and writing the zeros the host happened to be holding would make
   * the database quietly wrong rather than loudly empty.
   *
   * A no-op returning true off the GPU path, where the host's copy is the
   * only copy. Deliberately does not mark the rows dirty: this is the device
   * telling the host what it worked out, and echoing it back would push it
   * out again next frame.
   */
  async syncLearningToHost(): Promise<boolean> {
    if (!this.genomeOnGpu) return true;
    const store = this.agentStore;
    let maxSlot = -1;
    for (const a of this.agents.values()) if (a.slot > maxSlot) maxSlot = a.slot;
    if (maxSlot < 0) return true;
    const rows = await genomeGpu.drainLearn(maxSlot + 1);
    if (!rows) return false;
    const stride = genomeGpu.learnStride;
    const P = store.plasticAll;
    const T = store.traceAll;
    const C = store.criticAll;
    const V = store.prevValue;
    for (const a of this.agents.values()) {
      const s = a.slot;
      const o = s * stride;
      if (o + stride > rows.length) continue;
      const ps = s * PLASTIC_LEN;
      let on = 0;
      for (let k = 0; k < PLASTIC_LEN; k++) {
        const w = rows[o + LEARN_PLASTIC + k];
        P[ps + k] = w;
        T[ps + k] = rows[o + LEARN_TRACE + k];
        if (w !== 0) on = 1;
      }
      const cs = s * CRITIC_LEN;
      for (let k = 0; k < CRITIC_LEN; k++) C[cs + k] = rows[o + LEARN_CRITIC + k];
      V[s] = rows[o + LEARN_PREV_V];
      if (on) store.plasticOn[s] = 1;
    }
    return true;
  }

  /**
   * Put the rows that came back where the CPU keeps them.
   *
   * Deliberately does not mark them dirty: this is the device telling the
   * host what it worked out, and echoing it straight back would push it out
   * again every frame a rewrite is pending.
   */
  private scatterLearn(): void {
    const rows = genomeGpu.learnCount;
    if (rows === 0) return;
    const src = genomeGpu.learnOutData;
    const stride = genomeGpu.learnStride;
    const store = this.agentStore;
    const P = store.plasticAll;
    const T = store.traceAll;
    const C = store.criticAll;
    const V = store.prevValue;
    for (let i = 0; i < rows && i < this.learnAsked.length; i++) {
      const s = this.learnAsked[i];
      const o = i * stride;
      const ps = s * PLASTIC_LEN;
      let on = 0;
      for (let k = 0; k < PLASTIC_LEN; k++) {
        const w = src[o + LEARN_PLASTIC + k];
        P[ps + k] = w;
        T[ps + k] = src[o + LEARN_TRACE + k];
        if (w !== 0) on = 1;
      }
      const cs = s * CRITIC_LEN;
      for (let k = 0; k < CRITIC_LEN; k++) C[cs + k] = src[o + LEARN_CRITIC + k];
      V[s] = src[o + LEARN_PREV_V];
      if (on) store.plasticOn[s] = 1;
    }
    this.learnAsked.length = 0;
  }

  /**
   * Remember which body each list position was, for the unpack a frame later.
   *
   * Slot *and* id, because `AgentStore.release` does not clear a freed slot's
   * id — it drops the mapping and puts the slot on the free list. So a stale
   * slot still reads as whatever died there, and only the id says whether the
   * body that earned these results is still the one standing in that slot.
   */
  private genomeSlots(list: Agent[], n: number): void {
    if (this.genomeSlotBuf.length < n * 2) this.genomeSlotBuf = new Int32Array(n * 4);
    const b = this.genomeSlotBuf;
    for (let i = 0; i < n; i++) {
      b[i * 2] = list[i].slot;
      b[i * 2 + 1] = list[i].id;
    }
    this.genomeCount = n;
  }

  /**
   * Take the frame on the GPU.
   *
   * The first thing this has to do is close the force block, and not doing so
   * is what made the GPU path unusable. `beginFrame` leaves the block open on
   * purpose so the wasm solve can inherit the packed bodies — steer, port
   * torques, declutter, flocking and gravity all wrote their velocities into
   * that buffer and *not* into the agents. `solveFarNative` reads the buffer,
   * so it sees them. This packs from the agents, so it did not: it integrated
   * a frame using last frame's velocities, and then `endFrame`'s `syncForces`
   * unpacked the force-pass velocities straight over the top of the result.
   *
   * Positions from the GPU, velocities from before the solve — which is to say
   * the velocity feedback XPBD uses to hold a constraint was thrown away every
   * frame. That is an energy source. Wires grow without bound and the view
   * shakes, and it only happens at the zoom where the LOD lets the GPU take
   * the frame at all.
   *
   * `syncForces` hands the velocities to the agents and closes the block, so
   * the pack below sees this frame's forces and nothing overwrites the result.
   */
  private async solveFarGpu(params: Params, dt: number): Promise<boolean> {
    this.syncForces();
    const packed = this.packFar(params);
    if (!packed) {
      this.lastFarPath = 'none';
      return true;
    }
    // False means the device was lost or a pass threw and `farGpu` quietly ran
    // its own twin, which is a different solver at a very different speed. Say
    // which one actually took the frame rather than which one was asked.
    const onGpu = await farGpu.step(
      packed.data,
      packed.list.length,
      packed.wires,
      packed.nWires,
      dt,
      undefined,
      this.worldX,
      this.worldY,
      this.worldR,
    );
    this.unpackFar(packed.list, packed.data);
    this.lastFarPath = onGpu ? 'gpu' : 'js';
    return true;
  }

  /**
   * One WASM call for all eight substeps: integrate, XPBD wires, grab, SAT on
   * any detailed pair, disc on FAR-FAR, finalize. Hertzian audio and wire
   * clearance stay in JS, once per frame after unpack.
   */
  private solveNearNative(params: Params, dt: number): boolean {
    if (!nativeSolver.ready || !nativeSolver.bodies || !nativeSolver.wiresNear || !nativeSolver.nodes) {
      return false;
    }
    const list = this.forceList();
    const n = list.length;
    if (n === 0) return true;

    /*
     * The cached list itself, unfiltered: every wire's endpoints are in the
     * roster, so nothing here can drop one, and the endpoint indices come
     * with it.
     */
    const wireList = this.wireListResolved();
    const wai = this.wireAI;
    const wbi = this.wireBI;
    let nNodes = 0;
    for (let k = 0; k < wireList.length; k++) {
      const w = wireList[k];
      if (this.wireSimulatesRope(w) && w.nodes.length > 0) nNodes += w.nodes.length;
    }
    const nWires = wireList.length;
    if (!nativeSolver.canNear(n, nWires, nNodes)) return false;

    this.packNearMeta(list, wireList, wai, wbi, params);
    this.packNearState(list, wireList);

    const ropeKeep = Math.exp(-Math.max(0, params.springDamp) * (dt / Sim.SUBSTEPS));
    const heldId = this.grabbed?.id ?? -1;
    const heldBody = heldId < 0 ? undefined : this.agents.get(heldId);
    const heldIndex = heldBody ? this.slotIndex[heldBody.slot] : -1;
    const gx = this.grabbed?.x ?? 0;
    const gy = this.grabbed?.y ?? 0;
    if (!nativeSolver.stepNear(
      n, nWires, dt, Sim.SUBSTEPS, ropeKeep, heldIndex, Sim.GRAB_MAX_SPEED, gx, gy,
      this.worldX, this.worldY, this.worldR,
    )) {
      return false;
    }
    this.unpackNearState(list, wireList);
    this.releaseForceBlock();
    this.emitNativeHits(list);
    if (params.wireClear > 0) this.clearWires(params);
    for (const a of list) {
      a.stun = Math.max(0, a.stun - dt);
      wrapPos(a, this.w, this.h);
    }
    return true;
  }

  /** Hertzian from WASM SAT, using kinematics snapshotted at each hit. */
  private emitNativeHits(list: Agent[]): void {
    const count = nativeSolver.hitCount();
    const hits = nativeSolver.hits;
    if (!hits || count <= 0) return;
    const n = list.length;
    for (let k = 0; k < count; k++) {
      const o = k * HIT_STRIDE;
      const i = hits[o + HIT.a] | 0;
      const j = hits[o + HIT.b] | 0;
      if (i < 0 || j < 0 || i >= n || j >= n) continue;
      const A = list[i];
      const B = list[j];
      if (!A || !B) continue;
      const hit: Hit = {
        nx: hits[o + HIT.nx],
        ny: hits[o + HIT.ny],
        overlap: hits[o + HIT.overlap],
        px: hits[o + HIT.px],
        py: hits[o + HIT.py],
      };
      this.emitCollision(A, B, hit, {
        effMass: hits[o + HIT.effMass],
        vN: hits[o + HIT.vN],
        vT: hits[o + HIT.vT],
      });
    }
  }

  private packNearMeta(
    list: Agent[],
    wireList: Wire[],
    wai: Int32Array,
    wbi: Int32Array,
    params: Params,
  ): void {
    const bodies = nativeSolver.bodies!;
    const invI = nativeSolver.invInertia!;
    const sc = nativeSolver.scale!;
    const kinds = nativeSolver.kind!;
    const det = nativeSolver.detailed!;
    const frozen = this.rewriteFrozen;
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      const locked = poseHeld(a);
      bodies[o + FAR.invMass] = locked ? 0 : 1 / Math.max(0.08, a.mass);
      bodies[o + FAR.radius] = boundRadius(a);
      bodies[o + FAR.locked] = locked ? 1 : 0;
      invI[i] = locked ? 0 : 1 / Math.max(1e-4, momentOfInertia(a));
      sc[i] = a.scale;
      kinds[i] = this.kindCode(a.kind);
      det[i] = this.agentDetailed(a.id) ? 1 : 0;
    }
    const wires = nativeSolver.wiresNear!;
    const nodes = nativeSolver.nodes!;
    let nodeAt = 0;
    for (let k = 0; k < wireList.length; k++) {
      const w = wireList[k];
      const o = k * WIRE_NEAR_STRIDE;
      const ai = wai[k];
      const bi = wbi[k];
      const A = list[ai];
      const B = list[bi];
      // A rewrite's own wire is the one it may pull on; its leftovers are not.
      const frozenEnds = (frozen.has(A.id) || frozen.has(B.id)) && !this.rewriteWires.has(w.id);
      const skip = (poseHeld(A) && poseHeld(B)) || frozenEnds;
      const full = this.wireSimulatesRope(w) && w.nodes.length > 0;
      const stiff = this.graph.stiffnessOf(w, this.time, params);
      let flags = 0;
      if (full) flags |= WF_FULL;
      if (skip) flags |= WF_SKIP;
      if (this.rewriteWires.has(w.id)) flags |= WF_TETHER;
      if (frozenEnds) flags |= WF_HOLD;
      if (full && w.ropePath === 'full' && w.shape.length === w.nodes.length) flags |= WF_SHAPE;
      wires[o + WN.a] = ai;
      wires[o + WN.b] = bi;
      wires[o + WN.rest] = w.rest;
      wires[o + WN.rope] = w.ropeLen;
      wires[o + WN.scale] = stiff.scale;
      wires[o + WN.slack] = stiff.slack;
      wires[o + WN.aSlot] = this.slotCode(w.a.slot);
      wires[o + WN.bSlot] = this.slotCode(w.b.slot);
      wires[o + WN.node0] = full ? nodeAt : 0;
      wires[o + WN.nNodes] = full ? w.nodes.length : 0;
      wires[o + WN.flags] = flags;
      if (full) {
        const hasShape = (flags & WF_SHAPE) !== 0;
        for (let i = 0; i < w.nodes.length; i++) {
          const no = (nodeAt + i) * NODE_STRIDE;
          if (hasShape) {
            nodes[no + ND.shapeX] = w.shape[i].x;
            nodes[no + ND.shapeY] = w.shape[i].y;
          }
        }
        nodeAt += w.nodes.length;
      }
    }
  }

  private packNearState(list: Agent[], wireList: Wire[]): void {
    const bodies = nativeSolver.bodies!;
    // The force block leaves the pose here already; only the previous-frame
    // fields, which it never touches, still have to come across.
    const inherited = this.forceBlock;
    if (inherited && Sim.auditForceBlock) this.auditPack(list, bodies, 'NEAR');
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      if (!inherited) {
        bodies[o + FAR.x] = a.x;
        bodies[o + FAR.y] = a.y;
        bodies[o + FAR.vx] = a.vx;
        bodies[o + FAR.vy] = a.vy;
        bodies[o + FAR.heading] = a.heading;
        bodies[o + FAR.omega] = a.omega;
      }
      bodies[o + FAR.prevX] = a.prevX;
      bodies[o + FAR.prevY] = a.prevY;
      bodies[o + FAR.prevHeading] = a.prevHeading;
    }
    const nodes = nativeSolver.nodes!;
    let nodeAt = 0;
    for (const w of wireList) {
      if (!this.wireSimulatesRope(w) || w.nodes.length === 0) continue;
      for (let i = 0; i < w.nodes.length; i++) {
        const nd = w.nodes[i];
        const o = (nodeAt + i) * NODE_STRIDE;
        nodes[o + ND.x] = nd.x;
        nodes[o + ND.y] = nd.y;
        nodes[o + ND.vx] = nd.vx;
        nodes[o + ND.vy] = nd.vy;
        nodes[o + ND.prevX] = nd.prevX;
        nodes[o + ND.prevY] = nd.prevY;
      }
      nodeAt += w.nodes.length;
    }
  }

  private unpackNearState(list: Agent[], wireList: Wire[]): void {
    const bodies = nativeSolver.bodies!;
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      a.x = bodies[o + FAR.x];
      a.y = bodies[o + FAR.y];
      a.vx = bodies[o + FAR.vx];
      a.vy = bodies[o + FAR.vy];
      a.heading = bodies[o + FAR.heading];
      a.omega = bodies[o + FAR.omega];
      a.prevX = bodies[o + FAR.prevX];
      a.prevY = bodies[o + FAR.prevY];
      a.prevHeading = bodies[o + FAR.prevHeading];
    }
    const nodes = nativeSolver.nodes!;
    let nodeAt = 0;
    for (const w of wireList) {
      if (!this.wireSimulatesRope(w) || w.nodes.length === 0) continue;
      for (let i = 0; i < w.nodes.length; i++) {
        const nd = w.nodes[i];
        const o = (nodeAt + i) * NODE_STRIDE;
        nd.x = nodes[o + ND.x];
        nd.y = nodes[o + ND.y];
        nd.vx = nodes[o + ND.vx];
        nd.vy = nodes[o + ND.vy];
        nd.prevX = nodes[o + ND.prevX];
        nd.prevY = nodes[o + ND.prevY];
      }
      nodeAt += w.nodes.length;
    }
  }

  /** Approach speed that retriggers a strike while two bodies already touch. */
  private static readonly CONTACT_RETRIGGER = 3.2;
  /** Fraction of contact momentum radiated as sound instead of bounce. */
  private static readonly RADIATION = 0.035;

  private emitCollision(
    A: Agent,
    B: Agent,
    hit: Hit,
    mechanics?: { effMass: number; vN: number; vT: number },
  ): void {
    const key = contactKey(A.id, B.id);
    const m = mechanics ?? contactMechanics(A, B, hit);
    this.noteContact(A, B, hit.overlap, m.vT);

    if (this.contactAudioNow.has(key)) return;
    this.contactAudioNow.add(key);

    if (m.vN < 0.8 && hit.overlap < 1.2) return;
    if (this.contactAudioPrev.has(key) && m.vN < Sim.CONTACT_RETRIGGER) return;

    const ev: CollisionEvent = {
      type: 'collision',
      agentA: A.id,
      agentB: B.id,
      kindA: A.kind,
      kindB: B.kind,
      impact: Math.max(m.vN, hit.overlap * 4),
      overlap: hit.overlap,
      effMass: m.effMass,
      vN: m.vN,
      vT: m.vT,
      nx: hit.nx,
      ny: hit.ny,
      headingA: A.heading,
      headingB: B.heading,
      spin: Math.abs(A.omega) + Math.abs(B.omega),
    };
    audio.push(ev, this.graph, this.agents);
    this.noteRadiation(A, B, hit, m.effMass, m.vN);
  }

  /**
   * Keep the deepest overlap of the frame, with vT signed in canonical id order
   * so the worklet applies +F on A and −F on B consistently across substeps.
   */
  private noteContact(A: Agent, B: Agent, overlap: number, vT: number): void {
    const lo = A.id < B.id ? A.id : B.id;
    const hi = A.id < B.id ? B.id : A.id;
    const signed = A.id < B.id ? vT : -vT;
    const key = lo * CONTACT_KEY_WIDTH + hi;
    const prev = this.contacts.get(key);
    if (prev && prev.overlap >= overlap) return;
    this.contacts.set(key, { agentA: lo, agentB: hi, overlap, vT: signed });
  }

  /**
   * Energy that leaves as sound has to leave the bodies too. Without this the
   * audio is a passive read-out; with it, a collision that rings loudly is
   * measurably less bouncy than one that does not.
   */
  private noteRadiation(A: Agent, B: Agent, hit: Hit, effMass: number, vN: number): void {
    const j = effMass * Math.abs(vN) * Sim.RADIATION;
    if (j <= 0) return;
    const add = (agent: Agent, sx: number, sy: number) => {
      if (poseHeld(agent)) return;
      const cur = this.radiated.get(agent.id) ?? { x: 0, y: 0 };
      cur.x += sx;
      cur.y += sy;
      this.radiated.set(agent.id, cur);
    };
    add(A, hit.nx * j, hit.ny * j);
    add(B, -hit.nx * j, -hit.ny * j);
  }

  /**
   * Rebuild the body broad-phase from current positions. Cheap enough to redo
   * every substep, which keeps it exact rather than relying on a motion margin.
   */
  private rebuildBodyGrid(cellSize: number): Agent[] {
    const list = this.forceList();
    const n = list.length;
    if (this.gx.length < n) {
      this.gx = new Array(n * 2);
      this.gy = new Array(n * 2);
    }
    for (let i = 0; i < n; i++) {
      this.gx[i] = list[i].x;
      this.gy[i] = list[i].y;
    }
    this.bodyGrid.build(this.gx, this.gy, n, cellSize);
    return list;
  }

  private solveContacts(h: number): void {
    let reach = 0;
    for (const a of this.agents.values()) reach = Math.max(reach, boundRadius(a));
    const list = this.rebuildBodyGrid(reach * 2 + SLOP + 4);
    this.bodyGrid.forEachPair((i, j) => {
      const A = list[i];
      const B = list[j];
      if (poseHeld(A) && poseHeld(B)) return;
      const detailed = this.agentDetailed(A.id) || this.agentDetailed(B.id);
      // Wired pairs collide on both branches. The cheap one used to skip
      // them, on the grounds that the wire already held the gap — but the
      // detailed branch never skipped them, so a latched pair sat differently
      // depending on where the camera was.
      const hit = detailed
        ? queryHit(A, B, this.w, this.h)
        : queryDiscHit(A, B, this.w, this.h);
      if (!hit) return;
      if (detailed) this.emitCollision(A, B, hit);
      solveContact(A, B, hit, SLOP, h);
    });
  }

  /**
   * Applied once, after the substeps, so it never fights the position solver.
   * Capped at a fraction of the body's own speed: sound can slow a collision,
   * never reverse it.
   */
  private applyRadiationLoss(): void {
    for (const [id, imp] of this.radiated) {
      const agent = this.agents.get(id);
      if (!agent || poseHeld(agent)) continue;
      const im = 1 / Math.max(0.08, agent.mass);
      const dvx = imp.x * im;
      const dvy = imp.y * im;
      const speed = Math.hypot(agent.vx, agent.vy);
      const mag = Math.hypot(dvx, dvy);
      const scale = mag > speed * 0.25 ? (speed * 0.25) / Math.max(1e-9, mag) : 1;
      agent.vx += dvx * scale;
      agent.vy += dvy * scale;
    }
  }

  /** The three species' bounded waves for one body, reused. See `speciesWaves`. */
  private readonly waveScratch = new Float64Array(3);

  /** Rate steps the drag table divides itself into. See `dampVelocities`. */
  private static readonly GRIP_STEPS = 256;

  /** Per-frame `exp(-rate * dt)` by total rate, rebuilt only when it varies. */
  private readonly gripKeep = new Float64Array(Sim.GRIP_STEPS + 1);

  /** The two actuator mixtures, turned by this body's depth. See `rotateMix`. */
  private readonly swRot = new Float64Array(3);
  private readonly gwRot = new Float64Array(3);
  /** `[stroke, grip]` out of `actuatorPair`. */
  private readonly actPair = new Float64Array(2);

  /**
   * One step of the body's reactor, and what this frame's stroke comes to.
   *
   * Three chemicals in a well-stirred vat, and a fourth that is the gut.
   * `docs/scratch.txt` §3 gives four; A, its primary fuel, is what the body
   * has swallowed off the grid and not yet used, so it is `store.gut` and the
   * reaction that consumes it is digestion:
   *
   *     r1   A -> B          `Sim.runDigestion`, at the recipe's own rate
   *     r2   -> C            metabolicCat    * B * (base + C) / (1 + sigma*C)
   *     r3   C -> D          metabolicReset  * C
   *     r4   B + D ->        metabolicQuench * B * D
   *     work B ->            metabolicWork * gaitSwell * |wave|
   *
   * with a uniform outflow on all three. **The reactor is fed by eating, not
   * by buying.** It used to purchase fuel out of the tank at a price, so food
   * ran grid -> gut -> tank -> reactor and the tank sat in the middle of a
   * round trip; now `runDigestion` splits what it converts between the tank
   * and B, and `intake` is the share a body routes to its reactor. Two things
   * follow. A body with a full tank and an empty gut has no clock, so the
   * gait depends on eating rather than on having. And the cost of running a
   * metabolism needs no price of its own, because it is food the body did not
   * bank — `metabolicSupply` and `metabolicCost` are gone.
   *
   * The oscillation is the loop `B -> C -> D -| B`, which is a three-stage
   * negative feedback carrying a positive self-loop on C. Whether it
   * oscillates at all is a narrow question and the doc's own constants answer
   * it wrongly; `src/pond/spectrum.ts` carries the stability analysis and the two
   * conditions that matter.
   *
   * **The fuel window has both edges, and that is the gate.** Starved, the
   * reactor sits empty and still; fed, it runs a limit cycle whose period
   * shortens as the fuel rises; over-fed, it saturates and goes still again.
   * Nothing had to be added to get "a starving body does not undulate".
   *
   * The wave is `2C/(metabolicWave + C) - 1`: bounded, signed, and no
   * arbitrary normalisation, so a body empty of catalyst reads -1 and a
   * saturated one approaches +1.
   */
  private advanceGait(params: Params, dt: number): void {
    const store = this.agentStore;
    const R = store.react;
    const CHEM = store.chemAll;
    const WV = this.waveScratch;
    const WAVE = store.gaitWave;
    const GRIPW = store.gaitGrip;
    const GA = store.gaitAnchor;
    const ANCHOR = store.anchor;
    const rate = params.metabolicRate;
    if (!(rate > 0)) {
      for (const agent of this.agents.values()) {
        ANCHOR[agent.slot] = 0;
        WAVE[agent.slot] = 0;
        GRIPW[agent.slot] = 0;
      }
      return;
    }

    const k2 = params.metabolicCat;
    const k3 = params.metabolicReset;
    const k4 = params.metabolicQuench;
    const sig = params.metabolicSigma;
    const dec = params.metabolicDecay;
    const base = params.metabolicBase;
    const waveK = params.metabolicWave > 0 ? params.metabolicWave : 1e-6;
    const workRate = params.metabolicWork * params.gaitSwell;
    const STARVE = store.starve;
    const DEPTH = store.depth;
    const profile = params.gaitProfile;
    const SWROT = this.swRot;
    const GWROT = this.gwRot;
    const PAIR = this.actPair;
    const h = rate * dt;
    const steps = Math.max(1, Math.ceil(h / REACT_H));
    const hs = h / steps;

    for (const agent of this.agents.values()) {
      const s = agent.slot;
      const o = s * REACT_SPECIES;
      let B = R[o + REACT_B];
      let C = R[o + REACT_C];
      let D = R[o + REACT_D];
      for (let k = 0; k < steps; k++) {
        const r2 = (k2 * B * (base + C)) / (1 + sig * C);
        const r3 = k3 * C;
        const r4 = k4 * B * D;
        const w = (2 * C) / (waveK + C) - 1;
        const work = workRate * (w < 0 ? -w : w);
        B = B + (-r4 - dec * B - work) * hs;
        C = C + (r2 - r3 - dec * C) * hs;
        D = D + (r3 - dec * D) * hs;
        if (B < 0) B = 0;
        else if (B > REACT_CAP) B = REACT_CAP;
        if (C < 0) C = 0;
        else if (C > REACT_CAP) C = REACT_CAP;
        if (D < 0) D = 0;
        else if (D > REACT_CAP) D = REACT_CAP;
      }
      R[o + REACT_B] = B;
      R[o + REACT_C] = C;
      R[o + REACT_D] = D;
      /*
       * The doc's §7.2 clock: how long this body has been at absolute
       * depletion, **less how long it has since been fed.**
       *
       * It used to reset outright on any primer at all, which made it a
       * *continuous* window: a body that hit empty, caught one crumb and hit
       * empty again started again from zero, so chronic scarcity never
       * accumulated and only an unbroken famine could kill. Measured on a
       * crowded net, bodies reached 24.9 s of a 25 s window and reset. The
       * pond had a rule for starving and no rule for going short.
       *
       * Now the pressure unwinds instead of vanishing, at `STARVE_RECOVER`
       * per second of being fed against one per second of being empty. At 1
       * that reads as: **a second empty costs a second fed**, so a body that
       * spends more than half its time with nothing dies of it however the
       * gaps are arranged, and one that is fed more than half the time is
       * safe however ragged its supply. Nothing decays here — the clock winds
       * back only against the thing that caused it, which is the difference
       * between a credit window and a forgetting.
       */
      if (B > STARVE_EPS) {
        const t = STARVE[s] - dt * STARVE_RECOVER;
        STARVE[s] = t > 0 ? t : 0;
      } else {
        STARVE[s] += dt;
      }
      speciesWaves(B, C, D, k2, k3, dec, waveK, WV);
      const g = s * CHEM_LEN;
      /*
       * When in its own cycle this body acts, from how far it sits behind its
       * net's nose. The wavelength used to be global — `metabolicDiffuse` sets
       * a lag per wire and nothing about a net's own shape or about where the
       * food is reached it — so a net could undulate and could not aim. This
       * is the whole of what the claim relay was built for.
       *
       * A *delay* with depth, so the stroke starts at the nose and runs back:
       * a fixed phase difference per unit of depth is a travelling wave, and
       * along a body that is peristalsis.
       */
      actuatorPair(CHEM, g, WV, profile * DEPTH[s], SWROT, GWROT, PAIR);
      WAVE[s] = PAIR[0];
      ANCHOR[s] = GA[s] * PAIR[0];
      GRIPW[s] = PAIR[1];
    }

    /*
     * The coupling: what a body broadcasts down its wire, and which species.
     *
     * The doc's §4.1 transmission presets, as genes rather than a rule keyed
     * on kind. A Con broadcasts the catalyst C and drives excitation down the
     * mesh; a Dup broadcasts the inhibitor D and resets the wave front; an
     * **Era broadcasts A and B — the food it ate and the primer it made** —
     * which is §6.1, and it is what makes a leaf its net's feeder rather than
     * one more oscillator. A body broadcasts out of its principal port only,
     * so it has one mouth and up to two ears, and an Era has nothing but a
     * principal.
     *
     * Rectified rather than a Heaviside step, which is what the doc's own
     * python does and is the better reading: a step rings under explicit
     * Euler and gives a gate nothing to move along. Mass action on the
     * concentration, so the impulse is in the chemistry — a body sends most
     * at its catalyst's peak and nothing at its trough, with no clock and no
     * threshold needed to make it a pulse.
     *
     * Bipartite on purpose. A wire pass computes each wire's fluxes once and
     * writes them on the wire; a body pass reads the three wires on its own
     * ports. So the two ends of a transfer read one number and cannot
     * disagree about it, and nothing is written to a body until every wire
     * has spoken, so the order of the wire map cannot matter.
     */
    const spread = params.metabolicDiffuse;
    if (spread > 0) {
      const GUT = store.gut;
      const PW = store.portWire;
      const agents = this.agents;
      const wires = this.graph.wires;
      const k = spread * dt;
      const CAP = store.energyCap;
      const gutSize = params.gutSize;
      for (const wire of wires.values()) {
        const A = agents.get(wire.a.id);
        const B = agents.get(wire.b.id);
        if (!A || !B || A === B) {
          wire.fluxGut = 0;
          wire.fluxB = 0;
          wire.fluxC = 0;
          wire.fluxD = 0;
          continue;
        }
        const sa = A.slot;
        const sb = B.slot;
        // A principal end is a mouth; an auxiliary one is only an ear.
        const ga = wire.a.slot === 'p' ? sa * CHEM_LEN : -1;
        const gb = wire.b.slot === 'p' ? sb * CHEM_LEN : -1;
        // The doc's A: the ground, which is the one thing a gut holds.
        wire.fluxGut = crossing(GUT, CHEM, sa, sb, ga, gb, 1, 0, TX_A, CAP[sb] * gutSize, k);
        /*
         * **And it kicks, the way every other transfer of matter does.**
         *
         * `recoil` was reachable from exactly one place — the demand
         * gradient's `flowChargesFast` — so `extra` moving between neighbours
         * shoved them apart and an Era pushing food down its principal moved
         * the same substance along the same wire and shoved nothing. One road
         * moved matter and kicked, the other moved matter and did not.
         *
         * It is also the impulse the gait can be *timed by*. A cyclic grip
         * only pays if it is phase-locked to whatever delivers the impulse,
         * and the demand gradient is not: measured on the worm bench, the
         * reactor ran a real travelling wave (6.6 frames a segment, 0.67
         * agreeing) and `gripSwing` still cost, because the kicks arrive on
         * the economy's clock and the grip swings on the chemistry's. This
         * transfer is the one that swings with the reactor.
         */
        if (wire.fluxGut !== 0) {
          const send = wire.fluxGut > 0 ? A : B;
          const recv = wire.fluxGut > 0 ? B : A;
          this.recoil(send, recv, Math.abs(wire.fluxGut));
        }
        wire.fluxB = crossing(R, CHEM, sa, sb, ga, gb, REACT_SPECIES, REACT_B, TX_B, REACT_CAP, k);
        wire.fluxC = crossing(R, CHEM, sa, sb, ga, gb, REACT_SPECIES, REACT_C, TX_C, REACT_CAP, k);
        wire.fluxD = crossing(R, CHEM, sa, sb, ga, gb, REACT_SPECIES, REACT_D, TX_D, REACT_CAP, k);
      }
      for (const agent of agents.values()) {
        const s = agent.slot;
        const id = agent.id;
        const o = s * REACT_SPECIES;
        for (let port = 0; port < 3; port++) {
          const wid = PW[s * 3 + port];
          if (wid < 0) continue;
          const wire = wires.get(wid);
          if (!wire) continue;
          // Signed `a` toward `b`, and this body is one of the two.
          const sign = wire.a.id === id ? -1 : 1;
          if (wire.fluxGut !== 0) GUT[s] += sign * wire.fluxGut;
          if (wire.fluxB !== 0) R[o + REACT_B] += sign * wire.fluxB;
          if (wire.fluxC !== 0) R[o + REACT_C] += sign * wire.fluxC;
          if (wire.fluxD !== 0) R[o + REACT_D] += sign * wire.fluxD;
        }
        // The wave follows the catalyst, so a body that has just been driven
        // strokes as what it now is rather than what it was.
        speciesWaves(R[o + REACT_B], R[o + REACT_C], R[o + REACT_D], k2, k3, dec, waveK, WV);
        // Through the same `actuatorPair` the first read uses, depth profile
        // and all. This is the later of the two and it wins, so anything the
        // first one does and this one does not is not done at all.
        actuatorPair(CHEM, s * CHEM_LEN, WV, profile * DEPTH[s], SWROT, GWROT, PAIR);
        WAVE[s] = PAIR[0];
        ANCHOR[s] = GA[s] * PAIR[0];
        GRIPW[s] = PAIR[1];
      }
      // Something arrived in a gut, so the digestion pass has work again.
      this.gutLive = true;
    }
  }

  /**
   * One drag law for bodies; the rope is damped inside the substep loop.
   *
   * A body's rate is `drag + grip * fullness + anchor`. `grip` makes it
   * depend on the tank, which is what lets a net's internal pumping carry it
   * anywhere — see the parameter, which has the argument and the arithmetic.
   * `anchor` is this frame's point in the gait, and is what turns a standing
   * asymmetry into a stroke; see `advanceGait`.
   *
   * The exponential is read off a table rather than taken per body.
   * `Math.exp` is by an order of magnitude the most expensive thing in this
   * loop and `dt` is the same for everyone, so 256 of them serve fifty
   * thousand bodies. The table is over the *total rate* rather than over
   * fullness, which it used to be: with a gait the rate is no longer a
   * function of one bounded scalar, so there is nothing narrower to key on.
   * 256 steps rather than 64 because the span is now several times wider and
   * the resolution should not go backwards — at the shipping dials it is
   * 0.057/s a step against 0.031 before.
   *
   * Interpolated rather than snapped: two bodies a hair apart in rate should
   * stay a hair apart in drag, and a step function there would quietly sort
   * the pond into 256 kinds. With `grip` and `gaitRate` both at 0 the loop
   * below takes the flat path and never builds the table — a dial at its
   * off value should cost nothing.
   */
  private dampVelocities(params: Params, dt: number): void {
    const angKeep = Math.exp(-Math.max(0, params.angDrag) * dt);
    const store = this.agentStore;
    const LOCKED = store.locked;
    const PINNED = store.pinned;
    const VX = store.vx;
    const VY = store.vy;
    const OMEGA = store.omega;
    const base = Math.max(0, params.drag);
    const grip = params.grip;
    const gait = params.metabolicRate > 0;
    if (grip === 0 && !gait) {
      const linKeep = Math.exp(-base * dt);
      for (const agent of this.agents.values()) {
        const s = agent.slot;
        if (LOCKED[s] || PINNED[s]) continue;
        VX[s] *= linKeep;
        VY[s] *= linKeep;
        OMEGA[s] *= angKeep;
      }
      return;
    }
    const steps = Sim.GRIP_STEPS;
    const table = this.gripKeep;
    // The ceiling the table spans. Both terms are bounded — `grip` by its
    // slider and `anchor` by the head's own clamp — so this is the largest
    // rate any body can ask for, and anything past it saturates rather than
    // reading off the end.
    const swing = gait ? Math.max(0, params.gripSwing) : 0;
    const hi = base + Math.max(0, grip) * (1 + swing) + (gait ? GAIT_ANCHOR_MAX : 0);
    const span = hi > 1e-9 ? hi : 1;
    for (let i = 0; i <= steps; i++) {
      // Built over a rate that is already non-negative: a negative total is
      // clamped at the lookup instead, because `exp` of one silently
      // amplifies rather than failing.
      table[i] = Math.exp(-((span * i) / steps) * dt);
    }
    const EXTRA = store.extra;
    const CAP = store.energyCap;
    const ANCHOR = store.anchor;
    const GRIPW = store.gaitGrip;
    for (const agent of this.agents.values()) {
      const s = agent.slot;
      if (LOCKED[s] || PINNED[s]) continue;
      // The same clamp the genome's `IN_FULL` gets, so grip and the body's own
      // sense of how full it is never disagree.
      const cap = CAP[s];
      const raw = cap > 0 ? EXTRA[s] / cap : 0;
      const full = raw <= 0 ? 0 : raw >= 1 ? 1 : raw;
      /*
       * `grip * full` is the standing asymmetry that turns a transport kick
       * into travel; the swing is the cyclic one, from D. Two actuators at a
       * phase the chemistry sets — see `params.gripSwing`.
       */
      const rate = base + grip * full * (1 + swing * GRIPW[s]) + ANCHOR[s];
      const r = rate <= 0 ? 0 : rate >= span ? span : rate;
      const u = (r / span) * steps;
      const i = u < steps ? u | 0 : steps - 1;
      const keep = table[i] + (table[i + 1] - table[i]) * (u - i);
      VX[s] *= keep;
      VY[s] *= keep;
      OMEGA[s] *= angKeep;
    }
  }

  /**
   * Wires that have been stretched past `wireSnap` tear loose.
   *
   * Measured against the same live-length-over-rest ratio the rope coarsening
   * uses, so a wire that is merely taut is left alone and one that is being
   * pulled apart is not. A wire either end of which is mid-rewrite is spared:
   * the rewrite is about to rebuild that neighbourhood anyway, and cutting one
   * of its wires out from under it leaves the graph in a shape it did not
   * expect.
   */
  private readonly snapping: number[] = [];

  private snapTautWires(params: Params): void {
    const limit = params.wireSnap;
    if (limit <= 0) return;
    const doomed = this.snapping;
    doomed.length = 0;
    for (const wire of this.graph.wires.values()) {
      if (this.rewriteFrozen.has(wire.a.id) || this.rewriteFrozen.has(wire.b.id)) continue;
      const rest = this.graph.restLength(wire, this.time, params);
      if (!(rest > 0) || !Number.isFinite(wire.lastLen)) continue;
      if (wire.lastLen / rest > limit) doomed.push(wire.id);
    }
    for (let i = 0; i < doomed.length; i++) this.graph.detach(doomed[i]);
    this.tally.snaps += doomed.length;
  }

  /** Drop a free forager near the flock every spawnInterval seconds. */
  private autoSpawn(params: Params, dt: number): void {
    const interval = params.spawnInterval;
    if (interval <= 0 || !this.canSpawn(params)) return;
    this.spawnAcc += dt;
    while (this.spawnAcc >= interval && this.canSpawn(params)) {
      this.spawnAcc -= interval;
      const roll = Math.random();
      const kind: AgentKind = roll < 0.34 ? 'era' : roll < 0.67 ? 'dup' : 'con';
      const com = this.centerOfMass() ?? { x: this.w * 0.5, y: this.h * 0.5 };
      const reach = Math.min(
        Sim.SPAWN_REACH_CAP,
        0.28 * Math.min(this.coverW, this.coverH),
      );
      const r = 36 + Math.random() * Math.max(40, reach);
      const a = Math.random() * Math.PI * 2;
      const px = com.x + Math.cos(a) * r;
      const py = com.y + Math.sin(a) * r;
      if (this.worldR > 0) {
        const dx = px - this.worldX;
        const dy = py - this.worldY;
        if (dx * dx + dy * dy > this.worldR * this.worldR) continue;
      }
      this.spawn(kind, px, py, Math.random() * Math.PI * 2, params);
    }
  }

  private dropAcc = 0;

  /**
   * Lay a fresh patch of ground somewhere every `groundDropEvery` seconds.
   *
   * Beside `autoSpawn` because it is the same shape and the same argument from
   * the other side: immigration keeps bodies arriving so the pond is never a
   * closed population, and this keeps *ground* arriving so the dish is never a
   * finished landscape. `groundPatches` decides the grain once and then
   * grazing and diffusion wear it flat; regrowth cannot put it back, because
   * `Fields.grow` skips a cell at zero and so heals only what is still alive.
   *
   * The accumulator is host-side and stepped by the frame's own clamped dt.
   * A device-side clock would tick on the GPU's frames rather than the sim's
   * and the two ponds would part company over a long run, which is the one
   * class of divergence nothing on the page would show.
   *
   * `while`, not `if`, for the same reason `autoSpawn` uses one: a frame that
   * swallowed several intervals — a tab returning from the background, a long
   * synchronous pause — owes the dish every drop it missed, and a pond that
   * quietly skips its income while hidden is a different pond.
   */
  private dropGround(params: Params, dt: number): void {
    const every = params.groundDropEvery;
    if (every <= 0 || params.groundDropMass <= 0) {
      this.dropAcc = 0;
      return;
    }
    this.dropAcc += dt;
    // A clamp on the catch-up, so a pause cannot hand the dish a year of food
    // in one frame. Eight drops is well past any frame the page produces.
    if (this.dropAcc > every * 8) this.dropAcc = every * 8;
    while (this.dropAcc >= every) {
      this.dropAcc -= every;
      this.energy.dropSomewhere(params.groundDropMass);
    }
  }

  /**
   * Laying scent, in the solver.
   *
   * The field goes across once, the deposit writes it in place, and it comes
   * back once — arithmetic over a grid the solver already holds a mirror of.
   *
   * This used to stamp a wall mask alongside the deposit, rasterizing every
   * wire so scent could not diffuse across it. That went when the field became
   * world-fixed: a cell is 160 world units and a wire is 40, so a dense net put
   * a wire in nearly every cell it occupied and scent stopped moving through
   * tissue at all. Packing the polylines to achieve that was also the single
   * most expensive piece of host work left in the frame.
   */
  private scentWriteNative(params: Params): boolean {
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    const free = nativeSolver.portFree;
    const kinds = nativeSolver.kind;
    const sc = nativeSolver.scale;
    const emit = nativeSolver.bodyEmit;
    const EMITS = this.agentStore.emitAll;
    if (!free || !kinds || !sc || !emit) return false;

    const list = this.forceList();
    const n = list.length;
    if (!nativeSolver.canNear(n, 0, 0)) return false;
    /*
     * Grow the live box to cover the bodies before anything crosses. The
     * solver deposits and samples inside wasm, so `Fields.deposit` is never
     * called on this path and the box would never learn where the pond is —
     * and the box is what decides which rows get copied across.
     */
    for (let i = 0; i < n; i++) this.fields.touchWorld(list[i].x, list[i].y);
    if (!nativeSolver.loadScent(this.fields)) return false;
    Sim.phase('scent:load');

    const bodies = nativeSolver.bodies;
    if (!bodies) return false;
    /*
     * This runs in endFrame, after rewrites and upkeep have had their turn at
     * the roster, so the flag computed back in beginFrame cannot be trusted
     * here — the key is asked again.
     */
    const freeFresh = nativeSolver.scratchHolds(
      this.simId,
      this.graph.version,
      this.rosterVersion,
      n,
    );
    // Out of the store, for the reason `packPose` gives: these accessors all
    // reach through the same two properties to the same arrays, and the kind
    // is already the integer the solver wants.
    const st = this.agentStore;
    const X = st.x;
    const Y = st.y;
    const HD = st.heading;
    const LK = st.locked;
    const SC = st.scale;
    const KC = st.kindCode;
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const sl = a.slot;
      const o = i * FAR_STRIDE;
      bodies[o + FAR.x] = X[sl];
      bodies[o + FAR.y] = Y[sl];
      bodies[o + FAR.heading] = HD[sl];
      bodies[o + FAR.locked] = LK[sl] !== 0 ? 1 : 0;
      kinds[i] = KC[sl];
      sc[i] = SC[sl];
      // Materialised by `updateState`; `CH.energy` and `CH.aux` are masked
      // here rather than in the vector, because the vector is the budget and
      // the deposit path is the thing that must never see the ground or
      // counterfeit its smell. See `effEmit` for both reasons.
      const eo = sl * 4;
      for (let c = 0; c < 4; c++) {
        emit[i * 4 + c] = c === CH.energy || c === CH.aux ? 0 : EMITS[eo + c];
      }
      if (!freeFresh) free[i] = this.freePortMask(a);
    }
    Sim.phase('scent:bodyPack');
    // Here rather than in the steer pass, though they share one array: this is
    // the pass that reads it, and a value written by another pass is a value
    // that depends on the order the two happen to run in.
    const spDep = nativeSolver.steerParams;
    if (spDep) {
      spDep[STEER_PARAM.portLeak] = params.portLeak;
      spDep[STEER_PARAM.auxLeak] = params.auxLeak;
    }
    nativeSolver.deposit(n, params.deposit);
    Sim.phase('scent:deposit');

    nativeSolver.storeScent(this.fields);
    Sim.phase('scent:store');
    return true;
  }

  /*
   * A body says what it is; a free port says that it is free. Those were one
   * thing and are now two.
   *
   * Every port used to be skipped unless it was unattached, which meant a
   * fully wired body emitted nothing whatever — so a net was mute, and its
   * voice was not its own but its periphery's. A net of forty bodies with
   * four free ports spoke exactly as loudly as four loose ones. That is the
   * wrong shape for the thing this world is supposed to be about: a net is
   * the organism, and it could not be heard.
   *
   * So the principal lays this body's voice whether or not it is attached.
   * Volume is per body, not per free port, so a net's carrying distance now
   * scales with how much of it there is — which is what makes "there is
   * something large over there" a thing a stranger can smell at all.
   *
   * The free-port marker keeps its gate, because it is not a voice. It means
   * "there is somewhere to attach here", and that has to stay false when there
   * is not, or the one kind-independent signal in the field stops being true.
   * Emitting it from a filled port would advertise a socket that is not there
   * and every latch-seeking body in range would come and find nothing.
   *
   * **It is no longer on `CH.aux`**, which is the ground's now — `Fields.grow`
   * mints it in proportion to the food standing in a cell, and the channel
   * with the longest reach belongs to the fixed thing worth walking toward
   * from far away. What replaces it is not one marker but two, and which
   * channel each uses is decided by the seed's own arithmetic rather than
   * chosen here.
   *
   * **A free principal leaks its own kind's channel: a Con `conP`, a Dup
   * `dupP`, an Era nothing.** Look at what `seedChem` does and it is forced.
   * A Con emits ch0 and tastes ch1 and ch3; a Dup emits ch1 and tastes ch0 and
   * ch3; **neither tastes the channel it speaks on**. So a kind advertising on
   * its own channel is invisible to itself and to its own kind — a Con cannot
   * follow its own trail and cannot be drawn into a huddle of Cons — while
   * being loud to exactly the two kinds that want it: a Dup at
   * `attractMedium`, and an Era, which tastes both at `attractStrong`. And a
   * free principal is precisely what the other body needs, because a redex is
   * two principals nose to nose: Con-Dup commutes, Era-anything erases. The
   * signal and the rule it serves line up without either being bent.
   *
   * An Era leaks nothing because it has nothing to advertise. Its one port is
   * how it eats and how it erases; it is the kind that goes looking, not the
   * kind that is looked for.
   *
   * **A free auxiliary leaks `auxLeak` of both**, which is the old
   * kind-independent "there is somewhere to attach here". Self-following is
   * not a worry there the way it would be on a principal: an aux port is not
   * where a body senses from, and a socket is a near-field cue that wants to
   * be found by anybody rather than by one kind.
   *
   * Making either a gene — `E[aux][BOUND]`, so a body advertises as loudly as
   * its lineage has learned to — is still not done, and the reason still
   * stands: a body's voice is already emitted from its principal, so a second
   * genetic term at the port positions is either double-counting or a separate
   * gene, and "separate gene" is not the subsumption it was billed as.
   */
  private deposit(params: Params): void {
    const EMITS = this.agentStore.emitAll;
    const p = this.portScratch;
    const leak = params.deposit * params.portLeak;
    const aux = params.deposit * params.auxLeak;
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      const own = ownChannel(agent.kind);
      const slots = agent.kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
      for (let k = 0; k < slots.length; k++) {
        const slot = slots[k];
        const free = this.graph.isFreeAt(agent.id, slot);
        if (slot === 'p') {
          portWorldInto(agent, slot, this.w, this.h, p);
          const eo = agent.slot * 4;
          for (let ch = 0; ch < 4; ch++) {
            // The ground and its smell are not a body's to lay. See `effEmit`.
            if (ch === CH.energy || ch === CH.aux) continue;
            const w = EMITS[eo + ch];
            if (w !== 0) this.fields.deposit(ch, p.x, p.y, params.deposit * w);
          }
          // "My principal is free", on the one channel this kind is deaf to.
          if (free && leak > 0 && own >= 0) this.fields.deposit(own, p.x, p.y, leak);
        } else if (free && aux > 0) {
          portWorldInto(agent, slot, this.w, this.h, p);
          // "There is somewhere to attach here", to anybody.
          this.fields.deposit(CH.conP, p.x, p.y, aux);
          this.fields.deposit(CH.dupP, p.x, p.y, aux);
        }
      }
    }
  }

  /**
   * Dotted against the materialised taste vector rather than through
   * `mixScent`, which rebuilds it from the genome on every call — and this is
   * called three times a body a frame, once per sensor and once for the trail.
   * `mixScent` stays as the readable statement of what the dot product is, and
   * is what the tests exercise.
   */
  private scentAt(agent: Agent, x: number, y: number, _params: Params): number {
    const t = this.agentStore.tasteAll;
    const o = agent.slot * 4;
    const gain = this.tasteGain(agent.slot);
    if (gain === 0) return 0;
    return (
      gain *
      (t[o] * this.fields.sample(0, x, y) +
        t[o + 1] * this.fields.sample(1, x, y) +
        t[o + 2] * this.fields.sample(2, x, y) * this.groundScale +
        t[o + 3] * this.fields.sample(3, x, y))
    );
  }

  /**
   * What a full cell of ground reads as, once scaled: 1.
   *
   * Zero when there is no ground to speak of — an `ambientEnergy` of 0, which
   * is how most of the energy tests set up a barren world. A body cannot
   * smell what is not there, and this keeps that from being a division by it.
   */
  private get groundScale(): number {
    const cap = this.energy.cellCap;
    return cap > 1e-12 ? 1 / cap : 0;
  }

  
  /**
   * Steering in WASM. False when the scene will not pack, or when the scent
   * window is bigger than the solver's buffer.
   */
  private steerNative(params: Params, dt: number): boolean {
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    const sp = nativeSolver.steerParams;
    const flags = nativeSolver.steerFlags;
    const pwire = nativeSolver.steerPwire;
    const noise = nativeSolver.steerNoise;
    const drive = nativeSolver.bodyDrive;
    const taste = nativeSolver.bodyTaste;
    const cruiseArr = nativeSolver.bodyCruise;
    const turnArr = nativeSolver.bodyTurn;
    const CRUISE_OF = this.agentStore.cruise;
    const TURN_OF = this.agentStore.turn;
    const trail = nativeSolver.bodyTrail;
    const kinds = nativeSolver.kind;
    const sc = nativeSolver.scale;
    if (!sp || !flags || !pwire || !noise || !drive || !trail || !kinds || !sc || !taste) {
      return false;
    }
    const list = this.forceList();
    const n = list.length;
    if (n === 0) return true;
    if (!nativeSolver.canNear(n, 0, 0)) return false;
    /*
     * The field crosses only when the solver is going to sample it. On the
     * GPU path the three readings a body steers on came back with last
     * frame's probe and sit in the store by slot; the box copy used to happen
     * regardless, which with the field on the GPU is the whole disk — a
     * twelve-megabyte memcpy into wasm every frame that nothing then read.
     *
     * Packed by slot into list order here, rather than written into the
     * solver's array in list order when they arrived, because the list can
     * move between the two — see `AgentStore.steerAll`.
     */
    if (this.steerFromSamples) {
      const out = nativeSolver.steerSamples;
      if (!out) return false;
      const STEER = this.agentStore.steerAll;
      for (let i = 0; i < n; i++) {
        const so = list[i].slot * 3;
        out[i * 3] = STEER[so];
        out[i * 3 + 1] = STEER[so + 1];
        out[i * 3 + 2] = STEER[so + 2];
      }
      nativeSolver.useSamples(true);
    } else {
      if (!nativeSolver.loadScent(this.fields)) return false;
      nativeSolver.useSamples(false);
    }
    if (!this.forceBlock && !this.packPose(list)) return false;

    /*
     * Bit 0 of `flags` and the whole of `pwire` come from the topology, so
     * refreshForceScratch has already written them and they survive until a
     * wire changes. What is left here is genuinely per-frame: the stun bit,
     * the drive level, and three fresh random numbers.
     */
    /*
     * The per-body pack, straight out of the store.
     *
     * It was fourteen accessor calls and a method call a body: `tasteOf` was
     * four of those, each one a call that re-read `a.slot` to index an array
     * this can hold directly, and the ground scale it applies is a constant
     * for the frame on one known channel. Measured over fifty thousand
     * bodies, back to back on the same list: **6.88 ms as written, 1.98 ms
     * from the store**, of which 1.07 ms is the three random numbers, which
     * have to be drawn host-side so a seeded run stays reproducible.
     */
    const TASTE_OF = this.agentStore.tasteAll;
    const DRIVE_OF = this.agentStore.drive;
    const TRAIL_OF = this.agentStore.trail;
    const STUN_OF = this.agentStore.stun;
    const SCALE_OF = this.agentStore.scale;
    const KIND_OF = this.agentStore.kindCode;
    const groundScale = this.groundScale;
    const packed = !this.forceBlock;
    if (this.scratchFresh) {
      for (let i = 0; i < n; i++) {
        const sl = list[i].slot;
        flags[i] = (flags[i] & STEER_FLAG.pFree) | (STUN_OF[sl] > 0 ? STEER_FLAG.stunned : 0);
        // Under a force block packPose has already written these; without one
        // nothing else does, and scale moves every frame as bodies grow.
        if (packed) {
          kinds[i] = KIND_OF[sl];
          sc[i] = SCALE_OF[sl];
        }
        drive[i] = DRIVE_OF[sl];
        packTaste(taste, i * 4, TASTE_OF, sl, groundScale, this.tasteGain(sl));
        if (cruiseArr && turnArr) {
          cruiseArr[i] = CRUISE_OF[sl];
          turnArr[i] = TURN_OF[sl];
        }
        // Drawn host-side so a seeded run stays reproducible; the solver only
        // consumes them.
        noise[i * 3] = Math.random();
        noise[i * 3 + 1] = Math.random();
        noise[i * 3 + 2] = Math.random();
      }
    } else {
      const at = this.slotIndex;
      for (let i = 0; i < n; i++) {
        const a = list[i];
        const sl = a.slot;
        const pFree = this.graph.isFreeAtSlot(sl, 0);
        flags[i] = (pFree ? STEER_FLAG.pFree : 0) | (STUN_OF[sl] > 0 ? STEER_FLAG.stunned : 0);
        if (packed) {
          kinds[i] = KIND_OF[sl];
          sc[i] = SCALE_OF[sl];
        }
        drive[i] = DRIVE_OF[sl];
        packTaste(taste, i * 4, TASTE_OF, sl, groundScale, this.tasteGain(sl));
        if (cruiseArr && turnArr) {
          cruiseArr[i] = CRUISE_OF[sl];
          turnArr[i] = TURN_OF[sl];
        }
        const pw = this.graph.wireAtSlot(a.id, 'p');
        let pj = -1;
        let pslot = 0;
        if (pw) {
          const other = pw.a.id === a.id ? pw.b : pw.a;
          const ob = this.agents.get(other.id);
          pj = ob ? at[ob.slot] : -1;
          pslot = this.slotCode(other.slot);
        }
        pwire[i * 2] = pj;
        pwire[i * 2 + 1] = pslot;
        noise[i * 3] = Math.random();
        noise[i * 3 + 1] = Math.random();
        noise[i * 3 + 2] = Math.random();
      }
    }
    const SP = STEER_PARAM;
    sp[SP.faceRadius] = params.faceRadius;
    sp[SP.snapRadius] = params.snapRadius;
    sp[SP.snapArc] = params.snapArc;
    sp[SP.faceAttract] = params.faceAttract;
    sp[SP.snapWell] = params.snapWell;
    sp[SP.sensorAngle] = params.sensorAngle;
    sp[SP.sensorDist] = params.sensorDist;
    sp[SP.sense] = params.sense;
    sp[SP.turnRate] = params.turnRate;
    sp[SP.stepSpeed] = params.stepSpeed;
    sp[SP.swimTau] = params.swimTau;
    sp[SP.swimNoise] = params.swimNoise;
    // `unused12` and `unused13` are dead: they carried `attractStrong` and
    // `attractMedium`, which are seeds read once by `seedChem` and by nothing
    // in the solver. Reserved rather than reclaimed; see the note in solver.c.
    sp[SP.senseSpan] = SENSE_SPAN;
    nativeSolver.steer(n, dt);
    // Back the same way: two setter calls a body wrote these two arrays.
    for (let i = 0; i < n; i++) {
      const sl = list[i].slot;
      DRIVE_OF[sl] = drive[i];
      TRAIL_OF[sl] = trail[i];
    }
    if (!this.forceBlock) this.unpackDrift(list);
    return true;
  }

  private steer(params: Params, dt: number): void {
    if (this.steerNative(params, dt)) return;
    const { w, h } = this;
    // Face-attraction and the snap well both cut off at faceRadius, so this only
    // ever needed nearby agents; it used to walk the whole population per agent.
    const near = Math.max(params.faceRadius, params.snapRadius);
    const list = this.rebuildBodyGrid(Math.max(1, near));

    // Every direct field touch below goes through the store instead of
    // Agent's accessors — measured the single hottest pass in the sim at
    // scale (profiled ~32% of a frame). `agent` itself is still threaded
    // through to portWorld/scentAt/inSnapArc/graph.*, which need the
    // Agent shape (kind, chem) and are out of scope here.
    const store = this.agentStore;
    const X = store.x;
    const Y = store.y;
    const HEADING = store.heading;
    const OMEGA = store.omega;
    const VX = store.vx;
    const VY = store.vy;
    const DRIVE = store.drive;
    const TRAIL = store.trail;
    const LOCKED = store.locked;
    const PINNED = store.pinned;
    const STUN = store.stun;
    const ID = store.id;
    const CRUISE_JS = store.cruise;
    const TURN_JS = store.turn;

    for (const agent of list) {
      const s = agent.slot;
      if (LOCKED[s] || PINNED[s]) continue;
      const ax = X[s];
      const ay = Y[s];
      const aHeading = HEADING[s];
      const aId = ID[s];

      let biasX = 0;
      let biasY = 0;
      const tip = portWorld(agent, 'p', w, h);
      const pWire = this.graph.wireAt({ id: aId, slot: 'p' });
      if (pWire) {
        const other = pWire.a.id === aId ? pWire.b : pWire.a;
        const otherA = this.agents.get(other.id);
        if (otherA) {
          const op = portWorld(otherA, other.slot, w, h);
          const d = wrapDeltaVec(tip.x, tip.y, op.x, op.y, w, h);
          biasX += d.x;
          biasY += d.y;
        }
      }

      if (STUN[s] <= 0 && this.graph.isFreeAt(aId, 'p')) {
        const aCos = Math.cos(aHeading);
        const aSin = Math.sin(aHeading);
        this.bodyGrid.forEachNear(ax, ay, near, (idx) => {
          const other = list[idx];
          const os = other.slot;
          const oId = ID[os];
          if (oId === aId || LOCKED[os] || STUN[os] > 0) return;
          if (!this.graph.isFreeAt(oId, 'p')) return;
          const d = wrapDeltaVec(ax, ay, X[os], Y[os], w, h);
          const dist = Math.hypot(d.x, d.y);
          if (dist < 1e-4 || dist > params.faceRadius) return;
          const nx = d.x / dist;
          const ny = d.y / dist;
          const oHeading = HEADING[os];
          const aFace = aCos * nx + aSin * ny;
          const bFace = Math.cos(oHeading) * -nx + Math.sin(oHeading) * -ny;
          if (aFace > 0.35 && bFace > 0.35) {
            biasX += nx * params.faceAttract * aFace * bFace;
            biasY += ny * params.faceAttract * aFace * bFace;
          }
          const op = portWorld(other, 'p', w, h);
          if (inSnapArc(agent, 'p', op.x, op.y, w, h, params.snapRadius, params.snapArc)) {
            const pd = wrapDeltaVec(tip.x, tip.y, op.x, op.y, w, h);
            const pdist = Math.hypot(pd.x, pd.y) || 1;
            const well = (1 - pdist / params.snapRadius) * params.snapWell;
            biasX += (pd.x / pdist) * well;
            biasY += (pd.y / pdist) * well;
          }
        });
      }

      const arc = params.sensorAngle;
      const dist = params.sensorDist;
      const leftA = aHeading - arc;
      const rightA = aHeading + arc;
      const gain = 0.35 + params.sense / 500;
      const bm = Math.hypot(biasX, biasY);
      const scoreAt = (a: number): number => {
        const sx = ax + Math.cos(a) * dist;
        const sy = ay + Math.sin(a) * dist;
        let score = this.scentAt(agent, sx, sy, params) * gain;
        if (bm > 1e-6) {
          score += (1.6 * (biasX * Math.cos(a) + biasY * Math.sin(a))) / bm;
        }
        return score;
      };
      const left = scoreAt(leftA);
      const right = scoreAt(rightA);
      // Proportional past a deadband; see the C twin in solver_steer.
      const diff = right - left;
      const mag = Math.abs(left) + Math.abs(right) + 1e-6;
      const rel = diff / mag;
      const dz = 0.05 + 0.03 / mag;
      const over = Math.abs(rel) - dz;
      let t = 0;
      if (over > 0) {
        const span = SENSE_SPAN - dz;
        t = span > 1e-6 ? Math.min(1, over / span) : 1;
        if (rel < 0) t = -t;
      }
      const err = angleDelta(aHeading, aHeading + arc * t);
      const trail = this.scentAt(agent, ax, ay, params);
      TRAIL[s] = trail;
      const slow = scentSlowFactor(trail);
      const turnBoost = scentTurnBoost(trail);
      const bodyTurn = TURN_JS[s];
      const kp = bodyTurn * 6 * turnBoost;
      const kd = (bodyTurn * 2) / Math.sqrt(turnBoost);
      const principalFree = this.graph.isFreeAt(aId, 'p');
      if (principalFree) {
        OMEGA[s] += (kp * err - kd * OMEGA[s]) * dt;
      }
      const bodyCruise = CRUISE_JS[s];
      if (principalFree && bodyCruise > 0) {
        // Active Ornstein-Uhlenbeck propulsion: the drive decays toward cruise
        // with a persistence time while coloured noise kicks it. A swimmer
        // surges and eases the way a crawling cell does, where the servo this
        // replaces — drive velocity straight at a setpoint — reads mechanical.
        const cruise = bodyCruise * slow;
        const tau = Math.max(0.05, params.swimTau);
        const kick =
          params.swimNoise * cruise * Math.sqrt(dt / tau) *
          (Math.random() + Math.random() + Math.random() - 1.5) * 2;
        let drive = DRIVE[s] + ((cruise - DRIVE[s]) / tau) * dt + kick;
        drive = clamp(drive, -cruise * 0.4, cruise * 2.2);
        DRIVE[s] = drive;
        const hx = Math.cos(aHeading);
        const hy = Math.sin(aHeading);
        const along = VX[s] * hx + VY[s] * hy;
        const blend = 1 - Math.exp(-6 * dt);
        const dAlong = (drive - along) * blend;
        VX[s] += dAlong * hx;
        VY[s] += dAlong * hy;
      }
    }
  }

  /**
   * Active locomotion: accelerate only along the principal port.
   * Latched principals don't swim; they just follow the pull.
   */
  private locomote(id: number, slot: number, wishX: number, wishY: number, turnK: number): void {
    if (!this.graph.isFreeAt(id, 'p')) return;
    const store = this.agentStore;
    if (store.locked[slot] || store.pinned[slot]) return;
    const heading = store.heading[slot];
    const hx = Math.cos(heading);
    const hy = Math.sin(heading);
    const ahead = wishX * hx + wishY * hy;
    if (ahead > 0) {
      store.vx[slot] += ahead * hx;
      store.vy[slot] += ahead * hy;
    }
    const mag = Math.hypot(wishX, wishY);
    if (mag > 1e-8 && turnK !== 0) {
      store.omega[slot] += turnK * angleDelta(heading, Math.atan2(wishY, wishX));
    }
  }

  /** Flock / constraint pulls on wired cargo (no principal swim). */
  private netPull(slot: number, wishX: number, wishY: number, _turnK: number): void {
    const store = this.agentStore;
    if (store.locked[slot] || store.pinned[slot]) return;
    store.vx[slot] += wishX;
    store.vy[slot] += wishY;
  }

  private netForce(id: number, slot: number, wishX: number, wishY: number, turnK: number): void {
    if (this.graph.isFreeAt(id, 'p')) this.locomote(id, slot, wishX, wishY, turnK);
    else this.netPull(slot, wishX, wishY, turnK);
  }

  private flockNative(
    list: Agent[],
    swim: Uint8Array,
    n: number,
    align: number,
    sep: number,
    dt: number,
    turnRate: number,
    desired: number,
    maxHops: number,
    reuse: boolean,
  ): boolean {
    const bodies = nativeSolver.bodies;
    const adjOff = nativeSolver.adjOff;
    const adjNei = nativeSolver.adjNei;
    const ids = nativeSolver.flockId;
    const mass = nativeSolver.flockMass;
    const fa = nativeSolver.flockAlign;
    const fs = nativeSolver.flockSep;
    const sw = nativeSolver.swim;
    if (!bodies || !adjOff || !adjNei || !ids || !mass || !sw || !fa || !fs) return false;

    // On a reuse frame the solver never reads the adjacency — it replays the
    // pair list it already built from it — so neither the fit check nor the
    // copy has anything to do.
    if (!reuse) {
      const off = this.adjOff;
      const nei = this.adjNei;
      const nAdj = off[n];
      if (!nativeSolver.canFlock(n, nAdj)) return false;
      // Straight across: both sides are the same CSR, in the same order.
      adjOff.set(off.subarray(0, n + 1));
      adjNei.set(nei.subarray(0, nAdj));
    }

    const shared = this.forceBlock;
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      if (!shared) {
        bodies[o + FAR.x] = a.x;
        bodies[o + FAR.y] = a.y;
        bodies[o + FAR.vx] = a.vx;
        bodies[o + FAR.vy] = a.vy;
        bodies[o + FAR.heading] = a.heading;
        bodies[o + FAR.omega] = a.omega;
        bodies[o + FAR.locked] = poseHeld(a) ? 1 : 0;
      }
      ids[i] = a.id;
      mass[i] = a.mass;
      fa[i] = flockGain(a.flockAlign);
      fs[i] = flockGain(a.flockSep);
      sw[i] = swim[i];
    }
    if (
      !nativeSolver.flock(
        n, align, sep, dt, turnRate, desired, maxHops,
        this.simId, this.graph.version, this.rosterVersion,
      )
    ) {
      return false;
    }
    if (!shared) this.unpackDrift(list);
    return true;
  }

  /**
   * Boids on the net. Weight is 1/hops out to FLOCK_HOPS; farther and
   * disconnected pairs are ignored. Meridians align nematically (parallel,
   * either polarity), velocities match, clumps separate, and junctions sit
   * toward neighbor centroids so wires straighten.
   */
  private flock(params: Params, dt: number): void {
    /*
     * The gains are per body now, so the params are only a gate: a pond whose
     * sliders are both zero still has bodies carrying whatever their lineage
     * bred, and the largest of those decides whether the pass runs at all.
     * The values passed down are the maxima, used for nothing but that test —
     * the force itself reads each pair's own mean.
     */
    let align = params.flockAlign;
    let sep = params.flockSep;
    {
      const FLOCK_ALIGN = this.agentStore.flockAlign;
      const FLOCK_SEP = this.agentStore.flockSep;
      for (const a of this.agents.values()) {
        const ga = flockGain(FLOCK_ALIGN[a.slot]);
        const gs = flockGain(FLOCK_SEP[a.slot]);
        if (ga > align) align = ga;
        if (gs > sep) sep = gs;
      }
    }
    if ((align <= 0 && sep <= 0) || dt <= 0) return;
    const list = this.forceList();
    const n = list.length;
    if (n === 0) return;

    const maxHops = Sim.FLOCK_HOPS;
    /*
     * Adjacency, the index map and the swim flags are all functions of the
     * topology and the roster, so on a frame where neither moved they are
     * already correct from last time — as is the neighbourhood cache inside
     * the solver, which is keyed on exactly the same thing. Rebuilding them
     * anyway was most of what this pass cost at scale.
     */
    const reuse = nativeSolver.flockCacheHolds(
      this.simId,
      this.graph.version,
      this.rosterVersion,
      n,
      maxHops,
    );
    /*
     * The body adjacency CSR, which is the same wires in the same order on the
     * same key. This pass used to build an array-of-arrays and then flatten
     * it into exactly this shape to hand to wasm — a push per wire end and
     * then a copy of the lot — for a structure two other passes were already
     * keeping.
     */
    this.refreshBodyAdjacency();
    const off = this.adjOff;
    const nei = this.adjNei;

    if (this.flockDist.length < n) {
      const cap = Math.max(n * 2, 16);
      this.flockDist = new Int32Array(cap);
      this.flockDist.fill(-1);
      this.flockQ = new Int32Array(cap);
    }
    if (this.flockSwim.length < n) this.flockSwim = new Uint8Array(n * 2);
    if (this.flockSlot.length < n) this.flockSlot = new Int32Array(n * 2);
    const dist = this.flockDist;
    const q = this.flockQ;
    const swim = this.flockSwim;
    const slotOf = this.flockSlot;
    dist.fill(-1, 0, n);
    for (let i = 0; i < n; i++) slotOf[i] = list[i].slot;
    if (!reuse) {
      for (let i = 0; i < n; i++) {
        swim[i] = this.graph.isFreeAt(list[i].id, 'p') ? 1 : 0;
      }
    }

    const desired = Math.max(18, params.wireMinRest * 0.9);
    const turnRate = params.turnRate;
    if (this.flockNative(list, swim, n, align, sep, dt, turnRate, desired, maxHops, reuse)) {
      return;
    }

    const store = this.agentStore;
    const ID = store.id;
    const LOCKED = store.locked;
    const PINNED = store.pinned;
    const MASS = store.mass;
    const X = store.x;
    const Y = store.y;
    const VX = store.vx;
    const VY = store.vy;
    const FLOCK_ALIGN = store.flockAlign;
    const FLOCK_SEP = store.flockSep;

    const seen = this.flockSeen;

    for (let start = 0; start < n; start++) {
      const sA = slotOf[start];
      if (LOCKED[sA] || PINNED[sA]) continue;
      const idA = ID[sA];
      seen.length = 0;
      dist[start] = 0;
      seen.push(start);
      let qh = 0;
      let qt = 0;
      q[qt++] = start;
      while (qh < qt) {
        const u = q[qh++];
        const du = dist[u];
        if (du >= maxHops) continue;
        const e0 = off[u];
        const e1 = off[u + 1];
        for (let k = e0; k < e1; k++) {
          const v = nei[k];
          if (dist[v] >= 0) continue;
          const d = du + 1;
          dist[v] = d;
          seen.push(v);
          q[qt++] = v;
          const sB = slotOf[v];
          const idB = ID[sB];
          if (LOCKED[sB] || idB <= idA) continue;
          const w = 1 / d;
          const mA = Math.max(0.08, MASS[sA]);
          const mB = Math.max(0.08, MASS[sB]);
          const mSum = mA + mB;
          const dx = X[sB] - X[sA];
          const dy = Y[sB] - Y[sA];
          const gap = Math.hypot(dx, dy) || 1e-6;
          const nx = dx / gap;
          const ny = dy / gap;

          // The pair's mean of the clamped gains, matching the solver.
          const pairAlign = 0.5 * (flockGain(FLOCK_ALIGN[sA]) + flockGain(FLOCK_ALIGN[sB]));
          const pairSep = 0.5 * (flockGain(FLOCK_SEP[sA]) + flockGain(FLOCK_SEP[sB]));
          if (pairAlign > 0) {
            const kAlign = pairAlign * w * dt;
            const dvx = VX[sB] - VX[sA];
            const dvy = VY[sB] - VY[sA];
            this.netForce(idA, sA, dvx * kAlign * (mB / mSum), dvy * kAlign * (mB / mSum), 0);
            this.netForce(idB, sB, -dvx * kAlign * (mA / mSum), -dvy * kAlign * (mA / mSum), 0);
          }

          if (pairSep > 0 && d > 1) {
            const want = 22 + (d - 1) * desired;
            if (gap < want) {
              const mag = pairSep * w * (want - gap);
              const ax = nx * mag * dt;
              const ay = ny * mag * dt;
              const turn = turnRate * w * 0.25;
              this.netForce(idA, sA, -ax * (mB / mSum), -ay * (mB / mSum), swim[start] ? 0 : turn);
              this.netForce(idB, sB, ax * (mA / mSum), ay * (mA / mSum), swim[v] ? 0 : turn);
            }
          }
        }
      }
      for (let i = 0; i < seen.length; i++) dist[seen[i]] = -1;
    }
  }

  /**
   * Ease the home point toward the live centre of mass, until the world is
   * pinned. After that, home is the fixed anchor the scent field, the energy
   * grid, and the pull back home are all cut against — letting it keep
   * easing toward the crowd would slide the pond off its own grid, one
   * easing step at a time, which is exactly the drift pinning was meant to
   * end.
   */
  private trackHome(dt: number): void {
    if (this.worldPinned) return;
    const com = this.centerOfMass();
    if (!com) return;
    if (!this.home) {
      this.home = { x: com.x, y: com.y };
      return;
    }
    // A rewrite can delete two agents at once, which moves the true centre of
    // mass discontinuously; pulling toward that unsmoothed would kick the
    // survivors. Half a second of lag makes home a place, not an instant value.
    const k = 1 - Math.exp(-2 * dt);
    this.home.x += (com.x - this.home.x) * k;
    this.home.y += (com.y - this.home.y) * k;
  }

  momentum(): { px: number; py: number; L: number } {
    const com = this.centerOfMass() ?? { x: 0, y: 0 };
    let px = 0;
    let py = 0;
    let L = 0;
    for (const a of this.agents.values()) {
      px += a.mass * a.vx;
      py += a.mass * a.vy;
      L += a.mass * ((a.x - com.x) * a.vy - (a.y - com.y) * a.vx) + momentOfInertia(a) * a.omega;
    }
    for (const wire of this.graph.wires.values()) {
      for (const n of wire.nodes) {
        px += CHAIN_MASS * n.vx;
        py += CHAIN_MASS * n.vy;
        L += CHAIN_MASS * ((n.x - com.x) * n.vy - (n.y - com.y) * n.vx);
      }
    }
    return { px, py, L };
  }

  kineticEnergy(): number {
    let e = 0;
    for (const a of this.agents.values()) {
      e += 0.5 * a.mass * (a.vx * a.vx + a.vy * a.vy);
      e += 0.5 * momentOfInertia(a) * a.omega * a.omega;
    }
    for (const wire of this.graph.wires.values()) {
      for (const n of wire.nodes) {
        e += 0.5 * CHAIN_MASS * (n.vx * n.vx + n.vy * n.vy);
      }
    }
    return e;
  }

  /**
   * Which solver actually took the last frame. There are four ways a frame
   * can be integrated and no way to tell from the outside which one ran,
   * which matters most for the GPU path: it is gated on the whole pond
   * being FAR tier, so forcing `farGpuMode` on while zoomed in changes
   * nothing and looks identical to a path that is broken.
   */
  lastFarPath: 'gpu' | 'wasm' | 'js' | 'none' = 'none';

  /**
   * Fastest body in the pond. A health signal rather than a statistic: the
   * failure the GPU FAR path showed the one time it ran was bodies being
   * flung apart, and that shows up here long before it is legible on
   * screen at a zoom far enough out for the path to engage at all.
   */
  peakSpeed(): number {
    const store = this.agentStore;
    const VX = store.vx;
    const VY = store.vy;
    let peak = 0;
    for (const a of this.agents.values()) {
      const s = a.slot;
      const v = VX[s] * VX[s] + VY[s] * VY[s];
      if (v > peak) peak = v;
    }
    return Math.sqrt(peak);
  }

  totalFree(): number {
    let n = 0;
    for (const a of this.agents.values()) n += Math.max(0, a.extra);
    return n;
  }

  /**
   * Everything swallowed and not yet digested, across the pond.
   *
   * Anything totalling the pond has to add this or it will read a mouthful in
   * transit as matter that went missing — the same reason `escrowTotal` is
   * here. It is in neither the ground nor a tank, and it is not nothing.
   */
  totalGut(): number {
    const store = this.agentStore;
    let total = 0;
    for (const a of this.agents.values()) total += store.gutTotal(a.slot);
    return total;
  }

  totalBound(): number {
    let n = 0;
    for (const a of this.agents.values()) n += agentValue(a.kind);
    return n;
  }

  /**
   * A rewrite is a principal-port meeting, not a fully dressed net.
   *
   * Waiting until both agents are saturated, the shrink curve has finished,
   * and the rope has settled onto `wireMinRest` meant the solver did the
   * haul that the rewrite animation is for, and Dup/Con almost never fired
   * because their aux ports were still free. Leftovers may be null; that is
   * a legal net. We only wait until shrink has *started* so a latch is
   * visible for a beat, and so tests that stretch `wireShrink` still hold.
   */
  private static readonly REWRITE_SHRINK_READY = 0.45;

  private principalRedexReady(wire: Wire, A: Agent, B: Agent, params: Params): boolean {
    if (wire.a.slot !== 'p' || wire.b.slot !== 'p') return false;
    if (A.locked || B.locked || A.stun > 0 || B.stun > 0) return false;
    if (this.graph.shrinkU(wire, this.time, params) < Sim.REWRITE_SHRINK_READY) return false;
    const len = this.wireSimulatesRope(wire)
      ? this.graph.curveLength(wire, this.agents, this.w, this.h)
      : this.graph.stemSpan(wire, this.agents, this.w, this.h);
    /*
     * Not the rest-length sit: the collapse hauls them the rest of the way.
     * Still skip a cable that has barely started to take, so the pull is a
     * close and not a fling.
     *
     * **Against the wire's own rest, not the world's.** This used to read
     * `params.wireMinRest * 1.3`, one global distance in pixels — so every
     * wire in the pond had to arrive at the same length before its ends could
     * rewrite, and anything that shortened or lengthened a wire was competing
     * with breeding for the same variable. That is the second half of why
     * `wireTug` came out: not that it pulled, but that pulling moved the wire
     * away from the one number the rewrite gate was measured against. A ratio
     * against `restBase` is scale-free, so a short wire is simply a short
     * wire and a tension term can move `rest` without deciding who breeds.
     */
    return len <= wire.restBase * 1.3;
  }

  /**
   * Bank what a ready redex can afford now, so meeting poor is not the same as
   * never meeting.
   *
   * Affordability used to be an instantaneous gate. A pair became ready, was
   * asked for both shares in that one frame, and if it could not pay, the
   * meeting was lost — the need it had posted evaporated with it and the two
   * drifted on. That made "rich at the right instant" the selected trait
   * rather than "able to get rich", which is a much less interesting thing for
   * a soup to be good at.
   *
   * Two facts make an escrow the natural fix. `shrinkProgress` is monotone in
   * the wire's age, so a ready redex does not become unready by getting older;
   * and `pulseRequests` is already pumping energy toward exactly these ends.
   * What the pair lacked was somewhere to put it as it arrived.
   *
   * The pot belongs to the *wire*, not to the moment. It survives a lapse in
   * readiness — ends jostled apart by declutter, a stun — because the wire
   * surviving is what still makes the two a redex. It is refunded when the
   * wire goes, which `kill` triggers through `detachAgent`, so a partner dying
   * mid-accrual returns the stake instead of burning it.
   *
   * Runs after `pulseRequests` and before `startRewrites`, which keeps the
   * invariant the phase order was chosen for: energy that arrives to complete
   * a redex is spent on the frame it lands.
   */
  private accrueRedexes(params: Params): void {
    // Turned off mid-run: hand back what is held rather than freezing it in a
    // pot nothing will ever spend.
    if (params.rewriteDuration <= 0) {
      this.refundEscrows();
      return;
    }
    const ready = this.readyRedexes;
    for (let k = 0; k < ready.length; k++) {
      const wire = ready[k];
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      if (rewriteCost(detectRule(A.kind, B.kind)) <= 0) continue;
      let e = this.escrow.get(wire.id);
      // A wire id outliving a `rebind` would otherwise hand one pair's stake
      // to another. Refund and start the pot again under the new ends.
      if (e && (e.a !== A.id || e.b !== B.id)) {
        this.refundEscrow(e);
        e = undefined;
      }
      if (!e) {
        e = { a: A.id, b: B.id, paidA: 0, paidB: 0, x: A.x, y: A.y };
        this.escrow.set(wire.id, e);
      }
      // Where to put a refund if both ends are gone by the time it lapses.
      e.x = A.x;
      e.y = A.y;
      /*
       * A body clawing its way out of debt keeps what reaches it.
       *
       * Without this the pot swallows rescue energy on the frame it lands —
       * `pulseRequests` runs first, so `flowCharges` has just delivered it —
       * and the body can never climb back out. It would ask louder, be fed,
       * be emptied again, and starve with a full stake to its name. The
       * `recovering` latch is already exactly this question, held across the
       * zero crossing by `rescueNeed`, so paying is simply suspended until it
       * is back on its feet.
       */
      if (!A.recovering) e.paidA += payToward(A, rewriteShareOf(A) - e.paidA);
      if (!B.recovering) e.paidB += payToward(B, rewriteShareOf(B) - e.paidB);
    }
    this.sweepEscrows();
  }

  /** Drop the pots whose wire has gone, handing back what they held. */
  private sweepEscrows(): void {
    if (this.escrow.size === 0) return;
    for (const [id, e] of this.escrow) {
      if (this.graph.wires.has(id)) continue;
      this.refundEscrow(e);
      this.escrow.delete(id);
    }
  }

  /** Hand back every stake and forget the pots. */
  private refundEscrows(): void {
    if (this.escrow.size === 0) return;
    for (const e of this.escrow.values()) this.refundEscrow(e);
    this.escrow.clear();
  }

  private refundEscrow(e: RedexEscrow): void {
    this.repay(e.a, e.paidA, e.x, e.y);
    this.repay(e.b, e.paidB, e.x, e.y);
  }

  /**
   * Give a stake back to the body that put it up, or to the ground where the
   * redex stood if that body is gone.
   *
   * Overfill spills onto the ground rather than evaporating, for the reason
   * `tickUpkeep` gives at the only other site that can push a body past its
   * cap: a hole in the conservation the rest of the economy is careful about
   * is exactly how a mechanism quietly stops being worth anything.
   */
  private repay(id: number, amount: number, x: number, y: number): void {
    if (!(amount > 0)) return;
    const a = this.agents.get(id);
    if (!a) {
      this.energy.addAt(x, y, amount);
      return;
    }
    const room = a.energyCap - a.extra;
    const give = amount < room ? amount : room;
    if (give > 0) a.extra += give;
    if (amount - give > 0) this.energy.addAt(a.x, a.y, amount - give);
  }

  /**
   * Pay out the grazing the shader did at the end of the last frame.
   *
   * The plan was built then, from the gut (metered) or `extra` (unmetered)
   * as they stood at the end of that frame, and nothing between there and
   * here touches either — digestion, excretion and a corpse's spill all run
   * before the plan is built, and this is the first thing in `endFrame` that
   * writes to a gut. So the room each body was credited against is exactly
   * the room it would have had if the take had happened now: the pipelining
   * is exact, not an approximation anyone has to budget error for, and a gut
   * cannot be credited past `gutSize` times its cap.
   *
   * The id check is not currently reachable and is here because the invariant
   * it guards is a scheduling one. A plan spans a frame boundary, and if a
   * body ever comes to die or be born between the dispatch and this, the slot
   * it vacated would be paid its meal. Silent, and it would look like a
   * conservation leak somewhere else entirely.
   */
  private creditHarvest(): void {
    if (!this.harvestPending) return;
    this.harvestPending = false;
    const plan = this.harvestPlan;
    const got = fieldGpu.gotData;
    const store = this.agentStore;
    const EXTRA = store.extra;
    const GUT = store.gut;
    const CAP = store.energyCap;
    const IDS = store.id;
    const metered = plan.metered;
    for (let e = 0; e < plan.nEntries; e++) {
      const slot = plan.slots[e];
      if (IDS[slot] !== plan.ids[e]) continue;
      // One row per entry, one `got`: a body eats the ground and nothing else.
      // Metered it lands in the gut and `runDigestion` decides what becomes
      // tank and what becomes primer; unmetered it is money the moment it is
      // swallowed, which is what this has always done.
      const g = got[e * HARVEST_STRIDE + HARVEST_GOT];
      if (metered) {
        if (g > 0) {
          GUT[slot] += g;
          this.gutLive = true;
        }
        continue;
      }
      if (!(g > 0)) continue;
      const cap = CAP[slot];
      const next = EXTRA[slot] + g;
      EXTRA[slot] = next > cap ? cap : next;
    }
  }

  /**
   * Energy held in redex escrows, which is in neither a body nor the ground.
   *
   * Anything totalling the pond has to add this or it will read a pair saving
   * up for a commute as a leak.
   */
  escrowTotal(): number {
    let total = 0;
    for (const e of this.escrow.values()) total += e.paidA + e.paidB;
    return total;
  }

  /**
   * The net's demand for energy, as one field, and one hop of flow along it.
   *
   * Two things want energy and they compete on the same scale. A body that has
   * fallen into debt asks until it is back on its feet — up to its own
   * rescue fill, not merely up to zero, which is the difference between an
   * ambulance and a refill and the reason a surplus at one end of a net drains
   * toward a starving end at all. A stalled redex needs whatever each end is
   * short of a full extra.
   *
   * Both are magnitudes in the same units, so nothing has to be ranked by
   * policy: a body two hops away and 0.9 short outpulls a redex next door
   * that is 0.1 short, and a surplus at one end of a net finds a shortage at
   * the other end by following the gradient a wire at a time.
   */
  /**
   * A snapshot of who is alive and how they differ, for telling drift from
   * selection.
   *
   * Deliberately not called by anything per frame. It walks the whole roster
   * and allocates, and it exists for a bench or a console to ask occasionally
   * — the cost of answering "is this evolving or wandering?" belongs to
   * whoever asks the question, not to every frame that does not.
   *
   * `lines` is the count of distinct surviving founders, which is the number
   * that actually distinguishes the two. A population under selection loses
   * lines: some founders leave descendants and most do not. A population under
   * pure drift with a steady stream of fresh immigrants keeps roughly as many
   * lines as have arrived, however far the mean of any trait has wandered.
   *
   * **Give it a warm-up before believing any of it.** A preset drops its whole
   * population in at once as founders, so `bornMean` and `bornMax` start at
   * exactly zero by construction and every early reading is measuring the
   * seeding rather than the dynamics — `lines / bodies` starts at 1 for the
   * same reason. Under a minute of simulated time these say almost nothing
   * about a steady state, and it is very easy to read "has not got going yet"
   * as "does not work". Trait means and standard deviations are only slightly
   * better off: they begin as the spread `seedChem` and the sliders put there.
   *
   * What *is* fair early is a comparison between two runs at the same age and
   * the same seed, which is what the numbers on `declutter` are. An absolute
   * claim about whether a pond sustains evolution needs it run out properly.
   */
  census(): {
    bodies: number;
    lines: number;
    bornMax: number;
    bornMean: number;
    /**
     * Commutes per latch, cumulative — the tripwire on whether polluting the
     * two principal channels is cheap.
     *
     * Latching is proximity plus an arc
     * test and scent never gates it, and net reduction is confluent, so a net
     * is a fixed budget of evolutionary events spent down: scent cannot touch
     * that budget, only how nets acquire new structure. Well above 1 means
     * nets are doing real internal computation and chemistry on species 0 and
     * 1 costs little. At or below 1 every commute is roughly paid for by a
     * fresh latch and encounter rate is load-bearing after all.
     *
     * **Cumulative, and therefore unreadable on its own.** `tally` counts from
     * the last `clear`, and a soup's opening is a latch storm — measured over
     * an 18-trial sweep, 6,775 latches against 1,095 commutes in the first
     * thirty seconds. That start drags the cumulative ratio below 1 for
     * minutes whatever the pond then does. Difference two samples: over the
     * same sweep the windowed ratio ran 0.16, 0.52, 1.02, 1.43, 1.67, 2.06
     * across six thirty-second windows and was still climbing. Same shape, and
     * the same trap, as `rewriteMix`'s U — see its note, which says so at
     * length about the number next door.
     *
     * Null before anything has latched, because a ratio over nothing is not
     * zero — it is undecided, and reporting zero would read as the bad case.
     */
    commutesPerLatch: number | null;
    trait: Record<string, { mean: number; sd: number }>;
  } {
    const lines = new Set<number>();
    let bornSum = 0;
    let bornMax = 0;
    const sums: Record<string, number> = {};
    const sqs: Record<string, number> = {};
    const keys = [...TRAIT_KEYS] as string[];
    for (const k of keys) {
      sums[k] = 0;
      sqs[k] = 0;
    }
    let n = 0;
    for (const a of this.agents.values()) {
      n++;
      lines.add(a.lineage);
      bornSum += a.born;
      if (a.born > bornMax) bornMax = a.born;
      for (const k of keys) {
        const v = (a as unknown as Record<string, number>)[k];
        sums[k] += v;
        sqs[k] += v * v;
      }
    }
    const trait: Record<string, { mean: number; sd: number }> = {};
    for (const k of keys) {
      const mean = n > 0 ? sums[k] / n : 0;
      const varr = n > 0 ? Math.max(0, sqs[k] / n - mean * mean) : 0;
      trait[k] = { mean, sd: Math.sqrt(varr) };
    }
    return {
      bodies: n,
      lines: lines.size,
      bornMax,
      bornMean: n > 0 ? bornSum / n : 0,
      commutesPerLatch: this.tally.latches > 0 ? this.tally.commutes / this.tally.latches : null,
      trait,
    };
  }

  /**
   * How selective latching currently is, measured against chance.
   *
   * The three rewrite rules do very different things to a population: a
   * commute makes four bodies out of two, an annihilation takes two away, an
   * erase takes what it meets. So the *mix* of rules is what decides whether a
   * pond grows or consumes itself, and the mix is decided by which kinds end
   * up principal-to-principal.
   *
   * The number that matters is not the commute share on its own but the share
   * against what proximity alone would give. If bodies latch with whoever is
   * adjacent, the mix is the one you would get by drawing two bodies at random
   * from the current kind census — that is `chance` here, computed by running
   * `detectRule` over the nine kind pairs weighted by the census and bucketed
   * exactly as `tally` buckets them. If bodies steer, face and choose before
   * they latch, the share rises above it.
   *
   * Measured at thirty thousand, this traces the pond's opening as a clean U:
   * 0.224 during the initial latch storm (chance is 0.22 — the dish is so
   * crowded that latching is pure proximity), down to 0.103 as the net works
   * through the destructive pairs it made, then back up through the chance
   * line to 0.251 as the cull opens space and sensing starts to decide who
   * meets whom. `selectivity` is that distance from chance, and it going
   * positive is the moment the pond stops being a soup.
   *
   * Counts are cumulative since the last `clear`, as the rest of `tally` is;
   * difference two samples to get a window.
   */
  rewriteMix(): {
    commutes: number;
    erases: number;
    annihilations: number;
    rewrites: number;
    share: number | null;
    chance: number;
    selectivity: number | null;
  } {
    const KC = this.agentStore.kindCode;
    const count = [0, 0, 0];
    let n = 0;
    for (const a of this.agents.values()) {
      count[KC[a.slot]]++;
      n++;
    }
    let chance = 0;
    if (n > 0) {
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (detectRule(CODE_KIND[i], CODE_KIND[j]) !== 'commute') continue;
          chance += (count[i] / n) * (count[j] / n);
        }
      }
    }
    const t = this.tally;
    const rewrites = t.commutes + t.erases + t.annihilations;
    const share = rewrites > 0 ? t.commutes / rewrites : null;
    return {
      commutes: t.commutes,
      erases: t.erases,
      annihilations: t.annihilations,
      rewrites,
      share,
      chance,
      selectivity: share === null ? null : share - chance,
    };
  }

  private pulseRequests(params: Params): void {
    /*
     * Three claims are made on a body here, and the largest wins. That used to
     * be collected in a `Map` from agent id to the running maximum, cleared
     * and refilled every frame, and then drained with a roster lookup per
     * entry — at fifty thousand bodies, some fifty thousand map writes, a map
     * iteration and fifty thousand `agents.get` a frame.
     *
     * None of it was needed. `seedRequest` is itself a maximum into a field
     * that had just been zeroed, so taking the largest of three claims and
     * seeding once is the same number as seeding three times in any order. The
     * map is gone and each claim writes where it is made.
     *
     * The zeroing is folded in for the same reason: it was its own walk of
     * every body to write a zero, and this walks every body immediately after
     * and knows what the value should be. Locked bodies make no claim, so they
     * get the zero that walk would have given them.
     */
    const list = this.forceList();
    const REQUEST = this.agentStore.request;
    const LOCKED = this.agentStore.locked;
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      const slot = a.slot;
      if (LOCKED[slot]) {
        REQUEST[slot] = 0;
        continue;
      }
      // Side-effecting: `rescueNeed` is what latches and clears `recovering`.
      const h = rescueNeed(a);
      REQUEST[slot] = h > 0 ? h : 0;
    }

    if (params.rewriteDuration > 0) {
      const ready = this.readyRedexes;
      for (let k = 0; k < ready.length; k++) {
        const wire = ready[k];
        const A = this.agents.get(wire.a.id);
        const B = this.agents.get(wire.b.id);
        if (!A || !B) continue;
        if (rewriteCost(detectRule(A.kind, B.kind)) <= 0) continue;
        // Per end against its own stake, not the old pair-wide share count:
        // an end that has already banked its half should stop asking while its
        // partner keeps on, which is the whole point of accumulating.
        const e = this.escrow.get(wire.id);
        const paidA = e ? e.paidA : 0;
        const paidB = e ? e.paidB : 0;
        if (stakeMet(A, paidA) && stakeMet(B, paidB)) continue;
        seedRequest(A, redexNeed(A, paidA));
        seedRequest(B, redexNeed(B, paidB));
      }
    }
    const adj = this.wireAdjacency();
    Sim.phase('pulse:seed');
    /*
     * The need field, to its fixpoint, every frame.
     *
     * Queue-driven: one visit per body whose value actually improved, which is
     * `O(n + wires)`. `params.requestReach` used to offer the alternative —
     * advance a fixed number of one-hop steps and stop, so that demand took
     * time to cross a net and had a front to travel on — and that is the only
     * thing this cannot do: the field it leaves has no history, so nothing
     * propagates. The dial shipped at 0, so no pond ever took that trade, and
     * iterating the one-hop step to the same answer was quadratic in the
     * roster — the frame itself at a few hundred founders.
     */
    relaxRequestsFast(list, this.agentStore, adj);
    Sim.phase('pulse:spread');
    // No `quantum` here: each sender uses its own, seeded from the parameter
    // at birth exactly as `transportRecoil` is, so a net's rhythm can be a
    // property of the net rather than of the dish.
    flowChargesFast(
      list,
      this.agentStore,
      adj,
      (from, to, amount) => {
        this.recoil(from, to, amount);
        // What the wire between them carried, for `wireTug`. A body has three
        // ports and a port holds one wire, so the wire is three lookups away.
        const w = this.graph.wireBetween(from.id, to.id);
        if (w) w.carried += amount;
      },
      { grid: this.energy },
    );
    Sim.phase('pulse:flow');
    // Before the state pass, and over the same adjacency it is about to walk.
    this.relayDepth(list, adj, params);
    Sim.phase('pulse:depth');
    this.updateState(list, adj, params);
    Sim.phase('state');
  }

  /* ---- the claim relay: where a net's nose is, and how deep each body sits
     behind it ------------------------------------------------------------ */

  /** A body with no reading of its own cannot enter the competition. */
  private static readonly BLIND_BIG = 1e9;

  /** Head claim, by slot: the source's reading, the distance to it, its id. */
  private hVal = new Float64Array(0);
  private hDst = new Float64Array(0);
  private hSrc = new Int32Array(0);
  private hVo = new Float64Array(0);
  private hDo = new Float64Array(0);
  private hSo = new Int32Array(0);
  /** The same machinery on negated readings: the tail. */
  private tVal = new Float64Array(0);
  private tDst = new Float64Array(0);
  private tSrc = new Int32Array(0);
  private tVo = new Float64Array(0);
  private tDo = new Float64Array(0);
  private tSo = new Int32Array(0);

  private growRelay(cap: number): void {
    if (this.hVal.length >= cap) return;
    const f = (old: Float64Array): Float64Array<ArrayBuffer> => {
      const next = new Float64Array(cap);
      next.set(old);
      return next;
    };
    const i = (old: Int32Array): Int32Array<ArrayBuffer> => {
      const next = new Int32Array(cap);
      next.set(old);
      return next;
    };
    this.hVal = f(this.hVal); this.hDst = f(this.hDst); this.hSrc = i(this.hSrc);
    this.hVo = f(this.hVo); this.hDo = f(this.hDo); this.hSo = i(this.hSo);
    this.tVal = f(this.tVal); this.tDst = f(this.tDst); this.tSrc = i(this.tSrc);
    this.tVo = f(this.tVo); this.tDo = f(this.tDo); this.tSo = i(this.tSo);
    this.relayRange = f(this.relayRange); this.relayExtent = f(this.relayExtent);
  }

  /**
   * What the relay carries, and why it is a logarithm.
   *
   * The body's own taste score at its own position — `store.trail`, already
   * computed once a body a frame by the steer pass and correct on both field
   * paths. **Its own**, not a channel: a lineage that weights `aux` noses
   * toward food and one that weights `conP` noses toward other bodies'
   * terminals, so what the nose points at is a gene rather than a decision
   * taken here. That is the whole of "port scent competes with the ground for
   * navigation" — one nose, one field, and the taste vector arbitrates.
   *
   * Negated, so that *less* is *better* and the min-plus machinery below is
   * the inchworm's unchanged.
   *
   * And a logarithm, which is the one place the bench's algebra does not
   * transfer. Its sensor is `|p - food|`, which is 1-Lipschitz on the graph
   * metric — `s_i <= s_j + l_ij` on every edge — and that is what makes the
   * relay exact rather than tuned: at gamma >= 1 it is provably the identity,
   * and since `s_i - s_j = l * cos(alpha)`, **gamma is a cosine threshold on
   * edge alignment**. A diffused scent is not eikonal. Its gradient is steep
   * beside a patch and flat away from one, so one global gamma would mean
   * seventy degrees in one part of the dish and "everything is a head" in
   * another. Diffusion against decay gives roughly `exp(-r/lambda)`, so the
   * log of it has a near-constant slope `1/lambda` over a wide range, and
   * gamma gets its geometry back.
   */
  private relayRead(slot: number): number {
    if (this.agentStore.wires[slot] > 1) return Sim.BLIND_BIG;
    const t = this.agentStore.trail[slot];
    return -Math.log1p(t > 0 ? t : 0);
  }

  /**
   * Transmission cost over the total distance back to a claim's source.
   *
   * `p = 1` is the plain discount. Above it, short relays get cheaper and long
   * ones dearer than linear, which is what isolates a distant extremity into
   * its own basin without flattening the local gradient. The bench measured
   * that this is what separates a lobe from a wiggle: on its U-shaped body,
   * `p = 1` gave a bearing-dependent scatter of one to four heads and
   * `p = 2` a stable two at nearly every bearing.
   */
  private relayCost(d: number, slot: number, params: Params): number {
    /*
     * In units of the net's own reach, against the net's own reading range —
     * and neither of those normalisations is optional.
     *
     * `depthCost` is a discount per unit length weighed against a reading, so
     * it only means anything if the two are in comparable units. On the bench
     * they are by construction: its length unit *is* one edge rest length and
     * its reading is a distance in the same unit, so `s_i - s_j = l*cos(alpha)`
     * and gamma is a cosine threshold. Here a wire is forty to sixty pixels
     * and the reading is the log of a scent whose head-to-tail spread across a
     * net might be a tenth. Left raw, a claim crossing eight bodies cost about
     * 1500 against readings that differed by 0.18, so every body claimed
     * itself, every body was its own head, and the coordinate came out
     * uniformly zero — which reads as the relay being broken rather than as
     * two unit errors.
     *
     * Both scales come out of the relay itself, one frame old: `range` is what
     * this body's own head and tail claims differ by, and `extent` is how far
     * apart they are. So the cost of crossing the *whole* net is `depthCost`
     * times the range the net actually spans, whatever the pond's units are
     * and whatever the gradient happens to be that second. At 0.35 a claim
     * from the nose gives up about a third of the range by the time it reaches
     * the tail, which still beats the tail's own reading — so the tail is not
     * its own head — while a body sitting on a local bump bigger than the
     * discount it has accumulated *is*. That is the bench's rule, restated in
     * the only units this pond has.
     *
     * Both are zero on a body's first frame, which makes the cost zero, which
     * lets the claims propagate freely and fills in the scales for the next
     * one. It bootstraps rather than needing a seed.
     *
     * **`extent` is clamped by `depthReach`, and that clamp is what kills a
     * stale cycle.** The bench's claims self-correct because distance
     * accumulates around a loop until the cost of carrying one exceeds the
     * holder's own reading — the cost has to *grow* without bound for that to
     * work. Normalising by the relay's own reach feeds the output back into
     * its own yardstick, so a runaway distance normalises itself away and the
     * cost never catches it: measured on an eight-body chain, the tail claim
     * ping-ponged between the last two bodies and carried 3500 px of
     * accumulated distance across a net 479 px long, with every body reading a
     * depth of nearly zero. Past the reach the yardstick stops growing, the
     * cost resumes climbing, and the loop dies the way it is supposed to.
     */
    const extent = this.relayExtent[slot];
    if (!(extent > 0)) return 0;
    const u = d / extent;
    const g = params.depthCost * this.relayRange[slot];
    return params.depthConvex === 1 ? g * u : g * Math.pow(u, params.depthConvex);
  }

  /** What this body's head and tail claims differ by, and how far apart. */
  private relayRange = new Float64Array(0);
  private relayExtent = new Float64Array(0);

  /**
   * One hop of the claim relay, over the whole roster.
   *
   * Each body keeps the cheapest of its own reading (distance 0, itself the
   * source) and each wired neighbour's claim carried one more hop. Run once
   * on the readings and once on their negation, and what comes out is where
   * the net's nose is and where its tail is.
   *
   * **A head is a body that claims itself** — exact, and not a threshold. And
   * after `interiorTaste`, a body with more than one wire has no reading to
   * enter, so every head is a leaf by construction rather than by tuning the
   * cost curve. That is the whole reason the interior was blinded first.
   *
   * **It cannot latch.** The claim carries the *source's live reading* rather
   * than an already-combined value, so every source re-asserts itself each
   * frame and a claim is only ever as stale as its hop count. Stale cycles die
   * on their own, because distance accumulates around a loop until the cost
   * exceeds the holder's own reading. A plain scalar min-plus relay does latch
   * — min propagation is monotone decreasing — and needs an explicit leak;
   * this form does not, and that is why the source travels with the value.
   *
   * The source is an **id**, never an index. Slots are reused the moment a
   * body dies and ids are not, and the only two things done with a source are
   * equality tests: "is this me" and "is this the one I was already holding".
   *
   * Double-buffered, for the reason `updateState` is: reading a neighbour's
   * value after it has been updated this frame builds a sequential algorithm
   * whose answer depends on the roster's order.
   */
  private relayPass(
    list: Agent[],
    adj: WireAdjacency,
    params: Params,
    sgn: number,
    val: Float64Array,
    dst: Float64Array,
    src: Int32Array,
    vo: Float64Array,
    dof: Float64Array,
    so: Int32Array,
  ): void {
    const n = list.length;
    const X = this.agentStore.x;
    const Y = this.agentStore.y;
    const mu = params.depthHold;
    const reach = params.depthReach * Math.max(1, params.wireMinRest);
    for (let i = 0; i < n; i++) {
      const s = list[i].slot;
      vo[s] = val[s];
      dof[s] = dst[s];
      so[s] = src[s];
    }
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const s = a.slot;
      const read = this.relayRead(s);
      const own = read >= Sim.BLIND_BIG ? Sim.BLIND_BIG : sgn * read;
      const holding = so[s];
      let bV = own;
      let bD = 0;
      let bS = a.id;
      // The incumbent keeps a margin, so a basin does not flicker between two
      // sources that read within noise of each other.
      let best = own - (holding === a.id ? mu : 0);
      const lo = adj.off[i];
      const hi = adj.off[i + 1];
      for (let k = lo; k < hi; k++) {
        const j = adj.nei[k];
        const t = list[j].slot;
        const gap = wrapDeltaVec(X[s], Y[s], X[t], Y[t], this.w, this.h);
        const D = dof[t] + Math.hypot(gap.x, gap.y);
        // A claim carries this far and no further, which is the other half of
        // what stops a stale one going round a loop forever.
        if (D > reach) continue;
        let sc = vo[t] + this.relayCost(D, s, params);
        if (so[t] === holding) sc -= mu;
        if (sc < best) {
          best = sc;
          bV = vo[t];
          bD = D;
          bS = so[t];
        }
      }
      // A claim only stands while it is cheaper than speaking for yourself.
      if (bV + this.relayCost(bD, s, params) > own) {
        bV = own;
        bD = 0;
        bS = a.id;
      }
      val[s] = bV;
      dst[s] = bD;
      src[s] = bS;
    }
  }

  /**
   * The nose-to-tail coordinate, once a frame.
   *
   * `rho` is the transmission cost back to this body's head and `rhoHat`
   * normalises it against the cost onward to its tail, so it lands in [0, 1]
   * out of relayed scalars alone — **no body needs to know the size of the net
   * it is in**, which is what lets one profile over it mean the same thing on
   * a net of six and a net of three hundred.
   *
   * The blind-mode formula, which is the one this pond is in: with the
   * interior dark, depth is the distance back to a lit end rather than a
   * difference of readings.
   */
  private relayDepth(list: Agent[], adj: WireAdjacency, params: Params): void {
    if (!(params.depthCost > 0)) return;
    this.growRelay(this.agentStore.capacity);
    // Last frame's answer is this frame's yardstick. See `relayCost`.
    const reach = params.depthReach * Math.max(1, params.wireMinRest);
    for (let i = 0; i < list.length; i++) {
      const s = list[i].slot;
      const range = -this.tVal[s] - this.hVal[s];
      this.relayRange[s] = range > 0 ? range : 0;
      const extent = this.hDst[s] + this.tDst[s];
      this.relayExtent[s] = extent > 0 ? (extent < reach ? extent : reach) : 0;
    }
    this.relayPass(list, adj, params, 1, this.hVal, this.hDst, this.hSrc, this.hVo, this.hDo, this.hSo);
    this.relayPass(list, adj, params, -1, this.tVal, this.tDst, this.tSrc, this.tVo, this.tDo, this.tSo);
    const DEPTH = this.agentStore.depth;
    const HEAD = this.agentStore.depthHead;
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      const s = a.slot;
      const rho = this.relayCost(this.hDst[s], s, params);
      const tail = this.relayCost(this.tDst[s], s, params);
      const span = rho + tail;
      DEPTH[s] = span > 1e-9 ? rho / span : 0;
      HEAD[s] = this.hSrc[s] === a.id && this.agentStore.wires[s] <= 1 ? 1 : 0;
    }
  }

  /**
   * One round of message passing: every body's `h` from its inputs, its own
   * last value, and the mean of its wired neighbours'.
   *
   *     h <- phi( Wx.x + Wh.h + Wn.mean(h_j) + b )
   *
   * One round per frame, not a relaxation to convergence. The recurrence is
   * across frames rather than within one, which is both cheaper and more
   * expressive than iterating: state persists, so a body can integrate over
   * time rather than recomputing itself from scratch. Information travels one
   * wire-hop a frame, sixty hops a second, which crosses any net worth having.
   *
   * *Mean* of the neighbours, not sum. A sum scales with degree, so the same
   * genome saturates `phi` at a hub and barely moves a leaf — behaviour
   * differing by position for a reason that is not about position. `BOUND`
   * already carries degree, bounded and on purpose.
   *
   * Read *after* `spreadRequests` and `flowCharges`, so `DEMAND` and `FULL` are
   * this frame's. `sense` and `trail` are last frame's, written by the steer
   * pass — a frame of latency in smell that steering has always had.
   *
   * `hPrev` because every body has to see the same generation of its
   * neighbours. Updating in place would make the answer depend on iteration
   * order, and the order is the roster, which changes whenever anything is
   * born.
   */
  private hPrev = new Float64Array(0);
  private readonly stateInput = new Float64Array(IN_DIMS);
  private readonly stateMean = new Float64Array(STATE_DIMS);
  /** `chem + plastic` over the state matrices, for a body that has learned. */
  private readonly effWeights = new Float32Array(PLASTIC_LEN);
  /** This body's fifteen head displacements, redrawn every frame. */
  private readonly exploreDraw = new Float64Array(HEAD_ROWS);
  /**
   * Whether last frame drew anything, so that turning exploration off clears
   * the scratch once instead of every body zeroing it every frame.
   */
  private exploreWasOn = false;
  /**
   * Frames since the pond began, wrapped at `FRAME_WRAP`. The only thing that
   * reads it is `exploreAt`, which needs a counter both paths agree on; the
   * device gets it in the genome uniform.
   */
  private frames = 0;
  /** `phi'` per state dim, and the inputs each learned weight multiplies. */
  private readonly learnPost = new Float64Array(STATE_DIMS);
  private readonly learnPre = new Float64Array(IN_DIMS + 2 * STATE_DIMS);

  private updateState(list: Agent[], adj: WireAdjacency, params: Params): void {
    const n = list.length;
    if (n === 0) return;
    if (this.genomeOnGpu) {
      // The shader does all of this, learning included — `genome.wgsl` carries
      // the learned block against its own resident row, and this only unpacks
      // what came back. The learning row stays there: `syncLearn` fetches the
      // handful of parents a rewrite needs, and `syncLearningToHost` fetches
      // the rest when something means to measure it.
      this.unpackGenome();
      return;
    }
    const store = this.agentStore;
    const H = store.hAll;
    const SENSE = store.senseAll;
    /*
     * Straight into `chemAll` with a slot offset, rather than through
     * `Agent.chem`.
     *
     * That accessor hands back a `subarray` view per body, and this reads
     * eighty weights out of it per body per frame. Through the view the pass
     * cost 16.5 ms at 20k bodies — 825 ns a body for eighty multiplies, which
     * is an order of magnitude off what the arithmetic is worth. It is the
     * same finding the store conversions in this file's history keep making,
     * and the same fix.
     */
    const CHEM = store.chemAll;
    const S = STATE_DIMS;
    if (this.hPrev.length < n * S) this.hPrev = new Float64Array(n * S);
    const prev = this.hPrev;
    const slotOf = this.slotBuf.length >= n ? this.slotBuf : (this.slotBuf = new Int32Array(n));
    for (let i = 0; i < n; i++) {
      const s = list[i].slot;
      slotOf[i] = s;
      const po = i * S;
      const ho = s * S;
      for (let d = 0; d < S; d++) prev[po + d] = H[ho + d];
    }

    const { off, nei } = adj;
    const EXTRA = store.extra;
    const CAP = store.energyCap;
    const REQUEST = store.request;
    const BOUND_OF = store.bound;
    const X = store.x;
    const Y = store.y;
    const gScale = this.groundScale;
    const sScale = 1 / params.senseScale;
    const READS = store.readsField;
    const x = this.stateInput;
    const mean = this.stateMean;
    const EMITS = store.emitAll;
    const TASTES = store.tasteAll;
    const CRUISE = store.cruise;
    const TURN = store.turn;
    const FA = store.flockAlign;
    const FS = store.flockSep;
    const TT = store.transportThrust;
    const TR = store.transportRecoil;
    const GA = store.gaitAnchor;
    const PLASTIC = store.plasticAll;
    const TRACE = store.traceAll;
    const CRITIC = store.criticAll;
    const PREV_V = store.prevValue;
    const PREV_FULL = store.prevFull;
    const PLASTIC_ON = store.plasticOn;
    const post = this.learnPost;
    const pre = this.learnPre;
    const learn = params.learnRate > 0;
    const etaM = params.learnRate;
    const etaC = params.learnCritic;
    const lam = params.learnTrace;
    const discount = params.learnDiscount;
    const mix = params.learnReward;
    /*
     * Exploration only while there is a learner to use it. Noise a body cannot
     * learn from is noise, and this pond has `swimNoise` for that already.
     */
    const sigma = learn ? params.learnExplore : 0;
    const sigmaWasOn = this.exploreWasOn;
    this.exploreWasOn = sigma > 0;
    // The hold, not the frame — the two paths share one key. See `exploreKey`.
    const frame = exploreKey(this.frames, params.learnHold);
    const invDt = this.frameDt > 0 ? 1 / this.frameDt : 0;
    const MAXW = CHEM_TASTE_MAX;
    for (let i = 0; i < n; i++) {
      const slot = slotOf[i];
      const g = slot * CHEM_LEN;

      /*
       * Sampling is skipped unless this genome reads the field **or anything
       * is learning**. A bilinear sample is four scattered reads into a
       * sixteen-megabyte array that will not be in cache, once a body, for a
       * value nearly every genome multiplies by zero — `Wx`'s sense columns
       * seed to zero, because the one seeded pathway runs through `IN_DEMAND`.
       * The inputs are zeroed rather than left reading a stale `SENSE`, so
       * behaviour never depends on when a body last happened to sample.
       *
       * `|| learn` because the gate was only ever an optimisation and stopped
       * being invisible when learning arrived. Two things go wrong without it.
       * The eligibility trace is `phi'(v) * pre`, and `pre` is this input — a
       * weight of zero zeroes the forward pass but not the *update*, so a
       * gated body learns a different sense weight from an ungated one holding
       * the same genome. And the gate self-locks: `READS` is raised when a
       * sense weight becomes non-zero, but with the input pinned at zero no
       * sense weight can ever move, so the line below claiming a body can
       * learn its way into seeing was unreachable. `genome.wgsl` never had the
       * gate — it argued a zero weight made the paths agree, which is true of
       * the forward pass and false of the trace — so this is also what makes
       * the two paths learn the same thing, which is how it was found.
       */
      if (READS[slot] || learn) {
        const so = slot * 4;
        /*
         * On the GPU path `gpuFieldStep` filled these at the end of last
         * frame, already scaled, straight out of the buffer that holds the
         * live field; `fields.data` is a stale copy there and sampling it
         * reads garbage. Here the sample is taken and scaled at the write,
         * so the store holds the same thing on both paths.
         *
         * Both scales bring an input onto the range the other four already
         * occupy. The ground is a quantity per cell, so a full cell reads one;
         * the signal channels are accumulated deposits, so a strong local
         * reading reads about one. Without the second, a sense gene had seven
         * times the mutation leverage of every other input gene and `phi` was
         * pinned across all but about 7% of its legal range.
         */
        if (!this.fieldOnGpu) {
          this.fields.sampleAll(X[slot], Y[slot], SENSE, so);
          SENSE[so] *= sScale;
          SENSE[so + 1] *= sScale;
          SENSE[so + 3] *= sScale;
          SENSE[so + CH.energy] *= gScale;
        }
        const g = this.tasteGain(slot);
        x[IN_SENSE] = SENSE[so] * g;
        x[IN_SENSE + 1] = SENSE[so + 1] * g;
        x[IN_SENSE + 2] = SENSE[so + 2] * g;
        x[IN_SENSE + 3] = SENSE[so + 3] * g;
      } else {
        x[IN_SENSE] = 0;
        x[IN_SENSE + 1] = 0;
        x[IN_SENSE + 2] = 0;
        x[IN_SENSE + 3] = 0;
      }
      const cap = CAP[slot];
      const full = cap > 0 ? EXTRA[slot] / cap : 0;
      x[IN_FULL] = full <= 0 ? 0 : full >= 1 ? 1 : full;
      x[IN_BOUND] = BOUND_OF[slot];
      const r = REQUEST[slot];
      x[IN_DEMAND] = r <= 0 ? 0 : r >= 1 ? 1 : r;

      /*
       * The neighbour mean is a property of the body, not of the dimension
       * being computed, so it is gathered once rather than inside the `d`
       * loop. The obvious way round cost four times as much, because each of
       * the four output dimensions re-walked the whole adjacency.
       */
      const lo = off[i];
      const hi = off[i + 1];
      const deg = hi - lo;
      if (deg > 0) {
        for (let k = 0; k < S; k++) mean[k] = 0;
        for (let e = lo; e < hi; e++) {
          const base = nei[e] * S;
          for (let k = 0; k < S; k++) mean[k] += prev[base + k];
        }
        for (let k = 0; k < S; k++) mean[k] /= deg;
      }

      /*
       * Unrolled over the four state dimensions and the fifteen weights each
       * reads. `STATE_DIMS` and `IN_DIMS` are compile-time constants, and the
       * loop overhead around sixty multiply-adds is most of what this pass
       * costs — the same finding, and the same fix, as the channel loop in
       * `Fields.diffuse`.
       *
       * A neighbour mean of zero when there are no neighbours rather than a
       * branch inside the arithmetic: `Wn` times nothing is nothing, and the
       * branch was per dimension.
       */
      const po = i * S;
      const ho = slot * S;
      const p0 = prev[po];
      const p1 = prev[po + 1];
      const p2 = prev[po + 2];
      const p3 = prev[po + 3];
      const m0 = deg > 0 ? mean[0] : 0;
      const m1 = deg > 0 ? mean[1] : 0;
      const m2 = deg > 0 ? mean[2] : 0;
      const m3 = deg > 0 ? mean[3] : 0;
      const x0 = x[0];
      const x1 = x[1];
      const x2 = x[2];
      const x3 = x[3];
      const x4 = x[4];
      const x5 = x[5];
      const x6 = x[6];
      /*
       * The state matrices as this body actually has them: its genome plus
       * whatever it has learned since it was born. `plasticOn` is monotone —
       * learned weights never decay, so a body that has learned anything has
       * learned it for good — which keeps the sum off the path of a pond
       * where nothing has.
       */
      let W = CHEM;
      let wo = g + W_IN;
      if (PLASTIC_ON[slot]) {
        const plo = slot * PLASTIC_LEN;
        const eff = this.effWeights;
        for (let k = 0; k < PLASTIC_LEN; k++) eff[k] = CHEM[wo + k] + PLASTIC[plo + k];
        W = eff;
        wo = 0;
      }
      const gi = wo;
      const gs = wo + W_SELF - W_IN;
      const gn = wo + W_NET - W_IN;
      const gb = wo + B_STATE - W_IN;
      let v0 = 0;
      let v1 = 0;
      let v2 = 0;
      let v3 = 0;
      {
        const wi = gi + 0;
        const ws = gs + 0;
        const wn = gn + 0;
        const v =
          W[gb + 0] +
          W[wi] * x0 + W[wi + 1] * x1 + W[wi + 2] * x2 + W[wi + 3] * x3 +
          W[wi + 4] * x4 + W[wi + 5] * x5 + W[wi + 6] * x6 +
          W[ws] * p0 + W[ws + 1] * p1 + W[ws + 2] * p2 + W[ws + 3] * p3 +
          W[wn] * m0 + W[wn + 1] * m1 + W[wn + 2] * m2 + W[wn + 3] * m3;
        v0 = v;
        H[ho + 0] = v / (1 + (v < 0 ? -v : v));
      }
      {
        const wi = gi + 7;
        const ws = gs + 4;
        const wn = gn + 4;
        const v =
          W[gb + 1] +
          W[wi] * x0 + W[wi + 1] * x1 + W[wi + 2] * x2 + W[wi + 3] * x3 +
          W[wi + 4] * x4 + W[wi + 5] * x5 + W[wi + 6] * x6 +
          W[ws] * p0 + W[ws + 1] * p1 + W[ws + 2] * p2 + W[ws + 3] * p3 +
          W[wn] * m0 + W[wn + 1] * m1 + W[wn + 2] * m2 + W[wn + 3] * m3;
        v1 = v;
        H[ho + 1] = v / (1 + (v < 0 ? -v : v));
      }
      {
        const wi = gi + 14;
        const ws = gs + 8;
        const wn = gn + 8;
        const v =
          W[gb + 2] +
          W[wi] * x0 + W[wi + 1] * x1 + W[wi + 2] * x2 + W[wi + 3] * x3 +
          W[wi + 4] * x4 + W[wi + 5] * x5 + W[wi + 6] * x6 +
          W[ws] * p0 + W[ws + 1] * p1 + W[ws + 2] * p2 + W[ws + 3] * p3 +
          W[wn] * m0 + W[wn + 1] * m1 + W[wn + 2] * m2 + W[wn + 3] * m3;
        v2 = v;
        H[ho + 2] = v / (1 + (v < 0 ? -v : v));
      }
      {
        const wi = gi + 21;
        const ws = gs + 12;
        const wn = gn + 12;
        const v =
          W[gb + 3] +
          W[wi] * x0 + W[wi + 1] * x1 + W[wi + 2] * x2 + W[wi + 3] * x3 +
          W[wi + 4] * x4 + W[wi + 5] * x5 + W[wi + 6] * x6 +
          W[ws] * p0 + W[ws + 1] * p1 + W[ws + 2] * p2 + W[ws + 3] * p3 +
          W[wn] * m0 + W[wn + 1] * m1 + W[wn + 2] * m2 + W[wn + 3] * m3;
        v3 = v;
        H[ho + 3] = v / (1 + (v < 0 ? -v : v));
      }

      /*
       * The output heads, off this frame's state.
       *
       * Written into the same store fields the traits used to live in, so
       * every consumer downstream — the native flock packing, the JS pair
       * force, `recoil`, `state-hash` — reads what it always read and does not
       * need to know these stopped being constants. What changed is that they
       * are now a phenotype computed from `h` rather than a number a body
       * carries for life, so a lineage can shoal while fed and scatter while
       * starving instead of having to pick one.
       *
       * Clamped to the ranges the heritable versions were bred inside, because
       * those bounds were about what the *forces* survive, not about what the
       * genome was allowed to say. Alignment past its ceiling is negative
       * damping; separation past its own has no equilibrium to settle at.
       */
      /*
       * Emit and taste materialised here, once, instead of by each consumer.
       *
       * Both are pure functions of `h`, which was computed three lines up, and
       * both were being rebuilt from the genome four times a body a frame —
       * `effEmit` in the scent pass and `tasteOf` in steer, 13.8 ms between
       * them at 20k. Emit is normalised on the way in, which is the only place
       * all four channels are known at once and therefore the only place the
       * unit budget can actually be enforced.
       *
       * One behaviour change worth naming: `taste` is now this frame's rather
       * than last frame's. `updateState` runs in `endFrame`, so the scent pass
       * already saw this frame's `h` while `steer` and `flock` ran in the next
       * frame's `beginFrame` and saw the previous. That asymmetry between what
       * a body says and what it listens for is gone, which is almost certainly
       * an improvement and is definitely a change — it moves `state-hash`.
       */
      /*
       * This frame's exploration, one draw per head output, in the genome's
       * own row order: `E` 0-3, `T` 4-7, `F` align 8 sep 9, `P` thrust 10
       * recoil 11, `L` cruise 12 turn 13, `G` anchor 14.
       *
       * Drawn once and used twice — displacing the output below, then standing
       * as the post-synaptic factor in that row's eligibility. The two *must*
       * be the same number or the learner credits a displacement that never
       * happened, which is the one way node perturbation fails silently: it
       * still moves weights, just in a direction uncorrelated with anything.
       */
      const XI = this.exploreDraw;
      if (sigma > 0) {
        for (let r = 0; r < HEAD_ROWS; r++) XI[r] = sigma * exploreAt(slot, frame, r);
      } else if (sigmaWasOn) {
        XI.fill(0);
      }

      /*
       * The heads read `W`/`wb`, not the genome: with learning on, `W` is the
       * effective array — genome plus what this body has learned — and the
       * heads are inside the learned block now. `wb` is `wo` shifted so a
       * genome offset indexes it directly, which is the same trick the state
       * matrices above use and the reason the block starts at `W_IN`.
       *
       * `emit` and `taste` take their bases from the genome and their matrices
       * from `W`, because those two bases are the only part of a head that
       * does not learn; see `PLASTIC_LEN`.
       */
      const wb = wo - W_IN;
      emitVector(W, wb, CHEM, g, H, ho, XI, 0, EMITS, slot * 4);
      tasteVector(W, wb, CHEM, g, H, ho, XI, 4, TASTES, slot * 4);

      CRUISE[slot] = clamp((headAt(W, wb, L_OUT, L_BASE, 0, H, ho, S) + XI[12]) * HEAD_SCALE.cruise, HEAD_RANGE.cruise.min, HEAD_RANGE.cruise.max);
      TURN[slot] = clamp((headAt(W, wb, L_OUT, L_BASE, 1, H, ho, S) + XI[13]) * HEAD_SCALE.turn, HEAD_RANGE.turn.min, HEAD_RANGE.turn.max);
      FA[slot] = clamp((headAt(W, wb, F_OUT, F_BASE, 0, H, ho, S) + XI[8]) * HEAD_SCALE.align, HEAD_RANGE.align.min, HEAD_RANGE.align.max);
      FS[slot] = clamp((headAt(W, wb, F_OUT, F_BASE, 1, H, ho, S) + XI[9]) * HEAD_SCALE.sep, HEAD_RANGE.sep.min, HEAD_RANGE.sep.max);
      TT[slot] = clamp((headAt(W, wb, P_OUT, P_BASE, 0, H, ho, S) + XI[10]) * HEAD_SCALE.thrust, HEAD_RANGE.thrust.min, HEAD_RANGE.thrust.max);
      TR[slot] = clamp((headAt(W, wb, P_OUT, P_BASE, 1, H, ho, S) + XI[11]) * HEAD_SCALE.recoil, HEAD_RANGE.recoil.min, HEAD_RANGE.recoil.max);
      // Signed both ways on purpose: a body that lets go where its neighbour
      // holds walks the other way, and that is a lineage's to choose.
      GA[slot] = clamp(
        (headAt(W, wb, G_OUT, G_BASE, 0, H, ho, S) + XI[14]) * HEAD_SCALE.anchor,
        -GAIT_ANCHOR_MAX,
        GAIT_ANCHOR_MAX,
      );

      /*
       * What this body learns from the frame it has just had.
       *
       * Three factors, every one of them local to the body: an eligibility
       * trace per weight saying what that weight was lately doing, `phi'`
       * saying how much the state would have moved had the weight been
       * different, and one scalar saying whether things went better than
       * expected. The scalar is a temporal-difference error from the body's
       * own critic, and it is the only part of this that is a gradient
       * rather than a correlation — without it a Hebbian rule cannot tell a
       * useful coincidence from any other, and everything that fires
       * together grows together until it all saturates.
       *
       * The cost is the body's **own** tank, two ways, blended by
       * `learnReward`. `x4 - 1` is the *level*: zero when full, -1 when empty,
       * and it saturates — a third of a live soup sits at exactly zero and has
       * no gradient. The *rate* is the change in `x4` per second, clipped to
       * the same range, which still reads for a body at its cap. Local either
       * way, by decision. `x6` is the same shortfall relaxed over the wire
       * graph and is the other candidate teacher — it would make the net's
       * condition the thing a body learns about rather than its own — and
       * swapping it in is one line, since it is already in `x`.
       *
       * Nothing here decays except the trace, which is a credit window and
       * not a memory. A learned weight is the body's for life and travels
       * with it into whatever net it latches into next; that carriage is the
       * point of learning rather than only breeding.
       */
      if (learn) {
        const plo = slot * PLASTIC_LEN;
        const cro = slot * CRITIC_LEN;
        const value =
          CRITIC[cro] * H[ho] +
          CRITIC[cro + 1] * H[ho + 1] +
          CRITIC[cro + 2] * H[ho + 2] +
          CRITIC[cro + 3] * H[ho + 3] +
          CRITIC[cro + 4];
        const lvl = x4 - 1;
        /*
         * A zero `prev` is "no history": the slot is fresh, and without this a
         * body's first frame reads a full-scale rate and is rewarded for
         * existing. A genuinely empty body loses nothing by it — its rate that
         * frame is about zero anyway — and `extra` is a float, so landing on
         * exactly zero is otherwise rare. The device buffer is zeroed and
         * never seeded per body, so this is the sentinel both paths can share.
         */
        const was = PREV_FULL[slot];
        let rate = 0;
        if (was > 0) {
          rate = (x4 - was) * invDt;
          rate = rate <= -1 ? -1 : rate >= 1 ? 1 : rate;
        }
        PREV_FULL[slot] = x4;
        const reward = lvl + mix * (rate - lvl);
        const dlt = reward + discount * value - PREV_V[slot];
        PREV_V[slot] = value;
        /*
         * The critic's own update is an ordinary delta rule. Its estimate a
         * frame ago was a dot product with `h` a frame ago, so `h` a frame
         * ago is the gradient, and `prev` still holds it.
         */
        const kc = etaC * dlt;
        CRITIC[cro] += kc * p0;
        CRITIC[cro + 1] += kc * p1;
        CRITIC[cro + 2] += kc * p2;
        CRITIC[cro + 3] += kc * p3;
        CRITIC[cro + 4] += kc;

        // phi'(v) = 1 / (1 + |v|)^2. A saturated dimension has almost none
        // of it, which is what stops a pinned state dragging its inputs.
        const q0 = 1 / (1 + (v0 < 0 ? -v0 : v0));
        const q1 = 1 / (1 + (v1 < 0 ? -v1 : v1));
        const q2 = 1 / (1 + (v2 < 0 ? -v2 : v2));
        const q3 = 1 / (1 + (v3 < 0 ? -v3 : v3));
        post[0] = q0 * q0;
        post[1] = q1 * q1;
        post[2] = q2 * q2;
        post[3] = q3 * q3;
        pre[0] = x0;
        pre[1] = x1;
        pre[2] = x2;
        pre[3] = x3;
        pre[4] = x4;
        pre[5] = x5;
        pre[6] = x6;
        pre[7] = p0;
        pre[8] = p1;
        pre[9] = p2;
        pre[10] = p3;
        pre[11] = m0;
        pre[12] = m1;
        pre[13] = m2;
        pre[14] = m3;
        const step = etaM * dlt;
        const gW = g + W_IN;
        let touched = 0;
        let at = 0;
        /*
         * `Wx`, then `Wh`, then `Wn`, then `b` — the order the genome lays
         * them in, so one running index serves the trace, the learned delta
         * and the gene it is added to. The clamp is against the sum, because
         * what has to stay in range is the weight the state pass reads.
         */
        for (let d = 0; d < S; d++) {
          const pd = post[d];
          for (let k = 0; k < IN_DIMS; k++, at++) {
            const ti = plo + at;
            const tr = lam * TRACE[ti] + pd * pre[k];
            TRACE[ti] = tr;
            const base = CHEM[gW + at];
            let w = PLASTIC[ti] + step * tr;
            if (w < -MAXW - base) w = -MAXW - base;
            else if (w > MAXW - base) w = MAXW - base;
            PLASTIC[ti] = w;
            if (w !== 0) {
              touched = 1;
              /*
               * A learned sense weight turns a body that could not look at
               * the field into one that can, and the gate saying so is
               * otherwise settled at birth. Monotone, which is exact here
               * precisely because nothing decays back to zero.
               */
              if (k < 4) READS[slot] = 1;
            }
          }
        }
        for (let d = 0; d < S; d++) {
          const pd = post[d];
          for (let k = 0; k < S; k++, at++) {
            const ti = plo + at;
            const tr = lam * TRACE[ti] + pd * pre[7 + k];
            TRACE[ti] = tr;
            const base = CHEM[gW + at];
            let w = PLASTIC[ti] + step * tr;
            if (w < -MAXW - base) w = -MAXW - base;
            else if (w > MAXW - base) w = MAXW - base;
            PLASTIC[ti] = w;
            if (w !== 0) touched = 1;
          }
        }
        for (let d = 0; d < S; d++) {
          const pd = post[d];
          for (let k = 0; k < S; k++, at++) {
            const ti = plo + at;
            const tr = lam * TRACE[ti] + pd * pre[11 + k];
            TRACE[ti] = tr;
            const base = CHEM[gW + at];
            let w = PLASTIC[ti] + step * tr;
            if (w < -MAXW - base) w = -MAXW - base;
            else if (w > MAXW - base) w = MAXW - base;
            PLASTIC[ti] = w;
            if (w !== 0) touched = 1;
          }
        }
        // The bias, whose input is one.
        for (let d = 0; d < S; d++, at++) {
          const ti = plo + at;
          const tr = lam * TRACE[ti] + post[d];
          TRACE[ti] = tr;
          const base = CHEM[gW + at];
          let w = PLASTIC[ti] + step * tr;
          if (w < -MAXW - base) w = -MAXW - base;
          else if (w > MAXW - base) w = MAXW - base;
          PLASTIC[ti] = w;
          if (w !== 0) touched = 1;
        }

        /*
         * And the heads, on the same `delta` and the same trace, with one
         * factor swapped.
         *
         * The core's post-synaptic factor is `phi'(v)`, the gradient of `h`
         * with respect to that weight. A head is linear, so its equivalent is
         * 1 for every row — and a rule with the same factor everywhere moves
         * every head of the body in lockstep on the critic's sign, which can
         * never discover that cruise should rise while turn falls. So the
         * factor here is the row's *own* displacement `XI[row]`: the body
         * actually swam at `head + xi` this frame, and if the critic then says
         * things went better than expected, the weights that produced that
         * displacement are the ones to keep. Correlating a perturbation with
         * what followed it is node perturbation, and it is the standard answer
         * for a policy with a scalar reward and no target vector.
         *
         * `sigma` at zero makes every `XI` zero, so the trace decays and
         * nothing here moves — the heads go back to being genome-only without
         * a second code path saying so.
         *
         * `hc` and not `pre[7..]`: a head reads *this* frame's state, the one
         * the rows above have just written, while the core's own inputs are
         * the previous frame's. The eligibility has to name the number the
         * output was actually computed from.
         */
        let row = 0;
        for (let hi = 0; hi < HEAD_TABLE.length; hi++) {
          const head = HEAD_TABLE[hi];
          const hb = head.base;
          for (let r = 0; r < head.rows; r++, row++) {
            const xr = XI[row];
            const mo = head.at + r * S;
            for (let d = 0; d < S; d++) {
              const ti = plo + mo + d;
              const tr = lam * TRACE[ti] + xr * H[ho + d];
              TRACE[ti] = tr;
              const base = CHEM[gW + mo + d];
              let w = PLASTIC[ti] + step * tr;
              if (w < -MAXW - base) w = -MAXW - base;
              else if (w > MAXW - base) w = MAXW - base;
              PLASTIC[ti] = w;
              if (w !== 0) touched = 1;
            }
            // The bias, whose input is one — where the head has one inside the
            // block. `emit` and `taste` do not; see `PLASTIC_LEN`.
            if (hb >= 0) {
              const ti = plo + hb + r;
              const tr = lam * TRACE[ti] + xr;
              TRACE[ti] = tr;
              const base = CHEM[gW + hb + r];
              let w = PLASTIC[ti] + step * tr;
              if (w < -MAXW - base) w = -MAXW - base;
              else if (w > MAXW - base) w = MAXW - base;
              PLASTIC[ti] = w;
              if (w !== 0) touched = 1;
            }
          }
        }
        if (touched) PLASTIC_ON[slot] = 1;
      }
    }
  }


  /**
   * Spend the genome pass the shader ran at the end of the last frame.
   *
   * The store fields written here are exactly the ones `updateState` writes,
   * so every consumer downstream — the flock packing, the pair force,
   * `recoil`, the scent deposit, `state-hash` — reads what it always read and
   * has no idea the arithmetic moved.
   *
   * Slots are checked against the list the dispatch was packed from, not the
   * list now: a body that died between the two would otherwise have its
   * results paid to whichever body inherited its slot. That is the same
   * hazard `creditHarvest` guards, and here it is reachable rather than
   * theoretical, because rewrites settle between the dispatch and this.
   */
  private unpackGenome(): void {
    if (!this.genomePending) return;
    this.genomePending = false;
    const out = genomeGpu.outData;
    const stride = genomeGpu.outStride;
    const store = this.agentStore;
    const H = store.hAll;
    const EMITS = store.emitAll;
    const TASTES = store.tasteAll;
    const slots = this.genomeSlotBuf;
    const live = store.id;
    for (let i = 0; i < this.genomeCount; i++) {
      const slot = slots[i * 2];
      if (live[slot] !== slots[i * 2 + 1]) continue;
      const o = i * stride;
      const ho = slot * STATE_DIMS;
      H[ho] = out[o];
      H[ho + 1] = out[o + 1];
      H[ho + 2] = out[o + 2];
      H[ho + 3] = out[o + 3];
      const eo = slot * 4;
      EMITS[eo] = out[o + 4];
      EMITS[eo + 1] = out[o + 5];
      EMITS[eo + 2] = out[o + 6];
      EMITS[eo + 3] = out[o + 7];
      TASTES[eo] = out[o + 8];
      TASTES[eo + 1] = out[o + 9];
      TASTES[eo + 2] = out[o + 10];
      TASTES[eo + 3] = out[o + 11];
      store.cruise[slot] = out[o + 12];
      store.turn[slot] = out[o + 13];
      store.flockAlign[slot] = out[o + 14];
      store.flockSep[slot] = out[o + 15];
      store.transportThrust[slot] = out[o + 16];
      store.transportRecoil[slot] = out[o + 17];
      store.gaitAnchor[slot] = out[o + 18];
    }
  }

  /**
   * The sender's own `transportRecoil` is the kick's size — it is the one
   * doing the pumping — and the receiver's own `transportThrust` decides how
   * much of that kick it keeps versus hands back down the wire. Both are
   * heritable, so a lineage of strong, low-thrust pumps feeding a lineage of
   * high-thrust receivers drifts a net's swimming stroke somewhere neither
   * parent species swims alone.
   */
  private recoil(from: { id: number }, to: { id: number }, amount: number): void {
    const A = this.agents.get(from.id);
    const B = this.agents.get(to.id);
    if (A && B && A.transportRecoil > 0) {
      applyTransportRecoil(A, B, amount, A.transportRecoil, this.w, this.h, B.transportThrust);
    }
  }

  private startRewrites(params: Params): void {
    if (params.rewriteDuration <= 0) return;
    const ready = this.readyRedexes;
    for (let k = 0; k < ready.length; k++) {
      const wire = ready[k];
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      const rule = detectRule(A.kind, B.kind);
      if (rewriteCost(rule) > 0) {
        // Already paid, or not yet. `accrueRedexes` ran this frame and took
        // whatever the two could spare, so the pot is as full as it can be —
        // a pair rich enough to cover both shares outright filled it on the
        // frame it became ready and fires here exactly as it always did.
        const e = this.escrow.get(wire.id);
        if (!e || !stakeMet(A, e.paidA) || !stakeMet(B, e.paidB)) continue;
        // Spent: the stake becomes the new bodies, so it must not be refunded.
        this.escrow.delete(wire.id);
      }
      // Cancel only what the two are doing relative to each other. Zeroing
      // both outright pinned a collapsing pair to the world for the whole
      // rewrite while the rest of the soup drifted past it.
      const cx = (A.vx + B.vx) * 0.5;
      const cy = (A.vy + B.vy) * 0.5;
      A.vx = cx;
      A.vy = cy;
      B.vx = cx;
      B.vy = cy;
      A.omega = 0;
      B.omega = 0;
      const rw = beginRewrite(
        A,
        B,
        this.graph,
        this.agents,
        this.w,
        this.h,
        params.rewriteDuration,
        wire.id,
        params.rewritePull > 0,
      );
      /*
       * With learning on the device, these two bodies' learned weights are
       * there and not here, and `commitRewrite` is about to consolidate them
       * into four children's genomes on this side. Asking now gives the round
       * trip the whole length of the rewrite to complete.
       */
      if (this.genomeOnGpu && params.learnRate > 0) {
        this.learnWanted.push(A.slot, B.slot);
      }
      this.rewrites.push(rw);
      audio.push(rewriteAudio(rw, 'begin', wire.id, []), this.graph, this.agents);
    }
  }

  /**
   * Draw a collapsing wire's rope onto the chord between its two bodies.
   *
   * The bodies are moved kinematically during a rewrite, so the solver never
   * sees the motion that ought to be dragging the rope along with them. Left
   * to itself the rope is whipped by a shrinking rest length against nodes
   * that are still where they were, and its arc length climbs to 80 px
   * between two bodies touching each other — a knot, and a wire that sounds
   * lower the closer they get.
   */
  private reelRope(wire: Wire, pull: number): void {
    const A = this.agents.get(wire.a.id);
    const B = this.agents.get(wire.b.id);
    if (!A || !B) return;
    const sA = stemWorldInto(A, wire.a.slot, this.w, this.h, this.tmpStemA);
    const sB = stemWorldInto(B, wire.b.slot, this.w, this.h, this.tmpStemB);
    this.reelRopeTo(wire, sA.x, sA.y, sB.x, sB.y, pull);
  }

  private reelRopeTo(
    wire: Wire,
    ax: number,
    ay: number,
    bx: number,
    by: number,
    pull: number,
  ): void {
    const n = wire.nodes.length;
    if (n === 0 || pull <= 0) return;
    for (let i = 0; i < n; i++) {
      const node = wire.nodes[i];
      const t = (i + 1) / (n + 1);
      const tx = ax + (bx - ax) * t;
      const ty = ay + (by - ay) * t;
      node.x += (tx - node.x) * pull;
      node.y += (ty - node.y) * pull;
      node.vx *= 1 - pull;
      node.vy *= 1 - pull;
      node.prevX = node.x;
      node.prevY = node.y;
    }
    if (pull >= 1) {
      let len = 0;
      let px = ax;
      let py = ay;
      for (const node of wire.nodes) {
        len += Math.hypot(node.x - px, node.y - py);
        px = node.x;
        py = node.y;
      }
      len += Math.hypot(bx - px, by - py);
      wire.lastLen = len;
    }
  }

  /**
   * Leftover ropes on a rewriting pair are not the collapsing principal, but
   * they share a kinematic end. Unreeled they freeze in world space while the
   * stem slams, and the polyline grows into a several-hundred-pixel knot.
   * Snap them onto the handoff chord (leftover ↔ ghost, or leftover ↔ leftover
   * mate) so commit does not jump the endpoint.
   */
  private reelRewriteLeftovers(rw: Rewrite): void {
    const seen = new Set<number>();
    for (let e = 0; e < 2; e++) {
      const id = e === 0 ? rw.a : rw.b;
      const ag = this.agents.get(id);
      if (!ag) continue;
      const slots = ag.kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
      for (let k = 0; k < slots.length; k++) {
        const wire = this.graph.wireAtSlot(id, slots[k]);
        if (!wire || wire.id === rw.wireId || seen.has(wire.id)) continue;
        seen.add(wire.id);
        const handoff = rewriteHandoffStems(rw, wire, this.agents, this.w, this.h);
        if (handoff) this.reelRopeTo(wire, handoff.ax, handoff.ay, handoff.bx, handoff.by, 1);
        else this.reelRope(wire, 1);
      }
    }
  }

  /**
   * The knock of two bodies meeting at the end of a rewrite's pull.
   *
   * The pair is locked and moved kinematically, so the ordinary contact path
   * never sees them touch — but they visibly do, and an annihilation that
   * ends in silence at the moment of impact reads as a glitch. The closing
   * speed is the one the animation actually produces: the gap closes as
   * gap0*(1 - (t/PULL_END)^2), so at contact it is shutting at
   * 2*gap0/(PULL_END*duration).
   */
  private emitRewriteContact(rw: Rewrite): void {
    const A = this.agents.get(rw.a);
    const B = this.agents.get(rw.b);
    if (!A || !B) return;
    const gap0 = Math.hypot(rw.bx - rw.ax, rw.by - rw.ay);
    if (gap0 < 1e-3) return;
    const nx = (rw.bx - rw.ax) / gap0;
    const ny = (rw.by - rw.ay) / gap0;
    const vN = (2 * gap0) / Math.max(0.05, PULL_END * rw.duration);
    const mA = Math.max(0.08, A.mass);
    const mB = Math.max(0.08, B.mass);
    audio.push(
      {
        type: 'collision',
        agentA: A.id,
        agentB: B.id,
        kindA: A.kind,
        kindB: B.kind,
        impact: vN,
        overlap: boundRadius(A) + boundRadius(B),
        // Head-on and central: no lever arm, so the reduced mass is the whole
        // of the generalized effective mass.
        effMass: (mA * mB) / (mA + mB),
        vN,
        vT: 0,
        nx,
        ny,
        headingA: A.heading,
        headingB: B.heading,
        spin: 0,
      },
      this.graph,
      this.agents,
    );
  }

  /**
   * Whether anything in the pond is holding something undigested.
   *
   * Set where a mouthful lands and cleared by the pass that finds every gut
   * empty. It is a hint, not a fact: over-setting it costs one pass over the
   * roster, and the only thing that must never happen — a gut nothing drains
   * — needs it to be under-set, which nothing here does.
   */
  /** This frame's step, for the passes that are called without it. */
  private frameDt = 1 / 60;

  private gutLive = false;
  private readonly starving: number[] = [];


  /**
   * The gut: what a body swallowed becomes what a body has.
   *
   * One reaction, `r1`, and it splits two ways. What `intake` asks for goes to
   * the reactor as primer; the rest is banked, as far as the tank has room.
   * What neither takes stays in the gut, where it goes on bounding the next
   * mouthful — which is satiety, and it is three mechanisms deep rather than a
   * clamp: a body that cannot bank cannot clear its gut, so it cannot eat.
   *
   * It used to be four species and three rules. The harvest was a *sample of
   * the water*, so a body held species it could not touch; an uptake row was
   * its recipe for each, `catCoSubstrate` made the ground a reagent the others
   * were converted with, and whatever a body could not convert waited for
   * excretion to clear it. A body eats the ground and nothing else now. There
   * is nothing in a gut its owner cannot digest, so there is no waste, no
   * recipe to express, no co-substrate to pair, and no excretion pass — the
   * three rules collapse into the one reaction above.
   */
  private runDigestion(params: Params, t: number): void {
    const rate = params.digestRate;
    // Nothing has ever been swallowed, which is every frame of a pond running
    // at `uptakeVmax` 0. Guarded rather
    // than walked, for the reason `HarvestPlan.build` guards on `meter`: a
    // per-body pass that always finds zero is still a per-body pass.
    if (!(rate > 0) || !this.gutLive) return;
    const store = this.agentStore;
    const GUT = store.gut;
    const EXTRA = store.extra;
    const CAP = store.energyCap;
    const R = store.react;
    const INTAKE = store.intake;
    /*
     * What a body routes to its reactor instead of banking, and what a unit
     * of banked food is worth in reactor units.
     *
     * This is the doc's `r1`, `A -> B`, and the whole of the reactor's supply.
     * It used to *buy* its fuel out of the tank at `metabolicCost`, so food
     * ran grid -> gut -> tank -> reactor and the tank sat in the middle of a
     * round trip; the split is what cuts the tank out of it. A body with a
     * full tank and an empty gut now has no clock.
     *
     * The yield is a unit conversion and it is large, because a reactor turns
     * its pool over many times per unit of matter — that is what a currency
     * is. The old `metabolicCost` of 0.01 was the same conversion written as
     * a price; this is its reciprocal, and it is one dial instead of two.
     */
    const yieldB = params.metabolicYield;
    /*
     * The reactor's pools are in its own units and outside the pond's books —
     * a reactor turns its pool over many times per unit of matter, so the two
     * scales differ by orders. What crosses the boundary is the routed food,
     * and it leaves through the same road rent uses, so metabolising is
     * fertilising and the dish's total does not move. At `upkeepExcrete` 0 it
     * is destroyed instead, which is what the old buy-from-the-tank path did
     * with the price it charged; the conservation suite turns it on.
     */
    const back = params.upkeepExcrete;
    const grid = this.energy;
    const X = store.x;
    const Y = store.y;
    const full = 1 - Math.exp(-rate * t);
    let live = false;
    for (const a of this.agents.values()) {
      const s = a.slot;
      if (store.gutTotal(s) <= 0) continue;
      live = true;
      /*
       * **The tank bounds what is banked; the reactor's share flows anyway.**
       *
       * Digestion used to stop outright at a full tank, which was satiety
       * three mechanisms deep — a full body cannot digest, so its gut fills,
       * so it cannot eat. That worked while the tank had a continuous outflow
       * to keep it off the ceiling, and it had two: the standing rent and the
       * excretion rows. Both are gone, so every body would fill, stop
       * digesting and lose its clock — a well-fed body starving its own
       * metabolism, which is backwards.
       *
       * So the two halves of `r1` are bounded separately. What `intake` routes
       * to the reactor is consumed the moment it is made and needs no room;it
       * is only the banked half that has to fit. A sated body therefore keeps
       * exactly the clock its `intake` pays for and banks nothing, and what it
       * does not digest stays in its gut, where it goes on bounding the next
       * mouthful. Nothing spills and nothing is destroyed.
       */
      // A gut holds ground and a body can always digest ground, so the recipe
      // rows the uptake half used to gate this on have nothing left to say.
      const have = GUT[s];
      if (have <= 0) continue;
      /*
       * Mass action, as an exponential rather than a product, so a rate above
       * one frame's worth cannot take more than the body is holding. Snapped
       * to empty below `EXTRA_FULL_EPS`, because mass action never reaches
       * zero on its own and `gutLive` would then never clear: a gut holding a
       * crumb forever is a roster walk forever.
       */
      let take = have * full;
      if (have - take < EXTRA_FULL_EPS) take = have;
      /*
       * **`intake` is a rate, not a share: how fast this body feeds its
       * reactor, in matter per second.**
       *
       * It used to be a share of the mouthful, and removing the tank's drains
       * is what exposed that as the wrong quantity. While rent and the
       * excretion rows drained the tank a body's throughput was small, and a
       * quarter of it happened to be about what the reactor consumed; with
       * neither, throughput is whatever the ground supplies, and a quarter of
       * that put every reactor past the top of its fuel window — saturated and
       * still, which is the glutted end rather than the fed one.
       *
       * A share makes the clock a property of the dish. A rate makes it a
       * property of the body: the influx `j` that decides whether a body
       * paces, relays or sits saturated is `intake * metabolicYield /
       * metabolicRate`, and every term in that is the body's own or the
       * world's constant. `docs/metabolism-spec.md` §1 asks for exactly this
       * and calls it first among the four — a period that is a consequence of
       * what a lineage controls.
       *
       * Still fed by *eating*: it can only route what it has digested, so a
       * body with a full tank and an empty gut has no clock however high it
       * sets this.
       */
      const want = INTAKE[s] * t;
      let toReactor = take < want ? take : want;
      if (toReactor < 0) toReactor = 0;
      let toBank = take - toReactor;
      const room = Math.max(0, CAP[s] - EXTRA[s]);
      if (toBank > room) toBank = room;
      const moved = toReactor + toBank;
      if (!(moved > 0)) continue;
      GUT[s] = have - moved;
      if (toReactor > 0) {
        // Split, not taxed: what feeds the reactor is food the body did not
        // bank, so running a metabolism costs exactly what it consumes and
        // needs no price of its own.
        const r = s * REACT_SPECIES + REACT_B;
        const b = R[r] + toReactor * yieldB;
        R[r] = b > REACT_CAP ? REACT_CAP : b;
        if (back > 0) grid.addAt(X[s], Y[s], toReactor * back);
      }
      if (toBank > 0) EXTRA[s] += toBank;
    }
    /*
     * Read before this pass moved anything, so a pond that has just finished
     * digesting the last of it runs one more empty pass and then stops. The
     * error is one frame in the safe direction; the other direction would
     * leave a gut that nothing ever drained. The snap above is what makes
     * "finished" a state a gut can actually reach.
     */
    this.gutLive = live;
  }

  

  private tickRewrites(params: Params, dt: number): void {
    const done: Rewrite[] = [];
    for (const rw of this.rewrites) {
      if (advanceRewrite(rw, this.agents, this.w, this.h, dt, params.rewritePull)) done.push(rw);
      // The wire is what pulls them together, so shorten it in step with the
      // pull. It retracts into the pair and is gone by the time they touch,
      // and because it stays taut on the way its pitch rises instead of
      // sagging the way a slackening rope's does.
      //
      // The same accelerating ease `advanceRewrite` used to close the pair on,
      // and they have to be the same curve now that the wire is the thing
      // doing it. They were not: the animation pulled on `easeIn` and the
      // collapse was linear, so at a tenth of the way through a rewrite the
      // wire was asking for seven times the closing the picture showed. That
      // only stayed invisible while the wire was skipped by the solver, and it
      // is a wire releasing its tension — it should arrive faster than it sets
      // off, not set off at full speed.
      const raw = clamp(rw.t / PULL_END, 0, 1);
      const pull = raw * raw;
      if (rw.wireId >= 0) {
        const wire = this.graph.wires.get(rw.wireId);
        if (wire) {
          wire.collapse = pull;
          this.reelRope(wire, pull);
          if (pull >= 1 && !rw.struck) {
            rw.struck = true;
            this.emitRewriteContact(rw);
          }
        }
      }
      this.reelRewriteLeftovers(rw);
    }
    for (const rw of done) {
      audio.push(
        rewriteAudio(rw, 'commit', rw.wireId, leftoverAgentIds(rw)),
        this.graph,
        this.agents,
      );
      const dyingA = this.agents.get(rw.a);
      const dyingB = this.agents.get(rw.b);
      // Signed: a pair that annihilates while in debt releases less than a
      // well-fed one, and their debt dies with them rather than being minted
      // away. `rewriteYield` has already counted the bodies themselves.
      let pool = rewriteYield(rw.rule, this.bodyValue);
      if (dyingA) pool += dyingA.extra;
      if (dyingB) pool += dyingB.extra;
      // And what either had swallowed and not yet turned into anything goes
      // back as itself, the way `kill` does it: `commitRewrite` clears the
      // slots, and a rewrite that kept the guts would leak a mouthful on every
      // commute in a metered pond.
      if (dyingA) this.spillGut(dyingA.slot, dyingA.x, dyingA.y);
      if (dyingB) this.spillGut(dyingB.slot, dyingB.x, dyingB.y);
      pool = Math.max(0, pool);
      const leftoverIds = leftoverAgentIds(rw);
      const beforeId = this.nextId;
      this.nextId = commitRewrite(
        rw,
        this.agents,
        this.graph,
        params,
        this.nextId,
        this.time,
        this.w,
        this.h,
        this.agentStore,
        this.breed,
      );
      // It writes into the agents Map itself, so nothing else knows the roster
      // moved. Anything keyed on it — the body list, the flocking pair list,
      // the force scratch — is stale until this line.
      this.noteRosterChange();
      this.tally.born += this.nextId - beforeId;
      if (rw.rule === 'commute') this.tally.commutes++;
      else if (rw.rule === 'erase') this.tally.erases++;
      else this.tally.annihilations++;
      const recipients: Agent[] = [];
      for (const id of leftoverIds) {
        const ag = this.agents.get(id);
        if (ag) recipients.push(ag);
      }
      for (let id = beforeId; id < this.nextId; id++) {
        const ag = this.agents.get(id);
        if (ag) recipients.push(ag);
      }
      settlePool(pool, recipients, this.energy, rw.midX, rw.midY);
    }
    if (done.length) this.rewrites = this.rewrites.filter((rw) => !done.includes(rw));
  }
}

/**
 * Pumping energy along a wire shoves the two bodies apart along it.
 *
 * A body that ejects energy east recoils west, and the body that absorbs it is
 * pushed east, as an impulse, so a light Era twitches where a Con barely stirs.
 *
 * `thrust` is how much of the receiver's kick is withheld. At 0 the pair is
 * equal and opposite and the flock's centre of mass never moves. Above 0 the
 * pair keeps a net impulse of `thrust * gain * amount` pointing back down the
 * wire, *against* the direction the energy travelled — so a net with a standing
 * gradient, surplus at one end and shortage at the other, swims away from its
 * own supply. That is a momentum pump on purpose: it is the one force in here
 * that a net can only generate by moving energy through itself, which makes
 * transport something a body does rather than something that happens to it.
 * Nothing runs away, because fluid drag turns a sustained pump into a terminal
 * drift rather than an acceleration.
 *
 * What it buys, at the default gain, is a twitch on the *events*: a fresh
 * latch beside a charged body, a rescue, a pair refilling after a commute —
 * transfers of most of a unit, which nudge an Era about 56 px/s against
 * settled speeds around 50. It does not show up in steady state, and the
 * reason is the economy rather than the constant: with everyone nearly topped
 * up, a frame's transfer is about 4e-4, three orders of magnitude below a
 * one-off. Measured over a seeded 30 s soup, net openness at gains 0 / 6 / 12 /
 * 25 is 80 / 86 / 79 / 89 px, which is seed noise. Moving more energy per
 * frame is what would make the tissue breathe; raising this alone will not.
 * The same arithmetic bounds the swimming: a steady-state transfer of ~4e-4 a
 * frame at gain 20 and full thrust is under 1 px/s of drift, so the stroke
 * shows up on the events — a rescue, a refill after a commute — and on a net
 * held under a real gradient, not on a soup that is already topped up.
 */
export function applyTransportRecoil(
  A: { x: number; y: number; vx: number; vy: number; mass: number; locked: boolean },
  B: { x: number; y: number; vx: number; vy: number; mass: number; locked: boolean },
  amount: number,
  gain: number,
  w: number,
  h: number,
  thrust = 0,
): void {
  if (A.locked || B.locked || !(amount > 0) || !(gain > 0)) return;
  const d = wrapDeltaVec(A.x, A.y, B.x, B.y, w, h);
  const dist = Math.hypot(d.x, d.y);
  if (!(dist > 1e-6)) return;
  const p = gain * amount;
  // The sender's recoil is the whole impulse and never scales — thrust only
  // decides how much of it the receiver cancels, so dialling thrust up changes
  // where the pair ends up without changing how hard the pump kicks.
  const catches = 1 - Math.min(1, Math.max(0, thrust));
  const nx = d.x / dist;
  const ny = d.y / dist;
  const wA = 1 / Math.max(0.08, A.mass);
  const wB = 1 / Math.max(0.08, B.mass);
  A.vx -= nx * p * wA;
  A.vy -= ny * p * wA;
  B.vx += nx * p * catches * wB;
  B.vy += ny * p * catches * wB;
}

/** Move the closest point on polyline segment `seg` by (ux, uy). Stems stay put. */
function moveRopeClosest(wire: Wire, seg: number, t: number, ux: number, uy: number): void {
  const n = wire.nodes.length;
  const a = seg === 0 ? null : wire.nodes[seg - 1];
  const b = seg === n ? null : wire.nodes[seg];
  if (a && b) {
    const u = 1 - t;
    const denom = u * u + t * t;
    const wa = denom > 1e-9 ? u / denom : 0.5;
    const wb = denom > 1e-9 ? t / denom : 0.5;
    a.x += ux * wa;
    a.y += uy * wa;
    b.x += ux * wb;
    b.y += uy * wb;
    return;
  }
  if (a) {
    a.x += ux;
    a.y += uy;
  } else if (b) {
    b.x += ux;
    b.y += uy;
  }
}

function leftoverAgentIds(rw: Rewrite): number[] {
  const ids: number[] = [];
  const ports = [rw.leftoverAL, rw.leftoverAR, rw.leftoverBL, rw.leftoverBR];
  for (const p of ports) {
    if (!p || p.id === rw.a || p.id === rw.b) continue;
    if (!ids.includes(p.id)) ids.push(p.id);
  }
  return ids;
}

function rewriteAudio(
  rw: Rewrite,
  phase: 'begin' | 'commit',
  wireId: number,
  leftovers: number[],
): RewriteEvent {
  const kindA: AgentKind =
    rw.a === rw.eraId ? 'era' : rw.a === rw.conId ? 'con' : rw.a === rw.dupId ? 'dup' : 'era';
  const kindB: AgentKind =
    rw.b === rw.eraId ? 'era' : rw.b === rw.conId ? 'con' : rw.b === rw.dupId ? 'dup' : 'era';
  return {
    type: 'rewrite',
    phase,
    rule: rw.rule,
    agentA: rw.a,
    agentB: rw.b,
    kindA,
    kindB,
    wireId,
    leftovers,
  };
}

/**
 * One row of an output head, read straight out of the store arrays.
 *
 * The twin of `agents.ts`'s `head`, which goes through `Agent.chem` and
 * `Agent.h`. This one exists for `updateState`'s inner loop, where those two
 * accessors are the whole cost — see the note on reading `chemAll` directly.
 */
/**
 * What crosses a wire for one species this frame, signed from `a` toward `b`.
 *
 * The doc's §4: a node broadcasts out of its principal port at `Tx[slot]`
 * times however much it is holding above its gate `Gx[slot]`, which is
 * `H(x - G)` with the step softened into a rectifier. `ga` and `gb` are each
 * end's genome offset, or -1 when that end is not a principal. `pool` is the
 * array the species lives in and `stride` its width, because the doc's A is
 * the gut and the other three are the reactor.
 *
 * Bounded once on the net by a quarter of the source's stock and a quarter of
 * the sink's room: a body's species can be drawn on by its own broadcast and
 * by the far end of each of its three wires, so a quarter is what makes an
 * overdraw impossible and lets the apply need no clamp of its own.
 */
/**
 * The three reactor species, each in units its own balance makes natural, and
 * each through the same bounded signed map as the stroke has always used.
 *
 * `B · k₂/(k₃+d)`, `C`, and `D · d/k₃` all sit near one at a fixed point —
 * they are what each species would be if the loop stopped — so one
 * `metabolicWave` serves all three and no species needs a constant of its own.
 * Written into `out` in the reactor's order, B then C then D.
 */
function speciesWaves(
  B: number,
  C: number,
  D: number,
  cat: number,
  reset: number,
  decay: number,
  waveK: number,
  out: Float64Array,
): void {
  const bc = reset + decay > 0 ? (B * cat) / (reset + decay) : 0;
  const dc = reset > 0 ? (D * decay) / reset : 0;
  out[0] = (2 * bc) / (waveK + bc) - 1;
  out[1] = (2 * C) / (waveK + C) - 1;
  out[2] = (2 * dc) / (waveK + dc) - 1;
}

/*
 * Where each species sits on the cycle, relative to the catalyst.
 *
 * Three phase-shifted copies of one oscillation, so they span a plane and any
 * two of them are a basis for it: a weighted sum of the three reaches any
 * phase, and the same phase can be written more than one way. `docs/concepts.md`
 * has where the numbers come from — B leads C by 84 degrees, near quadrature,
 * and D trails it by `atan(w/d)`, which runs 51 degrees at the bottom of the
 * fuel window to 65 at the top and measured 53.5 on a lone fed body. Nobody
 * chose them; they fall out of `dD/dt = k3*C - d*D` being a first-order lag.
 *
 * Constants here rather than per-body because the spread across the fuel
 * window is a seventh of the angle and a body's own operating point moves
 * within it every cycle. `src/pond/spectrum.ts` derives them in closed form if
 * a body-by-body version is ever wanted.
 */
const PHASE_B = (84 * Math.PI) / 180;
const PHASE_C = 0;
const PHASE_D = (-53.5 * Math.PI) / 180;

/**
 * The same actuator, delayed by `theta`, written in the C-D basis.
 *
 * **A rotation, and not a blend toward the other actuator.** The obvious way
 * to give the stroke a phase that varies along a body is to slide its mixture
 * from `Sw` toward `Gw`, and it is wrong: at the far end the two coincide, the
 * stroke and the grip are then in phase, the loop in (shape, grip) has zero
 * area, and that end of the net contributes no displacement at all. Purcell,
 * which is the whole reason the 53 degrees between them exists. A blend
 * travels along a chord between two points; what is wanted is a turn.
 *
 * So: read the mixture as a phasor against the angles above, turn it by
 * `theta`, and write it back onto C and D — which span the plane, so the
 * output waveform is exactly the original delayed, at the same amplitude, and
 * whatever B component a lineage had is not lost but re-expressed. Both
 * actuators are turned by the same angle, so the angle *between* them — the
 * loop's area, the thing that makes a body move at all — is untouched, and
 * what varies along the body is only when in the cycle each segment acts.
 * That is peristalsis rather than a gradient of enthusiasm.
 */
function rotateMix(chem: Float32Array, base: number, theta: number, out: Float64Array): void {
  const re =
    chem[base] * Math.cos(PHASE_B) + chem[base + 1] * Math.cos(PHASE_C) + chem[base + 2] * Math.cos(PHASE_D);
  const im =
    chem[base] * Math.sin(PHASE_B) + chem[base + 1] * Math.sin(PHASE_C) + chem[base + 2] * Math.sin(PHASE_D);
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  // Delay, so a deeper body acts later: multiply by exp(-i*theta).
  const rr = re * ct + im * st;
  const ri = im * ct - re * st;
  /*
   * Back onto C and D. With C at 0 and D at `PHASE_D`, the pair (c, d) that
   * reproduces a phasor is the two-phase synthesis: `d` carries the whole of
   * the component D alone can express and `c` makes up the rest.
   */
  const sd = Math.sin(PHASE_D);
  let d = ri / sd;
  let c = rr - d * Math.cos(PHASE_D);
  /*
   * And rescaled to the weight the lineage actually asked for.
   *
   * The synthesis above is exact only if the three waves are unit-amplitude
   * sinusoids at those angles. They are neither: `speciesWaves` pushes scaled
   * B, C and D through a saturating map, so their swings differ and B's and
   * D's carry a rate-constant ratio in front of them. Left unscaled, a turn of
   * 126 degrees on a seeded pure-C stroke came out as coefficients of −1.19
   * and 1.01 and the actuator spent most of its cycle pinned at the clamp —
   * measured on the crawl bench, half the chain reading exactly 1.000.
   *
   * So the *direction* is taken from the turn and the *magnitude* from the
   * gene. A lineage that asked for a weak stroke keeps a weak one however deep
   * the body sits, and the profile changes when a segment acts and not how
   * hard, which is the whole distinction between a travelling wave and a
   * gradient of enthusiasm.
   */
  const was = Math.abs(chem[base]) + Math.abs(chem[base + 1]) + Math.abs(chem[base + 2]);
  const now = Math.abs(c) + Math.abs(d);
  if (now > 1e-9) {
    const k = was / now;
    c *= k;
    d *= k;
  }
  out[0] = 0;
  out[1] = c;
  out[2] = d;
}

/**
 * One actuator's reading: its own mixture of the three, bounded.
 *
 * The weights are a direction in (B, C, D) and the globals — `gaitSwell` for
 * the stroke, `gripSwing` for the grip — are how hard it is pulled. Clamped to
 * the same `[-1, 1]` a single species gave, so everything downstream keeps the
 * bounds it was written against however a lineage mixes.
 */
function actuatorOf(chem: Float32Array, base: number, w: Float64Array): number {
  const v = chem[base] * w[0] + chem[base + 1] * w[1] + chem[base + 2] * w[2];
  return v <= -1 ? -1 : v >= 1 ? 1 : v;
}

/** `actuatorOf` against a mixture held somewhere other than the genome. */
function actuatorOfVec(mix: Float64Array, w: Float64Array): number {
  const v = mix[0] * w[0] + mix[1] * w[1] + mix[2] * w[2];
  return v <= -1 ? -1 : v >= 1 ? 1 : v;
}

/**
 * Both actuators for one body, turned by `theta`. `out` is [stroke, grip].
 *
 * One function because `advanceGait` reads them **twice** — once after
 * integrating the reactor, and again after the coupling pass has moved species
 * across the wires, so that "a body that has just been driven strokes as what
 * it now is rather than what it was". The second read used to be written out
 * longhand, and when the depth profile went into the first one only, the
 * second quietly overwrote it for every body whose wire carried anything: the
 * stroke came out bit-identical at `gaitProfile` 0 and 4.4 on a bench built to
 * tell them apart, which reads as the mechanism doing nothing rather than as
 * one of two call sites being stale.
 */
function actuatorPair(
  chem: Float32Array,
  g: number,
  wv: Float64Array,
  theta: number,
  swRot: Float64Array,
  gwRot: Float64Array,
  out: Float64Array,
): void {
  if (theta === 0) {
    out[0] = actuatorOf(chem, g + SW_BASE, wv);
    out[1] = actuatorOf(chem, g + GW_BASE, wv);
    return;
  }
  rotateMix(chem, g + SW_BASE, theta, swRot);
  rotateMix(chem, g + GW_BASE, theta, gwRot);
  out[0] = actuatorOfVec(swRot, wv);
  out[1] = actuatorOfVec(gwRot, wv);
}

function crossing(
  pool: Float64Array,
  CHEM: Float32Array,
  sa: number,
  sb: number,
  ga: number,
  gb: number,
  stride: number,
  species: number,
  slot: number,
  cap: number,
  k: number,
): number {
  const ia = sa * stride + species;
  const ib = sb * stride + species;
  let f = 0;
  if (ga >= 0) {
    const rate = CHEM[ga + TX_BASE + slot];
    if (rate > 0) {
      const drive = pool[ia] - CHEM[ga + GX_BASE + slot];
      if (drive > 0) f += rate * drive * k;
    }
  }
  if (gb >= 0) {
    const rate = CHEM[gb + TX_BASE + slot];
    if (rate > 0) {
      const drive = pool[ib] - CHEM[gb + GX_BASE + slot];
      if (drive > 0) f -= rate * drive * k;
    }
  }
  if (f > 0) {
    const have = pool[ia] * 0.25;
    const room = (cap - pool[ib]) * 0.25;
    const lim = have < room ? have : room;
    return f > lim ? (lim > 0 ? lim : 0) : f;
  }
  if (f < 0) {
    const have = pool[ib] * 0.25;
    const room = (cap - pool[ia]) * 0.25;
    const lim = have < room ? have : room;
    return -f > lim ? (lim > 0 ? -lim : 0) : f;
  }
  return 0;
}

function headAt(
  chem: Float32Array,
  g: number,
  matrix: number,
  base: number,
  row: number,
  h: Float64Array,
  ho: number,
  dims: number,
): number {
  const o = g + matrix + row * dims;
  return (
    chem[g + base + row] +
    chem[o] * h[ho] +
    chem[o + 1] * h[ho + 1] +
    chem[o + 2] * h[ho + 2] +
    chem[o + 3] * h[ho + 3]
  );
}
