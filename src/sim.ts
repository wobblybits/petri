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
  HEAD_SCALE,
  L_BASE,
  L_OUT,
  P_BASE,
  P_OUT,
  PLASTIC_LEN,
  CRITIC_LEN,
  LEARN_CRITIC,
  LEARN_PREV_V,
  LEARN_TRACE,
  IN_BOUND,
  IN_DEMAND,
  IN_DIMS,
  IN_FULL,
  IN_SENSE,
  SENSE_SCALE,
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
import { closestTOnSegment, segmentsIntersect, WIRE_RADIUS, wireBowBudget, bounceOffDisk } from './geom.ts';
import { PairGrid } from './grid.ts';
import { CHAIN_MASS, contactMechanics, portExitAngle, solveContact } from './chain.ts';
import { CH, FERTILISE_CH, FIELD_CELL, FIELD_CELLS, Fields, worldBoundRadius } from './fields.ts';
import { Graph, ropeIsLive, wrapPos, type Wire } from './graph.ts';
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
  rewriteYield,
  seedRequest,
  settlePool,
  stakeMet,
  spreadRequestsFast,
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
  WF_FULL,
  WF_HOLD,
  WF_SHAPE,
  WF_SKIP,
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
function packTaste(
  out: Float32Array | Float64Array,
  at: number,
  tasteAll: Float64Array,
  slot: number,
  groundScale: number,
): void {
  const from = slot * 4;
  out[at] = tasteAll[from];
  out[at + 1] = tasteAll[from + 1];
  out[at + 2] = tasteAll[from + 2];
  out[at + 3] = tasteAll[from + 3];
  out[at + CH.energy] = tasteAll[from + CH.energy] * groundScale;
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

  /*
   * Activity LOD — dual-rate islands.
   *
   * The view LOD demotes a body for being far away. This demotes it for being
   * still: a taut, aged, calm net runs the same cheap disc+span path even on
   * screen and close up, so only the parts of the pond actually doing
   * something pay NEAR prices. The two compose in `agentDetailed` and either
   * can veto.
   *
   * It is the one change here that reduces how many bodies do expensive work
   * rather than making the work per body cheaper — and the one that changes
   * behaviour, because a sleeping body is off the SAT path and settled tissue
   * can therefore rest a little closer together than it otherwise would.
   */
  private sleepActive = false;
  private readonly awakeAgents = new Set<number>();
  /** id -> time until which it stays awake, so the boundary cannot flicker. */
  private readonly holdAwake = new Map<number, number>();
  /** Contact pairs from the collision pass, flattened, drained each frame. */
  private readonly hitWake: number[] = [];

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

  constructor(w: number, h: number, fieldCells = Sim.defaultFieldCells) {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    this.coverW = this.w;
    this.coverH = this.h;
    this.fields = new Fields(fieldCells, fieldCells * FIELD_CELL);
    this.graph = new Graph(this.agentStore);
    this.graph.onLatch = (ev) => {
      this.tally.latches++;
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
    this.sleepActive = false;
    this.awakeAgents.clear();
    this.holdAwake.clear();
    this.hitWake.length = 0;
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
    this.energy.addAt(agent.x, agent.y, deathYield(agent, this.bodyValue));
    this.graph.detachAgent(id);
    this.agents.delete(id);
    this.agentStore.release(id);
    this.rosterVersion++;
    this.tally.died++;
    if (this.grabbed?.id === id) this.grabbed = null;
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
    const t = this.beginFrame(dt, params);
    this.collectRewriteFrozen();
    Sim.phase('collectRewriteFrozen');
    this.assignPhysicsLod(view);
    Sim.phase('assignPhysicsLod');
    this.assignActivityLod(params);
    Sim.phase('assignActivityLod');
    this.graph.syncRest(this.time, params, this.wireDetailed);
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
    const block = this.openForceBlock(params);
    this.refreshForceScratch(this.forceList());
    this.refreshBound();
    Sim.phase('openForceBlock');
    this.steer(params, t);
    Sim.phase('steer');
    this.portTorques(params, t);
    Sim.phase('portTorques');
    this.declutter(params, t);
    Sim.phase('declutter');
    this.uncrossPrincipals(params, t);
    Sim.phase('uncrossPrincipals');
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
    this.energy.configure(params.energyCell, params.ambientEnergy);
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
      // above it, uptake is a rate against the block's own density. The GPU
      // twin applies the same limit inside the shader — see `field.wgsl`'s
      // `harvest` and `energy.ts:uptakeRate`.
      harvestSlotsFast(this.agents.values(), this.agentStore, this.energy, this.harvestPlan, {
        // `t` is the frame's clamped dt — see `beginFrame`.
        cap: params.uptakeVmax * t,
        ks: params.uptakeKs,
      });
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
     * Farming: stock converted into ground, one for one.
     *
     * Nothing is created at the transfer — this moves energy out of a tank and
     * onto the dish. The return comes from `grow` afterwards, which is
     * logistic and so proportional to what a cell already holds: a cell grazed
     * to zero is stuck there forever, and seeding it with anything restarts
     * growth that carries it back toward capacity. Investing in a scar returns
     * much more than it cost, and the dish's capacity bounds the whole thing,
     * so it is production and not a mint.
     *
     * Capped at what the body actually has above break-even. A body cannot
     * farm itself into debt, which would turn the most useful thing in the
     * economy into a way to die.
     */
    if (params.farmRate > 0) {
      const rate = params.farmRate * t;
      for (const a of this.agents.values()) {
        if (a.locked) continue;
        const w = this.agentStore.emitAll[a.slot * 4 + CH.energy];
        if (w <= 0) continue;
        const want = rate * w;
        const have = a.extra > 0 ? a.extra : 0;
        const give = want < have ? want : have;
        if (give <= 0) continue;
        a.extra -= give;
        this.energy.addAt(a.x, a.y, give);
      }
    }
    for (const id of this.contactDamage(params, t)) this.kill(id);
    for (const id of tickUpkeepFast(this.agents.values(), this.agentStore, t, params.upkeep, this.energy, {
      excrete: params.upkeepExcrete,
      eraRatio: params.eraUpkeepRatio,
    })) {
      this.kill(id);
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
    if (!this.fieldOnGpu) {
      if (!this.scentWriteNative(params)) this.deposit(params);
      Sim.phase('scentWrite');
      this.fields.diffuse(params.diffuse);
      Sim.phase('field:diffuse1');
      this.fields.diffuse(params.diffuse * 0.65);
      Sim.phase('field:diffuse2');
      // After the spreading and before the decay: Gray-Scott's own kill term
      // is `feed + kill`, and `decay` then acts on both channels on top of it,
      // so the effective kill is larger than `reactKill` alone.
      this.fields.react(CH.conP, CH.dupP, params.reactFeed, params.reactKill, t);
      Sim.phase('field:react');
      this.fields.decay(params.decay);
      Sim.phase('field:decay');
      // After the passes that move it, so a cell grows from what it kept
      // rather than from what it was about to lose.
      this.fields.grow(
        CH.energy,
        params.energyRegrow * t,
        this.energy.cellCap,
        FERTILISE_CH,
        params.fertilise,
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
        const held = poseHeld(a);
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
      if (poseHeld(agent)) return;
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
   * Only when every pass in that run is native: `uncrossPrincipals` is still
   * JS, so a scene that uses it would have that pass read stale velocities and
   * then have its own writes overwritten on unpack. It is off by default, and
   * when it is on each pass falls back to copying for itself.
   */
  private openForceBlock(params: Params): boolean {
    this.forceBlock = false;
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    if (params.uncross > 0) return false;
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
      const held = poseHeldAt(st, sl);
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
   * ask a `Map` for it, twice a wire: the flocking adjacency, the wake
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

  private refreshBound(): void {
    if (this.boundVersion === this.graph.version && this.boundRoster === this.rosterVersion) {
      return;
    }
    this.boundVersion = this.graph.version;
    this.boundRoster = this.rosterVersion;
    const g = this.graph;
    const BOUND_OF = this.agentStore.bound;
    for (const a of this.agents.values()) {
      const n = a.kind === 'era' ? 1 : 3;
      const sl = a.slot;
      let filled = 0;
      for (let k = 0; k < n; k++) if (!g.isFreeAtSlot(sl, k)) filled++;
      BOUND_OF[sl] = filled / n;
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
     * The wake graph's CSR again, aliased rather than copied.
     *
     * Four passes wanted the wire graph as neighbour lists — the two LOD
     * passes, flocking, and this one for the need field and the genome — and
     * each built its own from the same wires in the same order on the same
     * key. `refreshWakeGraph` is the one that builds it; this hands the same
     * two arrays to the energy passes, which only read them. The queue
     * scratch stays this object's own, since the relaxation does write that.
     */
    this.refreshWakeGraph();
    this.wireAdj.off = this.wakeOff;
    this.wireAdj.nei = this.wakeNei;
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
      flags[i] = pFree ? 1 : 0;
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

  private collectRewriteFrozen(): void {
    this.rewriteFrozen.clear();
    for (const rw of this.rewrites) {
      this.rewriteFrozen.add(rw.a);
      this.rewriteFrozen.add(rw.b);
    }
  }

  private agentDetailed(id: number): boolean {
    if (this.lodActive && !this.detailedAgents.has(id)) return false;
    if (this.sleepActive && !this.awakeAgents.has(id)) return false;
    return true;
  }

  private static readonly WAKE_HOPS = 2;
  private static readonly WAKE_HOLD = 0.4;
  private static readonly LATCH_WAKE = 0.5;

  private wakeGraphVersion = -1;
  private wakeGraphRoster = -1;
  private readonly wakeList: Agent[] = [];
  private wakeOff = new Int32Array(1);
  private wakeNei = new Int32Array(0);
  private wakeDist = new Int32Array(0);
  private wakeQ = new Int32Array(0);

  /**
   * Bodies and their wire adjacency in CSR, rebuilt only when topology moves.
   *
   * The version this was ported from rebuilt an array-of-arrays every frame —
   * the same per-frame rebuild of topology-derived scratch that flocking and
   * the force passes were each doing, and which cost more than the work it fed.
   */
  private refreshWakeGraph(): number {
    const list = this.wakeList;
    if (
      this.wakeGraphVersion === this.graph.version &&
      this.wakeGraphRoster === this.rosterVersion
    ) {
      return list.length;
    }
    // The shared list, so the wake graph is cut against exactly the order
    // every other pass indexes by.
    const src = this.forceList();
    list.length = 0;
    for (let i = 0; i < src.length; i++) list.push(src[i]);
    const n = list.length;
    if (this.wakeOff.length < n + 2) this.wakeOff = new Int32Array(n * 2 + 4);
    if (this.wakeDist.length < n) {
      this.wakeDist = new Int32Array(n * 2);
      this.wakeQ = new Int32Array(n * 2);
    }
    const off = this.wakeOff;
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
    if (this.wakeNei.length < edges) this.wakeNei = new Int32Array(edges * 2);
    const nei = this.wakeNei;
    const cursor = this.wakeQ;
    for (let i = 0; i < n; i++) cursor[i] = off[i];
    for (let k = 0; k < wires.length; k++) {
      const ia = wai[k];
      const ib = wbi[k];
      if (ia < 0 || ib < 0 || ia === ib) continue;
      nei[cursor[ia]++] = ib;
      nei[cursor[ib]++] = ia;
    }
    this.wakeGraphVersion = this.graph.version;
    this.wakeGraphRoster = this.rosterVersion;
    return n;
  }

  /**
   * Dual-rate islands. A taut, aged, calm net does not need SAT or a live rope
   * — the same cheap path the view LOD already uses when zoomed out — even
   * when it is on screen. Live ropes, loners, grabs, rewrites and fresh
   * latches stay NEAR. A contact against an already-awake body expands the
   * set; nothing here ever demotes a live rope.
   */
  private assignActivityLod(params: Params): void {
    this.awakeAgents.clear();
    this.sleepActive = params.nearBudget > 0;
    if (!this.sleepActive) {
      this.holdAwake.clear();
      this.hitWake.length = 0;
      return;
    }
    const n = this.refreshWakeGraph();
    if (n === 0) {
      this.hitWake.length = 0;
      return;
    }
    const at = this.slotIndex;
    const list = this.wakeList;
    const hop = this.wakeDist;
    const q = this.wakeQ;
    hop.fill(-1, 0, n);
    let qt = 0;
    const now = this.time;
    const sticky: number[] = [];

    /**
     * Seed by index. `hold` keeps it awake past the reason that woke it.
     *
     * Returns whether this call actually woke something new — which the
     * contact loop below needs, because it runs to a fixed point. A seed that
     * changes nothing must not count as progress: `hitWake` is filled during
     * the collision pass and drained here, so it can name a body that has
     * since been rewritten away and is no longer in the index at all. Reported
     * as progress, that pair alone spins the loop forever.
     */
    const seed = (id: number, hold: boolean): boolean => {
      const a = this.agents.get(id);
      if (!a) return false;
      const i = at[a.slot];
      if (i < 0) return false;
      if (hop[i] >= 0) {
        if (hold && !this.holdAwake.has(id)) sticky.push(id);
        return false;
      }
      if (hold) sticky.push(id);
      hop[i] = 0;
      q[qt++] = i;
      this.awakeAgents.add(id);
      return true;
    };

    if (this.grabbed) seed(this.grabbed.id, true);
    for (const id of this.rewriteFrozen) seed(id, true);
    for (const [id, until] of this.holdAwake) {
      if (until < now) this.holdAwake.delete(id);
      else seed(id, false);
    }
    // A body with nothing attached is a swimmer: it needs real collision, and
    // there is no island for it to be the still interior of.
    for (let i = 0; i < n; i++) {
      const a = list[i];
      if (!a.locked && !this.graph.isWired(a)) seed(a.id, true);
    }
    // A rope that is still being simulated, or a latch too young to have
    // settled, means the geometry there is not done moving.
    for (const w of this.graph.wires.values()) {
      if (w.ropePath !== 'span' || now - w.born < Sim.LATCH_WAKE) {
        seed(w.a.id, true);
        seed(w.b.id, true);
      }
    }

    /*
     * Contact propagation, to a fixed point. If exactly one side of a contact
     * is awake, wake the other — repeatedly, so a wake travels the whole chain
     * of touching bodies inside one frame rather than one link per frame. The
     * budget is the only thing that stops it, which is what makes `nearBudget`
     * a budget rather than a threshold.
     */
    const hits = this.hitWake;
    const budget = params.nearBudget;
    let grew = true;
    while (grew && this.awakeAgents.size < budget) {
      grew = false;
      for (let k = 0; k < hits.length; k += 2) {
        if (this.awakeAgents.size >= budget) break;
        const a = hits[k];
        const b = hits[k + 1];
        const aOn = this.awakeAgents.has(a);
        const bOn = this.awakeAgents.has(b);
        if (aOn === bOn) continue;
        if (seed(aOn ? b : a, true)) grew = true;
      }
    }
    hits.length = 0;

    /*
     * Spread out to WAKE_HOPS along the wires. Without the halo a body pops to
     * NEAR while its immediate neighbours stay coarse, and the constraint
     * between them is then being solved by two different solvers.
     */
    let qh = 0;
    while (qh < qt) {
      const u = q[qh++];
      const du = hop[u];
      if (du >= Sim.WAKE_HOPS) continue;
      const a0 = this.wakeOff[u];
      const a1 = this.wakeOff[u + 1];
      for (let k = a0; k < a1; k++) {
        const v = this.wakeNei[k];
        if (hop[v] >= 0) continue;
        hop[v] = du + 1;
        q[qt++] = v;
        this.awakeAgents.add(list[v].id);
      }
    }

    const holdUntil = now + Sim.WAKE_HOLD;
    for (const id of sticky) this.holdAwake.set(id, holdUntil);
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
     * The wake graph's CSR rather than an adjacency of its own.
     *
     * This used to build an array-of-arrays every frame — a push per wire
     * end into one of ten thousand arrays — over the same wires, in the same
     * order, that `refreshWakeGraph` already keeps as a flat CSR keyed on the
     * graph and the roster. Worse, it built it into `flockAdj`, which
     * `flock` caches on exactly that key and would happily have gone on
     * using: the two agreed only because they were building the same thing.
     */
    const n = this.refreshWakeGraph();
    const list = this.wakeList;
    const at = this.slotIndex;
    const off = this.wakeOff;
    const nei = this.wakeNei;
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
    this.collectRewriteFrozen();
    this.assignPhysicsLod(view);
    this.assignActivityLod(params);
    this.graph.syncRest(this.time, params, this.wireDetailed);
    this.graph.applyRopePaths(this.agents, this.w, this.h, this.time, params);
    this.graph.syncRopeShape(this.agents, this.w, this.h, this.wireDetailed);
    this.buildClearPairs(params);
    for (let sub = 0; sub < Sim.SUBSTEPS; sub++) {
      this.graph.solveWires(this.agents, params, h, this.time, this.rewriteFrozen, this.wireDetailed);
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
   * Finds wires whose chords cross a principal connection and eases the pair
   * apart. A principal wire is the one that matters: it is the redex, and a
   * wire lying across it keeps the two agents from ever meeting cleanly.
   *
   * The response is lateral — each crossed wire's endpoints slide away from the
   * other wire's line. Pulling the agents along their own wire instead (the
   * obvious "move forward") mostly just shortens the chord and leaves the
   * crossing where it was.
   */
  private uncrossPrincipals(params: Params, dt: number): void {
    const gain = params.uncross;
    if (gain <= 0 || dt <= 0) return;
    type Chord = {
      wireA: Agent;
      wireB: Agent;
      ax: number;
      ay: number;
      bx: number;
      by: number;
      principal: boolean;
    };
    const chords: Chord[] = [];
    for (const wire of this.graph.wires.values()) {
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      const sa = stemWorld(A, wire.a.slot, this.w, this.h);
      const sb = stemWorld(B, wire.b.slot, this.w, this.h);
      chords.push({
        wireA: A,
        wireB: B,
        ax: sa.x,
        ay: sa.y,
        bx: sb.x,
        by: sb.y,
        principal: wire.a.slot === 'p' || wire.b.slot === 'p',
      });
    }
    for (let i = 0; i < chords.length; i++) {
      for (let j = i + 1; j < chords.length; j++) {
        const S = chords[i];
        const T = chords[j];
        if (!S.principal && !T.principal) continue;
        if (
          S.wireA === T.wireA ||
          S.wireA === T.wireB ||
          S.wireB === T.wireA ||
          S.wireB === T.wireB
        ) {
          continue;
        }
        if (
          Math.max(S.ax, S.bx) < Math.min(T.ax, T.bx) ||
          Math.max(T.ax, T.bx) < Math.min(S.ax, S.bx) ||
          Math.max(S.ay, S.by) < Math.min(T.ay, T.by) ||
          Math.max(T.ay, T.by) < Math.min(S.ay, S.by)
        ) {
          continue;
        }
        if (!segmentsIntersect(S.ax, S.ay, S.bx, S.by, T.ax, T.ay, T.bx, T.by)) continue;
        // Draw the crossed principal's ends toward each other. A shorter chord
        // spans less, so it tends to slip out from under the wire lying over it.
        if (S.principal) this.reelIn(S, gain * dt);
        if (T.principal) this.reelIn(T, gain * dt);
      }
    }
  }

  /** Draw a wire's two agents toward each other along its own chord. */
  private reelIn(
    chord: { wireA: Agent; wireB: Agent; ax: number; ay: number; bx: number; by: number },
    k: number,
  ): void {
    const dx = chord.bx - chord.ax;
    const dy = chord.by - chord.ay;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) return;
    const ux = (dx / len) * k * 26;
    const uy = (dy / len) * k * 26;
    if (!poseHeld(chord.wireA)) {
      chord.wireA.vx += ux;
      chord.wireA.vy += uy;
    }
    if (!poseHeld(chord.wireB)) {
      chord.wireB.vx -= ux;
      chord.wireB.vy -= uy;
    }
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
    const LOCKED = store.locked;
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

      this.graph.solveWires(this.agents, params, h, this.time, frozen, this.wireDetailed);
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
    this.energy.configure(this.energyCell, this.energyAmbient);
    this.energy.seedGround();
  }

  /*
   * Last configure, kept so `pinWorld` can seed the ground at the size and
   * capacity the sliders are actually set to. `step` writes these every frame
   * before it harvests; `pinWorld` can run before the first of those.
   */
  private energyCell = 40;
  private energyAmbient = 1;

  /**
   * Tell the field what each channel is, from the sliders, every frame.
   *
   * Cheap — four numbers — and it has to be per frame because two of them are
   * live sliders. Energy is the odd one: it never decays, because it is a
   * quantity and the economy is supposed to conserve it, and it spreads at a
   * fraction of the signal rate so that local scarcity survives long enough
   * to forage against.
   */
  private tuneChannels(params: Params): void {
    const f = this.fields;
    f.decayRate[CH.energy] = 0;
    f.diffuseRate[CH.energy] = Math.max(0, params.energyDiffuse);
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
    if (dev && (await genomeGpu.init(dev))) this.genomeOnGpu = true;
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
    plan.build(this.agents.values(), this.agentStore, this.energy);
    Sim.phase('gpu:plan');
    // Three ports a body, plus whatever died, farmed, spilled or was refunded
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
    let nDep = 0;
    /*
     * Two shared constants rather than `slotsFor(a.kind)`, which builds a
     * fresh array per body and then a fresh iterator to walk it. Five thousand
     * bodies is ten thousand objects a frame for a value that has exactly two
     * possible answers.
     */
    const scratch = this.portScratch;
    for (let i = 0; i < n; i++) {
      const a = list[i];
      if (a.locked) continue;
      const ports = a.kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
      for (let pi = 0; pi < ports.length; pi++) {
        const slot = ports[pi];
        // See `deposit`: a voice carries whether or not the port is attached,
        // an aux marker does not.
        const free = this.graph.isFreeAt(a.id, slot);
        if (slot !== 'p' && !free) continue;
        const w = portWorldInto(a, slot, this.w, this.h, scratch);
        const o = nDep * dStride;
        dep[o] = w.x;
        dep[o + 1] = w.y;
        dep[o + 2] = 0; // a density; `depositScale` is already in `amt`
        if (slot === 'p') {
          const eo = a.slot * 4;
          dep[o + 4] = amt * EMITS[eo];
          dep[o + 5] = amt * EMITS[eo + 1];
          dep[o + 6] = 0;
          dep[o + 7] = amt * EMITS[eo + 3];
        } else {
          dep[o + 4] = 0;
          dep[o + 5] = 0;
          dep[o + 6] = 0;
          dep[o + 7] = amt * 0.7;
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
      dep[o] = pend[k * 3];
      dep[o + 1] = pend[k * 3 + 1];
      dep[o + 2] = 1;
      dep[o + 4] = 0;
      dep[o + 5] = 0;
      dep[o + 6] = pend[k * 3 + 2];
      dep[o + 7] = 0;
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
      packTaste(pro, o + 8, TASTE_OF, sl, groundScale);
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
      const rooms = fieldGpu.roomData;
      for (let e = 0; e < plan.nEntries; e++) rooms[e] = plan.rooms[e];
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
        catCh: FERTILISE_CH,
        gamma: params.fertilise,
      },
      {
        u: CH.conP,
        v: CH.dupP,
        feed: params.reactFeed,
        kill: params.reactKill,
        dt,
      },
      {
        ch: CH.energy,
        blocks: plan.nBlocks,
        entries: plan.nEntries,
        uptakeCap: params.uptakeVmax * dt,
        uptakeKs: params.uptakeKs,
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
    const sScale = 1 / SENSE_SCALE;
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
   */
  private dropFieldGpu(): void {
    this.harvestPending = false;
    this.genomePending = false;
    this.fieldOnGpu = false;
    this.genomeOnGpu = false;
    this.steerFromSamples = false;
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
      !genomeGpu.submit(samples, n, nNei, this.groundScale, CH.energy, {
        rate: params.learnRate,
        critic: params.learnCritic,
        trace: params.learnTrace,
        discount: params.learnDiscount,
        maxWeight: CHEM_TASTE_MAX,
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
          up[o + k] = P[ps + k];
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
        const w = src[o + k];
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
      const frozenEnds = frozen.has(A.id) || frozen.has(B.id);
      const skip = (poseHeld(A) && poseHeld(B)) || frozenEnds;
      const full = this.wireSimulatesRope(w) && w.nodes.length > 0;
      const stiff = this.graph.stiffnessOf(w, this.time, params);
      let flags = 0;
      if (full) flags |= WF_FULL;
      if (skip) flags |= WF_SKIP;
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
    // A contact is the one wake signal that cannot be derived from topology:
    // it is how a moving body tells sleeping tissue that it is coming.
    if (this.sleepActive) this.hitWake.push(A.id, B.id);
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
      // Bound discs are fatter than SAT triangles. A wired pair is already
      // held by the chord; colliding them shoves the net apart of the rest.
      if (!detailed && this.graph.sharesWire(A.id, B.id)) return;
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

  /** One drag law for bodies; the rope is damped inside the substep loop. */
  private dampVelocities(params: Params, dt: number): void {
    const linKeep = Math.exp(-Math.max(0, params.drag) * dt);
    const angKeep = Math.exp(-Math.max(0, params.angDrag) * dt);
    const store = this.agentStore;
    const LOCKED = store.locked;
    const PINNED = store.pinned;
    const VX = store.vx;
    const VY = store.vy;
    const OMEGA = store.omega;
    for (const agent of this.agents.values()) {
      const s = agent.slot;
      if (LOCKED[s] || PINNED[s]) continue;
      VX[s] *= linKeep;
      VY[s] *= linKeep;
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

  /**
   * Being hit costs energy, proportional to how deeply the pair interpenetrate.
   *
   * Damage, not death: it returns whoever it pushed past the floor so the
   * caller kills them the same way starvation does, and every lethal thing in
   * the sim keeps going through the one tank. Overlap is the severity proxy —
   * a harder collision penetrates further before the solver can resolve it —
   * and it is what `LiveContact` already carries for the audio.
   */
  private readonly bruised: number[] = [];

  private contactDamage(params: Params, dt: number): number[] {
    const k = params.contactCost;
    const dead = this.bruised;
    dead.length = 0;
    if (k <= 0 || dt <= 0) return dead;
    const hurt = k * dt;
    for (const c of this.contacts.values()) {
      const A = this.agents.get(c.agentA);
      const B = this.agents.get(c.agentB);
      const bite = hurt * c.overlap;
      if (A && !A.locked && A.extra > A.debtCap) {
        A.extra -= bite;
        if (A.extra <= A.debtCap) dead.push(A.id);
      }
      if (B && !B.locked && B.extra > B.debtCap) {
        B.extra -= bite;
        if (B.extra <= B.debtCap) dead.push(B.id);
      }
    }
    return dead;
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
      // Materialised by `updateState`; `CH.energy` is masked here rather than
      // in the vector, because the vector is the budget and the deposit path
      // is the thing that must never see the ground. See `effEmit`.
      const eo = sl * 4;
      for (let c = 0; c < 4; c++) emit[i * 4 + c] = c === CH.energy ? 0 : EMITS[eo + c];
      if (!freeFresh) free[i] = this.freePortMask(a);
    }
    Sim.phase('scent:bodyPack');
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
   * The aux marker keeps its gate, because it is not a voice. `CH.aux` means
   * "there is somewhere to attach here", and that has to stay false when
   * there is not, or the one kind-independent signal in the field stops being
   * true. Emitting it from a filled port would advertise a socket that is not
   * there and every latch-seeking body in range would come and find nothing.
   *
   * Its magnitude is still the hardcoded 0.7, and the plan was to make that a
   * gene — `E[aux][BOUND]`, so a body advertises its sockets as loudly as its
   * lineage has learned to. It is not done here because the clean version is
   * not obvious: a body's ch3 voice is already emitted from its principal, so
   * a second genetic ch3 term at the port positions is either double-counting
   * or a separate gene, and "separate gene" is not the subsumption it was
   * billed as. The position genuinely cannot be folded in — "the socket is
   * *here*" is information a body-centred voice cannot carry — so what is left
   * to move is one scalar, and it can wait for a reason to be a particular
   * shape rather than being changed because it was on a list.
   */
  private deposit(params: Params): void {
    const EMITS = this.agentStore.emitAll;
    const p = this.portScratch;
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      const slots = agent.kind === 'era' ? ERA_SLOTS : NODE_SLOTS;
      for (let k = 0; k < slots.length; k++) {
        const slot = slots[k];
        if (slot === 'p') {
          portWorldInto(agent, slot, this.w, this.h, p);
          const eo = agent.slot * 4;
          for (let ch = 0; ch < 4; ch++) {
            if (ch === CH.energy) continue;
            const w = EMITS[eo + ch];
            if (w !== 0) this.fields.deposit(ch, p.x, p.y, params.deposit * w);
          }
        } else if (this.graph.isFreeAt(agent.id, slot)) {
          portWorldInto(agent, slot, this.w, this.h, p);
          this.fields.deposit(CH.aux, p.x, p.y, params.deposit * 0.7);
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
    return (
      t[o] * this.fields.sample(0, x, y) +
      t[o + 1] * this.fields.sample(1, x, y) +
      t[o + 2] * this.fields.sample(2, x, y) * this.groundScale +
      t[o + 3] * this.fields.sample(3, x, y)
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
        flags[i] = (flags[i] & 1) | (STUN_OF[sl] > 0 ? 4 : 0);
        // Under a force block packPose has already written these; without one
        // nothing else does, and scale moves every frame as bodies grow.
        if (packed) {
          kinds[i] = KIND_OF[sl];
          sc[i] = SCALE_OF[sl];
        }
        drive[i] = DRIVE_OF[sl];
        packTaste(taste, i * 4, TASTE_OF, sl, groundScale);
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
        flags[i] = (pFree ? 1 : 0) | (STUN_OF[sl] > 0 ? 4 : 0);
        if (packed) {
          kinds[i] = KIND_OF[sl];
          sc[i] = SCALE_OF[sl];
        }
        drive[i] = DRIVE_OF[sl];
        packTaste(taste, i * 4, TASTE_OF, sl, groundScale);
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
    sp[0] = params.faceRadius;
    sp[1] = params.snapRadius;
    sp[2] = params.snapArc;
    sp[3] = params.faceAttract;
    sp[4] = params.snapWell;
    sp[5] = params.sensorAngle;
    sp[6] = params.sensorDist;
    sp[7] = params.sense;
    sp[8] = params.turnRate;
    sp[9] = params.stepSpeed;
    sp[10] = params.swimTau;
    sp[11] = params.swimNoise;
    // 12 and 13 are dead: they carried `attractStrong`/`attractMedium`, which
    // are seeds read once by `seedChem` and by nothing in the solver. The slots
    // stay reserved; see the note in solver.c.
    sp[14] = SENSE_SPAN;
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
      const off = this.wakeOff;
      const nei = this.wakeNei;
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
     * The wake graph's CSR, which is the same wires in the same order on the
     * same key. This pass used to build an array-of-arrays and then flatten
     * it into exactly this shape to hand to wasm — a push per wire end and
     * then a copy of the lot — for a structure two other passes were already
     * keeping.
     */
    this.refreshWakeGraph();
    const off = this.wakeOff;
    const nei = this.wakeNei;

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
    // Not the rest-length sit: the collapse hauls them the rest of the way.
    // Still skip a cable that has barely started to take, so the pull is a
    // close and not a fling.
    return len <= params.wireMinRest * 1.3;
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
   * The plan was built then, from `extra` as it stood at the end of that
   * frame, and nothing between there and here touches `extra` — harvest is
   * the first thing in `endFrame` that does. So the room each body was
   * credited against is exactly the room it would have had if the take had
   * happened now: the pipelining is exact, not an approximation anyone has to
   * budget error for.
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
    const CAP = store.energyCap;
    const IDS = store.id;
    for (let e = 0; e < plan.nEntries; e++) {
      const g = got[e];
      if (!(g > 0)) continue;
      const slot = plan.slots[e];
      if (IDS[slot] !== plan.ids[e]) continue;
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
     * `docs/energy-chemistry-plan.md` §8. Latching is proximity plus an arc
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

    /*
     * And what a body wants, not only what it is short of.
     *
     * `rescueNeed` and `redexNeed` are both shortfalls — about to die, about
     * to reproduce. Neither says anything about where the net should be, so a
     * net had no way to spend energy on going somewhere. This is the third
     * claim: a body that can swim asks in proportion to how much it likes
     * what it can smell, which is its `trail` read through its own taste
     * weights, so the ask is already in the currency of that body's genome.
     *
     * `trail` is last frame's reading, written by `steer`. A frame of latency
     * in an appetite is not a thing anything can perceive, and it saves
     * sampling the field a second time for every body in the pond.
     *
     * Clamped at zero because taste is signed: a body that is repelled by
     * everything around it has a negative trail, and that is a reason to
     * leave rather than a reason to be fed.
     */
    if (params.forageAsk > 0) {
      for (let i = 0; i < list.length; i++) {
        const a = list[i];
        if (LOCKED[a.slot] || !this.graph.isFreeAt(a.id, 'p')) continue;
        const trail = a.trail;
        seedRequest(a, params.forageAsk * (trail > 0 ? trail : 0));
      }
    }
    const adj = this.wireAdjacency();
    Sim.phase('pulse:seed');
    spreadRequestsFast(list, this.agentStore, adj);
    Sim.phase('pulse:spread');
    flowChargesFast(list, this.agentStore, adj, (from, to, amount) => this.recoil(from, to, amount));
    Sim.phase('pulse:flow');
    this.updateState(list, adj, params);
    Sim.phase('state');
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
  /** `phi'` per state dim, and the inputs each learned weight multiplies. */
  private readonly learnPost = new Float64Array(STATE_DIMS);
  private readonly learnPre = new Float64Array(IN_DIMS + 2 * STATE_DIMS);

  private updateState(list: Agent[], adj: WireAdjacency, params: Params): void {
    const n = list.length;
    if (n === 0) return;
    if (this.genomeOnGpu) {
      // Learning does not happen on this path yet: the shader computes `h`
      // and this only unpacks it. See `docs/plasticity-plan.md` phase 4 —
      // it wants a binding merge before the learned block will fit.
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
    const sScale = 1 / SENSE_SCALE;
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
    const PLASTIC = store.plasticAll;
    const TRACE = store.traceAll;
    const CRITIC = store.criticAll;
    const PREV_V = store.prevValue;
    const PLASTIC_ON = store.plasticOn;
    const post = this.learnPost;
    const pre = this.learnPre;
    const learn = params.learnRate > 0;
    const etaM = params.learnRate;
    const etaC = params.learnCritic;
    const lam = params.learnTrace;
    const discount = params.learnDiscount;
    const MAXW = CHEM_TASTE_MAX;
    for (let i = 0; i < n; i++) {
      const slot = slotOf[i];
      const g = slot * CHEM_LEN;

      /*
       * Sampling is skipped entirely unless this genome reads the field. A
       * bilinear sample is four scattered reads into a sixteen-megabyte array
       * that will not be in cache, once a body, for a value nearly every
       * genome multiplies by zero — `Wx`'s sense columns seed to zero, because
       * the one seeded pathway runs through `IN_DEMAND`. The inputs are zeroed
       * rather than left reading a stale `SENSE`, so behaviour never depends
       * on when a body last happened to sample.
       */
      if (READS[slot]) {
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
        x[IN_SENSE] = SENSE[so];
        x[IN_SENSE + 1] = SENSE[so + 1];
        x[IN_SENSE + 2] = SENSE[so + 2];
        x[IN_SENSE + 3] = SENSE[so + 3];
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
      emitVector(CHEM, g, H, ho, EMITS, slot * 4);
      tasteVector(CHEM, g, H, ho, TASTES, slot * 4);

      CRUISE[slot] = clamp(headAt(CHEM, g, L_OUT, L_BASE, 0, H, ho, S) * HEAD_SCALE.cruise, 0, 180);
      TURN[slot] = clamp(headAt(CHEM, g, L_OUT, L_BASE, 1, H, ho, S) * HEAD_SCALE.turn, 0, 8);
      FA[slot] = clamp(headAt(CHEM, g, F_OUT, F_BASE, 0, H, ho, S) * HEAD_SCALE.align, -8, 16);
      FS[slot] = clamp(headAt(CHEM, g, F_OUT, F_BASE, 1, H, ho, S) * HEAD_SCALE.sep, -60, 120);
      TT[slot] = clamp(headAt(CHEM, g, P_OUT, P_BASE, 0, H, ho, S) * HEAD_SCALE.thrust, 0, 1);
      TR[slot] = clamp(headAt(CHEM, g, P_OUT, P_BASE, 1, H, ho, S) * HEAD_SCALE.recoil, 0, 200);

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
       * The cost is the body's **own** tank: `x4` is `IN_FULL`, clamped to
       * [0,1], so `x4 - 1` is zero when full and -1 when empty. Local, by
       * decision. `x6` is the same shortfall relaxed over the wire graph and
       * is the other candidate teacher — it would make the net's condition
       * the thing a body learns about rather than its own — and swapping
       * them is this one line, since both are already in `x`.
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
        const dlt = x4 - 1 + discount * value - PREV_V[slot];
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

  private tickRewrites(params: Params, dt: number): void {
    const done: Rewrite[] = [];
    for (const rw of this.rewrites) {
      if (advanceRewrite(rw, this.agents, this.w, this.h, dt)) done.push(rw);
      // The wire is what pulls them together, so shorten it in step with the
      // pull. It retracts into the pair and is gone by the time they touch,
      // and because it stays taut on the way its pitch rises instead of
      // sagging the way a slackening rope's does.
      const pull = clamp(rw.t / PULL_END, 0, 1);
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
