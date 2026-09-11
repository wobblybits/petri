import { boundRadius, discRadius, createAgent, ERA_RADIUS, inSnapArc, momentOfInertia, momentOfInertiaAt, poseHeld, poseHeldAt, syncHeadingCosSin, ERA_SLOTS, NODE_SLOTS, portWorld, portWorldInto, stemRoot, stemOffset, CHEM_LEN, CHEM_SPECIES, PLASTIC_LEN, CRITIC_LEN, LEARN_CRITIC, LEARN_PREV_V, LEARN_TRACE, STATE_DIMS, effTaste, flockGain, stemOffsetInto, stemWorld, stemWorldInto, type Agent, type AgentKind, type PortSlot } from './agents.ts';
import { AgentStore, CODE_KIND } from './agent-store.ts';
import { queryHit, queryDiscHit, SLOP, type Hit } from './collide.ts';
import { closestTOnSegment, WIRE_RADIUS, wireBowBudget, bounceOffDisk } from './geom.ts';
import { PairGrid } from './grid.ts';
import { CHAIN_MASS, contactMechanics, portExitAngle, solveContact } from './chain.ts';
import { CH, CHANNELS, FERTILISE_CH, FIELD_CELL, FIELD_CELLS, Fields, worldBoundRadius } from './fields.ts';
import { Graph, ropeIsLive, wrapPos, type Wire } from './graph.ts';
import { LarvalWindow } from './larval.ts';
import type { Params } from './params.ts';
import { advanceRewrite, beginRewrite, commitRewrite, detectRule, PULL_END, rewriteHandoffStems, CHEM_TASTE_MAX, TRAIT_KEYS, type Rewrite } from './rewrite.ts';
import { agentValue, BODY_VALUE, deathYield, EnergyGrid, flowChargesFast, harvestSlotsFast, HarvestPlan, payToward, rescueNeed, redexNeed, rewriteCost, rewriteShareOf, PENDING_STRIDE, HARVEST_GOT, HARVEST_STRIDE, rewriteYield, seedRequest, settlePool, stakeMet, relaxRequestsFast, spreadRequestsFast, tickUpkeepFast, WireAdjacency } from './energy.ts';
import { audio } from './audio/engine.ts';
import type { CollisionEvent, LiveContact, PanView, RewriteEvent } from './audio/types.ts';
import { AGENT_BAND, LOD_FAR, LodSelector, agentKey, apparentPx, onScreen, wiresDrawable } from './audio/lod.ts';
import { farGpu } from './gpu/far-gpu.ts';
import { fieldGpu } from './gpu/field-gpu.ts';
import { genomeGpu } from './gpu/genome-gpu.ts';
import { FAR, FAR_STRIDE, packFarWire } from './gpu/far-kernel.ts';
import { KIND_CON, KIND_DUP, KIND_ERA, nativeSolver, ND, NODE_STRIDE, HIT, HIT_STRIDE, WF_FULL, WF_HOLD, WF_SHAPE, WF_SKIP, WIRE_NEAR_STRIDE, WN } from './native/solver.ts';
import { angleDelta, clamp, wrapAngle, wrapDeltaVec } from './wrap.ts';
import { updateState } from './state.ts';
import { refreshExpression, runDigestion, runExcretion, scentMints } from './body-chemistry.ts';
import { advanceMetabolism, dampVelocities } from './metabolism.ts';

/**
 * Normalised sensor asymmetry `(right - left) / (|right| + |left|)`, in [-1, 1],
 * that earns a full-arc turn; below it the turn ramps. Passed to the solver
 * through `sparams` so the constant is stated once on both sides of the wasm wall.
 */
export const SENSE_SPAN = 0.1;

/**
 * What a body smells at a point: its taste weights dotted with the four channels.
 * `groundScale` is `1 / cellCap`, so the energy channel reads as "how full is
 * it here", 0 to 1, and a taste gene means the same at every `ambientEnergy`.
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

/** A contact pair as one number. Ids below 2^26 pack exactly inside the safe-integer range. */
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
 * The Hill coefficient as both field paths use it: plain Monod (1) for anything
 * not above zero, resolved once so the shader and `runHarvestPlan` agree.
 */
function hillOf(params: Params): number {
  return params.hillN > 0 ? params.hillN : 1;
}

/**
 * One body's four taste weights, laid out at `out[at..]` for a consumer of the
 * field, with the ground scale folded in here so the JS dot product, the wasm
 * pack and the GPU probe steer alike.
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
   * reference JS implementation; the two agree to a few parts in 10^4, not
   * bit-for-bit (the C side is f32 with its own libm).
   */
  static nativeForces = true;

  /**
   * Check, at the solver, that the packed bodies still describe the agents.
   * The force block hands its packed copy to the solver on the grounds that
   * nothing between the force passes and the solve moves a body; if that stops
   * being true the pose goes stale silently. Debug aid, off in normal running.
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
   * Speed cap on a held agent. The hold is positional, so its correction
   * reappears as velocity at 1/h and the agent would rocket away on release.
   */
  private static readonly GRAB_MAX_SPEED = 160;

  w: number;
  h: number;
  coverW: number;
  coverH: number;
  time = 0;
  nextId = 1;
  /**
   * Bumped whenever the agent roster changes. Caches keyed on list index need
   * this as well as `graph.version`: deleting one body renumbers every body
   * after it without touching a wire.
   */
  rosterVersion = 0;
  /** What a body is worth dead, latched from `params.bodyValue` each frame; `kill` has no `Params`. */
  bodyValue = BODY_VALUE;

  /**
   * Register a roster change made without going through `spawn` or `kill`
   * (`commitRewrite`, the lambda loader). Every cache keyed on the roster is
   * silently stale until this is called.
   */
  noteRosterChange(): void {
    this.rosterVersion++;
  }
  /** Identifies this Sim to the shared wasm solver's flocking cache. */
  private readonly simId = nextSimId++;

  /**
   * Which FAR solver to prefer when both can take a frame. 'auto' picks on body
   * count; 'on' and 'off' force it either way. There is no WebGPU under Node,
   * so the GPU path is only ever verified in a browser against the twin.
   */
  static farGpuMode: 'auto' | 'on' | 'off' = 'auto';

  /**
   * How far under wasm's capacity 'auto' has to fall before it hands the FAR
   * solve back; the switch up is the capacity itself. The two solvers do not
   * agree to the digit (Jacobi on the GPU, Gauss-Seidel in the twin), so
   * flapping on the cap would show as a shimmer.
   */
  static FAR_GPU_RELEASE = 0.95;

  /** Which side of the hysteresis band 'auto' is currently latched to. */
  private farGpuLatched = false;

  /** Per-phase frame timings. Set to a Map to collect; phases accumulate until it is cleared. */
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
   * step off; reading it live would let one body's need race several hops in
   * a pass, in roster order.
   */
  private requestPrev = new Float64Array(0);

  /** Per-agent unmet need this frame, rebuilt by `pulseRequests`. */
  private readonly wireAdj = new WireAdjacency();
  /**
   * Which connected component each body is in, as a position in
   * `forceList()`, keyed on the graph and roster versions. Consumers only
   * compare roots for equality; `Graph.componentIds` stays for the audio
   * shards, which want ids.
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
  /** Bodies currently overlapping, keyed by canonical `lo:hi` id pair. */
  contacts = new Map<number, LiveContact>();
  /** Momentum lost to sound this frame, applied once after the substeps. */
  private radiated = new Map<number, { x: number; y: number }>();
  /**
   * Agent held by the pointer, and where. Solved as a constraint inside the
   * substep loop; assigning a pose directly would drag whole nets through
   * their joints at 1/h velocity.
   */
  grabbed: { id: number; x: number; y: number } | null = null;
  /** Agents in an active rewrite — their incident ropes are kinematic. */
  private rewriteFrozen = new Set<number>();
  /**
   * Energy banked against a ready redex, keyed by wire id. See `accrueRedexes`.
   * Real energy in neither body nor ground, so any pond total must count it
   * (`escrowTotal`).
   */
  private readonly escrow = new Map<number, RedexEscrow>();
  /**
   * Who grazes which block this frame, shared by both field paths. On the GPU
   * path it is built at the end of a frame and spent at the top of the next
   * — see `creditHarvest`.
   */
  private readonly harvestPlan = new HarvestPlan();
  /** Reused by the GPU deposit pack; see `portWorldInto`. */
  private readonly portScratch = { x: 0, y: 0 };
  /** True once a GPU harvest has been dispatched and not yet credited. */
  private harvestPending = false;
  /**
   * Ask for the GPU field to be copied back into `fields.data` each frame.
   * Opt-in: a sixteen-megabyte copy wanted only by the debug overlays, which
   * `render.ts` sets while one is showing. Ignored when the field is on the CPU.
   */
  wantFieldReadback = false;
  /**
   * True once the genome pass runs on the GPU. Strictly downstream of
   * `fieldOnGpu`: it reads the field probe's output buffer for its sense
   * inputs. When on, `updateState` unpacks instead of computing.
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
   * Slots whose learning the CPU needs back from the device: a rewrite's two
   * parents, whose learning `inheritChem` consolidates into their children.
   * `startRewrites` marks the pair; the rows are back long before commit.
   */
  private readonly learnWanted: number[] = [];
  private readonly learnAsked: number[] = [];
  /**
   * Whether the solver steers from readings handed to it (the GPU probe's)
   * rather than sampling a field it holds. Mirrors the process-wide
   * `nativeSolver.useSamples` for this Sim.
   */
  private steerFromSamples = false;
  /** Which topology and roster `wireAdj` describes. See `wireAdjacency`. */
  private adjGraphVersion = -1;
  private adjRosterVersion = -1;
  /** Principal pairs ready to rewrite this frame. See `collectReadyRedexes`. */
  private readonly readyRedexes: Wire[] = [];
  /**
   * Running counts of the events selection acts through, since `clear()`.
   * Turnover, which `census()` cannot give. Read by the experiment harness.
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
   * Whichever physics LOD tier this agent last settled into. A pure read:
   * `peek` does not feed the hysteresis the way `tier()` does. Never tiered
   * reads as NEAR, the safe default.
   */
  isFarTier(agentId: number): boolean {
    return this.physLod.peek(agentKey(agentId)) === LOD_FAR;
  }


  /** Eased centre of mass. Rewrites delete agents, which jumps the true COM. */
  private home: { x: number; y: number } | null = null;

  /**
   * Cells a side of the field a new Sim gets when the constructor is not told.
   * The test suite sets a quarter: the cell stays the same size, so it is a
   * smaller dish, not a coarser one.
   */
  static defaultFieldCells = FIELD_CELLS;

  /** Time from arrival to first latch, and how many never got there. Cumulative over the run. */
  readonly larval = new LarvalWindow();

  /**
   * A body's first wire, in the terms `larval` counts. `arrivedAt` doubles as
   * the flag: `-1` means "has latched", so a re-latch is not counted twice.
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

  /** Size of the visible area, in world units; auto-spawn places new bodies within a fraction of it. */
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
    // Before `clear`, which blanks port occupancy on whichever store it holds.
    this.graph.useStore(this.agentStore);
    this.graph.clear();
    this.rewrites = [];
    // Not `refundEscrows`: bodies and ground are both thrown away below.
    this.escrow.clear();
    this.fields.clear();
    this.energy.clear();
    this.time = 0;
    // A fresh store starts every slot at `arrivedAt = 0`; the histogram goes with it.
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
     * Everything the GPU path carries across a frame boundary: a harvest or
     * genome dispatched against the old roster would be paid to the new
     * pond's bodies in the same slots (ids restart at one, so the id check
     * cannot tell them apart), and `fields.clear()` only zeroes the CPU mirror.
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
   * A body that has run a whole unit into debt is gone. What it was made of
   * goes back to the ground it died on: a full body's worth plus whatever it
   * still held, onto the grid rather than into the net.
   */
  kill(id: number): void {
    const agent = this.agents.get(id);
    if (!agent) return;
    const slot = this.agentStore.slotFor(id);
    if (slot !== undefined && this.agentStore.arrivedAt[slot] >= 0) this.larval.diedAlone();
    this.energy.addAt(agent.x, agent.y, deathYield(agent, this.bodyValue));
    // And whatever it had swallowed and not yet digested, as itself; a corpse that kept its gut would be a leak.
    if (slot !== undefined) this.spillGut(slot, agent.x, agent.y);
    this.graph.detachAgent(id);
    this.agents.delete(id);
    this.agentStore.release(id);
    this.rosterVersion++;
    this.tally.died++;
    if (this.grabbed?.id === id) this.grabbed = null;
  }

  /** Its own, not `excreteScratch`: a death can land inside any pass. */
  private readonly deathScratch = new Float64Array(CHEM_SPECIES);

  /** Put one body's undigested holdings back on the ground it is standing on. */
  private spillGut(slot: number, x: number, y: number): void {
    const GUT = this.agentStore.gut;
    const go = slot * CHEM_SPECIES;
    const w = this.deathScratch;
    let held = 0;
    for (let c = 0; c < CHEM_SPECIES; c++) {
      const v = GUT[go + c];
      w[c] = v > 0 ? v : 0;
      held += w[c];
    }
    if (held <= 0) return;
    GUT.fill(0, go, go + CHEM_SPECIES);
    this.energy.addSpeciesAt(x, y, w);
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

  /** Everything before the solve, shared by the two twins below. Returns the clamped frame step. */
  private openFrame(dt: number, params: Params, view: PanView | null | undefined): number {
    Sim.phaseStart();
    const t = this.beginFrame(dt, params);
    this.collectRewriteFrozen();
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
     * then the GPU again for the frames wasm turned down (over its body cap
     * the GPU is the only real solver left), then the TS twin.
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
    // The store stamps a new slot with this, so every creation path records an arrival. See `larval.ts`.
    this.agentStore.now = this.time;
    this.trackHome(t);
    /*
     * One centre for the whole world grid, pinned once at the first centre of
     * mass (unless a preset already pinned) and never moved, so the wall has
     * something to bounce off. Latched before the pin: `pinWorld` lays the
     * ground down at whatever capacity these say, and can fire this frame.
     */
    this.energyCell = params.energyCell;
    this.energyAmbient = params.ambientEnergy;
    // `kill` is public and takes no `Params`, so this must be a fact about the Sim before anything can die.
    this.bodyValue = params.bodyValue;
    const h = this.home;
    if (h && !this.worldPinned) this.pinWorld(h.x, h.y, params);
    this.contactAudioNow.clear();
    this.contacts.clear();
    this.radiated.clear();
    Sim.phase('beginFrame:setup');

    // The force passes share one packed copy of the bodies.
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
    // Left open on purpose: nothing between here and the solver moves a body,
    // so the solver inherits the packed bodies. Whoever consumes them closes
    // it; `syncForces` is the backstop.
    void block;
    return t;
  }

  private endFrame(params: Params, t: number): void {
    // Backstop: a path that did not end in a native solve still owes the agents their velocities.
    this.syncForces();
    Sim.phase('syncForces');
    this.applyRadiationLoss();
    advanceMetabolism(this, params, t);
    dampVelocities(this, params, t);
    Sim.phase('damp');

    this.graph.refreshLengths(this.agents, this.w, this.h, this.rewriteFrozen, this.wireDetailed);
    this.snapTautWires(params);
    Sim.phase('refreshLengths');
    // Earn, distribute, spend, then pay rent. Rent-last is only safe
    // because `EXTRA_CAP` has headroom over `REWRITE_SHARE`.
    this.energy.configure(params.energyCell, params.ambientEnergy);
    // GPU path: the shader grazed at the end of the last frame; pay it out
    // before upkeep so a body cannot die owing itself a served meal.
    if (this.fieldOnGpu) this.creditHarvest();
    else {
      // `uptakeVmax` at zero is take-what-fits; above it, one mouthful a
      // frame shared across the four species. Must match `field.wgsl`'s `harvest`.
      harvestSlotsFast(this.agents.values(), this.agentStore, this.energy, this.harvestPlan, {
        cap: params.uptakeVmax * t,
        ks: params.uptakeKs,
        yDirect: params.yDirect,
        yEra: params.yEra,
        hillN: hillOf(params),
        gutSize: params.gutSize,
      });
      // A metered mouthful lands in a gut, so something now has to digest it.
      if (this.harvestPlan.metered && this.harvestPlan.nEntries > 0) this.gutLive = true;
    }
    Sim.phase('harvestSlots');
    this.latchPass(params);
    Sim.phase('snap');
    // Which principal pairs are ready is asked once, here, and read by the
    // three passes after it; the set cannot change between them.
    this.collectReadyRedexes(params);
    this.pulseRequests(params);
    this.accrueRedexes(params);
    this.startRewrites(params);
    this.tickRewrites(params, t);
    Sim.phase('rewrites');
    // The body reaction table, in three passes: express, digest, excrete.
    this.expressed = refreshExpression(this, params, t);
    this.gutLive = runDigestion(this, params, t, this.gutLive);
    runExcretion(this, params, t);
    Sim.phase('excrete');
    for (const id of tickUpkeepFast(this.agents.values(), this.agentStore, t, params.upkeep, this.energy, {
      rentBack: params.upkeepExcrete,
      eraRatio: params.eraUpkeepRatio,
      // The same condition `refreshExpression` returns early on.
      expressed: this.expressed,
    })) {
      this.kill(id);
    }
    Sim.phase('upkeep');
    // When the GPU owns the field, `gpuFieldStep` at the end of `stepAsync`
    // does the deposit, diffusion and decay instead of the branch below.
    // `tuneChannels` runs on both paths: `field-gpu.ts` reads the
    // `fields.diffuseRate` / `fields.decayRate` arrays it writes.
    this.tuneChannels(params);
    if (!this.fieldOnGpu) {
      if (scentMints(params) && !this.scentWriteNative(params)) this.deposit(params);
      Sim.phase('scentWrite');
      this.fields.diffuse(params.diffuse);
      Sim.phase('field:diffuse1');
      this.fields.diffuse(params.diffuse * 0.65);
      Sim.phase('field:diffuse2');
      // After the spreading and before the decay; `decay` then acts on both
      // channels on top of Gray-Scott's own `feed + kill`.
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
   * Pack the pose and inertia every force pass needs, plus the wire
   * endpoints. Returns the packed count, or -1 when the scene will not fit
   * and the caller should stay on the JS path.
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
    // When a block owns the pack the bodies are already there; rewriting
    // them would discard what the earlier passes accumulated.
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
    // Not cacheable: `wiresNear` and `wires` are two views over the same wasm
    // buffer and the FAR and NEAR solves write their own layout through it
    // every frame.
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
      // Critically damped: a bare proportional torque windmills.
      const damp = 1.8 * Math.sqrt(gain * I);
      // Aux ports aim slightly off their neighbour, toward their own side of
      // the body, so the uncrossed pose is the stable one.
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
   * Open a shared copy for the run of WASM force passes. Only when every pass
   * in that run is native: a JS pass in the middle would read stale
   * velocities and have its own writes overwritten on unpack.
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
   * them.
   */
  private packPose(list: Agent[]): boolean {
    const bodies = nativeSolver.bodies;
    const bm = nativeSolver.bodyMass;
    const invI = nativeSolver.invInertia;
    const kinds = nativeSolver.kind;
    const sc = nativeSolver.scale;
    if (!bodies || !bm || !invI || !kinds || !sc) return false;
    if (!nativeSolver.canNear(list.length, 0, 0)) return false;
    // Straight out of the store, not through the flyweight: this is the
    // largest JavaScript phase in the frame.
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
      // Radius is absent: no force pass reads it.
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
   * The bodies, in the order every pass indexes them by: roster insertion
   * order, so ascending id. Rebuilt only when `rosterVersion` moves, so every
   * writer of the roster must bump it (`spawn`, `kill`, `clear`,
   * `noteRosterChange`). No pass may sort or filter the shared array:
   * `wireAdjacency` and `unpackGenome` assume this order.
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
   * The latch pass, handed the resolved wire endpoints so it can refuse a
   * latch whose chord would cross a standing wire. Tests call this rather
   * than `graph.snap` so there is one path.
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
  /** The same two endpoints as positions in `forceList()`, or -1. */
  private wireAI = new Int32Array(0);
  private wireBI = new Int32Array(0);

  /**
   * The wires in Map order, with both endpoints resolved to agents. Keyed on
   * the roster as well as the graph version: otherwise a dead agent's wires
   * would keep handing out a body that is no longer in the sim.
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

  /** Topology this body's `bound` was last computed for; -1 forces a rebuild. */
  private boundVersion = -1;
  private boundRoster = -1;

  /** Union-find over the wires; component root per `forceList()` index. */
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
    // Flattened, so every later read is one array access.
    for (let i = 0; i < n; i++) parent[i] = find(i);
    this.compVersion = this.graph.version;
    this.compRoster = this.rosterVersion;
    return parent;
  }

  /**
   * Fill every body's `bound`, the fraction of its ports that are attached.
   * Its own pass, not part of `refreshForceScratch`: that one gives up when
   * the native solver is absent, and `BOUND` is a term in the chemistry, so
   * a zero there would be a different genome expressing itself.
   */
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

  /** Bit per unattached port: principal 1, left 2, right 4. */
  private freePortMask(a: Agent): number {
    const g = this.graph;
    const sl = a.slot;
    const p = g.isFreeAtSlot(sl, 0) ? 1 : 0;
    if (a.kind === 'era') return p;
    return p | (g.isFreeAtSlot(sl, 1) ? 2 : 0) | (g.isFreeAtSlot(sl, 2) ? 4 : 0);
  }

  /**
   * The wire graph as flat neighbour lists over `forceList()` order, rebuilt
   * only when the topology or the roster has moved.
   */
  private wireAdjacency(): WireAdjacency {
    if (this.adjGraphVersion === this.graph.version && this.adjRosterVersion === this.rosterVersion) {
      return this.wireAdj;
    }
    // The wake graph's CSR, aliased rather than copied; the energy passes
    // only read it. The queue scratch stays this object's own, since the
    // relaxation writes that.
    this.refreshWakeGraph();
    this.wireAdj.off = this.wakeOff;
    this.wireAdj.nei = this.wakeNei;
    this.adjGraphVersion = this.graph.version;
    this.adjRosterVersion = this.rosterVersion;
    return this.wireAdj;
  }

  /**
   * Every wire whose two principals are ready to rewrite, with neither end
   * already in a rewrite. Read by `pulseRequests`, `accrueRedexes` and
   * `startRewrites`; nothing adds to `rewrites` before the last of those,
   * and a principal has one wire, so the busy set need not grow meanwhile.
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
      // Bitmask of free ports, for the scent deposit in endFrame.
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
      // Only when the topology cache could not be claimed.
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
    return true;
  }


  private wakeGraphVersion = -1;
  private wakeGraphRoster = -1;
  private readonly wakeList: Agent[] = [];
  private wakeOff = new Int32Array(1);
  private wakeNei = new Int32Array(0);
  private wakeQ = new Int32Array(0);

  /** Bodies and their wire adjacency in CSR, rebuilt only when topology moves. */
  private refreshWakeGraph(): number {
    const list = this.wakeList;
    if (
      this.wakeGraphVersion === this.graph.version &&
      this.wakeGraphRoster === this.rosterVersion
    ) {
      return list.length;
    }
    // The shared list, so the wake graph uses the order every pass indexes by.
    const src = this.forceList();
    list.length = 0;
    for (let i = 0; i < src.length; i++) list.push(src[i]);
    const n = list.length;
    if (this.wakeOff.length < n + 2) this.wakeOff = new Int32Array(n * 2 + 4);
    if (this.wakeQ.length < n) this.wakeQ = new Int32Array(n * 2);
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


  /** True when this body is on the SAT / XPBD rope path. */
  isPhysicsDetailed(id: number): boolean {
    return this.agentDetailed(id);
  }

  /**
   * A wire keeps its live XPBD rope when either end is detailed *and* the
   * stroke is still wide enough to see the rope in. A wire test, not a body
   * test: stroke width must not decide body physics.
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
   * Size on screen, then lift a small neighbourhood (`PHYS_HOPS`) so a machine
   * under the cursor does not have half its ropes on the cheap path.
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
    // Every body is tiered every frame even when the answer is FAR, so the
    // hysteresis in the selector has the history it needs. Reads the store
    // directly; boundRadius's formula is inlined for the same reason.
    const store = this.agentStore;
    const ID = store.id;
    const KIND_CODE = store.kindCode;
    const SCALE = store.scale;
    const X = store.x;
    const Y = store.y;
    for (const a of this.agents.values()) {
      const s = a.slot;
      // boundRadius's own formula: era is 9px (agentSize), everything else 16px.
      const size =
        (KIND_CODE[s] === KIND_ERA ? ERA_RADIUS + 1.2 : 16 * 1.12) * SCALE[s] * 2;
      const px = apparentPx(size, view);
      const vis = onScreen(X[s], Y[s], size, view);
      if (this.physLod.tier(agentKey(ID[s]), px, vis, AGENT_BAND) !== LOD_FAR) seeds.push(ID[s]);
    }
    // A grab needs a neighbourhood so a pointer drag does not punch through
    // the cheap path. In-flight rewrites do not get one: lifting their ropes
    // onto XPBD while the kinematic pull drives them whips the ropes into knots.
    if (this.grabbed) seeds.push(this.grabbed.id);
    if (seeds.length === 0) {
      this.physLod.sweep();
      return;
    }

    // BFS over the wake graph's CSR, `PHYS_HOPS` deep from the seeds.
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
   * detected for friction audio but not displaced. Segment vs the body's
   * bounding circle, one-way: a rope never moves an agent, or the solver
   * explodes. Runs inside the substep loop so the rope constraints re-settle
   * around the push, and is rate limited (`WIRE_CLEAR_STEP`) because a
   * displacement resolved in one substep becomes that times 1/h in velocity.
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
      // Bounds inline rather than through ropeAabb: no per-pair allocation.
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
      // The chord is the same for every node, so its projection basis is hoisted.
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
   * Wire/body pairs close enough to be worth testing, rebuilt once per frame
   * for the narrow phase that runs every substep. The margins are generous
   * enough that a frame of drift cannot smuggle a pair past it.
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
   * arrange a net while the sim is paused. Velocities are cleared afterwards so
   * unpausing does not release stored-up correction as a kick.
   */
  dragStep(params: Params, dt: number, view?: PanView | null): void {
    if (!this.grabbed || dt <= 0) return;
    const h = dt / Sim.SUBSTEPS;
    this.collectRewriteFrozen();
    this.assignPhysicsLod(view);
    this.graph.syncRest(this.time, params, this.agents, this.agentStore.gaitWave, this.wireDetailed);
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

  /** Pull a held agent toward the pointer, rate limited by `GRAB_STEP`. */
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
   * against agents from *other* nets. Same-net crowding is left to flocking
   * separation. Equal and opposite, so it never moves the centre of mass.
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
        // Floored so the law cannot run away at touching distance.
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
   * Constrained integration (XPBD, small steps): wires, port axes and contacts
   * are compliant constraints solved inside this one loop; nothing outside it
   * writes a pose, and velocity is derived from the result.
   */
  private solve(params: Params, dt: number): void {
    if (dt <= 0) return;
    // buildClearPairs reads positions, which the force passes never move, so
    // the force block can stay open through the native attempt.
    this.buildClearPairs(params);
    if (this.solveNearNative(params, dt)) return;
    // The JS fallback reads the agents outright, so they get their velocities
    // back here, including after a native attempt that packed and bailed.
    this.syncForces();
    const frozen = this.rewriteFrozen;
    const h = dt / Sim.SUBSTEPS;
    const invH = 1 / h;
    const held = this.grabbed?.id ?? -1;
    // Rope velocity is re-derived every substep, so damping must be per substep.
    const ropeKeep = Math.exp(-Math.max(0, params.springDamp) * h);
    const list = this.forceList();
    const n = list.length;

    // Store arrays indexed by slot, not Agent accessors: hot loop.
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
   * whatever the mode says. 'auto' means the GPU only where wasm will not go
   * (the pack outgrowing its caps, with `FAR_GPU_RELEASE` hysteresis). A
   * missing wasm module reports zero capacity, which is why the check is
   * capacity rather than a constant.
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
    // Its own wire array, not the shared `wirePack`: this list drops
    // self-wires, and a filtered list in the shared array would leave it
    // stamped current while pairing wire k with another wire's endpoints.
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
      // FAR never runs SAT, so the contact radius is the glyph-area disc.
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
   * FAR straight against the solver's own buffers. When the force block is
   * still open the pose is in there already and only the per-frame metadata
   * is written.
   */
  private solveFarNative(params: Params, dt: number): boolean {
    if (!nativeSolver.ready || !this.canFarPacked()) return false;
    const data = nativeSolver.bodies;
    const wires = nativeSolver.wires;
    if (!data || !wires) return false;

    const list = this.forceList();
    const n = list.length;
    if (n === 0) return true;
    // Its own array, not the cached `wirePack`; see `packFar`.
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
      // `stiffnessOf` inlined; must match it.
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
    // Pinning is when the ground gets laid down, and how much ground there is
    // is a slider; callers outside a frame must pass `params` or the ground is
    // seeded from the construction fallbacks.
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
    // The ground moves onto the field and gets laid down. Until the first
    // pin the grid answers out of its sparse map; `bind` is not undone by
    // `clear`, so a cleared world has no ground until it is seeded again.
    this.energy.bind(this.fields);
    this.energy.configure(this.energyCell, this.energyAmbient);
    this.energy.seedGround();
  }

  /** Last configure, so `pinWorld` can seed the ground at the slider values. */
  private energyCell = 40;
  private energyAmbient = 1;

  /**
   * Per-channel field rates from the sliders, every frame. Energy never
   * decays: it is a conserved quantity.
   */
  private tuneChannels(params: Params): void {
    const f = this.fields;
    f.decayRate[CH.energy] = 0;
    f.diffuseRate[CH.energy] = Math.max(0, params.energyDiffuse);
  }

  /** True once a device exists and the field has moved there for good. */
  private fieldOnGpu = false;

  /**
   * Move the field to the GPU if there is one. Call once, at startup. One
   * way for the session: the only path back to the CPU field is a device
   * loss, which drops to it for good.
   *
   * Once open, the GPU holds the live field and only deposits and probes
   * cross; `fields.data` is a CPU copy that is refreshed only while
   * `wantFieldReadback` is set for the debug overlays. Anything else reading
   * or writing that array on this path is reading a stale copy.
   */
  async openFieldGpu(): Promise<boolean> {
    if (this.fieldOnGpu) return true;
    if (!(await fieldGpu.init(this.fields.cols))) return false;
    // `fieldGpu` is a module singleton whose buffers outlive the last `Sim`
    // that used them; a second pond in the same process must not inherit
    // the first one's field.
    fieldGpu.clear();
    // Everything that writes the ground has to be told before the first
    // frame: a queued seed is only picked up by `gpuFieldStep`.
    this.energy.deferAdds(true);
    this.energy.seedGround();
    this.fieldOnGpu = true;
    // The genome is strictly downstream: it reads the probe's output buffer
    // for its sense inputs. If it declines, the field still runs.
    const dev = fieldGpu.gpuDevice;
    if (dev && (await genomeGpu.init(dev))) this.genomeOnGpu = true;
    return true;
  }

  /**
   * Hand the field a frame's worth of work and take back what steering needs.
   * Port and sensor positions are computed here with the CPU path's helpers;
   * `field.wgsl` sees bare world coordinates. The samples that come back
   * describe the pose this frame started with, and steering reads them at
   * the top of the next one.
   */
  private async gpuFieldStep(params: Params, dt: number): Promise<void> {
    const list = this.forceList();
    const n = list.length;
    // The harvest plan is built here, at the end of the frame, and spent at
    // the top of the next.
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
    // Three ports a body, plus the frame's queued energy adds. Reserve before
    // taking references: `reserve` reallocates the staging arrays when it
    // grows them, and a stale reference writes into a discarded buffer.
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
    const scratch = this.portScratch;
    // The minted voice, or nothing at all: under the reaction table a body's
    // output leaves its tank through `runExcretion` instead. See `scentMints`.
    const mints = scentMints(params);
    for (let i = 0; mints && i < n; i++) {
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

    // The frame's queued energy adds, on the same scatter as the voices but
    // with the conserve flag: these are counts, handed over raw, and the
    // shader spreads each over the cells the disk will take rather than
    // dropping the share outside the rim (`Fields.addAt`, not `deposit`).
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
      // tank and each has to survive the rim the way a quantity does.
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
    // The two sensor directions come from the store's memoised cos/sin of
    // the heading by the angle-sum identity.
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
      !genomeGpu.submit(samples, n, nNei, this.groundScale, CH.energy, params.senseScale, {
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
   * Bring every body's learning back from the device.
   *
   * On the GPU path the learning row is device-resident and the host's copy
   * is fresh only for rewrite parents — `syncLearn` reads back a handful a
   * frame because a handful a frame is all the simulation needs. Anything
   * that *measures* what the pond has learned has to ask for the rest, and
   * `docs/history/plasticity-plan.md` phase 5 says so in as many words.
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
        const w = rows[o + k];
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
      // One row per entry, one `got` per species. Metered, each lands in the
      // gut as itself and `runDigestion` decides what any of it is worth;
      // unmetered, only the ground's is ever non-zero and it is money the
      // moment it is swallowed, which is what this has always done.
      const ro = e * HARVEST_STRIDE + HARVEST_GOT;
      if (metered) {
        const go = slot * CHEM_SPECIES;
        for (let c = 0; c < CHEM_SPECIES; c++) {
          const g = got[ro + c];
          if (g > 0) {
            GUT[go + c] += g;
            this.gutLive = true;
          }
        }
        continue;
      }
      let g = 0;
      for (let c = 0; c < CHANNELS; c++) g += got[ro + c];
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
     * `docs/history/energy-chemistry-plan.md` §8. Latching is proximity plus an arc
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
    // Before the claims overwrite it: this frame's relay is off last frame's
    // field, which is what makes demand travel a hop at a time.
    if (this.requestPrev.length < list.length) {
      this.requestPrev = new Float64Array(Math.max(64, list.length * 2));
    }
    const prev = this.requestPrev;
    for (let i = 0; i < list.length; i++) prev[i] = REQUEST[list[i].slot];
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
     * How far demand gets this frame. `requestReach` 0 means as far as it
     * goes: iterating the one-hop step past the longest possible path is the
     * fixpoint the old in-frame relaxation solved, so the field settles
     * exactly where it used to. Above 0 it advances that many hops and stops,
     * and the rest of the journey happens on later frames — which is what
     * gives demand a front to travel on. The parameter has the trade.
     *
     * Re-snapshotting between hops is what keeps each one a *step*: without
     * it a value would race down the list in whatever order the roster
     * happens to be in, and the reach would depend on the sort.
     */
    const reach = params.requestReach;
    if (reach <= 0) {
      // The default, and a different algorithm rather than this one run to
      // convergence: queue-driven, one visit per body that actually improved.
      // Iterating the one-hop step until it is certainly settled costs a
      // sweep per hop and is quadratic in the roster — measured as the frame
      // at a few hundred founders, which is how it was found.
      relaxRequestsFast(list, this.agentStore, adj);
    } else {
      for (let h = 0; h < reach; h++) {
        // Re-snapshot between hops so each is a *step*: reading the live
        // array would let one body's need race down the list in whatever
        // order the roster happens to be in.
        if (h > 0) for (let i = 0; i < list.length; i++) prev[i] = REQUEST[list[i].slot];
        spreadRequestsFast(list, this.agentStore, adj, prev);
      }
    }
    Sim.phase('pulse:spread');
    // No `quantum` here: each sender uses its own, seeded from the parameter
    // at birth exactly as `transportRecoil` is, so a net's rhythm can be a
    // property of the net rather than of the dish.
    flowChargesFast(list, this.agentStore, adj, (from, to, amount) => this.recoil(from, to, amount), {
      grid: this.energy,
    });
    Sim.phase('pulse:flow');
    if (list.length > 0) {
      if (this.genomeOnGpu) this.unpackGenome();
      else {
        updateState(
          { agentStore: this.agentStore, fields: this.fields, fieldOnGpu: this.fieldOnGpu, groundScale: this.groundScale },
          list,
          adj,
          params,
        );
      }
    }
    Sim.phase('state');
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
      store.transportRecoil[slot] = out[o + 16];
      store.gaitAnchor[slot] = out[o + 17];
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
      applyTransportRecoil(A, B, amount, A.transportRecoil, this.w, this.h);
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

  /**
   * Whether the last `refreshExpression` filled the vectors, which is the one
   * predicate for "is anything expressed". Passed to whoever reads the rows
   * outside this pass (`tickUpkeepFast`) rather than re-derived from eight
   * floats a body: `expressVector` always writes a unit-sum vector, so the
   * rows themselves cannot say whether they were written this frame.
   */
  private expressed = false;
  /**
   * Whether anything in the pond is holding something undigested.
   *
   * Set where a mouthful lands and cleared by the pass that finds every gut
   * empty. It is a hint, not a fact: over-setting it costs one pass over the
   * roster, and the only thing that must never happen — a gut nothing drains
   * — needs it to be under-set, which nothing here does.
   */
  private gutLive = false;

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
): void {
  if (A.locked || B.locked || !(amount > 0) || !(gain > 0)) return;
  const d = wrapDeltaVec(A.x, A.y, B.x, B.y, w, h);
  const dist = Math.hypot(d.x, d.y);
  if (!(dist > 1e-6)) return;
  const p = gain * amount;
  const nx = d.x / dist;
  const ny = d.y / dist;
  const wA = 1 / Math.max(0.08, A.mass);
  const wB = 1 / Math.max(0.08, B.mass);
  A.vx -= nx * p * wA;
  A.vy -= ny * p * wA;
  B.vx += nx * p * wB;
  B.vy += ny * p * wB;
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
