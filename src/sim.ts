import {
  boundRadius,
  discRadius,
  createAgent,
  inSnapArc,
  momentOfInertia,
  portWorld,
  slotsFor,
  stemRoot,
  stemOffset,
  stemOffsetInto,
  stemWorld,
  stemWorldInto,
  type Agent,
  type AgentKind,
  type PortSlot,
} from './agents.ts';
import { queryHit, queryDiscHit, SLOP, type Hit } from './collide.ts';
import { closestTOnSegment, segmentsIntersect, WIRE_RADIUS, wireBowBudget } from './geom.ts';
import { PairGrid } from './grid.ts';
import { CHAIN_MASS, contactMechanics, portExitAngle, solveContact } from './chain.ts';
import { CH, Fields } from './fields.ts';
import { Graph, ropeIsLive, wrapPos, type Wire } from './graph.ts';
import type { Params } from './params.ts';
import {
  advanceRewrite,
  beginRewrite,
  commitRewrite,
  detectRule,
  PULL_END,
  rewriteHandoffStems,
  type Rewrite,
} from './rewrite.ts';
import {
  agentValue,
  deathYield,
  EnergyGrid,
  extrasOf,
  canPayShare,
  flowCharges,
  harvestSlots,
  hungerNeed,
  redexNeed,
  resetRequests,
  rewriteCost,
  rewriteYield,
  seedRequest,
  settlePool,
  spendExtra,
  spreadRequests,
  tickUpkeep,
  WireAdjacency,
} from './energy.ts';
import { audio } from './audio/engine.ts';
import type { CollisionEvent, LiveContact, PanView, RewriteEvent } from './audio/types.ts';
import { AGENT_BAND, LOD_FAR, LodSelector, agentKey, apparentPx, onScreen, wiresDrawable } from './audio/lod.ts';
import { farGpu } from './gpu/far-gpu.ts';
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
export function mixScent(
  kind: AgentKind,
  con: number,
  dup: number,
  aux: number,
  params: Params,
): number {
  const S = params.attractStrong;
  const M = params.attractMedium;
  if (kind === 'era') return S * (con + dup) + M * aux;
  if (kind === 'dup') return M * (con + aux);
  return M * (dup + aux);
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

export class Sim {
  /** Substeps per frame. One constraint iteration each. */
  private static readonly SUBSTEPS = 8;

  /** Hop radius for flocking. Farther pairs are 1/hops anyway. */
  private static readonly FLOCK_HOPS = 6;
  /** Neighbourhood lifted onto the detailed physics path around a NEAR/MID body. */
  private static readonly PHYS_HOPS = 2;

  /**
   * Gravity only touches components this small, so a finished net does not
   * fall into its own centre of mass. 1 = loners, 2 = a fresh latch.
   */
  private static readonly GRAV_MAX_COMP = 2;

  /** Distance past which the gravity pull stops growing. */
  private static readonly HOME_REACH = 320;
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

  /** Scratch: connected-component size per root, rebuilt in `gravitate`. */
  private compSize = new Map<number, number>();

  w: number;
  h: number;
  coverW: number;
  coverH: number;
  time = 0;
  nextId = 1;
  spawnAcc = 0;
  agents = new Map<number, Agent>();
  graph = new Graph();
  fields: Fields;
  rewrites: Rewrite[] = [];
  energy = new EnergyGrid(48, 0.1);
  /** Per-agent unmet need this frame, rebuilt by `pulseRequests`. */
  private readonly needOf = new Map<number, number>();
  private readonly wireAdj = new WireAdjacency();
  /** Connected-component root per agent, refreshed once per frame. */
  private components = new Map<number, number>();
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
  private packIndex = new Map<number, number>();
  private clearWireList: Wire[] = [];
  private wallPts: { x: number; y: number }[] = [];
  private flockAdj: number[][] = [];
  private flockIndex = new Map<number, number>();
  private flockDist = new Int32Array(0);
  private flockQ = new Int32Array(0);
  private flockSeen: number[] = [];
  private flockSwim = new Uint8Array(0);
  private tmpStemA = { x: 0, y: 0 };
  private tmpStemB = { x: 0, y: 0 };
  private satBuf = new Uint8Array(0);
  private compBuf = new Int32Array(0);
  /** Pairs in contact last frame — a strike fires on onset, contact continues. */
  private contactAudioPrev = new Set<string>();
  private contactAudioNow = new Set<string>();
  /** Wire/body bows this frame, keyed by `wireId:agentId`. */
  /** Bodies currently overlapping, keyed by canonical `lo:hi` id pair. */
  contacts = new Map<string, LiveContact>();
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
   * Physics detail. When a view is passed, FAR agents keep disc contacts and a
   * chord constraint but skip SAT, rope XPBD, wire clearance, and Hertzian.
   * Missing view = everyone NEAR, which is what tests and a paused layout want.
   */
  private lodActive = false;
  /** False when zoom has shrunk the wire stroke below a visible hairline. */
  private ropesDrawable = true;
  private readonly physLod = new LodSelector();
  private readonly detailedAgents = new Set<number>();

  /** Eased centre of mass. Rewrites delete agents, which jumps the true COM. */
  private home: { x: number; y: number } | null = null;

  constructor(w: number, h: number) {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    this.coverW = this.w;
    this.coverH = this.h;
    this.fields = new Fields(this.w, this.h);
    this.graph.onLatch = (ev) => audio.push(ev, this.graph, this.agents);
    audio.contacts = this.contacts;
  }

  /** Viewport / spawn-box size. World coordinates are not scaled. */
  resize(w: number, h: number): void {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
  }

  setFieldCover(w: number, h: number): void {
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
    this.graph.clear();
    this.rewrites = [];
    this.fields.clear();
    this.energy.clear();
    this.time = 0;
    this.nextId = 1;
    this.spawnAcc = 0;
    this.home = null;
    this.contactAudioPrev.clear();
    this.contactAudioNow.clear();
    this.contacts.clear();
    this.physLod.clear();
    this.detailedAgents.clear();
    this.ropesDrawable = true;
    this.lodActive = false;
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
    );
    this.agents.set(a.id, a);
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
    this.energy.addAt(agent.x, agent.y, deathYield(agent));
    this.graph.detachAgent(id);
    this.agents.delete(id);
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

  step(dt: number, params: Params, view?: PanView | null): void {
    const t = this.beginFrame(dt, params);
    this.collectRewriteFrozen();
    this.assignPhysicsLod(view);
    this.graph.syncRest(this.time, params, this.wireDetailed);
    this.graph.applyRopePaths(this.agents, this.w, this.h, this.time, params);
    this.graph.syncRopeShape(this.agents, this.w, this.h, this.wireDetailed);
    if (this.solveFarNative(params, t)) this.finishIntegrate(t);
    else this.solve(params, t);
    this.endFrame(params, t);
  }

  /** Same as `step`, but will wait on the WebGPU FAR pass when WASM is not live. */
  async stepAsync(dt: number, params: Params, view?: PanView | null): Promise<void> {
    const t = this.beginFrame(dt, params);
    this.collectRewriteFrozen();
    this.assignPhysicsLod(view);
    this.graph.syncRest(this.time, params, this.wireDetailed);
    this.graph.applyRopePaths(this.agents, this.w, this.h, this.time, params);
    this.graph.syncRopeShape(this.agents, this.w, this.h, this.wireDetailed);
    if (this.solveFarNative(params, t)) {
      this.finishIntegrate(t);
    } else if (this.canFarGpu()) {
      await this.solveFarGpu(params, t);
      this.finishIntegrate(t);
    } else {
      this.solve(params, t);
    }
    this.endFrame(params, t);
  }

  private beginFrame(dt: number, params: Params): number {
    const t = clamp(dt, 0, 0.05);
    this.time += t;
    const com = this.centerOfMass();
    this.fields.cover(
      com?.x ?? this.w * 0.5,
      com?.y ?? this.h * 0.5,
      this.coverW,
      this.coverH,
    );
    this.components = this.graph.componentIds(this.agents);
    this.trackHome(t);
    this.contactAudioNow.clear();
    this.contacts.clear();
    this.radiated.clear();

    // The whole force phase now lives in WASM, so it shares one copy of the
    // bodies instead of each pass making its own — which was the entire cost
    // of moving them over: measured at 9600 agents, a pass that dropped from
    // 13 ms to 1.6 ms of compute still cost 7 ms because it packed 6 ms of
    // bodies to get there.
    const block = this.openForceBlock(params);
    this.steer(params, t);
    this.portTorques(params, t);
    this.declutter(params, t);
    this.uncrossPrincipals(params, t);
    this.flock(params, t);
    this.gravitate(params, t);
    // Left open on purpose. Nothing between here and the solver moves a body —
    // the LOD pass and the rope passes read positions and write wires — so the
    // solver can inherit the packed bodies instead of copying them in again.
    // Whoever consumes them closes it; `syncForces` is the backstop.
    void block;
    return t;
  }

  private endFrame(params: Params, t: number): void {
    // Backstop: every path that did not end in a native solve still owes the
    // agents their velocities.
    this.syncForces();
    this.applyRadiationLoss();
    this.dampVelocities(params, t);

    this.graph.refreshLengths(this.agents, this.w, this.h, this.rewriteFrozen, this.wireDetailed);
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
    harvestSlots(this.agents.values(), this.energy);
    this.graph.snap(this.agents, this.w, this.h, params, this.time);
    this.components = this.graph.componentIds(this.agents);
    this.pulseRequests(params);
    this.startRewrites(params);
    this.tickRewrites(params, t);
    for (const id of tickUpkeep(this.agents.values(), t, params.upkeep)) {
      this.kill(id);
    }
    this.components = this.graph.componentIds(this.agents);
    if (!this.scentWriteNative(params)) {
      this.deposit(params);
      this.paintScentWalls();
    }
    this.fields.diffuse(params.diffuse);
    this.fields.diffuse(params.diffuse * 0.65);
    this.fields.decay(params.decay);
    this.autoSpawn(params, t);

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
    const index = this.packIndex;
    const shared = this.forceBlock;
    if (!shared) index.clear();
    // Everything but the wires is already there when a block owns the pack;
    // writing it again would discard what the earlier passes accumulated.
    if (!shared) {
      for (let i = 0; i < n; i++) {
        const a = list[i];
        index.set(a.id, i);
        const o = i * FAR_STRIDE;
        bodies[o + FAR.x] = a.x;
        bodies[o + FAR.y] = a.y;
        bodies[o + FAR.vx] = a.vx;
        bodies[o + FAR.vy] = a.vy;
        bodies[o + FAR.heading] = a.heading;
        bodies[o + FAR.omega] = a.omega;
        bodies[o + FAR.locked] = a.locked ? 1 : 0;
        bodies[o + FAR.invMass] = a.locked ? 0 : 1 / Math.max(0.08, a.mass);
        invI[i] = a.locked ? 0 : 1 / Math.max(1e-4, momentOfInertia(a));
        kinds[i] = this.kindCode(a.kind);
        sc[i] = a.scale;
      }
    }
    const wires = nativeSolver.wiresNear;
    let k = 0;
    for (const w of wireList) {
      const ia = index.get(w.a.id);
      const ib = index.get(w.b.id);
      if (ia === undefined || ib === undefined) continue;
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
      if (a.locked) continue;
      a.omega = bodies[i * FAR_STRIDE + FAR.omega];
    }
  }

  private portTorques(params: Params, dt: number): void {
    const gain = params.portStiff * 320;
    if (gain <= 0 || dt <= 0) return;
    const splay = params.auxSpread * 0.35;
    if (this.portTorquesNative(gain, splay, dt)) return;
    const aim = (agent: Agent, slot: PortSlot, target: { x: number; y: number }): void => {
      if (agent.locked) return;
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
    const index = this.packIndex;
    index.clear();
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      bodies[o + FAR.x] = a.x;
      bodies[o + FAR.y] = a.y;
      bodies[o + FAR.vx] = a.vx;
      bodies[o + FAR.vy] = a.vy;
      bodies[o + FAR.heading] = a.heading;
      bodies[o + FAR.omega] = a.omega;
      bodies[o + FAR.invMass] = a.locked ? 0 : 1 / Math.max(0.08, a.mass);
      bodies[o + FAR.locked] = a.locked ? 1 : 0;
      // Radius is deliberately absent: no force pass reads it. The broad
      // phases here take their cell size as an argument.
      bm[i] = a.mass;
      invI[i] = a.locked ? 0 : 1 / Math.max(1e-4, momentOfInertia(a));
      kinds[i] = this.kindCode(a.kind);
      sc[i] = a.scale;
      index.set(a.id, i);
    }
    return true;
  }

  /** Read back what a force pass writes: linear and angular velocity. */
  private unpackDrift(list: Agent[]): void {
    const bodies = nativeSolver.bodies!;
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      if (a.locked) continue;
      const o = i * FAR_STRIDE;
      a.vx = bodies[o + FAR.vx];
      a.vy = bodies[o + FAR.vy];
      a.omega = bodies[o + FAR.omega];
    }
  }

  private forceList(): Agent[] {
    const list = this.agentList;
    list.length = 0;
    for (const a of this.agents.values()) list.push(a);
    return list;
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
    for (let i = 0; i < n; i++) {
      const a = list[i];
      sat[i] = this.graph.portsFilled(a) ? 1 : 0;
      comp[i] = this.components.get(a.id) ?? -1 - i;
    }
    nativeSolver.declutter(n, reach, atReach, cutoff, Sim.DECLUTTER_FLOOR, dt);
    if (!this.forceBlock) this.unpackDrift(list);
    return true;
  }

  /** Gravitate in WASM. `declSat` carries "this body's net is small enough". */
  private gravitateNative(
    cx: number,
    cy: number,
    base: number,
    cap: number,
    dt: number,
  ): boolean {
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    const sat = nativeSolver.declSat;
    if (!sat) return false;
    const list = this.forceList();
    const n = list.length;
    if (n === 0) return true;
    if (!this.forceBlock && !this.packPose(list)) return false;
    const sizes = this.compSize;
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const root = this.components.get(a.id) ?? a.id;
      sat[i] = (sizes.get(root) ?? 1) > cap ? 0 : 1;
    }
    nativeSolver.gravitate(n, cx, cy, base, Sim.HOME_REACH, cap, dt);
    if (!this.forceBlock) this.unpackDrift(list);
    return true;
  }

  /** The same pass in WASM. False when the scene will not pack. */
  private portTorquesNative(gain: number, splay: number, dt: number): boolean {
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    const list = this.agentList;
    list.length = 0;
    for (const a of this.agents.values()) list.push(a);
    if (list.length === 0) return true;
    const wireList = this.wirePack;
    wireList.length = 0;
    for (const w of this.graph.wires.values()) wireList.push(w);
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
    return !this.lodActive || this.detailedAgents.has(id);
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
    for (const a of this.agents.values()) {
      const size = boundRadius(a) * 2;
      const px = apparentPx(size, view);
      const vis = onScreen(a.x, a.y, size, view);
      if (this.physLod.tier(agentKey(a.id), px, vis, AGENT_BAND) !== LOD_FAR) seeds.push(a.id);
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

    const index = this.packIndex;
    index.clear();
    const list = this.agentList;
    list.length = 0;
    for (const a of this.agents.values()) {
      index.set(a.id, list.length);
      list.push(a);
    }
    const n = list.length;
    const adj = this.flockAdj;
    while (adj.length < n) adj.push([]);
    for (let i = 0; i < n; i++) adj[i].length = 0;
    for (const w of this.graph.wires.values()) {
      const ia = index.get(w.a.id);
      const ib = index.get(w.b.id);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      adj[ia].push(ib);
      adj[ib].push(ia);
    }
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
      const i = index.get(id);
      if (i === undefined || hop[i] >= 0) continue;
      hop[i] = 0;
      q[qt++] = i;
      this.detailedAgents.add(list[i].id);
    }
    let qh = 0;
    while (qh < qt) {
      const u = q[qh++];
      const du = hop[u];
      if (du >= Sim.PHYS_HOPS) continue;
      const nei = adj[u];
      for (let k = 0; k < nei.length; k++) {
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
    this.components = this.graph.componentIds(this.agents);
    this.collectRewriteFrozen();
    this.assignPhysicsLod(view);
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
    if (!agent || agent.locked) return;
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
    }
    const sat = this.satBuf;
    const comp = this.compBuf;
    for (let i = 0; i < n; i++) {
      sat[i] = this.graph.portsFilled(list[i]) ? 1 : 0;
      comp[i] = this.components.get(list[i].id) ?? -1 - i;
    }
    this.bodyGrid.forEachPair((i, j) => {
      {
        if (!sat[i] && !sat[j]) return;
        if (comp[i] === comp[j]) return;
        const A = list[i];
        const B = list[j];
        const dx = B.x - A.x;
        const dy = B.y - A.y;
        const dist = Math.hypot(dx, dy);
        if (dist > cutoff || dist < 1e-6) return;
        // Floored so the law cannot run away at touching distance; contacts own
        // that range anyway.
        const ratio = reach / Math.max(dist, floor);
        const force = atReach * ratio * ratio;
        const nx = dx / dist;
        const ny = dy / dist;
        if (!A.locked) {
          const invM = 1 / Math.max(0.08, A.mass);
          A.vx -= nx * force * invM * dt;
          A.vy -= ny * force * invM * dt;
        }
        if (!B.locked) {
          const invM = 1 / Math.max(0.08, B.mass);
          B.vx += nx * force * invM * dt;
          B.vy += ny * force * invM * dt;
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
    if (!chord.wireA.locked) {
      chord.wireA.vx += ux;
      chord.wireA.vy += uy;
    }
    if (!chord.wireB.locked) {
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
    const list = this.agentList;
    list.length = 0;
    for (const a of this.agents.values()) list.push(a);

    for (let sub = 0; sub < Sim.SUBSTEPS; sub++) {
      for (const a of list) {
        a.prevX = a.x;
        a.prevY = a.y;
        a.prevHeading = a.heading;
        if (a.locked) continue;
        a.x += a.vx * h;
        a.y += a.vy * h;
        a.heading = wrapAngle(a.heading + a.omega * h);
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

      for (const a of list) {
        if (a.locked) {
          a.vx = 0;
          a.vy = 0;
          a.omega = 0;
          continue;
        }
        a.vx = (a.x - a.prevX) * invH;
        a.vy = (a.y - a.prevY) * invH;
        a.omega = wrapAngle(a.heading - a.prevHeading) * invH;
        if (held === a.id) {
          const speed = Math.hypot(a.vx, a.vy);
          if (speed > Sim.GRAB_MAX_SPEED) {
            const k = Sim.GRAB_MAX_SPEED / speed;
            a.vx *= k;
            a.vy *= k;
          }
        }
      }
      for (const wire of this.graph.wires.values()) {
        if (!this.wireSimulatesRope(wire)) continue;
        for (const node of wire.nodes) {
          node.vx = (node.x - node.prevX) * invH * ropeKeep;
          node.vy = (node.y - node.prevY) * invH * ropeKeep;
        }
      }
    }

    for (const a of list) {
      a.stun = Math.max(0, a.stun - dt);
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

  private packFar(params: Params): {
    list: Agent[];
    data: Float32Array;
    wires: Float32Array;
    nWires: number;
  } | null {
    const list = this.agentList;
    list.length = 0;
    for (const a of this.agents.values()) list.push(a);
    const n = list.length;
    if (n === 0) return null;
    const index = this.packIndex;
    index.clear();
    for (let i = 0; i < n; i++) index.set(list[i].id, i);
    const wireList = this.wirePack;
    wireList.length = 0;
    for (const w of this.graph.wires.values()) {
      if (index.has(w.a.id) && index.has(w.b.id) && w.a.id !== w.b.id) wireList.push(w);
    }
    const { data, wires } = farGpu.packTarget(n, wireList.length);
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      const locked = a.locked || this.rewriteFrozen.has(a.id);
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
        index.get(w.a.id)!,
        index.get(w.b.id)!,
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
      if (a.locked || this.rewriteFrozen.has(a.id)) continue;
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

    const list = this.agentList;
    if (!this.forceBlock) {
      list.length = 0;
      for (const a of this.agents.values()) list.push(a);
    }
    const n = list.length;
    if (n === 0) return true;
    const index = this.packIndex;
    // The force block built this from the same list; no agent has come or
    // gone since, so rebuilding it is nine thousand Map writes for nothing.
    if (!this.forceBlock) {
      index.clear();
      for (let i = 0; i < n; i++) index.set(list[i].id, i);
    }
    const wireList = this.wirePack;
    wireList.length = 0;
    for (const w of this.graph.wires.values()) {
      if (index.has(w.a.id) && index.has(w.b.id) && w.a.id !== w.b.id) wireList.push(w);
    }
    if (n > nativeSolver.bodyCap || wireList.length > nativeSolver.wireCap) return false;

    const inherited = this.forceBlock;
    if (inherited && Sim.auditForceBlock) this.auditPack(list, data, 'FAR');
    const scale = 12 / Math.max(1, params.springK);
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      const locked = a.locked || this.rewriteFrozen.has(a.id);
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
        index.get(w.a.id)!,
        index.get(w.b.id)!,
        w.rest,
        this.tmpStemA.x,
        this.tmpStemA.y,
        this.tmpStemB.x,
        this.tmpStemB.y,
        soft,
      );
    }
    if (!nativeSolver.stepFarInPlace(n, wireList.length, dt)) return false;
    this.unpackFar(list, data);
    this.releaseForceBlock();
    return true;
  }

  /** Packed FAR pass on the GPU. True when the kernel ran. */
  private async solveFarGpu(params: Params, dt: number): Promise<boolean> {
    const packed = this.packFar(params);
    if (!packed) return true;
    await farGpu.step(packed.data, packed.list.length, packed.wires, packed.nWires, dt);
    this.unpackFar(packed.list, packed.data);
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
    const list = this.agentList;
    if (!this.forceBlock) {
      list.length = 0;
      for (const a of this.agents.values()) list.push(a);
    }
    const n = list.length;
    if (n === 0) return true;

    const index = this.packIndex;
    index.clear();
    for (let i = 0; i < n; i++) index.set(list[i].id, i);

    const wireList = this.wirePack;
    wireList.length = 0;
    let nNodes = 0;
    for (const w of this.graph.wires.values()) {
      if (!index.has(w.a.id) || !index.has(w.b.id)) continue;
      wireList.push(w);
      if (this.wireSimulatesRope(w) && w.nodes.length > 0) nNodes += w.nodes.length;
    }
    const nWires = wireList.length;
    if (!nativeSolver.canNear(n, nWires, nNodes)) return false;

    this.packNearMeta(list, wireList, index, params);
    this.packNearState(list, wireList);

    const ropeKeep = Math.exp(-Math.max(0, params.springDamp) * (dt / Sim.SUBSTEPS));
    const heldId = this.grabbed?.id ?? -1;
    const heldIndex = heldId < 0 ? -1 : (index.get(heldId) ?? -1);
    const gx = this.grabbed?.x ?? 0;
    const gy = this.grabbed?.y ?? 0;
    if (!nativeSolver.stepNear(n, nWires, dt, Sim.SUBSTEPS, ropeKeep, heldIndex, Sim.GRAB_MAX_SPEED, gx, gy)) {
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
    index: Map<number, number>,
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
      const locked = a.locked;
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
      const ai = index.get(w.a.id)!;
      const bi = index.get(w.b.id)!;
      const A = list[ai];
      const B = list[bi];
      const frozenEnds = frozen.has(A.id) || frozen.has(B.id);
      const skip = (A.locked && B.locked) || frozenEnds;
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
    const key = A.id < B.id ? `${A.id}:${B.id}` : `${B.id}:${A.id}`;
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
    const key = `${lo}:${hi}`;
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
      if (agent.locked) return;
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
    const list = this.agentList;
    list.length = 0;
    for (const a of this.agents.values()) list.push(a);
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
      if (A.locked && B.locked) return;
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
      if (!agent || agent.locked) continue;
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
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      agent.vx *= linKeep;
      agent.vy *= linKeep;
      agent.omega *= angKeep;
    }
  }

  /** Rasterize wire chains so scent diffusion cannot cross them. */
  private paintScentWalls(): void {
    this.fields.clearWalls();
    const pts = this.wallPts;
    for (const wire of this.graph.wires.values()) {
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      const n = this.wireSimulatesRope(wire) ? wire.nodes.length : 0;
      const need = n + 2;
      while (pts.length < need) pts.push({ x: 0, y: 0 });
      stemWorldInto(A, wire.a.slot, this.w, this.h, pts[0]);
      for (let i = 0; i < n; i++) {
        pts[i + 1].x = wire.nodes[i].x;
        pts[i + 1].y = wire.nodes[i].y;
      }
      stemWorldInto(B, wire.b.slot, this.w, this.h, pts[n + 1]);
      for (let i = 0; i < need - 1; i++) {
        this.fields.markSegment(pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y);
      }
    }
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
      this.spawn(
        kind,
        com.x + Math.cos(a) * r,
        com.y + Math.sin(a) * r,
        Math.random() * Math.PI * 2,
        params,
      );
    }
  }

  /**
   * Laying scent and stamping the wire walls, both in the solver.
   *
   * They go together because they share the copy: the field goes across once,
   * both passes write it in place, and it comes back once. Separately they
   * were the two most expensive things left in a settled frame — 4.5 ms of
   * wall marking and 2.5 ms of deposit at 9600 bodies — and both are
   * arithmetic over a grid the solver already holds a mirror of.
   */
  private scentWriteNative(params: Params): boolean {
    if (!Sim.nativeForces || !nativeSolver.ready) return false;
    const free = nativeSolver.portFree;
    const pts = nativeSolver.wallPts;
    const runs = nativeSolver.wallRuns;
    const kinds = nativeSolver.kind;
    const sc = nativeSolver.scale;
    if (!free || !pts || !runs || !kinds || !sc) return false;

    const list = this.forceList();
    const n = list.length;
    if (!nativeSolver.canNear(n, 0, 0)) return false;
    if (!nativeSolver.loadScent(this.fields)) return false;

    for (let i = 0; i < n; i++) {
      const a = list[i];
      const o = i * FAR_STRIDE;
      nativeSolver.bodies![o + FAR.x] = a.x;
      nativeSolver.bodies![o + FAR.y] = a.y;
      nativeSolver.bodies![o + FAR.heading] = a.heading;
      nativeSolver.bodies![o + FAR.locked] = a.locked ? 1 : 0;
      kinds[i] = this.kindCode(a.kind);
      sc[i] = a.scale;
      let mask = 0;
      for (const slot of slotsFor(a.kind)) {
        if (this.graph.isFreeAt(a.id, slot)) mask |= 1 << this.slotCode(slot);
      }
      free[i] = mask;
    }
    nativeSolver.deposit(n, params.deposit);

    // Polylines for the wall mask. The host packs them because it owns the
    // rope nodes; the marking walk is what costs.
    let at = 0;
    let nRuns = 0;
    const cap = nativeSolver.wallPtCap;
    for (const wire of this.graph.wires.values()) {
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      const mid = this.wireSimulatesRope(wire) ? wire.nodes.length : 0;
      if (at + mid + 2 > cap || nRuns >= runs.length) break;
      const sa = stemWorldInto(A, wire.a.slot, this.w, this.h, this.tmpStemA);
      pts[at * 2] = sa.x;
      pts[at * 2 + 1] = sa.y;
      at++;
      for (let i = 0; i < mid; i++) {
        pts[at * 2] = wire.nodes[i].x;
        pts[at * 2 + 1] = wire.nodes[i].y;
        at++;
      }
      const sb = stemWorldInto(B, wire.b.slot, this.w, this.h, this.tmpStemB);
      pts[at * 2] = sb.x;
      pts[at * 2 + 1] = sb.y;
      at++;
      runs[nRuns++] = mid + 2;
    }
    nativeSolver.paintWalls(nRuns);

    nativeSolver.storeScent(this.fields);
    nativeSolver.storeWalls(this.fields);
    return true;
  }

  private deposit(params: Params): void {
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      for (const slot of slotsFor(agent.kind)) {
        if (!this.graph.isFreeAt(agent.id, slot)) continue;
        const p = portWorld(agent, slot, this.w, this.h);
        const ch =
          slot === 'p'
            ? agent.kind === 'con'
              ? CH.conP
              : agent.kind === 'dup'
                ? CH.dupP
                : CH.eraP
            : CH.aux;
        const amt = slot === 'p' ? params.deposit : params.deposit * 0.7;
        this.fields.deposit(ch, p.x, p.y, amt);
      }
    }
  }

  private scentAt(agent: Agent, x: number, y: number, params: Params): number {
    return mixScent(
      agent.kind,
      this.fields.sample(CH.conP, x, y),
      this.fields.sample(CH.dupP, x, y),
      this.fields.sample(CH.aux, x, y),
      params,
    );
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
    const trail = nativeSolver.bodyTrail;
    const kinds = nativeSolver.kind;
    const sc = nativeSolver.scale;
    if (!sp || !flags || !pwire || !noise || !drive || !trail || !kinds || !sc) return false;
    const list = this.forceList();
    const n = list.length;
    if (n === 0) return true;
    if (!nativeSolver.canNear(n, 0, 0)) return false;
    if (!nativeSolver.loadScent(this.fields)) return false;
    if (!this.forceBlock && !this.packPose(list)) return false;

    const index = this.packIndex;
    if (!this.forceBlock) {
      index.clear();
      for (let i = 0; i < n; i++) index.set(list[i].id, i);
    }
    for (let i = 0; i < n; i++) {
      const a = list[i];
      const pFree = this.graph.isFreeAt(a.id, 'p');
      flags[i] = (pFree ? 1 : 0) | (a.stun > 0 ? 4 : 0);
      if (!this.forceBlock) {
        kinds[i] = this.kindCode(a.kind);
        sc[i] = a.scale;
      }
      drive[i] = a.drive;
      const pw = this.graph.wireAt({ id: a.id, slot: 'p' });
      let pj = -1;
      let pslot = 0;
      if (pw) {
        const other = pw.a.id === a.id ? pw.b : pw.a;
        pj = index.get(other.id) ?? -1;
        pslot = this.slotCode(other.slot);
      }
      pwire[i * 2] = pj;
      pwire[i * 2 + 1] = pslot;
      // Drawn host-side so a seeded run stays reproducible; the solver only
      // consumes them.
      noise[i * 3] = Math.random();
      noise[i * 3 + 1] = Math.random();
      noise[i * 3 + 2] = Math.random();
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
    sp[12] = params.attractStrong;
    sp[13] = params.attractMedium;
    nativeSolver.steer(n, dt);
    for (let i = 0; i < n; i++) {
      const a = list[i];
      a.drive = drive[i];
      a.trail = trail[i];
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

    for (const agent of list) {
      if (agent.locked) continue;

      let biasX = 0;
      let biasY = 0;
      const tip = portWorld(agent, 'p', w, h);
      const pWire = this.graph.wireAt({ id: agent.id, slot: 'p' });
      if (pWire) {
        const other = pWire.a.id === agent.id ? pWire.b : pWire.a;
        const otherA = this.agents.get(other.id);
        if (otherA) {
          const op = portWorld(otherA, other.slot, w, h);
          const d = wrapDeltaVec(tip.x, tip.y, op.x, op.y, w, h);
          biasX += d.x;
          biasY += d.y;
        }
      }

      if (agent.stun <= 0 && this.graph.isFreeAt(agent.id, 'p')) {
        this.bodyGrid.forEachNear(agent.x, agent.y, near, (idx) => {
          const other = list[idx];
          if (other.id === agent.id || other.locked || other.stun > 0) return;
          if (!this.graph.isFreeAt(other.id, 'p')) return;
          const d = wrapDeltaVec(agent.x, agent.y, other.x, other.y, w, h);
          const dist = Math.hypot(d.x, d.y);
          if (dist < 1e-4 || dist > params.faceRadius) return;
          const nx = d.x / dist;
          const ny = d.y / dist;
          const aFace = Math.cos(agent.heading) * nx + Math.sin(agent.heading) * ny;
          const bFace = Math.cos(other.heading) * -nx + Math.sin(other.heading) * -ny;
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
      const leftA = agent.heading - arc;
      const rightA = agent.heading + arc;
      const gain = 0.35 + params.sense / 500;
      const bm = Math.hypot(biasX, biasY);
      const scoreAt = (a: number): number => {
        const sx = agent.x + Math.cos(a) * dist;
        const sy = agent.y + Math.sin(a) * dist;
        let score = this.scentAt(agent, sx, sy, params) * gain;
        if (bm > 1e-6) {
          score += (1.6 * (biasX * Math.cos(a) + biasY * Math.sin(a))) / bm;
        }
        return score;
      };
      const left = scoreAt(leftA);
      const right = scoreAt(rightA);
      const dead = 0.05 * (Math.abs(left) + Math.abs(right)) + 0.03;
      let bestHeading = agent.heading;
      if (left > right + dead) bestHeading = leftA;
      else if (right > left + dead) bestHeading = rightA;

      const err = angleDelta(agent.heading, bestHeading);
      const trail = this.scentAt(agent, agent.x, agent.y, params);
      agent.trail = trail;
      const slow = scentSlowFactor(trail);
      const turnBoost = scentTurnBoost(trail);
      const kp = params.turnRate * 6 * turnBoost;
      const kd = (params.turnRate * 2) / Math.sqrt(turnBoost);
      const principalFree = this.graph.isFreeAt(agent.id, 'p');
      if (principalFree) {
        agent.omega += (kp * err - kd * agent.omega) * dt;
      }
      if (principalFree && params.stepSpeed > 0) {
        // Active Ornstein-Uhlenbeck propulsion: the drive decays toward cruise
        // with a persistence time while coloured noise kicks it. A swimmer
        // surges and eases the way a crawling cell does, where the servo this
        // replaces — drive velocity straight at a setpoint — reads mechanical.
        const cruise = params.stepSpeed * slow;
        const tau = Math.max(0.05, params.swimTau);
        const kick =
          params.swimNoise * cruise * Math.sqrt(dt / tau) *
          (Math.random() + Math.random() + Math.random() - 1.5) * 2;
        agent.drive += ((cruise - agent.drive) / tau) * dt + kick;
        agent.drive = clamp(agent.drive, -cruise * 0.4, cruise * 2.2);
        const hx = Math.cos(agent.heading);
        const hy = Math.sin(agent.heading);
        const along = agent.vx * hx + agent.vy * hy;
        const blend = 1 - Math.exp(-6 * dt);
        const dAlong = (agent.drive - along) * blend;
        agent.vx += dAlong * hx;
        agent.vy += dAlong * hy;
      }
    }
  }

  /**
   * Active locomotion: accelerate only along the principal port.
   * Latched principals don't swim; they just follow the pull.
   */
  private locomote(agent: Agent, wishX: number, wishY: number, turnK: number): void {
    if (!this.graph.isFreeAt(agent.id, 'p')) return;
    const hx = Math.cos(agent.heading);
    const hy = Math.sin(agent.heading);
    const ahead = wishX * hx + wishY * hy;
    if (ahead > 0) {
      agent.vx += ahead * hx;
      agent.vy += ahead * hy;
    }
    const mag = Math.hypot(wishX, wishY);
    if (mag > 1e-8 && turnK !== 0) {
      agent.omega += turnK * angleDelta(agent.heading, Math.atan2(wishY, wishX));
    }
  }

  /** Flock / constraint pulls on wired cargo (no principal swim). */
  private netPull(agent: Agent, wishX: number, wishY: number, _turnK: number): void {
    if (agent.locked) return;
    agent.vx += wishX;
    agent.vy += wishY;
  }

  private netForce(agent: Agent, wishX: number, wishY: number, turnK: number): void {
    if (this.graph.isFreeAt(agent.id, 'p')) this.locomote(agent, wishX, wishY, turnK);
    else this.netPull(agent, wishX, wishY, turnK);
  }

  private flockNative(
    list: Agent[],
    adj: number[][],
    swim: Uint8Array,
    n: number,
    align: number,
    sep: number,
    dt: number,
    turnRate: number,
    desired: number,
    maxHops: number,
  ): boolean {
    const bodies = nativeSolver.bodies;
    const adjOff = nativeSolver.adjOff;
    const adjNei = nativeSolver.adjNei;
    const ids = nativeSolver.flockId;
    const mass = nativeSolver.flockMass;
    const sw = nativeSolver.swim;
    if (!bodies || !adjOff || !adjNei || !ids || !mass || !sw) return false;

    let nAdj = 0;
    for (let i = 0; i < n; i++) nAdj += adj[i].length;
    if (!nativeSolver.canFlock(n, nAdj)) return false;
    adjOff[0] = 0;
    let at = 0;
    for (let i = 0; i < n; i++) {
      const nei = adj[i];
      for (let k = 0; k < nei.length; k++) adjNei[at++] = nei[k];
      adjOff[i + 1] = at;
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
        bodies[o + FAR.locked] = a.locked ? 1 : 0;
      }
      ids[i] = a.id;
      mass[i] = a.mass;
      sw[i] = swim[i];
    }
    if (!nativeSolver.flock(n, align, sep, dt, turnRate, desired, maxHops)) return false;
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
    const align = params.flockAlign;
    const sep = params.flockSep;
    if ((align <= 0 && sep <= 0) || dt <= 0) return;
    const list = this.agentList;
    list.length = 0;
    for (const a of this.agents.values()) list.push(a);
    const n = list.length;
    if (n === 0) return;

    const idx = this.flockIndex;
    idx.clear();
    for (let i = 0; i < n; i++) idx.set(list[i].id, i);

    const adj = this.flockAdj;
    while (adj.length < n) adj.push([]);
    for (let i = 0; i < n; i++) adj[i].length = 0;
    for (const wire of this.graph.wires.values()) {
      const ia = idx.get(wire.a.id);
      const ib = idx.get(wire.b.id);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      adj[ia].push(ib);
      adj[ib].push(ia);
    }

    if (this.flockDist.length < n) {
      const cap = Math.max(n * 2, 16);
      this.flockDist = new Int32Array(cap);
      this.flockDist.fill(-1);
      this.flockQ = new Int32Array(cap);
    }
    if (this.flockSwim.length < n) this.flockSwim = new Uint8Array(n * 2);
    const dist = this.flockDist;
    const q = this.flockQ;
    const swim = this.flockSwim;
    dist.fill(-1, 0, n);
    for (let i = 0; i < n; i++) {
      swim[i] = this.graph.isFreeAt(list[i].id, 'p') ? 1 : 0;
    }

    const maxHops = Sim.FLOCK_HOPS;
    const desired = Math.max(18, params.wireMinRest * 0.9);
    const turnRate = params.turnRate;
    if (this.flockNative(list, adj, swim, n, align, sep, dt, turnRate, desired, maxHops)) return;

    const seen = this.flockSeen;

    for (let start = 0; start < n; start++) {
      const A = list[start];
      if (A.locked) continue;
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
        const nei = adj[u];
        for (let k = 0; k < nei.length; k++) {
          const v = nei[k];
          if (dist[v] >= 0) continue;
          const d = du + 1;
          dist[v] = d;
          seen.push(v);
          q[qt++] = v;
          const B = list[v];
          if (B.locked || B.id <= A.id) continue;
          const w = 1 / d;
          const mA = Math.max(0.08, A.mass);
          const mB = Math.max(0.08, B.mass);
          const mSum = mA + mB;
          const dx = B.x - A.x;
          const dy = B.y - A.y;
          const gap = Math.hypot(dx, dy) || 1e-6;
          const nx = dx / gap;
          const ny = dy / gap;

          if (align > 0) {
            const kAlign = align * w * dt;
            const dvx = B.vx - A.vx;
            const dvy = B.vy - A.vy;
            this.netForce(A, dvx * kAlign * (mB / mSum), dvy * kAlign * (mB / mSum), 0);
            this.netForce(B, -dvx * kAlign * (mA / mSum), -dvy * kAlign * (mA / mSum), 0);
          }

          if (sep > 0 && d > 1) {
            const want = 22 + (d - 1) * desired;
            if (gap < want) {
              const mag = sep * w * (want - gap);
              const ax = nx * mag * dt;
              const ay = ny * mag * dt;
              const turn = turnRate * w * 0.25;
              this.netForce(A, -ax * (mB / mSum), -ay * (mB / mSum), swim[start] ? 0 : turn);
              this.netForce(B, ax * (mA / mSum), ay * (mA / mSum), swim[v] ? 0 : turn);
            }
          }
        }
      }
      for (let i = 0; i < seen.length; i++) dist[seen[i]] = -1;
    }
  }

  /** Ease the home point toward the live centre of mass. */
  private trackHome(dt: number): void {
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

  /**
   * Optional cohesion toward the flock centre, for loners and tiny latches
   * only. Larger nets keep the shape the wires gave them — a pull toward
   * their own centre of mass is what used to crumple machines into a ball.
   */
  private gravitate(params: Params, dt: number): void {
    const base = params.gravity;
    if (base <= 0 || dt <= 0) return;
    const com = this.home ?? this.centerOfMass();
    if (!com) return;
    const sizes = this.compSize;
    sizes.clear();
    for (const root of this.components.values()) {
      sizes.set(root, (sizes.get(root) ?? 0) + 1);
    }
    const cap = Sim.GRAV_MAX_COMP;
    if (this.gravitateNative(com.x, com.y, base, cap, dt)) return;
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      const root = this.components.get(agent.id) ?? agent.id;
      if ((sizes.get(root) ?? 1) > cap) continue;
      const dx = com.x - agent.x;
      const dy = com.y - agent.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 1e-6) continue;
      const pull = (base * Math.min(dist, Sim.HOME_REACH)) / dist;
      agent.vx += dx * pull * dt;
      agent.vy += dy * pull * dt;
    }
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

  private rewriteBusy(): Set<number> {
    const busy = new Set<number>();
    for (const rw of this.rewrites) {
      busy.add(rw.a);
      busy.add(rw.b);
    }
    return busy;
  }

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
   * The net's demand for energy, as one field, and one hop of flow along it.
   *
   * Two things want energy and they compete on the same scale. A body that
   * cannot pay its next bill needs whatever it is short by — read from the
   * upkeep accumulator, so the ask goes out while the body is still wired
   * rather than after `extra` has gone negative and the wires are already
   * gone. A stalled redex needs whatever each end is short of a full extra.
   *
   * Both are magnitudes in the same units, so nothing has to be ranked by
   * policy: a body two hops away and 0.9 short outpulls a redex next door
   * that is 0.1 short, and a surplus at one end of a net finds a shortage at
   * the other end by following the gradient a wire at a time.
   */
  private pulseRequests(params: Params): void {
    resetRequests(this.agents.values());
    const need = this.needOf;
    need.clear();

    for (const a of this.agents.values()) {
      if (a.locked) continue;
      const h = hungerNeed(a);
      if (h > 0) need.set(a.id, h);
    }

    if (params.rewriteDuration > 0) {
      const busy = this.rewriteBusy();
      for (const wire of this.graph.wires.values()) {
        const A = this.agents.get(wire.a.id);
        const B = this.agents.get(wire.b.id);
        if (!A || !B) continue;
        if (busy.has(A.id) || busy.has(B.id)) continue;
        if (!this.principalRedexReady(wire, A, B, params)) continue;
        const cost = rewriteCost(detectRule(A.kind, B.kind));
        if (cost <= 0 || extrasOf(A, B) >= cost) continue;
        for (const end of [A, B]) {
          const r = redexNeed(end);
          if (r > (need.get(end.id) ?? 0)) need.set(end.id, r);
        }
      }
    }

    for (const [id, n] of need) {
      const a = this.agents.get(id);
      if (a) seedRequest(a, n);
    }
    // Dense list and index, reused rather than rebuilt: the force block has
    // already made both, and a Map of neighbour arrays cost an array per body
    // per frame for a structure thrown away at the end of it.
    const list = this.forceList();
    const index = this.packIndex;
    if (!this.forceBlock) {
      index.clear();
      for (let i = 0; i < list.length; i++) index.set(list[i].id, i);
    }
    const adj = this.wireAdj;
    adj.build(list.length, index, () => this.graph.wires.values());
    spreadRequests(list, adj);
    const recoil = params.transportRecoil;
    flowCharges(
      list,
      adj,
      recoil > 0 ? (from, to, amount) => this.recoil(from, to, amount, recoil) : undefined,
    );
  }

  private recoil(
    from: { id: number },
    to: { id: number },
    amount: number,
    gain: number,
  ): void {
    const A = this.agents.get(from.id);
    const B = this.agents.get(to.id);
    if (A && B) applyTransportRecoil(A, B, amount, gain, this.w, this.h);
  }

  private startRewrites(params: Params): void {
    if (params.rewriteDuration <= 0) return;
    const busy = this.rewriteBusy();
    for (const wire of this.graph.wires.values()) {
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      if (busy.has(A.id) || busy.has(B.id)) continue;
      if (!this.principalRedexReady(wire, A, B, params)) continue;
      const rule = detectRule(A.kind, B.kind);
      const cost = rewriteCost(rule);
      if (cost > 0) {
        if (extrasOf(A, B) < cost) continue;
        let need = cost;
        if (canPayShare(A) && need > 0) {
          spendExtra(A);
          need--;
        }
        if (canPayShare(B) && need > 0) {
          spendExtra(B);
          need--;
        }
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
      this.rewrites.push(rw);
      audio.push(rewriteAudio(rw, 'begin', wire.id, []), this.graph, this.agents);
      busy.add(A.id);
      busy.add(B.id);
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
    for (const id of [rw.a, rw.b]) {
      const ag = this.agents.get(id);
      if (!ag) continue;
      for (const slot of slotsFor(ag.kind)) {
        const wire = this.graph.wireAt({ id, slot });
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
      let pool = rewriteYield(rw.rule);
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
      );
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
 * pushed east — equal and opposite, as an impulse, so a light Era twitches
 * where a Con barely stirs and the flock's centre of mass never moves. Pure
 * recoil without the matching kick would be a momentum pump: a net with a
 * standing gradient, surplus at one end and shortage at the other, would
 * accelerate in one direction forever.
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
