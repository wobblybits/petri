import {
  boundRadius,
  createAgent,
  inSnapArc,
  momentOfInertia,
  portWorld,
  slotsFor,
  stemRoot,
  stemWorld,
  type Agent,
  type AgentKind,
  type PortSlot,
} from './agents.ts';
import { queryHit, SLOP, type Hit } from './collide.ts';
import { segmentsIntersect, WIRE_RADIUS } from './geom.ts';
import { PairGrid } from './grid.ts';
import { CHAIN_MASS, contactMechanics, portExitAngle, solveContact, unwrapPoints } from './chain.ts';
import { CH, Fields } from './fields.ts';
import { Graph, wrapPos, type Wire } from './graph.ts';
import type { Params } from './params.ts';
import {
  advanceRewrite,
  beginRewrite,
  commitRewrite,
  type Rewrite,
} from './rewrite.ts';
import { audio } from './audio/engine.ts';
import type { CollisionEvent, LiveContact, RewriteEvent } from './audio/types.ts';
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

  /** Local scent below which an agent counts as having lost the trail. */
  private static readonly HOME_SCENT = 0.35;

  /** Distance past which the pull home stops growing. */
  private static readonly HOME_REACH = 320;

  /** Repulsion in px/s² at exactly one wire's length; inverse-square inside that. */
  static DECLUTTER_FORCE = 200;

  /** Range of the repulsion, in wire lengths. */
  private static readonly DECLUTTER_CUTOFF = 2;

  /** Closest distance the inverse-square law is evaluated at, in wire lengths. */
  private static readonly DECLUTTER_FLOOR = 0.4;

  /** Most a rope node may be pushed clear in one substep, in px. */
  static WIRE_CLEAR_STEP = 0.35;

  /** Share of the homing pull that still applies to a wired agent. */
  private static readonly HOME_WIRED = 0.2;

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
  /** Connected-component root per agent, refreshed once per frame. */
  private components = new Map<number, number>();
  /** Broad-phase results for wire clearance, flattened pairs, rebuilt per frame. */
  private clearWirePairs: unknown[] = [];
  private clearBodyPairs: unknown[] = [];
  /** Broad-phase grids and their scratch coordinate arrays. */
  private bodyGrid = new PairGrid();
  private wireGrid = new PairGrid();
  private gx: number[] = [];
  private gy: number[] = [];
  private wx: number[] = [];
  private wy: number[] = [];
  private agentList: Agent[] = [];
  /** Pairs in contact last frame — a strike fires on onset, contact continues. */
  private contactAudioPrev = new Set<string>();
  private contactAudioNow = new Set<string>();
  /** Bodies sliding against something this frame, and how fast. */
  /** Bodies currently overlapping, keyed by canonical `lo:hi` id pair. */
  contacts = new Map<string, LiveContact>();
  /** Momentum lost to sound this frame, applied once after the substeps. */
  private radiated = new Map<number, { x: number; y: number }>();
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
    this.time = 0;
    this.nextId = 1;
    this.spawnAcc = 0;
    this.home = null;
    this.contactAudioPrev.clear();
    this.contactAudioNow.clear();
    this.contacts.clear();
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
    return a;
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

  step(dt: number, params: Params): void {
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

    this.steer(params, t);
    this.portTorques(params, t);
    this.declutter(params, t);
    this.uncrossPrincipals(params, t);
    this.flock(params, t);
    this.gravitate(params, t);
    this.solve(params, t);
    this.applyRadiationLoss();
    this.dampVelocities(params, t);

    this.graph.refreshLengths(this.agents, this.w, this.h);
    this.graph.snap(this.agents, this.w, this.h, params, this.time);
    this.startRewrites(params);
    this.tickRewrites(params, t);
    this.deposit(params);
    this.paintScentWalls();
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
  private portTorques(params: Params, dt: number): void {
    const gain = params.portStiff * 320;
    if (gain <= 0 || dt <= 0) return;
    const splay = params.auxSpread * 0.35;
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
   * Ropes push off other ropes, and off bodies they are not attached to.
   *
   * Node-only and one-way: a rope never moves an agent. That is what makes it
   * safe — letting a wire shove its own anchors is exactly the coupling that
   * made the early drafts of this solver explode. It separates wires that would
   * otherwise be drawn through each other; it does not forbid a crossing
   * topologically, since two wires that genuinely cross will bow apart and
   * still cross.
   *
   * Solved inside the substep loop rather than after the frame, so the link,
   * bend and shape constraints get to re-settle the rope around the push
   * instead of the rope ending each frame off its own manifold. It is rate
   * limited for the same reason everything else here is: a displacement
   * resolved in one substep becomes that displacement times 1/h in velocity.
   */
  /**
   * Wire pairs and wire/body pairs close enough to be worth testing, rebuilt
   * once per frame. The broad phase is O(wires²) and the narrow phase runs every
   * substep, so pairing them up each substep costs eight times what it needs to;
   * the margins here are generous enough that a frame of drift cannot smuggle a
   * pair past it.
   */
  private buildClearPairs(params: Params): void {
    this.clearWirePairs.length = 0;
    this.clearBodyPairs.length = 0;
    if (params.wireClear <= 0) return;
    const ropeGap = WIRE_RADIUS * 3;
    const slack = params.wireMinRest;

    const wires: Wire[] = [];
    let maxRope = 0;
    for (const wire of this.graph.wires.values()) {
      if (wire.nodes.length === 0) continue;
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
    // A cell this wide guarantees any pair whose reach could overlap lands in
    // the same cell or an adjacent one.
    this.wireGrid.build(this.wx, this.wy, m, maxRope + ropeGap + slack);

    this.wireGrid.forEachPair((i, j) => {
      const P = wires[i];
      const Q = wires[j];
      if (
        P.a.id === Q.a.id ||
        P.a.id === Q.b.id ||
        P.b.id === Q.a.id ||
        P.b.id === Q.b.id
      ) {
        return;
      }
      const span = (P.ropeLen + Q.ropeLen) * 0.5 + ropeGap + slack;
      const dx = this.wx[j] - this.wx[i];
      const dy = this.wy[j] - this.wy[i];
      if (dx * dx + dy * dy > span * span) return;
      this.clearWirePairs.push(P, Q);
    });

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
        const span = P.ropeLen * 0.5 + boundRadius(agent) + WIRE_RADIUS + slack;
        const dx = agent.x - px;
        const dy = agent.y - py;
        if (dx * dx + dy * dy > span * span) return;
        this.clearBodyPairs.push(P, agent);
      });
    }
  }


  private clearWires(params: Params): void {
    const gain = params.wireClear;
    if (gain <= 0) return;
    const ropeGap = WIRE_RADIUS * 3;
    const cap = Sim.WIRE_CLEAR_STEP;

    for (let k = 0; k < this.clearWirePairs.length; k += 2) {
      const P = this.clearWirePairs[k] as Wire;
      const Q = this.clearWirePairs[k + 1] as Wire;
      {
        for (const p of P.nodes) {
          for (const q of Q.nodes) {
            const dx = q.x - p.x;
            const dy = q.y - p.y;
            const d2 = dx * dx + dy * dy;
            if (d2 >= ropeGap * ropeGap || d2 < 1e-9) continue;
            const d = Math.sqrt(d2);
            const step = Math.min((ropeGap - d) * 0.5 * gain, cap);
            const ux = (dx / d) * step;
            const uy = (dy / d) * step;
            p.x -= ux;
            p.y -= uy;
            q.x += ux;
            q.y += uy;
          }
        }
      }
    }

    for (let k = 0; k < this.clearBodyPairs.length; k += 2) {
      const wire = this.clearBodyPairs[k] as Wire;
      const agent = this.clearBodyPairs[k + 1] as Agent;
      const keep = boundRadius(agent) + WIRE_RADIUS;
      for (const node of wire.nodes) {
        const dx = node.x - agent.x;
        const dy = node.y - agent.y;
        const d2 = dx * dx + dy * dy;
        if (d2 >= keep * keep || d2 < 1e-9) continue;
        const d = Math.sqrt(d2);
        const step = Math.min((keep - d) * gain, cap);
        node.x += (dx / d) * step;
        node.y += (dy / d) * step;
      }
    }
  }

  /**
   * Personal space around a fully wired agent: a soft inverse-square push
   * against agents from *other* nets.
   *
   * A saturated agent has nothing left to join, so a stranger drifting close is
   * pure crowding, and crowding is what tangles nets. Nothing else does this —
   * flocking separation reads `hopDistances`, which has no entry for an agent
   * in another component, so those pairs are skipped entirely and separate nets
   * have never repelled at all. Meanwhile homing actively pulls them together.
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
    const list = this.rebuildBodyGrid(cutoff);
    const n = list.length;
    // Hoisted out of the inner loop: both were map lookups per pair.
    const sat = new Uint8Array(n);
    const comp = new Int32Array(n);
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
    this.graph.syncRest(this.time, params);
    this.graph.syncRopeShape(this.agents, this.w, this.h);
    this.buildClearPairs(params);
    const h = dt / Sim.SUBSTEPS;
    const invH = 1 / h;
    // Rope velocity is re-derived every substep, so a nudge of e px becomes
    // e/h — damping it once per frame is far too late to keep a slack rope calm.
    const ropeKeep = Math.exp(-Math.max(0, params.springDamp) * h);
    const list = [...this.agents.values()];

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
        for (const node of wire.nodes) {
          node.prevX = node.x;
          node.prevY = node.y;
          node.x += node.vx * h;
          node.y += node.vy * h;
        }
      }

      this.graph.solveWires(this.agents, params, h, this.time);
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
      }
      for (const wire of this.graph.wires.values()) {
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
  }

  /** Approach speed that retriggers a strike while two bodies already touch. */
  private static readonly CONTACT_RETRIGGER = 3.2;
  /** Fraction of contact momentum radiated as sound instead of bounce. */
  private static readonly RADIATION = 0.035;

  private emitCollision(A: Agent, B: Agent, hit: Hit): void {
    const key = A.id < B.id ? `${A.id}:${B.id}` : `${B.id}:${A.id}`;
    const m = contactMechanics(A, B, hit);
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
      const hit = queryHit(A, B, this.w, this.h);
      if (hit) {
        this.emitCollision(A, B, hit);
        solveContact(A, B, hit, SLOP, h);
      }
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
    for (const wire of this.graph.wires.values()) {
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B) continue;
      const raw = [
        stemWorld(A, wire.a.slot, this.w, this.h),
        ...wire.nodes,
        stemWorld(B, wire.b.slot, this.w, this.h),
      ];
      const pts = unwrapPoints(raw, this.w, this.h);
      for (let i = 0; i < pts.length - 1; i++) {
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
      const reach = 0.28 * Math.min(this.coverW, this.coverH);
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

  private deposit(params: Params): void {
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      for (const slot of slotsFor(agent.kind)) {
        if (!this.graph.isFree({ id: agent.id, slot })) continue;
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

  private steer(params: Params, dt: number): void {
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

      if (agent.stun <= 0 && this.graph.isFree({ id: agent.id, slot: 'p' })) {
        this.bodyGrid.forEachNear(agent.x, agent.y, near, (idx) => {
          const other = list[idx];
          if (other.id === agent.id || other.locked || other.stun > 0) return;
          if (!this.graph.isFree({ id: other.id, slot: 'p' })) return;
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
      const principalFree = this.graph.isFree({ id: agent.id, slot: 'p' });
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
    if (!this.graph.isFree({ id: agent.id, slot: 'p' })) return;
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
    if (this.graph.isFree({ id: agent.id, slot: 'p' })) this.locomote(agent, wishX, wishY, turnK);
    else this.netPull(agent, wishX, wishY, turnK);
  }

  /**
   * Boids on the net. Weight is 1/hops; disconnected pairs are ignored.
   * Meridians align nematically (parallel, either polarity), velocities match,
   * clumps separate, and junctions sit toward neighbor centroids so wires straighten.
   */
  private flock(params: Params, dt: number): void {
    const align = params.flockAlign;
    const sep = params.flockSep;
    if ((align <= 0 && sep <= 0) || dt <= 0) return;
    const hops = this.graph.hopDistances(this.agents);
    const list = [...this.agents.values()];
    const byId = this.agents;
    const desired = Math.max(18, params.wireMinRest * 0.9);
    const nbrs = new Map<number, number[]>();
    for (const a of list) nbrs.set(a.id, []);
    for (const wire of this.graph.wires.values()) {
      if (wire.a.id === wire.b.id) continue;
      if (!byId.has(wire.a.id) || !byId.has(wire.b.id)) continue;
      nbrs.get(wire.a.id)!.push(wire.b.id);
      nbrs.get(wire.b.id)!.push(wire.a.id);
    }

    for (let i = 0; i < list.length; i++) {
      const A = list[i];
      if (A.locked) continue;
      const fromA = hops.get(A.id);
      if (!fromA) continue;
      for (let j = i + 1; j < list.length; j++) {
        const B = list[j];
        if (B.locked) continue;
        const d = fromA.get(B.id);
        if (d === undefined || d < 1) continue;
        const w = 1 / d;
        const mA = Math.max(0.08, A.mass);
        const mB = Math.max(0.08, B.mass);
        const mSum = mA + mB;
        const delta = wrapDeltaVec(A.x, A.y, B.x, B.y, this.w, this.h);
        const dist = Math.hypot(delta.x, delta.y) || 1e-6;
        const nx = delta.x / dist;
        const ny = delta.y / dist;

        if (align > 0) {
          const k = align * w * dt;
          const dvx = B.vx - A.vx;
          const dvy = B.vy - A.vy;
          this.netForce(A, dvx * k * (mB / mSum), dvy * k * (mB / mSum), 0);
          this.netForce(B, -dvx * k * (mA / mSum), -dvy * k * (mA / mSum), 0);
        }

        if (sep > 0 && d > 1) {
          const want = 22 + (d - 1) * desired;
          if (dist < want) {
            const mag = sep * w * (want - dist);
            const ax = nx * mag * dt;
            const ay = ny * mag * dt;
            const turn = params.turnRate * w * 0.25;
            const aSwims = this.graph.isFree({ id: A.id, slot: 'p' });
            const bSwims = this.graph.isFree({ id: B.id, slot: 'p' });
            this.netForce(A, -ax * (mB / mSum), -ay * (mB / mSum), aSwims ? 0 : turn);
            this.netForce(B, ax * (mA / mSum), ay * (mA / mSum), bSwims ? 0 : turn);
          }
        }
      }
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
   * Cohesion toward the flock's centre — which is the centre of the map as
   * seen, since the camera and the scent field are both built around it.
   *
   * The homing share is scaled by how little an agent can smell. A swimmer on a
   * trail is left alone to follow it; one that has lost the scent completely
   * turns for home instead of wandering off. Measured over a minute of soup,
   * agents within 200 px of the flock sit around 0.2 scent while everything
   * past 600 px reads exactly zero, so the two cases separate cleanly.
   */
  private gravitate(params: Params, dt: number): void {
    const base = params.gravity;
    const home = params.homing;
    if ((base <= 0 && home <= 0) || dt <= 0) return;
    const com = this.home ?? this.centerOfMass();
    if (!com) return;
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      const dx = com.x - agent.x;
      const dy = com.y - agent.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 1e-6) continue;
      let k = base;
      if (home > 0) {
        // Mostly for loose foragers. A wired agent is already held in place by
        // its net, and hauling whole nets inward just crowds the flock, which
        // is what pushes aux wires across each other.
        const anchored = this.graph.isWired(agent) ? Sim.HOME_WIRED : 1;
        k += (home * anchored) / (1 + agent.trail / Sim.HOME_SCENT);
      }
      // Saturating: a spring close in, a steady walk home from far out, so a
      // stray is not slingshot back through the flock.
      const pull = (k * Math.min(dist, Sim.HOME_REACH)) / dist;
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

  private startRewrites(params: Params): void {
    if (params.rewriteDuration <= 0) return;
    const busy = new Set<number>();
    for (const rw of this.rewrites) {
      busy.add(rw.a);
      busy.add(rw.b);
    }
    for (const wire of this.graph.wires.values()) {
      if (wire.a.slot !== 'p' || wire.b.slot !== 'p') continue;
      const A = this.agents.get(wire.a.id);
      const B = this.agents.get(wire.b.id);
      if (!A || !B || A.locked || B.locked || A.stun > 0 || B.stun > 0) continue;
      if (busy.has(A.id) || busy.has(B.id)) continue;
      if (!this.graph.portsFilled(A) || !this.graph.portsFilled(B)) continue;
      const shrinkU = this.graph.shrinkU(wire, this.time, params);
      if (shrinkU < 1) continue;
      if (this.time - wire.born < params.wireShrink + 0.2) continue;
      const len = this.graph.curveLength(wire, this.agents, this.w, this.h);
      if (len > params.wireMinRest + 3) continue;
      A.vx = 0;
      A.vy = 0;
      A.omega = 0;
      B.vx = 0;
      B.vy = 0;
      B.omega = 0;
      const rw = beginRewrite(A, B, this.graph, this.agents, this.w, this.h, params.rewriteDuration);
      this.rewrites.push(rw);
      audio.push(rewriteAudio(rw, 'begin', wire.id, []), this.graph, this.agents);
      busy.add(A.id);
      busy.add(B.id);
    }
  }

  private tickRewrites(params: Params, dt: number): void {
    const done: Rewrite[] = [];
    for (const rw of this.rewrites) {
      if (advanceRewrite(rw, this.agents, this.w, this.h, dt)) done.push(rw);
    }
    for (const rw of done) {
      audio.push(rewriteAudio(rw, 'commit', 0, leftoverAgentIds(rw)), this.graph, this.agents);
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
    }
    if (done.length) this.rewrites = this.rewrites.filter((rw) => !done.includes(rw));
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
