import {
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
import { queryHit, SLOP } from './collide.ts';
import { CHAIN_MASS, portExitAngle, solveContact, unwrapPoints } from './chain.ts';
import { CH, Fields } from './fields.ts';
import { Graph, wrapPos } from './graph.ts';
import type { Params } from './params.ts';
import {
  advanceRewrite,
  beginRewrite,
  commitRewrite,
  type Rewrite,
} from './rewrite.ts';
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
  masses = new Map<number, number>();
  /** Eased centre of mass. Rewrites delete agents, which jumps the true COM. */
  private home: { x: number; y: number } | null = null;

  constructor(w: number, h: number) {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    this.coverW = this.w;
    this.coverH = this.h;
    this.fields = new Fields(this.w, this.h);
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
    this.masses = this.graph.componentMass(this.agents);
    this.trackHome(t);

    this.steer(params, t);
    this.portTorques(params, t);
    this.flock(params, t);
    this.gravitate(params, t);
    this.solve(params, t);
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

  private solveContacts(h: number): void {
    const list = [...this.agents.values()];
    for (let i = 0; i < list.length; i++) {
      const A = list[i];
      for (let j = i + 1; j < list.length; j++) {
        const B = list[j];
        if (A.locked && B.locked) continue;
        const hit = queryHit(A, B, this.w, this.h);
        if (hit) solveContact(A, B, hit, SLOP, h);
      }
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
    const list = [...this.agents.values()];

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
        for (const other of list) {
          if (other.id === agent.id || other.locked || other.stun > 0) continue;
          if (!this.graph.isFree({ id: other.id, slot: 'p' })) continue;
          const d = wrapDeltaVec(agent.x, agent.y, other.x, other.y, w, h);
          const dist = Math.hypot(d.x, d.y);
          if (dist < 1e-4 || dist > params.faceRadius) continue;
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
        }
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
      this.rewrites.push(
        beginRewrite(A, B, this.graph, this.agents, this.w, this.h, params.rewriteDuration),
      );
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
