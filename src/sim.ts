import {
  createAgent,
  inSnapArc,
  momentOfInertia,
  portWorld,
  slotsFor,
  type Agent,
  type AgentKind,
  type PortSlot,
} from './agents.ts';
import { collideWireAgents, collideWires, queryHit, resolveHit } from './collide.ts';
import { CHAIN_MASS } from './chain.ts';
import { CH, Fields } from './fields.ts';
import { Graph, wrapPos } from './graph.ts';
import type { Params } from './params.ts';
import {
  advanceRewrite,
  beginRewrite,
  commitRewrite,
  type Rewrite,
} from './rewrite.ts';
import {
  angleDelta,
  clamp,
  nematicDelta,
  wrapAngle,
  wrapDeltaVec,
} from './wrap.ts';

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

export class Sim {
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
    this.steer(params, t);
    this.flock(params, t);
    this.gravitate(params, t);
    this.integrate(params, t);
    this.graph.applySprings(this.agents, this.w, this.h, params, t, this.time);
    this.separate();
    this.reconstruct(t);
    this.damp(params, t);
    this.graph.snap(this.agents, this.w, this.h, params, this.time);
    this.startRewrites(params);
    this.tickRewrites(params, t);
    this.deposit(params);
    this.fields.diffuse(params.diffuse);
    this.fields.diffuse(params.diffuse * 0.65);
    this.fields.decay(params.decay);
    this.autoSpawn(params, t);
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
      const Mcomp = this.masses.get(agent.id) ?? agent.mass;

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
      const kp = params.turnRate * 6;
      const kd = params.turnRate * 2;
      const principalFree = this.graph.isFree({ id: agent.id, slot: 'p' });
      if (principalFree) {
        agent.omega += (kp * err - kd * agent.omega) * dt;
        if (params.stepSpeed > 0) {
          const cruise = params.stepSpeed * (agent.mass / Math.max(0.2, Mcomp));
          const hx = Math.cos(agent.heading);
          const hy = Math.sin(agent.heading);
          const along = agent.vx * hx + agent.vy * hy;
          const blend = 1 - Math.exp(-10 * dt);
          const dAlong = (cruise - along) * blend;
          agent.vx += dAlong * hx;
          agent.vy += dAlong * hy;
        }
      }
    }
  }

  /**
   * Nematic turn: parallel is enough, either polarity.
   * Latched agents still steer this way — it is facing, not swimming.
   */
  private alignHeading(agent: Agent, target: number, gain: number, dt: number): void {
    if (gain <= 0 || agent.locked) return;
    const I = Math.max(1e-4, momentOfInertia(agent));
    agent.omega += (gain * nematicDelta(agent.heading, target) * dt) / I;
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
    const portToward = new Map<string, PortSlot>();
    for (const a of list) nbrs.set(a.id, []);
    for (const wire of this.graph.wires.values()) {
      if (wire.a.id === wire.b.id) continue;
      if (!byId.has(wire.a.id) || !byId.has(wire.b.id)) continue;
      nbrs.get(wire.a.id)!.push(wire.b.id);
      nbrs.get(wire.b.id)!.push(wire.a.id);
      portToward.set(`${wire.a.id}:${wire.b.id}`, wire.a.slot);
      portToward.set(`${wire.b.id}:${wire.a.id}`, wire.b.slot);
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
          const turn = params.turnRate * w * 0.35;
          const dvx = B.vx - A.vx;
          const dvy = B.vy - A.vy;
          this.locomote(A, dvx * k * (mB / mSum), dvy * k * (mB / mSum), turn);
          this.locomote(B, -dvx * k * (mA / mSum), -dvy * k * (mA / mSum), turn);

          const slotA = portToward.get(`${A.id}:${B.id}`);
          const slotB = portToward.get(`${B.id}:${A.id}`);
          if (d === 1) {
            const axis = Math.atan2(ny, nx);
            const chord = align * 2.6;
            if (slotA === 'p') this.alignHeading(A, axis, chord, dt);
            if (slotB === 'p') this.alignHeading(B, axis + Math.PI, chord, dt);
          }
          this.alignHeading(A, B.heading, align * w * 2.1, dt);
          this.alignHeading(B, A.heading, align * w * 2.1, dt);

          const spd = Math.hypot(A.vx + B.vx, A.vy + B.vy);
          if (spd > 5) {
            const motion = Math.atan2(A.vy + B.vy, A.vx + B.vx);
            const face = align * w * 0.85;
            this.alignHeading(A, motion, face, dt);
            this.alignHeading(B, motion, face, dt);
          }
        }

        if (sep > 0) {
          const want = 22 + (d - 1) * desired;
          if (dist < want) {
            const mag = sep * w * (want - dist);
            const ax = nx * mag * dt;
            const ay = ny * mag * dt;
            const turn = params.turnRate * w * 0.25;
            this.locomote(A, -ax * (mB / mSum), -ay * (mB / mSum), turn);
            this.locomote(B, ax * (mA / mSum), ay * (mA / mSum), turn);
          }
        }
      }
    }

    if (align <= 0) return;
    const straight = align * 0.65;
    for (const agent of list) {
      if (agent.locked) continue;
      const ids = nbrs.get(agent.id);
      if (!ids || ids.length < 2) continue;
      let cx = 0;
      let cy = 0;
      let massN = 0;
      const others: Agent[] = [];
      for (const id of ids) {
        const n = byId.get(id);
        if (!n || n.locked) continue;
        others.push(n);
        cx += n.x;
        cy += n.y;
        massN += Math.max(0.08, n.mass);
      }
      if (others.length < 2) continue;
      cx /= others.length;
      cy /= others.length;
      const dx = cx - agent.x;
      const dy = cy - agent.y;
      const mI = Math.max(0.08, agent.mass);
      const k = straight * dt;
      const turn = params.turnRate * 0.2;
      this.locomote(agent, dx * k, dy * k, turn);
      const share = (mI * k) / massN;
      for (const n of others) {
        this.locomote(n, -dx * share, -dy * share, turn);
      }
      if (Math.hypot(dx, dy) > 1e-4) {
        this.alignHeading(agent, Math.atan2(dy, dx), align * 1.1, dt);
      }
      let cxH = 0;
      let syH = 0;
      for (const n of others) {
        const h = agent.heading + nematicDelta(agent.heading, n.heading);
        cxH += Math.cos(h);
        syH += Math.sin(h);
      }
      this.alignHeading(agent, Math.atan2(syH, cxH), align * 1.4, dt);
    }
  }

  private gravitate(params: Params, dt: number): void {
    const g = params.gravity;
    if (g <= 0) return;
    const com = this.centerOfMass();
    if (!com) return;
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      agent.vx += -g * (agent.x - com.x) * dt;
      agent.vy += -g * (agent.y - com.y) * dt;
    }
  }

  private integrate(params: Params, dt: number): void {
    for (const agent of this.agents.values()) {
      agent.prevX = agent.x;
      agent.prevY = agent.y;
      agent.prevHeading = agent.heading;
      agent.integVx = agent.vx;
      agent.integVy = agent.vy;
      agent.integOmega = agent.omega;
      if (agent.locked) {
        wrapPos(agent, this.w, this.h);
        continue;
      }
      agent.x += agent.vx * dt;
      agent.y += agent.vy * dt;
      agent.heading = wrapAngle(agent.heading + agent.omega * dt);
      agent.stun = Math.max(0, agent.stun - dt);
      wrapPos(agent, this.w, this.h);
    }
    for (const wire of this.graph.wires.values()) {
      for (const node of wire.nodes) {
        node.prevX = node.x;
        node.prevY = node.y;
        node.integVx = node.vx;
        node.integVy = node.vy;
      }
    }
  }

  private reconstruct(dt: number): void {
    const invDt = 1 / Math.max(1e-6, dt);
    const slack = 3;
    const wSlack = 1.5;
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      agent.vx = (agent.x - agent.prevX) * invDt;
      agent.vy = (agent.y - agent.prevY) * invDt;
      agent.omega = wrapAngle(agent.heading - agent.prevHeading) * invDt;
      const cap = Math.hypot(agent.integVx, agent.integVy) + slack;
      const speed = Math.hypot(agent.vx, agent.vy);
      if (speed > cap && speed > 1e-8) {
        const s = cap / speed;
        agent.vx *= s;
        agent.vy *= s;
      }
      const wCap = Math.abs(agent.integOmega) + wSlack;
      if (Math.abs(agent.omega) > wCap) {
        agent.omega = Math.sign(agent.omega) * wCap;
      }
    }
    for (const wire of this.graph.wires.values()) {
      for (const node of wire.nodes) {
        node.vx = (node.x - node.prevX) * invDt;
        node.vy = (node.y - node.prevY) * invDt;
        const cap = Math.hypot(node.integVx, node.integVy) + slack;
        const speed = Math.hypot(node.vx, node.vy);
        if (speed > cap && speed > 1e-8) {
          const s = cap / speed;
          node.vx *= s;
          node.vy *= s;
        }
      }
    }
  }

  private damp(params: Params, dt: number): void {
    const nodeKeep = Math.exp(-Math.max(0, params.drag) * dt);
    for (const agent of this.agents.values()) {
      if (agent.locked) continue;
      const cargo = !this.graph.isFree({ id: agent.id, slot: 'p' });
      const drag = Math.max(0, params.drag) + (cargo ? 3.5 : 0);
      const linKeep = Math.exp(-drag * dt);
      const angKeep = Math.exp(-Math.max(0, params.angDrag) * dt * (cargo ? 2.2 : 1));
      const shape = agent.kind === 'era' ? 0.75 : 1;
      agent.vx *= Math.pow(linKeep, shape);
      agent.vy *= Math.pow(linKeep, shape);
      agent.omega *= Math.pow(angKeep, shape);
    }
    for (const wire of this.graph.wires.values()) {
      for (const node of wire.nodes) {
        node.vx *= nodeKeep;
        node.vy *= nodeKeep;
      }
    }
  }

  private separate(): void {
    const list = [...this.agents.values()];
    for (let pass = 0; pass < 6; pass++) {
      for (let i = 0; i < list.length; i++) {
        const A = list[i];
        if (A.locked) continue;
        for (let j = i + 1; j < list.length; j++) {
          const B = list[j];
          if (B.locked) continue;
          const hit = queryHit(A, B, this.w, this.h);
          if (!hit) continue;
          resolveHit(A, B, hit, this.w, this.h);
        }
      }
    }
    collideWireAgents(this.graph, this.agents, this.w, this.h);
    collideWires(this.graph, this.agents, this.w, this.h);
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
      const len = this.graph.curveLength(wire, this.agents, this.w, this.h);
      if (len > params.wireMinRest + 3) continue;
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
