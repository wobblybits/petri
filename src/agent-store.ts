import { KIND_CON, KIND_DUP, KIND_ERA } from './native/solver.ts';
import type { AgentKind } from './agents.ts';

/*
 * Backing storage for Agent, as parallel typed arrays indexed by a dense
 * "slot" rather than one JS object per agent. See agents.ts's Agent class,
 * which is a thin (store, slot) flyweight over this.
 *
 * Reuses the WASM solver's own kind coding (KIND_ERA/DUP/CON) rather than
 * inventing a second one — the two are already the same concept, just for
 * different memory.
 *
 * `chem` is a duplicate of agents.ts's CHEM_LEN rather than an import, to
 * keep the dependency direction one-way: agents.ts builds its flyweight on
 * top of this module, so this module cannot import back from agents.ts.
 * Thirty-two is a structural fact about the chem layout (emit/taste, then a
 * slopes, four channels each), not a tunable — safe to duplicate.
 */
const CHEM_LEN = 104;
/** Width of the recurrent state `h`, and of one body's cached scent reading. */
const STATE_W = 4;
const SENSE_W = 4;

export const KIND_CODE: Record<AgentKind, number> = { era: KIND_ERA, dup: KIND_DUP, con: KIND_CON };
export const CODE_KIND: AgentKind[] = [];
CODE_KIND[KIND_ERA] = 'era';
CODE_KIND[KIND_DUP] = 'dup';
CODE_KIND[KIND_CON] = 'con';

export class AgentStore {
  capacity = 0;
  /**
   * Bumped every time the backing arrays are reallocated (growth). A cached
   * view into `chemAll` (or any other array) taken before a grow points at
   * an abandoned buffer afterward — callers that cache a view must check
   * this first. See Agent.chem in agents.ts.
   */
  generation = 0;

  id!: Int32Array;
  kindCode!: Uint8Array;
  x!: Float64Array;
  y!: Float64Array;
  vx!: Float64Array;
  vy!: Float64Array;
  heading!: Float64Array;
  omega!: Float64Array;
  mass!: Float64Array;
  alpha!: Float64Array;
  scale!: Float64Array;
  locked!: Uint8Array;
  pinned!: Uint8Array;
  stun!: Float64Array;
  drive!: Float64Array;
  trail!: Float64Array;
  prevX!: Float64Array;
  prevY!: Float64Array;
  prevHeading!: Float64Array;
  integVx!: Float64Array;
  integVy!: Float64Array;
  integOmega!: Float64Array;
  extra!: Float64Array;
  request!: Float64Array;
  flockAlign!: Float64Array;
  flockSep!: Float64Array;
  /**
   * `chem`, all agents': slot `i`'s 32 floats live at `chemAll[i*32 .. i*32+32)`.
   *
   * Float32, not Float64 like every other field here: this matches the
   * original plain-object `Agent`'s own `chem: Float32Array` (a deliberate
   * memory tradeoff predating this store, since it's 16 floats per agent
   * rather than one). Storing it at full float64 precision instead would be
   * a real, if tiny, behavior change from what Phase 1 promises to preserve
   * exactly — confirmed by a stateHash before/after mismatch that traced
   * back to exactly this.
   */
  chemAll!: Float32Array;
  recovering!: Uint8Array;
  requestDecay!: Float64Array;
  energyCap!: Float64Array;
  debtCap!: Float64Array;
  rescueTo!: Float64Array;
  transportThrust!: Float64Array;
  transportRecoil!: Float64Array;
  csHeading!: Float64Array;
  csCos!: Float64Array;
  csSin!: Float64Array;
  /**
   * Ancestry, which nothing in the sim reads — it exists to be measured.
   *
   * Every heritable trait in this project drifts as well as adapts, and with
   * `CHEM_MUTATE` on thirty-two genes the drift is not small. Nothing here
   * could previously tell the two apart: a test could show a genome had moved
   * away from its seed, which is what the scent-genome test does, and that is
   * equally consistent with selection and with a random walk. `born` and
   * `lineage` are what make the question answerable — how many rewrites deep a
   * body is, and which founder it came from.
   *
   * `Int32Array` and never read per frame, so this is 8 bytes a body and no
   * cost in the hot path.
   */
  born!: Int32Array;
  lineage!: Int32Array;
  /**
   * Fraction of this body's ports that are attached, 0 to 1.
   *
   * Derived from the graph, cached here because `chemState` reads it per
   * channel per body and walking the wire adjacency that often would cost more
   * than the whole state vector. Refreshed by `Sim.refreshBound` only when the
   * topology actually changes, which is what it depends on.
   */
  bound!: Float64Array;
  /**
   * The recurrent internal state, `STATE_W` floats a body.
   *
   * State, not genome: it is not inherited and not mutated, it is what the
   * genome's matrices compute from one frame to the next. Zero for a fresh
   * body, which with zero-seeded matrices makes a newborn behave exactly as a
   * bodiless one did.
   */
  hAll!: Float64Array;
  /** Last frame's four raw channel readings at each body's position. */
  senseAll!: Float64Array;

  /** Slots < highWater have been allocated at least once (live or freed). */
  private highWater = 0;
  /** Released slots available for reuse, all < highWater. */
  private free: number[] = [];
  private idToSlot = new Map<number, number>();

  constructor(initialCapacity = 64) {
    this.growTo(Math.max(1, initialCapacity));
  }

  /** Live agent count. */
  get size(): number {
    return this.highWater - this.free.length;
  }

  slotFor(id: number): number | undefined {
    return this.idToSlot.get(id);
  }

  /** A fresh slot, registered under `id`. Every field starts zeroed/false — callers fill them in. */
  allocate(id: number): number {
    let slot: number;
    if (this.free.length > 0) {
      slot = this.free.pop()!;
      this.clearSlot(slot);
    } else {
      if (this.highWater >= this.capacity) this.growTo(this.capacity * 2);
      slot = this.highWater++;
    }
    this.id[slot] = id;
    this.idToSlot.set(id, slot);
    return slot;
  }

  /** Frees `id`'s slot for reuse. A no-op if `id` isn't live (defensive — callers should not double-release). */
  release(id: number): void {
    const slot = this.idToSlot.get(id);
    if (slot === undefined) return;
    this.idToSlot.delete(id);
    this.free.push(slot);
  }

  /**
   * Zeroed rather than left with the previous occupant's data — a reused
   * slot must never leak state between two unrelated agents. `chemAll`'s
   * slice is zeroed too, via `.fill(0, ...)` on the same span the getter
   * views.
   */
  private clearSlot(slot: number): void {
    this.id[slot] = 0;
    this.kindCode[slot] = 0;
    this.x[slot] = 0;
    this.y[slot] = 0;
    this.vx[slot] = 0;
    this.vy[slot] = 0;
    this.heading[slot] = 0;
    this.omega[slot] = 0;
    this.mass[slot] = 0;
    this.alpha[slot] = 0;
    this.scale[slot] = 0;
    this.locked[slot] = 0;
    this.pinned[slot] = 0;
    this.stun[slot] = 0;
    this.drive[slot] = 0;
    this.trail[slot] = 0;
    this.prevX[slot] = 0;
    this.prevY[slot] = 0;
    this.prevHeading[slot] = 0;
    this.integVx[slot] = 0;
    this.integVy[slot] = 0;
    this.integOmega[slot] = 0;
    this.extra[slot] = 0;
    this.request[slot] = 0;
    this.flockAlign[slot] = 0;
    this.flockSep[slot] = 0;
    this.chemAll.fill(0, slot * CHEM_LEN, slot * CHEM_LEN + CHEM_LEN);
    this.recovering[slot] = 0;
    this.requestDecay[slot] = 0;
    this.energyCap[slot] = 0;
    this.debtCap[slot] = 0;
    this.rescueTo[slot] = 0;
    this.transportThrust[slot] = 0;
    this.transportRecoil[slot] = 0;
    this.csHeading[slot] = 0;
    this.csCos[slot] = 0;
    this.csSin[slot] = 0;
    this.born[slot] = 0;
    this.lineage[slot] = 0;
    this.bound[slot] = 0;
    this.hAll.fill(0, slot * STATE_W, slot * STATE_W + STATE_W);
    this.senseAll.fill(0, slot * SENSE_W, slot * SENSE_W + SENSE_W);
  }

  private growTo(newCapacity: number): void {
    const oldCapacity = this.capacity;
    const live = this.highWater;

    const growF64 = (old: Float64Array | undefined): Float64Array => {
      const next = new Float64Array(newCapacity);
      if (old) next.set(old.subarray(0, live));
      return next;
    };
    const growU8 = (old: Uint8Array | undefined): Uint8Array => {
      const next = new Uint8Array(newCapacity);
      if (old) next.set(old.subarray(0, live));
      return next;
    };
    const growI32 = (old: Int32Array | undefined): Int32Array => {
      const next = new Int32Array(newCapacity);
      if (old) next.set(old.subarray(0, live));
      return next;
    };

    this.id = growI32(this.id);
    this.kindCode = growU8(this.kindCode);
    this.x = growF64(this.x);
    this.y = growF64(this.y);
    this.vx = growF64(this.vx);
    this.vy = growF64(this.vy);
    this.heading = growF64(this.heading);
    this.omega = growF64(this.omega);
    this.mass = growF64(this.mass);
    this.alpha = growF64(this.alpha);
    this.scale = growF64(this.scale);
    this.locked = growU8(this.locked);
    this.pinned = growU8(this.pinned);
    this.stun = growF64(this.stun);
    this.drive = growF64(this.drive);
    this.trail = growF64(this.trail);
    this.prevX = growF64(this.prevX);
    this.prevY = growF64(this.prevY);
    this.prevHeading = growF64(this.prevHeading);
    this.integVx = growF64(this.integVx);
    this.integVy = growF64(this.integVy);
    this.integOmega = growF64(this.integOmega);
    this.extra = growF64(this.extra);
    this.request = growF64(this.request);
    this.flockAlign = growF64(this.flockAlign);
    this.flockSep = growF64(this.flockSep);
    const newChemAll = new Float32Array(newCapacity * CHEM_LEN);
    if (this.chemAll) newChemAll.set(this.chemAll.subarray(0, live * CHEM_LEN));
    this.chemAll = newChemAll;
    this.recovering = growU8(this.recovering);
    this.requestDecay = growF64(this.requestDecay);
    this.energyCap = growF64(this.energyCap);
    this.debtCap = growF64(this.debtCap);
    this.rescueTo = growF64(this.rescueTo);
    this.transportThrust = growF64(this.transportThrust);
    this.transportRecoil = growF64(this.transportRecoil);
    this.csHeading = growF64(this.csHeading);
    this.csCos = growF64(this.csCos);
    this.csSin = growF64(this.csSin);
    this.born = growI32(this.born);
    this.lineage = growI32(this.lineage);
    this.bound = growF64(this.bound);
    const newH = new Float64Array(newCapacity * STATE_W);
    if (this.hAll) newH.set(this.hAll.subarray(0, live * STATE_W));
    this.hAll = newH;
    const newSense = new Float64Array(newCapacity * SENSE_W);
    if (this.senseAll) newSense.set(this.senseAll.subarray(0, live * SENSE_W));
    this.senseAll = newSense;

    this.capacity = newCapacity;
    if (oldCapacity > 0) this.generation++;
  }
}
