import { KIND_CON, KIND_DUP, KIND_ERA } from './native/solver.ts';
import { CHEM_LEN, CHEM_SPECIES, CRITIC_LEN, PLASTIC_LEN, ROW_COUNT, STATE_DIMS as STATE_W } from './chem-layout.ts';
import type { AgentKind } from './agents.ts';

/*
 * Backing storage for Agent, as parallel typed arrays indexed by a dense
 * "slot"; agents.ts's Agent class is a thin (store, slot) flyweight over
 * this. Kind coding is the WASM solver's own (KIND_ERA/DUP/CON). `chem`'s
 * width comes from `chem-layout.ts` and is never written down here.
 */
const SENSE_W = 4;

/** Process-wide, so two stores never hand out the same `chemVersion`. */
let nextChemVersion = 1;

export const KIND_CODE: Record<AgentKind, number> = { era: KIND_ERA, dup: KIND_DUP, con: KIND_CON };
export const CODE_KIND: AgentKind[] = [];
CODE_KIND[KIND_ERA] = 'era';
CODE_KIND[KIND_DUP] = 'dup';
CODE_KIND[KIND_CON] = 'con';

export class AgentStore {
  capacity = 0;
  /**
   * Bumped every time the backing arrays are reallocated. A cached view into
   * any array taken before a grow points at an abandoned buffer afterward;
   * callers that cache a view must check this first. See Agent.chem.
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
  extra!: Float64Array;
  request!: Float64Array;
  flockAlign!: Float64Array;
  flockSep!: Float64Array;
  /**
   * `chem`, all agents': slot `i`'s `CHEM_LEN` floats live at
   * `chemAll[i*CHEM_LEN .. (i+1)*CHEM_LEN)`. Float32, not Float64: the
   * genome's precision is part of the pond's behaviour and `stateHash` pins it.
   */
  chemAll!: Float32Array;
  recovering!: Uint8Array;
  requestDecay!: Float64Array;
  energyCap!: Float64Array;
  debtCap!: Float64Array;
  rescueTo!: Float64Array;
  /** How particulate this lineage's inheritance is. See `assortChance`. */
  assort!: Float64Array;
  transportRecoil!: Float64Array;
  /**
   * The gait, and the metabolism that is its clock. `sub` is the pathway's
   * upstream metabolite, bought out of the tank. `atp` is the charged part
   * of a conserved adenylate pool of `adenylate`, so `adp` is
   * `adenylate - atp` and is never stored: the pool is cycled, never minted.
   * `gaitWave` is the bare `cos(phase)`, which a wire's rest length rides on
   * in `Graph.syncRest`; `gaitAnchor` is the amplitude the `G` head reads
   * off `h`, and `anchor` is `gaitAnchor * gaitWave`, added to the drag rate.
   */
  sub!: Float64Array;
  atp!: Float64Array;
  adenylate!: Float64Array;
  gaitWave!: Float64Array;
  gaitAnchor!: Float64Array;
  anchor!: Float64Array;
  /** Whole units this body sends in one transfer. See `params.transportQuantum`. */
  transportQuantum!: Float64Array;
  /**
   * Which wire holds each of this body's three ports, or -1 for a free one.
   * Three entries a body: `slot * 3 + 0/1/2` for principal, left, right.
   * The graph owns the meaning; the store owns the storage so `clearSlot`
   * resets it and a newborn cannot inherit a corpse's wires.
   */
  portWire!: Int32Array;
  csHeading!: Float64Array;
  csCos!: Float64Array;
  csSin!: Float64Array;
  /**
   * Ancestry, which nothing in the sim reads — it exists to be measured:
   * how many rewrites deep a body is, and which founder it came from.
   */
  born!: Int32Array;
  lineage!: Int32Array;
  /**
   * Simulated seconds at which this slot's body arrived, or `-1` once it has
   * latched at least once. Written once by `allocate` from `now`, so every
   * creation path stamps it. Read only when a body latches or dies. See
   * `src/larval.ts`.
   */
  arrivedAt!: Float64Array;
  /** The simulated time `allocate` stamps onto a new slot; the Sim sets it once a frame. */
  now = 0;
  /**
   * Fraction of this body's ports that are attached, 0 to 1. Derived from
   * the graph; refreshed by `Sim.refreshBound` only when the topology changes.
   */
  bound!: Float64Array;
  /**
   * The recurrent internal state, `STATE_W` floats a body. State, not
   * genome: not inherited, not mutated. Zero for a fresh body.
   */
  hAll!: Float64Array;
  /** Last frame's four raw channel readings at each body's position. */
  senseAll!: Float64Array;
  /**
   * Last frame's three steering readings — left sensor, right sensor, own
   * position — each already collapsed against the body's taste, when the
   * field lives on the GPU. By slot, not list order: the list is rebuilt
   * whenever the roster moves between the probe and the steer.
   */
  steerAll!: Float64Array;
  /**
   * This frame's realised emit and taste vectors, four channels each,
   * materialised once by `updateState`. Emit is the normalised vector,
   * summing to one unit across the four channels: that is the budget.
   */
  emitAll!: Float64Array;
  tasteAll!: Float64Array;
  /**
   * This frame's expression vector: how the body divides one unit of chemical
   * effort across the reaction table's `ROW_COUNT` rows. See `expressVector`.
   * Computed on the host on both field paths: it is a pure function of `chem`
   * and `h`, and `unpackGenome` brings `h` back every frame.
   */
  expressAll!: Float64Array;
  /** What this body excreted this frame, per species. Absolute, not a rate. */
  excreteAll!: Float64Array;
  /**
   * What this body has swallowed and not yet turned into anything, per
   * species. The harvest swallows a sample it cannot choose
   * (`runHarvestPlan`), `Sim.runDigestion` converts what the recipe can, and
   * `Sim.runExcretion` dumps the rest back as itself: waste is the gap
   * between the sample and the recipe. Not on the genome shader and not in
   * the net blob; a captured net starts hungry.
   */
  gut!: Float64Array;
  /**
   * Whether this body's genome reads the scent field at all. Set by
   * `refreshReadsField`, which must run whenever `chem` is written.
   */
  readsField!: Uint8Array;
  /** This frame's locomotion head: cruise speed and turn gain, per body. */
  cruise!: Float64Array;
  turn!: Float64Array;
  /**
   * What this body has learned since it was born: a delta on the state
   * matrices, `PLASTIC_LEN` floats laid out exactly as `chem` lays the same
   * weights. The effective weight is `chem[k] + plastic[k]`. Separate from
   * `chem` so learned and inherited drift can be told apart and inheritance
   * has something to scale. Nothing here ever decays.
   */
  plasticAll!: Float32Array;
  /**
   * The eligibility trace, same shape as `plastic`. The one thing that
   * decays, at `params.learnTrace` a frame: the credit window, not forgetting.
   */
  traceAll!: Float32Array;
  /** The critic's weights on `h`, and its bias. See `CRITIC_LEN`. */
  criticAll!: Float64Array;
  /** Last frame's value estimate, for the temporal-difference error. */
  prevValue!: Float64Array;
  /**
   * Whether this body has learned anything at all yet. Monotone: set the
   * first time a learned weight goes non-zero and never cleared, which is
   * exact because nothing decays.
   */
  plasticOn!: Uint8Array;

  /**
   * Which slots' genomes have changed since a consumer last looked, as a
   * dirty range. Stamped from `refreshReadsField`, which must run after
   * anything writes `chem`. `chemVersion` is process-wide so a fresh store
   * after `Sim.clear()` cannot collide with a version a consumer remembers.
   */
  chemVersion = nextChemVersion++;
  chemDirtyLo = 0;
  chemDirtyHi = 0;

  /** Note that slot `slot`'s genome changed. */
  markChem(slot: number): void {
    if (this.chemDirtyHi <= this.chemDirtyLo) {
      this.chemDirtyLo = slot;
      this.chemDirtyHi = slot + 1;
    } else {
      if (slot < this.chemDirtyLo) this.chemDirtyLo = slot;
      if (slot + 1 > this.chemDirtyHi) this.chemDirtyHi = slot + 1;
    }
    this.chemVersion = nextChemVersion++;
  }

  /** Consumer has caught up to `chemVersion`; nothing is dirty. */
  clearChemDirty(): void {
    this.chemDirtyLo = 0;
    this.chemDirtyHi = 0;
  }

  /**
   * The same, for the learning state. On the GPU path the host only writes
   * it to zero a recycled slot, and that write must reach the device or the
   * previous occupant's experience goes to whoever moved in.
   */
  learnVersion = nextChemVersion++;
  learnDirtyLo = 0;
  learnDirtyHi = 0;

  /*
   * Which slots' learning rows the host has written, as a list: recycled
   * slots are scattered wherever the free list handed them out. The span is
   * the fallback for when the list overruns.
   */
  private static readonly LEARN_DIRTY_CAP = 512;
  learnDirtySlots = new Int32Array(AgentStore.LEARN_DIRTY_CAP);
  learnDirtyCount = 0;
  /** True when the list overran and the span is the only usable record. */
  learnDirtyAll = false;
  /** Per-slot stamp, so a slot marked twice in a frame is listed once. */
  private learnDirtyStamp!: Int32Array;
  private learnDirtyEpoch = 1;

  markLearn(slot: number): void {
    if (this.learnDirtyHi <= this.learnDirtyLo) {
      this.learnDirtyLo = slot;
      this.learnDirtyHi = slot + 1;
    } else {
      if (slot < this.learnDirtyLo) this.learnDirtyLo = slot;
      if (slot + 1 > this.learnDirtyHi) this.learnDirtyHi = slot + 1;
    }
    if (!this.learnDirtyAll) {
      if (this.learnDirtyStamp[slot] !== this.learnDirtyEpoch) {
        this.learnDirtyStamp[slot] = this.learnDirtyEpoch;
        if (this.learnDirtyCount < AgentStore.LEARN_DIRTY_CAP) {
          this.learnDirtySlots[this.learnDirtyCount++] = slot;
        } else {
          this.learnDirtyAll = true;
        }
      }
    }
    this.learnVersion = nextChemVersion++;
  }

  clearLearnDirty(): void {
    this.learnDirtyLo = 0;
    this.learnDirtyHi = 0;
    this.learnDirtyCount = 0;
    this.learnDirtyAll = false;
    // Bumping the epoch retires every stamp at once, so nothing has to be
    // cleared and a slot marked last frame is not mistaken for marked now.
    this.learnDirtyEpoch++;
  }

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
    this.arrivedAt[slot] = this.now;
    return slot;
  }

  /** Frees `id`'s slot for reuse. A no-op if `id` isn't live (defensive — callers should not double-release). */
  release(id: number): void {
    const slot = this.idToSlot.get(id);
    if (slot === undefined) return;
    this.idToSlot.delete(id);
    this.free.push(slot);
  }

  /** A reused slot must never leak state between two unrelated agents. */
  private clearSlot(slot: number): void {
    this.portWire[slot * 3] = -1;
    this.portWire[slot * 3 + 1] = -1;
    this.portWire[slot * 3 + 2] = -1;
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
    this.extra[slot] = 0;
    this.request[slot] = 0;
    this.flockAlign[slot] = 0;
    this.flockSep[slot] = 0;
    this.chemAll.fill(0, slot * CHEM_LEN, slot * CHEM_LEN + CHEM_LEN);
    this.markChem(slot);
    this.recovering[slot] = 0;
    this.requestDecay[slot] = 0;
    this.energyCap[slot] = 0;
    this.debtCap[slot] = 0;
    this.rescueTo[slot] = 0;
    this.assort[slot] = 0;
    this.transportRecoil[slot] = 0;
    this.sub[slot] = 0;
    this.atp[slot] = 0;
    this.adenylate[slot] = 0;
    this.gaitWave[slot] = 0;
    this.gaitAnchor[slot] = 0;
    this.anchor[slot] = 0;
    this.transportQuantum[slot] = 0;
    this.csHeading[slot] = 0;
    this.csCos[slot] = 0;
    this.csSin[slot] = 0;
    this.born[slot] = 0;
    this.lineage[slot] = 0;
    this.arrivedAt[slot] = 0;
    this.bound[slot] = 0;
    this.hAll.fill(0, slot * STATE_W, slot * STATE_W + STATE_W);
    this.senseAll.fill(0, slot * SENSE_W, slot * SENSE_W + SENSE_W);
    this.steerAll.fill(0, slot * 3, slot * 3 + 3);
    this.readsField[slot] = 0;
    this.cruise[slot] = 0;
    this.turn[slot] = 0;
    this.plasticAll.fill(0, slot * PLASTIC_LEN, slot * PLASTIC_LEN + PLASTIC_LEN);
    this.traceAll.fill(0, slot * PLASTIC_LEN, slot * PLASTIC_LEN + PLASTIC_LEN);
    this.criticAll.fill(0, slot * CRITIC_LEN, slot * CRITIC_LEN + CRITIC_LEN);
    this.prevValue[slot] = 0;
    this.plasticOn[slot] = 0;
    this.markLearn(slot);
    this.emitAll.fill(0, slot * 4, slot * 4 + 4);
    this.tasteAll.fill(0, slot * 4, slot * 4 + 4);
    this.expressAll.fill(0, slot * ROW_COUNT, slot * ROW_COUNT + ROW_COUNT);
    this.excreteAll.fill(0, slot * CHEM_SPECIES, slot * CHEM_SPECIES + CHEM_SPECIES);
    this.gut.fill(0, slot * CHEM_SPECIES, slot * CHEM_SPECIES + CHEM_SPECIES);
  }

  /** Everything this body is holding undigested, across all four species. */
  gutTotal(slot: number): number {
    const o = slot * CHEM_SPECIES;
    let n = 0;
    for (let c = 0; c < CHEM_SPECIES; c++) n += this.gut[o + c];
    return n;
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
    this.assort = growF64(this.assort);
    this.transportRecoil = growF64(this.transportRecoil);
    this.sub = growF64(this.sub);
    this.atp = growF64(this.atp);
    this.adenylate = growF64(this.adenylate);
    this.gaitWave = growF64(this.gaitWave);
    this.gaitAnchor = growF64(this.gaitAnchor);
    this.anchor = growF64(this.anchor);
    this.transportQuantum = growF64(this.transportQuantum);
    // Three a body, and -1 rather than 0 is the free marker, so a fresh tail
    // cannot read as "port held by wire 0".
    const newPortWire = new Int32Array(newCapacity * 3).fill(-1);
    if (this.portWire) newPortWire.set(this.portWire.subarray(0, live * 3));
    this.portWire = newPortWire;
    // Stamps are epoch-compared, so a grown tail of zeros reads as "not marked
    // this epoch" for any epoch past zero — which `learnDirtyEpoch` starts at.
    this.learnDirtyStamp = growI32(this.learnDirtyStamp);
    this.csHeading = growF64(this.csHeading);
    this.csCos = growF64(this.csCos);
    this.csSin = growF64(this.csSin);
    this.born = growI32(this.born);
    this.lineage = growI32(this.lineage);
    this.arrivedAt = growF64(this.arrivedAt);
    this.bound = growF64(this.bound);
    const newH = new Float64Array(newCapacity * STATE_W);
    if (this.hAll) newH.set(this.hAll.subarray(0, live * STATE_W));
    this.hAll = newH;
    const newSense = new Float64Array(newCapacity * SENSE_W);
    if (this.senseAll) newSense.set(this.senseAll.subarray(0, live * SENSE_W));
    this.senseAll = newSense;
    const newSteer = new Float64Array(newCapacity * 3);
    if (this.steerAll) newSteer.set(this.steerAll.subarray(0, live * 3));
    this.steerAll = newSteer;
    this.readsField = growU8(this.readsField);
    this.cruise = growF64(this.cruise);
    this.turn = growF64(this.turn);
    const newPlastic = new Float32Array(newCapacity * PLASTIC_LEN);
    if (this.plasticAll) newPlastic.set(this.plasticAll.subarray(0, live * PLASTIC_LEN));
    this.plasticAll = newPlastic;
    const newTrace = new Float32Array(newCapacity * PLASTIC_LEN);
    if (this.traceAll) newTrace.set(this.traceAll.subarray(0, live * PLASTIC_LEN));
    this.traceAll = newTrace;
    const newCritic = new Float64Array(newCapacity * CRITIC_LEN);
    if (this.criticAll) newCritic.set(this.criticAll.subarray(0, live * CRITIC_LEN));
    this.criticAll = newCritic;
    this.prevValue = growF64(this.prevValue);
    this.plasticOn = growU8(this.plasticOn);
    const newEmit = new Float64Array(newCapacity * 4);
    if (this.emitAll) newEmit.set(this.emitAll.subarray(0, live * 4));
    this.emitAll = newEmit;
    const newTaste = new Float64Array(newCapacity * 4);
    if (this.tasteAll) newTaste.set(this.tasteAll.subarray(0, live * 4));
    this.tasteAll = newTaste;
    const newExpress = new Float64Array(newCapacity * ROW_COUNT);
    if (this.expressAll) newExpress.set(this.expressAll.subarray(0, live * ROW_COUNT));
    this.expressAll = newExpress;
    const newExcrete = new Float64Array(newCapacity * CHEM_SPECIES);
    if (this.excreteAll) newExcrete.set(this.excreteAll.subarray(0, live * CHEM_SPECIES));
    this.excreteAll = newExcrete;
    const newGut = new Float64Array(newCapacity * CHEM_SPECIES);
    if (this.gut) newGut.set(this.gut.subarray(0, live * CHEM_SPECIES));
    this.gut = newGut;

    this.capacity = newCapacity;
    if (oldCapacity > 0) this.generation++;
  }
}
