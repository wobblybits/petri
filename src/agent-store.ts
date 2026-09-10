import { KIND_CON, KIND_DUP, KIND_ERA } from './native/solver.ts';
import { CHEM_LEN, CHEM_SPECIES, CRITIC_LEN, PLASTIC_LEN, ROW_COUNT, STATE_DIMS as STATE_W } from './chem-layout.ts';
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
 * `chem`'s width comes from `chem-layout.ts`, which exists so that it can.
 * It used to be hand-copied here, on the grounds that a structural fact is
 * safe to duplicate; it drifted twice in one session and the second time
 * shipped a `RangeError` out of every spawn.
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
  extra!: Float64Array;
  request!: Float64Array;
  flockAlign!: Float64Array;
  flockSep!: Float64Array;
  /**
   * `chem`, all agents': slot `i`'s `CHEM_LEN` floats live at
   * `chemAll[i*CHEM_LEN .. (i+1)*CHEM_LEN)`. The width comes from
   * `chem-layout.ts` and is not written down here — see the note above.
   *
   * Float32, not Float64 like every other field here: this matches the
   * original plain-object `Agent`'s own `chem: Float32Array` (a deliberate
   * memory tradeoff predating this store, when it was sixteen floats a body
   * rather than the current 134). Storing it at full float64 precision would be
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
  /** How particulate this lineage's inheritance is. See `assortChance`. */
  assort!: Float64Array;
  transportThrust!: Float64Array;
  transportRecoil!: Float64Array;
  /**
   * The gait, and the metabolism that is now its clock.
   *
   * `atp` and `adp` are Selkov's two metabolites — substrate and product,
   * where the product activates the enzyme that makes it. `gaitAnchor` is
   * the amplitude the `G` head reads off `h`, and `anchor` is what it comes
   * to this frame: `gaitAnchor * gaitWave`, added to this body's drag rate.
   *
   * `gaitWave` is the bare `cos(phase)`, with no amplitude in it, and is what
   * a wire's rest length rides on in `Graph.syncRest`.
   *
   * One actuator, not two. `rest` is what this engine already moves things
   * with — `wireShrink` reels a latch in with it, `Wire.collapse` hauls a
   * rewrite's ends together with it, `wireBreathe` makes tissue move with it
   * — because the span constraint *serves* it rather than fighting it. The
   * gait is the fourth thing that writes it. The travel is then whatever
   * `anchor` and `grip` leave of the velocity that correction induces, which
   * costs no second mechanism.
   */
  atp!: Float64Array;
  adp!: Float64Array;
  gaitWave!: Float64Array;
  gaitAnchor!: Float64Array;
  anchor!: Float64Array;
  /** Whole units this body sends in one transfer. See `params.transportQuantum`. */
  transportQuantum!: Float64Array;
  /**
   * Which wire holds each of this body's three ports, or -1 for a free one.
   * Three entries a body: `slot * 3 + 0/1/2` for principal, left, right.
   *
   * The graph owns the meaning; the store owns the storage, because the
   * question "is this port free" is asked of a *slot* tens of thousands of
   * times a frame — the latch pass asks it of every port of every body, and
   * both GPU packs ask it again for the free-port mask. It was a `Map` keyed
   * on `agentId * 3 + slot`, and at thirty thousand bodies the latch pass
   * alone spent 3.8 ms a frame hashing into it.
   *
   * Living here also closes a hazard rather than opening one: `clearSlot`
   * runs when a slot is recycled, so a newborn cannot inherit a corpse's
   * wires the way it could when the map was keyed on an id nobody cleared.
   */
  portWire!: Int32Array;
  csHeading!: Float64Array;
  csCos!: Float64Array;
  csSin!: Float64Array;
  /**
   * Ancestry, which nothing in the sim reads — it exists to be measured.
   *
   * Every heritable trait in this project drifts as well as adapts, and with
   * `CHEM_MUTATE` across a genome this size the drift is not small. Nothing here
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
   * Simulated seconds at which this slot's body arrived, or `-1` once it has
   * latched at least once.
   *
   * The larval-window question — does a body reach a net before its tank runs
   * out — needs the age of a body that has never joined anything, and nothing
   * else here records when a body began. Written once by `allocate` from
   * `now`, which the Sim sets at the top of a frame, so no creation path has
   * to be found and threaded: `createAgent`, a rewrite's children and a
   * planted net all go through `allocate`. Read only when a body latches or
   * dies. See `src/larval.ts`.
   */
  arrivedAt!: Float64Array;
  /**
   * The simulated time `allocate` stamps onto a new slot. The Sim owns the
   * clock; this is the store's copy of it, set once a frame.
   */
  now = 0;
  /**
   * Fraction of this body's ports that are attached, 0 to 1.
   *
   * Derived from the graph, cached here because `updateState` reads it per
   * body per frame as `IN_BOUND` and walking the wire adjacency that often
   * would cost more than the state update itself. Refreshed by
   * `Sim.refreshBound` only when the topology actually changes, which is what
   * it depends on.
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
  /**
   * Last frame's three steering readings — left sensor, right sensor, own
   * position — each already collapsed against the body's taste, when the
   * field lives on the GPU and the probe brings them back.
   *
   * By slot, like `senseAll`, and for a reason that bit: they used to go
   * straight into the solver's array in *list* order, and the list is rebuilt
   * whenever the roster moves. A body spawned or erased between the probe at
   * the end of one frame and the steer at the top of the next shifted every
   * body after it onto a neighbour's readings, and a body born past the old
   * end of the list steered on whatever the buffer last held there.
   */
  steerAll!: Float64Array;
  /**
   * This frame's realised emit and taste vectors, four channels each.
   *
   * Materialised once by `updateState` rather than recomputed by each
   * consumer. The scent pass asked for emit four times a body and the steer
   * pass asked for taste four times a body, each call walking the genome
   * again — 13.8 ms a frame at 20k between them, for two vectors that are pure
   * functions of `h` and could not have changed since it was computed.
   *
   * Emit here is the *normalised* vector: what the body actually says, summing
   * to one unit across the four channels. That is the budget, and it is
   * enforced here because here is the only place all four are known at once.
   */
  emitAll!: Float64Array;
  tasteAll!: Float64Array;
  /**
   * This frame's expression vector: how the body divides one unit of chemical
   * effort across the reaction table's `ROW_COUNT` rows. See `expressVector`
   * and `docs/energy-chemistry-plan.md` §3.
   *
   * Materialised here for the same reason `emitAll` is — two passes want it in
   * the same frame, excretion and uptake, and recomputing thirty-two
   * multiply-adds a body twice is the shape of cost this store exists to
   * remove.
   *
   * Computed on the host on **both** field paths, because it is a pure
   * function of `chem` and `h` and `unpackGenome` brings `h` back every frame.
   * That is what keeps the whole reaction table off the genome shader: it
   * needs no new binding and no new output slot, and the arithmetic is a
   * rounding error beside the passes that already run here.
   */
  expressAll!: Float64Array;
  /** What this body excreted this frame, per species. Absolute, not a rate. */
  excreteAll!: Float64Array;
  /**
   * Whether this body's genome reads the scent field at all.
   *
   * `updateState` re-derived it every body every frame — sixteen `Float32`
   * reads to answer a question whose answer cannot change, because a genome is
   * fixed for a body's life. All three kinds seed with every `Wx` sense column
   * at zero, so for a fresh pond the answer is always no and the whole gate was
   * overhead. Set by `refreshReadsField` whenever `chem` is written.
   */
  readsField!: Uint8Array;
  /** This frame's locomotion head: cruise speed and turn gain, per body. */
  cruise!: Float64Array;
  turn!: Float64Array;
  /**
   * What this body has learned since it was born: a delta on the state
   * matrices, `PLASTIC_LEN` floats laid out exactly as `chem` lays the same
   * weights. The effective weight is `chem[k] + plastic[k]`.
   *
   * Separate from `chem` rather than written into it so that learned drift
   * and inherited drift can be told apart by anything measuring the pond,
   * and so inheritance has something to scale. **Nothing here ever decays.**
   * A body carries what it learned into whatever net it latches into next,
   * and that transfer is what lets one net's experience reach another.
   */
  plasticAll!: Float32Array;
  /**
   * The eligibility trace, same shape as `plastic`.
   *
   * This one does decay, at `params.learnTrace` a frame, and that is a
   * different thing from forgetting: it is the credit window, about the time
   * a transfer takes to show up in the tank, so that a weight is rewarded
   * for what it was doing shortly before things improved.
   */
  traceAll!: Float32Array;
  /** The critic's weights on `h`, and its bias. See `CRITIC_LEN`. */
  criticAll!: Float64Array;
  /** Last frame's value estimate, for the temporal-difference error. */
  prevValue!: Float64Array;
  /**
   * Whether this body has learned anything at all yet.
   *
   * Monotone: set the first time a learned weight goes non-zero and never
   * cleared, which is exact precisely because nothing decays. A pond with
   * learning switched off, or one whose bodies have not learned yet, reads
   * its genome the way it always did and pays one branch a body for the
   * privilege.
   */
  plasticOn!: Uint8Array;

  /**
   * Which slots' genomes have changed since a consumer last looked.
   *
   * The GPU genome pass reads `chemAll` by slot and was uploading the whole
   * table every frame — 2.7 MB at five thousand bodies, 27 MB at fifty — on
   * the grounds that a missed invalidation would be a body silently running
   * somebody else's genome. The invalidation has one choke point already:
   * `refreshReadsField` must be called after anything writes `chem`, or the
   * sense gate goes stale, so it is the right place to stamp this too.
   *
   * A dirty *range* rather than a flag, so a frame with one birth uploads one
   * genome. `chemVersion` is a process-wide counter rather than a per-store
   * one so a fresh store after `Sim.clear()` cannot collide with the version a
   * consumer remembers from the store it replaced.
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
   * The same, for the learning state.
   *
   * On the GPU path the device owns this and the host only ever writes it to
   * zero a slot — but that write matters more than most: a recycled slot
   * whose learning was left on the device would hand the previous occupant's
   * experience to whoever moved in.
   */
  learnVersion = nextChemVersion++;
  learnDirtyLo = 0;
  learnDirtyHi = 0;

  /*
   * Which slots' learning rows the host has written, as a list rather than a
   * span.
   *
   * A span is the wrong shape for this. The host writes a learning row in one
   * place — zeroing a slot that has just been recycled — so the dirty slots
   * are wherever the free list happened to hand out, which is everywhere.
   * Measured on a grown pond: **62 dirty slots a frame, in 62 separate runs,
   * spanning 24,088 slots.** The span carried three hundred and eighty-nine
   * times more than it needed to, and interleaving it into the upload buffer
   * cost 11.9 ms a frame — most of the genome pack.
   *
   * The span is kept as the fallback for when the list overruns, which is the
   * case a list is bad at and a span is fine at.
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

  /**
   * Zeroed rather than left with the previous occupant's data — a reused
   * slot must never leak state between two unrelated agents. `chemAll`'s
   * slice is zeroed too, via `.fill(0, ...)` on the same span the getter
   * views.
   */
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
    this.transportThrust[slot] = 0;
    this.transportRecoil[slot] = 0;
    // Off both axes, or the reaction has nothing to start from: `adp` at
    // zero makes the autocatalytic term zero and the pathway never lights.
    this.atp[slot] = 1;
    this.adp[slot] = 0.5;
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
    this.transportThrust = growF64(this.transportThrust);
    this.transportRecoil = growF64(this.transportRecoil);
    this.atp = growF64(this.atp);
    this.adp = growF64(this.adp);
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

    this.capacity = newCapacity;
    if (oldCapacity > 0) this.generation++;
  }
}
