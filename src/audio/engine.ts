import type { Agent } from '../agents.ts';
import type { Graph } from '../graph.ts';
import {
  planCollisionMessages,
  planLatchMessages,
  planRewriteMessages,
  planSpawnMessages,
  planContactMessage,
  planAirMessage,
} from './dispatch.ts';
import { buildTopology } from './topology.ts';
import { LodSelector } from './lod.ts';
import { setSampleRate } from './presets.ts';
import { makeReverbIR } from './reverb.ts';
import type { AudioEvent, LiveContact, NetTopology, PanView, WaveSnapshot, WorkletInMessage } from './types.ts';
import workletUrl from './worklet/net-processor.ts?url';
import workerUrl from './worklet/net-worker.ts?url';
import { AudioRing, ringBytes } from './ring.ts';
import {
  SHARD_COUNT,
  SOUP_SLOT,
  ShardAssigner,
  emptyTopology,
  partitionPairs,
  splitTopology,
} from './shards.ts';

/**
 * Structural events (latch, rewrite, spawn, pluck) allowed per frame.
 * Collisions are not in this budget — a visible knock has to sound even when
 * the pond is busy rewriting.
 */
const EVENTS_PER_FRAME = 5;
/**
 * Simultaneous knocks that may fire in one frame, loudest first.
 *
 * The sim already emits at most one strike per pair per contact episode, so
 * this is a pile-up valve, not a musical thin. It used to share the five-event
 * structural budget, which is how a latching soup went visually busy and
 * audibly mute.
 */
const COLLISION_BUDGET = 24;
/** Seconds an agent stays quiet after a structural event. Collisions skip this. */
const AGENT_COOLDOWN = 0.09;
const WIRE_COOLDOWN = 0.05;
/** Traveling-wave records kept for the renderer, across every shard. */
const WAVE_VIZ_CAP = 8;

/**
 * Frames of buffer between the worker and the audio callback.
 *
 * This is the latency, exactly: 1024 frames is 21 ms at 48 kHz. It is also
 * the whole protection budget — a producer that stalls has this long to
 * recover before the callback runs dry — so the two cannot be traded
 * separately. Wires can be plucked and bodies dragged, and interaction starts
 * to feel detached somewhere north of 30 ms, which is what sets the ceiling.
 */
const RING_FRAMES = 1024;

/** SharedArrayBuffer needs a cross-origin-isolated page (COOP + COEP). */
function sharedMemoryAvailable(): boolean {
  return (
    typeof SharedArrayBuffer !== 'undefined' &&
    typeof Worker !== 'undefined' &&
    typeof globalThis.crossOriginIsolated === 'boolean' &&
    globalThis.crossOriginIsolated
  );
}

function eventGain(ev: AudioEvent): number {
  if (ev.type === 'latch') return 1.5;
  if (ev.type === 'rewrite') return ev.phase === 'commit' ? 1.4 : 1;
  // A spawn outranks a light bump but yields to a latch, so a busy frame keeps
  // the structural events and drops the incidental ones.
  if (ev.type === 'spawn') return 1.2;
  return Math.min(1, ev.impact / 40);
}

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  /** Soup slots pack small islands; dedicated slots are self-contained nets. Empty until workers start. */
  private workers: Worker[] = [];
  private rings: AudioRing[] = [];
  private master: GainNode | null = null;
  private ready: Promise<boolean> | null = null;
  private muted = false;
  private armed = false;
  private events: AudioEvent[] = [];
  private graph: Graph | null = null;
  private agents: Map<number, Agent> | null = null;
  /** Wall clock in seconds, advanced by frame(); drives the cooldowns. */
  private now = 0;
  private lastAgent = new Map<number, number>();
  private lastWire = new Map<number, number>();
  /** Detail tiers carry frame-to-frame, so a boundary-sitting wire holds still. */
  private lod = new LodSelector();
  private readonly assigner = new ShardAssigner();
  private shardOf = new Map<number, number>();
  private parts: NetTopology[] = [];
  private readonly topoKeys = new Array<number>(SHARD_COUNT).fill(-1);
  private readonly poseKeys = new Array<number>(SHARD_COUNT).fill(-1);
  private readonly tuneKeys = new Array<number>(SHARD_COUNT).fill(-1);
  private readonly contactKeys = new Array<number>(SHARD_COUNT).fill(-1);
  private readonly airKeys = new Array<number>(SHARD_COUNT).fill(-1);
  private readonly contactedLast = new Array<boolean>(SHARD_COUNT).fill(false);
  private readonly airedLast = new Array<boolean>(SHARD_COUNT).fill(false);
  onPost: ((msg: WorkletInMessage) => void) | null = null;
  /** Latest traveling-wave snapshot from the worklet. Null until the first one. */
  private wavePacked: Float32Array | null = null;
  private waveIndex = new Map<number, number>();
  private readonly waveBySlot: (Float32Array | null)[] = new Array(SHARD_COUNT).fill(null);
  private readonly waveMerged = new Float32Array(2 + WAVE_VIZ_CAP * (2 + 32 * 2));

  get waves(): WaveSnapshot | null {
    return this.wavePacked ? { packed: this.wavePacked, index: this.waveIndex } : null;
  }

  /** True when each large net has its own synth worker. */
  get sharded(): boolean {
    return this.workers.length === SHARD_COUNT;
  }

  /** @internal Tests feed a snapshot without a worklet. */
  acceptWaves(packed: Float32Array, slot = SOUP_SLOT): void {
    if (slot < 0 || slot >= SHARD_COUNT) slot = SOUP_SLOT;
    this.waveBySlot[slot] = packed;
    this.rebuildWaves();
  }

  private rebuildWaves(): void {
    this.waveIndex.clear();
    let bins = 32;
    type Rec = { env: number; src: Float32Array; offset: number };
    const recs: Rec[] = [];
    // Dedicated nets first: those are the machines whose travelling waves
    // the renderer is trying to show. Soup slots follow, high index first.
    for (let s = SHARD_COUNT - 1; s >= 0; s--) {
      const packed = this.waveBySlot[s];
      if (!packed || packed.length < 2) continue;
      const n = packed[0] | 0;
      if (n <= 0) continue;
      bins = packed[1] | 0;
      const stride = 2 + bins * 2;
      let o = 2;
      for (let i = 0; i < n; i++) {
        if (o + stride > packed.length) break;
        recs.push({ env: packed[o + 1], src: packed, offset: o });
        o += stride;
      }
    }
    if (recs.length === 0) {
      this.wavePacked = null;
      return;
    }
    const stride = 2 + bins * 2;
    const take = recs.length < WAVE_VIZ_CAP ? recs.length : WAVE_VIZ_CAP;
    this.waveMerged[0] = take;
    this.waveMerged[1] = bins;
    let o = 2;
    for (let i = 0; i < take; i++) {
      const r = recs[i];
      this.waveMerged.set(r.src.subarray(r.offset, r.offset + stride), o);
      this.waveIndex.set(this.waveMerged[o] | 0, o);
      o += stride;
    }
    this.wavePacked = this.waveMerged.subarray(0, o);
  }

  get isMuted(): boolean {
    return this.muted;
  }

  get isArmed(): boolean {
    return this.armed;
  }

  setMuted(m: boolean): void {
    this.muted = m;
    // Ramp rather than step: a jump on master.gain is an audible click.
    if (this.master && this.ctx) {
      this.master.gain.setTargetAtTime(m ? 0 : 1, this.ctx.currentTime, 0.02);
    }
    this.post({ type: 'gain', master: m ? 0 : 1 });
  }

  push(ev: AudioEvent, graph: Graph, agents: Map<number, Agent>): void {
    this.graph = graph;
    this.agents = agents;
    if (!this.armed || this.muted) return;
    // Queue only. frame() drains collisions separately from structural events
    // so a busy rewrite cannot swallow a visible knock.
    if (this.events.length < 128) this.events.push(ev);
  }

  async boot(): Promise<boolean> {
    if (typeof window === 'undefined' || !window.AudioContext) return false;
    const ok = await this.initAudio();
    if (!ok || !this.ctx) return false;
    await this.ctx.resume();
    this.armed = true;
    this.setMuted(this.muted);
    return true;
  }

  private async initAudio(): Promise<boolean> {
    if (this.ctx && this.node) return true;
    if (this.ready) {
      const prior = await this.ready;
      // A failed attempt must not poison every later one.
      if (prior) return true;
      this.ready = null;
    }
    this.ready = this.createContext();
    return this.ready;
  }

  private async createContext(): Promise<boolean> {
    // No forced sampleRate: pinning 48 kHz makes 44.1 kHz hardware resample the
    // whole output, and throws outright on some configurations.
    const ctx = new AudioContext({ latencyHint: 'interactive' });
    try {
      setSampleRate(ctx.sampleRate);
      await ctx.audioWorklet.addModule(workletUrl);
      const node = new AudioWorkletNode(ctx, 'net-processor', {
        numberOfInputs: 0,
        numberOfOutputs: 2,
        outputChannelCount: [2, 2],
      });
      node.onprocessorerror = (ev) => {
        console.error('audio worklet error', ev);
      };
      node.port.onmessage = (ev: MessageEvent) => {
        const msg = ev.data;
        if (msg && msg.type === 'waves' && msg.packed instanceof Float32Array) {
          this.acceptWaves(msg.packed);
        }
        if (msg && msg.type === 'error') {
          console.error('audio worklet', msg.message);
        }
      };

      // Two buses from the worklet: dry (falls with distance) and a wet send
      // (almost flat). The hall is roughly uniform, so D/R is the distance cue.
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 110;

      const dry = ctx.createGain();
      // Dry and wet sum at the master, and each bus is soft-clipped to 1 in the
      // worklet, so their gains have to sum to 1 for the destination to be
      // safe at every instant. 0.85 + 0.7 could reach 1.55 and hard clip on a
      // dense moment. Same ratio between them, just scaled to fit.
      dry.gain.value = this.busGains().dry;
      const wet = ctx.createGain();
      wet.gain.value = this.busGains().wet;

      const verb = ctx.createConvolver();
      verb.normalize = true;
      verb.buffer = makeReverbIR(ctx);

      const master = ctx.createGain();
      master.gain.value = this.muted ? 0 : 1;

      node.connect(hp, 0);
      hp.connect(dry);
      dry.connect(master);
      node.connect(verb, 1);
      verb.connect(wet);
      wet.connect(master);
      master.connect(ctx.destination);

      this.ctx = ctx;
      this.node = node;
      this.master = master;
      this.startWorkers(ctx, node);
      this.post({ type: 'gain', master: this.muted ? 0 : 1 });
      return true;
    } catch (err) {
      console.warn('audio init failed', err);
      await ctx.close();
      this.ctx = null;
      this.node = null;
      this.master = null;
      return false;
    }
  }

  private post(msg: WorkletInMessage, slot?: number): void {
    this.onPost?.(msg);
    // Whoever owns the net gets the control messages. In ring mode the
    // worklet owns nothing but the copy out of shared memory.
    if (this.workers.length > 0) {
      if (slot === undefined) {
        for (const w of this.workers) w.postMessage(msg);
        return;
      }
      this.workers[slot]?.postMessage(msg);
      return;
    }
    if (!this.node) return;
    this.node.port.postMessage(msg);
  }

  private slotOf(id: number): number {
    if (!this.sharded) return SOUP_SLOT;
    return this.shardOf.get(id) ?? SOUP_SLOT;
  }

  /**
   * How full the ring is, 0..1, or null when rendering in the worklet.
   *
   * With several workers this is the emptiest ring — the one that will
   * underrun first. A dedicated net that is keeping up does not hide a
   * starving soup.
   */
  get ringFill(): number | null {
    if (this.rings.length === 0) return null;
    let worst = 1;
    for (const r of this.rings) {
      const f = r.fill();
      if (f < worst) worst = f;
    }
    return worst;
  }

  /** Per-slot fill, soup first. Null when rendering in the worklet. */
  get shardFills(): number[] | null {
    if (this.rings.length === 0) return null;
    return this.rings.map((r) => r.fill());
  }

  /** Quanta the callback has had to pad with silence. Should stay at zero. */
  get underruns(): number | null {
    if (this.rings.length === 0) return null;
    let n = 0;
    for (const r of this.rings) n += r.underruns();
    return n;
  }

  /**
   * Move synthesis onto a pool of Workers, each feeding a shared ring.
   *
   * Slot 0..SOUP_COUNT-1 pack small islands. The rest are for nets that have
   * grown large enough to be their own instrument. Returns false when the
   * page is not cross-origin-isolated, or a Worker cannot be built — in which
   * case the caller keeps the in-worklet path, which still works and is what
   * everything did before.
   *
   * Each Worker is started from a bootstrap that sets `sampleRate` and only
   * then pulls in the bundle. waveguide.ts reads that global once, at load,
   * and bakes filter coefficients out of it; an ordinary module import would
   * evaluate first and tune the whole net for 48 kHz on 44.1 kHz hardware.
   */
  private startWorkers(ctx: AudioContext, node: AudioWorkletNode): boolean {
    if (!sharedMemoryAvailable()) {
      const isolated = globalThis.crossOriginIsolated === true;
      console.warn(
        isolated
          ? 'audio workers unavailable (SharedArrayBuffer or Worker missing)'
          : 'page is not cross-origin isolated; synthesis stays on the audio thread. Need COOP same-origin + COEP require-corp on this document.',
      );
      return false;
    }
    let url = '';
    const workers: Worker[] = [];
    const rings: AudioRing[] = [];
    try {
      const target = new URL(workerUrl, location.href).href;
      const boot = `self.sampleRate=${ctx.sampleRate};importScripts(${JSON.stringify(target)});`;
      url = URL.createObjectURL(new Blob([boot], { type: 'text/javascript' }));
      for (let slot = 0; slot < SHARD_COUNT; slot++) {
        const worker = new Worker(url);
        const sab = new SharedArrayBuffer(ringBytes(RING_FRAMES));
        const ring = new AudioRing(sab, RING_FRAMES);
        const captured = slot;
        worker.onmessage = (ev: MessageEvent) => {
          const msg = ev.data;
          if (msg && msg.type === '__ready') {
            // Only now does the callback start reading this slot. Handing it
            // the ring at once means it drains an empty buffer while the
            // worker is still rendering its first pass.
            node.port.postMessage({ type: '__ring', sab, capacity: RING_FRAMES, slot: captured });
            return;
          }
          if (msg && msg.type === 'waves' && msg.packed instanceof Float32Array) {
            this.acceptWaves(msg.packed, captured);
          }
          if (msg && msg.type === 'error') console.error('audio worker', captured, msg.message);
        };
        worker.onerror = (ev) => console.error('audio worker', captured, ev.message);
        worker.postMessage({
          type: '__ring',
          sab,
          capacity: RING_FRAMES,
          sampleRate: ctx.sampleRate,
        });
        workers.push(worker);
        rings.push(ring);
      }
      this.workers = workers;
      this.rings = rings;
      return true;
    } catch (err) {
      console.warn('audio worker unavailable, rendering in the worklet', err);
      for (const w of workers) w.terminate();
      this.workers = [];
      this.rings = [];
      return false;
    } finally {
      if (url) URL.revokeObjectURL(url);
    }
  }

  /** Cooldowns keep one agent or wire from retriggering every frame. */
  private allow(ev: AudioEvent): boolean {
    if (ev.type === 'spawn') return true;
    if (ev.type === 'latch') {
      if ((this.lastWire.get(ev.wireId) ?? -1) > this.now - WIRE_COOLDOWN) return false;
      this.lastWire.set(ev.wireId, this.now);
      return true;
    }
    const a = ev.agentA;
    const b = ev.agentB;
    if ((this.lastAgent.get(a) ?? -1) > this.now - AGENT_COOLDOWN) return false;
    if ((this.lastAgent.get(b) ?? -1) > this.now - AGENT_COOLDOWN) return false;
    this.lastAgent.set(a, this.now);
    this.lastAgent.set(b, this.now);
    return true;
  }

  private drainEvents(): void {
    if (!this.armed || !this.node || !this.graph || !this.agents) return;
    if (this.events.length === 0) return;
    const pending = this.events.splice(0, this.events.length);
    const knocks: AudioEvent[] = [];
    const structural: AudioEvent[] = [];
    for (const ev of pending) {
      if (ev.type === 'collision') knocks.push(ev);
      else structural.push(ev);
    }
    structural.sort((x, y) => eventGain(y) - eventGain(x));
    let fired = 0;
    for (const ev of structural) {
      if (fired >= EVENTS_PER_FRAME) break;
      if (!this.allow(ev)) continue;
      this.dispatch(ev);
      fired++;
    }
    knocks.sort((x, y) => eventGain(y) - eventGain(x));
    let struck = 0;
    for (const ev of knocks) {
      if (struck >= COLLISION_BUDGET) break;
      this.dispatch(ev);
      struck++;
    }
    if (this.lastAgent.size > 512) this.lastAgent.clear();
    if (this.lastWire.size > 512) this.lastWire.clear();
  }

  /**
   * Graph shape: who is wired to whom, and which stems are open. Delay and
   * damping are not here — a stretching rope must not rebuild the net.
   */
  private static topologyKey(topo: NetTopology): number {
    let h = 2166136261;
    for (const w of topo.wires) h = mix(mix(mix(h, w.id), w.agentA), w.agentB);
    for (const a of topo.agents) {
      h = mix(mix(h, a.id), a.openPorts);
      h = mix(h, a.compN ?? 1);
      h = mix(h, a.tissue ? 1 : 0);
      h = mix(h, Math.round((a.tissueY ?? 0) * 20));
      if (a.stubs) for (const st of a.stubs) h = mix(h, st.slot);
    }
    if (topo.tissues) {
      for (const t of topo.tissues) h = mix(mix(h, t.id), t.n);
    }
    return h;
  }

  /** Rope delay and damping, quantized to a sample / a twentieth. */
  private static tuneKey(topo: NetTopology): number {
    let h = 2166136261;
    for (const w of topo.wires) {
      h = mix(mix(mix(h, w.id), Math.round(w.length)), Math.round((w.damp ?? 0) * 20));
      h = mix(h, Math.round((w.loss ?? 1) * 2000));
    }
    return h;
  }

  /** Listener pose: height, pan, distance. Cheap to send, changes as you look. */
  private static poseKey(topo: NetTopology): number {
    let h = mix(2166136261, Math.round((topo.height ?? 0) * 20));
    // lod is in the key on its own account: dist is quantised to 1/20 here,
    // so a tier change near a threshold can leave the rounded pose identical
    // and the worklet would never be told the object changed representation.
    for (const w of topo.wires) {
      h = mix(mix(mix(h, w.id), Math.round((w.pan ?? 0) * 10)), Math.round((w.dist ?? 0) * 20));
      h = mix(h, w.lod ?? 0);
    }
    for (const a of topo.agents) {
      h = mix(mix(mix(h, a.id), Math.round((a.pan ?? 0) * 10)), Math.round((a.dist ?? 0) * 20));
      h = mix(h, a.lod ?? 0);
    }
    return h;
  }

  private forceSkin(): Set<number> {
    const ids = new Set<number>();
    for (const ev of this.events) {
      if (ev.type === 'spawn') {
        ids.add(ev.agent);
        continue;
      }
      if ('agentA' in ev) ids.add(ev.agentA);
      if ('agentB' in ev) ids.add(ev.agentB);
      if (ev.type === 'rewrite') for (const id of ev.leftovers) ids.add(id);
    }
    if (this.contacts) {
      for (const c of this.contacts.values()) {
        ids.add(c.agentA);
        ids.add(c.agentB);
      }
    }
    return ids;
  }

  /**
   * Touching pairs for this frame. Set by the sim before frame(); contact is a
   * state, not an event, so it is resent every frame and stops the moment the
   * sim stops reporting it. `vT` is signed sliding speed in px/s.
   */
  contacts: Map<string, LiveContact> | null = null;
  /** Scraping wire pairs for this frame. Same contract as `contacts`. */

  frame(graph: Graph, agents: Map<number, Agent>, dt = 1 / 60, view?: PanView | null): void {
    this.graph = graph;
    this.agents = agents;
    if (!this.armed) return;
    this.now += dt;

    if (this.ctx?.state === 'suspended') void this.ctx.resume();

    // Muted: nothing to hear, so skip the whole per-frame rebuild.
    if (this.muted) {
      this.events.length = 0;
      return;
    }

    const topo = buildTopology(graph, agents, view, this.lod, this.forceSkin());
    const slots = this.sharded ? SHARD_COUNT : 1;
    if (this.sharded) {
      this.shardOf = this.assigner.assign(graph.componentIds(agents));
      this.parts = splitTopology(topo, this.shardOf, SHARD_COUNT);
    } else {
      this.shardOf = new Map();
      this.parts = [topo];
    }
    for (let s = 0; s < slots; s++) this.syncShard(s, this.parts[s] ?? emptyTopology(topo.height));
    this.drainEvents();
    this.syncContacts(slots);
    this.syncAir(agents, slots);
  }

  private syncShard(slot: number, topo: NetTopology): void {
    const key = AudioEngine.topologyKey(topo);
    const pose = AudioEngine.poseKey(topo);
    const tune = AudioEngine.tuneKey(topo);
    if (key !== this.topoKeys[slot]) {
      this.topoKeys[slot] = key;
      this.poseKeys[slot] = pose;
      this.tuneKeys[slot] = tune;
      this.post({ type: 'topology', topo }, slot);
      return;
    }
    if (tune !== this.tuneKeys[slot]) {
      this.tuneKeys[slot] = tune;
      this.post(
        {
          type: 'tune',
          wires: topo.wires.map((w) => ({
            id: w.id,
            length: w.length,
            damp: w.damp,
            loss: w.loss,
            bend: w.bend,
          })),
        },
        slot,
      );
    }
    if (pose !== this.poseKeys[slot]) {
      this.poseKeys[slot] = pose;
      this.post(
        {
          type: 'listen',
          height: topo.height ?? 0,
          wires: topo.wires.map((w) => ({
            id: w.id,
            pan: w.pan ?? 0,
            dist: w.dist ?? 0,
            lod: w.lod ?? 0,
          })),
          agents: topo.agents.map((a) => ({
            id: a.id,
            pan: a.pan ?? 0,
            dist: a.dist ?? 0,
            lod: a.lod ?? 0,
          })),
        },
        slot,
      );
    }
  }

  private syncContacts(slots: number): void {
    if (this.contacts && (this.contacts.size > 0 || this.contactedLast.some(Boolean))) {
      const msg = planContactMessage(this.contacts);
      const parts = this.sharded
        ? partitionPairs(msg.items, this.shardOf, SHARD_COUNT)
        : [msg.items];
      for (let s = 0; s < slots; s++) {
        const items = parts[s] ?? [];
        const cKey = contactKeyOf(items);
        if (cKey !== this.contactKeys[s]) {
          this.contactKeys[s] = cKey;
          this.post({ type: 'contact', items }, s);
        }
        this.contactedLast[s] = items.length > 0;
      }
    }

  }

  private syncAir(agents: Map<number, Agent>, slots: number): void {
    const air = planAirMessage(agents);
    const parts = this.sharded ? partitionPairs(air.items, this.shardOf, SHARD_COUNT) : [air.items];
    for (let s = 0; s < slots; s++) {
      const items = parts[s] ?? [];
      const key = airKeyOf(items);
      if (key === this.airKeys[s]) continue;
      this.airKeys[s] = key;
      if (items.length > 0 || this.airedLast[s]) {
        this.post({ type: 'air', items }, s);
        this.airedLast[s] = items.length > 0;
      }
    }
  }

  /** @internal The dry/wet split, so a test can assert it leaves headroom. */
  busGains(): { dry: number; wet: number } {
    return { dry: 0.55, wet: 0.45 };
  }

  invalidateTopology(): void {
    this.topoKeys.fill(-1);
    this.poseKeys.fill(-1);
    this.tuneKeys.fill(-1);
    this.contactKeys.fill(-1);
    this.airKeys.fill(-1);
    this.airedLast.fill(false);
    this.contactedLast.fill(false);
    this.events.length = 0;
    this.lastAgent.clear();
    this.lastWire.clear();
    this.wavePacked = null;
    this.waveIndex.clear();
    this.waveBySlot.fill(null);
  }

  /** @internal Arm for unit tests without Web Audio. */
  armWithoutAudio(): void {
    this.armed = true;
    this.node = { port: { postMessage: () => {} } } as unknown as AudioWorkletNode;
  }

  private dispatch(ev: AudioEvent): void {
    const graph = this.graph!;
    const agents = this.agents!;
    if (ev.type === 'latch') {
      const slot = this.slotOf(ev.agentA);
      const topo = this.parts[slot];
      for (const msg of planLatchMessages(ev, graph, agents, topo)) this.post(msg, slot);
      return;
    }
    if (ev.type === 'spawn') {
      const slot = this.slotOf(ev.agent);
      for (const msg of planSpawnMessages(ev)) this.post(msg, slot);
      return;
    }
    if (ev.type === 'rewrite') {
      const slot = this.slotOf(ev.agentA);
      for (const msg of planRewriteMessages(ev)) this.post(msg, slot);
      return;
    }
    for (const msg of planCollisionMessages(ev)) {
      this.post(msg, this.slotOf(msg.agentId));
    }
  }
}

function airKeyOf(items: { agentA: number; agentB: number; length: number; gain: number }[]): number {
  let h = 2166136261;
  for (const it of items) {
    h = mix(mix(mix(mix(h, it.agentA), it.agentB), it.length | 0), (it.gain * 200) | 0);
  }
  return h;
}

function contactKeyOf(items: { agentA: number; agentB: number; load: number; slide: number }[]): number {
  let h = 2166136261;
  for (const it of items) {
    h = mix(mix(mix(mix(h, it.agentA), it.agentB), (it.load * 20) | 0), (it.slide * 40) | 0);
  }
  return h;
}


/**
 * FNV-1a step. These four keys exist only to answer "did anything change", and
 * they used to answer it by building a string per object and joining — hundreds
 * of allocations a frame for a boolean. A rolling hash gives the same answer
 * with no garbage at all.
 */
function mix(h: number, v: number): number {
  return Math.imul(h ^ (v | 0), 16777619) >>> 0;
}

export const audio = new AudioEngine();
