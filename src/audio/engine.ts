import type { Agent } from '../agents.ts';
import type { Graph } from '../graph.ts';
import {
  planCollisionMessages,
  planLatchMessages,
  planRewriteMessages,
  planSpawnMessages,
  planContactMessage,
  planWireContactMessage,
  planAirMessage,
} from './dispatch.ts';
import { buildTopology } from './topology.ts';
import { LodSelector } from './lod.ts';
import { setSampleRate } from './presets.ts';
import { makeReverbIR } from './reverb.ts';
import type { AudioEvent, LiveContact, LiveWireContact, NetTopology, PanView, WaveSnapshot, WorkletInMessage } from './types.ts';
import workletUrl from './worklet/net-processor.ts?url';
import workerUrl from './worklet/net-worker.ts?url';
import { AudioRing, ringBytes } from './ring.ts';

/** Excitations allowed per frame. Past this a busy net is a rattle, not music. */
const EVENTS_PER_FRAME = 5;
/** Seconds an agent stays quiet after sounding, so contacts do not machine-gun. */
const AGENT_COOLDOWN = 0.09;
const WIRE_COOLDOWN = 0.05;

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
  if (ev.type === 'pluck') return Math.min(1.4, 0.4 + ev.gain);
  if (ev.type === 'rewrite') return ev.phase === 'commit' ? 1.4 : 1;
  // A spawn outranks a light bump but yields to a latch, so a busy frame keeps
  // the structural events and drops the incidental ones.
  if (ev.type === 'spawn') return 1.2;
  return Math.min(1, ev.impact / 40);
}

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  /** Set only when synthesis runs off the audio thread. */
  private worker: Worker | null = null;
  private ring: AudioRing | null = null;
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
  private topoKey = -1;
  private poseKey = -1;
  private tuneKey = -1;
  private contactKey = -1;
  private wireContactKey = -1;
  onPost: ((msg: WorkletInMessage) => void) | null = null;
  /** Latest traveling-wave snapshot from the worklet. Null until the first one. */
  private wavePacked: Float32Array | null = null;
  private waveIndex = new Map<number, number>();

  get waves(): WaveSnapshot | null {
    return this.wavePacked ? { packed: this.wavePacked, index: this.waveIndex } : null;
  }

  /** @internal Tests feed a snapshot without a worklet. */
  acceptWaves(packed: Float32Array): void {
    this.waveIndex.clear();
    if (packed.length < 2) {
      this.wavePacked = null;
      return;
    }
    const n = packed[0] | 0;
    if (n <= 0) {
      this.wavePacked = null;
      return;
    }
    this.wavePacked = packed;
    const bins = packed[1] | 0;
    const stride = 2 + bins * 2;
    let o = 2;
    for (let i = 0; i < n; i++) {
      if (o + stride > packed.length) break;
      this.waveIndex.set(packed[o] | 0, o);
      o += stride;
    }
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
    // Queue only. frame() picks the loudest few, so a burst of contacts in one
    // step cannot fire thirty excitations at once.
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
      this.startWorker(ctx, node);
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

  private post(msg: WorkletInMessage): void {
    this.onPost?.(msg);
    // Whoever owns the net gets the control messages. In ring mode the
    // worklet owns nothing but the copy out of shared memory.
    if (this.worker) {
      this.worker.postMessage(msg);
      return;
    }
    if (!this.node) return;
    this.node.port.postMessage(msg);
  }

  /**
   * How full the ring is, 0..1, or null when rendering in the worklet.
   *
   * This is the load signal worth watching: it sags before anything is
   * dropped, where an underrun count only rises afterwards.
   */
  get ringFill(): number | null {
    return this.ring ? this.ring.fill() : null;
  }

  /** Quanta the callback has had to pad with silence. Should stay at zero. */
  get underruns(): number | null {
    return this.ring ? this.ring.underruns() : null;
  }

  /**
   * Move synthesis into a Worker feeding a shared ring.
   *
   * Returns false when the page is not cross-origin-isolated, or the Worker
   * cannot be built — in which case the caller keeps the in-worklet path,
   * which still works and is what everything did before.
   *
   * The Worker is started from a bootstrap that sets `sampleRate` and only
   * then pulls in the bundle. waveguide.ts reads that global once, at load,
   * and bakes filter coefficients out of it; an ordinary module import would
   * evaluate first and tune the whole net for 48 kHz on 44.1 kHz hardware.
   */
  private startWorker(ctx: AudioContext, node: AudioWorkletNode): boolean {
    if (!sharedMemoryAvailable()) return false;
    let url = '';
    try {
      const target = new URL(workerUrl, location.href).href;
      const boot = `self.sampleRate=${ctx.sampleRate};importScripts(${JSON.stringify(target)});`;
      url = URL.createObjectURL(new Blob([boot], { type: 'text/javascript' }));
      const worker = new Worker(url);
      const sab = new SharedArrayBuffer(ringBytes(RING_FRAMES));
      const ring = new AudioRing(sab, RING_FRAMES);
      worker.onmessage = (ev: MessageEvent) => {
        const msg = ev.data;
        if (msg && msg.type === '__ready') {
          // Only now does the callback start reading. Handing it the ring at
          // once means it drains an empty buffer while the worker is still
          // rendering its first pass.
          node.port.postMessage({ type: '__ring', sab, capacity: RING_FRAMES });
          return;
        }
        if (msg && msg.type === 'waves' && msg.packed instanceof Float32Array) {
          this.acceptWaves(msg.packed);
        }
        if (msg && msg.type === 'error') console.error('audio worker', msg.message);
      };
      worker.onerror = (ev) => console.error('audio worker', ev.message);
      worker.postMessage({ type: '__ring', sab, capacity: RING_FRAMES, sampleRate: ctx.sampleRate });
      this.worker = worker;
      this.ring = ring;
      return true;
    } catch (err) {
      console.warn('audio worker unavailable, rendering in the worklet', err);
      this.worker = null;
      this.ring = null;
      return false;
    } finally {
      if (url) URL.revokeObjectURL(url);
    }
  }

  /** Cooldowns keep one agent or wire from retriggering every frame. */
  private allow(ev: AudioEvent): boolean {
    if (ev.type === 'spawn') return true;
    if (ev.type === 'latch' || ev.type === 'pluck') {
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

  private drainEvents(topo?: NetTopology): void {
    if (!this.armed || !this.node || !this.graph || !this.agents) return;
    if (this.events.length === 0) return;
    const pending = this.events.splice(0, this.events.length);
    // Loudest first, so thinning drops the events nobody would have missed.
    pending.sort((x, y) => eventGain(y) - eventGain(x));
    let fired = 0;
    for (const ev of pending) {
      if (fired >= EVENTS_PER_FRAME) break;
      if (!this.allow(ev)) continue;
      this.dispatch(ev, this.graph, this.agents, topo);
      fired++;
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
      if (a.stubs) for (const st of a.stubs) h = mix(h, st.slot);
    }
    return h;
  }

  /** Rope delay and damping, quantized to a sample / a twentieth. */
  private static tuneKey(topo: NetTopology): number {
    let h = 2166136261;
    for (const w of topo.wires) {
      h = mix(mix(mix(h, w.id), Math.round(w.length)), Math.round((w.damp ?? 0) * 20));
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

  /**
   * Touching pairs for this frame. Set by the sim before frame(); contact is a
   * state, not an event, so it is resent every frame and stops the moment the
   * sim stops reporting it. `vT` is signed sliding speed in px/s.
   */
  contacts: Map<string, LiveContact> | null = null;
  /** Scraping wire pairs for this frame. Same contract as `contacts`. */
  wireContacts: Map<string, LiveWireContact> | null = null;

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

    const topo = buildTopology(graph, agents, view, this.lod);
    const key = AudioEngine.topologyKey(topo);
    const pose = AudioEngine.poseKey(topo);
    const tune = AudioEngine.tuneKey(topo);
    if (key !== this.topoKey) {
      this.topoKey = key;
      this.poseKey = pose;
      this.tuneKey = tune;
      this.post({ type: 'topology', topo });
    } else {
      if (tune !== this.tuneKey) {
        this.tuneKey = tune;
        this.post({
          type: 'tune',
          wires: topo.wires.map((w) => ({
            id: w.id,
            length: w.length,
            damp: w.damp,
            loss: w.loss,
            bend: w.bend,
          })),
        });
      }
      if (pose !== this.poseKey) {
        this.poseKey = pose;
        this.post({
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
        });
      }
    }
    this.drainEvents(topo);

    // Contact is continuous, so it bypasses the event budget: it is one message
    // describing every touching pair, and an empty list is how they separate.
    if (this.contacts && (this.contacts.size > 0 || this.contactedLast)) {
      const msg = planContactMessage(this.contacts);
      const cKey = contactKeyOf(msg.items);
      if (cKey !== this.contactKey) {
        this.contactKey = cKey;
        this.post(msg);
      }
      this.contactedLast = this.contacts.size > 0;
    }

    if (this.wireContacts && (this.wireContacts.size > 0 || this.wiredLast)) {
      const msg = planWireContactMessage(this.wireContacts);
      const wKey = wireContactKeyOf(msg.items);
      if (wKey !== this.wireContactKey) {
        this.wireContactKey = wKey;
        this.post(msg);
      }
      this.wiredLast = this.wireContacts.size > 0;
    }

    const air = planAirMessage(agents);
    const airKey = airKeyOf(air.items);
    if (airKey !== this.airKey) {
      this.airKey = airKey;
      if (air.items.length > 0 || this.airedLast) {
        this.post(air);
        this.airedLast = air.items.length > 0;
      }
    }
  }

  private contactedLast = false;
  private wiredLast = false;
  private airedLast = false;
  private airKey = -1;

  /** @internal The dry/wet split, so a test can assert it leaves headroom. */
  busGains(): { dry: number; wet: number } {
    return { dry: 0.55, wet: 0.45 };
  }

  invalidateTopology(): void {
    this.topoKey = -1;
    this.poseKey = -1;
    this.tuneKey = -1;
    this.contactKey = -1;
    this.wireContactKey = -1;
    this.airKey = -1;
    this.airedLast = false;
    this.wiredLast = false;
    this.events.length = 0;
    this.lastAgent.clear();
    this.lastWire.clear();
    this.wavePacked = null;
    this.waveIndex.clear();
  }

  /** @internal Arm for unit tests without Web Audio. */
  armWithoutAudio(): void {
    this.armed = true;
    this.node = { port: { postMessage: () => {} } } as unknown as AudioWorkletNode;
  }

  private dispatch(
    ev: AudioEvent,
    graph: Graph,
    agents: Map<number, Agent>,
    topo?: NetTopology,
  ): void {
    if (ev.type === 'latch') {
      for (const msg of planLatchMessages(ev, graph, agents, topo)) this.post(msg);
      return;
    }
    if (ev.type === 'spawn') {
      for (const msg of planSpawnMessages(ev)) this.post(msg);
      return;
    }
    if (ev.type === 'rewrite') {
      for (const msg of planRewriteMessages(ev)) this.post(msg);
      return;
    }
    if (ev.type === 'pluck') {
      this.post({ type: 'pluck', wireId: ev.wireId, gain: ev.gain, samples: ev.samples });
      return;
    }
    for (const msg of planCollisionMessages(ev)) this.post(msg);
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

function wireContactKeyOf(
  items: { wireA: number; wireB: number; load: number; slide: number; atA: number; atB: number }[],
): number {
  let h = 2166136261;
  for (const it of items) {
    h = mix(
      mix(
        mix(mix(mix(mix(h, it.wireA), it.wireB), (it.load * 20) | 0), (it.slide * 40) | 0),
        (it.atA * 20) | 0,
      ),
      (it.atB * 20) | 0,
    );
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
