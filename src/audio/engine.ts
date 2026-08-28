import type { Agent } from '../agents.ts';
import type { Graph } from '../graph.ts';
import {
  planCollisionMessages,
  planLatchMessages,
  planRewriteMessages,
  planContactMessage,
  planAirMessage,
} from './dispatch.ts';
import { buildTopology } from './topology.ts';
import { setSampleRate } from './presets.ts';
import { makeReverbIR } from './reverb.ts';
import type { AudioEvent, LiveContact, NetTopology, PanView, WaveSnapshot, WorkletInMessage } from './types.ts';
import workletUrl from './worklet/net-processor.ts?url';

/** Excitations allowed per frame. Past this a busy net is a rattle, not music. */
const EVENTS_PER_FRAME = 5;
/** Seconds an agent stays quiet after sounding, so contacts do not machine-gun. */
const AGENT_COOLDOWN = 0.09;
const WIRE_COOLDOWN = 0.05;

function eventGain(ev: AudioEvent): number {
  if (ev.type === 'latch') return 1.5;
  if (ev.type === 'rewrite') return ev.phase === 'commit' ? 1.4 : 1;
  return Math.min(1, ev.impact / 40);
}

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
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
  private topoKey = '';
  onPost: ((msg: WorkletInMessage) => void) | null = null;
  /** Latest traveling-wave snapshot from the worklet. Null until the first one. */
  private wavePacked: Float32Array | null = null;
  private waveIndex = new Map<number, number>();

  get waves(): WaveSnapshot | null {
    return this.wavePacked ? { packed: this.wavePacked, index: this.waveIndex } : null;
  }

  /** @internal Tests feed a snapshot without a worklet. */
  acceptWaves(packed: Float32Array): void {
    this.wavePacked = packed;
    this.waveIndex.clear();
    if (packed.length < 2) return;
    const n = packed[0] | 0;
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
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });
      node.onprocessorerror = (ev) => {
        console.error('audio worklet error', ev);
      };
      node.port.onmessage = (ev: MessageEvent) => {
        const msg = ev.data;
        if (msg && msg.type === 'waves' && msg.packed instanceof Float32Array) {
          this.acceptWaves(msg.packed);
        }
      };

      // Tone shaping, then a dry/wet split into a convolution tail. The dry
      // waveguide stays short; the reverb carries the sustain, and a reverb
      // tail is guaranteed to decay to silence in a way a feedback loop is not.
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 110;
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 6500;
      lp.Q.value = 0.5;

      const dry = ctx.createGain();
      dry.gain.value = 0.68;
      const send = ctx.createGain();
      send.gain.value = 0.55;
      const wet = ctx.createGain();
      wet.gain.value = 0.75;

      const verb = ctx.createConvolver();
      verb.normalize = true;
      verb.buffer = makeReverbIR(ctx);

      const master = ctx.createGain();
      master.gain.value = this.muted ? 0 : 1;

      node.connect(hp);
      hp.connect(lp);
      lp.connect(dry);
      lp.connect(send);
      send.connect(verb);
      verb.connect(wet);
      dry.connect(master);
      wet.connect(master);
      master.connect(ctx.destination);

      this.ctx = ctx;
      this.node = node;
      this.master = master;
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
    if (!this.node) return;
    this.node.port.postMessage(msg);
  }

  /** Cooldowns keep one agent or wire from retriggering every frame. */
  private allow(ev: AudioEvent): boolean {
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
   * Everything the worklet would actually act on: which wires exist, delay
   * length to a sample, pan to a tenth, and each agent's open-port count.
   * Rope length drifts every frame but the delay only posts when it crosses
   * a sample, so this is not 60 topology messages a second.
   */
  private static topologyKey(topo: NetTopology): string {
    const parts: string[] = [];
    for (const w of topo.wires) {
      parts.push(
        `${w.id}:${w.agentA}>${w.agentB}:${Math.round(w.length)}:${Math.round((w.pan ?? 0) * 10)}:${Math.round((w.damp ?? 0) * 20)}`,
      );
    }
    for (const a of topo.agents) parts.push(`a${a.id}:${a.openPorts}`);
    return parts.join('|');
  }

  /**
   * Touching pairs for this frame. Set by the sim before frame(); contact is a
   * state, not an event, so it is resent every frame and stops the moment the
   * sim stops reporting it. `vT` is signed sliding speed in px/s.
   */
  contacts: Map<string, LiveContact> | null = null;

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

    const topo = buildTopology(graph, agents, view);
    const key = AudioEngine.topologyKey(topo);
    if (key !== this.topoKey) {
      this.topoKey = key;
      this.post({ type: 'topology', topo });
    }
    this.drainEvents(topo);

    // Contact is continuous, so it bypasses the event budget: it is one message
    // describing every touching pair, and an empty list is how they separate.
    if (this.contacts && (this.contacts.size > 0 || this.contactedLast)) {
      this.post(planContactMessage(this.contacts));
      this.contactedLast = this.contacts.size > 0;
    }

    const air = planAirMessage(agents);
    if (air.items.length > 0 || this.airedLast) {
      this.post(air);
      this.airedLast = air.items.length > 0;
    }
  }

  private contactedLast = false;
  private airedLast = false;

  invalidateTopology(): void {
    this.topoKey = '';
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
    if (ev.type === 'rewrite') {
      for (const msg of planRewriteMessages(ev)) this.post(msg);
      return;
    }
    for (const msg of planCollisionMessages(ev)) this.post(msg);
  }
}

export const audio = new AudioEngine();
