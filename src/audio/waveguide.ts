import type { AgentTopo, NetTopology } from './types.ts';

/**
 * The worklet is inlined as one import-free file, so it cannot read the shared
 * sample-rate helper. AudioWorkletGlobalScope defines `sampleRate`; in tests it
 * does not exist, so fall back to the rate the engine asks for.
 */
declare const sampleRate: number | undefined;
const SAMPLE_RATE = typeof sampleRate === 'number' ? sampleRate : 48000;

type AgentTopoLike = AgentTopo;

export const MAX_WIRES = 64;
export const MAX_DELAY = 4096;
export const MAX_AGENTS = 256;
export const MAX_PORTS = 4;
export const MAX_CONTACTS = 48;
export const MAX_AIR = 48;
export const MAX_AIR_DELAY = 512;
export const MAX_STUBS = 96;
export const MAX_STUB_DELAY = 64;
export const IMPULSE_TAPS = 12;

/** Spatial bins in a traveling-wave snapshot. t=0 is end A, t=1 is end B. */
export const WAVE_BINS = 32;
/** Packed record: id, env, fwd[WAVE_BINS], back[WAVE_BINS]. */
export const WAVE_STRIDE = 2 + WAVE_BINS * 2;

/** Delay length glides toward its target instead of stepping (~5 ms). */
const LENGTH_GLIDE = 0.004;
/** Below this pickup envelope a wire is zeroed so silence is really silent. */
const QUIET_FLOOR = 3e-6;
/** Junction incoming bled into body modes. Mix-only — never written back. */
const BODY_FROM_STRING = 0.1;
/** Active wires quieter than this can be stolen for a new latch. */
const STEAL_ENV = 0.002;
/** Friction / Hertzian force into the junction — a bow, not a scrape-pluck. */
const RUB_TO_JUNCTION = 0.22;
/** How hard a contact force drives the plate modes. They are quiet at contact-scale gain. */
const BOW_TO_MODE = 6;
/** Incoming waveguide velocity, scaled into the same units as modal velocity. */
const JUNCTION_VEL = 0.012;
/** Hertzian stiffness: vibrational displacement difference → restoring force. */
const CONTACT_K = 2.2;
/** Hertzian dashpot: keeps the contact spring from howling. */
const CONTACT_C = 0.28;
/**
 * How loudly surface asperities radiate, per unit load and slip speed.
 *
 * Kept deliberately low. Roughness exists to break the exact antisymmetry of
 * the contact force so a same-kind pair does not cancel to silence — nothing
 * more. Turned up, it swamps the friction's own limit cycle: a bowed wire stops
 * entraining to its round trip and locks onto the body mode instead, and the
 * whole thing reads as metallic scraping rather than bowing. At this level a
 * wired pair holds the string's period and stays tonal.
 */
const ROUGHNESS = 4.2;

/** Air pressure into plate modes. Weaker than a bow so the gap is a halo, not a second instrument. */
const AIR_TO_BODY = 2.2;
/** Air pressure into the junction. */
const AIR_TO_JUNCTION = 0.08;
/** Open-lip volume flow mixed into the body's air radiation. */
const STUB_TO_AIR = 0.1;
/** One-pole at the lip: 1 is a perfect inversion, lower leaks highs into air. */
const OPEN_END_DAMP = 0.62;
const OPEN_END_LOSS = 0.97;
const MU_STATIC = 0.82;
const MU_KINETIC = 0.34;
const V_STRIBECK = 0.01;
const V_STICK = 0.0012;

export interface WireState {
  active: boolean;
  /** Zeroed and skipped once the pickup envelope falls under QUIET_FLOOR. */
  quiet: boolean;
  wireId: number;
  length: number;
  lengthTarget: number;
  loss: number;
  bend: number;
  /** One-pole damping coefficient, 1 = bright, toward 0 = dark. */
  damp: number;
  /** First-order allpass coefficient. Nonzero stretches partials (bar/bell). */
  disp: number;
  /** -1 hard left .. +1 hard right. */
  pan: number;
  /** Listener distance, 0 = on top of you. */
  dist: number;
  dry: number;
  wet: number;
  /** One-pole air absorption on the pickup. */
  airDamp: number;
  airLp: number;
  exAt: number;
  exWidth: number;
  zA: number;
  zB: number;
  agentA: number;
  agentB: number;
  pos: number;
  burstPos: number;
  env: number;
  lpFwd: number;
  lpBack: number;
  dcXFwd: number;
  dcYFwd: number;
  dcXBack: number;
  dcYBack: number;
  apXFwd: number;
  apYFwd: number;
  apXBack: number;
  apYBack: number;
  bufFwd: Float32Array;
  bufBack: Float32Array;
  burstFwd: Float32Array;
  burstBack: Float32Array;
  exciteA: number;
  exciteB: number;
  inA: number;
  inB: number;
  outA: number;
  outB: number;
}

export const BODY_MODES = 3;

export interface AgentState {
  active: boolean;
  id: number;
  openPorts: number;
  portCount: number;
  loadY: number;
  excite: number;
  pan: number;
  dist: number;
  dry: number;
  wet: number;
  airDamp: number;
  airLp: number;
  coupling: number;
  wireIdx: Int16Array;
  wireEnd: Int8Array;
  admittance: Float32Array;
  stubCount: number;
  stubIdx: Int16Array;

  /** Hertzian contact in progress: a force pulse, not an instantaneous spike. */
  strikePeak: number;
  strikePos: number;
  strikeDur: number;
  strikeSharp: number;

  /** True while this body is in at least one contact pair. */
  touching: boolean;
  /** Scratch: junction admittance sum, incoming, this-sample strike and contact force. */
  sumY: number;
  sumYIn: number;
  strikeNow: number;
  contactF: number;
  /** Delayed air arriving this sample. */
  airIn: number;
  /** What this body radiated last sample — the air source. */
  radiate: number;

  /** Body modes — a knocked body rings whether or not a wire is tied to it. */
  modeA1: Float32Array;
  modeA2: Float32Array;
  modeGain: Float32Array;
  modeY1: Float32Array;
  modeY2: Float32Array;
  bodyEnv: number;
  /** Differentiated contact force — the transient that radiates directly. */
  prevForce: number;
  dForce: number;
  /** This body's own surface-roughness stream, independent of any partner's. */
  noise: number;
}

export type WorkletMessage =
  | { type: 'topology'; topo: NetTopology }
  | { type: 'impulse'; wireId: number; end: 0 | 1; gain: number }
  | { type: 'junction'; agentId: number; gain: number }
  | { type: 'strike'; agentId: number; peak: number; dur: number; sharp: number }
  | { type: 'contact'; items: { agentA: number; agentB: number; load: number; slide: number }[] }
  | { type: 'air'; items: { agentA: number; agentB: number; length: number; gain: number; damp: number }[] }
  | { type: 'latch'; topo: NetTopology; wireId: number; gain: number }
  | { type: 'rewrite'; phase: 0 | 1; wireId: number; agentA: number; agentB: number; leftovers: number[]; gain: number }
  | { type: 'gain'; master: number }
  | {
      type: 'listen';
      height?: number;
      wires: { id: number; pan?: number; dist?: number }[];
      agents: { id: number; pan?: number; dist?: number }[];
    }
  | {
      type: 'tune';
      wires: { id: number; length: number; damp?: number; loss?: number; bend?: number }[];
    };

function makeWire(): WireState {
  return {
    active: false,
    quiet: true,
    wireId: -1,
    length: 64,
    lengthTarget: 64,
    loss: 0.999,
    bend: 0.05,
    damp: 0.5,
    disp: 0,
    pan: 0,
    dist: 0,
    dry: 1,
    wet: 0.85,
    airDamp: 1,
    airLp: 0,
    exAt: 0.16,
    exWidth: 1,
    zA: 1,
    zB: 1,
    agentA: -1,
    agentB: -1,
    pos: 0,
    burstPos: 0,
    env: 0,
    lpFwd: 0,
    lpBack: 0,
    dcXFwd: 0,
    dcYFwd: 0,
    dcXBack: 0,
    dcYBack: 0,
    apXFwd: 0,
    apYFwd: 0,
    apXBack: 0,
    apYBack: 0,
    bufFwd: new Float32Array(MAX_DELAY),
    bufBack: new Float32Array(MAX_DELAY),
    burstFwd: new Float32Array(IMPULSE_TAPS),
    burstBack: new Float32Array(IMPULSE_TAPS),
    exciteA: 0,
    exciteB: 0,
    inA: 0,
    inB: 0,
    outA: 0,
    outB: 0,
  };
}

function makeAgent(): AgentState {
  return {
    active: false,
    id: -1,
    openPorts: 0,
    portCount: 0,
    loadY: 0,
    excite: 0,
    pan: 0,
    dist: 0,
    dry: 1,
    wet: 0.85,
    airDamp: 1,
    airLp: 0,
    coupling: 1,
    wireIdx: new Int16Array(MAX_PORTS),
    wireEnd: new Int8Array(MAX_PORTS),
    admittance: new Float32Array(MAX_PORTS),
    stubCount: 0,
    stubIdx: new Int16Array(MAX_PORTS),
    strikePeak: 0,
    strikePos: 0,
    strikeDur: 0,
    strikeSharp: 0.5,
    touching: false,
    sumY: 0,
    sumYIn: 0,
    strikeNow: 0,
    contactF: 0,
    airIn: 0,
    radiate: 0,
    modeA1: new Float32Array(BODY_MODES),
    modeA2: new Float32Array(BODY_MODES),
    modeGain: new Float32Array(BODY_MODES),
    modeY1: new Float32Array(BODY_MODES),
    modeY2: new Float32Array(BODY_MODES),
    bodyEnv: 0,
    prevForce: 0,
    dForce: 0,
    noise: 1,
  };
}

interface ContactState {
  active: boolean;
  idxA: number;
  idxB: number;
  idA: number;
  idB: number;
  load: number;
  slide: number;
  noise: number;
}

function makeContact(): ContactState {
  return {
    active: false,
    idxA: 0,
    idxB: 0,
    idA: -1,
    idB: -1,
    load: 0,
    slide: 0,
    noise: 1,
  };
}

interface AirState {
  active: boolean;
  keep: boolean;
  idxA: number;
  idxB: number;
  idA: number;
  idB: number;
  length: number;
  lengthTarget: number;
  gain: number;
  damp: number;
  pos: number;
  lpFwd: number;
  lpBack: number;
  bufFwd: Float32Array;
  bufBack: Float32Array;
}

function makeAir(): AirState {
  return {
    active: false,
    keep: false,
    idxA: 0,
    idxB: 0,
    idA: -1,
    idB: -1,
    length: 16,
    lengthTarget: 16,
    gain: 0,
    damp: 0.5,
    pos: 0,
    lpFwd: 0,
    lpBack: 0,
    bufFwd: new Float32Array(MAX_AIR_DELAY),
    bufBack: new Float32Array(MAX_AIR_DELAY),
  };
}

interface StubState {
  active: boolean;
  keep: boolean;
  agentId: number;
  agentIdx: number;
  slot: number;
  length: number;
  lengthTarget: number;
  z: number;
  y: number;
  pos: number;
  lp: number;
  inJ: number;
  outJ: number;
  bufFwd: Float32Array;
  bufBack: Float32Array;
}

function makeStub(): StubState {
  return {
    active: false,
    keep: false,
    agentId: -1,
    agentIdx: 0,
    slot: 0,
    length: 8,
    lengthTarget: 8,
    z: 1,
    y: 1,
    pos: 0,
    lp: 0,
    inJ: 0,
    outJ: 0,
    bufFwd: new Float32Array(MAX_STUB_DELAY),
    bufBack: new Float32Array(MAX_STUB_DELAY),
  };
}

function clampDelay(length: number): number {
  if (!Number.isFinite(length)) return 64;
  return Math.max(8, Math.min(MAX_DELAY - 1, length));
}

function clampAirDelay(length: number): number {
  if (!Number.isFinite(length)) return 16;
  return Math.max(4, Math.min(MAX_AIR_DELAY - 1, length));
}

function clampStubDelay(length: number): number {
  if (!Number.isFinite(length)) return 8;
  return Math.max(4, Math.min(MAX_STUB_DELAY - 1, length));
}

function wrapDelayIndex(pos: number, delay: number): number {
  if (!Number.isFinite(pos) || !Number.isFinite(delay)) return 0;
  let r = pos - delay;
  r %= MAX_DELAY;
  if (r < 0) r += MAX_DELAY;
  if (!Number.isFinite(r)) return 0;
  return r;
}

function wrapAirIndex(pos: number, delay: number): number {
  if (!Number.isFinite(pos) || !Number.isFinite(delay)) return 0;
  let r = pos - delay;
  r %= MAX_AIR_DELAY;
  if (r < 0) r += MAX_AIR_DELAY;
  if (!Number.isFinite(r)) return 0;
  return r;
}

function wrapStubIndex(pos: number, delay: number): number {
  if (!Number.isFinite(pos) || !Number.isFinite(delay)) return 0;
  let r = pos - delay;
  r %= MAX_STUB_DELAY;
  if (r < 0) r += MAX_STUB_DELAY;
  if (!Number.isFinite(r)) return 0;
  return r;
}

function sameAirPair(s: AirState, a: number, b: number): boolean {
  return (s.idA === a && s.idB === b) || (s.idA === b && s.idB === a);
}

/** Keep in step with presets.ts distanceDry / distanceWet / distanceRoom / distanceDamp. */
export function distDry(d: number): number {
  return 1 / (1 + 1.15 * Math.max(0, d));
}
export function distWet(r: number): number {
  return 0.85 / (1 + 0.12 * Math.max(0, r));
}
export function distRoom(h: number): number {
  return 1 / (1 + 0.9 * Math.max(0, h));
}
export function distDamp(d: number): number {
  return Math.max(0.06, 1 / (1 + 1.9 * Math.max(0, d) * Math.max(0, d)));
}

function listen(
  s: { dist: number; dry: number; wet: number; airDamp: number },
  d: number | undefined,
  height: number,
): void {
  const h = Number.isFinite(height) ? Math.max(0, height) : 0;
  const x = d !== undefined && Number.isFinite(d) ? Math.max(0, d) : h;
  s.dist = x;
  s.dry = distDry(x);
  const r = Math.sqrt(Math.max(0, x * x - h * h));
  s.wet = distWet(r) * distRoom(h);
  s.airDamp = distDamp(x);
}

function softClip(x: number): number {
  if (x > -1 && x < 1) return x;
  return Math.tanh(x);
}

/** Kill denormals and non-finite values so a NaN cannot poison the net. */
function flush(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return x > -1e-25 && x < 1e-25 ? 0 : x;
}

/**
 * Resistive load at a junction. The body always leaks a little so a cycle
 * cannot sit on the clip rail. Open stems are stubs with an inverting lip,
 * not extra shunt here.
 */
export function junctionLoadY(_openPorts = 0): number {
  return 0.012;
}

/**
 * Bidirectional waveguide net. Runtime-import-free so the AudioWorklet
 * plugin can inline this file.
 */
export class WaveguideNet {
  wires: WireState[] = [];
  wireById = new Map<number, number>();
  agents: AgentState[] = [];
  agentById = new Map<number, number>();
  contacts: ContactState[] = [];
  airs: AirState[] = [];
  stubs: StubState[] = [];
  master = 1;
  dryL = 0;
  dryR = 0;
  wetL = 0;
  wetR = 0;
  snap = 0;
  snapLp = 0;
  outL = 0;
  outR = 0;
  outDryL = 0;
  outDryR = 0;
  outWetL = 0;
  outWetR = 0;
  env = 0;
  dcDryX = 0;
  dcDryY = 0;
  dcDryX2 = 0;
  dcDryY2 = 0;
  dcWetX = 0;
  dcWetY = 0;
  dcWetX2 = 0;
  dcWetY2 = 0;
  /** Listener height for this topology; wet falls as this rises. */
  listenH = 0;

  /**
   * Dense indices of the objects that actually exist, so the per-sample loops
   * never walk the fixed tables.
   *
   * The tables are sized for the worst case (256 agents, 96 stubs, ...), and
   * scanning them cost ~154 us per 128-sample quantum — about 6% of the whole
   * audio budget — even with the net completely silent, because the cost is set
   * by table size rather than by load. These lists are rebuilt only when the
   * set of objects changes, never per sample. `quiet` still gates work inside
   * the loops; this only removes the walk over slots that hold nothing.
   */
  private liveWires = new Int32Array(MAX_WIRES);
  private liveWireN = 0;
  private liveAgents = new Int32Array(MAX_AGENTS);
  private liveAgentN = 0;
  private liveStubs = new Int32Array(MAX_STUBS);
  private liveStubN = 0;
  private liveAirs = new Int32Array(MAX_AIR);
  private liveAirN = 0;
  private liveContacts = new Int32Array(MAX_CONTACTS);
  private liveContactN = 0;

  /** Scratch reused by applyTopology so the audio thread never allocates. */
  private seen = new Set<number>();
  private keep = new Set<number>();
  private portHead = new Map<number, number>();
  private portWire = new Int16Array(MAX_WIRES * 2);
  private portEnd = new Int8Array(MAX_WIRES * 2);
  private portNext = new Int16Array(MAX_WIRES * 2);
  private savedExcite = new Map<number, number>();
  /** Preallocated traveling-wave snapshot. [n, bins, ...records of WAVE_STRIDE]. */
  private waveSnap = new Float32Array(2 + MAX_WIRES * WAVE_STRIDE);

  constructor() {
    for (let i = 0; i < MAX_WIRES; i++) this.wires.push(makeWire());
    for (let i = 0; i < MAX_AGENTS; i++) this.agents.push(makeAgent());
    for (let i = 0; i < MAX_CONTACTS; i++) this.contacts.push(makeContact());
    for (let i = 0; i < MAX_AIR; i++) this.airs.push(makeAir());
    for (let i = 0; i < MAX_STUBS; i++) this.stubs.push(makeStub());
  }

  handle(msg: WorkletMessage): void {
    try {
      if (msg.type === 'gain') {
        this.master = msg.master;
        return;
      }
      if (msg.type === 'topology') {
        this.applyTopology(msg.topo);
        return;
      }
      if (msg.type === 'latch') {
        this.applyTopology(msg.topo);
        // The pluck is the whole excitation: the voice's `width` already sets
        // how sharp the attack is, and an extra burst on top just adds a click.
        this.injectPluck(msg.wireId, msg.gain);
        return;
      }
      if (msg.type === 'impulse') {
        this.injectImpulse(msg.wireId, msg.end, msg.gain);
        return;
      }
      if (msg.type === 'junction') {
        this.addJunction(msg.agentId, msg.gain);
        return;
      }
      if (msg.type === 'strike') {
        this.addStrike(msg.agentId, msg.peak, msg.dur, msg.sharp);
        return;
      }
      if (msg.type === 'contact') {
        this.setContacts(msg.items);
        return;
      }
      if (msg.type === 'air') {
        this.setAir(msg.items);
        return;
      }
      if (msg.type === 'listen') {
        this.applyListen(msg);
        return;
      }
      if (msg.type === 'tune') {
        this.applyTune(msg.wires);
        return;
      }
      if (msg.type === 'rewrite') {
        this.handleRewrite(msg);
      }
    } catch {
      // Keep processor alive on bad input.
    }
  }

  /** Called whenever the set of live objects changes — never per sample. */
  private refreshLiveWires(): void {
    let n = 0;
    for (let i = 0; i < MAX_WIRES; i++) if (this.wires[i].active) this.liveWires[n++] = i;
    this.liveWireN = n;
  }

  private refreshLiveAgents(): void {
    let n = 0;
    for (let i = 0; i < MAX_AGENTS; i++) if (this.agents[i].active) this.liveAgents[n++] = i;
    this.liveAgentN = n;
  }

  private refreshLiveStubs(): void {
    let n = 0;
    for (let i = 0; i < MAX_STUBS; i++) if (this.stubs[i].active) this.liveStubs[n++] = i;
    this.liveStubN = n;
  }

  private refreshLiveAirs(): void {
    let n = 0;
    for (let i = 0; i < MAX_AIR; i++) if (this.airs[i].active) this.liveAirs[n++] = i;
    this.liveAirN = n;
  }

  private refreshLiveContacts(): void {
    let n = 0;
    for (let i = 0; i < MAX_CONTACTS; i++) {
      if (this.contacts[i].active) this.liveContacts[n++] = i;
    }
    this.liveContactN = n;
  }

  read(buf: Float32Array, pos: number, delay: number): number {
    const r = wrapDelayIndex(pos, delay);
    const i0 = r | 0;
    const frac = r - i0;
    const i1 = i0 + 1 === MAX_DELAY ? 0 : i0 + 1;
    return buf[i0] + frac * (buf[i1] - buf[i0]);
  }

  readAir(buf: Float32Array, pos: number, delay: number): number {
    const r = wrapAirIndex(pos, delay);
    const i0 = r | 0;
    const frac = r - i0;
    const i1 = i0 + 1 === MAX_AIR_DELAY ? 0 : i0 + 1;
    return buf[i0] + frac * (buf[i1] - buf[i0]);
  }

  readStub(buf: Float32Array, pos: number, delay: number): number {
    const r = wrapStubIndex(pos, delay);
    const i0 = r | 0;
    const frac = r - i0;
    const i1 = i0 + 1 === MAX_STUB_DELAY ? 0 : i0 + 1;
    return buf[i0] + frac * (buf[i1] - buf[i0]);
  }

  /** Raised-cosine burst at one end. Never a bare single-sample spike. */
  injectImpulse(wireId: number, end: 0 | 1, gain: number): boolean {
    const idx = this.wireById.get(wireId);
    if (idx === undefined) return false;
    const w = this.wires[idx];
    w.quiet = false;
    // Seed the envelope with what we are about to inject, otherwise the quiet
    // gate fires on tick 1 — the excitation has not reached the pickup yet.
    w.env = Math.max(w.env, Math.abs(gain));
    const burst = end === 1 ? w.burstBack : w.burstFwd;
    for (let k = 0; k < IMPULSE_TAPS; k++) {
      const phase = k / (IMPULSE_TAPS - 1);
      const shape = 0.5 - 0.5 * Math.cos(2 * Math.PI * phase);
      burst[(w.burstPos + k) % IMPULSE_TAPS] += gain * shape * 0.5;
    }
    return true;
  }

  /**
   * Excitation shape along the wire, x in [0,1].
   * width >= 1 gives the classic plucked triangle peaked at `pos`;
   * width < 1 gives a narrow raised-cosine strike (mallet).
   */
  private shapeAt(x: number, pos: number, width: number): number {
    if (width >= 1) {
      if (x < pos) return pos > 0 ? x / pos : 0;
      return pos < 1 ? (1 - x) / (1 - pos) : 0;
    }
    const half = Math.max(1e-4, width * 0.5);
    const d = (x - pos) / half;
    if (d <= -1 || d >= 1) return 0;
    return 0.5 + 0.5 * Math.cos(Math.PI * d);
  }

  /**
   * Set the string's initial displacement. Each delay line carries half of it,
   * and because the lines run in opposite directions, sample k of the forward
   * line and sample L-k of the backward line are the same point on the wire.
   * The shape is made zero-mean first: with near-unity reflections at both
   * terminations, any DC in the initial condition is the longest-lived thing
   * in the loop and would sit under everything as a slow pedestal.
   */
  injectPluck(wireId: number, gain: number, at?: number, width?: number): boolean {
    // `at` near an end drives the fundamental; `at` at 0.5 cancels it. See the
    // note on Voice in presets.ts — the terminations do not invert, so the mode
    // shapes are cosines rather than a fixed-fixed string's sines.
    const idx = this.wireById.get(wireId);
    if (idx === undefined) return false;
    const w = this.wires[idx];
    if (at === undefined) at = w.exAt;
    if (width === undefined) width = w.exWidth;
    const L = Math.max(8, w.length | 0);

    let mean = 0;
    for (let k = 0; k < L; k++) mean += this.shapeAt(k / L, at, width);
    mean /= L;

    const amp = gain * 0.5;
    for (let k = 1; k < L; k++) {
      const s = (this.shapeAt(k / L, at, width) - mean) * amp;
      w.bufFwd[wrapDelayIndex(w.pos, k) | 0] += s;
      w.bufBack[wrapDelayIndex(w.pos, L - k) | 0] += s;
    }
    w.quiet = false;
    w.env = Math.max(w.env, Math.abs(gain));
    return true;
  }

  addJunction(agentId: number, gain: number): void {
    const aIdx = this.agentById.get(agentId);
    if (aIdx === undefined) return;
    const a = this.agents[aIdx];
    a.excite += gain;
    for (let p = 0; p < a.portCount; p++) {
      const w = this.wires[a.wireIdx[p]];
      w.quiet = false;
      w.env = Math.max(w.env, Math.abs(gain));
    }
  }

  /**
   * Start a Hertzian contact. `dur` is the contact duration in samples, which
   * the sim derives from the effective mass and the closing speed — a harder
   * hit is a *shorter* contact and therefore a brighter one, which is the whole
   * reason a strike carries a duration instead of just a gain.
   */
  addStrike(agentId: number, peak: number, dur: number, sharp: number): void {
    const aIdx = this.agentById.get(agentId);
    if (aIdx === undefined) return;
    const a = this.agents[aIdx];
    // Overlapping contacts merge into the louder, shorter one.
    if (a.strikePos < a.strikeDur && a.strikePeak > peak) return;
    a.strikePeak = peak;
    a.strikeDur = Math.max(2, Math.min(2048, dur));
    a.strikePos = 0;
    a.strikeSharp = Math.max(0, Math.min(1, sharp));
    a.bodyEnv = Math.max(a.bodyEnv, peak);
    for (let p = 0; p < a.portCount; p++) {
      const w = this.wires[a.wireIdx[p]];
      w.quiet = false;
      w.env = Math.max(w.env, peak);
    }
  }

  /** Replace the set of touching pairs. Anything not listed has separated. */
  setContacts(items: { agentA: number; agentB: number; load: number; slide: number }[]): void {
    for (const a of this.agents) {
      if (a.active) a.touching = false;
    }
    let n = 0;
    for (const it of items) {
      if (n >= MAX_CONTACTS) break;
      const ia = this.agentById.get(it.agentA);
      const ib = this.agentById.get(it.agentB);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      const c = this.contacts[n];
      c.active = true;
      c.idxA = ia;
      c.idxB = ib;
      c.idA = it.agentA;
      c.idB = it.agentB;
      c.load = Math.max(0, Math.min(1, it.load));
      c.slide = Math.max(-0.08, Math.min(0.08, it.slide));
      n++;
      const A = this.agents[ia];
      const B = this.agents[ib];
      A.touching = true;
      B.touching = true;
      if (c.load > 0) {
        A.bodyEnv = Math.max(A.bodyEnv, c.load * 0.5);
        B.bodyEnv = Math.max(B.bodyEnv, c.load * 0.5);
        this.wakeWires(A, c.load);
        this.wakeWires(B, c.load);
      }
    }
    for (; n < MAX_CONTACTS; n++) {
      const c = this.contacts[n];
      c.active = false;
      c.idA = -1;
      c.idB = -1;
    }
    this.refreshLiveContacts();
  }

  /** Replace the set of line-of-sight air paths. Matching pairs keep their delay lines. */
  setAir(
    items: { agentA: number; agentB: number; length: number; gain: number; damp: number }[],
  ): void {
    for (const a of this.airs) a.keep = false;

    // Match existing pairs first so a new neighbour cannot steal a live buffer.
    for (const it of items) {
      const ia = this.agentById.get(it.agentA);
      const ib = this.agentById.get(it.agentB);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      for (let i = 0; i < MAX_AIR; i++) {
        const a = this.airs[i];
        if (!a.active || a.keep || !sameAirPair(a, it.agentA, it.agentB)) continue;
        this.bindAir(a, ia, ib, it, false);
        a.keep = true;
        break;
      }
    }

    for (const it of items) {
      const ia = this.agentById.get(it.agentA);
      const ib = this.agentById.get(it.agentB);
      if (ia === undefined || ib === undefined || ia === ib) continue;
      let taken = false;
      for (let i = 0; i < MAX_AIR; i++) {
        if (this.airs[i].keep && sameAirPair(this.airs[i], it.agentA, it.agentB)) {
          taken = true;
          break;
        }
      }
      if (taken) continue;
      let slot = -1;
      for (let i = 0; i < MAX_AIR; i++) {
        if (!this.airs[i].active) {
          slot = i;
          break;
        }
      }
      if (slot < 0) {
        for (let i = 0; i < MAX_AIR; i++) {
          if (!this.airs[i].keep) {
            slot = i;
            break;
          }
        }
      }
      if (slot < 0) continue;
      this.bindAir(this.airs[slot], ia, ib, it, true);
      this.airs[slot].keep = true;
    }

    for (const a of this.airs) {
      if (!a.keep) this.retireAir(a);
    }
    this.refreshLiveAirs();
  }

  private bindAir(
    a: AirState,
    ia: number,
    ib: number,
    it: { agentA: number; agentB: number; length: number; gain: number; damp: number },
    fresh: boolean,
  ): void {
    if (fresh) this.retireAir(a);
    a.active = true;
    a.idxA = ia;
    a.idxB = ib;
    a.idA = it.agentA;
    a.idB = it.agentB;
    a.lengthTarget = clampAirDelay(it.length);
    a.gain = Number.isFinite(it.gain) ? Math.max(0, Math.min(1, it.gain)) : 0;
    a.damp = Number.isFinite(it.damp) ? Math.max(0.02, Math.min(0.95, it.damp)) : 0.5;
    if (fresh) a.length = a.lengthTarget;
  }

  private retireAir(a: AirState): void {
    a.active = false;
    a.keep = false;
    a.idA = -1;
    a.idB = -1;
    a.gain = 0;
    a.lpFwd = 0;
    a.lpBack = 0;
    a.pos = 0;
    a.bufFwd.fill(0);
    a.bufBack.fill(0);
  }

  /**
   * Read delayed air into both bodies. Length glides like a wire so a walk
   * does not zipper the delay.
   */
  private applyAirReads(): void {
    for (let k = 0; k < this.liveAirN; k++) {
      const a = this.airs[this.liveAirs[k]];
      const ia = this.agentById.get(a.idA);
      const ib = this.agentById.get(a.idB);
      if (ia === undefined || ib === undefined) continue;
      a.idxA = ia;
      a.idxB = ib;
      const A = this.agents[ia];
      const B = this.agents[ib];
      if (!A.active || !B.active) continue;
      const d = a.lengthTarget - a.length;
      if (d !== 0) a.length += d * LENGTH_GLIDE;
      if (!Number.isFinite(a.length)) a.length = a.lengthTarget;
      const yFwd = flush(this.readAir(a.bufFwd, a.pos, a.length));
      const yBack = flush(this.readAir(a.bufBack, a.pos, a.length));
      A.airIn += yBack;
      B.airIn += yFwd;
    }
  }

  /** Body radiation this sample goes into the air lines for the other end. */
  private applyAirWrites(): void {
    for (let k = 0; k < this.liveAirN; k++) {
      const a = this.airs[this.liveAirs[k]];
      const A = this.agents[a.idxA];
      const B = this.agents[a.idxB];
      const xA = A && A.active && A.id === a.idA ? A.radiate * a.gain : 0;
      const xB = B && B.active && B.id === a.idB ? B.radiate * a.gain : 0;
      a.lpFwd += a.damp * (xA - a.lpFwd);
      a.lpBack += a.damp * (xB - a.lpBack);
      a.bufFwd[a.pos] = flush(a.lpFwd);
      a.bufBack[a.pos] = flush(a.lpBack);
      a.pos++;
      if (a.pos >= MAX_AIR_DELAY) a.pos = 0;
    }
  }

  /**
   * One open pipe per free stem. Matching (agent, slot) keeps the buffer so a
   * heading change does not click; a latch retires that slot's stub.
   */
  private bindStubs(topo: NetTopology): void {
    for (const s of this.stubs) s.keep = false;

    for (const spec of topo.agents) {
      const ia = this.agentById.get(spec.id);
      if (ia === undefined) continue;
      const list = spec.stubs;
      if (!list) continue;
      for (let k = 0; k < list.length; k++) {
        const it = list[k];
        for (let i = 0; i < MAX_STUBS; i++) {
          const s = this.stubs[i];
          if (!s.active || s.keep || s.agentId !== spec.id || s.slot !== it.slot) continue;
          this.bindStub(s, ia, spec.id, it, false);
          s.keep = true;
          break;
        }
      }
    }

    for (const spec of topo.agents) {
      const ia = this.agentById.get(spec.id);
      if (ia === undefined) continue;
      const list = spec.stubs;
      if (!list) continue;
      for (let k = 0; k < list.length; k++) {
        const it = list[k];
        let taken = false;
        for (let i = 0; i < MAX_STUBS; i++) {
          if (this.stubs[i].keep && this.stubs[i].agentId === spec.id && this.stubs[i].slot === it.slot) {
            taken = true;
            break;
          }
        }
        if (taken) continue;
        let slot = -1;
        for (let i = 0; i < MAX_STUBS; i++) {
          if (!this.stubs[i].active) {
            slot = i;
            break;
          }
        }
        if (slot < 0) {
          for (let i = 0; i < MAX_STUBS; i++) {
            if (!this.stubs[i].keep) {
              slot = i;
              break;
            }
          }
        }
        if (slot < 0) continue;
        this.bindStub(this.stubs[slot], ia, spec.id, it, true);
        this.stubs[slot].keep = true;
      }
    }

    for (const s of this.stubs) {
      if (!s.keep) this.retireStub(s);
    }

    for (const a of this.agents) a.stubCount = 0;
    for (let i = 0; i < MAX_STUBS; i++) {
      const s = this.stubs[i];
      if (!s.active) continue;
      const a = this.agents[s.agentIdx];
      if (!a || !a.active || a.stubCount >= MAX_PORTS) continue;
      a.stubIdx[a.stubCount] = i;
      a.stubCount++;
    }
    this.refreshLiveStubs();
  }

  private bindStub(
    s: StubState,
    ia: number,
    agentId: number,
    it: { slot: number; length: number; z: number },
    fresh: boolean,
  ): void {
    if (fresh) this.retireStub(s);
    s.active = true;
    s.agentId = agentId;
    s.agentIdx = ia;
    s.slot = it.slot;
    s.lengthTarget = clampStubDelay(it.length);
    s.z = it.z > 0 && Number.isFinite(it.z) ? it.z : 1;
    s.y = 1 / Math.max(0.05, s.z);
    if (fresh) s.length = s.lengthTarget;
  }

  private retireStub(s: StubState): void {
    s.active = false;
    s.keep = false;
    s.agentId = -1;
    s.lp = 0;
    s.pos = 0;
    s.inJ = 0;
    s.outJ = 0;
    s.bufFwd.fill(0);
    s.bufBack.fill(0);
  }

  private applyStubReads(): void {
    for (let k = 0; k < this.liveStubN; k++) {
      const s = this.stubs[this.liveStubs[k]];
      const d = s.lengthTarget - s.length;
      if (d !== 0) s.length += d * LENGTH_GLIDE;
      if (!Number.isFinite(s.length)) s.length = s.lengthTarget;
      s.inJ = flush(this.readStub(s.bufBack, s.pos, s.length));
    }
  }

  /** Invert at the lip (pressure release) and write both directions. */
  private applyStubWrites(): void {
    for (let k = 0; k < this.liveStubN; k++) {
      const s = this.stubs[this.liveStubs[k]];
      const yJ = flush(s.outJ);
      s.bufFwd[s.pos] = yJ;
      const lipIn = flush(this.readStub(s.bufFwd, s.pos, s.length));
      const inv = -lipIn;
      s.lp += OPEN_END_DAMP * (inv - s.lp);
      s.lp = flush(s.lp);
      const lipOut = OPEN_END_LOSS * s.lp;
      s.bufBack[s.pos] = lipOut;
      const flow = lipIn - lipOut;
      const a = this.agents[s.agentIdx];
      if (a && a.active && a.id === s.agentId) {
        a.radiate += flow * STUB_TO_AIR;
      }
      s.pos++;
      if (s.pos >= MAX_STUB_DELAY) s.pos = 0;
    }
  }

  private wakeWires(a: AgentState, load: number): void {
    for (let p = 0; p < a.portCount; p++) {
      const w = this.wires[a.wireIdx[p]];
      if (!w) continue;
      w.quiet = false;
      w.env = Math.max(w.env, load * 0.3);
    }
  }

  /**
   * Hertzian strike force for this sample. Rubbing is a separate friction loop
   * on the resonator — not this pulse, and not dForce.
   */
  private strikeForce(a: AgentState): number {
    if (a.strikePos >= a.strikeDur) return 0;
    const u = a.strikePos / a.strikeDur;
    const s = Math.sin(Math.PI * u);
    a.strikePos++;
    return a.strikePeak * s * Math.sqrt(s);
  }

  /** Modal + junction velocity at the contact, in per-sample units. */
  private surfaceVel(a: AgentState, sumYIn: number): number {
    let v = 0;
    for (let m = 0; m < BODY_MODES; m++) v += a.modeY1[m] - a.modeY2[m];
    return v + sumYIn * JUNCTION_VEL;
  }

  /** Modal displacement at the contact. Junction waves are velocity-like; omit them. */
  private surfaceDisp(a: AgentState): number {
    let u = 0;
    for (let m = 0; m < BODY_MODES; m++) u += a.modeY1[m];
    return u;
  }

  /** Per-body surface roughness. Each body carries its own asperity stream. */
  private surfaceNoise(a: AgentState): number {
    a.noise = (Math.imul(a.noise, 1664525) + 1013904223) >>> 0;
    return (a.noise / 4294967296) * 2 - 1;
  }

  /**
   * Stribeck friction. Sign convention: F is the force on A, so F_B = −F.
   *
   * On why rubbing sounds like scraping rather than bowing, which is deliberate
   * and has been checked: a body offers up to six resonators at once, and none
   * of them agree. Con carries three plate modes at 145/248/380 Hz — ratios
   * 1 : 1.71 : 2.62, inharmonic by construction — plus one quarter-wave stub
   * per open port, at 563/669/669 Hz, which sound odd harmonics only. Friction
   * excites all of them together, and several mutually inharmonic resonators
   * driven at once is what a struck metal plate is. A bowed instrument has one
   * resonator and one harmonic series.
   *
   * Give the pair a wire and the friction has something to lock to; with a
   * single dominant resonator it entrains to the round trip and produces real
   * Helmholtz motion (see the entrainment test in audio.test.ts). So an
   * untethered pair scrapes and a wired one sings, which is both the honest
   * physics and a reason to want the net to wire itself up.
   * vSlip is A's material velocity minus B's, rigid slide included.
   * F_A = −μN tanh(vSlip) drags A toward stick (vSlip → 0).
   */
  private friction(vSlip: number, N: number, c: ContactState): number {
    if (N <= 1e-6) return 0;
    let v = vSlip;
    c.noise = (Math.imul(c.noise, 1664525) + 1013904223) >>> 0;
    v += ((c.noise / 4294967296) * 2 - 1) * N * 0.012;
    const ax = v < 0 ? -v : v;
    const mu = MU_KINETIC + (MU_STATIC - MU_KINETIC) * Math.exp(-ax / V_STRIBECK);
    let F = -mu * N * Math.tanh(v / V_STICK);
    const max = MU_STATIC * N;
    if (F > max) F = max;
    else if (F < -max) F = -max;
    return F;
  }

  /**
   * Shared contact force: Hertzian spring always, friction while sliding.
   * Equal and opposite, from the previous sample's surface state.
   */
  private applyContacts(): void {
    for (let k = 0; k < this.liveContactN; k++) {
      const c = this.contacts[this.liveContacts[k]];
      if (c.load <= 1e-6) continue;
      const A = this.agents[c.idxA];
      const B = this.agents[c.idxB];
      if (!A || !B || !A.active || !B.active) continue;
      const vA = this.surfaceVel(A, A.sumYIn);
      const vB = this.surfaceVel(B, B.sumYIn);
      const du = this.surfaceDisp(A) - this.surfaceDisp(B);
      const dv = vA - vB;
      const Fn = -CONTACT_K * c.load * du - CONTACT_C * c.load * dv;
      const slide = c.slide;
      const sliding = slide > 1e-6 || slide < -1e-6;
      const Ft = sliding ? this.friction(slide + dv, c.load, c) : 0;
      const F = Fn + Ft;
      A.contactF += F;
      B.contactF -= F;

      // Newton's third law makes the force above exactly antisymmetric, which
      // is right for the dynamics and wrong for what radiates: two identical
      // bodies rubbing would cancel to literal silence at the pickup. What you
      // actually hear from a slide is each surface's own asperities exciting
      // its own body — correlated between the two, never identical. So the
      // roughness is drawn per body and added with the *same* sign, leaving the
      // coupled dynamics untouched.
      if (sliding) {
        const rough = c.load * (slide < 0 ? -slide : slide) * ROUGHNESS;
        if (rough > 0) {
          A.contactF += this.surfaceNoise(A) * rough;
          B.contactF += this.surfaceNoise(B) * rough;
        }
      }
    }
  }

  /**
   * One sample of the agent's own body ringing.
   *
   * Strike goes through the differentiator (contact duration → brightness) and
   * the modes. Contact force (Hertzian spring + friction) drives the modes
   * only — never dForce — so a pair can couple and a slide can lock instead of
   * becoming a click train. String bleed is mix-only and never written back
   * into the junction. Air is a delayed pressure into the same modes.
   */
  private bodyVoice(a: AgentState, contact: number, stringIn: number, force = 0, air = 0): number {
    const bleed = stringIn * BODY_FROM_STRING;
    const drive = contact + bleed + force * BOW_TO_MODE + air * AIR_TO_BODY;
    if (drive === 0 && a.bodyEnv < QUIET_FLOOR && a.dForce === 0) return 0;
    const d = contact - a.prevForce;
    a.prevForce = contact;
    a.dForce = flush(a.dForce * 0.72 + d);
    let sum = a.dForce * 0.5 * (0.6 + a.strikeSharp * 0.8) + bleed;
    for (let m = 0; m < BODY_MODES; m++) {
      const y =
        a.modeA1[m] * a.modeY1[m] - a.modeA2[m] * a.modeY2[m] + a.modeGain[m] * drive;
      a.modeY2[m] = a.modeY1[m];
      a.modeY1[m] = flush(y);
      sum += y;
    }
    const abs = sum < 0 ? -sum : sum;
    a.bodyEnv += (abs > a.bodyEnv ? 0.01 : 0.0002) * (abs - a.bodyEnv);
    // While a contact is down or air is arriving, do not wipe the modes.
    const airLive = air > QUIET_FLOOR || air < -QUIET_FLOOR;
    if (!Number.isFinite(a.bodyEnv) || (a.bodyEnv < QUIET_FLOOR && !a.touching && !airLive)) {
      a.bodyEnv = 0;
      a.dForce = 0;
      a.prevForce = 0;
      for (let m = 0; m < BODY_MODES; m++) {
        a.modeY1[m] = 0;
        a.modeY2[m] = 0;
      }
      return 0;
    }
    return sum;
  }

  handleRewrite(msg: Extract<WorkletMessage, { type: 'rewrite' }>): void {
    if (msg.phase === 0) {
      // Begin: a soft, wide excitation low on the wire — more breath than click.
      this.injectPluck(msg.wireId, msg.gain * 0.7, 0.2, 0.8);
      this.addJunction(msg.agentA, msg.gain * 0.3);
      this.addJunction(msg.agentB, msg.gain * 0.3);
      return;
    }
    this.snap += msg.gain * 0.5;
    this.addJunction(msg.agentA, msg.gain * 0.25);
    this.addJunction(msg.agentB, msg.gain * 0.25);
    for (let i = 0; i < msg.leftovers.length; i++) {
      this.addJunction(msg.leftovers[i], msg.gain * 0.5);
    }
  }

  private retireWire(w: WireState): void {
    w.active = false;
    w.quiet = true;
    w.exciteA = 0;
    w.exciteB = 0;
    w.env = 0;
    w.bufFwd.fill(0);
    w.bufBack.fill(0);
    w.burstFwd.fill(0);
    w.burstBack.fill(0);
    if (w.wireId >= 0) this.wireById.delete(w.wireId);
    w.wireId = -1;
  }

  /** Fold leftover ringing into the two agents, then free the slot. */
  private dumpAndRetire(w: WireState): void {
    // Do not scan the delay line here: wireEnergy walks every sample, and a
    // rewrite that retires several wires would blow the audio callback.
    if (w.env > 0) {
      const g = Math.min(0.85, w.env * 0.25);
      if (w.agentA >= 0) {
        this.savedExcite.set(w.agentA, (this.savedExcite.get(w.agentA) ?? 0) + g);
      }
      if (w.agentB >= 0 && w.agentB !== w.agentA) {
        this.savedExcite.set(w.agentB, (this.savedExcite.get(w.agentB) ?? 0) + g);
      }
    }
    this.retireWire(w);
  }

  private isStealable(w: WireState): boolean {
    return w.quiet || w.env < STEAL_ENV;
  }

  /**
   * Which of this update's wires get a slot. Loud existing wires keep theirs;
   * new latches take quiet slots; leftover quiet wires fill whatever remains.
   */
  private chooseKeep(topo: NetTopology): void {
    this.keep.clear();
    for (const w of this.wires) {
      if (!w.active || this.isStealable(w)) continue;
      if (this.keep.size >= MAX_WIRES) break;
      this.keep.add(w.wireId);
    }
    for (const spec of topo.wires) {
      if (this.keep.size >= MAX_WIRES) break;
      const idx = this.wireById.get(spec.id);
      if (idx !== undefined && this.wires[idx].active) continue;
      this.keep.add(spec.id);
    }
    for (const spec of topo.wires) {
      if (this.keep.size >= MAX_WIRES) break;
      this.keep.add(spec.id);
    }
  }

  applyTopology(topo: NetTopology): void {
    this.savedExcite.clear();
    for (const [id, idx] of this.agentById) {
      this.savedExcite.set(id, this.agents[idx].excite);
    }

    this.seen.clear();
    for (let i = 0; i < topo.wires.length; i++) this.seen.add(topo.wires[i].id);
    for (const w of this.wires) {
      if (w.active && !this.seen.has(w.wireId)) this.dumpAndRetire(w);
    }

    this.chooseKeep(topo);
    for (const w of this.wires) {
      if (w.active && !this.keep.has(w.wireId)) this.dumpAndRetire(w);
    }

    for (const a of this.agents) {
      a.active = false;
      a.portCount = 0;
      a.stubCount = 0;
      a.id = -1;
    }
    this.agentById.clear();
    this.listenH =
      topo.height !== undefined && Number.isFinite(topo.height) ? Math.max(0, topo.height) : 0;

    let placed = 0;
    for (const spec of topo.wires) {
      if (placed >= MAX_WIRES) break;
      if (!this.keep.has(spec.id)) continue;
      let slot = this.wireById.get(spec.id);
      if (slot === undefined) {
        slot = this.wires.findIndex((w) => !w.active);
        if (slot < 0) continue;
      }
      placed++;

      const w = this.wires[slot];
      const fresh = !w.active || w.wireId !== spec.id;
      w.active = true;
      w.wireId = spec.id;
      w.lengthTarget = clampDelay(spec.length);
      w.loss = Number.isFinite(spec.loss) ? spec.loss : 0.99;
      w.bend = Number.isFinite(spec.bend) ? spec.bend : 0.05;
      w.damp = spec.damp !== undefined && Number.isFinite(spec.damp) ? spec.damp : Math.max(0.05, 1 - spec.bend * 2.4);
      w.disp = spec.disp !== undefined && Number.isFinite(spec.disp) ? spec.disp : 0;
      w.pan = spec.pan !== undefined && Number.isFinite(spec.pan) ? spec.pan : 0;
      listen(w, spec.dist, this.listenH);
      w.exAt = spec.exAt !== undefined ? spec.exAt : 0.16;
      w.exWidth = spec.exWidth !== undefined ? spec.exWidth : 1;
      w.zA = spec.zA && spec.zA > 0 ? spec.zA : 1;
      w.zB = spec.zB && spec.zB > 0 ? spec.zB : 1;
      w.agentA = spec.agentA;
      w.agentB = spec.agentB;
      if (fresh) {
        w.length = w.lengthTarget;
        w.quiet = true;
        w.pos = 0;
        w.burstPos = 0;
        w.env = 0;
        w.lpFwd = 0;
        w.lpBack = 0;
        w.dcXFwd = 0;
        w.dcYFwd = 0;
        w.dcXBack = 0;
        w.dcYBack = 0;
        w.apXFwd = 0;
        w.apYFwd = 0;
        w.apXBack = 0;
        w.apYBack = 0;
        w.exciteA = 0;
        w.exciteB = 0;
        w.inA = 0;
        w.inB = 0;
        w.outA = 0;
        w.outB = 0;
        w.airLp = 0;
        w.bufFwd.fill(0);
        w.bufBack.fill(0);
        w.burstFwd.fill(0);
        w.burstBack.fill(0);
      }
      this.wireById.set(spec.id, slot);
    }

    // Intrusive linked lists over preallocated arrays: no per-agent garbage.
    this.portHead.clear();
    let n = 0;
    for (const w of this.wires) {
      if (!w.active) continue;
      const idx = this.wireById.get(w.wireId)!;
      for (let e = 0 as 0 | 1; e <= 1; e = (e + 1) as 0 | 1) {
        const agentId = e === 0 ? w.agentA : w.agentB;
        this.portWire[n] = idx;
        this.portEnd[n] = e;
        const head = this.portHead.get(agentId);
        this.portNext[n] = head === undefined ? -1 : head;
        this.portHead.set(agentId, n);
        n++;
      }
    }

    let ai = 0;
    for (const spec of topo.agents) {
      if (ai >= MAX_AGENTS) break;
      // Every agent gets a slot, wired or not. Skipping the unwired ones meant
      // a knock on a loose body was silent, which made collisions audible only
      // by accident of the net's shape.
      let node = this.portHead.get(spec.id);
      if (node === undefined) node = -1;
      const a = this.agents[ai];
      const same = a.id === spec.id;
      a.active = true;
      a.id = spec.id;
      a.openPorts = spec.openPorts;
      a.pan = spec.pan !== undefined ? spec.pan : 0;
      listen(a, spec.dist, this.listenH);
      if (!same) a.airLp = 0;
      a.coupling = spec.coupling !== undefined ? spec.coupling : 1;
      this.setBodyModes(a, spec, same);
      // Scale the resistive load by the agent's own admittance, so a heavy Con
      // and a light Dup do not leak identically. Without this, `impedance` was
      // computed by the topology builder every frame and then dropped.
      a.loadY = junctionLoadY() / Math.max(0.05, spec.impedance || 1);
      a.excite = this.savedExcite.get(spec.id) ?? 0;
      let p = 0;
      while (node >= 0 && p < MAX_PORTS) {
        const wi = this.portWire[node];
        const end = this.portEnd[node] as 0 | 1;
        const w = this.wires[wi];
        a.wireIdx[p] = wi;
        a.wireEnd[p] = end;
        a.admittance[p] = 1 / Math.max(0.05, end === 0 ? w.zA : w.zB);
        p++;
        node = this.portNext[node];
      }
      a.portCount = p;
      this.agentById.set(spec.id, ai);
      ai++;
    }
    this.bindStubs(topo);
    this.refreshLiveWires();
    this.refreshLiveAgents();
  }

  /**
   * Pan, distance, and listener height. Camera motion used to send a full
   * topology for this, which rebuilt every delay line on the audio thread and
   * blew the callback budget — that is the dropout.
   */
  applyListen(msg: {
    height?: number;
    wires: { id: number; pan?: number; dist?: number }[];
    agents: { id: number; pan?: number; dist?: number }[];
  }): void {
    this.listenH =
      msg.height !== undefined && Number.isFinite(msg.height) ? Math.max(0, msg.height) : this.listenH;
    for (let i = 0; i < msg.wires.length; i++) {
      const spec = msg.wires[i];
      const idx = this.wireById.get(spec.id);
      if (idx === undefined) continue;
      const w = this.wires[idx];
      if (!w.active) continue;
      if (spec.pan !== undefined && Number.isFinite(spec.pan)) w.pan = spec.pan;
      listen(w, spec.dist, this.listenH);
    }
    for (let i = 0; i < msg.agents.length; i++) {
      const spec = msg.agents[i];
      const idx = this.agentById.get(spec.id);
      if (idx === undefined) continue;
      const a = this.agents[idx];
      if (!a.active) continue;
      if (spec.pan !== undefined && Number.isFinite(spec.pan)) a.pan = spec.pan;
      listen(a, spec.dist, this.listenH);
    }
  }

  /** Delay and damping only. Must not touch buffers or port lists. */
  applyTune(
    wires: { id: number; length: number; damp?: number; loss?: number; bend?: number }[],
  ): void {
    for (let i = 0; i < wires.length; i++) {
      const spec = wires[i];
      const idx = this.wireById.get(spec.id);
      if (idx === undefined) continue;
      const w = this.wires[idx];
      if (!w.active) continue;
      w.lengthTarget = clampDelay(spec.length);
      if (spec.damp !== undefined && Number.isFinite(spec.damp)) w.damp = spec.damp;
      if (spec.loss !== undefined && Number.isFinite(spec.loss)) w.loss = spec.loss;
      if (spec.bend !== undefined && Number.isFinite(spec.bend)) w.bend = spec.bend;
    }
  }

  /**
   * Resonator coefficients for the body's modes. `keep` preserves the ringing
   * state when an agent keeps its slot across a topology update.
   */
  private setBodyModes(a: AgentState, spec: AgentTopoLike, keep: boolean): void {
    const hz = spec.modeHz;
    const t60 = spec.modeT60;
    const gain = spec.modeGain;
    if (!hz || !t60 || !gain) return;
    for (let m = 0; m < BODY_MODES; m++) {
      const f = hz[m] !== undefined ? hz[m] : 200;
      const d = t60[m] !== undefined ? t60[m] : 0.2;
      const w = (2 * Math.PI * f) / SAMPLE_RATE;
      // Pole radius for a given T60: r^(T60*fs) = 1e-3.
      const r = Math.exp(-6.9078 / Math.max(1, d * SAMPLE_RATE));
      a.modeA1[m] = 2 * r * Math.cos(w);
      a.modeA2[m] = r * r;
      // Normalize so peak response is independent of Q.
      a.modeGain[m] = (gain[m] !== undefined ? gain[m] : 0.5) * (1 - r) * Math.sin(w) * 4;
      if (!keep) {
        a.modeY1[m] = 0;
        a.modeY2[m] = 0;
      }
    }
    if (!keep) a.bodyEnv = 0;
  }

  /**
   * Listen to one source: air-absorption lowpass, then split into a dry bus
   * (falls with distance) and a wet send (almost flat). D/R is the distance
   * cue; the two gains are what produce it.
   */
  private place(
    x: number,
    pan: number,
    src: { dry: number; wet: number; airDamp: number; airLp: number },
  ): void {
    src.airLp += src.airDamp * (x - src.airLp);
    const y = flush(src.airLp);
    const p = pan > 1 ? 1 : pan < -1 ? -1 : pan;
    const gl = Math.sqrt(0.5 * (1 - p));
    const gr = Math.sqrt(0.5 * (1 + p));
    this.dryL += y * gl * src.dry;
    this.dryR += y * gr * src.dry;
    this.wetL += y * gl * src.wet;
    this.wetR += y * gr * src.wet;
  }

  /**
   * Everything that happens once per one-way trip, lumped at the delay output:
   * frequency-dependent damping, DC rejection, optional dispersion, loss.
   * Losses are LTI so they commute with the delay and belong at a single point
   * rather than being spread over every sample of the line.
   */
  private travel(w: WireState, x: number, dir: 0 | 1): number {
    let lp = dir === 0 ? w.lpFwd : w.lpBack;
    lp += w.damp * (x - lp);
    lp = flush(lp);

    let dcX = dir === 0 ? w.dcXFwd : w.dcXBack;
    let dcY = dir === 0 ? w.dcYFwd : w.dcYBack;
    const blocked = flush(lp - dcX + 0.999 * dcY);
    dcX = lp;
    dcY = blocked;

    let y = blocked;
    if (w.disp !== 0) {
      const apX = dir === 0 ? w.apXFwd : w.apXBack;
      const apY = dir === 0 ? w.apYFwd : w.apYBack;
      y = flush(w.disp * (y - apY) + apX);
      if (dir === 0) {
        w.apXFwd = blocked;
        w.apYFwd = y;
      } else {
        w.apXBack = blocked;
        w.apYBack = y;
      }
    }

    if (dir === 0) {
      w.lpFwd = lp;
      w.dcXFwd = dcX;
      w.dcYFwd = dcY;
    } else {
      w.lpBack = lp;
      w.dcXBack = dcX;
      w.dcYBack = dcY;
    }
    return w.loss * y;
  }

  /** Advance one sample. Fills outL/outR; returns the mono sum. */
  tick(): number {
    for (let k = 0; k < this.liveWireN; k++) {
      const w = this.wires[this.liveWires[k]];
      if (w.quiet) continue;
      const d = w.lengthTarget - w.length;
      if (d !== 0) w.length += d * LENGTH_GLIDE;
      if (!Number.isFinite(w.length)) w.length = w.lengthTarget;
      w.inB = this.travel(w, this.read(w.bufFwd, w.pos, w.length), 0);
      w.inA = this.travel(w, this.read(w.bufBack, w.pos, w.length), 1);
      w.outA = w.inA;
      w.outB = w.inB;
    }

    this.applyStubReads();

    this.dryL = 0;
    this.dryR = 0;
    this.wetL = 0;
    this.wetR = 0;
    for (let k = 0; k < this.liveAgentN; k++) {
      const agent = this.agents[this.liveAgents[k]];
      agent.strikeNow = this.strikeForce(agent);
      let sumY = agent.loadY;
      let sumYIn = 0;
      for (let p = 0; p < agent.portCount; p++) {
        const w = this.wires[agent.wireIdx[p]];
        if (!w) continue;
        const y = agent.admittance[p];
        sumY += y;
        sumYIn += y * (agent.wireEnd[p] === 0 ? w.inA : w.inB);
      }
      for (let s = 0; s < agent.stubCount; s++) {
        const st = this.stubs[agent.stubIdx[s]];
        if (!st) continue;
        sumY += st.y;
        sumYIn += st.y * st.inJ;
      }
      agent.sumY = sumY;
      agent.sumYIn = sumYIn;
      agent.contactF = 0;
      agent.airIn = 0;
      agent.radiate = 0;
    }

    this.applyAirReads();
    this.applyContacts();

    for (let k = 0; k < this.liveAgentN; k++) {
      const agent = this.agents[this.liveAgents[k]];
      const strike = agent.strikeNow;
      const force = agent.contactF;
      const air = agent.airIn;
      if (air > QUIET_FLOOR || air < -QUIET_FLOOR) {
        agent.bodyEnv = Math.max(agent.bodyEnv, air < 0 ? -air : air);
        this.wakeWires(agent, Math.min(1, air < 0 ? -air : air));
      }
      const body = this.bodyVoice(
        agent,
        strike,
        agent.sumYIn + (agent.portCount === 0 && agent.stubCount === 0 ? agent.excite : 0),
        force,
        air,
      );
      agent.radiate = body;
      if (body !== 0) this.place(body * 1.5, agent.pan, agent);

      const drive =
        agent.excite + strike * agent.coupling + force * RUB_TO_JUNCTION + air * AIR_TO_JUNCTION;
      const pJ = (2 * agent.sumYIn + drive) / Math.max(1e-6, agent.sumY);
      agent.excite = 0;
      for (let p = 0; p < agent.portCount; p++) {
        const w = this.wires[agent.wireIdx[p]];
        if (!w) continue;
        const incoming = agent.wireEnd[p] === 0 ? w.inA : w.inB;
        const outgoing = pJ - incoming;
        if (agent.wireEnd[p] === 0) w.outA = outgoing;
        else w.outB = outgoing;
        // A gated wire has to wake up when a neighbour scatters into it,
        // otherwise energy cannot cross a junction onto a silent wire.
        if (w.quiet) {
          const m = outgoing < 0 ? -outgoing : outgoing;
          if (m > QUIET_FLOOR) {
            w.quiet = false;
            w.env = Math.max(w.env, m);
          }
        }
      }
      for (let s = 0; s < agent.stubCount; s++) {
        const st = this.stubs[agent.stubIdx[s]];
        if (!st) continue;
        st.outJ = pJ - st.inJ;
      }
    }

    this.applyStubWrites();
    this.applyAirWrites();

    for (let k = 0; k < this.liveWireN; k++) {
      const w = this.wires[this.liveWires[k]];
      if (w.quiet) continue;
      const extraA = w.burstFwd[w.burstPos] + w.exciteA;
      const extraB = w.burstBack[w.burstPos] + w.exciteB;
      w.burstFwd[w.burstPos] = 0;
      w.burstBack[w.burstPos] = 0;
      w.burstPos = w.burstPos + 1 === IMPULSE_TAPS ? 0 : w.burstPos + 1;
      const yA = softClip(w.outA + extraA);
      const yB = softClip(w.outB + extraB);
      w.bufFwd[w.pos] = yA;
      w.bufBack[w.pos] = yB;
      w.exciteA = 0;
      w.exciteB = 0;
      w.pos = w.pos + 1 === MAX_DELAY ? 0 : w.pos + 1;

      // One pickup, at end A. The two terminations are L apart, so for mode n
      // they differ in phase by n*pi: summing both ends cancels every odd
      // harmonic and doubles every even one, which sounds an octave high and
      // hollow. Pan the single pickup instead of mixing in the far end.
      const p = yA + w.inA;
      const a = p < 0 ? -p : p;
      w.env += (a > w.env ? 0.01 : 0.0002) * (a - w.env);
      if (!Number.isFinite(w.env) || w.env < QUIET_FLOOR) {
        w.quiet = true;
        w.bufFwd.fill(0);
        w.bufBack.fill(0);
        w.inA = 0;
        w.inB = 0;
        w.outA = 0;
        w.outB = 0;
        continue;
      }
      this.place(p, w.pan, w);
    }

    // Rewrite-commit "snap": a short bandlimited puff, not a DC thump.
    if (this.snap !== 0 || this.snapLp !== 0) {
      const noise = (Math.random() * 2 - 1) * this.snap;
      this.snapLp = flush(this.snapLp + 0.35 * (noise - this.snapLp));
      this.dryL += this.snapLp;
      this.dryR += this.snapLp;
      this.snap = flush(this.snap * 0.9992);
    }

    let dL = this.dryL - this.dcDryX + 0.9995 * this.dcDryY;
    this.dcDryX = this.dryL;
    this.dcDryY = flush(dL);
    dL = this.dcDryY * 0.5;
    let dR = this.dryR - this.dcDryX2 + 0.9995 * this.dcDryY2;
    this.dcDryX2 = this.dryR;
    this.dcDryY2 = flush(dR);
    dR = this.dcDryY2 * 0.5;
    let wL = this.wetL - this.dcWetX + 0.9995 * this.dcWetY;
    this.dcWetX = this.wetL;
    this.dcWetY = flush(wL);
    wL = this.dcWetY * 0.5;
    let wR = this.wetR - this.dcWetX2 + 0.9995 * this.dcWetY2;
    this.dcWetX2 = this.wetR;
    this.dcWetY2 = flush(wR);
    wR = this.dcWetY2 * 0.5;

    const peak = Math.max(
      dL < 0 ? -dL : dL,
      dR < 0 ? -dR : dR,
      wL < 0 ? -wL : wL,
      wR < 0 ? -wR : wR,
    );
    this.env += (peak > this.env ? 0.00417 : 0.00007) * (peak - this.env);
    this.env = flush(this.env);
    // Ceiling only: a compressor with makeup was pulling quiet (far) mixes
    // back up to the same loudness as near ones.
    const ceil = 0.95;
    const g = this.env > ceil ? ceil / this.env : 1;
    const k = g * this.master;
    this.outDryL = flush(dL * k);
    this.outDryR = flush(dR * k);
    this.outWetL = flush(wL * k);
    this.outWetR = flush(wR * k);
    this.outL = flush(Math.tanh(this.outDryL + this.outWetL));
    this.outR = flush(Math.tanh(this.outDryR + this.outWetR));
    return (this.outL + this.outR) * 0.5;
  }

  /**
   * Downsample each live delay line onto WAVE_BINS along A→B.
   * Forward is the wave leaving A; backward is the wave leaving B.
   * Writes into a reused buffer so the audio thread does not allocate.
   *
   * Layout: packed[0] = nWires, packed[1] = bins;
   * then nWires records of (2 + bins*2): id, env, fwd[bins], back[bins].
   */
  fillWaveSnapshot(): Float32Array {
    let n = 0;
    let o = 2;
    for (let k = 0; k < this.liveWireN; k++) {
      const w = this.wires[this.liveWires[k]];
      if (!w.active || w.quiet) continue;
      const L = Math.max(1, w.length);
      const span = Math.max(1, L - 1);
      this.waveSnap[o] = w.wireId;
      this.waveSnap[o + 1] = w.env;
      const fwdBase = o + 2;
      const backBase = fwdBase + WAVE_BINS;
      for (let i = 0; i < WAVE_BINS; i++) {
        const t = i / (WAVE_BINS - 1);
        // Delay 1 is the newest sample at the write end; delay L is the far end.
        this.waveSnap[fwdBase + i] = this.read(w.bufFwd, w.pos, 1 + t * span);
        this.waveSnap[backBase + i] = this.read(w.bufBack, w.pos, 1 + (1 - t) * span);
      }
      o += WAVE_STRIDE;
      n++;
      if (n >= 8) break;
    }
    this.waveSnap[0] = n;
    this.waveSnap[1] = WAVE_BINS;
    return this.waveSnap;
  }

  /** Traveling-wave energy in the live delay window of one wire. */
  wireEnergy(wireId: number): number {
    const idx = this.wireById.get(wireId);
    if (idx === undefined) return 0;
    const w = this.wires[idx];
    let e = 0;
    const steps = Math.max(1, w.length | 0);
    for (let d = 0; d < steps; d++) {
      e += Math.abs(this.read(w.bufFwd, w.pos, d));
      e += Math.abs(this.read(w.bufBack, w.pos, d));
    }
    return e;
  }

  energy(): number {
    let e = 0;
    for (const w of this.wires) {
      if (w.active && !w.quiet) e += this.wireEnergy(w.wireId);
    }
    return e;
  }

  /** Active delay lines, including ones currently gated silent. */
  get liveCount(): number {
    return this.liveWireN;
  }

  activeWireCount(): number {
    let n = 0;
    for (const w of this.wires) if (w.active) n++;
    return n;
  }
}
