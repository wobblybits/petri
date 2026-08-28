import type { AgentKind, PortSlot } from '../agents.ts';

export type KindCode = 0 | 1 | 2;

export function kindCode(kind: AgentKind): KindCode {
  if (kind === 'era') return 0;
  if (kind === 'dup') return 1;
  return 2;
}

/** Camera-space stereo: pan from where the agent sits on screen. */
export interface PanView {
  x: number;
  y: number;
  zoom: number;
  viewW: number;
}

export interface WireTopo {
  id: number;
  /** One-way delay in samples. Pitch is sampleRate / (2 * length): long is low. */
  length: number;
  /** Loop-filter loss (0–1, higher = brighter). */
  loss: number;
  agentA: number;
  agentB: number;
  /** Characteristic impedance at the A end. */
  zA?: number;
  /** Characteristic impedance at the B end. */
  zB?: number;
  /** Inline loss from rope curvature / node count. */
  bend: number;
  /** One-pole damping coefficient per traversal. 1 = bright, toward 0 = dark. */
  damp?: number;
  /** First-order allpass coefficient. Nonzero stretches partials toward a bar. */
  disp?: number;
  /** Stereo position of this wire's pickup, -1 left .. +1 right. */
  pan?: number;
  /** Where along the wire excitation lands, 0..1. */
  exAt?: number;
  /** Excitation footprint: <1 is a mallet strike, 1 is a full pluck. */
  exWidth?: number;
}

export interface AgentTopo {
  id: number;
  kind: KindCode;
  /** Count of free (open) ports — radiates energy. */
  openPorts: number;
  impedance: number;
  /** Stereo position of this body, -1 left .. +1 right. */
  pan?: number;
  /** Body mode frequencies in Hz. */
  modeHz?: number[];
  /** Per-mode T60 in seconds. */
  modeT60?: number[];
  /** Per-mode amplitude. */
  modeGain?: number[];
  /** How hard this body drives the wires tied to it. */
  coupling?: number;
}

export interface NetTopology {
  wires: WireTopo[];
  agents: AgentTopo[];
}

export interface LatchEvent {
  type: 'latch';
  wireId: number;
  agentA: number;
  agentB: number;
  slotA: PortSlot;
  slotB: PortSlot;
  kindA: AgentKind;
  kindB: AgentKind;
  rest: number;
  latchLen: number;
}

export interface CollisionEvent {
  type: 'collision';
  agentA: number;
  agentB: number;
  kindA: AgentKind;
  kindB: AgentKind;
  impact: number;
  overlap: number;
  /** Generalized effective mass at the contact, lever arm included. */
  effMass: number;
  /** Closing speed along the normal. */
  vN: number;
  /** Sliding speed along the tangent. */
  vT: number;
  /** Contact normal, world space. With the headings this gives which feature
   *  of each body actually made contact — vertex, edge, or face. */
  nx: number;
  ny: number;
  headingA: number;
  headingB: number;
  /** Relative spin at the contact; a turning body scrapes as well as strikes. */
  spin: number;
}

export interface RewriteEvent {
  type: 'rewrite';
  phase: 'begin' | 'commit';
  rule: string;
  agentA: number;
  agentB: number;
  kindA: AgentKind;
  kindB: AgentKind;
  wireId: number;
  leftovers: number[];
}

export type AudioEvent = LatchEvent | CollisionEvent | RewriteEvent;

/** One touching pair, sent every frame while the overlap lasts. */
export interface ContactItem {
  agentA: number;
  agentB: number;
  /** Hertzian overlap as a 0..1 load — stiffness and friction share it. */
  load: number;
  /** Signed bow velocity (A relative to B). Zero when they are not sliding. */
  slide: number;
}

/** Live contact as the sim reports it, before load/slide mapping. */
export interface LiveContact {
  agentA: number;
  agentB: number;
  overlap: number;
  /** Signed tangent speed of A relative to B, px/s. */
  vT: number;
}

/** Direct air path between two bodies. Delay is travel time, gain dies with distance. */
export interface AirItem {
  agentA: number;
  agentB: number;
  length: number;
  gain: number;
  damp: number;
}

export type WorkletInMessage =
  | { type: 'topology'; topo: NetTopology }
  | { type: 'latch'; topo: NetTopology; wireId: number; gain: number }
  | { type: 'impulse'; wireId: number; end: 0 | 1; gain: number }
  | { type: 'junction'; agentId: number; gain: number }
  | {
      /** A real contact: a force of `peak` newtons-ish held for `dur` samples. */
      type: 'strike';
      agentId: number;
      peak: number;
      dur: number;
      /** 0 = struck on a flat face, 1 = struck on a vertex. */
      sharp: number;
    }
  | {
      /** Touching pairs this frame. Empty list = nothing in contact. */
      type: 'contact';
      items: ContactItem[];
    }
  | {
      /** Line-of-sight air paths this frame. Empty list = nobody in range. */
      type: 'air';
      items: AirItem[];
    }
  | {
      type: 'rewrite';
      phase: 0 | 1;
      wireId: number;
      agentA: number;
      agentB: number;
      leftovers: number[];
      gain: number;
    }
  | { type: 'gain'; master: number };

export type WorkletOutMessage = { type: 'waves'; packed: Float32Array };

/** Latest traveling-wave snapshot from the worklet. `index` maps wireId → record offset. */
export interface WaveSnapshot {
  packed: Float32Array;
  index: Map<number, number>;
}
