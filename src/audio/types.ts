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
  viewH: number;
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
  /** Distance from the listener: 0 on top of them, 1 a comfortable way off. */
  dist?: number;
  /** Where along the wire excitation lands, 0..1. */
  exAt?: number;
  /** Excitation footprint: <1 is a mallet strike, 1 is a full pluck. */
  exWidth?: number;
  /** Detail tier from apparent size: 0 full waveguide, 1 modal, 2 ensemble. */
  lod?: number;
}

/** One unused stem, a short open pipe from the body junction to the lip. */
export interface StubTopo {
  /** 0 = p, 1 = l, 2 = r */
  slot: 0 | 1 | 2;
  /** One-way delay in samples. */
  length: number;
  z: number;
}

export interface AgentTopo {
  id: number;
  kind: KindCode;
  /** Count of free (open) ports — each is a stub, not a shunt. */
  openPorts: number;
  /** Open-ended bore for every unattached stem. */
  stubs?: StubTopo[];
  impedance: number;
  /** Stereo position of this body, -1 left .. +1 right. */
  pan?: number;
  /** Distance from the listener: 0 on top of them, 1 a comfortable way off. */
  dist?: number;
  /** Body mode frequencies in Hz. */
  modeHz?: number[];
  /** Per-mode T60 in seconds. */
  modeT60?: number[];
  /** Per-mode amplitude. */
  modeGain?: number[];
  /** How hard this body drives the wires tied to it. */
  coupling?: number;
  /** Agents in this body's connected component. 1 = unwired, 2 = a pair. */
  compN?: number;
  /** Detail tier from apparent size: 0 full modal body, 1 reduced, 2 ensemble. */
  lod?: number;
  /**
   * True when this body is the interior of a commute-mesh: no delay lines,
   * no stubs, strikes dump into `tissueId`.
   */
  tissue?: boolean;
  /** Component resonator this body belongs to, when `tissue` or on the skin boundary. */
  tissueId?: number;
  /**
   * Extra junction admittance into that resonator. Skin agents on the edge of
   * tissue present this as one more port; fully-skin and fully-tissue are 0.
   */
  tissueY?: number;
}

/**
 * One collapsed commute-mesh interior. A handful of lattice modes standing
 * in for the delay lines we are not running. Driven by dumped wires, by
 * strikes on tissue bodies, and by the boundary junctions of the skin.
 */
export interface TissueTopo {
  /** Stable id: the smallest agent id in the component. */
  id: number;
  n: number;
  /** Characteristic one-way delay in samples (median of the mesh wires). */
  delay: number;
  pan?: number;
  dist?: number;
  lod?: number;
}

export interface NetTopology {
  wires: WireTopo[];
  agents: AgentTopo[];
  tissues?: TissueTopo[];
  /** Listener height above the pond, in the same units as `dist`. */
  height?: number;
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

export interface SpawnEvent {
  type: 'spawn';
  agent: number;
  kind: AgentKind;
}

export type AudioEvent = LatchEvent | CollisionEvent | RewriteEvent | SpawnEvent;

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
  | { type: 'gain'; master: number }
  | {
      /** Camera pose only. Must not rebuild delay lines. */
      type: 'listen';
      height?: number;
      wires: { id: number; pan?: number; dist?: number; lod?: number }[];
      agents: { id: number; pan?: number; dist?: number; lod?: number }[];
    }
  | {
      /** Rope delay and damping. The graph shape is unchanged. */
      type: 'tune';
      wires: { id: number; length: number; damp?: number; loss?: number; bend?: number }[];
    };

export type WorkletOutMessage =
  | { type: 'waves'; packed: Float32Array }
  | { type: 'error'; message: string };

/** Latest traveling-wave snapshot from the worklet. `index` maps wireId → record offset. */
export interface WaveSnapshot {
  packed: Float32Array;
  index: Map<number, number>;
}
