import { boundRadius, portLocal, slotsFor, type Agent, type AgentKind, type PortSlot } from '../agents.ts';
import type { Graph, Wire } from '../graph.ts';
import {
  bendLoss,
  delaySamplesForPath,
  feasibleDamp,
  lossForT60,
  portImpedance,
  stubDelaySamples,
  tautBrighten,
  tautness,
} from './presets.ts';
import type { AgentTopo, NetTopology, PanView, StubTopo, WireTopo } from './types.ts';
import { bodyCoupling, bodyTone } from './body.ts';
import { blendVoices, voiceFromAgent } from './voice.ts';
import { kindCode } from './types.ts';

/** Half-width of the stereo field in world pixels, when no camera is given. */
const PAN_SPREAD = 260;

/** World pixels that read as one unit of distance from the listener. */
const DIST_REF_PX = 320;
/**
 * Fallback height when there is no camera, in dist units. Keeps a body at the
 * origin from sitting on the listener.
 */
const LISTENER_HEIGHT = 0.45;

/**
 * How far the listener floats above the pond. An ortho camera has no real
 * frustum, so this is the dolly: half the world-space view, in dist units.
 * Zooming in lowers you onto the surface; zooming out is flying up, which is
 * the gesture that has to change loudness.
 */
function listenerHeight(view: PanView | null | undefined): number {
  if (view && view.zoom > 0 && view.viewW > 0 && view.viewH > 0) {
    const half = 0.5 * Math.min(view.viewW, view.viewH) / view.zoom;
    return Math.max(0.12, Math.min(8, half / DIST_REF_PX));
  }
  return LISTENER_HEIGHT;
}

/**
 * Distance from the listener: 0 is on top of them, 1 is a comfortable way off.
 *
 * Planar offset is world pixels, not screen pixels — a body at the edge of a
 * zoomed-out view really is further away. Height is the camera dolly, so the
 * cluster you are looking at also recedes when you scroll out.
 */
function listenerDistance(
  wx: number,
  wy: number,
  view: PanView | null | undefined,
  centreX: number,
  centreY: number,
  height: number,
): number {
  const ox = view ? view.x : centreX;
  const oy = view ? view.y : centreY;
  const r = Math.hypot(wx - ox, wy - oy) / DIST_REF_PX;
  return Math.hypot(r, height);
}

function stereoPan(wx: number, view: PanView | null | undefined, centreX: number): number {
  if (view && view.viewW > 0 && view.zoom > 0) {
    const sx = (wx - view.x) * view.zoom;
    return Math.max(-1, Math.min(1, sx / (view.viewW * 0.5))) * 0.8;
  }
  return Math.max(-1, Math.min(1, (wx - centreX) / PAN_SPREAD)) * 0.8;
}

function wireTopo(
  wire: Wire,
  agents: Map<number, Agent>,
  centreX: number,
  centreY: number,
  brightness: number,
  view: PanView | null | undefined,
  height: number,
): WireTopo | null {
  const A = agents.get(wire.a.id);
  const B = agents.get(wire.b.id);
  if (!A || !B) return null;

  const voice = blendVoices(voiceFromAgent(A, wire.a.slot), voiceFromAgent(B, wire.b.slot));
  const taut = tautness(wire.lastLen, wire.rest, wire.ropeLen);
  // Travel time along the rope, not a pitch we assigned to rest length.
  const length = delaySamplesForPath(Math.max(1, wire.ropeLen), taut, voice.disp);
  const bend = bendLoss(wire.nodes.length, wire.ropeLen, wire.rest);
  // Slack rope and a busy net both darken the wire; taut ropes brighten;
  // feasibleDamp then lifts it back to whatever the T60 target can support.
  const wanted = tautBrighten(Math.max(0.04, voice.damp * (1 - bend * 1.6) * brightness), taut);
  const damp = feasibleDamp(wanted, length, voice.t60);

  return {
    id: wire.id,
    length,
    // T60-based, so a wire rings for the same wall-clock time whatever its
    // pitch. A flat per-sample factor made bass ring 2.4x longer than treble.
    loss: lossForT60(length, voice.t60, damp),
    agentA: wire.a.id,
    agentB: wire.b.id,
    zA: portImpedance(A.kind, wire.a.slot),
    zB: portImpedance(B.kind, wire.b.slot),
    bend,
    damp,
    disp: voice.disp,
    pan: stereoPan((A.x + B.x) * 0.5, view, centreX),
    dist: listenerDistance((A.x + B.x) * 0.5, (A.y + B.y) * 0.5, view, centreX, centreY, height),
    exAt: voice.at,
    exWidth: voice.width,
  };
}

function slotCode(slot: PortSlot): 0 | 1 | 2 {
  if (slot === 'p') return 0;
  if (slot === 'l') return 1;
  return 2;
}

function stubFor(agent: Agent, slot: PortSlot): StubTopo {
  const tip = portLocal(agent.kind, slot);
  const path = Math.hypot(tip.x, tip.y) + boundRadius(agent) * 0.6;
  return {
    slot: slotCode(slot),
    length: stubDelaySamples(path),
    z: portImpedance(agent.kind, slot),
  };
}

function agentTopo(
  agent: Agent,
  graph: Graph,
  centreX: number,
  centreY: number,
  view: PanView | null | undefined,
  height: number,
): AgentTopo {
  const slots = slotsFor(agent.kind);
  const open = slots.filter((slot) => graph.isFree({ id: agent.id, slot }));
  const z =
    slots.reduce((sum, slot) => sum + portImpedance(agent.kind, slot), 0) / Math.max(1, slots.length);
  const tone = bodyTone(agent);
  const stubs = open.map((slot) => stubFor(agent, slot));
  return {
    id: agent.id,
    kind: kindCode(agent.kind),
    openPorts: open.length,
    stubs,
    impedance: z,
    pan: stereoPan(agent.x, view, centreX),
    dist: listenerDistance(agent.x, agent.y, view, centreX, centreY, height),
    modeHz: tone.freq,
    modeT60: tone.decay,
    modeGain: tone.gain,
    coupling: bodyCoupling(agent),
  };
}

export function buildTopology(
  graph: Graph,
  agents: Map<number, Agent>,
  view?: PanView | null,
): NetTopology {
  let centreX = 0;
  let centreY = 0;
  let n = 0;
  for (const agent of agents.values()) {
    centreX += agent.x;
    centreY += agent.y;
    n++;
  }
  if (n > 0) {
    centreX /= n;
    centreY /= n;
  }

  // Thin the top end as the net gets busy: 30 wires ringing at full brightness
  // is a wash, and pulling the highs is what keeps a dense net ambient.
  const brightness = 1 / (1 + graph.wires.size * 0.012);
  const height = listenerHeight(view);

  const wires: WireTopo[] = [];
  for (const wire of graph.wires.values()) {
    const t = wireTopo(wire, agents, centreX, centreY, brightness, view, height);
    if (t) wires.push(t);
  }
  const agentList: AgentTopo[] = [];
  for (const agent of agents.values()) {
    agentList.push(agentTopo(agent, graph, centreX, centreY, view, height));
  }
  return { wires, agents: agentList, height };
}

export function kindBrightness(kind: AgentKind): number {
  if (kind === 'era') return 1.1;
  if (kind === 'dup') return 0.95;
  return 0.75;
}
