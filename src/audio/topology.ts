import { slotsFor, type Agent, type AgentKind } from '../agents.ts';
import type { Graph, Wire } from '../graph.ts';
import {
  bendLoss,
  delaySamplesForPath,
  feasibleDamp,
  lossForT60,
  openPortRadiation,
  portImpedance,
  tautBrighten,
  tautness,
} from './presets.ts';
import type { AgentTopo, NetTopology, PanView, WireTopo } from './types.ts';
import { bodyCoupling, bodyTone } from './body.ts';
import { blendVoices, voiceFromAgent } from './voice.ts';
import { kindCode } from './types.ts';

/** Half-width of the stereo field in world pixels, when no camera is given. */
const PAN_SPREAD = 260;

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
  brightness: number,
  view: PanView | null | undefined,
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
    exAt: voice.at,
    exWidth: voice.width,
  };
}

function agentTopo(
  agent: Agent,
  graph: Graph,
  centreX: number,
  view: PanView | null | undefined,
): AgentTopo {
  const slots = slotsFor(agent.kind);
  const open = slots.filter((slot) => graph.isFree({ id: agent.id, slot }));
  const z =
    slots.reduce((sum, slot) => sum + portImpedance(agent.kind, slot), 0) / Math.max(1, slots.length);
  const tone = bodyTone(agent);
  return {
    id: agent.id,
    kind: kindCode(agent.kind),
    openPorts: open.length,
    impedance: z * (1 + openPortRadiation(open.length)),
    pan: stereoPan(agent.x, view, centreX),
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
  let n = 0;
  for (const agent of agents.values()) {
    centreX += agent.x;
    n++;
  }
  if (n > 0) centreX /= n;

  // Thin the top end as the net gets busy: 30 wires ringing at full brightness
  // is a wash, and pulling the highs is what keeps a dense net ambient.
  const brightness = 1 / (1 + graph.wires.size * 0.012);

  const wires: WireTopo[] = [];
  for (const wire of graph.wires.values()) {
    const t = wireTopo(wire, agents, centreX, brightness, view);
    if (t) wires.push(t);
  }
  const agentList: AgentTopo[] = [];
  for (const agent of agents.values()) {
    agentList.push(agentTopo(agent, graph, centreX, view));
  }
  return { wires, agents: agentList };
}

export function kindBrightness(kind: AgentKind): number {
  if (kind === 'era') return 1.1;
  if (kind === 'dup') return 0.95;
  return 0.75;
}
