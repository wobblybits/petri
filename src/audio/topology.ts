import { boundRadius, portLocal, slotsFor, type Agent, type AgentKind, type PortSlot } from '../agents.ts';
import type { Graph, Wire } from '../graph.ts';
import {
  bendLoss,
  delaySamplesForPath,
  feasibleDamp,
  lossForT60,
  componentLoadScale,
  portImpedance,
  stubDelaySamples,
  tautBrighten,
  tautness,
} from './presets.ts';
import type { AgentTopo, NetTopology, PanView, StubTopo, TissueTopo, WireTopo } from './types.ts';
import { bodyCoupling, bodyTone } from './body.ts';
import {
  AGENT_BAND,
  COST_US,
  LOD_FAR,
  LOD_NEAR,
  LodSelector,
  WIRE_BAND,
  agentKey,
  apparentPx,
  assign,
  onScreen,
  wireKey,
  type LodCandidate,
} from './lod.ts';
import { blendVoices, voiceFromAgent } from './voice.ts';
import { kindCode } from './types.ts';
import { clockRoots, median, meshRoots, skinAgents } from './motif.ts';

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
  cands: LodCandidate[],
  compN: number,
): WireTopo | null {
  const A = agents.get(wire.a.id);
  const B = agents.get(wire.b.id);
  if (!A || !B) return null;

  const midX = (A.x + B.x) * 0.5;
  const midY = (A.y + B.y) * 0.5;
  const voice = blendVoices(voiceFromAgent(A, wire.a.slot), voiceFromAgent(B, wire.b.slot));
  const taut = tautness(wire.lastLen, wire.rest, wire.ropeLen);
  // Travel time along the live rope, not the rest cubic. A body sitting on
  // the wire lengthens the path; a yank also raises tautness. Using the cubic
  // made collisions inaudible on an already-ringing string.
  const path = Math.max(1, wire.pitchFloor, wire.lastLen > 1 ? wire.lastLen : wire.ropeLen);
  const length = delaySamplesForPath(path, taut, voice.disp);
  const bend = bendLoss(wire.nodes.length, wire.ropeLen, wire.rest);
  // Slack rope and a busy net both darken the wire; taut ropes brighten.
  // Component load then shortens T60 so a clump is a heavier structure, not
  // N copies of a pair — feasibleDamp is asked to hit that shorter ring.
  const t60 = voice.t60 * componentLoadScale(compN);
  const wanted = tautBrighten(Math.max(0.04, voice.damp * (1 - bend * 1.6) * brightness), taut);
  const damp = feasibleDamp(wanted, length, t60);

  cands.push({
    key: wireKey(wire.id),
    px: apparentPx(path, view),
    visible: onScreen(midX, midY, path, view),
    band: WIRE_BAND,
    nearUs: COST_US.nearWire,
    midUs: COST_US.midWire,
  });

  return {
    id: wire.id,
    length,
    // T60-based, so a wire rings for the same wall-clock time whatever its
    // pitch. A flat per-sample factor made bass ring 2.4x longer than treble.
    loss: lossForT60(length, t60, damp),
    agentA: wire.a.id,
    agentB: wire.b.id,
    zA: portImpedance(A.kind, wire.a.slot),
    zB: portImpedance(B.kind, wire.b.slot),
    bend,
    damp,
    disp: voice.disp,
    pan: stereoPan(midX, view, centreX),
    dist: listenerDistance(midX, midY, view, centreX, centreY, height),
    exAt: voice.at,
    exWidth: voice.width,
    lod: LOD_NEAR,
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
  cands: LodCandidate[],
  compN: number,
): AgentTopo {
  const slots = slotsFor(agent.kind);
  const open = slots.filter((slot) => graph.isFree({ id: agent.id, slot }));
  const z =
    slots.reduce((sum, slot) => sum + portImpedance(agent.kind, slot), 0) / Math.max(1, slots.length);
  const tone = bodyTone(agent);
  const load = componentLoadScale(compN);
  const decay = tone.decay.map((d) => d * load);
  const stubs = open.map((slot) => stubFor(agent, slot));
  const size = boundRadius(agent) * 2;
  cands.push({
    key: agentKey(agent.id),
    px: apparentPx(size, view),
    visible: onScreen(agent.x, agent.y, size, view),
    band: AGENT_BAND,
    nearUs: COST_US.nearAgent,
    midUs: COST_US.midAgent,
  });
  return {
    id: agent.id,
    kind: kindCode(agent.kind),
    openPorts: open.length,
    stubs,
    impedance: z,
    pan: stereoPan(agent.x, view, centreX),
    dist: listenerDistance(agent.x, agent.y, view, centreX, centreY, height),
    modeHz: tone.freq,
    modeT60: decay,
    modeGain: tone.gain,
    coupling: bodyCoupling(agent),
    compN,
    lod: LOD_NEAR,
  };
}

export function buildTopology(
  graph: Graph,
  agents: Map<number, Agent>,
  view?: PanView | null,
  lod: LodSelector | null = null,
  forceSkin?: Set<number> | null,
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
  // is a wash, and pulling the highs is what keeps a dense net ambient. This
  // is colour, not Q — duration is componentLoadScale, below.
  const brightness = 1 / (1 + graph.wires.size * 0.012);
  const height = listenerHeight(view);
  const roots = graph.componentIds(agents);
  const compN = new Map<number, number>();
  const counts = new Map<number, number>();
  for (const root of roots.values()) counts.set(root, (counts.get(root) ?? 0) + 1);
  for (const [id, root] of roots) compN.set(id, counts.get(root) ?? 1);

  // Tiers are a whole-frame decision, not a per-object one: the budget is
  // shared, so nothing can be tiered until every candidate is on the table.
  const cands: LodCandidate[] = [];
  const wires: WireTopo[] = [];
  for (const wire of graph.wires.values()) {
    const t = wireTopo(
      wire,
      agents,
      centreX,
      centreY,
      brightness,
      view,
      height,
      cands,
      compN.get(wire.a.id) ?? 1,
    );
    if (t) wires.push(t);
  }
  const agentList: AgentTopo[] = [];
  for (const agent of agents.values()) {
    agentList.push(agentTopo(agent, graph, centreX, centreY, view, height, cands, compN.get(agent.id) ?? 1));
  }

  const clocks = view
    ? clockRoots(
        roots,
        (id) => {
          const a = agents.get(id);
          return a ? kindCode(a.kind) : 0;
        },
        (id) => {
          const a = agents.get(id);
          if (!a) return 0;
          let n = 0;
          for (const slot of slotsFor(a.kind)) {
            if (graph.isFree({ id, slot })) n++;
          }
          return n;
        },
      )
    : new Set<number>();
  if (clocks.size) promoteClockCands(cands, clocks, roots, wires, forceSkin);

  const tiers = assign(cands, lod);
  for (const w of wires) w.lod = tiers.get(wireKey(w.id)) ?? LOD_NEAR;
  for (const a of agentList) a.lod = tiers.get(agentKey(a.id)) ?? LOD_NEAR;
  lod?.sweep();

  const tissues = view ? collapseMesh(roots, agents, wires, agentList, forceSkin) : [];
  return { wires, agents: agentList, tissues, height };
}

/**
 * A clock is one instrument. If any cell is on screen (or force-skinned),
 * the rest of the component is voiced at that same apparent size so a far
 * leftover Era is not an ensemble bed while you look at the pair.
 */
function promoteClockCands(
  cands: LodCandidate[],
  clocks: Set<number>,
  roots: Map<number, number>,
  wires: WireTopo[],
  forceSkin?: Set<number> | null,
): void {
  const byKey = new Map<number, LodCandidate>();
  for (const c of cands) byKey.set(c.key, c);
  const idsByRoot = new Map<number, number[]>();
  for (const [id, root] of roots) {
    if (!clocks.has(root)) continue;
    const list = idsByRoot.get(root);
    if (list) list.push(id);
    else idsByRoot.set(root, [id]);
  }
  for (const [root, ids] of idsByRoot) {
    const keys: number[] = [];
    for (const id of ids) keys.push(agentKey(id));
    for (const w of wires) {
      if (roots.get(w.agentA) === root) keys.push(wireKey(w.id));
    }
    let maxPx = 0;
    let vis = false;
    let forced = false;
    if (forceSkin) for (const id of ids) if (forceSkin.has(id)) forced = true;
    for (const k of keys) {
      const c = byKey.get(k);
      if (!c) continue;
      if (c.px > maxPx) maxPx = c.px;
      if (c.visible) vis = true;
    }
    if (forced) {
      vis = true;
      maxPx = Math.max(maxPx, AGENT_BAND.near, WIRE_BAND.near);
    }
    if (!vis) continue;
    for (const k of keys) {
      const c = byKey.get(k);
      if (!c) continue;
      c.visible = true;
      if (c.px < maxPx) c.px = maxPx;
    }
  }
}

/**
 * Collapse the interior of each commute-mesh. Mutates `wires` (drops tissue
 * strings) and `agentList` (flags, tissueY, no stubs on interior bodies).
 */
function collapseMesh(
  roots: Map<number, number>,
  agents: Map<number, Agent>,
  wires: WireTopo[],
  agentList: AgentTopo[],
  forceSkin?: Set<number> | null,
): TissueTopo[] {
  const meshes = meshRoots(roots, (id) => {
    const a = agents.get(id);
    return a ? kindCode(a.kind) : 0;
  });
  if (meshes.size === 0) return [];

  const seed: number[] = [];
  if (forceSkin) for (const id of forceSkin) seed.push(id);
  for (const a of agentList) {
    if ((a.lod ?? 0) === LOD_NEAR) seed.push(a.id);
  }
  for (const w of wires) {
    if ((w.lod ?? 0) === LOD_NEAR) {
      seed.push(w.agentA);
      seed.push(w.agentB);
    }
  }
  const skin = skinAgents(meshes, roots, seed, wires);

  const delayByRoot = new Map<number, number[]>();
  const panByRoot = new Map<number, { pan: number; dist: number; n: number; minId: number }>();
  for (const w of wires) {
    const root = roots.get(w.agentA);
    if (root === undefined || !meshes.has(root)) continue;
    const list = delayByRoot.get(root);
    if (list) list.push(w.length);
    else delayByRoot.set(root, [w.length]);
  }

  const keep = new Set<number>();
  const tissueY = new Map<number, number>();
  for (const w of wires) {
    const root = roots.get(w.agentA);
    if (root === undefined || !meshes.has(root)) {
      keep.add(w.id);
      continue;
    }
    const aSkin = skin.has(w.agentA);
    const bSkin = skin.has(w.agentB);
    if (aSkin && bSkin) {
      keep.add(w.id);
      continue;
    }
    if (aSkin && !bSkin) {
      const y = 1 / Math.max(0.05, w.zA ?? 1);
      tissueY.set(w.agentA, (tissueY.get(w.agentA) ?? 0) + y);
    } else if (bSkin && !aSkin) {
      const y = 1 / Math.max(0.05, w.zB ?? 1);
      tissueY.set(w.agentB, (tissueY.get(w.agentB) ?? 0) + y);
    }
  }

  let nW = 0;
  for (const w of wires) {
    if (keep.has(w.id)) wires[nW++] = w;
  }
  wires.length = nW;

  const tissueAgents = new Map<number, number>();
  for (const a of agentList) {
    const root = roots.get(a.id);
    if (root === undefined || !meshes.has(root)) continue;
    let acc = panByRoot.get(root);
    if (!acc) {
      acc = { pan: 0, dist: 0, n: 0, minId: a.id };
      panByRoot.set(root, acc);
    }
    acc.pan += a.pan ?? 0;
    acc.dist += a.dist ?? 0;
    acc.n++;
    if (a.id < acc.minId) acc.minId = a.id;
    if (skin.has(a.id)) {
      a.tissue = false;
      a.tissueY = tissueY.get(a.id) ?? 0;
    } else {
      a.tissue = true;
      a.stubs = [];
      a.openPorts = 0;
      a.tissueY = 0;
      tissueAgents.set(root, (tissueAgents.get(root) ?? 0) + 1);
    }
  }

  const tissues: TissueTopo[] = [];
  for (const root of meshes) {
    if ((tissueAgents.get(root) ?? 0) === 0) continue;
    const acc = panByRoot.get(root);
    const tid = acc?.minId ?? root;
    for (const a of agentList) {
      if (roots.get(a.id) === root) a.tissueId = tid;
    }
    const delay = median(delayByRoot.get(root) ?? []);
    tissues.push({
      id: tid,
      n: acc?.n ?? 0,
      delay: delay > 0 ? delay : 64,
      pan: acc && acc.n > 0 ? acc.pan / acc.n : 0,
      dist: acc && acc.n > 0 ? acc.dist / acc.n : 0.5,
      lod: LOD_FAR,
    });
  }
  return tissues;
}

export function kindBrightness(kind: AgentKind): number {
  if (kind === 'era') return 1.1;
  if (kind === 'dup') return 0.95;
  return 0.75;
}
