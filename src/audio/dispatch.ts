import type { Graph } from '../graph.ts';
import type { Agent } from '../agents.ts';
import {
  contactPeak,
  bowSpeed,
  contactSeconds,
  getSampleRate,
  latchGain,
  rewriteBeginGain,
  rewriteCommitGain,
  airDelaySamples,
  airGain,
  airDamp,
  AIR_CUTOFF_PX,
  AIR_MIN_PX,
  MAX_AIR_PATHS,
} from './presets.ts';
import { albedo, strikeSharpness } from './voice.ts';
import { buildTopology, kindBrightness } from './topology.ts';
import type { AudioEvent, LiveContact, WorkletInMessage } from './types.ts';
import type { NetTopology } from './types.ts';

/** Pure dispatch logic — maps sim events to worklet messages (testable without AudioContext). */
export function planLatchMessages(
  ev: Extract<AudioEvent, { type: 'latch' }>,
  graph: Graph,
  agents: Map<number, Agent>,
  prebuilt?: NetTopology,
): WorkletInMessage[] {
  // Levels dropped across the board: with the pickup no longer cancelling the
  // fundamental, the same message is far louder than it used to be.
  const g = latchGain(ev.latchLen) * 0.85 * (kindBrightness(ev.kindA) + kindBrightness(ev.kindB)) * 0.5;
  const topo = prebuilt ?? buildTopology(graph, agents);
  return [{ type: 'latch', topo, wireId: ev.wireId, gain: g }];
}

/**
 * A collision becomes a contact force, not a gain.
 *
 * The duration comes from Hertzian contact mechanics — heavier pairs stay in
 * contact longer, faster impacts are over sooner — and the peak follows from
 * the momentum that has to be reversed over that duration. So a hard knock is
 * shorter as well as louder, which is what makes it brighter, and none of the
 * numbers here are chosen by ear.
 */
export function planCollisionMessages(
  ev: Extract<AudioEvent, { type: 'collision' }>,
): WorkletInMessage[] {
  const tau = contactSeconds(ev.effMass, ev.vN);
  const dur = Math.max(2, Math.round(tau * getSampleRate()));
  const peak = contactPeak(ev.effMass, ev.vN, tau);

  // Which feature of each body actually made contact. A circle answers the
  // same from every direction; a triangle hit on a vertex is a click and hit
  // on a face is a thud, so two agents colliding twice never sound the same.
  const sa = strikeSharpness({ kind: ev.kindA, heading: ev.headingA } as never, ev.nx, ev.ny);
  const sb = strikeSharpness({ kind: ev.kindB, heading: ev.headingB } as never, -ev.nx, -ev.ny);
  // A glancing, spinning contact sheds energy sideways instead of into the body.
  const glance = 1 / (1 + Math.abs(ev.vT) / Math.max(1, Math.abs(ev.vN)));

  const strike = (agentId: number, sharp: number, kind: typeof ev.kindA, trim: number) => ({
    type: 'strike' as const,
    agentId,
    // A vertex concentrates the same momentum into a shorter contact.
    peak: Math.min(6, peak * (0.7 + albedo(kind) * 0.4) * glance * trim),
    dur: Math.max(2, Math.round(dur * (1 - sharp * 0.45))),
    sharp,
  });

  return [strike(ev.agentA, sa, ev.kindA, 1), strike(ev.agentB, sb, ev.kindB, 0.85)];
}

/** Below this, the pair is touching but not sliding — Hertzian spring only. */
const SLIDE_DEADZONE = 1.5;

/**
 * One message a frame carrying every touching pair. Resting contact is included
 * (slide = 0) so vibration can cross the surface without a scrape.
 */
export function planContactMessage(
  contacts: Map<string, LiveContact>,
): Extract<WorkletInMessage, { type: 'contact' }> {
  const items: Extract<WorkletInMessage, { type: 'contact' }>['items'] = [];
  for (const c of contacts.values()) {
    items.push({
      agentA: c.agentA,
      agentB: c.agentB,
      load: Math.max(0, Math.min(1, c.overlap / 1.6)),
      slide: Math.abs(c.vT) < SLIDE_DEADZONE ? 0 : bowSpeed(c.vT),
    });
  }
  return { type: 'contact', items };
}

/**
 * Direct line-of-sight between nearby bodies. Closest pairs first, capped so a
 * soup does not become N² delay lines. Empty list is how air coupling stops.
 */
export function planAirMessage(
  agents: Map<number, Agent>,
): Extract<WorkletInMessage, { type: 'air' }> {
  const list: Agent[] = [];
  for (const a of agents.values()) list.push(a);
  const cand: { a: number; b: number; d: number }[] = [];
  for (let i = 0; i < list.length; i++) {
    const A = list[i];
    for (let j = i + 1; j < list.length; j++) {
      const B = list[j];
      const d = Math.hypot(B.x - A.x, B.y - A.y);
      if (d >= AIR_MIN_PX && d <= AIR_CUTOFF_PX) cand.push({ a: A.id, b: B.id, d });
    }
  }
  cand.sort((x, y) => x.d - y.d);
  const n = Math.min(MAX_AIR_PATHS, cand.length);
  const items: Extract<WorkletInMessage, { type: 'air' }>['items'] = [];
  for (let i = 0; i < n; i++) {
    const c = cand[i];
    items.push({
      agentA: c.a,
      agentB: c.b,
      length: airDelaySamples(c.d),
      gain: airGain(c.d),
      damp: airDamp(c.d),
    });
  }
  return { type: 'air', items };
}

export function planRewriteMessages(
  ev: Extract<AudioEvent, { type: 'rewrite' }>,
): WorkletInMessage[] {
  const bright = (kindBrightness(ev.kindA) + kindBrightness(ev.kindB)) * 0.5;
  const g =
    (ev.phase === 'begin' ? rewriteBeginGain(ev.rule) : rewriteCommitGain(ev.rule)) * 0.6 * bright;
  return [
    {
      type: 'rewrite',
      phase: ev.phase === 'begin' ? 0 : 1,
      wireId: ev.wireId,
      agentA: ev.agentA,
      agentB: ev.agentB,
      leftovers: ev.leftovers,
      gain: g,
    },
  ];
}

export function planTopologyMessage(
  graph: Graph,
  agents: Map<number, Agent>,
): { type: 'topology'; topo: NetTopology } {
  return { type: 'topology', topo: buildTopology(graph, agents) };
}
