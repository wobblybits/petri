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
import type { AudioEvent, LiveContact, LiveWireContact, WorkletInMessage } from './types.ts';
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
 * One message a frame for every scraping wire pair. Empty list is how they
 * separate. Both strings get the same load and slide — a symmetric bow.
 */
export function planWireContactMessage(
  contacts: Map<string, LiveWireContact>,
): Extract<WorkletInMessage, { type: 'wireContact' }> {
  const items: Extract<WorkletInMessage, { type: 'wireContact' }>['items'] = [];
  for (const c of contacts.values()) {
    items.push({
      wireA: c.wireA,
      wireB: c.wireB,
      load: Math.max(0, Math.min(1, c.overlap / 1.6)),
      slide: Math.abs(c.vT) < SLIDE_DEADZONE ? 0 : bowSpeed(c.vT),
      atA: Math.max(0.02, Math.min(0.98, c.atA)),
      atB: Math.max(0.02, Math.min(0.98, c.atB)),
    });
  }
  return { type: 'wireContact', items };
}

/**
 * Scratch for planAirMessage, kept at module scope so a per-frame call does not
 * allocate. Main thread only, and the whole selection finishes inside one call.
 */
const airBestA = new Int32Array(MAX_AIR_PATHS);
const airBestB = new Int32Array(MAX_AIR_PATHS);
const airBestD2 = new Float64Array(MAX_AIR_PATHS);
const airXs: number[] = [];
const airYs: number[] = [];
const airIds: number[] = [];

const AIR_MIN_SQ = AIR_MIN_PX * AIR_MIN_PX;
const AIR_CUTOFF_SQ = AIR_CUTOFF_PX * AIR_CUTOFF_PX;

/**
 * Direct line-of-sight between nearby bodies. Closest pairs first, capped so a
 * soup does not become N² delay lines. Empty list is how air coupling stops.
 *
 * Still O(n²) in pair tests — a real fix needs a broadphase the sim does not
 * have yet — but the two things that actually dominated are gone. Distances are
 * compared squared, so the sqrt is paid only for the handful of pairs kept
 * rather than all of them. And the nearest pairs are selected by insertion into
 * a fixed 48-slot array instead of sorting every candidate: at 120 agents that
 * sort ranked 4084 entries to use 49 of them, so 98.8% of it was wasted.
 */
export function planAirMessage(
  agents: Map<number, Agent>,
): Extract<WorkletInMessage, { type: 'air' }> {
  airXs.length = 0;
  airYs.length = 0;
  airIds.length = 0;
  for (const a of agents.values()) {
    airXs.push(a.x);
    airYs.push(a.y);
    airIds.push(a.id);
  }

  const n = airIds.length;
  let kept = 0;
  // Once the table is full this is the distance a pair has to beat to matter,
  // which rejects the overwhelming majority on one compare.
  let worst = Infinity;

  // Deliberately all-pairs. A broad phase was tried here and made it slower:
  // the air cutoff is 240 px, comparable to the whole swarm, so every body
  // lands in the same cell or its neighbour and the grid rejects nothing while
  // still charging for the rebuild. The sim's contact solver is the opposite
  // case — a ~40 px cell against the same swarm — which is why the grid pays
  // off there and not here.
  for (let i = 0; i < n; i++) {
    const ax = airXs[i];
    const ay = airYs[i];
    for (let j = i + 1; j < n; j++) {
      const dx = airXs[j] - ax;
      const dy = airYs[j] - ay;
      const d2 = dx * dx + dy * dy;
      if (d2 < AIR_MIN_SQ || d2 > AIR_CUTOFF_SQ) continue;
      if (kept === MAX_AIR_PATHS && d2 >= worst) continue;

      let at = kept < MAX_AIR_PATHS ? kept++ : MAX_AIR_PATHS - 1;
      while (at > 0 && airBestD2[at - 1] > d2) {
        airBestD2[at] = airBestD2[at - 1];
        airBestA[at] = airBestA[at - 1];
        airBestB[at] = airBestB[at - 1];
        at--;
      }
      airBestD2[at] = d2;
      airBestA[at] = airIds[i];
      airBestB[at] = airIds[j];
      if (kept === MAX_AIR_PATHS) worst = airBestD2[MAX_AIR_PATHS - 1];
    }
  }

  const items: Extract<WorkletInMessage, { type: 'air' }>['items'] = [];
  for (let i = 0; i < kept; i++) {
    const d = Math.sqrt(airBestD2[i]);
    items.push({
      agentA: airBestA[i],
      agentB: airBestB[i],
      length: airDelaySamples(d),
      gain: airGain(d),
      damp: airDamp(d),
    });
  }
  return { type: 'air', items };
}

/**
 * A body arriving. Visually this is one of the loudest things that happens —
 * an agent simply appears — and it was the last such event making no sound at
 * all. A long, soft, blunt contact rather than a strike: an entrance, not a hit.
 */
export function planSpawnMessages(
  ev: Extract<AudioEvent, { type: 'spawn' }>,
): WorkletInMessage[] {
  return [
    {
      type: 'strike',
      agentId: ev.agent,
      // The dark body arrives more softly than the light one, same as
      // everywhere else: what you see is what you hear.
      peak: 0.16 + albedo(ev.kind) * 0.12,
      dur: Math.round(0.006 * getSampleRate()),
      sharp: 0.1,
    },
  ];
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
