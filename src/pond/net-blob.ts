import { CODE_KIND, KIND_CODE } from '../agent-store.ts';
import {
  CHEM_LEN,
  CHEM_SEGMENTS,
  CRITIC_LEN,
  PLASTIC_BASE,
  PLASTIC_LEN,
  STATE_DIMS,
  type ChemSegment,
} from '../chem-layout.ts';
import type { AgentKind, PortSlot } from '../agents.ts';

/*
 * One net, on disk: the topology, and every number each of its bodies carries
 * that is not recomputable from those two things.
 *
 * The shape of the problem is that a net is mostly typed arrays — a genome is
 * 188 floats and a learned delta is 64 more, times however many bodies — and
 * mostly *not* JSON. So this is a JSON header describing a binary payload,
 * which gives both halves what they want: the header is readable with `.dump`
 * in `sqlite3` and tells you what you are looking at, and the payload is the
 * store's own arrays copied out flat.
 *
 * ## Why it is self-describing
 *
 * The header carries the genome's dimensions and every section's name, type,
 * offset and per-body stride. That is not decoration. `chem-layout.ts` exists
 * because a hand-copied `CHEM_LEN` drifted twice in one session, and the
 * second time shipped; a blob written before such a drift and read after it
 * would not crash, it would silently hand every body a genome cut at the
 * wrong offsets — a lineage's evolved behaviour quietly replaced by garbage
 * that still runs. A stored pond is a long-lived artifact and the layout is
 * not.
 *
 * It also means a reader does not have to be this module. The lab page can
 * walk `sections` with a `DataView` and render `energyCap` by name without
 * importing anything from the simulation.
 *
 * ## Versioning
 *
 * Format 1 refused any blob whose four dimensions differed from the build's,
 * and that turned out to be the wrong grain: the layout changed twice in a
 * week, each time by a head appended to the end, and every stored net became
 * unreadable for a change that had touched none of its numbers.
 *
 * Format 2 writes the genome's **segment map** (`CHEM_SEGMENTS`) into the
 * header, and `compatibility` lines it up with this build's by name:
 *
 *   - a segment with the same name and length is carried, wherever it moved;
 *   - one this build has and the blob does not is seeded for the body's kind;
 *   - one the blob has and this build does not is dropped;
 *   - one whose length changed refuses the net, naming it;
 *   - and the learned block (`plastic`, `trace`) follows its segments, so a
 *     weight that was learnable and no longer is refuses too.
 *
 * The recurrent state's width, the critic and the learned block's width are
 * not migratable: change any of them and nothing a net learned means the same
 * thing, so those refuse outright. A format 1 blob carries no map and is
 * therefore readable only at an exact match, as before.
 *
 * `decodeNet` hands back the blob at *its own* layout, with `layout`,
 * `segments` and `plasticAt` on the `NetData` saying so; `migrateNet` turns
 * that into this build's, given a seeder for the missing segments; and
 * `plantNet` calls it, so a caller that only ever plants never sees any of
 * this. The header also carries provenance — commit, time, and which run and
 * net a blob came from — for reading back, not for gating.
 *
 * ## What is not in here
 *
 * Agent ids: they are handed out per pond and mean nothing in another one, so
 * wires address bodies by their index in this blob. Mass, scale and alpha:
 * derived from kind and params at birth. Velocity, drive, stun, trail, the
 * rope node positions and the gait phase: a planted net is a net dropped into
 * different water, and none of that survives the move in any meaningful
 * sense.
 *
 * `trace` and `h` *are* in here, though neither is inherited and both are
 * gone within a second of simulated time. They cost 68 floats a body and
 * they are the difference between reloading a net and reloading the net that
 * was actually running — worth having for the one case where a run is
 * continued immediately rather than crossed into a new pond.
 */

/** Bumped when the payload's meaning changes in a way a reader must notice. */
export const NET_FORMAT = 2;

const MAGIC = 'petri-net';

/** Wires address ports by index here, because `PortSlot` is a string. */
export const SLOT_CODE: Record<PortSlot, number> = { p: 0, l: 1, r: 2 };
export const CODE_SLOT: PortSlot[] = ['p', 'l', 'r'];

/**
 * The heritable scalars, in the order the `scalar` section stores them.
 *
 * Deliberately written out rather than imported from `TRAIT_KEYS`: that list
 * is a live thing the simulation reorders when a trait becomes an output head
 * (four of them already have), and a stored blob's field order must not move
 * underneath it. The suite asserts the two still cover the same set, so the
 * day a trait is added the test says so rather than the pond quietly losing
 * it.
 */
export const SCALAR_FIELDS = [
  'extra',
  'requestDecay',
  'energyCap',
  'debtCap',
  'rescueTo',
  'assort',
  // Appended, never inserted: a stored blob's field order is its layout, and
  // moving one would make every net in the library decode as something else.
  // `NET_FORMAT` goes up with it so an older blob is refused rather than
  // silently read one scalar short.
  'adenylate',
] as const;
export type ScalarField = (typeof SCALAR_FIELDS)[number];

export const POSE_FIELDS = ['x', 'y', 'heading'] as const;
export const ANCESTRY_FIELDS = ['born', 'lineage'] as const;
export const WIRE_FIELDS = ['a', 'aSlot', 'b', 'bSlot'] as const;

export type SectionType = 'u8' | 'i32' | 'f32' | 'f64';

export interface Section {
  name: string;
  type: SectionType;
  /** Bytes from the start of the payload. */
  offset: number;
  /** Elements, not bytes: `stride * rows`. */
  count: number;
  /** Elements per row (per body, or per wire for `wire`). */
  stride: number;
  /** Present when the stride is a named tuple rather than a vector. */
  fields?: readonly string[];
  /** Present when the values are an enumeration; the index is the code. */
  values?: readonly string[];
}

/** The genome dimensions a blob was written against. */
export interface NetLayout {
  chem: number;
  plastic: number;
  critic: number;
  state: number;
}

/** Where a blob came from, when the writer knew. */
export interface NetSource {
  run?: number;
  net?: number;
  /** Simulated seconds into that run. */
  t?: number;
  file?: string;
}

/** Provenance a writer can attach. Read back, never gated on. */
export interface NetMeta {
  /** `9a3f21c`, or `9a3f21c+dirty`; null when there was no repository. */
  commit?: string | null;
  /** ISO 8601. Left out by default so the same net encodes to the same bytes. */
  written?: string;
  source?: NetSource;
  note?: string;
}

export interface NetHeader extends NetMeta {
  magic: string;
  format: number;
  layout: NetLayout;
  bodies: number;
  wires: number;
  sections: Section[];
  /** Format 2 and up: the genome's segment map, and where the learned block starts in it. */
  segments?: ChemSegment[];
  plasticAt?: number;
}

/** What this build's `chem-layout.ts` says. Written into every blob. */
export function currentLayout(): NetLayout {
  return { chem: CHEM_LEN, plastic: PLASTIC_LEN, critic: CRITIC_LEN, state: STATE_DIMS };
}

/** One body, as the encoder is handed it and the decoder hands it back. */
export interface NetBody {
  kind: AgentKind;
  x: number;
  y: number;
  heading: number;
  extra: number;
  requestDecay: number;
  energyCap: number;
  debtCap: number;
  rescueTo: number;
  assort: number;
  /** Working capital: the adenylate pool. See `Sim.advanceGait`. */
  adenylate: number;
  born: number;
  lineage: number;
  /** `layout.chem` floats. Copied, never a live view into a store. */
  chem: Float32Array;
  /** `layout.plastic` floats: the delta on the state matrices this body learned. */
  plastic: Float32Array;
  /** `layout.plastic` floats: the eligibility trace. */
  trace: Float32Array;
  /** `layout.critic` floats: weights on `h`, then the bias. */
  critic: Float64Array;
  prevValue: number;
  /** `layout.state` floats: the recurrent state. */
  h: Float64Array;
}

export interface NetWire {
  /** Index into `bodies`, not an agent id. */
  a: number;
  aSlot: PortSlot;
  b: number;
  bSlot: PortSlot;
}

export interface NetData {
  bodies: NetBody[];
  wires: NetWire[];
  /**
   * What the bodies' arrays are laid out as. Absent means this build's own,
   * which is what `captureNets` produces; `decodeNet` always fills it in from
   * the header, and `migrateNet` is how a foreign one becomes current.
   */
  layout?: NetLayout;
  segments?: ChemSegment[];
  plasticAt?: number;
}

const BYTES: Record<SectionType, number> = { u8: 1, i32: 4, f32: 4, f64: 8 };

function align8(n: number): number {
  return (n + 7) & ~7;
}

/**
 * Whether a segment map describes a genome of `chemLen` floats with a learned
 * block of `plasticLen` at `plasticAt`: sorted, contiguous, no gaps, no
 * overlaps, no duplicate names, and the learned block a run of whole segments.
 * Returns the complaint, or null.
 */
export function segmentsComplaint(
  segments: readonly ChemSegment[],
  chemLen: number,
  plasticAt: number,
  plasticLen: number,
): string | null {
  const seen = new Set<string>();
  let at = 0;
  for (const s of segments) {
    if (seen.has(s.name)) return `segment ${s.name} is listed twice`;
    seen.add(s.name);
    if (!Number.isInteger(s.at) || !Number.isInteger(s.len) || s.len <= 0) {
      return `segment ${s.name} has a bad extent (${s.at}, ${s.len})`;
    }
    if (s.at !== at) return `segment ${s.name} starts at ${s.at}, expected ${at}`;
    at += s.len;
  }
  if (at !== chemLen) return `segments cover ${at} floats of a ${chemLen}-float genome`;
  const end = plasticAt + plasticLen;
  if (!segments.some((s) => s.at === plasticAt)) return `the learned block starts inside a segment (at ${plasticAt})`;
  if (!segments.some((s) => s.at + s.len === end)) return `the learned block ends inside a segment (at ${end})`;
  return null;
}

/**
 * Whether this build's own layout is describable, which is what makes it
 * storable. Null when it is; the CLI refuses to start otherwise, and the
 * encoder refuses to write.
 */
export function layoutSelfCheck(): string | null {
  const c = segmentsComplaint(CHEM_SEGMENTS, CHEM_LEN, PLASTIC_BASE, PLASTIC_LEN);
  return c ? `this build's genome layout is not storable: ${c} (add the head to CHEM_SEGMENTS in chem-layout.ts)` : null;
}

function sameSegments(a: readonly ChemSegment[], b: readonly ChemSegment[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].name !== b[i].name || a[i].at !== b[i].at || a[i].len !== b[i].len) return false;
  }
  return true;
}

function within(s: ChemSegment, at: number, len: number): boolean {
  return s.at >= at && s.at + s.len <= at + len;
}

export type Compatibility =
  | { kind: 'exact' }
  | { kind: 'migratable'; notes: string[] }
  | { kind: 'refused'; reason: string };

/** The shape `compatibility` reads; a header is one, and so is a decoded net. */
export interface LayoutLike {
  format?: number;
  layout: NetLayout;
  segments?: readonly ChemSegment[];
  plasticAt?: number;
}

/**
 * Whether this build can plant what a header describes, and at what cost.
 *
 * `exact` is byte-for-byte this build's layout. `migratable` carries a note
 * per segment that will move, be seeded or be dropped, in the words
 * `migrateNet` will act on. `refused` names the reason, and the reason
 * always begins with the same phrase so a caller can grep for the class.
 */
export function compatibility(h: LayoutLike): Compatibility {
  if ((h.format ?? NET_FORMAT) > NET_FORMAT) {
    return { kind: 'refused', reason: `net format ${h.format} is newer than this build's ${NET_FORMAT}` };
  }
  const now = currentLayout();
  const L = h.layout;
  const keys = Object.keys(now) as (keyof NetLayout)[];
  const off = keys.filter((k) => L[k] !== now[k]);
  const segs = h.segments;
  const sameMap = segs === undefined || (sameSegments(segs, CHEM_SEGMENTS) && (h.plasticAt ?? PLASTIC_BASE) === PLASTIC_BASE);
  if (off.length === 0 && sameMap) return { kind: 'exact' };

  const changed = 'genome layout has changed since this net was stored: ' + off.map((k) => `${k} ${L[k]} -> ${now[k]}`).join(', ');
  if (L.state !== now.state || L.critic !== now.critic || L.plastic !== now.plastic) {
    return {
      kind: 'refused',
      reason: `${changed}; the recurrent state or its learned block changed width, so nothing this net learned means the same thing`,
    };
  }
  if (!segs) {
    return {
      kind: 'refused',
      reason: `${changed}; written before the genome carried a segment map (format ${h.format ?? 1}), so only an exact layout can be read`,
    };
  }
  if (h.plasticAt === undefined) {
    return { kind: 'refused', reason: 'net blob header carries a segment map but not where its learned block starts' };
  }
  const bad = segmentsComplaint(segs, L.chem, h.plasticAt, L.plastic);
  if (bad) return { kind: 'refused', reason: `net blob header is inconsistent: ${bad}` };

  const notes: string[] = [];
  const byName = new Map(segs.map((s) => [s.name, s]));
  for (const cur of CHEM_SEGMENTS) {
    const old = byName.get(cur.name);
    if (!old) {
      notes.push(`${cur.name} seeded (new in this build)`);
      continue;
    }
    if (old.len !== cur.len) {
      return {
        kind: 'refused',
        reason: `genome layout has changed since this net was stored: segment ${cur.name} was ${old.len} floats and is ${cur.len}`,
      };
    }
    const wasLearned = within(old, h.plasticAt, L.plastic);
    const isLearned = within(cur, PLASTIC_BASE, PLASTIC_LEN);
    if (wasLearned !== isLearned) {
      return {
        kind: 'refused',
        reason:
          `genome layout has changed since this net was stored: segment ${cur.name} ` +
          (wasLearned ? 'was learned and is not' : 'is learned and was not') +
          ', so its learned delta has nowhere to go',
      };
    }
    if (old.at !== cur.at) notes.push(`${cur.name} moved ${old.at} -> ${cur.at}`);
  }
  for (const old of segs) {
    if (!CHEM_SEGMENTS.some((s) => s.name === old.name)) notes.push(`${old.name} dropped (gone from this build)`);
  }
  return { kind: 'migratable', notes };
}

/** Whether a `NetData` is at this build's layout, so it can be encoded or planted as is. */
export function isCurrent(net: NetData): boolean {
  if (!net.layout) return true;
  return compatibility({ layout: net.layout, segments: net.segments, plasticAt: net.plasticAt }).kind === 'exact';
}

/**
 * A net at another build's layout, brought to this one.
 *
 * `seed` supplies a fresh genome for a kind — `seedChem` with the run's
 * params, which is exactly what a body born in this pond would carry — and
 * every carried segment is written over it. The learned delta and its trace
 * follow their segments; everything else on a body is untouched. The notes
 * are `compatibility`'s, so a caller can print what changed.
 *
 * Identity on a net that is already current, and throws on one that cannot
 * be brought across, with the reason.
 */
export function migrateNet(net: NetData, seed: (kind: AgentKind) => Float32Array): { net: NetData; notes: string[] } {
  if (isCurrent(net)) return { net, notes: [] };
  const c = compatibility({ layout: net.layout!, segments: net.segments, plasticAt: net.plasticAt });
  if (c.kind === 'refused') throw new Error(`pond: ${c.reason}`);
  if (c.kind === 'exact') return { net, notes: [] };
  const oldSegs = net.segments!;
  const oldPlasticAt = net.plasticAt!;
  const byName = new Map(oldSegs.map((s) => [s.name, s]));
  const moves: { from: number; to: number; len: number; learned: boolean }[] = [];
  for (const cur of CHEM_SEGMENTS) {
    const old = byName.get(cur.name);
    if (!old) continue;
    moves.push({ from: old.at, to: cur.at, len: cur.len, learned: within(cur, PLASTIC_BASE, PLASTIC_LEN) });
  }
  const bodies: NetBody[] = net.bodies.map((b) => {
    const chem = seed(b.kind);
    if (chem.length !== CHEM_LEN) {
      throw new Error(`pond: migrateNet's seeder returned ${chem.length} floats, not ${CHEM_LEN}`);
    }
    const plastic = new Float32Array(PLASTIC_LEN);
    const trace = new Float32Array(PLASTIC_LEN);
    for (const m of moves) {
      chem.set(b.chem.subarray(m.from, m.from + m.len), m.to);
      if (m.learned) {
        const src = m.from - oldPlasticAt;
        const dst = m.to - PLASTIC_BASE;
        plastic.set(b.plastic.subarray(src, src + m.len), dst);
        trace.set(b.trace.subarray(src, src + m.len), dst);
      }
    }
    return { ...b, chem, plastic, trace };
  });
  return {
    net: {
      bodies,
      wires: net.wires,
      layout: currentLayout(),
      segments: CHEM_SEGMENTS.map((s) => ({ ...s })),
      plasticAt: PLASTIC_BASE,
    },
    notes: c.notes,
  };
}

/**
 * Pack a net into one buffer, at this build's layout.
 *
 * Sections are laid out largest-alignment-first so every one lands on its own
 * natural boundary without padding between them, which is what lets `decodeNet`
 * hand back typed-array views straight onto the blob instead of copying it.
 *
 * Refuses a net that is at another layout — `migrateNet` first — and refuses
 * to write at all if this build's own layout is not describable, since a
 * blob without a true segment map is one the next layout change strands.
 */
export function encodeNet(net: NetData, meta: NetMeta = {}): Uint8Array {
  const self = layoutSelfCheck();
  if (self) throw new Error(`pond: ${self}`);
  if (!isCurrent(net)) {
    throw new Error("pond: encodeNet was handed a net at another build's layout; migrateNet it first");
  }
  return encodeNetAs(net, currentLayout(), CHEM_SEGMENTS, PLASTIC_BASE, meta);
}

/**
 * Pack a net as a build with the given layout would have.
 *
 * For tests of the migration and for nothing else: the bodies' arrays must
 * already be the widths `layout` says, and nothing here checks that they
 * mean anything. `encodeNet` is the one to call.
 */
export function encodeNetAs(
  net: NetData,
  L: NetLayout,
  segments: readonly ChemSegment[],
  plasticAt: number,
  meta: NetMeta = {},
): Uint8Array {
  const n = net.bodies.length;
  const w = net.wires.length;
  const plan: Omit<Section, 'offset'>[] = [
    { name: 'pose', type: 'f64', count: n * POSE_FIELDS.length, stride: POSE_FIELDS.length, fields: POSE_FIELDS },
    { name: 'scalar', type: 'f64', count: n * SCALAR_FIELDS.length, stride: SCALAR_FIELDS.length, fields: SCALAR_FIELDS },
    { name: 'critic', type: 'f64', count: n * L.critic, stride: L.critic },
    { name: 'prevValue', type: 'f64', count: n, stride: 1 },
    { name: 'h', type: 'f64', count: n * L.state, stride: L.state },
    { name: 'chem', type: 'f32', count: n * L.chem, stride: L.chem },
    { name: 'plastic', type: 'f32', count: n * L.plastic, stride: L.plastic },
    { name: 'trace', type: 'f32', count: n * L.plastic, stride: L.plastic },
    { name: 'ancestry', type: 'i32', count: n * ANCESTRY_FIELDS.length, stride: ANCESTRY_FIELDS.length, fields: ANCESTRY_FIELDS },
    { name: 'wire', type: 'i32', count: w * WIRE_FIELDS.length, stride: WIRE_FIELDS.length, fields: WIRE_FIELDS },
    { name: 'kind', type: 'u8', count: n, stride: 1, values: CODE_KIND },
  ];

  const sections: Section[] = [];
  let at = 0;
  for (const s of plan) {
    sections.push({ ...s, offset: at });
    at += s.count * BYTES[s.type];
  }
  const payloadBytes = align8(at);

  const header: NetHeader = {
    magic: MAGIC,
    format: NET_FORMAT,
    layout: { ...L },
    bodies: n,
    wires: w,
    sections,
    segments: segments.map((s) => ({ name: s.name, at: s.at, len: s.len })),
    plasticAt,
  };
  // Only what the writer actually knew: an absent key stays absent, so a net
  // encoded with no meta is the same bytes every time.
  if (meta.commit !== undefined) header.commit = meta.commit;
  if (meta.written !== undefined) header.written = meta.written;
  if (meta.source !== undefined) header.source = { ...meta.source };
  if (meta.note !== undefined) header.note = meta.note;

  const json = new TextEncoder().encode(JSON.stringify(header));
  const payloadAt = align8(4 + json.length);
  const buf = new Uint8Array(payloadAt + payloadBytes);
  new DataView(buf.buffer).setUint32(0, json.length, true);
  buf.set(json, 4);

  const view = (name: string) => {
    const s = sections.find((x) => x.name === name)!;
    const start = payloadAt + s.offset;
    if (s.type === 'u8') return new Uint8Array(buf.buffer, start, s.count);
    if (s.type === 'i32') return new Int32Array(buf.buffer, start, s.count);
    if (s.type === 'f32') return new Float32Array(buf.buffer, start, s.count);
    return new Float64Array(buf.buffer, start, s.count);
  };

  const kind = view('kind') as Uint8Array;
  const pose = view('pose') as Float64Array;
  const scalar = view('scalar') as Float64Array;
  const ancestry = view('ancestry') as Int32Array;
  const chem = view('chem') as Float32Array;
  const plastic = view('plastic') as Float32Array;
  const trace = view('trace') as Float32Array;
  const critic = view('critic') as Float64Array;
  const prevValue = view('prevValue') as Float64Array;
  const h = view('h') as Float64Array;

  for (let i = 0; i < n; i++) {
    const b = net.bodies[i];
    if (b.chem.length !== L.chem || b.plastic.length !== L.plastic || b.trace.length !== L.plastic) {
      throw new Error(`pond: body ${i} is not at the layout being written (chem ${b.chem.length} for ${L.chem})`);
    }
    kind[i] = KIND_CODE[b.kind];
    pose[i * 3] = b.x;
    pose[i * 3 + 1] = b.y;
    pose[i * 3 + 2] = b.heading;
    const so = i * SCALAR_FIELDS.length;
    for (let k = 0; k < SCALAR_FIELDS.length; k++) scalar[so + k] = b[SCALAR_FIELDS[k]];
    ancestry[i * 2] = b.born;
    ancestry[i * 2 + 1] = b.lineage;
    chem.set(b.chem, i * L.chem);
    plastic.set(b.plastic, i * L.plastic);
    trace.set(b.trace, i * L.plastic);
    critic.set(b.critic, i * L.critic);
    prevValue[i] = b.prevValue;
    h.set(b.h, i * L.state);
  }

  const wire = view('wire') as Int32Array;
  for (let i = 0; i < w; i++) {
    const e = net.wires[i];
    wire[i * 4] = e.a;
    wire[i * 4 + 1] = SLOT_CODE[e.aSlot];
    wire[i * 4 + 2] = e.b;
    wire[i * 4 + 3] = SLOT_CODE[e.bSlot];
  }
  return buf;
}

/**
 * Read the header without touching the payload.
 *
 * Cheap enough to run over every row of a query — the lab page listing a
 * hundred nets wants their sizes, not their genomes.
 */
export function readHeader(blob: Uint8Array): NetHeader {
  if (blob.byteLength < 4) throw new Error('pond: net blob is truncated');
  const len = new DataView(blob.buffer, blob.byteOffset, blob.byteLength).getUint32(0, true);
  if (len === 0 || 4 + len > blob.byteLength) throw new Error('pond: net blob header length is out of range');
  const header = JSON.parse(new TextDecoder().decode(blob.subarray(4, 4 + len))) as NetHeader;
  if (header.magic !== MAGIC) throw new Error(`pond: not a net blob (magic ${JSON.stringify(header.magic)})`);
  return header;
}

/**
 * Whether this build can read a blob at all, and why not when it cannot.
 *
 * Returns the complaint rather than throwing, so a caller listing a database
 * written by an older build can show which rows it cannot open instead of
 * dying on the first one. Null for anything `decodeNet` will accept, which
 * since format 2 includes blobs that need `migrateNet`; `compatibility` is
 * the finer reading.
 */
export function layoutComplaint(header: NetHeader): string | null {
  const c = compatibility(header);
  return c.kind === 'refused' ? c.reason : null;
}

/**
 * Unpack a blob, at the layout it was written in.
 *
 * The returned genome and matrix views alias it, so a caller that plants a
 * net and then keeps the arrays is looking at the blob, not at the pond. The
 * result says what layout it is at; `isCurrent` tells whether it can be used
 * as is, and `migrateNet` (which `plantNet` calls) brings it across when it
 * cannot. Throws only on what cannot be brought across, with the reason.
 *
 * A blob whose start is not eight-byte aligned is copied first: `sqlite3`
 * hands back a view into a shared read buffer at whatever offset the row
 * happened to land on, and a `Float64Array` cannot be built over an odd one.
 */
export function decodeNet(input: Uint8Array): NetData {
  const blob = input.byteOffset % 8 === 0 ? input : new Uint8Array(input);
  const header = readHeader(blob);
  const c = compatibility(header);
  if (c.kind === 'refused') throw new Error(`pond: ${c.reason}`);

  const len = new DataView(blob.buffer, blob.byteOffset, blob.byteLength).getUint32(0, true);
  const payloadAt = blob.byteOffset + align8(4 + len);
  const byName = new Map(header.sections.map((s) => [s.name, s]));
  const need = (name: string): Section => {
    const s = byName.get(name);
    if (!s) throw new Error(`pond: net blob has no ${name} section`);
    return s;
  };
  const f64 = (name: string) => {
    const s = need(name);
    return new Float64Array(blob.buffer, payloadAt + s.offset, s.count);
  };
  const f32 = (name: string) => {
    const s = need(name);
    return new Float32Array(blob.buffer, payloadAt + s.offset, s.count);
  };
  const i32 = (name: string) => {
    const s = need(name);
    return new Int32Array(blob.buffer, payloadAt + s.offset, s.count);
  };

  const L = header.layout;
  const kindS = need('kind');
  const kind = new Uint8Array(blob.buffer, payloadAt + kindS.offset, kindS.count);
  const pose = f64('pose');
  const scalar = f64('scalar');
  const critic = f64('critic');
  const prevValue = f64('prevValue');
  const h = f64('h');
  const chem = f32('chem');
  const plastic = f32('plastic');
  const trace = f32('trace');
  const ancestry = i32('ancestry');
  const wire = i32('wire');

  const bodies: NetBody[] = [];
  for (let i = 0; i < header.bodies; i++) {
    const so = i * SCALAR_FIELDS.length;
    bodies.push({
      kind: CODE_KIND[kind[i]],
      x: pose[i * 3],
      y: pose[i * 3 + 1],
      heading: pose[i * 3 + 2],
      extra: scalar[so],
      requestDecay: scalar[so + 1],
      energyCap: scalar[so + 2],
      debtCap: scalar[so + 3],
      rescueTo: scalar[so + 4],
      assort: scalar[so + 5],
      adenylate: scalar[so + 6],
      born: ancestry[i * 2],
      lineage: ancestry[i * 2 + 1],
      chem: chem.subarray(i * L.chem, (i + 1) * L.chem),
      plastic: plastic.subarray(i * L.plastic, (i + 1) * L.plastic),
      trace: trace.subarray(i * L.plastic, (i + 1) * L.plastic),
      critic: critic.subarray(i * L.critic, (i + 1) * L.critic),
      prevValue: prevValue[i],
      h: h.subarray(i * L.state, (i + 1) * L.state),
    });
  }

  const wires: NetWire[] = [];
  for (let i = 0; i < header.wires; i++) {
    wires.push({
      a: wire[i * 4],
      aSlot: CODE_SLOT[wire[i * 4 + 1]],
      b: wire[i * 4 + 2],
      bSlot: CODE_SLOT[wire[i * 4 + 3]],
    });
  }
  return {
    bodies,
    wires,
    layout: { ...L },
    segments: header.segments?.map((s) => ({ ...s })),
    plasticAt: header.plasticAt,
  };
}
