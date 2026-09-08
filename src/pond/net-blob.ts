import { CODE_KIND, KIND_CODE } from '../agent-store.ts';
import { CHEM_LEN, CRITIC_LEN, PLASTIC_LEN, STATE_DIMS } from '../chem-layout.ts';
import type { AgentKind, PortSlot } from '../agents.ts';

/*
 * One net, on disk: the topology, and every number each of its bodies carries
 * that is not recomputable from those two things.
 *
 * The shape of the problem is that a net is mostly typed arrays — a genome is
 * 134 floats and a learned delta is 64 more, times however many bodies — and
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
 * that still runs. `decodeNet` compares the header's dimensions against the
 * ones this build was compiled with and refuses the blob if they differ,
 * naming both. A stored pond is a long-lived artifact and the layout is not.
 *
 * It also means a reader does not have to be this module. The lab page can
 * walk `sections` with a `DataView` and render `energyCap` by name without
 * importing anything from the simulation.
 *
 * ## What is not in here
 *
 * Agent ids: they are handed out per pond and mean nothing in another one, so
 * wires address bodies by their index in this blob. Mass, scale and alpha:
 * derived from kind and params at birth. Velocity, drive, stun, trail, the
 * rope node positions: a planted net is a net dropped into different water,
 * and none of that survives the move in any meaningful sense.
 *
 * `trace` and `h` *are* in here, though neither is inherited and both are
 * gone within a second of simulated time. They cost 68 floats a body and
 * they are the difference between reloading a net and reloading the net that
 * was actually running — worth having for the one case where a run is
 * continued immediately rather than crossed into a new pond.
 */

/** Bumped when the payload's meaning changes in a way a reader must notice. */
export const NET_FORMAT = 1;

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

export interface NetHeader {
  magic: string;
  format: number;
  layout: NetLayout;
  bodies: number;
  wires: number;
  sections: Section[];
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
  born: number;
  lineage: number;
  /** `CHEM_LEN` floats. Copied, never a live view into a store. */
  chem: Float32Array;
  /** `PLASTIC_LEN` floats: the delta on the state matrices this body learned. */
  plastic: Float32Array;
  /** `PLASTIC_LEN` floats: the eligibility trace. */
  trace: Float32Array;
  /** `CRITIC_LEN` floats: weights on `h`, then the bias. */
  critic: Float64Array;
  prevValue: number;
  /** `STATE_DIMS` floats: the recurrent state. */
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
}

const BYTES: Record<SectionType, number> = { u8: 1, i32: 4, f32: 4, f64: 8 };

function align8(n: number): number {
  return (n + 7) & ~7;
}

/**
 * Pack a net into one buffer.
 *
 * Sections are laid out largest-alignment-first so every one lands on its own
 * natural boundary without padding between them, which is what lets `decodeNet`
 * hand back typed-array views straight onto the blob instead of copying it.
 */
export function encodeNet(net: NetData): Uint8Array {
  const n = net.bodies.length;
  const w = net.wires.length;
  const L = currentLayout();
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
    layout: L,
    bodies: n,
    wires: w,
    sections,
  };
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
 * dying on the first one.
 */
export function layoutComplaint(header: NetHeader): string | null {
  if (header.format > NET_FORMAT) {
    return `net format ${header.format} is newer than this build's ${NET_FORMAT}`;
  }
  const now = currentLayout();
  const keys = Object.keys(now) as (keyof NetLayout)[];
  const off = keys.filter((k) => header.layout[k] !== now[k]);
  if (off.length === 0) return null;
  return (
    'genome layout has changed since this net was stored: ' +
    off.map((k) => `${k} ${header.layout[k]} -> ${now[k]}`).join(', ')
  );
}

/**
 * Unpack a blob. The returned genome and matrix views alias it, so a caller
 * that plants a net and then keeps the arrays is looking at the blob, not at
 * the pond.
 *
 * A blob whose start is not eight-byte aligned is copied first: `sqlite3`
 * hands back a view into a shared read buffer at whatever offset the row
 * happened to land on, and a `Float64Array` cannot be built over an odd one.
 */
export function decodeNet(input: Uint8Array): NetData {
  const blob = input.byteOffset % 8 === 0 ? input : new Uint8Array(input);
  const header = readHeader(blob);
  const complaint = layoutComplaint(header);
  if (complaint) throw new Error(`pond: ${complaint}`);

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
  return { bodies, wires };
}
