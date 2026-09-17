import type { Params } from './params.ts';
import type { Sim } from './sim.ts';
import { plantNet, prepareNet } from './pond/capture.ts';
import { compatibility, decodeNet, readHeader, type NetHeader } from './pond/net-blob.ts';

/*
 * A stored net, into the page.
 *
 * The runner grows nets and the database keeps them; this is how one gets
 * onto the screen, which is where a grown net is judged. Three ways in, all
 * ending here: a `.petrinet` file dropped on the dish, one picked with the
 * button, or one named in the URL (`?net=mixed-308` fetches
 * `nets/mixed-308.petrinet`, which Vite serves from the repository root in
 * dev). The bytes are the same blob the database holds, so a net written at
 * an older genome layout is brought across on the way in, exactly as
 * `plantNet` does for the runner, and the notes say what changed.
 *
 * Browser-safe on purpose: nothing here touches the filesystem, and the
 * decoder and planter it uses do not either.
 */

interface PlantedNet {
  /** The new agent ids, in blob order. Empty means it did not fit. */
  ids: number[];
  header: NetHeader;
  /** What the migration changed; empty when the blob was already current. */
  notes: string[];
}

let planted = 0;

/**
 * Plant a blob's net centred on `(x, y)`.
 *
 * Throws, with the reason, on a blob this build cannot read; returns empty
 * `ids` when the whole net would not fit under `maxAgents`, the same all-or-
 * nothing rule the runner uses. Each net planted this way gets its own
 * negative founder line, so the census counts it as one arrival.
 */
export function plantNetBytes(sim: Sim, params: Params, bytes: ArrayBuffer | Uint8Array, x: number, y: number): PlantedNet {
  const blob = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const header = readHeader(blob);
  const c = compatibility(header);
  if (c.kind === 'refused') throw new Error(c.reason);
  const { net, notes } = prepareNet(decodeNet(blob), params);
  planted += 1;
  const ids = plantNet(sim, params, net, x, y, { lineage: -planted });
  return { ids, header, notes };
}

/** Fetch a `.petrinet` by URL. Throws on anything but a 2xx. */
export async function fetchNet(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return new Uint8Array(await res.arrayBuffer());
}

/**
 * Where `?net=` points. A bare name is a file under `nets/`; anything with a
 * slash or the extension is taken as given.
 */
export function netUrlFromQuery(search: string): string | null {
  const q = new URLSearchParams(search).get('net');
  if (!q) return null;
  if (q.includes('/') || q.endsWith('.petrinet')) return q;
  return `nets/${q}.petrinet`;
}

/** One line for the panel: what went in, and what the migration changed. */
export function describePlant(name: string, p: PlantedNet): string {
  if (p.ids.length === 0) return `${name}: ${p.header.bodies} bodies would not fit under maxAgents`;
  const from = p.header.commit ? ` (grown at ${p.header.commit})` : '';
  const notes = p.notes.length ? ` — migrated: ${p.notes.join('; ')}` : '';
  return `${name}: ${p.ids.length} bodies, ${p.header.wires} wires${from}${notes}`;
}
