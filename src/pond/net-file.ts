import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Params } from '../params.ts';
import { prepareNet } from './capture.ts';
import {
  compatibility,
  decodeNet,
  encodeNet,
  readHeader,
  type Compatibility,
  type NetData,
  type NetHeader,
  type NetMeta,
} from './net-blob.ts';

/*
 * Nets as files.
 *
 * The database is where a run leaves what it grew, and it is gitignored:
 * megabytes of evolved genome that belong to the machine that grew them. A
 * file is the same blob outside it — something a test can read from the
 * repository, a page can fetch, and a person can hand to someone else. The
 * header carries everything the row did that matters for reading it back:
 * the layout it was written at, the commit, the time, and which run and net
 * it came from.
 *
 * `nets/` at the repository root is where checked-in ones live. A test that
 * wants a grown net rather than a founder soup starts from one of these, by
 * name, and `loadFixture` brings it to this build's layout on the way in —
 * so a fixture written before a head was appended keeps working, and one
 * written before the state changed width says so instead of silently
 * misreading.
 */

export const NET_FILE_EXT = '.petrinet';

/** `<repo>/nets/`, wherever this module is loaded from. */
export const NETS_DIR = fileURLToPath(new URL('../../nets/', import.meta.url));

export interface LoadedNet {
  /** At this build's layout. */
  net: NetData;
  header: NetHeader;
  /** What the migration changed; empty when the file was already current. */
  notes: string[];
  path: string;
}

export function writeNetFile(path: string, blob: Uint8Array): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, blob);
}

/** Encode at this build's layout, stamp the time, and write. Returns the bytes written. */
export function saveNet(path: string, net: NetData, meta: NetMeta = {}): Uint8Array {
  const blob = encodeNet(net, { written: new Date().toISOString(), ...meta });
  writeNetFile(path, blob);
  return blob;
}

export function readNetBlob(path: string): Uint8Array {
  const b = readFileSync(path);
  return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
}

/** The header and whether this build can plant it, without decoding the payload. */
export function inspectNetFile(path: string): { header: NetHeader; compatibility: Compatibility } {
  const header = readHeader(readNetBlob(path));
  return { header, compatibility: compatibility(header) };
}

/**
 * A file, decoded and brought to this build's layout under `params`.
 * Throws with the reason when it cannot be.
 */
export function loadNet(path: string, params: Params): LoadedNet {
  const blob = readNetBlob(path);
  const header = readHeader(blob);
  const { net, notes } = prepareNet(decodeNet(blob), params);
  return { net, header, notes, path };
}

/** `nets/<name>.petrinet`; the extension may be given or not. */
export function fixturePath(name: string): string {
  return resolve(NETS_DIR, name.endsWith(NET_FILE_EXT) ? name : name + NET_FILE_EXT);
}

/** The checked-in nets, by name. */
export function listFixtures(): string[] {
  if (!existsSync(NETS_DIR)) return [];
  return readdirSync(NETS_DIR)
    .filter((f) => f.endsWith(NET_FILE_EXT))
    .map((f) => basename(f, NET_FILE_EXT))
    .sort();
}

/** A checked-in net, at this build's layout. */
export function loadFixture(name: string, params: Params): LoadedNet {
  const path = fixturePath(name);
  if (!existsSync(path)) {
    const have = listFixtures();
    throw new Error(`pond: no fixture ${name} in ${NETS_DIR}` + (have.length ? ` (have: ${have.join(', ')})` : ''));
  }
  return loadNet(path, params);
}
