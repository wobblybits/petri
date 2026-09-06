export const GALLERY_KEY = 'swimmers.design.gallery.v1';

export interface GalleryPiece {
  id: string;
  name: string;
  text: string;
  savedAt: number;
}

export interface GalleryStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const NAME_MAX = 28;

/** Compact one-line label from an HVM2 net. */
export function pieceNameFromNet(text: string): string {
  const compact = text.replace(/\s+/g, ' ').trim();
  if (!compact) return 'piece';
  if (compact.length <= NAME_MAX) return compact;
  return `${compact.slice(0, NAME_MAX - 1)}…`;
}

function newId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

function isPiece(value: unknown): value is GalleryPiece {
  if (!value || typeof value !== 'object') return false;
  const p = value as GalleryPiece;
  return (
    typeof p.id === 'string' &&
    typeof p.name === 'string' &&
    typeof p.text === 'string' &&
    typeof p.savedAt === 'number'
  );
}

export function parseGallery(raw: string | null): GalleryPiece[] {
  if (!raw) return [];
  try {
    const data = JSON.parse(raw) as unknown;
    if (!Array.isArray(data)) return [];
    return data.filter(isPiece);
  } catch {
    return [];
  }
}

export function serializeGallery(pieces: GalleryPiece[]): string {
  return JSON.stringify(pieces);
}

export function loadGallery(storage: GalleryStore = localStorage): GalleryPiece[] {
  try {
    return parseGallery(storage.getItem(GALLERY_KEY));
  } catch {
    return [];
  }
}

export function storeGallery(pieces: GalleryPiece[], storage: GalleryStore = localStorage): void {
  storage.setItem(GALLERY_KEY, serializeGallery(pieces));
}

/** Newest first. Skips empty nets. */
export function addPiece(
  pieces: GalleryPiece[],
  text: string,
  name?: string,
  now = Date.now(),
): GalleryPiece[] {
  const trimmed = text.trim();
  if (!trimmed) return pieces;
  const piece: GalleryPiece = {
    id: newId(),
    name: (name ?? pieceNameFromNet(trimmed)).trim() || pieceNameFromNet(trimmed),
    text: trimmed,
    savedAt: now,
  };
  return [piece, ...pieces];
}

export function removePiece(pieces: GalleryPiece[], id: string): GalleryPiece[] {
  return pieces.filter((p) => p.id !== id);
}

export function renamePiece(pieces: GalleryPiece[], id: string, name: string): GalleryPiece[] {
  const next = name.trim();
  if (!next) return pieces;
  return pieces.map((p) => (p.id === id ? { ...p, name: next } : p));
}
