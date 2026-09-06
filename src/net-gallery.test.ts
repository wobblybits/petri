import { describe, expect, it } from 'vitest';
import {
  addPiece,
  loadGallery,
  parseGallery,
  pieceNameFromNet,
  removePiece,
  renamePiece,
  storeGallery,
  type GalleryStore,
} from './net-gallery.ts';

function memory(): GalleryStore & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => {
      data.set(key, value);
    },
  };
}

describe('pieceNameFromNet', () => {
  it('compacts whitespace and truncates long nets', () => {
    expect(pieceNameFromNet('& (* *) ~ {* *}')).toBe('& (* *) ~ {* *}');
    expect(pieceNameFromNet('  (a   b)  ')).toBe('(a b)');
    expect(pieceNameFromNet('')).toBe('piece');
    const long = '& ((a (b (c d))) (e f)) ~ {g h}';
    const name = pieceNameFromNet(long);
    expect(name.endsWith('…')).toBe(true);
    expect(name.length).toBeLessThanOrEqual(28);
  });
});

describe('gallery store', () => {
  it('ignores junk and empty storage', () => {
    expect(parseGallery(null)).toEqual([]);
    expect(parseGallery('not json')).toEqual([]);
    expect(parseGallery('{"x":1}')).toEqual([]);
  });

  it('prepends a piece and skips empty nets', () => {
    const first = addPiece([], '& (* *) ~ {* *}', 'commute', 10);
    expect(first).toHaveLength(1);
    expect(first[0]!.name).toBe('commute');
    expect(first[0]!.text).toBe('& (* *) ~ {* *}');
    expect(first[0]!.savedAt).toBe(10);
    const two = addPiece(first, '*', undefined, 20);
    expect(two[0]!.name).toBe('*');
    expect(two[1]!.name).toBe('commute');
    expect(addPiece(two, '  \n  ')).toHaveLength(2);
  });

  it('renames and removes by id', () => {
    const pieces = addPiece([], '*', 'era', 1);
    const id = pieces[0]!.id;
    const renamed = renamePiece(pieces, id, '  eraser  ');
    expect(renamed[0]!.name).toBe('eraser');
    expect(renamePiece(renamed, id, '   ')).toEqual(renamed);
    expect(removePiece(renamed, id)).toEqual([]);
  });

  it('round-trips through storage', () => {
    const storage = memory();
    const pieces = addPiece([], '(a b)', 'con');
    storeGallery(pieces, storage);
    expect(loadGallery(storage)).toEqual(pieces);
  });
});
