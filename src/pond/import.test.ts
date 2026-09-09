import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { defaultParams } from '../params.ts';
import { PondDb } from './db.ts';
import { importRuns } from './import.ts';

/*
 * The importer's whole job is to survive files it did not write: older
 * schemas, runs that reference each other, and being pointed at the same file
 * twice. Each of those is here, and each of them is something the real
 * library actually had.
 */

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'pond-import-'));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function seed(path: string, count: number, opts: { parents?: boolean } = {}): number[] {
  const db = new PondDb(path);
  const ids: number[] = [];
  try {
    for (let i = 0; i < count; i++) {
      const params = defaultParams();
      params.excreteRate = 0.01 * i;
      const id = db.startRun({
        seed: i,
        seconds: 60,
        dt: 1 / 60,
        world: { w: 1600, h: 1200 },
        fieldCells: 256,
        preset: 'soup',
        soupCount: 100,
        parentRun: opts.parents && ids.length > 0 ? ids[ids.length - 1] : null,
        params,
        commit: null,
        note: `run ${i}`,
        sweep: 'x',
        point: { excreteRate: 0.01 * i },
      });
      db.addSample(id, {
        t: 10, bodies: 100 + i, wires: 50, lines: 3, bornMean: 1, bornMax: 2, spawned: 0,
        born: 1, died: 0, commutes: 5, erases: 0, annihilations: 0, latches: 9, snaps: 0,
        free: 1, ground: 2, escrow: 0, meanExtra: 1, canPay: 1, ppWires: 1, conDupWires: 1,
        commuteShare: null, commuteChance: 0.2, commuteEdge: null, matrixDrift: 0,
        diversity: { netFst: 0.1 },
      });
      db.finishRun(id, 600, 100, {});
      ids.push(id);
    }
  } finally {
    db.close();
  }
  return ids;
}

describe('importing one library into another', () => {
  it('copies runs and their samples, and remembers where they came from', () => {
    const from = join(dir, 'old.db');
    seed(from, 3);
    const db = new PondDb(join(dir, 'lib.db'));
    try {
      const r = importRuns(db, from);
      expect(r).toMatchObject({ file: 'old', runs: 3, skipped: 0, samples: 3 });
      const rows = db.runs();
      expect(rows).toHaveLength(3);
      expect(rows.map((x) => x.origin).sort()).toEqual(['old#1', 'old#2', 'old#3']);
      expect(rows.map((x) => x.note).sort()).toEqual(['run 0', 'run 1', 'run 2']);
      // The samples came with them, attached to the new ids.
      for (const row of rows) expect(db.samples(row.id)).toHaveLength(1);
    } finally {
      db.close();
    }
  });

  it('is a no-op the second time, so a directory can be re-imported at will', () => {
    const from = join(dir, 'old.db');
    seed(from, 3);
    const db = new PondDb(join(dir, 'lib.db'));
    try {
      importRuns(db, from);
      const again = importRuns(db, from);
      expect(again).toMatchObject({ runs: 0, skipped: 3, samples: 0 });
      expect(db.runs()).toHaveLength(3);
    } finally {
      db.close();
    }
  });

  it('keeps two files apart even where their run ids collide', () => {
    seed(join(dir, 'a.db'), 2);
    seed(join(dir, 'b.db'), 2);
    const db = new PondDb(join(dir, 'lib.db'));
    try {
      importRuns(db, join(dir, 'a.db'));
      importRuns(db, join(dir, 'b.db'));
      expect(db.runs()).toHaveLength(4);
      expect(db.runs().map((r) => r.origin).sort()).toEqual(['a#1', 'a#2', 'b#1', 'b#2']);
    } finally {
      db.close();
    }
  });

  it('rewrites a run that points at another run in the same file', () => {
    const from = join(dir, 'chain.db');
    seed(from, 3, { parents: true });
    const db = new PondDb(join(dir, 'lib.db'));
    try {
      importRuns(db, from);
      const rows = db.runs().sort((a, b) => a.id - b.id);
      // The chain survives, pointing at the NEW ids, not the old ones.
      expect(rows[0].parent_run).toBeNull();
      expect(rows[1].parent_run).toBe(rows[0].id);
      expect(rows[2].parent_run).toBe(rows[1].id);
    } finally {
      db.close();
    }
  });

  it('reads a file that predates a column, and does not write to it', () => {
    const from = join(dir, 'old.db');
    seed(from, 2);
    // Take a column away, the way a file written before `net_fst` existed
    // would have been. SQLite can drop it; the importer must not care.
    const raw = new DatabaseSync(from);
    raw.exec('ALTER TABLE sample DROP COLUMN net_fst');
    raw.exec('ALTER TABLE run DROP COLUMN point');
    raw.exec('DROP INDEX run_by_origin');
    raw.exec('ALTER TABLE run DROP COLUMN origin');
    raw.close();

    const db = new PondDb(join(dir, 'lib.db'));
    try {
      expect(importRuns(db, from)).toMatchObject({ runs: 2, samples: 2 });
      const id = db.runs()[0].id;
      expect(db.samples(id)[0].net_fst).toBeNull();
      expect(db.runs()[0].point).toBeNull();
    } finally {
      db.close();
    }
    // Read-only really is read-only: the source never gained the columns the
    // destination has, so importing cannot damage what it reads.
    const after = new DatabaseSync(from, { readOnly: true });
    const cols = (after.prepare('PRAGMA table_info(run)').all() as { name: string }[]).map((c) => c.name);
    after.close();
    expect(cols).not.toContain('origin');
    expect(cols).not.toContain('point');
  });

  it('leaves the library untouched when a file cannot be read whole', () => {
    const from = join(dir, 'bad.db');
    seed(from, 3);
    // A source too old for a column the destination requires. Drift in a
    // nullable column is survivable, above; drift in a NOT NULL one is not,
    // and what matters is that it takes nothing with it.
    const raw = new DatabaseSync(from);
    raw.exec('ALTER TABLE sample DROP COLUMN snaps');
    raw.close();

    const db = new PondDb(join(dir, 'lib.db'));
    try {
      expect(() => importRuns(db, from)).toThrow(/snaps/);
      // Not one of the three runs landed: a file is all or nothing. A
      // half-import would be worse than none, because the origins of the runs
      // that made it would make the next attempt skip them and strand their
      // samples for good.
      expect(db.runs()).toHaveLength(0);
    } finally {
      db.close();
    }
  });
});
