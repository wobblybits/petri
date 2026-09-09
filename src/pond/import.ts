import { DatabaseSync } from 'node:sqlite';
import { basename } from 'node:path';
import { PondDb, SCHEMA_VERSION } from './db.ts';

/*
 * Fold one library file into another.
 *
 * The runs from a single session ended up in thirty-eight separate database
 * files, one per question, which is the habit the format was built to break:
 * a run in a file of its own informs the sweep it belonged to and nothing
 * else, while a run in the library joins every regression asked of it
 * afterwards. `explore` reads the whole library, so this is what makes the
 * history worth anything.
 *
 * Two things it has to survive, both of which the real files have:
 *
 * - **Schema drift.** A file written before `net_fst` existed has no such
 *   column, and a file written before `origin` did has no such column either.
 *   Columns are intersected per table rather than assumed, and the source is
 *   opened read-only so importing never migrates — or damages — the thing it
 *   is reading.
 * - **Re-import.** Every run remembers where it came from as `file#id`, and a
 *   run already carrying that origin is skipped. Importing the same file
 *   twice is a no-op, which means this can be run over a directory whenever,
 *   without keeping track of what has already been folded in.
 */

export interface ImportResult {
  file: string;
  runs: number;
  skipped: number;
  nets: number;
  samples: number;
  plants: number;
}

function columns(db: DatabaseSync, table: string): string[] {
  return (db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[]).map((c) => c.name);
}

/** Columns both sides have, in the destination's order, minus the ones given. */
function shared(src: string[], dst: string[], drop: string[]): string[] {
  const have = new Set(src);
  return dst.filter((c) => have.has(c) && !drop.includes(c));
}

export function importRuns(dest: PondDb, path: string): ImportResult {
  const src = new DatabaseSync(path, { readOnly: true });
  const tag = basename(path).replace(/\.db$/, '');
  const out: ImportResult = { file: tag, runs: 0, skipped: 0, nets: 0, samples: 0, plants: 0 };
  try {
    const version = (src.prepare("SELECT value FROM meta WHERE key = 'schema_version'").get() as
      | { value: string }
      | undefined)?.value;
    if (version !== undefined && Number(version) > SCHEMA_VERSION) {
      throw new Error(`pond: ${path} is schema version ${version}, newer than this build's ${SCHEMA_VERSION}`);
    }
    const runCols = shared(columns(src, 'run'), columns(dest.db, 'run'), ['id', 'origin']);
    const netCols = shared(columns(src, 'net'), columns(dest.db, 'net'), ['id']);
    const sampleCols = shared(columns(src, 'sample'), columns(dest.db, 'sample'), []);
    const plantCols = shared(columns(src, 'plant'), columns(dest.db, 'plant'), []);

    const already = new Set(
      (dest.db.prepare('SELECT origin FROM run WHERE origin IS NOT NULL').all() as { origin: string }[])
        .map((r) => r.origin),
    );
    const insertRun = dest.db.prepare(
      `INSERT INTO run (${runCols.join(', ')}, origin) VALUES (${runCols.map(() => '?').join(',')}, ?)`,
    );
    const insertNet = dest.db.prepare(
      `INSERT INTO net (${netCols.join(', ')}) VALUES (${netCols.map(() => '?').join(',')})`,
    );
    const insertSample = dest.db.prepare(
      `INSERT OR REPLACE INTO sample (${sampleCols.join(', ')}) VALUES (${sampleCols.map(() => '?').join(',')})`,
    );
    const insertPlant = dest.db.prepare(
      `INSERT OR REPLACE INTO plant (${plantCols.join(', ')}) VALUES (${plantCols.map(() => '?').join(',')})`,
    );
    const lastId = dest.db.prepare('SELECT last_insert_rowid() AS id');
    const newId = (): number => Number((lastId.get() as { id: number }).id);

    /*
     * One transaction for the file. A half-imported library is worse than an
     * un-imported one: the origins of the runs that landed would make a
     * second attempt skip them and leave their samples behind forever.
     */
    dest.db.exec('BEGIN');
    try {
      const runMap = new Map<number, number>();
      /*
       * By id, so a run whose `parent_run` is in the same file meets its
       * parent already mapped. A forward reference would have to be a cycle,
       * which the runner cannot write.
       */
      for (const r of src.prepare('SELECT * FROM run ORDER BY id').all() as Record<string, unknown>[]) {
        const origin = `${tag}#${r.id}`;
        if (already.has(origin)) {
          out.skipped++;
          continue;
        }
        const parent = r.parent_run === null || r.parent_run === undefined
          ? null
          : runMap.get(Number(r.parent_run)) ?? null;
        insertRun.run(
          ...runCols.map((c) => (c === 'parent_run' ? parent : (r[c] as never))),
          origin,
        );
        runMap.set(Number(r.id), newId());
        out.runs++;
      }
      /*
       * Nothing new in this file. Fall through to the commit rather than
       * returning: an early return here left the transaction open and the
       * next file's BEGIN failed with "cannot start a transaction within a
       * transaction", which is how importing a directory died on its second
       * already-imported file.
       */
      const netMap = new Map<number, number>();
      const netRows = runMap.size === 0
        ? []
        : (src.prepare('SELECT * FROM net ORDER BY id').all() as Record<string, unknown>[]);
      for (const n of netRows) {
        const run = runMap.get(Number(n.run_id));
        if (run === undefined) continue;
        const parent = n.parent_net === null || n.parent_net === undefined
          ? null
          : netMap.get(Number(n.parent_net)) ?? null;
        insertNet.run(
          ...netCols.map((c) =>
            c === 'run_id' ? run : c === 'parent_net' ? parent : (n[c] as never),
          ),
        );
        netMap.set(Number(n.id), newId());
        out.nets++;
      }
      const sampleRows = runMap.size === 0
        ? []
        : (src.prepare('SELECT * FROM sample').all() as Record<string, unknown>[]);
      for (const s of sampleRows) {
        const run = runMap.get(Number(s.run_id));
        if (run === undefined) continue;
        insertSample.run(...sampleCols.map((c) => (c === 'run_id' ? run : (s[c] as never))));
        out.samples++;
      }
      const plantRows = runMap.size === 0
        ? []
        : (src.prepare('SELECT * FROM plant').all() as Record<string, unknown>[]);
      for (const p of plantRows) {
        const run = runMap.get(Number(p.run_id));
        const net = netMap.get(Number(p.net_id));
        // A plant whose net came from a file not yet imported has nothing to
        // point at; the run it seeded is still worth keeping.
        if (run === undefined || net === undefined) continue;
        insertPlant.run(
          ...plantCols.map((c) => (c === 'run_id' ? run : c === 'net_id' ? net : (p[c] as never))),
        );
        out.plants++;
      }
      dest.db.exec('COMMIT');
    } catch (e) {
      dest.db.exec('ROLLBACK');
      throw e;
    }
  } finally {
    src.close();
  }
  return out;
}
