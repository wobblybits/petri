import { DatabaseSync, type StatementSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { CHEM_LEN } from '../chem-layout.ts';
import type { Params } from '../params.ts';
import { NET_FORMAT, decodeNet, readHeader, type NetData, type NetHeader } from './net-blob.ts';
import type { CapturedNet, NetStats } from './capture.ts';

/*
 * The pond library: one SQLite file holding runs, the nets they produced, and
 * the timeline each run was sampled into.
 *
 * SQLite because `node:sqlite` is in Node itself — this project has no runtime
 * dependencies and a store for evolved genomes is not the place to acquire the
 * first one — and because the questions worth asking of a library of ponds are
 * queries. "The twenty deepest nets across every run since Tuesday" is one
 * line of SQL and no line of JavaScript.
 *
 * Everything heavy is one BLOB per net (see `net-blob.ts`); everything a query
 * might filter or sort on is a column beside it. That split is what lets the
 * lab page list a hundred nets without reading a hundred genomes.
 *
 * ## Reading this from the browser
 *
 * Nothing in this module is importable from a page — `node:sqlite` is not. The
 * format is split so it does not have to be: `net-blob.ts` is pure and
 * portable, and the lab page gets its rows either through a dev-server
 * endpoint that runs this module, or by fetching the `.db` file and opening it
 * with a wasm SQLite build. Either way the blob it decodes is the same one,
 * with the same reader.
 */

/** Bumped when a migration is needed; stored in `meta`. */
export const SCHEMA_VERSION = 2;

const SCHEMA = `
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- One row per invocation of the runner.
CREATE TABLE IF NOT EXISTS run (
  id          INTEGER PRIMARY KEY,
  started_at  TEXT    NOT NULL,           -- ISO 8601, UTC
  finished_at TEXT,                       -- NULL while running, or if it died
  seed        INTEGER NOT NULL,
  seconds     REAL    NOT NULL,           -- simulated seconds asked for
  dt          REAL    NOT NULL,
  frames      INTEGER NOT NULL DEFAULT 0,
  wall_ms     REAL,
  world_w     REAL    NOT NULL,
  world_h     REAL    NOT NULL,
  field_cells INTEGER NOT NULL,
  preset      TEXT,                       -- NULL when the pond was seeded from stored nets
  soup_count  INTEGER NOT NULL,
  parent_run  INTEGER REFERENCES run(id),
  params      TEXT    NOT NULL,           -- JSON: the whole Params, not a diff
  commit_hash TEXT,                       -- the working tree this ran from
  net_format  INTEGER NOT NULL,
  chem_len    INTEGER NOT NULL,           -- the genome width these blobs were written at
  note        TEXT,
  census      TEXT,                       -- JSON: the closing census and tally
  sweep       TEXT,                       -- name of the sweep this run belongs to
  point       TEXT                        -- JSON: the grid point, for grouping
);

-- One row per connected component saved out of a run.
CREATE TABLE IF NOT EXISTS net (
  id             INTEGER PRIMARY KEY,
  run_id         INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  at             TEXT    NOT NULL,        -- wallclock of the harvest
  t              REAL    NOT NULL,        -- simulated seconds into the run
  frame          INTEGER NOT NULL,
  bodies         INTEGER NOT NULL,
  wires          INTEGER NOT NULL,
  era            INTEGER NOT NULL,
  dup            INTEGER NOT NULL,
  con            INTEGER NOT NULL,
  pp_wires       INTEGER NOT NULL,
  con_dup_wires  INTEGER NOT NULL,
  born_max       INTEGER NOT NULL,
  born_mean      REAL    NOT NULL,
  lines          INTEGER NOT NULL,
  dominant       INTEGER NOT NULL,
  dominant_share REAL    NOT NULL,
  energy         REAL    NOT NULL,
  learned        REAL    NOT NULL,
  plastic_mean   REAL    NOT NULL,
  matrix_drift   REAL    NOT NULL,
  parent_net     INTEGER REFERENCES net(id),  -- the planted net this one descends from
  text           TEXT    NOT NULL,        -- HVM2 net IR; topology only
  blob           BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS net_by_run  ON net(run_id, t);
CREATE INDEX IF NOT EXISTS net_by_size ON net(bodies DESC);
CREATE INDEX IF NOT EXISTS net_by_depth ON net(born_max DESC);

-- Which stored net was planted into which run, and under what founder line.
CREATE TABLE IF NOT EXISTS plant (
  run_id  INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  lineage INTEGER NOT NULL,               -- negative; see PlantOptions.lineage
  net_id  INTEGER NOT NULL REFERENCES net(id),
  bodies  INTEGER NOT NULL,
  PRIMARY KEY (run_id, lineage)
);

-- The timeline. One row per sample point, mirroring the experiment harness.
CREATE TABLE IF NOT EXISTS sample (
  run_id         INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  t              REAL    NOT NULL,
  bodies         INTEGER NOT NULL,
  wires          INTEGER NOT NULL,
  lines          INTEGER NOT NULL,
  born_mean      REAL    NOT NULL,
  born_max       INTEGER NOT NULL,
  spawned        INTEGER NOT NULL,
  born           INTEGER NOT NULL,
  died           INTEGER NOT NULL,
  commutes       INTEGER NOT NULL,
  erases         INTEGER NOT NULL,
  annihilations  INTEGER NOT NULL,
  latches        INTEGER NOT NULL,
  snaps          INTEGER NOT NULL,
  free           REAL    NOT NULL,
  ground         REAL    NOT NULL,
  escrow         REAL    NOT NULL,
  mean_extra     REAL    NOT NULL,
  can_pay        REAL    NOT NULL,
  pp_wires       INTEGER NOT NULL,
  con_dup_wires  INTEGER NOT NULL,
  commute_share  REAL,
  commute_chance REAL    NOT NULL,
  commute_edge   REAL,
  matrix_drift   REAL    NOT NULL,
  -- Diversity and divergence; see measure.ts. Nullable because a pond with
  -- one net has no between-net variance and reporting zero would read as an
  -- answer rather than as nothing to compare.
  lines_effective  REAL,
  line_dominance   REAL,
  nets             INTEGER,
  nets_effective   REAL,
  net_dominance    REAL,
  kinds_effective  REAL,
  var_drifted      REAL,
  var_seeded       REAL,
  net_fst          REAL,
  line_fst         REAL,
  commutes_per_latch REAL,
  signal_total     REAL,
  forage_ratio     REAL,
  json           TEXT    NOT NULL,        -- the whole Sample, for what has no column
  PRIMARY KEY (run_id, t)
);
`;

/**
 * Columns added after a database in the wild already had the table.
 *
 * `CREATE TABLE IF NOT EXISTS` does nothing to a table that exists, so a
 * library grown before a measure existed would keep its old shape and every
 * insert naming the new column would fail. Adding them one at a time, guarded
 * by what `PRAGMA table_info` reports, means an old file keeps its runs and
 * gains the columns — which matters here more than usual, because the whole
 * point of the library is that knowledge accumulates across sessions.
 */
const MIGRATIONS: { table: string; column: string; decl: string }[] = [
  { table: 'run', column: 'sweep', decl: 'TEXT' },
  { table: 'run', column: 'point', decl: 'TEXT' },
  { table: 'sample', column: 'lines_effective', decl: 'REAL' },
  { table: 'sample', column: 'line_dominance', decl: 'REAL' },
  { table: 'sample', column: 'nets', decl: 'INTEGER' },
  { table: 'sample', column: 'nets_effective', decl: 'REAL' },
  { table: 'sample', column: 'net_dominance', decl: 'REAL' },
  { table: 'sample', column: 'kinds_effective', decl: 'REAL' },
  { table: 'sample', column: 'var_drifted', decl: 'REAL' },
  { table: 'sample', column: 'var_seeded', decl: 'REAL' },
  { table: 'sample', column: 'net_fst', decl: 'REAL' },
  { table: 'sample', column: 'line_fst', decl: 'REAL' },
  { table: 'sample', column: 'commutes_per_latch', decl: 'REAL' },
  { table: 'sample', column: 'signal_total', decl: 'REAL' },
  { table: 'sample', column: 'forage_ratio', decl: 'REAL' },
];

/** The eleven diversity columns, in the order `insertSample` binds them. */
function divCols(d: Record<string, unknown> | undefined): (number | null)[] {
  const num = (k: string): number | null => {
    const v = d?.[k];
    return v === null || v === undefined || Number.isNaN(v) ? null : Number(v);
  };
  return [
    num('linesEffective'),
    num('lineDominance'),
    num('nets'),
    num('netsEffective'),
    num('netDominance'),
    num('kindsEffective'),
    num('varianceDrifted'),
    num('varianceSeeded'),
    num('netFst'),
    num('lineFst'),
    num('commutesPerLatch'),
    num('signalTotal'),
    num('forageRatio'),
  ];
}

export interface RunSpec {
  seed: number;
  seconds: number;
  dt: number;
  world: { w: number; h: number };
  fieldCells: number;
  preset: string | null;
  soupCount: number;
  parentRun: number | null;
  params: Params;
  commit: string | null;
  note: string | null;
  /** Set when this run is one cell of a sweep; see `sweep.ts`. */
  sweep?: string | null;
  point?: Record<string, number> | null;
}

export interface RunRow {
  id: number;
  started_at: string;
  finished_at: string | null;
  seed: number;
  seconds: number;
  dt: number;
  frames: number;
  wall_ms: number | null;
  world_w: number;
  world_h: number;
  field_cells: number;
  preset: string | null;
  soup_count: number;
  parent_run: number | null;
  params: string;
  commit_hash: string | null;
  net_format: number;
  chem_len: number;
  note: string | null;
  census: string | null;
}

export interface NetRow extends NetStats {
  id: number;
  run_id: number;
  at: string;
  t: number;
  frame: number;
  parent_net: number | null;
  text: string;
}

/** A `net` row's summary columns, mapped back onto the names `NetStats` uses. */
function toNetRow(r: Record<string, unknown>): NetRow {
  return {
    id: Number(r.id),
    run_id: Number(r.run_id),
    at: String(r.at),
    t: Number(r.t),
    frame: Number(r.frame),
    parent_net: r.parent_net === null || r.parent_net === undefined ? null : Number(r.parent_net),
    text: String(r.text),
    bodies: Number(r.bodies),
    wires: Number(r.wires),
    era: Number(r.era),
    dup: Number(r.dup),
    con: Number(r.con),
    ppWires: Number(r.pp_wires),
    conDupWires: Number(r.con_dup_wires),
    bornMax: Number(r.born_max),
    bornMean: Number(r.born_mean),
    lines: Number(r.lines),
    dominant: Number(r.dominant),
    dominantShare: Number(r.dominant_share),
    energy: Number(r.energy),
    learned: Number(r.learned),
    plasticMean: Number(r.plastic_mean),
    matrixDrift: Number(r.matrix_drift),
  };
}

export class PondDb {
  readonly db: DatabaseSync;
  private readonly insertNet: StatementSync;
  private readonly insertSample: StatementSync;

  constructor(path: string) {
    if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true });
    this.db = new DatabaseSync(path);
    // WAL so a long run's writes do not block the lab page reading the same
    // file, and foreign keys on so `ON DELETE CASCADE` actually cascades —
    // SQLite leaves them off per connection, which is a footgun rather than a
    // default.
    this.db.exec('PRAGMA journal_mode = WAL');
    this.db.exec('PRAGMA foreign_keys = ON');
    this.db.exec(SCHEMA);
    for (const m of MIGRATIONS) {
      const cols = this.db.prepare(`PRAGMA table_info(${m.table})`).all() as { name: string }[];
      if (!cols.some((c) => c.name === m.column)) {
        this.db.exec(`ALTER TABLE ${m.table} ADD COLUMN ${m.column} ${m.decl}`);
      }
    }
    const have = this.meta('schema_version');
    if (have === null) this.setMeta('schema_version', String(SCHEMA_VERSION));
    else if (Number(have) < SCHEMA_VERSION) this.setMeta('schema_version', String(SCHEMA_VERSION));
    else if (Number(have) > SCHEMA_VERSION) {
      throw new Error(
        `pond: ${path} is schema version ${have}, newer than this build's ${SCHEMA_VERSION}`,
      );
    }
    this.insertNet = this.db.prepare(`
      INSERT INTO net (run_id, at, t, frame, bodies, wires, era, dup, con, pp_wires,
        con_dup_wires, born_max, born_mean, lines, dominant, dominant_share, energy,
        learned, plastic_mean, matrix_drift, parent_net, text, blob)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    `);
    this.insertSample = this.db.prepare(`
      INSERT OR REPLACE INTO sample (run_id, t, bodies, wires, lines, born_mean, born_max,
        spawned, born, died, commutes, erases, annihilations, latches, snaps,
        free, ground, escrow, mean_extra, can_pay, pp_wires, con_dup_wires,
        commute_share, commute_chance, commute_edge, matrix_drift,
        lines_effective, line_dominance, nets, nets_effective, net_dominance,
        kinds_effective, var_drifted, var_seeded, net_fst, line_fst, commutes_per_latch,
        signal_total, forage_ratio, json)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
              ?,?,?,?,?,?,?,?,?,?,?,?,?,
              ?)
    `);
  }

  close(): void {
    this.db.close();
  }

  meta(key: string): string | null {
    const row = this.db.prepare('SELECT value FROM meta WHERE key = ?').get(key) as
      | { value: string }
      | undefined;
    return row ? row.value : null;
  }

  setMeta(key: string, value: string): void {
    this.db.prepare('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)').run(key, value);
  }

  /** Open a run. Returns its id; `finishRun` closes it. */
  startRun(spec: RunSpec): number {
    const r = this.db
      .prepare(`
        INSERT INTO run (started_at, seed, seconds, dt, world_w, world_h, field_cells,
          preset, soup_count, parent_run, params, commit_hash, net_format, chem_len, note,
          sweep, point)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
      `)
      .run(
        new Date().toISOString(),
        spec.seed,
        spec.seconds,
        spec.dt,
        spec.world.w,
        spec.world.h,
        spec.fieldCells,
        spec.preset,
        spec.soupCount,
        spec.parentRun,
        JSON.stringify(spec.params),
        spec.commit,
        NET_FORMAT,
        CHEM_LEN,
        spec.note,
        spec.sweep ?? null,
        spec.point ? JSON.stringify(spec.point) : null,
      );
    return Number(r.lastInsertRowid);
  }

  finishRun(id: number, frames: number, wallMs: number, census: unknown): void {
    this.db
      .prepare('UPDATE run SET finished_at = ?, frames = ?, wall_ms = ?, census = ? WHERE id = ?')
      .run(new Date().toISOString(), frames, wallMs, JSON.stringify(census), id);
  }

  /** Record that stored net `netId` was planted into `runId` under `lineage`. */
  addPlant(runId: number, lineage: number, netId: number, bodies: number): void {
    this.db
      .prepare('INSERT OR REPLACE INTO plant (run_id, lineage, net_id, bodies) VALUES (?,?,?,?)')
      .run(runId, lineage, netId, bodies);
  }

  /**
   * Store one harvested net, and attribute it to its ancestor when it has one.
   *
   * A body planted from the database carries a negative founder line (see
   * `PlantOptions.lineage`), and a net's dominant line is therefore either a
   * positive id — it grew here — or a negative one naming a `plant` row. That
   * is the whole ancestry mechanism: no ids are carried across ponds, and a
   * net that has drifted onto a different founder is not falsely attributed.
   */
  addNet(runId: number, t: number, frame: number, net: CapturedNet, blob: Uint8Array): number {
    const s = net.stats;
    let parent: number | null = null;
    if (s.dominant < 0) {
      const row = this.db
        .prepare('SELECT net_id FROM plant WHERE run_id = ? AND lineage = ?')
        .get(runId, s.dominant) as { net_id: number } | undefined;
      parent = row ? Number(row.net_id) : null;
    }
    const r = this.insertNet.run(
      runId,
      new Date().toISOString(),
      t,
      frame,
      s.bodies,
      s.wires,
      s.era,
      s.dup,
      s.con,
      s.ppWires,
      s.conDupWires,
      s.bornMax,
      s.bornMean,
      s.lines,
      s.dominant,
      s.dominantShare,
      s.energy,
      s.learned,
      s.plasticMean,
      s.matrixDrift,
      parent,
      net.text,
      blob,
    );
    return Number(r.lastInsertRowid);
  }

  addSample(runId: number, s: Record<string, unknown>): void {
    this.insertSample.run(
      runId,
      Number(s.t),
      Number(s.bodies),
      Number(s.wires),
      Number(s.lines),
      Number(s.bornMean),
      Number(s.bornMax),
      Number(s.spawned),
      Number(s.born),
      Number(s.died),
      Number(s.commutes),
      Number(s.erases),
      Number(s.annihilations),
      Number(s.latches),
      Number(s.snaps),
      Number(s.free),
      Number(s.ground),
      Number(s.escrow),
      Number(s.meanExtra),
      Number(s.canPay),
      Number(s.ppWires),
      Number(s.conDupWires),
      s.commuteShare === null ? null : Number(s.commuteShare),
      Number(s.commuteChance),
      s.commuteEdge === null ? null : Number(s.commuteEdge),
      Number(s.matrixDrift),
      ...divCols(s.diversity as Record<string, unknown> | undefined),
      JSON.stringify(s),
    );
  }

  /** Everything but the blobs, newest first. */
  runs(limit = 50): RunRow[] {
    return this.db
      .prepare('SELECT * FROM run ORDER BY id DESC LIMIT ?')
      .all(limit) as unknown as RunRow[];
  }

  run(id: number): RunRow | null {
    const r = this.db.prepare('SELECT * FROM run WHERE id = ?').get(id) as unknown;
    return (r as RunRow) ?? null;
  }

  /** Net summaries without the blob. `runId` null means every run. */
  nets(opts: { runId?: number | null; limit?: number; order?: 'bodies' | 'depth' | 'recent' } = {}): NetRow[] {
    const order =
      opts.order === 'depth' ? 'born_max DESC, bodies DESC' : opts.order === 'recent' ? 'id DESC' : 'bodies DESC, id DESC';
    const where = opts.runId === undefined || opts.runId === null ? '' : 'WHERE run_id = ?';
    const args: unknown[] = [];
    if (where) args.push(opts.runId);
    args.push(opts.limit ?? 50);
    const rows = this.db
      .prepare(`SELECT id, run_id, at, t, frame, bodies, wires, era, dup, con, pp_wires,
        con_dup_wires, born_max, born_mean, lines, dominant, dominant_share, energy,
        learned, plastic_mean, matrix_drift, parent_net, text
        FROM net ${where} ORDER BY ${order} LIMIT ?`)
      .all(...(args as never[])) as Record<string, unknown>[];
    return rows.map(toNetRow);
  }

  /** One net's summary columns, without its blob. */
  netRow(id: number): NetRow | null {
    const r = this.db
      .prepare(`SELECT id, run_id, at, t, frame, bodies, wires, era, dup, con, pp_wires,
        con_dup_wires, born_max, born_mean, lines, dominant, dominant_share, energy,
        learned, plastic_mean, matrix_drift, parent_net, text FROM net WHERE id = ?`)
      .get(id) as Record<string, unknown> | undefined;
    return r ? toNetRow(r) : null;
  }

  /** The stored blob for one net, exactly as written. */
  blob(netId: number): Uint8Array | null {
    const row = this.db.prepare('SELECT blob FROM net WHERE id = ?').get(netId) as
      | { blob: Uint8Array }
      | undefined;
    return row ? row.blob : null;
  }

  /** The header alone, for listing without decoding. */
  header(netId: number): NetHeader | null {
    const b = this.blob(netId);
    return b ? readHeader(b) : null;
  }

  /** A net ready to plant. Throws if this build's genome layout has moved. */
  net(netId: number): NetData | null {
    const b = this.blob(netId);
    return b ? decodeNet(b) : null;
  }

  samples(runId: number): Record<string, unknown>[] {
    return this.db
      .prepare('SELECT * FROM sample WHERE run_id = ? ORDER BY t')
      .all(runId) as unknown as Record<string, unknown>[];
  }

  /** Run `fn` in one transaction. A run that dies mid-harvest stores nothing. */
  transaction<T>(fn: () => T): T {
    this.db.exec('BEGIN');
    try {
      const out = fn();
      this.db.exec('COMMIT');
      return out;
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }
}
