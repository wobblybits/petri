import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { CHEM_LEN, PLASTIC_LEN } from '../chem-layout.ts';
import { fieldGpu } from '../gpu/field-gpu.ts';
import { nativeSolver } from '../native/solver.ts';
import { PondDb, type NetRow } from './db.ts';
import { NET_FORMAT, layoutComplaint, readHeader } from './net-blob.ts';
import { parseGround } from './ground.ts';
import { paramsWith, runPond, type SeedNet } from './run.ts';
import { effectTable, effects, pointTable, sweepTrials } from './analyze.ts';
import { gridPoints, runSweep } from './sweep.ts';
import { closeWebGpu } from './webgpu-node.ts';

/*
 * `npm run pond -- <command>`
 *
 * A pond you can leave running. The page is the instrument for watching one;
 * this is for growing one — overnight, at a size and a length no tab wants to
 * hold — and for reopening what the last one grew.
 *
 * Every run writes a row saying when it started, what it ran, and which
 * commit it ran from, so a database is a record of an experiment rather than
 * a heap of genomes with no provenance.
 */

const USAGE = `
petri pond — a headless pond, and the library it writes to

  npm run pond -- run    [options]      grow a pond and store what it grew
  npm run pond -- runs   [--limit n]    list runs, newest first
  npm run pond -- nets   [options]      list stored nets
  npm run pond -- show   <net-id>       one net: stats, and its HVM2 topology
  npm run pond -- export <net-id> [f]   write one net's blob to a file
  npm run pond -- sweep  --name n ...   run a parameter grid into the library
  npm run pond -- analyze --name n      what the sweep says, and which dial did it

run options
  --db <path>          database file           (default ponds/pond.db)
  --seconds <n>        simulated seconds       (default 120)
  --dt <n>             frame step              (default 1/60)
  --seed <n>           RNG seed                (default 1)
  --bodies <n>         founders in the opening soup (default 400, or 0 when planting)
  --world <w>x<h>      spawn box               (default 1600x1200)
  --field <n>          field cells a side      (default 512)
  --sample-every <n>   simulated seconds between timeline samples (default 10)
  --harvest-every <n>  also store nets this often, not just at the close
  --min-bodies <n>     smallest component worth storing  (default 2)
  --limit <n>          store only the n largest components per harvest
  --gpu auto|on|off    field and genome on the GPU, via Dawn  (default auto)
  --ground <spec>      uniform | none | patches[:n]   (default uniform)
                       the same total mass, arranged differently
  --set <key>=<value>  override one Params field; repeatable
  --note <text>        free text on the run row

seeding from the library
  --from-run <id>      plant the nets that run stored (with --top)
  --from-net <ids>     plant these nets, comma separated
  --top <n>            with --from-run: the n largest             (default 8)

nets options
  --db <path>   --run <id>   --limit <n>   --order bodies|depth|recent

sweep options
  --name <text>        sweep name; runs are tagged with it        (required)
  --axis <key>=<a,b,c> one parameter axis; repeatable
  --seeds <a,b,c>      seeds per grid point                       (default 1,2,3)
  --seconds <n>        simulated seconds per trial                (default 120)
  --bodies <n>         founders per trial                         (default 400)
  --field <n>          field cells a side                         (default 512)
  --sample-every <n>   simulated seconds between samples          (default 10)
  --set <key>=<value>  applied to every point, before the axes
  --keep-nets          store the largest nets of each trial too
  --gpu auto|on|off    as for run
  --dry-run            print the grid and the cost, run nothing

analyze options
  --name <text>        sweep to read; omit to list what is in the library
  --metric <a,b,c>     restrict to these metrics
  --limit <n>          rows of the effect table                   (default 25)
`.trim();

interface Args {
  command: string;
  rest: string[];
  flags: Map<string, string>;
  sets: Map<string, number>;
  /** Repeatable `--axis key=v1,v2`, in the order given. */
  axes: string[];
}

function parseArgs(argv: string[]): Args {
  const flags = new Map<string, string>();
  const sets = new Map<string, number>();
  const axes: string[] = [];
  const rest: string[] = [];
  let command = 'run';
  let i = 0;
  if (argv[0] && !argv[0].startsWith('-')) command = argv[i++];
  for (; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) {
      rest.push(a);
      continue;
    }
    const key = a.slice(2);
    if (key === 'help' || key === 'keep-nets' || key === 'dry-run') {
      flags.set(key, '1');
      continue;
    }
    const value = argv[++i];
    if (value === undefined) throw new Error(`pond: ${a} needs a value`);
    if (key === 'axis') {
      axes.push(value);
    } else if (key === 'set') {
      const eq = value.indexOf('=');
      if (eq < 0) throw new Error(`pond: --set wants key=value, got ${JSON.stringify(value)}`);
      const n = Number(value.slice(eq + 1));
      if (!Number.isFinite(n)) throw new Error(`pond: --set ${value} is not a number`);
      sets.set(value.slice(0, eq), n);
    } else {
      flags.set(key, value);
    }
  }
  return { command, rest, flags, sets, axes };
}

function num(flags: Map<string, string>, key: string, fallback: number): number {
  const v = flags.get(key);
  if (v === undefined) return fallback;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`pond: --${key} wants a number, got ${JSON.stringify(v)}`);
  return n;
}

function gitCommit(): string | null {
  try {
    return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

/** Right-align numbers, left-align everything else, one space of gutter. */
function table(head: string[], rows: (string | number | null)[][]): string {
  const cells = rows.map((r) => r.map((c) => (c === null ? '-' : typeof c === 'number' ? fmt(c) : c)));
  const widths = head.map((h, i) => Math.max(h.length, ...cells.map((r) => (r[i] ?? '').length)));
  const numeric = head.map((_, i) => rows.every((r) => r[i] === null || typeof r[i] === 'number'));
  const line = (r: string[]) =>
    r.map((c, i) => (numeric[i] ? c.padStart(widths[i]) : c.padEnd(widths[i]))).join('  ').trimEnd();
  return [line(head), line(widths.map((w) => '-'.repeat(w))), ...cells.map(line)].join('\n');
}

function fmt(n: number): string {
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(Math.abs(n) < 1 ? 3 : 2);
}

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function netRows(rows: NetRow[]): string {
  return table(
    ['id', 'run', 't', 'bodies', 'wires', 'era/dup/con', 'depth', 'lines', 'drift', 'learned', 'from'],
    rows.map((r) => [
      r.id,
      r.run_id,
      r.t,
      r.bodies,
      r.wires,
      `${r.era}/${r.dup}/${r.con}`,
      r.bornMax,
      r.lines,
      r.matrixDrift,
      r.learned,
      r.parent_net,
    ]),
  );
}

async function cmdRun(args: Args): Promise<void> {
  const { flags, sets } = args;
  /*
   * Turn the solver on.
   *
   * `Sim.nativeForces` is true by default but `nativeSolver.ready` is not:
   * the wasm is instantiated by an explicit `init()`, which the three pages
   * each call at startup and which nothing headless did. Without it every
   * guarded call falls through to the JavaScript twin, silently and
   * correctly — the pond is right, it is just running the reference
   * implementation of a solver that exists in C.
   *
   * Measured back to back at 1,800 bodies: 57.0 ms a frame on the twin,
   * 16.9 ms with this line, and the solve phase alone 37.6 -> 4.8 ms.
   */
  const wasm = await nativeSolver.init();
  if (!wasm) {
    process.stderr.write(`pond: wasm solver unavailable (${nativeSolver.lastError}); running the JS twin, which is ~3x slower\n`);
  }
  const db = new PondDb(flags.get('db') ?? 'ponds/pond.db');
  try {
    const seedNets: SeedNet[] = [];
    let parentRun: number | null = null;

    const fromNet = flags.get('from-net');
    const fromRun = flags.get('from-run');
    if (fromNet !== undefined && fromRun !== undefined) {
      throw new Error('pond: use --from-run or --from-net, not both');
    }
    let ids: number[] = [];
    if (fromNet !== undefined) {
      ids = fromNet.split(',').map((s) => {
        const n = Number(s.trim());
        if (!Number.isInteger(n)) throw new Error(`pond: --from-net wants net ids, got ${JSON.stringify(s)}`);
        return n;
      });
    } else if (fromRun !== undefined) {
      parentRun = Number(fromRun);
      if (!Number.isInteger(parentRun)) throw new Error('pond: --from-run wants a run id');
      const top = num(flags, 'top', 8);
      // The close of that run, not every harvest it made: continuing from an
      // hour-old snapshot of the same net as well as its final state would
      // plant the same lineage twice and call it two.
      const last = db.db
        .prepare('SELECT MAX(t) AS t FROM net WHERE run_id = ?')
        .get(parentRun) as { t: number | null } | undefined;
      if (!last || last.t === null) throw new Error(`pond: run ${parentRun} stored no nets`);
      ids = (
        db.db
          .prepare('SELECT id FROM net WHERE run_id = ? AND t = ? ORDER BY bodies DESC LIMIT ?')
          .all(parentRun, last.t, top) as { id: number }[]
      ).map((r) => Number(r.id));
      if (ids.length === 0) throw new Error(`pond: run ${parentRun} stored no nets`);
    }
    for (const id of ids) {
      const blob = db.blob(id);
      if (!blob) throw new Error(`pond: no net ${id}`);
      const complaint = layoutComplaint(readHeader(blob));
      if (complaint) throw new Error(`pond: net ${id}: ${complaint}`);
      seedNets.push({ netId: id, data: db.net(id)! });
    }

    const gpuFlag = flags.get('gpu') ?? 'auto';
    if (gpuFlag !== 'auto' && gpuFlag !== 'on' && gpuFlag !== 'off') {
      throw new Error('pond: --gpu wants auto, on or off');
    }
    const gpu: 'auto' | 'on' | 'off' = gpuFlag;
    const params = paramsWith(sets);
    const worldFlag = flags.get('world') ?? '1600x1200';
    const [ww, wh] = worldFlag.split('x').map(Number);
    if (!Number.isFinite(ww) || !Number.isFinite(wh)) {
      throw new Error(`pond: --world wants <w>x<h>, got ${JSON.stringify(worldFlag)}`);
    }
    const spec = {
      seconds: num(flags, 'seconds', 120),
      dt: num(flags, 'dt', 1 / 60),
      seed: num(flags, 'seed', 1),
      world: { w: ww, h: wh },
      fieldCells: num(flags, 'field', 512),
      params,
      soupCount: num(flags, 'bodies', seedNets.length > 0 ? 0 : 400),
      seeds: seedNets,
      sampleEvery: num(flags, 'sample-every', 10),
      harvestEvery: num(flags, 'harvest-every', 0),
      minBodies: num(flags, 'min-bodies', 2),
      limit: flags.has('limit') ? num(flags, 'limit', 0) : null,
      gpu,
      ground: parseGround(flags.get('ground') ?? 'uniform'),
    };

    const runId = db.startRun({
      seed: spec.seed,
      seconds: spec.seconds,
      dt: spec.dt,
      world: spec.world,
      fieldCells: spec.fieldCells,
      // A pond seeded only from the library did not come from a preset, and
      // saying `soup` would make the row claim something untrue about how it
      // started.
      preset: spec.soupCount > 0 ? 'soup' : null,
      soupCount: spec.soupCount,
      parentRun,
      params,
      commit: gitCommit(),
      note: flags.get('note') ?? null,
    });

    process.stderr.write(
      `run ${runId}: ${spec.seconds}s at seed ${spec.seed}, ` +
        `${spec.soupCount} founders` +
        (seedNets.length > 0 ? ` + ${seedNets.length} planted net(s)` : '') +
        `, field ${spec.fieldCells}\n`,
    );

    const result = await runPond(db, runId, spec, {
      onSample: (s) => {
        process.stderr.write(
          `  t=${s.t.toFixed(0).padStart(5)}s  bodies ${String(s.bodies).padStart(6)}  ` +
            `wires ${String(s.wires).padStart(6)}  lines ${String(s.lines).padStart(5)}  ` +
            `depth ${s.bornMean.toFixed(2)}  drift ${s.matrixDrift.toFixed(4)}\n`,
        );
      },
      onHarvest: (t, nets) => {
        process.stderr.write(`  harvest at ${t.toFixed(0)}s: ${nets.length} net(s)\n`);
      },
    });
    db.finishRun(runId, result.frames, result.wallMs, result.census);

    const stored = result.harvests.reduce((a, h) => a + h.nets, 0);
    const size = result.harvests.reduce((a, h) => a + h.bytes, 0);
    const on = [
      result.paths.wasm ? 'wasm solve' : 'JS solve',
      result.paths.fieldGpu ? 'GPU field' : 'CPU field',
      result.paths.genomeGpu ? 'GPU genome' : 'CPU genome',
    ].join(', ');
    process.stdout.write(
      `\nrun ${runId} done: ${result.frames} frames in ${(result.wallMs / 1000).toFixed(1)}s ` +
        `(${((result.wallMs / result.frames) || 0).toFixed(2)} ms/frame), ` +
        `${stored} net(s) stored (${bytes(size)})\n` +
        `  ${on}${result.paths.adapter ? ` on ${result.paths.adapter}` : ''}\n\n`,
    );
    const top = db.nets({ runId, limit: 10 });
    if (top.length > 0) process.stdout.write(`${netRows(top)}\n`);
  } finally {
    db.close();
  }
}

function cmdRuns(args: Args): void {
  const db = new PondDb(args.flags.get('db') ?? 'ponds/pond.db');
  try {
    const rows = db.runs(num(args.flags, 'limit', 25));
    if (rows.length === 0) {
      process.stdout.write('no runs\n');
      return;
    }
    process.stdout.write(
      `${table(
        ['id', 'started', 'seed', 'secs', 'soup', 'from', 'nets', 'bodies', 'note'],
        rows.map((r) => {
          const nets = db.db
            .prepare('SELECT COUNT(*) AS n, MAX(bodies) AS b FROM net WHERE run_id = ?')
            .get(r.id) as { n: number; b: number | null };
          return [
            r.id,
            r.started_at.replace('T', ' ').slice(0, 19),
            r.seed,
            r.seconds,
            r.soup_count,
            r.parent_run,
            Number(nets.n),
            nets.b === null ? null : Number(nets.b),
            r.note ?? '',
          ];
        }),
      )}\n`,
    );
  } finally {
    db.close();
  }
}

function cmdNets(args: Args): void {
  const db = new PondDb(args.flags.get('db') ?? 'ponds/pond.db');
  try {
    const order = args.flags.get('order') ?? 'bodies';
    if (order !== 'bodies' && order !== 'depth' && order !== 'recent') {
      throw new Error('pond: --order wants bodies, depth or recent');
    }
    const rows = db.nets({
      runId: args.flags.has('run') ? num(args.flags, 'run', 0) : null,
      limit: num(args.flags, 'limit', 25),
      order,
    });
    process.stdout.write(rows.length === 0 ? 'no nets\n' : `${netRows(rows)}\n`);
  } finally {
    db.close();
  }
}

function cmdShow(args: Args): void {
  const id = Number(args.rest[0]);
  if (!Number.isInteger(id)) throw new Error('pond: show wants a net id');
  const db = new PondDb(args.flags.get('db') ?? 'ponds/pond.db');
  try {
    const row = db.netRow(id);
    if (!row) throw new Error(`pond: no net ${id}`);
    const blob = db.blob(id)!;
    const header = readHeader(blob);
    const complaint = layoutComplaint(header);
    process.stdout.write(
      `net ${row.id}  run ${row.run_id}  t=${row.t.toFixed(1)}s  ${row.at}\n` +
        `  ${row.bodies} bodies (${row.era} era, ${row.dup} dup, ${row.con} con), ${row.wires} wires\n` +
        `  ${row.ppWires} principal-to-principal, ${row.conDupWires} of them Con-Dup\n` +
        `  depth ${row.bornMax} max / ${row.bornMean.toFixed(2)} mean, ${row.lines} line(s), ` +
        `dominant ${row.dominant} at ${(row.dominantShare * 100).toFixed(0)}%\n` +
        `  energy ${row.energy.toFixed(2)}, matrix drift ${row.matrixDrift.toFixed(4)}, ` +
        `learned ${(row.learned * 100).toFixed(0)}% of bodies (mean |delta| ${row.plasticMean.toFixed(5)})\n` +
        `  descends from net ${row.parent_net ?? '-'}\n` +
        `  blob ${bytes(blob.byteLength)}, format ${header.format}, ` +
        `genome ${header.layout.chem} + ${header.layout.plastic} learned\n` +
        (complaint ? `  UNREADABLE HERE: ${complaint}\n` : '') +
        `\n${row.text}\n`,
    );
  } finally {
    db.close();
  }
}

function cmdExport(args: Args): void {
  const id = Number(args.rest[0]);
  if (!Number.isInteger(id)) throw new Error('pond: export wants a net id');
  const db = new PondDb(args.flags.get('db') ?? 'ponds/pond.db');
  try {
    const blob = db.blob(id);
    if (!blob) throw new Error(`pond: no net ${id}`);
    const path = args.rest[1] ?? `net-${id}.petrinet`;
    writeFileSync(path, blob);
    process.stdout.write(`${path}  ${bytes(blob.byteLength)}\n`);
  } finally {
    db.close();
  }
}


function cmdAnalyze(args: Args): void {
  const db = new PondDb(args.flags.get('db') ?? 'ponds/pond.db');
  try {
    const name = args.flags.get('name');
    if (!name) {
      const rows = db.db
        .prepare(`SELECT sweep, COUNT(*) AS runs, MIN(started_at) AS first, MAX(started_at) AS last
                  FROM run WHERE sweep IS NOT NULL GROUP BY sweep ORDER BY last DESC`)
        .all() as { sweep: string; runs: number; first: string; last: string }[];
      process.stdout.write(
        rows.length === 0
          ? 'no sweeps in this library\n'
          : `${table(
              ['sweep', 'runs', 'first', 'last'],
              rows.map((r) => [
                r.sweep,
                Number(r.runs),
                r.first.slice(0, 19).replace('T', ' '),
                r.last.slice(0, 19).replace('T', ' '),
              ]),
            )}\n`,
      );
      return;
    }
    const trials = sweepTrials(db, name);
    if (trials.length === 0) throw new Error(`pond: no runs tagged ${JSON.stringify(name)}`);
    const only = args.flags.get('metric');
    const metrics = only ? only.split(',').map((m) => m.trim()) : undefined;
    process.stdout.write(`sweep ${name}: ${trials.length} trial(s)\n\n`);
    process.stdout.write(`${pointTable(trials, metrics)}\n\n`);
    /*
     * The ranking is the point. The per-point table is what a sweep produced;
     * the effect table is what it *found* — which dial moved which measure,
     * and by how much against the disagreement between seeds.
     */
    process.stdout.write('effect of each axis, by share of variance explained\n');
    process.stdout.write(`${effectTable(effects(trials, metrics), num(args.flags, 'limit', 25))}\n`);
  } finally {
    db.close();
  }
}

async function cmdSweep(args: Args): Promise<void> {
  const { flags, sets } = args;
  const name = flags.get('name');
  if (!name) throw new Error('pond: sweep needs --name');
  const grid: Record<string, number[]> = {};
  for (const spec of args.axes) {
    const eq = spec.indexOf('=');
    if (eq < 0) throw new Error(`pond: --axis wants key=v1,v2, got ${JSON.stringify(spec)}`);
    const key = spec.slice(0, eq);
    const values = spec
      .slice(eq + 1)
      .split(',')
      .map((v) => {
        const n = Number(v.trim());
        if (!Number.isFinite(n)) throw new Error(`pond: --axis ${spec}: ${JSON.stringify(v)} is not a number`);
        return n;
      });
    if (values.length === 0) throw new Error(`pond: --axis ${spec} has no values`);
    grid[key] = values;
  }
  if (Object.keys(grid).length === 0) throw new Error('pond: sweep needs at least one --axis');
  /*
   * Validated here rather than inside the loop. A typo in an axis name should
   * fail before the first trial, not after the twentieth — `paramsWith` throws
   * on a key `Params` does not have, which is the whole check.
   */
  const probe = new Map<string, number>(Object.entries(Object.fromEntries(sets)));
  for (const [k, v] of Object.entries(grid)) probe.set(k, v[0]);
  paramsWith(probe);

  const seeds = (flags.get('seeds') ?? '1,2,3').split(',').map((v) => Number(v.trim()));
  if (seeds.some((v) => !Number.isFinite(v))) throw new Error('pond: --seeds wants numbers');
  const gpuFlag = flags.get('gpu') ?? 'auto';
  if (gpuFlag !== 'auto' && gpuFlag !== 'on' && gpuFlag !== 'off') {
    throw new Error('pond: --gpu wants auto, on or off');
  }
  const worldFlag = flags.get('world') ?? '1600x1200';
  const [ww, wh] = worldFlag.split('x').map(Number);
  if (!Number.isFinite(ww) || !Number.isFinite(wh)) {
    throw new Error(`pond: --world wants <w>x<h>, got ${JSON.stringify(worldFlag)}`);
  }
  const spec = {
    name,
    grid,
    seeds,
    base: Object.fromEntries(sets),
    seconds: num(flags, 'seconds', 120),
    dt: num(flags, 'dt', 1 / 60),
    soupCount: num(flags, 'bodies', 400),
    fieldCells: num(flags, 'field', 512),
    world: { w: ww, h: wh },
    sampleEvery: num(flags, 'sample-every', 10),
    gpu: gpuFlag as 'auto' | 'on' | 'off',
    ground: parseGround(flags.get('ground') ?? 'uniform'),
    keepNets: flags.has('keep-nets'),
    note: flags.get('note') ?? null,
  };
  const points = gridPoints(grid);
  const total = points.length * seeds.length;
  process.stderr.write(
    `sweep ${name}: ${points.length} point(s) x ${seeds.length} seed(s) = ${total} trial(s), ` +
      `${spec.seconds}s each\n`,
  );
  if (flags.has('dry-run')) {
    for (const p of points) process.stdout.write(`${JSON.stringify(p)}\n`);
    return;
  }

  const wasm = await nativeSolver.init();
  if (!wasm) process.stderr.write(`pond: wasm solver unavailable (${nativeSolver.lastError})\n`);
  const db = new PondDb(flags.get('db') ?? 'ponds/pond.db');
  try {
    const t0 = Date.now();
    await runSweep(db, spec, {
      onTrial: (row, done) => {
        const per = (Date.now() - t0) / done;
        const left = ((total - done) * per) / 1000;
        process.stderr.write(
          `  [${String(done).padStart(3)}/${total}] run ${row.runId} ${JSON.stringify(row.point)} ` +
            `seed=${row.seed} ${(row.wallMs / 1000).toFixed(1)}s` +
            (done < total ? `  ~${left < 90 ? `${left.toFixed(0)}s` : `${(left / 60).toFixed(1)}m`} left` : '') +
            '\n',
        );
      },
    });
    const trials = sweepTrials(db, name);
    process.stdout.write(`\nsweep ${name}: ${trials.length} trial(s)\n\n${pointTable(trials)}\n\n`);
    process.stdout.write('effect of each axis, by share of variance explained\n');
    process.stdout.write(`${effectTable(effects(trials))}\n`);
  } finally {
    db.close();
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.flags.has('help')) {
    process.stdout.write(`${USAGE}\n`);
    return;
  }
  if (args.command === 'run') await cmdRun(args);
  else if (args.command === 'runs') cmdRuns(args);
  else if (args.command === 'nets') cmdNets(args);
  else if (args.command === 'show') cmdShow(args);
  else if (args.command === 'export') cmdExport(args);
  else if (args.command === 'sweep') await cmdSweep(args);
  else if (args.command === 'analyze' || args.command === 'analyse') cmdAnalyze(args);
  else if (args.command === 'format') {
    process.stdout.write(
      `net format ${NET_FORMAT}: genome ${CHEM_LEN} floats, learned ${PLASTIC_LEN}\n`,
    );
  } else {
    process.stderr.write(`${USAGE}\n`);
    process.exitCode = 2;
  }
}

/**
 * Run, then let go of the device.
 *
 * A live `GPUDevice` holds Node's event loop open, so a finished run with the
 * database closed and nothing left to do would simply sit there. Destroying it
 * is the polite way out; the explicit exit is the one that works whether or
 * not Dawn has anything else outstanding, and it happens after stdout has
 * drained so the summary is not truncated on a pipe.
 */
main()
  .catch((e: unknown) => {
    process.stderr.write(`${e instanceof Error ? e.message : String(e)}\n`);
    process.exitCode = 1;
  })
  .finally(() => {
    closeWebGpu(fieldGpu.gpuDevice);
    const code = process.exitCode ?? 0;
    if (process.stdout.writableLength === 0) process.exit(code);
    else process.stdout.once('drain', () => process.exit(code));
  });
