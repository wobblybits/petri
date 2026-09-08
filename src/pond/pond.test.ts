import { describe, expect, it } from 'vitest';
import { CHEM_LEN, CRITIC_LEN, PLASTIC_LEN, STATE_DIMS } from '../chem-layout.ts';
import { defaultParams, type Params } from '../params.ts';
import { loadPreset } from '../presets.ts';
import { TRAIT_KEYS } from '../rewrite.ts';
import { Sim } from '../sim.ts';
import { captureNets, plantNet } from './capture.ts';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PondDb, SCHEMA_VERSION } from './db.ts';
import { effects, sweepTrials } from './analyze.ts';
import { gridPoints } from './sweep.ts';
import {
  SCALAR_FIELDS,
  currentLayout,
  decodeNet,
  encodeNet,
  layoutComplaint,
  readHeader,
} from './net-blob.ts';

/*
 * What these are for.
 *
 * A pond database is a long-lived artifact and the genome layout is not: the
 * whole point of the header is that a blob written before a layout change is
 * refused rather than silently misread. So the tests that matter here are the
 * ones about *identity* — a net that goes out and comes back is the same net,
 * down to the bit — and the one about refusal.
 */

function learningPond(soup: number, frames: number): { sim: Sim; params: Params } {
  const params = defaultParams();
  params.soupCount = soup;
  // The dials that ship at zero. A pond with learning off stores an all-zero
  // plastic block, which would let a broken round trip pass.
  params.learnRate = 0.01;
  params.spawnInterval = 0;
  const sim = new Sim(1600, 1200, 128);
  loadPreset(sim, 'soup', params);
  for (let i = 0; i < frames; i++) sim.step(1 / 60, params);
  return { sim, params };
}

describe('net blob', () => {
  it('round-trips a net bit for bit', () => {
    const { sim } = learningPond(120, 400);
    const nets = captureNets(sim);
    expect(nets.length).toBeGreaterThan(0);
    const net = nets[0];
    expect(net.stats.bodies).toBeGreaterThan(1);

    const back = decodeNet(encodeNet(net.data));
    expect(back.bodies.length).toBe(net.data.bodies.length);
    expect(back.wires).toEqual(net.data.wires);
    for (let i = 0; i < back.bodies.length; i++) {
      const a = net.data.bodies[i];
      const b = back.bodies[i];
      expect(b.kind).toBe(a.kind);
      // Exact, not close: pose and traits are stored at the f64 they live at,
      // and the genome at the f32 the store holds. Nothing here rounds.
      for (const k of ['x', 'y', 'heading', ...SCALAR_FIELDS, 'born', 'lineage', 'prevValue'] as const) {
        expect(b[k]).toBe(a[k]);
      }
      expect([...b.chem]).toEqual([...a.chem]);
      expect([...b.plastic]).toEqual([...a.plastic]);
      expect([...b.trace]).toEqual([...a.trace]);
      expect([...b.critic]).toEqual([...a.critic]);
      expect([...b.h]).toEqual([...a.h]);
    }
  });

  it('stores something a body actually learned', () => {
    const { sim } = learningPond(120, 400);
    const nets = captureNets(sim);
    const moved = nets.some((n) => n.stats.plasticMean > 0);
    expect(moved).toBe(true);
    const net = nets.find((n) => n.stats.plasticMean > 0)!;
    const back = decodeNet(encodeNet(net.data));
    expect(back.bodies.some((b) => b.plastic.some((v) => v !== 0))).toBe(true);
  });

  it('names the header dimensions this build compiled with', () => {
    const { sim } = learningPond(60, 120);
    const header = readHeader(encodeNet(captureNets(sim)[0].data));
    expect(header.layout).toEqual({
      chem: CHEM_LEN,
      plastic: PLASTIC_LEN,
      critic: CRITIC_LEN,
      state: STATE_DIMS,
    });
    expect(layoutComplaint(header)).toBeNull();
    // Every section says its own type, offset and per-body stride, so a
    // reader that is not this module can walk it.
    const chem = header.sections.find((s) => s.name === 'chem')!;
    expect(chem.type).toBe('f32');
    expect(chem.stride).toBe(CHEM_LEN);
    expect(chem.count).toBe(CHEM_LEN * header.bodies);
  });

  it('refuses a blob whose genome layout has moved', () => {
    const { sim } = learningPond(60, 120);
    const blob = encodeNet(captureNets(sim)[0].data);
    const len = new DataView(blob.buffer, blob.byteOffset, blob.byteLength).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(blob.subarray(4, 4 + len)));
    header.layout.chem = CHEM_LEN + 10;
    // Re-encode in place; JSON.stringify of the same object is the same
    // length here because only a digit count changed, so pad to be sure.
    const json = new TextEncoder().encode(JSON.stringify(header).padEnd(len, ' '));
    expect(json.length).toBe(len);
    blob.set(json, 4);
    expect(() => decodeNet(blob)).toThrow(/genome layout has changed/);
    // The complaint names both widths, since the only way to act on it is to
    // know which build wrote the file.
    expect(layoutComplaint(readHeader(blob))).toContain(`chem ${CHEM_LEN + 10} -> ${CHEM_LEN}`);
  });

  it('decodes a blob that does not start on an eight-byte boundary', () => {
    // What a page gets when it slices a fetched ArrayBuffer, and what
    // `sqlite3` hands back from its shared read buffer: a view at whatever
    // offset the row landed on. A `Float64Array` cannot be built over an odd
    // one, so the decoder has to notice.
    const { sim } = learningPond(60, 200);
    const net = captureNets(sim)[0];
    const blob = encodeNet(net.data);
    const shifted = new Uint8Array(blob.byteLength + 3);
    shifted.set(blob, 3);
    const view = shifted.subarray(3);
    expect(view.byteOffset % 8).not.toBe(0);

    const back = decodeNet(view);
    expect(back.bodies.length).toBe(net.data.bodies.length);
    expect([...back.bodies[0].chem]).toEqual([...net.data.bodies[0].chem]);
    expect([...back.bodies[0].critic]).toEqual([...net.data.bodies[0].critic]);
    expect(back.wires).toEqual(net.data.wires);
  });

  it('keeps every heritable scalar the simulation breeds', () => {
    // SCALAR_FIELDS is written out by hand so the on-disk field order cannot
    // move; this is what makes that safe. A trait added to TRAIT_KEYS and not
    // here would be silently dropped from every stored net.
    for (const key of TRAIT_KEYS) expect(SCALAR_FIELDS).toContain(key);
    expect(currentLayout().chem).toBe(CHEM_LEN);
  });
});

describe('plant', () => {
  it('puts back the net that was taken out', () => {
    const { sim, params } = learningPond(120, 400);
    const net = captureNets(sim)[0];
    const source = decodeNet(encodeNet(net.data));

    sim.clear();
    loadPreset(sim, 'soup', { ...params, soupCount: 0 });
    const ids = plantNet(sim, params, source, sim.w * 0.5, sim.h * 0.5);
    expect(ids.length).toBe(source.bodies.length);

    const store = sim.agentStore;
    for (let i = 0; i < ids.length; i++) {
      const a = sim.agents.get(ids[i])!;
      const b = source.bodies[i];
      expect(a.kind).toBe(b.kind);
      expect(a.born).toBe(b.born);
      for (const k of TRAIT_KEYS) expect(a[k]).toBe(b[k]);
      const s = a.slot;
      expect([...store.chemAll.subarray(s * CHEM_LEN, (s + 1) * CHEM_LEN)]).toEqual([...b.chem]);
      expect([...store.plasticAll.subarray(s * PLASTIC_LEN, (s + 1) * PLASTIC_LEN)]).toEqual([...b.plastic]);
      expect([...store.criticAll.subarray(s * CRITIC_LEN, (s + 1) * CRITIC_LEN)]).toEqual([...b.critic]);
      // Monotone in the store and set by hand on a plant: a body that arrives
      // holding a learned delta must not run its bare genome.
      const learned = b.plastic.some((v) => v !== 0);
      expect(Boolean(store.plasticOn[s])).toBe(learned);
    }

    // Same topology, re-indexed onto the new ids.
    const replanted = captureNets(sim);
    expect(replanted.length).toBe(1);
    expect(replanted[0].data.wires.length).toBe(source.wires.length);
    expect(replanted[0].text).toBe(net.text);
  });

  it('plants nothing at all when the net will not fit', () => {
    const params = defaultParams();
    params.soupCount = 40;
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    for (let i = 0; i < 240; i++) sim.step(1 / 60, params);
    const net = decodeNet(encodeNet(captureNets(sim)[0].data));
    expect(net.bodies.length).toBeGreaterThan(2);

    // One body short of the net, so placement would stop part way through if
    // the cap were checked a body at a time. A torn piece of a net is worse
    // than no net, so the answer is nothing placed.
    const tight = { ...params, maxAgents: net.bodies.length - 1, soupCount: 0 };
    const full = new Sim(1600, 1200, 128);
    loadPreset(full, 'soup', tight);
    expect(plantNet(full, tight, net, full.w * 0.5, full.h * 0.5)).toEqual([]);
    expect(full.agents.size).toBe(0);

    // Exactly enough room, and it goes in whole.
    const roomy = { ...tight, maxAgents: net.bodies.length };
    const fits = new Sim(1600, 1200, 128);
    loadPreset(fits, 'soup', roomy);
    expect(plantNet(fits, roomy, net, fits.w * 0.5, fits.h * 0.5).length).toBe(net.bodies.length);
  });

  it('marks planted bodies with the founder line it is given', () => {
    const { sim, params } = learningPond(80, 240);
    const net = decodeNet(encodeNet(captureNets(sim)[0].data));
    const fresh = new Sim(1600, 1200, 128);
    loadPreset(fresh, 'soup', { ...params, soupCount: 0 });
    const ids = plantNet(fresh, params, net, fresh.w * 0.5, fresh.h * 0.5, { lineage: -3 });
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) expect(fresh.agents.get(id)!.lineage).toBe(-3);
    expect(fresh.census().lines).toBe(1);
  });
});

describe('learning readback', () => {
  it('is a no-op that succeeds when the genome is on the CPU', async () => {
    // The contract the harvest depends on: off the GPU path the host's copy is
    // the only copy, so asking for it must succeed and change nothing. A
    // `false` here would fail every headless harvest on a machine with no
    // device.
    const { sim } = learningPond(80, 300);
    const before = captureNets(sim)[0];
    expect(await sim.syncLearningToHost()).toBe(true);
    const after = captureNets(sim)[0];
    expect([...after.data.bodies[0].plastic]).toEqual([...before.data.bodies[0].plastic]);
    expect(after.stats.plasticMean).toBe(before.stats.plasticMean);
  });

  it('succeeds on an empty pond', async () => {
    const sim = new Sim(1600, 1200, 128);
    expect(await sim.syncLearningToHost()).toBe(true);
  });
});

describe('pond database', () => {
  it('stores a net and hands back the same one', () => {
    const { sim } = learningPond(120, 400);
    const net = captureNets(sim)[0];
    const db = new PondDb(':memory:');
    try {
      const params = defaultParams();
      const runId = db.startRun({
        seed: 1,
        seconds: 10,
        dt: 1 / 60,
        world: { w: 1600, h: 1200 },
        fieldCells: 128,
        preset: 'soup',
        soupCount: 120,
        parentRun: null,
        params,
        commit: 'abc1234',
        note: null,
      });
      const netId = db.addNet(runId, 6.5, 390, net, encodeNet(net.data));

      const row = db.netRow(netId)!;
      expect(row.bodies).toBe(net.stats.bodies);
      expect(row.wires).toBe(net.stats.wires);
      expect(row.text).toBe(net.text);
      expect(row.parent_net).toBeNull();

      const back = db.net(netId)!;
      expect(back.bodies.length).toBe(net.data.bodies.length);
      expect([...back.bodies[0].chem]).toEqual([...net.data.bodies[0].chem]);
      expect([...back.bodies[0].plastic]).toEqual([...net.data.bodies[0].plastic]);

      // The run row is the provenance: what it ran, and what it ran from.
      const run = db.run(runId)!;
      expect(JSON.parse(run.params).learnRate).toBe(params.learnRate);
      expect(run.commit_hash).toBe('abc1234');
      expect(run.chem_len).toBe(CHEM_LEN);
    } finally {
      db.close();
    }
  });

  it('attributes a net to the stored net it was planted from', () => {
    const { sim } = learningPond(120, 400);
    const net = captureNets(sim)[0];
    const db = new PondDb(':memory:');
    try {
      const spec = {
        seed: 1,
        seconds: 10,
        dt: 1 / 60,
        world: { w: 1600, h: 1200 },
        fieldCells: 128,
        preset: 'soup' as string | null,
        soupCount: 120,
        parentRun: null as number | null,
        params: defaultParams(),
        commit: null,
        note: null,
      };
      const first = db.startRun(spec);
      const seedNet = db.addNet(first, 10, 600, net, encodeNet(net.data));

      const second = db.startRun({ ...spec, preset: null, soupCount: 0, parentRun: first });
      db.addPlant(second, -1, seedNet, net.stats.bodies);

      // A descendant carries the negative founder line the plant handed out;
      // a net that grew here carries a positive one and is nobody's child.
      const child = { ...net, stats: { ...net.stats, dominant: -1 } };
      const native = { ...net, stats: { ...net.stats, dominant: 7 } };
      expect(db.netRow(db.addNet(second, 10, 600, child, encodeNet(net.data)))!.parent_net).toBe(seedNet);
      expect(db.netRow(db.addNet(second, 10, 600, native, encodeNet(net.data)))!.parent_net).toBeNull();
    } finally {
      db.close();
    }
  });
});

describe('sweeps', () => {
  it('crosses every axis with every seed, in a stable order', () => {
    const points = gridPoints({ a: [1, 2], b: [10, 20, 30] });
    expect(points).toHaveLength(6);
    expect(points[0]).toEqual({ a: 1, b: 10 });
    expect(points[5]).toEqual({ a: 2, b: 30 });
    // Stable, because a resumed or re-run sweep should line up against the
    // rows already in the library rather than against a different ordering.
    expect(gridPoints({ a: [1, 2], b: [10, 20, 30] })).toEqual(points);
    expect(gridPoints({})).toEqual([{}]);
  });

  it('tags its runs and reads them back as trials', () => {
    const db = new PondDb(':memory:');
    try {
      const base = {
        seed: 1,
        seconds: 10,
        dt: 1 / 60,
        world: { w: 1600, h: 1200 },
        fieldCells: 128,
        preset: 'soup' as string | null,
        soupCount: 10,
        parentRun: null,
        params: defaultParams(),
        commit: null,
        note: null,
      };
      const mk = (point: Record<string, number>, seed: number, bodies: number, fst: number | null) => {
        const id = db.startRun({ ...base, seed, sweep: 'demo', point });
        db.addSample(id, {
          t: 10, bodies, wires: 0, lines: 1, bornMean: 0, bornMax: 0,
          spawned: 0, born: 0, died: 0, commutes: 0, erases: 0, annihilations: 0,
          latches: 0, snaps: 0, free: 0, ground: 0, escrow: 0, meanExtra: 0,
          canPay: 0, ppWires: 0, conDupWires: 0, commuteShare: null,
          commuteChance: 0, commuteEdge: null, matrixDrift: 0,
          diversity: { netFst: fst },
        });
      };
      mk({ x: 0 }, 1, 10, 0.1);
      mk({ x: 0 }, 2, 12, 0.1);
      mk({ x: 1 }, 1, 30, 0.9);
      mk({ x: 1 }, 2, 32, null);

      const trials = sweepTrials(db, 'demo');
      expect(trials).toHaveLength(4);
      expect(trials[0].point).toEqual({ x: 0 });
      expect(trials[0].values.bodies).toBe(10);
      // Null survives as null all the way through the column and back.
      expect(trials[3].values.net_fst).toBeNull();
      expect(sweepTrials(db, 'nothing-by-that-name')).toEqual([]);

      const ranked = effects(trials, ['bodies', 'net_fst']);
      const byBodies = ranked.find((e) => e.metric === 'bodies')!;
      expect(byBodies.axis).toBe('x');
      // Twenty apart between levels against two within: nearly all of it.
      expect(byBodies.eta2).toBeGreaterThan(0.98);
      expect(byBodies.high.level).toBe(1);
      const byFst = ranked.find((e) => e.metric === 'net_fst')!;
      expect(byFst.trials, 'the null trial should be dropped, not zeroed').toBe(3);
    } finally {
      db.close();
    }
  });

  it('migrates a library written before the diversity columns existed', () => {
    /*
     * `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
     * exists, so a library grown before a measure was added would keep its old
     * shape and every insert naming the new column would fail. The whole point
     * of the library is that it accumulates across sessions, so opening an old
     * one has to work — which means a real file and a second handle, since an
     * in-memory database is gone the moment the first one closes.
     */
    const dir = mkdtempSync(join(tmpdir(), 'pond-migrate-'));
    const path = join(dir, 'old.db');
    try {
      const first = new PondDb(path);
      const runId = first.startRun({
        seed: 1,
        seconds: 1,
        dt: 1 / 60,
        world: { w: 100, h: 100 },
        fieldCells: 64,
        preset: 'soup',
        soupCount: 1,
        parentRun: null,
        params: defaultParams(),
        commit: null,
        note: null,
      });
      // Age the file: strip the columns a later build added, and the version
      // stamp with them, so this is the shape a v1 library actually had.
      for (const col of ['net_fst', 'line_fst', 'lines_effective']) {
        first.db.exec(`ALTER TABLE sample DROP COLUMN ${col}`);
      }
      first.db.exec('ALTER TABLE run DROP COLUMN sweep');
      first.setMeta('schema_version', '1');
      first.close();

      const second = new PondDb(path);
      try {
        const cols = (table: string) =>
          (second.db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[]).map((c) => c.name);
        expect(cols('sample')).toEqual(expect.arrayContaining(['net_fst', 'line_fst', 'lines_effective']));
        expect(cols('run')).toContain('sweep');
        expect(second.meta('schema_version')).toBe(String(SCHEMA_VERSION));
        // The run that was already there is still there, and a sample naming
        // the new columns now inserts against it.
        expect(second.run(runId)).not.toBeNull();
        second.addSample(runId, {
          t: 0, bodies: 1, wires: 0, lines: 1, bornMean: 0, bornMax: 0,
          spawned: 0, born: 0, died: 0, commutes: 0, erases: 0, annihilations: 0,
          latches: 0, snaps: 0, free: 0, ground: 0, escrow: 0, meanExtra: 0,
          canPay: 0, ppWires: 0, conDupWires: 0, commuteShare: null,
          commuteChance: 0, commuteEdge: null, matrixDrift: 0,
          diversity: { netFst: 0.5 },
        });
        expect(second.samples(runId)).toHaveLength(1);
        expect(Number(second.samples(runId)[0].net_fst)).toBeCloseTo(0.5, 10);
      } finally {
        second.close();
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
