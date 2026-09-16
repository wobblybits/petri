import { describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { seedChem } from '../agents.ts';
import { CHEM_LEN, CHEM_SEGMENTS, PLASTIC_BASE, PLASTIC_LEN, type ChemSegment } from '../chem-layout.ts';
import { defaultParams, type Params } from '../params.ts';
import { TRAIT_RANGE } from '../rewrite.ts';
import { loadPreset } from '../presets.ts';
import { Sim } from '../sim.ts';
import { captureNets, plantNet, prepareNet } from './capture.ts';
import {
  SCALAR_FIELDS,
  compatibility,
  currentLayout,
  decodeNet,
  encodeNet,
  encodeNetAs,
  isCurrent,
  layoutSelfCheck,
  migrateNet,
  readHeader,
  segmentsComplaint,
  storedScalars,
  type NetData,
  type ScalarField,
  type NetHeader,
} from './net-blob.ts';
import { fixturePath, inspectNetFile, listFixtures, loadFixture, loadNet, saveNet } from './net-file.ts';

/*
 * A stored net has to outlive the genome layout that stored it, and has to
 * say so when it cannot. These are the cases: a head appended (what happened
 * twice in the week this was written), segments reordered, one renamed, and
 * the changes that are refused because the numbers would no longer mean what
 * they meant. Plus the files, and the checked-in nets the suite starts from.
 */

function learningPond(soup: number, frames: number): { sim: Sim; params: Params } {
  const params = defaultParams();
  params.soupCount = soup;
  params.learnRate = 0.01;
  params.spawnInterval = 0;
  const sim = new Sim(1600, 1200, 128);
  loadPreset(sim, 'soup', params);
  for (let i = 0; i < frames; i++) sim.step(1 / 60, params);
  return { sim, params };
}

function segment(name: string): ChemSegment {
  const s = CHEM_SEGMENTS.find((x) => x.name === name);
  if (!s) throw new Error(`no segment ${name}`);
  return s;
}

/**
 * The net as a build with segments in `names`' order (a subset of this
 * build's, optionally renamed) would have stored it: every segment copied to
 * where that build kept it. The learned block has to stay a contiguous run
 * for this to be a valid layout, which every case below arranges.
 */
function asOldBuild(
  net: NetData,
  names: string[],
  rename: Record<string, string> = {},
): { blob: Uint8Array; segments: ChemSegment[]; chem: number } {
  const segments: ChemSegment[] = [];
  let at = 0;
  for (const n of names) {
    const cur = segment(n);
    segments.push({ name: rename[n] ?? n, at, len: cur.len });
    at += cur.len;
  }
  const chem = at;
  const plasticAt = segments[names.indexOf('Wx')].at;
  const bodies = net.bodies.map((b) => {
    const c = new Float32Array(chem);
    for (let i = 0; i < names.length; i++) {
      const cur = segment(names[i]);
      c.set(b.chem.subarray(cur.at, cur.at + cur.len), segments[i].at);
    }
    return { ...b, chem: c };
  });
  const blob = encodeNetAs({ bodies, wires: net.wires }, { ...currentLayout(), chem }, segments, plasticAt);
  return { blob, segments, chem };
}

/** Rewrite a blob's header JSON in place, padded to the same length. */
function patchHeader(blob: Uint8Array, edit: (h: NetHeader) => void): Uint8Array {
  const len = new DataView(blob.buffer, blob.byteOffset, blob.byteLength).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(blob.subarray(4, 4 + len))) as NetHeader;
  edit(header);
  const text = JSON.stringify(header);
  if (text.length > len) throw new Error('patched header would not fit');
  blob.set(new TextEncoder().encode(text.padEnd(len, ' ')), 4);
  return blob;
}

const NAMES = CHEM_SEGMENTS.map((s) => s.name);

describe('the segment map', () => {
  it("describes this build's genome exactly", () => {
    expect(layoutSelfCheck()).toBeNull();
    let at = 0;
    for (const s of CHEM_SEGMENTS) {
      expect(s.at).toBe(at);
      at += s.len;
    }
    expect(at).toBe(CHEM_LEN);
    // The learned block is a run of whole segments, so a delta can follow them.
    expect(CHEM_SEGMENTS.some((s) => s.at === PLASTIC_BASE)).toBe(true);
    expect(CHEM_SEGMENTS.some((s) => s.at + s.len === PLASTIC_BASE + PLASTIC_LEN)).toBe(true);
  });

  it('names what is wrong with a map that does not tile', () => {
    const segs = CHEM_SEGMENTS.map((s) => ({ ...s }));
    expect(segmentsComplaint(segs, CHEM_LEN, PLASTIC_BASE, PLASTIC_LEN)).toBeNull();
    expect(segmentsComplaint(segs, CHEM_LEN + 1, PLASTIC_BASE, PLASTIC_LEN)).toMatch(/cover/);
    const gap = segs.map((s, i) => (i === 3 ? { ...s, at: s.at + 1 } : s));
    expect(segmentsComplaint(gap, CHEM_LEN, PLASTIC_BASE, PLASTIC_LEN)).toMatch(/starts at/);
    const dup = segs.map((s, i) => (i === 3 ? { ...s, name: segs[2].name } : s));
    expect(segmentsComplaint(dup, CHEM_LEN, PLASTIC_BASE, PLASTIC_LEN)).toMatch(/twice/);
    expect(segmentsComplaint(segs, CHEM_LEN, PLASTIC_BASE + 1, PLASTIC_LEN)).toMatch(/learned block starts/);
    expect(segmentsComplaint(segs, CHEM_LEN, PLASTIC_BASE, PLASTIC_LEN - 1)).toMatch(/learned block ends/);
  });
});

describe('the header', () => {
  it('carries the map, the learned offset, and whatever provenance it was given', () => {
    const { sim } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    const plain = readHeader(encodeNet(net));
    expect(plain.format).toBe(2);
    expect(plain.segments).toEqual(CHEM_SEGMENTS.map((s) => ({ ...s })));
    expect(plain.plasticAt).toBe(PLASTIC_BASE);
    expect(plain.commit).toBeUndefined();
    expect(plain.written).toBeUndefined();

    const meta = { commit: 'abc1234+dirty', written: '2026-09-09T00:00:00.000Z', source: { run: 7, net: 12, t: 300 }, note: 'hi' };
    const full = readHeader(encodeNet(net, meta));
    expect(full.commit).toBe(meta.commit);
    expect(full.written).toBe(meta.written);
    expect(full.source).toEqual(meta.source);
    expect(full.note).toBe('hi');
  });

  it('encodes the same net to the same bytes when given no provenance', () => {
    const { sim } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    expect([...encodeNet(net)]).toEqual([...encodeNet(net)]);
  });

  it('is exact for a blob this build wrote, and decoding says so', () => {
    const { sim, params } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    const blob = encodeNet(net);
    expect(compatibility(readHeader(blob))).toEqual({ kind: 'exact' });
    const back = decodeNet(blob);
    expect(back.layout).toEqual(currentLayout());
    expect(isCurrent(back)).toBe(true);
    const { net: same, notes } = prepareNet(back, params);
    expect(same).toBe(back);
    expect(notes).toEqual([]);
  });

  it('refuses to encode a net at another layout', () => {
    const { sim } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    const old = decodeNet(asOldBuild(net, NAMES.filter((n) => n !== 'Sw' && n !== 'Gw')).blob);
    expect(() => encodeNet(old)).toThrow(/migrateNet/);
  });
});

describe('migration', () => {
  it('seeds a head this build appended, and carries everything else bit for bit', () => {
    // The case that stranded every net in the library: `G` and `g0` went on
    // the end, the width grew, and nothing a stored body carried had changed.
    // The tail is whatever this build appended last — `Sw` and `Gw` now —
    // so the test follows the layout rather than naming the head of the day.
    const { sim, params } = learningPond(120, 400);
    const net = captureNets(sim)[0].data;
    const { blob, chem } = asOldBuild(
      net,
      NAMES.filter((n) => n !== 'Sw' && n !== 'Gw'),
    );
    expect(chem).toBe(CHEM_LEN - segment('Sw').len - segment('Gw').len);

    const c = compatibility(readHeader(blob));
    expect(c).toEqual({ kind: 'migratable', notes: ['Sw seeded (new in this build)', 'Gw seeded (new in this build)'] });

    const old = decodeNet(blob);
    expect(old.bodies[0].chem.length).toBe(chem);
    expect(isCurrent(old)).toBe(false);

    const { net: now, notes } = prepareNet(old, params);
    expect(notes).toEqual(c.kind === 'migratable' ? c.notes : []);
    expect(isCurrent(now)).toBe(true);
    const G = segment('Sw');
    for (let i = 0; i < net.bodies.length; i++) {
      const was = net.bodies[i];
      const is = now.bodies[i];
      expect(is.chem.length).toBe(CHEM_LEN);
      expect([...is.chem.subarray(0, G.at)]).toEqual([...was.chem.subarray(0, G.at)]);
      // The seeded tail is exactly what a body of this kind born under these
      // params would carry — not zeros, and not what the old body had.
      expect([...is.chem.subarray(G.at)]).toEqual([...seedChem(was.kind, params).subarray(G.at)]);
      expect([...is.plastic]).toEqual([...was.plastic]);
      expect([...is.trace]).toEqual([...was.trace]);
      expect([...is.critic]).toEqual([...was.critic]);
      expect([...is.h]).toEqual([...was.h]);
      expect(is.kind).toBe(was.kind);
      expect(is.born).toBe(was.born);
    }
    // And the migrated net encodes, as a current one.
    expect(compatibility(readHeader(encodeNet(now)))).toEqual({ kind: 'exact' });
  });

  it('puts reordered segments back where this build keeps them', () => {
    const { sim, params } = learningPond(120, 400);
    const net = captureNets(sim)[0].data;
    // A build that kept `ksg` ahead of the locomotion heads.
    const order = NAMES.filter((n) => n !== 'ksg');
    order.splice(order.indexOf('L'), 0, 'ksg');
    const { blob, segments: was } = asOldBuild(net, order);
    const c = compatibility(readHeader(blob));
    expect(c.kind).toBe('migratable');
    if (c.kind !== 'migratable') return;
    const at = (name: string) => was.find((s) => s.name === name)!.at;
    expect(c.notes).toEqual([
      `L moved ${at('L')} -> ${segment('L').at}`,
      `l0 moved ${at('l0')} -> ${segment('l0').at}`,
      `ksg moved ${at('ksg')} -> ${segment('ksg').at}`,
    ]);
    const { net: now } = prepareNet(decodeNet(blob), params);
    for (let i = 0; i < net.bodies.length; i++) {
      expect([...now.bodies[i].chem]).toEqual([...net.bodies[i].chem]);
      expect([...now.bodies[i].plastic]).toEqual([...net.bodies[i].plastic]);
    }
  });

  it('follows the learned block when its segments move', () => {
    const { sim, params } = learningPond(120, 400);
    const net = captureNets(sim)[0].data;
    expect(net.bodies.some((b) => b.plastic.some((v) => v !== 0))).toBe(true);
    // The four learned segments after the heads rather than before them.
    const learned = ['Wx', 'Wh', 'Wn', 'b'];
    const order = [...NAMES.filter((n) => !learned.includes(n)), ...learned];
    const { blob } = asOldBuild(net, order);
    const c = compatibility(readHeader(blob));
    expect(c.kind).toBe('migratable');
    const { net: now } = prepareNet(decodeNet(blob), params);
    for (let i = 0; i < net.bodies.length; i++) {
      expect([...now.bodies[i].chem]).toEqual([...net.bodies[i].chem]);
      expect([...now.bodies[i].plastic]).toEqual([...net.bodies[i].plastic]);
      expect([...now.bodies[i].trace]).toEqual([...net.bodies[i].trace]);
    }
  });

  it('drops a segment this build lost and seeds one it gained', () => {
    const { sim, params } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    const { blob } = asOldBuild(net, NAMES, { ksg: 'affinity' });
    const c = compatibility(readHeader(blob));
    expect(c).toEqual({
      kind: 'migratable',
      notes: ['ksg seeded (new in this build)', 'affinity dropped (gone from this build)'],
    });
    const { net: now } = prepareNet(decodeNet(blob), params);
    const ks = segment('ksg');
    for (let i = 0; i < net.bodies.length; i++) {
      const seed = seedChem(net.bodies[i].kind, params);
      expect([...now.bodies[i].chem.subarray(ks.at, ks.at + ks.len)]).toEqual([...seed.subarray(ks.at, ks.at + ks.len)]);
      expect([...now.bodies[i].chem.subarray(0, ks.at)]).toEqual([...net.bodies[i].chem.subarray(0, ks.at)]);
    }
  });

  it('refuses when the recurrent state changed width', () => {
    const { sim } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    const L = currentLayout();
    const blob = encodeNetAs(net, { ...L, state: L.state + 1, critic: L.critic + 1 }, CHEM_SEGMENTS, PLASTIC_BASE);
    const c = compatibility(readHeader(blob));
    expect(c.kind).toBe('refused');
    if (c.kind === 'refused') expect(c.reason).toMatch(/recurrent state/);
    expect(() => decodeNet(blob)).toThrow(/genome layout has changed/);
  });

  it('refuses a segment that changed length, by name', () => {
    const { sim } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    // Same total width, two floats moved from `Gx` to `Tx`.
    const segs = CHEM_SEGMENTS.map((s) => ({ ...s }));
    const tx = segs.find((s) => s.name === 'Tx')!;
    const gx = segs.find((s) => s.name === 'Gx')!;
    tx.len += 2;
    gx.at += 2;
    gx.len -= 2;
    const blob = encodeNetAs(net, currentLayout(), segs, PLASTIC_BASE);
    const c = compatibility(readHeader(blob));
    expect(c).toEqual({
      kind: 'refused',
      reason: 'genome layout has changed since this net was stored: segment Tx was 6 floats and is 4',
    });
  });

  it('refuses a segment that crossed into or out of the learned block', () => {
    const { sim } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    // `b` out of the learned run and two bases into it, at the same width.
    const order = NAMES.filter((n) => !['b', 'f0', 'l0'].includes(n));
    order.splice(order.indexOf('Wn') + 1, 0, 'f0', 'l0', 'b');
    const { blob } = asOldBuild(net, order);
    const c = compatibility(readHeader(blob));
    expect(c.kind).toBe('refused');
    if (c.kind === 'refused') expect(c.reason).toMatch(/segment b is learned and was not/);
  });

  it('refuses a format 1 blob at any other width, saying why', () => {
    const { sim } = learningPond(60, 120);
    const blob = patchHeader(encodeNet(captureNets(sim)[0].data), (h) => {
      h.format = 1;
      h.layout.chem = CHEM_LEN - 10;
      delete h.segments;
      delete h.plasticAt;
    });
    const c = compatibility(readHeader(blob));
    expect(c.kind).toBe('refused');
    if (c.kind === 'refused') {
      expect(c.reason).toContain(`chem ${CHEM_LEN - 10} -> ${CHEM_LEN}`);
      expect(c.reason).toMatch(/segment map/);
    }
  });

  it('refuses a header whose map does not describe its own genome', () => {
    const { sim } = learningPond(60, 120);
    const blob = patchHeader(encodeNet(captureNets(sim)[0].data), (h) => {
      h.layout.chem = CHEM_LEN + 10;
    });
    const c = compatibility(readHeader(blob));
    expect(c.kind).toBe('refused');
    if (c.kind === 'refused') expect(c.reason).toMatch(/inconsistent/);
  });

  it('refuses a blob from a newer format', () => {
    const { sim } = learningPond(60, 120);
    const blob = patchHeader(encodeNet(captureNets(sim)[0].data), (h) => {
      h.format = 9;
    });
    expect(compatibility(readHeader(blob))).toEqual({ kind: 'refused', reason: "net format 9 is newer than this build's 2" });
  });

  it('is what plantNet does on the way in', () => {
    const { sim, params } = learningPond(120, 400);
    const net = captureNets(sim)[0].data;
    const old = decodeNet(asOldBuild(net, NAMES.filter((n) => n !== 'Sw' && n !== 'Gw')).blob);

    const fresh = new Sim(1600, 1200, 128);
    loadPreset(fresh, 'soup', { ...params, soupCount: 0 });
    const ids = plantNet(fresh, params, old, fresh.w * 0.5, fresh.h * 0.5);
    expect(ids.length).toBe(net.bodies.length);
    const G = segment('Sw');
    const store = fresh.agentStore;
    for (let i = 0; i < ids.length; i++) {
      const a = fresh.agents.get(ids[i])!;
      const chem = store.chemAll.subarray(a.slot * CHEM_LEN, (a.slot + 1) * CHEM_LEN);
      expect([...chem.subarray(0, G.at)]).toEqual([...net.bodies[i].chem.subarray(0, G.at)]);
      expect([...chem.subarray(G.at)]).toEqual([...seedChem(a.kind, params).subarray(G.at)]);
    }
    // And the pond it was planted into runs.
    for (let i = 0; i < 30; i++) fresh.step(1 / 60, params);
  });

  it('seeds through migrateNet with whatever seeder it is handed', () => {
    const { sim } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    const old = decodeNet(asOldBuild(net, NAMES.filter((n) => n !== 'Gw')).blob);
    const marker = new Float32Array(CHEM_LEN).fill(-7);
    const { net: now, notes } = migrateNet(old, () => marker.slice());
    expect(notes).toEqual(['Gw seeded (new in this build)']);
    const g0 = segment('Gw');
    expect([...now.bodies[0].chem.subarray(g0.at)]).toEqual(new Array(g0.len).fill(-7));
    expect(() => migrateNet(old, () => new Float32Array(3))).toThrow(new RegExp(`not ${CHEM_LEN}`));
  });
});

describe('the scalar list', () => {
  /*
   * The genome is not the only append-only list in a blob. `SCALAR_FIELDS`
   * grew an `intake` and the decoder went on reading by fixed index, so
   * every net stored before that field existed came back shifted by one from
   * body one onward, with the tail reading off the end of the section. Both
   * nets in `nets/` are such blobs, and planted they arrived with each
   * other's tanks and pools that were negative or not numbers — which read
   * as the gait being broken.
   *
   * The section has always declared its own field names. These say the
   * decoder uses them.
   */
  function asSixField(net: NetData): Uint8Array {
    const six = SCALAR_FIELDS.filter((f) => f !== 'intake') as ScalarField[];
    expect(six.length).toBe(SCALAR_FIELDS.length - 1);
    return encodeNetAs(net, currentLayout(), CHEM_SEGMENTS, PLASTIC_BASE, {}, six);
  }

  it('reads a blob by the names its section declares, not by index', () => {
    const { sim, params } = learningPond(120, 400);
    const net = captureNets(sim)[0].data;
    const blob = asSixField(net);
    expect(storedScalars(readHeader(blob))).not.toContain('intake');

    const old = decodeNet(blob);
    expect(old.scalars).not.toContain('intake');
    // Every body, not just the first: reading at this build's width shifted
    // everything after body zero, which is why body zero looked fine.
    for (let i = 0; i < net.bodies.length; i++) {
      const was = net.bodies[i];
      const is = old.bodies[i];
      expect(is.extra, `body ${i} extra`).toBe(was.extra);
      expect(is.requestDecay, `body ${i} requestDecay`).toBe(was.requestDecay);
      expect(is.energyCap, `body ${i} energyCap`).toBe(was.energyCap);
      expect(is.debtCap, `body ${i} debtCap`).toBe(was.debtCap);
      expect(is.rescueTo, `body ${i} rescueTo`).toBe(was.rescueTo);
      expect(is.assort, `body ${i} assort`).toBe(was.assort);
      // Not carried, so not invented: NaN is the decoder's marker and
      // `migrateNet` is what seeds it.
      expect(is.intake, `body ${i} intake`).toBeNaN();
    }

    // Migratable, with the note, and seeded to what a fresh body would carry.
    const c = compatibility(readHeader(blob));
    expect(c).toEqual({ kind: 'migratable', notes: ['intake seeded (new in this build)'] });
    const { net: now, notes } = prepareNet(old, params);
    expect(notes).toEqual(['intake seeded (new in this build)']);
    expect(isCurrent(now)).toBe(true);
    for (const b of now.bodies) expect(b.intake).toBe(params.intake);
    // And it encodes again as a current net.
    expect(compatibility(readHeader(encodeNet(now)))).toEqual({ kind: 'exact' });
  });

  it('refuses a scalar name this build cannot place', () => {
    const { sim } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    const blob = patchHeader(encodeNet(net), (h) => {
      const s = h.sections.find((x) => x.name === 'scalar')!;
      s.fields = [...SCALAR_FIELDS.slice(0, SCALAR_FIELDS.length - 1), 'rigor'];
    });
    const c = compatibility(readHeader(blob));
    expect(c.kind).toBe('refused');
    if (c.kind === 'refused') expect(c.reason).toMatch(/scalars this build does not know: rigor/);
  });

  it('plants the checked-in nets with every scalar inside its own gene range', () => {
    // The end of the chain this was found at: the two nets in the library are
    // six-scalar blobs, and before the fix most of their bodies had no usable
    // intake pool, so `advanceGait` skipped them and a planted net stood
    // still whatever the dials said.
    const params = defaultParams();
    for (const name of listFixtures()) {
      const { net } = loadFixture(name, params);
      for (const b of net.bodies) {
        expect(b.intake, `${name} intake`).toBeGreaterThanOrEqual(TRAIT_RANGE.intake.min);
        expect(b.intake, `${name} intake`).toBeLessThanOrEqual(TRAIT_RANGE.intake.max);
        expect(b.energyCap, `${name} energyCap`).toBeGreaterThan(0);
        expect(b.debtCap, `${name} debtCap`).toBeLessThan(0);
        expect(b.requestDecay, `${name} requestDecay`).toBeGreaterThanOrEqual(TRAIT_RANGE.requestDecay.min);
        expect(b.requestDecay, `${name} requestDecay`).toBeLessThanOrEqual(TRAIT_RANGE.requestDecay.max);
      }
      // And with the pathway on, every planted body's clock actually runs.
      const sim = new Sim(1600, 1200, 128);
      loadPreset(sim, 'soup', { ...params, soupCount: 0, spawnInterval: 0 });
      plantNet(sim, params, net, sim.w * 0.5, sim.h * 0.5);
      const on = { ...params, soupCount: 0, spawnInterval: 0, metabolicRate: 15 };
      for (let f = 0; f < 59; f++) sim.step(1 / 60, on);
      // The roster from before the last step. A body a rewrite made during
      // that step has not been through `advanceGait` yet, so it still reads
      // the zero the store resets it to — which is a birth and not a dead
      // clock, and counting it made this assertion a test of whether the net
      // happened to rewrite on the last frame.
      const grown = [...sim.agents.values()].map((a) => a.id);
      sim.step(1 / 60, on);
      const store = sim.agentStore;
      let still = 0;
      for (const id of grown) {
        const a = sim.agents.get(id);
        if (a && store.gaitWave[a.slot] === 0) still++;
      }
      expect(still, `${name}: bodies with a dead pathway`).toBe(0);
    }
  });
});

describe('net files', () => {
  it('round-trips through a file, stamped with when and where from', () => {
    const { sim, params } = learningPond(120, 400);
    const net = captureNets(sim)[0].data;
    const dir = mkdtempSync(join(tmpdir(), 'petri-nets-'));
    try {
      const path = join(dir, 'one.petrinet');
      saveNet(path, net, { commit: 'abc1234', source: { run: 3, net: 9, t: 120 } });
      const { header, compatibility: c } = inspectNetFile(path);
      expect(c).toEqual({ kind: 'exact' });
      expect(header.commit).toBe('abc1234');
      expect(header.written).toMatch(/^\d{4}-\d{2}-\d{2}T/);
      expect(header.source).toEqual({ run: 3, net: 9, t: 120 });
      const loaded = loadNet(path, params);
      expect(loaded.notes).toEqual([]);
      expect(loaded.net.bodies.length).toBe(net.bodies.length);
      expect(loaded.net.wires).toEqual(net.wires);
      expect([...loaded.net.bodies[0].chem]).toEqual([...net.bodies[0].chem]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('brings an old file across on the way in', () => {
    const { sim, params } = learningPond(60, 120);
    const net = captureNets(sim)[0].data;
    const dir = mkdtempSync(join(tmpdir(), 'petri-nets-'));
    try {
      const path = join(dir, 'old.petrinet');
      const { blob } = asOldBuild(net, NAMES.filter((n) => n !== 'Sw' && n !== 'Gw'));
      writeFileSync(path, blob);
      expect(inspectNetFile(path).compatibility.kind).toBe('migratable');
      const loaded = loadNet(path, params);
      expect(loaded.notes.length).toBe(2);
      expect(isCurrent(loaded.net)).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('names a fixture that is not there, and the ones that are', () => {
    expect(() => loadFixture('no-such-net', defaultParams())).toThrow(/no fixture no-such-net/);
    expect(fixturePath('x')).toMatch(/nets[\\/]x\.petrinet$/);
    expect(fixturePath('x.petrinet')).toMatch(/nets[\\/]x\.petrinet$/);
  });
});

describe('the checked-in nets', () => {
  const names = listFixtures();

  // No fixture yet is a state the repository can be in; the test that would
  // read one is then simply not there, rather than red for a missing file.
  if (names.length === 0) it.todo('nets/ has no .petrinet files to plant');

  for (const name of names) {
    it(`${name}: reads on this build, plants, and runs`, () => {
      // Under a megabyte each, so the repository stays a repository.
      expect(statSync(fixturePath(name)).size).toBeLessThan(1 << 20);
      const { compatibility: c } = inspectNetFile(fixturePath(name));
      expect(c.kind).not.toBe('refused');

      const params = defaultParams();
      params.soupCount = 0;
      params.spawnInterval = 0;
      const loaded = loadFixture(name, params);
      const sim = new Sim(1600, 1200, 128);
      loadPreset(sim, 'soup', params);
      const ids = plantNet(sim, params, loaded.net, sim.w * 0.5, sim.h * 0.5, { lineage: -1 });
      expect(ids.length).toBe(loaded.net.bodies.length);
      expect(loaded.net.bodies.length).toBeGreaterThan(2);
      // A grown net, not a founder soup: something in it has bred.
      expect(Math.max(...loaded.net.bodies.map((b) => b.born))).toBeGreaterThan(0);
      for (let i = 0; i < 60; i++) sim.step(1 / 60, params);
      expect(sim.agents.size).toBeGreaterThan(0);
    });
  }
});
