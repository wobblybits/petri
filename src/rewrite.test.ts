import { afterEach, beforeEach, describe, expect, it } from 'vitest';

// Seeded. The oscillator swims, and swimming draws coloured noise from
// Math.random, so an unseeded run is a different trajectory every time — and
// worse, vitest reuses a worker across files, so whichever file ran before
// this one decided the stream. Two of these tests measure rope geometry
// during a commute to within a few pixels, which is not a question an
// arbitrary random walk can answer repeatably.
const realRandom = Math.random;
beforeEach(() => {
  let s = 20260902 >>> 0;
  Math.random = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
});
afterEach(() => {
  Math.random = realRandom;
});
import { bezierLength } from './curve.ts';
import { portLocal, stemRoot, stemWorld } from './agents.ts';
import { segmentsIntersect } from './geom.ts';
import { loadPreset } from './presets.ts';
import {
  applyRewrite,
  beginRewrite,
  BLEND_WIDEN,
  commitRewrite,
  leftoverOf,
  portsConnected,
  rewriteHandoffStems,
  COLLAPSE_START,
  COMMUTE_ACROSS_MIN,
  PULL_END,
  TRAIT_KEYS,
  TRAIT_RANGE,
  type NetSnapshot,
} from './rewrite.ts';
import { EXTRA_CAP, extraCapFor } from './energy.ts';
import { Sim } from './sim.ts';
import { defaultParams, type Params } from './params.ts';
import { buildTopology } from './audio/topology.ts';
import { audio } from './audio/engine.ts';
import type { WorkletInMessage } from './audio/types.ts';
import { wrap, wrapDelta, wrapDeltaVec, wrapDist, wrapMid, nematicDelta } from './wrap.ts';

describe('euclidean vectors', () => {
  it('does not wrap positions', () => {
    expect(wrap(10, 10)).toBe(10);
    expect(wrap(-1, 10)).toBe(-1);
    expect(wrap(21, 10)).toBe(21);
  });

  it('uses a straight delta', () => {
    expect(wrapDelta(1, 9, 10)).toBe(8);
    expect(wrapDelta(9, 1, 10)).toBe(-8);
    expect(wrapDelta(0, 5, 10)).toBe(5);
  });

  it('midpoint and distance on the plane', () => {
    const d = wrapDeltaVec(2, 2, 98, 2, 100, 100);
    expect(d.x).toBe(96);
    expect(d.y).toBe(0);
    expect(wrapDist(2, 2, 98, 2, 100, 100)).toBe(96);
    const m = wrapMid(2, 2, 98, 2, 100, 100);
    expect(m.x).toBe(50);
    expect(m.y).toBe(2);
  });

  it('treats opposite headings as already parallel', () => {
    expect(nematicDelta(0, Math.PI)).toBeCloseTo(0, 5);
    expect(nematicDelta(0, 0.4)).toBeCloseTo(0.4, 5);
    expect(Math.abs(nematicDelta(0, 3))).toBeLessThan(0.2);
    expect(Math.abs(nematicDelta(0, Math.PI / 2))).toBeCloseTo(Math.PI / 2, 5);
  });
});

describe('bezier wire length', () => {
  it('measures a straight cubic as the end-to-end distance', () => {
    const p0 = { x: 0, y: 0 };
    const p1 = { x: 10, y: 0 };
    const p2 = { x: 20, y: 0 };
    const p3 = { x: 30, y: 0 };
    expect(bezierLength(p0, p1, p2, p3)).toBeCloseTo(30, 1);
  });
});

describe('canonical port geometry', () => {
  it('extrudes the principal from the apex along heading', () => {
    const root = stemRoot('con', 'p');
    const tip = portLocal('con', 'p');
    expect(root.y).toBe(0);
    expect(tip.y).toBe(0);
    expect(tip.x).toBeGreaterThan(root.x);
  });

  it('draws aux ports as parallel backward legs on the base, not at vertices', () => {
    const s = 16;
    const halfBase = s * 0.82;
    const lRoot = stemRoot('dup', 'l');
    const rRoot = stemRoot('dup', 'r');
    const lTip = portLocal('dup', 'l');
    const rTip = portLocal('dup', 'r');
    const baseX = -s * 0.55;
    expect(lRoot.x).toBeCloseTo(baseX);
    expect(rRoot.x).toBeCloseTo(baseX);
    expect(lRoot.y).toBeCloseTo(-rRoot.y);
    expect(Math.abs(lRoot.y)).toBeGreaterThan(halfBase * 0.5);
    expect(Math.abs(lRoot.y)).toBeLessThan(halfBase - 1e-6);
    expect(lTip.y).toBeCloseTo(lRoot.y);
    expect(rTip.y).toBeCloseTo(rRoot.y);
    expect(lTip.x).toBeLessThan(lRoot.x);
    expect(rTip.x).toBeLessThan(rRoot.x);
    expect(lTip.x).toBeCloseTo(rTip.x);
  });
});

function net(agents: NetSnapshot['agents'], wires: NetSnapshot['wires']): NetSnapshot {
  return { agents, wires };
}

function expectWire(result: NetSnapshot, a: number, as: 'p' | 'l' | 'r', b: number, bs: 'p' | 'l' | 'r') {
  expect(portsConnected(result, { id: a, slot: as }, { id: b, slot: bs })).toBe(true);
}

describe('Lafont reconnect', () => {
  it('ε–ε annihilates both agents', () => {
    const start = net(
      [
        { id: 1, kind: 'era' },
        { id: 2, kind: 'era' },
      ],
      [{ a: { id: 1, slot: 'p' }, b: { id: 2, slot: 'p' } }],
    );
    const out = applyRewrite(start, 'era-era', 1, 2, 10);
    expect(out.net.agents).toEqual([]);
    expect(out.net.wires).toEqual([]);
  });

  it('δ–δ reconnects left–left and right–right', () => {
    const start = net(
      [
        { id: 1, kind: 'dup' },
        { id: 2, kind: 'dup' },
        { id: 11, kind: 'era' },
        { id: 12, kind: 'era' },
        { id: 13, kind: 'era' },
        { id: 14, kind: 'era' },
      ],
      [
        { a: { id: 1, slot: 'p' }, b: { id: 2, slot: 'p' } },
        { a: { id: 1, slot: 'l' }, b: { id: 11, slot: 'p' } },
        { a: { id: 1, slot: 'r' }, b: { id: 12, slot: 'p' } },
        { a: { id: 2, slot: 'l' }, b: { id: 13, slot: 'p' } },
        { a: { id: 2, slot: 'r' }, b: { id: 14, slot: 'p' } },
      ],
    );
    const out = applyRewrite(start, 'annihilate-dup', 1, 2, 20);
    expect(out.net.agents.map((a) => a.id).sort((a, b) => a - b)).toEqual([11, 12, 13, 14]);
    expectWire(out.net, 11, 'p', 13, 'p');
    expectWire(out.net, 12, 'p', 14, 'p');
    expect(portsConnected(out.net, { id: 11, slot: 'p' }, { id: 14, slot: 'p' })).toBe(false);
  });

  it('γ–γ reconnects with a port swap (left–right)', () => {
    const start = net(
      [
        { id: 1, kind: 'con' },
        { id: 2, kind: 'con' },
        { id: 11, kind: 'era' },
        { id: 12, kind: 'era' },
        { id: 13, kind: 'era' },
        { id: 14, kind: 'era' },
      ],
      [
        { a: { id: 1, slot: 'p' }, b: { id: 2, slot: 'p' } },
        { a: { id: 1, slot: 'l' }, b: { id: 11, slot: 'p' } },
        { a: { id: 1, slot: 'r' }, b: { id: 12, slot: 'p' } },
        { a: { id: 2, slot: 'l' }, b: { id: 13, slot: 'p' } },
        { a: { id: 2, slot: 'r' }, b: { id: 14, slot: 'p' } },
      ],
    );
    const out = applyRewrite(start, 'annihilate-con', 1, 2, 20);
    expect(out.net.agents.map((a) => a.id).sort((a, b) => a - b)).toEqual([11, 12, 13, 14]);
    expectWire(out.net, 11, 'p', 14, 'p');
    expectWire(out.net, 12, 'p', 13, 'p');
    expect(portsConnected(out.net, { id: 11, slot: 'p' }, { id: 13, slot: 'p' })).toBe(false);
  });

  it('ε–γ erases the constructor and hangs two new erasers on leftovers', () => {
    const start = net(
      [
        { id: 1, kind: 'era' },
        { id: 2, kind: 'con' },
        { id: 11, kind: 'era' },
        { id: 12, kind: 'era' },
      ],
      [
        { a: { id: 1, slot: 'p' }, b: { id: 2, slot: 'p' } },
        { a: { id: 2, slot: 'l' }, b: { id: 11, slot: 'p' } },
        { a: { id: 2, slot: 'r' }, b: { id: 12, slot: 'p' } },
      ],
    );
    const out = applyRewrite(start, 'erase', 1, 2, 50);
    expect(out.spawned.map((s) => s.kind)).toEqual(['era', 'era']);
    const ids = out.net.agents.map((a) => a.id).sort((a, b) => a - b);
    expect(ids).toEqual([11, 12, 50, 51]);
    expectWire(out.net, 11, 'p', 50, 'p');
    expectWire(out.net, 12, 'p', 51, 'p');
  });

  it('ε–δ erases the duplicator the same way', () => {
    const start = net(
      [
        { id: 1, kind: 'era' },
        { id: 2, kind: 'dup' },
        { id: 11, kind: 'era' },
        { id: 12, kind: 'era' },
      ],
      [
        { a: { id: 1, slot: 'p' }, b: { id: 2, slot: 'p' } },
        { a: { id: 2, slot: 'l' }, b: { id: 11, slot: 'p' } },
        { a: { id: 2, slot: 'r' }, b: { id: 12, slot: 'p' } },
      ],
    );
    const out = applyRewrite(start, 'erase', 1, 2, 7);
    expect(out.spawned).toHaveLength(2);
    expectWire(out.net, 11, 'p', 7, 'p');
    expectWire(out.net, 12, 'p', 8, 'p');
  });

  it('γ–δ commutation copies into two cons and two dups with the square wiring', () => {
    const start = net(
      [
        { id: 1, kind: 'con' },
        { id: 2, kind: 'dup' },
        { id: 11, kind: 'era' },
        { id: 12, kind: 'era' },
        { id: 13, kind: 'era' },
        { id: 14, kind: 'era' },
      ],
      [
        { a: { id: 1, slot: 'p' }, b: { id: 2, slot: 'p' } },
        { a: { id: 1, slot: 'l' }, b: { id: 11, slot: 'p' } },
        { a: { id: 1, slot: 'r' }, b: { id: 12, slot: 'p' } },
        { a: { id: 2, slot: 'l' }, b: { id: 13, slot: 'p' } },
        { a: { id: 2, slot: 'r' }, b: { id: 14, slot: 'p' } },
      ],
    );
    const out = applyRewrite(start, 'commute', 1, 2, 100);
    const Cu = out.spawned.find((s) => s.role === 'con-u')!;
    const Cv = out.spawned.find((s) => s.role === 'con-v')!;
    const Dx = out.spawned.find((s) => s.role === 'dup-x')!;
    const Dy = out.spawned.find((s) => s.role === 'dup-y')!;
    expect(out.net.agents).toHaveLength(8);
    expectWire(out.net, Cu.id, 'p', 13, 'p');
    expectWire(out.net, Cv.id, 'p', 14, 'p');
    expectWire(out.net, Dx.id, 'p', 11, 'p');
    expectWire(out.net, Dy.id, 'p', 12, 'p');
    expectWire(out.net, Cu.id, 'l', Dx.id, 'l');
    expectWire(out.net, Cu.id, 'r', Dy.id, 'l');
    expectWire(out.net, Cv.id, 'l', Dx.id, 'r');
    expectWire(out.net, Cv.id, 'r', Dy.id, 'r');
  });

  it('treats leftovers on dying agents as free', () => {
    const wires = [
      { a: { id: 1, slot: 'l' as const }, b: { id: 2, slot: 'r' as const } },
    ];
    expect(leftoverOf(wires, 1, 'l', new Set([1, 2]))).toBeNull();
  });
});

describe('annihilation staging', () => {
  function annihilating(): { sim: Sim; params: Params; a: number; b: number } {
    const sim = new Sim(800, 600);
    const params = defaultParams();
    params.spawnInterval = 0;
    const a = sim.spawn('era', 380, 300, 0, params, true)!;
    const b = sim.spawn('era', 460, 300, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    for (let f = 0; f < 900 && sim.rewrites.length === 0; f++) sim.step(1 / 60, params);
    expect(sim.rewrites.length).toBe(1);
    return { sim, params, a: a.id, b: b.id };
  }

  it('holds full size and opacity until the bodies have met', () => {
    const { sim, params, a } = annihilating();
    let sawPull = false;
    for (let f = 0; f < 200; f++) {
      const rw = sim.rewrites[0];
      if (!rw) break;
      const A = sim.agents.get(a);
      if (!A) break;
      if (rw.t < COLLAPSE_START) {
        // Nothing has happened yet, so nothing should look like it has.
        expect(A.scale).toBe(1);
        expect(A.alpha).toBe(1);
        if (rw.t > 0.2) sawPull = true;
      }
      sim.step(1 / 60, params);
    }
    expect(sawPull).toBe(true);
  });

  it('closes the gap by the end of the pull, and only then collapses', () => {
    const { sim, params, a, b } = annihilating();
    let gapAtPull = Infinity;
    let scaleAtEnd = 1;
    for (let f = 0; f < 200; f++) {
      const rw = sim.rewrites[0];
      if (!rw) break;
      const A = sim.agents.get(a);
      const B = sim.agents.get(b);
      if (A && B && rw.t >= PULL_END && gapAtPull === Infinity) {
        gapAtPull = Math.hypot(A.x - B.x, A.y - B.y);
      }
      if (A && rw.t > 0.95) scaleAtEnd = A.scale;
      sim.step(1 / 60, params);
    }
    expect(gapAtPull).toBeLessThan(2);
    expect(scaleAtEnd).toBeLessThan(0.2);
  });

  it('keeps the rope on the chord instead of whipping it', () => {
    // The bodies move kinematically, so the solver never sees the motion that
    // should drag the rope along. Unreeled, its arc length climbed past 80 px
    // between two bodies already touching.
    const { sim, params } = annihilating();
    let worst = 0;
    for (let f = 0; f < 200; f++) {
      const rw = sim.rewrites[0];
      if (!rw) break;
      const wire = sim.graph.wires.get(rw.wireId);
      if (wire) worst = Math.max(worst, wire.lastLen);
      sim.step(1 / 60, params);
    }
    expect(worst).toBeLessThan(60);
  });

  it('lets the wire tighten and rise, without a multi-octave squeal', () => {
    const { sim, params } = annihilating();
    const pitches: number[] = [];
    for (let f = 0; f < 200; f++) {
      const rw = sim.rewrites[0];
      if (!rw) break;
      const w = buildTopology(sim.graph, sim.agents, null).wires[0];
      if (w) pitches.push(48000 / (2 * w.length));
      sim.step(1 / 60, params);
    }
    const first = pitches[0];
    const top = Math.max(...pitches);
    expect(top).toBeGreaterThan(first * 1.5);
    expect(top).toBeLessThan(first * 3.2);
  });

  it('carries the pair along with the soup instead of pinning it', () => {
    const { sim, params, a, b } = annihilating();
    const A0 = sim.agents.get(a)!;
    const B0 = sim.agents.get(b)!;
    const rw = sim.rewrites[0];
    // Give the pair a shared drift the way the flock would.
    A0.vx = B0.vx = 40;
    A0.vy = B0.vy = 0;
    rw.vx = 40;
    rw.vy = 0;
    const startMidX = (A0.x + B0.x) * 0.5;
    for (let f = 0; f < 20; f++) sim.step(1 / 60, params);
    const A = sim.agents.get(a);
    const B = sim.agents.get(b);
    if (!A || !B) return;
    expect((A.x + B.x) * 0.5).toBeGreaterThan(startMidX + 4);
  });
});

describe('annihilation contact', () => {
  /** The sim posts to the module singleton, not to an engine you construct. */
  function capture(run: (sim: Sim, params: Params) => void): WorkletInMessage[] {
    const posted: WorkletInMessage[] = [];
    const prevPost = audio.onPost;
    audio.armWithoutAudio();
    audio.onPost = (m) => posted.push(m);
    try {
      const sim = new Sim(800, 600);
      const params = defaultParams();
      params.spawnInterval = 0;
      run(sim, params);
    } finally {
      audio.onPost = prevPost;
    }
    return posted;
  }

  it('knocks when the two bodies meet', () => {
    const posted = capture((sim, params) => {
      const a = sim.spawn('era', 380, 300, 0, params, true)!;
      const b = sim.spawn('era', 460, 300, Math.PI, params, true)!;
      sim.wire(a.id, 'p', b.id, 'p', params);
      for (let f = 0; f < 900; f++) {
        sim.step(1 / 60, params);
        audio.frame(sim.graph, sim.agents, 1 / 60, null);
      }
    });
    const strikes = posted.filter((m) => m.type === 'strike');
    // One per body. The pair is locked and moved kinematically, so the
    // ordinary contact path never sees them touch — this is the only thing
    // that makes the meeting audible.
    expect(strikes.length).toBeGreaterThanOrEqual(2);
    for (const s of strikes) {
      expect(s.type === 'strike' && s.peak).toBeGreaterThan(0);
      expect(s.type === 'strike' && s.peak).toBeLessThanOrEqual(6);
    }
  });

  it('fires once, on the onset, not every frame after contact', () => {
    // Count only strikes posted while a rewrite is running, so the ordinary
    // collision the pair makes on its way to latching is not mistaken for it.
    let duringRewrite = 0;
    const prevPost = audio.onPost;
    audio.armWithoutAudio();
    const sim = new Sim(800, 600);
    const params = defaultParams();
    params.spawnInterval = 0;
    let live = false;
    audio.onPost = (m) => {
      if (live && m.type === 'strike') duringRewrite++;
    };
    try {
      const a = sim.spawn('era', 380, 300, 0, params, true)!;
      const b = sim.spawn('era', 460, 300, Math.PI, params, true)!;
      sim.wire(a.id, 'p', b.id, 'p', params);
      for (let f = 0; f < 900; f++) {
        sim.step(1 / 60, params);
        live = sim.rewrites.length > 0;
        audio.frame(sim.graph, sim.agents, 1 / 60, null);
      }
    } finally {
      audio.onPost = prevPost;
    }
    // One per body, on the frame the gap closes. Not one a frame for the rest
    // of the collapse.
    expect(duringRewrite).toBe(2);
  });

  it('rewrites a principal meeting without a dressed net', () => {
    const sim = new Sim(800, 600);
    const params = defaultParams();
    params.spawnInterval = 0;
    const a = sim.spawn('dup', 380, 300, 0, params, true)!;
    const b = sim.spawn('dup', 460, 300, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    expect(sim.graph.portsFilled(a)).toBe(false);
    expect(sim.graph.portsFilled(b)).toBe(false);
    for (let f = 0; f < 180 && sim.rewrites.length === 0 && sim.agents.size === 2; f++) {
      sim.step(1 / 60, params);
    }
    expect(sim.rewrites.length > 0 || sim.agents.size !== 2).toBe(true);
  });
});

describe('commute birth layout', () => {
  function oscillatorPair(): { sim: Sim; params: Params } {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    return { sim, params };
  }

  function triangleBbox(xs: number[], ys: number[]): { w: number; h: number } {
    return {
      w: Math.max(...xs) - Math.min(...xs),
      h: Math.max(...ys) - Math.min(...ys),
    };
  }

  function wireCrossings(sim: Sim, trianglesOnly = false): number {
    const chords: { ax: number; ay: number; bx: number; by: number }[] = [];
    for (const w of sim.graph.wires.values()) {
      const A = sim.agents.get(w.a.id);
      const B = sim.agents.get(w.b.id);
      if (!A || !B) continue;
      if (trianglesOnly && (A.kind === 'era' || B.kind === 'era')) continue;
      const sa = stemWorld(A, w.a.slot, sim.w, sim.h);
      const sb = stemWorld(B, w.b.slot, sim.w, sim.h);
      chords.push({ ax: sa.x, ay: sa.y, bx: sb.x, by: sb.y });
    }
    let n = 0;
    for (let i = 0; i < chords.length; i++) {
      for (let j = i + 1; j < chords.length; j++) {
        const a = chords[i];
        const b = chords[j];
        if (segmentsIntersect(a.ax, a.ay, a.bx, a.by, b.ax, b.ay, b.bx, b.by)) n++;
      }
    }
    return n;
  }

  it('opens the oscillator commute into a rectangle, not a pancake', () => {
    const { sim } = oscillatorPair();
    const dup = [...sim.agents.values()].find((a) => a.kind === 'dup')!;
    const con = [...sim.agents.values()].find((a) => a.kind === 'con')!;
    const rw = beginRewrite(dup, con, sim.graph, sim.agents, sim.w, sim.h, 1);
    expect(rw.targets).toHaveLength(4);
    const box = triangleBbox(
      rw.targets.map((g) => g.x),
      rw.targets.map((g) => g.y),
    );
    expect(box.h).toBeGreaterThanOrEqual(COMMUTE_ACROSS_MIN * 2 - 1);
    expect(box.w).toBeGreaterThan(box.h * 0.6);
    const eras = [...sim.agents.values()].filter((a) => a.kind === 'era');
    const minX = Math.min(...rw.targets.map((g) => g.x));
    const maxX = Math.max(...rw.targets.map((g) => g.x));
    const minY = Math.min(...rw.targets.map((g) => g.y));
    const maxY = Math.max(...rw.targets.map((g) => g.y));
    for (const era of eras) {
      const inside =
        era.x > minX + 1 && era.x < maxX - 1 && era.y > minY + 1 && era.y < maxY - 1;
      expect(inside).toBe(false);
    }
  });

  it('commits one interior crossing, not a stack of chords', () => {
    const { sim, params } = oscillatorPair();
    let committed = false;
    for (let f = 0; f < 300; f++) {
      const nRw = sim.rewrites.length;
      sim.step(1 / 60, params);
      if (nRw > 0 && sim.rewrites.length === 0) {
        committed = true;
        break;
      }
    }
    expect(committed).toBe(true);
    const triangles = [...sim.agents.values()].filter((a) => a.kind !== 'era');
    expect(triangles).toHaveLength(4);
    const box = triangleBbox(
      triangles.map((a) => a.x),
      triangles.map((a) => a.y),
    );
    expect(box.h).toBeGreaterThanOrEqual(COMMUTE_ACROSS_MIN * 2 - 8);
    expect(wireCrossings(sim, true)).toBe(1);
  });
});

describe('heritable traits', () => {
  function oscillatorPair(): { sim: Sim; params: Params } {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    return { sim, params };
  }

  it('recombines both parents into every commute child, whichever mechanism its kind uses', () => {
    const { sim, params } = oscillatorPair();
    const dup = [...sim.agents.values()].find((a) => a.kind === 'dup')!;
    const con = [...sim.agents.values()].find((a) => a.kind === 'con')!;
    dup.requestDecay = 0.55;
    con.requestDecay = 0.95;
    dup.conductSpeed = 5;
    con.conductSpeed = 55;

    const before = new Set(sim.agents.keys());
    const rw = beginRewrite(dup, con, sim.graph, sim.agents, sim.w, sim.h, 1);
    sim.nextId = commitRewrite(
      rw,
      sim.agents,
      sim.graph,
      params,
      sim.nextId,
      sim.time,
      sim.w,
      sim.h,
      sim.agentStore,
    );

    const children = [...sim.agents.values()].filter((a) => !before.has(a.id));
    expect(children).toHaveLength(4);
    for (const c of children) {
      expect(c.requestDecay).toBeGreaterThanOrEqual(0.5);
      expect(c.requestDecay).toBeLessThanOrEqual(0.98);
      expect(c.conductSpeed).toBeGreaterThanOrEqual(0);
      expect(c.conductSpeed).toBeLessThanOrEqual(200);
    }
    // Neither parent's exact value survives untouched, and the four siblings
    // do not all land on the same blend of the same two numbers.
    const decays = new Set(children.map((c) => c.requestDecay));
    expect(decays.size, 'siblings recombine independently').toBeGreaterThan(1);
    expect(children.some((c) => c.requestDecay === dup.requestDecay)).toBe(false);
    expect(children.some((c) => c.requestDecay === con.requestDecay)).toBe(false);
  });

  it('assorts a Dup child from one whole parent per trait, instead of blending like a Con child', () => {
    // A Dup duplicates a value; it does not combine two of them. So unlike a
    // Con child — which can land anywhere between its parents — a Dup child's
    // own trait has to sit within mutation range of one parent's *exact*
    // value, on every trait, no matter which way the coin fell.
    const { sim, params } = oscillatorPair();
    const dup = [...sim.agents.values()].find((a) => a.kind === 'dup')!;
    const con = [...sim.agents.values()].find((a) => a.kind === 'con')!;
    dup.requestDecay = 0.55;
    con.requestDecay = 0.95;
    dup.energyCap = EXTRA_CAP * 0.6;
    con.energyCap = EXTRA_CAP * 1.8;
    dup.rescueTo = 0.05;
    con.rescueTo = 0.95;
    dup.conductSpeed = 10;
    con.conductSpeed = 190;

    const before = new Set(sim.agents.keys());
    const rw = beginRewrite(dup, con, sim.graph, sim.agents, sim.w, sim.h, 1);
    sim.nextId = commitRewrite(
      rw,
      sim.agents,
      sim.graph,
      params,
      sim.nextId,
      sim.time,
      sim.w,
      sim.h,
      sim.agentStore,
    );

    const children = [...sim.agents.values()].filter((a) => !before.has(a.id));
    const dupChildren = children.filter((a) => a.kind === 'dup');
    const conChildren = children.filter((a) => a.kind === 'con');
    expect(dupChildren).toHaveLength(2);
    expect(conChildren).toHaveLength(2);

    for (const key of TRAIT_KEYS) {
      const mutate = TRAIT_RANGE[key].mutate;
      for (const c of dupChildren) {
        const nearDup = Math.abs(c[key] - dup[key]) <= mutate + 1e-9;
        const nearCon = Math.abs(c[key] - con[key]) <= mutate + 1e-9;
        expect(
          nearDup || nearCon,
          `dup child's ${key} = ${c[key]} should copy one whole parent (dup ${dup[key]}, con ${con[key]})`,
        ).toBe(true);
      }
      /*
       * A blended Con child is not pinned to either endpoint, and is not
       * confined between them either. `blend` is BLX-alpha: it draws from the
       * parents' interval widened by `BLEND_WIDEN` on each side, precisely so
       * that a blending lineage can explore past the range its ancestors
       * spanned and so that offspring variance does not collapse toward the
       * mean every generation. Then a mutation nudge on top, then the trait's
       * own clamp.
       */
      const span = Math.abs(dup[key] - con[key]);
      const lo = Math.max(
        TRAIT_RANGE[key].min,
        Math.min(dup[key], con[key]) - BLEND_WIDEN * span - mutate,
      );
      const hi = Math.min(
        TRAIT_RANGE[key].max,
        Math.max(dup[key], con[key]) + BLEND_WIDEN * span + mutate,
      );
      for (const c of conChildren) {
        expect(c[key]).toBeGreaterThanOrEqual(lo);
        expect(c[key]).toBeLessThanOrEqual(hi);
      }
    }
  });

  /*
   * Assortment is a trait now, not a fact about the calculus.
   *
   * The rule was `child.kind === 'dup'` — absolute, and the one thing about a
   * body that could never evolve, in a system whose premise is that nothing
   * should be true by fiat. Blending and assortment differ sharply in how fast
   * variance is lost and how easily two lines can pull apart, which is exactly
   * the sort of question selection is for.
   *
   * `assortChance` takes the parents' mean and offsets it by the child's kind,
   * so the seeded 0.5 reproduces the old rule exactly and nothing moves until
   * the trait drifts. The test above this one is the proof of that: it asserts
   * the old behaviour and still passes untouched.
   */
  it('assorts every gene when the lineage has drifted all the way up', () => {
    const { sim, params } = oscillatorPair();
    const dup = [...sim.agents.values()].find((a) => a.kind === 'dup')!;
    const con = [...sim.agents.values()].find((a) => a.kind === 'con')!;
    dup.requestDecay = 0.55;
    con.requestDecay = 0.95;
    // 1.0 on both parents puts a Con child at 0.5 and a Dup child at 1 — so
    // every Dup gene is copied whole, and half of a Con's are.
    dup.assort = 1;
    con.assort = 1;

    const before = new Set(sim.agents.keys());
    const rw = beginRewrite(dup, con, sim.graph, sim.agents, sim.w, sim.h, 1);
    sim.nextId = commitRewrite(rw, sim.agents, sim.graph, params, sim.nextId, sim.time, sim.w, sim.h, sim.agentStore);
    const children = [...sim.agents.values()].filter((a) => !before.has(a.id));
    const mutate = TRAIT_RANGE.requestDecay.mutate;
    for (const c of children.filter((a) => a.kind === 'dup')) {
      const near =
        Math.abs(c.requestDecay - dup.requestDecay) <= mutate + 1e-9 ||
        Math.abs(c.requestDecay - con.requestDecay) <= mutate + 1e-9;
      expect(near, `dup child ${c.requestDecay} should copy a parent whole`).toBe(true);
    }
  });

  it('blends a Dup child too, once the lineage has drifted all the way down', () => {
    /*
     * The other end, and the one the old rule could not express at all: a
     * lineage for which duplication can combine rather than only copy.
     *
     * Stated as a comparison between the two ends, and measured over many
     * rewrites, because neither is true of a single child. `assortChance`
     * offsets the parents' mean by the child's kind — `mean + 0.5` for a Dup —
     * so parents at 0 give a Dup child a chance of **0.5**, not 0: about half
     * its genes are still copied whole. Asserting off one rewrite that both
     * children blended was therefore a coin flip landing right, and it duly
     * came up tails the first time `CHEM_LEN` grew and `inheritChem` started
     * drawing more numbers per child.
     *
     * What is actually true, and is what the rule is for: at the top of the
     * range a Dup child never blends, and at the bottom it often does.
     */
    const m = TRAIT_RANGE.requestDecay.mutate;
    const blendedShare = (assort: number): number => {
      let children = 0;
      let blended = 0;
      for (let trial = 0; trial < 30; trial++) {
        const { sim, params } = oscillatorPair();
        const dup = [...sim.agents.values()].find((a) => a.kind === 'dup')!;
        const con = [...sim.agents.values()].find((a) => a.kind === 'con')!;
        dup.requestDecay = 0.55;
        con.requestDecay = 0.95;
        dup.assort = assort;
        con.assort = assort;

        const before = new Set(sim.agents.keys());
        const rw = beginRewrite(dup, con, sim.graph, sim.agents, sim.w, sim.h, 1);
        sim.nextId = commitRewrite(rw, sim.agents, sim.graph, params, sim.nextId, sim.time, sim.w, sim.h, sim.agentStore);
        const dups = [...sim.agents.values()]
          .filter((a) => !before.has(a.id))
          .filter((a) => a.kind === 'dup');
        expect(dups).toHaveLength(2);
        for (const c of dups) {
          children++;
          const pinned =
            Math.abs(c.requestDecay - 0.55) <= m + 1e-9 || Math.abs(c.requestDecay - 0.95) <= m + 1e-9;
          if (!pinned) blended++;
        }
      }
      return blended / children;
    };

    const atTop = blendedShare(1);
    const atBottom = blendedShare(0);
    // Copied whole, every gene, every child: the mutation nudge is the only
    // thing that moves a Dup child at all up here.
    expect(atTop, `assort 1 blended ${atTop}`).toBeLessThan(0.05);
    // And down here it is a real fraction. Half the genes are still assorted
    // and a blend can land near a parent or on the range's clamp, so this is
    // nothing like 1 — the claim is that it is not nothing.
    expect(atBottom, `assort 0 blended ${atBottom}`).toBeGreaterThan(0.2);
  });

  it('clones an Era through erase, with a mutation nudge', () => {
    const sim = new Sim(320, 240);
    const params = defaultParams();
    const era = sim.spawn('era', 100, 100, 0, params, true)!;
    const bin = sim.spawn('con', 130, 100, Math.PI, params, true)!;
    sim.wire(era.id, 'p', bin.id, 'p', params);
    era.requestDecay = 0.55;
    era.debtCap = -1.8;
    era.rescueTo = 0.2;
    era.energyCap = EXTRA_CAP * 1.6;
    const parentDecay = era.requestDecay;
    const rw = beginRewrite(era, bin, sim.graph, sim.agents, sim.w, sim.h, 1);
    sim.nextId = commitRewrite(
      rw,
      sim.agents,
      sim.graph,
      params,
      sim.nextId,
      sim.time,
      sim.w,
      sim.h,
      sim.agentStore,
    );
    const spawned = [...sim.agents.values()].filter((a) => a.kind === 'era');
    expect(spawned).toHaveLength(2);
    for (const e of spawned) {
      expect(e.debtCap).toBeLessThan(0);
      expect(e.rescueTo).toBeGreaterThanOrEqual(0);
      expect(e.rescueTo).toBeLessThanOrEqual(1);
      // Near the Era parent, not the slider default, on every trait that
      // parent had moved off the seed.
      const decayRange = TRAIT_RANGE.requestDecay;
      expect(e.requestDecay).toBeGreaterThanOrEqual(parentDecay - decayRange.mutate - 1e-9);
      expect(e.requestDecay).toBeLessThanOrEqual(parentDecay + decayRange.mutate + 1e-9);
      expect(e.requestDecay, 'must not reset to the slider').not.toBe(params.requestDecay);
    }
  });

  it('skips trait inheritance when breeding is off', () => {
    const { sim, params } = oscillatorPair();
    const dup = [...sim.agents.values()].find((a) => a.kind === 'dup')!;
    const con = [...sim.agents.values()].find((a) => a.kind === 'con')!;
    dup.requestDecay = 0.55;
    con.requestDecay = 0.95;
    dup.energyCap = EXTRA_CAP * 0.6;
    con.energyCap = EXTRA_CAP * 1.8;
    const before = new Set(sim.agents.keys());
    const rw = beginRewrite(dup, con, sim.graph, sim.agents, sim.w, sim.h, 1);
    sim.nextId = commitRewrite(
      rw,
      sim.agents,
      sim.graph,
      params,
      sim.nextId,
      sim.time,
      sim.w,
      sim.h,
      sim.agentStore,
      false,
    );
    const children = [...sim.agents.values()].filter((a) => !before.has(a.id));
    expect(children).toHaveLength(4);
    for (const c of children) {
      expect(c.requestDecay).toBe(params.requestDecay);
      expect(c.energyCap).toBe(extraCapFor(c.kind));
    }
  });
});

describe('rewrite leftover ropes', () => {
  function incidentWires(sim: Sim, a: number, b: number) {
    return [...sim.graph.wires.values()].filter((w) => {
      const ids = [w.a.id, w.b.id];
      return ids.includes(a) || ids.includes(b);
    });
  }

  function chordSpan(sim: Sim, wire: { a: { id: number; slot: 'p' | 'l' | 'r' }; b: { id: number; slot: 'p' | 'l' | 'r' } }): number {
    const A = sim.agents.get(wire.a.id);
    const B = sim.agents.get(wire.b.id);
    if (!A || !B) return 0;
    const sa = stemWorld(A, wire.a.slot, sim.w, sim.h);
    const sb = stemWorld(B, wire.b.slot, sim.w, sim.h);
    return Math.hypot(sb.x - sa.x, sb.y - sa.y);
  }

  it('keeps oscillator leftovers on their chords instead of whipping them', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    let rewrites = 0;
    let worstRatio = 0;
    for (let f = 0; f < 400; f++) {
      const nRw = sim.rewrites.length;
      sim.step(1 / 60, params);
      if (nRw === 0 && sim.rewrites.length > 0) rewrites++;
      for (const rw of sim.rewrites) {
        for (const wire of incidentWires(sim, rw.a, rw.b)) {
          const handoff = rewriteHandoffStems(rw, wire, sim.agents, sim.w, sim.h);
          const span = handoff
            ? Math.hypot(handoff.bx - handoff.ax, handoff.by - handoff.ay)
            : chordSpan(sim, wire);
          if (span > 1) worstRatio = Math.max(worstRatio, wire.lastLen / span);
        }
      }
      if (rewrites >= 2 && sim.rewrites.length === 0) break;
    }
    expect(rewrites).toBeGreaterThanOrEqual(1);
    // A ratio, not a pixel count. How much rope a leftover carries scales with
    // the wire it came from, and energy changed *when* a commute can fire —
    // this net now reaches its second one around frame 115 with a 30 px chord
    // rather than frame 20 with a 7 px one. The bow is the same shape either
    // way: measured 2.67x the chord with the energy economy on and 2.67x with
    // it switched off entirely. A whip is an order of magnitude, not a third.
    expect(worstRatio, `leftover rope ${worstRatio.toFixed(2)}x its chord`).toBeLessThan(4);
  });

  it('does not haul leftover eras into the collapsing pair', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    // Measured across the rewrite, not from the start of the run. The eras
    // swim on their own the whole time, and energy moved the first commute
    // from about frame 20 to about frame 115, so drift from t=0 would be
    // mostly ninety frames of ordinary foraging. What this test is about is
    // whether the collapsing pair *hauls* them, which is a question about the
    // window the rewrite is open.
    let era0: { id: number; x: number; y: number; toPair: number }[] = [];
    let mid = { x: 0, y: 0 };
    for (let f = 0; f < 300; f++) {
      const nRw = sim.rewrites.length;
      sim.step(1 / 60, params);
      if (nRw === 0 && sim.rewrites.length > 0) {
        const rw = sim.rewrites[0];
        mid = { x: rw.midX, y: rw.midY };
        era0 = [...sim.agents.values()]
          .filter((a) => a.kind === 'era')
          .map((a) => ({
            id: a.id,
            x: a.x,
            y: a.y,
            toPair: Math.hypot(a.x - mid.x, a.y - mid.y),
          }));
      }
      if (nRw > 0 && sim.rewrites.length === 0) break;
    }
    expect(era0.length, 'a rewrite has to have started').toBeGreaterThan(0);
    // How much *closer to the pair* they got, not how far they moved. The eras
    // forage the whole time the rewrite is open, and that motion is isotropic;
    // hauling is the directed part. Measuring total displacement counted a
    // swim away from the pair as if it were a pull toward it.
    let pulled = 0;
    for (const e of era0) {
      const a = sim.agents.get(e.id);
      if (!a) continue;
      pulled = Math.max(pulled, e.toPair - Math.hypot(a.x - mid.x, a.y - mid.y));
    }
    expect(pulled, `worst era pulled ${pulled.toFixed(1)} px toward the pair`).toBeLessThan(20);
  });

  it('hands leftover chords to the birth poses instead of jumping at commit', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    const eraIds = [...sim.agents.values()].filter((a) => a.kind === 'era').map((a) => a.id);
    let pre: { era: number; len: number }[] = [];
    for (let f = 0; f < 300; f++) {
      const nRw = sim.rewrites.length;
      sim.step(1 / 60, params);
      if (sim.rewrites.length) {
        const rw = sim.rewrites[0];
        pre = [];
        for (const wire of incidentWires(sim, rw.a, rw.b)) {
          const ends = [wire.a.id, wire.b.id];
          const era = eraIds.find((id) => ends.includes(id));
          if (era === undefined) continue;
          const handoff = rewriteHandoffStems(rw, wire, sim.agents, sim.w, sim.h);
          const len = handoff
            ? Math.hypot(handoff.bx - handoff.ax, handoff.by - handoff.ay)
            : wire.lastLen;
          pre.push({ era, len });
        }
      }
      if (nRw > 0 && sim.rewrites.length === 0) break;
    }
    expect(pre.length).toBeGreaterThan(0);
    for (const p of pre) {
      const era = sim.agents.get(p.era);
      expect(era).toBeTruthy();
      const wire = [...sim.graph.wires.values()].find(
        (w) => w.a.id === p.era || w.b.id === p.era,
      );
      expect(wire).toBeTruthy();
      const A = sim.agents.get(wire!.a.id)!;
      const B = sim.agents.get(wire!.b.id)!;
      const sa = stemWorld(A, wire!.a.slot, sim.w, sim.h);
      const sb = stemWorld(B, wire!.b.slot, sim.w, sim.h);
      const span = Math.hypot(sb.x - sa.x, sb.y - sa.y);
      expect(Math.abs(span - p.len), `leftover jump ${span.toFixed(1)} vs ${p.len.toFixed(1)}`).toBeLessThan(18);
    }
  });

  it('does not fire latch audio for rewrite-born wires', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    const latches: number[] = [];
    const prev = sim.graph.onLatch;
    sim.graph.onLatch = (ev) => {
      latches.push(ev.wireId);
      prev?.(ev);
    };
    let sawRewrite = false;
    for (let f = 0; f < 300; f++) {
      const nRw = sim.rewrites.length;
      sim.step(1 / 60, params);
      if (nRw > 0) sawRewrite = true;
      if (sawRewrite && sim.rewrites.length === 0) break;
    }
    expect(sawRewrite).toBe(true);
    expect(latches).toHaveLength(0);
  });

  it('keeps leftover wire ids across an oscillator commute', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    const dup = [...sim.agents.values()].find((a) => a.kind === 'dup')!;
    const con = [...sim.agents.values()].find((a) => a.kind === 'con')!;
    const principal = [...sim.graph.wires.values()].find(
      (w) =>
        (w.a.id === dup.id && w.a.slot === 'p' && w.b.id === con.id && w.b.slot === 'p') ||
        (w.b.id === dup.id && w.b.slot === 'p' && w.a.id === con.id && w.a.slot === 'p'),
    )!;
    const leftovers = [...sim.graph.wires.values()]
      .filter((w) => w.id !== principal.id)
      .map((w) => w.id);
    expect(leftovers.length).toBe(3);
    let committed = false;
    for (let f = 0; f < 300; f++) {
      const nRw = sim.rewrites.length;
      sim.step(1 / 60, params);
      if (nRw > 0 && sim.rewrites.length === 0) {
        committed = true;
        break;
      }
    }
    expect(committed).toBe(true);
    expect(sim.graph.wires.has(principal.id)).toBe(false);
    for (const id of leftovers) {
      expect(sim.graph.wires.has(id), `leftover ${id} was reminted`).toBe(true);
    }
  });
});
