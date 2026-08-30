import { describe, expect, it } from 'vitest';
import { bezierLength } from './curve.ts';
import { portLocal, stemRoot } from './agents.ts';
import {
  applyRewrite,
  leftoverOf,
  portsConnected,
  COLLAPSE_START,
  PULL_END,
  type NetSnapshot,
} from './rewrite.ts';
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
    expect(worst).toBeLessThan(45);
  });

  it('lets the wire tighten and rise, but not past an octave', () => {
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
    expect(top).toBeLessThan(first * 2.3);
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
});
