import { describe, expect, it } from 'vitest';
import { render, wireStrokePoints, type ViewOpts } from './render.ts';
import { Camera } from './camera.ts';
import { stemWorld } from './agents.ts';
import { rewriteHandoffStems } from './rewrite.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * `drawWires` draws the chord case out of a reused buffer rather than minting
 * a polyline per wire, and puts every wire in one path instead of stroking
 * each. Both are meant to be invisible, so the guard is the path itself:
 * capture what the canvas is told to do and compare it against the geometry
 * `wireStrokePoints` -- which is untouched, and which the length tests already
 * pin -- says each wire should have.
 */
interface Call {
  op: string;
  x: number;
  y: number;
}

/**
 * Just the wire pass. Wires are drawn before ghosts and bodies and now end in
 * a single `stroke`, so everything up to the first one is theirs -- and
 * `drawAgent` puts moveTo/lineTo on the canvas too, which would otherwise land
 * in the comparison.
 */
function wirePass(calls: Call[]): Call[] {
  const end = calls.findIndex((c) => c.op === 'stroke');
  return (end < 0 ? calls : calls.slice(0, end)).filter(
    (c) => c.op === 'moveTo' || c.op === 'lineTo',
  );
}

function stubCtx(calls: Call[]): CanvasRenderingContext2D {
  const noop = (): void => {};
  const ctx = {
    canvas: { width: 800, height: 600 },
    save: noop,
    restore: noop,
    setTransform: noop,
    translate: noop,
    scale: noop,
    rotate: noop,
    beginPath: () => calls.push({ op: 'beginPath', x: 0, y: 0 }),
    stroke: () => calls.push({ op: 'stroke', x: 0, y: 0 }),
    moveTo: (x: number, y: number) => calls.push({ op: 'moveTo', x, y }),
    lineTo: (x: number, y: number) => calls.push({ op: 'lineTo', x, y }),
    closePath: noop,
    fill: noop,
    fillRect: noop,
    arc: noop,
    ellipse: noop,
    quadraticCurveTo: noop,
    bezierCurveTo: noop,
    drawImage: noop,
    createImageData: noop,
    putImageData: noop,
    fillText: noop,
    measureText: () => ({ width: 0 }),
    lineWidth: 1,
    strokeStyle: '',
    fillStyle: '',
    lineCap: '',
    lineJoin: '',
    globalAlpha: 1,
    imageSmoothingEnabled: false,
    font: '',
    textAlign: '',
    textBaseline: '',
  };
  return ctx as unknown as CanvasRenderingContext2D;
}

function pond(): { sim: Sim; params: ReturnType<typeof defaultParams> } {
  const params = defaultParams();
  params.stepSpeed = 0;
  const sim = new Sim(400, 300);
  let seed = 4242;
  const rnd = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const kinds = ['era', 'dup', 'con'] as const;
  for (let i = 0; i < 90; i++) {
    sim.spawn(kinds[i % 3], rnd() * 400, rnd() * 300, rnd() * 6.28, params, true);
  }
  // Zoomed out, so the LOD puts wires on the chord path rather than a live
  // rope -- that is the branch the reused buffer serves, and the branch a
  // close-up pond never reaches.
  const far = { x: 200, y: 150, zoom: 0.12, viewW: 400, viewH: 300 };
  sim.setViewExtent((far.viewW / far.zoom) * 1.7, (far.viewH / far.zoom) * 1.7);
  for (let i = 0; i < 30; i++) sim.step(1 / 60, params, far);
  return { sim, params };
}

const view: ViewOpts = {
  overlay: false,
  energyGrid: false,
  energyCircles: false,
  kindColors: true,
  gpuAgents: true,
};

describe('wire stroking', () => {
  it('draws every wire once, in one path', () => {
    const { sim } = pond();
    expect(sim.graph.wires.size).toBeGreaterThan(5);
    const calls: Call[] = [];
    const camera = new Camera();
    camera.setView(400, 300);
    camera.snap(200, 150);
    render(stubCtx(calls), sim, camera, view);

    // Every wire opens a subpath, and they are all stroked together: the
    // first stroke of the frame closes the whole wire pass.
    const moves = wirePass(calls).filter((c) => c.op === 'moveTo').length;
    expect(moves).toBe(sim.graph.wires.size);
  });

  it('puts the same points on the canvas as wireStrokePoints, zoomed in', () => {
    const { sim } = pond();
    const calls: Call[] = [];
    const camera = new Camera();
    camera.setView(400, 300);
    camera.snap(200, 150);
    /*
     * Zoomed in, where the chord level of detail asks for every segment.
     *
     * `strokeWire` cuts a chord into fewer segments the smaller it is on
     * screen, and at the camera's default zoom of 0.065 a wire is about a
     * pixel long and draws as one straight line. This test is about the
     * geometry agreeing with the reference, so it asks at a zoom where the
     * detail is not being dropped; the test below covers the dropping.
     */
    camera.zoom = 40;
    render(stubCtx(calls), sim, camera, view);

    /*
     * Rebuild the expected path from the untouched reference geometry — and
     * the reference deliberately keeps the exhaustive scan over every rewrite
     * that `drawWires` no longer does. That scan is what the body-indexed
     * lookup replaced, so this comparison is what says the two agree.
     */
    let withHandoff = 0;
    const expected: Call[] = [];
    for (const wire of sim.graph.wires.values()) {
      const A = sim.agents.get(wire.a.id);
      const B = sim.agents.get(wire.b.id);
      if (!A || !B) continue;
      const rope = sim.wireSimulatesRope(wire);
      // A wire whose endpoint is being consumed is drawn from the rewrite's
      // handoff stems, so the reference has to ask for them too.
      let stemA: { x: number; y: number } | undefined;
      let stemB: { x: number; y: number } | undefined;
      for (const rw of sim.rewrites) {
        const handoff = rewriteHandoffStems(rw, wire, sim.agents, sim.w, sim.h);
        if (!handoff) continue;
        stemA = { x: handoff.ax, y: handoff.ay };
        stemB = { x: handoff.bx, y: handoff.by };
        withHandoff++;
        break;
      }
      const pts = wireStrokePoints(A, B, wire, sim.w, sim.h, stemA, stemB, rope);
      if (pts.length === 0) continue;
      expected.push({ op: 'moveTo', x: pts[0].x, y: pts[0].y });
      for (let i = 1; i < pts.length; i++) {
        expected.push({ op: 'lineTo', x: pts[i].x, y: pts[i].y });
      }
    }

    // Otherwise the handoff half of this is vacuous: no wire would be drawn
    // from a rewrite's stems and the index would never be consulted.
    expect(withHandoff, 'no wire is mid-rewrite, so the handoff path is untested')
      .toBeGreaterThan(0);
    expect(sim.rewrites.length, 'no rewrites in flight').toBeGreaterThan(1);

    const drawn = wirePass(calls);
    expect(drawn.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(drawn[i].op, `call ${i}`).toBe(expected[i].op);
      expect(drawn[i].x, `call ${i} x`).toBeCloseTo(expected[i].x, 9);
      expect(drawn[i].y, `call ${i} y`).toBeCloseTo(expected[i].y, 9);
    }
  });

  it('cuts a chord into far fewer segments once it is a pixel on screen', () => {
    const { sim } = pond();
    const camera = new Camera();
    camera.setView(400, 300);
    camera.snap(200, 150);

    const pathAt = (zoom: number): Call[] => {
      camera.zoom = zoom;
      const calls: Call[] = [];
      render(stubCtx(calls), sim, camera, view);
      return wirePass(calls);
    };

    // Zoomed in, every chord gets its full sixteen segments.
    const near = pathAt(40);
    // The camera's default. A whole dish in view puts a wire at about a pixel.
    const far = pathAt(0.065);

    const moves = far.filter((c) => c.op === 'moveTo').length;
    expect(moves, 'nothing was drawn, so this proves nothing').toBeGreaterThan(5);
    expect(near.filter((c) => c.op === 'moveTo').length, 'same wires either way').toBe(moves);
    // The point of the whole exercise.
    expect(far.length * 5, `${near.length} calls zoomed in, ${far.length} out`)
      .toBeLessThan(near.length);
    // And most of them are down to a single straight segment.
    expect(far.length).toBeLessThan(moves * 3);

    /*
     * A one-segment wire still runs stem to stem: the shortcut drops the curve
     * between the ends, not the wire's position. Only the single-segment
     * subpaths are checked, because a wire long enough to keep its segments is
     * still drawn along the cubic.
     */
    const ends = new Set<string>();
    const key = (x: number, y: number, u: number, v: number): string =>
      `${x.toFixed(6)},${y.toFixed(6)}|${u.toFixed(6)},${v.toFixed(6)}`;
    for (const wr of sim.graph.wires.values()) {
      const A = sim.agents.get(wr.a.id);
      const B = sim.agents.get(wr.b.id);
      if (!A || !B) continue;
      const sa = stemWorld(A, wr.a.slot, sim.w, sim.h);
      const sb = stemWorld(B, wr.b.slot, sim.w, sim.h);
      ends.add(key(sa.x, sa.y, sb.x, sb.y));
    }
    let singles = 0;
    for (let i = 0; i < far.length; i++) {
      if (far[i].op !== 'moveTo') continue;
      const isSingle = i + 1 < far.length && far[i + 1].op === 'lineTo' &&
        (i + 2 >= far.length || far[i + 2].op === 'moveTo');
      if (!isSingle) continue;
      singles++;
      const k = key(far[i].x, far[i].y, far[i + 1].x, far[i + 1].y);
      expect(ends.has(k), `single segment at call ${i} does not run stem to stem`).toBe(true);
    }
    expect(singles, 'no wire collapsed to one segment').toBeGreaterThan(5);
  });
});
