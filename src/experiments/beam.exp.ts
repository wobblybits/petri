import { describe, it } from 'vitest';
import { defaultParams, type Params } from '../params.ts';
import { Sim } from '../sim.ts';
import { buildLadder, buildWorm, capWithEras, motorsOff } from './organism.ts';
import { seededRandom } from './harness.ts';

/*
 * Does a body plan have bending stiffness, and where does it come from?
 *
 * The gait work stalled on this. A phase wave along a chain produced no
 * directed motion at any drive, because there is no drive at which a chain
 * holds a *controlled* curve: gentle and it stays straight and swallows the
 * actuator, hard and it coils. The reason is that a joint between two bodies
 * has no rest angle and no bending stiffness — `portTorques` is a servo aiming
 * each port at its neighbour's stem, which is an aiming controller and not a
 * beam.
 *
 * The obvious fix is a new primitive: an angular spring on the joint. This
 * asks whether the topology can pay for it instead. Bending a ladder means
 * lengthening one rail and shortening the other, and rail length is held by
 * the span constraint, which is the stiffest thing in the solver. So a chain
 * should be a string and a ladder a beam, with no new force at all — and the
 * stiffness should scale with the rail separation, which is a number the body
 * plan sets and evolution could therefore reach.
 *
 * The measurement is a cantilever. Pin the rear, settle, then push the front
 * sideways with a fixed total force and read how far it goes. More deflection
 * is a floppier structure. Then let go and see how much comes back, which
 * separates a spring from a hinge that simply stayed where it was put.
 *
 *     npm run experiment -- beam
 */

const SETTLE = 3;
const PUSH = 4;
const RELEASE = 6;

function bench(): Params {
  return {
    ...defaultParams(),
    ...motorsOff(),
    upkeep: 0,
    ambientEnergy: 0,
    spawnInterval: 0,
  };
}

interface Shape {
  label: string;
  build: (sim: Sim, params: Params) => { ids: number[]; heads: number[]; tails: number[] };
}

/** Lateral offset of the head group from the line the tail group defines. */
function deflection(sim: Sim, heads: number[], tails: number[], axis: { x: number; y: number }): number {
  const mean = (list: number[]) => {
    let x = 0;
    let y = 0;
    let n = 0;
    for (const id of list) {
      const a = sim.agents.get(id);
      if (!a) continue;
      x += a.x;
      y += a.y;
      n++;
    }
    return n > 0 ? { x: x / n, y: y / n } : null;
  };
  const h = mean(heads);
  const t = mean(tails);
  if (!h || !t) return NaN;
  const dx = h.x - t.x;
  const dy = h.y - t.y;
  return -dx * axis.y + dy * axis.x;
}

function cantilever(shape: Shape, force: number, seed: number): {
  deflect: number;
  recovered: number;
  bodies: number;
} {
  const realRandom = Math.random;
  Math.random = seededRandom(seed);
  try {
    const params = bench();
    const sim = new Sim(6000, 6000, 256);
    sim.pinWorld(3000, 3000, params);
    const { ids, heads, tails } = shape.build(sim, params);
    const axis = { x: 1, y: 0 };
    const dt = 1 / 60;
    for (let f = 0; f < SETTLE * 60; f++) sim.step(dt, params);
    // Pinned after settling, so the structure is held in the pose it relaxed
    // into rather than the one it was built in.
    for (const id of tails) {
      const a = sim.agents.get(id);
      if (a) a.pinned = true;
    }
    const rest = deflection(sim, heads, tails, axis);
    const share = force / Math.max(1, heads.length);
    for (let f = 0; f < PUSH * 60; f++) {
      for (const id of heads) {
        const a = sim.agents.get(id);
        if (a) a.vy += (share * dt) / Math.max(0.08, a.mass);
      }
      sim.step(dt, params);
    }
    const bent = deflection(sim, heads, tails, axis);
    for (let f = 0; f < RELEASE * 60; f++) sim.step(dt, params);
    const back = deflection(sim, heads, tails, axis);
    const deflect = bent - rest;
    return {
      deflect,
      recovered: Math.abs(deflect) > 1e-6 ? 1 - (back - rest) / deflect : 0,
      bodies: ids.length,
    };
  } finally {
    Math.random = realRandom;
  }
}

function report(title: string, shapes: Shape[], force: number): void {
  const seeds = [1, 2, 3];
  const rows: string[] = [];
  for (const shape of shapes) {
    const runs = seeds.map((s) => cantilever(shape, force, s));
    const avg = (f: (r: (typeof runs)[0]) => number) => runs.reduce((s, r) => s + f(r), 0) / runs.length;
    const d = avg((r) => r.deflect);
    const sd = Math.sqrt(Math.max(0, avg((r) => r.deflect * r.deflect) - d * d));
    rows.push(
      `${shape.label.padEnd(24)} ${d.toFixed(1).padStart(9)} +/-${sd.toFixed(1).padStart(6)}  ` +
        `recovered ${avg((r) => r.recovered).toFixed(2).padStart(6)}  bodies ${String(runs[0].bodies).padStart(3)}`,
    );
  }
  console.log(`\n${title}  (force ${force}, mean of ${seeds.length} seeds)`);
  console.log(`${'shape'.padEnd(24)} ${'deflect'.padStart(9)}`);
  console.log(rows.join('\n'));
}

const SEGMENTS = Number(process.env.EXP_SEGMENTS ?? 8);

function chainShape(label: string, eras = false): Shape {
  return {
    label,
    build: (sim, params) => {
      const ids = buildWorm(sim, params, {
        segments: SEGMENTS,
        kinds: 'con',
        heading: 0,
        jitter: 0.02,
        x: 3000,
        y: 3000,
      });
      const extra = eras ? capWithEras(sim, params, ids, 'r') : [];
      return { ids: [...ids, ...extra], heads: [ids[ids.length - 1]!], tails: [ids[0]!] };
    },
  };
}

function ladderShape(label: string, railGap: number, rails = 2): Shape {
  return {
    label,
    build: (sim, params) => {
      const ids = buildLadder(sim, params, {
        segments: SEGMENTS,
        kinds: 'con',
        heading: 0,
        jitter: 0.02,
        x: 3000,
        y: 3000,
        rails,
        railGap,
      });
      const heads: number[] = [];
      const tails: number[] = [];
      for (let r = 0; r < rails; r++) {
        tails.push(ids[r * SEGMENTS]!);
        heads.push(ids[r * SEGMENTS + SEGMENTS - 1]!);
      }
      return { ids, heads, tails };
    },
  };
}

describe('experiment: where bending stiffness comes from', () => {
  it('checks the deflection is in a regime worth calling stiffness', () => {
    /*
     * A cantilever number only means stiffness while deflection is
     * proportional to load. The first run of this pushed the tip more than
     * half a body length sideways, which is folding rather than bending — and
     * it showed: recovery came back *negative*, meaning the shape kept drifting
     * after the load came off instead of springing back at all.
     */
    for (const force of [25, 100, 400]) {
      report(
        'load sweep',
        [chainShape('chain'), ladderShape('ladder, gap 144', 144)],
        force,
      );
    }
  });

  it('compares a chain against a ladder', () => {
    report(
      'cantilever: same length, pushed sideways at the free end',
      [
        chainShape('chain'),
        chainShape('chain + Era fins', true),
        ladderShape('ladder, gap 48', 48),
        ladderShape('ladder, gap 96', 96),
        ladderShape('ladder, gap 144', 144),
        ladderShape('3 rails, gap 48', 48, 3),
      ],
      400,
    );
  });
});
