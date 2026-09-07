import { describe, expect, it } from 'vitest';
import { FAR, FAR_STRIDE } from '../gpu/far-kernel.ts';
import { SLOP } from '../collide.ts';
import { NativeSolver, KIND_ERA } from './solver.ts';

/**
 * The far tier builds its candidate pair list once a frame and reuses it for
 * every substep, the way `solver_step_near` always has.
 *
 * That is only sound because the list is a *superset*: the cell it bins into
 * is two body radii plus slop plus a four-pixel margin, and `disc` still
 * distance-tests every pair it is handed, every substep. If the reuse ever
 * became a filter — a real contact dropped because the body that would have
 * been in it moved cells after substep zero — bodies would sink into each
 * other and stay there.
 *
 * So this does not check that the list is identical. It checks the property
 * the broad phase exists for: after a frame of solving, a dense pile is not
 * interpenetrating. It is run on a pile tight enough that most bodies start
 * overlapping and have to be pushed apart, and with enough speed on them that
 * some do change cells mid-frame.
 */

const R = 8; // era disc radius at scale 1
const KEEP = 2 * R;

/**
 * Two clumps, far enough apart that nothing bins them together to begin with,
 * closing on each other.
 *
 * This is the shape the test needs. A single dense pile does not test the
 * rebuild at all: at that density the cell lists already hold nearly every
 * pair, so a list frozen at frame zero still resolves it — measured, and it
 * passed, which is why the test is not written that way. Bodies have to
 * arrive at neighbours they did not have when the list was built.
 */
function twoClumps(side: number, gap: number, speed: number): Float32Array {
  const per = side * side;
  const n = per * 2;
  const data = new Float32Array(n * FAR_STRIDE);
  let seed = 12345;
  const rnd = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  // A jittered lattice at more than a body diameter, so the opening state has
  // nothing to resolve and whatever shows up at the end came from the two
  // clumps meeting.
  const pitch = 2 * R + 10;
  const jitter = 4;
  for (let i = 0; i < n; i++) {
    const o = i * FAR_STRIDE;
    const right = i >= per;
    const k = right ? i - per : i;
    const col = k % side;
    const row = (k - col) / side;
    data[o + FAR.x] =
      (right ? gap : 0) + col * pitch + (rnd() - 0.5) * jitter;
    data[o + FAR.y] = row * pitch + (rnd() - 0.5) * jitter;
    data[o + FAR.vx] = right ? -speed : speed;
    data[o + FAR.vy] = (rnd() - 0.5) * 4;
    data[o + FAR.heading] = rnd() * 6.28;
    data[o + FAR.omega] = 0;
    data[o + FAR.invMass] = 1;
    data[o + FAR.locked] = 0;
    data[o + FAR.radius] = R;
  }
  return data;
}

/** The deepest overlap in the pack, in pixels past the slop the solver allows. */
function worstPenetration(data: Float32Array, n: number): number {
  let worst = 0;
  for (let i = 0; i < n; i++) {
    const oi = i * FAR_STRIDE;
    for (let j = i + 1; j < n; j++) {
      const oj = j * FAR_STRIDE;
      const dx = data[oj + FAR.x] - data[oi + FAR.x];
      const dy = data[oj + FAR.y] - data[oi + FAR.y];
      const d = Math.hypot(dx, dy);
      const pen = KEEP - SLOP - d;
      if (pen > worst) worst = pen;
    }
  }
  return worst;
}

/** Mean x of a clump, so the two can be watched for passing through. */
function centroidX(data: Float32Array, from: number, to: number): number {
  let sum = 0;
  for (let i = from; i < to; i++) sum += data[i * FAR_STRIDE + FAR.x];
  return sum / (to - from);
}

describe('far tier pair list reused across substeps', () => {
  it('resolves contacts between bodies that were not neighbours when it was built', async () => {
    const native = new NativeSolver();
    expect(await native.init(), native.lastError).toBe(true);
    const side = 9;
    const per = side * side;
    const n = per * 2;
    const data = twoClumps(side, 460, 260);
    const kinds = native.kind!;
    for (let i = 0; i < n; i++) kinds[i] = KIND_ERA;
    expect(worstPenetration(data, n), 'the clumps must start apart').toBeLessThan(R);
    expect(centroidX(data, 0, per)).toBeLessThan(centroidX(data, per, n));

    /*
     * Watched every frame, not just at the end. A broad phase that never
     * rebuilds lets the clumps go straight through each other and come apart
     * on the far side, so the closing state looks pristine — measured, and an
     * end-state assertion passed on a deliberately broken build. What gives it
     * away is the crossing itself.
     */
    const empty = new Float32Array(0);
    let worst = 0;
    let crossed = false;
    for (let f = 0; f < 120; f++) {
      expect(native.stepFar(data, n, empty, 0, 1 / 60, 8)).toBe(true);
      const pen = worstPenetration(data, n);
      if (pen > worst) worst = pen;
      if (centroidX(data, 0, per) > centroidX(data, per, n)) crossed = true;
    }

    for (let i = 0; i < n; i++) {
      const o = i * FAR_STRIDE;
      expect(Number.isFinite(data[o + FAR.x]), `body ${i} x`).toBe(true);
      expect(Number.isFinite(data[o + FAR.y]), `body ${i} y`).toBe(true);
    }
    expect(crossed, 'the clumps swapped sides, so contacts between them were missed')
      .toBe(false);
    expect(worst, `deepest overlap at any frame: ${worst.toFixed(1)}`).toBeLessThan(R);
  });
});
