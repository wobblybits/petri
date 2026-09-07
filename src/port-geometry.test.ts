import { describe, expect, it } from 'vitest';
import {
  ERA_SLOTS,
  NODE_SLOTS,
  portFrameInto,
  portWorld,
  portWorldInto,
  slotsFor,
} from './agents.ts';
import type { AgentKind, PortSlot } from './agents.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/** A real body, because the geometry reads `kind`, `scale` and `heading`. */
function body(sim: Sim, kind: AgentKind, params: ReturnType<typeof defaultParams>) {
  return sim.spawn(kind, 5000, 5000, 0, params, true)!;
}

/**
 * `portWorldInto` against the allocating `portWorld` it stands in for.
 *
 * The into-form is hand-flattened: it inlines `stemRoot`, `portLocal`,
 * `rotate` and the two `wrap` calls to avoid four objects a call, and it is on
 * the path that decides where every voice in the pond is deposited. A sign
 * slip in the rotation would move every scent blob a few units and look
 * exactly like the simulation being chaotic.
 */
describe('portWorldInto', () => {
  it('lands where portWorld lands, for every kind, slot and pose', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    const sim = new Sim(20000, 20000);
    const out = { x: 0, y: 0 };
    let checked = 0;
    let worst = 0;
    for (const kind of ['con', 'dup', 'era'] as AgentKind[]) {
      const a = body(sim, kind, params);
      for (const slot of slotsFor(kind)) {
        for (let k = 0; k < 24; k++) {
          a.x = 1000 + k * 37.5;
          a.y = -400 + k * 61.25;
          a.heading = (k / 24) * Math.PI * 2 - Math.PI;
          const want = portWorld(a, slot, 10000, 8000);
          const got = portWorldInto(a, slot, 10000, 8000, out);
          worst = Math.max(worst, Math.abs(want.x - got.x), Math.abs(want.y - got.y));
          checked++;
        }
      }
    }
    expect(checked, 'the sweep covered nothing').toBeGreaterThan(100);
    expect(worst, 'the flattened form drifted from the one it replaces').toBeLessThan(1e-9);
  });

  it('scales with the body, which is where a flattened rotation goes wrong', () => {
    // `portOffset` scales the *local* offset and then rotates. Doing it the
    // other way round agrees at scale 1 and nowhere else, so the sweep above
    // would not catch it on its own.
    const params = defaultParams();
    params.spawnInterval = 0;
    const sim = new Sim(20000, 20000);
    const out = { x: 0, y: 0 };
    let worst = 0;
    const a = body(sim, 'con', params);
    for (const scale of [0.4, 1, 2.5]) {
      a.x = 300;
      a.y = 700;
      a.heading = 0.9;
      a.scale = scale;
      for (const slot of NODE_SLOTS) {
        const want = portWorld(a, slot, 10000, 8000);
        const got = portWorldInto(a, slot, 10000, 8000, out);
        worst = Math.max(worst, Math.abs(want.x - got.x), Math.abs(want.y - got.y));
      }
    }
    expect(worst).toBeLessThan(1e-9);
  });

  it('shares one list per kind rather than building one per call', () => {
    // The other half of the saving, and the half a correctness test would
    // miss: `slotsFor` returns a fresh array every time.
    expect([...ERA_SLOTS]).toEqual(slotsFor('era'));
    expect([...NODE_SLOTS]).toEqual(slotsFor('con'));
    expect(ERA_SLOTS, 'a shared list has to be the same object each time').toBe(ERA_SLOTS);
    expect(slotsFor('con'), 'and slotsFor is deliberately not').not.toBe(slotsFor('con'));
    expect(() => {
      (NODE_SLOTS as PortSlot[]).push('p');
    }, 'a shared list a caller can mutate is worse than an allocation').toThrow();
  });
});

/**
 * `portFrameInto` against the two functions it replaced.
 *
 * The latch pass no longer calls `portWorldInto` per port or the old
 * `inSnapArcAt` per candidate pair; it turns each body's heading into a sine
 * and a cosine once and this builds both the port's position and the outward
 * axis the snap arc is measured from. The axis formula below is the deleted
 * `inSnapArcAt`'s, kept here so the sign convention has a witness: a flipped
 * sign would let ports latch to whatever is behind them, which looks like the
 * pond being unusually social rather than like a bug.
 */
describe('portFrameInto', () => {
  it('agrees with portWorldInto and the arc axis, for every kind, slot and pose', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    const sim = new Sim(20000, 20000);
    const kinds: AgentKind[] = ['era', 'dup', 'con'];
    const into = { x: 0, y: 0 };
    const frame = { x: 0, y: 0, ax: 0, ay: 0 };
    for (const kind of kinds) {
      const a = body(sim, kind, params);
      for (const heading of [0, 0.3, 1.9, -2.7, Math.PI, 6.1]) {
        for (const scale of [0.6, 1, 1.7]) {
          a.heading = heading;
          a.scale = scale;
          a.x = 4231.5;
          a.y = 6104.25;
          const cos = Math.cos(heading);
          const sin = Math.sin(heading);
          for (const slot of slotsFor(kind) as PortSlot[]) {
            portWorldInto(a, slot, 20000, 20000, into);
            portFrameInto(kind, slot, a.x, a.y, scale, cos, sin, frame);
            expect(frame.x, `${kind}.${slot} x`).toBe(into.x);
            expect(frame.y, `${kind}.${slot} y`).toBe(into.y);
            // The deleted `inSnapArcAt`'s axis, verbatim.
            const sign = kind === 'era' || slot === 'p' ? 1 : -1;
            expect(frame.ax, `${kind}.${slot} ax`).toBe(sign * cos);
            expect(frame.ay, `${kind}.${slot} ay`).toBe(sign * sin);
          }
        }
      }
    }
  });
});
