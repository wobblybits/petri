import { describe, expect, it } from 'vitest';
import { slotIndex, type PortSlot } from './agents.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * Port occupancy is a slot-keyed array in `AgentStore`, not a map keyed on
 * agent id, because the latch pass asks "is this port free" of every port of
 * every body and both GPU packs ask again for the free-port mask.
 *
 * The array and the wire list are two views of one fact, and nothing enforces
 * that they agree — `attach`, `detach`, `rebind`, `rotatePorts` and
 * `detachAgent` each write both. So the guard is the agreement itself, checked
 * on a pond that has actually latched, rewritten and killed things.
 */
function occupancyAgrees(sim: Sim): string | null {
  const PW = sim.agentStore.portWire;
  const expected = new Map<number, number>();
  for (const w of sim.graph.wires.values()) {
    for (const p of [w.a, w.b]) {
      const a = sim.agents.get(p.id);
      if (!a) return `wire ${w.id} names body ${p.id}, which is not in the roster`;
      const at = a.slot * 3 + slotIndex(p.slot);
      const clash = expected.get(at);
      if (clash !== undefined) {
        return `port ${p.id}.${p.slot} is held by wires ${clash} and ${w.id}`;
      }
      expected.set(at, w.id);
    }
  }
  for (const a of sim.agents.values()) {
    for (const slot of ['p', 'l', 'r'] as PortSlot[]) {
      const at = a.slot * 3 + slotIndex(slot);
      const want = expected.get(at) ?? -1;
      if (PW[at] !== want) {
        return `port ${a.id}.${slot} (slot ${a.slot}) reads ${PW[at]}, wires say ${want}`;
      }
      // And the two lookup forms have to say the same thing.
      if (sim.graph.isFreeAt(a.id, slot) !== (want < 0)) {
        return `isFreeAt disagrees for ${a.id}.${slot}`;
      }
      if (sim.graph.isFreeAtSlot(a.slot, slotIndex(slot)) !== (want < 0)) {
        return `isFreeAtSlot disagrees for ${a.id}.${slot}`;
      }
    }
  }
  return null;
}

describe('port occupancy', () => {
  it('agrees with the wire list through latching, rewriting and death', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    const sim = new Sim(1400, 1000);
    let seed = 424242;
    const rnd = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const kinds = ['era', 'dup', 'con'] as const;
    for (let i = 0; i < 220; i++) {
      sim.spawn(kinds[i % 3], rnd() * 1400, rnd() * 1000, rnd() * 6.28, params, true);
    }
    expect(occupancyAgrees(sim), 'before any step').toBeNull();

    let sawWires = 0;
    for (let f = 0; f < 240; f++) {
      sim.step(1 / 60, params);
      sawWires = Math.max(sawWires, sim.graph.wires.size);
      if (f % 20 === 0) expect(occupancyAgrees(sim), `frame ${f}`).toBeNull();
    }
    // Vacuous if nothing ever latched.
    expect(sawWires, 'nothing ever latched, so this proves nothing').toBeGreaterThan(5);
    expect(sim.tally.born + sim.tally.died, 'no births or deaths to exercise recycling')
      .toBeGreaterThan(0);
    expect(occupancyAgrees(sim), 'at the end').toBeNull();
  });

  it('gives a recycled slot no memory of the wires its last tenant held', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    const sim = new Sim(600, 600);
    const a = sim.spawn('con', 300, 300, 0, params, true)!;
    const b = sim.spawn('con', 340, 300, Math.PI, params, true)!;
    sim.wire(a.id, 'p', b.id, 'p', params);
    expect(sim.graph.isFreeAt(a.id, 'p')).toBe(false);
    const slot = a.slot;

    sim.kill(a.id);
    // The slot goes back on the free list; the next spawn takes it.
    const c = sim.spawn('dup', 300, 300, 0, params, true)!;
    expect(c.slot, 'the test needs the slot to actually be reused').toBe(slot);
    expect(sim.graph.isFreeAt(c.id, 'p'), 'newborn inherited a corpse wire').toBe(true);
    expect(occupancyAgrees(sim)).toBeNull();
  });
});
