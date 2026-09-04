import { describe, expect, it } from 'vitest';
import { AgentStore } from './agent-store.ts';
import { createAgent } from './agents.ts';
import { defaultParams } from './params.ts';

/**
 * `AgentStore.growTo` reallocates every field array and reassigns it onto
 * `store`; an Agent created before a grow has to keep tracking the field
 * it's actually attached to, or it silently freezes at whatever value it
 * held the instant capacity ran out — while code that indexes the store
 * directly (sim.ts's converted solve/declutter/flock, render.ts's
 * buildFarInstances) keeps seeing the live value. That divergence is what
 * showed up as agents rendering in stale positions once the population grew
 * past the store's starting capacity.
 */
describe('AgentStore growth', () => {
  it('keeps every field live across a grow, for agents created before it', () => {
    const store = new AgentStore(2);
    const params = defaultParams();
    const early = createAgent(1, 'con', 10, 20, 0.5, params, store);
    // Capacity starts at 2; two more agents force at least one growTo.
    createAgent(2, 'con', 0, 0, 0, params, store);
    createAgent(3, 'con', 0, 0, 0, params, store);
    expect(store.capacity).toBeGreaterThan(2);

    // A write straight to the (now-grown) store, as the converted hot loops
    // make, must be visible through the pre-grow Agent's own accessor.
    store.x[early.slot] = 999;
    store.heading[early.slot] = 1.25;
    store.locked[early.slot] = 1;
    expect(early.x).toBe(999);
    expect(early.heading).toBe(1.25);
    expect(early.locked).toBe(true);

    // And the reverse: writing through the pre-grow Agent must land in the
    // current array, not an abandoned one.
    early.x = 42;
    early.locked = false;
    expect(store.x[early.slot]).toBe(42);
    expect(store.locked[early.slot]).toBe(0);
  });

  it('keeps chem live across a grow too', () => {
    const store = new AgentStore(1);
    const params = defaultParams();
    const early = createAgent(1, 'con', 0, 0, 0, params, store);
    createAgent(2, 'con', 0, 0, 0, params, store);
    expect(store.capacity).toBeGreaterThan(1);

    early.chem[0] = 0.75;
    expect(store.chemAll[early.slot * 16]).toBe(0.75);
  });
});
