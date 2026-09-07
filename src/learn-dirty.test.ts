import { describe, expect, it } from 'vitest';
import { AgentStore } from './agent-store.ts';

/**
 * Which learning rows the host has written, tracked as a list and not a span.
 *
 * The host writes a learning row in exactly one place — zeroing a slot the
 * free list has just handed back — so the dirty slots are scattered wherever
 * bodies happened to die. On a grown pond that measured 62 dirty slots a
 * frame in 62 separate runs, spanning 24,088 slots: carrying the span into the
 * upload buffer cost 11.9 ms a frame to deliver 62 rows.
 *
 * The span is still kept, because it is the right shape for the one case a
 * list is bad at, and `syncLearn` falls back to it when the list overruns.
 */
describe('learning dirty tracking', () => {
  /** `markLearn` is what `clearSlot` calls; reach it the way the store does. */
  function marked(store: AgentStore): number[] {
    return Array.from(store.learnDirtySlots.subarray(0, store.learnDirtyCount));
  }

  it('lists the slots that were written, not the span between them', () => {
    const store = new AgentStore(2048);
    store.clearLearnDirty();
    for (const slot of [1900, 3, 700]) store.markLearn(slot);
    expect(marked(store)).toEqual([1900, 3, 700]);
    expect(store.learnDirtyAll).toBe(false);
    // And the span still describes them, for the fallback.
    expect(store.learnDirtyLo).toBe(3);
    expect(store.learnDirtyHi).toBe(1901);
  });

  it('lists a slot once however many times it is written', () => {
    const store = new AgentStore(64);
    store.clearLearnDirty();
    store.markLearn(7);
    store.markLearn(7);
    store.markLearn(9);
    store.markLearn(7);
    expect(marked(store)).toEqual([7, 9]);
  });

  it('gives up on the list rather than growing it without bound', () => {
    const store = new AgentStore(4096);
    store.clearLearnDirty();
    for (let i = 0; i < 4000; i++) store.markLearn(i);
    expect(store.learnDirtyAll, 'should have fallen back to the span').toBe(true);
    // The span still covers everything marked, which is what the fallback reads.
    expect(store.learnDirtyLo).toBe(0);
    expect(store.learnDirtyHi).toBe(4000);
  });

  it('forgets last frame, without having to walk what it forgot', () => {
    const store = new AgentStore(64);
    store.clearLearnDirty();
    store.markLearn(5);
    expect(marked(store)).toEqual([5]);
    store.clearLearnDirty();
    expect(store.learnDirtyCount).toBe(0);
    expect(store.learnDirtyAll).toBe(false);
    // The same slot again: the stamp from last frame must not suppress it.
    store.markLearn(5);
    expect(marked(store), 'a slot marked last frame was swallowed').toEqual([5]);
  });

  it('marks the slot a recycled body lands in, so its rows are zeroed', () => {
    const store = new AgentStore(16);
    const slot = store.allocate(1);
    store.plasticAll[slot * 4] = 0.5;
    store.release(1);
    store.clearLearnDirty();
    const again = store.allocate(2);
    expect(again, 'the test needs the slot reused').toBe(slot);
    expect(marked(store), 'a recycled slot must be pushed to the device').toContain(slot);
    expect(store.plasticAll[slot * 4], 'and cleared host-side').toBe(0);
  });
});
