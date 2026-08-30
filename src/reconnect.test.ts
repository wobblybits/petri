import { describe, expect, it } from 'vitest';
import { applyRewrite, type NetSnapshot } from './rewrite.ts';

/**
 * Reconnection when agents die. The interesting cases are the ones where a
 * dying agent's ports lead back into the dying pair rather than straight out —
 * one hop is not enough to find the surviving ends.
 */

const con = (id: number) => ({ id, kind: 'con' as const });
const dup = (id: number) => ({ id, kind: 'dup' as const });
const era = (id: number) => ({ id, kind: 'era' as const });
const p = (id: number, slot: 'p' | 'l' | 'r') => ({ id, slot });

/** Canonical `a.slot-b.slot` strings, sorted, so comparisons are order-free. */
function links(net: NetSnapshot): string[] {
  return net.wires
    .map((w) => {
      const x = `${w.a.id}.${w.a.slot}`;
      const y = `${w.b.id}.${w.b.slot}`;
      return x < y ? `${x}-${y}` : `${y}-${x}`;
    })
    .sort();
}

describe('annihilation reconnects surviving ends', () => {
  it('joins the four outside neighbours, crossed for Con', () => {
    const net: NetSnapshot = {
      agents: [con(1), con(2), con(5), con(6), con(7), con(8)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(1, 'l'), b: p(5, 'l') },
        { a: p(1, 'r'), b: p(6, 'l') },
        { a: p(2, 'l'), b: p(7, 'l') },
        { a: p(2, 'r'), b: p(8, 'l') },
      ],
    };
    const out = applyRewrite(net, 'annihilate-con', 1, 2, 20);
    // crossed: 1.l joins 2.r's neighbour, 1.r joins 2.l's neighbour
    expect(links(out.net)).toEqual(['5.l-8.l', '6.l-7.l']);
  });

  it('joins straight through for Dup', () => {
    const net: NetSnapshot = {
      agents: [dup(1), dup(2), con(5), con(6), con(7), con(8)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(1, 'l'), b: p(5, 'l') },
        { a: p(1, 'r'), b: p(6, 'l') },
        { a: p(2, 'l'), b: p(7, 'l') },
        { a: p(2, 'r'), b: p(8, 'l') },
      ],
    };
    const out = applyRewrite(net, 'annihilate-dup', 1, 2, 20);
    expect(links(out.net)).toEqual(['5.l-7.l', '6.l-8.l']);
  });

  it('chases through a node whose own aux ports are wired together', () => {
    // This is the identity function, λx.x — a Con with l joined to r. Applying
    // it should hand the argument straight back to the caller. Following one
    // wire from each side finds only the dying node itself and gives up, which
    // used to drop both connections.
    const net: NetSnapshot = {
      agents: [con(1), con(2), con(5), con(6)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(2, 'l'), b: p(2, 'r') },
        { a: p(1, 'l'), b: p(5, 'l') },
        { a: p(1, 'r'), b: p(6, 'l') },
      ],
    };
    const out = applyRewrite(net, 'annihilate-con', 1, 2, 20);
    expect(links(out.net)).toEqual(['5.l-6.l']);
  });

  it('chases through a chain of several dying ports', () => {
    // 1.l-2.r and 2.l-1.r: everything inside is dying, so the two outside ends
    // 5.l and 6.l must find each other across two hops.
    const net: NetSnapshot = {
      agents: [con(1), con(2), con(5), con(6)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(1, 'l'), b: p(2, 'l') },
        { a: p(1, 'r'), b: p(5, 'l') },
        { a: p(2, 'r'), b: p(6, 'l') },
      ],
    };
    const out = applyRewrite(net, 'annihilate-con', 1, 2, 20);
    expect(links(out.net)).toEqual(['5.l-6.l']);
  });

  it('drops a closed loop that reaches nothing outside', () => {
    const net: NetSnapshot = {
      agents: [con(1), con(2)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(1, 'l'), b: p(2, 'r') },
        { a: p(1, 'r'), b: p(2, 'l') },
      ],
    };
    const out = applyRewrite(net, 'annihilate-con', 1, 2, 20);
    expect(out.net.agents).toEqual([]);
    expect(links(out.net)).toEqual([]);
  });

  it('leaves a free end free rather than inventing a partner', () => {
    const net: NetSnapshot = {
      agents: [con(1), con(2), con(5)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(1, 'l'), b: p(5, 'l') },
        // 1.r, 2.l and 2.r are all unwired
      ],
    };
    const out = applyRewrite(net, 'annihilate-con', 1, 2, 20);
    expect(links(out.net)).toEqual([]);
  });
});

describe('erasure and commutation reconnect too', () => {
  it('sends an eraser into both aux ports of the node it meets', () => {
    const net: NetSnapshot = {
      agents: [era(1), con(2), con(5), con(6)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(2, 'l'), b: p(5, 'l') },
        { a: p(2, 'r'), b: p(6, 'l') },
      ],
    };
    const out = applyRewrite(net, 'erase', 1, 2, 20);
    const kinds = out.net.agents.map((a) => a.kind).sort();
    expect(kinds).toEqual(['con', 'con', 'era', 'era']);
    expect(links(out.net)).toEqual(['20.p-5.l', '21.p-6.l']);
  });

  it('pairs the two new erasers when the node it meets is a loop', () => {
    const net: NetSnapshot = {
      agents: [era(1), con(2)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(2, 'l'), b: p(2, 'r') },
      ],
    };
    const out = applyRewrite(net, 'erase', 1, 2, 20);
    // The loop is gone; the two erasers face each other and will annihilate.
    expect(links(out.net)).toEqual(['20.p-21.p']);
  });

  it('commutation keeps all four outside connections', () => {
    const net: NetSnapshot = {
      agents: [con(1), dup(2), con(5), con(6), con(7), con(8)],
      wires: [
        { a: p(1, 'p'), b: p(2, 'p') },
        { a: p(1, 'l'), b: p(5, 'l') },
        { a: p(1, 'r'), b: p(6, 'l') },
        { a: p(2, 'l'), b: p(7, 'l') },
        { a: p(2, 'r'), b: p(8, 'l') },
      ],
    };
    const out = applyRewrite(net, 'commute', 1, 2, 20);
    expect(out.net.agents).toHaveLength(8);
    // Every outside neighbour still has exactly one wire.
    for (const outside of [5, 6, 7, 8]) {
      const touching = out.net.wires.filter(
        (w) => w.a.id === outside || w.b.id === outside,
      );
      expect(touching, `agent ${outside} lost its wire`).toHaveLength(1);
    }
  });
});
