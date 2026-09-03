import { describe, expect, it } from 'vitest';
import {
  ap,
  church,
  compile,
  decodeChurch,
  evalChurch,
  injectTerm,
  lam,
  layoutCrossings,
  layoutNet,
  normalize,
  readChurch,
  v,
  MULT,
  PLUS,
} from './lambda.ts';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

describe('compiling', () => {
  it('rejects a free variable rather than leaving a dangling wire', () => {
    expect(() => compile(v('nope'))).toThrow(/unbound/);
  });

  it('gives an unused binder an eraser and a shared one a dup tree', () => {
    const none = compile(lam('x', lam('y', v('y'))));
    expect(none.net.agents.filter((a) => a.kind === 'era')).toHaveLength(1);

    const twice = compile(church(2));
    expect(twice.net.agents.filter((a) => a.kind === 'dup')).toHaveLength(1);

    const thrice = compile(church(3));
    expect(thrice.net.agents.filter((a) => a.kind === 'dup')).toHaveLength(2);
  });

  it('wires every port it claims to, and only the interface stays free', () => {
    const c = compile(ap(PLUS, church(2), church(3)));
    const seen = new Set<string>();
    for (const w of c.net.wires) {
      for (const p of [w.a, w.b]) {
        const k = `${p.id}.${p.slot}`;
        expect(seen.has(k), `port ${k} is wired twice`).toBe(false);
        seen.add(k);
      }
    }
    // The marker's own principal is what keeps it inert; it must stay free.
    expect(seen.has(`${c.root.id}.p`)).toBe(false);
  });
});

describe('reducing', () => {
  it('reads Church numerals straight back before any reduction', () => {
    for (let n = 0; n <= 6; n++) {
      const c = compile(church(n));
      expect(decodeChurch(c.net, c.root), `church(${n})`).toBe(n);
    }
  });

  it('normalises the identity applied to a numeral', () => {
    const c = compile(ap(lam('x', v('x')), church(3)));
    const out = normalize(c);
    expect(out.done).toBe(true);
    expect(decodeChurch(out.net, c.root)).toBe(3);
  });

  it('adds', () => {
    for (let a = 0; a <= 5; a++) {
      for (let b = 0; b <= 5; b++) {
        const r = evalChurch(ap(PLUS, church(a), church(b)));
        expect(r.done, `${a}+${b} ran out of steps`).toBe(true);
        expect(r.value, `${a}+${b}`).toBe(a + b);
      }
    }
  });

  it('multiplies only while one side is trivial, and says null otherwise', () => {
    // Multiplication duplicates a term that itself duplicates. The plain
    // Con/Dup/Era encoding has no bookkeeping for that, so the normal form
    // keeps a Dup that readback cannot see through. What matters is that this
    // reports "not a numeral" rather than a confidently wrong number.
    expect(evalChurch(ap(MULT, church(0), church(4))).value).toBe(0);
    expect(evalChurch(ap(MULT, church(1), church(4))).value).toBe(4);
    expect(evalChurch(ap(MULT, church(3), church(3))).value).toBeNull();
  });

  it('does not mistake a non-numeral for one', () => {
    const notANumber = compile(lam('f', lam('x', v('f'))));
    expect(decodeChurch(notANumber.net, notANumber.root)).toBeNull();
  });
});

describe('layout', () => {
  function crossingsOf(term: ReturnType<typeof church> | typeof PLUS): number {
    const c = compile(term);
    return layoutCrossings(c.net, layoutNet(c.net, c.root, 0, 0));
  }

  it('gives every agent a distinct pose', () => {
    const c = compile(ap(PLUS, church(2), church(3)));
    const poses = layoutNet(c.net, c.root, 400, 300);
    expect(poses).toHaveLength(c.net.agents.length);
    const keys = new Set(poses.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`));
    expect(keys.size).toBe(poses.length);
  });

  it('places church numerals without crossed wires', () => {
    for (let n = 0; n <= 4; n++) {
      expect(crossingsOf(church(n)), `church(${n})`).toBe(0);
    }
  });

  it('places plus and a sum without crossed wires', () => {
    expect(crossingsOf(PLUS), '+').toBe(0);
    expect(crossingsOf(ap(PLUS, church(2), church(3))), '2+3').toBe(0);
    expect(crossingsOf(ap(PLUS, church(0), church(1))), '0+1').toBe(0);
  });
});

describe('running in the simulation', () => {
  function run(a: number, b: number, seconds = 60) {
    const params = defaultParams();
    params.spawnInterval = 0;
    const sim = new Sim(900, 700);
    sim.setViewExtent(900, 700);
    const { root } = injectTerm(sim, ap(PLUS, church(a), church(b)), 450, 350, params);
    for (let f = 0; f < 60 * seconds; f++) {
      sim.step(1 / 60, params);
      const answer = readChurch(sim.agents, sim.graph, root);
      if (answer !== null) return { answer, frames: f };
    }
    return { answer: null as number | null, frames: 60 * seconds };
  }

  it('reduces 2 + 3 to 5 under the physics', () => {
    const { answer, frames } = run(2, 3);
    expect(answer, `settled after ${frames} frames`).toBe(5);
  });

  it('reduces a term containing zero', () => {
    expect(run(0, 2).answer).toBe(2);
  });

  it('seals the interface so the term cannot latch onto a passer-by', () => {
    const params = defaultParams();
    params.spawnInterval = 0;
    const sim = new Sim(900, 700);
    const { root } = injectTerm(sim, church(2), 450, 350, params);
    // A free agent parked right on the interface must not be able to join it.
    sim.spawn('era', 450, 350, 0, params, true);
    for (let f = 0; f < 60 * 20; f++) sim.step(1 / 60, params);
    expect(sim.graph.sealed.size).toBeGreaterThan(0);
    expect(readChurch(sim.agents, sim.graph, root), 'term was corrupted').toBe(2);
  });
});
