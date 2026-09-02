import { describe, expect, it } from 'vitest';
import { PairGrid } from './grid.ts';

function scatter(n: number, w: number, h: number, seed: number) {
  let s = seed >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(rnd() * w);
    ys.push(rnd() * h);
  }
  return { xs, ys };
}

function key(i: number, j: number): string {
  return i < j ? `${i}:${j}` : `${j}:${i}`;
}

/** Every pair closer than `r`, by brute force. */
function bruteForce(xs: number[], ys: number[], r: number): Set<string> {
  const out = new Set<string>();
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      if (Math.hypot(xs[j] - xs[i], ys[j] - ys[i]) <= r) out.add(key(i, j));
    }
  }
  return out;
}

describe('PairGrid', () => {
  it('finds every pair inside the cell size, and no pair twice', () => {
    const cell = 40;
    for (const seed of [1, 2, 3, 7, 99]) {
      const { xs, ys } = scatter(300, 900, 600, seed);
      const grid = new PairGrid();
      grid.build(xs, ys, xs.length, cell);

      const seen: string[] = [];
      grid.forEachPair((i, j) => seen.push(key(i, j)));
      const unique = new Set(seen);
      expect(unique.size, `seed ${seed}: a pair was visited twice`).toBe(seen.length);

      const expected = bruteForce(xs, ys, cell);
      for (const k of expected) {
        expect(unique.has(k), `seed ${seed}: missed close pair ${k}`).toBe(true);
      }
    }
  });

  it('narrows the field enough to be worth it', () => {
    const { xs, ys } = scatter(400, 3000, 1500, 5);
    const grid = new PairGrid();
    grid.build(xs, ys, xs.length, 48);
    let visited = 0;
    grid.forEachPair(() => visited++);
    const allPairs = (400 * 399) / 2;
    expect(visited, `visited ${visited} of ${allPairs}`).toBeLessThan(allPairs * 0.05);
  });

  it('degrades safely when everything is stacked in one spot', () => {
    const xs = new Array(50).fill(10);
    const ys = new Array(50).fill(10);
    const grid = new PairGrid();
    grid.build(xs, ys, xs.length, 32);
    let visited = 0;
    grid.forEachPair(() => visited++);
    expect(visited).toBe((50 * 49) / 2);
  });

  it('handles a wildly spread swarm without allocating a huge grid', () => {
    const { xs, ys } = scatter(200, 4_000_000, 4_000_000, 11);
    const grid = new PairGrid();
    grid.build(xs, ys, xs.length, 8);
    let visited = 0;
    grid.forEachPair(() => visited++);
    // Coarsening kicks in; the point is that it returns rather than trying to
    // allocate 250 billion cells.
    expect(visited).toBeGreaterThanOrEqual(0);
  });

  it('finds neighbours within a radius', () => {
    const { xs, ys } = scatter(250, 800, 800, 4);
    const grid = new PairGrid();
    grid.build(xs, ys, xs.length, 50);
    const r = 45;
    for (const probe of [0, 17, 123, 249]) {
      const found = new Set<number>();
      grid.forEachNear(xs[probe], ys[probe], r, (i) => found.add(i));
      for (let i = 0; i < xs.length; i++) {
        if (Math.hypot(xs[i] - xs[probe], ys[i] - ys[probe]) <= r) {
          expect(found.has(i), `missed neighbour ${i} of ${probe}`).toBe(true);
        }
      }
    }
  });

  it('survives an empty set', () => {
    const grid = new PairGrid();
    grid.build([], [], 0, 10);
    let visited = 0;
    grid.forEachPair(() => visited++);
    grid.forEachNear(0, 0, 10, () => visited++);
    expect(visited).toBe(0);
  });

  it('does not hang when every position is NaN', () => {
    const grid = new PairGrid();
    const t0 = Date.now();
    grid.build([NaN, NaN, NaN], [NaN, NaN, NaN], 3, 10);
    let visited = 0;
    grid.forEachPair(() => visited++);
    expect(Date.now() - t0).toBeLessThan(100);
    expect(visited).toBeGreaterThanOrEqual(0);
  });
});
