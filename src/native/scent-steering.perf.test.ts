import { describe, expect, it } from 'vitest';
import { CH, FIELD_CELL } from '../fields.ts';
import { defaultParams } from '../params.ts';
import { Sim } from '../sim.ts';
import { nativeSolver } from './solver.ts';

/**
 * Can a body find a scent source, and how well?
 *
 * The measurement that any change to steering has to answer. Free bodies start
 * in a ring facing random ways, a broad source sits off to one side, and the
 * score is how much of their initial distance they close. Chance is roughly
 * zero: without steering they wander and the mean distance barely moves.
 *
 * It exists because steering has been broken twice by changes that looked
 * local — once by coarsening the field until the sensors straddled a seventh
 * of a cell, once by a deposit constant that disagreed across the wasm wall by
 * a factor of four. Neither showed up in a pass/fail test.
 */

function climb(frames = 600): { closed: number; meanEnd: number } {
  const params = defaultParams();
  params.maxAgents = 200;
  params.spawnInterval = 0;
  params.upkeep = 0;
  params.ambientEnergy = 0;
  params.rewriteDuration = 0;
  params.snapRadius = 0;
  params.deposit = 0; // no self-trail: this measures gradient following alone

  const sim = new Sim(4000, 4000);
  const ids: number[] = [];
  for (let i = 0; i < 24; i++) {
    const a = sim.spawn('dup', 0, (i - 12) * 40, (i * 0.79) % 6.28, params, true)!;
    ids.push(a.id);
  }
  sim.step(1 / 60, params);

  // A broad source 60 cells east, restamped each frame so decay does not eat
  // it, and wide enough that the bodies start well inside its skirt — placed
  // beyond its reach they sit in a flat zero and there is nothing to measure.
  // Amounts are asked for as a field density, not a raw deposit.
  const srcX = FIELD_CELL * 60;
  const srcY = 0;
  const stamp = (): void => {
    const unit = 2 / sim.fields.depositScale;
    for (let r = 0; r < FIELD_CELL * 90; r += FIELD_CELL * 3) {
      for (let a = 0; a < 6.283; a += 0.2) {
        sim.fields.deposit(
          CH.conP,
          srcX + Math.cos(a) * r,
          srcY + Math.sin(a) * r,
          unit / (1 + r / (FIELD_CELL * 30)),
        );
      }
    }
  };

  const dist = (): number => {
    let d = 0;
    let n = 0;
    for (const id of ids) {
      const a = sim.agents.get(id);
      if (!a) continue;
      d += Math.hypot(a.x - srcX, a.y - srcY);
      n++;
    }
    return n ? d / n : 0;
  };

  stamp();
  const start = dist();
  for (let f = 0; f < frames; f++) {
    stamp();
    sim.step(1 / 60, params);
  }
  const end = dist();
  return { closed: (start - end) / start, meanEnd: end };
}

describe('scent steering', () => {
  it('closes on a source it can smell', async () => {
    expect(await nativeSolver.init(), nativeSolver.lastError).toBe(true);
    const r = climb();
    console.log(
      `\ngradient climbing over 10s\n` +
        `  closed ${(r.closed * 100).toFixed(0)}% of the distance ` +
        `(mean ${r.meanEnd.toFixed(0)} from the source)\n`,
    );
    /*
     * A tripwire for steering being off, not a pin on how well it works. With
     * 24 bodies and one run the figure moves several points between runs: the
     * SENSE_SPAN sweep read 2, 5, 10, 6, 8 percent at 0.25, 0.15, 0.10, 0.07
     * and 0.05, which is not monotonic and should not be read as one. What it
     * does establish is that the proportional response at 0.10 is level with
     * the three-way choice it replaced, which was the bar — the change is for
     * what a graded turn lets a signal say, not for better foraging.
     */
    expect(r.closed, `only closed ${(r.closed * 100).toFixed(0)}%`).toBeGreaterThan(0.04);
  }, 900_000);
});
