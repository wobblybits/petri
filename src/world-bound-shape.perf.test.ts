import { describe, expect, it } from 'vitest';
import { defaultParams } from './params.ts';
import { Sim } from './sim.ts';

/**
 * The world bound must not have corners.
 *
 * It was per-axis, which makes the boundary a square, and a square boundary
 * with a restoring force has four attractors: on an edge one velocity
 * component is cancelled and a body slides along the other, but in a corner
 * both are cancelled and it is stuck. Measured that way, 21 of 60 free bodies
 * released on the boundary were sitting in corners a minute later and not one
 * had come back inside — which is what a pond looks like when its Eras keep
 * collecting in the corners of the field.
 *
 * Radially it is 0 of 60. A disk inscribed in the square grid, so nothing
 * colliding with it can leave the field, and a body pressed outward by its
 * own swimming slides around the rim rather than wedging.
 */
describe('world bound shape', () => {
  it('does not collect free bodies in corners', () => {
    const params = defaultParams();
    params.maxAgents = 5000;
    params.spawnInterval = 0;
    params.upkeep = 0;
    params.ambientEnergy = 0;
    params.snapRadius = 0;
    params.rewriteDuration = 0;
    const sim = new Sim(800, 600);
    sim.pinWorld(400, 300);
    for (let i = 0; i < 20; i++) sim.spawn('con', 400 + (i % 5) * 40, 300 + ((i / 5) | 0) * 40, 0, params, true);
    const hx = sim.worldX;
    const hy = sim.worldY;
    const R = sim.worldR;
    const ids: number[] = [];
    for (let k = 0; k < 60; k++) {
      const a = (k / 60) * Math.PI * 2;
      const r = R * 0.98;
      const body = sim.spawn('era', hx + Math.cos(a) * r, hy + Math.sin(a) * r, a, params, true)!;
      ids.push(body.id);
    }
    for (let f = 0; f < 3600; f++) {
      sim.step(1 / 60, params);
    }

    let cornerish = 0;
    let edgeish = 0;
    let inside = 0;
    let maxR = 0;
    for (const id of ids) {
      const a = sim.agents.get(id);
      if (!a) continue;
      const ax = Math.abs(a.x - hx) / R;
      const ay = Math.abs(a.y - hy) / R;
      maxR = Math.max(maxR, Math.hypot(ax, ay));
      if (ax > 0.9 && ay > 0.9) cornerish++;
      else if (ax > 0.9 || ay > 0.9) edgeish++;
      else inside++;
    }
    console.log(
      `\n60 free eras released on the boundary, 60s later\n` +
        `  in a corner (both axes past 90%): ${cornerish}\n` +
        `  on an edge  (one axis past 90%):  ${edgeish}\n` +
        `  back inside:                      ${inside}\n` +
        `  furthest normalised radius:        ${maxR.toFixed(2)}\n`,
    );
    expect(cornerish + edgeish + inside, 'nothing survived to measure').toBeGreaterThan(30);
    expect(cornerish, `${cornerish} bodies wedged in corners`).toBe(0);
    expect(maxR, `furthest body at ${maxR.toFixed(2)} of the bound`).toBeLessThan(1.05);
  }, 300_000);
});
