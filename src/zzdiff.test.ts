import { describe, expect, it } from 'vitest';
import { Fields } from './fields.ts';

/** How far does a point source actually reach, at various settings? */
describe('diffusion reach', () => {
  it('measures spread and stability', () => {
    const rows: string[] = [];
    for (const [mix, decay, frames] of [
      [0.28, 0.018, 600],
      [1.0, 0.018, 600],
      [1.0, 0.001, 600],
      [1.0, 0.0, 600],
      [1.0, 0.0, 3000],
      [0.5, 0.0, 3000],
    ] as number[][]) {
      const f = new Fields(256);
      const cx = f.originX + f.worldW * 0.5;
      const cy = f.originY + f.worldW * 0.5;
      const cell = f.cellSize;
      for (let t = 0; t < frames; t++) {
        // A steady source at the middle, like a body sitting still.
        f.deposit(0, cx, cy, 1 / f.depositScale);
        f.diffuse(mix);
        f.diffuse(mix * 0.65);
        f.decay(decay);
      }
      // Reach: furthest cell along +x still holding 1% of the peak.
      const peak = f.sample(0, cx, cy);
      let reach = 0;
      for (let k = 1; k < 120; k++) {
        if (f.sample(0, cx + k * cell, cy) > peak * 0.01) reach = k;
      }
      // Checkerboard: neighbouring cells disagreeing in sign or wildly in size
      // is the high-frequency mode, not smoothing.
      let flips = 0;
      for (let k = 1; k < 40; k++) {
        const a = f.sample(0, cx + k * cell, cy);
        const b = f.sample(0, cx + (k + 1) * cell, cy);
        if (a > 0 && b > 0 && (a / b > 3 || b / a > 3)) flips++;
      }
      rows.push(
        `  mix ${String(mix).padEnd(4)} decay ${String(decay).padEnd(6)} ${String(frames).padStart(4)}f:` +
          ` peak ${peak.toFixed(1).padStart(9)}  reach ${String(reach).padStart(3)} cells` +
          ` (${(reach * cell).toFixed(0).padStart(5)} units)  roughness ${flips}`,
      );
    }
    console.log(`\ndiffusion reach, 256-cell grid, 40-unit cells\n${rows.join('\n')}\n`);
    expect(rows.length).toBe(6);
  }, 300_000);
});
