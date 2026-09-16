import { describe, it } from 'vitest';
import { defaultParams, type Params } from './params.ts';
import { loadPreset } from './presets.ts';
import { Sim } from './sim.ts';
import { seededRandom } from './experiments/harness.ts';
import { measureDiversity } from './pond/measure.ts';

type Row = { bodies: number; lines: number; depth: number; speed: number; forage: number };

function one(tweak: (p: Params) => void, seed: number, seconds: number): Row {
  const real = Math.random;
  Math.random = seededRandom(seed);
  try {
    const params = defaultParams();
    params.soupCount = 400;
    tweak(params);
    const sim = new Sim(1600, 1200, 128);
    loadPreset(sim, 'soup', params);
    for (let i = 0; i < seconds * 60; i++) sim.step(1 / 60, params);
    const bodies = [...sim.agents.values()];
    let speed = 0;
    for (const b of bodies) speed += Math.hypot(b.vx, b.vy) / bodies.length;
    return {
      bodies: bodies.length,
      lines: new Set(bodies.map((b) => b.lineage)).size,
      depth: sim.graph.wires.size / Math.max(1, bodies.length),
      speed,
      forage: measureDiversity(sim).forageRatio ?? 0,
    };
  } finally {
    Math.random = real;
  }
}

function arm(label: string, tweak: (p: Params) => void, seeds = [1, 2, 3], seconds = 120): void {
  const rows = seeds.map((s) => one(tweak, s, seconds));
  const m = (f: (r: Row) => number): string => (rows.reduce((a, r) => a + f(r), 0) / rows.length).toFixed(3).padStart(8);
  console.log(
    ` ${label.padEnd(20)} bodies ${m((r) => r.bodies)} [${rows.map((r) => r.bodies).join('/')}]  lines ${m((r) => r.lines)}` +
      `  wires/body ${m((r) => r.depth)}  speed ${m((r) => r.speed)}  forage ${m((r) => r.forage)}`,
  );
}

describe('probe', () => {
  it('D', { timeout: 2_400_000 }, () => {
    console.log('');
    arm('shipped', () => {});
    arm('thrust 0.5', (p) => { p.transportThrust = 0.5; });
    arm('excrete 1', (p) => { p.upkeepExcrete = 1; });
  });
});
