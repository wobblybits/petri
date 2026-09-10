import { describe, expect, it } from 'vitest';
import { defaultParams } from '../params.ts';
import { PondDb } from './db.ts';
import {
  ARM_AXIS,
  PROTOCOLS,
  armSweepName,
  describeProtocol,
  evaluatePreconditions,
  findProtocol,
  planProtocol,
  preflight,
  protocolReport,
  readOutcomes,
  type Protocol,
} from './protocol.ts';
import type { TrialRow } from './analyze.ts';

/*
 * A protocol is an experiment decided in advance, and the registry is a set
 * of decisions this project has already made. So the first test is the one
 * that matters: every registered protocol passes its own preflight with no
 * errors and no unaccepted coupling. A protocol that would run one arm blind
 * cannot be committed.
 */
describe('registry', () => {
  it('has unique names and every entry passes preflight clean', () => {
    const names = new Set<string>();
    for (const p of PROTOCOLS) {
      expect(names.has(p.name), `${p.name} twice`).toBe(false);
      names.add(p.name);
      const pf = preflight(p);
      expect(pf.errors, `${p.name}: ${pf.errors.join('; ')}`).toEqual([]);
      expect(pf.warnings, `${p.name}: ${pf.warnings.join('; ')}`).toEqual([]);
    }
    expect(findProtocol('forage-engages')).toBeDefined();
    expect(findProtocol('nothing')).toBeUndefined();
  });

  it('records what each one knowingly holds', () => {
    // The Baldwin protocol holds the learning horizon across a layout axis on
    // purpose; the reason is the prediction, and preflight carries it forward.
    const pf = preflight(findProtocol('baldwin-hunger')!);
    expect(pf.accepted.some((a) => a.startsWith('groundPatches/learnDiscount'))).toBe(true);
    // Every arm-based chemistry protocol carries the deposit acknowledgement.
    expect(preflight(findProtocol('conserved-signal-price')!).accepted.some((a) => a.startsWith('excreteRate/deposit'))).toBe(true);
  });

  it('describes itself', () => {
    const text = describeProtocol(findProtocol('contested-ground')!);
    expect(text).toContain('arm 0 minted');
    expect(text).toContain('arm 1 conserved');
    expect(text).toContain('ambientEnergy=1,0.25');
  });
});

const tiny: Protocol = {
  name: 'tiny',
  question: 'q',
  prediction: 'p',
  nullReading: 'n',
  arms: [{ name: 'a', set: { excreteRate: 0 } }, { name: 'b', set: { excreteRate: 0.015 } }],
  axes: { groundPatches: [0, 8] },
  // Fixed weights, like the registered layout protocols: a learning horizon
  // held across a patch-spacing axis is a coupling, and this fixture is for
  // testing the excreteRate ones.
  base: { energyRegrow: 0, learnRate: 0 },
  seeds: 3,
  seconds: 120,
  soupCount: 100,
  outcomes: [{ metric: 'net_fst', expect: 'up' }],
  preconditions: [{ metric: 'demand_mean', summary: 'peak', min: 0.1, why: 'w' }],
};

describe('planning', () => {
  it('expands arms by points by seeds, and names the sweeps', () => {
    const plan = planProtocol(tiny);
    expect(plan.arms).toHaveLength(2);
    expect(plan.points).toBe(2);
    expect(plan.trials).toBe(12);
    expect(plan.arms[0].sweep).toBe('tiny/a');
    expect(plan.arms[1].base).toEqual({ energyRegrow: 0, learnRate: 0, excreteRate: 0.015 });
    expect(plan.arms[0].seeds).toEqual([1, 2, 3]);
    expect(planProtocol(tiny, { seeds: 5 }).trials).toBe(20);
  });

  it('makes a smoke run one seed, short, and tagged apart', () => {
    const plan = planProtocol(tiny, { smoke: true });
    expect(plan.smoke).toBe(true);
    expect(plan.trials).toBe(4);
    expect(plan.arms[0].seconds).toBeLessThanOrEqual(20);
    expect(plan.arms[0].sweep).toBe('tiny~smoke/a');
    expect(armSweepName('x', 'y', true)).toBe('x~smoke/y');
  });
});

describe('preflight', () => {
  it('flags an arm that varies excreteRate without re-choosing senseScale', () => {
    const pf = preflight(tiny);
    expect(pf.errors).toEqual([]);
    expect(pf.warnings.some((w) => w.includes('excreteRate/senseScale'))).toBe(true);
    // `excreteRate/uptakeVmax` used to warn here too, until uptake stopped
    // meaning a different mechanism on each side of the excretion switch. See
    // the retirement note above `COUPLINGS`.
    expect(pf.warnings.some((w) => w.includes('excreteRate/uptakeVmax'))).toBe(false);
  });

  it('is satisfied when the arms carry their own constants, or when the hold is accepted in writing', () => {
    const fixed: Protocol = {
      ...tiny,
      arms: [
        { name: 'a', set: { excreteRate: 0, senseScale: 4.3, uptakeVmax: 0 } },
        { name: 'b', set: { excreteRate: 0.015, senseScale: 0.002, uptakeVmax: 6 } },
      ],
      accepts: [{ axis: 'excreteRate', constant: 'deposit', because: 'inert above zero' }],
    };
    const pf = preflight(fixed);
    expect(pf.warnings).toEqual([]);
    expect(pf.accepted).toEqual(['excreteRate/deposit: inert above zero']);
  });

  it('warns about an acceptance nothing tripped', () => {
    const pf = preflight({ ...tiny, arms: [{ name: 'a', set: {} }], accepts: [{ axis: 'drag', constant: 'turnRate', because: 'x' }] });
    expect(pf.warnings.some((w) => w.includes('did not fire'))).toBe(true);
  });

  it('refuses unknown parameters, unknown metrics, bad scopes and too few seeds', () => {
    const bad: Protocol = {
      ...tiny,
      arms: [{ name: 'a', set: { nope: 1 } }],
      axes: { groundPatches: [0] },
      base: { alsoNope: 2 },
      seeds: 2,
      outcomes: [{ metric: 'net_fst', along: 'flockAlign' }, { metric: 'not_a_metric' }],
      preconditions: [
        { metric: 'demand_mean', why: 'no bound' },
        { metric: 'bodies', min: 1, arms: ['zzz'], why: 'no such arm' },
        { metric: 'bodies', min: 1, where: { drag: 1 }, why: 'not an axis' },
      ],
    };
    const pf = preflight(bad);
    const text = pf.errors.join('\n');
    expect(text).toContain('unknown parameter "nope"');
    expect(text).toContain('unknown parameter "alsoNope"');
    expect(text).toContain('at least two levels');
    expect(text).toContain('fewer than three is a smoke run');
    expect(text).toContain('not an axis');
    expect(text).toContain('unknown metric "not_a_metric"');
    expect(text).toContain('no bound');
    expect(text).toContain('no arm "zzz"');
    expect(text).toContain('where drag is not an axis');
    // The same protocol is allowed as a smoke run.
    expect(preflight({ ...tiny, seeds: 1 }, { smoke: true }).errors).toEqual([]);
  });

  it('warns rather than refuses when the budget is under the protocol minimum', () => {
    const pf = preflight(tiny, { seeds: 3, seconds: 60 });
    expect(pf.errors).toEqual([]);
    expect(pf.warnings.some((w) => w.includes('warm-up'))).toBe(true);
  });
});

function trial(arm: number, point: Record<string, number>, seed: number, values: Record<string, number | null>): TrialRow {
  return { runId: 0, seed, point: { ...point, [ARM_AXIS]: arm }, values, params: {} };
}

describe('reading', () => {
  it('scopes a gauge to its arms and grid points', () => {
    const p: Protocol = {
      ...tiny,
      preconditions: [
        { metric: 'demand_mean', summary: 'peak', min: 0.1, why: 'all' },
        { metric: 'demand_mean', summary: 'peak', min: 0.1, arms: ['b'], why: 'arm b' },
        { metric: 'demand_mean', summary: 'peak', min: 0.1, where: { groundPatches: 8 }, why: 'patches' },
      ],
    };
    const trials = [
      trial(0, { groundPatches: 0 }, 1, { demand_mean: 0 }),
      trial(0, { groundPatches: 8 }, 1, { demand_mean: 0.5 }),
      trial(1, { groundPatches: 0 }, 1, { demand_mean: 0.4 }),
      trial(1, { groundPatches: 8 }, 1, { demand_mean: null }),
    ];
    const [all, armB, patches] = evaluatePreconditions(p, trials);
    expect([all.passed, all.total]).toEqual([2, 4]);
    expect([armB.passed, armB.total]).toEqual([1, 2]);
    expect(armB.scope).toBe('arm b');
    expect([patches.passed, patches.total]).toEqual([1, 2]);
    expect(patches.mean).toBeCloseTo(0.5, 12);
  });

  it('reads an outcome along the arms, first to last, and against its expectation', () => {
    const trials = [1, 2, 3].flatMap((seed) => [
      trial(0, {}, seed, { net_fst: 0.1 + seed * 0.01 }),
      trial(1, {}, seed, { net_fst: 0.4 + seed * 0.01 }),
    ]);
    const [r] = readOutcomes({ ...tiny, axes: {} }, trials);
    expect(r.along).toBe(ARM_AXIS);
    expect(r.status).toBe('resolved');
    expect(r.direction).toBe('up');
    expect(r.matches).toBe(true);
    const [down] = readOutcomes({ ...tiny, axes: {}, outcomes: [{ metric: 'net_fst', expect: 'down' }] }, trials);
    expect(down.matches).toBe(false);
  });

  it('leaves the verdict open when the read is thin or unresolved', () => {
    const one = [trial(0, {}, 1, { net_fst: 0.1 }), trial(1, {}, 1, { net_fst: 0.4 })];
    const [thin] = readOutcomes({ ...tiny, axes: {} }, one);
    expect(thin.status).toBe('thin');
    expect(thin.matches).toBeNull();
    const [none] = readOutcomes({ ...tiny, axes: {} }, []);
    expect(none.status).toBe('no data');
  });
});

describe('report', () => {
  it('reads a protocol back out of the library in its own terms', () => {
    const db = new PondDb(':memory:');
    try {
      const p = findProtocol('conserved-signal-price')!;
      const plan = planProtocol(p, { smoke: true });
      for (const arm of plan.arms) {
        const params = { ...defaultParams(), ...arm.base };
        const id = db.startRun({
          seed: 1, seconds: 20, dt: 1 / 60, world: { w: 100, h: 100 }, fieldCells: 64,
          preset: 'soup', soupCount: 100, parentRun: null, params, commit: null, note: arm.note,
          sweep: arm.sweep, point: {},
        });
        for (const t of [0, 10, 20]) {
          db.addSample(id, {
            t, bodies: 100, wires: 0, lines: 1, bornMean: t / 10, bornMax: 0,
            spawned: 0, born: 0, died: 0, commutes: t, erases: 0, annihilations: 0,
            latches: t * 2, snaps: 0, free: 0, ground: 0, escrow: 0, meanExtra: 0,
            canPay: 0, ppWires: 0, conDupWires: 0, commuteShare: null,
            commuteChance: 0, commuteEdge: null, matrixDrift: 0,
            diversity: { netFst: 0.2, linesEffective: 3, signalTotal: arm.arm.name === 'minted' ? 500 : 30, signalP90: arm.arm.name === 'minted' ? 1 : 500 },
          });
        }
      }
      const text = protocolReport(db, p, { smoke: true });
      expect(text).toContain('SMOKE RUN');
      expect(text).toContain(p.question);
      expect(text).toContain('was the mechanism engaged?');
      // The per-arm scope and the sense-read gauge, computed against each arm's own senseScale.
      expect(text).toContain('arm conserved');
      expect(text).toContain('senseRead');
      // One seed an arm: thin, so the null reading is printed.
      expect(text).toContain('thin');
      expect(text).toContain('null reading, as written before the run');
      // Levels are named after the arms, not numbered.
      expect(text).toContain('arm=minted');
      // Nothing in the library for the real thing yet.
      expect(protocolReport(db, p)).toContain('no runs in this library');
    } finally {
      db.close();
    }
  });
});
