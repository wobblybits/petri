import { defaultParams } from '../params.ts';
import { checkCouplings, type CouplingWarning } from './couplings.ts';
import {
  METRICS,
  RESOLVED_ETA2,
  SUMMARIES,
  THIN_SEEDS,
  effectTable,
  effects,
  metricKey,
  metricLabel,
  pointTable,
  readEffect,
  sweepTrials,
  table,
  type Effect,
  type Summary,
  type TrialRow,
} from './analyze.ts';
import type { PondDb } from './db.ts';

/*
 * An experiment written down before it runs.
 *
 * A sweep is a grid and a list of seeds. It does not know what question it is
 * asking, which constants it is holding that mean different things at
 * different points, whether the mechanism the question is about was switched
 * on in the ponds it ran, which measure the answer should be read off, how
 * many seeds that measure needs, or what a flat result would mean. Every one
 * of those was got wrong at least once in the first two days of sweeping —
 * `docs/experiments.md` §1 is the list — and every one of them is a thing
 * that can be decided in advance and checked by the machine.
 *
 * So a protocol is the sweep plus those decisions, as data. `preflight`
 * checks the parameter and metric names, the seed and duration minima, and
 * the couplings (`couplings.ts`) against what the protocol holds; a coupling
 * the author knowingly holds is listed in `accepts`, with the reason, and the
 * reason is what gets recorded. Each arm runs as its own sweep, named
 * `<protocol>/<arm>`, so it lands in the library like any other sweep and can
 * be read with `analyze --name`. `protocolReport` then reads the arms back
 * together — the arm is an axis — evaluates the preconditions per trial, and
 * reads each outcome against its prediction as resolved, unresolved or thin.
 * The null reading is printed whenever an outcome is not resolved, so the
 * story written before the run is the one that gets told after it.
 *
 * `PROTOCOLS` is the registry. `protocol.test.ts` runs every entry through
 * its own preflight, so a protocol that would trip a coupling or name a
 * missing metric cannot be committed.
 */

export interface Arm {
  /** Short, lowercase; it becomes part of the sweep name. */
  name: string;
  /** The regime's constants: everything this arm sets differently. */
  set: Record<string, number>;
}

export interface Outcome {
  metric: string;
  summary?: Summary;
  /**
   * Which way the prediction says the metric moves along `along`, read from
   * the axis's smallest level to its largest — for `arm`, from the first arm
   * to the last. `differs` claims a difference without a direction.
   */
  expect?: 'up' | 'down' | 'differs';
  /** The axis the expectation is about. Default: `arm` with several arms, else the first axis. */
  along?: string;
}

export interface Precondition {
  metric: string;
  summary?: Summary;
  min?: number;
  max?: number;
  /** Arms this gauge applies to. Default: all. */
  arms?: string[];
  /** Grid points this gauge applies to, as axis values. Default: all. */
  where?: Record<string, number>;
  why: string;
}

export interface Accepted {
  axis: string;
  constant: string;
  because: string;
}

export interface Protocol {
  name: string;
  /** One sentence, about the pond, not about a dial. */
  question: string;
  /** Directional and falsifiable. */
  prediction: string;
  /** What an unresolved or flat result means, decided before the run. */
  nullReading: string;
  arms: Arm[];
  /** Crossed within every arm. */
  axes: Record<string, number[]>;
  /** Held constants common to every arm. */
  base?: Record<string, number>;
  /** Couplings this protocol knowingly holds. Unaccepted ones fail preflight. */
  accepts?: Accepted[];
  /** Minimum seeds per grid point per arm. The CLI can raise it. */
  seeds: number;
  /** Minimum simulated seconds per trial. */
  seconds: number;
  soupCount: number;
  /** Simulated seconds between samples; default 5 under three minutes, else 10. */
  sampleEvery?: number;
  keepNets?: boolean;
  outcomes: Outcome[];
  preconditions: Precondition[];
  notes?: string;
}

/** The axis name the arm index is filed under when arms are read together. */
export const ARM_AXIS = 'arm';

/*
 * The two chemistry regimes as arms, because they are the canonical case of
 * constants that cannot be shared: `senseScale` is three orders apart between
 * them, `uptakeVmax` meters one species in one and four in the other, and
 * `deposit` exists in only one. Shared by three protocols below.
 */
const MINTED: Arm = { name: 'minted', set: { excreteRate: 0, senseScale: 4.3, uptakeVmax: 0, catCoSubstrate: 0 } };
const CONSERVED: Arm = {
  name: 'conserved',
  set: { excreteRate: 0.015, senseScale: 0.002, uptakeVmax: 6, catCoSubstrate: 1 },
};
const ACCEPT_DEPOSIT: Accepted = {
  axis: 'excreteRate',
  constant: 'deposit',
  because: 'deposit is inert in the conserved arm and the minted arm keeps its default; there is nothing to re-choose',
};

export const PROTOCOLS: Protocol[] = [
  /*
   * The precondition for every question about resource structure, asked on
   * its own because it is cheap: a spatial statistic on one frame, CV ~0.18,
   * against depth's 0.73. Measured 2026-09-08: uniform never exceeds 1
   * (0.96 ± 0.05); eight patches peak at 1.95 ± 0.50; forty-eight at 1.36.
   * The effect is a transient — bodies spawn empty, forage, fill, stop — so
   * the outcome is the peak, and the run is short. `energyRegrow` is held at
   * zero because it is the only clean control for a layout comparison: with
   * growth on, a uniform dish at cap produces nothing and patches manufacture
   * ground, and the two layouts end 50% apart in mass.
   */
  {
    name: 'forage-engages',
    question: 'Do hungry bodies find structured ground, or graze wherever they happen to stand?',
    prediction:
      'forage_ratio peaks well above 1 where the ground is in patches and stays near 1 where it is uniform, ' +
      'at equal total mass; demand_mean rises above 0.1 in every arm because bodies spawn empty.',
    nullReading:
      'If demand never rose, the seeded pathway was never engaged and the run says nothing about foraging: ' +
      'look at uptake and ambient before anything else. If demand rose and forage stayed near 1 on patches, ' +
      'the seeded pathway does not steer at this patch spacing — compare sensorDist with the patch size ' +
      'before touching the genome.',
    arms: [{ name: 'default', set: {} }],
    axes: { groundPatches: [0, 8, 48] },
    base: { energyRegrow: 0 },
    seeds: 5,
    seconds: 180,
    soupCount: 500,
    sampleEvery: 5,
    outcomes: [
      { metric: 'forage_ratio', summary: 'peak', expect: 'differs', along: 'groundPatches' },
      { metric: 'demand_mean', summary: 'peak' },
      { metric: 'full_mean', summary: 'trough' },
    ],
    preconditions: [
      {
        metric: 'demand_mean',
        summary: 'peak',
        min: 0.1,
        why: 'the seeded foraging pathway is gated on DEMAND; a pond that was never hungry cannot forage',
      },
    ],
  },

  /*
   * The question the structure sweeps were trying to ask and could not,
   * because the bodies were never hungry (uptakeVmax 6 kept every tank at
   * 1.20 of 1.25) and because depth was the outcome. `netFst` is the
   * divergence number and its CV (~0.26) puts a 30% effect within reach of
   * eight seeds. The forage precondition on the patchy level is protocol one's
   * result made a gate: if the bodies never used the structure, the question
   * was not asked.
   */
  {
    name: 'structure-shapes-nets',
    question: 'Does where the food is change how nets develop — do nets become more different from each other?',
    prediction:
      'net_fst at the close is higher on patchy ground than on uniform ground at equal total mass and no regrowth, ' +
      'and nets_effective is not lower.',
    nullReading:
      'netFst wants about eight seeds for a 30% difference; unresolved means unresolved, not absent. ' +
      'Read ground and forage peak alongside: if forage never exceeded 1.2 on patches the bodies never used ' +
      'the structure. If they did and netFst did not move, structure at this scale is not what differentiates ' +
      'nets, and the next lever is scarcity (contested-ground), not layout.',
    arms: [{ name: 'default', set: {} }],
    axes: { groundPatches: [0, 8] },
    base: { energyRegrow: 0 },
    seeds: 8,
    seconds: 600,
    soupCount: 500,
    outcomes: [
      { metric: 'net_fst', expect: 'up' },
      { metric: 'nets_effective' },
      { metric: 'born_mean' },
      { metric: 'ground' },
      { metric: 'forage_ratio', summary: 'peak' },
    ],
    preconditions: [
      {
        metric: 'forage_ratio',
        summary: 'peak',
        min: 1.2,
        where: { groundPatches: 8 },
        why: 'bodies have to have found the patches for structure to have had a chance to matter',
      },
      { metric: 'demand_mean', summary: 'peak', min: 0.1, why: 'the pathway that finds food is gated on demand' },
    ],
  },

  /*
   * The Baldwin question, put as a claim about three named genes rather than
   * an aggregate. The learning horizon defaults to a third of a second and a
   * foraging trip is five to twelve, so the horizon is raised to cover a trip
   * (0.999 a frame is ~17 s); that is a knowing hold of the groundPatches
   * coupling, accepted below, and the asymmetry it creates is the prediction.
   * Within-life mastery is implausible — seven trips in an 83-second tank —
   * so the mechanism is inheritance: `inheritLearned` writes a parent's
   * learned bias into its children, and a lineage compounds it. That is why
   * the outcomes are the inherited loci and why nets are kept.
   */
  {
    name: 'baldwin-hunger',
    question:
      'Where acting on hunger pays, does a lineage accumulate a bias to act on it — a longer hunger memory ' +
      'and a stronger demand pathway — through learning consolidated into the genome?',
    prediction:
      'locus_self_00 (Wh[0][0]) and locus_demand_h0 (Wx[0][DEMAND]) drift upward on patchy ground and not on ' +
      'uniform ground, with learning on and a critic horizon that covers a trip.',
    nullReading:
      'Both loci drifting equally in both arms is drift, which every unseeded weight does. Neither moving means ' +
      'the horizon or the episode count is too small even with inheritance. Ten trials on a single locus is ' +
      'thin; if the gap is small say unresolved, do not reach for it.',
    arms: [{ name: 'default', set: {} }],
    axes: { groundPatches: [0, 8] },
    base: { learnRate: 0.01, learnDiscount: 0.999, learnTrace: 0.99, energyRegrow: 0 },
    accepts: [
      {
        axis: 'groundPatches',
        constant: 'learnDiscount',
        because:
          'the horizon is chosen for the patchy trip and held on uniform ground, where there is no trip; the ' +
          'prediction is exactly that the same horizon pays in one arm and not the other',
      },
      { axis: 'groundPatches', constant: 'learnTrace', because: 'as learnDiscount' },
    ],
    seeds: 5,
    seconds: 600,
    soupCount: 500,
    keepNets: true,
    outcomes: [
      { metric: 'locus_self_00', expect: 'up' },
      { metric: 'locus_demand_h0', expect: 'up' },
      { metric: 'locus_self_00', summary: 'slope' },
      { metric: 'net_fst' },
      { metric: 'born_mean' },
    ],
    preconditions: [
      {
        metric: 'forage_ratio',
        summary: 'peak',
        min: 1.2,
        where: { groundPatches: 8 },
        why: 'acting on hunger has to have paid somewhere for a bias toward it to be selected',
      },
      { metric: 'born_mean', min: 3, why: 'inheritance needs generations; under three deep there is nothing to compound' },
    ],
  },

  /*
   * The eighteen-fold claim from the params comment, measured properly. The
   * shipped spacing forces were tuned for how a pond looks; the measurement
   * that they cost reproduction was 45 seconds by hand. `canPay` is the gauge
   * that separates a meeting problem from an economy problem: rich idle
   * bodies that are not meeting is the mechanism claimed, so canPay has to be
   * high for the claim to be about spacing at all.
   */
  {
    name: 'spacing-costs-breeding',
    question: 'Do personal space and alignment cost reproduction at equal age, and only together?',
    prediction:
      'the commute rate and depth are highest with both forces at zero and near-unchanged with either alone; ' +
      'canPay stays high throughout, so the shortfall is meeting, not energy.',
    nullReading:
      'If canPay is low the pond is poor and spacing is not what is being measured. If both-off does not beat ' +
      'the others the earlier 45-second reading was warm-up, which its own comment allowed for.',
    arms: [{ name: 'default', set: {} }],
    axes: { declutter: [0, 1.4], flockAlign: [0, 5.5] },
    seeds: 5,
    seconds: 300,
    soupCount: 300,
    outcomes: [
      { metric: 'commutes', summary: 'window', expect: 'down', along: 'declutter' },
      { metric: 'commutes', summary: 'window', expect: 'down', along: 'flockAlign' },
      { metric: 'born_mean' },
      { metric: 'con_dup_wires' },
      { metric: 'commute_edge' },
    ],
    preconditions: [
      { metric: 'can_pay', summary: 'mean', min: 0.5, why: 'the claim is that bodies are rich and idle; a poor pond is a different claim' },
    ],
  },

  /*
   * What conserved signalling costs, at the regime the sweeps picked, with
   * the thing the sweeps forgot to measure at first — whether anything is
   * left in the water to hear. Depth is the outcome the author asked about so
   * it stays, at a seed count that can only resolve a large gap; signal and
   * netFst are the cheaper reads. `sense_read_p90` is the gauge that the two
   * arms are each seeing their own signal at a scale phi can resolve.
   */
  {
    name: 'conserved-signal-price',
    question: 'What does paying for what you say cost reproduction, and does it leave anything in the water to hear?',
    prediction:
      'the conserved arm reproduces at no worse than half the minted arm\'s depth, keeps signal_total well above ' +
      'zero, and is at least as differentiated (net_fst).',
    nullReading:
      'Depth at CV 0.73 will not resolve a twofold gap at five seeds unless the gap is large; read the commute ' +
      'rate and netFst first. If signal_total is near zero in the conserved arm the pond is not a cheaper ' +
      'signalling pond, it is the pre-chemistry pond with extra machinery.',
    arms: [MINTED, CONSERVED],
    axes: {},
    accepts: [ACCEPT_DEPOSIT],
    seeds: 5,
    seconds: 600,
    soupCount: 500,
    outcomes: [
      { metric: 'born_mean', expect: 'down' },
      { metric: 'commutes', summary: 'window', expect: 'down' },
      { metric: 'signal_total' },
      { metric: 'net_fst' },
      { metric: 'lines_effective' },
    ],
    preconditions: [
      { metric: 'signal_total', min: 1, arms: ['conserved'], why: 'a silent conserved pond is not the mechanism' },
      {
        metric: 'sense_read_p90',
        min: 0.1,
        max: 10,
        why: 'each arm must read its own signal near the range phi resolves; see couplings excreteRate/senseScale',
      },
    ],
  },

  /*
   * Phase 7b of the chemistry plan: no sweep so far has produced a contested
   * dish — 500 bodies dent an ungrowing default dish by 5% a minute and never
   * bare a cell. Under strict conservation eating a signalling species cannot
   * beat eating the ground unless the ground is locally scarce, so this is a
   * change to the *conditions*: a thin, patchy, non-regrowing dish. The
   * preconditions are that scarcity actually happened — the ground fell and
   * somebody died — because a scarce arm in which nothing died is the puddle
   * with a smaller number. `uptakeKs` is held across the ambient axis on
   * purpose: slower uptake at low density is the scarcity under test.
   */
  {
    name: 'contested-ground',
    question: 'When the ground is genuinely scarce, does anyone die, and does reaching past the ground start to pay?',
    prediction:
      'in the scarce condition the ground falls over the run and the death rate is above zero; the conserved ' +
      'arm is more differentiated (net_fst) than the minted arm only where the ground is scarce.',
    nullReading:
      'If nothing dies at ambientEnergy 0.25 the dish is still not contested and the axis has to go lower or ' +
      'regrowth has to stay off longer; the mechanism was not tested. If deaths occur and netFst does not ' +
      'separate the arms, access to a second pool does not differentiate nets at this scale.',
    arms: [MINTED, CONSERVED],
    axes: { ambientEnergy: [1, 0.25] },
    base: { energyRegrow: 0, groundPatches: 8 },
    accepts: [
      ACCEPT_DEPOSIT,
      {
        axis: 'ambientEnergy',
        constant: 'uptakeKs',
        because: 'scarcity is the question; uptake slowing at low density is the mechanism under test, not a confound',
      },
    ],
    seeds: 5,
    seconds: 600,
    soupCount: 500,
    outcomes: [
      { metric: 'died', summary: 'window', expect: 'down', along: 'ambientEnergy' },
      { metric: 'ground', summary: 'slope', along: 'ambientEnergy' },
      { metric: 'net_fst', expect: 'up', along: ARM_AXIS },
      { metric: 'born_mean' },
      { metric: 'bodies' },
    ],
    preconditions: [
      { metric: 'died', summary: 'window', min: 1e-9, where: { ambientEnergy: 0.25 }, why: 'scarcity that kills nobody is not scarcity' },
      { metric: 'ground', summary: 'slope', max: 0, where: { ambientEnergy: 0.25 }, why: 'the dish has to be being spent down' },
    ],
  },
];

export function findProtocol(name: string): Protocol | undefined {
  return PROTOCOLS.find((p) => p.name === name);
}

/** The sweep an arm's runs are tagged with. Smoke runs get their own name so they never mix with findings. */
export function armSweepName(protocol: string, arm: string, smoke = false): string {
  return `${protocol}${smoke ? '~smoke' : ''}/${arm}`;
}

export interface PlanOptions {
  seeds?: number;
  seconds?: number;
  smoke?: boolean;
}

export interface ArmPlan {
  arm: Arm;
  sweep: string;
  base: Record<string, number>;
  grid: Record<string, number[]>;
  seeds: number[];
  seconds: number;
  soupCount: number;
  sampleEvery: number;
  keepNets: boolean;
  note: string;
}

export interface Plan {
  arms: ArmPlan[];
  points: number;
  trials: number;
  smoke: boolean;
}

function gridSize(axes: Record<string, number[]>): number {
  let n = 1;
  for (const v of Object.values(axes)) n *= v.length;
  return n;
}

/** What running this protocol would do, arm by arm. */
export function planProtocol(p: Protocol, opts: PlanOptions = {}): Plan {
  const smoke = opts.smoke === true;
  const seedCount = smoke ? 1 : (opts.seeds ?? p.seeds);
  const seconds = smoke ? Math.min(20, p.seconds) : (opts.seconds ?? p.seconds);
  const sampleEvery = smoke ? 5 : (p.sampleEvery ?? (seconds <= 180 ? 5 : 10));
  const seeds = Array.from({ length: Math.max(1, seedCount) }, (_, i) => i + 1);
  const note = `${p.name}: ${p.question} | predicts: ${p.prediction}`;
  const arms = p.arms.map((arm) => ({
    arm,
    sweep: armSweepName(p.name, arm.name, smoke),
    base: { ...(p.base ?? {}), ...arm.set },
    grid: p.axes,
    seeds,
    seconds,
    soupCount: p.soupCount,
    sampleEvery,
    keepNets: p.keepNets === true,
    note,
  }));
  const points = gridSize(p.axes);
  return { arms, points, trials: arms.length * points * seeds.length, smoke };
}

export interface Preflight {
  /** Anything here and the protocol must not run. */
  errors: string[];
  /** Worth reading before running. */
  warnings: string[];
  /** Couplings held on purpose, with the reasons — recorded, not suppressed. */
  accepted: string[];
}

/** Parameter keys set differently between arms: the arm axis, as parameters. */
export function variedAcrossArms(p: Protocol): Set<string> {
  const out = new Set<string>();
  const keys = new Set<string>();
  for (const a of p.arms) for (const k of Object.keys(a.set)) keys.add(k);
  for (const k of keys) {
    const values = p.arms.map((a) => a.set[k]);
    if (values.some((v) => v !== values[0])) out.add(k);
  }
  return out;
}

/**
 * Everything that can be checked before a trial runs.
 *
 * Names first, because a typo in a parameter should fail before the first
 * trial and not after the twentieth. Then the budget: fewer than three seeds
 * is a smoke run whatever it is called, and under two simulated minutes is
 * warm-up. Then the couplings, per arm, against what that arm holds — which
 * is where a protocol most often needs to say something in `accepts`.
 */
export function preflight(p: Protocol, opts: PlanOptions = {}): Preflight {
  const errors: string[] = [];
  const warnings: string[] = [];
  const accepted: string[] = [];
  const smoke = opts.smoke === true;
  const known = new Set(Object.keys(defaultParams()));

  if (!/^[a-z0-9][a-z0-9-]*$/.test(p.name)) errors.push(`name ${JSON.stringify(p.name)}: lowercase letters, digits and dashes`);
  if (p.arms.length === 0) errors.push('a protocol needs at least one arm');
  const armNames = new Set<string>();
  for (const a of p.arms) {
    if (!/^[a-z0-9][a-z0-9-]*$/.test(a.name)) errors.push(`arm ${JSON.stringify(a.name)}: lowercase letters, digits and dashes`);
    if (armNames.has(a.name)) errors.push(`arm ${JSON.stringify(a.name)} appears twice`);
    armNames.add(a.name);
    for (const k of Object.keys(a.set)) if (!known.has(k)) errors.push(`arm ${a.name} sets unknown parameter ${JSON.stringify(k)}`);
  }
  for (const k of Object.keys(p.base ?? {})) if (!known.has(k)) errors.push(`base sets unknown parameter ${JSON.stringify(k)}`);
  for (const [k, v] of Object.entries(p.axes)) {
    if (!known.has(k)) errors.push(`axis ${JSON.stringify(k)} is not a parameter`);
    if (v.length < 2) errors.push(`axis ${k} needs at least two levels`);
    if (k === ARM_AXIS) errors.push(`${ARM_AXIS} is reserved for the arms`);
  }

  const seedCount = smoke ? 1 : (opts.seeds ?? p.seeds);
  if (!smoke && seedCount < 3) errors.push(`${seedCount} seed(s): fewer than three is a smoke run; say --smoke`);
  if (!smoke && (opts.seeds ?? p.seeds) < p.seeds) {
    warnings.push(`${opts.seeds} seeds is under the protocol's minimum of ${p.seeds}; read the result as thin`);
  }
  const seconds = smoke ? Math.min(20, p.seconds) : (opts.seconds ?? p.seconds);
  if (!smoke && seconds < 120) warnings.push(`${seconds} s is warm-up for this pond; the protocol asks for ${p.seconds}`);
  if (!smoke && seconds < p.seconds) warnings.push(`${seconds} s is under the protocol's minimum of ${p.seconds}`);

  const axisKeys = new Set([ARM_AXIS, ...Object.keys(p.axes)]);
  const checkMetric = (where: string, metric: string, summary?: Summary): void => {
    if (!METRICS[metric]) errors.push(`${where}: unknown metric ${JSON.stringify(metric)}`);
    if (summary !== undefined && !SUMMARIES.includes(summary)) errors.push(`${where}: unknown summary ${JSON.stringify(summary)}`);
  };
  for (const o of p.outcomes) {
    checkMetric(`outcome ${o.metric}`, o.metric, o.summary);
    if (o.along !== undefined && !axisKeys.has(o.along)) errors.push(`outcome ${o.metric}: along ${JSON.stringify(o.along)} is not an axis`);
    if (o.along === ARM_AXIS && p.arms.length < 2) errors.push(`outcome ${o.metric}: along arm, but there is one arm`);
    if (o.expect && o.along === undefined && p.arms.length < 2 && Object.keys(p.axes).length === 0) {
      errors.push(`outcome ${o.metric}: expects a direction but there is nothing to read it along`);
    }
  }
  for (const c of p.preconditions) {
    checkMetric(`precondition ${c.metric}`, c.metric, c.summary);
    if (c.min === undefined && c.max === undefined) errors.push(`precondition ${c.metric}: no bound`);
    for (const a of c.arms ?? []) if (!armNames.has(a)) errors.push(`precondition ${c.metric}: no arm ${JSON.stringify(a)}`);
    for (const k of Object.keys(c.where ?? {})) if (!(k in p.axes)) errors.push(`precondition ${c.metric}: where ${k} is not an axis`);
  }

  /*
   * Couplings, per arm. The moving set is the grid's axes plus whatever the
   * arms set differently between them; what is held is the arm's whole
   * parameter set, so a coupling made moot by a neutral setting is skipped.
   */
  const varied = variedAcrossArms(p);
  const axes = new Set([...Object.keys(p.axes), ...varied]);
  const fired = new Map<string, CouplingWarning>();
  for (const a of p.arms) {
    const held = { ...defaultParams(), ...(p.base ?? {}), ...a.set } as unknown as Record<string, number>;
    for (const w of checkCouplings(axes, varied, held)) fired.set(`${w.axis}/${w.constant}`, w);
  }
  const accepts = p.accepts ?? [];
  for (const acc of accepts) {
    if (!known.has(acc.axis) || !known.has(acc.constant)) errors.push(`accepts ${acc.axis}/${acc.constant}: not parameters`);
  }
  for (const [key, w] of fired) {
    const acc = accepts.find((x) => `${x.axis}/${x.constant}` === key);
    if (acc) accepted.push(`${key}: ${acc.because}`);
    else warnings.push(`coupling ${key} is held across its axis.\n    why: ${w.why}\n    fix: ${w.fix}\n    (or list it in accepts, with the reason)`);
  }
  for (const acc of accepts) {
    if (!fired.has(`${acc.axis}/${acc.constant}`)) warnings.push(`accepts ${acc.axis}/${acc.constant}, which did not fire`);
  }
  return { errors, warnings, accepted };
}

/** The metric keys a protocol reads: outcomes, then preconditions, deduplicated. */
export function protocolKeys(p: Protocol): string[] {
  const keys: string[] = [];
  const add = (metric: string, summary?: Summary) => {
    const k = metricKey(metric, summary);
    if (!keys.includes(k)) keys.push(k);
  };
  for (const o of p.outcomes) add(o.metric, o.summary);
  for (const c of p.preconditions) add(c.metric, c.summary);
  return keys;
}

/** Every arm's trials, with the arm filed as an axis. */
export function protocolTrials(db: PondDb, p: Protocol, opts: { smoke?: boolean; warmup?: number } = {}): TrialRow[] {
  const keys = protocolKeys(p);
  const out: TrialRow[] = [];
  p.arms.forEach((arm, i) => {
    for (const t of sweepTrials(db, armSweepName(p.name, arm.name, opts.smoke === true), { metrics: keys, warmup: opts.warmup })) {
      out.push({ ...t, point: { ...t.point, [ARM_AXIS]: i } });
    }
  });
  return out;
}

export interface PreconditionResult {
  pre: Precondition;
  key: string;
  scope: string;
  passed: number;
  total: number;
  mean: number | null;
}

function inScope(p: Protocol, c: Precondition, t: TrialRow): boolean {
  if (c.arms) {
    const arm = p.arms[t.point[ARM_AXIS]];
    if (!arm || !c.arms.includes(arm.name)) return false;
  }
  for (const [k, v] of Object.entries(c.where ?? {})) if (t.point[k] !== v) return false;
  return true;
}

/** Each gauge against each trial it applies to. */
export function evaluatePreconditions(p: Protocol, trials: TrialRow[]): PreconditionResult[] {
  return p.preconditions.map((c) => {
    const key = metricKey(c.metric, c.summary);
    const scoped = trials.filter((t) => inScope(p, c, t));
    let passed = 0;
    let sum = 0;
    let count = 0;
    for (const t of scoped) {
      const v = t.values[key];
      if (v === null || v === undefined || !Number.isFinite(v)) continue;
      sum += v;
      count++;
      if ((c.min === undefined || v >= c.min) && (c.max === undefined || v <= c.max)) passed++;
    }
    const parts: string[] = [];
    if (c.arms) parts.push(`arm ${c.arms.join('|')}`);
    for (const [k, v] of Object.entries(c.where ?? {})) parts.push(`${k}=${v}`);
    return { pre: c, key, scope: parts.length > 0 ? parts.join(' ') : 'all', passed, total: scoped.length, mean: count > 0 ? sum / count : null };
  });
}

export interface Reading {
  outcome: Outcome;
  key: string;
  along: string;
  effect: Effect | null;
  direction: 'up' | 'down' | 'flat' | null;
  status: 'resolved' | 'unresolved' | 'thin' | 'no data';
  /** True or false against `expect`; null when there is no expectation or nothing resolved. */
  matches: boolean | null;
}

function defaultAlong(p: Protocol): string | null {
  if (p.arms.length > 1) return ARM_AXIS;
  const first = Object.keys(p.axes)[0];
  return first ?? null;
}

/** Each outcome against its prediction. */
export function readOutcomes(p: Protocol, trials: TrialRow[]): Reading[] {
  return p.outcomes.map((o) => {
    const key = metricKey(o.metric, o.summary);
    const along = o.along ?? defaultAlong(p) ?? ARM_AXIS;
    const effect = effects(trials, [key]).find((e) => e.axis === along) ?? null;
    let status: Reading['status'] = 'no data';
    let direction: Reading['direction'] = null;
    if (effect) {
      direction = effect.last.mean > effect.first.mean ? 'up' : effect.last.mean < effect.first.mean ? 'down' : 'flat';
      const perLevel = effect.trials / effect.levels;
      status = perLevel < THIN_SEEDS ? 'thin' : effect.eta2 >= RESOLVED_ETA2 ? 'resolved' : 'unresolved';
    }
    let matches: boolean | null = null;
    if (o.expect && status === 'resolved') {
      matches = o.expect === 'differs' ? true : direction === o.expect;
    }
    return { outcome: o, key, along, effect, direction, status, matches };
  });
}

function fmt(n: number | null): string {
  if (n === null || !Number.isFinite(n)) return '-';
  if (Number.isInteger(n)) return String(n);
  const a = Math.abs(n);
  return a >= 100 ? n.toFixed(0) : a >= 1 ? n.toFixed(2) : n.toFixed(4);
}

function setText(set: Record<string, number>): string {
  const parts = Object.entries(set).map(([k, v]) => `${k}=${v}`);
  return parts.length > 0 ? parts.join(' ') : '(defaults)';
}

/** The protocol as a paragraph, for `protocols` and the head of a report. */
export function describeProtocol(p: Protocol): string {
  const lines = [
    `protocol ${p.name}`,
    `  question:   ${p.question}`,
    `  prediction: ${p.prediction}`,
    `  null:       ${p.nullReading}`,
  ];
  p.arms.forEach((a, i) => lines.push(`  arm ${i} ${a.name}: ${setText(a.set)}`));
  const axes = Object.entries(p.axes).map(([k, v]) => `${k}=${v.join(',')}`);
  lines.push(`  axes:       ${axes.length > 0 ? axes.join('  ') : '(none; the arms are the axis)'}`);
  lines.push(`  held:       ${setText(p.base ?? {})}`);
  lines.push(`  budget:     ${p.seeds} seeds x ${gridSize(p.axes)} point(s) x ${p.arms.length} arm(s), ${p.seconds} s, ${p.soupCount} founders`);
  const gauges = p.preconditions.map((c) => {
    const bound = [c.min !== undefined ? `>= ${c.min}` : '', c.max !== undefined ? `<= ${c.max}` : ''].filter(Boolean).join(' and ');
    return `${metricKey(c.metric, c.summary)} ${bound}`;
  });
  lines.push(`  gauges:     ${gauges.length > 0 ? gauges.join('; ') : '(none)'}`);
  lines.push(`  outcomes:   ${p.outcomes.map((o) => metricKey(o.metric, o.summary) + (o.expect ? ` ${o.expect}` : '')).join('; ')}`);
  if (p.notes) lines.push(`  notes:      ${p.notes}`);
  return lines.join('\n');
}

/**
 * Read a protocol's runs back in its own terms.
 *
 * Preconditions first, per trial, because an outcome in an arm whose
 * mechanism never engaged is not an outcome. Then the point table and the
 * effect table with the arm as an axis. Then each outcome against its
 * prediction, and the null reading written before the run whenever an
 * outcome is not resolved.
 */
export function protocolReport(db: PondDb, p: Protocol, opts: { smoke?: boolean; warmup?: number } = {}): string {
  const smoke = opts.smoke === true;
  const trials = protocolTrials(db, p, opts);
  const out: string[] = [describeProtocol(p), ''];
  if (smoke) out.push('SMOKE RUN: plumbing only. These numbers are not findings and are tagged apart from them.', '');
  if (trials.length === 0) {
    out.push(`no runs in this library for ${p.name}${smoke ? ' (smoke)' : ''}. Run: npm run pond -- protocol ${p.name}${smoke ? ' --smoke' : ''}`);
    return out.join('\n');
  }
  const perArm = p.arms.map((_, i) => trials.filter((t) => t.point[ARM_AXIS] === i).length);
  out.push(`trials: ${trials.length} (${perArm.map((n, i) => `${p.arms[i].name} ${n}`).join(', ')})`, '');

  const levelName = (axis: string, level: number): string =>
    axis === ARM_AXIS ? (p.arms[level]?.name ?? String(level)) : fmt(level);

  const pres = evaluatePreconditions(p, trials);
  if (pres.length > 0) {
    out.push('preconditions: was the mechanism engaged?');
    out.push(
      table(
        ['gauge', 'bound', 'scope', 'met', 'mean', 'why'],
        pres.map((r) => [
          metricLabel(r.key),
          [r.pre.min !== undefined ? `>= ${r.pre.min}` : '', r.pre.max !== undefined ? `<= ${r.pre.max}` : ''].filter(Boolean).join(' and '),
          r.scope,
          `${r.passed}/${r.total}${r.total > 0 && r.passed === r.total ? ' MET' : r.passed === 0 ? ' UNMET' : ' partial'}`,
          fmt(r.mean),
          r.pre.why,
        ]),
      ),
      '',
    );
  }

  const keys = protocolKeys(p);
  out.push('per point', pointTable(trials, keys, { levelName }), '');
  const effs = effects(trials, keys);
  out.push('effect of each axis, by share of variance explained', effectTable(effs, 40, { levelName }), '');

  const readings = readOutcomes(p, trials);
  out.push('reading, against the prediction');
  out.push(
    table(
      ['outcome', 'along', 'first -> last', 'eta2', 'read', 'prediction'],
      readings.map((r) => {
        const e = r.effect;
        const span = e ? `${levelName(r.along, e.first.level)}: ${fmt(e.first.mean)} -> ${levelName(r.along, e.last.level)}: ${fmt(e.last.mean)}` : '-';
        const verdict =
          r.outcome.expect === undefined
            ? '(no expectation)'
            : r.matches === null
              ? `expected ${r.outcome.expect}; ${r.status}`
              : r.matches
                ? `expected ${r.outcome.expect}: MATCHES`
                : `expected ${r.outcome.expect}, saw ${r.direction}: DOES NOT MATCH`;
        return [metricLabel(r.key), r.along, span, e ? fmt(e.eta2) : '-', e ? readEffect(e) : r.status, verdict];
      }),
    ),
    '',
  );
  const unmet = pres.filter((r) => r.total > 0 && r.passed < r.total);
  if (unmet.length > 0) {
    out.push(`gauges unmet in ${unmet.map((r) => `${metricLabel(r.key)} (${r.total - r.passed} trial(s))`).join(', ')}: outcomes there are not readings about the mechanism.`);
  }
  if (readings.some((r) => r.status !== 'resolved')) {
    out.push(`null reading, as written before the run: ${p.nullReading}`);
  }
  return out.join('\n');
}
