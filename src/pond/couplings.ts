import type { Params } from '../params.ts';

/*
 * Dials whose meaning depends on another dial. A sweep holds every parameter
 * it is not varying at one value and assumes that value means the same thing
 * at every point of the grid; the harness cannot detect where it does not,
 * so the model's couplings are written down here. `sweep` and `protocol`
 * warn when an axis is crossed with a coupled constant that does not move
 * with it; a protocol that knowingly holds one says so in `accepts`, with
 * the reason. Add a row the moment a held constant is found to mean two things.
 */
export interface Coupling {
  /** The dial whose setting changes what `constant` means. */
  axis: keyof Params;
  /** The dial that has to be re-chosen per level of `axis`. */
  constant: keyof Params;
  /** Why one value cannot serve both ends of the axis. */
  why: string;
  /** What to do instead. */
  fix: string;
  /** A held setting at which the coupling is moot: usually the constant's own neutral value, sometimes another dial's. */
  unless?: { key: keyof Params; is: number };
}

export const COUPLINGS: readonly Coupling[] = [
  {
    axis: 'excreteRate',
    constant: 'senseScale',
    why:
      'minted signal reads p90 ~4.3 at a body, conserved excretion ~0.002; ' +
      'one scale leaves one arm reading three orders outside what phi can resolve',
    fix: 'senseScale 4.3 when excreteRate is 0, ~0.002 when it is on; run the two regimes as arms',
  },
  {
    axis: 'uptakeVmax',
    constant: 'digestRate',
    why:
      'nothing fills a gut at uptakeVmax 0, so digestRate is inert there and decides how much ' +
      'of a mouthful is ever worth anything above it',
    fix: 'read digestRate within a metered arm, never across the uptakeVmax 0 boundary',
  },
  {
    axis: 'uptakeVmax',
    constant: 'gutSize',
    why: 'nothing fills a gut at uptakeVmax 0, so gutSize bounds nothing there and bounds every mouthful above it',
    fix: 'read gutSize within a metered arm, never across the uptakeVmax 0 boundary',
  },
  {
    axis: 'excreteRate',
    constant: 'deposit',
    why: 'the minted deposit is off whenever excretion is on, so deposit is inert above zero',
    fix: 'do not sweep deposit against excreteRate; it only exists in the minted arm',
  },
  {
    axis: 'groundPatches',
    constant: 'energyRegrow',
    why:
      'logistic growth is zero at cap and Fields.grow skips a cell at e >= cap, so a uniform dish ' +
      'seeded at cellCap produces nothing while patches drag their edges through the productive ' +
      'band; layouts seeded at equal mass ended 50% apart',
    fix: 'energyRegrow = 0 is the clean control for a layout question',
    unless: { key: 'energyRegrow', is: 0 },
  },
  {
    axis: 'groundPatches',
    constant: 'learnDiscount',
    why:
      'the critic horizon has to cover the trip to the reward, and patch spacing sets the trip: ' +
      '5-12 s against a default horizon of 0.33 s (0.95 a frame)',
    fix: 'choose learnDiscount and learnTrace from the trip time, and accept the coupling in writing',
    unless: { key: 'learnRate', is: 0 },
  },
  {
    axis: 'groundPatches',
    constant: 'learnTrace',
    why: 'the eligibility window has to reach back to the action that started the trip; see learnDiscount',
    fix: 'as learnDiscount',
    unless: { key: 'learnRate', is: 0 },
  },
  {
    axis: 'ambientEnergy',
    constant: 'uptakeKs',
    why: 'a half-saturation constant only means something relative to the ground density it is measured against',
    fix: 'scale uptakeKs with ambientEnergy, or accept it and read uptake as the mechanism rather than a confound',
  },
  {
    axis: 'soupCount',
    constant: 'declutter',
    why:
      'the spacing forces gate the opening scramble, whose severity is set by density; ' +
      'the 18x on depth was measured at 250 bodies',
    fix: 'sweep spacing at one density, or put soupCount on its own axis and read the interaction',
  },
  {
    axis: 'soupCount',
    constant: 'flockAlign',
    why: 'as declutter: alignment turns a head-on approach into a shoal, and how often that happens is density',
    fix: 'as declutter',
  },
  {
    axis: 'energyRegrow',
    constant: 'fertilise',
    why: 'fertilise scales the regrowth rate, and scaling zero is zero',
    fix: 'fertilise is only readable where energyRegrow is non-zero',
    unless: { key: 'fertilise', is: 0 },
  },
  {
    axis: 'eraUpkeepRatio',
    constant: 'excreteRate',
    why:
      'a seeded Era earns ~0.003/s from the producer discount at -0.2 and nothing at all at 1, ' +
      'and excreteRate is what it spends making ground; above its income it starves',
    fix: 'choose excreteRate against the producer income each arm actually has',
    unless: { key: 'excreteRate', is: 0 },
  },
  {
    axis: 'grip',
    constant: 'transportRecoil',
    why:
      'grip only turns an impulse into displacement, and the impulse along a wire is the recoil; ' +
      'at recoil 0 grip is not a net stroke at all but a change to how far a lone body coasts, ' +
      'which is a different mechanism answering a different question',
    fix: 'hold transportRecoil above zero when asking whether grip carries a net; run recoil 0 as the loner control',
    unless: { key: 'grip', is: 0 },
  },
  {
    axis: 'transportQuantum',
    constant: 'transportRecoil',
    why:
      'the kick is recoil times the amount moved, and quantising changes the amount by three orders: ' +
      'a continuous transfer is ~4e-4 and a 0.5 packet is 0.5, so one recoil means a nudge of 0.04 in ' +
      'one arm and 50 in the other',
    fix: 'scale transportRecoil down as the quantum goes up, or read the two arms as different ponds',
    unless: { key: 'transportRecoil', is: 0 },
  },
];

export interface CouplingWarning {
  axis: string;
  constant: string;
  why: string;
  fix: string;
}

/**
 * Which couplings a sweep trips. `axes` are the parameters the sweep varies;
 * `varied` are parameters that move with them by some other route; `held` is
 * what everything else is set to. A warning is a coupled constant of some
 * axis that is in neither moving set and not at a moot setting.
 */
export function checkCouplings(
  axes: Iterable<string>,
  varied: Iterable<string> = [],
  held: Partial<Record<string, number>> = {},
): CouplingWarning[] {
  const moving = new Set<string>(varied);
  for (const a of axes) moving.add(a);
  const out: CouplingWarning[] = [];
  for (const axis of new Set(axes)) {
    for (const c of COUPLINGS) {
      if (c.axis !== axis || moving.has(c.constant)) continue;
      if (c.unless && !moving.has(c.unless.key) && held[c.unless.key] === c.unless.is) continue;
      out.push({ axis, constant: c.constant, why: c.why, fix: c.fix });
    }
  }
  return out;
}

/** One line per warning, for stderr. */
export function formatCouplings(list: CouplingWarning[]): string {
  return list
    .map((w) => `coupling: ${w.axis} is on an axis but ${w.constant} is held.\n    why: ${w.why}\n    fix: ${w.fix}`)
    .join('\n');
}
