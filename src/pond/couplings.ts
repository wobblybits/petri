import type { Params } from '../params.ts';

/*
 * Dials whose meaning depends on another dial.
 *
 * A sweep holds every parameter it is not varying at one value and assumes
 * that value means the same thing at every point of the grid. It does not.
 * `senseScale` is the right gain for a minted pond and three orders too large
 * for an excreting one; `energyRegrow` is a producer on a patchy dish and
 * nothing at all on a uniform one seeded at capacity. Both were held constant
 * across the axis that changed their meaning, in sweeps about exactly the
 * thing they blinded — four sweeps and two afternoons, in one session.
 *
 * The harness cannot detect this. The assumption is about the model, not the
 * grid, so the model's own couplings have to be written down where the
 * harness can read them. That is this table. `sweep` and `protocol` warn when
 * an axis is crossed with a coupled constant that does not move with it; a
 * protocol that knowingly holds one says so in `accepts`, with the reason,
 * and the reason is recorded rather than the warning suppressed.
 *
 * Add a row the moment a held constant is found to mean two things. The
 * table is the memory this project did not have.
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
  /**
   * A held setting at which the coupling is moot: a learning horizon is
   * irrelevant with `learnRate` at 0. Usually the constant's own neutral
   * value, sometimes another dial's.
   */
  unless?: { key: keyof Params; is: number };
}

/*
 * Retired, all three: `excreteRate` x `senseScale`, x `deposit`, and x the
 * producer's income.
 *
 * They were the couplings of a pond where a body paid for its voice out of its
 * tank, so the signal field was matter and its scale moved three orders
 * between the two regimes. A body eats the ground and excretes nothing now, so
 * a signal is always minted, `excreteRate` is gone, and there is no second
 * regime for anything to be coupled across. Kept as a note because a coupling
 * that quietly stops being one is as misleading as one that is missed.
 */
export const COUPLINGS: readonly Coupling[] = [
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
    axis: 'groundDropEvery',
    constant: 'energyRegrow',
    why:
      'at groundDropEvery 0 regrowth is the whole income and above it there are two, and they are ' +
      'not the same income: regrowth heals living ground in place and can never re-green a cell at ' +
      'zero, a drop puts new ground where there was none. Held across the axis, the fast end of it ' +
      'is a richer pond as well as a lumpier one',
    fix: 'choose energyRegrow per arm so the per-second income matches, and read structure not quantity',
    unless: { key: 'energyRegrow', is: 0 },
  },
  {
    axis: 'groundDropEvery',
    constant: 'groundPatches',
    why:
      'a drop is one patch at the current grain — Energy.patchRadius and patchMass are both ' +
      'divided by groundPatches — so the same drop interval lays a quarter-disk meadow at 1 patch ' +
      'and a freckle at 128',
    fix: 'hold groundPatches within an arm; crossing both axes is a grid, not a held constant',
    unless: { key: 'groundDropEvery', is: 0 },
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
 * Which couplings a sweep trips.
 *
 * `axes` are the parameters the sweep varies; `varied` are parameters that
 * move with them by some other route — set differently per arm, say; `held`
 * is what everything else is set to. A warning is a coupled constant of some
 * axis that is in neither moving set and not at a setting that makes the
 * coupling moot: held at one value across an axis that changes what the value
 * means.
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
