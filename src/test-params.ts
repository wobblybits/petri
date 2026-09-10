import { defaultParams, type Params } from './params.ts';
import { rewriteCost } from './energy.ts';
import { CHANNELS } from './fields.ts';
import type { Sim } from './sim.ts';

/**
 * The shipping pond with lifetime learning switched off.
 *
 * `npm test` is a change detector for mechanical rewrites, and nearly every
 * assertion in it is about what a *fixed* genome does: a weight that stays at
 * zero, a body that cruises at its seeded speed, momentum that is conserved
 * because nothing is steering. `learnRate` above zero makes all of those
 * probabilistic — the weights those assertions are about now move while the
 * test runs, so a test either passes or fails on how far a body happened to
 * drift before the third frame.
 *
 * That is not a reason to leave learning off in the pond. It is a reason for
 * a test that is not about learning to say so, by pinning the rate here
 * rather than inheriting whatever `defaultParams` currently ships. The tests
 * that *are* about learning — `plasticity.test.ts`, `learn-dirty.test.ts` —
 * keep calling `defaultParams` and set their own rate, which is what makes
 * them the only place the default's value can break the suite.
 *
 * Everything else is the shipping default, deliberately: pinning one dial to
 * keep a measurement honest is different from running the suite against a
 * pond nobody ships.
 */
export function fixedParams(): Params {
  const params = defaultParams();
  params.learnRate = 0;
  return params;
}

/** Standing stock of one species across the whole field. */
export function channelTotal(sim: Sim, ch: number): number {
  const d = sim.fields.data;
  let s = 0;
  for (let k = ch; k < d.length; k += CHANNELS) s += d[k];
  return s;
}

/** Every species, every cell. */
export function fieldTotal(sim: Sim): number {
  const d = sim.fields.data;
  let s = 0;
  for (let k = 0; k < d.length; k++) s += d[k];
  return s;
}

/**
 * Everything in the pond: what bodies are made of and hold, what is in
 * flight, in escrow, or in a gut, and every species in the field.
 *
 * A body's *existence* has to be in here, not just its stock. Deaths and
 * rewrites move `bodyValue` between the two, and a total that counted only
 * `extra` would read every commute as matter appearing — which is exactly
 * what it read, at four per cent over three simulated seconds, before this
 * counted bodies. Pair it with `bodyValue = REWRITE_SHARE`, which is what
 * makes those transfers conservative in the first place. Debt counts against
 * it: a body one unit into debt holds `bodyValue - 1` of real matter, and
 * `deathYield` releases exactly that when it dies, so flooring at zero the
 * way `totalFree` does would make every starvation look like matter
 * vanishing when what vanished was a debt.
 *
 * In flight: a rewrite charges its pair `rewriteCost` when it *begins* and
 * pays the pool out when it *commits*, and in between the shares are held by
 * the `Rewrite` itself — not by a body, not by the escrow map. Only a commute
 * costs shares up front; an erase or an annihilation is free to start and
 * pays out on commit.
 *
 * The whole field, every channel: rent leaves through the excretion rows, so
 * a Con pays it in `conP` and `aux`, and a total that counted the ground
 * alone would read that as matter going missing. Matter is matter whatever
 * molecule it is in. And the gut, which is in neither a tank nor the ground
 * and is not nothing — see `Sim.totalGut`.
 */
export function pondMatter(sim: Sim, bodyValue: number): number {
  let held = 0;
  for (const a of sim.agents.values()) held += bodyValue + a.extra;
  let inFlight = 0;
  for (const rw of sim.rewrites) inFlight += rewriteCost(rw.rule);
  return held + inFlight + sim.escrowTotal() + sim.totalGut() + fieldTotal(sim);
}
