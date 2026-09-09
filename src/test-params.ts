import { defaultParams, type Params } from './params.ts';

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
