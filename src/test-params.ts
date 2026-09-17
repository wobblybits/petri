import { defaultParams, type Params } from './params.ts';
import { rewriteCost } from './energy.ts';
import { CH, CHANNELS } from './fields.ts';
import type { Sim } from './sim.ts';

/**
 * The shipping pond with lifetime learning and the metabolic pathway switched
 * off.
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
 * `metabolicRate` is pinned for the same reason and it is the stronger case.
 * The pathway *spends*: every body buys substrate out of its own tank in
 * proportion to how discharged it is, and the price leaves through the same
 * road rent does. So with it running, "a still body loses nothing", "a full
 * body has nothing to learn from", "nothing else should be draining it" and
 * "full ground should read about 1" are all false — not because the mechanism
 * they test broke, but because a second spender arrived and the body then ate
 * to cover it. It also swings every wire's rest length, which moves a chord
 * and a bound-disc radius by a few per cent.
 *
 * A test about the pathway says so by turning it on: `gait.test.ts` is the
 * one place it runs, and it sets its own rate. That is the same arrangement
 * learning has, and for the same reason.
 *
 * `uptakeVmax` is pinned for a third reason, and it is
 * about isolation rather than noise. They switch the *whole* food path: above
 * zero a mouthful is a sample of four species landing in a gut, which a
 * recipe then digests, and what a body says leaves its tank conserved instead
 * of being minted. Almost every assertion in this suite about a tank, a
 * commute's stake or a rescue was written against the take-what-fits path,
 * and the two paths are not small variations on each other. A test about
 * *eating* says so by turning them on — `chemistry.test.ts` does, and it is
 * the file that owns them.
 *
 * `groundPatches` is pinned for a fourth reason, and it is the plainest of
 * the four: it decides *where the food is*. The shipped dish gathers its mass
 * into blobs over a quarter of the disk, so three quarters of it is bare —
 * which means a body spawned at a coordinate a test picked is, four times out
 * of five, standing on nothing. Every assertion about a body that eats, and
 * every assertion about a body that moves without being steered, is then
 * about whether that coordinate happened to land in a patch. Flat is not the
 * pond; it is the control, and it is what an isolated mechanic needs
 * underneath it for the same reason a bench pins everything but its axis.
 *
 * A test about foraging says so by asking for patches: `pond/ground.test.ts`
 * and the forage measures in `pond/measure.test.ts` do, and they own it.
 *
 * Everything else is the shipping default, deliberately: pinning five dials
 * to keep a measurement honest is different from running the suite against a
 * pond nobody ships.
 */
export function fixedParams(): Params {
  const params = defaultParams();
  params.learnRate = 0;
  params.metabolicRate = 0;
  params.uptakeVmax = 0;
  params.groundPatches = 0;
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
function fieldTotal(sim: Sim): number {
  const d = sim.fields.data;
  let s = 0;
  // The ground alone. A signal is not matter: nothing eats it, nothing
  // excretes it, and `params.deposit` mints it out of nothing on purpose.
  for (let k = CH.energy; k < d.length; k += CHANNELS) s += d[k];
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
 * The field's **ground channel only**. A body eats the ground and nothing
 * else, and it excretes nothing at all, so the three signalling channels never
 * hold matter — they are minted by `params.deposit` and smelled, and counting
 * them would read every shout as creation. One substance, one column. And the
 * gut, which is in neither a tank nor the ground and is not nothing — see
 * `Sim.totalGut`.
 */
export function pondMatter(sim: Sim, bodyValue: number): number {
  let held = 0;
  for (const a of sim.agents.values()) held += bodyValue + a.extra;
  let inFlight = 0;
  for (const rw of sim.rewrites) inFlight += rewriteCost(rw.rule);
  return held + inFlight + sim.escrowTotal() + sim.totalGut() + fieldTotal(sim);
}
