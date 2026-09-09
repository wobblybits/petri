import { execFileSync } from 'node:child_process';

/**
 * The working tree a run came from — `9a3f21c`, or `9a3f21c+dirty`.
 *
 * A run row records the commit so the library can be read a month later, and
 * an uncommitted edit to `src/` makes that hash a lie about what ran. It
 * happened within a day of the library existing: a physics sweep inherited
 * `learnRate 0.02` and `learnDiscount 0.99` from a browser session someone
 * was in the middle of, held them as constants across every trial, and
 * recorded a commit where those are 0 and 0.95. The full parameters are
 * stored, so nothing was lost — but nothing said to go and look, either.
 *
 * Only `src/` counts: a note or a plan edited mid-run changes no arithmetic.
 *
 * In its own module because there were two copies of this function, one in
 * `cli.ts` and one in `sweep.ts`, and fixing the wrong one is how the first
 * sweep after the fix still recorded a clean hash from a dirty tree. The
 * same shape as the hand-copied `CHEM_LEN` that drifted twice.
 */
export function gitCommit(): string | null {
  try {
    const head = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
    const dirty = execFileSync('git', ['status', '--porcelain', '--', 'src'], { encoding: 'utf8' }).trim();
    return dirty ? `${head}+dirty` : head;
  } catch {
    return null;
  }
}
