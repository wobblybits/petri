import { execFileSync } from 'node:child_process';

/**
 * The working tree a run came from: `9a3f21c`, or `9a3f21c+dirty` when
 * `src/` has uncommitted edits, which would make the hash a lie about what
 * ran. Only `src/` counts: a note edited mid-run changes no arithmetic.
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
