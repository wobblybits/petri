import { readFileSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { fileURLToPath } from 'node:url';

/*
 * What Vite does for the browser, for `node --experimental-strip-types`.
 *
 * The headless runner (`npm run pond`) is the same simulation the page runs,
 * so it imports the same modules — and two of those modules import files Node
 * has no idea what to do with: `./field.wgsl?raw` and
 * `./worklet/net-processor.ts?url`. Neither is reachable from the CPU step,
 * but both are on the module graph above it, and an unknown extension is a
 * load error whether or not anything calls into it.
 *
 * So: `?raw` becomes the file's text and `?url` becomes its `file:` URL,
 * which is exactly Vite's contract for both. Synchronous `registerHooks`
 * rather than the worker-thread `register`, so this is one file with no
 * bootstrap beside it.
 *
 * The alternative was to bundle the CLI through Vite before running it, which
 * would put a build step between an edit and a pond. This keeps `src/` the
 * only thing that runs.
 */

const QUERY = /\?(raw|url)$/;

registerHooks({
  resolve(spec, ctx, next) {
    const m = QUERY.exec(spec);
    if (!m) return next(spec, ctx);
    const resolved = next(spec.slice(0, m.index), ctx);
    return { ...resolved, url: `${resolved.url}?${m[1]}`, format: 'module', shortCircuit: true };
  },
  load(url, ctx, next) {
    const m = QUERY.exec(url);
    if (!m) return next(url, ctx);
    const file = url.slice(0, m.index);
    const value =
      m[1] === 'raw' ? readFileSync(fileURLToPath(file), 'utf8') : file;
    return { format: 'module', shortCircuit: true, source: `export default ${JSON.stringify(value)};` };
  },
});
