import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { defineConfig, type Plugin } from 'vitest/config';

const root = path.dirname(fileURLToPath(import.meta.url));
const processorPath = path.resolve(root, 'src/audio/worklet/net-processor.ts');
const workerPath = path.resolve(root, 'src/audio/worklet/net-worker.ts');
const waveguidePath = path.resolve(root, 'src/audio/waveguide.ts');
const ringPath = path.resolve(root, 'src/audio/ring.ts');
const fillerPath = path.resolve(root, 'src/audio/filler.ts');
const VIRTUAL_URL = '\0audio-worklet-url';
const VIRTUAL_WORKER_URL = '\0audio-worker-url';
const DEV_PATH = '/__audio_worklet';
const DEV_WORKER_PATH = '/__audio_worker';
const ISO_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
};

function stripModuleChrome(src: string): string {
  return src
    .replace(/^\/\/\/ <reference path="[^"]+" \/>\n/gm, '')
    .replace(/^import type [^;]+;\n/gm, '')
    .replace(/^import \{[^}]+\} from [^;]+;\n/gm, '')
    .replace(/^export /gm, '');
}

function inlineAll(paths: string[]): string {
  return paths.map((p) => stripModuleChrome(fs.readFileSync(p, 'utf8'))).join('\n');
}

function compileWorklet(): string {
  // ring.ts first: the processor constructs an AudioRing when it is handed
  // shared memory, and the concatenation has no module graph to order it.
  return assertInlined(inlineAll([ringPath, waveguidePath, processorPath]), 'net-processor.ts');
}

/**
 * stripModuleChrome is a regex, not a parser. A value import it fails to
 * recognise would be silently dropped, and the bundle would die at runtime
 * with a ReferenceError instead of here, at build time.
 */
function assertInlined(bundled: string, fileName: string): string {
  const stray = bundled.match(/^\s*(import|export)\s.*$/m);
  if (stray) {
    throw new Error(
      `audio-worklet: module syntax survived inlining into ${fileName}: ${stray[0].trim()}\n` +
        'These bundles must be single import-free files — inline the dependency ' +
        'or extend stripModuleChrome.',
    );
  }
  return transpile(bundled, fileName);
}

/**
 * The worker that renders when synthesis runs off the audio thread.
 *
 * Bundled the same import-free way as the worklet, and for a related reason:
 * it is loaded with importScripts after a bootstrap has set `sampleRate`.
 * waveguide.ts reads that global once, at load, and derives its filter
 * coefficients from it — a module import would evaluate before the bootstrap
 * could set anything, and the whole net would be tuned for 48 kHz on hardware
 * running at 44.1.
 */
function compileWorker(): string {
  return assertInlined(
    inlineAll([ringPath, waveguidePath, fillerPath, workerPath]),
    'net-worker.ts',
  );
}

function transpile(bundled: string, fileName: string): string {
  const result = ts.transpileModule(bundled, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
      strict: false,
    },
    fileName,
  });
  return result.outputText.replace(/^export \{\};\s*$/m, '');
}

/** Inline waveguide.ts into the worklet so addModule sees one file, no imports. */
function audioWorklet(): Plugin {
  let isBuild = false;
  let base = '/';
  return {
    name: 'audio-worklet',
    enforce: 'pre',
    configResolved(config) {
      isBuild = config.command === 'build';
      base = config.base;
    },
    resolveId(id, importer) {
      if (!/\?url\b/.test(id)) return null;
      const spec = id.split('?')[0];
      const abs = path.isAbsolute(spec)
        ? spec
        : path.resolve(importer ? path.dirname(importer) : root, spec);
      if (abs === processorPath) return VIRTUAL_URL;
      if (abs === workerPath) return VIRTUAL_WORKER_URL;
      return null;
    },
    load(id) {
      if (id === VIRTUAL_URL) {
        if (!isBuild) return `export default ${JSON.stringify(DEV_PATH)}`;
        return `export default ${JSON.stringify(`${base}assets/net-processor.js`)}`;
      }
      if (id === VIRTUAL_WORKER_URL) {
        if (!isBuild) return `export default ${JSON.stringify(DEV_WORKER_PATH)}`;
        return `export default ${JSON.stringify(`${base}assets/net-worker.js`)}`;
      }
      return null;
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split('?')[0];
        const body =
          url === DEV_PATH ? compileWorklet() : url === DEV_WORKER_PATH ? compileWorker() : null;
        if (body === null) {
          next();
          return;
        }
        res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache');
        res.end(body);
      });
    },
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'assets/net-processor.js',
        source: compileWorklet(),
      });
      this.emitFile({
        type: 'asset',
        fileName: 'assets/net-worker.js',
        source: compileWorker(),
      });
    },
  };
}

const isDemo = process.env.VITE_APP === 'demo';

/** Pages serves index.html; Vite's demo input is demo.html. */
function demoAsIndex(): Plugin {
  return {
    name: 'demo-as-index',
    closeBundle() {
      const from = path.resolve(root, 'dist/demo.html');
      const to = path.resolve(root, 'dist/index.html');
      if (fs.existsSync(from)) fs.renameSync(from, to);
    },
  };
}

export default defineConfig({
  base: process.env.BASE_PATH || '/',
  plugins: isDemo ? [audioWorklet(), demoAsIndex()] : [audioWorklet()],
  // SharedArrayBuffer (and the worker ring) only exist on a cross-origin
  // isolated page. Both headers have to be on the document; CORP on the
  // responses lets the isolated page load its own worklet and workers.
  // `vite` used to set these in middleware, which `vite preview` never ran.
  server: { headers: ISO_HEADERS },
  preview: { headers: ISO_HEADERS },
  build: {
    // Worklet must be a real file URL — never inline raw source as a data: URL.
    assetsInlineLimit: 0,
    rollupOptions: isDemo ? { input: path.resolve(root, 'demo.html') } : undefined,
  },
  test: {
    /*
     * Raised from 60s for the minute-long soak tests in net-health, which
     * simulate 3,600 frames of real soup and now spend most of that in the
     * scent field. Making the field world-fixed at one wire length per cell
     * took it from 14,400 cells to 1,048,576, and its two diffusion passes and
     * a decay from 0.25ms a frame to 13.7ms — 49 seconds of diffusion alone
     * across a minute of simulated pond. That is the designed cost of a grid
     * fine enough to steer by, and the reason the field wants to be on the GPU;
     * it is not a regression to tune away here.
     */
    testTimeout: 150_000,
    // Applied to both projects: seeds Math.random and resets the Sim statics
    // per test, so a file's result does not depend on which file ran before it.
    setupFiles: ['./src/test-setup.ts'],
    // Two projects, because they need opposite things from the runner.
    //
    // `suite` is correctness and runs its files in parallel. `bench` asserts
    // against a frame budget, so it cannot share the machine with three other
    // worker processes — measured 25 ms/frame contended against 13 ms alone,
    // which made the suite fail at random. Random failure is worse than no
    // assertion, so the budget moved to `npm run bench`, on its own.
    projects: [
      {
        extends: true,
        test: {
          name: 'suite',
          include: ['src/**/*.test.ts'],
          exclude: ['**/node_modules/**', '**/dist/**', 'src/**/*.perf.test.ts'],
        },
      },
      {
        extends: true,
        test: {
          name: 'bench',
          include: ['src/**/*.perf.test.ts'],
          fileParallelism: false,
          testTimeout: 300_000,
        },
      },
    ],
  },
});
