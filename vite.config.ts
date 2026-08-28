import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { defineConfig, type Plugin } from 'vitest/config';

const root = path.dirname(fileURLToPath(import.meta.url));
const processorPath = path.resolve(root, 'src/audio/worklet/net-processor.ts');
const waveguidePath = path.resolve(root, 'src/audio/waveguide.ts');
const VIRTUAL_URL = '\0audio-worklet-url';
const DEV_PATH = '/__audio_worklet';

function stripModuleChrome(src: string): string {
  return src
    .replace(/^\/\/\/ <reference path="[^"]+" \/>\n/gm, '')
    .replace(/^import type [^;]+;\n/gm, '')
    .replace(/^import \{[^}]+\} from [^;]+;\n/gm, '')
    .replace(/^export /gm, '');
}

function compileWorklet(): string {
  const bundled =
    stripModuleChrome(fs.readFileSync(waveguidePath, 'utf8')) +
    '\n' +
    stripModuleChrome(fs.readFileSync(processorPath, 'utf8'));
  // stripModuleChrome is a regex, not a parser. A value import it fails to
  // recognise would be silently dropped and the worklet would die at runtime
  // with a ReferenceError instead of here, at build time.
  const stray = bundled.match(/^\s*(import|export)\s.*$/m);
  if (stray) {
    throw new Error(
      `audio-worklet: module syntax survived inlining: ${stray[0].trim()}\n` +
        'The worklet must be a single import-free file — inline the dependency ' +
        'or extend stripModuleChrome.',
    );
  }
  const result = ts.transpileModule(bundled, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
      strict: false,
    },
    fileName: 'net-processor.ts',
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
      if (abs !== processorPath) return null;
      return VIRTUAL_URL;
    },
    load(id) {
      if (id !== VIRTUAL_URL) return null;
      if (!isBuild) return `export default ${JSON.stringify(DEV_PATH)}`;
      return `export default ${JSON.stringify(`${base}assets/net-processor.js`)}`;
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url?.split('?')[0] !== DEV_PATH) {
          next();
          return;
        }
        res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache');
        res.end(compileWorklet());
      });
    },
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'assets/net-processor.js',
        source: compileWorklet(),
      });
    },
  };
}

export default defineConfig({
  plugins: [audioWorklet()],
  build: {
    // Worklet must be a real file URL — never inline raw source as a data: URL.
    assetsInlineLimit: 0,
  },
  test: {
    include: ['src/**/*.test.ts'],
    testTimeout: 60_000,
  },
});
