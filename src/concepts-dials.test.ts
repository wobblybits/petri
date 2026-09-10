import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { defaultParams, SLIDERS } from './params.ts';

/*
 * `docs/concepts.md` names, per heading, the dials that serve it. A dial that
 * is deleted or renamed has to be deleted or renamed there too, or the
 * document quietly describes a mechanism that no longer exists.
 */

function dialsNamedInConcepts(): Set<string> {
  const doc = readFileSync(new URL('../docs/concepts.md', import.meta.url), 'utf8');
  const named = new Set<string>();
  for (const line of doc.split('\n')) {
    if (!line.startsWith('|')) continue;
    const cells = line.split('|').map((c) => c.trim());
    // heading | mechanic | dials | default | gauge | not yet true
    if (cells.length < 4) continue;
    for (const m of cells[3].matchAll(/`([A-Za-z_][A-Za-z0-9_]*)`/g)) named.add(m[1]);
  }
  return named;
}

describe('concepts.md', () => {
  it('names only dials that exist', () => {
    const keys = new Set(Object.keys(defaultParams()));
    const missing = [...dialsNamedInConcepts()].filter((d) => !keys.has(d));
    expect(missing, 'named in concepts.md but not in Params').toEqual([]);
  });

  it('names every slider somewhere', () => {
    const named = dialsNamedInConcepts();
    const unnamed = SLIDERS.map((s) => s.key).filter((k) => !named.has(k));
    expect(unnamed, 'a slider no heading claims').toEqual([]);
  });
});
