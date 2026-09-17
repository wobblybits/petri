import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import farShader from '../gpu/far.wgsl?raw';
import { ERA_RADIUS, TRI_DISC_RATIO } from '../agents.ts';
import { SKIN, SLOP } from '../collide.ts';
import {
  FAR,
  FAR_CONTACT_COMP,
  FAR_SKIN,
  FAR_SLOP,
  FAR_SPAN_COMP,
  FAR_STRIDE,
  FAR_WIRE_STRIDE,
} from '../gpu/far-kernel.ts';

/*
 * The contact, in four languages.
 *
 * `collide.ts` says of `SKIN` that it is "exported because three other copies
 * of the contact — the WASM solver, the GPU kernel and its twin — have to
 * carry the same number, and a mirror that drops it moves the pond by 1.7 px
 * the moment the camera changes which tier a body is in." That was the intent
 * and the export was never taken up: `far-kernel.ts` declares its own 0.85,
 * `far.wgsl` declares a third, and `solver.c` a fourth, and nothing compared
 * any of them.
 *
 * Two of the four cannot import — a shader and a C file — and the third must
 * not: `far-kernel.ts` importing `collide.ts` closes a cycle through
 * `agents.ts` and `native/solver.ts`, and a `const` initialised inside an ES
 * module cycle reads `undefined` rather than failing. So the binding is here,
 * where text is text.
 *
 * The body layout matters more than the skin and is quieter about being wrong.
 * `FAR_X..FAR_PREVHEAD` in `solver.c` and `FAR` here index the same packed
 * floats from two languages; a field inserted in one shifts every read in that
 * one and nothing else, and what comes out is a pond where velocity is read as
 * position.
 */

const solverSrc = readFileSync(new URL('../../native/solver.c', import.meta.url), 'utf8');

/** One `#define NAME <number>` from solver.c, `f` suffix and all. */
function cDefine(name: string): number {
  const m = new RegExp(`^#define\\s+${name}\\s+(-?[\\d.eE+-]+)f?\\s*(?:/\\*|$)`, 'm').exec(solverSrc);
  if (!m) throw new Error(`no #define ${name} in native/solver.c`);
  return Number(m[1]);
}

/** One `const NAME: f32 = <number>;` from far.wgsl. */
function wgslF32(name: string): number {
  const m = new RegExp(`const\\s+${name}\\s*:\\s*f32\\s*=\\s*(-?[\\d.]+)\\s*;`).exec(farShader);
  if (!m) throw new Error(`no f32 const ${name} in far.wgsl`);
  return Number(m[1]);
}

describe('the contact constants agree across the four tiers', () => {
  it('gives every copy of the skin and the slop the same number', () => {
    expect(FAR_SKIN, "far-kernel.ts's skin against collide.ts").toBe(SKIN);
    expect(FAR_SLOP, "far-kernel.ts's slop against collide.ts").toBe(SLOP);
    expect(wgslF32('SKIN'), "far.wgsl's skin").toBe(SKIN);
    expect(cDefine('SKIN'), "solver.c's skin").toBe(SKIN);
    expect(cDefine('SLOP'), "solver.c's slop").toBe(SLOP);
  });

  it('gives both solvers the same compliances', () => {
    // These two have no `collide.ts` copy: the discrete tiers do not solve a
    // constraint softly, so the number exists only where a substep does.
    expect(cDefine('CONTACT_COMP'), "solver.c's contact compliance").toBe(FAR_CONTACT_COMP);
    expect(cDefine('SPAN_COMP'), "solver.c's span compliance").toBe(FAR_SPAN_COMP);
  });

  it('gives both solvers the same body radii', () => {
    expect(cDefine('ERA_R'), "solver.c's Era radius").toBe(ERA_RADIUS);
    /*
     * `solver.c` carries the glyph disc as a radius at s = 16 and `agents.ts`
     * as the ratio it is derived from; the C copy is a rounded literal, so
     * this compares them at the literal's own four decimal places. The comment
     * over `TRI_DISC_R` already claims the two "must agree" — this is the
     * first thing to check it.
     */
    expect(cDefine('TRI_DISC_R'), "solver.c's glyph disc radius at s = 16").toBeCloseTo(
      TRI_DISC_RATIO * 16,
      4,
    );
  });

  it('indexes the packed body the same way in both languages', () => {
    expect(cDefine('STRIDE'), "solver.c's body stride").toBe(FAR_STRIDE);
    expect(cDefine('WIRE_FAR'), "solver.c's far wire stride").toBe(FAR_WIRE_STRIDE);
    const C_NAMES: Record<keyof typeof FAR, string> = {
      x: 'FAR_X',
      y: 'FAR_Y',
      vx: 'FAR_VX',
      vy: 'FAR_VY',
      heading: 'FAR_HEADING',
      omega: 'FAR_OMEGA',
      invMass: 'FAR_INVMASS',
      radius: 'FAR_RADIUS',
      locked: 'FAR_LOCKED',
      prevX: 'FAR_PREVX',
      prevY: 'FAR_PREVY',
      prevHeading: 'FAR_PREVHEAD',
    };
    for (const [field, define] of Object.entries(C_NAMES)) {
      expect(cDefine(define), `solver.c's ${define} against FAR.${field}`).toBe(
        FAR[field as keyof typeof FAR],
      );
    }
    // Every slot accounted for, so a thirteenth field added to one side is a
    // failure rather than a silent extra.
    expect(Object.keys(C_NAMES).length, 'FAR does not tile its own stride').toBe(FAR_STRIDE);
  });
});
