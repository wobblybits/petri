import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import farShader from '../gpu/far.wgsl?raw';
import { ERA_RADIUS, TRI_DISC_RATIO } from '../agents.ts';
import { SKIN, SLOP } from '../collide.ts';
import {
  HIT_STRIDE,
  ND,
  NODE_STRIDE,
  STEER_FLAG,
  STEER_PARAM,
  WF_FULL,
  WF_HOLD,
  WF_SHAPE,
  WF_SKIP,
  WIRE_NEAR_STRIDE,
  WN,
} from './solver.ts';
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

describe('the wasm wall agrees about every index that crosses it', () => {
  /*
   * `solver.c` says twice what this is for, and both notes are the reason it
   * exists rather than a renumbering.
   *
   * Over `SP_UNUSED_12`: two slots that carry nothing are still reserved,
   * "because every index above them is a hardcoded number on both sides of
   * this wall and renumbering to recover two floats is how that kind of thing
   * goes wrong."
   *
   * Over `SP_SENSE_SPAN`: "a constant that has to agree across the wasm wall
   * and is written down twice eventually disagrees, which is how the deposit
   * normalisation came to be 20 on one side and 10 on the other."
   *
   * Both are true and neither was checkable. Every map below already had names
   * on the TypeScript side; what was missing was anything comparing them to
   * the `#define`s they were transcribed from. A field inserted into one of
   * these blocks shifts every read above it in that language only, and what
   * comes out is a pond that steers on its own sensor distance or reads a
   * wire's rest length as its rope length.
   */
  it('packs the steer parameters where the solver reads them', () => {
    const C: Record<keyof typeof STEER_PARAM, string> = {
      faceRadius: 'SP_FACE_RADIUS',
      snapRadius: 'SP_SNAP_RADIUS',
      snapArc: 'SP_SNAP_ARC',
      faceAttract: 'SP_FACE_ATTRACT',
      snapWell: 'SP_SNAP_WELL',
      sensorAngle: 'SP_SENSOR_ANGLE',
      sensorDist: 'SP_SENSOR_DIST',
      sense: 'SP_SENSE',
      turnRate: 'SP_TURN_RATE',
      stepSpeed: 'SP_STEP_SPEED',
      swimTau: 'SP_SWIM_TAU',
      swimNoise: 'SP_SWIM_NOISE',
      unused12: 'SP_UNUSED_12',
      unused13: 'SP_UNUSED_13',
      senseSpan: 'SP_SENSE_SPAN',
      portLeak: 'SP_PORT_LEAK',
      auxLeak: 'SP_AUX_LEAK',
    };
    for (const [field, define] of Object.entries(C)) {
      expect(cDefine(define), `solver.c's ${define} against STEER_PARAM.${field}`).toBe(
        STEER_PARAM[field as keyof typeof STEER_PARAM],
      );
    }
  });

  it('agrees about the steer flags', () => {
    expect(cDefine('SF_P_FREE')).toBe(STEER_FLAG.pFree);
    expect(cDefine('SF_STARVING')).toBe(STEER_FLAG.starving);
    expect(cDefine('SF_STUNNED')).toBe(STEER_FLAG.stunned);
    // A bitfield, so they must also be distinct powers of two on both sides.
    const bits = Object.values(STEER_FLAG);
    expect(new Set(bits).size, 'two flags share a bit').toBe(bits.length);
    for (const b of bits) expect(b & (b - 1), `${b} is not a single bit`).toBe(0);
  });

  it('indexes a near wire and its rope nodes the same way', () => {
    expect(cDefine('WIRE_NEAR'), "solver.c's near wire stride").toBe(WIRE_NEAR_STRIDE);
    expect(cDefine('NODE_STRIDE'), "solver.c's rope node stride").toBe(NODE_STRIDE);
    expect(cDefine('HIT_STRIDE'), "solver.c's hit stride").toBe(HIT_STRIDE);

    const WIRE: Record<keyof typeof WN, string> = {
      a: 'WN_A',
      b: 'WN_B',
      rest: 'WN_REST',
      rope: 'WN_ROPE',
      scale: 'WN_SCALE',
      slack: 'WN_SLACK',
      aSlot: 'WN_ASLOT',
      bSlot: 'WN_BSLOT',
      node0: 'WN_NODE0',
      nNodes: 'WN_NNODES',
      flags: 'WN_FLAGS',
    };
    for (const [field, define] of Object.entries(WIRE)) {
      expect(cDefine(define), `solver.c's ${define} against WN.${field}`).toBe(
        WN[field as keyof typeof WN],
      );
    }

    const NODE: Record<keyof typeof ND, string> = {
      x: 'ND_X',
      y: 'ND_Y',
      vx: 'ND_VX',
      vy: 'ND_VY',
      prevX: 'ND_PREVX',
      prevY: 'ND_PREVY',
      shapeX: 'ND_SX',
      shapeY: 'ND_SY',
    };
    for (const [field, define] of Object.entries(NODE)) {
      expect(cDefine(define), `solver.c's ${define} against ND.${field}`).toBe(
        ND[field as keyof typeof ND],
      );
    }
  });

  it('agrees about the wire flags', () => {
    expect(cDefine('WF_FULL')).toBe(WF_FULL);
    expect(cDefine('WF_SKIP')).toBe(WF_SKIP);
    expect(cDefine('WF_SHAPE')).toBe(WF_SHAPE);
    expect(cDefine('WF_HOLD')).toBe(WF_HOLD);
  });
});
