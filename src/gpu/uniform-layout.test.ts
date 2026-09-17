import { describe, expect, it } from 'vitest';
import fieldShader from './field.wgsl?raw';
import farShader from './far.wgsl?raw';
import genomeShader from './genome.wgsl?raw';
import { FIELD_U } from './field-gpu.ts';
import { FAR_U } from './far-gpu.ts';
import { GENOME_U } from './genome-gpu.ts';

/*
 * Every uniform this program uploads is declared by name in WGSL and written
 * by slot in TypeScript, and until these maps existed the only thing holding
 * the two orders together was counting.
 *
 * That is the defect this project keeps paying for — one number living in two
 * languages with nothing binding them — and it has already been paid for once
 * in exactly this shape: a stored net's scalars were declared by name and
 * decoded by index, every net came back shifted, and it read as the gait being
 * broken. `HEAD_RANGE` is the same fix for the head clamps. This is it for the
 * thing that carries them.
 *
 * Counting was least safe in `FieldParams`, and not because it is the longest.
 * Three `vec4f` sit in its middle, and a `vec4f` in a uniform aligns to sixteen
 * bytes, so a scalar inserted anywhere above `mix` moves twelve slots by four
 * rather than by one — silently, and only on the device. So this walks the
 * struct under WGSL's own alignment rules rather than trusting either side's
 * arithmetic.
 *
 * It needs no device. A struct declaration and an object literal are both text.
 */

/** Size and alignment in bytes, for the types these three structs use. */
const TYPES: Record<string, { size: number; align: number }> = {
  f32: { size: 4, align: 4 },
  u32: { size: 4, align: 4 },
  i32: { size: 4, align: 4 },
  vec2f: { size: 8, align: 8 },
  // vec3 is the trap in every std140-shaped layout: twelve bytes of size and
  // sixteen of alignment, so it is not three scalars.
  vec3f: { size: 12, align: 16 },
  vec4f: { size: 16, align: 16 },
};

interface Field {
  name: string;
  type: string;
  /** Byte offset divided by four: the index the host writes at. */
  slot: number;
}

/**
 * The fields of one struct, in declaration order, each with the slot WGSL puts
 * it at.
 *
 * Comments come out first, because both `//` and `/* *\/` forms appear inside
 * these declarations and a comment mentioning a type would otherwise parse as
 * a field.
 */
function parseStruct(src: string, name: string): { fields: Field[]; bytes: number } {
  const open = new RegExp(`struct\\s+${name}\\s*\\{`).exec(src);
  if (!open) throw new Error(`no struct ${name}`);
  const from = open.index + open[0].length;
  const close = src.indexOf('\n}', from);
  if (close < 0) throw new Error(`struct ${name} is not closed`);
  const body = src
    .slice(from, close)
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n]*/g, '');

  const fields: Field[] = [];
  let offset = 0;
  let maxAlign = 1;
  for (const m of body.matchAll(/([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z0-9_<>]+)\s*,/g)) {
    const type = m[2];
    const t = TYPES[type];
    if (!t) throw new Error(`${name}.${m[1]}: unhandled type ${type}`);
    offset = Math.ceil(offset / t.align) * t.align;
    if (offset % 4 !== 0) throw new Error(`${name}.${m[1]} is not slot-aligned`);
    fields.push({ name: m[1], type, slot: offset / 4 });
    offset += t.size;
    maxAlign = Math.max(maxAlign, t.align);
  }
  // A uniform struct is rounded up to sixteen, which is where the trailing
  // `padN` fields in all three of these come from.
  const bytes = Math.ceil(offset / Math.max(16, maxAlign)) * Math.max(16, maxAlign);
  return { fields, bytes };
}

/** Each struct, the map the host writes it with, and that host's buffer size. */
const CASES = [
  { name: 'GenomeParams', src: genomeShader, map: GENOME_U, host: 'genome-gpu.ts', bytes: 96 },
  { name: 'SimParams', src: farShader, map: FAR_U, host: 'far-gpu.ts', bytes: 256 },
  { name: 'FieldParams', src: fieldShader, map: FIELD_U, host: 'field-gpu.ts', bytes: 176 },
] as const;

describe('the uniform maps match the structs they are written into', () => {
  for (const c of CASES) {
    it(`${c.host} agrees with ${c.name}`, () => {
      const { fields, bytes } = parseStruct(c.src, c.name);
      const map: Record<string, number> = c.map;

      // Order as well as content: a map that had the right names at the right
      // slots but listed them in another order would still read wrongly to
      // anyone maintaining it beside the struct.
      expect(fields.map((f) => f.name), `${c.name} field order`).toEqual(Object.keys(map));

      for (const f of fields) {
        expect(map[f.name], `${c.host}'s slot for ${c.name}.${f.name} (${f.type})`).toBe(f.slot);
      }

      // The buffer has to hold the struct. `far-gpu.ts` over-allocates on
      // purpose and that is fine; too small is not.
      expect(c.bytes, `${c.host}'s UNIFORM_BYTES against ${c.name}`).toBeGreaterThanOrEqual(bytes);
      expect(c.bytes % 16, `${c.host}'s UNIFORM_BYTES is not a whole number of blocks`).toBe(0);
    });
  }

  it('walks vec4 alignment rather than assuming four scalars', () => {
    /*
     * The parser is the thing being trusted above, so it is checked on the one
     * case that matters. `FieldParams` has eight scalars, then four u32, then
     * `mix` — so `mix` lands at 48 bytes with no padding — and the assertion
     * that earns its keep is the next one: a scalar inserted before a `vec4f`
     * pads to the next sixteen, which is how a one-field edit moves twelve
     * slots.
     */
    const flat = parseStruct('struct S {\n  a: f32,\n  v: vec4f,\n}\n', 'S');
    expect(flat.fields.map((f) => f.slot)).toEqual([0, 4]);
    expect(flat.bytes).toBe(32);

    const vec3 = parseStruct('struct S {\n  a: f32,\n  v: vec3f,\n  b: f32,\n}\n', 'S');
    expect(vec3.fields.map((f) => f.slot), 'vec3f aligns to 16 and sizes to 12').toEqual([0, 4, 7]);
  });
});
