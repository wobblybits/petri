import { describe, expect, it } from 'vitest';
import { AudioRing, RING_CHANNELS, ringBytes } from './ring.ts';

function makeRing(capacity: number): AudioRing {
  return new AudioRing(new SharedArrayBuffer(ringBytes(capacity)), capacity);
}

function planes(n: number, fill = (c: number, i: number) => c * 1000 + i): Float32Array[] {
  const out: Float32Array[] = [];
  for (let c = 0; c < RING_CHANNELS; c++) {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = fill(c, i);
    out.push(a);
  }
  return out;
}

function empty(n: number): Float32Array[] {
  const out: Float32Array[] = [];
  for (let c = 0; c < RING_CHANNELS; c++) out.push(new Float32Array(n));
  return out;
}

describe('audio ring', () => {
  it('rejects a capacity that is not a power of two', () => {
    expect(() => makeRing(100)).toThrow(/power of two/);
  });

  it('starts empty and reports its room', () => {
    const ring = makeRing(64);
    expect(ring.available()).toBe(0);
    expect(ring.writable()).toBe(64);
    expect(ring.fill()).toBe(0);
  });

  it('round-trips samples on every channel', () => {
    const ring = makeRing(64);
    expect(ring.write(planes(16), 16)).toBe(16);
    const out = empty(16);
    expect(ring.read(out, 16)).toBe(16);
    for (let c = 0; c < RING_CHANNELS; c++) {
      for (let i = 0; i < 16; i++) expect(out[c][i]).toBe(c * 1000 + i);
    }
  });

  it('wraps without losing or reordering a sample', () => {
    // Capacity 8, pushed 40 frames through in odd-sized chunks, so the
    // split-at-the-end path runs many times over.
    const ring = makeRing(8);
    let written = 0;
    let read = 0;
    const seen: number[] = [];
    while (read < 40) {
      const chunk = Math.min(3, 40 - written);
      if (chunk > 0) {
        const src = planes(chunk, (c, i) => (c === 0 ? written + i : 0));
        written += ring.write(src, chunk);
      }
      const out = empty(2);
      const got = ring.read(out, 2);
      for (let i = 0; i < got; i++) seen.push(out[0][i]);
      read += got;
      if (got === 0 && written >= 40) break;
    }
    expect(seen.length).toBeGreaterThanOrEqual(40);
    for (let i = 0; i < 40; i++) expect(seen[i]).toBe(i);
  });

  it('never overwrites audio the consumer has not taken', () => {
    const ring = makeRing(8);
    expect(ring.write(planes(8), 8)).toBe(8);
    expect(ring.writable()).toBe(0);
    // A full ring accepts nothing, rather than clobbering the oldest frame.
    expect(ring.write(planes(4), 4)).toBe(0);
    const out = empty(8);
    ring.read(out, 8);
    for (let i = 0; i < 8; i++) expect(out[0][i]).toBe(i);
  });

  it('distinguishes full from empty', () => {
    const ring = makeRing(8);
    ring.write(planes(8), 8);
    expect(ring.available()).toBe(8);
    expect(ring.fill()).toBe(1);
    ring.read(empty(8), 8);
    expect(ring.available()).toBe(0);
    expect(ring.fill()).toBe(0);
  });

  it('pads with silence and counts the shortfall rather than repeating audio', () => {
    const ring = makeRing(64);
    ring.write(planes(4, () => 0.5), 4);
    const out = empty(16);
    expect(ring.read(out, 16)).toBe(4);
    for (let i = 0; i < 4; i++) expect(out[0][i]).toBe(0.5);
    // The rest is silence, not the last frame held or the buffer repeated.
    for (let i = 4; i < 16; i++) expect(out[0][i]).toBe(0);
    expect(ring.underruns()).toBe(1);
    expect(ring.starvedFrames()).toBe(12);
  });

  it('tolerates a missing output channel', () => {
    const ring = makeRing(64);
    ring.write(planes(8), 8);
    const out: (Float32Array | undefined)[] = [new Float32Array(8), undefined, undefined, undefined];
    expect(ring.read(out, 8)).toBe(8);
    expect(out[0]![7]).toBe(7);
  });

  it('survives far more frames than the index range, without drift', () => {
    // Indices live in [0, 2*cap); this is the wrap that used to be an Int32
    // overflow waiting to happen.
    const ring = makeRing(16);
    let next = 0;
    let expected = 0;
    for (let round = 0; round < 5000; round++) {
      const src = planes(8, (c, i) => (c === 0 ? (next + i) % 997 : 0));
      const wrote = ring.write(src, 8);
      next += wrote;
      const out = empty(8);
      const got = ring.read(out, 8);
      for (let i = 0; i < got; i++) {
        expect(out[0][i]).toBe(expected % 997);
        expected++;
      }
    }
    expect(expected).toBeGreaterThan(30000);
  });

  it('sees writes made through a second view of the same memory', () => {
    // This is the whole point: the worker and the worklet hold different
    // AudioRing objects over one SharedArrayBuffer.
    const sab = new SharedArrayBuffer(ringBytes(32));
    const producer = new AudioRing(sab, 32);
    const consumer = new AudioRing(sab, 32);
    producer.write(planes(8, (_c, i) => i + 1), 8);
    expect(consumer.available()).toBe(8);
    const out = empty(8);
    consumer.read(out, 8);
    for (let i = 0; i < 8; i++) expect(out[0][i]).toBe(i + 1);
    expect(producer.available()).toBe(0);
  });

  it('resets to empty', () => {
    const ring = makeRing(16);
    ring.write(planes(8), 8);
    ring.read(empty(16), 16);
    expect(ring.underruns()).toBe(1);
    ring.reset();
    expect(ring.available()).toBe(0);
    expect(ring.underruns()).toBe(0);
    expect(ring.starvedFrames()).toBe(0);
  });
});
