/**
 * Lock-free audio ring over a SharedArrayBuffer.
 *
 * The point is to get synthesis off the audio callback. Rendering inside
 * `process()` means one slow quantum is one audible click, with no way to
 * borrow time from a quantum that finished early. A ring lets the producer run
 * ahead: an overrun eats buffer instead of making a hole, and the fill level
 * becomes a load signal that leads rather than lags — unlike the audio clock,
 * which only diverges once a quantum has already been missed.
 *
 * Single producer, single consumer, no locks. Both indices live in [0, 2*cap)
 * rather than counting monotonically, so the arithmetic cannot overflow an
 * Int32 after a few hours of playback; the doubled range is what keeps "full"
 * distinguishable from "empty" when the two indices coincide.
 *
 * No imports: this file is inlined into the worklet alongside waveguide.ts.
 */

/** dryL, dryR, wetL, wetR. */
export const RING_CHANNELS = 4;

const CTRL_READ = 0;
const CTRL_WRITE = 1;
const CTRL_UNDERRUNS = 2;
const CTRL_STARVED = 3;
const CTRL_LEN = 8;

/** Bytes needed to back a ring of `capacity` frames. Capacity must be 2^n. */
export function ringBytes(capacity: number): number {
  return CTRL_LEN * 4 + RING_CHANNELS * capacity * 4;
}

export class AudioRing {
  readonly capacity: number;
  private readonly mask: number;
  private readonly span: number;
  private readonly ctrl: Int32Array;
  private readonly planes: Float32Array[];

  constructor(buffer: ArrayBufferLike, capacity: number) {
    if ((capacity & (capacity - 1)) !== 0 || capacity <= 0) {
      throw new Error(`ring capacity must be a power of two, got ${capacity}`);
    }
    this.capacity = capacity;
    this.mask = capacity - 1;
    this.span = capacity * 2;
    this.ctrl = new Int32Array(buffer, 0, CTRL_LEN);
    this.planes = [];
    for (let c = 0; c < RING_CHANNELS; c++) {
      this.planes.push(new Float32Array(buffer, CTRL_LEN * 4 + c * capacity * 4, capacity));
    }
  }

  /** Frames the consumer has not taken yet. */
  available(): number {
    const w = Atomics.load(this.ctrl, CTRL_WRITE);
    const r = Atomics.load(this.ctrl, CTRL_READ);
    return (w - r + this.span) % this.span;
  }

  /** Frames the producer may still add without overwriting unread audio. */
  writable(): number {
    return this.capacity - this.available();
  }

  /**
   * Producer. Copies `n` frames from four planar sources. Returns how many
   * were taken — short only when the ring is nearly full, which is the normal
   * way a producer running ahead of the clock gets told to stop.
   */
  write(src: Float32Array[], n: number): number {
    const room = this.writable();
    const count = n < room ? n : room;
    if (count <= 0) return 0;
    let w = Atomics.load(this.ctrl, CTRL_WRITE);
    const start = w & this.mask;
    const first = Math.min(count, this.capacity - start);
    for (let c = 0; c < RING_CHANNELS; c++) {
      const plane = this.planes[c];
      const from = src[c];
      plane.set(from.subarray(0, first), start);
      if (first < count) plane.set(from.subarray(first, count), 0);
    }
    w = (w + count) % this.span;
    Atomics.store(this.ctrl, CTRL_WRITE, w);
    return count;
  }

  /**
   * Consumer. Fills `n` frames into four planar destinations, padding with
   * silence if the producer has fallen behind, and counts that as an underrun
   * so the shortfall is visible rather than merely audible.
   */
  read(dst: (Float32Array | undefined)[], n: number): number {
    const have = this.available();
    const count = n < have ? n : have;
    if (count > 0) {
      const r = Atomics.load(this.ctrl, CTRL_READ);
      const start = r & this.mask;
      const first = Math.min(count, this.capacity - start);
      for (let c = 0; c < RING_CHANNELS; c++) {
        const out = dst[c];
        if (!out) continue;
        const plane = this.planes[c];
        out.set(plane.subarray(start, start + first), 0);
        if (first < count) out.set(plane.subarray(0, count - first), first);
      }
      Atomics.store(this.ctrl, CTRL_READ, (r + count) % this.span);
    }
    if (count < n) {
      for (let c = 0; c < RING_CHANNELS; c++) dst[c]?.fill(0, count, n);
      Atomics.add(this.ctrl, CTRL_UNDERRUNS, 1);
      Atomics.add(this.ctrl, CTRL_STARVED, n - count);
    }
    return count;
  }

  /** Quanta the consumer has had to pad. Non-zero means the producer is losing. */
  underruns(): number {
    return Atomics.load(this.ctrl, CTRL_UNDERRUNS);
  }

  /** Frames of silence padded in total. */
  starvedFrames(): number {
    return Atomics.load(this.ctrl, CTRL_STARVED);
  }

  /**
   * How full the ring is, 0..1. This is the leading load signal: it sags
   * before anything is dropped, where an underrun count only rises after.
   */
  fill(): number {
    return this.available() / this.capacity;
  }

  reset(): void {
    Atomics.store(this.ctrl, CTRL_READ, 0);
    Atomics.store(this.ctrl, CTRL_WRITE, 0);
    Atomics.store(this.ctrl, CTRL_UNDERRUNS, 0);
    Atomics.store(this.ctrl, CTRL_STARVED, 0);
    for (const p of this.planes) p.fill(0);
  }
}
