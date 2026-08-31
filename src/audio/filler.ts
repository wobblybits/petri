import { AudioRing, RING_CHANNELS } from './ring.ts';
import type { WaveguideNet } from './waveguide.ts';

/** Frames rendered per inner pass. Matches a render quantum. */
const BLOCK = 128;

/**
 * Drives a WaveguideNet into an audio ring.
 *
 * Kept apart from the Worker that hosts it so the part with the logic in it
 * can be tested without a Worker, a SharedArrayBuffer message, or an
 * AudioContext. The Worker is a shell: messages in, `fill` on a timer.
 */
export class NetFiller {
  private readonly scratch: Float32Array[] = [];
  readonly net: WaveguideNet;
  readonly ring: AudioRing;
  /** Frames rendered since construction. Diagnostics only. */
  frames = 0;

  constructor(net: WaveguideNet, ring: AudioRing) {
    this.net = net;
    this.ring = ring;
    for (let c = 0; c < RING_CHANNELS; c++) this.scratch.push(new Float32Array(BLOCK));
  }

  /**
   * Render until the ring is full, or `maxFrames` have been produced.
   *
   * The cap matters: without it a producer that has fallen a long way behind
   * would try to make the whole shortfall back in one go, which on a machine
   * already too slow is how a small stall becomes a long one.
   */
  fill(maxFrames = this.ring.capacity): number {
    let made = 0;
    while (made < maxFrames) {
      const room = this.ring.writable();
      if (room <= 0) break;
      const want = Math.min(BLOCK, room, maxFrames - made);
      const dryL = this.scratch[0];
      const dryR = this.scratch[1];
      const wetL = this.scratch[2];
      const wetR = this.scratch[3];
      const net = this.net;
      // `shed` is the ring's own verdict on whether it is keeping up, which
      // is a leading signal — the audio clock could only report a quantum
      // already missed.
      const shed = this.ring.fill() < 0.25;
      for (let i = 0; i < want; i++) {
        net.tick(shed);
        dryL[i] = net.outDryL;
        dryR[i] = net.outDryR;
        wetL[i] = net.outWetL;
        wetR[i] = net.outWetR;
      }
      const wrote = this.ring.write(this.scratch, want);
      made += wrote;
      this.frames += wrote;
      // `want` never exceeds the room measured a moment ago, and the only
      // other party can free space but not take it, so a short write is not
      // reachable. If it ever became reachable it would mean rendered samples
      // were dropped on the floor, so stop rather than carry on unaware.
      if (wrote < want) break;
    }
    return made;
  }

  /** Milliseconds of audio buffered. What the latency actually is. */
  bufferedMs(sampleRate: number): number {
    return (this.ring.available() / Math.max(1, sampleRate)) * 1000;
  }
}
