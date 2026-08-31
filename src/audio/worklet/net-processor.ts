/// <reference path="./worklet-env.d.ts" />
import { AudioRing } from '../ring.ts';
import { WaveguideNet } from '../waveguide.ts';

/**
 * Two modes.
 *
 * With a ring, synthesis happens in a Worker and this is a copy: read four
 * planes, hand them to the outputs, wake the producer. A quantum that the
 * producer was late for costs buffered audio rather than a click, and the
 * buffer is what lets a slow pass be paid for out of a fast one.
 *
 * Without a ring — no SharedArrayBuffer, or a page that is not
 * cross-origin-isolated — it runs the net here, exactly as before. That path
 * is the fallback rather than the plan, but it has to keep working, because
 * cross-origin isolation is a deployment property nobody can promise.
 */
class NetProcessor extends AudioWorkletProcessor {
  net = new WaveguideNet();
  private ring: AudioRing | null = null;
  private out: (Float32Array | undefined)[] = [undefined, undefined, undefined, undefined];
  private vizAcc = 0;
  private errSent = 0;
  private prevTime = -1;

  constructor() {
    super();
    this.port.onmessage = (ev: MessageEvent) => {
      try {
        const msg = ev.data;
        if (msg && msg.type === '__ring') {
          this.ring = new AudioRing(msg.sab, msg.capacity);
          return;
        }
        this.net.handle(msg);
      } catch (err) {
        this.report(err);
      }
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const dry = outputs[0];
    const wet = outputs[1];
    const dryL = dry && dry[0];
    if (!dryL) return true;
    const dryR = dry[1];
    const wetL = wet && wet[0];
    const wetR = wet && wet[1];
    const n = dryL.length;

    if (this.ring) {
      this.out[0] = dryL;
      this.out[1] = dryR;
      this.out[2] = wetL;
      this.out[3] = wetR;
      try {
        this.ring.read(this.out, n);
      } catch (err) {
        this.report(err);
      }
      return true;
    }

    try {
      // `currentTime` is a global in AudioWorkletGlobalScope, not a property
      // of the processor. Reading it off `this` gave undefined, so `late` was
      // permanently false and the overload shedding never once fired.
      const now = currentTime;
      const budget = n / sampleRate;
      const late = this.prevTime >= 0 && now - this.prevTime > budget * 1.35;
      this.prevTime = now;

      for (let i = 0; i < n; i++) {
        this.net.tick(late);
        dryL[i] = this.net.outDryL;
        if (dryR) dryR[i] = this.net.outDryR;
        if (wetL) wetL[i] = this.net.outWetL;
        if (wetR) wetR[i] = this.net.outWetR;
      }

      this.vizAcc += n;
      const period = sampleRate / 30;
      if (this.vizAcc >= period) {
        this.vizAcc -= period;
        // Always post, including n=0: skipping left the last wiggle on screen
        // after the net went quiet. A late quantum skips the delay-line walk
        // (that is the overrun) and posts an empty header so the draw path
        // drops offsets instead of freezing them.
        const packed = late ? this.net.zeroWaveSnapshot() : this.net.fillWaveSnapshot();
        this.port.postMessage({ type: 'waves', packed });
      }
    } catch (err) {
      this.report(err);
    }
    return true;
  }

  private report(err: unknown): void {
    if (this.errSent >= 8) return;
    this.errSent++;
    const message = err instanceof Error ? err.message : String(err);
    this.port.postMessage({ type: 'error', message });
  }
}

registerProcessor('net-processor', NetProcessor);
