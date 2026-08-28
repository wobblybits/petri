/// <reference path="./worklet-env.d.ts" />
import { WaveguideNet } from '../waveguide.ts';

class NetProcessor extends AudioWorkletProcessor {
  net = new WaveguideNet();
  private vizAcc = 0;
  private errSent = 0;

  constructor() {
    super();
    this.port.onmessage = (ev: MessageEvent) => {
      try {
        this.net.handle(ev.data);
      } catch (err) {
        this.report(err);
      }
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const outL = outputs[0] && outputs[0][0];
    if (!outL) return true;
    const outR = outputs[0][1];
    const n = outL.length;
    // Keep rendering even if the viz snapshot throws: wiping the block after
    // a successful tick is what made the output cut in and out at 60 Hz.
    try {
      for (let i = 0; i < n; i++) {
        this.net.tick();
        outL[i] = this.net.outL;
        if (outR) outR[i] = this.net.outR;
      }
    } catch (err) {
      this.report(err);
      outL.fill(0);
      if (outR) outR.fill(0);
      return true;
    }
    this.vizAcc += n;
    const period = sampleRate / 60;
    if (this.vizAcc >= period) {
      this.vizAcc -= period;
      try {
        const packed = this.net.fillWaveSnapshot();
        // Structured clone copies this; do not slice (alloc on the audio thread)
        // and do not transfer (that would neuter the live buffer).
        this.port.postMessage({ type: 'waves', packed });
      } catch (err) {
        this.report(err);
      }
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
