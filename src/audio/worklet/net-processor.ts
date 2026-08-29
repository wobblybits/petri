/// <reference path="./worklet-env.d.ts" />
import { WaveguideNet } from '../waveguide.ts';

class NetProcessor extends AudioWorkletProcessor {
  net = new WaveguideNet();
  private vizAcc = 0;
  private errSent = 0;
  private prevTime = -1;

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
    const dry = outputs[0];
    const wet = outputs[1];
    const dryL = dry && dry[0];
    if (!dryL) return true;
    const dryR = dry[1];
    const wetL = wet && wet[0];
    const wetR = wet && wet[1];
    const n = dryL.length;

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
