/// <reference path="./worklet-env.d.ts" />
import { WaveguideNet } from '../waveguide.ts';

class NetProcessor extends AudioWorkletProcessor {
  net = new WaveguideNet();
  private vizAcc = 0;

  constructor() {
    super();
    this.port.onmessage = (ev: MessageEvent) => this.net.handle(ev.data);
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const outL = outputs[0] && outputs[0][0];
    if (!outL) return true;
    const outR = outputs[0][1];
    const n = outL.length;
    try {
      for (let i = 0; i < n; i++) {
        this.net.tick();
        outL[i] = this.net.outL;
        if (outR) outR[i] = this.net.outR;
      }
      this.vizAcc += n;
      const period = sampleRate / 60;
      if (this.vizAcc >= period) {
        this.vizAcc -= period;
        const packed = this.net.fillWaveSnapshot();
        const count = packed[0] | 0;
        const end = Math.min(packed.length, 2 + count * (2 + (packed[1] | 0) * 2));
        this.port.postMessage({ type: 'waves', packed: packed.slice(0, end) });
      }
    } catch {
      outL.fill(0);
      if (outR) outR.fill(0);
    }
    return true;
  }
}

registerProcessor('net-processor', NetProcessor);
