/// <reference path="./worklet-env.d.ts" />
import { AudioRing } from '../ring.ts';
import { WaveguideNet } from '../waveguide.ts';

function mixClip(x: number): number {
  if (x > -1 && x < 1) return x;
  return Math.tanh(x);
}

/**
 * Two modes.
 *
 * With rings, synthesis happens in one or more Workers and this is a mixer:
 * read four planes from each ring, sum them, hand them to the outputs. A
 * quantum that one producer was late for costs that ring's buffer rather than
 * a click, and the other nets still speak — which is the point of sharding.
 *
 * Without a ring — no SharedArrayBuffer, or a page that is not
 * cross-origin-isolated — it runs the net here, exactly as before. That path
 * is the fallback rather than the plan, but it has to keep working, because
 * cross-origin isolation is a deployment property nobody can promise.
 */
class NetProcessor extends AudioWorkletProcessor {
  net = new WaveguideNet();
  private rings: (AudioRing | null)[] = [];
  private mix: Float32Array[] = [];
  private vizAcc = 0;
  private errSent = 0;
  private prevTime = -1;

  constructor() {
    super();
    this.port.onmessage = (ev: MessageEvent) => {
      try {
        const msg = ev.data;
        if (msg && msg.type === '__ring') {
          const slot = Number.isFinite(msg.slot) ? msg.slot | 0 : 0;
          while (this.rings.length <= slot) this.rings.push(null);
          this.rings[slot] = new AudioRing(msg.sab, msg.capacity);
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

    if (this.hasRing()) {
      this.mixRings(n, dryL, dryR, wetL, wetR);
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

  private hasRing(): boolean {
    for (let i = 0; i < this.rings.length; i++) if (this.rings[i]) return true;
    return false;
  }

  private ensureMix(n: number): void {
    if (this.mix.length === 4 && this.mix[0].length >= n) return;
    this.mix = [
      new Float32Array(n),
      new Float32Array(n),
      new Float32Array(n),
      new Float32Array(n),
    ];
  }

  /**
   * Sum every live ring into the worklet outputs.
   *
   * Each net already limits itself. The sum of independent instruments can
   * still exceed 1, so the last stage is the same soft clip the waveguide
   * uses — linear below unity, then it rounds off rather than folds.
   */
  private mixRings(
    n: number,
    dryL: Float32Array,
    dryR: Float32Array | undefined,
    wetL: Float32Array | undefined,
    wetR: Float32Array | undefined,
  ): void {
    dryL.fill(0);
    dryR?.fill(0);
    wetL?.fill(0);
    wetR?.fill(0);
    this.ensureMix(n);
    try {
      for (let s = 0; s < this.rings.length; s++) {
        const ring = this.rings[s];
        if (!ring) continue;
        ring.read(this.mix, n);
        for (let i = 0; i < n; i++) {
          dryL[i] += this.mix[0][i];
          if (dryR) dryR[i] += this.mix[1][i];
          if (wetL) wetL[i] += this.mix[2][i];
          if (wetR) wetR[i] += this.mix[3][i];
        }
      }
      for (let i = 0; i < n; i++) {
        dryL[i] = mixClip(dryL[i]);
        if (dryR) dryR[i] = mixClip(dryR[i]);
        if (wetL) wetL[i] = mixClip(wetL[i]);
        if (wetR) wetR[i] = mixClip(wetR[i]);
      }
    } catch (err) {
      this.report(err);
    }
  }

  private report(err: unknown): void {
    if (this.errSent >= 8) return;
    this.errSent++;
    const message = err instanceof Error ? err.message : String(err);
    this.port.postMessage({ type: 'error', message });
  }
}

registerProcessor('net-processor', NetProcessor);
