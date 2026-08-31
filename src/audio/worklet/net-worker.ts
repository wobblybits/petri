/// <reference path="./worker-env.d.ts" />
import { AudioRing } from '../ring.ts';
import { NetFiller } from '../filler.ts';
import { WaveguideNet } from '../waveguide.ts';

/**
 * Host for the waveguide when synthesis runs off the audio thread.
 *
 * A shell by design: everything with logic in it lives in NetFiller, which is
 * testable without a Worker. All this does is own the net, forward control
 * messages to it, and top the ring up on a timer.
 *
 * The timer is not laziness. A worker that blocks — on Atomics.wait, or in a
 * spin loop — never returns to its event loop, so no control message would
 * ever be delivered: the net would render on forever with the topology it had
 * at startup. Yielding between passes is what keeps messages flowing.
 */

const net = new WaveguideNet();
let filler: NetFiller | null = null;
let rate = 48000;
let running = false;
let primed = false;
let vizAcc = 0;
let errSent = 0;

/** Frames produced per pass. Eight quanta keeps the wakeup rate modest. */
const PASS_FRAMES = 128 * 8;

function report(err: unknown): void {
  if (errSent >= 8) return;
  errSent++;
  const message = err instanceof Error ? err.message : String(err);
  self.postMessage({ type: 'error', message });
}

/**
 * How long to sleep before topping the ring up again.
 *
 * Sleeping half of what is buffered sounds right and is not: setTimeout
 * granularity and scheduler jitter both overshoot, and the ring was observed
 * dipping to a quarter full — which is the shedding threshold, so the net
 * would start dropping air on a machine that was actually keeping up. Aim to
 * wake with the ring still comfortably above that, and cap the sleep so a
 * long timer cannot turn one late wakeup into an underrun.
 */
function nextDelayMs(): number {
  if (!filler) return 4;
  const capacityMs = (filler.ring.capacity / Math.max(1, rate)) * 1000;
  const slack = filler.bufferedMs(rate) - capacityMs * 0.6;
  if (slack <= 1) return 1;
  return Math.min(8, Math.floor(slack));
}

function pump(): void {
  if (!running || !filler) return;
  try {
    const made = filler.fill(PASS_FRAMES);

    // Tell the engine only once there is a buffer to read, so the callback is
    // not handed an empty ring and made to pad. Otherwise every session opens
    // with a couple of dozen underruns, which is harmless in itself but
    // leaves the underrun count useless as a health signal.
    if (!primed && filler.ring.fill() >= 0.5) {
      primed = true;
      self.postMessage({ type: '__ready' });
    }

    // Frames actually produced, not frames asked for. A pass that finds the
    // ring nearly full makes far fewer, and counting the request instead was
    // posting snapshots at twice the intended rate — each one a 17 kB array
    // through structured clone.
    vizAcc += made;
    const period = rate / 30;
    if (vizAcc >= period) {
      vizAcc -= period;
      // Starving means the delay-line walk this needs is the last thing worth
      // spending time on, so post an empty header and let the draw path drop
      // its offsets rather than freeze them.
      const starving = filler.ring.fill() < 0.25;
      const packed = starving ? net.zeroWaveSnapshot() : net.fillWaveSnapshot();
      self.postMessage({ type: 'waves', packed });
    }
  } catch (err) {
    report(err);
  }
  setTimeout(pump, nextDelayMs());
}

self.onmessage = (ev: MessageEvent): void => {
  const msg = ev.data;
  if (!msg) return;
  try {
    if (msg.type === '__ring') {
      rate = Number.isFinite(msg.sampleRate) && msg.sampleRate > 0 ? msg.sampleRate : 48000;
      filler = new NetFiller(net, new AudioRing(msg.sab, msg.capacity));
      if (!running) {
        running = true;
        pump();
      }
      return;
    }
    if (msg.type === '__stop') {
      running = false;
      return;
    }
    net.handle(msg);
  } catch (err) {
    report(err);
  }
};
