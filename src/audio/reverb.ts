/**
 * Short stereo impulse response for a warm hall tail. Decorrelated per channel,
 * with a brief build-up before the exponential decay so it reads as a room
 * rather than a gated noise burst.
 */
export function makeReverbIR(ctx: BaseAudioContext, seconds = 2.6): AudioBuffer {
  const rate = ctx.sampleRate;
  const len = Math.max(1, Math.floor(rate * seconds));
  const buf = ctx.createBuffer(2, len, rate);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    let lp = 0;
    for (let i = 0; i < len; i++) {
      const t = i / rate;
      const white = Math.random() * 2 - 1;
      // Damp the tail progressively: late reflections should be darker.
      const damp = 0.35 - 0.22 * (t / seconds);
      lp += damp * (white - lp);
      const attack = 1 - Math.exp(-t / 0.012);
      const decay = Math.exp((-6.5 * t) / seconds);
      data[i] = lp * attack * decay;
    }
  }
  return buf;
}
