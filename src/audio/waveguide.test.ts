import { describe, expect, it } from 'vitest';
import { WaveguideNet, junctionLoadY } from './waveguide.ts';
import { bowSpeed } from './presets.ts';
import type { NetTopology } from './types.ts';

function sampleTopo(wireId: number, agentA: number, agentB: number, length = 120): NetTopology {
  return {
    wires: [
      {
        id: wireId,
        length,
        loss: 0.999,
        bend: 0.05,
        agentA,
        agentB,
        zA: 1,
        zB: 1,
      },
    ],
    agents: [
      { id: agentA, kind: 0, openPorts: 0, impedance: 1 },
      { id: agentB, kind: 2, openPorts: 0, impedance: 1.2 },
    ],
  };
}

function collect(net: WaveguideNet, n: number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = net.tick();
  return out;
}

function peak(samples: number[]): number {
  let p = 0;
  for (const s of samples) p = Math.max(p, Math.abs(s));
  return p;
}

function autocorrLag(samples: number[], minLag: number, maxLag: number): number {
  let bestLag = minLag;
  let best = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0;
    for (let i = 0; i + lag < samples.length; i++) s += samples[i] * samples[i + lag];
    if (s > best) {
      best = s;
      bestLag = lag;
    }
  }
  return bestLag;
}

/** Magnitude at one frequency, Hann-windowed. */
function binMag(x: number[], hz: number, sr = 48000): number {
  let re = 0;
  let im = 0;
  for (let n = 0; n < x.length; n++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (x.length - 1));
    re += x[n] * w * Math.cos((2 * Math.PI * hz * n) / sr);
    im -= x[n] * w * Math.sin((2 * Math.PI * hz * n) / sr);
  }
  return Math.hypot(re, im) / x.length;
}

describe('WaveguideNet', () => {
  it('a closed junction still leaks, but less than an open port', () => {
    expect(junctionLoadY(0)).toBeGreaterThan(0.004);
    expect(junctionLoadY(0)).toBeLessThan(junctionLoadY(1) * 0.5);
    expect(junctionLoadY(2)).toBeGreaterThan(junctionLoadY(1));
  });

  it('a NaN delay does not poison the output', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 10, 20, Number.NaN);
    net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
    const x = collect(net, 512);
    expect(x.every(Number.isFinite)).toBe(true);
    expect(peak(x)).toBeGreaterThan(0);
  });

  it('sounds its fundamental, not the octave above it', () => {
    // Regression: mixing both terminations into the output cancelled every odd
    // harmonic and doubled every even one, so a wire sounded an octave high
    // with no fundamental at all. h2 measured ~11800x h1.
    const net = new WaveguideNet();
    const L = 120;
    net.handle({ type: 'latch', topo: sampleTopo(1, 10, 20, L), wireId: 1, gain: 1 });
    const x = collect(net, 4096);
    const f0 = 48000 / (2 * L);
    const h1 = binMag(x, f0);
    expect(h1).toBeGreaterThan(binMag(x, f0 * 2));
    expect(h1).toBeGreaterThan(binMag(x, f0 * 3));
    expect(h1).toBeGreaterThan(binMag(x, f0 * 4));
  });

  it('excitation is zero-mean: no DC pedestal circulates in the loop', () => {
    // Regression: the pluck was an all-positive triangle and both terminations
    // reflect with near-unity gain, so DC was the longest-lived thing in the
    // system. Every sample in the delay lines used to share one sign.
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(1, 10, 20, 120), wireId: 1, gain: 1 });
    for (let i = 0; i < 2000; i++) net.tick();
    const w = net.wires[net.wireById.get(1)!];
    let dc = 0;
    let abs = 0;
    for (let d = 0; d < 120; d++) {
      const a = net.read(w.bufFwd, w.pos, d);
      const b = net.read(w.bufBack, w.pos, d);
      dc += a + b;
      abs += Math.abs(a) + Math.abs(b);
    }
    expect(abs).toBeGreaterThan(0);
    expect(Math.abs(dc / abs)).toBeLessThan(0.15);
  });

  it('decays to exact silence and stays there', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 10, 20, 120);
    topo.wires[0].loss = 0.9;
    net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
    expect(peak(collect(net, 400))).toBeGreaterThan(0.01);
    for (let i = 0; i < 48000 * 3; i++) net.tick();
    // The delay lines are zeroed outright, so no denormal grind and no tail
    // that creeps back up. The output stage still has one asymptotic DC-blocker
    // pole, which is inaudible rather than bit-exact zero.
    expect(net.energy()).toBe(0);
    expect(peak(collect(net, 2000))).toBeLessThan(1e-12);
  });

  it('pans a wire without changing what it plays', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 10, 20, 120);
    topo.wires[0].pan = -0.85;
    net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
    let l = 0;
    let r = 0;
    for (let i = 0; i < 4000; i++) {
      net.tick();
      l += Math.abs(net.outL);
      r += Math.abs(net.outR);
    }
    expect(l).toBeGreaterThan(r * 2);
    expect(r).toBeGreaterThan(0);
  });

  it('topology alone does not inject energy (silent until an impulse)', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleTopo(1, 10, 20) });
    expect(net.activeWireCount()).toBe(1);

    let p = 0;
    for (let i = 0; i < 500; i++) p = Math.max(p, Math.abs(net.tick()));
    expect(p).toBeLessThan(1e-6);
  });

  it('latch message applies topology then injects in one atomic step', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(7, 1, 2, 80), wireId: 7, gain: 0.8 });

    expect(peak(collect(net, 200))).toBeGreaterThan(0.01);
  });

  it('impulse before topology is ignored; after topology it rings', () => {
    const net = new WaveguideNet();
    expect(net.injectImpulse(3, 0, 1)).toBe(false);

    net.handle({ type: 'topology', topo: sampleTopo(3, 5, 6) });
    expect(net.injectImpulse(3, 0, 0.6)).toBe(true);

    expect(peak(collect(net, 300))).toBeGreaterThan(0.005);
  });

  it('many latches in sequence all remain audible', () => {
    const net = new WaveguideNet();
    const peaks: number[] = [];
    for (let id = 1; id <= 8; id++) {
      net.handle({
        type: 'latch',
        topo: {
          wires: [
            {
              id,
              length: 60 + id * 4,
              loss: 0.999,
              bend: 0.05,
              agentA: id * 2,
              agentB: id * 2 + 1,
              zA: 1,
              zB: 1,
            },
          ],
          agents: [
            { id: id * 2, kind: 0, openPorts: 0, impedance: 1 },
            { id: id * 2 + 1, kind: 1, openPorts: 0, impedance: 1 },
          ],
        },
        wireId: id,
        gain: 0.7,
      });
      peaks.push(peak(collect(net, 150)));
      for (let i = 0; i < 3000; i++) net.tick();
    }
    expect(peaks.every((p) => p > 0.008)).toBe(true);
  });

  it('energy decays over time (release, not sustain)', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(4, 1, 2, 100);
    topo.wires[0].loss = 0.992;
    net.handle({ type: 'latch', topo, wireId: 4, gain: 1 });
    const peakEarly = peak(collect(net, 800));
    for (let i = 0; i < 48000; i++) net.tick();
    const peakLate = peak(collect(net, 800));
    expect(peakEarly).toBeGreaterThan(0.01);
    expect(peakLate).toBeLessThan(peakEarly * 0.75);
  });

  it('bare topology refresh between latches does not silence later rings', () => {
    const net = new WaveguideNet();
    const topo1 = sampleTopo(1, 10, 20, 80);
    net.handle({ type: 'latch', topo: topo1, wireId: 1, gain: 0.8 });
    for (let i = 0; i < 2000; i++) net.tick();

    net.handle({ type: 'topology', topo: topo1 });
    net.handle({
      type: 'latch',
      topo: {
        wires: [
          { id: 1, length: 80, loss: 0.999, bend: 0.05, agentA: 10, agentB: 20, zA: 1, zB: 1 },
          { id: 2, length: 96, loss: 0.999, bend: 0.05, agentA: 30, agentB: 40, zA: 1, zB: 1 },
        ],
        agents: [
          { id: 10, kind: 0, openPorts: 0, impedance: 1 },
          { id: 20, kind: 1, openPorts: 0, impedance: 1 },
          { id: 30, kind: 0, openPorts: 0, impedance: 1 },
          { id: 40, kind: 2, openPorts: 0, impedance: 1 },
        ],
      },
      wireId: 2,
      gain: 0.8,
    });

    expect(peak(collect(net, 200))).toBeGreaterThan(0.01);
  });

  it('reused slot clears buffer on fresh wire id', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(1, 1, 2), wireId: 1, gain: 1 });
    for (let i = 0; i < 100; i++) net.tick();

    net.handle({ type: 'topology', topo: { wires: [], agents: [] } });
    net.handle({ type: 'latch', topo: sampleTopo(99, 3, 4), wireId: 99, gain: 1 });

    expect(peak(collect(net, 30))).toBeGreaterThan(0.01);
  });

  it('scatters energy through a shared junction onto another wire', () => {
    const net = new WaveguideNet();
    net.handle({
      type: 'topology',
      topo: {
        wires: [
          { id: 1, length: 48, loss: 0.9995, bend: 0.02, agentA: 1, agentB: 2, zA: 1, zB: 1 },
          { id: 2, length: 48, loss: 0.9995, bend: 0.02, agentA: 2, agentB: 3, zA: 1, zB: 1 },
        ],
        agents: [
          { id: 1, kind: 0, openPorts: 0, impedance: 1 },
          { id: 2, kind: 1, openPorts: 0, impedance: 1 },
          { id: 3, kind: 2, openPorts: 0, impedance: 1 },
        ],
      },
    });
    net.injectImpulse(1, 0, 1);
    for (let i = 0; i < 120; i++) net.tick();
    expect(net.wireEnergy(2)).toBeGreaterThan(net.wireEnergy(1) * 0.15);
  });

  it('live length change retunes the round-trip', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 1, 2, 64);
    net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
    for (let i = 0; i < 400; i++) net.tick();
    const longPeriod = autocorrLag(collect(net, 1500), 90, 160);

    topo.wires[0].length = 40;
    net.handle({ type: 'topology', topo });
    for (let i = 0; i < 400; i++) net.tick();
    const shortPeriod = autocorrLag(collect(net, 1500), 50, 160);

    expect(longPeriod).toBeGreaterThan(110);
    expect(shortPeriod).toBeLessThan(100);
    expect(shortPeriod).toBeLessThan(longPeriod - 20);
  });

  it('live topology update does not mute a ringing wire', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 1, 2, 80);
    net.handle({ type: 'latch', topo, wireId: 1, gain: 0.9 });
    for (let i = 0; i < 80; i++) net.tick();
    topo.wires[0].length = 70;
    topo.wires[0].bend = 0.12;
    topo.wires[0].loss = 0.998;
    net.handle({ type: 'topology', topo });
    expect(peak(collect(net, 200))).toBeGreaterThan(0.01);
  });

  it('does not self-oscillate: energy falls after a pluck with no further input', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 1, 2, 80);
    topo.wires[0].loss = 0.9;
    net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
    for (let i = 0; i < 400; i++) net.tick();
    const eEarly = net.energy();
    for (let i = 0; i < 4000; i++) net.tick();
    const eMid = net.energy();
    for (let i = 0; i < 48000 * 3; i++) net.tick();
    expect(eEarly).toBeGreaterThan(0);
    expect(eMid).toBeLessThan(eEarly);
    expect(net.energy()).toBe(0);
  });

  it('length wobble does not pump energy', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 1, 2, 80);
    net.handle({ type: 'latch', topo, wireId: 1, gain: 0.8 });
    for (let i = 0; i < 300; i++) net.tick();
    const e0 = net.energy();
    for (let k = 0; k < 40; k++) {
      topo.wires[0].length = k % 2 === 0 ? 88 : 72;
      net.handle({ type: 'topology', topo });
      for (let i = 0; i < 200; i++) net.tick();
    }
    expect(net.energy()).toBeLessThan(e0 * 1.5);
  });

  it('rewrite commit still clicks after the wire is gone', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(1, 1, 2, 80), wireId: 1, gain: 0.5 });
    net.handle({ type: 'topology', topo: { wires: [], agents: [] } });
    net.handle({
      type: 'rewrite',
      phase: 1,
      wireId: 1,
      agentA: 1,
      agentB: 2,
      leftovers: [],
      gain: 0.9,
    });
    expect(peak(collect(net, 40))).toBeGreaterThan(0.05);
  });

  it('rewrite begin plucks the principal wire', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleTopo(4, 1, 2, 90) });
    net.handle({
      type: 'rewrite',
      phase: 0,
      wireId: 4,
      agentA: 1,
      agentB: 2,
      leftovers: [],
      gain: 0.7,
    });
    expect(peak(collect(net, 200))).toBeGreaterThan(0.01);
  });
});

function snapCopy(net: WaveguideNet): Float32Array {
  return new Float32Array(net.fillWaveSnapshot());
}

function waveChannel(packed: Float32Array, wireId: number, dir: 'fwd' | 'back'): Float32Array {
  const n = packed[0] | 0;
  const bins = packed[1] | 0;
  const stride = 2 + bins * 2;
  let o = 2;
  for (let i = 0; i < n; i++) {
    if ((packed[o] | 0) === wireId) {
      const base = o + 2 + (dir === 'back' ? bins : 0);
      return packed.subarray(base, base + bins);
    }
    o += stride;
  }
  throw new Error(`wire ${wireId} missing from snapshot`);
}

function absEnergy(xs: ArrayLike<number>): number {
  let e = 0;
  for (let i = 0; i < xs.length; i++) e += Math.abs(xs[i]);
  return e;
}

function argmaxAbs(xs: ArrayLike<number>): number {
  let b = 0;
  let v = -1;
  for (let i = 0; i < xs.length; i++) {
    const a = Math.abs(xs[i]);
    if (a > v) {
      v = a;
      b = i;
    }
  }
  return b;
}

describe('traveling-wave snapshot', () => {
  it('omits a silent wire and tags bins in the header', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleTopo(1, 10, 20, 64) });
    const packed = snapCopy(net);
    expect(packed[0]).toBe(0);
    expect(packed[1]).toBeGreaterThan(1);
  });

  it('a pluck appears in both directions', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(1, 10, 20, 80), wireId: 1, gain: 1 });
    const packed = snapCopy(net);
    expect(packed[0]).toBe(1);
    expect(absEnergy(waveChannel(packed, 1, 'fwd'))).toBeGreaterThan(0.1);
    expect(absEnergy(waveChannel(packed, 1, 'back'))).toBeGreaterThan(0.1);
  });

  it('an impulse at A travels toward B on the forward line', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleTopo(1, 10, 20, 64) });
    net.injectImpulse(1, 0, 1);
    for (let i = 0; i < 12; i++) net.tick();
    const early = snapCopy(net);
    const fwdEarly = waveChannel(early, 1, 'fwd');
    expect(absEnergy(fwdEarly)).toBeGreaterThan(absEnergy(waveChannel(early, 1, 'back')) * 2);
    const peakEarly = argmaxAbs(fwdEarly);

    for (let i = 0; i < 24; i++) net.tick();
    const later = snapCopy(net);
    const peakLater = argmaxAbs(waveChannel(later, 1, 'fwd'));
    expect(peakLater).toBeGreaterThan(peakEarly);
  });

  it('an impulse at B travels toward A on the backward line', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleTopo(1, 10, 20, 64) });
    net.injectImpulse(1, 1, 1);
    for (let i = 0; i < 12; i++) net.tick();
    const packed = snapCopy(net);
    const back = waveChannel(packed, 1, 'back');
    expect(absEnergy(back)).toBeGreaterThan(absEnergy(waveChannel(packed, 1, 'fwd')) * 2);
    expect(argmaxAbs(back)).toBeGreaterThan((packed[1] | 0) * 0.5);
  });
});

function ringingBody(id: number, kind: 0 | 1 | 2): NetTopology['agents'][number] {
  return {
    id,
    kind,
    openPorts: 0,
    impedance: 1,
    modeHz: [180, 380, 620],
    modeT60: [0.3, 0.18, 0.1],
    modeGain: [0.7, 0.4, 0.22],
  };
}

describe('net as one instrument', () => {
  it('a plucked wire makes the bodies hum without a strike', () => {
    const net = new WaveguideNet();
    const topo: NetTopology = {
      wires: [sampleTopo(1, 10, 20, 80).wires[0]],
      agents: [ringingBody(10, 0), ringingBody(20, 2)],
    };
    net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
    for (let i = 0; i < 200; i++) net.tick();
    const a = net.agents[net.agentById.get(10)!];
    expect(a.bodyEnv).toBeGreaterThan(1e-4);
  });

  it('a retired ringing wire dumps into the bodies', () => {
    const net = new WaveguideNet();
    const topo: NetTopology = {
      wires: [sampleTopo(1, 10, 20, 80).wires[0]],
      agents: [ringingBody(10, 0), ringingBody(20, 2)],
    };
    net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
    for (let i = 0; i < 200; i++) net.tick();
    expect(net.energy()).toBeGreaterThan(0);
    net.handle({
      type: 'topology',
      topo: { wires: [], agents: [ringingBody(10, 0), ringingBody(20, 2)] },
    });
    expect(peak(collect(net, 80))).toBeGreaterThan(0.001);
  });

  it('a new wire steals a quiet slot once the table is full', () => {
    const net = new WaveguideNet();
    const wires: NetTopology['wires'] = [];
    const agents: NetTopology['agents'] = [];
    for (let i = 0; i < 64; i++) {
      const a = i * 2 + 1;
      const b = i * 2 + 2;
      wires.push({
        id: i + 1,
        length: 32,
        loss: 0.99,
        bend: 0.05,
        agentA: a,
        agentB: b,
        zA: 1,
        zB: 1,
      });
      agents.push({ id: a, kind: 0, openPorts: 0, impedance: 1 });
      agents.push({ id: b, kind: 1, openPorts: 0, impedance: 1 });
    }
    net.handle({ type: 'topology', topo: { wires, agents } });
    expect(net.activeWireCount()).toBe(64);

    wires.push({
      id: 65,
      length: 32,
      loss: 0.99,
      bend: 0.05,
      agentA: 200,
      agentB: 201,
      zA: 1,
      zB: 1,
    });
    agents.push({ id: 200, kind: 0, openPorts: 0, impedance: 1 });
    agents.push({ id: 201, kind: 1, openPorts: 0, impedance: 1 });
    net.handle({ type: 'topology', topo: { wires, agents } });

    expect(net.activeWireCount()).toBe(64);
    expect(net.wireById.has(65)).toBe(true);
    expect(net.wireById.has(64)).toBe(false);
  });
});

function plate(id: number, hz: number[]): NetTopology['agents'][number] {
  return {
    id,
    kind: 0,
    openPorts: 3,
    impedance: 1,
    pan: 0,
    modeHz: hz,
    modeT60: [0.4, 0.22, 0.12],
    modeGain: [0.7, 0.4, 0.22],
  };
}

function inert(id: number): NetTopology['agents'][number] {
  return { id, kind: 1, openPorts: 0, impedance: 1 };
}

function bowedAgainstRail(): NetTopology {
  return { wires: [], agents: [plate(1, [180, 380, 620]), inert(2)] };
}

function twoPlates(): NetTopology {
  return { wires: [], agents: [plate(1, [180, 380, 620]), plate(2, [260, 510, 780])] };
}

function touching(
  a: number,
  b: number,
  load: number,
  slide: number,
): { type: 'contact'; items: { agentA: number; agentB: number; load: number; slide: number }[] } {
  return { type: 'contact', items: [{ agentA: a, agentB: b, load, slide }] };
}

describe('friction bowing', () => {
  it('an unwired body sings near its plate mode, not the slide rate', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: bowedAgainstRail() });
    net.handle(touching(1, 2, 0.8, bowSpeed(40)));
    collect(net, 8000);
    const x = collect(net, 4096);
    expect(peak(x)).toBeGreaterThan(0.01);
    const f0 = binMag(x, 180);
    expect(f0).toBeGreaterThan(binMag(x, 90));
    expect(f0).toBeGreaterThan(binMag(x, 700));
  });

  it('a faster slide is still the plate, not a high grind', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: bowedAgainstRail() });
    net.handle(touching(1, 2, 0.85, bowSpeed(90)));
    collect(net, 8000);
    const x = collect(net, 4096);
    expect(peak(x)).toBeGreaterThan(0.01);
    expect(binMag(x, 180) + binMag(x, 380)).toBeGreaterThan(binMag(x, 1600));
  });

  it('goes quiet after the bow lifts', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: bowedAgainstRail() });
    net.handle(touching(1, 2, 0.8, bowSpeed(40)));
    collect(net, 8000);
    expect(peak(collect(net, 512))).toBeGreaterThan(0.01);
    net.handle({ type: 'contact', items: [] });
    collect(net, 48000 * 2);
    expect(peak(collect(net, 2048))).toBeLessThan(1e-4);
  });

  it('a Hertzian strike still knocks an unwired body', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: bowedAgainstRail() });
    net.handle({ type: 'strike', agentId: 1, peak: 1.2, dur: 40, sharp: 0.6 });
    expect(peak(collect(net, 512))).toBeGreaterThan(0.05);
  });

  it('friction at a junction puts traveling-wave energy on the wire', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 10, 20, 80);
    topo.agents[0] = { ...ringingBody(10, 0), impedance: 1 };
    topo.agents[1] = { ...ringingBody(20, 2), impedance: 1.2 };
    net.handle({ type: 'topology', topo });
    net.handle(touching(10, 20, 0.85, bowSpeed(40)));
    collect(net, 4000);
    expect(net.wireEnergy(1)).toBeGreaterThan(1e-6);
  });

  it('a ringing body in resting contact drives the other', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: twoPlates() });
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 48, sharp: 0.5 });
    collect(net, 200);
    net.handle(touching(1, 2, 0.9, 0));
    collect(net, 4000);
    const b = net.agents[net.agentById.get(2)!];
    expect(b.bodyEnv).toBeGreaterThan(1e-4);
  });

  it('resting contact does not howl', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: twoPlates() });
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 48, sharp: 0.5 });
    net.handle(touching(1, 2, 0.9, 0));
    const early = peak(collect(net, 2048));
    collect(net, 24000);
    const late = peak(collect(net, 2048));
    expect(early).toBeGreaterThan(0.01);
    expect(late).toBeLessThan(early * 0.5);
  });
});

function airPath(
  a: number,
  b: number,
  length: number,
  gain = 0.18,
  damp = 0.8,
): { type: 'air'; items: { agentA: number; agentB: number; length: number; gain: number; damp: number }[] } {
  return { type: 'air', items: [{ agentA: a, agentB: b, length, gain, damp }] };
}

describe('air coupling', () => {
  it('a strike on A rings B after the air delay, not before', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: twoPlates() });
    net.handle(airPath(1, 2, 64));
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 24, sharp: 0.5 });
    collect(net, 24);
    const earlyB = net.agents[net.agentById.get(2)!].bodyEnv;
    collect(net, 120);
    const lateB = net.agents[net.agentById.get(2)!].bodyEnv;
    expect(earlyB).toBeLessThan(1e-5);
    expect(lateB).toBeGreaterThan(1e-4);
  });

  it('without an air path, a strike on A leaves B quiet', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: twoPlates() });
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 24, sharp: 0.5 });
    collect(net, 400);
    expect(net.agents[net.agentById.get(2)!].bodyEnv).toBeLessThan(1e-6);
  });

  it('clearing the air list stops the coupling', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: twoPlates() });
    net.handle(airPath(1, 2, 32));
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 24, sharp: 0.5 });
    collect(net, 200);
    expect(net.agents[net.agentById.get(2)!].bodyEnv).toBeGreaterThan(1e-4);
    net.handle({ type: 'air', items: [] });
    const b = net.agents[net.agentById.get(2)!];
    const held = b.bodyEnv;
    collect(net, 400);
    expect(b.bodyEnv).toBeLessThanOrEqual(held);
  });

  it('air coupling decays instead of howling', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: twoPlates() });
    net.handle(airPath(1, 2, 48, 0.18, 0.75));
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 40, sharp: 0.5 });
    const early = peak(collect(net, 2048));
    collect(net, 24000);
    const late = peak(collect(net, 2048));
    expect(early).toBeGreaterThan(0.01);
    expect(late).toBeLessThan(early * 0.6);
  });

  it('a NaN air delay does not poison the output', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: twoPlates() });
    net.handle(airPath(1, 2, Number.NaN));
    net.handle({ type: 'strike', agentId: 1, peak: 1.2, dur: 24, sharp: 0.5 });
    const x = collect(net, 512);
    expect(x.every(Number.isFinite)).toBe(true);
  });
});
