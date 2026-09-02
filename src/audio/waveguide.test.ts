import { describe, expect, it } from 'vitest';
import { WaveguideNet, WAVE_BINS, junctionLoadY } from './waveguide.ts';
import { bowSpeed, componentLoadScale } from './presets.ts';
import type { AgentTopo, NetTopology } from './types.ts';
import { LOD_FAR, LOD_MID, LOD_NEAR } from './lod.ts';
import { buildTopology } from './topology.ts';
import { bodyTone } from './body.ts';
import { Sim } from '../sim.ts';
import { defaultParams } from '../params.ts';
import { loadPreset } from '../presets.ts';

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

describe('voice ceiling', () => {
  it('a cascade is reclaimed down to the voice budget', () => {
    // Regression: this was a rising threshold that ratcheted to its ceiling
    // whenever a busy soup held the awake count near the cap, and then silenced
    // every body below it. A full soup went ~20x quieter a few seconds in and
    // stayed there. Ranking removes exactly the overflow and cannot run away.
    // Nothing else bounds concurrent voices: the net stays inside its callback
    // budget only because gating happens to keep counts low. Strike far more
    // bodies than the budget allows and the quietest have to be stolen back.
    const N = 200;
    const agents: AgentTopo[] = [];
    for (let i = 1; i <= N; i++) {
      agents.push({
        id: i,
        kind: (i % 3) as 0 | 1 | 2,
        openPorts: 2,
        stubs: [
          { slot: 0 as const, length: 10, z: 1 },
          { slot: 1 as const, length: 8, z: 0.62 },
        ],
        impedance: 1,
        pan: 0,
        dist: 0.5,
        modeHz: [180, 380, 620],
        modeT60: [0.9, 0.5, 0.3],
        modeGain: [0.7, 0.4, 0.22],
        coupling: 1,
      });
    }
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: { wires: [], agents } });
    for (let i = 1; i <= N; i++) {
      net.handle({ type: 'strike', agentId: i, peak: 1.2, dur: 60, sharp: 0.6 });
    }

    const awake = () => net.agents.filter((a) => a.active && !a.quiet).length;
    expect(awake()).toBe(N);

    // Bounded to MAX_STEALS per quantum, so a 200-body cascade needs a couple
    // of hundred quanta to drain. Half a second is ample.
    for (let i = 0; i < 48000 * 0.5; i++) net.tick(false);
    expect(awake()).toBeLessThanOrEqual(44);

    // And the survivors still decay to silence rather than being pinned alive.
    for (let i = 0; i < 48000 * 6; i++) net.tick(false);
    expect(awake()).toBe(0);
  });
});

describe('stale index recovery', () => {
  it('a contact follows its bodies when agent slots are reshuffled', () => {
    // Agent slots are packed in topo order, so erasing one body shifts every
    // later body down a slot. A contact captured before that points at the
    // wrong pair, and the force it applies is a real Hertzian spring plus
    // friction between two bodies that are not touching.
    const body = (id: number) => ({
      id,
      kind: 0 as const,
      openPorts: 0,
      impedance: 1,
      pan: 0,
      dist: 0.5,
      modeHz: [180, 380, 620],
      modeT60: [0.25, 0.14, 0.08],
      modeGain: [0.7, 0.4, 0.22],
      coupling: 1,
    });
    const topoOf = (ids: number[]) => ({ wires: [], agents: ids.map(body) });

    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: topoOf([1, 2, 3]) });
    net.handle({ type: 'contact', items: [{ agentA: 2, agentB: 3, load: 1, slide: 0.02 }] });

    const c = net.contacts.find((x) => x.active)!;
    expect(c.idA).toBe(2);
    expect(c.idB).toBe(3);

    // Agent 1 is erased: 2 and 3 shift down, 4 arrives.
    net.handle({ type: 'topology', topo: topoOf([2, 3, 4]) });
    net.tick(false);

    expect(net.agents[c.idxA].id).toBe(c.idA);
    expect(net.agents[c.idxB].id).toBe(c.idB);
  });
});

describe('WaveguideNet', () => {
  it('a closed junction still leaks a little', () => {
    expect(junctionLoadY(0)).toBeGreaterThan(0.004);
    expect(junctionLoadY(0)).toBeLessThan(0.05);
    expect(junctionLoadY(0, 1)).toBe(junctionLoadY(0, 2));
    expect(junctionLoadY(0, 32)).toBeGreaterThan(junctionLoadY(0, 2) * 1.4);
    expect(junctionLoadY(0, 32)).toBeLessThan(0.05);
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

  it('a tune message changes delay without rebuilding the net', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 1, 2, 80);
    net.handle({ type: 'latch', topo, wireId: 1, gain: 0.8 });
    collect(net, 400);
    const e0 = net.energy();
    net.handle({ type: 'tune', wires: [{ id: 1, length: 120, damp: 0.4, loss: 0.998 }] });
    expect(net.energy()).toBeGreaterThan(e0 * 0.5);
    expect(peak(collect(net, 200))).toBeGreaterThan(0.001);
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

  it('zeroWaveSnapshot is an empty header so a late callback can drop offsets', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(1, 10, 20, 80), wireId: 1, gain: 1 });
    expect(snapCopy(net)[0]).toBe(1);
    const z = new Float32Array(net.zeroWaveSnapshot());
    expect(z[0]).toBe(0);
    expect(z[1]).toBe(WAVE_BINS);
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
    // A scrape is deliberately the quietest thing in the mix — under a bowed
    // string, well under a collision — so this only checks that it speaks at
    // all. The assertion that matters is the silence below.
    expect(peak(collect(net, 512))).toBeGreaterThan(0.003);
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

function openStub(length = 8): { slot: 0; length: number; z: number } {
  return { slot: 0, length, z: 1 };
}

function withOpenEnd(topo: NetTopology, agentId: number, length = 8): NetTopology {
  return {
    ...topo,
    agents: topo.agents.map((a) =>
      a.id === agentId ? { ...a, openPorts: 1, stubs: [openStub(length)] } : a,
    ),
  };
}

describe('open stems', () => {
  it('a closed-open rope is an octave below a closed-closed rope', () => {
    const L = 80;
    const fHalf = 48000 / (2 * L);
    const fQuarter = 48000 / (4 * L);

    const closed = new WaveguideNet();
    closed.handle({ type: 'latch', topo: sampleTopo(1, 10, 20, L), wireId: 1, gain: 1 });
    collect(closed, 400);
    const xClosed = collect(closed, 4096);

    const mixed = new WaveguideNet();
    mixed.handle({
      type: 'latch',
      topo: withOpenEnd(sampleTopo(1, 10, 20, L), 20),
      wireId: 1,
      gain: 1,
    });
    collect(mixed, 400);
    const xMixed = collect(mixed, 4096);

    expect(binMag(xClosed, fHalf)).toBeGreaterThan(binMag(xClosed, fQuarter));
    expect(binMag(xMixed, fQuarter)).toBeGreaterThan(binMag(xMixed, fHalf));
  });

  it('latching a stem retires its stub', () => {
    const net = new WaveguideNet();
    const open: NetTopology = {
      wires: [],
      agents: [{ id: 1, kind: 1, openPorts: 3, impedance: 1, stubs: [
        { slot: 0, length: 8, z: 1 },
        { slot: 1, length: 8, z: 0.62 },
        { slot: 2, length: 8, z: 0.62 },
      ] }],
    };
    net.handle({ type: 'topology', topo: open });
    expect(net.stubs.filter((s) => s.active).length).toBe(3);
    net.handle({
      type: 'topology',
      topo: {
        wires: [sampleTopo(1, 1, 2, 80).wires[0]],
        agents: [
          { id: 1, kind: 1, openPorts: 2, impedance: 1, stubs: [
            { slot: 1, length: 8, z: 0.62 },
            { slot: 2, length: 8, z: 0.62 },
          ] },
          { id: 2, kind: 0, openPorts: 0, impedance: 1 },
        ],
      },
    });
    const live = net.stubs.filter((s) => s.active);
    expect(live).toHaveLength(2);
    expect(live.every((s) => s.slot !== 0)).toBe(true);
  });

  it('an open lip leaks instead of howling', () => {
    const net = new WaveguideNet();
    net.handle({
      type: 'latch',
      topo: withOpenEnd(sampleTopo(1, 10, 20, 80), 20),
      wireId: 1,
      gain: 1,
    });
    collect(net, 400);
    const early = peak(collect(net, 512));
    collect(net, 24000);
    const late = peak(collect(net, 512));
    expect(early).toBeGreaterThan(0.01);
    expect(late).toBeLessThan(early * 0.5);
  });

  it('a NaN stub delay does not poison the output', () => {
    const net = new WaveguideNet();
    net.handle({
      type: 'latch',
      topo: withOpenEnd(sampleTopo(1, 10, 20, 80), 20, Number.NaN),
      wireId: 1,
      gain: 1,
    });
    const x = collect(net, 512);
    expect(x.every(Number.isFinite)).toBe(true);
    expect(peak(x)).toBeGreaterThan(0);
  });
});

describe('listener distance mix', () => {
  function knock(dist: number, height = 0): { dry: number; wet: number } {
    const net = new WaveguideNet();
    const a = plate(1, [180, 380, 620]);
    a.dist = dist;
    net.handle({ type: 'topology', topo: { wires: [], agents: [a], height } });
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 40, sharp: 0.5 });
    let dry = 0;
    let wet = 0;
    for (let i = 0; i < 512; i++) {
      net.tick();
      dry = Math.max(dry, Math.abs(net.outDryL) + Math.abs(net.outDryR));
      wet = Math.max(wet, Math.abs(net.outWetL) + Math.abs(net.outWetR));
    }
    return { dry, wet };
  }

  it('a far body is quieter on the dry bus, and wetter relative to dry', () => {
    const near = knock(0);
    const far = knock(2);
    expect(near.dry).toBeGreaterThan(0.01);
    expect(far.dry).toBeLessThan(near.dry * 0.7);
    expect(far.wet / Math.max(1e-9, far.dry)).toBeGreaterThan(near.wet / Math.max(1e-9, near.dry));
  });

  it('pulling the listener up quiets dry and wet together', () => {
    const low = knock(0.2, 0.2);
    const high = knock(5, 5);
    expect(high.dry).toBeLessThan(low.dry * 0.55);
    expect(high.wet).toBeLessThan(low.wet * 0.55);
    expect(high.dry + high.wet).toBeLessThan((low.dry + low.wet) * 0.5);
  });

  it('a listen message changes distance without rebuilding the net', () => {
    const near = knock(0.2, 0.2);
    const net = new WaveguideNet();
    const a = plate(1, [180, 380, 620]);
    a.dist = 0.2;
    net.handle({ type: 'topology', topo: { wires: [], agents: [a], height: 0.2 } });
    net.handle({
      type: 'listen',
      height: 5,
      wires: [],
      agents: [{ id: 1, pan: 0, dist: 5 }],
    });
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 40, sharp: 0.5 });
    let dry = 0;
    let wet = 0;
    for (let i = 0; i < 512; i++) {
      net.tick();
      dry = Math.max(dry, Math.abs(net.outDryL) + Math.abs(net.outDryR));
      wet = Math.max(wet, Math.abs(net.outWetL) + Math.abs(net.outWetR));
    }
    expect(dry).toBeLessThan(near.dry * 0.55);
    expect(wet).toBeLessThan(near.wet * 0.55);
  });
});

describe('silent-body skip', () => {
  it('a strike on one body in a crowd still rings, and the others stay quiet', () => {
    const agents: NetTopology['agents'] = [];
    for (let i = 1; i <= 40; i++) agents.push(ringingBody(i, 0));
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: { wires: [], agents } });
    net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 40, sharp: 0.5 });
    const x = collect(net, 512);
    expect(peak(x)).toBeGreaterThan(0.01);
    expect(net.agents[net.agentById.get(1)!].quiet).toBe(false);
    expect(net.agents[net.agentById.get(2)!].quiet).toBe(true);
    expect(net.agents[net.agentById.get(2)!].bodyEnv).toBeLessThan(1e-6);
  });

  it('a latch still sounds if topology was already applied', () => {
    const net = new WaveguideNet();
    const topo = sampleTopo(1, 10, 20, 80);
    net.handle({ type: 'topology', topo });
    net.handle({ type: 'latch', topo, wireId: 1, gain: 1 });
    expect(peak(collect(net, 256))).toBeGreaterThan(0.01);
  });

  it('shedding air does not mute a ringing wire', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(1, 10, 20, 80), wireId: 1, gain: 1 });
    const x = collect(net, 64);
    for (let i = 0; i < 64; i++) net.tick(true);
    expect(peak(x)).toBeGreaterThan(0.01);
    expect(peak(collect(net, 64))).toBeGreaterThan(0.01);
  });
});

describe('event ducking', () => {
  it('an event steps the sustained bed aside, then lets it back', () => {
    // Bows, scrapes and air are continuous; collisions are short. Measured
    // against a running bed a collision cleared it by only ~4 dB, which is
    // where a transient stops being heard — not because it was quiet, but
    // because it had nothing to stand out from.
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleTopo(1, 10, 20, 150) });
    net.handle({
      type: 'contact',
      items: [{ agentA: 10, agentB: 20, load: 0.8, slide: bowSpeed(120) }],
    });
    collect(net, 24000);
    expect(net.duck).toBe(1);

    net.handle({ type: 'strike', agentId: 10, peak: 0.6, dur: 90, sharp: 0.6 });
    expect(net.duck).toBeLessThan(0.8);
    expect(net.duck).toBeGreaterThanOrEqual(0.25);

    // Recovers within a few hundred ms rather than gating the bed shut.
    collect(net, 24000);
    expect(net.duck).toBeGreaterThan(0.97);
  });

  it('a flurry ducks once rather than stacking to silence', () => {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: sampleTopo(1, 10, 20, 150) });
    for (let i = 0; i < 30; i++) {
      net.handle({ type: 'strike', agentId: 10, peak: 1.5, dur: 60, sharp: 0.8 });
    }
    expect(net.duck).toBeGreaterThanOrEqual(0.25);
  });
});

describe('loudness ordering', () => {
  it('collision is louder than a scrape', () => {
    // A deliberate mix decision, and a fragile one: the three levels are set by
    // constants that interact non-linearly, and two of them have already been
    // found to flip the order when nudged. Measured at matched drive so the
    // comparison is about the sources, not about how often each happens.
    const body = (id: number, hz: number[]) => ({
      id,
      kind: 0 as const,
      openPorts: 3,
      impedance: 1,
      pan: 0,
      dist: 0.5,
      modeHz: hz,
      modeT60: [0.4, 0.22, 0.12],
      modeGain: [0.7, 0.4, 0.22],
      coupling: 1,
    });
    const rope = (id: number, a: number, b: number) => ({
      id, length: 150, loss: 0.9992, bend: 0.04, agentA: a, agentB: b,
      zA: 1, zB: 1, damp: 0.6, disp: 0, pan: 0, dist: 0.5, exAt: 0.16, exWidth: 1,
    });
    const topo = {
      wires: [rope(1, 1, 2), rope(2, 3, 4)],
      agents: [
        body(1, [180, 380, 620]), body(2, [200, 420, 700]),
        body(3, [170, 360, 590]), body(4, [210, 440, 720]),
      ],
    };
    const rmsOf = (drive: (n: WaveguideNet) => void) => {
      const net = new WaveguideNet();
      net.handle({ type: 'topology', topo });
      drive(net);
      let sum = 0;
      const n = 48000;
      for (let i = 0; i < n; i++) {
        net.tick(false);
        const v = net.outDryL + net.outWetL;
        sum += v * v;
      }
      return Math.sqrt(sum / n);
    };

    const collision = rmsOf((net) =>
      net.handle({ type: 'strike', agentId: 1, peak: 0.73, dur: 88, sharp: 0.6 }),
    );
    // Measured on bodies with nothing tied to them.
    const scrape = (() => {
      const net = new WaveguideNet();
      net.handle({
        type: 'topology',
        topo: { wires: [], agents: [body(1, [180, 380, 620]), body(2, [200, 420, 700])] },
      });
      net.handle({
        type: 'contact',
        items: [{ agentA: 1, agentB: 2, load: 0.8, slide: bowSpeed(120) }],
      });
      let sum = 0;
      const n = 48000;
      for (let i = 0; i < n; i++) {
        net.tick(false);
        const v = net.outDryL + net.outWetL;
        sum += v * v;
      }
      return Math.sqrt(sum / n);
    })();

    // Both still have to be audible.
    expect(scrape).toBeGreaterThan(0.004);
    expect(collision).toBeGreaterThan(scrape);
  });
});

describe('annihilation release', () => {
  /** Latch a wire, ring it, then take the whole net away. */
  function annihilate(length: number, gain = 0.9): WaveguideNet {
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(1, 1, 2, length), wireId: 1, gain: 0.6 });
    for (let i = 0; i < 2000; i++) net.tick();
    net.handle({ type: 'topology', topo: { wires: [], agents: [] } });
    net.handle({
      type: 'rewrite', phase: 1, wireId: 1, agentA: 1, agentB: 2, leftovers: [], gain,
    });
    return net;
  }

  it('sounds at the pitch of the string that died, not as noise', () => {
    // The old commit was filtered white noise, which has no period at all.
    // A release is the string letting go, so it still has the string's own.
    const net = annihilate(120);
    const out = collect(net, 3000);
    const lag = autocorrLag(out, 60, 400);
    expect(Math.abs(lag - 240)).toBeLessThan(24);
  });

  it('takes its pitch from the wire, so different wires end differently', () => {
    const shortLag = autocorrLag(collect(annihilate(80), 3000), 40, 400);
    const longLag = autocorrLag(collect(annihilate(200), 3000), 40, 500);
    expect(longLag).toBeGreaterThan(shortLag * 1.8);
  });

  it('decays away rather than ringing on, and gives the slot back', () => {
    const net = annihilate(120);
    expect(peak(collect(net, 2000))).toBeGreaterThan(0.01);
    // Run past RELEASE_SEC, then listen to what is left. The reverb has its
    // own tail, so this asks whether the source stopped, not the room.
    collect(net, 30000);
    expect(peak(collect(net, 4000))).toBeLessThan(1e-3);
    for (const w of net.wires) expect(w.active).toBe(false);
  });

  it('lets a wire go even when nothing announces it', () => {
    // No rewrite message at all: the topology just drops a ringing wire. It
    // should still be heard leaving instead of being cut off mid-ring.
    const net = new WaveguideNet();
    net.handle({ type: 'latch', topo: sampleTopo(1, 1, 2, 120), wireId: 1, gain: 0.6 });
    for (let i = 0; i < 2000; i++) net.tick();
    net.handle({ type: 'topology', topo: { wires: [], agents: [] } });
    expect(peak(collect(net, 1500))).toBeGreaterThan(0.005);
  });

  it('is placed where it happened, not dead centre', () => {
    const make = (pan: number) => {
      const topo = sampleTopo(1, 1, 2, 120);
      topo.wires[0].pan = pan;
      const net = new WaveguideNet();
      net.handle({ type: 'latch', topo, wireId: 1, gain: 0.6 });
      for (let i = 0; i < 2000; i++) net.tick();
      net.handle({ type: 'topology', topo: { wires: [], agents: [] } });
      let l = 0;
      let r = 0;
      for (let i = 0; i < 3000; i++) {
        net.tick();
        l = Math.max(l, Math.abs(net.outDryL));
        r = Math.max(r, Math.abs(net.outDryR));
      }
      return { l, r };
    };
    const left = make(-1);
    expect(left.l).toBeGreaterThan(0);
    expect(left.r).toBeLessThan(left.l * 0.1);
  });
});

describe('lod mix', () => {
  function agent(id: number, lod: number): AgentTopo {
    return {
      id,
      kind: (id % 3) as 0 | 1 | 2,
      openPorts: 0,
      impedance: 1,
      pan: 0,
      dist: 0.5,
      lod,
      modeHz: [180, 380, 620],
      modeT60: [0.4, 0.22, 0.12],
      modeGain: [0.7, 0.4, 0.22],
      coupling: 1,
    };
  }

  function islands(n: number, lod: number): NetTopology {
    const wires = [];
    const agents: AgentTopo[] = [];
    for (let i = 0; i < n; i++) {
      const a = i * 2 + 1;
      const b = i * 2 + 2;
      wires.push({
        id: i + 1,
        length: 140,
        loss: 0.9992,
        bend: 0.04,
        agentA: a,
        agentB: b,
        zA: 1,
        zB: 1,
        damp: 0.6,
        disp: 0,
        pan: 0,
        dist: 0.5,
        exAt: 0.16,
        exWidth: 1,
        lod,
      });
      agents.push(agent(a, lod), agent(b, lod));
    }
    return { wires, agents, height: 0.45 };
  }

  function drive(topo: NetTopology, plucked: number[], n = 8192): { rms: number; peak: number; rail: number } {
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo });
    for (const id of plucked) net.injectPluck(id, 0.9);
    let sum = 0;
    let pk = 0;
    let rail = 0;
    for (let i = 0; i < n; i++) {
      net.tick();
      const d = Math.max(Math.abs(net.outDryL), Math.abs(net.outDryR));
      const w = Math.max(Math.abs(net.outWetL), Math.abs(net.outWetR));
      const v = net.outDryL + net.outWetL;
      sum += v * v;
      if (d > pk) pk = d;
      if (w > pk) pk = w;
      if (d > 0.98 || w > 0.98) rail++;
    }
    return { rms: Math.sqrt(sum / n), peak: pk, rail: rail / n };
  }

  it('a NEAR pluck is not turned down by silent FAR objects', () => {
    const solo = drive(islands(1, LOD_NEAR), [1]);
    const withFar = new WaveguideNet();
    const topo = islands(16, LOD_FAR);
    topo.wires[0].lod = LOD_NEAR;
    topo.agents[0].lod = LOD_NEAR;
    topo.agents[1].lod = LOD_NEAR;
    withFar.handle({ type: 'topology', topo });
    withFar.injectPluck(1, 0.9);
    let sum = 0;
    for (let i = 0; i < 8192; i++) {
      withFar.tick();
      const v = withFar.outDryL + withFar.outWetL;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / 8192);
    expect(rms).toBeGreaterThan(solo.rms * 0.9);
    expect(rms).toBeLessThan(solo.rms * 1.15);
  });

  it('sixteen FAR plucks stay off the clip rail', () => {
    const far = drive(islands(16, LOD_FAR), Array.from({ length: 16 }, (_, i) => i + 1));
    expect(far.peak).toBeLessThan(0.95);
    expect(far.rail).toBe(0);
    expect(far.rms).toBeGreaterThan(0.01);
  });

  it('sixteen FAR plucks do not turn a NEAR pluck down', () => {
    const nearOnly = drive(islands(1, LOD_NEAR), [1]);
    const mixedTopo = islands(16, LOD_FAR);
    mixedTopo.wires[0].lod = LOD_NEAR;
    mixedTopo.agents[0].lod = LOD_NEAR;
    mixedTopo.agents[1].lod = LOD_NEAR;
    const mixed = drive(mixedTopo, Array.from({ length: 16 }, (_, i) => i + 1));
    // FAR adds a bed, so the mix can be louder — it must not be quieter.
    expect(mixed.rms).toBeGreaterThan(nearOnly.rms * 0.9);
    expect(mixed.peak).toBeLessThan(0.95);
  });

  it('a MID clump is quieter than the same clump at NEAR', () => {
    const ids = Array.from({ length: 8 }, (_, i) => i + 1);
    const near = drive(islands(8, LOD_NEAR), ids);
    const mid = drive(islands(8, LOD_MID), ids);
    expect(mid.rms).toBeLessThan(near.rms * 0.85);
    expect(mid.rms).toBeGreaterThan(0.01);
  });
});

describe('component loading', () => {
  it('leaves a pair and an unwired body unscaled', () => {
    expect(componentLoadScale(1)).toBe(1);
    expect(componentLoadScale(2)).toBe(1);
    expect(componentLoadScale(32)).toBeLessThan(0.45);
    expect(componentLoadScale(32)).toBeGreaterThan(0.3);
  });

  function chain(n: number): Sim {
    const sim = new Sim(1600, 400);
    const params = defaultParams();
    params.spawnInterval = 0;
    params.maxAgents = n;
    const made = [];
    for (let i = 0; i < n; i++) {
      made.push(sim.spawn('dup', 40 + i * 36, 180, 0, params, true)!);
    }
    for (let i = 0; i < n - 1; i++) {
      sim.wire(made[i].id, i === 0 ? 'p' : 'l', made[i + 1].id, 'p', params);
    }
    return sim;
  }

  it('a lone pair keeps the free-body T60, a 32-agent clump does not', () => {
    const pairSim = chain(2);
    const clumpSim = chain(32);
    const pair = buildTopology(pairSim.graph, pairSim.agents);
    const clump = buildTopology(clumpSim.graph, clumpSim.agents);
    const pairAgent = [...pairSim.agents.values()][0];
    const free = bodyTone(pairAgent).decay[0];
    expect(pair.agents[0].compN).toBe(2);
    expect(clump.agents[0].compN).toBe(32);
    expect(pair.agents[0].modeT60![0]).toBeCloseTo(free, 5);
    expect(clump.agents[0].modeT60![0]).toBeLessThan(free * 0.5);
    expect(clump.agents[0].modeT60![0]).toBeCloseTo(free * componentLoadScale(32), 5);
  });

  it('the bed of a large component is quieter; a knock still hits', () => {
    const strikePeak = (sim: Sim) => {
      const topo = buildTopology(sim.graph, sim.agents);
      const net = new WaveguideNet();
      net.handle({ type: 'topology', topo });
      net.handle({ type: 'strike', agentId: topo.agents[0].id, peak: 1.2, dur: 48, sharp: 0.6 });
      let peak = 0;
      for (let i = 0; i < 400; i++) {
        net.tick();
        peak = Math.max(peak, Math.abs(net.outDryL), Math.abs(net.outDryR));
      }
      return peak;
    };
    const pluckTail = (sim: Sim) => {
      const topo = buildTopology(sim.graph, sim.agents);
      const net = new WaveguideNet();
      net.handle({ type: 'topology', topo });
      if (topo.wires[0]) net.injectPluck(topo.wires[0].id, 0.9);
      for (let i = 0; i < 2000; i++) net.tick();
      let sum = 0;
      for (let i = 0; i < 4000; i++) {
        net.tick();
        const v = net.outDryL + net.outWetL;
        sum += v * v;
      }
      return Math.sqrt(sum / 4000);
    };

    const pairSim = chain(2);
    const clumpSim = chain(32);
    const pairPeak = strikePeak(pairSim);
    const clumpPeak = strikePeak(clumpSim);
    expect(clumpPeak).toBeGreaterThan(pairPeak * 0.7);
    expect(pluckTail(clumpSim)).toBeLessThan(pluckTail(pairSim) * 0.75);
  });
});

describe('tissue resonator', () => {
  function skinWire(id: number, a: number, b: number): NetTopology['wires'][number] {
    return { id, length: 80, loss: 0.999, bend: 0.04, agentA: a, agentB: b, zA: 1, zB: 1 };
  }

  function body(id: number, tissue = false, tissueId?: number, tissueY = 0): AgentTopo {
    return {
      id,
      kind: 1,
      openPorts: tissue ? 0 : 1,
      impedance: 1,
      pan: 0,
      dist: 0.5,
      lod: tissue ? 2 : 0,
      tissue,
      tissueId,
      tissueY,
      modeHz: [180, 380, 620],
      modeT60: [0.4, 0.22, 0.12],
      modeGain: [0.7, 0.4, 0.22],
      coupling: 1,
    };
  }

  it('collapses interior strings so only the skin wire is a delay line', () => {
    const net = new WaveguideNet();
    net.handle({
      type: 'topology',
      topo: {
        wires: [skinWire(1, 1, 2)],
        agents: [
          body(1, false, 1, 0.4),
          body(2, false, 1, 0.4),
          body(3, true, 1),
          body(4, true, 1),
        ],
        tissues: [{ id: 1, n: 4, delay: 64, lod: 2 }],
      },
    });
    expect(net.liveCount).toBe(1);
    expect(net.tissues.filter((t) => t.active)).toHaveLength(1);
  });

  it('a knock on a tissue body is still audible', () => {
    const net = new WaveguideNet();
    net.handle({
      type: 'topology',
      topo: {
        wires: [],
        agents: [body(1, true, 1), body(2, true, 1)],
        tissues: [{ id: 1, n: 8, delay: 80, pan: 0, dist: 0.4, lod: 2 }],
      },
    });
    net.handle({ type: 'strike', agentId: 1, peak: 1.4, dur: 48, sharp: 0.5 });
    expect(peak(collect(net, 800))).toBeGreaterThan(0.01);
  });

  it('folding a ringing skin wire into tissue does not release it', () => {
    const skin: NetTopology = {
      wires: [skinWire(1, 1, 2), skinWire(2, 2, 3), skinWire(3, 3, 4)],
      agents: [body(1), body(2), body(3), body(4)],
    };
    const folded: NetTopology = {
      wires: [skinWire(1, 1, 2)],
      agents: [body(1, false, 1, 0.5), body(2, false, 1, 0.5), body(3, true, 1), body(4, true, 1)],
      tissues: [{ id: 1, n: 4, delay: 64, lod: 2 }],
    };
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: skin });
    net.injectPluck(2, 0.9);
    collect(net, 240);
    const ringing = net.wires.find((w) => w.wireId === 2);
    expect(ringing?.env ?? 0).toBeGreaterThan(0.05);
    net.handle({ type: 'topology', topo: folded });
    expect(net.wires.some((w) => w.releasing)).toBe(false);
    expect(net.wires.find((w) => w.wireId === 2)?.active ?? false).toBe(false);
    expect(net.liveCount).toBe(1);
    const tissue = net.tissues.find((t) => t.active);
    expect(tissue?.env ?? 0).toBeGreaterThan(0);
    net.injectPluck(1, 0.9);
    expect(peak(collect(net, 400))).toBeGreaterThan(0.02);
  });
});

describe('leftover delay lines', () => {
  it('a wire that keeps its id keeps its ringing across a topology rebuild', () => {
    const net = new WaveguideNet();
    net.handle({
      type: 'topology',
      topo: sampleTopo(7, 1, 2, 90),
    });
    net.injectPluck(7, 0.9);
    collect(net, 240);
    const before = net.wires.find((w) => w.wireId === 7)!;
    expect(before.env).toBeGreaterThan(0.05);
    const env = before.env;
    net.handle({
      type: 'topology',
      topo: {
        wires: [
          {
            id: 7,
            length: 90,
            loss: 0.999,
            bend: 0.05,
            agentA: 1,
            agentB: 5,
            zA: 1,
            zB: 1,
          },
        ],
        agents: [
          { id: 1, kind: 0, openPorts: 0, impedance: 1 },
          { id: 5, kind: 1, openPorts: 0, impedance: 1 },
        ],
      },
    });
    const after = net.wires.find((w) => w.wireId === 7)!;
    expect(after.active).toBe(true);
    expect(after.env).toBeGreaterThan(env * 0.5);
    expect(after.agentB).toBe(5);
    expect(net.wires.some((w) => w.releasing)).toBe(false);
  });

  it('an oscillator leftover still rings after the commute', () => {
    const sim = new Sim(480, 320);
    const params = defaultParams();
    params.spawnInterval = 0;
    loadPreset(sim, 'oscillator', params);
    const era = [...sim.agents.values()].find((a) => a.kind === 'era')!;
    const leftover = [...sim.graph.wires.values()].find((w) => w.a.id === era.id || w.b.id === era.id)!;
    const topo0 = buildTopology(sim.graph, sim.agents);
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: topo0 });
    net.injectPluck(leftover.id, 0.9);
    collect(net, 200);
    expect(net.wires.find((w) => w.wireId === leftover.id)?.env ?? 0).toBeGreaterThan(0.05);
    for (let f = 0; f < 300; f++) {
      const nRw = sim.rewrites.length;
      sim.step(1 / 60, params);
      if (nRw > 0 && sim.rewrites.length === 0) break;
    }
    expect(sim.graph.wires.has(leftover.id)).toBe(true);
    net.handle({ type: 'topology', topo: buildTopology(sim.graph, sim.agents) });
    const kept = net.wires.find((w) => w.wireId === leftover.id);
    expect(kept?.active).toBe(true);
    expect(kept?.env ?? 0).toBeGreaterThan(0.02);
    expect(net.wires.some((w) => w.releasing && w.wireId === leftover.id)).toBe(false);
  });
});
