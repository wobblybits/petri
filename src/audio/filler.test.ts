import { describe, expect, it } from 'vitest';
import { AudioRing, RING_CHANNELS, ringBytes } from './ring.ts';
import { NetFiller } from './filler.ts';
import { WaveguideNet } from './waveguide.ts';
import type { AgentTopo, NetTopology, WireTopo } from './types.ts';

function agentSpec(id: number): AgentTopo {
  return {
    id,
    kind: (id % 3) as 0 | 1 | 2,
    openPorts: 0,
    impedance: 1,
    pan: 0,
    dist: 0.5,
    modeHz: [180, 380, 620],
    modeT60: [0.25, 0.14, 0.08],
    modeGain: [0.7, 0.4, 0.22],
    coupling: 1,
  };
}

function topo(): NetTopology {
  const wires: WireTopo[] = [
    { id: 1, length: 120, loss: 0.999, bend: 0.05, agentA: 1, agentB: 2, zA: 1, zB: 1, damp: 0.5 },
  ];
  return { wires, agents: [agentSpec(1), agentSpec(2)], height: 0.45 };
}

function ringOf(capacity: number): AudioRing {
  return new AudioRing(new SharedArrayBuffer(ringBytes(capacity)), capacity);
}

function excited(): WaveguideNet {
  const net = new WaveguideNet();
  net.handle({ type: 'topology', topo: topo() });
  net.injectPluck(1, 0.9);
  return net;
}

function drain(ring: AudioRing, n: number): Float32Array {
  const dst: Float32Array[] = [];
  for (let c = 0; c < RING_CHANNELS; c++) dst.push(new Float32Array(n));
  ring.read(dst, n);
  return dst[0];
}

describe('net filler', () => {
  it('fills the ring and then stops', () => {
    const ring = ringOf(512);
    const filler = new NetFiller(excited(), ring);
    expect(filler.fill()).toBe(512);
    expect(ring.available()).toBe(512);
    // Nothing has been consumed, so a second pass has nowhere to put anything.
    expect(filler.fill()).toBe(0);
  });

  it('honours the per-pass cap so a stalled producer cannot monopolise', () => {
    const ring = ringOf(4096);
    const filler = new NetFiller(excited(), ring);
    expect(filler.fill(256)).toBe(256);
    expect(ring.available()).toBe(256);
  });

  it('tops up as the consumer drains, without losing a sample', () => {
    const ring = ringOf(512);
    const filler = new NetFiller(excited(), ring);
    filler.fill();
    let consumed = 0;
    for (let i = 0; i < 40; i++) {
      drain(ring, 128);
      consumed += 128;
      filler.fill();
    }
    expect(consumed).toBe(5120);
    expect(ring.underruns()).toBe(0);
    expect(filler.frames).toBe(512 + consumed);
  });

  it('produces exactly what ticking the net directly produces', () => {
    // The ring is a transport, not a process. Any difference here would be a
    // sample being dropped, duplicated, or reordered on the way through.
    const direct = excited();
    const expected = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) {
      direct.tick(false);
      expected[i] = direct.outDryL;
    }

    const ring = ringOf(2048);
    const filler = new NetFiller(excited(), ring);
    filler.fill(1024);
    const got = drain(ring, 1024);
    for (let i = 0; i < 1024; i++) expect(got[i]).toBe(expected[i]);
  });

  it('reports how much audio is buffered, which is the latency', () => {
    const ring = ringOf(1024);
    const filler = new NetFiller(excited(), ring);
    filler.fill();
    expect(filler.bufferedMs(48000)).toBeCloseTo((1024 / 48000) * 1000, 6);
    drain(ring, 512);
    expect(filler.bufferedMs(48000)).toBeCloseTo((512 / 48000) * 1000, 6);
  });

  it('sheds air only once the ring is genuinely starving', () => {
    // Shedding is driven by ring fill, not by the audio clock — the point of
    // the buffer is that this decision can be made before anything is lost.
    const ring = ringOf(1024);
    const seen: boolean[] = [];
    const net = excited();
    const realTick = net.tick.bind(net);
    net.tick = (shed?: boolean) => {
      seen.push(shed === true);
      return realTick(shed);
    };
    const filler = new NetFiller(net, ring);
    // Ring empty: below the quarter mark, so shed.
    filler.fill(128);
    expect(seen[0]).toBe(true);
    // Ring full: comfortably above it, so do not.
    seen.length = 0;
    filler.fill();
    expect(seen[seen.length - 1]).toBe(false);
  });
});
