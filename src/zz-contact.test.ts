import { it } from 'vitest';
import { WaveguideNet } from './audio/waveguide.ts';
import { contactPeak, contactSeconds, getSampleRate, bowSpeed } from './audio/presets.ts';
import { Sim } from './sim.ts';
import { defaultParams } from './params.ts';
import { bodyTone } from './audio/body.ts';
const SR = getSampleRate();
function centroidHz(x: number[]) {
  // spectral centroid via a coarse filterbank
  let num = 0, den = 0;
  for (let f = 80; f < 8000; f *= 1.12) {
    let re = 0, im = 0;
    for (let n = 0; n < x.length; n++) {
      const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / (x.length - 1));
      re += x[n] * w * Math.cos(2 * Math.PI * f * n / SR);
      im -= x[n] * w * Math.sin(2 * Math.PI * f * n / SR);
    }
    const m = Math.hypot(re, im); num += m * f; den += m;
  }
  return den > 0 ? num / den : 0;
}
function bodyTopo(id: number, agent: any) {
  const t = bodyTone(agent);
  return { wires: [], agents: [{ id, kind: 0 as const, openPorts: 3, impedance: 1,
    pan: 0, modeHz: t.freq, modeT60: t.decay, modeGain: t.gain, coupling: 1 }] };
}
function pairTopo(a: any, b: any) {
  const ta = bodyTone(a);
  const tb = bodyTone(b);
  return { wires: [], agents: [
    { id: 1, kind: 0 as const, openPorts: 3, impedance: 1, pan: -0.4, modeHz: ta.freq, modeT60: ta.decay, modeGain: ta.gain, coupling: 1 },
    { id: 2, kind: 2 as const, openPorts: 3, impedance: 1, pan: 0.4, modeHz: tb.freq, modeT60: tb.decay, modeGain: tb.gain, coupling: 1 },
  ]};
}
it('contact physics', () => {
  const sim = new Sim(400, 300); const p = defaultParams();
  const era = sim.spawn('era', 40, 40, 0, p, true)!;
  const con = sim.spawn('con', 140, 40, 0, p, true)!;

  console.log('\n=== Hertzian contact: harder hits are SHORTER, hence brighter ===');
  console.log('  effMass  vN     tau(ms)  peakF   ->  attackCentroid  peak out');
  for (const [m, v] of [[1.4, 5], [1.4, 20], [1.4, 80], [1.4, 300], [4.0, 80]] as const) {
    const tau = contactSeconds(m, v);
    const dur = Math.round(tau * SR);
    const pk = contactPeak(m, v, tau);
    const net = new WaveguideNet();
    net.handle({ type: 'topology', topo: bodyTopo(1, era) as never });
    net.handle({ type: 'strike', agentId: 1, peak: pk, dur, sharp: 0.5 });
    const x: number[] = []; let out = 0;
    for (let i = 0; i < 8192; i++) { const s = net.tick(); x.push(s); out = Math.max(out, Math.abs(s)); }
    console.log(`  ${m.toFixed(1).padStart(6)}  ${String(v).padStart(4)}  ${(tau*1000).toFixed(2).padStart(6)}  ${pk.toFixed(3).padStart(6)}   ->  ${centroidHz(x.slice(0,600)).toFixed(0).padStart(8)}      ${out.toFixed(3)}`);
  }

  console.log('\n=== body modes: size sets pitch, what you see is what you hear ===');
  for (const [name, ag] of [['era (circle r=9)', era], ['con (triangle r=18)', con]] as const) {
    const t = bodyTone(ag);
    console.log(`  ${name.padEnd(21)} modes ${t.freq.map(f=>f.toFixed(0)+'Hz').join(' ')}  T60 ${t.decay.map(d=>d.toFixed(2)+'s').join(' ')}`);
  }

  console.log('\n=== an UNWIRED body now sounds when knocked (used to be silent) ===');
  const net = new WaveguideNet();
  net.handle({ type: 'topology', topo: bodyTopo(1, con) as never });
  net.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 60, sharp: 0.8 });
  let pk = 0; for (let i = 0; i < 24000; i++) pk = Math.max(pk, Math.abs(net.tick()));
  console.log(`  wires=0, knocked: peak = ${pk.toFixed(4)}`);
  for (let i = 0; i < SR * 8; i++) net.tick();
  let tail = 0; for (let i = 0; i < 4800; i++) tail = Math.max(tail, Math.abs(net.tick()));
  console.log(`  after 8 s of silence: tail = ${tail.toExponential(2)}`);

  console.log('\n=== bowing: pair friction, resonator is the clock ===');
  const r = new WaveguideNet();
  r.handle({ type: 'topology', topo: pairTopo(era, con) as never });
  for (const v of [4, 20, 90]) {
    r.handle({ type: 'contact', items: [{ agentA: 1, agentB: 2, load: Math.min(1, Math.max(0, (v-1.5)/55)), slide: bowSpeed(v) }] });
    const x: number[] = []; let out = 0;
    for (let i = 0; i < 12000; i++) { const s = r.tick(); x.push(s); out = Math.max(out, Math.abs(s)); }
    console.log(`  slide ${String(v).padStart(3)} px/s: bow=${bowSpeed(v).toExponential(2)}  peak=${out.toFixed(4)}  centroid=${centroidHz(x.slice(0,8192)).toFixed(0)}Hz`);
  }
  r.handle({ type: 'contact', items: [] });
  for (let i = 0; i < SR * 6; i++) r.tick();
  let stop = 0; for (let i = 0; i < 4800; i++) stop = Math.max(stop, Math.abs(r.tick()));
  console.log(`  sliding stops -> ${stop.toExponential(2)}`);

  console.log('\n=== resting contact: a knock on one body reaches the other ===');
  const c = new WaveguideNet();
  c.handle({ type: 'topology', topo: pairTopo(era, con) as never });
  c.handle({ type: 'strike', agentId: 1, peak: 1.5, dur: 60, sharp: 0.5 });
  c.handle({ type: 'contact', items: [{ agentA: 1, agentB: 2, load: 0.85, slide: 0 }] });
  let envB = 0;
  for (let i = 0; i < 8000; i++) c.tick();
  envB = c.agents[c.agentById.get(2)!].bodyEnv;
  console.log(`  B.bodyEnv after knock on A = ${envB.toExponential(2)}`);
});
