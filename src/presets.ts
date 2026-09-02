import { portWorld, stemWorld, type Agent, type PortSlot } from './agents.ts';
import type { Params } from './params.ts';
import type { Sim } from './sim.ts';

export type PresetName = 'soup' | 'commute' | 'annihilate-con' | 'annihilate-dup' | 'oscillator';

function plugAux(sim: Sim, agent: Agent, slot: PortSlot, params: Params): void {
  const root = stemWorld(agent, slot, sim.w, sim.h);
  const tip = portWorld(agent, slot, sim.w, sim.h);
  const dx = tip.x - root.x;
  const dy = tip.y - root.y;
  const era = sim.spawn(
    'era',
    tip.x + dx * 1.2,
    tip.y + dy * 1.2,
    Math.atan2(-dy, -dx),
    params,
    true,
  );
  if (era) sim.wire(agent.id, slot, era.id, 'p', params);
}

export function loadPreset(sim: Sim, name: PresetName, params: Params): void {
  sim.clear();
  const cx = sim.w * 0.5;
  const cy = sim.h * 0.5;
  if (name === 'soup') {
    const n = Math.min(params.soupCount, params.maxAgents);
    for (let i = 0; i < n; i++) {
      const roll = Math.random();
      const kind = roll < 0.34 ? 'era' : roll < 0.67 ? 'dup' : 'con';
      sim.spawn(
        kind,
        Math.random() * sim.w,
        Math.random() * sim.h,
        Math.random() * Math.PI * 2,
        params,
        true,
      );
    }
    return;
  }
  if (name === 'commute') {
    const c = sim.spawn('con', cx - 34, cy, 0, params, true);
    const d = sim.spawn('dup', cx + 34, cy, Math.PI, params, true);
    if (!c || !d) return;
    sim.wire(c.id, 'p', d.id, 'p', params);
    c.extra = 1;
    d.extra = 1;
    plugAux(sim, c, 'l', params);
    plugAux(sim, c, 'r', params);
    plugAux(sim, d, 'l', params);
    plugAux(sim, d, 'r', params);
    return;
  }
  if (name === 'annihilate-con' || name === 'annihilate-dup') {
    const kind = name === 'annihilate-con' ? 'con' : 'dup';
    const a = sim.spawn(kind, cx - 34, cy, 0, params, true);
    const b = sim.spawn(kind, cx + 34, cy, Math.PI, params, true);
    if (!a || !b) return;
    sim.wire(a.id, 'p', b.id, 'p', params);
    plugAux(sim, a, 'l', params);
    plugAux(sim, a, 'r', params);
    plugAux(sim, b, 'l', params);
    plugAux(sim, b, 'r', params);
    return;
  }

  const dup = sim.spawn('dup', cx - 40, cy, 0, params, true);
  const con = sim.spawn('con', cx + 40, cy, Math.PI, params, true);
  const era1 = sim.spawn('era', cx - 70, cy - 28, Math.PI, params, true);
  const era2 = sim.spawn('era', cx + 70, cy - 28, 0, params, true);
  if (!dup || !con || !era1 || !era2) return;
  sim.wire(dup.id, 'p', con.id, 'p', params);
  sim.wire(dup.id, 'l', era1.id, 'p', params);
  sim.wire(con.id, 'r', era2.id, 'p', params);
  sim.wire(dup.id, 'r', con.id, 'l', params);
  dup.extra = 1;
  con.extra = 1;
}
