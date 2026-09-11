import type { Sim } from './sim.ts';

/**
 * A hash of everything a frame can move, over the raw f64 bits, so a
 * refactor that shifts a single ulp shows up. Outside any test file so
 * several suites can import it.
 */

/** FNV-1a over the raw bits of every number fed in. Order matters and is part of what is checked. */
class BitHash {
  private h = 0x811c9dc5;
  private readonly buf = new DataView(new ArrayBuffer(8));

  num(v: number): void {
    // Normalize the two zeros and every NaN payload.
    this.buf.setFloat64(0, v === 0 ? 0 : Number.isNaN(v) ? NaN : v);
    for (let i = 0; i < 8; i++) {
      this.h ^= this.buf.getUint8(i);
      this.h = Math.imul(this.h, 0x01000193) >>> 0;
    }
  }

  text(s: string): void {
    for (let i = 0; i < s.length; i++) {
      this.h ^= s.charCodeAt(i);
      this.h = Math.imul(this.h, 0x01000193) >>> 0;
    }
  }

  get hex(): string {
    return (this.h >>> 0).toString(16).padStart(8, '0');
  }
}

/** Every piece of simulation state that a frame can move. */
export function stateHash(sim: Sim): string {
  const h = new BitHash();
  const ids = [...sim.agents.keys()].sort((a, b) => a - b);
  h.num(ids.length);
  for (const id of ids) {
    const a = sim.agents.get(id)!;
    h.num(id);
    h.text(a.kind);
    for (const v of [
      a.x, a.y, a.vx, a.vy, a.heading, a.omega,
      a.extra, a.request, a.recovering ? 1 : 0,
      a.scale, a.alpha, a.stun, a.drive, a.trail, a.mass,
      a.requestDecay, a.energyCap, a.debtCap, a.rescueTo, a.assort, a.transportRecoil,
    ]) {
      h.num(v);
    }
    h.num(a.locked ? 1 : 0);
    h.num(a.pinned ? 1 : 0);
  }
  const wires = [...sim.graph.wires.values()].sort((p, q) => p.id - q.id);
  h.num(wires.length);
  for (const w of wires) {
    h.num(w.id);
    h.num(w.a.id);
    h.text(w.a.slot);
    h.num(w.b.id);
    h.text(w.b.slot);
    for (const v of [w.rest, w.ropeLen, w.collapse, w.born, w.lastLen]) h.num(v);
    h.num(w.nodes.length);
    for (const n of w.nodes) {
      h.num(n.x);
      h.num(n.y);
      h.num(n.vx);
      h.num(n.vy);
    }
  }
  return h.hex;
}
