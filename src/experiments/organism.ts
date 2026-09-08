import { Sim } from '../sim.ts';
import { defaultParams, type Params } from '../params.ts';
import { refreshReadsField, type Agent, type AgentKind, type PortSlot } from '../agents.ts';
import { EMIT, TASTE, HEAD_SCALE, P_BASE, L_BASE, F_BASE, CHEM_LEN } from '../chem-layout.ts';
import { CH } from '../fields.ts';
import { seededRandom } from './harness.ts';

/*
 * A bench for one hand-built organism, as opposed to `harness.ts`, which
 * studies a pond.
 *
 * The two ask different questions and so they measure different things. A
 * sweep over a soup wants population statistics — how many lines survive, what
 * the genome drift is — and every number in `Sample` is an average over
 * thousands of bodies. None of that says whether *this* net does *that*
 * thing. A worm is one organism with an inside: segments in an order, a
 * gradient along it, a stroke with a phase. Averaging it away is exactly the
 * wrong move.
 *
 * So a trial here is a single net, built to a spec, run with the rest of the
 * pond switched off, and sampled per segment. Nothing passes or fails; the
 * output is a trajectory and a per-segment timeline you can read a gait off.
 */

// --------------------------------------------------------------- body plans

export interface WormSpec {
  segments: number;
  /**
   * Kind per segment, rear first. A bare kind fills the chain; a short list
   * repeats its last entry, so `['con']` and `'con'` mean the same thing and
   * `['con', 'dup']` means "one Con at the tail, Dups the rest of the way".
   */
  kinds?: AgentKind | AgentKind[];
  /** Centre-to-centre spacing at build. Defaults to `wireMinRest`, which is where a wire settles. */
  spacing?: number;
  /**
   * Which aux port the segment ahead offers to the one behind.
   *
   * The spine is principal-to-aux and it has to be: a principal's axis points
   * along the body's heading and an aux's points against it, so `p(i) -> l(i+1)`
   * is the one pairing whose two port torques agree about where the other body
   * should be — ahead, in line, both facing the same way. Wire two aux ports
   * together instead and each end wants the other *behind* it, which no chain
   * of three can satisfy; it buckles into a rosette. (That is worth knowing
   * rather than avoiding — it is the obvious way to build a spring — but it is
   * not a worm.)
   */
  spine?: 'l' | 'r';
  x?: number;
  y?: number;
  /** Facing of every segment at build, and the axis displacement is measured along. */
  heading?: number;
  /**
   * Radians of random heading noise per segment at build.
   *
   * A worm built perfectly straight in a rotationally symmetric world is a
   * degenerate trial: nothing in `motorsOff` consumes `Math.random`, so every
   * seed produces the identical run and a sweep over seeds measures one thing
   * five times. This is the perturbation that makes a seed mean something —
   * and it is the honest one, because "is this gait stable against a body that
   * is not perfectly straight" is the question that separates swimming from
   * tumbling.
   */
  jitter?: number;
}

function kindAt(spec: WormSpec, i: number): AgentKind {
  const k = spec.kinds ?? 'con';
  if (typeof k === 'string') return k;
  if (k.length === 0) return 'con';
  return k[Math.min(i, k.length - 1)]!;
}

/**
 * A chain of segments, rear to front, wired principal-into-aux.
 *
 * Returns the ids in that order, so index 0 is the tail and the last index is
 * the head — the one segment left with a free principal, and so the only one
 * that could swim under its own power if swimming were switched on.
 *
 * Placed at exactly `spacing` apart, which should be `wireMinRest`: a wire
 * remembers the distance it latched at and shrinks from there toward the
 * minimum over `wireShrink` seconds, so building at the rest length starts the
 * worm settled instead of having it haul itself shorter for the first second.
 */
export function buildWorm(sim: Sim, params: Params, spec: WormSpec): number[] {
  const n = spec.segments;
  const gap = spec.spacing ?? params.wireMinRest;
  const heading = spec.heading ?? 0;
  const cx = spec.x ?? sim.w * 0.5;
  const cy = spec.y ?? sim.h * 0.5;
  const ux = Math.cos(heading);
  const uy = Math.sin(heading);
  // Centred on (cx, cy) so the worm's centre of mass starts where the world
  // was pinned, and the ground it is standing on is the ground it was given.
  const back = ((n - 1) * gap) / 2;
  const ids: number[] = [];
  for (let i = 0; i < n; i++) {
    const d = i * gap - back;
    const wobble = spec.jitter ? (Math.random() * 2 - 1) * spec.jitter : 0;
    const a = sim.spawn(kindAt(spec, i), cx + ux * d, cy + uy * d, heading + wobble, params, true);
    if (!a) throw new Error(`buildWorm: spawn failed at segment ${i}`);
    a.extra = a.energyCap;
    ids.push(a.id);
  }
  const spine: PortSlot = spec.spine ?? 'l';
  for (let i = 0; i + 1 < n; i++) {
    if (kindAt(spec, i + 1) === 'era') throw new Error('buildWorm: an Era has no aux port to receive a spine');
    sim.wire(ids[i]!, 'p', ids[i + 1]!, spine, params);
  }
  return ids;
}

// ----------------------------------------------------------------- genomes

/**
 * Overwrite the parts of a genome this bench hand-writes.
 *
 * Bases only. Every matrix stays where `seedChem` left it, which is zero bar
 * the two seeded pathways, so a dressed segment computes `h = phi(0) = 0` and
 * its heads are exactly the constants written here — no recurrent state, no
 * drift, nothing to disentangle from the mechanism under test. A controller
 * goes in later, on top, by writing the matrices instead.
 *
 * Head values are in natural units (px/s, gain) and converted through
 * `HEAD_SCALE`, because that is the conversion the sim reads them back with.
 */
export interface Dress {
  /** Emit bases per channel, in `CH` order. Normalised at read, so these are shares. */
  emit?: Partial<Record<keyof typeof CH, number>>;
  taste?: Partial<Record<keyof typeof CH, number>>;
  /** Transport: how much of the sender's kick the receiver cancels, and the kick itself. */
  thrust?: number;
  recoil?: number;
  cruise?: number;
  turn?: number;
  align?: number;
  sep?: number;
  extra?: number;
  energyCap?: number;
  debtCap?: number;
  rescueTo?: number;
  requestDecay?: number;
}

export function dress(a: Agent, d: Dress): void {
  const c = a.chem;
  if (d.emit) {
    for (let k = 0; k < 4; k++) c[EMIT + k] = 0;
    for (const [name, v] of Object.entries(d.emit)) c[EMIT + CH[name as keyof typeof CH]] = v!;
  }
  if (d.taste) {
    for (let k = 0; k < 4; k++) c[TASTE + k] = 0;
    for (const [name, v] of Object.entries(d.taste)) c[TASTE + CH[name as keyof typeof CH]] = v!;
  }
  if (d.thrust !== undefined) c[P_BASE] = d.thrust / HEAD_SCALE.thrust;
  if (d.recoil !== undefined) c[P_BASE + 1] = d.recoil / HEAD_SCALE.recoil;
  if (d.cruise !== undefined) c[L_BASE] = d.cruise / HEAD_SCALE.cruise;
  if (d.turn !== undefined) c[L_BASE + 1] = d.turn / HEAD_SCALE.turn;
  if (d.align !== undefined) c[F_BASE] = d.align / HEAD_SCALE.align;
  if (d.sep !== undefined) c[F_BASE + 1] = d.sep / HEAD_SCALE.sep;
  if (d.energyCap !== undefined) a.energyCap = d.energyCap;
  if (d.debtCap !== undefined) a.debtCap = d.debtCap;
  if (d.rescueTo !== undefined) a.rescueTo = d.rescueTo;
  if (d.requestDecay !== undefined) a.requestDecay = d.requestDecay;
  // Last, so a cap written above is the one a full tank is measured against.
  if (d.extra !== undefined) a.extra = d.extra;
  refreshReadsField(a);
}

/**
 * Everything off but transport.
 *
 * The point of the bench is to attribute motion, and the sim has four other
 * things that move a body. Each is switched off at its own dial rather than by
 * special-casing anything, so a run with these overrides is the same code path
 * the pond runs, with the other motors set to zero:
 *
 *   stepSpeed  the OU swim drive, the only self-propulsion a free principal has
 *   turnRate   the steering servo, which would turn the worm toward food
 *   swimNoise  the coloured kick on the drive; zero drive still gets kicked
 *   flockAlign/flockSep  pair forces between bodies, which a chain is full of
 *   wireBreathe  a global sinusoid on every rest length — a passive jiggle that
 *                would show up as a stroke in exactly the metrics being read
 *   declutter    the crowding push
 *   snapRadius   latching. A worm has a free principal at the nose and free aux
 *                ports along it, and principal-principal is the *most*
 *                preferred latch there is, so a worm that can latch bends round
 *                and eats itself. Zero returns from `Graph.snap` immediately.
 *   spawnInterval  immigration, which would drop strangers into the dish
 *
 * What is left that can move a body: the XPBD constraints (internal, and they
 * conserve the pair's momentum), the port torques (internal torque), contacts
 * (nothing to touch), the wall, and `applyTransportRecoil`. So displacement is
 * the pump, and the `recoil: 0` control run proves it.
 */
export function motorsOff(): Partial<Params> {
  return {
    stepSpeed: 0,
    turnRate: 0,
    swimNoise: 0,
    flockAlign: 0,
    flockSep: 0,
    wireBreathe: 0,
    declutter: 0,
    snapRadius: 0,
    spawnInterval: 0,
  };
}

// ----------------------------------------------------------------- sampling

export interface SegmentSample {
  extra: number;
  request: number;
  /** Heading relative to the worm's build axis, wrapped to (-pi, pi]. */
  bend: number;
  /** Centre-to-centre distance to the segment ahead; NaN at the head. */
  gap: number;
}

export interface OrganismSample {
  t: number;
  /** Centre of mass of the surviving segments, and its displacement from the start. */
  x: number;
  y: number;
  along: number;
  perp: number;
  /** Momentum of the whole worm along the build axis, which is what the pump adds to. */
  momentum: number;
  alive: number;
  wires: number;
  /** Cumulative, from `Sim.tally`. */
  moved: number;
  hops: number;
  pumpImpulse: number;
  segments: SegmentSample[];
}

export interface OrganismTrial {
  spec: OrganismSpec;
  samples: OrganismSample[];
  /** Path length of the centre of mass, integrated every frame rather than per sample. */
  pathLen: number;
  /**
   * Path-weighted mean cosine between the centre of mass's step and the worm's
   * own tail-to-head axis, over every frame it moved.
   *
   * The metric `along` cannot answer this. A worm that turns and then swims
   * beautifully in the new direction scores badly on the build axis, and a
   * worm tumbling end over end scores whatever its last lurch happened to give.
   * This asks the question that actually distinguishes a gait: +1 is swimming
   * nose-first, -1 is being pushed backwards, 0 is going wherever.
   */
  headward: number;
  wallMs: number;
}

export interface OrganismSpec {
  seconds: number;
  dt?: number;
  seed: number;
  params?: Partial<Params>;
  worm: WormSpec;
  sampleEvery?: number;
  /** Field cells a side. 256 is plenty for one worm and a fifth of 512's field cost. */
  fieldCells?: number;
  world?: { w: number; h: number };
  /** Hand-write genomes once, after the worm is built and before it is stepped. */
  dressWorm?: (sim: Sim, ids: number[], params: Params) => void;
  /** Called every frame before the step. For feeders and other declared puppetry. */
  drive?: (sim: Sim, ids: number[], t: number, params: Params) => void;
}

function wrapPi(a: number): number {
  let v = a;
  while (v > Math.PI) v -= 2 * Math.PI;
  while (v <= -Math.PI) v += 2 * Math.PI;
  return v;
}

function sampleOrganism(
  sim: Sim,
  ids: number[],
  t: number,
  origin: { x: number; y: number },
  axis: { x: number; y: number },
): OrganismSample {
  let mx = 0;
  let my = 0;
  let m = 0;
  let px = 0;
  let py = 0;
  let alive = 0;
  const segments: SegmentSample[] = [];
  for (let i = 0; i < ids.length; i++) {
    const a = sim.agents.get(ids[i]!);
    if (!a) {
      segments.push({ extra: NaN, request: NaN, bend: NaN, gap: NaN });
      continue;
    }
    alive++;
    mx += a.x * a.mass;
    my += a.y * a.mass;
    m += a.mass;
    px += a.vx * a.mass;
    py += a.vy * a.mass;
    const ahead = sim.agents.get(ids[i + 1]!);
    segments.push({
      extra: a.extra,
      request: a.request,
      bend: wrapPi(a.heading - Math.atan2(axis.y, axis.x)),
      gap: ahead ? Math.hypot(ahead.x - a.x, ahead.y - a.y) : NaN,
    });
  }
  const cx = m > 0 ? mx / m : origin.x;
  const cy = m > 0 ? my / m : origin.y;
  const dx = cx - origin.x;
  const dy = cy - origin.y;
  let wires = 0;
  const set = new Set(ids);
  for (const w of sim.graph.wires.values()) {
    if (set.has(w.a.id) && set.has(w.b.id)) wires++;
  }
  return {
    t,
    x: cx,
    y: cy,
    along: dx * axis.x + dy * axis.y,
    perp: -dx * axis.y + dy * axis.x,
    momentum: px * axis.x + py * axis.y,
    alive,
    wires,
    moved: sim.tally.moved,
    hops: sim.tally.hops,
    pumpImpulse: sim.tally.pumpImpulse,
    segments,
  };
}

/** Build one organism, run it, and sample it on a schedule. */
export function runOrganism(spec: OrganismSpec): OrganismTrial {
  const realRandom = Math.random;
  Math.random = seededRandom(spec.seed);
  const t0 = performance.now();
  try {
    const params: Params = { ...defaultParams(), ...(spec.params ?? {}) };
    const world = spec.world ?? { w: 4000, h: 4000 };
    const sim = new Sim(world.w, world.h, spec.fieldCells ?? 256);
    const cx = world.w * 0.5;
    const cy = world.h * 0.5;
    // Before the worm, so the ground it is standing on exists when it is built.
    sim.pinWorld(cx, cy, params);
    const ids = buildWorm(sim, params, { x: cx, y: cy, ...spec.worm });
    spec.dressWorm?.(sim, ids, params);

    const heading = spec.worm.heading ?? 0;
    const axis = { x: Math.cos(heading), y: Math.sin(heading) };
    const first = sampleOrganism(sim, ids, 0, { x: cx, y: cy }, axis);
    const origin = { x: first.x, y: first.y };

    const dt = spec.dt ?? 1 / 60;
    const every = spec.sampleEvery ?? 2;
    const samples: OrganismSample[] = [];
    let nextSample = 0;
    let t = 0;
    let pathLen = 0;
    let headwardSum = 0;
    let lastX = origin.x;
    let lastY = origin.y;
    while (t < spec.seconds - 1e-9) {
      if (t >= nextSample - 1e-9) {
        samples.push(sampleOrganism(sim, ids, t, origin, axis));
        nextSample += every;
      }
      spec.drive?.(sim, ids, t, params);
      sim.step(dt, params);
      t += dt;
      // Every frame, not every sample: a stroke that goes back and forth
      // between samples is exactly what a straightness number has to see.
      const now = sampleCom(sim, ids);
      if (now) {
        const sx = now.x - lastX;
        const sy = now.y - lastY;
        const step = Math.hypot(sx, sy);
        pathLen += step;
        const body = bodyAxis(sim, ids);
        if (body && step > 1e-9) headwardSum += (sx * body.x + sy * body.y) / step * step;
        lastX = now.x;
        lastY = now.y;
      }
    }
    samples.push(sampleOrganism(sim, ids, t, origin, axis));
    return {
      spec,
      samples,
      pathLen,
      headward: pathLen > 1e-9 ? headwardSum / pathLen : 0,
      wallMs: performance.now() - t0,
    };
  } finally {
    Math.random = realRandom;
  }
}

/** Unit vector from the rearmost living segment to the foremost. The worm's own forward. */
function bodyAxis(sim: Sim, ids: number[]): { x: number; y: number } | null {
  let tail: Agent | undefined;
  let head: Agent | undefined;
  for (const id of ids) {
    const a = sim.agents.get(id);
    if (!a) continue;
    if (!tail) tail = a;
    head = a;
  }
  if (!tail || !head || tail === head) return null;
  const dx = head.x - tail.x;
  const dy = head.y - tail.y;
  const len = Math.hypot(dx, dy);
  return len > 1e-6 ? { x: dx / len, y: dy / len } : null;
}

function sampleCom(sim: Sim, ids: number[]): { x: number; y: number } | null {
  let mx = 0;
  let my = 0;
  let m = 0;
  for (const id of ids) {
    const a = sim.agents.get(id);
    if (!a) continue;
    mx += a.x * a.mass;
    my += a.y * a.mass;
    m += a.mass;
  }
  return m > 0 ? { x: mx / m, y: my / m } : null;
}

// ------------------------------------------------------------------ metrics

export interface Gait {
  /** Net displacement along the build axis, and mean speed over the run. */
  along: number;
  perp: number;
  speed: number;
  /** |net displacement| / path length. 1 is a straight line, 0 is a stroke going nowhere. */
  straightness: number;
  /** Energy that crossed a wire, transfers, and the momentum those transfers injected. */
  moved: number;
  hops: number;
  pumpImpulse: number;
  /**
   * Energy moved per unit distance travelled: this motor's cost of transport,
   * in the only currency the pond has. The number to compare gaits on.
   */
  costOfTransport: number;
  /** Impulse that ended up as displacement rather than cancelling. */
  pumpEfficiency: number;
  /** Path-weighted mean cosine between travel and the body's own forward. See `OrganismTrial`. */
  headward: number;
  /** Peak-to-peak of each segment's bend and of each gap, averaged over segments. */
  bendSwing: number;
  gapSwing: number;
  intact: boolean;
}

export function gaitOf(trial: OrganismTrial): Gait {
  const last = trial.samples[trial.samples.length - 1]!;
  const first = trial.samples[0]!;
  const seconds = last.t - first.t || 1;
  const dist = Math.hypot(last.along, last.perp);
  const n = first.segments.length;
  let bendSwing = 0;
  let gapSwing = 0;
  let bendCount = 0;
  let gapCount = 0;
  for (let i = 0; i < n; i++) {
    let bLo = Infinity;
    let bHi = -Infinity;
    let gLo = Infinity;
    let gHi = -Infinity;
    for (const s of trial.samples) {
      const seg = s.segments[i];
      if (!seg) continue;
      if (Number.isFinite(seg.bend)) {
        bLo = Math.min(bLo, seg.bend);
        bHi = Math.max(bHi, seg.bend);
      }
      if (Number.isFinite(seg.gap)) {
        gLo = Math.min(gLo, seg.gap);
        gHi = Math.max(gHi, seg.gap);
      }
    }
    if (bHi >= bLo) {
      bendSwing += bHi - bLo;
      bendCount++;
    }
    if (gHi >= gLo) {
      gapSwing += gHi - gLo;
      gapCount++;
    }
  }
  const moved = last.moved - first.moved;
  return {
    along: last.along,
    perp: last.perp,
    speed: dist / seconds,
    straightness: trial.pathLen > 1e-9 ? dist / trial.pathLen : 0,
    moved,
    hops: last.hops - first.hops,
    pumpImpulse: last.pumpImpulse - first.pumpImpulse,
    costOfTransport: dist > 1e-6 ? moved / dist : Infinity,
    pumpEfficiency: last.pumpImpulse > 1e-9 ? dist / last.pumpImpulse : 0,
    headward: trial.headward,
    bendSwing: bendCount > 0 ? bendSwing / bendCount : 0,
    gapSwing: gapCount > 0 ? gapSwing / gapCount : 0,
    intact: last.alive === n && last.wires === first.wires,
  };
}

/** Fixed-width table of gaits, one row a condition. */
export function gaitTable(rows: { label: string; gait: Gait }[]): string {
  const cols: [string, (g: Gait) => string][] = [
    ['along', (g) => g.along.toFixed(1)],
    ['perp', (g) => g.perp.toFixed(1)],
    ['px/s', (g) => g.speed.toFixed(2)],
    ['straight', (g) => g.straightness.toFixed(2)],
    ['headward', (g) => g.headward.toFixed(2)],
    ['moved', (g) => g.moved.toFixed(2)],
    ['hops', (g) => String(g.hops)],
    ['impulse', (g) => g.pumpImpulse.toFixed(1)],
    ['cot', (g) => (Number.isFinite(g.costOfTransport) ? g.costOfTransport.toFixed(4) : '-')],
    ['bend', (g) => g.bendSwing.toFixed(2)],
    ['gap', (g) => g.gapSwing.toFixed(1)],
    ['intact', (g) => (g.intact ? 'y' : 'n')],
  ];
  const width = Math.max(6, ...rows.map((r) => r.label.length));
  const head = ['condition'.padEnd(width), ...cols.map(([h]) => h.padStart(9))].join(' ');
  const body = rows.map(
    (r) => [r.label.padEnd(width), ...cols.map(([, f]) => f(r.gait).padStart(9))].join(' '),
  );
  return [head, ...body].join('\n');
}

export { CHEM_LEN };
