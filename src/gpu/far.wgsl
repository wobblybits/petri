// FAR body step: integrate, translation-only disc XPBD, stem span. Matches
// src/gpu/far-kernel.ts. The packed radius is the glyph-area disc and span
// compliance is per wire, so this tier settles where NEAR settles.

struct SimParams {
  n: u32,
  nWires: u32,
  cols: u32,
  rows: u32,
  h: f32,
  slop: f32,
  contactComp: f32,
  spanComp: f32,
  gridMinX: f32,
  gridMinY: f32,
  invCell: f32,
  pad0: f32,
  worldX: f32,
  worldY: f32,
  boundR: f32,
  pad1: f32,
}

struct Particle {
  x: f32,
  y: f32,
  vx: f32,
  vy: f32,
  heading: f32,
  omega: f32,
  invMass: f32,
  radius: f32,
  locked: f32,
  prevX: f32,
  prevY: f32,
  prevHeading: f32,
}

struct Wire {
  a: f32,
  b: f32,
  rest: f32,
  // Per-wire softness: params.springK scale times the birth-slack ramp.
  soft: f32,
  oax: f32,
  oay: f32,
  obx: f32,
  oby: f32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(2) var<storage, read_write> delta: array<vec2f>;
@group(0) @binding(3) var<storage, read> wires: array<Wire>;
// Uniform-grid broadphase, rebuilt every substep because bodies move.
@group(0) @binding(4) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cellBodies: array<u32>;
// Wire indices touching each body, built once per step.
@group(0) @binding(6) var<storage, read_write> nei: array<u32>;
@group(0) @binding(7) var<storage, read_write> neiCount: array<atomic<u32>>;

/** Bodies recorded per cell. Beyond this the cell drops contacts. */
const CELL_CAP: u32 = 64u;
/** An agent has three ports, so three wires. The fourth slot is slack. */
const NEI_CAP: u32 = 4u;

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;
// Exactly the largest finite f32, in hex because the decimal form rounds up.
const F32_MAX: f32 = 0x1.fffffep+127;

// `Number.isFinite`. A magnitude test rather than `x - x == 0`, which a
// compiler may fold to `true`; NaN fails every comparison.
fn isFinite(x: f32) -> bool {
  return abs(x) <= F32_MAX;
}

// Grid column/row for a point, clamped: monotone, so an off-grid body still collides at the edge.
fn cellXY(x: f32, y: f32) -> vec2i {
  if (!isFinite(x) || !isFinite(y)) { return vec2i(0, 0); }
  let cx = clamp(i32(floor((x - params.gridMinX) * params.invCell)), 0, i32(params.cols) - 1);
  let cy = clamp(i32(floor((y - params.gridMinY) * params.invCell)), 0, i32(params.rows) - 1);
  return vec2i(cx, cy);
}

@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= params.cols * params.rows) { return; }
  atomicStore(&cellCount[c], 0u);
}

@compute @workgroup_size(64)
fn binBodies(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let p = parts[i];
  let cc = cellXY(p.x, p.y);
  let c = u32(cc.y) * params.cols + u32(cc.x);
  let slot = atomicAdd(&cellCount[c], 1u);
  if (slot < CELL_CAP) { cellBodies[c * CELL_CAP + slot] = i; }
}

@compute @workgroup_size(64)
fn clearNei(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  atomicStore(&neiCount[i], 0u);
}

@compute @workgroup_size(64)
fn buildNei(@builtin(global_invocation_id) gid: vec3u) {
  let w = gid.x;
  if (w >= params.nWires) { return; }
  let wire = wires[w];
  // Exactly `fillDiscNeighbours`'s gate: a wire with a bad rest still marks the pair wired.
  if (!isFinite(wire.a) || !isFinite(wire.b)) { return; }
  let ia = i32(wire.a);
  let ib = i32(wire.b);
  let sn = i32(params.n);
  if (ia < 0 || ib < 0 || ia >= sn || ib >= sn || ia == ib) { return; }
  let sa = atomicAdd(&neiCount[u32(ia)], 1u);
  if (sa < NEI_CAP) { nei[u32(ia) * NEI_CAP + sa] = w; }
  let sb = atomicAdd(&neiCount[u32(ib)], 1u);
  if (sb < NEI_CAP) { nei[u32(ib) * NEI_CAP + sb] = w; }
}

fn wrapAngle(a: f32) -> f32 {
  if (a != a) { return 0.0; }
  if (a >= -PI && a < PI) { return a; }
  var r = a % TAU;
  if (r >= PI) { r -= TAU; }
  if (r < -PI) { r += TAU; }
  return r;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  var p = parts[i];
  p.prevX = p.x;
  p.prevY = p.y;
  p.prevHeading = p.heading;
  if (p.locked < 0.5) {
    p.x += p.vx * params.h;
    p.y += p.vy * params.h;
    p.heading = wrapAngle(p.heading + p.omega * params.h);
  }
  parts[i] = p;
  delta[i] = vec2f(0.0, 0.0);
}

@compute @workgroup_size(64)
fn disc(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let pi = parts[i];
  if (pi.locked >= 0.5 || pi.invMass <= 0.0) { return; }
  var push = vec2f(0.0, 0.0);
  let alpha = params.contactComp / max(1e-12, params.h * params.h);
  // Span owns wired gaps; the wired partners come off the prebuilt table.
  var nb = array<u32, 4>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  let ncnt = min(atomicLoad(&neiCount[i]), NEI_CAP);
  for (var k = 0u; k < ncnt; k++) {
    let wire = wires[nei[i * NEI_CAP + k]];
    let ia = u32(i32(wire.a));
    let ib = u32(i32(wire.b));
    nb[k] = select(ia, ib, ia == i);
  }
  // Cell size is at least the widest contact gap, so a touch is at most one cell away.
  let cc = cellXY(pi.x, pi.y);
  let cols = i32(params.cols);
  let rows = i32(params.rows);
  for (var dy = -1; dy <= 1; dy++) {
    let ny = cc.y + dy;
    if (ny < 0 || ny >= rows) { continue; }
    for (var dx = -1; dx <= 1; dx++) {
      let nx = cc.x + dx;
      if (nx < 0 || nx >= cols) { continue; }
      let c = u32(ny) * params.cols + u32(nx);
      let cnt = min(atomicLoad(&cellCount[c]), CELL_CAP);
      for (var s = 0u; s < cnt; s++) {
        let j = cellBodies[c * CELL_CAP + s];
        if (j == i) { continue; }
        let pj = parts[j];
        var d = vec2f(pj.x - pi.x, pj.y - pi.y);
        var dist = length(d);
        let keep = pi.radius + pj.radius;
        if (dist >= keep) { continue; }
        let wired = j == nb[0] || j == nb[1] || j == nb[2] || j == nb[3];
        if (dist < 1e-6) {
          d = vec2f(select(-1.0, 1.0, i < j), 0.0);
          dist = 1.0;
        } else if (wired) {
          continue;
        }
        let depth = keep - dist - params.slop;
        if (depth <= 0.0) { continue; }
        let nrm = d / dist;
        let denom = pi.invMass + pj.invMass + alpha;
        if (denom < 1e-12) { continue; }
        let lam = depth / denom;
        push -= nrm * (lam * pi.invMass);
      }
    }
  }
  delta[i] = push;
}

@compute @workgroup_size(64)
fn span(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let pi = parts[i];
  if (pi.locked >= 0.5 || pi.invMass <= 0.0) { return; }
  var push = vec2f(0.0, 0.0);
  var count = 0u;
  let si = i32(i);
  // `buildNei` has gated endpoints; a negative or non-finite rest makes C meaningless.
  let ncnt = min(atomicLoad(&neiCount[i]), NEI_CAP);
  for (var k = 0u; k < ncnt; k++) {
    let wire = wires[nei[i * NEI_CAP + k]];
    if (!(wire.rest >= 0.0) || !isFinite(wire.rest)) { continue; }
    let ia = i32(wire.a);
    let ib = i32(wire.b);
    var j: u32;
    var oix: f32;
    var oiy: f32;
    var ojx: f32;
    var ojy: f32;
    if (ia == si) {
      j = u32(ib);
      oix = wire.oax; oiy = wire.oay;
      ojx = wire.obx; ojy = wire.oby;
    } else if (ib == si) {
      j = u32(ia);
      oix = wire.obx; oiy = wire.oby;
      ojx = wire.oax; ojy = wire.oay;
    } else { continue; }
    let pj = parts[j];
    var d = vec2f(pj.x + ojx - (pi.x + oix), pj.y + ojy - (pi.y + oiy));
    var dist = length(d);
    // A body far enough out overflows the length to +inf, and inf/inf is NaN.
    if (!isFinite(dist)) { continue; }
    if (dist < 1e-6) {
      d = vec2f(select(-1.0, 1.0, i < j), 0.0);
      dist = 1.0;
    }
    let nrm = d / dist;
    let C = dist - wire.rest;
    var soft = wire.soft;
    if (!(soft > 0.0)) { soft = 1.0; }
    let alpha = params.spanComp * soft / max(1e-12, params.h * params.h);
    let denom = pi.invMass + pj.invMass + alpha;
    if (denom < 1e-12) { continue; }
    let lam = -C / denom;
    // n from i toward j. A (i) is pushed with -n * lambda.
    push += -nrm * (lam * pi.invMass);
    count = count + 1u;
  }
  // Every wire is solved against the same start pose, so k wires summed
  // would overshoot k-fold. Averaging (Jacobi) cannot move where the net
  // settles, only how fast it walks there; hence FAR_SUBSTEPS passes.
  if (count > 1u) { push /= f32(count); }
  delta[i] = push;
}

@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  var p = parts[i];
  if (p.locked < 0.5) {
    p.x += delta[i].x;
    p.y += delta[i].y;
  }
  parts[i] = p;
  delta[i] = vec2f(0.0, 0.0);
}

@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  var p = parts[i];
  let invH = 1.0 / params.h;
  if (p.locked >= 0.5) {
    p.vx = 0.0;
    p.vy = 0.0;
    p.omega = 0.0;
  } else {
    p.vx = (p.x - p.prevX) * invH;
    p.vy = (p.y - p.prevY) * invH;
    p.omega = wrapAngle(p.heading - p.prevHeading) * invH;
  }
  parts[i] = p;
}

@compute @workgroup_size(64)
fn wall(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  if (params.boundR <= 0.0) { return; }
  var p = parts[i];
  if (p.locked >= 0.5) { return; }
  var maxr = params.boundR - p.radius;
  if (maxr < 0.0) { maxr = 0.0; }
  let dx = p.x - params.worldX;
  let dy = p.y - params.worldY;
  let dist = sqrt(dx * dx + dy * dy);
  if (dist <= maxr || dist < 1e-6) { return; }
  let inv = 1.0 / dist;
  let ux = dx * inv;
  let uy = dy * inv;
  p.x = params.worldX + ux * maxr;
  p.y = params.worldY + uy * maxr;
  let vn = p.vx * ux + p.vy * uy;
  if (vn > 0.0) {
    p.vx -= vn * ux;
    p.vy -= vn * uy;
  }
  parts[i] = p;
}
