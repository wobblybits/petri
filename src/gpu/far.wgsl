// FAR body step: integrate, translation-only disc XPBD, stem span.
// Matches src/gpu/far-kernel.ts. No SAT, no rope nodes, no torque.
// The packed radius is the glyph-area disc, not the SAT bound, and span
// compliance is per wire, so this tier settles where NEAR settles.

struct SimParams {
  n: u32,
  nWires: u32,
  pad0: u32,
  pad1: u32,
  h: f32,
  slop: f32,
  contactComp: f32,
  spanComp: f32,
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

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;
// Exactly the largest finite f32. Spelled in hex because the decimal form
// rounds up and is rejected as unrepresentable.
const F32_MAX: f32 = 0x1.fffffep+127;

/**
 * `Number.isFinite`, which the twin leans on in several places and WGSL has no
 * builtin for. Written as a magnitude test rather than the usual `x - x == 0`
 * because that identity is exactly what a shader compiler is free to fold to
 * `true`. A comparison cannot be folded, and NaN fails every comparison, so
 * this rejects NaN and both infinities.
 */
fn isFinite(x: f32) -> bool {
  return abs(x) <= F32_MAX;
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
  // Span owns wired gaps. Bound discs are fatter than SAT, so colliding a
  // neighbour the chord is holding fights the rest length.
  var n0: u32 = 0xffffffffu;
  var n1: u32 = 0xffffffffu;
  var n2: u32 = 0xffffffffu;
  var k: u32 = 0u;
  let si = i32(i);
  let sn = i32(params.n);
  for (var w = 0u; w < params.nWires; w++) {
    let wire = wires[w];
    // Same validity gate `fillDiscNeighbours` applies. Guarding span but not
    // this would be worse than guarding neither: a malformed wire would be
    // skipped by the chord and still mark a body wired here, which is how a
    // pair silently loses its contact.
    if (!isFinite(wire.a) || !isFinite(wire.b)) { continue; }
    let ia = i32(wire.a);
    let ib = i32(wire.b);
    if (ia < 0 || ib < 0 || ia >= sn || ib >= sn || ia == ib) { continue; }
    var other: u32 = 0xffffffffu;
    if (ia == si) { other = u32(ib); }
    else if (ib == si) { other = u32(ia); }
    else { continue; }
    if (k == 0u) { n0 = other; }
    else if (k == 1u) { n1 = other; }
    else { n2 = other; }
    k = k + 1u;
  }
  for (var j = 0u; j < params.n; j++) {
    if (j == i) { continue; }
    let pj = parts[j];
    var d = vec2f(pj.x - pi.x, pj.y - pi.y);
    var dist = length(d);
    let keep = pi.radius + pj.radius;
    if (dist >= keep) { continue; }
    let wired = j == n0 || j == n1 || j == n2;
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
  let sn = i32(params.n);
  for (var w = 0u; w < params.nWires; w++) {
    let wire = wires[w];
    // The twin's guards, which this had been missing. A rest that is negative
    // or non-finite makes C meaningless, and an endpoint outside the pack
    // reads a body that is not there -- indeterminate here rather than merely
    // wrong, since a float that far out of range converts to whatever the
    // hardware does with it.
    if (!(wire.rest >= 0.0) || !isFinite(wire.rest)) { continue; }
    if (!isFinite(wire.a) || !isFinite(wire.b)) { continue; }
    let ia = i32(wire.a);
    let ib = i32(wire.b);
    if (ia < 0 || ib < 0 || ia >= sn || ib >= sn || ia == ib) { continue; }
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
    // Was `dist != dist`, which is only the NaN half of the twin's
    // `Number.isFinite`. A body far enough out overflows the length to +inf,
    // and inf/inf is a NaN normal that poisons the whole sum.
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
  // The twin solves wires in sequence, so each one sees the last one's
  // result. One thread per body cannot: every wire here is solved against
  // the same start pose, so k wires on a body each correct the whole error
  // and the sum overshoots k-fold. Three wires between one pair -- legal,
  // Con and Dup have three ports each -- diverged outright.
  //
  // Averaging is the standard Jacobi fix and cannot move where the net
  // settles: at rest every correction is zero, and zero averages to zero.
  // It only walks there in smaller steps, which is why this tier gets
  // FAR_SUBSTEPS passes at it.
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
