// The scent field: scatter, diffuse, decay, gather.
//
// Deliberately geometry-free. The host hands over world positions to deposit
// at and world positions to sample from, already computed; nothing in here
// knows what a port or a sensor is. Port placement exists in two places
// already (Sim and the wasm solver) and putting it in a third is how the
// deposit normalisation came to be 20 on one side and 10 on the other.
//
// Layout matches Fields exactly: FIELD_CELLS squared, four channels
// interleaved per cell, origin at home minus half the extent.

// 144 bytes. Every vec4f sits on a sixteen-byte boundary because WGSL demands
// it, which is why the scalars are grouped in fours rather than in the order
// anyone would write them down.
struct FieldParams {
  cols: u32,
  rows: u32,
  nDeposit: u32,
  nProbe: u32,
  originX: f32,
  originY: f32,
  extent: f32,
  fixedScale: f32,
  worldX: f32,
  worldY: f32,
  boundR: f32,
  growCh: f32,
  // Per channel, not scalar: energy is conserved where a signal fades, and a
  // reaction only patterns when its two species move at different speeds.
  mix: vec4f,
  mix2: vec4f,
  keep: vec4f,
  growR: f32,
  growCap: f32,
  growGamma: f32,
  growCat: f32,
  reactF: f32,
  reactKV: f32,
  reactDt: f32,
  pad0: f32,  // reserved; keeps reactU on its own sixteen-byte boundary
  reactU: f32,
  reactV: f32,
  nBlocks: u32,
  harvestCh: f32,
}

// A world position and what to add there, per channel.
struct Deposit {
  pos: vec2f,
  pad: vec2f,
  w: vec4f,
}

// Three world positions to sample, and the weights to combine them with.
struct Probe {
  left: vec2f,
  right: vec2f,
  own: vec2f,
  pad: vec2f,
  taste: vec4f,
}

@group(0) @binding(0) var<uniform> P: FieldParams;
@group(0) @binding(1) var<storage, read_write> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
// Fixed-point, because WGSL atomics are integer only and many bodies land in
// one cell. One atomic per channel, so the stride here is 4x the cell index.
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> deposits: array<Deposit>;
@group(0) @binding(5) var<storage, read> probes: array<Probe>;
@group(0) @binding(6) var<storage, read_write> samples: array<vec4f>;

/*
 * Harvest, which is grazing rather than sampling and so needs its own shape.
 *
 * An energy cell is a block of field cells — `EnergyGrid.span` squared, which
 * is sixteen at the shipped sizes — and the bodies standing in one block eat
 * from it in turn. Both of those orderings are load-bearing, so the host does
 * the binning (it already builds exactly this list, to group bodies by cell)
 * and hands over one work item per occupied block with its bodies already in
 * order. One thread per block, blocks independent, no atomics needed: two
 * bodies never contend because they are never in the same thread's block.
 */
struct HarvestBlock {
  fi: u32,
  fj: u32,
  wi: u32,
  wj: u32,
  first: u32,
  count: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(7) var<storage, read> hBlocks: array<HarvestBlock>;
// Room in each body's tank, in the order its block feeds them.
@group(0) @binding(8) var<storage, read> hRooms: array<f32>;
// What each body actually got, same indexing. Read back and credited on the
// next frame; see `Sim.gpuFieldStep`.
@group(0) @binding(9) var<storage, read_write> hGot: array<f32>;

// Matches `FLOW_EPS` in energy.ts: below this a take is not worth a cell walk.
const FLOW_EPS: f32 = 1e-9;

fn cellOf(p: vec2f) -> vec2f {
  let g = (p - vec2f(P.originX, P.originY)) / P.extent;
  return g * f32(P.cols);
}

@compute @workgroup_size(64)
fn clearAcc(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.cols * P.rows * 4u) { return; }
  atomicStore(&acc[i], 0);
}

// Bilinear splat, matching Fields.deposit: the four cells around the point,
// weighted by the fractional part.
fn cellOut(i: i32, j: i32) -> bool {
  if (P.boundR <= 0.0) { return false; }
  let cs = P.extent / f32(P.cols);
  let x = P.originX + (f32(i) + 0.5) * cs;
  let y = P.originY + (f32(j) + 0.5) * cs;
  let dx = x - P.worldX;
  let dy = y - P.worldY;
  return dx * dx + dy * dy > P.boundR * P.boundR;
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= P.nDeposit) { return; }
  let d = deposits[k];
  let g = cellOf(d.pos);
  let i0 = i32(floor(g.x));
  let j0 = i32(floor(g.y));
  let tx = g.x - f32(i0);
  let ty = g.y - f32(j0);
  for (var dj = 0; dj < 2; dj++) {
    for (var di = 0; di < 2; di++) {
      let i = i0 + di;
      let j = j0 + dj;
      if (i < 0 || j < 0 || i >= i32(P.cols) || j >= i32(P.rows)) { continue; }
      if (cellOut(i, j)) { continue; }
      var wx = 1.0 - tx;
      if (di == 1) { wx = tx; }
      var wy = 1.0 - ty;
      if (dj == 1) { wy = ty; }
      let w = wx * wy;
      let base = (u32(j) * P.cols + u32(i)) * 4u;
      for (var c = 0u; c < 4u; c++) {
        let v = d.w[c] * w;
        if (v != 0.0) {
          // Rounded, not truncated. i32() truncates toward zero, so every
          // contribution loses up to one unit in the same direction and the
          // bias accumulates with the number of bodies landing in a cell —
          // measured at nearly twice the tolerance over two hundred deposits.
          atomicAdd(&acc[base + c], i32(round(v * P.fixedScale)));
        }
      }
    }
  }
}

// Fold the accumulator into the field and clear it, one thread per cell.
@compute @workgroup_size(64)
fn applyAcc(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.cols * P.rows) { return; }
  let base = i * 4u;
  var add = vec4f(0.0);
  for (var c = 0u; c < 4u; c++) {
    add[c] = f32(atomicExchange(&acc[base + c], 0)) / P.fixedScale;
  }
  src[i] += add;
}

// One diffusion pass, src into dst. With no bound, a missing neighbour falls
// back to this cell (Neumann). With a disk bound, cells whose centres sit
// outside are zero and so are their contributions (Dirichlet).
fn diffuseAt(idx: u32, m: vec4f) {
  let i = i32(idx % P.cols);
  let j = i32(idx / P.cols);
  if (cellOut(i, j)) {
    dst[idx] = vec4f(0.0);
    return;
  }
  let here = src[idx];
  let dirichlet = P.boundR > 0.0;
  var a = here;
  var b = here;
  var c = here;
  var e = here;
  if (dirichlet) {
    a = vec4f(0.0);
    b = vec4f(0.0);
    c = vec4f(0.0);
    e = vec4f(0.0);
    if (i > 0 && !cellOut(i - 1, j)) { a = src[idx - 1u]; }
    if (i + 1 < i32(P.cols) && !cellOut(i + 1, j)) { b = src[idx + 1u]; }
    if (j > 0 && !cellOut(i, j - 1)) { c = src[idx - P.cols]; }
    if (j + 1 < i32(P.rows) && !cellOut(i, j + 1)) { e = src[idx + P.cols]; }
  } else {
    if (i > 0) { a = src[idx - 1u]; }
    if (i + 1 < i32(P.cols)) { b = src[idx + 1u]; }
    if (j > 0) { c = src[idx - P.cols]; }
    if (j + 1 < i32(P.rows)) { e = src[idx + P.cols]; }
  }
  dst[idx] = (vec4f(1.0) - m) * here + m * (a + b + c + e) * 0.25;
}

// Two entry points rather than one dispatched twice, because the host runs the
// pass at two different mixes in a frame and a uniform written once is read by
// every pass in the submission. Two entries beats a dynamic-offset binding for
// something this small.
@compute @workgroup_size(64)
fn diffuse(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.cols * P.rows) { return; }
  diffuseAt(gid.x, P.mix);
}

@compute @workgroup_size(64)
fn diffuse2(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.cols * P.rows) { return; }
  diffuseAt(gid.x, P.mix2);
}

@compute @workgroup_size(64)
fn decay(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.cols * P.rows) { return; }
  let ci = i32(i % P.cols);
  let cj = i32(i / P.cols);
  if (cellOut(ci, cj)) {
    src[i] = vec4f(0.0);
    return;
  }
  src[i] *= P.keep;
}

// Logistic regrowth on one channel, optionally catalysed by another.
//
// Growth is proportional to what is already in the cell, so zero is a fixed
// point: a cell grazed to the floor cannot recover on its own and has to be
// recolonised by diffusion. The catalyst scales the *rate* and is clamped at
// zero, so an inhibitor can stall regrowth but never run it backwards — ground
// destroyed by being smelled at would be a hole in the conservation the whole
// economy rests on. The result is clamped at capacity because one explicit
// step can overshoot it once the catalyst multiplies the rate.
@compute @workgroup_size(64)
fn grow(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.cols * P.rows) { return; }
  if (P.growR <= 0.0 || P.growCap <= 0.0) { return; }
  let ci = i32(i % P.cols);
  let cj = i32(i / P.cols);
  if (cellOut(ci, cj)) { return; }
  let ch = u32(P.growCh);
  var v = src[i];
  let e = v[ch];
  if (e <= 0.0 || e >= P.growCap) { return; }
  var r = P.growR;
  let cat = i32(P.growCat);
  if (cat >= 0 && u32(cat) != ch && P.growGamma != 0.0) {
    r = r * (1.0 + P.growGamma * v[u32(cat)]);
  }
  if (r <= 0.0) { return; }
  let next = e + r * e * (1.0 - e / P.growCap);
  v[ch] = min(next, P.growCap);
  src[i] = v;
}

// Gray-Scott between two channels: u + 2v -> 3v, fed and killed. `reactF` and
// `reactKV` arrive already multiplied by dt, matching the host.
@compute @workgroup_size(64)
fn react(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.cols * P.rows) { return; }
  if (P.reactDt <= 0.0) { return; }
  if (P.reactF <= 0.0 && P.reactKV <= 0.0) { return; }
  let uc = u32(P.reactU);
  let vc = u32(P.reactV);
  if (uc == vc) { return; }
  let ci = i32(i % P.cols);
  let cj = i32(i / P.cols);
  if (cellOut(ci, cj)) { return; }
  var cell = src[i];
  let u = cell[uc];
  let v = cell[vc];
  let uvv = u * v * v * P.reactDt;
  let nu = u - uvv + P.reactF * (1.0 - u);
  let nv = v + uvv - P.reactKV * v;
  cell[uc] = max(nu, 0.0);
  cell[vc] = max(nv, 0.0);
  src[i] = cell;
}

fn sampleAt(p: vec2f) -> vec4f {
  let g = cellOf(p);
  let i0 = i32(floor(g.x));
  let j0 = i32(floor(g.y));
  let tx = g.x - f32(i0);
  let ty = g.y - f32(j0);
  var out = vec4f(0.0);
  for (var dj = 0; dj < 2; dj++) {
    for (var di = 0; di < 2; di++) {
      let i = i0 + di;
      let j = j0 + dj;
      if (i < 0 || j < 0 || i >= i32(P.cols) || j >= i32(P.rows)) { continue; }
      var wx = 1.0 - tx;
      if (di == 1) { wx = tx; }
      var wy = 1.0 - ty;
      if (dj == 1) { wy = ty; }
      out += src[u32(j) * P.cols + u32(i)] * (wx * wy);
    }
  }
  return out;
}

// Two vec4f per body.
//
// The first is the three sensor readings steering wants, each collapsed
// against the body's taste weights — the reason the field can stay here and
// only a handful of scalars per body go back.
//
// The second is the raw four channels under the body, which is a different
// consumer with a different need: the genome's `Wx` sense columns multiply
// each channel by its own weight, so a taste-collapsed scalar cannot serve
// them. `Sim.updateState` used to get this from `Fields.sampleAll` on the CPU,
// out of an array the GPU never writes to — the quietest of the reasons
// `openFieldGpu` refuses. It costs nothing to add here: `own` is already
// sampled for the third scalar above.
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= P.nProbe) { return; }
  let pr = probes[k];
  let own = sampleAt(pr.own);
  samples[k * 2u] = vec4f(
    dot(pr.taste, sampleAt(pr.left)),
    dot(pr.taste, sampleAt(pr.right)),
    dot(pr.taste, own),
    0.0,
  );
  samples[k * 2u + 1u] = own;
}

/*
 * A frame of grazing, and a faithful port of `EnergyGrid.take` rather than the
 * obvious parallel one.
 *
 * The obvious one gives every body a proportional share of its block. That is
 * wrong here, and wrong in a way the CPU version says out loud: draining cells
 * one at a time leaves an uneven floor, and an uneven floor is what diffusion
 * has a gradient to work against. A flat share keeps the block uniform and
 * there is nothing left to flow. So this walks the block in raster order,
 * emptying each cell before moving to the next, exactly as `take` does.
 *
 * The scan restarts at the top of the block for each body rather than carrying
 * a cursor, which is also what `take` does — cells already emptied are skipped
 * by the `have <= 0` test, so a fresh scan finds the same place a cursor would.
 * `wi * wj` is sixteen.
 *
 * `if (got <= 0)` ends the block: the ground under it is gone and every body
 * still queued gets nothing. Their entries are zeroed rather than left, since
 * the buffer outlives a frame.
 */
@compute @workgroup_size(64)
fn harvest(@builtin(global_invocation_id) gid: vec3u) {
  let b = gid.x;
  if (b >= P.nBlocks) { return; }
  let blk = hBlocks[b];
  let ch = u32(P.harvestCh);
  for (var e = 0u; e < blk.count; e++) {
    var want = hRooms[blk.first + e];
    if (want <= FLOW_EPS) {
      hGot[blk.first + e] = 0.0;
      continue;
    }
    var got = 0.0;
    for (var y = 0u; y < blk.wj; y++) {
      if (want <= FLOW_EPS) { break; }
      for (var x = 0u; x < blk.wi; x++) {
        if (want <= FLOW_EPS) { break; }
        let idx = (blk.fj + y) * P.cols + (blk.fi + x);
        var v = src[idx];
        let have = v[ch];
        if (have <= 0.0) { continue; }
        var g = want;
        if (have < want) { g = have; }
        v[ch] = have - g;
        src[idx] = v;
        got += g;
        want -= g;
      }
    }
    hGot[blk.first + e] = got;
    if (got <= 0.0) {
      for (var r = e + 1u; r < blk.count; r++) { hGot[blk.first + r] = 0.0; }
      break;
    }
  }
}
