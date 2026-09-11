// The scent field: scatter, diffuse, decay, gather. Geometry-free: the host
// hands over world positions to deposit at and to sample from; nothing here
// knows what a port or a sensor is. Layout matches Fields exactly:
// FIELD_CELLS squared, four channels interleaved per cell, origin at home
// minus half the extent.

// 160 bytes. Every vec4f must sit on a sixteen-byte boundary, which is why
// the scalars are grouped in fours.
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
  // Per channel: energy is conserved where a signal fades, and a reaction
  // only patterns when its two species move at different speeds.
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
  fillCh: f32,
  fillValue: f32,
  // Monod uptake. `uptakeCap` is `params.uptakeVmax * dt`; zero means
  // unmetered, the take-what-fits path. See `energy.ts:uptakeRate`.
  uptakeCap: f32,
  uptakeKs: f32,
  // Hill coefficient on uptake; 1 is plain Monod. See `UptakeKinetics.hillN`.
  hillN: f32,
  // Reserved; the uniform is a whole number of sixteen-byte blocks, packed by index.
  pad5: f32,
  pad6: f32,
  pad7: f32,
}

// A world position and what to add there, per channel. `conserve` picks
// which of the two adds `Fields.addAt` has: a scent deposit is a density and
// a share landing in a cell the disk rejects is not deposited; energy is a
// count and its share is spread over whichever cells will take it, since a
// point inside the disk can straddle cells whose centres are outside it.
struct Deposit {
  pos: vec2f,
  conserve: f32,
  pad: f32,
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
// Fixed-point, because WGSL atomics are integer only. One atomic per channel,
// so the stride here is 4x the cell index.
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> deposits: array<Deposit>;
@group(0) @binding(5) var<storage, read> probes: array<Probe>;
@group(0) @binding(6) var<storage, read_write> samples: array<vec4f>;

/*
 * Harvest work item: one per occupied energy block (`EnergyGrid.span` squared
 * field cells), with its bodies already in eating order. The host does the
 * binning; both orderings are load-bearing. One thread per block, so two
 * bodies never contend and no atomics are needed.
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
/*
 * Room in each body's gut (its tank, on the unmetered path), and what it got
 * out per species, in the same row. One buffer rather than two because WebGPU
 * guarantees only eight storage buffers per compute stage. In place is safe:
 * an entry belongs to one block and a block is one thread.
 */
@group(0) @binding(8) var<storage, read_write> hFlow: array<f32>;

// One entry's row in `hFlow`, from `energy.ts`: the room in that body's gut,
// the frame's whole uptake budget, then an affinity per species. `got`
// overwrites the affinities on the way back out; they are read first.
// Literals here, pinned to `energy.ts`'s exports by `field-kernel.test.ts`.
const HARVEST_ROOM: u32 = 0u;
const HARVEST_TOTAL: u32 = 1u;
const HARVEST_KS: u32 = 2u;
const HARVEST_GOT: u32 = 2u;
const HARVEST_STRIDE: u32 = 6u;

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

  // Renormalise a conserved channel over the cells that will take it; only
  // the rim ever has a total below one.
  var norm = 1.0;
  if (d.conserve > 0.5) {
    var legal = 0.0;
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
        legal += wx * wy;
      }
    }
    // Outside in every sense; `Fields.addAt` drops it here too.
    if (legal <= 0.0) { return; }
    norm = 1.0 / legal;
  }

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
      let w = wx * wy * norm;
      let base = (u32(j) * P.cols + u32(i)) * 4u;
      for (var c = 0u; c < 4u; c++) {
        let v = d.w[c] * w;
        if (v != 0.0) {
          // Rounded, not truncated: i32() truncates toward zero and the bias
          // accumulates with the number of bodies landing in a cell.
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
    // A missing neighbour reads as zero on the three signal channels (an
    // absorbing rim) and as this cell's own value on `CH.energy` (reflecting):
    // nothing may destroy the substance. Kept in step with `Fields.diffuse`
    // by `field-kernel.test.ts`.
    let miss = vec4f(0.0, 0.0, here.z, 0.0);
    a = miss;
    b = miss;
    c = miss;
    e = miss;
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

// Two entry points because the host runs the pass at two mixes in a frame
// and a uniform written once is read by every pass in the submission.
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

// Logistic regrowth on one channel, optionally catalysed by another. Zero is
// a fixed point: a cell grazed to the floor is recolonised by diffusion only.
// The catalysed rate is clamped at zero so an inhibitor can stall regrowth
// but never run it backwards; the result is clamped at capacity because one
// explicit step can overshoot.
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

// Two vec4f per body: the three sensor readings steering wants, each
// collapsed against the body's taste weights, then the raw four channels
// under the body for the genome's `Wx` sense columns.
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
 * Drain up to `want` of one species from a block, cell by cell in raster
 * order, and return what was taken. Cell by cell rather than proportionally:
 * grazing has to leave an uneven floor for diffusion to have a gradient to
 * work against. `EnergyGrid.takeFrom` is the same loop, and
 * `field-kernel.test.ts` holds the mirror that says so.
 */
fn drain(blk: HarvestBlock, ch: u32, want0: f32) -> f32 {
  var want = want0;
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
  return got;
}

@compute @workgroup_size(64)
fn harvest(@builtin(global_invocation_id) gid: vec3u) {
  let b = gid.x;
  if (b >= P.nBlocks) { return; }
  let blk = hBlocks[b];
  if (P.uptakeCap <= 0.0) {
    // The single-species path: take what fits, from the ground, to saturation.
    // Once a body gets nothing the block is empty; the queued entries are
    // zeroed rather than left, since the buffer outlives a frame.
    let ch = u32(P.harvestCh);
    for (var e = 0u; e < blk.count; e++) {
      let ro = (blk.first + e) * HARVEST_STRIDE;
      var want = hFlow[ro + HARVEST_ROOM];
      hFlow[ro + HARVEST_GOT] = 0.0;
      hFlow[ro + HARVEST_GOT + 1u] = 0.0;
      hFlow[ro + HARVEST_GOT + 2u] = 0.0;
      hFlow[ro + HARVEST_GOT + 3u] = 0.0;
      if (want <= FLOW_EPS) { continue; }
      let got = drain(blk, ch, want);
      hFlow[ro + HARVEST_GOT + ch] = got;
      if (got <= 0.0) {
        for (var r = e + 1u; r < blk.count; r++) {
          let z = (blk.first + r) * HARVEST_STRIDE;
          hFlow[z + HARVEST_GOT] = 0.0;
          hFlow[z + HARVEST_GOT + 1u] = 0.0;
          hFlow[z + HARVEST_GOT + 2u] = 0.0;
          hFlow[z + HARVEST_GOT + 3u] = 0.0;
        }
        break;
      }
    }
    return;
  }

  /*
   * The four uptake rows, drawn as one mouthful. Densities are read once for
   * the block, before anybody eats, so visit order cannot decide what a body
   * may draw. `total` is the frame's budget and each species may have at most
   * its share, `total * density / stock`; `left` is the gut room after the
   * species already taken. Species in index order, matching `runHarvestPlan`.
   */
  var density = vec4f(0.0);
  let cells = f32(blk.wi * blk.wj);
  if (cells > 0.0) {
    var sum = vec4f(0.0);
    for (var y = 0u; y < blk.wj; y++) {
      for (var x = 0u; x < blk.wi; x++) {
        sum += src[(blk.fj + y) * P.cols + (blk.fi + x)];
      }
    }
    density = sum / cells;
  }
  // A cell can sit below zero after a diffusion step; the host counts the
  // positive densities alone.
  let stock = dot(max(density, vec4f(0.0)), vec4f(1.0));
  for (var e = 0u; e < blk.count; e++) {
    let ro = (blk.first + e) * HARVEST_STRIDE;
    var left = hFlow[ro + HARVEST_ROOM];
    let total = hFlow[ro + HARVEST_TOTAL];
    // Read before the writeback: `got` shares the affinities' slots.
    let ks = vec4f(
      hFlow[ro + HARVEST_KS],
      hFlow[ro + HARVEST_KS + 1u],
      hFlow[ro + HARVEST_KS + 2u],
      hFlow[ro + HARVEST_KS + 3u],
    );
    for (var c = 0u; c < 4u; c++) {
      var got = 0.0;
      if (left > FLOW_EPS && total > 0.0 && density[c] > 0.0 && stock > 0.0) {
        // `total` stands in for `vmax` on every species. Hill at `n`; 1 is
        // plain Monod, and the host resolves a non-positive `hillN` to 1
        // before upload, so `!= 1.0` is the whole test. Kept in step with
        // `runHarvestPlan` by hand.
        var sN = density[c];
        var kN = ks[c];
        if (P.hillN != 1.0) {
          sN = pow(density[c], P.hillN);
          kN = pow(ks[c], P.hillN);
        }
        let rate = total * sN / (kN + sN);
        // This species' share of one budget is the ceiling however good the
        // body's transporter for it is.
        let share = total * density[c] / stock;
        var want = min(rate, share);
        if (left < want) { want = left; }
        if (want > FLOW_EPS) {
          got = drain(blk, c, want);
          left -= got;
        }
      }
      hFlow[ro + HARVEST_GOT + c] = got;
    }
  }
}


// Lay one channel down across the disk, the device-side twin of
// `Fields.fillDisk`: the value inside the disk and nothing written outside it.
@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.cols * P.rows) { return; }
  let x = i32(i % P.cols);
  let y = i32(i / P.cols);
  if (cellOut(x, y)) { return; }
  var v = src[i];
  v[u32(P.fillCh)] = P.fillValue;
  src[i] = v;
}
