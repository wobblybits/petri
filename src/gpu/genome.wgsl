// The genome: one small recurrent network per body, over the wire graph.
//
//     x  = [ sense(4) , FULL , BOUND , DEMAND ]                    in R^7
//     h <- phi( Wx.x + Wh.h + Wn.mean(h of wired neighbours) + b )  in R^4
//     emit  = normalise(relu( E.h + e0 ))                          in R^4
//     taste = T.h + t0                                             in R^4
//     [cruise,turn] = L.h + l0, and the same shape for F and P
//
// A line-for-line port of `Sim.updateState`, and it has to stay one: the CPU
// version is the reference, `genome-kernel.test.ts` mirrors this against it,
// and the layout constants below are `chem-layout.ts` transcribed. Changing
// the genome's shape means changing all three.
//
// Everything is in *list order*, not slot order — the host packs `hPrev` and
// the adjacency the same way it builds them, so nothing here has to know what
// a slot is except to find the genome, which is why `slot` is an input.

const STATE_DIMS: u32 = 4u;
const IN_DIMS: u32 = 7u;

// chem-layout.ts, derived the same way and in the same order.
const EMIT: u32 = 0u;
const TASTE: u32 = 4u;
const E_OUT: u32 = 8u;
const T_OUT: u32 = 24u;
const W_IN: u32 = 40u;
const W_SELF: u32 = 68u;
const W_NET: u32 = 84u;
const B_STATE: u32 = 100u;
const F_OUT: u32 = 104u;
const F_BASE: u32 = 112u;
const P_OUT: u32 = 114u;
const P_BASE: u32 = 122u;
const L_OUT: u32 = 124u;
const L_BASE: u32 = 132u;
// The gait head sits past the chemistry genes, at the end of the genome.
const G_OUT: u32 = 178u;
const G_BASE: u32 = 182u;
// `chem-layout.ts`'s GAIT_ANCHOR_MAX, transcribed with the offsets above.
const GAIT_ANCHOR_MAX: f32 = 8.0;

// The learning row, from `chem-layout.ts`: learned deltas on the state
// matrices, then their eligibility traces, then the critic, then last
// frame's value estimate. Indexed by slot, like the genome, because it is
// state a body keeps rather than something the host packs each frame.
const PLASTIC_LEN: u32 = 64u;
const LEARN_TRACE: u32 = 64u;
const LEARN_CRITIC: u32 = 128u;
const LEARN_PREV_V: u32 = 133u;
const LEARN_STRIDE: u32 = 134u;

// Floats written per body: h(4), emit(4), taste(4), then the six heads.
const OUT_STRIDE: u32 = 19u;

struct GenomeParams {
  n: u32,
  chemLen: u32,
  senseScale: f32,   // 1 / SENSE_SCALE
  groundScale: f32,  // 1 / cellCap
  // HEAD_SCALE, in the order the heads are written out.
  sCruise: f32,
  sTurn: f32,
  sAlign: f32,
  sSep: f32,
  sThrust: f32,
  sRecoil: f32,
  energyCh: f32,
  // Learning. `learnRate` at zero is the whole thing switched off, and the
  // block at the end of `state` is then never entered.
  learnRate: f32,
  learnCritic: f32,
  learnTrace: f32,
  learnDiscount: f32,
  maxWeight: f32,
  sAnchor: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
}

@group(0) @binding(0) var<uniform> G: GenomeParams;
// The field probe's output. Two vec4f a body; the second is the raw four
// channels under it, which is what the sense columns read.
@group(0) @binding(1) var<storage, read> samples: array<vec4f>;
@group(0) @binding(2) var<storage, read> chem: array<f32>;
// This frame's state, in list order. Read only: every body reads its
// neighbours' *previous* state, so the pass cannot write here.
@group(0) @binding(3) var<storage, read> hPrev: array<f32>;
// Two vec4f a body: [full, bound, demand, readsField] then [slot, _, _, _].
@group(0) @binding(4) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(5) var<storage, read> adjOff: array<u32>;
@group(0) @binding(6) var<storage, read> adjNei: array<u32>;
@group(0) @binding(7) var<storage, read_write> outv: array<f32>;
/*
 * What each body has learned, and is learning.
 *
 * The eighth storage buffer, against a per-stage guarantee of eight — the
 * field pass next door is the one that had to merge two bindings to fit, and
 * this one had a slot free. Read-write and *resident*: the host writes it
 * only to zero a recycled slot, and reads it only for the two bodies of a
 * rewrite about to commit, which need their learning consolidated into their
 * children's genome on the CPU where inheritance lives.
 */
@group(0) @binding(8) var<storage, read_write> learn: array<f32>;

// Bounded, signed, and no reflecting barrier at zero. Chosen over tanh on
// measurement: 0.373ms against 0.581 for 80k calls.
fn phi(v: f32) -> f32 {
  return v / (1.0 + abs(v));
}

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
  return min(max(v, lo), hi);
}

/** One row of an output head: `base[row] + matrix[row] . h`. */
fn headAt(g: u32, matrix: u32, base: u32, row: u32, h: vec4f) -> f32 {
  let o = g + matrix + row * STATE_DIMS;
  return chem[g + base + row]
    + chem[o] * h.x + chem[o + 1u] * h.y + chem[o + 2u] * h.z + chem[o + 3u] * h.w;
}

@compute @workgroup_size(64)
fn state(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= G.n) { return; }

  let facts = inputs[i * 2u];
  let slot = u32(inputs[i * 2u + 1u].x);
  let g = slot * G.chemLen;

  /*
   * The sense columns, or zero.
   *
   * `readsField` is the same gate the CPU uses, and it means the same thing
   * here even though the sample is already computed: the inputs are *zeroed*
   * rather than left holding a stale reading, so behaviour never depends on
   * when a body last happened to sample. The two scales bring the four onto
   * the range the other three inputs already occupy — without them a sense
   * gene has seven times the mutation leverage of every other input gene.
   */
  /*
   * No sense gate here, unlike the CPU pass.
   *
   * There it saves a bilinear sample into a sixteen-megabyte array for the
   * many genomes that multiply the result by zero. Here `gather` has already
   * taken that sample for every body — it is sitting in `samples` either way
   * — so the gate would save nothing at all. It would also go stale: it is
   * settled at birth from the genome's sense columns, and a body that
   * *learns* a sense weight would still be told it cannot see. The two paths
   * agree wherever the gate is right, because a gate of zero means every
   * weight multiplying the reading is zero.
   */
  let raw = samples[i * 2u + 1u];
  var s = raw * G.senseScale;
  let ec = u32(G.energyCh);
  s[ec] = raw[ec] * G.groundScale;

  let x0 = s.x;
  let x1 = s.y;
  let x2 = s.z;
  let x3 = s.w;
  let x4 = facts.x;  // FULL, already clamped by the host
  let x5 = facts.y;  // BOUND
  let x6 = facts.z;  // DEMAND, already clamped

  let po = i * STATE_DIMS;
  // `var`, not `let`: the learning block indexes it by a loop variable.
  var p = vec4f(hPrev[po], hPrev[po + 1u], hPrev[po + 2u], hPrev[po + 3u]);

  /*
   * The neighbour mean, gathered once for the body rather than once per
   * output dimension — the obvious way round costs four times as much,
   * because each of the four dimensions re-walks the whole adjacency.
   *
   * Mean and not sum: a sum scales with degree, so a hub saturates `phi` and
   * a leaf barely moves for a reason that is not about position. `BOUND`
   * carries degree already, bounded and on purpose.
   */
  var m = vec4f(0.0);
  let lo = adjOff[i];
  let hi = adjOff[i + 1u];
  let deg = hi - lo;
  if (deg > 0u) {
    for (var e = lo; e < hi; e++) {
      let b = adjNei[e] * STATE_DIMS;
      m += vec4f(hPrev[b], hPrev[b + 1u], hPrev[b + 2u], hPrev[b + 3u]);
    }
    m = m / f32(deg);
  }

  /*
   * The state matrices as the body actually has them: its genome plus what
   * it has learned. Added unconditionally rather than behind the CPU's
   * `plasticOn` branch — sixty-four more loads a body is nothing here, and a
   * body that has learned nothing is adding zeros.
   *
   * `vv` keeps the pre-activations, which the learning block needs for
   * `phi'`.
   */
  let lb = slot * LEARN_STRIDE;
  var h = vec4f(0.0);
  var vv = vec4f(0.0);
  for (var d = 0u; d < STATE_DIMS; d++) {
    let wi = g + W_IN + d * IN_DIMS;
    let ws = g + W_SELF + d * STATE_DIMS;
    let wn = g + W_NET + d * STATE_DIMS;
    // The same weights' offsets inside the learned span, which starts at W_IN.
    let li = lb + d * IN_DIMS;
    let ls = lb + W_SELF - W_IN + d * STATE_DIMS;
    let ln = lb + W_NET - W_IN + d * STATE_DIMS;
    let v = (chem[g + B_STATE + d] + learn[lb + B_STATE - W_IN + d])
      + (chem[wi] + learn[li]) * x0
      + (chem[wi + 1u] + learn[li + 1u]) * x1
      + (chem[wi + 2u] + learn[li + 2u]) * x2
      + (chem[wi + 3u] + learn[li + 3u]) * x3
      + (chem[wi + 4u] + learn[li + 4u]) * x4
      + (chem[wi + 5u] + learn[li + 5u]) * x5
      + (chem[wi + 6u] + learn[li + 6u]) * x6
      + (chem[ws] + learn[ls]) * p.x
      + (chem[ws + 1u] + learn[ls + 1u]) * p.y
      + (chem[ws + 2u] + learn[ls + 2u]) * p.z
      + (chem[ws + 3u] + learn[ls + 3u]) * p.w
      + (chem[wn] + learn[ln]) * m.x
      + (chem[wn + 1u] + learn[ln + 1u]) * m.y
      + (chem[wn + 2u] + learn[ln + 2u]) * m.z
      + (chem[wn + 3u] + learn[ln + 3u]) * m.w;
    vv[d] = v;
    h[d] = phi(v);
  }

  // Emit: relu, then normalised to a unit budget. That budget is the honesty
  // mechanism — feeding the dish and being heard come out of the same purse —
  // and this is the only place all four channels are known at once, so it is
  // the only place it can be enforced.
  var emit = vec4f(0.0);
  var sum = 0.0;
  for (var c = 0u; c < 4u; c++) {
    let o = g + E_OUT + c * STATE_DIMS;
    let v = chem[g + EMIT + c]
      + chem[o] * h.x + chem[o + 1u] * h.y + chem[o + 2u] * h.z + chem[o + 3u] * h.w;
    let w = max(v, 0.0);
    emit[c] = w;
    sum += w;
  }
  if (sum > 1e-6) { emit = emit / sum; }

  // Taste is signed and not normalised: a taste weight is compared against
  // other taste weights rather than spent, so there is no budget.
  var taste = vec4f(0.0);
  for (var c = 0u; c < 4u; c++) {
    let o = g + T_OUT + c * STATE_DIMS;
    taste[c] = chem[g + TASTE + c]
      + chem[o] * h.x + chem[o + 1u] * h.y + chem[o + 2u] * h.z + chem[o + 3u] * h.w;
  }

  let o = i * OUT_STRIDE;
  outv[o] = h.x;
  outv[o + 1u] = h.y;
  outv[o + 2u] = h.z;
  outv[o + 3u] = h.w;
  outv[o + 4u] = emit.x;
  outv[o + 5u] = emit.y;
  outv[o + 6u] = emit.z;
  outv[o + 7u] = emit.w;
  outv[o + 8u] = taste.x;
  outv[o + 9u] = taste.y;
  outv[o + 10u] = taste.z;
  outv[o + 11u] = taste.w;
  // Clamped to the ranges the heritable versions were bred inside: those
  // bounds are about what the forces survive, not about what a genome may say.
  outv[o + 12u] = clampf(headAt(g, L_OUT, L_BASE, 0u, h) * G.sCruise, 0.0, 180.0);
  outv[o + 13u] = clampf(headAt(g, L_OUT, L_BASE, 1u, h) * G.sTurn, 0.0, 8.0);
  outv[o + 14u] = clampf(headAt(g, F_OUT, F_BASE, 0u, h) * G.sAlign, -8.0, 16.0);
  outv[o + 15u] = clampf(headAt(g, F_OUT, F_BASE, 1u, h) * G.sSep, -60.0, 120.0);
  outv[o + 16u] = clampf(headAt(g, P_OUT, P_BASE, 0u, h) * G.sThrust, 0.0, 1.0);
  outv[o + 17u] = clampf(headAt(g, P_OUT, P_BASE, 1u, h) * G.sRecoil, 0.0, 200.0);
  // The gait's grip. Signed both ways: a body that lets go where its
  // neighbour holds walks the other way.
  outv[o + 18u] = clampf(headAt(g, G_OUT, G_BASE, 0u, h) * G.sAnchor, -GAIT_ANCHOR_MAX, GAIT_ANCHOR_MAX);

  /*
   * What this body learns from the frame it has just had. A line-for-line
   * port of the block at the end of `Sim.updateState`, which is the
   * reference; `genome-kernel.test.ts` is what holds the two together.
   *
   * Three factors, all local to the body: an eligibility trace per weight,
   * `phi'` for how much the state would have moved had that weight been
   * different, and one temporal-difference error from the body's own critic
   * saying whether things went better than predicted. The cost is the body's
   * own tank, which is `x4`.
   *
   * Nothing decays but the trace. A learned weight stays with the body and
   * goes with it into the next net it latches into.
   */
  if (G.learnRate > 0.0) {
    let cr = lb + LEARN_CRITIC;
    let value = learn[cr] * h.x + learn[cr + 1u] * h.y + learn[cr + 2u] * h.z
      + learn[cr + 3u] * h.w + learn[cr + 4u];
    let dlt = x4 - 1.0 + G.learnDiscount * value - learn[lb + LEARN_PREV_V];
    learn[lb + LEARN_PREV_V] = value;
    // The critic's own delta rule: its last estimate was a dot product with
    // last frame's `h`, so last frame's `h` is the gradient.
    let kc = G.learnCritic * dlt;
    learn[cr] = learn[cr] + kc * p.x;
    learn[cr + 1u] = learn[cr + 1u] + kc * p.y;
    learn[cr + 2u] = learn[cr + 2u] + kc * p.z;
    learn[cr + 3u] = learn[cr + 3u] + kc * p.w;
    learn[cr + 4u] = learn[cr + 4u] + kc;

    // phi'(v) = 1 / (1 + |v|)^2.
    let q = vec4f(1.0) / (vec4f(1.0) + abs(vv));
    var post = q * q;
    var xs = array<f32, 7>(x0, x1, x2, x3, x4, x5, x6);
    let step = G.learnRate * dlt;
    // Wx, Wh, Wn, then b — the order the genome lays them in, so one running
    // index serves the trace, the learned delta and the gene it adds to.
    var at = 0u;
    for (var d = 0u; d < STATE_DIMS; d++) {
      let pd = post[d];
      for (var k = 0u; k < IN_DIMS; k++) {
        let ti = lb + LEARN_TRACE + at;
        let tr = G.learnTrace * learn[ti] + pd * xs[k];
        learn[ti] = tr;
        let base = chem[g + W_IN + at];
        learn[lb + at] = clamp(learn[lb + at] + step * tr, -G.maxWeight - base, G.maxWeight - base);
        at = at + 1u;
      }
    }
    for (var d = 0u; d < STATE_DIMS; d++) {
      let pd = post[d];
      for (var k = 0u; k < STATE_DIMS; k++) {
        let ti = lb + LEARN_TRACE + at;
        let tr = G.learnTrace * learn[ti] + pd * p[k];
        learn[ti] = tr;
        let base = chem[g + W_IN + at];
        learn[lb + at] = clamp(learn[lb + at] + step * tr, -G.maxWeight - base, G.maxWeight - base);
        at = at + 1u;
      }
    }
    for (var d = 0u; d < STATE_DIMS; d++) {
      let pd = post[d];
      for (var k = 0u; k < STATE_DIMS; k++) {
        let ti = lb + LEARN_TRACE + at;
        let tr = G.learnTrace * learn[ti] + pd * m[k];
        learn[ti] = tr;
        let base = chem[g + W_IN + at];
        learn[lb + at] = clamp(learn[lb + at] + step * tr, -G.maxWeight - base, G.maxWeight - base);
        at = at + 1u;
      }
    }
    // The bias, whose input is one.
    for (var d = 0u; d < STATE_DIMS; d++) {
      let ti = lb + LEARN_TRACE + at;
      let tr = G.learnTrace * learn[ti] + post[d];
      learn[ti] = tr;
      let base = chem[g + W_IN + at];
      learn[lb + at] = clamp(learn[lb + at] + step * tr, -G.maxWeight - base, G.maxWeight - base);
      at = at + 1u;
    }
  }
}
