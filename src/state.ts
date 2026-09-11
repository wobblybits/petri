import { AgentStore } from './agent-store.ts';
import { type Agent, B_STATE, CHEM_LEN, CRITIC_LEN, F_BASE, F_OUT, GAIT_ANCHOR_MAX, G_BASE, G_OUT, HEAD_SCALE, IN_BOUND, IN_DEMAND, IN_DIMS, IN_FULL, IN_SENSE, L_BASE, L_OUT, PLASTIC_LEN, P_BASE, P_OUT, STATE_DIMS, W_IN, W_NET, W_SELF, emitVector, tasteVector } from './agents.ts';
import { WireAdjacency } from './energy.ts';
import { CH, Fields } from './fields.ts';
import { type Params } from './params.ts';
import { CHEM_TASTE_MAX } from './rewrite.ts';
import { clamp } from './wrap.ts';

  /**
   * One round of message passing per frame: every body's `h` from its
   * inputs, its own last value, and the mean of its wired neighbours'.
   *
   *     h <- phi( Wx.x + Wh.h + Wn.mean(h_j) + b )
   *
   * Mean, not sum: a sum scales with degree, and `BOUND` already carries it.
   *
   * Runs after `spreadRequests` and `flowCharges`, so `DEMAND` and `FULL`
   * are this frame's; `sense` and `trail` are last frame's, written by the
   * steer pass.
   *
   * `hPrev` so every body sees the same generation of its neighbours;
   * updating in place would make the answer depend on roster order.
   */
let hPrev = new Float64Array(0);
let slotBuf = new Int32Array(0);
const stateInput = new Float64Array(IN_DIMS);
const stateMean = new Float64Array(STATE_DIMS);
/** `chem + plastic` over the state matrices, for a body that has learned. */
const effWeights = new Float32Array(PLASTIC_LEN);
/** `phi'` per state dim, and the inputs each learned weight multiplies. */
const learnPost = new Float64Array(STATE_DIMS);
const learnPre = new Float64Array(IN_DIMS + 2 * STATE_DIMS);

/** What the state pass reads off the simulation. */
export interface StateHost {
  agentStore: AgentStore;
  fields: Fields;
  /** On the GPU field path the sense samples are already in the store. */
  fieldOnGpu: boolean;
  groundScale: number;
}

export function updateState(host: StateHost, list: Agent[], adj: WireAdjacency, params: Params): void {
  const n = list.length;
  const store = host.agentStore;
  const H = store.hAll;
  const SENSE = store.senseAll;
  // Straight into `chemAll` with a slot offset: the `Agent.chem` subarray
  // view costs more than the eighty multiplies it serves.
  const CHEM = store.chemAll;
  const S = STATE_DIMS;
  if (hPrev.length < n * S) hPrev = new Float64Array(n * S);
  const prev = hPrev;
  const slotOf = slotBuf.length >= n ? slotBuf : (slotBuf = new Int32Array(n));
  for (let i = 0; i < n; i++) {
    const s = list[i].slot;
    slotOf[i] = s;
    const po = i * S;
    const ho = s * S;
    for (let d = 0; d < S; d++) prev[po + d] = H[ho + d];
  }

  const { off, nei } = adj;
  const EXTRA = store.extra;
  const CAP = store.energyCap;
  const REQUEST = store.request;
  const BOUND_OF = store.bound;
  const X = store.x;
  const Y = store.y;
  const gScale = host.groundScale;
  const sScale = 1 / params.senseScale;
  const READS = store.readsField;
  const x = stateInput;
  const mean = stateMean;
  const EMITS = store.emitAll;
  const TASTES = store.tasteAll;
  const CRUISE = store.cruise;
  const TURN = store.turn;
  const FA = store.flockAlign;
  const FS = store.flockSep;
  const TR = store.transportRecoil;
  const GA = store.gaitAnchor;
  const PLASTIC = store.plasticAll;
  const TRACE = store.traceAll;
  const CRITIC = store.criticAll;
  const PREV_V = store.prevValue;
  const PLASTIC_ON = store.plasticOn;
  const post = learnPost;
  const pre = learnPre;
  const learn = params.learnRate > 0;
  const etaM = params.learnRate;
  const etaC = params.learnCritic;
  const lam = params.learnTrace;
  const discount = params.learnDiscount;
  const MAXW = CHEM_TASTE_MAX;
  for (let i = 0; i < n; i++) {
    const slot = slotOf[i];
    const g = slot * CHEM_LEN;

    /*
     * Sampling is skipped unless this genome reads the field. The inputs are
     * zeroed rather than left reading a stale `SENSE`, so behaviour never
     * depends on when a body last happened to sample.
     */
    if (READS[slot]) {
      const so = slot * 4;
      /*
       * On the GPU path `gpuFieldStep` filled these at the end of last
       * frame, already scaled; `fields.data` is a stale copy there. Both
       * scales bring an input onto the [0, 1] range the other inputs
       * occupy: a full ground cell reads one, a strong local signal about
       * one.
       */
      if (!host.fieldOnGpu) {
        host.fields.sampleAll(X[slot], Y[slot], SENSE, so);
        SENSE[so] *= sScale;
        SENSE[so + 1] *= sScale;
        SENSE[so + 3] *= sScale;
        SENSE[so + CH.energy] *= gScale;
      }
      x[IN_SENSE] = SENSE[so];
      x[IN_SENSE + 1] = SENSE[so + 1];
      x[IN_SENSE + 2] = SENSE[so + 2];
      x[IN_SENSE + 3] = SENSE[so + 3];
    } else {
      x[IN_SENSE] = 0;
      x[IN_SENSE + 1] = 0;
      x[IN_SENSE + 2] = 0;
      x[IN_SENSE + 3] = 0;
    }
    const cap = CAP[slot];
    const full = cap > 0 ? EXTRA[slot] / cap : 0;
    x[IN_FULL] = full <= 0 ? 0 : full >= 1 ? 1 : full;
    x[IN_BOUND] = BOUND_OF[slot];
    const r = REQUEST[slot];
    x[IN_DEMAND] = r <= 0 ? 0 : r >= 1 ? 1 : r;

    // Neighbour mean once per body, not once per output dimension.
    const lo = off[i];
    const hi = off[i + 1];
    const deg = hi - lo;
    if (deg > 0) {
      for (let k = 0; k < S; k++) mean[k] = 0;
      for (let e = lo; e < hi; e++) {
        const base = nei[e] * S;
        for (let k = 0; k < S; k++) mean[k] += prev[base + k];
      }
      for (let k = 0; k < S; k++) mean[k] /= deg;
    }

    // Unrolled over the four state dimensions; a neighbour mean of zero when
    // there are no neighbours, since `Wn` times nothing is nothing.
    const po = i * S;
    const ho = slot * S;
    const p0 = prev[po];
    const p1 = prev[po + 1];
    const p2 = prev[po + 2];
    const p3 = prev[po + 3];
    const m0 = deg > 0 ? mean[0] : 0;
    const m1 = deg > 0 ? mean[1] : 0;
    const m2 = deg > 0 ? mean[2] : 0;
    const m3 = deg > 0 ? mean[3] : 0;
    const x0 = x[0];
    const x1 = x[1];
    const x2 = x[2];
    const x3 = x[3];
    const x4 = x[4];
    const x5 = x[5];
    const x6 = x[6];
    /*
     * Genome plus whatever the body has learned. `plasticOn` is monotone —
     * learned weights never decay — so the sum is off the path of a body
     * that has not learned.
     */
    let W = CHEM;
    let wo = g + W_IN;
    if (PLASTIC_ON[slot]) {
      const plo = slot * PLASTIC_LEN;
      const eff = effWeights;
      for (let k = 0; k < PLASTIC_LEN; k++) eff[k] = CHEM[wo + k] + PLASTIC[plo + k];
      W = eff;
      wo = 0;
    }
    const gi = wo;
    const gs = wo + W_SELF - W_IN;
    const gn = wo + W_NET - W_IN;
    const gb = wo + B_STATE - W_IN;
    let v0 = 0;
    let v1 = 0;
    let v2 = 0;
    let v3 = 0;
    {
      const wi = gi + 0;
      const ws = gs + 0;
      const wn = gn + 0;
      const v =
        W[gb + 0] +
        W[wi] * x0 + W[wi + 1] * x1 + W[wi + 2] * x2 + W[wi + 3] * x3 +
        W[wi + 4] * x4 + W[wi + 5] * x5 + W[wi + 6] * x6 +
        W[ws] * p0 + W[ws + 1] * p1 + W[ws + 2] * p2 + W[ws + 3] * p3 +
        W[wn] * m0 + W[wn + 1] * m1 + W[wn + 2] * m2 + W[wn + 3] * m3;
      v0 = v;
      H[ho + 0] = v / (1 + (v < 0 ? -v : v));
    }
    {
      const wi = gi + 7;
      const ws = gs + 4;
      const wn = gn + 4;
      const v =
        W[gb + 1] +
        W[wi] * x0 + W[wi + 1] * x1 + W[wi + 2] * x2 + W[wi + 3] * x3 +
        W[wi + 4] * x4 + W[wi + 5] * x5 + W[wi + 6] * x6 +
        W[ws] * p0 + W[ws + 1] * p1 + W[ws + 2] * p2 + W[ws + 3] * p3 +
        W[wn] * m0 + W[wn + 1] * m1 + W[wn + 2] * m2 + W[wn + 3] * m3;
      v1 = v;
      H[ho + 1] = v / (1 + (v < 0 ? -v : v));
    }
    {
      const wi = gi + 14;
      const ws = gs + 8;
      const wn = gn + 8;
      const v =
        W[gb + 2] +
        W[wi] * x0 + W[wi + 1] * x1 + W[wi + 2] * x2 + W[wi + 3] * x3 +
        W[wi + 4] * x4 + W[wi + 5] * x5 + W[wi + 6] * x6 +
        W[ws] * p0 + W[ws + 1] * p1 + W[ws + 2] * p2 + W[ws + 3] * p3 +
        W[wn] * m0 + W[wn + 1] * m1 + W[wn + 2] * m2 + W[wn + 3] * m3;
      v2 = v;
      H[ho + 2] = v / (1 + (v < 0 ? -v : v));
    }
    {
      const wi = gi + 21;
      const ws = gs + 12;
      const wn = gn + 12;
      const v =
        W[gb + 3] +
        W[wi] * x0 + W[wi + 1] * x1 + W[wi + 2] * x2 + W[wi + 3] * x3 +
        W[wi + 4] * x4 + W[wi + 5] * x5 + W[wi + 6] * x6 +
        W[ws] * p0 + W[ws + 1] * p1 + W[ws + 2] * p2 + W[ws + 3] * p3 +
        W[wn] * m0 + W[wn + 1] * m1 + W[wn + 2] * m2 + W[wn + 3] * m3;
      v3 = v;
      H[ho + 3] = v / (1 + (v < 0 ? -v : v));
    }

    /*
     * The output heads off this frame's state, written into the store
     * fields every consumer reads. Clamped to what the forces survive:
     * alignment past its ceiling is negative damping; separation past its
     * own has no equilibrium to settle at. Emit and taste are materialised
     * here once; emit is normalised on the way in, the one place all four
     * channels are known together.
     */
    emitVector(CHEM, g, H, ho, EMITS, slot * 4);
    tasteVector(CHEM, g, H, ho, TASTES, slot * 4);

    CRUISE[slot] = clamp(headAt(CHEM, g, L_OUT, L_BASE, 0, H, ho, S) * HEAD_SCALE.cruise, 0, 180);
    TURN[slot] = clamp(headAt(CHEM, g, L_OUT, L_BASE, 1, H, ho, S) * HEAD_SCALE.turn, 0, 8);
    FA[slot] = clamp(headAt(CHEM, g, F_OUT, F_BASE, 0, H, ho, S) * HEAD_SCALE.align, -8, 16);
    FS[slot] = clamp(headAt(CHEM, g, F_OUT, F_BASE, 1, H, ho, S) * HEAD_SCALE.sep, -60, 120);
    TR[slot] = clamp(headAt(CHEM, g, P_OUT, P_BASE, 0, H, ho, S) * HEAD_SCALE.recoil, 0, 200);
    // Signed both ways on purpose: a body that lets go where its neighbour
    // holds walks the other way, and that is a lineage's to choose.
    GA[slot] = clamp(headAt(CHEM, g, G_OUT, G_BASE, 0, H, ho, S) * HEAD_SCALE.anchor, -GAIT_ANCHOR_MAX, GAIT_ANCHOR_MAX);

    /*
     * Three-factor Hebbian learning, every factor local to the body: an
     * eligibility trace per weight, `phi'` per state dim, and a
     * temporal-difference error from the body's own critic. The cost is the
     * body's own tank: `x4` is `IN_FULL` in [0, 1], so `x4 - 1` is zero
     * when full and -1 when empty. Nothing decays but the trace; a learned
     * weight is the body's for life and travels with it into its next net.
     */
    if (learn) {
      const plo = slot * PLASTIC_LEN;
      const cro = slot * CRITIC_LEN;
      const value =
        CRITIC[cro] * H[ho] +
        CRITIC[cro + 1] * H[ho + 1] +
        CRITIC[cro + 2] * H[ho + 2] +
        CRITIC[cro + 3] * H[ho + 3] +
        CRITIC[cro + 4];
      const dlt = x4 - 1 + discount * value - PREV_V[slot];
      PREV_V[slot] = value;
      // Critic delta rule: its last estimate was a dot with last frame's
      // `h`, which `prev` still holds.
      const kc = etaC * dlt;
      CRITIC[cro] += kc * p0;
      CRITIC[cro + 1] += kc * p1;
      CRITIC[cro + 2] += kc * p2;
      CRITIC[cro + 3] += kc * p3;
      CRITIC[cro + 4] += kc;

      // phi'(v) = 1 / (1 + |v|)^2. A saturated dimension has almost none
      // of it, which is what stops a pinned state dragging its inputs.
      const q0 = 1 / (1 + (v0 < 0 ? -v0 : v0));
      const q1 = 1 / (1 + (v1 < 0 ? -v1 : v1));
      const q2 = 1 / (1 + (v2 < 0 ? -v2 : v2));
      const q3 = 1 / (1 + (v3 < 0 ? -v3 : v3));
      post[0] = q0 * q0;
      post[1] = q1 * q1;
      post[2] = q2 * q2;
      post[3] = q3 * q3;
      pre[0] = x0;
      pre[1] = x1;
      pre[2] = x2;
      pre[3] = x3;
      pre[4] = x4;
      pre[5] = x5;
      pre[6] = x6;
      pre[7] = p0;
      pre[8] = p1;
      pre[9] = p2;
      pre[10] = p3;
      pre[11] = m0;
      pre[12] = m1;
      pre[13] = m2;
      pre[14] = m3;
      const step = etaM * dlt;
      const gW = g + W_IN;
      let touched = 0;
      let at = 0;
      // `Wx`, `Wh`, `Wn`, `b` in genome order, so one index serves trace,
      // delta and gene. The clamp is against the sum the state pass reads.
      for (let d = 0; d < S; d++) {
        const pd = post[d];
        for (let k = 0; k < IN_DIMS; k++, at++) {
          const ti = plo + at;
          const tr = lam * TRACE[ti] + pd * pre[k];
          TRACE[ti] = tr;
          const base = CHEM[gW + at];
          let w = PLASTIC[ti] + step * tr;
          if (w < -MAXW - base) w = -MAXW - base;
          else if (w > MAXW - base) w = MAXW - base;
          PLASTIC[ti] = w;
          if (w !== 0) {
            touched = 1;
            // A learned sense weight makes a field reader; monotone, exact
            // because nothing decays back to zero.
            if (k < 4) READS[slot] = 1;
          }
        }
      }
      for (let d = 0; d < S; d++) {
        const pd = post[d];
        for (let k = 0; k < S; k++, at++) {
          const ti = plo + at;
          const tr = lam * TRACE[ti] + pd * pre[7 + k];
          TRACE[ti] = tr;
          const base = CHEM[gW + at];
          let w = PLASTIC[ti] + step * tr;
          if (w < -MAXW - base) w = -MAXW - base;
          else if (w > MAXW - base) w = MAXW - base;
          PLASTIC[ti] = w;
          if (w !== 0) touched = 1;
        }
      }
      for (let d = 0; d < S; d++) {
        const pd = post[d];
        for (let k = 0; k < S; k++, at++) {
          const ti = plo + at;
          const tr = lam * TRACE[ti] + pd * pre[11 + k];
          TRACE[ti] = tr;
          const base = CHEM[gW + at];
          let w = PLASTIC[ti] + step * tr;
          if (w < -MAXW - base) w = -MAXW - base;
          else if (w > MAXW - base) w = MAXW - base;
          PLASTIC[ti] = w;
          if (w !== 0) touched = 1;
        }
      }
      // The bias, whose input is one.
      for (let d = 0; d < S; d++, at++) {
        const ti = plo + at;
        const tr = lam * TRACE[ti] + post[d];
        TRACE[ti] = tr;
        const base = CHEM[gW + at];
        let w = PLASTIC[ti] + step * tr;
        if (w < -MAXW - base) w = -MAXW - base;
        else if (w > MAXW - base) w = MAXW - base;
        PLASTIC[ti] = w;
        if (w !== 0) touched = 1;
      }
      if (touched) PLASTIC_ON[slot] = 1;
    }
  }
}

/**
 * One row of an output head, read straight out of the store arrays: the
 * twin of `agents.ts`'s `head` for `updateState`'s inner loop.
 */
function headAt(
  chem: Float32Array,
  g: number,
  matrix: number,
  base: number,
  row: number,
  h: Float64Array,
  ho: number,
  dims: number,
): number {
  const o = g + matrix + row * dims;
  return (
    chem[g + base + row] +
    chem[o] * h[ho] +
    chem[o + 1] * h[ho + 1] +
    chem[o + 2] * h[ho + 2] +
    chem[o + 3] * h[ho + 3]
  );
}
