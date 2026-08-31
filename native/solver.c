/*
 * Packed FAR solver + NEAR XPBD + SAT + scent field. No libc heap; static
 * buffers.
 *
 * Body layout matches src/gpu/far-kernel.ts (12 floats / body).
 * FAR wires are 4 floats. NEAR wires are 12 floats in the same buffer.
 * Discs use a spatial hash, not N². FAR chord span is one Jacobi impulse.
 * NEAR wires are Gauss-Seidel XPBD (span + links + bend + shape) matching
 * src/chain.ts. Detailed pairs use SAT + angular contact matching
 * src/collide.ts and src/chain.ts solveContact. FAR-FAR stays disc.
 *
 * Rebuild: npm run wasm
 *
 * Compiled with -msimd128. Independent wires are batched 4-wide when they
 * commute with the original Gauss-Seidel order (no shared bodies with any
 * skipped predecessor). SAT stays sequential GS; triangle math is f32.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define HAVE_SIMD 1
typedef v128_t v128;
#else
#define HAVE_SIMD 0
#endif

#define MAX_BODIES 16384
#define MAX_WIRES 16384
#define MAX_NODES 131072
#define MAX_CELLS 65536
#define MAX_PAIRS 262144
#define STRIDE 12
#define WIRE_FAR 4
#define WIRE_NEAR 12
#define NODE_STRIDE 8
#define MAX_COLS 160
#define MAX_ROWS 512
#define CHANNELS 4
#define MAX_SCENT (MAX_COLS * MAX_ROWS * CHANNELS)

#define FAR_X 0
#define FAR_Y 1
#define FAR_VX 2
#define FAR_VY 3
#define FAR_HEADING 4
#define FAR_OMEGA 5
#define FAR_INVMASS 6
#define FAR_RADIUS 7
#define FAR_LOCKED 8
#define FAR_PREVX 9
#define FAR_PREVY 10
#define FAR_PREVHEAD 11

#define WN_A 0
#define WN_B 1
#define WN_REST 2
#define WN_ROPE 3
#define WN_SCALE 4
#define WN_SLACK 5
#define WN_ASLOT 6
#define WN_BSLOT 7
#define WN_NODE0 8
#define WN_NNODES 9
#define WN_FLAGS 10

#define WF_FULL 1
#define WF_SKIP 2
#define WF_SHAPE 4
#define WF_HOLD 8

#define ND_X 0
#define ND_Y 1
#define ND_VX 2
#define ND_VY 3
#define ND_PREVX 4
#define ND_PREVY 5
#define ND_SX 6
#define ND_SY 7

#define SLOP 0.35f
#define SKIN 0.85f
#define ERA_R 8.0f
#define CONTACT_COMP 4.0e-6f
#define SPAN_COMP 3.0e-6f
#define LINK_COMP 2.0e-6f
#define BEND_COMP 1.5e-4f
#define SHAPE_COMP 3.0e-4f
#define CHAIN_MASS 0.08f
#define SLACK_RATIO 2.0f
#define GRAB_COMP 1.0e-6f
#define GRAB_STEP 1.5f
#define PI 3.14159265f
#define TAU 6.2831853f
#define HIT_STRIDE 10
#define MAX_HITS 8192

static float bodies[MAX_BODIES * STRIDE];
static float wires[MAX_WIRES * WIRE_NEAR];
static float nodes[MAX_NODES * NODE_STRIDE];
static float inv_inertia[MAX_BODIES];
static float scale[MAX_BODIES];
static uint8_t kind[MAX_BODIES];
static uint8_t detailed[MAX_BODIES];
static float delta[MAX_BODIES * 2];
static float scent[MAX_SCENT];
static float scent_tmp[MAX_SCENT];
static uint8_t walls[MAX_COLS * MAX_ROWS];
static int g_pairs = 0;

static int32_t cell_of[MAX_BODIES];
static int32_t order[MAX_BODIES];
static int32_t start[MAX_CELLS + 1];
static int32_t pair_a[MAX_PAIRS];
static int32_t pair_b[MAX_PAIRS];
static float hits[MAX_HITS * HIT_STRIDE];
static int g_hits = 0;
static int32_t adj_off[MAX_BODIES + 1];
static int32_t adj_nei[MAX_WIRES * 2];
static int32_t flock_id[MAX_BODIES];
static int32_t flock_dist[MAX_BODIES];
static int32_t flock_q[MAX_BODIES];
static int32_t flock_seen[MAX_BODIES];
static float flock_mass[MAX_BODIES];
static uint8_t swim[MAX_BODIES];
static float port_rx[MAX_BODIES * 3];
static float port_ry[MAX_BODIES * 3];
static float cs_c[MAX_BODIES];
static float cs_s[MAX_BODIES];
static float cs_a[MAX_BODIES];
static uint8_t cs_ok[MAX_BODIES];
static uint8_t pose_ok[MAX_BODIES];
static uint8_t wdone[MAX_WIRES];

static float wrap_angle(float a) {
  if (a >= -PI && a < PI) return a;
  float r = fmodf(a, TAU);
  if (r >= PI) r -= TAU;
  if (r < -PI) r += TAU;
  return r;
}

static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }

static void ensure_cs(int i) {
  float a = bodies[i * STRIDE + FAR_HEADING];
  if (cs_ok[i] && a == cs_a[i]) return;
  cs_c[i] = cosf(a);
  cs_s[i] = sinf(a);
  cs_a[i] = a;
  cs_ok[i] = 1;
}

static void invalidate_pose(int i) {
  cs_ok[i] = 0;
  pose_ok[i] = 0;
}

static int collect_pairs(int n, float cell_size) {
  if (n <= 0) return 0;
  float minx = bodies[FAR_X], maxx = minx;
  float miny = bodies[FAR_Y], maxy = miny;
  for (int i = 1; i < n; i++) {
    float x = bodies[i * STRIDE + FAR_X];
    float y = bodies[i * STRIDE + FAR_Y];
    if (x < minx) minx = x;
    if (x > maxx) maxx = x;
    if (y < miny) miny = y;
    if (y > maxy) maxy = y;
  }
  float cell = cell_size > 1e-3f ? cell_size : 1e-3f;
  int max_cells = imin(MAX_CELLS, imax(256, n * 32));
  int cols = (int)((maxx - minx) / cell) + 1;
  int rows = (int)((maxy - miny) / cell) + 1;
  if (cols < 1) cols = 1;
  if (rows < 1) rows = 1;
  while ((int64_t)cols * rows > max_cells) {
    cell *= 2.f;
    cols = (int)((maxx - minx) / cell) + 1;
    rows = (int)((maxy - miny) / cell) + 1;
    if (cols < 1) cols = 1;
    if (rows < 1) rows = 1;
  }
  int cells = cols * rows;
  if (cells > MAX_CELLS) cells = MAX_CELLS;
  memset(start, 0, (size_t)(cells + 1) * sizeof(int32_t));
  float inv = 1.f / cell;
  for (int i = 0; i < n; i++) {
    float x = bodies[i * STRIDE + FAR_X];
    float y = bodies[i * STRIDE + FAR_Y];
    int cx = (int)((x - minx) * inv);
    int cy = (int)((y - miny) * inv);
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx >= cols) cx = cols - 1;
    if (cy >= rows) cy = rows - 1;
    int c = cy * cols + cx;
    cell_of[i] = c;
    start[c + 1]++;
  }
  for (int c = 0; c < cells; c++) start[c + 1] += start[c];
  for (int i = 0; i < n; i++) order[start[cell_of[i]]++] = i;
  for (int c = cells; c > 0; c--) start[c] = start[c - 1];
  start[0] = 0;

  int np = 0;
  const int dxn[4] = {1, -1, 0, 1};
  const int dyn[4] = {0, 1, 1, 1};
  for (int cy = 0; cy < rows; cy++) {
    for (int cx = 0; cx < cols; cx++) {
      int c = cy * cols + cx;
      int a0 = start[c], a1 = start[c + 1];
      if (a0 == a1) continue;
      for (int a = a0; a < a1; a++) {
        for (int b = a + 1; b < a1 && np < MAX_PAIRS; b++) {
          pair_a[np] = order[a];
          pair_b[np] = order[b];
          np++;
        }
      }
      for (int k = 0; k < 4; k++) {
        int nx = cx + dxn[k];
        int ny = cy + dyn[k];
        if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
        int nb = ny * cols + nx;
        int b0 = start[nb], b1 = start[nb + 1];
        for (int a = a0; a < a1; a++) {
          for (int b = b0; b < b1 && np < MAX_PAIRS; b++) {
            pair_a[np] = order[a];
            pair_b[np] = order[b];
            np++;
          }
        }
      }
    }
  }
  return np;
}

static void integrate(int n, float h) {
  for (int i = 0; i < n; i++) {
    float *p = bodies + i * STRIDE;
    p[FAR_PREVX] = p[FAR_X];
    p[FAR_PREVY] = p[FAR_Y];
    p[FAR_PREVHEAD] = p[FAR_HEADING];
    if (p[FAR_LOCKED] >= 0.5f) continue;
    p[FAR_X] += p[FAR_VX] * h;
    p[FAR_Y] += p[FAR_VY] * h;
    p[FAR_HEADING] = wrap_angle(p[FAR_HEADING] + p[FAR_OMEGA] * h);
  }
}

static void disc(int n, float h) {
  memset(delta, 0, (size_t)n * 2 * sizeof(float));
  float maxr = 0.f;
  for (int i = 0; i < n; i++) {
    float r = bodies[i * STRIDE + FAR_RADIUS];
    if (r > maxr) maxr = r;
  }
  int np = collect_pairs(n, maxr * 2.f + SLOP + 4.f);
  float alpha = CONTACT_COMP / fmaxf(1e-12f, h * h);
  for (int p = 0; p < np; p++) {
    int i = pair_a[p], j = pair_b[p];
    float *pi = bodies + i * STRIDE;
    float *pj = bodies + j * STRIDE;
    if (pi[FAR_LOCKED] >= 0.5f && pj[FAR_LOCKED] >= 0.5f) continue;
    float dx = pj[FAR_X] - pi[FAR_X];
    float dy = pj[FAR_Y] - pi[FAR_Y];
    float dist = sqrtf(dx * dx + dy * dy);
    float keep = pi[FAR_RADIUS] + pj[FAR_RADIUS];
    if (dist >= keep || dist < 1e-6f) continue;
    float depth = keep - dist - SLOP;
    if (depth <= 0.f) continue;
    float wA = pi[FAR_INVMASS], wB = pj[FAR_INVMASS];
    float denom = wA + wB + alpha;
    if (denom < 1e-12f) continue;
    float lam = depth / denom;
    float s = lam / dist;
    if (pi[FAR_LOCKED] < 0.5f && wA > 0.f) {
      delta[i * 2] -= dx * s * wA;
      delta[i * 2 + 1] -= dy * s * wA;
    }
    if (pj[FAR_LOCKED] < 0.5f && wB > 0.f) {
      delta[j * 2] += dx * s * wB;
      delta[j * 2 + 1] += dy * s * wB;
    }
  }
}

static void span(int n, int n_wires, float h) {
  memset(delta, 0, (size_t)n * 2 * sizeof(float));
  float alpha = SPAN_COMP / fmaxf(1e-12f, h * h);
  for (int w = 0; w < n_wires; w++) {
    int i = (int)wires[w * 4];
    int j = (int)wires[w * 4 + 1];
    float rest = wires[w * 4 + 2];
    if (i < 0 || j < 0 || i >= n || j >= n || i == j) continue;
    float *pi = bodies + i * STRIDE;
    float *pj = bodies + j * STRIDE;
    float dx = pj[FAR_X] - pi[FAR_X];
    float dy = pj[FAR_Y] - pi[FAR_Y];
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist < 1e-9f) continue;
    float C = dist - rest;
    float wA = pi[FAR_INVMASS], wB = pj[FAR_INVMASS];
    float denom = wA + wB + alpha;
    if (denom < 1e-12f) continue;
    float lam = -C / denom;
    float s = lam / dist;
    if (pi[FAR_LOCKED] < 0.5f && wA > 0.f) {
      delta[i * 2] -= dx * s * wA;
      delta[i * 2 + 1] -= dy * s * wA;
    }
    if (pj[FAR_LOCKED] < 0.5f && wB > 0.f) {
      delta[j * 2] += dx * s * wB;
      delta[j * 2 + 1] += dy * s * wB;
    }
  }
}

static void apply(int n) {
  for (int i = 0; i < n; i++) {
    float *p = bodies + i * STRIDE;
    if (p[FAR_LOCKED] >= 0.5f) continue;
    p[FAR_X] += delta[i * 2];
    p[FAR_Y] += delta[i * 2 + 1];
  }
}

static void finalize(int n, float h) {
  float invh = 1.f / h;
  for (int i = 0; i < n; i++) {
    float *p = bodies + i * STRIDE;
    if (p[FAR_LOCKED] >= 0.5f) {
      p[FAR_VX] = 0.f;
      p[FAR_VY] = 0.f;
      p[FAR_OMEGA] = 0.f;
      continue;
    }
    p[FAR_VX] = (p[FAR_X] - p[FAR_PREVX]) * invh;
    p[FAR_VY] = (p[FAR_Y] - p[FAR_PREVY]) * invh;
    p[FAR_OMEGA] = wrap_angle(p[FAR_HEADING] - p[FAR_PREVHEAD]) * invh;
  }
}

static void stem_local(uint8_t k, int slot, float sc, float *lx, float *ly) {
  if (k == 0) {
    *lx = slot == 0 ? 8.f * sc : 0.f;
    *ly = 0.f;
    return;
  }
  float s = 16.f * sc;
  if (slot == 0) {
    *lx = s * 1.05f;
    *ly = 0.f;
    return;
  }
  *lx = -s * 0.55f;
  *ly = (slot == 1 ? -1.f : 1.f) * s * 0.82f * 0.7f;
}

static void refresh_pose(int i) {
  ensure_cs(i);
  float c = cs_c[i], s = cs_s[i];
  for (int slot = 0; slot < 3; slot++) {
    float lx, ly;
    stem_local(kind[i], slot, scale[i], &lx, &ly);
    port_rx[i * 3 + slot] = lx * c - ly * s;
    port_ry[i * 3 + slot] = lx * s + ly * c;
  }
  pose_ok[i] = 1;
}

static void refresh_poses(int n) {
  for (int i = 0; i < n; i++) refresh_pose(i);
}

static void attach(int i, int slot, float *rx, float *ry) {
  if (slot < 0) slot = 0;
  if (slot > 2) slot = 2;
  float a = bodies[i * STRIDE + FAR_HEADING];
  if (!pose_ok[i] || !cs_ok[i] || a != cs_a[i]) refresh_pose(i);
  int k = i * 3 + slot;
  *rx = port_rx[k];
  *ry = port_ry[k];
}

static float gen_inv(int i, float rx, float ry, float nx, float ny) {
  float rxn = rx * ny - ry * nx;
  return bodies[i * STRIDE + FAR_INVMASS] + inv_inertia[i] * rxn * rxn;
}

static void apply_imp(int i, float rx, float ry, float nx, float ny, float lambda) {
  float *p = bodies + i * STRIDE;
  if (p[FAR_LOCKED] >= 0.5f || lambda == 0.f) return;
  float im = p[FAR_INVMASS];
  p[FAR_X] += im * lambda * nx;
  p[FAR_Y] += im * lambda * ny;
  p[FAR_HEADING] = wrap_angle(p[FAR_HEADING] + inv_inertia[i] * (rx * ny - ry * nx) * lambda);
  invalidate_pose(i);
}

static void solve_span(int i, int si, int j, int sj, float rest, float alpha) {
  float rAx, rAy, rBx, rBy;
  attach(i, si, &rAx, &rAy);
  attach(j, sj, &rBx, &rBy);
  float *pi = bodies + i * STRIDE;
  float *pj = bodies + j * STRIDE;
  float dx = (pj[FAR_X] + rBx) - (pi[FAR_X] + rAx);
  float dy = (pj[FAR_Y] + rBy) - (pi[FAR_Y] + rAy);
  float dist = sqrtf(dx * dx + dy * dy);
  if (dist < 1e-9f) return;
  float nx = dx / dist, ny = dy / dist;
  float C = dist - rest;
  float denom = gen_inv(i, rAx, rAy, nx, ny) + gen_inv(j, rBx, rBy, nx, ny) + alpha;
  if (denom < 1e-12f) return;
  float lambda = -C / denom;
  apply_imp(i, rAx, rAy, -nx, -ny, lambda);
  apply_imp(j, rBx, rBy, nx, ny, lambda);
}

static void solve_body_node(int i, int slot, float *node, float rest, float alpha) {
  float rx, ry;
  attach(i, slot, &rx, &ry);
  float *p = bodies + i * STRIDE;
  float dx = node[ND_X] - (p[FAR_X] + rx);
  float dy = node[ND_Y] - (p[FAR_Y] + ry);
  float dist = sqrtf(dx * dx + dy * dy);
  if (dist < 1e-9f) return;
  float C = dist - rest;
  float wNode = 1.f / CHAIN_MASS;
  float denom = wNode + (C < 0.f ? alpha * SLACK_RATIO : alpha);
  if (denom < 1e-12f) return;
  float lambda = -C / denom;
  node[ND_X] += (wNode * lambda * dx) / dist;
  node[ND_Y] += (wNode * lambda * dy) / dist;
}

static void solve_node_link(float *a, float *b, float rest, float alpha) {
  float dx = b[ND_X] - a[ND_X];
  float dy = b[ND_Y] - a[ND_Y];
  float dist = sqrtf(dx * dx + dy * dy);
  if (dist < 1e-9f) return;
  float nx = dx / dist, ny = dy / dist;
  float C = dist - rest;
  float w = 1.f / CHAIN_MASS;
  float lambda = -C / (2.f * w + (C < 0.f ? alpha * SLACK_RATIO : alpha));
  a[ND_X] -= w * lambda * nx;
  a[ND_Y] -= w * lambda * ny;
  b[ND_X] += w * lambda * nx;
  b[ND_Y] += w * lambda * ny;
}

static void solve_bend(float ax, float ay, float *b, float cx, float cy, float wA, float wC, float alpha) {
  float Cx = ax - 2.f * b[ND_X] + cx;
  float Cy = ay - 2.f * b[ND_Y] + cy;
  float wB = 1.f / CHAIN_MASS;
  float denom = wA + 4.f * wB + wC + alpha;
  if (denom < 1e-12f) return;
  b[ND_X] += 2.f * wB * (Cx / denom);
  b[ND_Y] += 2.f * wB * (Cy / denom);
}

static void solve_shape(float *node, float tx, float ty, float alpha) {
  float dx = node[ND_X] - tx;
  float dy = node[ND_Y] - ty;
  float C = sqrtf(dx * dx + dy * dy);
  if (C < 1e-9f) return;
  float w = 1.f / CHAIN_MASS;
  float lambda = -C / (w + alpha);
  node[ND_X] += (w * lambda * dx) / C;
  node[ND_Y] += (w * lambda * dy) / C;
}

static void solve_one_wire(int n, float *W, float h) {
  int flags = (int)W[WN_FLAGS];
  if (flags & WF_SKIP) return;
  int i = (int)W[WN_A];
  int j = (int)W[WN_B];
  if (i < 0 || j < 0 || i >= n || j >= n || i == j) return;
  float invH2 = 1.f / fmaxf(1e-12f, h * h);
  float soft = W[WN_SCALE] * W[WN_SLACK];
  float aSpan = SPAN_COMP * soft * invH2;
  int si = (int)W[WN_ASLOT];
  int sj = (int)W[WN_BSLOT];
  float rest = W[WN_REST];
  int nn = (int)W[WN_NNODES];
  if (!(flags & WF_FULL) || nn <= 0) {
    solve_span(i, si, j, sj, rest, aSpan);
    return;
  }
  int n0 = (int)W[WN_NODE0];
  if (n0 < 0 || n0 + nn > MAX_NODES) return;
  float aLink = LINK_COMP * soft * invH2;
  float aBend = BEND_COMP * soft * invH2;
  float aShape = SHAPE_COMP * soft * invH2;
  solve_span(i, si, j, sj, rest, aSpan);
  float linkRest = W[WN_ROPE] / (float)(nn + 1);
  float *nd0 = nodes + n0 * NODE_STRIDE;
  solve_body_node(i, si, nd0, linkRest, aLink);
  for (int k = 0; k < nn - 1; k++) {
    solve_node_link(nodes + (n0 + k) * NODE_STRIDE, nodes + (n0 + k + 1) * NODE_STRIDE, linkRest, aLink);
  }
  solve_body_node(j, sj, nodes + (n0 + nn - 1) * NODE_STRIDE, linkRest, aLink);

  float rAx, rAy, rBx, rBy;
  attach(i, si, &rAx, &rAy);
  attach(j, sj, &rBx, &rBy);
  float sAx = bodies[i * STRIDE + FAR_X] + rAx;
  float sAy = bodies[i * STRIDE + FAR_Y] + rAy;
  float sBx = bodies[j * STRIDE + FAR_X] + rBx;
  float sBy = bodies[j * STRIDE + FAR_Y] + rBy;
  float wNode = 1.f / CHAIN_MASS;
  for (int k = 0; k < nn; k++) {
    float *b = nodes + (n0 + k) * NODE_STRIDE;
    float px, py, nx, ny, wP, wN;
    if (k == 0) {
      px = sAx;
      py = sAy;
      wP = 0.f;
    } else {
      float *p = nodes + (n0 + k - 1) * NODE_STRIDE;
      px = p[ND_X];
      py = p[ND_Y];
      wP = wNode;
    }
    if (k == nn - 1) {
      nx = sBx;
      ny = sBy;
      wN = 0.f;
    } else {
      float *q = nodes + (n0 + k + 1) * NODE_STRIDE;
      nx = q[ND_X];
      ny = q[ND_Y];
      wN = wNode;
    }
    solve_bend(px, py, b, nx, ny, wP, wN, aBend);
  }
  if (flags & WF_SHAPE) {
    for (int k = 0; k < nn; k++) {
      float *b = nodes + (n0 + k) * NODE_STRIDE;
      solve_shape(b, b[ND_SX], b[ND_SY], aShape);
    }
  }
}

#if HAVE_SIMD
static v128 load4_off(const float *p0, const float *p1, const float *p2, const float *p3, int off) {
  return wasm_f32x4_make(p0[off], p1[off], p2[off], p3[off]);
}

static void store4_off(float *p0, float *p1, float *p2, float *p3, int off, v128 v) {
  p0[off] = wasm_f32x4_extract_lane(v, 0);
  p1[off] = wasm_f32x4_extract_lane(v, 1);
  p2[off] = wasm_f32x4_extract_lane(v, 2);
  p3[off] = wasm_f32x4_extract_lane(v, 3);
}

static void solve_node_link4(float *a0, float *a1, float *a2, float *a3,
                             float *b0, float *b1, float *b2, float *b3,
                             v128 rest, v128 alpha) {
  v128 ax = load4_off(a0, a1, a2, a3, ND_X);
  v128 ay = load4_off(a0, a1, a2, a3, ND_Y);
  v128 bx = load4_off(b0, b1, b2, b3, ND_X);
  v128 by = load4_off(b0, b1, b2, b3, ND_Y);
  v128 dx = wasm_f32x4_sub(bx, ax);
  v128 dy = wasm_f32x4_sub(by, ay);
  v128 dist = wasm_f32x4_sqrt(wasm_f32x4_add(wasm_f32x4_mul(dx, dx), wasm_f32x4_mul(dy, dy)));
  v128 ok = wasm_f32x4_gt(dist, wasm_f32x4_splat(1e-9f));
  v128 dist_s = wasm_v128_bitselect(dist, wasm_f32x4_splat(1.f), ok);
  v128 C = wasm_f32x4_sub(dist, rest);
  v128 w = wasm_f32x4_splat(1.f / CHAIN_MASS);
  v128 slack = wasm_f32x4_lt(C, wasm_f32x4_splat(0.f));
  v128 a_use = wasm_v128_bitselect(wasm_f32x4_mul(alpha, wasm_f32x4_splat(SLACK_RATIO)), alpha, slack);
  v128 denom = wasm_f32x4_add(wasm_f32x4_add(w, w), a_use);
  v128 lambda = wasm_f32x4_div(wasm_f32x4_neg(C), denom);
  v128 nx = wasm_f32x4_div(dx, dist_s);
  v128 ny = wasm_f32x4_div(dy, dist_s);
  v128 s = wasm_v128_bitselect(wasm_f32x4_mul(w, lambda), wasm_f32x4_splat(0.f), ok);
  store4_off(a0, a1, a2, a3, ND_X, wasm_f32x4_sub(ax, wasm_f32x4_mul(s, nx)));
  store4_off(a0, a1, a2, a3, ND_Y, wasm_f32x4_sub(ay, wasm_f32x4_mul(s, ny)));
  store4_off(b0, b1, b2, b3, ND_X, wasm_f32x4_add(bx, wasm_f32x4_mul(s, nx)));
  store4_off(b0, b1, b2, b3, ND_Y, wasm_f32x4_add(by, wasm_f32x4_mul(s, ny)));
}

static void solve_bend4(float *b0, float *b1, float *b2, float *b3,
                        v128 ax, v128 ay, v128 cx, v128 cy, v128 wA, v128 wC, v128 alpha) {
  v128 bx = load4_off(b0, b1, b2, b3, ND_X);
  v128 by = load4_off(b0, b1, b2, b3, ND_Y);
  v128 Cx = wasm_f32x4_add(wasm_f32x4_sub(ax, wasm_f32x4_mul(bx, wasm_f32x4_splat(2.f))), cx);
  v128 Cy = wasm_f32x4_add(wasm_f32x4_sub(ay, wasm_f32x4_mul(by, wasm_f32x4_splat(2.f))), cy);
  v128 wB = wasm_f32x4_splat(1.f / CHAIN_MASS);
  v128 denom = wasm_f32x4_add(wasm_f32x4_add(wA, wasm_f32x4_mul(wB, wasm_f32x4_splat(4.f))),
                              wasm_f32x4_add(wC, alpha));
  v128 ok = wasm_f32x4_gt(denom, wasm_f32x4_splat(1e-12f));
  v128 s = wasm_v128_bitselect(wasm_f32x4_div(wasm_f32x4_mul(wB, wasm_f32x4_splat(2.f)), denom),
                               wasm_f32x4_splat(0.f), ok);
  store4_off(b0, b1, b2, b3, ND_X, wasm_f32x4_add(bx, wasm_f32x4_mul(s, Cx)));
  store4_off(b0, b1, b2, b3, ND_Y, wasm_f32x4_add(by, wasm_f32x4_mul(s, Cy)));
}

static void solve_shape4(float *n0, float *n1, float *n2, float *n3, v128 alpha) {
  v128 px = load4_off(n0, n1, n2, n3, ND_X);
  v128 py = load4_off(n0, n1, n2, n3, ND_Y);
  v128 tx = load4_off(n0, n1, n2, n3, ND_SX);
  v128 ty = load4_off(n0, n1, n2, n3, ND_SY);
  v128 dx = wasm_f32x4_sub(px, tx);
  v128 dy = wasm_f32x4_sub(py, ty);
  v128 C = wasm_f32x4_sqrt(wasm_f32x4_add(wasm_f32x4_mul(dx, dx), wasm_f32x4_mul(dy, dy)));
  v128 ok = wasm_f32x4_gt(C, wasm_f32x4_splat(1e-9f));
  v128 Cs = wasm_v128_bitselect(C, wasm_f32x4_splat(1.f), ok);
  v128 w = wasm_f32x4_splat(1.f / CHAIN_MASS);
  v128 lambda = wasm_f32x4_div(wasm_f32x4_neg(C), wasm_f32x4_add(w, alpha));
  v128 s = wasm_v128_bitselect(wasm_f32x4_div(wasm_f32x4_mul(w, lambda), Cs), wasm_f32x4_splat(0.f), ok);
  store4_off(n0, n1, n2, n3, ND_X, wasm_f32x4_add(px, wasm_f32x4_mul(s, dx)));
  store4_off(n0, n1, n2, n3, ND_Y, wasm_f32x4_add(py, wasm_f32x4_mul(s, dy)));
}

static void solve_four_full(int n, const int batch[4], float h) {
  float *W[4];
  int ia[4], ja[4], si[4], sj[4], n0[4];
  float aSpan[4], aLink[4], aBend[4], aShape[4], rest[4], linkRest[4];
  int nn = (int)wires[batch[0] * WIRE_NEAR + WN_NNODES];
  int shape_all = 1;
  float invH2 = 1.f / fmaxf(1e-12f, h * h);
  for (int b = 0; b < 4; b++) {
    W[b] = wires + batch[b] * WIRE_NEAR;
    ia[b] = (int)W[b][WN_A];
    ja[b] = (int)W[b][WN_B];
    if (ia[b] < 0 || ja[b] < 0 || ia[b] >= n || ja[b] >= n || ia[b] == ja[b]) {
      for (int k = 0; k < 4; k++) solve_one_wire(n, wires + batch[k] * WIRE_NEAR, h);
      return;
    }
    si[b] = (int)W[b][WN_ASLOT];
    sj[b] = (int)W[b][WN_BSLOT];
    n0[b] = (int)W[b][WN_NODE0];
    if (n0[b] < 0 || n0[b] + nn > MAX_NODES) {
      for (int k = 0; k < 4; k++) solve_one_wire(n, wires + batch[k] * WIRE_NEAR, h);
      return;
    }
    float soft = W[b][WN_SCALE] * W[b][WN_SLACK];
    aSpan[b] = SPAN_COMP * soft * invH2;
    aLink[b] = LINK_COMP * soft * invH2;
    aBend[b] = BEND_COMP * soft * invH2;
    aShape[b] = SHAPE_COMP * soft * invH2;
    rest[b] = W[b][WN_REST];
    linkRest[b] = W[b][WN_ROPE] / (float)(nn + 1);
    if (!((int)W[b][WN_FLAGS] & WF_SHAPE)) shape_all = 0;
  }
  for (int b = 0; b < 4; b++) solve_span(ia[b], si[b], ja[b], sj[b], rest[b], aSpan[b]);
  for (int b = 0; b < 4; b++) {
    solve_body_node(ia[b], si[b], nodes + n0[b] * NODE_STRIDE, linkRest[b], aLink[b]);
  }
  v128 rest4 = wasm_f32x4_make(linkRest[0], linkRest[1], linkRest[2], linkRest[3]);
  v128 link4 = wasm_f32x4_make(aLink[0], aLink[1], aLink[2], aLink[3]);
  v128 bend4 = wasm_f32x4_make(aBend[0], aBend[1], aBend[2], aBend[3]);
  v128 shape4 = wasm_f32x4_make(aShape[0], aShape[1], aShape[2], aShape[3]);
  for (int k = 0; k < nn - 1; k++) {
    solve_node_link4(nodes + (n0[0] + k) * NODE_STRIDE, nodes + (n0[1] + k) * NODE_STRIDE,
                     nodes + (n0[2] + k) * NODE_STRIDE, nodes + (n0[3] + k) * NODE_STRIDE,
                     nodes + (n0[0] + k + 1) * NODE_STRIDE, nodes + (n0[1] + k + 1) * NODE_STRIDE,
                     nodes + (n0[2] + k + 1) * NODE_STRIDE, nodes + (n0[3] + k + 1) * NODE_STRIDE,
                     rest4, link4);
  }
  for (int b = 0; b < 4; b++) {
    solve_body_node(ja[b], sj[b], nodes + (n0[b] + nn - 1) * NODE_STRIDE, linkRest[b], aLink[b]);
  }
  float sAx[4], sAy[4], sBx[4], sBy[4];
  for (int b = 0; b < 4; b++) {
    float rAx, rAy, rBx, rBy;
    attach(ia[b], si[b], &rAx, &rAy);
    attach(ja[b], sj[b], &rBx, &rBy);
    sAx[b] = bodies[ia[b] * STRIDE + FAR_X] + rAx;
    sAy[b] = bodies[ia[b] * STRIDE + FAR_Y] + rAy;
    sBx[b] = bodies[ja[b] * STRIDE + FAR_X] + rBx;
    sBy[b] = bodies[ja[b] * STRIDE + FAR_Y] + rBy;
  }
  v128 stemAx = wasm_f32x4_make(sAx[0], sAx[1], sAx[2], sAx[3]);
  v128 stemAy = wasm_f32x4_make(sAy[0], sAy[1], sAy[2], sAy[3]);
  v128 stemBx = wasm_f32x4_make(sBx[0], sBx[1], sBx[2], sBx[3]);
  v128 stemBy = wasm_f32x4_make(sBy[0], sBy[1], sBy[2], sBy[3]);
  v128 wNode = wasm_f32x4_splat(1.f / CHAIN_MASS);
  v128 z = wasm_f32x4_splat(0.f);
  for (int k = 0; k < nn; k++) {
    v128 px, py, nx, ny, wP, wN;
    if (k == 0) {
      px = stemAx;
      py = stemAy;
      wP = z;
    } else {
      px = load4_off(nodes + (n0[0] + k - 1) * NODE_STRIDE, nodes + (n0[1] + k - 1) * NODE_STRIDE,
                     nodes + (n0[2] + k - 1) * NODE_STRIDE, nodes + (n0[3] + k - 1) * NODE_STRIDE, ND_X);
      py = load4_off(nodes + (n0[0] + k - 1) * NODE_STRIDE, nodes + (n0[1] + k - 1) * NODE_STRIDE,
                     nodes + (n0[2] + k - 1) * NODE_STRIDE, nodes + (n0[3] + k - 1) * NODE_STRIDE, ND_Y);
      wP = wNode;
    }
    if (k == nn - 1) {
      nx = stemBx;
      ny = stemBy;
      wN = z;
    } else {
      nx = load4_off(nodes + (n0[0] + k + 1) * NODE_STRIDE, nodes + (n0[1] + k + 1) * NODE_STRIDE,
                     nodes + (n0[2] + k + 1) * NODE_STRIDE, nodes + (n0[3] + k + 1) * NODE_STRIDE, ND_X);
      ny = load4_off(nodes + (n0[0] + k + 1) * NODE_STRIDE, nodes + (n0[1] + k + 1) * NODE_STRIDE,
                     nodes + (n0[2] + k + 1) * NODE_STRIDE, nodes + (n0[3] + k + 1) * NODE_STRIDE, ND_Y);
      wN = wNode;
    }
    solve_bend4(nodes + (n0[0] + k) * NODE_STRIDE, nodes + (n0[1] + k) * NODE_STRIDE,
                nodes + (n0[2] + k) * NODE_STRIDE, nodes + (n0[3] + k) * NODE_STRIDE,
                px, py, nx, ny, wP, wN, bend4);
  }
  if (shape_all) {
    for (int k = 0; k < nn; k++) {
      solve_shape4(nodes + (n0[0] + k) * NODE_STRIDE, nodes + (n0[1] + k) * NODE_STRIDE,
                   nodes + (n0[2] + k) * NODE_STRIDE, nodes + (n0[3] + k) * NODE_STRIDE, shape4);
    }
  } else {
    for (int b = 0; b < 4; b++) {
      if (!((int)W[b][WN_FLAGS] & WF_SHAPE)) continue;
      for (int k = 0; k < nn; k++) {
        float *nd = nodes + (n0[b] + k) * NODE_STRIDE;
        solve_shape(nd, nd[ND_SX], nd[ND_SY], aShape[b]);
      }
    }
  }
}
#endif

static int wires_share(int u, int v) {
  int ua = (int)wires[u * WIRE_NEAR + WN_A], ub = (int)wires[u * WIRE_NEAR + WN_B];
  int va = (int)wires[v * WIRE_NEAR + WN_A], vb = (int)wires[v * WIRE_NEAR + WN_B];
  return ua == va || ua == vb || ub == va || ub == vb;
}

static void solve_wires_batched(int n, int n_wires, float h) {
  memset(wdone, 0, (size_t)n_wires);
  int cursor = 0;
  while (cursor < n_wires) {
    while (cursor < n_wires && wdone[cursor]) cursor++;
    if (cursor >= n_wires) break;
    if ((int)wires[cursor * WIRE_NEAR + WN_FLAGS] & WF_SKIP) {
      wdone[cursor] = 1;
      continue;
    }
    int batch[4];
    int nb = 0;
    for (int w = cursor; w < n_wires && nb < 4; w++) {
      if (wdone[w]) continue;
      if ((int)wires[w * WIRE_NEAR + WN_FLAGS] & WF_SKIP) {
        wdone[w] = 1;
        continue;
      }
      int conflict = 0;
      for (int k = cursor; k < w; k++) {
        if (wdone[k]) continue;
        int inb = 0;
        for (int b = 0; b < nb; b++) {
          if (batch[b] == k) {
            inb = 1;
            break;
          }
        }
        if (inb) continue;
        if (wires_share(k, w)) {
          conflict = 1;
          break;
        }
      }
      if (conflict) continue;
      for (int b = 0; b < nb; b++) {
        if (wires_share(batch[b], w)) {
          conflict = 1;
          break;
        }
      }
      if (conflict) continue;
      batch[nb++] = w;
    }
#if HAVE_SIMD
    if (nb == 4) {
      int nn0 = (int)wires[batch[0] * WIRE_NEAR + WN_NNODES];
      int ok = ((int)wires[batch[0] * WIRE_NEAR + WN_FLAGS] & WF_FULL) && nn0 >= 2;
      for (int b = 1; b < 4 && ok; b++) {
        int f = (int)wires[batch[b] * WIRE_NEAR + WN_FLAGS];
        int nn = (int)wires[batch[b] * WIRE_NEAR + WN_NNODES];
        if (!(f & WF_FULL) || nn != nn0) ok = 0;
      }
      if (ok) {
        solve_four_full(n, batch, h);
        for (int b = 0; b < 4; b++) wdone[batch[b]] = 1;
        continue;
      }
    }
#endif
    for (int b = 0; b < nb; b++) {
      solve_one_wire(n, wires + batch[b] * WIRE_NEAR, h);
      wdone[batch[b]] = 1;
    }
  }
}

static void integrate_nodes(int n_wires, float h) {
  for (int w = 0; w < n_wires; w++) {
    float *W = wires + w * WIRE_NEAR;
    int flags = (int)W[WN_FLAGS];
    if (!(flags & WF_FULL)) continue;
    int n0 = (int)W[WN_NODE0];
    int nn = (int)W[WN_NNODES];
    if (n0 < 0 || nn <= 0) continue;
    if (n0 + nn > MAX_NODES) nn = MAX_NODES - n0;
    int hold = flags & WF_HOLD;
    for (int k = 0; k < nn; k++) {
      float *nd = nodes + (n0 + k) * NODE_STRIDE;
      nd[ND_PREVX] = nd[ND_X];
      nd[ND_PREVY] = nd[ND_Y];
      if (hold) continue;
      nd[ND_X] += nd[ND_VX] * h;
      nd[ND_Y] += nd[ND_VY] * h;
    }
  }
}

static void finalize_nodes(int n_wires, float h, float rope_keep) {
  float invh = 1.f / h;
  for (int w = 0; w < n_wires; w++) {
    float *W = wires + w * WIRE_NEAR;
    int flags = (int)W[WN_FLAGS];
    if (!(flags & WF_FULL)) continue;
    int n0 = (int)W[WN_NODE0];
    int nn = (int)W[WN_NNODES];
    if (n0 < 0 || nn <= 0) continue;
    if (n0 + nn > MAX_NODES) nn = MAX_NODES - n0;
    for (int k = 0; k < nn; k++) {
      float *nd = nodes + (n0 + k) * NODE_STRIDE;
      nd[ND_VX] = (nd[ND_X] - nd[ND_PREVX]) * invh * rope_keep;
      nd[ND_VY] = (nd[ND_Y] - nd[ND_PREVY]) * invh * rope_keep;
    }
  }
}

static void tri_world_d(int i, double vx[3], double vy[3]) {
  double sc = (double)scale[i];
  double s = 16.0 * sc;
  double lx0 = s * 1.05, ly0 = 0.0;
  double lx1 = -s * 0.55, ly1 = -s * 0.82;
  double lx2 = -s * 0.55, ly2 = s * 0.82;
  double a = (double)bodies[i * STRIDE + FAR_HEADING];
  double c = cos(a), sn = sin(a);
  double x = (double)bodies[i * STRIDE + FAR_X];
  double y = (double)bodies[i * STRIDE + FAR_Y];
  vx[0] = x + lx0 * c - ly0 * sn;
  vy[0] = y + lx0 * sn + ly0 * c;
  vx[1] = x + lx1 * c - ly1 * sn;
  vy[1] = y + lx1 * sn + ly1 * c;
  vx[2] = x + lx2 * c - ly2 * sn;
  vy[2] = y + lx2 * sn + ly2 * c;
}

static void add_poly_axes_d(const double *vx, const double *vy, int n, double *ax, double *ay, int *nax) {
  for (int i = 0; i < n; i++) {
    int j = i + 1;
    if (j == n) j = 0;
    double ex = vx[j] - vx[i];
    double ey = vy[j] - vy[i];
    double len = sqrt(ex * ex + ey * ey);
    if (len < 1e-12) len = 1.0;
    int k = *nax;
    ax[k] = -ey / len;
    ay[k] = ex / len;
    *nax = k + 1;
  }
}

static void project_poly_d(const double *vx, const double *vy, int n, double nx, double ny, double *mn, double *mx) {
  double lo = vx[0] * nx + vy[0] * ny;
  double hi = lo;
  for (int i = 1; i < n; i++) {
    double d = vx[i] * nx + vy[i] * ny;
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  *mn = lo - (double)SKIN;
  *mx = hi + (double)SKIN;
}

static void project_circle_d(double x, double y, double r, double nx, double ny, double *mn, double *mx) {
  double m = x * nx + y * ny;
  double rad = r + (double)SKIN;
  *mn = m - rad;
  *mx = m + rad;
}

static int overlap_axis_d(double amin, double amax, double bmin, double bmax, double *o) {
  double left = bmax - amin;
  double right = amax - bmin;
  if (left <= 0.0 || right <= 0.0) return 0;
  *o = left < right ? left : right;
  return 1;
}

static void closest_on_seg_d(double px, double py, double ax, double ay, double bx, double by, double *qx, double *qy) {
  double abx = bx - ax;
  double aby = by - ay;
  double denom = abx * abx + aby * aby;
  if (denom < 1e-12) denom = 1.0;
  double t = ((px - ax) * abx + (py - ay) * aby) / denom;
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;
  *qx = ax + abx * t;
  *qy = ay + aby * t;
}

static void closest_on_poly_d(double px, double py, const double *vx, const double *vy, int n, double *qx, double *qy) {
  double bx, by;
  closest_on_seg_d(px, py, vx[0], vy[0], vx[1], vy[1], &bx, &by);
  double best = (bx - px) * (bx - px) + (by - py) * (by - py);
  *qx = bx;
  *qy = by;
  for (int i = 1; i < n; i++) {
    int j = i + 1;
    if (j == n) j = 0;
    closest_on_seg_d(px, py, vx[i], vy[i], vx[j], vy[j], &bx, &by);
    double d = (bx - px) * (bx - px) + (by - py) * (by - py);
    if (d < best) {
      best = d;
      *qx = bx;
      *qy = by;
    }
  }
}

static void support_poly_d(const double *vx, const double *vy, int n, double nx, double ny, double *px, double *py) {
  double best = vx[0] * nx + vy[0] * ny;
  int bi = 0;
  for (int i = 1; i < n; i++) {
    double d = vx[i] * nx + vy[i] * ny;
    if (d > best) {
      best = d;
      bi = i;
    }
  }
  *px = vx[bi];
  *py = vy[bi];
}

static int sat_hit(int i, int j, float *nx_out, float *ny_out, float *o_out, float *px_out, float *py_out) {
  float *pi = bodies + i * STRIDE;
  float *pj = bodies + j * STRIDE;
  double ax = (double)pi[FAR_X], ay = (double)pi[FAR_Y];
  double bx = (double)pj[FAR_X], by = (double)pj[FAR_Y];
  double toBx = bx - ax, toBy = by - ay;
  double dist = sqrt(toBx * toBx + toBy * toBy);
  if (dist > (double)pi[FAR_RADIUS] + (double)pj[FAR_RADIUS] + (double)SKIN * 2.0 + 2.0) return 0;

  int eraA = kind[i] == 0;
  int eraB = kind[j] == 0;
  double ra = (double)ERA_R * (double)scale[i];
  double rb = (double)ERA_R * (double)scale[j];

  if (eraA && eraB) {
    double minDist = ra + rb + (double)SKIN * 2.0;
    if (dist >= minDist) return 0;
    if (dist < 1e-6) {
      *nx_out = 1.f;
      *ny_out = 0.f;
      *o_out = (float)minDist;
      *px_out = (float)(ax + ra);
      *py_out = (float)ay;
      return 1;
    }
    double nx = toBx / dist, ny = toBy / dist;
    *nx_out = (float)nx;
    *ny_out = (float)ny;
    *o_out = (float)(minDist - dist);
    *px_out = (float)(ax + nx * ra);
    *py_out = (float)(ay + ny * ra);
    return 1;
  }

  double avx[3], avy[3], bvx[3], bvy[3];
  int na = 0, nb = 0;
  if (!eraA) {
    tri_world_d(i, avx, avy);
    na = 3;
  }
  if (!eraB) {
    tri_world_d(j, bvx, bvy);
    nb = 3;
  }

  double axes_x[8], axes_y[8];
  int nax = 0;
  if (na) add_poly_axes_d(avx, avy, na, axes_x, axes_y, &nax);
  if (nb) add_poly_axes_d(bvx, bvy, nb, axes_x, axes_y, &nax);
  if (eraA && nb) {
    double qx, qy;
    closest_on_poly_d(ax, ay, bvx, bvy, nb, &qx, &qy);
    double dx = qx - ax, dy = qy - ay;
    double len = sqrt(dx * dx + dy * dy);
    if (len > 1e-6) {
      axes_x[nax] = dx / len;
      axes_y[nax] = dy / len;
      nax++;
    }
  } else if (eraB && na) {
    double qx, qy;
    closest_on_poly_d(bx, by, avx, avy, na, &qx, &qy);
    double dx = qx - bx, dy = qy - by;
    double len = sqrt(dx * dx + dy * dy);
    if (len > 1e-6) {
      axes_x[nax] = dx / len;
      axes_y[nax] = dy / len;
      nax++;
    }
  }

  double bestO = 1e30;
  double nx = 1.0, ny = 0.0;
  for (int k = 0; k < nax; k++) {
    double ux = axes_x[k], uy = axes_y[k];
    double amin, amax, bmin, bmax, o;
    if (eraA) project_circle_d(ax, ay, ra, ux, uy, &amin, &amax);
    else project_poly_d(avx, avy, na, ux, uy, &amin, &amax);
    if (eraB) project_circle_d(bx, by, rb, ux, uy, &bmin, &bmax);
    else project_poly_d(bvx, bvy, nb, ux, uy, &bmin, &bmax);
    if (!overlap_axis_d(amin, amax, bmin, bmax, &o)) return 0;
    double len = sqrt(ux * ux + uy * uy);
    if (len < 1e-12) len = 1.0;
    ux /= len;
    uy /= len;
    if (ux * toBx + uy * toBy < 0.0) {
      ux = -ux;
      uy = -uy;
    }
    if (o < bestO) {
      bestO = o;
      nx = ux;
      ny = uy;
    }
  }
  if (!(bestO < 1e29) || bestO <= 0.0) return 0;

  double pax, pay, pbx, pby;
  if (eraA) {
    pax = ax + nx * ra;
    pay = ay + ny * ra;
  } else {
    support_poly_d(avx, avy, na, nx, ny, &pax, &pay);
  }
  if (eraB) {
    pbx = bx - nx * rb;
    pby = by - ny * rb;
  } else {
    support_poly_d(bvx, bvy, nb, -nx, -ny, &pbx, &pby);
  }
  *nx_out = (float)nx;
  *ny_out = (float)ny;
  *o_out = (float)bestO;
  *px_out = (float)((pax + pbx) * 0.5);
  *py_out = (float)((pay + pby) * 0.5);
  return 1;
}

static void record_hit(int i, int j, float nx, float ny, float overlap, float px, float py) {
  if (g_hits >= MAX_HITS) return;
  float *pi = bodies + i * STRIDE;
  float *pj = bodies + j * STRIDE;
  float rAx = px - pi[FAR_X], rAy = py - pi[FAR_Y];
  float rBx = px - pj[FAR_X], rBy = py - pj[FAR_Y];
  float wA = gen_inv(i, rAx, rAy, nx, ny);
  float wB = gen_inv(j, rBx, rBy, nx, ny);
  float eff = 1.f / fmaxf(1e-9f, wA + wB);
  float pax = pi[FAR_VX] - pi[FAR_OMEGA] * rAy;
  float pay = pi[FAR_VY] + pi[FAR_OMEGA] * rAx;
  float pbx = pj[FAR_VX] - pj[FAR_OMEGA] * rBy;
  float pby = pj[FAR_VY] + pj[FAR_OMEGA] * rBx;
  float rvx = pax - pbx;
  float rvy = pay - pby;
  float *h = hits + g_hits * HIT_STRIDE;
  h[0] = (float)i;
  h[1] = (float)j;
  h[2] = nx;
  h[3] = ny;
  h[4] = overlap;
  h[5] = px;
  h[6] = py;
  h[7] = -(rvx * nx + rvy * ny);
  h[8] = rvx * -ny + rvy * nx;
  h[9] = eff;
  g_hits++;
}

static void solve_sat_contact(int i, int j, float h) {
  float nx, ny, overlap, px, py;
  if (!sat_hit(i, j, &nx, &ny, &overlap, &px, &py)) return;
  record_hit(i, j, nx, ny, overlap, px, py);
  float depth = overlap - SLOP;
  if (depth <= 0.f) return;
  float *pi = bodies + i * STRIDE;
  float *pj = bodies + j * STRIDE;
  float rAx = px - pi[FAR_X], rAy = py - pi[FAR_Y];
  float rBx = px - pj[FAR_X], rBy = py - pj[FAR_Y];
  float wA = gen_inv(i, rAx, rAy, nx, ny);
  float wB = gen_inv(j, rBx, rBy, nx, ny);
  float denom = wA + wB + CONTACT_COMP / fmaxf(1e-12f, h * h);
  if (denom < 1e-12f) return;
  float lambda = depth / denom;
  apply_imp(i, rAx, rAy, -nx, -ny, lambda);
  apply_imp(j, rBx, rBy, nx, ny, lambda);
}

static void solve_grab(int held, float gx, float gy, float h) {
  if (held < 0 || held >= MAX_BODIES) return;
  float *p = bodies + held * STRIDE;
  if (p[FAR_LOCKED] >= 0.5f) return;
  float dx = gx - p[FAR_X];
  float dy = gy - p[FAR_Y];
  float dist = sqrtf(dx * dx + dy * dy);
  if (dist < 1e-6f) return;
  float w = p[FAR_INVMASS];
  if (w <= 0.f) return;
  float alpha = GRAB_COMP / fmaxf(1e-12f, h * h);
  float step = (dist * w) / (w + alpha);
  if (step > GRAB_STEP) step = GRAB_STEP;
  p[FAR_X] += (dx / dist) * step;
  p[FAR_Y] += (dy / dist) * step;
}

static void grab_cap(int n, int held, float grab_max) {
  if (held < 0 || held >= n) return;
  float *p = bodies + held * STRIDE;
  float speed = sqrtf(p[FAR_VX] * p[FAR_VX] + p[FAR_VY] * p[FAR_VY]);
  if (speed > grab_max && grab_max > 0.f) {
    float k = grab_max / speed;
    p[FAR_VX] *= k;
    p[FAR_VY] *= k;
  }
}

static void flock_locomote(int i, float wish_x, float wish_y, float turn_k) {
  if (!swim[i]) return;
  float *p = bodies + i * STRIDE;
  float a = p[FAR_HEADING];
  float hx = cosf(a), hy = sinf(a);
  float ahead = wish_x * hx + wish_y * hy;
  if (ahead > 0.f) {
    p[FAR_VX] += ahead * hx;
    p[FAR_VY] += ahead * hy;
  }
  float mag = sqrtf(wish_x * wish_x + wish_y * wish_y);
  if (mag > 1e-8f && turn_k != 0.f) {
    p[FAR_OMEGA] += turn_k * wrap_angle(atan2f(wish_y, wish_x) - a);
  }
}

static void flock_pull(int i, float wish_x, float wish_y) {
  float *p = bodies + i * STRIDE;
  if (p[FAR_LOCKED] >= 0.5f) return;
  p[FAR_VX] += wish_x;
  p[FAR_VY] += wish_y;
}

static void flock_force(int i, float wish_x, float wish_y, float turn_k) {
  if (swim[i]) flock_locomote(i, wish_x, wish_y, turn_k);
  else flock_pull(i, wish_x, wish_y);
}

void solver_flock(int n, float align, float sep, float dt, float turn_rate, float desired, int max_hops) {
  if (n <= 0 || dt <= 0.f) return;
  if (n > MAX_BODIES) n = MAX_BODIES;
  if (max_hops < 0) max_hops = 0;
  if (max_hops > 16) max_hops = 16;
  for (int i = 0; i < n; i++) flock_dist[i] = -1;
  for (int start = 0; start < n; start++) {
    if (bodies[start * STRIDE + FAR_LOCKED] >= 0.5f) continue;
    int seen_n = 0;
    flock_dist[start] = 0;
    flock_seen[seen_n++] = start;
    int qh = 0, qt = 0;
    flock_q[qt++] = start;
    while (qh < qt) {
      int u = flock_q[qh++];
      int du = flock_dist[u];
      if (du >= max_hops) continue;
      int a0 = adj_off[u], a1 = adj_off[u + 1];
      if (a0 < 0) a0 = 0;
      if (a1 > MAX_WIRES * 2) a1 = MAX_WIRES * 2;
      if (a0 > a1) continue;
      for (int k = a0; k < a1; k++) {
        int v = adj_nei[k];
        if (v < 0 || v >= n || flock_dist[v] >= 0) continue;
        int d = du + 1;
        flock_dist[v] = d;
        flock_seen[seen_n++] = v;
        flock_q[qt++] = v;
        if (bodies[v * STRIDE + FAR_LOCKED] >= 0.5f || flock_id[v] <= flock_id[start]) continue;
        float w = 1.f / (float)d;
        float mA = flock_mass[start];
        float mB = flock_mass[v];
        if (mA < 0.08f) mA = 0.08f;
        if (mB < 0.08f) mB = 0.08f;
        float mSum = mA + mB;
        float *A = bodies + start * STRIDE;
        float *B = bodies + v * STRIDE;
        float dx = B[FAR_X] - A[FAR_X];
        float dy = B[FAR_Y] - A[FAR_Y];
        float gap = sqrtf(dx * dx + dy * dy);
        if (gap < 1e-6f) gap = 1e-6f;
        float nx = dx / gap, ny = dy / gap;
        if (align > 0.f) {
          float kAlign = align * w * dt;
          float dvx = B[FAR_VX] - A[FAR_VX];
          float dvy = B[FAR_VY] - A[FAR_VY];
          flock_force(start, dvx * kAlign * (mB / mSum), dvy * kAlign * (mB / mSum), 0.f);
          flock_force(v, -dvx * kAlign * (mA / mSum), -dvy * kAlign * (mA / mSum), 0.f);
        }
        if (sep > 0.f && d > 1) {
          float want = 22.f + (float)(d - 1) * desired;
          if (gap < want) {
            float mag = sep * w * (want - gap);
            float ax = nx * mag * dt;
            float ay = ny * mag * dt;
            float turn = turn_rate * w * 0.25f;
            flock_force(start, -ax * (mB / mSum), -ay * (mB / mSum), swim[start] ? 0.f : turn);
            flock_force(v, ax * (mA / mSum), ay * (mA / mSum), swim[v] ? 0.f : turn);
          }
        }
      }
    }
    for (int i = 0; i < seen_n; i++) flock_dist[flock_seen[i]] = -1;
  }
}

static int near_contacts(int n, float h, int reset_hits, int rebuild_pairs) {
  if (reset_hits) g_hits = 0;
  if (n <= 0 || h <= 0.f) return 0;
  if (n > MAX_BODIES) n = MAX_BODIES;
  memset(delta, 0, (size_t)n * 2 * sizeof(float));
  int np = g_pairs;
  if (rebuild_pairs || np <= 0) {
    float maxr = 0.f;
    for (int i = 0; i < n; i++) {
      float r = bodies[i * STRIDE + FAR_RADIUS];
      if (r > maxr) maxr = r;
    }
    np = collect_pairs(n, maxr * 2.f + SLOP + 4.f);
    g_pairs = np;
  }
  float alpha = CONTACT_COMP / fmaxf(1e-12f, h * h);
  for (int p = 0; p < np; p++) {
    int i = pair_a[p], j = pair_b[p];
    float *pi = bodies + i * STRIDE;
    float *pj = bodies + j * STRIDE;
    if (pi[FAR_LOCKED] >= 0.5f && pj[FAR_LOCKED] >= 0.5f) continue;
    if (detailed[i] || detailed[j]) {
      solve_sat_contact(i, j, h);
      continue;
    }
    float dx = pj[FAR_X] - pi[FAR_X];
    float dy = pj[FAR_Y] - pi[FAR_Y];
    float dist = sqrtf(dx * dx + dy * dy);
    float keep = pi[FAR_RADIUS] + pj[FAR_RADIUS];
    if (dist >= keep || dist < 1e-6f) continue;
    float depth = keep - dist - SLOP;
    if (depth <= 0.f) continue;
    float wA = pi[FAR_INVMASS], wB = pj[FAR_INVMASS];
    float denom = wA + wB + alpha;
    if (denom < 1e-12f) continue;
    float lam = depth / denom;
    float s = lam / dist;
    if (pi[FAR_LOCKED] < 0.5f && wA > 0.f) {
      delta[i * 2] -= dx * s * wA;
      delta[i * 2 + 1] -= dy * s * wA;
    }
    if (pj[FAR_LOCKED] < 0.5f && wB > 0.f) {
      delta[j * 2] += dx * s * wB;
      delta[j * 2 + 1] += dy * s * wB;
    }
  }
  apply(n);
  return np;
}

float *solver_bodies(void) { return bodies; }
float *solver_wires(void) { return wires; }
float *solver_nodes(void) { return nodes; }
float *solver_inv_inertia(void) { return inv_inertia; }
float *solver_scale(void) { return scale; }
uint8_t *solver_kind(void) { return kind; }
uint8_t *solver_detailed(void) { return detailed; }
int32_t *solver_pair_a(void) { return pair_a; }
int32_t *solver_pair_b(void) { return pair_b; }
int solver_cap(void) { return MAX_BODIES; }
int solver_wire_cap(void) { return MAX_WIRES; }
int solver_node_cap(void) { return MAX_NODES; }
int solver_pair_cap(void) { return MAX_PAIRS; }
int solver_pair_count(void) { return g_pairs; }
int solver_wire_near_stride(void) { return WIRE_NEAR; }
int solver_node_stride(void) { return NODE_STRIDE; }
float *solver_hits(void) { return hits; }
int solver_hit_count(void) { return g_hits; }
int solver_hit_stride(void) { return HIT_STRIDE; }
int solver_hit_cap(void) { return MAX_HITS; }
int32_t *solver_adj_off(void) { return adj_off; }
int32_t *solver_adj_nei(void) { return adj_nei; }
int32_t *solver_flock_id(void) { return flock_id; }
float *solver_flock_mass(void) { return flock_mass; }
uint8_t *solver_swim(void) { return swim; }
int solver_adj_cap(void) { return MAX_WIRES * 2; }

void solver_step_far(int n, int n_wires, float dt, int substeps) {
  if (n <= 0 || dt <= 0.f || substeps <= 0) return;
  if (n > MAX_BODIES) n = MAX_BODIES;
  if (n_wires > MAX_WIRES) n_wires = MAX_WIRES;
  if (n_wires < 0) n_wires = 0;
  float h = dt / (float)substeps;
  for (int s = 0; s < substeps; s++) {
    integrate(n, h);
    disc(n, h);
    apply(n);
    if (n_wires > 0) {
      span(n, n_wires, h);
      apply(n);
    }
    finalize(n, h);
  }
}

void solver_near_integrate(int n, int n_wires, float h) {
  if (n <= 0 || h <= 0.f) return;
  if (n > MAX_BODIES) n = MAX_BODIES;
  if (n_wires > MAX_WIRES) n_wires = MAX_WIRES;
  if (n_wires < 0) n_wires = 0;
  integrate(n, h);
  integrate_nodes(n_wires, h);
}

void solver_near_wires(int n, int n_wires, float h) {
  if (n <= 0 || h <= 0.f || n_wires <= 0) return;
  if (n > MAX_BODIES) n = MAX_BODIES;
  if (n_wires > MAX_WIRES) n_wires = MAX_WIRES;
  refresh_poses(n);
  solve_wires_batched(n, n_wires, h);
}

int solver_near_disc(int n, float h) {
  g_pairs = 0;
  if (n <= 0 || h <= 0.f) return 0;
  if (n > MAX_BODIES) n = MAX_BODIES;
  memset(delta, 0, (size_t)n * 2 * sizeof(float));
  float maxr = 0.f;
  for (int i = 0; i < n; i++) {
    float r = bodies[i * STRIDE + FAR_RADIUS];
    if (r > maxr) maxr = r;
  }
  int np = collect_pairs(n, maxr * 2.f + SLOP + 4.f);
  g_pairs = np;
  float alpha = CONTACT_COMP / fmaxf(1e-12f, h * h);
  for (int p = 0; p < np; p++) {
    int i = pair_a[p], j = pair_b[p];
    if (detailed[i] || detailed[j]) continue;
    float *pi = bodies + i * STRIDE;
    float *pj = bodies + j * STRIDE;
    if (pi[FAR_LOCKED] >= 0.5f && pj[FAR_LOCKED] >= 0.5f) continue;
    float dx = pj[FAR_X] - pi[FAR_X];
    float dy = pj[FAR_Y] - pi[FAR_Y];
    float dist = sqrtf(dx * dx + dy * dy);
    float keep = pi[FAR_RADIUS] + pj[FAR_RADIUS];
    if (dist >= keep || dist < 1e-6f) continue;
    float depth = keep - dist - SLOP;
    if (depth <= 0.f) continue;
    float wA = pi[FAR_INVMASS], wB = pj[FAR_INVMASS];
    float denom = wA + wB + alpha;
    if (denom < 1e-12f) continue;
    float lam = depth / denom;
    float s = lam / dist;
    if (pi[FAR_LOCKED] < 0.5f && wA > 0.f) {
      delta[i * 2] -= dx * s * wA;
      delta[i * 2 + 1] -= dy * s * wA;
    }
    if (pj[FAR_LOCKED] < 0.5f && wB > 0.f) {
      delta[j * 2] += dx * s * wB;
      delta[j * 2 + 1] += dy * s * wB;
    }
  }
  apply(n);
  return np;
}

void solver_near_finalize(int n, int n_wires, float h, float rope_keep, int held, float grab_max) {
  if (n <= 0 || h <= 0.f) return;
  if (n > MAX_BODIES) n = MAX_BODIES;
  if (n_wires > MAX_WIRES) n_wires = MAX_WIRES;
  if (n_wires < 0) n_wires = 0;
  finalize(n, h);
  grab_cap(n, held, grab_max);
  finalize_nodes(n_wires, h, rope_keep);
}

int solver_near_contacts(int n, float h) {
  return near_contacts(n, h, 1, 1);
}

void solver_step_near(int n, int n_wires, float dt, int substeps,
                      float rope_keep, int held, float grab_max,
                      float gx, float gy) {
  if (n <= 0 || dt <= 0.f || substeps <= 0) return;
  if (n > MAX_BODIES) n = MAX_BODIES;
  if (n_wires > MAX_WIRES) n_wires = MAX_WIRES;
  if (n_wires < 0) n_wires = 0;
  float h = dt / (float)substeps;
  g_hits = 0;
  int have_pairs = 0;
  for (int s = 0; s < substeps; s++) {
    integrate(n, h);
    integrate_nodes(n_wires, h);
    refresh_poses(n);
    solve_wires_batched(n, n_wires, h);
    solve_grab(held, gx, gy, h);
    near_contacts(n, h, 0, !have_pairs);
    have_pairs = 1;
    finalize(n, h);
    grab_cap(n, held, grab_max);
    finalize_nodes(n_wires, h, rope_keep);
  }
}

float *solver_scent(void) { return scent; }
float *solver_scent_tmp(void) { return scent_tmp; }
uint8_t *solver_walls(void) { return walls; }
int solver_scent_cap(void) { return MAX_SCENT; }
int solver_wall_cap(void) { return MAX_COLS * MAX_ROWS; }

void solver_scent_diffuse(int cols, int rows, float mix) {
  if (mix <= 0.f || cols <= 0 || rows <= 0) return;
  if (cols > MAX_COLS) cols = MAX_COLS;
  if (rows > MAX_ROWS) rows = MAX_ROWS;
  float m = mix;
  float keep = 1.f - m;
  int row_stride = cols * CHANNELS;
  const float *src = scent;
  float *dst = scent_tmp;
  for (int j = 0; j < rows; j++) {
    int has_up = j > 0;
    int has_down = j < rows - 1;
    for (int i = 0; i < cols; i++) {
      int cell = j * cols + i;
      int base = cell * CHANNELS;
      if (walls[cell]) {
#if HAVE_SIMD
        wasm_v128_store(dst + base, wasm_v128_load(src + base));
#else
        dst[base] = src[base];
        dst[base + 1] = src[base + 1];
        dst[base + 2] = src[base + 2];
        dst[base + 3] = src[base + 3];
#endif
        continue;
      }
      int left = (i > 0 && !walls[cell - 1]) ? base - CHANNELS : -1;
      int right = (i < cols - 1 && !walls[cell + 1]) ? base + CHANNELS : -1;
      int up = (has_up && !walls[cell - cols]) ? base - row_stride : -1;
      int down = (has_down && !walls[cell + cols]) ? base + row_stride : -1;
#if HAVE_SIMD
      v128 self = wasm_v128_load(src + base);
      v128 a = left >= 0 ? wasm_v128_load(src + left) : self;
      v128 b = right >= 0 ? wasm_v128_load(src + right) : self;
      v128 c = up >= 0 ? wasm_v128_load(src + up) : self;
      v128 e = down >= 0 ? wasm_v128_load(src + down) : self;
      v128 sum = wasm_f32x4_add(wasm_f32x4_add(a, b), wasm_f32x4_add(c, e));
      v128 out = wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(keep), self),
                                wasm_f32x4_mul(wasm_f32x4_splat(m * 0.25f), sum));
      wasm_v128_store(dst + base, out);
#else
      for (int ch = 0; ch < CHANNELS; ch++) {
        int k = base + ch;
        float self = src[k];
        float a = left >= 0 ? src[left + ch] : self;
        float b = right >= 0 ? src[right + ch] : self;
        float c = up >= 0 ? src[up + ch] : self;
        float e = down >= 0 ? src[down + ch] : self;
        dst[k] = keep * self + m * (a + b + c + e) * 0.25f;
      }
#endif
    }
  }
  memcpy(scent, scent_tmp, (size_t)cols * (size_t)rows * CHANNELS * sizeof(float));
}

void solver_scent_decay(int n, float keep) {
  if (n <= 0) return;
  if (n > MAX_SCENT) n = MAX_SCENT;
  if (keep >= 1.f) return;
  if (keep <= 0.f) {
    memset(scent, 0, (size_t)n * sizeof(float));
    return;
  }
#if HAVE_SIMD
  v128 k4 = wasm_f32x4_splat(keep);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    wasm_v128_store(scent + i, wasm_f32x4_mul(wasm_v128_load(scent + i), k4));
  }
  for (; i < n; i++) scent[i] *= keep;
#else
  for (int i = 0; i < n; i++) scent[i] *= keep;
#endif
}
