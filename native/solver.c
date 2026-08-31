/*
 * Packed FAR solver + NEAR XPBD + scent field. No libc heap; static buffers.
 *
 * Body layout matches src/gpu/far-kernel.ts (12 floats / body).
 * FAR wires are 4 floats. NEAR wires are 12 floats in the same buffer.
 * Discs use a spatial hash, not N². FAR chord span is one Jacobi impulse.
 * NEAR wires are Gauss-Seidel XPBD (span + links + bend + shape) matching
 * src/chain.ts. FAR-FAR discs skip detailed bodies so JS can SAT those pairs.
 *
 * Rebuild: npm run wasm
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAX_BODIES 16384
#define MAX_WIRES 8192
#define MAX_NODES 65536
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
#define CONTACT_COMP 4.0e-6f
#define SPAN_COMP 3.0e-6f
#define LINK_COMP 2.0e-6f
#define BEND_COMP 1.5e-4f
#define SHAPE_COMP 3.0e-4f
#define CHAIN_MASS 0.08f
#define SLACK_RATIO 2.0f
#define PI 3.14159265f
#define TAU 6.2831853f

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

static float wrap_angle(float a) {
  if (a >= -PI && a < PI) return a;
  float r = fmodf(a, TAU);
  if (r >= PI) r -= TAU;
  if (r < -PI) r += TAU;
  return r;
}

static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }

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

static void attach(int i, int slot, float *rx, float *ry) {
  float lx, ly;
  stem_local(kind[i], slot, scale[i], &lx, &ly);
  float a = bodies[i * STRIDE + FAR_HEADING];
  float c = cosf(a), s = sinf(a);
  *rx = lx * c - ly * s;
  *ry = lx * s + ly * c;
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
  for (int w = 0; w < n_wires; w++) solve_one_wire(n, wires + w * WIRE_NEAR, h);
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
  if (held >= 0 && held < n) {
    float *p = bodies + held * STRIDE;
    float speed = sqrtf(p[FAR_VX] * p[FAR_VX] + p[FAR_VY] * p[FAR_VY]);
    if (speed > grab_max && grab_max > 0.f) {
      float k = grab_max / speed;
      p[FAR_VX] *= k;
      p[FAR_VY] *= k;
    }
  }
  finalize_nodes(n_wires, h, rope_keep);
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
        dst[base] = src[base];
        dst[base + 1] = src[base + 1];
        dst[base + 2] = src[base + 2];
        dst[base + 3] = src[base + 3];
        continue;
      }
      int left = (i > 0 && !walls[cell - 1]) ? base - CHANNELS : -1;
      int right = (i < cols - 1 && !walls[cell + 1]) ? base + CHANNELS : -1;
      int up = (has_up && !walls[cell - cols]) ? base - row_stride : -1;
      int down = (has_down && !walls[cell + cols]) ? base + row_stride : -1;
      for (int ch = 0; ch < CHANNELS; ch++) {
        int k = base + ch;
        float self = src[k];
        float a = left >= 0 ? src[left + ch] : self;
        float b = right >= 0 ? src[right + ch] : self;
        float c = up >= 0 ? src[up + ch] : self;
        float e = down >= 0 ? src[down + ch] : self;
        dst[k] = keep * self + m * (a + b + c + e) * 0.25f;
      }
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
  for (int i = 0; i < n; i++) scent[i] *= keep;
}
