/*
 * Instanced dot/triangle rendering for the FAR tier (see render.ts). One
 * quad per instance, corners from vertex_index, rotated to the body's
 * heading, then clipped to a circle or a triangle in the fragment shader.
 *
 * Instance layout, 9 floats (36 bytes) per body:
 *   [0] x, [1] y (world), [2] radius (world units), [3] heading (radians),
 *   [4] shape (0 = circle, 1 = triangle),
 *   [5] r, [6] g, [7] b, [8] a
 */
struct Camera {
  // world -> NDC: ndc = (world - origin) * scale, then the standard Y flip.
  originX: f32,
  originY: f32,
  scaleX: f32,
  scaleY: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<f32>;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
  @location(2) @interpolate(flat) shape: u32,
};

const CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

const SHAPE_CIRCLE = 0u;
const SHAPE_TRIANGLE = 1u;

const STRIDE = 9u;

@vertex
fn vs_main(@builtin(vertex_index) vIdx: u32, @builtin(instance_index) iIdx: u32) -> VsOut {
  let base = iIdx * STRIDE;
  let worldX = instances[base + 0u];
  let worldY = instances[base + 1u];
  let radius = instances[base + 2u];
  let heading = instances[base + 3u];
  let corner = CORNERS[vIdx];

  // Rotate the quad to the body's heading; the fragment shader's shape
  // tests stay in this un-rotated local frame.
  let c = cos(heading);
  let s = sin(heading);
  let rotated = vec2f(corner.x * c - corner.y * s, corner.x * s + corner.y * c);

  let px = worldX + rotated.x * radius;
  let py = worldY + rotated.y * radius;
  let ndcX = (px - camera.originX) * camera.scaleX;
  let ndcY = -(py - camera.originY) * camera.scaleY;

  var out: VsOut;
  out.pos = vec4f(ndcX, ndcY, 0.0, 1.0);
  out.uv = corner;
  out.color = vec4f(instances[base + 5u], instances[base + 6u], instances[base + 7u], instances[base + 8u]);
  out.shape = u32(instances[base + 4u] + 0.5);
  return out;
}

/** Signed perpendicular distance from the line a->b, positive to its left. */
fn edgeDist(a: vec2f, b: vec2f, p: vec2f) -> f32 {
  let e = b - a;
  let w = p - a;
  return (e.x * w.y - e.y * w.x) / length(e);
}

// A triangle pointing along local +X, the heading's own direction.
const TRI_TIP = vec2f(1.0, 0.0);
const TRI_BACK_L = vec2f(-0.7, 0.75);
const TRI_BACK_R = vec2f(-0.7, -0.75);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
  var edge: f32;
  if (in.shape == SHAPE_TRIANGLE) {
    let d = min(min(
      edgeDist(TRI_TIP, TRI_BACK_L, in.uv),
      edgeDist(TRI_BACK_L, TRI_BACK_R, in.uv)),
      edgeDist(TRI_BACK_R, TRI_TIP, in.uv));
    if (d < -0.08) {
      discard;
    }
    edge = smoothstep(-0.08, 0.0, d);
  } else {
    let r = length(in.uv);
    if (r > 1.0) {
      discard;
    }
    // A soft 1px-ish edge instead of a hard-aliased circle.
    edge = 1.0 - smoothstep(0.85, 1.0, r);
  }
  let a = in.color.a * edge;
  // Premultiplied, to match the canvas's alphaMode and the pipeline's blend
  // state (agents-gpu.ts); straight alpha here would multiply twice.
  return vec4f(in.color.rgb * a, a);
}
