import { describe, expect, it } from 'vitest';
import { Camera, CAMERA_MAX_ZOOM, CAMERA_MIN_ZOOM } from './camera.ts';

describe('Camera.zoomBy', () => {
  it('can zoom out past the old 0.18 floor', () => {
    const cam = new Camera();
    for (let i = 0; i < 40; i++) cam.zoomBy(0.7);
    expect(cam.zoom).toBe(CAMERA_MIN_ZOOM);
    expect(cam.zoom).toBeLessThan(0.18);
  });

  it('still caps zoom-in', () => {
    const cam = new Camera();
    for (let i = 0; i < 40; i++) cam.zoomBy(1.4);
    expect(cam.zoom).toBe(CAMERA_MAX_ZOOM);
  });
});

describe('Camera.fitDisk', () => {
  it('fills the shorter viewport edge with the disk diameter', () => {
    const cam = new Camera();
    cam.setView(800, 600);
    cam.fitDisk(400);
    expect(cam.zoom).toBeCloseTo(600 / 800, 10);

    cam.setView(400, 800);
    cam.fitDisk(400);
    expect(cam.zoom).toBeCloseTo(400 / 800, 10);
  });
});

describe('Camera.zoomAt', () => {
  it('keeps the world point under the cursor fixed', () => {
    const cam = new Camera();
    cam.setView(800, 600);
    cam.zoom = 1;
    cam.snap(100, 50);
    const sx = 600;
    const sy = 200;
    const before = cam.worldFromScreen(sx, sy);
    cam.zoomAt(2, sx, sy);
    const after = cam.worldFromScreen(sx, sy);
    expect(after.x).toBeCloseTo(before.x, 10);
    expect(after.y).toBeCloseTo(before.y, 10);
    expect(cam.zoom).toBe(2);
  });
});
