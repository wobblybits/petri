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
