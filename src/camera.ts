import { clamp } from './wrap.ts';

export const CAMERA_MIN_ZOOM = 0.02;
export const CAMERA_MAX_ZOOM = 6;

export class Camera {
  x = 0;
  y = 0;
  zoom = .065;
  viewW = 800;
  viewH = 600;

  setView(w: number, h: number): void {
    this.viewW = Math.max(1, w);
    this.viewH = Math.max(1, h);
  }

  snap(x: number, y: number): void {
    this.x = x;
    this.y = y;
  }

  follow(x: number, y: number, dt: number): void {
    const k = 1 - Math.exp(-5 * dt);
    this.x += (x - this.x) * k;
    this.y += (y - this.y) * k;
  }

  coverWidth(): number {
    return this.viewW / this.zoom;
  }

  coverHeight(): number {
    return this.viewH / this.zoom;
  }

  worldFromScreen(sx: number, sy: number): { x: number; y: number } {
    return {
      x: this.x + (sx - this.viewW * 0.5) / this.zoom,
      y: this.y + (sy - this.viewH * 0.5) / this.zoom,
    };
  }

  zoomBy(factor: number): void {
    this.zoom = clamp(this.zoom * factor, CAMERA_MIN_ZOOM, CAMERA_MAX_ZOOM);
  }

  /** Keep the world point under `(sx, sy)` fixed while the zoom changes. */
  zoomAt(factor: number, sx: number, sy: number): void {
    const before = this.worldFromScreen(sx, sy);
    this.zoomBy(factor);
    const after = this.worldFromScreen(sx, sy);
    this.x += before.x - after.x;
    this.y += before.y - after.y;
  }

  /** Frame a disk of `radius` so its diameter fills the shorter viewport edge. */
  fitDisk(radius: number): void {
    const span = 2 * radius;
    if (!(span > 0)) return;
    const minDim = Math.min(this.viewW, this.viewH);
    this.zoom = clamp(minDim / span, CAMERA_MIN_ZOOM, CAMERA_MAX_ZOOM);
  }

  apply(ctx: CanvasRenderingContext2D): void {
    ctx.translate(this.viewW * 0.5, this.viewH * 0.5);
    ctx.scale(this.zoom, this.zoom);
    ctx.translate(-this.x, -this.y);
  }
}
