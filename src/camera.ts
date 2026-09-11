import { clamp } from './wrap.ts';

export const CAMERA_MIN_ZOOM = 0.02;
export const CAMERA_MAX_ZOOM = 6;

export class Camera {
  x = 0;
  y = 0;
  zoom = .065;
  viewW = 800;
  viewH = 600;
  /** Screen-space top band the dish should sit below (demo chrome on phones). */
  insetTop = 0;

  setView(w: number, h: number): void {
    this.viewW = Math.max(1, w);
    this.viewH = Math.max(1, h);
  }

  get contentW(): number {
    return this.viewW;
  }

  get contentH(): number {
    return Math.max(1, this.viewH - this.insetTop);
  }

  get screenCX(): number {
    return this.viewW * 0.5;
  }

  get screenCY(): number {
    return this.insetTop + this.contentH * 0.5;
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
      x: this.x + (sx - this.screenCX) / this.zoom,
      y: this.y + (sy - this.screenCY) / this.zoom,
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

  /** Frame a disk of `radius` so its diameter fills the shorter content edge. */
  fitDisk(radius: number): void {
    const span = 2 * radius;
    if (!(span > 0)) return;
    const minDim = Math.min(this.contentW, this.contentH);
    this.zoom = clamp(minDim / span, CAMERA_MIN_ZOOM, CAMERA_MAX_ZOOM);
  }

  /** World-space origin the GPU path treats as the canvas centre, so a shifted 2d projection and the FAR dots line up. */
  gpuView(): { x: number; y: number; zoom: number; viewW: number; viewH: number } {
    return {
      x: this.x + (this.viewW * 0.5 - this.screenCX) / this.zoom,
      y: this.y + (this.viewH * 0.5 - this.screenCY) / this.zoom,
      zoom: this.zoom,
      viewW: this.viewW,
      viewH: this.viewH,
    };
  }

  apply(ctx: CanvasRenderingContext2D): void {
    ctx.translate(this.screenCX, this.screenCY);
    ctx.scale(this.zoom, this.zoom);
    ctx.translate(-this.x, -this.y);
  }
}
