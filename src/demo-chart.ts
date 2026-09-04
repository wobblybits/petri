import type { Sim } from './sim.ts';

export const CHART_MAX_SAMPLES = 180;

export type ChartSample = {
  total: number;
  era: number;
  dup: number;
  con: number;
  energy: number;
};

const COLORS = {
  total: '#ffffff',
  era: '#ffff00',
  dup: '#ff0000',
  con: '#0000ff',
  energy: '#3ddc84',
} as const;

/**
 * 1 Hz strip for the public demo. Not on the sim tick — sample() is what
 * walks the agents, and the caller fires it from a timer.
 */
export class DemoChart {
  readonly samples: ChartSample[] = [];
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('no 2d context for chart');
    this.ctx = ctx;
  }

  clear(): void {
    this.samples.length = 0;
    this.draw();
  }

  sample(sim: Sim): void {
    let era = 0;
    let dup = 0;
    let con = 0;
    for (const a of sim.agents.values()) {
      if (a.kind === 'era') era++;
      else if (a.kind === 'dup') dup++;
      else con++;
    }
    this.samples.push({
      total: sim.agents.size,
      era,
      dup,
      con,
      energy: sim.totalFree(),
    });
    if (this.samples.length > CHART_MAX_SAMPLES) this.samples.shift();
    this.draw();
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    this.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  draw(): void {
    const ctx = this.ctx;
    const w = this.canvas.getBoundingClientRect().width;
    const h = this.canvas.getBoundingClientRect().height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(8, 9, 12, 0.55)';
    ctx.fillRect(0, 0, w, h);

    const padL = 8;
    const padR = 8;
    const padT = 18;
    const padB = 6;
    const innerW = Math.max(1, w - padL - padR);
    const innerH = Math.max(1, h - padT - padB);
    const series = this.samples;
    ctx.fillStyle = '#8b8d96';
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'top';
    ctx.fillText('pop', padL, 4);
    ctx.fillStyle = COLORS.era;
    ctx.fillText('era', padL + 28, 4);
    ctx.fillStyle = COLORS.dup;
    ctx.fillText('dup', padL + 56, 4);
    ctx.fillStyle = COLORS.con;
    ctx.fillText('con', padL + 84, 4);
    ctx.fillStyle = COLORS.energy;
    ctx.fillText('energy', padL + 112, 4);
    if (series.length < 2) return;

    let popMax = 1;
    let energyMax = 1;
    for (const s of series) {
      popMax = Math.max(popMax, s.total, s.era, s.dup, s.con);
      energyMax = Math.max(energyMax, s.energy);
    }

    const xAt = (i: number) => padL + (i / (series.length - 1)) * innerW;
    const yPop = (v: number) => padT + innerH * (1 - v / popMax);
    const yEnergy = (v: number) => padT + innerH * (1 - v / energyMax);

    const stroke = (color: string, yOf: (s: ChartSample) => number) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.lineJoin = 'round';
      for (let i = 0; i < series.length; i++) {
        const x = xAt(i);
        const y = yOf(series[i]!);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    stroke(COLORS.energy, (s) => yEnergy(s.energy));
    stroke(COLORS.era, (s) => yPop(s.era));
    stroke(COLORS.dup, (s) => yPop(s.dup));
    stroke(COLORS.con, (s) => yPop(s.con));
    stroke(COLORS.total, (s) => yPop(s.total));
  }
}
