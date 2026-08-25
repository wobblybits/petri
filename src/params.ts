export interface Params {
  deposit: number;
  diffuse: number;
  decay: number;
  sense: number;
  attractStrong: number;
  attractMedium: number;
  snapRadius: number;
  snapArc: number;
  snapWell: number;
  faceRadius: number;
  faceAttract: number;
  rewriteDuration: number;
  springK: number;
  springDamp: number;
  wireMinRest: number;
  wireShrink: number;
  eraMass: number;
  nodeMass: number;
  turnRate: number;
  sensorAngle: number;
  sensorDist: number;
  stepSpeed: number;
  thrust: number;
  drag: number;
  angDrag: number;
  maxSpeed: number;
  wander: number;
  gravity: number;
  flockAlign: number;
  flockSep: number;
  maxAgents: number;
  soupCount: number;
  /** Seconds between automatic free-agent spawns (0 = off). */
  spawnInterval: number;
}

export function defaultParams(): Params {
  return {
    deposit: 2.4,
    diffuse: 0.28,
    decay: 0.018,
    sense: 520,
    attractStrong: 1.45,
    attractMedium: 0.72,
    snapRadius: 22,
    snapArc: 0.3,
    snapWell: 48,
    faceRadius: 90,
    faceAttract: 32,
    rewriteDuration: 0.7,
    springK: 12,
    springDamp: 18,
    wireMinRest: 40,
    wireShrink: 0.9,
    eraMass: 0.45,
    nodeMass: 1,
    turnRate: 1.6,
    sensorAngle: 0.48,
    sensorDist: 24,
    stepSpeed: 38,
    thrust: 95,
    drag: 0.55,
    angDrag: 2.4,
    maxSpeed: 70,
    wander: 0.12,
    gravity: 0.12,
    flockAlign: 5.5,
    flockSep: 36,
    maxAgents: 80,
    soupCount: 28,
    spawnInterval: 10,
  };
}

export interface SliderSpec {
  key: keyof Params;
  label: string;
  min: number;
  max: number;
  step: number;
}

export const SLIDERS: SliderSpec[] = [
  { key: 'deposit', label: 'Deposit', min: 0, max: 6, step: 0.05 },
  { key: 'diffuse', label: 'Diffuse', min: 0, max: 1, step: 0.01 },
  { key: 'decay', label: 'Decay', min: 0, max: 0.08, step: 0.001 },
  { key: 'sense', label: 'Sense', min: 0, max: 1200, step: 10 },
  { key: 'attractStrong', label: 'Strong attract', min: 0, max: 3, step: 0.05 },
  { key: 'attractMedium', label: 'Medium attract', min: 0, max: 2, step: 0.05 },
  { key: 'sensorAngle', label: 'Sensor arc', min: 0.1, max: 1.2, step: 0.02 },
  { key: 'stepSpeed', label: 'Step speed', min: 10, max: 180, step: 1 },
  { key: 'gravity', label: 'Gravity', min: 0, max: 0.8, step: 0.01 },
  { key: 'drag', label: 'Fluid drag', min: 0, max: 4, step: 0.05 },
  { key: 'angDrag', label: 'Spin damp', min: 0, max: 8, step: 0.05 },
  { key: 'flockAlign', label: 'Flock align', min: 0, max: 16, step: 0.1 },
  { key: 'flockSep', label: 'Flock separate', min: 0, max: 120, step: 1 },
  { key: 'snapRadius', label: 'Snap reach', min: 4, max: 48, step: 1 },
  { key: 'snapArc', label: 'Snap arc', min: 0.08, max: 1.2, step: 0.02 },
  { key: 'wireShrink', label: 'Wire shrink', min: 0.1, max: 3, step: 0.05 },
  { key: 'wireMinRest', label: 'Wire min length', min: 8, max: 48, step: 1 },
  { key: 'springK', label: 'Spring stiffness', min: 0, max: 80, step: 0.5 },
  { key: 'springDamp', label: 'Spring damp', min: 0, max: 40, step: 0.5 },
  { key: 'eraMass', label: 'Era mass', min: 0.15, max: 2, step: 0.05 },
  { key: 'nodeMass', label: 'Con/Dup mass', min: 0.3, max: 4, step: 0.05 },
  { key: 'maxAgents', label: 'Max agents', min: 8, max: 200, step: 1 },
  { key: 'spawnInterval', label: 'Auto spawn (s)', min: 0, max: 30, step: 0.5 },
];
