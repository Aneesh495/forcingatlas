/** Diverging anomaly scale. Missing values are not mapped to a color. */

const STOPS: Array<[number, [number, number, number]]> = [
  [0, [28, 58, 112]],
  [0.35, [148, 186, 214]],
  [0.5, [244, 239, 230]],
  [0.65, [222, 164, 132]],
  [1, [140, 47, 57]],
];

export const SCALE_MIN = -2;
export const SCALE_MAX = 2;

export function anomalyColor(value: number, alpha = 1): string {
  const t = Math.min(1, Math.max(0, (value - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)));
  let lower = STOPS[0];
  let upper = STOPS[STOPS.length - 1];
  for (let index = 1; index < STOPS.length; index += 1) {
    if (t <= STOPS[index][0]) {
      lower = STOPS[index - 1];
      upper = STOPS[index];
      break;
    }
  }
  const span = upper[0] - lower[0] || 1;
  const mix = (t - lower[0]) / span;
  const red = Math.round(lower[1][0] + (upper[1][0] - lower[1][0]) * mix);
  const green = Math.round(lower[1][1] + (upper[1][1] - lower[1][1]) * mix);
  const blue = Math.round(lower[1][2] + (upper[1][2] - lower[1][2]) * mix);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

export function finite(value: number): boolean {
  return Number.isFinite(value);
}
