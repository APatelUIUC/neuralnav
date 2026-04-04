/**
 * Viridis-inspired sequential color scale for the heatmap.
 * Dark purple -> teal -> bright yellow.
 */
const VIRIDIS_STOPS: [number, number, number][] = [
  [0.267, 0.004, 0.329], // dark purple
  [0.282, 0.140, 0.458],
  [0.253, 0.265, 0.530],
  [0.206, 0.372, 0.553],
  [0.163, 0.471, 0.558],
  [0.128, 0.567, 0.551],
  [0.134, 0.658, 0.518],
  [0.267, 0.749, 0.441],
  [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.150],
  [0.993, 0.906, 0.144], // bright yellow
];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function interpolateViridis(t: number): [number, number, number] {
  // Clamp t to [0, 1]
  const tc = Math.max(0, Math.min(1, t));
  const idx = tc * (VIRIDIS_STOPS.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, VIRIDIS_STOPS.length - 1);
  const frac = idx - lo;

  const a = VIRIDIS_STOPS[lo]!;
  const b = VIRIDIS_STOPS[hi]!;

  return [
    lerp(a[0], b[0], frac),
    lerp(a[1], b[1], frac),
    lerp(a[2], b[2], frac),
  ];
}

/**
 * Map a norm value to a CSS color string using a viridis-like scale.
 */
export function normToColor(value: number, min: number, max: number): string {
  const range = max - min;
  const t = range > 0 ? (value - min) / range : 0;
  const [r, g, b] = interpolateViridis(t);
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

/**
 * Get the raw RGB [0-255] for canvas rendering.
 */
export function normToRGB(
  value: number,
  min: number,
  max: number,
): [number, number, number] {
  const range = max - min;
  const t = range > 0 ? (value - min) / range : 0;
  const [r, g, b] = interpolateViridis(t);
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Generate an array of { color, value } stops for the color legend.
 */
export function colorLegendStops(
  min: number,
  max: number,
  steps: number = 10,
): { color: string; value: number }[] {
  const stops: { color: string; value: number }[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const value = min + t * (max - min);
    stops.push({ color: normToColor(value, min, max), value });
  }
  return stops;
}
