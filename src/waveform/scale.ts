import {
  CAPTURE_DB_MAX,
  CAPTURE_DB_MIN,
  DISPLAY_NOISE_FLOOR_DB,
  PLAYBACK_SCALE_PERCENTILE,
  RECORD_VISIBLE_DB_MAX,
  RECORD_VISIBLE_DB_MIN,
  TARGET_PEAK_FRACTION,
} from './config';
import type { WaveformSample } from './types';

export type DbRange = { minDb: number; maxDb: number };

/** Fixed range used during live recording. */
export const RECORD_DB_RANGE: DbRange = {
  minDb: RECORD_VISIBLE_DB_MIN,
  maxDb: RECORD_VISIBLE_DB_MAX,
};

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/**
 * Map a dB value into 0..1 for bar height within a visible range.
 * Values at/below noise floor → 0. Does not apply live autoscale.
 */
export function dbToHeight01(db: number, range: DbRange): number {
  if (!Number.isFinite(db) || db <= DISPLAY_NOISE_FLOOR_DB) return 0;
  const span = Math.max(1e-6, range.maxDb - range.minDb);
  return clamp((db - range.minDb) / span, 0, 1);
}

/**
 * Playback Y range from session peaks.
 * Uses a high percentile so a single shriek does not crush conversation features,
 * with headroom so that shriek still reads as taller (clamped at top).
 */
export function computePlaybackDbRange(samples: WaveformSample[]): DbRange {
  const peaks = samples
    .map((s) => s.peakDb)
    .filter((db) => Number.isFinite(db) && db > CAPTURE_DB_MIN + 1)
    .sort((a, b) => a - b);

  if (peaks.length === 0) {
    return { ...RECORD_DB_RANGE };
  }

  const idx = Math.min(
    peaks.length - 1,
    Math.max(0, Math.floor(peaks.length * PLAYBACK_SCALE_PERCENTILE)),
  );
  const p = peaks[idx];

  // Place the percentile peak at TARGET_PEAK_FRACTION of bar height (maxDb = 0):
  // (p - minDb) / (0 - minDb) = TARGET  →  minDb = p / (1 - TARGET)
  // A loud shriek above p still clips at the top; conversation below p stays visible.
  const maxDb = CAPTURE_DB_MAX;
  let minDb = p / (1 - TARGET_PEAK_FRACTION);
  // Stay within a useful speech window; do not reopen older ultra-wide -60 floors.
  minDb = clamp(minDb, CAPTURE_DB_MIN, RECORD_VISIBLE_DB_MIN);
  if (maxDb - minDb < 30) minDb = maxDb - 30;
  // Floor should sit near ambient; never lower than RECORD_VISIBLE for readability.
  if (minDb < RECORD_VISIBLE_DB_MIN - 10) minDb = RECORD_VISIBLE_DB_MIN - 10;

  return { minDb, maxDb };
}

/**
 * Scale factor for bipolar amp columns: percentile of |amp| → TARGET_PEAK_FRACTION.
 * Returns a divisor: renderAmp = amp / scale (clamped to -1..1).
 */
export function computeBipolarAmpScale(samples: WaveformSample[]): number {
  const amps = samples
    .flatMap((s) => [Math.abs(s.minAmp ?? 0), Math.abs(s.maxAmp ?? 0)])
    .filter((a) => a > 1e-6)
    .sort((a, b) => a - b);

  if (amps.length === 0) return 1;
  const idx = Math.min(
    amps.length - 1,
    Math.max(0, Math.floor(amps.length * PLAYBACK_SCALE_PERCENTILE)),
  );
  const p = amps[idx];
  return Math.max(1e-6, p / TARGET_PEAK_FRACTION);
}
