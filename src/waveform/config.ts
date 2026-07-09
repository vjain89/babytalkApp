/** Shared waveform capture + display constants. */

/** Native record-back callback interval (seconds). Must match BAR_MS. */
export const SUBSCRIPTION_SEC = 0.05;

/** One stored / displayed bar = 50ms of audio. */
export const BAR_MS = 50;

/** Recording live-tail window. */
export const RECORD_WINDOW_MS = 3_000; // 60 bars

/** Playback rolling window (fits ~160 bars on a phone without downsampling). */
export const PLAYBACK_WINDOW_MS = 8_000; // 160 bars

/** iOS metering floor (silence). Store raw values; do not clip on capture. */
export const CAPTURE_DB_MIN = -160;
export const CAPTURE_DB_MAX = 0;

/**
 * Fixed visible range while recording.
 * Capture still stores full -160..0; this only maps dB → bar height.
 */
export const RECORD_VISIBLE_DB_MIN = -60;
export const RECORD_VISIBLE_DB_MAX = 0;

/** Playback: map this percentile of peakDb to TARGET_PEAK_FRACTION of height. */
export const PLAYBACK_SCALE_PERCENTILE = 0.95;
export const TARGET_PEAK_FRACTION = 0.8;

/** Values at or below this (in the visible mapping) render as silence. */
export const DISPLAY_NOISE_FLOOR_DB = -58;

export const MIN_BAR_PX = 1;
export const WAVEFORM_HEIGHT = 64;

export const WAVEFORM_SCHEMA_VERSION = 2;
