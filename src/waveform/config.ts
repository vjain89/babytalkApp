/** Shared waveform capture + display constants. */

/** Native record-back callback interval (seconds). Must match BAR_MS. */
export const SUBSCRIPTION_SEC = 0.05;

/** One stored / displayed bar while recording = 50ms of audio. */
export const BAR_MS = 50;

/**
 * A1 playback detailed waveform: ~7 ms columns (min+max).
 * Recording live view stays at BAR_MS; playback Path B re-extracts at this size.
 */
export const PLAYBACK_BAR_MS = 7;

/** Recording live-tail window. */
export const RECORD_WINDOW_MS = 3_000; // 60 bars

/** Playback rolling window. */
export const PLAYBACK_WINDOW_MS = 8_000;

/** iOS metering floor (silence). Store raw values; do not clip on capture. */
export const CAPTURE_DB_MIN = -160;
export const CAPTURE_DB_MAX = 0;

/**
 * Fixed visible range while recording (Option 1 polish).
 * Narrower than full metering span so speech stands out vs room tone.
 * Capture still stores full -160..0.
 */
export const RECORD_VISIBLE_DB_MIN = -45;
export const RECORD_VISIBLE_DB_MAX = 0;

/** Playback: map this percentile of peakDb to TARGET_PEAK_FRACTION of height. */
export const PLAYBACK_SCALE_PERCENTILE = 0.95;
export const TARGET_PEAK_FRACTION = 0.8;

/**
 * Values at or below this (dB) render as silence.
 * Raised so ambient noise does not fill the strip during quiet moments.
 */
export const DISPLAY_NOISE_FLOOR_DB = -42;

/** Tag marker width on the time axis (was 500ms — ~17% of a 3s window). */
export const TAG_MARKER_MS = 80;

/** Minimum gap between bars so the strip reads as spikes, not a solid blob. */
export const BAR_GAP_PX = 1;

export const MIN_BAR_PX = 1;
export const WAVEFORM_HEIGHT = 72;

/** v3 adds optional minAmp/maxAmp for bipolar A1 playback. */
export const WAVEFORM_SCHEMA_VERSION = 3;
