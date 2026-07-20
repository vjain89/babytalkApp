import { BAR_MS, WAVEFORM_SCHEMA_VERSION } from './config';

/** One metering or file-extracted sample. */
export type WaveformSample = {
  /** Time from start of recording, milliseconds. */
  tMs: number;
  /** RMS / averagePower equivalent, dBFS. */
  avgDb: number;
  /** Peak amplitude, dBFS. */
  peakDb: number;
  /** A1 bipolar: min signed amplitude in bin (-1..1). */
  minAmp?: number;
  /** A1 bipolar: max signed amplitude in bin (-1..1). */
  maxAmp?: number;
};

export type WaveformSource = 'metering' | 'file';

/** Persisted waveform payload (JSON in recordings.waveform_data). */
export type WaveformPayload = {
  version: typeof WAVEFORM_SCHEMA_VERSION | number;
  barDurationMs: number;
  /** Where samples came from. Path B uses 'file'. */
  source?: WaveformSource;
  samples: WaveformSample[];
};

export function emptyWaveformPayload(source: WaveformSource = 'metering'): WaveformPayload {
  return {
    version: WAVEFORM_SCHEMA_VERSION,
    barDurationMs: BAR_MS,
    source,
    samples: [],
  };
}

/** One bipolar column for rendering: normalized -1..1 min/max. */
export type BipolarColumn = { min: number; max: number };
