import { BAR_MS, WAVEFORM_SCHEMA_VERSION } from './config';

/** One metering sample from a native record-back callback. */
export type WaveformSample = {
  /** Time from start of recording, milliseconds. */
  tMs: number;
  /** averagePower (iOS) or equivalent, dBFS. */
  avgDb: number;
  /** peakPower (iOS) or maxAmplitude-derived (Android), dBFS. */
  peakDb: number;
};

/** Persisted waveform payload (JSON in recordings.waveform_data). */
export type WaveformPayload = {
  version: typeof WAVEFORM_SCHEMA_VERSION | number;
  barDurationMs: number;
  samples: WaveformSample[];
};

export function emptyWaveformPayload(): WaveformPayload {
  return {
    version: WAVEFORM_SCHEMA_VERSION,
    barDurationMs: BAR_MS,
    samples: [],
  };
}
