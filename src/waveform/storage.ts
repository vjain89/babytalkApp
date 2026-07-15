import { BAR_MS, CAPTURE_DB_MIN, WAVEFORM_SCHEMA_VERSION } from './config';
import { emptyWaveformPayload, type WaveformPayload, type WaveformSample } from './types';

/**
 * Upsert a sample into a time-ordered sparse list.
 * If a sample already exists for the same 50ms bin, keep the louder peak/avg.
 */
export function upsertSample(samples: WaveformSample[], next: WaveformSample): WaveformSample[] {
  const bin = Math.round(next.tMs / BAR_MS) * BAR_MS;
  const sample: WaveformSample = {
    tMs: bin,
    avgDb: next.avgDb,
    peakDb: next.peakDb,
  };

  const last = samples[samples.length - 1];
  if (last && last.tMs === bin) {
    last.avgDb = Math.max(last.avgDb, sample.avgDb);
    last.peakDb = Math.max(last.peakDb, sample.peakDb);
    return samples;
  }

  // Rare out-of-order: insert sorted
  if (last && sample.tMs < last.tMs) {
    const idx = samples.findIndex((s) => s.tMs >= bin);
    if (idx === -1) {
      samples.push(sample);
    } else if (samples[idx].tMs === bin) {
      samples[idx].avgDb = Math.max(samples[idx].avgDb, sample.avgDb);
      samples[idx].peakDb = Math.max(samples[idx].peakDb, sample.peakDb);
    } else {
      samples.splice(idx, 0, sample);
    }
    return samples;
  }

  samples.push(sample);
  return samples;
}

/**
 * Dense peak array for a time window (one entry per BAR_MS).
 * Missing bins → silence (CAPTURE_DB_MIN). Used only for rendering.
 * startMs may be negative (left-pad silence so "now" stays at the right edge).
 */
export function densifyPeaks(
  samples: WaveformSample[],
  startMs: number,
  endMs: number,
  barDurationMs: number = BAR_MS,
): number[] {
  const startBin = Math.floor(startMs / barDurationMs);
  const endBin = Math.max(startBin, Math.ceil(endMs / barDurationMs));
  const count = Math.max(0, endBin - startBin);
  const out = new Array<number>(count).fill(CAPTURE_DB_MIN);

  if (samples.length === 0 || count === 0) return out;

  for (const s of samples) {
    const bin = Math.round(s.tMs / barDurationMs);
    if (bin < startBin) continue;
    if (bin >= endBin) continue;
    const idx = bin - startBin;
    out[idx] = Math.max(out[idx], s.peakDb);
  }
  return out;
}

export function serializeWaveform(payload: WaveformPayload): string {
  return JSON.stringify(payload);
}

/**
 * Parse DB JSON. Supports:
 * - v2: { version, barDurationMs, samples: [{tMs,avgDb,peakDb}] }
 * - legacy: number[] of normalized 0..1 peaks (best-effort)
 */
export function parseWaveformData(raw: string | null | undefined): WaveformPayload {
  if (!raw) return emptyWaveformPayload();

  try {
    const parsed = JSON.parse(raw);

    if (parsed && typeof parsed === 'object' && Array.isArray(parsed.samples)) {
      const samples: WaveformSample[] = parsed.samples.map((s: any) => ({
        tMs: Number(s.tMs) || 0,
        avgDb: Number.isFinite(Number(s.avgDb)) ? Number(s.avgDb) : CAPTURE_DB_MIN,
        peakDb: Number.isFinite(Number(s.peakDb)) ? Number(s.peakDb) : CAPTURE_DB_MIN,
      }));
      return {
        version: Number(parsed.version) || WAVEFORM_SCHEMA_VERSION,
        barDurationMs: Number(parsed.barDurationMs) || BAR_MS,
        samples,
      };
    }

    // Legacy: flat normalized 0..1 array → fake dB for display only
    if (Array.isArray(parsed)) {
      const samples: WaveformSample[] = parsed.map((v: any, i: number) => {
        const n = Math.max(0, Math.min(1, Number(v) || 0));
        // Invert old normalize: 0..1 was roughly (-60..0) with gamma — approximate
        const db = n <= 0 ? CAPTURE_DB_MIN : -60 + n * 60;
        return { tMs: i * BAR_MS, avgDb: db, peakDb: db };
      });
      return {
        version: 1,
        barDurationMs: BAR_MS,
        samples,
      };
    }
  } catch {
    // fall through
  }

  return emptyWaveformPayload();
}
