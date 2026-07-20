/** Session-kit schema shared by phone export, USB backup, and Mac tools. */

export const SESSION_KIT_SCHEMA_VERSION = 1;

export type SessionManifest = {
  schemaVersion: number;
  recordingId: number;
  recordingUuid: string;
  sessionName: string | null;
  filename: string;
  /** WAV basename inside the kit (transcoded from ALAC master). */
  audioFile: string;
  durationMs: number;
  createdAt: number;
  codec: string;
  sampleRate: number;
  bitDepth: number;
  channels: number;
  /** SHA-256 of the exported WAV (preferred re-link key). */
  audioContentHash: string;
  exportedAt: string;
};

export type KitTag = {
  uuid: string;
  label: string;
  startMs: number;
  endMs: number | null;
  source: 'user' | 'ml' | 'ml_confirmed';
  status: 'provisional' | 'confirmed';
  /** Legacy alias for startMs. */
  tMs: number;
};

export type KitAnnotation = KitTag & {
  score?: number;
};

export type ExportBatchManifest = {
  schemaVersion: number;
  exportedAt: string;
  sessions: Array<{
    folder: string;
    recordingUuid: string;
    audioContentHash: string;
  }>;
};
