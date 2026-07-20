import RNFS from 'react-native-fs';
import AudioRecorderPlayer from '../../react-native-audio-recorder-player';
import {
  CAPTURE_BIT_DEPTH,
  CAPTURE_CHANNELS,
  CAPTURE_CODEC,
  CAPTURE_SAMPLE_RATE,
} from '../audio/captureSettings';
import {
  getRecordingById,
  getTagsForRecording,
  updateAudioContentHash,
  type TagRow,
} from '../db';
import { resolveAudioUri } from '../waveform/audioPath';
import {
  SESSION_KIT_SCHEMA_VERSION,
  type ExportBatchManifest,
  type KitAnnotation,
  type KitTag,
  type SessionManifest,
} from './types';

const audioBridge = new AudioRecorderPlayer();

const stripFile = (p: string) => p.replace(/^file:\/\//, '');

export function sessionFolderName(recording: {
  id: number;
  uuid?: string;
  session_name?: string | null;
  created_at: number;
}): string {
  const stamp = new Date(recording.created_at).toISOString().replace(/[:.]/g, '-');
  const slug = (recording.session_name || `session-${recording.id}`)
    .replace(/[^\w\-]+/g, '_')
    .slice(0, 40);
  return `${stamp}_${slug}`;
}

function toKitTag(row: TagRow): KitTag {
  const startMs = row.start_ms ?? row.timestamp_ms;
  return {
    uuid: row.uuid,
    label: row.label,
    startMs,
    endMs: row.end_ms ?? null,
    source: row.source ?? 'user',
    status: row.status ?? 'confirmed',
    tMs: startMs,
  };
}

async function ensureDir(path: string) {
  if (!(await RNFS.exists(path))) {
    await RNFS.mkdir(path);
  }
}

async function writeJson(path: string, data: unknown) {
  await RNFS.writeFile(path, JSON.stringify(data, null, 2), 'utf8');
}

/**
 * Assemble one versioned session kit folder:
 *   manifest.json, audio.wav (from ALAC), tags.json, annotations.json, optional waveform.json
 */
export async function buildSessionKit(
  recordingId: number,
  destParentDir: string,
): Promise<{ kitDir: string; manifest: SessionManifest }> {
  const recording = await getRecordingById(recordingId);
  if (!recording) {
    throw new Error(`Recording ${recordingId} not found`);
  }

  const folder = sessionFolderName(recording);
  const kitDir = `${destParentDir}/${folder}`;
  await ensureDir(kitDir);

  const audioUri = await resolveAudioUri(recording.filename);
  if (!audioUri) {
    throw new Error(`Audio file missing for recording ${recordingId}`);
  }

  const wavName = 'audio.wav';
  const wavPath = `${kitDir}/${wavName}`;
  await audioBridge.transcodeToWav(audioUri, wavPath);

  const audioContentHash = await RNFS.hash(stripFile(wavPath), 'sha256');
  await updateAudioContentHash(recordingId, audioContentHash);

  const allTags = await getTagsForRecording(recordingId);
  const userTags = allTags
    .filter((t) => (t.source ?? 'user') === 'user')
    .map(toKitTag);
  const mlAnnotations: KitAnnotation[] = allTags
    .filter((t) => (t.source ?? 'user') !== 'user')
    .map(toKitTag);

  const manifest: SessionManifest = {
    schemaVersion: SESSION_KIT_SCHEMA_VERSION,
    recordingId: recording.id,
    recordingUuid: recording.uuid,
    sessionName: recording.session_name ?? null,
    filename: recording.filename,
    audioFile: wavName,
    durationMs: recording.duration_ms,
    createdAt: recording.created_at,
    codec: CAPTURE_CODEC,
    sampleRate: CAPTURE_SAMPLE_RATE,
    bitDepth: CAPTURE_BIT_DEPTH,
    channels: CAPTURE_CHANNELS,
    audioContentHash,
    exportedAt: new Date().toISOString(),
  };

  await writeJson(`${kitDir}/manifest.json`, manifest);
  await writeJson(`${kitDir}/tags.json`, { tags: userTags });
  await writeJson(`${kitDir}/annotations.json`, { annotations: mlAnnotations });

  if (recording.waveform_data) {
    await RNFS.writeFile(`${kitDir}/waveform.json`, recording.waveform_data, 'utf8');
  }

  return { kitDir, manifest };
}

/** Build kits for many recordings into one parent folder + export_manifest.json. */
export async function buildExportBatch(
  recordingIds: number[],
  destParentDir: string,
): Promise<{ batchDir: string; batchManifest: ExportBatchManifest }> {
  await ensureDir(destParentDir);
  const sessions: ExportBatchManifest['sessions'] = [];

  for (const id of recordingIds) {
    const { kitDir, manifest } = await buildSessionKit(id, destParentDir);
    sessions.push({
      folder: kitDir.split('/').pop()!,
      recordingUuid: manifest.recordingUuid,
      audioContentHash: manifest.audioContentHash,
    });
  }

  const batchManifest: ExportBatchManifest = {
    schemaVersion: SESSION_KIT_SCHEMA_VERSION,
    exportedAt: new Date().toISOString(),
    sessions,
  };
  await writeJson(`${destParentDir}/export_manifest.json`, batchManifest);
  return { batchDir: destParentDir, batchManifest };
}

/** Share-sheet helper: build one kit under Caches and return paths to share. */
export async function buildSessionKitForShare(recordingId: number): Promise<{
  kitDir: string;
  urls: string[];
}> {
  const parent = `${RNFS.CachesDirectoryPath}/session_kits`;
  await ensureDir(parent);
  const { kitDir, manifest } = await buildSessionKit(recordingId, parent);
  const urls = [
    `file://${kitDir}/${manifest.audioFile}`,
    `file://${kitDir}/manifest.json`,
    `file://${kitDir}/tags.json`,
    `file://${kitDir}/annotations.json`,
  ];
  const waveformPath = `${kitDir}/waveform.json`;
  if (await RNFS.exists(waveformPath)) {
    urls.push(`file://${waveformPath}`);
  }
  return { kitDir, urls };
}
