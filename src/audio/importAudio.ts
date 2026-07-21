import RNFS from 'react-native-fs';
import AudioRecorderPlayer from '../../react-native-audio-recorder-player';
import { addRecording, getTodayRecordingCount } from '../db';
import { IMPORT_DIR, SYSTEM_INBOX_DIR } from '../export/backup';
import { PLAYBACK_BAR_MS } from '../waveform/config';
import { serializeWaveform } from '../waveform/storage';
import { emptyWaveformPayload } from '../waveform/types';

const audioBridge = new AudioRecorderPlayer();

const AUDIO_EXTS = new Set([
  'm4a',
  'mp4',
  'aac',
  'wav',
  'caf',
  'aiff',
  'aif',
  'mp3',
  'flac',
]);

function extOf(name: string): string {
  const parts = name.split('.');
  return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
}

function basenameNoExt(name: string): string {
  return name.replace(/\.[^.]+$/, '');
}

function isAudioFilename(name: string): boolean {
  return AUDIO_EXTS.has(extOf(name));
}

/** Normalize file:// URIs and percent-encoding; try /private variants iOS uses. */
async function resolveExistingPath(pathOrUri: string): Promise<string | null> {
  let raw = pathOrUri.replace(/^file:\/\//, '');
  try {
    raw = decodeURIComponent(raw);
  } catch {
    // keep raw
  }

  const candidates = [raw];
  if (raw.startsWith('/private/')) {
    candidates.push(raw.replace(/^\/private/, ''));
  } else if (raw.startsWith('/var/')) {
    candidates.push(`/private${raw}`);
  }

  for (const c of candidates) {
    try {
      if (await RNFS.exists(c)) return c;
    } catch {
      // try next
    }
  }
  return null;
}

export type ImportedRecording = {
  recordingId: number;
  filename: string;
  sessionName: string;
  durationMs: number;
};

/**
 * Copy an external audio URI into Documents, extract waveform, and insert a recording row.
 * Works for Voice Memos (via Files picker / Open In), Downloads, AirDrop, etc.
 */
export async function importAudioFile(opts: {
  uri: string;
  name?: string;
  sessionName?: string;
}): Promise<ImportedRecording> {
  const srcPath = await resolveExistingPath(opts.uri);
  const originalName =
    opts.name ||
    (srcPath || opts.uri).replace(/^file:\/\//, '').split('/').pop() ||
    `import_${Date.now()}.m4a`;

  if (!srcPath) {
    throw new Error(
      `Source audio not found: ${originalName} (looked for ${opts.uri})`,
    );
  }

  const safeExt = extOf(originalName) || 'm4a';
  const destName = `imported_${Date.now()}_${Math.floor(Math.random() * 1e6)}.${safeExt}`;
  const destPath = `${RNFS.DocumentDirectoryPath}/${destName}`;

  // If the file is already a stable Documents import, move/rename instead of copy when possible.
  const alreadyInDocs =
    srcPath.startsWith(RNFS.DocumentDirectoryPath) ||
    srcPath.startsWith(`/private${RNFS.DocumentDirectoryPath}`);

  if (await RNFS.exists(destPath)) {
    await RNFS.unlink(destPath);
  }

  if (alreadyInDocs && srcPath.includes('/Import/')) {
    try {
      await RNFS.moveFile(srcPath, destPath);
    } catch {
      await RNFS.copyFile(srcPath, destPath);
    }
  } else {
    await RNFS.copyFile(srcPath, destPath);
  }

  let durationMs = 0;
  try {
    durationMs = Math.round(await audioBridge.getAudioDurationMs(destPath));
  } catch (err) {
    console.warn('⚠️ Could not read duration; estimating from waveform', err);
  }

  const payload = emptyWaveformPayload('file');
  payload.barDurationMs = PLAYBACK_BAR_MS;
  try {
    const peaks = await audioBridge.extractWaveformPeaks(destPath, PLAYBACK_BAR_MS, true);
    payload.samples = peaks;
    if (durationMs <= 0 && peaks.length > 0) {
      const last = peaks[peaks.length - 1];
      durationMs = Math.round(last.tMs + PLAYBACK_BAR_MS);
    }
  } catch (err) {
    console.warn('⚠️ Waveform extract failed on import:', err);
  }

  let sessionName = opts.sessionName?.trim();
  if (!sessionName) {
    // Prefer the original Voice Memo title (picker passes it separately).
    let fromFile = basenameNoExt(originalName).trim();
    fromFile = fromFile.replace(/^picked_\d+_/, '');
    if (fromFile && !/^recording_/i.test(fromFile) && fromFile !== 'New Recording') {
      sessionName = fromFile;
    } else {
      const count = await getTodayRecordingCount();
      sessionName = `Imported-${count + 1}`;
    }
  }

  const recordingId = await addRecording({
    filename: destName,
    sessionName,
    durationMs: Math.max(0, durationMs),
    waveformData: serializeWaveform(payload),
  });

  return {
    recordingId,
    filename: destName,
    sessionName,
    durationMs,
  };
}

/** Open the system audio picker (browse On My iPhone → Voice Memos, Files, etc.). */
export async function pickAndImportAudioFiles(allowMultiple = true): Promise<{
  imported: ImportedRecording[];
  cancelled: boolean;
  errors: string[];
}> {
  let picks: Array<{ uri: string; name: string }>;
  try {
    picks = await audioBridge.pickAudioFiles(allowMultiple);
  } catch (err: any) {
    const msg = String(err?.message || err);
    if (msg.toLowerCase().includes('cancel')) {
      return { imported: [], cancelled: true, errors: [] };
    }
    throw err;
  }

  const imported: ImportedRecording[] = [];
  const errors: string[] = [];
  for (const pick of picks) {
    try {
      imported.push(
        await importAudioFile({
          uri: pick.uri,
          name: pick.name,
        }),
      );
    } catch (err) {
      errors.push(`${pick.name}: ${String(err)}`);
    }
  }
  return { imported, cancelled: false, errors };
}

/**
 * Import loose audio from Documents/Import, Documents root, and the iOS system Inbox.
 * Call prepareIncomingSharedAudio() first (Share extension / Copy-to land here).
 */
export async function importAudioFromInbox(): Promise<{
  imported: ImportedRecording[];
  errors: string[];
}> {
  try {
    await audioBridge.prepareIncomingSharedAudio();
  } catch (err) {
    console.warn('prepareIncomingSharedAudio:', err);
  }

  const imported: ImportedRecording[] = [];
  const errors: string[] = [];

  const docsRoot = RNFS.DocumentDirectoryPath;
  for (const dir of [IMPORT_DIR, SYSTEM_INBOX_DIR, docsRoot]) {
    const dirPath = await resolveExistingPath(dir);
    if (!dirPath) continue;
    let entries: Awaited<ReturnType<typeof RNFS.readDir>> = [];
    try {
      entries = await RNFS.readDir(dirPath);
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (!entry.isFile() || !isAudioFilename(entry.name)) continue;
      if (entry.name === 'README.txt') continue;
      // Don't scoop up in-app recordings already living in Documents root.
      if (dirPath === docsRoot || dirPath === `/private${docsRoot}`) {
        const n = entry.name.toLowerCase();
        if (n.startsWith('recording_') || n.startsWith('imported_')) continue;
      }
      try {
        const result = await importAudioFile({
          uri: entry.path,
          name: entry.name,
        });
        imported.push(result);
        try {
          if (await RNFS.exists(entry.path)) await RNFS.unlink(entry.path);
        } catch {
          // keep source if delete fails
        }
      } catch (err) {
        errors.push(`${entry.name}: ${String(err)}`);
      }
    }
  }

  return { imported, errors };
}
