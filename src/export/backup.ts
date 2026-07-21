import RNFS from 'react-native-fs';
import {
  getRecordingByAudioHash,
  getRecordingByUuid,
  upsertAnnotationByUuid,
  type TagSource,
  type TagStatus,
} from '../db';
import { buildExportBatch, buildSessionKit, sessionFolderName } from './sessionKit';
import type { KitAnnotation, SessionManifest } from './types';
import { getAllRecordings } from '../db';

const DOCUMENTS = RNFS.DocumentDirectoryPath;

export const BACKUPS_DIR = `${DOCUMENTS}/Backups`;
/** App-owned drop folder for Finder / USB imports (kits, annotations, audio). */
export const IMPORT_DIR = `${DOCUMENTS}/Import`;
/**
 * iOS system Inbox for "Open In" shares. Do NOT mkdir — iOS creates it when needed
 * and rejects app-created Inbox folders.
 */
export const SYSTEM_INBOX_DIR = `${DOCUMENTS}/Inbox`;
/** @deprecated Use IMPORT_DIR or SYSTEM_INBOX_DIR. Kept as alias for Import. */
export const INBOX_DIR = IMPORT_DIR;

async function ensureDir(path: string) {
  if (!(await RNFS.exists(path))) {
    await RNFS.mkdir(path);
  }
}

export async function ensureBackupDirs(): Promise<void> {
  await ensureDir(BACKUPS_DIR);
  await ensureDir(IMPORT_DIR);
  // Never create Documents/Inbox — that path is reserved by iOS.
  const readme = `${IMPORT_DIR}/README.txt`;
  if (!(await RNFS.exists(readme))) {
    await RNFS.writeFile(
      readme,
      [
        'BabyTalk Import',
        '',
        'Mac USB sync drops kit folders here (manifest.json + tags.json).',
        'The app imports them automatically when it becomes active — no button required.',
        'You can still tap Retry Import Annotations as a manual retry.',
        '',
        'New recordings are auto-exported to Documents/Backups/sync/ for Mac Sync with iPhone.',
        'Audio: drop .m4a/.wav files here, then open the app (or Import Inbox Audio).',
        'Or use Import Voice Memos / Audio, or Share → BabyTalk from Voice Memos.',
        '',
        'Mac review: python3 tools/review_server.py',
        'Mac sync:   tools/.venv/bin/python tools/iphone_sync.py sync',
        '',
      ].join('\n'),
      'utf8',
    );
  }
}

async function listDirIfExists(
  path: string,
): Promise<Array<{ path: string; name: string; isFile: () => boolean; isDirectory: () => boolean }>> {
  if (!(await RNFS.exists(path))) return [];
  try {
    return await RNFS.readDir(path);
  } catch {
    return [];
  }
}

function backupDateFolder(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${y}-${m}-${day}_${h}${min}`;
}

/** Stable folder of kits kept ready for Mac USB sync (flat under Backups/sync/). */
export const SYNC_KITS_DIR = `${BACKUPS_DIR}/sync`;

async function listManifestPaths(root: string): Promise<string[]> {
  if (!(await RNFS.exists(root))) return [];
  const out: string[] = [];
  const walk = async (dir: string) => {
    let entries: Awaited<ReturnType<typeof RNFS.readDir>> = [];
    try {
      entries = await RNFS.readDir(dir);
    } catch {
      return;
    }
    for (const e of entries) {
      if (e.isDirectory()) {
        await walk(e.path);
      } else if (e.name === 'manifest.json') {
        out.push(e.path);
      }
    }
  };
  await walk(root);
  return out;
}

/** Collect recordingUuid / audioContentHash already present under Documents/Backups. */
async function indexExistingBackupKits(): Promise<{
  uuids: Set<string>;
  hashes: Set<string>;
}> {
  const uuids = new Set<string>();
  const hashes = new Set<string>();
  const manifests = await listManifestPaths(BACKUPS_DIR);
  for (const path of manifests) {
    const man = await readJson<SessionManifest>(path);
    if (!man) continue;
    if (man.recordingUuid) uuids.add(man.recordingUuid);
    if (man.audioContentHash) hashes.add(man.audioContentHash);
  }
  return { uuids, hashes };
}

/**
 * Ensure every local recording has a session kit under Documents/Backups/sync/
 * so Mac "Sync with iPhone" can pull new imports without tapping Prepare USB Backup.
 */
export async function ensureRecordingKitsForSync(): Promise<{
  created: string[];
  skipped: number;
  errors: string[];
}> {
  await ensureBackupDirs();
  await ensureDir(SYNC_KITS_DIR);

  const recordings = await getAllRecordings();
  const { uuids, hashes } = await indexExistingBackupKits();

  const created: string[] = [];
  const errors: string[] = [];
  let skipped = 0;

  for (const rec of recordings) {
    const uuid = rec.uuid as string | undefined;
    const hash = rec.audio_content_hash as string | undefined;
    if ((uuid && uuids.has(uuid)) || (hash && hashes.has(hash))) {
      skipped += 1;
      continue;
    }
    const folder = sessionFolderName(rec);
    const existingKit = `${SYNC_KITS_DIR}/${folder}`;
    if (await RNFS.exists(`${existingKit}/manifest.json`)) {
      skipped += 1;
      continue;
    }
    try {
      const { kitDir, manifest } = await buildSessionKit(rec.id, SYNC_KITS_DIR);
      created.push(kitDir.split('/').pop() || folder);
      if (manifest.recordingUuid) uuids.add(manifest.recordingUuid);
      if (manifest.audioContentHash) hashes.add(manifest.audioContentHash);
      console.log(`✅ Sync kit ready: ${folder}`);
    } catch (err) {
      errors.push(`${folder}: ${String(err)}`);
      console.warn('ensureRecordingKitsForSync failed', rec.id, err);
    }
  }

  return { created, skipped, errors };
}

/** Write dated session kits under Documents/Backups/<date>/ for Finder USB copy. */
export async function prepareBackup(recordingIds: number[]): Promise<string> {
  await ensureDir(BACKUPS_DIR);
  const dest = `${BACKUPS_DIR}/${backupDateFolder()}`;
  await ensureDir(dest);
  await buildExportBatch(recordingIds, dest);
  return dest;
}

export async function prepareSingleBackup(recordingId: number): Promise<string> {
  await ensureDir(BACKUPS_DIR);
  const dest = `${BACKUPS_DIR}/${backupDateFolder()}`;
  await ensureDir(dest);
  await buildSessionKit(recordingId, dest);
  return dest;
}

export type ImportSummary = {
  scanned: number;
  inserted: number;
  updated: number;
  skipped: number;
  unmatched: number;
  errors: string[];
};

async function readJson<T>(path: string): Promise<T | null> {
  try {
    const raw = await RNFS.readFile(path, 'utf8');
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function normalizeSource(s: unknown): TagSource {
  if (s === 'ml' || s === 'ml_confirmed' || s === 'user') return s;
  if (typeof s === 'string' && s.startsWith('ml')) return 'ml';
  return 'ml';
}

function normalizeStatus(s: unknown): TagStatus {
  return s === 'confirmed' ? 'confirmed' : 'provisional';
}

async function resolveRecordingId(opts: {
  recordingUuid?: string;
  audioContentHash?: string;
  recordingId?: number;
}): Promise<number | null> {
  if (opts.audioContentHash) {
    const byHash = await getRecordingByAudioHash(opts.audioContentHash);
    if (byHash) return byHash.id;
  }
  if (opts.recordingUuid) {
    const byUuid = await getRecordingByUuid(opts.recordingUuid);
    if (byUuid) return byUuid.id;
  }
  return null;
}

async function importTagList(
  list: KitAnnotation[],
  recordingId: number,
  summary: ImportSummary,
) {
  for (const ann of list) {
    summary.scanned += 1;
    if (!ann?.uuid) {
      summary.skipped += 1;
      continue;
    }
    if ((ann as { status?: string }).status === 'dismissed') {
      summary.skipped += 1;
      continue;
    }
    const startMs = ann.startMs ?? (ann as { tMs?: number }).tMs ?? 0;
    const result = await upsertAnnotationByUuid({
      recordingId,
      uuid: ann.uuid,
      label: ann.label || 'untitled',
      startMs,
      endMs: ann.endMs ?? null,
      source: normalizeSource(ann.source ?? 'user'),
      status: normalizeStatus(ann.status ?? 'confirmed'),
    });
    if (result === 'inserted') summary.inserted += 1;
    else if (result === 'updated') summary.updated += 1;
    else summary.skipped += 1;
  }
}

async function importAnnotationsFile(
  annotationsPath: string,
  recordingId: number,
  summary: ImportSummary,
) {
  const payload = await readJson<{ annotations?: KitAnnotation[] } | KitAnnotation[]>(
    annotationsPath,
  );
  if (!payload) {
    summary.errors.push(`Could not parse ${annotationsPath}`);
    return;
  }
  const list = Array.isArray(payload) ? payload : payload.annotations ?? [];
  await importTagList(list, recordingId, summary);
}

async function importTagsFile(
  tagsPath: string,
  recordingId: number,
  summary: ImportSummary,
) {
  const payload = await readJson<{ tags?: KitAnnotation[] } | KitAnnotation[]>(tagsPath);
  if (!payload) {
    summary.errors.push(`Could not parse ${tagsPath}`);
    return;
  }
  const list = Array.isArray(payload) ? payload : payload.tags ?? [];
  await importTagList(list, recordingId, summary);
}

async function importKitDir(kitDir: string, summary: ImportSummary) {
  const manifest = await readJson<SessionManifest>(`${kitDir}/manifest.json`);
  const recordingId = await resolveRecordingId({
    recordingUuid: manifest?.recordingUuid,
    audioContentHash: manifest?.audioContentHash,
    recordingId: manifest?.recordingId,
  });

  if (!recordingId) {
    summary.unmatched += 1;
    summary.errors.push(`No matching recording for kit ${kitDir.split('/').pop()}`);
    return;
  }

  const tagsPath = `${kitDir}/tags.json`;
  if (await RNFS.exists(tagsPath)) {
    await importTagsFile(tagsPath, recordingId, summary);
  }

  const annPath = `${kitDir}/annotations.json`;
  if (await RNFS.exists(annPath)) {
    await importAnnotationsFile(annPath, recordingId, summary);
  }
}

/**
 * Import annotations from Documents/Import (and system Inbox if present).
 * Merge rules: user wins; provisional ml may update while provisional; confirmed sticky.
 * Re-links by audio content hash first, then recording UUID.
 */
export async function importInboxAnnotations(): Promise<ImportSummary> {
  await ensureDir(IMPORT_DIR);
  const summary: ImportSummary = {
    scanned: 0,
    inserted: 0,
    updated: 0,
    skipped: 0,
    unmatched: 0,
    errors: [],
  };

  const roots = [IMPORT_DIR, SYSTEM_INBOX_DIR];
  for (const root of roots) {
    const entries = await listDirIfExists(root);
    for (const entry of entries) {
      if (entry.isDirectory()) {
        await importKitDir(entry.path, summary);
        continue;
      }
      if (entry.name === 'annotations.json' || entry.name.endsWith('_annotations.json')) {
        const parent = entry.path.replace(/\/[^/]+$/, '');
        const manifest = await readJson<SessionManifest>(`${parent}/manifest.json`);
        const recordingId = await resolveRecordingId({
          recordingUuid: manifest?.recordingUuid,
          audioContentHash: manifest?.audioContentHash,
        });
        if (!recordingId) {
          summary.unmatched += 1;
          summary.errors.push(`Loose annotations need a kit/manifest: ${entry.name}`);
          continue;
        }
        await importAnnotationsFile(entry.path, recordingId, summary);
      }
    }
  }

  return summary;
}
