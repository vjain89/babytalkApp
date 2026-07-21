import { Alert, DeviceEventEmitter } from 'react-native';
import { ensureBackupDirs, importInboxAnnotations } from './backup';
import { importAudioFile, importAudioFromInbox } from '../audio/importAudio';

export const RECORDINGS_CHANGED_EVENT = 'babytalk:recordings-changed';

function notifyRecordingsChanged() {
  DeviceEventEmitter.emit(RECORDINGS_CHANGED_EVENT);
}

let importingTags = false;
let importingAudio = false;
let lastTagAlertKey: string | null = null;

/**
 * Merge tags from Documents/Import (USB sync drop). Safe to call often.
 */
export async function runAutoImportAnnotations(opts?: {
  alertOnChange?: boolean;
}): Promise<{ changed: number; unmatched: number }> {
  if (importingTags) return { changed: 0, unmatched: 0 };
  importingTags = true;
  try {
    await ensureBackupDirs();
    const summary = await importInboxAnnotations();
    const changed = summary.inserted + summary.updated;
    if (changed > 0 && opts?.alertOnChange !== false) {
      const key = `${summary.inserted}:${summary.updated}:${summary.scanned}`;
      if (lastTagAlertKey !== key) {
        lastTagAlertKey = key;
        Alert.alert(
          'Tags imported',
          `+${summary.inserted} new · ${summary.updated} updated` +
            (summary.unmatched ? ` · ${summary.unmatched} unmatched` : ''),
        );
      }
      notifyRecordingsChanged();
    }
    return { changed, unmatched: summary.unmatched };
  } finally {
    importingTags = false;
  }
}

/** Import any pending shared / Inbox / Documents-root audio. */
export async function runAutoImportAudio(opts?: {
  alertOnChange?: boolean;
}): Promise<number> {
  if (importingAudio) return 0;
  importingAudio = true;
  try {
    const result = await importAudioFromInbox();
    if (result.imported.length > 0) {
      notifyRecordingsChanged();
      if (opts?.alertOnChange !== false) {
        const names = result.imported.map((r) => r.sessionName).join(', ');
        Alert.alert(
          'Audio imported',
          `Imported ${result.imported.length}: ${names}`,
        );
      }
    }
    return result.imported.length;
  } catch (err) {
    console.warn('Auto-import audio skipped:', err);
    return 0;
  } finally {
    importingAudio = false;
  }
}

/** Import a file:// or path shared via Share → Copy to BabyTalk. */
export async function importSharedAudioUrl(url: string): Promise<void> {
  // Prefer scanning Import — native AppDelegate already copied the file there.
  const n = await runAutoImportAudio({ alertOnChange: true });
  if (n > 0) return;

  const raw = url.replace(/^file:\/\//, '');
  let path = raw;
  try {
    path = decodeURIComponent(raw);
  } catch {
    // keep raw
  }
  if (!path || path.startsWith('babytalk:')) {
    await runAutoImportAudio({ alertOnChange: true });
    return;
  }
  const name = path.split('/').pop() || `shared_${Date.now()}.m4a`;
  const result = await importAudioFile({ uri: path, name });
  notifyRecordingsChanged();
  Alert.alert(
    'Imported audio',
    `“${result.sessionName}” (${(result.durationMs / 1000).toFixed(1)}s)`,
  );
}
