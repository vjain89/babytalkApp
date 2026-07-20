import SQLite from 'react-native-sqlite-storage';
import { uuidv4 } from './utils/uuid';

SQLite.enablePromise(true);

export type TagSource = 'user' | 'ml' | 'ml_confirmed';
export type TagStatus = 'provisional' | 'confirmed';

export type TagRow = {
  id: number;
  recording_id: number;
  timestamp_ms: number;
  label: string;
  uuid: string;
  start_ms: number;
  end_ms: number | null;
  source: TagSource;
  status: TagStatus;
};

// Open or create the database
export const getDb = async () => {
  return SQLite.openDatabase({ name: 'babytalk.db', location: 'default' });
};

const tableHasColumn = async (
  db: Awaited<ReturnType<typeof getDb>>,
  table: string,
  column: string,
): Promise<boolean> => {
  const [result] = await db.executeSql(`PRAGMA table_info(${table})`);
  for (let i = 0; i < result.rows.length; i++) {
    if (result.rows.item(i).name === column) return true;
  }
  return false;
};

const addColumnIfMissing = async (
  db: Awaited<ReturnType<typeof getDb>>,
  table: string,
  column: string,
  ddl: string,
) => {
  if (!(await tableHasColumn(db, table, column))) {
    await db.executeSql(`ALTER TABLE ${table} ADD COLUMN ${ddl}`);
  }
};

// Create tables if they don't exist
export const initDb = async () => {
  const db = await getDb();

  await db.executeSql(`
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            session_name TEXT,
            created_at INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL,
            waveform_data TEXT
        );
    `);

  await addColumnIfMissing(db, 'recordings', 'waveform_data', 'waveform_data TEXT');
  await addColumnIfMissing(db, 'recordings', 'uuid', 'uuid TEXT');
  await addColumnIfMissing(db, 'recordings', 'audio_content_hash', 'audio_content_hash TEXT');

  await db.executeSql(`
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            label TEXT NOT NULL,
            FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
        );
    `);

  // Annotation fields (roadmap §8). Keep int PK; timestamp_ms retained as start alias.
  await addColumnIfMissing(db, 'tags', 'uuid', 'uuid TEXT');
  await addColumnIfMissing(db, 'tags', 'start_ms', 'start_ms INTEGER');
  await addColumnIfMissing(db, 'tags', 'end_ms', 'end_ms INTEGER');
  await addColumnIfMissing(db, 'tags', 'source', "source TEXT DEFAULT 'user'");
  await addColumnIfMissing(db, 'tags', 'status', "status TEXT DEFAULT 'confirmed'");
  await addColumnIfMissing(db, 'tags', 'imported_at', 'imported_at INTEGER');

  // Backfill recordings.uuid
  const [recRows] = await db.executeSql(
    `SELECT id FROM recordings WHERE uuid IS NULL OR uuid = ''`,
  );
  for (let i = 0; i < recRows.rows.length; i++) {
    await db.executeSql(`UPDATE recordings SET uuid = ? WHERE id = ?`, [
      uuidv4(),
      recRows.rows.item(i).id,
    ]);
  }

  // Backfill tag annotation columns from legacy timestamp_ms / label rows
  const [tagRows] = await db.executeSql(
    `SELECT id, timestamp_ms FROM tags WHERE uuid IS NULL OR uuid = '' OR start_ms IS NULL`,
  );
  for (let i = 0; i < tagRows.rows.length; i++) {
    const row = tagRows.rows.item(i);
    await db.executeSql(
      `UPDATE tags SET
          uuid = COALESCE(NULLIF(uuid, ''), ?),
          start_ms = COALESCE(start_ms, ?),
          source = COALESCE(source, 'user'),
          status = COALESCE(status, 'confirmed')
        WHERE id = ?`,
      [uuidv4(), row.timestamp_ms, row.id],
    );
  }

  await db.executeSql(
    `CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_uuid ON tags(uuid)`,
  );
  await db.executeSql(
    `CREATE UNIQUE INDEX IF NOT EXISTS idx_recordings_uuid ON recordings(uuid)`,
  );

  console.log('✅ Database initialized');
};

// Add a new recording
export const addRecording = async ({
  filename,
  sessionName,
  durationMs,
  waveformData,
  audioContentHash,
}: {
  filename: string;
  sessionName?: string;
  durationMs: number;
  /** Serialized WaveformPayload JSON (raw dB samples + metadata), or null. */
  waveformData?: string | null;
  audioContentHash?: string | null;
}): Promise<number> => {
  const db = await getDb();
  const createdAt = Date.now();
  const waveformJson = waveformData ?? null;
  const recordingUuid = uuidv4();

  const [result] = await db.executeSql(
    `INSERT INTO recordings (filename, session_name, created_at, duration_ms, waveform_data, uuid, audio_content_hash)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
    [
      filename,
      sessionName || null,
      createdAt,
      durationMs,
      waveformJson,
      recordingUuid,
      audioContentHash ?? null,
    ],
  );

  return result.insertId!;
};

export const updateWaveformData = async (
  recordingId: number,
  waveformData: string,
): Promise<void> => {
  const db = await getDb();
  await db.executeSql(`UPDATE recordings SET waveform_data = ? WHERE id = ?`, [
    waveformData,
    recordingId,
  ]);
};

export const updateAudioContentHash = async (
  recordingId: number,
  hash: string,
): Promise<void> => {
  const db = await getDb();
  await db.executeSql(`UPDATE recordings SET audio_content_hash = ? WHERE id = ?`, [
    hash,
    recordingId,
  ]);
};

// Get all recordings, ordered by newest
export const getAllRecordings = async (
  sort: 'newest' | 'oldest' | 'longest' | 'shortest' = 'newest',
) => {
  const db = await getDb();

  let orderBy = 'created_at DESC';
  if (sort === 'oldest') orderBy = 'created_at ASC';
  if (sort === 'longest') orderBy = 'duration_ms DESC';
  if (sort === 'shortest') orderBy = 'duration_ms ASC';

  const [result] = await db.executeSql(`
        SELECT * FROM recordings ORDER BY ${orderBy}
    `);

  return result.rows.raw();
};

// Get a recording by ID
export const getRecordingById = async (id: number) => {
  const db = await getDb();

  const [result] = await db.executeSql(`SELECT * FROM recordings WHERE id = ?`, [id]);

  if (result.rows.length > 0) {
    return result.rows.item(0);
  }
  return null;
};

export const getRecordingByUuid = async (uuid: string) => {
  const db = await getDb();
  const [result] = await db.executeSql(`SELECT * FROM recordings WHERE uuid = ?`, [uuid]);
  if (result.rows.length > 0) return result.rows.item(0);
  return null;
};

export const getRecordingByAudioHash = async (hash: string) => {
  const db = await getDb();
  const [result] = await db.executeSql(
    `SELECT * FROM recordings WHERE audio_content_hash = ?`,
    [hash],
  );
  if (result.rows.length > 0) return result.rows.item(0);
  return null;
};

// Get the count of recordings created today
export const getTodayRecordingCount = async (): Promise<number> => {
  const db = await getDb();

  const startOfDay = new Date();
  startOfDay.setHours(0, 0, 0, 0);
  const startTimestamp = startOfDay.getTime();

  const [result] = await db.executeSql(
    `SELECT COUNT(*) as count FROM recordings WHERE created_at >= ?`,
    [startTimestamp],
  );

  return result.rows.item(0).count;
};

// Add a tag to a recording (user-confirmed by default)
export const addTag = async ({
  recordingId,
  timestampMs,
  label,
  endMs,
  source = 'user',
  status = 'confirmed',
  uuid,
}: {
  recordingId: number;
  timestampMs: number;
  label: string;
  endMs?: number | null;
  source?: TagSource;
  status?: TagStatus;
  uuid?: string;
}): Promise<number> => {
  const db = await getDb();
  const tagUuid = uuid ?? uuidv4();
  const startMs = timestampMs;

  const [result] = await db.executeSql(
    `INSERT INTO tags (recording_id, timestamp_ms, label, uuid, start_ms, end_ms, source, status)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [recordingId, startMs, label, tagUuid, startMs, endMs ?? null, source, status],
  );

  return result.insertId!;
};

/** Upsert an annotation by UUID (import path). User tags win; provisional ml can update. */
export const upsertAnnotationByUuid = async ({
  recordingId,
  uuid,
  label,
  startMs,
  endMs,
  source,
  status,
}: {
  recordingId: number;
  uuid: string;
  label: string;
  startMs: number;
  endMs?: number | null;
  source: TagSource;
  status: TagStatus;
}): Promise<'inserted' | 'updated' | 'skipped'> => {
  const db = await getDb();
  const [existing] = await db.executeSql(`SELECT * FROM tags WHERE uuid = ?`, [uuid]);

  if (existing.rows.length === 0) {
    await db.executeSql(
      `INSERT INTO tags (recording_id, timestamp_ms, label, uuid, start_ms, end_ms, source, status, imported_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        recordingId,
        startMs,
        label,
        uuid,
        startMs,
        endMs ?? null,
        source,
        status,
        Date.now(),
      ],
    );
    return 'inserted';
  }

  const row = existing.rows.item(0) as TagRow;
  // Never overwrite a human label with ML.
  if (row.source === 'user' && source !== 'user') {
    return 'skipped';
  }
  // Confirmed edits are sticky unless the incoming row is also confirmed user/ml_confirmed.
  if (row.status === 'confirmed' && status === 'provisional') {
    return 'skipped';
  }

  await db.executeSql(
    `UPDATE tags SET
        recording_id = ?,
        timestamp_ms = ?,
        label = ?,
        start_ms = ?,
        end_ms = ?,
        source = ?,
        status = ?,
        imported_at = ?
      WHERE uuid = ?`,
    [
      recordingId,
      startMs,
      label,
      startMs,
      endMs ?? null,
      source,
      status,
      Date.now(),
      uuid,
    ],
  );
  return 'updated';
};

// Get all tags for a recording
export const getTagsForRecording = async (recordingId: number): Promise<TagRow[]> => {
  const db = await getDb();

  const [result] = await db.executeSql(
    `SELECT * FROM tags WHERE recording_id = ? ORDER BY COALESCE(start_ms, timestamp_ms)`,
    [recordingId],
  );

  return result.rows.raw();
};

export const updateTagLabel = async (tagId: number, newLabel: string): Promise<void> => {
  const db = await getDb();
  await db.executeSql(`UPDATE tags SET label = ? WHERE id = ?`, [newLabel, tagId]);
};

// Get recordings by matching tag label (case-insensitive substring match)
export const getRecordingsByTagLabel = async (labelQuery: string) => {
  const db = await getDb();

  const [results] = await db.executeSql(
    `
        SELECT DISTINCT r.*
        FROM recordings r
        JOIN tags t ON r.id = t.recording_id
        WHERE LOWER(t.label) LIKE ?
        ORDER BY r.created_at DESC
        `,
    [`%${labelQuery.toLowerCase()}%`],
  );

  return results.rows.raw();
};

export const updateSessionName = async (id: number, newName: string) => {
  const db = await getDb();
  await db.executeSql(`UPDATE recordings SET session_name = ? WHERE id = ?`, [
    newName,
    id,
  ]);
};
