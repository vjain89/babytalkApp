import SQLite from 'react-native-sqlite-storage';

SQLite.enablePromise(true);

// Open or create the database
export const getDb = async () => {
    return SQLite.openDatabase({ name: 'babytalk.db', location: 'default' });
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

    // Migration: Add waveform_data column if it doesn't exist (for existing databases)
    try {
      await db.executeSql(`
        ALTER TABLE recordings ADD COLUMN waveform_data TEXT;
      `);
    } catch (err) {
      // Column already exists, ignore error
    }

    await db.executeSql(`
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            label TEXT NOT NULL,
            FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
        );
    `);

    console.log('✅ Database initialized');
    };

// Add a new recording
export const addRecording = async ({
    filename,
    sessionName,
    durationMs,
    waveformData,
}: {
    filename: string;
    sessionName?: string;
    durationMs: number;
    /** Serialized WaveformPayload JSON (raw dB samples + metadata), or null. */
    waveformData?: string | null;
}): Promise<number> => {
    const db = await getDb();
    const createdAt = Date.now();
    const waveformJson = waveformData ?? null;

    const [result] = await db.executeSql(
        `INSERT INTO recordings (filename, session_name, created_at, duration_ms, waveform_data)
         VALUES (?, ?, ?, ?, ?)`,
        [filename, sessionName || null, createdAt, durationMs, waveformJson]
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

// Get all recordings, ordered by newest
export const getAllRecordings = async (sort: 'newest' | 'oldest' | 'longest' | 'shortest' = 'newest') => {
    const db = await getDb();

    let orderBy = 'created_at DESC';
    if (sort === 'oldest') orderBy = 'created_at ASC';
    if (sort === 'longest') orderBy = 'duration_ms DESC';
    if (sort === 'shortest') orderBy = 'duration_ms ASC';

    const [result] = await db.executeSql(`
        SELECT * FROM recordings ORDER BY ${orderBy}
    `);

    return result.rows.raw(); // returns an array of { id, filename, session_name, ... }
};

// Get a recording by ID
export const getRecordingById = async (id: number) => {
    const db = await getDb();

    const [result] = await db.executeSql(
        `SELECT * FROM recordings WHERE id = ?`,
        [id]
    );

    if (result.rows.length > 0) {
        return result.rows.item(0);
    }
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
        [startTimestamp]
    );

    return result.rows.item(0).count;
};

// Add a tag to a recording
export const addTag = async ({
    recordingId,
    timestampMs,
    label,
}: {
    recordingId: number;
    timestampMs: number;
    label: string;
}): Promise<number> => {
    const db = await getDb();

    const [result] = await db.executeSql(
        `INSERT INTO tags (recording_id, timestamp_ms, label)
         VALUES (?, ?, ?)`,
        [recordingId, timestampMs, label]
    );

    return result.insertId!;
};

// Get all tags for a recording
export const getTagsForRecording = async (recordingId: number) => {
    const db = await getDb();

    const [result] = await db.executeSql(
        `SELECT * FROM tags WHERE recording_id = ? ORDER BY timestamp_ms`,
        [recordingId]
    );

    return result.rows.raw(); // returns array of tags
};

export const updateTagLabel = async (tagId: number, newLabel: string): Promise<void> => {
    const db = await getDb();
    await db.executeSql(
        `UPDATE tags SET label = ? WHERE id = ?`,
        [newLabel, tagId]
    );
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
        [`%${labelQuery.toLowerCase()}%`]
    );

    return results.rows.raw(); // same format as getAllRecordings
};

export const updateSessionName = async (id: number, newName: string) => {
    const db = await getDb();
    await db.executeSql(`UPDATE recordings SET session_name = ? WHERE id = ?`, [newName, id]);
};