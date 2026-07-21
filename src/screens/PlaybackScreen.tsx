import React, { useEffect, useMemo, useState } from 'react';
import {
  View,
  Text,
  Button,
  StyleSheet,
  FlatList,
  Alert,
  Platform,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import AudioRecorderPlayer from '../../react-native-audio-recorder-player';
import { RouteProp, useRoute } from '@react-navigation/native';
import {
  addTag,
  getTagsForRecording,
  getDb,
  updateTagLabel,
  updateWaveformData,
  type TagRow,
} from '../db';
import Share from 'react-native-share';
import Waveform from '../components/Waveform';
import RecordingLayout from '../components/RecordingLayout';
import CircularPlayButton from '../components/CircularPlayButton';
import { prepareSingleBackup } from '../export/backup';
import { buildSessionKitForShare } from '../export/sessionKit';
import { PLAYBACK_BAR_MS, PLAYBACK_WINDOW_MS, TAG_MARKER_MS } from '../waveform/config';
import {
  computeBipolarAmpScale,
  computePlaybackDbRange,
  type DbRange,
} from '../waveform/scale';
import {
  densifyBipolar,
  densifyPeaks,
  parseWaveformData,
  serializeWaveform,
} from '../waveform/storage';
import { resolveAudioUri } from '../waveform/audioPath';
import { emptyWaveformPayload, type WaveformSample } from '../waveform/types';

const audioPlayer = new AudioRecorderPlayer();

type PlaybackScreenProps = {
  route: RouteProp<
    { params: { recordingId: number; filePath: string; filename: string } },
    'params'
  >;
};

export default function PlaybackScreen() {
  const { recordingId, filePath, filename } = useRoute<PlaybackScreenProps['route']>().params;

  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackMs, setPlaybackMs] = useState(0);
  const [durationMs, setDurationMs] = useState(1);
  const [resolvedUri, setResolvedUri] = useState<string | null>(null);

  const [tags, setTags] = useState<TagRow[]>([]);
  const [highlightedTagId, setHighlightedTagId] = useState<number | null>(null);

  const [samples, setSamples] = useState<WaveformSample[]>([]);
  const [barDurationMs, setBarDurationMs] = useState(PLAYBACK_BAR_MS);
  const [dbRange, setDbRange] = useState<DbRange>(() => computePlaybackDbRange([]));
  const [bipolarScale, setBipolarScale] = useState(1);
  const [hasBipolar, setHasBipolar] = useState(false);
  const [waveformView, setWaveformView] = useState<'full' | 'rolling'>('rolling');
  const [scrubPreviewMs, setScrubPreviewMs] = useState<number | null>(null);
  const [scrubWindowEndMs, setScrubWindowEndMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  const displayMs = scrubPreviewMs ?? playbackMs;
  const rollingWindowEndMs = scrubWindowEndMs ?? playbackMs;

  const tagStartMs = (t: TagRow) => t.start_ms ?? t.timestamp_ms;

  const handleScrubChange = (ms: number | null) => {
    if (ms === null) {
      setScrubPreviewMs(null);
      setScrubWindowEndMs(null);
      return;
    }
    setScrubWindowEndMs((prev) => prev ?? playbackMs);
    setScrubPreviewMs(ms);
  };

  useEffect(() => {
    loadTags();
    loadRecordingWaveform();
    return () => {
      audioPlayer.removePlayBackListener();
      void audioPlayer.stopPlayer().catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadTags = async () => {
    try {
      const tagList = await getTagsForRecording(recordingId);
      setTags(tagList);
    } catch (err) {
      console.error('❌ Failed to load tags:', err);
    }
  };

  const loadRecordingWaveform = async () => {
    try {
      const uri = await resolveAudioUri(filePath);
      setResolvedUri(uri);

      const db = await getDb();
      const [res] = await db.executeSql(
        `SELECT duration_ms, waveform_data FROM recordings WHERE id = ? LIMIT 1`,
        [recordingId],
      );
      if (res.rows.length > 0) {
        const row = res.rows.item(0);
        if (typeof row.duration_ms === 'number' && row.duration_ms > 0) {
          setDurationMs(row.duration_ms);
        }

        let payload = parseWaveformData(row.waveform_data);
        const needsA1 =
          payload.source !== 'file' ||
          payload.samples.length === 0 ||
          payload.barDurationMs > PLAYBACK_BAR_MS ||
          !payload.samples.some((s) => s.minAmp != null && s.maxAmp != null);

        if (needsA1 && uri) {
          try {
            const extracted = await audioPlayer.extractWaveformPeaks(
              uri,
              PLAYBACK_BAR_MS,
              true,
            );
            if (extracted.length > 0) {
              payload = emptyWaveformPayload('file');
              payload.barDurationMs = PLAYBACK_BAR_MS;
              payload.samples = extracted;
              await updateWaveformData(recordingId, serializeWaveform(payload));
            }
          } catch (extractErr) {
            console.warn('⚠️ Playback file peak extract failed:', extractErr);
          }
        }

        setSamples(payload.samples);
        setBarDurationMs(payload.barDurationMs || PLAYBACK_BAR_MS);
        setDbRange(computePlaybackDbRange(payload.samples));
        setBipolarScale(computeBipolarAmpScale(payload.samples));
        setHasBipolar(payload.samples.some((s) => s.minAmp != null && s.maxAmp != null));

        // Fallback duration from waveform or native probe when DB has 0.
        if ((!row.duration_ms || row.duration_ms <= 0) && payload.samples.length > 0) {
          const last = payload.samples[payload.samples.length - 1];
          const fromWave = Math.round(last.tMs + (payload.barDurationMs || PLAYBACK_BAR_MS));
          if (fromWave > 0) setDurationMs(fromWave);
        } else if ((!row.duration_ms || row.duration_ms <= 0) && uri) {
          try {
            const ms = Math.round(await audioPlayer.getAudioDurationMs(uri));
            if (ms > 0) setDurationMs(ms);
          } catch {
            // leave duration as-is
          }
        }
      }
    } catch (err) {
      console.error('❌ Failed to load waveform_data:', err);
    }
  };

  const startPlaying = async () => {
    try {
      const uri = resolvedUri ?? (await resolveAudioUri(filePath));
      if (!uri) {
        Alert.alert('Missing audio', 'Could not find the audio file on disk.');
        return;
      }
      setResolvedUri(uri);

      await audioPlayer.startPlayer(uri);

      if (playbackMs > 0) {
        await audioPlayer.seekToPlayer(playbackMs);
      }

      audioPlayer.addPlayBackListener((e) => {
        if (typeof e.currentPosition === 'number') setPlaybackMs(e.currentPosition);
        if (typeof e.duration === 'number' && e.duration > 0 && e.duration !== durationMs) {
          setDurationMs(e.duration);
        }
        if (e.currentPosition >= e.duration) stopPlaying();
        return;
      });

      setIsPlaying(true);
    } catch (err) {
      console.error('❌ Failed to play audio:', err);
      Alert.alert('Playback failed', String(err));
    }
  };

  const stopPlaying = async () => {
    try {
      await audioPlayer.stopPlayer();
      audioPlayer.removePlayBackListener();
      setIsPlaying(false);
    } catch (err) {
      console.error('❌ Failed to stop audio:', err);
    }
  };

  const handlePlayPause = () => {
    if (isPlaying) stopPlaying();
    else startPlaying();
  };

  const seekTo = async (ms: number) => {
    const clamped = Math.max(0, Math.min(durationMs, Math.floor(ms)));
    setPlaybackMs(clamped);
    setScrubPreviewMs(null);
    setScrubWindowEndMs(null);
    setHighlightedTagId(null);
    try {
      await audioPlayer.seekToPlayer(clamped);
    } catch {
      // Player may not be started yet; position is stored for the next play.
    }
  };

  const handleTag = async () => {
    const ts = Math.floor(displayMs);
    let wasPlaying = false;

    if (isPlaying) {
      await audioPlayer.pausePlayer();
      setIsPlaying(false);
      wasPlaying = true;
    }

    const promptForLabel = async (label: string) => {
      try {
        await addTag({
          recordingId,
          timestampMs: ts,
          label: label.trim() || 'Untitled tag',
        });
        await loadTags();
      } catch (err) {
        console.error('❌ Failed to save tag:', err);
      }

      if (wasPlaying) {
        await audioPlayer.resumePlayer();
        setIsPlaying(true);
      }
    };

    if (Platform.OS === 'ios') {
      Alert.prompt('Add Tag', `Timestamp: ${(ts / 1000).toFixed(1)}s`, promptForLabel);
    } else {
      const label = prompt('Enter tag label');
      if (label !== null) promptForLabel(label);
    }
  };

  const handleExport = async () => {
    setBusy(true);
    try {
      const { urls } = await buildSessionKitForShare(recordingId);
      await Share.open({
        title: 'Export Session Kit',
        message: 'Session kit (WAV + manifest + tags)',
        urls,
        failOnCancel: false,
      });
    } catch (err) {
      console.error('❌ Failed to export:', err);
      Alert.alert('Export failed', String(err));
    } finally {
      setBusy(false);
    }
  };

  const handlePrepareBackup = async () => {
    setBusy(true);
    try {
      const dest = await prepareSingleBackup(recordingId);
      Alert.alert(
        'Backup ready',
        `Written to Documents/Backups.\nPlug into your Mac and copy via Finder:\n${dest.replace(
          /.*\/Documents\//,
          'Documents/',
        )}`,
      );
    } catch (err) {
      console.error('❌ Backup failed:', err);
      Alert.alert('Backup failed', String(err));
    } finally {
      setBusy(false);
    }
  };

  const peaksDb = useMemo(() => {
    if (hasBipolar || samples.length === 0) return [];

    if (waveformView === 'full') {
      const end = Math.max(durationMs, samples[samples.length - 1]?.tMs ?? 0) + barDurationMs;
      return densifyPeaks(samples, 0, end, barDurationMs);
    }

    return densifyPeaks(
      samples,
      rollingWindowEndMs - PLAYBACK_WINDOW_MS,
      rollingWindowEndMs,
      barDurationMs,
    );
  }, [samples, hasBipolar, waveformView, durationMs, rollingWindowEndMs, barDurationMs]);

  const bipolarColumns = useMemo(() => {
    if (!hasBipolar || samples.length === 0) return undefined;

    if (waveformView === 'full') {
      const end = Math.max(durationMs, samples[samples.length - 1]?.tMs ?? 0) + barDurationMs;
      return densifyBipolar(samples, 0, end, barDurationMs);
    }

    return densifyBipolar(
      samples,
      rollingWindowEndMs - PLAYBACK_WINDOW_MS,
      rollingWindowEndMs,
      barDurationMs,
    );
  }, [samples, hasBipolar, waveformView, durationMs, rollingWindowEndMs, barDurationMs]);

  return (
    <RecordingLayout
      title={filename}
      durationLabel={`⏱️ ${(displayMs / 1000).toFixed(1)}s / ${(durationMs / 1000).toFixed(1)}s`}
      waveform={
        samples.length > 0 ? (
          <Waveform
            peaksDb={peaksDb}
            bipolarColumns={bipolarColumns}
            bipolarScale={bipolarScale}
            barDurationMs={barDurationMs}
            progressMs={playbackMs}
            scrubPreviewMs={scrubPreviewMs}
            rollingWindowEndMs={rollingWindowEndMs}
            durationMs={durationMs}
            windowMs={PLAYBACK_WINDOW_MS}
            mode={waveformView}
            cursorMode="follow"
            showCursor={true}
            dbRange={dbRange}
            minBarPx={1}
            tagTimestamps={tags.map(tagStartMs)}
            tagWidthMs={TAG_MARKER_MS}
            seekable
            onSeek={seekTo}
            onScrubChange={handleScrubChange}
          />
        ) : (
          <Text style={{ textAlign: 'center', color: '#888' }}>
            No waveform data saved for this recording.
          </Text>
        )
      }
      controls={
        <View style={{ alignItems: 'center', gap: 12 }}>
          <CircularPlayButton isPlaying={isPlaying} onPress={handlePlayPause} />

          <TouchableOpacity
            onPress={() => setWaveformView((v) => (v === 'full' ? 'rolling' : 'full'))}
            style={styles.toggleBtn}
          >
            <Text style={styles.toggleText}>
              {waveformView === 'full' ? '📉 Switch to 8s Scroll View' : '🗺️ Switch to Full Waveform'}
            </Text>
          </TouchableOpacity>

          <Button title="🏷️ Tag This Moment" onPress={handleTag} />
          <Text style={styles.seekHint}>Tap or drag the waveform to seek</Text>
          {busy ? (
            <ActivityIndicator />
          ) : (
            <>
              <Button title="📤 Export Session Kit" onPress={handleExport} />
              <Button title="💾 Prepare USB Backup" onPress={handlePrepareBackup} />
            </>
          )}
        </View>
      }
    >
      <Text style={styles.sectionTitle}>Tags</Text>

      {tags.length === 0 ? (
        <Text style={styles.empty}>No tags yet.</Text>
      ) : (
        <FlatList
          data={tags}
          keyExtractor={(item) => item.uuid || item.id.toString()}
          renderItem={({ item }) => (
            <TouchableOpacity
              onPress={() => {
                void seekTo(tagStartMs(item));
                setHighlightedTagId(item.id);
              }}
              onLongPress={() => {
                Alert.alert('Tag Options', `"${item.label}"`, [
                  {
                    text: 'Edit',
                    onPress: () => {
                      if (Platform.OS === 'ios') {
                        Alert.prompt('Edit Tag', `Current: "${item.label}"`, async (newLabel) => {
                          if (newLabel && newLabel.trim()) {
                            await updateTagLabel(item.id, newLabel.trim());
                            await loadTags();
                          }
                        });
                      }
                    },
                  },
                  { text: 'Cancel', style: 'cancel' },
                ]);
              }}
              style={[styles.tagItem, item.id === highlightedTagId && { backgroundColor: '#fff3c4' }]}
            >
              <Text style={styles.tagLabel}>
                {item.label}
                {item.source && item.source !== 'user' ? ` · ${item.source}` : ''}
              </Text>
              <Text style={styles.tagTime}>{(tagStartMs(item) / 1000).toFixed(1)}s</Text>
            </TouchableOpacity>
          )}
        />
      )}
    </RecordingLayout>
  );
}

const styles = StyleSheet.create({
  sectionTitle: { fontSize: 18, marginTop: 30, marginBottom: 10, fontWeight: 'bold' },
  empty: { textAlign: 'center', color: '#888', marginTop: 20 },
  tagItem: {
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderColor: '#eee',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  tagLabel: { fontSize: 16 },
  tagTime: { color: '#666' },

  toggleBtn: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: '#eee',
  },
  toggleText: {
    color: '#007AFF',
    fontSize: 14,
  },
  seekHint: {
    fontSize: 12,
    color: '#888',
    textAlign: 'center',
  },
});
