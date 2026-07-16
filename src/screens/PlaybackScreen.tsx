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
} from 'react-native';
import AudioRecorderPlayer from '../../react-native-audio-recorder-player';
import { RouteProp, useRoute } from '@react-navigation/native';
import { addTag, getTagsForRecording, getDb, updateTagLabel } from '../db';
import RNFS from 'react-native-fs';
import Share from 'react-native-share';
import Waveform from '../components/Waveform';
import RecordingLayout from '../components/RecordingLayout';
import CircularPlayButton from '../components/CircularPlayButton';
import { BAR_MS, PLAYBACK_WINDOW_MS, TAG_MARKER_MS } from '../waveform/config';
import { computePlaybackDbRange, type DbRange } from '../waveform/scale';
import { densifyPeaks, parseWaveformData } from '../waveform/storage';
import type { WaveformSample } from '../waveform/types';

const audioPlayer = new AudioRecorderPlayer();

type PlaybackScreenProps = {
  route: RouteProp<{ params: { recordingId: number; filePath: string; filename: string } }, 'params'>;
};

type Tag = {
  id: number;
  label: string;
  timestamp_ms: number;
};

export default function PlaybackScreen() {
  const { recordingId, filePath, filename } = useRoute<PlaybackScreenProps['route']>().params;

  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackMs, setPlaybackMs] = useState(0);
  const [durationMs, setDurationMs] = useState(1);

  const [tags, setTags] = useState<Tag[]>([]);
  const [highlightedTagId, setHighlightedTagId] = useState<number | null>(null);

  const [samples, setSamples] = useState<WaveformSample[]>([]);
  const [barDurationMs, setBarDurationMs] = useState(BAR_MS);
  const [dbRange, setDbRange] = useState<DbRange>(() => computePlaybackDbRange([]));
  const [waveformView, setWaveformView] = useState<'full' | 'rolling'>('rolling');

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
      const db = await getDb();
      const [res] = await db.executeSql(
        `SELECT duration_ms, waveform_data FROM recordings WHERE id = ? LIMIT 1`,
        [recordingId]
      );
      if (res.rows.length > 0) {
        const row = res.rows.item(0);
        if (typeof row.duration_ms === 'number') setDurationMs(row.duration_ms);

        const payload = parseWaveformData(row.waveform_data);
        setSamples(payload.samples);
        setBarDurationMs(payload.barDurationMs || BAR_MS);
        setDbRange(computePlaybackDbRange(payload.samples));
      }
    } catch (err) {
      console.error('❌ Failed to load waveform_data:', err);
    }
  };

  const startPlaying = async () => {
    try {
      await audioPlayer.startPlayer(filePath);

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

  const handleTag = async () => {
    const ts = Math.floor(playbackMs);
    let wasPlaying = false;

    if (isPlaying) {
      await audioPlayer.pausePlayer();
      setIsPlaying(false);
      wasPlaying = true;
    }

    const promptForLabel = async (label: string) => {
      try {
        await addTag({ recordingId, timestampMs: ts, label: label.trim() || 'Untitled tag' });
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
    try {
      const exportData = { filename, recordingId, tags };
      const exportJson = JSON.stringify(exportData, null, 2);
      const exportPath = `${RNFS.CachesDirectoryPath}/${filename.replace(/\s/g, '_')}_tags.json`;
      await RNFS.writeFile(exportPath, exportJson, 'utf8');

      const audioPath = filePath.startsWith('file://') ? filePath : `file://${filePath}`;
      const tagPath = `file://${exportPath}`;

      await Share.open({
        title: 'Export Recording & Tags',
        message: 'Sharing the recording and associated tags',
        urls: [audioPath, tagPath],
        failOnCancel: false,
      });
    } catch (err) {
      console.error('❌ Failed to export:', err);
    }
  };

  const peaksDb = useMemo(() => {
    if (samples.length === 0) return [];

    if (waveformView === 'full') {
      const end = Math.max(durationMs, samples[samples.length - 1]?.tMs ?? 0) + barDurationMs;
      return densifyPeaks(samples, 0, end, barDurationMs);
    }

    // Trailing window ending at playhead (left-pad if near start).
    return densifyPeaks(
      samples,
      playbackMs - PLAYBACK_WINDOW_MS,
      playbackMs,
      barDurationMs
    );
  }, [samples, waveformView, durationMs, playbackMs, barDurationMs]);

  return (
    <RecordingLayout
      title={filename}
      durationLabel={`⏱️ ${(playbackMs / 1000).toFixed(1)}s`}
      waveform={
        samples.length > 0 ? (
          <Waveform
            peaksDb={peaksDb}
            barDurationMs={barDurationMs}
            progressMs={playbackMs}
            durationMs={durationMs}
            windowMs={PLAYBACK_WINDOW_MS}
            mode={waveformView}
            cursorMode="follow"
            showCursor={true}
            dbRange={dbRange}
            minBarPx={1}
            tagTimestamps={tags.map((t) => t.timestamp_ms)}
            tagWidthMs={TAG_MARKER_MS}
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
          <Button title="📤 Export Recording + Tags" onPress={handleExport} />
        </View>
      }
    >
      <Text style={styles.sectionTitle}>Tags</Text>

      {tags.length === 0 ? (
        <Text style={styles.empty}>No tags yet.</Text>
      ) : (
        <FlatList
          data={tags}
          keyExtractor={(item) => item.id.toString()}
          renderItem={({ item }) => (
            <TouchableOpacity
              onPress={() => {
                audioPlayer.seekToPlayer(item.timestamp_ms);
                setPlaybackMs(item.timestamp_ms);
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
              <Text style={styles.tagLabel}>{item.label}</Text>
              <Text style={styles.tagTime}>{(item.timestamp_ms / 1000).toFixed(1)}s</Text>
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
});
