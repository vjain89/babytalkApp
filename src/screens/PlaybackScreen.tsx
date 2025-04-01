import React, { useEffect, useState } from 'react';
import { View, Text, Button, FlatList, Alert, Platform, TouchableOpacity, StyleSheet } from 'react-native';
import { RouteProp, useRoute } from '@react-navigation/native';
import RNFS from 'react-native-fs';
import Share from 'react-native-share';
import AudioRecorderPlayer from '../../react-native-audio-recorder-player';
import { addTag, getTagsForRecording, getDb, updateTagLabel } from '../db';
import RecordingLayout from '../components/RecordingLayout';
import Waveform from '../components/Waveform';
import CircularPlayButton from '../components/CircularPlayButton';

const audioPlayer = new AudioRecorderPlayer();

export default function PlaybackScreen() {
  const { recordingId, filePath, filename } = useRoute<RouteProp<{ params: { recordingId: number; filePath: string; filename: string } }, 'params'>>().params;

  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSecs, setPlaybackSecs] = useState(0);
  const [durationMs, setDurationMs] = useState(1);
  const [tags, setTags] = useState([]);
  const [highlightedTagId, setHighlightedTagId] = useState(null);
  const [peaks, setPeaks] = useState<number[]>([]);

  const loadTags = async () => {
    try {
      const tagList = await getTagsForRecording(recordingId);
      setTags(tagList);
    } catch (err) {
      console.error('❌ Failed to load tags:', err);
    }
  };

  const loadPeaks = async () => {
    try {
        const db = await getDb();
        const result = await db.executeSql('SELECT peaks_json FROM recordings WHERE id = ?', [recordingId]);
        const json = result[0].rows.item(0).peaks_json;
        if (json) {
            const parsed = JSON.parse(json);
            console.log('📈 Loaded peaks:', parsed.length, parsed.slice(0, 10));
            setPeaks(parsed);
        }
    } catch (err) {
        console.error('❌ Failed to load waveform peaks:', err);
    }
  };

  useEffect(() => {
    loadTags();
    loadPeaks();
  }, []);

  useEffect(() => {
    (async () => {
        const db = await getDb();
        const result = await db.executeSql('SELECT * FROM recordings');
        const rows = result[0].rows;
        for (let i = 0; i < rows.length; i++) {
            console.log('📄 Recording:', rows.item(i));
        }
    })();
  }, []);

  const startPlaying = async () => {
    try {
      await audioPlayer.startPlayer(filePath);
      audioPlayer.addPlayBackListener((e) => {
        if (typeof e.currentPosition === 'number') {
          setPlaybackSecs(e.currentPosition);
        }
        if (typeof e.duration === 'number' && e.duration > 0) {
          setDurationMs(e.duration);
        }
        if (e.currentPosition >= e.duration) stopPlaying();
        return;
      });
      setIsPlaying(true);
    } catch (err) {
      console.error('❌ Failed to start playback:', err);
    }
  };

  const stopPlaying = async () => {
    try {
      await audioPlayer.stopPlayer();
      audioPlayer.removePlayBackListener();
      setIsPlaying(false);
    } catch (err) {
      console.error('❌ Failed to stop playback:', err);
    }
  };

  const handlePlayPause = () => {
    if (isPlaying) stopPlaying();
    else startPlaying();
  };

  const handleTag = async () => {
    const currentTimestamp = Math.floor(playbackSecs);
    let wasPlaying = false;

    if (isPlaying) {
      await audioPlayer.pausePlayer();
      setIsPlaying(false);
      wasPlaying = true;
    }

    const promptForLabel = async (label: string) => {
      try {
        await addTag({ recordingId, timestampMs: currentTimestamp, label: label.trim() || 'Untitled tag' });
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
      Alert.prompt('Add Tag', `Timestamp: ${(currentTimestamp / 1000).toFixed(1)}s`, promptForLabel);
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
      await Share.open({
        title: 'Export Recording & Tags',
        message: 'Sharing the recording and associated tags',
        urls: [`file://${filePath}`, `file://${exportPath}`],
        failOnCancel: false,
      });
    } catch (err) {
      console.error('❌ Failed to export:', err);
    }
  };

  return (
    <RecordingLayout
      title={filename}
      durationLabel={`⏱️ ${(playbackSecs / 1000).toFixed(1)}s`}
      waveform={
        <Waveform
            peaks={peaks}
            durationMs={durationMs}
            progressMs={playbackSecs}
            tagTimestamps={tags.map(t => t.timestamp_ms)}
            onSeek={(ms) => {
                audioPlayer.seekToPlayer(ms);
                setPlaybackSecs(ms);
                const matchedTag = tags.find(t => Math.abs(t.timestamp_ms - ms) < durationMs / peaks.length);
                setHighlightedTagId(matchedTag?.id ?? null);

                if (!isPlaying) {
                audioPlayer.resumePlayer();
                setIsPlaying(true);
                }
            }}
            highlightedTagId={highlightedTagId}
        />

      }
      controls={
        <>
          <View style={{ alignItems: 'center' }}>
            <CircularPlayButton isPlaying={isPlaying} onPress={handlePlayPause} />
          </View>
          <Button title="🏷️ Tag This Moment" onPress={handleTag} />
          <Button title="📤 Export Recording + Tags" onPress={handleExport} />
        </>
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
            onPress={async () => {
                await audioPlayer.stopPlayer();
                await audioPlayer.startPlayer(filePath);
              
                audioPlayer.addPlayBackListener((e) => {
                  if (typeof e.currentPosition === 'number') {
                    setPlaybackSecs(e.currentPosition);
                  }
                  if (typeof e.duration === 'number' && e.duration > 0) {
                    setDurationMs(e.duration);
                  }
                  if (typeof e.currentMetering === 'number') {
                    setPeaks(prev => [...prev.slice(-299), (e.currentMetering + 160) / 160]);
                  }
                  if (e.currentPosition >= e.duration) stopPlaying();
                  return;
                });
              
                await audioPlayer.seekToPlayer(item.timestamp_ms);
                setPlaybackSecs(item.timestamp_ms);
                setHighlightedTagId(item.id);
                setIsPlaying(true);
              }}
              
              onLongPress={() => {
                Alert.alert('Tag Options', `${item.label}`, [
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
                      } else {
                        const newLabel = prompt(`Edit tag "${item.label}"`);
                        if (newLabel && newLabel.trim()) {
                          updateTagLabel(item.id, newLabel.trim()).then(loadTags).catch(console.error);
                        }
                      }
                    }
                  },
                  {
                    text: 'Delete',
                    style: 'destructive',
                    onPress: async () => {
                      const db = await getDb();
                      await db.executeSql('DELETE FROM tags WHERE id = ?', [item.id]);
                      await loadTags();
                      setHighlightedTagId(null);
                    }
                  },
                  { text: 'Cancel', style: 'cancel' }
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
});
