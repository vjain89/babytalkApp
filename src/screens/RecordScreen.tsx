import React, { useRef, useState } from 'react';
import {
  View,
  Text,
  Button,
  Alert,
  Platform,
  Modal,
  TouchableOpacity,
  ScrollView,
  TextInput,
  StyleSheet,
} from 'react-native';
import AudioRecorderPlayer from '../../react-native-audio-recorder-player';
import { useNavigation } from '@react-navigation/native';
import { addRecording, addTag, getTodayRecordingCount } from '../db';
import CircularRecordButton from '../components/CircularRecordButton';
import RecordingLayout from '../components/RecordingLayout';
import Waveform from '../components/Waveform';

const audioRecorderPlayer = new AudioRecorderPlayer();
const TAG_SUGGESTIONS = ['hungry', 'tired', 'frustrated', 'playful', 'bored'];

const BAR_MS = 50;
const WINDOW_MS = 30_000;

export default function RecordScreen() {
  const [recording, setRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordMs, setRecordMs] = useState(0);
  const [filePath, setFilePath] = useState<string | null>(null);

  const [liveTags, setLiveTags] = useState<{ timestampMs: number; label: string }[]>([]);
  const [tagModalVisible, setTagModalVisible] = useState(false);
  const [customTag, setCustomTag] = useState('');

  // ✅ Fixed-grid waveform for entire session
  const waveformAllRef = useRef<number[]>([]);

  const navigation = useNavigation();

  const startRecording = async () => {
    try {
      waveformAllRef.current = [];
      setLiveTags([]);
      setRecordMs(0);

      const result = await audioRecorderPlayer.startRecorder(
        Platform.select({ ios: 'recording.m4a', android: undefined }),
        undefined,
        true
      );
      setFilePath(result);

      audioRecorderPlayer.addRecordBackListener((e) => {
        if (typeof e?.currentPosition === 'number') {
          setRecordMs(e.currentPosition);

          // Fixed-grid index for this time
          const idx = Math.max(0, Math.round(e.currentPosition / BAR_MS));

          // Normalize metering
          if (typeof e.currentMetering === 'number') {
            const minDb = -60;
            const maxDb = 0;
            const clamped = Math.max(minDb, Math.min(maxDb, e.currentMetering));
            let normalized = (clamped - minDb) / (maxDb - minDb); // 0..1
            normalized = Math.pow(normalized, 1.6);

            const arr = waveformAllRef.current;

            // Fill gaps with 0 so index==time bin
            while (arr.length < idx) arr.push(0);

            // Set the value at idx (use max to avoid losing peaks if listener fires multiple times per bin)
            const prev = arr[idx] ?? 0;
            arr[idx] = Math.max(prev, normalized);

            waveformAllRef.current = arr;
          }
        }
        return;
      });

      setRecording(true);
      setIsPaused(false);
    } catch (err) {
      console.error('❌ Failed to start recording:', err);
    }
  };

  const pauseRecording = async () => {
    try {
      await audioRecorderPlayer.pauseRecorder();
      setIsPaused(true);
    } catch (err) {
      console.error('❌ Failed to pause:', err);
    }
  };

  const resumeRecording = async () => {
    try {
      await audioRecorderPlayer.resumeRecorder();
      setIsPaused(false);
    } catch (err) {
      console.error('❌ Failed to resume:', err);
    }
  };

  const stopRecording = async () => {
    try {
      const result = await audioRecorderPlayer.stopRecorder();
      audioRecorderPlayer.removeRecordBackListener();
      setFilePath(result);
      setRecording(false);
      setIsPaused(false);
    } catch (err) {
      console.error('❌ Failed to stop recording:', err);
    }
  };

  const handleTagNow = () => {
    setCustomTag('');
    setTagModalVisible(true);
  };

  const saveTag = (label: string) => {
    setLiveTags((prev) => [...prev, { timestampMs: Math.floor(recordMs), label }]);
    setTagModalVisible(false);
  };

  const handleSave = async () => {
    if (recording || isPaused) await stopRecording();
    if (!filePath) return;

    const promptAndSave = async (enteredName: string | null) => {
      const durationMs = recordMs;
      const filename = filePath.split('/').pop() || `recording_${Date.now()}.m4a`;

      let finalSessionName = enteredName?.trim();
      if (!finalSessionName) {
        const count = await getTodayRecordingCount();
        finalSessionName = `Recording-${count + 1}`;
      }

      try {
        const recordingId = await addRecording({
          filename,
          sessionName: finalSessionName,
          durationMs,
          waveformData: waveformAllRef.current, // ✅ fixed-grid waveform
        });

        for (const tag of liveTags) {
          await addTag({ recordingId, timestampMs: tag.timestampMs, label: tag.label });
        }
      } catch (err) {
        console.error('❌ DB insert failed:', err);
      }

      // reset
      waveformAllRef.current = [];
      setRecordMs(0);
      setFilePath(null);
      setLiveTags([]);
      setRecording(false);
      setIsPaused(false);

      navigation.navigate('RecordingList' as never);
    };

    if (Platform.OS === 'ios') {
      Alert.prompt('Save Recording', 'Enter session name (optional):', promptAndSave);
    } else {
      const name = prompt('Enter session name (optional):');
      await promptAndSave(name);
    }
  };

  return (
    <>
      <RecordingLayout
        title="New Recording"
        durationLabel={`Duration: ${(recordMs / 1000).toFixed(1)}s`}
        waveform={
          <Waveform
            peaks={waveformAllRef.current}
            barDurationMs={BAR_MS}
            progressMs={recordMs}
            windowMs={WINDOW_MS}
            mode="rolling"
            cursorMode="pinned"
            showCursor={true}
            minBarPx={2}
            tagTimestamps={liveTags.map((t) => Number(t.timestampMs))}
            tagWidthMs={500}
          />
        }
        controls={
          <>
            <View style={{ alignItems: 'center' }}>
              <CircularRecordButton
                recording={recording}
                isPaused={isPaused}
                onStart={startRecording}
                onPause={pauseRecording}
                onResume={resumeRecording}
                onStop={stopRecording}
              />
            </View>

            <View style={{ marginTop: 20 }}>
              <Button title="🏷️ Tag Now" onPress={handleTagNow} />
              <View style={{ marginVertical: 6 }} />
              <Button title="💾 Save Recording" onPress={handleSave} disabled={!filePath} />
              <View style={{ marginTop: 20 }} />
              <Button
                title="📄 View All Recordings"
                onPress={() => navigation.navigate('RecordingList' as never)}
              />
            </View>
          </>
        }
      />

      <Modal visible={tagModalVisible} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Tag This Moment</Text>
            <ScrollView contentContainerStyle={styles.suggestionsContainer}>
              {TAG_SUGGESTIONS.map((label) => (
                <TouchableOpacity
                  key={label}
                  style={styles.suggestionButton}
                  onPress={() => saveTag(label)}
                >
                  <Text style={styles.suggestionText}>{label}</Text>
                </TouchableOpacity>
              ))}
              <TextInput
                style={styles.customInput}
                placeholder="Custom tag..."
                value={customTag}
                onChangeText={setCustomTag}
                onSubmitEditing={() => {
                  if (customTag.trim()) saveTag(customTag.trim());
                }}
              />
            </ScrollView>
            <Button title="Cancel" onPress={() => setTagModalVisible(false)} />
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 10,
    width: '80%',
    alignItems: 'stretch',
  },
  modalTitle: { fontSize: 18, fontWeight: 'bold', marginBottom: 10, textAlign: 'center' },
  suggestionsContainer: { gap: 10, paddingBottom: 12 },
  suggestionButton: {
    padding: 10,
    backgroundColor: '#eee',
    borderRadius: 6,
    alignItems: 'center',
  },
  suggestionText: { fontSize: 16 },
  customInput: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    borderRadius: 6,
  },
});
