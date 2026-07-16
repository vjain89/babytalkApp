import React, { useMemo, useRef, useState } from 'react';
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
import {
  BAR_MS,
  CAPTURE_DB_MIN,
  RECORD_WINDOW_MS,
  SUBSCRIPTION_SEC,
  TAG_MARKER_MS,
} from '../waveform/config';
import { RECORD_DB_RANGE } from '../waveform/scale';
import { densifyPeaks, serializeWaveform, upsertSample } from '../waveform/storage';
import { emptyWaveformPayload, type WaveformSample } from '../waveform/types';

const audioRecorderPlayer = new AudioRecorderPlayer();
const TAG_SUGGESTIONS = ['hungry', 'tired', 'frustrated', 'playful', 'bored'];

export default function RecordScreen() {
  const [recording, setRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordMs, setRecordMs] = useState(0);
  const [filePath, setFilePath] = useState<string | null>(null);

  const [liveTags, setLiveTags] = useState<{ timestampMs: number; label: string }[]>([]);
  const [tagModalVisible, setTagModalVisible] = useState(false);
  const [pendingTagMs, setPendingTagMs] = useState<number | null>(null);
  const [customTag, setCustomTag] = useState('');

  // Sparse capture list (mutated in place; displayPeaks is React state for re-renders)
  const samplesRef = useRef<WaveformSample[]>([]);
  const [displayPeaks, setDisplayPeaks] = useState<number[]>([]);

  const navigation = useNavigation();

  const startRecording = async () => {
    try {
      samplesRef.current = [];
      setDisplayPeaks([]);
      setLiveTags([]);
      setRecordMs(0);

      await audioRecorderPlayer.setSubscriptionDuration(SUBSCRIPTION_SEC);

      const result = await audioRecorderPlayer.startRecorder(
        Platform.select({ ios: 'recording.m4a', android: undefined }),
        undefined,
        true
      );
      setFilePath(result);

      audioRecorderPlayer.addRecordBackListener((e) => {
        if (typeof e?.currentPosition !== 'number') return;

        const tMs = e.currentPosition;
        setRecordMs(tMs);

        const avgDb =
          typeof e.currentMetering === 'number' ? e.currentMetering : CAPTURE_DB_MIN;
        const peakDb =
          typeof e.currentPeakMetering === 'number'
            ? e.currentPeakMetering
            : avgDb;

        upsertSample(samplesRef.current, { tMs, avgDb, peakDb });

        // Live tail: window ends at "now" so new bars appear at the right edge.
        // Negative start → left-padded silence until the recording fills 3s.
        setDisplayPeaks(
          densifyPeaks(samplesRef.current, tMs - RECORD_WINDOW_MS, tMs, BAR_MS)
        );
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
    // Stamp the recording clock immediately so label picking does not delay the mark.
    setPendingTagMs(Math.floor(recordMs));
    setCustomTag('');
    setTagModalVisible(true);
  };

  const saveTag = (label: string) => {
    const timestampMs = pendingTagMs ?? Math.floor(recordMs);
    setLiveTags((prev) => [...prev, { timestampMs, label }]);
    setPendingTagMs(null);
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
        const payload = emptyWaveformPayload();
        payload.samples = samplesRef.current.slice();
        const recordingId = await addRecording({
          filename,
          sessionName: finalSessionName,
          durationMs,
          waveformData: serializeWaveform(payload),
        });

        for (const tag of liveTags) {
          await addTag({ recordingId, timestampMs: tag.timestampMs, label: tag.label });
        }
      } catch (err) {
        console.error('❌ DB insert failed:', err);
      }

      samplesRef.current = [];
      setDisplayPeaks([]);
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

  const tagTimestampsInWindow = useMemo(() => {
    const start = Math.max(0, recordMs - RECORD_WINDOW_MS);
    return liveTags
      .map((t) => Number(t.timestampMs))
      .filter((ts) => ts >= start && ts <= recordMs);
  }, [liveTags, recordMs]);

  return (
    <>
      <RecordingLayout
        title="New Recording"
        durationLabel={`Duration: ${(recordMs / 1000).toFixed(1)}s`}
        waveform={
          <Waveform
            peaksDb={displayPeaks}
            barDurationMs={BAR_MS}
            progressMs={recordMs}
            windowMs={RECORD_WINDOW_MS}
            mode="rolling"
            cursorMode="pinned"
            showCursor={true}
            dbRange={RECORD_DB_RANGE}
            minBarPx={1}
            tagTimestamps={tagTimestampsInWindow}
            tagWidthMs={TAG_MARKER_MS}
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
            <Button
              title="Cancel"
              onPress={() => {
                setPendingTagMs(null);
                setTagModalVisible(false);
              }}
            />
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
