import React, { useEffect, useRef } from 'react';
import { AppState, Linking, type AppStateStatus } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import RecordScreen from './src/screens/RecordScreen';
import RecordingListScreen from './src/screens/RecordingListScreen';
import PlaybackScreen from './src/screens/PlaybackScreen';
import { initDb } from './src/db';
import { ensureBackupDirs, ensureRecordingKitsForSync } from './src/export/backup';
import {
  importSharedAudioUrl,
  runAutoImportAnnotations,
  runAutoImportAudio,
} from './src/export/autoImport';
import type { RootStackParamList } from './src/navigation/types';

const Stack = createNativeStackNavigator<RootStackParamList>();

function isImportableUrl(url: string): boolean {
  const lower = url.toLowerCase();
  if (lower.startsWith('babytalk:')) return true;
  return (
    lower.startsWith('file:') ||
    /\.(m4a|mp4|aac|wav|caf|aiff|aif|mp3|flac)(\?|$)/i.test(lower)
  );
}

export default function App() {
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const kitSyncBusy = useRef(false);

  useEffect(() => {
    const initialize = async () => {
      try {
        await initDb();
        await ensureBackupDirs();
        console.log('✅ Database initialized from App.tsx');
      } catch (err) {
        console.error('❌ Failed to initialize app:', err);
      }
    };

    initialize();
  }, []);

  // Export missing recordings into Documents/Backups/sync for Mac USB pull.
  useEffect(() => {
    const exportKits = async () => {
      if (kitSyncBusy.current) return;
      kitSyncBusy.current = true;
      try {
        const result = await ensureRecordingKitsForSync();
        if (result.created.length > 0) {
          console.log(
            `✅ Prepared ${result.created.length} kit(s) for Mac sync:`,
            result.created.join(', '),
          );
        }
      } catch (err) {
        console.warn('Sync kit export skipped:', err);
      } finally {
        kitSyncBusy.current = false;
      }
    };

    const tick = () => {
      void runAutoImportAnnotations({ alertOnChange: true });
      void (async () => {
        const n = await runAutoImportAudio({ alertOnChange: true });
        // New imports → rebuild kits so the next Mac Sync can pull them.
        if (n > 0) await exportKits();
      })();
    };

    const startPoll = () => {
      if (pollRef.current) return;
      pollRef.current = setInterval(tick, 4000);
    };
    const stopPoll = () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };

    tick();
    void exportKits();
    startPoll();

    const onChange = (state: AppStateStatus) => {
      if (state === 'active') {
        tick();
        void exportKits();
        startPoll();
      } else {
        stopPoll();
      }
    };
    const sub = AppState.addEventListener('change', onChange);
    return () => {
      stopPoll();
      sub.remove();
    };
  }, []);

  // Voice Memos / Files → Share → Copy to BabyTalk / babytalk://shared-import
  useEffect(() => {
    const handleUrl = async (url: string | null) => {
      if (!url || !isImportableUrl(url)) return;
      try {
        await importSharedAudioUrl(url);
      } catch (err) {
        console.warn('Shared audio import failed:', err);
      }
    };

    void Linking.getInitialURL().then(handleUrl);
    const sub = Linking.addEventListener('url', ({ url }) => {
      void handleUrl(url);
    });
    return () => sub.remove();
  }, []);

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Record">
        <Stack.Screen name="Record" component={RecordScreen} options={{ title: 'New Recording' }} />
        <Stack.Screen name="RecordingList" component={RecordingListScreen} options={{ title: 'All Recordings' }} />
        <Stack.Screen name="Playback" component={PlaybackScreen} options={{ title: 'Playback' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
