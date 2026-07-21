import React, { useEffect, useRef } from 'react';
import { AppState, type AppStateStatus } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import RecordScreen from './src/screens/RecordScreen';
import RecordingListScreen from './src/screens/RecordingListScreen';
import PlaybackScreen from './src/screens/PlaybackScreen';
import { initDb } from './src/db';
import { ensureBackupDirs, importInboxAnnotations } from './src/export/backup';
import type { RootStackParamList } from './src/navigation/types';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  const importingRef = useRef(false);

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

  useEffect(() => {
    const autoImport = async () => {
      if (importingRef.current) return;
      importingRef.current = true;
      try {
        await ensureBackupDirs();
        const summary = await importInboxAnnotations();
        const changed = summary.inserted + summary.updated;
        if (changed > 0) {
          console.log(
            `✅ Auto-imported annotations: +${summary.inserted} new, ${summary.updated} updated`,
          );
        }
      } catch (err) {
        console.warn('Auto-import annotations skipped:', err);
      } finally {
        importingRef.current = false;
      }
    };

    void autoImport();
    const onChange = (state: AppStateStatus) => {
      if (state === 'active') void autoImport();
    };
    const sub = AppState.addEventListener('change', onChange);
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
