import React, { useEffect, useState } from 'react';
import { SafeAreaView, View, Button, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import RecordScreen from './src/screens/RecordScreen';
import RecordingListScreen from './src/screens/RecordingListScreen';
import PlaybackScreen from './src/screens/PlaybackScreen';
import { initDb } from './src/db';
import type { RootStackParamList } from './src/navigation/types';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  useEffect(() => {
    const initialize = async () => {
      try {
        await initDb();
        console.log('✅ Database initialized from App.tsx');
      } catch (err) {
        console.error('❌ Failed to initialize DB:', err);
      }
    };

    initialize();
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
};