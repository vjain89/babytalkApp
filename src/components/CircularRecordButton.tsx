import React, { useState, useEffect } from 'react';
import { TouchableOpacity, StyleSheet, View } from 'react-native';

interface Props {
    recording: boolean;
    isPaused: boolean;
    onStart: () => void;
    onPause: () => void;
    onResume: () => void;
    onStop: () => void;
}

export default function CircularRecordButton({ 
    recording,
    isPaused,
    onStart,
    onPause,
    onResume,
    onStop,    
}: Props) {
    const [status, setStatus] = useState<'idle' | 'recording' | 'paused'>('idle');

    useEffect(() => {
        if (!recording) {
            setStatus('idle');
        } else if (isPaused) {
            setStatus('paused');
        } else {
            setStatus('recording');
        }
    }, [recording, isPaused]);

    const handlePress = () => {
        console.log('🔘 Pressed:', status);
        if (status === 'idle') {
            onStart();
        } else if (status === 'recording') {
            onPause();
        } else if (status === 'paused') {
            onResume();
        }
    };

    return (
        <TouchableOpacity onPress={handlePress} style={styles.wrapper}>
            <View 
                style={[
                    styles.circle, 
                    { backgroundColor: status === 'recording' ? 'red' : '#aaa' }
                ]} 
            />
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
    wrapper: {
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
    },
    circle: {
        width: 72,
        height: 72,
        borderRadius: 36,
    },
});