import React from 'react';
import { View, StyleSheet, Text } from 'react-native';

type Props = {
    title: string;
    durationLabel: string;
    waveform?: React.ReactNode;
    controls: React.ReactNode;
    children?: React.ReactNode;
};

export default function RecordingLayout({
    title,
    durationLabel,
    waveform,
    controls,
    children,
}: Props) {
    return (
        <View style={styles.container}>
            <Text style={styles.title}>{title}</Text>
            <Text style={styles.timer}>{durationLabel}</Text>

            {waveform ? (
                <View style={styles.waveform}>{waveform}</View>
            ) : (
                <View style={{ height: 60, marginBottom: 16 }} />
            )}

            <View style={styles.controls}>{controls}</View>

            {children}
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, padding: 20, justifyContent: 'flex-start' },
    title: { fontSize: 18, textAlign: 'center', marginBottom: 12 },
    timer: { fontSize: 16, textAlign: 'center', marginBottom: 16 },
    waveform: { marginBottom: 16 },
    controls: {
        width: '100%',
        alignItems: 'center',
        gap: 12,
        marginBottom: 16,
    },
});