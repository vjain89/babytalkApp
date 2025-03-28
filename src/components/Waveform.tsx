import React from 'react';
import { View, StyleSheet, TouchableOpacity } from 'react-native';

type Props = {
    peaks: number[];
    durationMs: number;
    progressMs: number;
    tagTimestamps?: number[];
    onSeek?: (ms: number) => void;
    highlightedTagId?: number;
};

export default function Waveform({
    peaks,
    durationMs,
    progressMs,
    tagTimestamps = [],
    onSeek,
}: Props) {
    const progressRatio = durationMs > 0 ? progressMs / durationMs : 0;

    return (
        <View style={styles.waveformWrapper}>
            {/* Waveform Bars */}
            <View style={styles.waveformContainer}>
                {peaks.map((height, index) => {
                    const barTimestamp = durationMs > 0
                        ? (index / peaks.length) * durationMs
                        : 0;

                    const isTagTick = tagTimestamps.some(
                        (t) => Math.abs(t - barTimestamp) < durationMs / peaks.length
                    );

                    return (
                        <TouchableOpacity
                            key={index}
                            onPress={() => onSeek?.(Math.floor(barTimestamp))}
                        >
                            <View
                                style={[
                                    styles.waveformBar,
                                    {
                                        height: `${height * 100}%`,
                                        backgroundColor: isTagTick ? 'blue' : '#ccc',
                                    },
                                ]}
                            />
                        </TouchableOpacity>
                    );
                })}
            </View>
            {/* Playback progress overlay */}
            <View
                style={[
                    styles.playbackProgress,
                    { left: `${progressRatio * 100}%` },
                ]}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    waveformWrapper: {
        position: 'relative',
        height: 60,
        marginBottom: 20,
    },
    waveformContainer: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        height: '100%',
        borderColor: '#ddd',
        borderWidth: 1,
        borderRadius: 4,
        overflow: 'hidden',
        paddingBottom: 1,
        backgroundColor: '#f8f8f8',
    },
    waveformBar: {
        width: 2,
        marginRight: 1,
    },
    playbackProgress: {
        position: 'absolute',
        top: 0,
        bottom: 0,
        width: 2,
        backgroundColor: 'red',
        zIndex: 10,
    },
});