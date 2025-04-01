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
    const visibleWindowMs = 30_000;
    const totalBars = peaks.length;
    const barDurationMs = durationMs / totalBars;
    const barsPerWindow = Math.floor(visibleWindowMs / barDurationMs);

    const currentBarIndex = Math.floor(progressMs / barDurationMs);
    const startBarIndex = Math.max(0, currentBarIndex - barsPerWindow);
    const visiblePeaks = peaks.slice(startBarIndex, currentBarIndex + 1);

    const progressRatioInWindow =
        (progressMs - startBarIndex * barDurationMs) / (visibleWindowMs || 1);

    const scrubLeft = `${Math.min(progressRatioInWindow * 100, 100)}%`;

    return (
        <View style={styles.waveformWrapper}>
            {/* Waveform Bars */}
            <View style={styles.waveformContainer}>
                {visiblePeaks.map((height, i) => {
                    const globalIndex = startBarIndex + i;
                    const barTimestamp = globalIndex * barDurationMs;

                    const isTagTick = tagTimestamps.some(
                        (t) => Math.abs(t - barTimestamp) < barDurationMs
                    );

                    return (
                        <TouchableOpacity
                            key={globalIndex}
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
            <View style={[styles.playbackProgress, { left: scrubLeft }]} />
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
