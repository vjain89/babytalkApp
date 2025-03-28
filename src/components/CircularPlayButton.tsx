import React from 'react';
import { TouchableOpacity, View, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

type Props = {
    isPlaying: boolean;
    onPress: () => void;
}

export default function CircularPlayButton({ isPlaying, onPress }: Props) {
    return (
        <TouchableOpacity onPress={onPress} style={styles.wrapper}>
            <View style={styles.circle}>
                {isPlaying ? (
                    <View style={styles.pauseIcon}>
                        <View style={styles.pauseBar} />
                        <View style={styles.pauseBar} />
                    </View>
                ) : (
                    <View style={styles.playTriangle} />
                )}
            </View>
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
        backgroundColor: '#ccc',
        alignItems: 'center',
        justifyContent: 'center',
    },
    playTriangle: {
        width: 0,
        height: 0,
        marginLeft: 4,
        borderTopWidth: 10,
        borderBottomWidth: 10,
        borderLeftWidth: 16,
        borderTopColor: 'transparent',
        borderBottomColor: 'transparent',
        borderLeftColor: 'black',
    },
    pauseIcon: {
        flexDirection: 'row',
        gap: 4,
    },
    pauseBar: {
        width: 4,
        height: 20,
        backgroundColor: 'black',
    }
});