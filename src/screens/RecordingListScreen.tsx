import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    SectionList,
    StyleSheet,
    TouchableOpacity,
    TextInput,
    Button,
    Alert,
    Platform,
} from 'react-native';
import { format } from 'date-fns';
import { getAllRecordings, updateSessionName, getRecordingsByTagLabel, getTagsForRecording } from '../db';
import { useNavigation } from '@react-navigation/native';
import RNFS from 'react-native-fs';
import Share from 'react-native-share';

type Recording = {
    id: number;
    filename: string;
    session_name: string;
    created_at: number;
    duration_ms: number;
};

type Section = {
    title: string;
    data: Recording[];
};

const groupRecordingsByDate = (recordings: Recording[]): Section[] => {
    const map = new Map<string, Recording[]>();

    recordings.forEach((r) => {
        const dateKey = format(new Date(r.created_at), 'EEEE, MMMM d, yyyy');
        if (!map.has(dateKey)) {
            map.set(dateKey, []);
        }
        map.get(dateKey)!.push(r);
    });

    return Array.from(map.entries())
        .sort((a, b) => new Date(b[0]).getTime() - new Date(a[0]).getTime())
        .map(([title, data]) => ({ title, data }));
};

export default function RecordingListScreen() {
    const [recordings, setRecordings] = useState<Recording[]>([]);
    const [filteredRecordings, setFilteredRecordings] = useState<Recording[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [sortOption, setSortOption] = useState<'date_desc' | 'date_asc' | 'duration_desc' | 'duration_asc'>('date_desc');
    const [tagQuery, setTagQuery] = useState('');

    const navigation = useNavigation();

    useEffect(() => {
        fetchData();
    }, [tagQuery]);

    useEffect(() => {
        applyFilters();
    }, [recordings, searchQuery, sortOption]);

    const fetchData = async () => {
        try {
            console.log('🔍 Fetching with tagQuery:', tagQuery);
    
            let results: Recording[] = [];
    
            if (tagQuery.trim()) {
                results = await getRecordingsByTagLabel(tagQuery.trim());
                console.log('🏷️ Tag matches:', results.length);
            } else {
                results = await getAllRecordings();
                console.log('📄 All session matches:', results.length);
            }
    
            setRecordings(results);
        } catch (err) {
            console.error('❌ Failed to fetch recordings:', err);
        }
    };

    const applyFilters = () => {
        let filtered = recordings.filter((r) =>
            (r.session_name ?? '').toLowerCase().includes(searchQuery.toLowerCase())
        );

        console.log('🔍 Applying filters to:', recordings.length);

        switch (sortOption) {
            case 'date_asc':
                filtered = filtered.sort((a, b) => a.created_at - b.created_at);
                break;
            case 'date_desc':
                filtered = filtered.sort((a, b) => b.created_at - a.created_at);
                break;
            case 'duration_asc':
                filtered = filtered.sort((a, b) => a.duration_ms - b.duration_ms);
                break;
            case 'duration_desc':
                filtered = filtered.sort((a, b) => b.duration_ms - a.duration_ms);
                break;
        }

        setFilteredRecordings(filtered);
    };

    const [exportingAll, setExportingAll] = useState(false);

    const resolveAudioPath = async (filename: string): Promise<string | null> => {
        const candidates = [
            filename,
            `${RNFS.CachesDirectoryPath}/${filename}`,
            `${RNFS.DocumentDirectoryPath}/${filename}`,
        ];
        for (const c of candidates) {
            const p = c.replace(/^file:\/\//, '');
            if (await RNFS.exists(p)) return c.startsWith('file://') ? c : `file://${p}`;
        }
        return null;
    };

    const handleExportAll = async () => {
        if (filteredRecordings.length === 0) return;
        setExportingAll(true);
        try {
            const exportData: { exportedAt: string; recordings: Array<{ filename: string; recordingId: number; session_name: string | null; duration_ms: number; created_at: number; tags: Array<{ id: number; label: string; timestamp_ms: number }> }> } = {
                exportedAt: new Date().toISOString(),
                recordings: [],
            };
            const urls: string[] = [];

            for (const r of filteredRecordings) {
                const tags = await getTagsForRecording(r.id);
                exportData.recordings.push({
                    filename: r.filename,
                    recordingId: r.id,
                    session_name: r.session_name ?? null,
                    duration_ms: r.duration_ms,
                    created_at: r.created_at,
                    tags,
                });
                const audioUrl = await resolveAudioPath(r.filename);
                if (audioUrl) urls.push(audioUrl);
            }

            const jsonPath = `${RNFS.CachesDirectoryPath}/babytalk_export_all_${Date.now()}.json`;
            await RNFS.writeFile(jsonPath, JSON.stringify(exportData, null, 2), 'utf8');
            urls.push(`file://${jsonPath}`);

            await Share.open({
                title: 'Export All Recordings & Tags',
                message: `Exporting ${exportData.recordings.length} recording(s) and tags`,
                urls,
                failOnCancel: false,
            });
        } catch (err) {
            console.error('❌ Export all failed:', err);
            if (Platform.OS === 'ios') {
                Alert.alert('Export failed', String(err));
            }
        } finally {
            setExportingAll(false);
        }
    };

    const renderItem = ({ item }: { item: Recording }) => (
        <TouchableOpacity
            style={styles.item}
            onPress={() =>
                navigation.navigate('Playback', {
                    recordingId: item.id,
                    filePath: item.filename,
                    filename: item.session_name ?? item.filename,
                })
            }
            onLongPress={() => {
                if (Platform.OS === 'ios') {
                    Alert.prompt(
                        'Rename Session',
                        'Enter a new session name:',
                        async (newName) => {
                            if (newName?.trim()) {
                                await updateSessionName(item.id, newName.trim());
                                fetchData(); // reload list
                            }
                        }
                    );
                } else {
                    const newName = prompt('Enter a new session name:');
                    if (newName?.trim()) {
                        updateSessionName(item.id, newName.trim()).then(fetchData);
                    }
                }
            }}
        >
            <Text style={styles.session}>{item.session_name || 'Untitled session'}</Text>
            <Text style={styles.details}>
                {(item.duration_ms / 1000).toFixed(1)}s · {new Date(item.created_at).toLocaleString()}
            </Text>
        </TouchableOpacity>
    );

    return (
        <View style={styles.container}>
            <TextInput
                placeholder="Search session names..."
                value={searchQuery}
                onChangeText={setSearchQuery}
                style={styles.searchInput}
            />

            <TextInput
                placeholder="Search tags..."
                value={tagQuery}
                onChangeText={setTagQuery}
                style={styles.searchInput}
            />

            <Button
                title={exportingAll ? 'Exporting…' : '📤 Export All Recordings + Tags'}
                onPress={handleExportAll}
                disabled={filteredRecordings.length === 0 || exportingAll}
            />

            {filteredRecordings.length === 0 ? (
                <Text style={styles.empty}>No recordings match.</Text>
            ) : (
                <SectionList
                    sections={groupRecordingsByDate(filteredRecordings)}
                    keyExtractor={(item) => item.id.toString()}
                    renderItem={renderItem}
                    renderSectionHeader={({ section: { title } }) => (
                        <Text style={styles.sectionHeader}>{title}</Text>
                    )}
                />
            )}
        </View>
    );
}


const styles = StyleSheet.create({
    container: { flex: 1, padding: 20 },
    searchInput: {
        borderWidth: 1,
        borderColor: '#ccc',
        marginBottom: 12,
        padding: 8,
        borderRadius: 6,
    },
    sortButtons: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 12,
        gap: 6,
    },
    item: {
        padding: 16,
        borderBottomWidth: 1,
        borderColor: '#eee',
    },
    session: { fontSize: 18 },
    details: { color: '#666', marginTop: 4 },
    empty: { textAlign: 'center', color: '#888', marginTop: 40 },
    sectionHeader: {
        fontSize: 16,
        fontWeight: 'bold',
        backgroundColor: '#f0f0f0',
        paddingVertical: 4,
        paddingHorizontal: 10,
        marginTop: 16,
    },
});