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
import { getAllRecordings, updateSessionName, getRecordingsByTagLabel } from '../db';
import { useNavigation } from '@react-navigation/native';
import { importInboxAnnotations, prepareBackup } from '../export/backup';
import { buildExportBatch } from '../export/sessionKit';
import {
    importAudioFromInbox,
    pickAndImportAudioFiles,
} from '../audio/importAudio';
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

    const handleExportAll = async () => {
        if (filteredRecordings.length === 0) return;
        setExportingAll(true);
        try {
            const parent = `${RNFS.CachesDirectoryPath}/export_all_${Date.now()}`;
            await buildExportBatch(
                filteredRecordings.map((r) => r.id),
                parent,
            );
            Alert.alert(
                'Export ready',
                `Built ${filteredRecordings.length} session kit(s) as a folder. Prepare USB Backup writes the same kits under Documents/Backups for Finder.`,
            );
            // Also offer share of the batch manifest as a pointer.
            await Share.open({
                title: 'Export All Session Kits',
                message: `Exported ${filteredRecordings.length} session kit(s)`,
                urls: [`file://${parent}/export_manifest.json`],
                failOnCancel: false,
            });
        } catch (err) {
            console.error('❌ Export all failed:', err);
            Alert.alert('Export failed', String(err));
        } finally {
            setExportingAll(false);
        }
    };

    const handlePrepareBackup = async () => {
        if (filteredRecordings.length === 0) return;
        setExportingAll(true);
        try {
            const dest = await prepareBackup(filteredRecordings.map((r) => r.id));
            Alert.alert(
                'Backup ready',
                `Copied ${filteredRecordings.length} kit(s) to Documents/Backups.\nPlug into your Mac → Finder → your iPhone → babytalkApp → Backups.\n\n${dest.replace(/.*\/Documents\//, 'Documents/')}`,
            );
        } catch (err) {
            console.error('❌ Backup failed:', err);
            Alert.alert('Backup failed', String(err));
        } finally {
            setExportingAll(false);
        }
    };

    const handleImportAnnotations = async () => {
        setExportingAll(true);
        try {
            const summary = await importInboxAnnotations();
            Alert.alert(
                'Import complete',
                `Scanned ${summary.scanned}: +${summary.inserted} new, ${summary.updated} updated, ${summary.skipped} skipped, ${summary.unmatched} unmatched.` +
                    (summary.errors.length ? `\n\n${summary.errors.slice(0, 3).join('\n')}` : ''),
            );
            fetchData();
        } catch (err) {
            console.error('❌ Import failed:', err);
            Alert.alert('Import failed', String(err));
        } finally {
            setExportingAll(false);
        }
    };

    const handleImportVoiceMemos = async () => {
        setExportingAll(true);
        try {
            const result = await pickAndImportAudioFiles(true);
            if (result.cancelled) return;
            const errTail = result.errors.length
                ? `\n\n${result.errors.slice(0, 3).join('\n')}`
                : '';
            Alert.alert(
                'Audio import',
                result.imported.length
                    ? `Imported ${result.imported.length} recording(s).${errTail}`
                    : `No files imported.${errTail || '\nTip: In the picker, open Browse → On My iPhone → Voice Memos.'}`,
            );
            fetchData();
        } catch (err) {
            console.error('❌ Audio import failed:', err);
            Alert.alert('Audio import failed', String(err));
        } finally {
            setExportingAll(false);
        }
    };

    const handleImportInboxAudio = async () => {
        setExportingAll(true);
        try {
            const result = await importAudioFromInbox();
            const errTail = result.errors.length
                ? `\n\n${result.errors.slice(0, 3).join('\n')}`
                : '';
            Alert.alert(
                'Inbox audio',
                result.imported.length
                    ? `Imported ${result.imported.length} file(s) from Documents/Import (and system Inbox if present).${errTail}`
                    : `No audio files found in Documents/Import or Inbox.${errTail}\n\nFrom Voice Memos: use Import Voice Memos / Audio, or Share → Open in babytalkApp.`,
            );
            fetchData();
        } catch (err) {
            console.error('❌ Inbox audio import failed:', err);
            Alert.alert('Inbox import failed', String(err));
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
                title={exportingAll ? 'Working…' : '🎙 Import Voice Memos / Audio'}
                onPress={handleImportVoiceMemos}
                disabled={exportingAll}
            />
            <View style={{ height: 8 }} />
            <Button
                title={exportingAll ? 'Working…' : '📂 Import Inbox Audio'}
                onPress={handleImportInboxAudio}
                disabled={exportingAll}
            />
            <View style={{ height: 8 }} />
            <Button
                title={exportingAll ? 'Working…' : '📤 Export All Session Kits'}
                onPress={handleExportAll}
                disabled={filteredRecordings.length === 0 || exportingAll}
            />
            <View style={{ height: 8 }} />
            <Button
                title={exportingAll ? 'Working…' : '💾 Prepare USB Backup'}
                onPress={handlePrepareBackup}
                disabled={filteredRecordings.length === 0 || exportingAll}
            />
            <View style={{ height: 8 }} />
            <Button
                title={exportingAll ? 'Working…' : '📥 Import Inbox Annotations'}
                onPress={handleImportAnnotations}
                disabled={exportingAll}
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