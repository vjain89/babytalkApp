# BabyTalk App - Project Architecture & Technical Summary

## Overview

BabyTalk is a React Native mobile application designed for recording and analyzing baby sounds. The app allows users to record audio sessions, tag specific moments during recordings, visualize audio waveforms, and manage a library of recordings with associated metadata.

## Technology Stack

### Core Framework
- **React Native 0.78.1**: Cross-platform mobile framework
- **React 19.0.0**: UI library
- **TypeScript 5.0.4**: Type-safe JavaScript
- **Hermes Engine**: JavaScript engine optimized for React Native

### Key Dependencies
- **react-native-audio-recorder-player**: Local fork for audio recording/playback
- **react-native-sqlite-storage**: Local SQLite database for metadata storage
- **@react-navigation/native**: Navigation framework
- **react-native-fs**: File system operations
- **react-native-share**: Sharing functionality

## Architecture Decisions

### React Native vs Native Swift/Kotlin

**Decision: React Native with TypeScript**

**Rationale:**
- **Cross-platform**: Single codebase for iOS and Android
- **Rapid development**: Faster iteration with hot reload
- **Type safety**: TypeScript catches errors at compile time
- **Ecosystem**: Rich library ecosystem for React Native
- **Maintainability**: Easier to maintain one codebase

**Trade-offs:**
- Performance: Slightly slower than pure native, but sufficient for this use case
- Native modules: Some functionality requires native modules (audio, file system)
- Platform-specific code: Minimal, mostly in native modules

### Audio Recording Architecture

**Decision: Local fork of `react-native-audio-recorder-player`**

The app uses a **local fork** of `react-native-audio-recorder-player` located at `./react-native-audio-recorder-player/`. This allows for:
- Custom modifications to audio recording behavior
- Control over metering data collection
- Platform-specific optimizations

**Implementation Details:**

#### iOS (Swift)
- Uses `AVAudioRecorder` from AVFoundation
- Native module: `RNAudioRecorderPlayer.swift`
- Bridge: `RNAudioRecorderPlayer.m` (Objective-C bridge)
- Event emitter: `RCTEventEmitter` for real-time callbacks

#### Android (Kotlin)
- Uses `MediaRecorder` from Android SDK
- Native module: `RNAudioRecorderPlayerModule.kt`
- Event emitter: `RCTDeviceEventEmitter` for callbacks

**Why not pure Swift/Kotlin?**
- React Native bridge provides seamless integration
- JavaScript layer handles business logic
- Native modules handle platform-specific audio APIs
- Best of both worlds: native performance + React Native flexibility

### Waveform Visualization

#### How It Works

The waveform visualizer is a **sophisticated React component** (`Waveform.tsx`) that displays audio amplitude over time using a **fixed-grid storage system** for perfect synchronization.

**Architecture: Fixed-Grid System**

The waveform uses a **fixed-grid index-based system** where each array index represents a fixed time bin (50ms by default). This ensures perfect synchronization between recording time and waveform data.

**Data Flow:**

1. **During Recording** (`RecordScreen.tsx`):
   ```typescript
   // Fixed-grid storage: index = time bin
   const idx = Math.round(e.currentPosition / BAR_MS); // 50ms bins
   waveformAllRef.current[idx] = normalized; // Store at index, fill gaps with 0
   ```
   - Each array index represents a 50ms time bin (20 bars/second)
   - Index `i` corresponds to time `i * 50ms`
   - Gaps are filled with `0` to maintain index==time relationship
   - Entire recording waveform stored (not just 30s window)
   - Normalization: dB (-60 to 0) → 0-1 range with gamma correction (power 1.6)

2. **Storage**:
   - Full waveform stored as JSON array in `recordings.waveform_data` column
   - Each value is a normalized amplitude (0-1)
   - Database migration handles adding column to existing databases
   - Stored when recording is saved

3. **Playback Loading** (`PlaybackScreen.tsx`):
   - Waveform data loaded from database on mount
   - Parsed from JSON string to number array
   - Falls back to empty array if no data (shows message)

4. **Visualization** (`Waveform.tsx`):
   - **Two modes**:
     - `'rolling'`: 30-second scrolling window (Voice Memos style)
     - `'full'`: Entire waveform visible at once
   - **Two cursor modes**:
     - `'pinned'`: Cursor fixed at right edge (recording mode)
     - `'follow'`: Cursor follows playback position (playback mode)
   - **Auto-scaling**: Uses 98th percentile of visible bars for dynamic scaling
   - **Downsampling**: Buckets samples to pixel width for efficient rendering
   - **Tag visualization**: Blue rectangles overlay at tag timestamps

**Key Features:**
- **Fixed-grid storage**: Perfect time synchronization (no drift)
- **Real-time visualization**: Updates as audio is recorded
- **Persistent storage**: Full waveform saved to database
- **Dual display modes**: Rolling window or full view
- **Tag markers**: Blue rectangles show tagged moments
- **Playback sync**: Cursor perfectly aligned with audio position
- **Auto-scaling**: Dynamic Y-axis scaling for optimal visualization
- **Efficient rendering**: Downsampled to pixel density

**Technical Details:**
- **Bar duration**: 50ms (20 bars/second, configurable via `barDurationMs`)
- **Window size**: 30 seconds (600 bars, configurable via `windowMs`)
- **Normalization**: dB values (-60 to 0) → 0-1 range with gamma correction (1.6)
- **Scaling**: 98th percentile auto-scaling every 500ms
- **Rendering**: Downsampled to match screen pixel density (`minBarPx=2`)

**Why Fixed-Grid System?**
- **Perfect sync**: Index directly maps to time (no calculation drift)
- **Efficient**: O(1) lookup by time → index
- **Simple**: No complex timebase tracking
- **Reliable**: Eliminates cursor lag from state update ordering

### Database Architecture

**Decision: SQLite via `react-native-sqlite-storage`**

**Schema:**

```sql
recordings (
  id INTEGER PRIMARY KEY,
  filename TEXT NOT NULL,
  session_name TEXT,
  created_at INTEGER,
  duration_ms INTEGER,
  waveform_data TEXT  -- JSON array, ready for future use
)

tags (
  id INTEGER PRIMARY KEY,
  recording_id INTEGER,
  timestamp_ms INTEGER,
  label TEXT,
  FOREIGN KEY(recording_id) REFERENCES recordings(id)
)
```

**Why SQLite?**
- **Local storage**: No server required
- **Relational data**: Perfect for recordings + tags relationship
- **Performance**: Fast queries for filtering/sorting
- **Offline-first**: Works without network
- **Lightweight**: Minimal overhead

**Data Flow:**
1. Recording saved → `addRecording()` inserts metadata
2. Tags added → `addTag()` links to recording via `recording_id`
3. Playback → `getRecordingById()` + `getTagsForRecording()` loads data
4. List view → `getAllRecordings()` with sorting options

### File Storage

**Decision: Platform-native file system**

- **iOS**: Files stored in app's cache directory
- **Android**: Files stored in app's internal storage
- **Format**: M4A (AAC) for iOS, platform default for Android
- **File management**: `react-native-fs` for file operations

**Why not cloud storage?**
- Privacy: Baby audio data stays on device
- Offline-first: No network dependency
- Performance: Instant access
- Cost: No storage fees

## Component Architecture

### Screen Components

1. **RecordScreen** (`src/screens/RecordScreen.tsx`)
   - Manages recording state
   - Collects waveform data in real-time
   - Handles tagging during recording
   - Saves recordings to database

2. **PlaybackScreen** (`src/screens/PlaybackScreen.tsx`)
   - Loads stored waveform from database
   - Displays waveform in 'full' or 'rolling' mode (toggleable)
   - Manages playback controls
   - Shows tags as blue rectangles on waveform
   - Allows adding tags during playback
   - Uses 'follow' cursor mode for playback visualization

3. **RecordingListScreen** (`src/screens/RecordingListScreen.tsx`)
   - Lists all recordings
   - Filtering and sorting
   - Navigation to playback

### Reusable Components

1. **Waveform** (`src/components/Waveform.tsx`)
   - Sophisticated waveform visualization component
   - Supports 'rolling' (30s window) and 'full' (entire waveform) modes
   - 'pinned' cursor mode (right edge) for recording
   - 'follow' cursor mode for playback
   - Auto-scaling based on percentile of visible data
   - Downsampling to pixel density for efficient rendering
   - Tag visualization as blue rectangles
   - Highly configurable via props

2. **CircularRecordButton** (`src/components/CircularRecordButton.tsx`)
   - Record/pause/stop controls
   - Visual state indicators

3. **CircularPlayButton** (`src/components/CircularPlayButton.tsx`)
   - Play/pause controls
   - Playback state management

4. **RecordingLayout** (`src/components/RecordingLayout.tsx`)
   - Common layout for recording/playback screens
   - Consistent UI structure

## Data Flow

### Recording Flow

```
User taps Record
  ↓
RecordScreen.startRecording()
  ↓
audioRecorderPlayer.startRecorder() [Native Module]
  ↓
Native: AVAudioRecorder/MediaRecorder starts
  ↓
addRecordBackListener() [Event Callback]
  ↓
Every callback: Fires with:
  - currentPosition (ms)
  - currentMetering (dB)
  ↓
JavaScript: Fixed-grid storage
  - Calculate index: idx = round(currentPosition / 50ms)
  - Fill gaps with 0: while (array.length < idx) array.push(0)
  - Normalize dB: (-60 to 0) → (0 to 1) with gamma 1.6
  - Store: waveformAllRef.current[idx] = normalized
  ↓
React: Re-render Waveform component
  - Mode: 'rolling' (30s window)
  - Cursor: 'pinned' (right edge)
  ↓
User taps Stop
  ↓
audioRecorderPlayer.stopRecorder()
  ↓
Save to database: addRecording({ waveformData: waveformAllRef.current })
  - Serialize array as JSON string
  - Store in recordings.waveform_data column
  ↓
Save tags: addTag() for each live tag
```

### Playback Flow

```
User selects recording
  ↓
PlaybackScreen loads on mount:
  - Recording metadata: getDb().executeSql('SELECT duration_ms, waveform_data...')
  - Tags: getTagsForRecording(recordingId)
  - Waveform: Parse JSON from waveform_data column
  ↓
Parse waveform_data JSON → number[] array
  ↓
If no waveform_data: Show "No waveform data" message
  ↓
User taps Play
  ↓
audioRecorderPlayer.startPlayer()
  ↓
addPlayBackListener() [Event Callback]
  ↓
Every callback: Fires with:
  - currentPosition (ms)
  - duration (ms)
  ↓
React: Update playbackMs state
  ↓
Waveform component re-renders:
  - Mode: 'full' or 'rolling' (user toggleable)
  - Cursor: 'follow' (tracks playback position)
  - Cursor X: (playbackMs / totalMs) * width (full mode)
              or (playbackMs - visibleStart) / windowMs * width (rolling mode)
  - Tag rectangles: Overlay at tag timestamps
  ↓
User can toggle between 'full' and 'rolling' view
```

## Performance Considerations

### Waveform Data Collection
- **Fixed-grid storage**: O(1) time lookup by index
- **Sparse array**: Gaps filled with 0, no wasted memory for silence
- **Normalization**: dB → 0-1 conversion with gamma correction (1.6) done in JavaScript
- **Memory**: Full waveform stored (reasonable for typical recording lengths)

### Database Queries
- **Indexed**: Primary keys for fast lookups
- **Filtered**: Queries use WHERE clauses for efficiency
- **JSON storage**: Waveform stored as compact JSON string
- **Pagination**: Can be added for large recording lists

### Rendering
- **Memoization**: `useMemo` for expensive calculations (visible range, downsampling, tag positions)
- **Downsampling**: Buckets samples to pixel density (`minBarPx=2`)
- **Auto-scaling**: Throttled to 500ms updates
- **Windowed**: Rolling mode only processes visible 30-second window
- **Efficient**: React Native handles efficient re-renders with memoized calculations

## Future Enhancements

### Waveform Features
- ✅ **Fixed-grid storage**: Implemented
- ✅ **Database persistence**: Implemented
- ✅ **Dual display modes**: Implemented (full/rolling)
- ✅ **Tag visualization**: Implemented (blue rectangles)
- Touch-to-seek on waveform (tap to jump to position)
- Horizontal scrolling for full mode on long recordings
- Waveform zoom in/out
- Export waveform as image
- Background waveform generation for old recordings without data

### Performance Optimizations
- ✅ **Downsampling**: Implemented (buckets to pixel density)
- ✅ **Auto-scaling**: Implemented (98th percentile)
- ✅ **Memoization**: Implemented (useMemo for calculations)
- Virtual scrolling for very long recordings (100+ minutes)
- Progressive loading of waveform data

## Development Workflow

### Local Module Development
The `react-native-audio-recorder-player` is a local fork, allowing:
- Direct modification of native code
- Testing changes immediately
- Custom features not in upstream

### Build Process
1. **iOS**: CocoaPods manages native dependencies
2. **Android**: Gradle manages dependencies
3. **Metro**: Bundles JavaScript/TypeScript
4. **Hermes**: Compiles JavaScript to bytecode

### Testing
- Unit tests: Jest (configured)
- Integration: Manual testing on simulators/devices
- Native modules: Tested via React Native bridge

## Key Takeaways

1. **Hybrid Architecture**: React Native + Native Modules provides best balance
2. **Real-time Data**: Event emitters enable live waveform updates
3. **Local-first**: SQLite + file system for offline functionality
4. **Type Safety**: TypeScript prevents runtime errors
5. **Modular Design**: Components are reusable and testable
6. **Performance**: Native modules handle heavy lifting (audio), React handles UI

## File Structure

```
babytalkApp-1/
├── src/
│   ├── components/        # Reusable UI components
│   ├── screens/          # Screen components
│   ├── navigation/       # Navigation types
│   └── db.ts            # Database operations
├── react-native-audio-recorder-player/  # Local fork
│   ├── ios/             # iOS native code
│   ├── android/         # Android native code
│   └── index.ts         # TypeScript interface
├── ios/                 # iOS project files
└── android/             # Android project files
```

This architecture provides a solid foundation for a mobile audio recording app with real-time visualization, while maintaining flexibility for future enhancements.

