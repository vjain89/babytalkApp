# BabyTalk App

React Native app for recording baby sounds, tagging moments, visualizing waveforms, and exporting sessions locally.

**Why this exists:** existing first-word trackers are diaries — a log of when words appeared. BabyTalk captures the audio moments themselves, tagged in real time with context and emotion, building a structured dataset of one child's speech development — with the longer-term goal of training models on it. Solo project.

## Features

- **Record / pause / resume / stop** with live metering
- **Waveform visualization** — 3s live tail while recording; 8s rolling window on playback (50ms bars, peak dB)
- **Tagging** during record and playback, with markers on the waveform
- **Session library** — search by name or tag, sort, rename
- **Export** single recording or Export All (audio + JSON) via the system share sheet

## Stack

| Piece | Choice |
|-------|--------|
| Framework | React Native 0.78 / React 19 / TypeScript |
| Audio | Local fork of `react-native-audio-recorder-player` |
| Storage | SQLite (`react-native-sqlite-storage`) + `.m4a` on disk |
| Navigation | React Navigation native stack |

## Project layout

```
src/
├── screens/          # Record, Playback, RecordingList
├── components/       # Waveform, buttons, layout
├── waveform/         # Capture/storage helpers, scale, config
├── db.ts             # SQLite API
└── navigation/
docs/
└── ARCHITECTURE_DEEP_DIVE.md   # Engine → bridge → capture → store → render → playback
```

## Setup

### Prerequisites

- Node ≥ 18
- Xcode (iOS) with CocoaPods
- Physical device: Xcode major version must support your iOS version (e.g. iOS 26.x needs Xcode 26.x / macOS Tahoe)

### Install

```sh
npm install
cd ios && bundle install && bundle exec pod install && cd ..
```

The audio module is linked as `file:../react-native-audio-recorder-player` (sibling checkout). Ensure that directory exists next to this repo (or adjust `package.json` / reinstall).

### Run

```sh
npm start
# other terminal:
npm run ios          # simulator
# or open in Xcode:
open ios/babytalkApp.xcworkspace
```

After **native** audio-module changes (Swift/Kotlin), do a Clean Build in Xcode (`Shift+Cmd+K`), then Run — Metro Fast Refresh alone is not enough.

### Android

```sh
npm run android
```

## Waveform model (summary)

| Phase | Window | Resolution | Y scale |
|-------|--------|------------|---------|
| Recording | last 3s | 50ms peak bars | Fixed −60…0 dB |
| Playback | last 8s | 50ms peak bars | Fixed per session (95th percentile) |

Samples are stored as sparse `{ tMs, avgDb, peakDb }` with metadata (`barDurationMs`, `version`) in SQLite. Details: [docs/ARCHITECTURE_DEEP_DIVE.md](docs/ARCHITECTURE_DEEP_DIVE.md).

## Docs

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE_DEEP_DIVE.md](docs/ARCHITECTURE_DEEP_DIVE.md) | File-tied architecture walkthrough |
| [INSTALL_TO_IPHONE.md](INSTALL_TO_IPHONE.md) | USB install via Xcode |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Historical status notes (may lag code) |

## Roadmap

- [x] Waveform rescope (50ms sparse raw-dB metering envelope)
- [x] Envelope UX polish (narrower tags, tighter Y / noise floor, gapped bars)
- [ ] File-derived peaks for playback (Path B — true envelope from `.m4a`)
- [ ] Structured export for offline/cloud processing
- [ ] Google Drive backup
- [ ] Optional PCM-style oscilloscope view
- [ ] Local and/or cloud audio processing

## Troubleshooting

| Issue | Likely cause |
|-------|----------------|
| Developer disk image / mount error on device | Xcode too old for the phone’s iOS — upgrade macOS/Xcode or use Simulator |
| Flat waveform while speaking | Mic permission denied, or native rebuild missing after metering changes |
| `react-native-sqlite-storage` CLI config warning | Harmless package metadata warning; ignore unless DB fails at runtime |
| Stub / no real audio | Wrong audio module path — confirm `node_modules` symlink and pods |

## License

Private project.
