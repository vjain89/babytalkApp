# BabyTalk App — Architecture Deep Dive

**Document version:** July 2026 (Path B + playback seek)  
**Branch context:** `feature/path-b-file-peaks` (file-derived peaks, waveform seek); merge to `main` when validated  
**Audience:** Developer reference — start with **§4** for the file-tied pipeline tour

---

## 1. Executive summary

BabyTalk is a React Native app for recording baby sounds, tagging moments, visualizing waveforms, and exporting sessions locally. The architecture deliberately **separates concerns**:

| Layer | Responsibility | Where it lives |
|-------|----------------|----------------|
| **Audio engine** | Record/play `.m4a`, position + metering, **file peak extract** | `react-native-audio-recorder-player` (Swift/Kotlin) |
| **Bridge** | Native ↔ JS events (~50ms) + async native methods | `index.ts` listeners + `rn-recordback` / `rn-playback` |
| **Waveform capture** | Live events → sparse `{tMs, avgDb, peakDb}` | `RecordScreen.tsx` + `waveform/storage.upsertSample` |
| **Waveform store** | Persist payload (`source`: metering \| file) + tags | `serializeWaveform` → `db.ts` → SQLite |
| **Waveform renderer** | Dense bars, cursor, tags, seek overlay, fixed Y | `Waveform.tsx` + `waveform/scale.ts` |
| **Playback** | Load / upgrade peaks, 8s window, play sync, **scrub seek** | `PlaybackScreen.tsx` |

The audio engine is **not** a waveform library. Visualization lives in `src/waveform/` and `src/components/Waveform.tsx`. Capture wiring is in the screens, not a separate `waveform/capture` module.

**What you see on screen** is a **loudness envelope** (VU-style bars), not a PCM oscilloscope. Bars come from either live metering (while recording) or **Path B** peaks decoded from the `.m4a` after save / on playback load.

**Read next:** §3 diagram → **§4 hop-by-hop walkthrough** → §8 Path A/B → §12 roadmap.

**Near-term product arc (planned, not all built):** structured “session kit” export → Finder USB backup + import → Mac offline analysis → Fitbit-style review of suggested moments → search / timeline / confidence. Google Drive is **out** of the near-term plan.

---

## 2. Technology stack

| Component | Choice | Role |
|-----------|--------|------|
| React Native 0.78.1 | Cross-platform shell | UI, navigation, business logic |
| TypeScript 5.0.4 | Language | Type safety |
| Hermes | JS engine | Runtime |
| react-native-audio-recorder-player | Local fork | Native record/play, metering, `extractWaveformPeaks` |
| react-native-sqlite-storage | SQLite | Metadata, tags, waveform JSON |
| react-native-fs | File I/O | Resolve audio paths, export files |
| react-native-share | Share sheet | Export recording + tags |
| @react-navigation/native-stack | Navigation | Record → List → Playback |

### Why React Native (not pure Swift)?

- Single codebase for iOS and Android
- Fast iteration with Metro hot reload
- Heavy lifting (microphone, filesystem, file decode) delegated to native modules
- Trade-off: bridge latency (~50ms ticks) vs 60fps native UI (Voice Memos)

---

## 3. System diagram

There is **no** separate `waveform/capture` module. Capture wiring lives in `RecordScreen.tsx`; helpers (`upsertSample`, `densifyPeaks`, parse/serialize) live in `src/waveform/storage.ts`. Path B decode lives in the **audio module**, invoked from Save and Playback.

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Native (JS/TS)                      │
├─────────────────────────────────────────────────────────────────┤
│  RecordScreen.tsx       PlaybackScreen.tsx   RecordingListScreen │
│   live metering          load + Path B upgrade  list / Export All │
│   Tag Now → pendingMs    seek / scrub window                     │
│   Save → extract peaks   play + seekToPlayer                     │
│         │                      │                                 │
│         ▼                      ▼                                 │
│  waveform/storage.ts     waveform/storage + scale + audioPath    │
│  upsert / densify        parse / densify / resolveAudioUri       │
│  serializeWaveform       computePlaybackDbRange                  │
│         │                      │                                 │
│         └──────────┬───────────┘                                 │
│                    ▼                                             │
│             Waveform.tsx (display + optional seek overlay)       │
│                    │                                             │
│                    ▼                                             │
│                  db.ts  ←→  babytalk.db (waveform_data JSON)      │
└────────────────────┬────────────────────────────────────────────┘
                     │ RN Bridge (~50ms events) + native methods
┌────────────────────▼────────────────────────────────────────────┐
│         react-native-audio-recorder-player                       │
│  index.ts: listeners + extractWaveformPeaks()                    │
│  iOS:  RNAudioRecorderPlayer.swift                               │
│        AVAudioRecorder (meter) + AVAudioFile (Path B peaks)      │
│  Android: MediaRecorder metering; Path B stub → [] for now       │
│  Events: rn-recordback , rn-playback                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  recording.m4a              babytalk.db
  (Caches directory)         tags + waveform_data
```

**Pipeline we follow below:**  
`engine → bridge → capture → store (+ Path B) → render → playback (+ seek)`

---

## 4. End-to-end walkthrough (tied to files)

This section follows **one tap of Record** through the stack, then **Save** (including Path B), then **Playback** (including seek). File paths are relative to the app root unless noted.

### Hop overview

| Hop | Layer | Primary files |
|-----|-------|---------------|
| 1 | Engine | `react-native-audio-recorder-player/ios/RNAudioRecorderPlayer.swift` |
| 2 | Bridge | same Swift + `react-native-audio-recorder-player/index.ts` |
| 3 | Capture | `src/screens/RecordScreen.tsx` + `src/waveform/storage.ts` (`upsertSample`) |
| 4 | Store + Path B | `RecordScreen.handleSave` → `extractWaveformPeaks` → `serializeWaveform` → `db.ts` |
| 5 | Render (live) | `storage.densifyPeaks` → `Waveform.tsx` + `scale.ts` |
| 6 | Playback | `PlaybackScreen` → parse / optional Path B upgrade → densify → seek → play |

---

### Hop 1 — Engine (native record + meter)

**What starts it (JS):** `RecordScreen.startRecording()` calls:

1. `audioRecorderPlayer.setSubscriptionDuration(SUBSCRIPTION_SEC)` — `0.05` from `src/waveform/config.ts`
2. `audioRecorderPlayer.startRecorder('recording.m4a', undefined, true)` — third arg enables metering

**What native does (iOS):** In `RNAudioRecorderPlayer.swift`:

- `startRecorder(...)` creates `AVAudioRecorder`, sets `isMeteringEnabled`, starts recording to Caches, then `startRecorderTimer()`.
- `startRecorderTimer()` schedules a repeating `Timer` with interval `subscriptionDuration` (50ms after the JS call above).
- Each tick calls `updateRecorderProgress(timer:)`:

```swift
// RNAudioRecorderPlayer.swift — updateRecorderProgress
audioRecorder.updateMeters()
currentMetering = audioRecorder.averagePower(forChannel: 0)
currentPeakMetering = audioRecorder.peakPower(forChannel: 0)
// currentPosition = audioRecorder.currentTime * 1000
sendEvent(withName: "rn-recordback", body: status)
```

**Android analog:** `RNAudioRecorderPlayerModule.kt` posts a `Runnable` every `subsDurationMillis`, reads `MediaRecorder.maxAmplitude`, converts to dB, and puts the **same** value in both `currentMetering` and `currentPeakMetering` (no separate peak API).

**What the engine owns:** writing `.m4a`, session clock (`currentPosition`), loudness snapshots (dB), and (separately) **decoding peaks from a finished file** via `extractWaveformPeaks`.  
**What it does not own:** bar layout, SQLite, tags, Y-scale, React UI, seek gesture logic.

**Module path caveat:** `package.json` points at `file:../react-native-audio-recorder-player`. Screens often import `../../react-native-audio-recorder-player` (in-repo nested copy). Keep sibling and nested copies in sync for native API changes. Swift/Kotlin changes need a **full native rebuild**, not Metro Fast Refresh alone.

---

### Hop 2 — Bridge (native event → JS listener)

**Bridge:** React Native serializes a small map from native → JS. Native cannot call React functions directly; it emits a named event.

**Emit (native):** `sendEvent(withName: "rn-recordback", body: { isRecording, currentPosition, currentMetering, currentPeakMetering })`.

**Subscribe (JS):** `react-native-audio-recorder-player/index.ts` — `addRecordBackListener(callback)`:

- iOS: `NativeEventEmitter(RNAudioRecorderPlayer).addListener('rn-recordback', callback)`
- Android: `DeviceEventEmitter.addListener('rn-recordback', callback)`

**Who registers the callback:** `RecordScreen.tsx` immediately after `startRecorder` succeeds:

```typescript
audioRecorderPlayer.addRecordBackListener((e) => { /* Hop 3 */ });
```

**Payload fields:**

| Field | Role |
|-------|------|
| `currentPosition` | ms on the recorder clock → timer, tags, bin index |
| `currentMetering` | averagePower (iOS) — stored as `avgDb` |
| `currentPeakMetering` | peakPower (iOS) — stored as `peakDb`, **used for bar height** |

**Alignment rule:** one sample per callback, snapped to **50ms** bins (`BAR_MS`). Callbacks can skip under load; the **file** stays continuous, the **metering stream** can have holes — that is why live capture is sparse.

**Playback side of the bridge:** `addPlayBackListener` stores a JS callback; native emits `rn-playback` with `{ currentPosition, duration, isFinished }` while playing. Seek uses the separate native method `seekToPlayer(ms)` (not an event).

---

### Hop 3 — Capture (JS turns events into sparse samples)

**File:** `src/screens/RecordScreen.tsx` (listener body) + `src/waveform/storage.ts` (`upsertSample`).

On every `rn-recordback`:

1. `tMs = e.currentPosition`; `setRecordMs(tMs)` (drives timer + re-render cadence).
2. Read `avgDb` / `peakDb` from metering fields (fallback `CAPTURE_DB_MIN` = -160).
3. `upsertSample(samplesRef.current, { tMs, avgDb, peakDb })`.

**`upsertSample` (`storage.ts`):**

- Snaps `tMs` to nearest `BAR_MS` bin: `round(tMs / 50) * 50`.
- If that bin already exists, keeps the **louder** avg/peak (`Math.max`).
- Otherwise appends (or inserts if rare out-of-order).

**In-memory shape during recording:**

```typescript
samplesRef: WaveformSample[]  // sparse, mutated in place — live metering truth until Save
displayPeaks: number[]        // React state — dense peak dB for the last 3s only
pendingTagMs: number | null   // stamped when user taps Tag Now (not when label is chosen)
```

Types live in `src/waveform/types.ts`:

```typescript
type WaveformSample = { tMs: number; avgDb: number; peakDb: number };
type WaveformSource = 'metering' | 'file';
```

Capture stores **raw dB** across the full -160…0 range. Display mapping happens later (Hop 5 / 6).

**Live tagging:** `handleTagNow` sets `pendingTagMs = floor(recordMs)` before the label modal opens, so picking a label does not slide the timestamp forward.

---

### Hop 4 — Store (persist payload + Path B on save)

**When:** user taps Save → `RecordScreen.handleSave` → stop recorder if needed → prompt for session name.

**Path B on save (iOS):** after stop, call `extractWaveformPeaks(filePath, BAR_MS)`. Native opens the `.m4a` with `AVAudioFile`, walks PCM in `barDurationMs` windows, and returns `{ tMs, avgDb, peakDb }[]`. If extraction succeeds and is non-empty, those samples replace live metering and `source = 'file'`. On failure (or Android stub returning `[]`), fall back to live metering with `source = 'metering'`.

**Build payload:**

```typescript
const payload = emptyWaveformPayload(source); // version, barDurationMs: 50, source
payload.samples = samples; // file peaks or metering
waveformData: serializeWaveform(payload) // JSON.stringify
```

**`WaveformPayload` v2** (written into SQLite `recordings.waveform_data`):

```json
{
  "version": 2,
  "barDurationMs": 50,
  "source": "file",
  "samples": [
    { "tMs": 0, "avgDb": -45.2, "peakDb": -38.1 },
    { "tMs": 50, "avgDb": -42.0, "peakDb": -35.0 }
  ]
}
```

**Why metadata?** Playback must know bins are 50ms and whether peaks came from the file. Older code guessed `durationMs / array.length`, which breaks on sparse samples or bin-size changes. `version` + `barDurationMs` (+ optional `source`) are the contract.

**DB write:** `src/db.ts` → `addRecording({ filename, sessionName, durationMs, waveformData })` inserts into `recordings`. Then `addTag(...)` for each live tag. `updateWaveformData` can rewrite `waveform_data` later (playback Path B upgrade).

**Also on disk:** the `.m4a` from Hop 1 (Caches). DB stores the **filename**; list/playback resolve the path via `react-native-fs` / `waveform/audioPath.resolveAudioUri` when needed.

**Legacy:** `parseWaveformData` still accepts old flat `number[]` (normalized 0–1) and approximates dB for display — new recordings write v2 with optional `source`.

---

### Hop 5 — Render while recording (live tail)

Still inside the same record-back listener, after `upsertSample`:

```typescript
setDisplayPeaks(
  densifyPeaks(samplesRef.current, tMs - RECORD_WINDOW_MS, tMs, BAR_MS)
);
```

**`densifyPeaks` (`storage.ts`):** sparse samples → dense `peakDb[]` for `[startMs, endMs)`. Missing bins → `CAPTURE_DB_MIN` (silence). `startMs` may be **negative** early in a session so the window is always 3s and “now” sits at the **right** edge (left-padded silence).

**Constants:** `RECORD_WINDOW_MS = 3000` → **60 bars** at 50ms.

**UI:** `RecordScreen` passes into `Waveform.tsx`:

| Prop | Value |
|------|--------|
| `peaksDb` | `displayPeaks` (already densified for the window) |
| `windowMs` | `RECORD_WINDOW_MS` (3s) |
| `progressMs` | `recordMs` |
| `mode` | `'rolling'` |
| `cursorMode` | `'pinned'` (cursor at right edge) |
| `dbRange` | `RECORD_DB_RANGE` → fixed **-45…0** (capture still stores -160…0) |
| `tagWidthMs` | `TAG_MARKER_MS` (80ms) |
| `seekable` | false (record screen does not scrub) |

**`Waveform.tsx`:** display-only for record. Maps each peak dB with `dbToHeight01(db, dbRange)` (`scale.ts`): noise floor (**-42 dB**) → flat hairline; no live autoscale. Bars use a **1px gap** (`BAR_GAP_PX`) and height **72** so the strip reads as spikes, not a solid blob. Draws tag markers from live tag timestamps.

Render cadence ≈ callback cadence (~50ms), because `setRecordMs` / `setDisplayPeaks` run on each event.

---

### Hop 6 — Playback (load → Path B upgrade → window → seek → play)

**Load (mount):** `PlaybackScreen.loadRecordingWaveform`:

1. SQL: `SELECT duration_ms, waveform_data FROM recordings WHERE id = ?`
2. `parseWaveformData(row.waveform_data)` → `{ samples, barDurationMs, source }`
3. **If `source !== 'file'` or samples empty:** resolve URI via `resolveAudioUri`, call `extractWaveformPeaks`, and if successful `updateWaveformData` so the DB is upgraded for next open
4. `setDbRange(computePlaybackDbRange(payload.samples))` — **once per session**

**`computePlaybackDbRange` (`scale.ts`):** sort `peakDb`, take **95th percentile**, map that level to **80%** of bar height (`TARGET_PEAK_FRACTION`). Shrieks above that clip at the top; quieter speech stays visible. No breathing autoscale during play.

**Play:** `startPlayer(filePath)`; if `playbackMs > 0`, `seekToPlayer(playbackMs)`; then `addPlayBackListener` → `setPlaybackMs(e.currentPosition)`.

**Window for drawing:**

- **Rolling (default):** densify `[rollingWindowEndMs - 8s, rollingWindowEndMs)`. While scrubbing, `rollingWindowEndMs` is **frozen** at the playhead where the gesture started so bars do not scroll under the finger.
- **Full:** densify `0…duration` (may downsample in `Waveform.tsx` if bars exceed pixel width).

**Seek / scrub:**

| Piece | Role |
|-------|------|
| `Waveform` `seekable` | PanResponder overlay maps touch X → time |
| `onScrubChange` | Preview ms + freezes rolling window end |
| `onSeek` / `seekTo` | Clamp, `setPlaybackMs`, `seekToPlayer` |
| Tag list tap | Same `seekTo` path |

Full view maps `x / width * duration`. Rolling maps within the **frozen** 8s window. Tag-this-moment uses the scrub preview time when dragging.

**UI:** `Waveform` with `cursorMode="follow"`, `scrubPreviewMs`, `rollingWindowEndMs`, tags from `getTagsForRecording`, duration label `elapsed / total`.

**Sync note:** `playbackMs` is the player clock delivered over the bridge (~50ms ticks + React render). Same module as recording → no dual-clock drift; not sample-accurate.

---

### Walkthrough summary (one sample’s life)

```
AVAudioRecorder (file + meters)
  → Timer 50ms → updateRecorderProgress
  → sendEvent("rn-recordback")
  → RecordScreen listener
  → upsertSample(samplesRef)          // sparse raw dB (live)
  → densifyPeaks → setDisplayPeaks     // live 3s tail
  → Waveform.tsx + RECORD_DB_RANGE     // fixed -45..0
  → [Tag Now] pendingTagMs stamp
  → [Save] extractWaveformPeaks (Path B) or keep metering
  → serializeWaveform(source) → db.addRecording
  → [Open] parseWaveformData → upgrade to file peaks if needed
  → computePlaybackDbRange → densifyPeaks(8s)
  → Waveform + follow cursor + seek overlay
  → startPlayer / seekToPlayer + rn-playback → playbackMs
```

---

## 5. Design principles and constants

### Locked principles (July 2026)

1. **Separate engine from visualization** — library records/plays/extracts; `src/waveform/*` + `Waveform.tsx` own viz and seek UX
2. **Capture and display both at 50ms** (may diverge later for PCM views)
3. **Render cadence aligned to callbacks** (~50ms state updates while recording)
4. **Fixed Y scale per session** — record: -45…0 visible; playback: percentile once at load
5. **Path B preferred** — after save (and on playback if missing), regenerate peaks from `.m4a`
6. **Envelope, not oscilloscope** — UI is loudness bars; bipolar PCM drawing is a future option

### Constants (`src/waveform/config.ts`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `SUBSCRIPTION_SEC` | 0.05 | Native callback interval |
| `BAR_MS` | 50 | One bar / one bin = 50ms |
| `RECORD_WINDOW_MS` | 3000 | Live tail: 60 bars |
| `PLAYBACK_WINDOW_MS` | 8000 | Rolling playback: 160 bars |
| `CAPTURE_DB_MIN/MAX` | -160 / 0 | Full range stored |
| `RECORD_VISIBLE_DB_MIN/MAX` | -45 / 0 | Fixed display while recording |
| `PLAYBACK_SCALE_PERCENTILE` | 0.95 | Session Y from 95th percentile peak |
| `TARGET_PEAK_FRACTION` | 0.8 | That percentile → 80% bar height |
| `DISPLAY_NOISE_FLOOR_DB` | -42 | At/below → silence hairline |
| `TAG_MARKER_MS` | 80 | Tag marker width on the time axis |
| `BAR_GAP_PX` | 1 | Gap between bars |
| `WAVEFORM_HEIGHT` | 72 | Strip height |
| `WAVEFORM_SCHEMA_VERSION` | 2 | Payload contract |

---

## 6. Sparse storage and densify (why this shape)

### Why sparse `{ tMs, avgDb, peakDb }[]`?

A dense array assumes slot `i` = time `i × 50ms`. Dropped live callbacks leave holes or force inventing silence in the stored array, and `length × 50 ≈ duration` breaks. Sparse storage only records measured ticks (or Path B bins that had audio). Display densifies on the fly.

Path B extraction typically produces a **dense-looking** list (one bin per 50ms of file), but still uses the same sparse-compatible sample shape and densify path.

### Why densify only for render?

`densifyPeaks` fills missing bins with silence **for drawing**. The DB keeps the honest sample list. You can change window size or display metric without rewriting stored data.

### Pixel budget (playback, no downsampling)

`bars = windowSeconds × 20` (at 50ms). On ~390px width with `minBarPx = 1`, **8s → 160 bars** fits. Longer full-clip views may downsample in `Waveform.tsx` only when `peaksDb.length > width / minBarPx`.

---

## 7. Parameter map (visualization success)

### Layer 1 — Signal acquisition (native)

| Parameter | Source | Notes |
|-----------|--------|-------|
| Callback interval | `setSubscriptionDuration(0.05)` | Max ~20 updates/sec (live) |
| averagePower | iOS AVAudioRecorder | Stored as avgDb |
| peakPower | iOS AVAudioRecorder | Displayed while recording |
| File peaks | iOS `AVAudioFile` in `extractWaveformPeaks` | Path B; Android stub returns `[]` |
| Capture dB range | -160..0 | Full range stored |

### Layer 2 — Capture & storage

| Parameter | Value | Notes |
|-----------|-------|-------|
| Bin size | 50ms | Matches callbacks / extract window |
| Structure | Sparse `{tMs, avgDb, peakDb}` | Robust to dropped ticks |
| Persisted format | WaveformPayload JSON | version + barDurationMs + source |

### Layer 3 — Time axis (X)

| Parameter | Recording | Playback |
|-----------|-----------|----------|
| Window | 3s tail | 8s rolling (or full clip) |
| Bars in window | 60 | 160 (rolling) |
| Cursor | Pinned right | Follows playhead (moves while scrubbing) |
| Seek | No | Tap/drag; freeze window while scrubbing |
| Downsampling | None (fits screen) | None for 8s @ 50ms |

### Layer 4 — Amplitude axis (Y)

| Phase | Strategy |
|-------|----------|
| Recording | Fixed -45..0 dB visible range |
| Playback | 95th percentile peak → 80% height; fixed for session |
| Noise floor | -42 dB → render as silence |

### Layer 5 — Sync

| Check | Status |
|-------|--------|
| Same audio clock for position | Yes (native module) |
| Bridge latency | Up to ~50ms + React render |
| Tag timestamps vs waveform | Aligned when barDurationMs metadata correct; Tag Now uses press-time stamp |
| Seek vs rolling window | Window frozen for duration of scrub gesture |

---

## 8. Path A vs Path B (waveform fidelity)

### Path A — Live metering (still used for the live strip)

- One loudness number every 50ms while recording
- Summarizes ~2,205 PCM samples at 44.1kHz (iOS) into a single meter reading
- **VU-meter trace**, not sample-accurate waveform
- Good enough for live feedback; may miss detail vs the finished file

### Path B — File analysis (current default after save / on upgrade)

- After stop (and on playback if DB still has metering-only data): decode `.m4a` → PCM → peak/RMS per 50ms window
- Reflects **actual audio content** in the file
- Can regenerate for old recordings on open
- iOS: `extractWaveformPeaks` in `RNAudioRecorderPlayer.swift`
- Android: stub returns empty array → keep metering until implemented

### What Path B is not

Still an **envelope** (one peak/avg per 50ms), not a bipolar PCM oscilloscope. A future “true waveform” view would decode samples and draw positive/negative amplitude — separate from Path B.

### Audio file temporal resolution

M4A/AAC typically encodes ~44,100 samples/second. You **hear** full resolution; the UI **shows** 20 envelope bars/second.

---

## 9. Database and files

### SQLite schema

**recordings**

| Column | Type | Purpose |
|--------|------|---------|
| id | INTEGER PK | |
| filename | TEXT | e.g. recording.m4a |
| session_name | TEXT | User label |
| created_at | INTEGER | Unix ms |
| duration_ms | INTEGER | Session length |
| waveform_data | TEXT | WaveformPayload JSON (may include `source`) |

**tags**

| Column | Type | Purpose |
|--------|------|---------|
| id | INTEGER PK | |
| recording_id | FK | |
| timestamp_ms | INTEGER | Moment in recording |
| label | TEXT | e.g. hungry, tired |

### File storage

- iOS: app Caches directory (typical for recordings)
- Format: M4A (AAC)
- Path resolution: `src/waveform/audioPath.ts` (`Caches` / `Documents` / absolute URI)
- Export today: Share sheet (single or Export All with thin tags JSON) — **not** yet the planned session-kit layout

---

## 10. Screen map

| Screen | File | Key behaviors |
|--------|------|---------------|
| Record | RecordScreen.tsx | Record, Tag Now (press-time stamp), 3s envelope, Save + Path B |
| List | RecordingListScreen.tsx | Search, sort, Export All |
| Playback | PlaybackScreen.tsx | Play, Path B upgrade, 8s/full waveform, seek/scrub, tag, export |

### Shared components

- `Waveform.tsx` — bars, cursor, tag markers, optional seek overlay
- `CircularRecordButton` / `CircularPlayButton`
- `RecordingLayout` — common chrome

---

## 11. Comparison to Voice Memos / phone call bar

| Feature | Voice Memos / system UI | BabyTalk (current) |
|---------|-------------------------|---------------------|
| Render thread | Native UI / 60fps | React state ~20Hz |
| Live input | High-frequency native metering | 50ms bridge callbacks |
| Playback waveform | From decoded file | Path B file peaks (envelope) when available |
| Seek on waveform | Yes | Yes (full + frozen rolling window) |
| Y scale | Fixed / slow AGC | Fixed per session |

---

## 12. Roadmap

### Done

- [x] 50ms sparse capture with raw dB
- [x] peak + avg metering from native
- [x] 3s record tail, 8s playback window
- [x] Envelope UX polish (narrower tags, tighter Y / noise floor, gapped bars)
- [x] Tag Now stamps time at button press
- [x] WaveformPayload v2 metadata (+ `source`)
- [x] Path B: `extractWaveformPeaks` on save + playback upgrade (iOS)
- [x] Playback seek / scrub (full view + frozen 8s rolling window)

### Next (product phases — see planning notes)

| Phase | Intent | User-facing outcome |
|-------|--------|---------------------|
| **1 Session kit** | Versioned export folder (audio + waveform + tags + manifest) | Complete packages for Mac / future ML |
| **2 USB in/out** | Finder File Sharing backup + import annotations | Copy kits to Mac; bring suggestions back |
| **3 Mac scripts** | Offline validate / propose moments / later classify | Suggestions written into the kit |
| **4 Review UI** | Fitbit-style confirm / edit / dismiss | Trustworthy tags without re-listening from zero |
| **5 Timeline** | Search + acquisition timeline + confidence | Cross-session insight |

**Explicitly deferred:** Google Drive sync, cloud training as default, PCM oscilloscope view, on-device heavy ML (may come later as an optimization of Phase 3).

### Optional polish

- [ ] Android Path B (`extractWaveformPeaks` real implementation)
- [ ] averagePower overlay on peak bars
- [ ] Device matrix checklist (simulator / physical iOS)

---

## 13. Build and deploy notes

### iOS device requirements (July 2026)

- **iOS 26.x device** requires **macOS Tahoe** and **Xcode 26.x** (Developer Disk Image mismatch on older Xcode)
- **Simulator** works for UI flow; mic and Path B behavior are limited vs device
- Native module changes (including `extractWaveformPeaks`) require **Clean Build** in Xcode, not Metro reload alone
- JS-only changes (Waveform seek, screens) → Metro reload (`r`) is enough

### Known warnings

- `react-native-sqlite-storage` invalid `dependency.platforms.ios.project` — CLI metadata warning; usually harmless if DB works
- `fmt` pod may need C++17 on Xcode 26 — forced in `ios/Podfile` for that pod only

---

## 14. File reference

```
src/
├── waveform/
│   ├── config.ts      # Constants (windows, dB floors, tag width, gaps)
│   ├── types.ts       # WaveformSample, WaveformPayload, WaveformSource
│   ├── storage.ts     # upsert, densify, parse, serialize
│   ├── scale.ts       # dbToHeight01, RECORD_DB_RANGE, computePlaybackDbRange
│   └── audioPath.ts   # resolveAudioUri for Path B / playback
├── components/
│   └── Waveform.tsx   # Envelope renderer + optional seek
├── screens/
│   ├── RecordScreen.tsx
│   ├── PlaybackScreen.tsx
│   └── RecordingListScreen.tsx
└── db.ts              # SQLite API (incl. updateWaveformData)

react-native-audio-recorder-player/   # In-repo copy (screen import path)
../react-native-audio-recorder-player/ # npm/CocoaPods linked copy — keep in sync
```

---

## 15. Glossary

| Term | Definition |
|------|------------|
| **Callback** | Native → JS event on a timer (record-back / playback) |
| **Bridge** | RN message channel between JS and native |
| **DDI** | Developer Disk Image — Xcode uses to debug on physical device |
| **Densify** | Convert sparse samples to fixed 50ms bar array for drawing |
| **Envelope** | Loudness-over-time bars (VU), not PCM oscilloscope |
| **Metering** | Loudness in dB from recorder, not PCM samples |
| **Path A** | Live metering samples while recording |
| **Path B** | Peaks/RMS extracted from the saved `.m4a` |
| **Scrub** | Drag on waveform to preview/seek time; rolling window frozen during gesture |
| **Session kit** | Planned complete export folder (not yet shipped) |
| **Sparse storage** | List of `{tMs, dB}` only where samples exist |
| **VU trace** | Path A envelope from live metering |

---

*Updated July 2026 for Path B, envelope polish, Tag Now stamp timing, and playback seek. Next doc refresh when session-kit export or USB import lands.*
