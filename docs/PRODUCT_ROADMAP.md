# BabyTalk — Product Roadmap & Scope

**Status:** planning, updated with scope decisions from the §12 review.
**Supersedes:** the earlier waveform-rebuild doc (its content lives here as Track A).
**How to read the flags:** `[confirm]` = I interpreted a decision, ratify it. `[needs input]` = still open. `[locked]` = decided.
**Read order:** §1 vision → §2 decisions ledger → §3 current state → §4 reframe → §5–§7 tracks → §8 foundations → §9 sequence → §10 tradeoffs → §11 risks → §12 scope decisions + open threads.

---

## 1. Product vision (as stated)

A **personal research instrument now**, built so it can grow into a tool for other parents. The thesis: help a parent *listen to their child better* and train a **per-child, personal model** that facilitates communication — not a universal baby translator. The parent records over time and tags words, actions, behaviors, and needs; a personal model emerges from that.

The iPhone is the **edge**: it detects features, the parent classifies them, and *features* (not raw audio) may be encrypted to a cloud — opt-in, for the broader study of infant speech, not a dump of every family's audio.

The emotional core is the **moments / timeline**: a family seeing how their child grows communicatively — a modality almost no one tracks but everyone loves to revisit. Searchable tags were the seed.

**Phasing:** Phase 1 = personal instrument until you have reps on the **record → annotate → analyze → publish** loop. Phase 2 = a small TestFlight cohort of families. Monetization is not a near-term focus.

---

## 2. Decisions ledger

| Decision | Status | Note |
|----------|--------|------|
| Capture **lossless ALAC** on device | `[locked]` | Resolves AAC-vs-WAV; lossless master |
| **44.1 kHz, 16-bit, mono** | `[confirm]` | Proposed params; ratify or adjust |
| ALAC master → **WAV at export** | `[confirm]` | Lossless→lossless; export owns the transcode |
| **iOS-only** near-term | `[locked]` | Android deferred |
| Backup via **Finder USB**, no Drive | `[locked]` | |
| Export kit = **folder, not zip** | `[locked]` | Behaves as a sync, not a one-shot download |
| Waveform playback = **Path B** (built) | `[locked]` | 50 ms envelope today |
| **A1** playback detailed waveform | `[locked]` | Playback-only for now |
| **A2** live high-res tap / real-time | `[locked]` deferred | Real-time is north-star; not near-term |
| Data model → **uuid / spans / source / status / free-form label** | `[confirm]` | `kind`/taxonomy column deferred (flat labels) |
| **Review workflow on the Mac** (research phase) | `[locked]` | Phone stays capture + quick-tag; parent-facing review is a later build |
| **Model** = classifier: label given spectrogram slice/segment; **Mac-side** training | `[locked]` | Pull features to device later |
| Near-term = **gather + annotate, no model** (cold start accepted) | `[locked]` | |
| Taxonomy = **single flat, free-form-then-cluster**; near-term unit = **words** | `[locked]` | Month-one target: word-frequency timeline |
| Single child / single caregiver now; multiple later, **labels merge** | `[locked]` | Schema already supports multi-source merge |
| Cloud = **opt-in** contribution to shared corpus / general model | `[locked]` direction | Features = metadata + embeddings + spectrogram slices; see privacy caveat §12 |
| Full **deletion** supported | `[locked]` | Local delete is total; cloud contribution is the boundary (§12) |

---

## 3. Where the app is now

You shipped **Path B** (decode file → peak/RMS per 50 ms in Swift), **playback seek/scrub**, **Tag Now** press-time stamping, and a `source` field. Google Drive is out of the near-term plan.

- On screen it's still a **50 ms loudness envelope** — better data than metering, same coarseness, single-height bars.
- You already built the expensive half of the *playback* detailed waveform: **file → PCM decode in Swift.** A1 is "keep min **and** max per ~7 ms column + bipolar renderer."
- The **live** high-res waveform (A2) is unbuilt and now explicitly deferred (real-time = north-star).

---

## 4. The reframe that drives sequencing

**The recordings are the only irreplaceable, time-sensitive thing here.** Everything else can be rebuilt later against recordings you already have. So the top priority is the boring, irreversible stuff: **capture quality** (done — ALAC), **durable backup** (P1), **low-friction tagging**.

Second reframe: the payoff (word-frequency timeline, with confidence) is gated on **confirmed labels**. Labels come from a human confirming candidates. The engine is a **labeling flywheel**: capture → propose → confirm → labeled data → (later) train → better proposals. Start the flywheel cheaply; build the classifier last.

---

## 5. Two tracks

**Track A — Waveform fidelity.** Independent. A1 (cheap playback upgrade) is in; A2 (live tap) is deferred.

**Track B — Data → export → USB → ML → review → timeline.** The product thesis.

They meet at the **data model** (§8), the **waveform as review surface**, and **format** (handled: ALAC in, WAV out). Reminder: `waveform.json` in the kit is **not** for ML — Mac tools decode the audio directly.

---

## 6. Track A — waveform fidelity

- **A1 — detailed playback waveform (cheap, in).** Reuse the Path B decode; min/max per ~7 ms column; bipolar render with Skia; Y from the file's loudness. Playback-only. `[locked]`
- **A2 — detailed live waveform (deferred).** AVAudioEngine tap streaming min/max columns in real time. Only if a live requirement returns; real-time inference is a north-star, so this waits. `[locked]` deferred

---

## 7. Track B — data → export → USB → ML → review → timeline

P0 gives one schema everything consumes; P1 closes the phone↔Mac loop and is durability; P2 turns the archive into suggestions (signal-based first, classification much later); P3 (**on the Mac**) converts suggestions into trusted labels; P4 is the word-frequency timeline. Full detail in §9.

---

## 8. Shared foundations

### Data model migration `[confirm]`
Extend `tags` (or add an `annotations` table) with:

| Field | Why |
|-------|-----|
| `uuid` | Stable identity across export/import; enables merge-by-UUID (and multi-caregiver merge later) |
| `start_ms` (+ optional `end_ms`) | Word attempts are spans, not points |
| `source` (`user` / `ml` / `ml_confirmed`) | Never let re-import overwrite human labels |
| `status` (`provisional` / `confirmed`) | Separates suggestions from trusted labels |
| `label` (free-form text) | Flat, free-form for now; cluster later. `kind`/taxonomy column deferred |

Keep the int PK; add columns + a migration.

### Capture format `[locked]` (params `[confirm]`)
ALAC master on device, **44.1 kHz / 16-bit / mono** `[confirm]`, transcode to WAV at export `[confirm]`. Storage (~5–6× AAC) is absorbed by the USB archive: capture lossless, offload masters, keep recent sessions hot.

---

## 9. Recommended sequence (detailed)

Ordered "protect the asset → start the flywheel → build the payoff." Each item is independently shippable.

**0 — Switch capture to lossless ALAC.**
Edit the `AVAudioRecorder` settings in the Swift fork: Apple Lossless, 44100, 16-bit, mono. Confirm the container/extension and `audioPath` resolution still hold. Native change → Clean Build. Verify: plays back, `extractWaveformPeaks`/`AVAudioFile` still decodes it, file is meaningfully larger. *Why first:* every session from now is permanent, and this is the one capture decision you can't walk back. `[confirm 44.1/16/mono]`

**1 — Data model migration (tags → annotations).**
Add `uuid`, `start_ms`, optional `end_ms`, `source`, `status`, free-form `label`; keep int PK; write a migration. Update `db.ts` and the tag-writing paths to populate `uuid` + `source=user`. Taxonomy is flat/free-form, so no `kind` vocabulary needed yet. *Why here:* everything downstream keys off stable IDs and the user-vs-ml distinction.

**2 — P0 structured session-kit export (as a synced folder).**
A `src/export` module assembling a versioned **folder** per session (not a zip — so it reads as a sync, not a download): `manifest.json` (schemaVersion, ids, durationMs, createdAt, codec, sampleRate, **audio content hash**), the audio, `tags.json` (user), optional `annotations.json` (ml). Emit **WAV** in the kit, transcoded from the ALAC master. One module for single + Export All (batch = folder of sessions + `export_manifest.json`). *Why here:* one schema that USB, ML, review, and re-import all consume. `[needs input: hash-based re-link (recommended) vs recordingId alone]`

**3 — P1 Finder USB backup + import.**
File Sharing (`UIFileSharingEnabled` + Info.plist) so Documents appear in Finder over USB. "Prepare backup" writes dated folders under `Documents/Backups/<date>/`. You copy to the Mac, process, drop updated `annotations.json` (or whole kits) into `Documents/Inbox/`. "Import annotations" **merges by UUID** — user tags win, ml-provisional can update while provisional, confirmed edits sticky; track `importedAt`/`source`. The merge/conflict logic is the real work. *Why here:* closes the loop **and** is your durability story. `[locked: Finder, no Drive; folder sync]`

**4 — A1 playback detailed waveform.**
Change the decode to keep min **and** max per ~7 ms column; render bipolar with `@shopify/react-native-skia`; recompute Y from the file's loudness. Playback-only. *Why here:* cheap (decode exists) and it makes the Mac review surface better. Touches the Swift extract fn, `Waveform.tsx`, `scale.ts`. `[locked: playback-only]`

**5 — P2 dataset hygiene + signal-based candidates (Mac, Python).**
`validate_export.py` (schema checks + coverage: durations, tag density, gaps) and `propose_candidates.py` (signal-based: onsets, loudness peaks, voiced/unvoiced, vocalization-vs-silence — oriented toward word-like vocalization events) writing `annotations.json` with `tMs`/`endMs`, `score`, `status=provisional`, `source=ml_v0`. Zero labels needed; works today; starts the flywheel. You can also start computing **embeddings** here — they're the substrate for the free-form-then-cluster taxonomy. Deliberately **not** classification yet. *Why here:* turns the archive into reviewable suggestions without a model.

**6 — P3 review tool (on the Mac).**
A Mac-side tool (e.g. a small local web app) reading the synced package folder: a queue of unreviewed candidates rendered as markers on the waveform; confirm (label free-form), adjust time/span, or dismiss. Confirmed → `status=confirmed`. Writes back into the package, which syncs to the phone. *Why here:* this is the crank that turns cheap candidates into trustworthy, trainable labels — higher priority than the classifier it feeds. **Product-phase note:** TestFlight parents won't have this Mac setup, so a parent-facing on-device (or hosted) review UI is a separate, later build.

**7 — P4 word-frequency timeline (and/or P2 classification).**
The near-term payoff: a **cross-session timeline of word frequency over time** — when a word first appeared and how often it's used week over week — plus search by label/status. Show model confidence only where it exists. In parallel, once enough confirmed labels exist, train a small **per-child classifier** (label ← spectrogram slice/segment) **Mac-side**, writing `label`+`confidence` onto candidates. Research-phase timeline can live on the Mac (part of analyze/publish); the parent-facing timeline is a Phase-2 build. *Why last:* gated on accumulated confirmed labels. `[needs input: what counts as "enough"; eval via confidence bounds — later]`

**8 — A2 live high-res waveform tap (deferred).**
Only if a live-waveform requirement returns. Real-time inference is a north-star, so this waits. `[locked: deferred]`

Items 0–3 are "secure the asset" and shouldn't wait. Item 4 is a cheap win. Items 5–7 are the flywheel and the payoff.

---

## 10. Complexity vs. value

| Feature | Effort | Value | Notes |
|---------|--------|-------|-------|
| ALAC capture switch | Trivial | High | Irreversible; now made |
| Data model migration | Medium | High | Unblocks four phases |
| P0 export (folder + WAV) | Low–Med | High | Serialization + one transcode |
| P1 USB backup + import | Medium | High | Merge logic; also durability |
| A1 playback waveform | Low–Med | Medium | Decode already built |
| P2 hygiene + candidates + embeddings | Low–Med | High | Zero-label; starts flywheel |
| P3 Mac review tool | Medium | High | Converts candidates → labels |
| P4 word-frequency timeline | Med | High | The near-term payoff |
| P2 classification | High | Med→High | Gated on labels; hard for infant speech |
| A2 live PCM tap | High | Deferred | Only if live returns |

---

## 11. Risks

- **Dataset loss** (highest). One device + manual backup. Mitigation: P1 early; consider a second copy destination + encryption-at-rest (§12).
- **Privacy leakage via "features."** Spectrogram slices are partially invertible to audio of a minor. Mitigation: per-session opt-in; decide knowingly what leaves the device (§12).
- **ML over-expectation.** Adult-trained models degrade on babble. Mitigation: signal-based + your own labels; classification later. (You've accepted this — it's the thesis.)
- **Scope sprawl.** Mitigation: the flywheel ships value at each step.
- **Format regret — mitigated** by ALAC, if item 0 lands before the archive grows.

---

## 12. Scope decisions and remaining open threads

Resolved from your review; a few threads still open (flagged).

### Product boundary
**Decided:** Phase 1 personal instrument until you have reps on record→annotate→analyze→publish; Phase 2 small TestFlight cohort. One child / one caregiver now; multiple later with labels **merging** (union). Monetization not a focus.
**Open:** what **"publish"** means in your loop (findings? a dataset? a shareable timeline?) — it's a core verb but undefined. Multi-caregiver merge rule when two labels disagree on the same moment (union is fine for free-form; revisit if you adopt a fixed taxonomy).

### Privacy & consent
**Decided:** caregiver consent for now. Raw audio on phone + Mac backup. Full deletion supported.
**Open / flags:**
- **Encryption at rest** unspecified. iOS protects files when the phone is locked; the Mac copy is only as protected as the disk (FileVault). Decide if that's sufficient for a child's audio.
- **Deletion boundary.** Local deletion must cascade (audio + DB + Mac packages). But once features are contributed to a shared/general model, you generally can't un-train them — so the honest promise is "erase everything local + stop future contribution," not "erase from the general model." Make cloud contribution **per-session opt-in** so nothing leaves without a deliberate act.
- **Incidental voices** (visitors, other children) resurface as an issue at the TestFlight stage.

### Cloud architecture
**Decided:** opt-in contribution to a shared research corpus / eventual general model; parent gets better/alternative recs back. Features = metadata + embeddings + spectrogram slices.
**Explainer:** an *embedding* is a learned numeric fingerprint of a sound segment; similar sounds get nearby vectors. It's the substrate that makes **free-form-then-cluster** work — so the embedding work and the taxonomy plan are one effort.
**Flag:** spectrogram slices and some embeddings are **partially invertible** back toward audio. "Features, not raw" therefore doesn't fully deliver the privacy it implies. Decide knowingly: accept it behind explicit opt-in, or restrict outbound data to less-invertible representations. Design extraction with this in mind (cloud is deferred, so not urgent).

### Taxonomy / units
**Decided:** single **flat** label set, **free-form-then-cluster**, near-term unit = **words**. Month-one target: a timeline of word frequency over time.
**Open:** what counts as a "word" for an infant (proto-words / approximations) — but since labels are free-form you decide per-tag; formalize only at the clustering step.

### Per-child model
**Decided:** classifier predicting a **label given a spectrogram slice/segment**. **Mac-side** training; pull features to the device later. Near-term = gather + annotate, **no model** (cold start accepted — value comes from candidates + manual tagging).
**Open:** evaluation — later; some form of **confidence bounds**.

### Actionable / live insights
**Decided:** real-time inference is a **north-star**, not near-term. Progress insight = **frequency of words/needs/behaviors over time**, helping a parent see what's becoming communicative.

### ML realism (infant speech)
**Decided:** signal-based detection + your own labels near-term; recognition/classification later. Confirmed as the core thesis.

---

## 13. Explicitly deferred

Google Drive / OAuth; cloud training as default; on-device heavy ML; Android parity; parent-facing (on-device) review UI; the live PCM tap (A2); real-time inference.

---

*Near-term build: items 0–4 (secure the asset + the cheap waveform win), then 5–7 toward the concrete month-one target — a Mac-reviewed, free-form-tagged **timeline of word frequency over time**. Define "publish" and settle encryption-at-rest when convenient; neither blocks the near-term build.*
