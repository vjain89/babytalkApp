# BabyTalk Mac tools

## One-time setup

```bash
cd /path/to/babytalkApp-1
python3 -m venv tools/.venv
tools/.venv/bin/pip install -r tools/requirements.txt
```

Plug in the iPhone, unlock it, and tap **Trust** if asked.

## Review studio

Default library (no path needed):

```bash
python3 tools/review_server.py
open http://127.0.0.1:8765
```

Kits live in `~/Documents/BabyTalk/Library/`. On first launch, kits are seeded from `~/Documents/BabyTalk/Backups/` if present.

Optional override:

```bash
python3 tools/review_server.py /path/to/some/kit-or-batch
```

### Tagging

1. Select a session → play → **drag** on the waveform → pick a **category** → optional **speaker** → **Add tag**
2. Categories: `verbal vocalization` · `non-verbal vocalization` · `non-vocal vegetative sound`
3. For **verbal vocalization**, enter **word** (required — intended/target word) and optional **phonetic** (how it sounded, casual orthography OK)
4. For **non-verbal vocalization**, optional **phonetic** (+ optional note); for **non-vocal vegetative sound**, optional free-form **note** only
5. Tags write live into that kit’s `tags.json` with `category`, optional `speaker`, `word`/`phonetic`/`note` as appropriate, and a composed `label` like `verbal vocalization · Baby · Lorenzo` (phone-compatible; phonetic is not folded into `label`)
6. Click **Sync with iPhone** (USB) to pull new kits and push tags

Speaker chips: Baby / Parent / Other, or type a custom name.

### USB sync (CLI)

```bash
tools/.venv/bin/python tools/iphone_sync.py status
tools/.venv/bin/python tools/iphone_sync.py sync   # pull kits + push tags.json
tools/.venv/bin/python tools/iphone_sync.py pull
tools/.venv/bin/python tools/iphone_sync.py push
```

**Pull:** copies new/changed kits from phone `Documents/Backups/` → Mac `Library/`  
**Push:** writes `manifest.json` + `tags.json` into phone `Documents/Import/<kit>/` (tags only)

Open **BabyTalk** on the phone after a push — it auto-imports Import folders when the app becomes active.

Bundle id default: `org.reactjs.native.example.babytalkApp`  
Override: `--bundle-id your.bundle.id`

## ML candidates: VAD → speaker diarization → candidates

Clicking **Find speech segments** (or running `vad_segments.py`) runs three stages:

| Stage | Where | What it does |
| --- | --- | --- |
| 1 · VAD + speech gate | `vad_segments.py`, `speechlike.py` | energy detection of louder-than-the-room regions, then an absolute "does this sound like a voice?" score that drops taps, doors, thumps and running water |
| 2 · Diarization | `diarize.py` | speaker embeddings + clustering across the session; cuts each region where the speaker changes |
| 3 · Refine + re-gate | `vad_segments.py` | pause-splits any same-speaker span still over 4s, applies the duration cap, re-scores each final candidate through the speech gate, writes `annotations.json` |

```bash
tools/.venv/bin/pip install -r tools/requirements.txt
python3 tools/vad_segments.py ~/Documents/BabyTalk/Library
# or per kit:
python3 tools/vad_segments.py ~/Documents/BabyTalk/Library/<kit-folder>
tools/.venv/bin/python tools/vad_segments.py --list-backends   # what's installed
```

Defaults: merge gaps ≤ **200 ms**, drop segments **&lt; 300 ms**, source `vad_v0` → `annotations.json` as provisional candidates.

Each run gets a flat **80ms** pad on both edges so soft onsets/offsets aren't clipped by the threshold crossing. On top of that, edges are extended through neighboring frames that clear a second, lower threshold (half the speech delta, capped at **120ms**) before the flat pad is added — a gradual attack/decay often sits above the noise floor for a while before it's "confidently" loud, and a single fixed threshold would otherwise cut into it. This is still level-based (not a derivative/onset detector), just two thresholds instead of one, and the reach is clamped so it can never cross into a neighboring run.

### Stage 1 — VAD + the speech gate

The energy VAD only answers *"is something louder than the room here?"*, which is equally true of a tap running and a door closing. Every region it finds is therefore scored by `speechlike.py` on whether it sounds like it came out of a **vocal tract**:

| Feature | Speech | Junk |
| --- | --- | --- |
| `voicing_peak_med` / `voiced_fraction` | periodic — clear autocorrelation peak at 70–600 Hz f0 | impacts and water are aperiodic |
| `speech_band_ratio` | most energy in 300–3400 Hz | thumps and hiss sit outside it |
| `low_band_ratio` | little sub-250 Hz | rumble dominates handling noise, footsteps, doors |

Score = `0.40·speech_band + 0.25·voicing_peak + 0.15·voiced_fraction + 0.20·(1 − low_band)`, and it is applied twice: leniently (< **0.42**) on whole regions before diarization, then per candidate after splitting — below **0.55** dropped, **0.55–0.68** kept but flagged `possible_non_speech` with a reason and down-ranked. Every candidate carries its `speechScore`, shown as a badge in the UI. `--keep-noise` / `rejectNonSpeech: false` turns dropping off and flags instead.

**Calibration.** Thresholds come from 457 spans this project's reviewer had already confirmed (206) or dismissed (251) across their own sessions, not from eyeballed constants: AUC **0.81** overall and stable per kit (0.77 / 0.83 / 0.88). Two things were tried and rejected — fitting logistic weights (best leave-one-kit-out AUC 0.79, i.e. *worse* than the fixed weights above, so the interpretable ones ship), and a 2–10 Hz syllabic-modulation feature (textbook speech cue, AUC 0.46 here — these segments are single short utterances, not continuous speech).

The older **relative** impulsive check (flatness + zero-crossing + spikiness z-scored against the same recording's other short segments) still runs as a second opinion on ≤700ms bursts. It stays secondary because a z-score can only ever flag a fixed slice of each session as "most noise-like" — on real kits it removed just 2 of 131 and 5 of 300 candidates, which is why noisy sessions still felt noisy.

This is signal processing, not a trained classifier: expect it to keep some junk and to occasionally drop a very quiet, whispered or heavily-clipped utterance.

Stage 1 says nothing about *who* is talking, so a 5–10s region can hold several turns. That's stage 2's job.

### Stage 2 — speaker diarization

Sliding 1.5s windows inside each VAD region are turned into **speaker embeddings**, clustered across the whole recording with cosine agglomerative clustering, and each region is cut wherever the winning cluster changes. Every candidate carries the cluster id as `speakerCluster` (`SPEAKER_00`, `SPEAKER_01`, …) and shows it as a badge in the UI.

`speakerCluster` is deliberately **not** written into the reviewer-facing `speaker` field: clusters group turns, they don't know which one is the baby, so Confirm still asks you.

Backends, picked automatically in this order:

- **`ecapa` (recommended)** — SpeechBrain ECAPA-TDNN embeddings. Needs `torch torchaudio speechbrain` (already in `requirements.txt`, ~1 GB of wheels). First run downloads the ~80 MB model to `~/.cache/babytalk/spkrec-ecapa-voxceleb` and takes an extra ~20s; after that a 12-minute session diarizes in ~20s. **No HuggingFace token needed.**
- **`melstats` (fallback)** — pure numpy/sklearn MFCC mean+std features with per-recording normalization. No downloads, always available, runs in ~3s, but noticeably weaker: on the test session it split one child into three "speakers". Fine for adult-vs-baby, unreliable for adult-vs-adult.
- **`pyannote` (opt-in)** — the full `pyannote.audio` speaker-diarization-3.1 pipeline. Best quality and some overlap handling, but you must `pip install pyannote.audio`, accept the model terms on huggingface.co, and export a token:

```bash
export BABYTALK_HF_TOKEN=hf_xxx
tools/.venv/bin/python tools/vad_segments.py <kit> --diarization pyannote
```

It is never selected automatically (slow on CPU, needs setup) — ask for it by name.

Useful flags:

```bash
--diarization auto|ecapa|melstats|pyannote|none   # backend choice
--no-diarization                                  # stage 1 + pause splitting only
--num-speakers 2                                  # force the speaker count
--speaker-distance 0.5                            # clustering cut (lower = more speakers)
--list-backends                                   # availability report
--dry-run                                         # counts only, don't write
```

**Graceful degradation:** if no backend is usable, or the model download fails, the pipeline still returns VAD-only candidates and reports why stage 2 was skipped — in the CLI output, and in the hint text next to the button. The Review Server never fails the request over a missing optional model.

### Stage 3 — refinement

Same-speaker spans still longer than **4s** are re-split at their deepest internal relative energy dip (one person, several sentences — a pause is a reasonable utterance boundary). An absolute 15s hard cap is the last resort if no split point is found, flagged `hard_capped`.

Each resulting candidate then goes back through the **speech gate** one more time. Stage 1 screened whole regions, which often mix a sentence with the clatter right after it; only now is each piece a single turn that can be judged on its own, so this is the pass that does most of the filtering. Measured on two of the reviewer's kits (candidates → junk they had previously dismissed that got re-proposed):

| Kit | Before | After |
| --- | --- | --- |
| `…_St_78` (5.6 min) | 131 candidates, 98% of known junk re-proposed | 59 candidates, 35% of known junk re-proposed, 85% of known speech still found |
| `…_St_75` (11.5 min) | 300 candidates, 98% of known junk re-proposed | 172 candidates, 52% of known junk re-proposed, 86% of known speech still found |

Finally, any candidate that overlaps an existing entry in `tags.json` (± a **75ms** margin) is dropped — re-running VAD never re-proposes something already reviewed. Point tags (no `endMs`) are treated as covering a short **500ms** window from their timestamp for this check. The number suppressed is reported as `tagOverlapSuppressed` in the API response and shown in the UI hint ("N skipped (already tagged)").

Re-run **Find speech segments** (or `vad_segments.py`) after changing any of this; it only replaces still-provisional VAD candidates, so confirmed/dismissed decisions **and existing tags** are preserved and skips spans already tagged.

**Known limits:** the speech gate still lets through roughly a third to a half of what a reviewer would dismiss, and costs ~14% of real speech at the current threshold — mostly quiet or breathy segments; overlapping speech is assigned to a single speaker; a toddler imitating a caregiver's pitch can land in the wrong cluster; clusters are per-session, so `SPEAKER_00` in one kit is unrelated to `SPEAKER_00` in another. `speakerCluster` groups turns by voice — it is **not** speaker identification and never decides who is the baby.

Diarization is a separate job from the **Clustering tab** — that groups *similar sounds/words* regardless of who said them, and is unaffected by any of this.

### Local Whisper suggestions (optional)

On each ML candidate or tag, **Suggest** runs offline Whisper (`faster-whisper`) on that time slice and stores the result under `asr` (never overwrites your `word` / `phonetic`). **Copy→word** pastes the model text into the Word field for you to edit before Save/Confirm.

```bash
tools/.venv/bin/pip install faster-whisper   # first time; downloads model weights on first Suggest
# optional: BABYTALK_WHISPER_MODEL=small tools/.venv/bin/python tools/review_server.py
```

CLI:

```bash
tools/.venv/bin/python tools/asr_suggest.py ~/Documents/BabyTalk/Library/<kit> --uuid <uuid>
tools/.venv/bin/python tools/asr_suggest.py ~/Documents/BabyTalk/Library/<kit> --start-ms 1200 --end-ms 2100
```

Expect weak results on toddler speech and Swiss German — useful for comparison, not as truth.

### Acoustic clustering (Review Browser tab)

Groups similar spans in one kit (confirmed tags + non-dismissed VAD). Labels live on the **cluster** only; members stay linked for training and future concepts.

```bash
tools/.venv/bin/pip install scikit-learn
tools/.venv/bin/python tools/cluster_sounds.py ~/Documents/BabyTalk/Library/<kit>
# or in the UI: Clustering tab → Run clustering
```

Writes `clusters.json` (+ optional `cluster_embeddings.npz` cache). Confidence shows tightness / size / separation / outliers. Exclude members; optionally promote annotation members to tags.

Legacy short-burst onset detector:

```bash
python3 tools/propose_candidates.py ~/Documents/BabyTalk/Library
python3 tools/validate_export.py ~/Documents/BabyTalk/Library
```
