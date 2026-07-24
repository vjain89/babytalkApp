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

## Optional ML candidates

Energy-based **speech segments** (recommended for Review Browser):

```bash
tools/.venv/bin/pip install numpy soundfile   # if not already in the venv
python3 tools/vad_segments.py ~/Documents/BabyTalk/Library
# or per kit:
python3 tools/vad_segments.py ~/Documents/BabyTalk/Library/<kit-folder>
```

Defaults: merge gaps ≤ **400 ms**, drop segments **&lt; 300 ms**, source `vad_v0` → `annotations.json` as provisional candidates.

In the Review UI, open a session and click **Find speech segments** (same pipeline via `POST /api/vad/run`). On confirm, assign a category (+ optional speaker; verbal: word + optional phonetic; non-verbal: optional phonetic); dismiss works like other ML candidates.

VAD is energy-based only — it detects speech-like activity, not who spoke (no diarization).

Legacy short-burst onset detector:

```bash
python3 tools/propose_candidates.py ~/Documents/BabyTalk/Library
python3 tools/validate_export.py ~/Documents/BabyTalk/Library
```
