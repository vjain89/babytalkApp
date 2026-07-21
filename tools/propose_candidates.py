"""Signal-based candidate proposals for BabyTalk session kits (ml_v0).

Reads audio.wav from a kit (or a folder of kits), detects loudness onsets /
vocalization-like bursts, and writes annotations.json with provisional ML tags.

Requires: numpy, soundfile (pip install numpy soundfile)

Usage:
  python3 tools/propose_candidates.py /path/to/kit_or_backup_folder
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Install deps: pip install numpy soundfile")
    raise SystemExit(1)


FRAME_MS = 20
HOP_MS = 10
# Relative to session median loudness
ONSET_DELTA_DB = 8.0
MIN_EVENT_MS = 80
MAX_EVENT_MS = 2500
MERGE_GAP_MS = 120


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def frame_rms_db(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    times = []
    dbs = []
    for start in range(0, max(1, n - frame), hop):
        chunk = audio[start : start + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-12)
        db = 20.0 * math.log10(rms)
        times.append(start * 1000.0 / sr)
        dbs.append(db)
    return np.asarray(times), np.asarray(dbs)


def propose(times: np.ndarray, dbs: np.ndarray) -> list[dict]:
    if len(dbs) == 0:
        return []
    floor = float(np.median(dbs))
    thresh = floor + ONSET_DELTA_DB
    active = dbs >= thresh

    events: list[tuple[float, float, float]] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        peak = dbs[i]
        while j < len(active) and active[j]:
            peak = max(peak, dbs[j])
            j += 1
        start_ms = float(times[i])
        end_ms = float(times[min(j, len(times) - 1)])
        dur = end_ms - start_ms
        if MIN_EVENT_MS <= dur <= MAX_EVENT_MS:
            score = float(min(1.0, max(0.0, (peak - thresh) / 20.0)))
            events.append((start_ms, end_ms, score))
        i = max(j, i + 1)

    # Merge close events
    merged: list[tuple[float, float, float]] = []
    for ev in events:
        if not merged:
            merged.append(ev)
            continue
        ps, pe, pscore = merged[-1]
        if ev[0] - pe <= MERGE_GAP_MS:
            merged[-1] = (ps, ev[1], max(pscore, ev[2]))
        else:
            merged.append(ev)

    out = []
    for start_ms, end_ms, score in merged:
        out.append(
            {
                "uuid": str(uuid.uuid4()),
                "label": "",
                "startMs": int(round(start_ms)),
                "endMs": int(round(end_ms)),
                "tMs": int(round(start_ms)),
                "source": "ml_v0",
                "status": "provisional",
                "score": round(score, 3),
            }
        )
    return out


def process_kit(kit: Path) -> int:
    manifest_path = kit / "manifest.json"
    if not manifest_path.exists():
        print(f"skip {kit.name}: no manifest")
        return 0
    manifest = load_json(manifest_path)
    audio_name = manifest.get("audioFile", "audio.wav")
    audio_path = kit / audio_name
    if not audio_path.exists():
        print(f"skip {kit.name}: missing {audio_name}")
        return 0

    audio, sr = sf.read(str(audio_path), always_2d=False)
    times, dbs = frame_rms_db(np.asarray(audio, dtype=np.float64), int(sr))
    anns = propose(times, dbs)
    write_json(kit / "annotations.json", {"annotations": anns})
    print(f"{kit.name}: wrote {len(anns)} provisional candidates")
    return len(anns)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = Path(argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"Path not found: {root}")
        return 1

    kits: list[Path]
    if (root / "manifest.json").exists():
        kits = [root]
    else:
        kits = sorted([p for p in root.iterdir() if p.is_dir()])

    total = 0
    for kit in kits:
        if (kit / "manifest.json").exists():
            total += process_kit(kit)
    print(f"Done. {total} candidates across kits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
