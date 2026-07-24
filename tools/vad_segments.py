"""Energy-based speech-segment VAD for BabyTalk session kits (vad_v0).

Detects continuous speech-like regions (not short onsets), merges gaps that
are likely the same utterance, drops brief noise blips, and writes provisional
annotations compatible with Review Browser / propose_candidates.py.

Requires: numpy, soundfile
  pip install numpy soundfile

Usage:
  python3 tools/vad_segments.py /path/to/kit_or_library
  python3 tools/vad_segments.py /path/to/kit --merge-gap-ms 400 --min-ms 300
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install deps: pip install numpy soundfile\n" + str(e)
    ) from e

# Defaults tuned for baby/caregiver utterances reviewed in the Mac UI.
FRAME_MS = 30
HOP_MS = 10
# Speech if frame is this many dB above a robust noise floor.
SPEECH_DELTA_DB = 6.0
# Pad each run so soft onsets/offsets aren't clipped.
PAD_MS = 80
# Merge segments separated by ≤ this gap (same utterance).
MERGE_GAP_MS = 400
# Drop segments shorter than this (likely clicks/noise).
MIN_DUR_MS = 300
# Soft cap — very long "speech" is usually ambient; split later if needed.
MAX_DUR_MS = 30_000
SOURCE = "vad_v0"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def frame_rms_db(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    times: list[float] = []
    dbs: list[float] = []
    for start in range(0, max(1, n - frame + 1), hop):
        chunk = audio[start : start + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-12)
        times.append(start * 1000.0 / sr)
        dbs.append(20.0 * math.log10(rms))
    return np.asarray(times), np.asarray(dbs)


def noise_floor_db(dbs: np.ndarray) -> float:
    """Robust floor: lower quartile of frames (silence-heavy sessions)."""
    if len(dbs) == 0:
        return -80.0
    return float(np.percentile(dbs, 25))


def detect_speech_segments(
    times: np.ndarray,
    dbs: np.ndarray,
    *,
    speech_delta_db: float = SPEECH_DELTA_DB,
    pad_ms: float = PAD_MS,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_dur_ms: float = MIN_DUR_MS,
    max_dur_ms: float = MAX_DUR_MS,
) -> list[tuple[float, float, float]]:
    """Return list of (start_ms, end_ms, score) after merge/filter."""
    if len(dbs) == 0:
        return []

    floor = noise_floor_db(dbs)
    thresh = floor + speech_delta_db
    active = dbs >= thresh

    # Collect contiguous active runs.
    raw: list[tuple[float, float, float]] = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        peak = float(dbs[i])
        while j < n and active[j]:
            peak = max(peak, float(dbs[j]))
            j += 1
        start_ms = float(times[i]) - pad_ms
        end_ms = float(times[min(j - 1, n - 1)]) + FRAME_MS + pad_ms
        score = float(min(1.0, max(0.0, (peak - thresh) / 18.0)))
        raw.append((start_ms, end_ms, score))
        i = max(j, i + 1)

    if not raw:
        return []

    # Merge close gaps.
    merged: list[tuple[float, float, float]] = [raw[0]]
    for start_ms, end_ms, score in raw[1:]:
        ps, pe, pscore = merged[-1]
        if start_ms - pe <= merge_gap_ms:
            merged[-1] = (ps, end_ms, max(pscore, score))
        else:
            merged.append((start_ms, end_ms, score))

    # Duration filter + clamp to non-negative.
    out: list[tuple[float, float, float]] = []
    for start_ms, end_ms, score in merged:
        start_ms = max(0.0, start_ms)
        end_ms = max(start_ms, end_ms)
        dur = end_ms - start_ms
        if dur < min_dur_ms:
            continue
        if dur > max_dur_ms:
            # Keep first max_dur_ms — reviewer can extend; avoids giant ambient bands.
            end_ms = start_ms + max_dur_ms
        out.append((start_ms, end_ms, score))
    return out


def segments_to_annotations(
    segments: list[tuple[float, float, float]],
) -> list[dict]:
    anns: list[dict] = []
    for start_ms, end_ms, score in segments:
        s = int(round(start_ms))
        e = int(round(end_ms))
        anns.append(
            {
                "uuid": str(uuid.uuid4()),
                "label": "",
                "startMs": s,
                "endMs": e,
                "tMs": s,
                "source": SOURCE,
                "status": "provisional",
                "score": round(score, 3),
            }
        )
    return anns


def merge_with_existing(
    existing: list,
    new_anns: list[dict],
    *,
    replace_provisional: bool = True,
) -> list:
    """Keep confirmed/dismissed (and optionally other provisional) intact."""
    kept: list = []
    for a in existing:
        status = a.get("status")
        source = a.get("source")
        if status in ("confirmed", "dismissed"):
            kept.append(a)
            continue
        if not replace_provisional:
            kept.append(a)
            continue
        # Drop prior provisional VAD (and legacy empty-label ml_v0 bursts if
        # re-running speech finder — user can still keep labeled provisional).
        if source == SOURCE:
            continue
        if source == "ml_v0" and not (a.get("label") or "").strip():
            continue
        kept.append(a)
    return kept + new_anns


def run_vad_on_audio(
    audio_path: Path,
    *,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_dur_ms: float = MIN_DUR_MS,
    speech_delta_db: float = SPEECH_DELTA_DB,
) -> list[dict]:
    audio, sr = sf.read(str(audio_path), always_2d=False)
    times, dbs = frame_rms_db(np.asarray(audio, dtype=np.float64), int(sr))
    segs = detect_speech_segments(
        times,
        dbs,
        speech_delta_db=speech_delta_db,
        merge_gap_ms=merge_gap_ms,
        min_dur_ms=min_dur_ms,
    )
    return segments_to_annotations(segs)


def process_kit(
    kit: Path,
    *,
    merge_gap_ms: float = MERGE_GAP_MS,
    min_dur_ms: float = MIN_DUR_MS,
    speech_delta_db: float = SPEECH_DELTA_DB,
    write: bool = True,
) -> dict:
    manifest_path = kit / "manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "error": "no manifest", "kit": kit.name}

    manifest = load_json(manifest_path)
    audio_name = manifest.get("audioFile", "audio.wav")
    audio_path = kit / audio_name
    if not audio_path.exists():
        return {"ok": False, "error": f"missing {audio_name}", "kit": kit.name}

    new_anns = run_vad_on_audio(
        audio_path,
        merge_gap_ms=merge_gap_ms,
        min_dur_ms=min_dur_ms,
        speech_delta_db=speech_delta_db,
    )
    existing = []
    ann_path = kit / "annotations.json"
    if ann_path.exists():
        ap = load_json(ann_path)
        existing = ap.get("annotations", ap if isinstance(ap, list) else [])

    merged = merge_with_existing(existing, new_anns, replace_provisional=True)
    if write:
        write_json(ann_path, {"annotations": merged})

    return {
        "ok": True,
        "kit": kit.name,
        "added": len(new_anns),
        "total": len(merged),
        "annotations": new_anns,
        "params": {
            "mergeGapMs": merge_gap_ms,
            "minDurMs": min_dur_ms,
            "speechDeltaDb": speech_delta_db,
            "source": SOURCE,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", help="Kit folder or library root")
    p.add_argument("--merge-gap-ms", type=float, default=MERGE_GAP_MS)
    p.add_argument("--min-ms", type=float, default=MIN_DUR_MS)
    p.add_argument("--speech-delta-db", type=float, default=SPEECH_DELTA_DB)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing annotations.json",
    )
    args = p.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Path not found: {root}")
        return 1

    if (root / "manifest.json").exists():
        kits = [root]
    else:
        kits = sorted(
            [x for x in root.iterdir() if x.is_dir() and (x / "manifest.json").exists()]
        )

    total = 0
    for kit in kits:
        result = process_kit(
            kit,
            merge_gap_ms=args.merge_gap_ms,
            min_dur_ms=args.min_ms,
            speech_delta_db=args.speech_delta_db,
            write=not args.dry_run,
        )
        if not result.get("ok"):
            print(f"skip {kit.name}: {result.get('error')}")
            continue
        total += int(result["added"])
        verb = "would write" if args.dry_run else "wrote"
        print(f"{kit.name}: {verb} {result['added']} speech segments")
    print(f"Done. {total} segments across {len(kits)} kit(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
