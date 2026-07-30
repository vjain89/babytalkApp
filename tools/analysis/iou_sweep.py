#!/usr/bin/env python3
"""Sweep DJW resegmentation knobs for manual-tag IoU ≥ 0.5.

Runs VAD+ECAPA once per curated kit, then re-applies resegment+gate under each
parameter combo so the expensive diarization is not repeated.

North-star metric: IoU≥0.5 on *manual* tags (source != ml_confirmed).

    tools/.venv/bin/python tools/analysis/iou_sweep.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

import numpy as np
import soundfile as sf

import resegment as reseg_mod
from ml_delta import LIBRARY, compare, load, pct
from vad_segments import (
    MIN_DUR_MS,
    SPEECH_DELTA_DB,
    SPLIT_TARGET_MS,
    apply_speech_gate,
    detect_speech_regions,
    frame_rms_db,
    refine_long_spans,
    segments_to_annotations,
    split_by_speaker,
)

OUT = Path(__file__).resolve().parent / "out" / "iou_sweep.json"


def curated_kits(lib: Path) -> list[Path]:
    kits = []
    for kit in sorted(lib.iterdir()):
        if not (kit / "manifest.json").exists():
            continue
        tags = load(kit / "tags.json", "tags")
        anns = load(kit / "annotations.json", "annotations")
        if len(tags) < 4:
            continue
        if not any(a.get("status") == "dismissed" for a in anns):
            continue
        kits.append(kit)
    return kits


def parents_for_kit(kit: Path, *, diarization: str = "ecapa") -> tuple[list[dict], np.ndarray, int, list]:
    man = json.loads((kit / "manifest.json").read_text())
    audio_path = kit / man.get("audioFile", "audio.wav")
    audio, sr = sf.read(str(audio_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    sr = int(sr)
    times, dbs = frame_rms_db(audio, sr)
    regions, _ = detect_speech_regions(
        times,
        dbs,
        audio=audio,
        sr=sr,
        speech_delta_db=SPEECH_DELTA_DB,
        reject_non_speech=True,
    )
    if diarization and diarization not in ("none", "off"):
        pieces, _ = split_by_speaker(regions, audio, sr, backend=diarization)
    else:
        pieces = regions
    pieces, _ = refine_long_spans(
        pieces, times, dbs, split_target_ms=SPLIT_TARGET_MS, min_dur_ms=MIN_DUR_MS
    )
    tags = load(kit / "tags.json", "tags")
    return pieces, audio, sr, tags


def finish_candidates(
    parents: list[dict],
    audio: np.ndarray,
    sr: int,
    *,
    target_ms: float,
    min_part_ms: float,
    word_sep_ms: float,
    word_dip_db: float,
) -> list[dict]:
    pieces, _ = reseg_mod.resegment_pieces(
        parents,
        audio,
        sr,
        enabled=True,
        target_ms=target_ms,
        min_part_ms=min_part_ms,
        word_split_min_sep_ms=word_sep_ms,
        word_split_min_dip_db=word_dip_db,
    )
    pieces, _ = apply_speech_gate(pieces, audio, sr, reject=True)
    return segments_to_annotations(pieces)


def score_manual(tags: list, cands: list[dict]) -> dict:
    cmp = compare(tags, cands, label="sweep")
    manual = (cmp.get("bySource") or {}).get("manual") or {}
    return {
        "nTags": cmp["nTags"],
        "nCands": cmp["nCands"],
        "anyOverlapPct": cmp["coverage"]["anyOverlapPct"],
        "iou50Pct": cmp["coverage"]["iou50Pct"],
        "manualN": manual.get("nTags", 0),
        "manualIou50Pct": manual.get("iou50Pct", 0.0),
        "manualMedianIou": (manual.get("iou") or {}).get("median"),
        "manualDurRatio": (manual.get("durRatio") or {}).get("median"),
        "medianCandDur": (cmp.get("boundary") or {}).get("candDur", {}).get("median"),
        "candTooLong2xPct": (cmp.get("boundary") or {}).get("candTooLong2xPct"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(LIBRARY))
    ap.add_argument("--diarization", default="ecapa")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    lib = Path(args.library).expanduser()
    kits = curated_kits(lib)
    print(f"curated kits: {len(kits)}", flush=True)

    # Cache parents once.
    cached = []
    for kit in kits:
        print(f"[parents] {kit.name}", flush=True)
        t0 = time.time()
        parents, audio, sr, tags = parents_for_kit(kit, diarization=args.diarization)
        print(
            f"  parents={len(parents)} tags={len(tags)} in {time.time()-t0:.1f}s",
            flush=True,
        )
        cached.append((kit, parents, audio, sr, tags))

    grid = {
        "target_ms": [1600.0, 1200.0, 1000.0, 800.0],
        "min_part_ms": [420.0, 350.0, 300.0],
        "word_sep_ms": [300.0, 220.0, 160.0],
        "word_dip_db": [4.0, 3.0, 2.5],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    # Always include baseline first.
    baseline = (1600.0, 420.0, 300.0, 4.0)
    if baseline in combos:
        combos.remove(baseline)
    combos = [baseline] + combos

    rows = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        print(f"[{i+1}/{len(combos)}] {params}", flush=True)
        kit_scores = []
        total_manual = 0
        total_manual_hit = 0
        total_cands = 0
        any_hits = 0
        any_n = 0
        for kit, parents, audio, sr, tags in cached:
            cands = finish_candidates(parents, audio, sr, **params)
            sc = score_manual(tags, cands)
            kit_scores.append({"kit": kit.name, **sc})
            total_manual += sc["manualN"]
            total_manual_hit += int(round(sc["manualIou50Pct"] * sc["manualN"] / 100.0))
            total_cands += sc["nCands"]
            any_hits += int(round(sc["anyOverlapPct"] * sc["nTags"] / 100.0))
            any_n += sc["nTags"]
        row = {
            "params": params,
            "pooledManualIou50Pct": pct(total_manual_hit, total_manual),
            "pooledAnyOverlapPct": pct(any_hits, any_n),
            "nManual": total_manual,
            "nCands": total_cands,
            "kits": kit_scores,
            "isBaseline": vals == baseline,
        }
        rows.append(row)
        print(
            f"  manualIoU50={row['pooledManualIou50Pct']}% any={row['pooledAnyOverlapPct']}% cands={total_cands}",
            flush=True,
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["pooledManualIou50Pct"],
            r["pooledAnyOverlapPct"],
            -r["nCands"],
        ),
        reverse=True,
    )
    out = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric": "manual tag IoU≥0.5 (north star)",
        "baseline": dict(zip(keys, baseline)),
        "best": rows_sorted[0],
        "top10": rows_sorted[:10],
        "all": rows_sorted,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nBEST:", json.dumps(rows_sorted[0]["params"], indent=2))
    print(
        f"manualIoU50 {rows_sorted[0]['pooledManualIou50Pct']}% "
        f"(baseline was {[r for r in rows if r['isBaseline']][0]['pooledManualIou50Pct']}%)"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
