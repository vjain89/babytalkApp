#!/usr/bin/env python3
"""Score VTC vs ECAPA on baby/adult *role* labels (not segment tightness).

This answers the stage-2 replacement question: given the same speech regions,
does Voice Type Classifier assign Baby vs Adult better than ECAPA clusters
plus a per-session manual/oracle speaker pick?

Evaluation unit = each human tag that already has a ``speaker`` field. For that
span we take the majority-overlapping VTC role (or ECAPA cluster), map to a
coarse class, and compute precision / recall / F1. Boundary IoU is deliberately
ignored.

    tools/.venv/bin/python tools/analysis/role_delta.py
    tools/.venv/bin/python tools/analysis/role_delta.py --no-ecapa   # VTC only

Writes ``tools/analysis/out/role_delta.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

LIBRARY = Path.home() / "Documents" / "BabyTalk" / "Library"
OUT_DIR = Path(__file__).resolve().parent / "out"

# Coarse classes used for the stage-2 decision.
BABY = "Baby"
ADULT = "Adult"
OTHER = "Other"  # other child / neither
CLASSES = (BABY, ADULT, OTHER)

# Tag speaker → coarse gold class.
TAG_TO_CLASS = {
    "Baby": BABY,
    "Parent": ADULT,
    "Other": OTHER,
}

# VTC role → coarse class.
VTC_TO_CLASS = {
    "KCHI": BABY,
    "FEM": ADULT,
    "MAL": ADULT,
    "OCH": OTHER,
}

POINT_TAG_SPAN_MS = 500.0
# If a second role covers at least this fraction of the tag, flag "overlap".
OVERLAP_FRAC = 0.25


def load_tags(kit: Path) -> list[dict]:
    path = kit / "tags.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    tags = data.get("tags", data if isinstance(data, list) else [])
    out = []
    for t in tags:
        if not t.get("speaker"):
            continue
        if t["speaker"] not in TAG_TO_CLASS:
            continue
        start = t.get("startMs")
        if start is None:
            start = t.get("tMs")
        if start is None:
            continue
        end = t.get("endMs")
        if end is None or end <= start:
            end = float(start) + POINT_TAG_SPAN_MS
        out.append({**t, "_start": float(start), "_end": float(end)})
    return out


def overlap_ms(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def majority_label(
    spans: list[tuple[float, float, str]],
    start: float,
    end: float,
) -> tuple[str | None, dict[str, float], bool]:
    """Return (majority_label, time_by_label_ms, is_overlap) inside [start,end]."""
    by: dict[str, float] = defaultdict(float)
    for s, e, lab in spans:
        ov = overlap_ms(start, end, s, e)
        if ov > 0:
            by[lab] += ov
    if not by:
        return None, {}, False
    ranked = sorted(by.items(), key=lambda kv: -kv[1])
    top_lab, top_ms = ranked[0]
    dur = max(end - start, 1e-6)
    is_overlap = False
    if len(ranked) > 1 and ranked[1][1] / dur >= OVERLAP_FRAC:
        is_overlap = True
    return top_lab, dict(by), is_overlap


def prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(100.0 * prec, 1),
        "recall": round(100.0 * rec, 1),
        "f1": round(100.0 * f1, 1),
        "support": tp + fn,
    }


def score_predictions(
    pairs: list[tuple[str, str | None]],
    *,
    labels: tuple[str, ...] = CLASSES,
) -> dict:
    """pairs = (gold_class, pred_class|None). None pred = always wrong (FN)."""
    per: dict[str, dict] = {}
    confusion: dict[str, Counter] = {g: Counter() for g in labels}
    confusion["MISS"] = Counter()  # unused key slot for clarity
    miss_as: Counter = Counter()

    for gold, pred in pairs:
        if gold not in confusion:
            confusion[gold] = Counter()
        if pred is None:
            miss_as[gold] += 1
            confusion[gold]["MISS"] += 1
        else:
            confusion[gold][pred] += 1

    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if p == lab and g != lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        per[lab] = prf(tp, fp, fn)

    # Macro over classes that appear in gold.
    present = [lab for lab in labels if per[lab]["support"] > 0]
    macro_f1 = (
        round(sum(per[lab]["f1"] for lab in present) / len(present), 1)
        if present
        else 0.0
    )
    agree = sum(1 for g, p in pairs if p is not None and g == p)
    n = len(pairs)
    return {
        "n": n,
        "agreePct": round(100.0 * agree / n, 1) if n else 0.0,
        "missPct": round(100.0 * sum(miss_as.values()) / n, 1) if n else 0.0,
        "macroF1": macro_f1,
        "perClass": per,
        "confusion": {g: dict(c) for g, c in confusion.items() if c},
        "missesByGold": dict(miss_as),
    }


def binary_baby_vs_rest(pairs: list[tuple[str, str | None]]) -> dict:
    """Collapse Adult+Other → NonBaby; score Baby detection."""
    bin_pairs = []
    for g, p in pairs:
        bg = BABY if g == BABY else "NonBaby"
        if p is None:
            bp = None
        else:
            bp = BABY if p == BABY else "NonBaby"
        bin_pairs.append((bg, bp))
    return score_predictions(bin_pairs, labels=(BABY, "NonBaby"))


def binary_baby_vs_adult(pairs: list[tuple[str, str | None]]) -> dict:
    """Drop Other-gold tags; score Baby vs Adult only."""
    filtered = []
    for g, p in pairs:
        if g == OTHER:
            continue
        if p == OTHER:
            # Predicting Other against Baby/Adult gold counts as wrong.
            filtered.append((g, "WRONG_OTHER"))
        else:
            filtered.append((g, p))
    # Remap WRONG_OTHER into a fake class that only creates FN/FP noise via != gold
    clean = [(g, None if p == "WRONG_OTHER" else p) for g, p in filtered]
    # Actually treating WRONG_OTHER as None (miss/wrong) is fine for P/R of Baby/Adult
    return score_predictions(clean, labels=(BABY, ADULT))


def run_vtc_spans(audio, sr) -> list[tuple[float, float, str]]:
    import vtc as vtc_mod

    result = vtc_mod.run_vtc_inference(audio, sr)
    if not result.ok:
        raise RuntimeError(result.error or "VTC failed")
    return [(t.start_ms, t.end_ms, t.role) for t in result.turns]


def run_ecapa_spans(audio, sr) -> list[tuple[float, float, str]]:
    """VAD regions → ECAPA turns (same stage-1+2 path as production)."""
    import numpy as np
    from diarize import diarize_regions
    from vad_segments import detect_speech_regions, frame_rms_db

    times, dbs = frame_rms_db(audio, sr)
    regions, _ = detect_speech_regions(times, dbs, audio=audio, sr=sr)
    if not regions:
        return []
    result = diarize_regions(
        audio,
        sr,
        [(r["start"], r["end"]) for r in regions],
        backend="ecapa",
    )
    if not result.ok:
        raise RuntimeError(result.error or "ECAPA failed")
    return [(t.start_ms, t.end_ms, t.speaker) for t in result.turns]


def oracle_cluster_map(
    tags: list[dict],
    cluster_spans: list[tuple[float, float, str]],
) -> dict[str, str]:
    """Per-kit majority vote: ECAPA cluster → coarse class (manual-pick oracle)."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for tag in tags:
        gold = TAG_TO_CLASS[tag["speaker"]]
        lab, _, _ = majority_label(cluster_spans, tag["_start"], tag["_end"])
        if lab is None:
            continue
        votes[lab][gold] += 1
    mapping: dict[str, str] = {}
    for cluster, ctr in votes.items():
        mapping[cluster] = ctr.most_common(1)[0][0]
    return mapping


def analyse_kit(kit: Path, *, run_ecapa: bool) -> dict:
    import numpy as np
    import soundfile as sf

    tags = load_tags(kit)
    man = json.loads((kit / "manifest.json").read_text())
    audio_path = kit / man.get("audioFile", "audio.wav")
    audio, sr = sf.read(str(audio_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    row: dict = {
        "kit": kit.name,
        "nTagsWithSpeaker": len(tags),
        "tagClasses": dict(Counter(TAG_TO_CLASS[t["speaker"]] for t in tags)),
    }
    if len(tags) < 1:
        row["skipped"] = "no speaker tags"
        return row

    t0 = time.time()
    vtc_spans = run_vtc_spans(audio, int(sr))
    row["vtcElapsedSec"] = round(time.time() - t0, 1)
    row["vtcTurns"] = len(vtc_spans)
    row["vtcRoles"] = dict(Counter(lab for _, _, lab in vtc_spans))

    vtc_pairs: list[tuple[str, str | None]] = []
    vtc_overlap_flags = 0
    for tag in tags:
        gold = TAG_TO_CLASS[tag["speaker"]]
        role, _, is_ov = majority_label(vtc_spans, tag["_start"], tag["_end"])
        if is_ov:
            vtc_overlap_flags += 1
        pred = VTC_TO_CLASS.get(role) if role else None
        vtc_pairs.append((gold, pred))
    row["vtcOverlapFlagged"] = vtc_overlap_flags
    row["vtc"] = {
        "ternary": score_predictions(vtc_pairs),
        "babyVsRest": binary_baby_vs_rest(vtc_pairs),
        "babyVsAdult": binary_baby_vs_adult(vtc_pairs),
    }

    if run_ecapa:
        t1 = time.time()
        try:
            ecapa_spans = run_ecapa_spans(audio, int(sr))
            row["ecapaOk"] = True
        except Exception as e:  # noqa: BLE001
            row["ecapaOk"] = False
            row["ecapaError"] = f"{type(e).__name__}: {e}"
            ecapa_spans = []
        row["ecapaElapsedSec"] = round(time.time() - t1, 1)
        row["ecapaTurns"] = len(ecapa_spans)
        row["ecapaClusters"] = dict(Counter(lab for _, _, lab in ecapa_spans))

        mapping = oracle_cluster_map(tags, ecapa_spans)
        row["ecapaOracleMap"] = mapping

        ecapa_pairs: list[tuple[str, str | None]] = []
        for tag in tags:
            gold = TAG_TO_CLASS[tag["speaker"]]
            cluster, _, _ = majority_label(ecapa_spans, tag["_start"], tag["_end"])
            pred = mapping.get(cluster) if cluster else None
            ecapa_pairs.append((gold, pred))
        row["ecapaOracle"] = {
            "ternary": score_predictions(ecapa_pairs),
            "babyVsRest": binary_baby_vs_rest(ecapa_pairs),
            "babyVsAdult": binary_baby_vs_adult(ecapa_pairs),
        }

        # Always-Baby prior (what you get if you never open the speaker chip).
        always = [(TAG_TO_CLASS[t["speaker"]], BABY) for t in tags]
        row["alwaysBaby"] = {
            "ternary": score_predictions(always),
            "babyVsRest": binary_baby_vs_rest(always),
            "babyVsAdult": binary_baby_vs_adult(always),
        }

    return row


def pool_pairs_from_kits(kits: list[dict], method_key: str, metric: str) -> dict | None:
    """Re-score by concatenating is awkward without raw pairs; use support-weighted F1.

    We stored per-kit confusion — rebuild pairs counts from confusion matrices.
    """
    conf: dict[str, Counter] = defaultdict(Counter)
    for k in kits:
        block = k.get(method_key) or {}
        metric_block = block.get(metric) or {}
        c = metric_block.get("confusion") or {}
        for gold, preds in c.items():
            for pred, n in preds.items():
                conf[gold][pred] += n
    if not conf:
        return None

    # Expand back to pairs for score_predictions.
    pairs: list[tuple[str, str | None]] = []
    labels = set()
    for gold, preds in conf.items():
        labels.add(gold)
        for pred, n in preds.items():
            if pred == "MISS":
                pairs.extend([(gold, None)] * n)
            else:
                labels.add(pred)
                pairs.extend([(gold, pred)] * n)
    # Keep stable class order when possible.
    order = [c for c in (BABY, ADULT, OTHER, "NonBaby") if c in labels]
    order += sorted(labels - set(order))
    return score_predictions(pairs, labels=tuple(order))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(LIBRARY))
    ap.add_argument("--no-ecapa", action="store_true")
    ap.add_argument("--min-tags", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_DIR / "role_delta.json"))
    args = ap.parse_args()

    lib = Path(args.library).expanduser()
    kits_out = []
    for kit in sorted(lib.iterdir()):
        if not (kit / "manifest.json").exists():
            continue
        tags = load_tags(kit)
        if len(tags) < args.min_tags:
            continue
        print(
            f"[role] {kit.name}  speaker-tags={len(tags)}",
            flush=True,
        )
        kits_out.append(analyse_kit(kit, run_ecapa=not args.no_ecapa))

    pooled = {
        "vtc": {
            "ternary": pool_pairs_from_kits(kits_out, "vtc", "ternary"),
            "babyVsRest": pool_pairs_from_kits(kits_out, "vtc", "babyVsRest"),
            "babyVsAdult": pool_pairs_from_kits(kits_out, "vtc", "babyVsAdult"),
        }
    }
    if not args.no_ecapa:
        pooled["ecapaOracle"] = {
            "ternary": pool_pairs_from_kits(kits_out, "ecapaOracle", "ternary"),
            "babyVsRest": pool_pairs_from_kits(kits_out, "ecapaOracle", "babyVsRest"),
            "babyVsAdult": pool_pairs_from_kits(kits_out, "ecapaOracle", "babyVsAdult"),
        }
        pooled["alwaysBaby"] = {
            "ternary": pool_pairs_from_kits(kits_out, "alwaysBaby", "ternary"),
            "babyVsRest": pool_pairs_from_kits(kits_out, "alwaysBaby", "babyVsRest"),
            "babyVsAdult": pool_pairs_from_kits(kits_out, "alwaysBaby", "babyVsAdult"),
        }

    out = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "library": str(lib),
        "task": "baby-vs-adult role label (not segment IoU)",
        "nKits": len(kits_out),
        "nTags": sum(k.get("nTagsWithSpeaker", 0) for k in kits_out),
        "tagClasses": dict(
            sum(
                (Counter(k.get("tagClasses") or {}) for k in kits_out),
                Counter(),
            )
        ),
        "kits": kits_out,
        "pooled": pooled,
        "notes": {
            "vtc": "KCHI→Baby, FEM/MAL→Adult, OCH→Other; no human mapping",
            "ecapaOracle": (
                "Per-kit majority vote cluster→class using the same tags "
                "(upper bound on 'ECAPA + manual speaker pick')"
            ),
            "alwaysBaby": "Predict Baby for every tag (majority-class prior)",
            "unit": "one human tag span with a speaker field",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(pooled, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
