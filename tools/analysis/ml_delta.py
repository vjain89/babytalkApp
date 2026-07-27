#!/usr/bin/env python3
"""Quantify the delta between Find-speech-segments ML candidates and final human tags.

Read-only analysis harness. Never writes into the library; the fresh pipeline
re-run goes through ``run_vad_on_audio`` directly so nothing touches
``annotations.json`` and no tag-overlap suppression is applied.

    tools/.venv/bin/python tools/analysis/ml_delta.py            # archival only
    tools/.venv/bin/python tools/analysis/ml_delta.py --fresh    # + re-run current VAD
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

LIBRARY = Path.home() / "Documents" / "BabyTalk" / "Library"
OUT_DIR = Path(__file__).resolve().parent / "out"

POINT_TAG_SPAN_MS = 500.0
VERBAL = "verbal vocalization"


def load(path: Path, key: str) -> list:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get(key, [])
    return data if isinstance(data, list) else []


def span(item: dict) -> tuple[float, float] | None:
    start = item.get("startMs")
    if start is None:
        start = item.get("tMs")
    if start is None:
        return None
    end = item.get("endMs")
    if end is None or end <= start:
        end = start + POINT_TAG_SPAN_MS
    return float(start), float(end)


def overlap_ms(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = overlap_ms(a, b)
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def quantiles(values: list[float], nd: int = 1) -> dict:
    if not values:
        return {}
    vals = sorted(values)

    def q(p: float) -> float:
        if len(vals) == 1:
            return round(vals[0], nd)
        idx = p * (len(vals) - 1)
        lo, hi = math.floor(idx), math.ceil(idx)
        return round(vals[lo] + (vals[hi] - vals[lo]) * (idx - lo), nd)

    return {
        "n": len(vals),
        "min": round(vals[0], nd),
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": round(vals[-1], nd),
        "mean": round(statistics.fmean(vals), nd),
    }


def histogram(values: list[float], edges: list[float]) -> list[dict]:
    """Bucket values into [edges[i], edges[i+1]) plus open tails."""
    buckets = []
    labels = []
    labels.append(f"< {edges[0]:g}")
    buckets.append(sum(1 for v in values if v < edges[0]))
    for lo, hi in zip(edges, edges[1:]):
        labels.append(f"{lo:g}–{hi:g}")
        buckets.append(sum(1 for v in values if lo <= v < hi))
    labels.append(f"≥ {edges[-1]:g}")
    buckets.append(sum(1 for v in values if v >= edges[-1]))
    return [{"bucket": lbl, "count": n} for lbl, n in zip(labels, buckets)]


# --------------------------------------------------------------------------
# core comparison
# --------------------------------------------------------------------------


def compare(tags: list[dict], cands: list[dict], *, label: str) -> dict:
    """Match every tag to its best-overlapping candidate and summarise the delta."""
    tspans = [(t, span(t)) for t in tags]
    tspans = [(t, s) for t, s in tspans if s]
    cspans = [(c, span(c)) for c in cands]
    cspans = [(c, s) for c, s in cspans if s]

    matches = []
    misses = []
    for tag, ts in tspans:
        best = None
        overlapping = []
        for cand, cs in cspans:
            ov = overlap_ms(ts, cs)
            if ov <= 0:
                continue
            overlapping.append((cand, cs, ov))
            score = iou(ts, cs)
            if best is None or score > best["iou"]:
                best = {
                    "iou": score,
                    "cand": cand,
                    "cspan": cs,
                    "overlap": ov,
                }
        if best is None:
            misses.append(tag)
            continue
        ts_dur = ts[1] - ts[0]
        cs = best["cspan"]
        cs_dur = cs[1] - cs[0]
        speakers = {
            t2.get("speaker")
            for t2, ts2 in tspans
            if overlap_ms(ts2, cs) > 0 and t2.get("speaker")
        }
        matches.append(
            {
                "tag": tag,
                "cand": best["cand"],
                "iou": best["iou"],
                "overlapFracTag": best["overlap"] / ts_dur if ts_dur else 0.0,
                "dStart": ts[0] - cs[0],
                "dEnd": ts[1] - cs[1],
                "tagDur": ts_dur,
                "candDur": cs_dur,
                "durRatio": cs_dur / ts_dur if ts_dur else 0.0,
                "nOverlappingCands": len(overlapping),
                "candSpeakerSpread": len(speakers),
                "ts": list(ts),
                "cs": list(cs),
            }
        )

    # candidate-side view: which candidates hit a tag at all
    cand_rows = []
    for cand, cs in cspans:
        hits = [(t, ts) for t, ts in tspans if overlap_ms(ts, cs) > 0]
        covered = sum(overlap_ms(ts, cs) for _, ts in hits)
        cs_dur = cs[1] - cs[0]
        cand_rows.append(
            {
                "cand": cand,
                "dur": cs_dur,
                "nTags": len(hits),
                "coveredFrac": min(1.0, covered / cs_dur) if cs_dur else 0.0,
                "speakers": sorted(
                    {t.get("speaker") for t, _ in hits if t.get("speaker")}
                ),
                "bestIou": max((iou(ts, cs) for _, ts in hits), default=0.0),
            }
        )

    n_tags = len(tspans)

    # Tags promoted straight from a candidate (`ml_confirmed`) are covered by
    # construction, so the honest coverage question is asked of manual tags —
    # the ones the reviewer drew by hand after refining or rejecting a span.
    by_source: dict[str, dict] = {}
    for src_key, want_ml in (("manual", False), ("mlConfirmed", True)):
        subset = [
            m
            for m in matches
            if (m["tag"].get("source") == "ml_confirmed") == want_ml
        ]
        n_src = sum(
            1
            for t, _ in tspans
            if (t.get("source") == "ml_confirmed") == want_ml
        )
        by_source[src_key] = {
            "nTags": n_src,
            "anyOverlapPct": pct(len(subset), n_src),
            "iou50Pct": pct(sum(1 for m in subset if m["iou"] >= 0.5), n_src),
            "iou70Pct": pct(sum(1 for m in subset if m["iou"] >= 0.7), n_src),
            "contains90Pct": pct(
                sum(1 for m in subset if m["overlapFracTag"] >= 0.9), n_src
            ),
            "missed": n_src - len(subset),
            "iou": quantiles([m["iou"] for m in subset], 3),
            "durRatio": quantiles([m["durRatio"] for m in subset], 2),
            "dStart": quantiles([m["dStart"] for m in subset]),
            "dEnd": quantiles([m["dEnd"] for m in subset]),
        }

    verbal = [m for m in matches if m["tag"].get("category") == VERBAL]
    baby = [m for m in matches if (m["tag"].get("speaker") or "") == "Baby"]
    verbal_tags = [t for t, _ in tspans if t.get("category") == VERBAL]
    baby_tags = [t for t, _ in tspans if (t.get("speaker") or "") == "Baby"]

    d_start = [m["dStart"] for m in matches]
    d_end = [m["dEnd"] for m in matches]

    fp = [r for r in cand_rows if r["nTags"] == 0]
    fp_dismissed = [r for r in fp if r["cand"].get("status") == "dismissed"]
    scores = [
        r["cand"].get("speechScore")
        for r in cand_rows
        if r["cand"].get("speechScore") is not None
    ]

    return {
        "label": label,
        "nTags": n_tags,
        "nCands": len(cspans),
        "coverage": {
            "anyOverlap": len(matches),
            "anyOverlapPct": pct(len(matches), n_tags),
            "contains50": sum(1 for m in matches if m["overlapFracTag"] >= 0.5),
            "contains50Pct": pct(
                sum(1 for m in matches if m["overlapFracTag"] >= 0.5), n_tags
            ),
            "contains90": sum(1 for m in matches if m["overlapFracTag"] >= 0.9),
            "contains90Pct": pct(
                sum(1 for m in matches if m["overlapFracTag"] >= 0.9), n_tags
            ),
            "iou50": sum(1 for m in matches if m["iou"] >= 0.5),
            "iou50Pct": pct(sum(1 for m in matches if m["iou"] >= 0.5), n_tags),
            "iou70": sum(1 for m in matches if m["iou"] >= 0.7),
            "iou70Pct": pct(sum(1 for m in matches if m["iou"] >= 0.7), n_tags),
            "verbalTags": len(verbal_tags),
            "verbalAnyPct": pct(len(verbal), len(verbal_tags)),
            "verbalIou50Pct": pct(
                sum(1 for m in verbal if m["iou"] >= 0.5), len(verbal_tags)
            ),
            "babyTags": len(baby_tags),
            "babyAnyPct": pct(len(baby), len(baby_tags)),
            "babyIou50Pct": pct(
                sum(1 for m in baby if m["iou"] >= 0.5), len(baby_tags)
            ),
        },
        "boundary": {
            "dStart": quantiles(d_start),
            "dEnd": quantiles(d_end),
            "iou": quantiles([m["iou"] for m in matches], 3),
            "durRatio": quantiles([m["durRatio"] for m in matches], 2),
            "tagDur": quantiles([m["tagDur"] for m in matches]),
            "candDur": quantiles([m["candDur"] for m in matches]),
            "lateStart": sum(1 for m in matches if m["dStart"] < -200),
            "lateStartPct": pct(
                sum(1 for m in matches if m["dStart"] < -200), len(matches)
            ),
            "earlyEnd": sum(1 for m in matches if m["dEnd"] > 200),
            "earlyEndPct": pct(
                sum(1 for m in matches if m["dEnd"] > 200), len(matches)
            ),
            "candTooLong2x": sum(1 for m in matches if m["durRatio"] >= 2.0),
            "candTooLong2xPct": pct(
                sum(1 for m in matches if m["durRatio"] >= 2.0), len(matches)
            ),
            "candTooLong4x": sum(1 for m in matches if m["durRatio"] >= 4.0),
            "candTooLong4xPct": pct(
                sum(1 for m in matches if m["durRatio"] >= 4.0), len(matches)
            ),
            "candTooShort": sum(1 for m in matches if m["durRatio"] < 0.8),
            "candTooShortPct": pct(
                sum(1 for m in matches if m["durRatio"] < 0.8), len(matches)
            ),
            "dStartHist": histogram(d_start, [-2000, -1000, -500, -200, 0, 200, 500]),
            "dEndHist": histogram(d_end, [-500, -200, 0, 200, 500, 1000, 2000]),
            "iouHist": histogram(
                [m["iou"] for m in matches], [0.1, 0.25, 0.5, 0.7, 0.9]
            ),
            "durRatioHist": histogram(
                [m["durRatio"] for m in matches], [0.8, 1.0, 1.5, 2.0, 4.0, 8.0]
            ),
        },
        "falsePositives": {
            "nNoTagOverlap": len(fp),
            "noTagOverlapPct": pct(len(fp), len(cspans)),
            "nDismissedNoOverlap": len(fp_dismissed),
            "durQuantiles": quantiles([r["dur"] for r in fp]),
            "durHist": histogram(
                [r["dur"] for r in fp], [500, 1000, 2000, 4000, 8000]
            ),
            "speechScoreHitVsMiss": {
                "hit": quantiles(
                    [
                        r["cand"]["speechScore"]
                        for r in cand_rows
                        if r["nTags"] > 0 and r["cand"].get("speechScore") is not None
                    ],
                    3,
                ),
                "miss": quantiles(
                    [
                        r["cand"]["speechScore"]
                        for r in cand_rows
                        if r["nTags"] == 0 and r["cand"].get("speechScore") is not None
                    ],
                    3,
                ),
                "available": len(scores),
            },
            "flaggedNonSpeech": sum(
                1 for r in cand_rows if r["cand"].get("nonSpeechReason")
            ),
        },
        "misses": {
            "n": len(misses),
            "pct": pct(len(misses), n_tags),
            "byCategory": tally(m.get("category") for m in misses),
            "bySpeaker": tally(m.get("speaker") for m in misses),
            "durQuantiles": quantiles(
                [span(m)[1] - span(m)[0] for m in misses if span(m)]
            ),
            "examples": [
                {
                    "startMs": m.get("startMs"),
                    "endMs": m.get("endMs"),
                    "category": m.get("category"),
                    "speaker": m.get("speaker"),
                    "word": m.get("word") or m.get("phonetic") or "",
                }
                for m in misses[:12]
            ],
        },
        "fragmentation": {
            "tagsWithMultiCand": sum(1 for m in matches if m["nOverlappingCands"] > 1),
            "tagsWithMultiCandPct": pct(
                sum(1 for m in matches if m["nOverlappingCands"] > 1), len(matches)
            ),
            "candsWith2PlusTags": sum(1 for r in cand_rows if r["nTags"] >= 2),
            "candsWith2PlusTagsPct": pct(
                sum(1 for r in cand_rows if r["nTags"] >= 2), len(cspans)
            ),
            "candsMultiSpeaker": sum(1 for r in cand_rows if len(r["speakers"]) >= 2),
            "candsMultiSpeakerPct": pct(
                sum(1 for r in cand_rows if len(r["speakers"]) >= 2), len(cspans)
            ),
            "tagsPerHitCand": quantiles(
                [float(r["nTags"]) for r in cand_rows if r["nTags"] > 0]
            ),
            "coveredFracOfHitCands": quantiles(
                [r["coveredFrac"] for r in cand_rows if r["nTags"] > 0], 3
            ),
        },
        "bySource": by_source,
        "_matches": matches,
        "_candRows": cand_rows,
        "_misses": misses,
    }


def constant_trim_sim(matches: list[dict]) -> dict:
    """Best achievable IoU if every candidate edge were shifted by one constant.

    Separates "the pad is simply too generous" (a global constant helps a lot)
    from "each candidate is wrong by a different amount" (only resegmentation
    helps). Trims are inward: ``a`` off the start, ``b`` off the end.
    """
    if not matches:
        return {}
    grid_a = list(range(0, 801, 50))
    grid_b = list(range(0, 2001, 50))
    baseline_ious = [m["iou"] for m in matches]
    baseline = {
        "medianIou": round(statistics.median(baseline_ious), 3),
        "iou50Pct": pct(sum(1 for v in baseline_ious if v >= 0.5), len(matches)),
    }

    best = None
    surface = []
    for a in grid_a:
        for b in grid_b:
            ious = []
            for m in matches:
                ts0, ts1 = m["ts"]
                cs0, cs1 = m["cs"][0] + a, m["cs"][1] - b
                if cs1 <= cs0:
                    ious.append(0.0)
                    continue
                inter = max(0.0, min(ts1, cs1) - max(ts0, cs0))
                union = (ts1 - ts0) + (cs1 - cs0) - inter
                ious.append(inter / union if union > 0 else 0.0)
            med = statistics.median(ious)
            hit50 = pct(sum(1 for v in ious if v >= 0.5), len(ious))
            if best is None or med > best["medianIou"]:
                best = {
                    "trimStartMs": a,
                    "trimEndMs": b,
                    "medianIou": round(med, 3),
                    "iou50Pct": hit50,
                }
            if a in (0, 100, 200) and b % 250 == 0:
                surface.append(
                    {
                        "trimStartMs": a,
                        "trimEndMs": b,
                        "medianIou": round(med, 3),
                        "iou50Pct": hit50,
                    }
                )
    return {"baseline": baseline, "best": best, "surface": surface}


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability a random tag-matching candidate outscores a random unused one."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(wins / (len(pos) * len(neg)), 3)


def tally(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        key = v or "(none)"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def confirm_edits(tags: list[dict], anns: list[dict]) -> dict:
    """Bound edits the reviewer made *after* accepting an ML candidate.

    Confirm copies the candidate span verbatim into tags.json; later span edits
    only touch tags.json. So for a shared uuid, tag − annotation is exactly the
    correction the human applied to the machine's boundary.
    """
    by_uuid = {a.get("uuid"): a for a in anns}
    rows = []
    for t in tags:
        if t.get("source") != "ml_confirmed":
            continue
        a = by_uuid.get(t.get("uuid"))
        if not a:
            continue
        ts, cs = span(t), span(a)
        if not ts or not cs:
            continue
        rows.append(
            {
                "dStart": ts[0] - cs[0],
                "dEnd": ts[1] - cs[1],
                "tagDur": ts[1] - ts[0],
                "candDur": cs[1] - cs[0],
                "edited": abs(ts[0] - cs[0]) > 1 or abs(ts[1] - cs[1]) > 1,
            }
        )
    edited = [r for r in rows if r["edited"]]
    return {
        "n": len(rows),
        "nEdited": len(edited),
        "editedPct": pct(len(edited), len(rows)),
        "dStart": quantiles([r["dStart"] for r in edited]),
        "dEnd": quantiles([r["dEnd"] for r in edited]),
        "shrinkPct": pct(
            sum(1 for r in edited if r["tagDur"] < r["candDur"]), len(edited)
        ),
    }


def workflow_signature(tags: list[dict], anns: list[dict]) -> dict:
    """dismiss-then-manual-tag vs confirm-ML-then-tag."""
    manual = [t for t in tags if t.get("source") != "ml_confirmed"]
    ml = [t for t in tags if t.get("source") == "ml_confirmed"]
    dismissed = [
        a for a in anns if a.get("status") == "dismissed" and a.get("source") == "vad_v0"
    ]
    dspans = [(a, span(a)) for a in dismissed]
    dspans = [(a, s) for a, s in dspans if s]

    rescued = 0
    for t in manual:
        ts = span(t)
        if not ts:
            continue
        if any(overlap_ms(ts, s) > 0 for _, s in dspans):
            rescued += 1
    return {
        "nTags": len(tags),
        "nManual": len(manual),
        "nMlConfirmed": len(ml),
        "manualPct": pct(len(manual), len(tags)),
        "mlConfirmedPct": pct(len(ml), len(tags)),
        "nDismissed": len(dismissed),
        "manualInsideDismissed": rescued,
        "manualInsideDismissedPct": pct(rescued, len(manual)),
        "manualNoCandidate": len(manual) - rescued,
    }


def vintage(anns: list[dict]) -> dict:
    """Candidates accumulate across pipeline versions; field presence dates them."""
    cur = sum(1 for a in anns if a.get("speechScore") is not None)
    diar = sum(
        1
        for a in anns
        if a.get("speakerCluster") is not None and a.get("speechScore") is None
    )
    old = sum(
        1
        for a in anns
        if a.get("speechScore") is None and a.get("speakerCluster") is None
    )
    return {"currentGate": cur, "diarizedOnly": diar, "preDiarization": old}


def analyse_kit(
    kit: Path, *, fresh: bool, diarization: str, segmentation: str = "vad"
) -> dict:
    tags = load(kit / "tags.json", "tags")
    anns = load(kit / "annotations.json", "annotations")
    manifest = json.loads((kit / "manifest.json").read_text())

    archival_cands = [
        a for a in anns if a.get("source") in ("vad_v0", "ml_v0", "ml_confirmed")
    ]

    audio_path = kit / manifest.get("audioFile", "audio.wav")
    duration_ms = manifest.get("durationMs")
    if not duration_ms and audio_path.exists():
        import soundfile as sf

        info = sf.info(str(audio_path))
        duration_ms = int(1000 * info.frames / info.samplerate)

    # A kit only supports false-positive claims if the reviewer actually walked
    # the candidate list; a dismissal is the proof they did.
    curated = any(a.get("status") == "dismissed" for a in anns)

    result = {
        "kit": kit.name,
        "curated": curated,
        "durationMs": duration_ms,
        "nTags": len(tags),
        "nAnnotations": len(anns),
        "tagCategories": tally(t.get("category") for t in tags),
        "tagSpeakers": tally(t.get("speaker") for t in tags),
        "candidateVintage": vintage([a for a in anns if a.get("source") == "vad_v0"]),
        "candidateStatus": tally(
            f"{a.get('source')}/{a.get('status')}" for a in anns
        ),
        "workflow": workflow_signature(tags, anns),
        "confirmEdits": confirm_edits(tags, anns),
        "archival": compare(tags, archival_cands, label="archival"),
        "tagDur": quantiles(
            [span(t)[1] - span(t)[0] for t in tags if span(t)]
        ),
    }

    if fresh:
        from vad_segments import run_vad_on_audio

        audio = kit / manifest.get("audioFile", "audio.wav")
        t0 = time.time()
        cands, stats = run_vad_on_audio(
            audio, diarization=diarization, segmentation=segmentation
        )
        diar = stats.get("diarization") or {}
        result["fresh"] = compare(tags, cands, label="fresh")
        result["fresh"]["roleAccuracy"] = role_accuracy(tags, cands)
        result["freshStats"] = {
            k: v for k, v in stats.items() if isinstance(v, (int, float, str))
        }
        result["freshStats"]["elapsedSec"] = round(time.time() - t0, 1)
        result["freshStats"]["segmentation"] = stats.get("segmentation", segmentation)
        result["freshStats"]["diarBackend"] = diar.get("backend")
        result["freshStats"]["diarOk"] = diar.get("ok")
        if diar.get("error"):
            result["freshStats"]["diarError"] = diar.get("error")
        result["freshCandDur"] = quantiles(
            [span(c)[1] - span(c)[0] for c in cands if span(c)]
        )
        result["freshCandDurHist"] = histogram(
            [span(c)[1] - span(c)[0] for c in cands if span(c)],
            [500, 1000, 2000, 4000, 8000],
        )
        result["freshScoreHist"] = histogram(
            [c.get("speechScore", 0.0) for c in cands],
            [0.55, 0.6, 0.68, 0.75, 0.85],
        )
    return result


def role_accuracy(tags: list[dict], cands: list[dict]) -> dict:
    """Where a candidate overlaps a tagged speaker, does the prefilled role match?"""
    pairs = []
    for tag in tags:
        ts = span(tag)
        if not ts or not tag.get("speaker"):
            continue
        best = None
        for cand in cands:
            cs = span(cand)
            if not cs:
                continue
            score = iou(ts, cs)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, cand)
        if best is None:
            continue
        cand = best[1]
        pred = cand.get("speaker") or ""
        if not pred:
            continue
        pairs.append((tag.get("speaker"), pred, best[0]))
    if not pairs:
        return {"n": 0, "agreePct": 0.0, "byTagSpeaker": {}}
    agree = sum(1 for t, p, _ in pairs if t == p)
    by: dict[str, dict] = {}
    for t, p, _ in pairs:
        slot = by.setdefault(t, {"n": 0, "agree": 0})
        slot["n"] += 1
        if t == p:
            slot["agree"] += 1
    return {
        "n": len(pairs),
        "agreePct": pct(agree, len(pairs)),
        "byTagSpeaker": {
            k: {"n": v["n"], "agreePct": pct(v["agree"], v["n"])} for k, v in by.items()
        },
    }


def strip_private(obj):
    if isinstance(obj, dict):
        return {
            k: strip_private(v) for k, v in obj.items() if not k.startswith("_")
        }
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def pooled(kits: list[dict], key: str) -> dict:
    """Re-run the comparison over every kit's raw match rows at once."""
    matches, cand_rows, misses = [], [], []
    n_tags = 0
    for k in kits:
        block = k.get(key)
        if not block:
            continue
        matches += block["_matches"]
        cand_rows += block["_candRows"]
        misses += block["_misses"]
        n_tags += block["nTags"]
    if not n_tags:
        return {}
    d_start = [m["dStart"] for m in matches]
    d_end = [m["dEnd"] for m in matches]
    fp = [r for r in cand_rows if r["nTags"] == 0]
    hit_scores = [
        r["cand"]["speechScore"]
        for r in cand_rows
        if r["nTags"] > 0 and r["cand"].get("speechScore") is not None
    ]
    miss_scores = [
        r["cand"]["speechScore"]
        for r in cand_rows
        if r["nTags"] == 0 and r["cand"].get("speechScore") is not None
    ]
    return {
        "nTags": n_tags,
        "nCands": len(cand_rows),
        "tagDur": quantiles([m["tagDur"] for m in matches]),
        "candDur": quantiles([r["dur"] for r in cand_rows]),
        "speechScoreAuc": auc(hit_scores, miss_scores),
        "coverage": {
            "anyOverlapPct": pct(len(matches), n_tags),
            "contains90Pct": pct(
                sum(1 for m in matches if m["overlapFracTag"] >= 0.9), n_tags
            ),
            "iou50Pct": pct(sum(1 for m in matches if m["iou"] >= 0.5), n_tags),
            "iou70Pct": pct(sum(1 for m in matches if m["iou"] >= 0.7), n_tags),
        },
        "bySource": {
            src: {
                "nTags": sum(
                    1
                    for t in [
                        m["tag"]
                        for k in kits
                        if k.get(key)
                        for m in k[key]["_matches"]
                    ]
                    if (t.get("source") == "ml_confirmed") == (src == "mlConfirmed")
                ),
                "iou50Pct": pct(
                    sum(
                        1
                        for m in matches
                        if (m["tag"].get("source") == "ml_confirmed")
                        == (src == "mlConfirmed")
                        and m["iou"] >= 0.5
                    ),
                    max(
                        1,
                        sum(
                            1
                            for m in matches
                            if (m["tag"].get("source") == "ml_confirmed")
                            == (src == "mlConfirmed")
                        ),
                    ),
                ),
                "iou": quantiles(
                    [
                        m["iou"]
                        for m in matches
                        if (m["tag"].get("source") == "ml_confirmed")
                        == (src == "mlConfirmed")
                    ],
                    3,
                ),
                "durRatio": quantiles(
                    [
                        m["durRatio"]
                        for m in matches
                        if (m["tag"].get("source") == "ml_confirmed")
                        == (src == "mlConfirmed")
                    ],
                    2,
                ),
            }
            for src in ("manual", "mlConfirmed")
        },
        "boundary": {
            "dStart": quantiles(d_start),
            "dEnd": quantiles(d_end),
            "iou": quantiles([m["iou"] for m in matches], 3),
            "durRatio": quantiles([m["durRatio"] for m in matches], 2),
            "dStartHist": histogram(d_start, [-2000, -1000, -500, -200, 0, 200, 500]),
            "dEndHist": histogram(d_end, [-500, -200, 0, 200, 500, 1000, 2000]),
            "iouHist": histogram(
                [m["iou"] for m in matches], [0.1, 0.25, 0.5, 0.7, 0.9]
            ),
            "durRatioHist": histogram(
                [m["durRatio"] for m in matches], [0.8, 1.0, 1.5, 2.0, 4.0, 8.0]
            ),
            "candTooLong2xPct": pct(
                sum(1 for m in matches if m["durRatio"] >= 2.0), len(matches)
            ),
            "lateStartPct": pct(
                sum(1 for m in matches if m["dStart"] < -200), len(matches)
            ),
            "earlyEndPct": pct(
                sum(1 for m in matches if m["dEnd"] > 200), len(matches)
            ),
        },
        "falsePositives": {
            "nNoTagOverlap": len(fp),
            "noTagOverlapPct": pct(len(fp), len(cand_rows)),
            "durHist": histogram([r["dur"] for r in fp], [500, 1000, 2000, 4000, 8000]),
            "durQuantiles": quantiles([r["dur"] for r in fp]),
            "hitScore": quantiles(
                [
                    r["cand"]["speechScore"]
                    for r in cand_rows
                    if r["nTags"] > 0 and r["cand"].get("speechScore") is not None
                ],
                3,
            ),
            "missScore": quantiles(
                [
                    r["cand"]["speechScore"]
                    for r in cand_rows
                    if r["nTags"] == 0 and r["cand"].get("speechScore") is not None
                ],
                3,
            ),
        },
        "misses": {
            "n": len(misses),
            "pct": pct(len(misses), n_tags),
            "byCategory": tally(m.get("category") for m in misses),
            "bySpeaker": tally(m.get("speaker") for m in misses),
            "durQuantiles": quantiles(
                [span(m)[1] - span(m)[0] for m in misses if span(m)]
            ),
            "examples": [
                {
                    "kit": next(
                        (
                            k["kit"]
                            for k in kits
                            if k.get(key) and m in k[key]["_misses"]
                        ),
                        "",
                    ),
                    "startMs": m.get("startMs"),
                    "endMs": m.get("endMs"),
                    "category": m.get("category"),
                    "speaker": m.get("speaker"),
                    "word": m.get("word") or m.get("phonetic") or m.get("note") or "",
                }
                for m in misses
            ],
        },
        "constantTrimSim": constant_trim_sim(matches),
        "tagDur": quantiles([m["tagDur"] for m in matches]),
        "candDur": quantiles([r["dur"] for r in cand_rows]),
        "speechScoreAuc": auc(
            [
                r["cand"]["speechScore"]
                for r in cand_rows
                if r["nTags"] > 0 and r["cand"].get("speechScore") is not None
            ],
            [
                r["cand"]["speechScore"]
                for r in cand_rows
                if r["nTags"] == 0 and r["cand"].get("speechScore") is not None
            ],
        ),
        "fragmentation": {
            "candsWith2PlusTagsPct": pct(
                sum(1 for r in cand_rows if r["nTags"] >= 2), len(cand_rows)
            ),
            "candsMultiSpeakerPct": pct(
                sum(1 for r in cand_rows if len(r["speakers"]) >= 2), len(cand_rows)
            ),
            "tagsWithMultiCandPct": pct(
                sum(1 for m in matches if m["nOverlappingCands"] > 1), len(matches)
            ),
            "coveredFracOfHitCands": quantiles(
                [r["coveredFrac"] for r in cand_rows if r["nTags"] > 0], 3
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(LIBRARY))
    ap.add_argument("--fresh", action="store_true", help="re-run current VAD pipeline")
    ap.add_argument("--diarization", default="auto")
    ap.add_argument(
        "--segmentation",
        default="vad",
        choices=["vad", "vtc-first"],
        help="Parent-span finder for --fresh (default: vad)",
    )
    ap.add_argument("--min-tags", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_DIR / "ml_delta.json"))
    args = ap.parse_args()

    lib = Path(args.library).expanduser()
    kits = []
    for kit in sorted(lib.iterdir()):
        if not (kit / "manifest.json").exists():
            continue
        tags = load(kit / "tags.json", "tags")
        anns = load(kit / "annotations.json", "annotations")
        # Without --fresh there is nothing to compare against unless the kit
        # already has saved candidates; with it, tag-only kits are the most
        # honest test set (they were labelled without ever seeing ML output).
        if len(tags) < args.min_tags or (not anns and not args.fresh):
            continue
        print(
            f"[analyse] {kit.name}  tags={len(tags)} anns={len(anns)}"
            f"  seg={args.segmentation} diar={args.diarization}",
            flush=True,
        )
        kits.append(
            analyse_kit(
                kit,
                fresh=args.fresh,
                diarization=args.diarization,
                segmentation=args.segmentation,
            )
        )

    curated = [k for k in kits if k.get("curated")]
    out = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "library": str(lib),
        "segmentation": args.segmentation,
        "diarization": args.diarization,
        "kits": kits,
        "curatedKits": [k["kit"] for k in curated],
        "pooledArchival": pooled(kits, "archival"),
        "pooledArchivalCurated": pooled(curated, "archival"),
    }
    if args.fresh:
        out["pooledFresh"] = pooled(kits, "fresh")
        out["pooledFreshCurated"] = pooled(curated, "fresh")
        # Pool role accuracy across curated kits when VTC prefills speakers.
        role_ns = [
            (k.get("fresh") or {}).get("roleAccuracy") or {}
            for k in curated
        ]
        n = sum(r.get("n", 0) for r in role_ns)
        if n:
            # Recompute from per-kit tallies when available.
            agree = 0
            total = 0
            by: dict[str, dict] = {}
            for k in curated:
                ra = (k.get("fresh") or {}).get("roleAccuracy") or {}
                for spk, slot in (ra.get("byTagSpeaker") or {}).items():
                    b = by.setdefault(spk, {"n": 0, "agree": 0})
                    b["n"] += slot.get("n", 0)
                    b["agree"] += int(round(slot.get("agreePct", 0) * slot.get("n", 0) / 100.0))
                total += ra.get("n", 0)
                agree += int(round(ra.get("agreePct", 0) * ra.get("n", 0) / 100.0))
            out["pooledRoleAccuracyCurated"] = {
                "n": total,
                "agreePct": pct(agree, total),
                "byTagSpeaker": {
                    k: {"n": v["n"], "agreePct": pct(v["agree"], v["n"])}
                    for k, v in by.items()
                },
            }

    out = strip_private(out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out.get("pooledArchival", {}), indent=2))
    if args.fresh:
        print(json.dumps(out.get("pooledFresh", {}), indent=2))
        if out.get("pooledRoleAccuracyCurated"):
            print(json.dumps(out["pooledRoleAccuracyCurated"], indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
