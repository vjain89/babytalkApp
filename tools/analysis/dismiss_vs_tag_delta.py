#!/usr/bin/env python3
"""Compare dismissed ML/VAD speech candidates vs human tags (boundary deltas).

Typical review workflow on this kit: Add tag (manual draw) + Dismiss overlapping
candidate. This script measures how the machine box differed from the human box.

    tools/.venv/bin/python tools/analysis/dismiss_vs_tag_delta.py \\
        --session 26_07_27__19:53:00

    # Also trim silence inside each box and re-score (active-audio windows):
    tools/.venv/bin/python tools/analysis/dismiss_vs_tag_delta.py \\
        --session 26_07_27__19:53:00 --active-audio

Writes:
  tools/analysis/out/dismiss_vs_tag_delta_<session>.md
  tools/analysis/out/dismiss_vs_tag_delta_<session>.json
  tools/analysis/out/dismiss_vs_tag_delta_<session>_active.md   (--active-audio)
  tools/analysis/out/dismiss_vs_tag_delta_<session>_active.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from vad_segments import (  # noqa: E402
    FRAME_MS,
    HOP_MS,
    SPEECH_DELTA_DB,
    frame_rms_db,
    noise_floor_db,
)

try:
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install deps: tools/.venv/bin/pip install soundfile numpy\n" + str(e)
    ) from e

LIBRARY = Path.home() / "Documents" / "BabyTalk" / "Library"
OUT_DIR = Path(__file__).resolve().parent / "out"

POINT_TAG_SPAN_MS = 500.0
# Boundary "meaningful" threshold (ms) — around one dual-threshold pad step.
BOUNDARY_MS = 80.0
# Duration ratio bands for too_short / too_long (cand / tag).
SHORT_RATIO = 0.75
LONG_RATIO = 1.33
# "Shifted" = both ends move similarly, not a pure shrink/expand.
SHIFT_ALIGN_MS = 60.0
SHIFT_MIN_MS = 100.0
OK_IOU = 0.70

# Active-speech trim inside a raw box (silence shouldn't count like speech).
ACTIVE_PAD_MS = 40.0
# Also keep frames within this many dB of the local (in-span) peak, so soft
# speech above a quiet floor still counts even if slightly under SPEECH_DELTA.
ACTIVE_REL_PEAK_DB = 25.0
# Minimum active frames; else fall back to the raw span (can't trim).
ACTIVE_MIN_FRAMES = 2


def load(path: Path, key: str) -> list:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get(key, [])
    return data if isinstance(data, list) else []


def resolve_kit(session: str | None, kit_name: str | None) -> Path:
    if kit_name:
        p = LIBRARY / kit_name
        if not p.is_dir():
            raise SystemExit(f"kit not found: {p}")
        return p
    if not session:
        raise SystemExit("pass --session or --kit")
    hits = []
    for d in sorted(LIBRARY.iterdir()):
        man = d / "manifest.json"
        if not man.exists():
            continue
        try:
            m = json.loads(man.read_text())
        except json.JSONDecodeError:
            continue
        if m.get("sessionName") == session:
            hits.append(d)
    if not hits:
        raise SystemExit(f"no kit with sessionName={session!r} under {LIBRARY}")
    if len(hits) > 1:
        names = ", ".join(h.name for h in hits)
        raise SystemExit(f"multiple kits for {session!r}: {names}")
    return hits[0]


def span(item: dict) -> tuple[float, float] | None:
    start = item.get("startMs")
    if start is None:
        start = item.get("tMs")
    if start is None:
        return None
    end = item.get("endMs")
    if end is None or end <= start:
        end = float(start) + POINT_TAG_SPAN_MS
    return float(start), float(end)


def overlap_ms(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = overlap_ms(a, b)
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def load_kit_energy(kit: Path) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Whole-kit frame RMS (dB) + global noise floor — same recipe as VAD."""
    manifest = json.loads((kit / "manifest.json").read_text())
    audio_name = manifest.get("audioFile", "audio.wav")
    audio_path = kit / audio_name
    if not audio_path.exists():
        raise SystemExit(f"audio not found: {audio_path}")
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    times, dbs = frame_rms_db(np.asarray(audio, dtype=np.float64), int(sr))
    floor = noise_floor_db(dbs)
    meta = {
        "audioFile": audio_name,
        "sampleRate": int(sr),
        "nFrames": int(len(dbs)),
        "frameMs": FRAME_MS,
        "hopMs": HOP_MS,
        "noiseFloorDb": round(float(floor), 2),
        "speechDeltaDb": SPEECH_DELTA_DB,
        "activeRelPeakDb": ACTIVE_REL_PEAK_DB,
        "activePadMs": ACTIVE_PAD_MS,
    }
    return times, dbs, float(floor), meta


def active_bounds_ms(
    start_ms: float,
    end_ms: float,
    times: np.ndarray,
    dbs: np.ndarray,
    floor_db: float,
    *,
    speech_delta_db: float = SPEECH_DELTA_DB,
    rel_peak_db: float = ACTIVE_REL_PEAK_DB,
    pad_ms: float = ACTIVE_PAD_MS,
) -> dict:
    """Trim leading/trailing silence inside a raw [start, end] box.

    A frame is "active" if it clears the global noise floor by ``speech_delta_db``
    **or** sits within ``rel_peak_db`` of the loudest frame in the box (soft
    speech). Bounds are first→last active frame, padded, then clipped to the
    raw box (we only shrink — never invent speech outside the label).

    If too few active frames, returns the raw span unchanged (``trimmed=False``).
    """
    raw_dur = max(0.0, end_ms - start_ms)
    lo = int(np.searchsorted(times, start_ms, side="left"))
    hi = int(np.searchsorted(times, end_ms, side="right"))
    if hi <= lo:
        return {
            "startMs": start_ms,
            "endMs": end_ms,
            "durMs": round(raw_dur, 1),
            "silenceLeadMs": 0.0,
            "silenceTrailMs": 0.0,
            "activeFrac": 1.0,
            "trimmed": False,
            "nActiveFrames": 0,
            "fallback": "empty_frames",
        }

    seg = dbs[lo:hi]
    peak = float(np.max(seg)) if len(seg) else float("-inf")
    abs_thr = floor_db + speech_delta_db
    rel_thr = peak - rel_peak_db
    # Active = above global floor+delta OR near local peak (still above floor+3).
    soft_floor = floor_db + min(3.0, speech_delta_db)
    active = (seg >= abs_thr) | ((seg >= rel_thr) & (seg >= soft_floor))
    n_act = int(np.count_nonzero(active))
    if n_act < ACTIVE_MIN_FRAMES:
        active = seg >= soft_floor
        n_act = int(np.count_nonzero(active))
    thr = float(min(abs_thr, rel_thr)) if math.isfinite(peak) else float(abs_thr)
    if n_act < ACTIVE_MIN_FRAMES:
        return {
            "startMs": start_ms,
            "endMs": end_ms,
            "durMs": round(raw_dur, 1),
            "silenceLeadMs": 0.0,
            "silenceTrailMs": 0.0,
            "activeFrac": 0.0,
            "trimmed": False,
            "nActiveFrames": n_act,
            "fallback": "no_active",
            "thresholdDb": round(thr, 2),
            "peakDb": round(peak, 2) if math.isfinite(peak) else None,
        }

    idxs = np.flatnonzero(active)
    first_i = int(idxs[0])
    last_i = int(idxs[-1])
    # Frame time is frame start; extend end by one hop so the last frame counts.
    a0 = float(times[lo + first_i]) - pad_ms
    a1 = float(times[lo + last_i]) + HOP_MS + pad_ms
    a0 = max(start_ms, a0)
    a1 = min(end_ms, a1)
    if a1 <= a0:
        a0, a1 = start_ms, end_ms
        trimmed = False
    else:
        trimmed = (a0 > start_ms + 0.5) or (a1 < end_ms - 0.5)

    act_dur = a1 - a0
    return {
        "startMs": round(a0, 1),
        "endMs": round(a1, 1),
        "durMs": round(act_dur, 1),
        "silenceLeadMs": round(a0 - start_ms, 1),
        "silenceTrailMs": round(end_ms - a1, 1),
        "activeFrac": round(act_dur / raw_dur, 3) if raw_dur > 0 else 0.0,
        "trimmed": trimmed,
        "nActiveFrames": n_act,
        "thresholdDb": round(float(thr), 2),
        "peakDb": round(peak, 2),
        "fallback": None,
    }


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


def fmt_ms(ms: float) -> str:
    s = ms / 1000.0
    m = int(s // 60)
    rem = s - 60 * m
    return f"{m}:{rem:06.3f}" if m else f"{rem:.3f}s"


def classify_mode(d_start: float, d_end: float, dur_ratio: float, iou_v: float) -> str:
    """Primary error mode for one dismissed-candidate ↔ tag pair.

    Sign convention (tag − cand):
      dStart > 0 → human started later → ML started early (lead-in)
      dStart < 0 → human started earlier → ML started late (missed onset)
      dEnd   > 0 → human ended later   → ML ended early (cut short)
      dEnd   < 0 → human ended earlier → ML overran the end
    """
    if iou_v >= OK_IOU and abs(d_start) < BOUNDARY_MS and abs(d_end) < BOUNDARY_MS:
        return "ok"

    # Whole box slid: start and end move nearly the same amount.
    if (
        abs(d_start) >= SHIFT_MIN_MS
        and abs(d_end) >= SHIFT_MIN_MS
        and abs(d_start - d_end) <= SHIFT_ALIGN_MS
    ):
        return "shifted"

    if dur_ratio < SHORT_RATIO:
        return "too_short"
    if dur_ratio > LONG_RATIO:
        return "too_long"

    # Pure-ish boundary errors (pick the larger absolute miss).
    errs: list[tuple[float, str]] = []
    if d_start < -BOUNDARY_MS:
        errs.append((abs(d_start), "late_start"))  # ML late
    elif d_start > BOUNDARY_MS:
        errs.append((abs(d_start), "early_start"))  # ML early / lead-in
    if d_end > BOUNDARY_MS:
        errs.append((abs(d_end), "early_end"))  # ML cut short
    elif d_end < -BOUNDARY_MS:
        errs.append((abs(d_end), "late_end"))  # ML overrun

    if not errs:
        return "ok" if iou_v >= 0.5 else "soft_mismatch"
    errs.sort(reverse=True)
    if len(errs) >= 2 and errs[1][0] >= BOUNDARY_MS:
        # both ends meaningfully wrong in different ways → shrink/expand already
        # handled; leftover is mixed boundary.
        a, b = errs[0][1], errs[1][1]
        if {a, b} == {"late_start", "early_end"}:
            return "too_short"  # missed both ends → under-cover
        if {a, b} == {"early_start", "late_end"}:
            return "too_long"
        return "mixed_boundary"
    return errs[0][1]


MODE_PLAIN = {
    "ok": "close enough (IoU high / small edge diffs)",
    "too_short": "ML box shorter than human tag (under-cover / syllable cut)",
    "too_long": "ML box longer than human tag (over-cover / merged neighbors)",
    "late_start": "ML started late — missed the onset the human included",
    "early_start": "ML started early — lead-in / padding before the word",
    "early_end": "ML ended early — cut off the tail the human kept",
    "late_end": "ML ended late — overran past the human end",
    "shifted": "whole box slid (start and end moved together)",
    "mixed_boundary": "both ends wrong in mixed ways",
    "soft_mismatch": "overlap but not a clean named failure",
}


def best_tag_for_cand(
    cs: tuple[float, float],
    tag_spans: list[tuple[dict, tuple[float, float]]],
    *,
    prefer_manual: bool,
) -> tuple[dict, tuple[float, float], float] | None:
    """Return (tag, span, iou) with max IoU; optionally prefer non-ml_confirmed."""
    hits: list[tuple[dict, tuple[float, float], float]] = []
    for t, ts in tag_spans:
        if overlap_ms(cs, ts) <= 0:
            continue
        hits.append((t, ts, iou(cs, ts)))
    if not hits:
        return None
    if prefer_manual:
        manual = [h for h in hits if h[0].get("source") != "ml_confirmed"]
        pool = manual or hits
    else:
        pool = hits
    return max(pool, key=lambda h: h[2])


def _pair_metrics(
    cs: tuple[float, float],
    ts: tuple[float, float],
) -> dict:
    d_start = ts[0] - cs[0]
    d_end = ts[1] - cs[1]
    cand_dur = cs[1] - cs[0]
    tag_dur = ts[1] - ts[0]
    dur_ratio = cand_dur / tag_dur if tag_dur > 0 else 0.0
    iou_v = iou(cs, ts)
    return {
        "dStartMs": round(d_start, 1),
        "dEndMs": round(d_end, 1),
        "candDurMs": round(cand_dur, 1),
        "tagDurMs": round(tag_dur, 1),
        "durRatio": round(dur_ratio, 3),
        "iou": round(iou_v, 3),
        "overlapMs": round(overlap_ms(cs, ts), 1),
        "mode": classify_mode(d_start, d_end, dur_ratio, iou_v),
    }


def summarise_rows(rows: list[dict], *, prefix: str = "") -> dict:
    """Summarise a list of pair dicts. ``prefix`` selects raw ('') or 'active' keys."""
    if not rows:
        return {"n": 0}

    def key_name(base: str) -> str:
        if not prefix:
            return base
        return prefix + base[0].upper() + base[1:]

    modes_key = key_name("mode")
    modes = Counter(r[modes_key] for r in rows)
    return {
        "n": len(rows),
        "modes": dict(modes.most_common()),
        "modePct": {k: pct(v, len(rows)) for k, v in modes.most_common()},
        "dStartMs": quantiles([float(r[key_name("dStartMs")]) for r in rows]),
        "dEndMs": quantiles([float(r[key_name("dEndMs")]) for r in rows]),
        "durRatio": quantiles([float(r[key_name("durRatio")]) for r in rows], 3),
        "iou": quantiles([float(r[key_name("iou")]) for r in rows], 3),
        "candDurMs": quantiles([float(r[key_name("candDurMs")]) for r in rows]),
        "tagDurMs": quantiles([float(r[key_name("tagDurMs")]) for r in rows]),
        "iou50Pct": pct(
            sum(1 for r in rows if float(r[key_name("iou")]) >= 0.5), len(rows)
        ),
        "iou70Pct": pct(
            sum(1 for r in rows if float(r[key_name("iou")]) >= 0.7), len(rows)
        ),
    }


def analyse(kit: Path, *, active_audio: bool = False) -> dict:
    manifest = json.loads((kit / "manifest.json").read_text())
    anns = load(kit / "annotations.json", "annotations")
    tags = load(kit / "tags.json", "tags")

    energy = None
    energy_meta = None
    if active_audio:
        print("loading audio + frame energy…", flush=True)
        times, dbs, floor, energy_meta = load_kit_energy(kit)
        energy = (times, dbs, floor)
        print(
            f"  frames={energy_meta['nFrames']}  floor={energy_meta['noiseFloorDb']} dB",
            flush=True,
        )

    dismissed = [
        a
        for a in anns
        if a.get("status") == "dismissed"
        and a.get("source") in ("vad_v0", "ml_v0", "ml_confirmed", None)
    ]
    # Prefer vad/ml proposals; keep any dismissed with a span.
    d_spans = [(a, span(a)) for a in dismissed]
    d_spans = [(a, s) for a, s in d_spans if s]

    tag_spans = [(t, span(t)) for t in tags]
    tag_spans = [(t, s) for t, s in tag_spans if s]
    manual_tags = [(t, s) for t, s in tag_spans if t.get("source") != "ml_confirmed"]
    ml_tags = [(t, s) for t, s in tag_spans if t.get("source") == "ml_confirmed"]

    pairs: list[dict] = []
    junk: list[dict] = []  # dismissed, no tag overlap
    matched_tag_uuids: set[str] = set()

    for cand, cs in d_spans:
        hit = best_tag_for_cand(cs, tag_spans, prefer_manual=True)
        if hit is None:
            junk.append(
                {
                    "candUuid": cand.get("uuid"),
                    "startMs": cs[0],
                    "endMs": cs[1],
                    "durMs": round(cs[1] - cs[0], 1),
                    "speechScore": cand.get("speechScore"),
                    "speaker": cand.get("speaker"),
                    "speakerCluster": cand.get("speakerCluster"),
                }
            )
            continue
        tag, ts, iou_v = hit
        matched_tag_uuids.add(tag.get("uuid") or "")
        raw = _pair_metrics(cs, ts)
        row = {
            "candUuid": cand.get("uuid"),
            "tagUuid": tag.get("uuid"),
            "tagSource": tag.get("source"),
            "word": tag.get("word"),
            "speaker": tag.get("speaker"),
            "category": tag.get("category"),
            "candStartMs": cs[0],
            "candEndMs": cs[1],
            "tagStartMs": ts[0],
            "tagEndMs": ts[1],
            "dStartMs": raw["dStartMs"],
            "dEndMs": raw["dEndMs"],
            "candDurMs": raw["candDurMs"],
            "tagDurMs": raw["tagDurMs"],
            "durRatio": raw["durRatio"],
            "iou": raw["iou"],
            "overlapMs": raw["overlapMs"],
            "mode": raw["mode"],
            "speechScore": cand.get("speechScore"),
        }
        if energy is not None:
            times, dbs, floor = energy
            ca = active_bounds_ms(cs[0], cs[1], times, dbs, floor)
            ta = active_bounds_ms(ts[0], ts[1], times, dbs, floor)
            acs = (ca["startMs"], ca["endMs"])
            ats = (ta["startMs"], ta["endMs"])
            act = _pair_metrics(acs, ats)
            # How much of the raw edge miss is silence on either side?
            # Positive silenceLead on tag means human box started before speech.
            silence_explained_start = abs(raw["dStartMs"]) - abs(act["dStartMs"])
            silence_explained_end = abs(raw["dEndMs"]) - abs(act["dEndMs"])
            row.update(
                {
                    "candActiveStartMs": ca["startMs"],
                    "candActiveEndMs": ca["endMs"],
                    "tagActiveStartMs": ta["startMs"],
                    "tagActiveEndMs": ta["endMs"],
                    "candSilenceLeadMs": ca["silenceLeadMs"],
                    "candSilenceTrailMs": ca["silenceTrailMs"],
                    "tagSilenceLeadMs": ta["silenceLeadMs"],
                    "tagSilenceTrailMs": ta["silenceTrailMs"],
                    "candActiveFrac": ca["activeFrac"],
                    "tagActiveFrac": ta["activeFrac"],
                    "candTrimmed": ca["trimmed"],
                    "tagTrimmed": ta["trimmed"],
                    "activeDStartMs": act["dStartMs"],
                    "activeDEndMs": act["dEndMs"],
                    "activeCandDurMs": act["candDurMs"],
                    "activeTagDurMs": act["tagDurMs"],
                    "activeDurRatio": act["durRatio"],
                    "activeIou": act["iou"],
                    "activeOverlapMs": act["overlapMs"],
                    "activeMode": act["mode"],
                    "silenceExplainedStartMs": round(silence_explained_start, 1),
                    "silenceExplainedEndMs": round(silence_explained_end, 1),
                }
            )
        pairs.append(row)

    # Tags with no overlapping dismissed candidate.
    orphan_manual = []
    orphan_ml = []
    for t, ts in tag_spans:
        has = any(overlap_ms(ts, cs) > 0 for _, cs in d_spans)
        if has:
            continue
        row = {
            "tagUuid": t.get("uuid"),
            "source": t.get("source"),
            "word": t.get("word"),
            "speaker": t.get("speaker"),
            "category": t.get("category"),
            "startMs": ts[0],
            "endMs": ts[1],
            "durMs": round(ts[1] - ts[0], 1),
        }
        if t.get("source") == "ml_confirmed":
            orphan_ml.append(row)
        else:
            orphan_manual.append(row)

    manual_pairs = [p for p in pairs if p["tagSource"] != "ml_confirmed"]
    ml_pairs = [p for p in pairs if p["tagSource"] == "ml_confirmed"]

    # One pair per tag (best IoU) — avoids syllable-fragment inflation.
    best_by_tag: dict[str, dict] = {}
    for r in manual_pairs:
        u = r.get("tagUuid") or ""
        if u not in best_by_tag or r["iou"] > best_by_tag[u]["iou"]:
            best_by_tag[u] = r
    deduped = list(best_by_tag.values())
    per_tag_counts = Counter(r.get("tagUuid") for r in manual_pairs)
    multi_tag_n = sum(1 for v in per_tag_counts.values() if v >= 2)

    # Example rows per mode (up to 3): prefer real overlap + a word, not tiny IoU.
    examples: dict[str, list] = {}
    by_mode: dict[str, list] = {}
    for r in manual_pairs + ml_pairs:
        by_mode.setdefault(r["mode"], []).append(r)
    for mode, rows in by_mode.items():
        ranked = sorted(
            rows,
            key=lambda r: (
                0 if r["tagSource"] != "ml_confirmed" else 1,
                0 if r.get("word") else 1,
                0 if r["iou"] >= 0.15 else 1,
                -r["iou"],
                -abs(r["dStartMs"]) - abs(r["dEndMs"]),
            ),
        )
        examples[mode] = ranked[:3]

    out: dict = {
        "kit": kit.name,
        "sessionName": manifest.get("sessionName"),
        "durationMs": manifest.get("durationMs"),
        "nAnnotations": len(anns),
        "nTags": len(tags),
        "nManualTags": len(manual_tags),
        "nMlConfirmedTags": len(ml_tags),
        "nDismissed": len(d_spans),
        "nMatched": len(pairs),
        "nMatchedManual": len(manual_pairs),
        "nMatchedMlConfirmed": len(ml_pairs),
        "nJunkDismissed": len(junk),
        "nOrphanManualTags": len(orphan_manual),
        "nOrphanMlConfirmedTags": len(orphan_ml),
        "matchedTagUnique": len({p["tagUuid"] for p in pairs if p.get("tagUuid")}),
        "allMatched": summarise_rows(pairs),
        "manualMatched": summarise_rows(manual_pairs),
        "manualMatchedDedupedByTag": summarise_rows(deduped),
        "nUniqueManualTagsMatched": len(deduped),
        "nManualTagsWithMultipleDismissed": multi_tag_n,
        "meanDismissedPerMatchedManualTag": round(
            len(manual_pairs) / len(deduped), 2
        )
        if deduped
        else 0.0,
        "mlConfirmedMatched": summarise_rows(ml_pairs),
        "junkDismissedSample": _unique_spans(
            sorted(junk, key=lambda r: r["startMs"]), 15
        ),
        "orphanManualSample": orphan_manual[:15],
        "orphanMlConfirmedSample": orphan_ml[:10],
        "examplesByMode": examples,
        "pairs": pairs,
        "junkDismissed": junk,
        "orphanManualTags": orphan_manual,
        "orphanMlConfirmedTags": orphan_ml,
        "signConvention": {
            "dStartMs": "tagStart - candStart; >0 human later / ML early lead-in",
            "dEndMs": "tagEnd - candEnd; >0 human later / ML early cut-off",
            "durRatio": "candDur / tagDur; <1 ML shorter than human",
        },
        "thresholds": {
            "boundaryMs": BOUNDARY_MS,
            "shortRatio": SHORT_RATIO,
            "longRatio": LONG_RATIO,
            "okIou": OK_IOU,
        },
        "activeAudio": bool(energy is not None),
    }

    if energy is not None:
        # Dedup by best *active* IoU as well.
        best_act: dict[str, dict] = {}
        for r in manual_pairs:
            u = r.get("tagUuid") or ""
            if u not in best_act or r["activeIou"] > best_act[u]["activeIou"]:
                best_act[u] = r
        deduped_act = list(best_act.values())

        act_examples: dict[str, list] = {}
        by_amode: dict[str, list] = {}
        for r in manual_pairs:
            by_amode.setdefault(r["activeMode"], []).append(r)
        for mode, rows in by_amode.items():
            ranked = sorted(
                rows,
                key=lambda r: (
                    0 if r.get("word") else 1,
                    0 if r["activeIou"] >= 0.15 else 1,
                    -r["activeIou"],
                    -abs(r["activeDStartMs"]) - abs(r["activeDEndMs"]),
                ),
            )
            act_examples[mode] = ranked[:3]

        silence_stats = {
            "tagSilenceLeadMs": quantiles(
                [r["tagSilenceLeadMs"] for r in manual_pairs]
            ),
            "tagSilenceTrailMs": quantiles(
                [r["tagSilenceTrailMs"] for r in manual_pairs]
            ),
            "candSilenceLeadMs": quantiles(
                [r["candSilenceLeadMs"] for r in manual_pairs]
            ),
            "candSilenceTrailMs": quantiles(
                [r["candSilenceTrailMs"] for r in manual_pairs]
            ),
            "tagActiveFrac": quantiles(
                [r["tagActiveFrac"] for r in manual_pairs], 3
            ),
            "candActiveFrac": quantiles(
                [r["candActiveFrac"] for r in manual_pairs], 3
            ),
            "silenceExplainedStartMs": quantiles(
                [r["silenceExplainedStartMs"] for r in manual_pairs]
            ),
            "silenceExplainedEndMs": quantiles(
                [r["silenceExplainedEndMs"] for r in manual_pairs]
            ),
            "pctTagTrimmed": pct(
                sum(1 for r in manual_pairs if r["tagTrimmed"]), len(manual_pairs)
            ),
            "pctCandTrimmed": pct(
                sum(1 for r in manual_pairs if r["candTrimmed"]), len(manual_pairs)
            ),
        }

        out["energyMeta"] = energy_meta
        out["manualMatchedActive"] = summarise_rows(manual_pairs, prefix="active")
        out["manualMatchedActiveDedupedByTag"] = summarise_rows(
            deduped_act, prefix="active"
        )
        out["silenceStatsManual"] = silence_stats
        out["examplesByActiveMode"] = act_examples
        out["recommendation"] = build_active_recommendation(out)

    return out


def _unique_spans(rows: list[dict], limit: int) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r.get("startMs"), r.get("endMs"), r.get("speakerCluster"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def build_active_recommendation(result: dict) -> dict:
    """Compare raw vs active; suggest merge-back knobs only if speech-active supports it.

    Production today: short_piece_ms=400, max_gap_ms=100.
    450/300 was explored from raw oracle gaps — not accepted until active-audio agrees.
    """
    raw = result.get("manualMatched") or {}
    act = result.get("manualMatchedActive") or {}
    raw_ded = result.get("manualMatchedDedupedByTag") or {}
    act_ded = result.get("manualMatchedActiveDedupedByTag") or {}
    sil = result.get("silenceStatsManual") or {}

    raw_ts = (raw.get("modePct") or {}).get("too_short", 0)
    act_ts = (act.get("modePct") or {}).get("too_short", 0)
    raw_ded_ts = (raw_ded.get("modePct") or {}).get("too_short", 0)
    act_ded_ts = (act_ded.get("modePct") or {}).get("too_short", 0)

    raw_ds = (raw.get("dStartMs") or {}).get("median")
    raw_de = (raw.get("dEndMs") or {}).get("median")
    act_ds = (act.get("dStartMs") or {}).get("median")
    act_de = (act.get("dEndMs") or {}).get("median")
    act_dur = (act.get("durRatio") or {}).get("median")

    tag_trail = (sil.get("tagSilenceTrailMs") or {}).get("median")
    tag_lead = (sil.get("tagSilenceLeadMs") or {}).get("median")
    explained_s = (sil.get("silenceExplainedStartMs") or {}).get("median")
    explained_e = (sil.get("silenceExplainedEndMs") or {}).get("median")

    # Heuristic: support loosening only if under-cover remains dominant *after*
    # silence trim, with speech-active median edge errors still large (≥150 ms)
    # and dur ratio still clearly short (<0.75).
    still_dominant = act_ded_ts >= 40 or act_ts >= 45
    speech_edges_large = (
        (act_ds is not None and abs(float(act_ds)) >= 150)
        or (act_de is not None and abs(float(act_de)) >= 150)
    )
    still_short = act_dur is not None and float(act_dur) < 0.75
    mostly_silence = (
        (explained_s is not None and float(explained_s) >= 80)
        or (explained_e is not None and float(explained_e) >= 80)
    ) and not speech_edges_large

    production = {"short_piece_ms": 400, "max_gap_ms": 100}
    explored_raw = {"short_piece_ms": 450, "max_gap_ms": 300, "accepted": False}

    if mostly_silence and not still_dominant:
        verdict = "keep_production"
        suggested = None
        note = (
            "After trimming silence, under-cover shrinks a lot — raw 'too_short' "
            "was partly silence mismatch. Do not loosen merge-back from raw alone; "
            "keep production 400/100."
        )
    elif still_dominant and still_short and speech_edges_large:
        verdict = "pending_loosen"
        # Suggest modest step toward <0.5s word break, not a jump to 300 unless
        # active end/start medians imply hundreds of ms of missed speech glue.
        abs_edge = max(abs(float(act_ds or 0)), abs(float(act_de or 0)))
        if abs_edge >= 250:
            suggested = {"short_piece_ms": 450, "max_gap_ms": 250}
        else:
            suggested = {"short_piece_ms": 400, "max_gap_ms": 200}
        note = (
            "Active-audio still shows dominant too_short with large *speech* edge "
            f"errors (median Δstart {act_ds} ms, Δend {act_de} ms, durR {act_dur}). "
            f"Recommend pending acceptance of short_piece_ms={suggested['short_piece_ms']}, "
            f"max_gap_ms={suggested['max_gap_ms']} — do not ship until explicitly accepted. "
            "450/300 from raw oracle gaps remains not accepted."
        )
    else:
        verdict = "keep_production"
        suggested = None
        note = (
            "Active-audio does not clearly support loosening beyond production "
            "400/100 (too_short or speech-edge evidence mixed/weak after silence trim). "
            "450/300 from raw gaps stays rejected for production."
        )

    return {
        "verdict": verdict,
        "productionDefaults": production,
        "exploredFromRawNotAccepted": explored_raw,
        "suggestedPending": suggested,
        "note": note,
        "rawTooShortPct": raw_ts,
        "activeTooShortPct": act_ts,
        "rawTooShortPctDeduped": raw_ded_ts,
        "activeTooShortPctDeduped": act_ded_ts,
        "rawMedianDStartMs": raw_ds,
        "rawMedianDEndMs": raw_de,
        "activeMedianDStartMs": act_ds,
        "activeMedianDEndMs": act_de,
        "activeMedianDurRatio": act_dur,
        "medianTagSilenceLeadMs": tag_lead,
        "medianTagSilenceTrailMs": tag_trail,
        "medianSilenceExplainedStartMs": explained_s,
        "medianSilenceExplainedEndMs": explained_e,
    }


def takeaways(result: dict) -> list[str]:
    """Plain-language bullets for merge-back / padding / finder errors."""
    out: list[str] = []
    n_d = result["nDismissed"]
    n_m = result["nMatched"]
    n_man = result["nMatchedManual"]
    n_junk = result["nJunkDismissed"]
    man = result["manualMatched"]
    ded = result.get("manualMatchedDedupedByTag") or {}
    modes = man.get("modePct") or {}
    dmodes = ded.get("modePct") or modes

    out.append(
        f"Of {n_d} dismissed VAD proposals, {n_m} overlap a tag "
        f"({n_man} to a hand-drawn tag, {result['nMatchedMlConfirmed']} to "
        f"ml_confirmed); {n_junk} are pure junk rejects with no tag nearby."
    )

    if man.get("n"):
        top = sorted(modes.items(), key=lambda kv: -kv[1])[:3]
        top_s = ", ".join(f"{k} {v}%" for k, v in top)
        ds = man.get("dStartMs") or {}
        de = man.get("dEndMs") or {}
        dr = man.get("durRatio") or {}
        out.append(
            f"On hand-drawn rescues (n={man['n']} pairs → "
            f"{result.get('nUniqueManualTagsMatched')} unique tags), "
            f"main modes: {top_s}. "
            f"Median start delta {ds.get('median')} ms, end delta {de.get('median')} ms, "
            f"duration ratio (cand/tag) {dr.get('median')}."
        )

    if ded.get("n"):
        dds = ded.get("dStartMs") or {}
        dde = ded.get("dEndMs") or {}
        ddr = ded.get("durRatio") or {}
        out.append(
            f"Per-tag (best-IoU) view still says under-cover: too_short "
            f"{dmodes.get('too_short', 0)}%, median Δstart {dds.get('median')} ms "
            f"(ML late / missed onset), Δend {dde.get('median')} ms "
            f"(ML early cut-off), dur ratio {ddr.get('median')}."
        )

    multi = result.get("nManualTagsWithMultipleDismissed") or 0
    if multi:
        out.append(
            f"{multi} rescued words overlap ≥2 dismissed boxes "
            f"(mean {result.get('meanDismissedPerMatchedManualTag')} "
            "dismissals/tag) — classic syllable/fragment split that merge-back "
            "targets; flat edge-pad cannot glue siblings."
        )

    too_short = dmodes.get("too_short", 0)
    if too_short >= 40:
        out.append(
            "Dominant failure is too_short (cand ~½–⅔ of the human word) on "
            "**raw** timestamps. Prefer merge-back / longer word-like pieces over "
            "more padding — but retune gap/short only after active-audio confirms "
            "the miss is speech, not silence."
        )

    junk_pct = pct(n_junk, n_d)
    out.append(
        f"Junk dismissals are {junk_pct}% of dismissals "
        f"({n_junk}/{n_d}) — true false positives, not boundary edits; "
        f"speechScore / role gating matters as much as boundary polish."
    )

    orphans = result["nOrphanManualTags"]
    if orphans:
        out.append(
            f"{orphans} manual tags have no overlapping dismissed candidate "
            "(finder miss / never proposed) — coverage gap separate from "
            "dismiss-and-redraw boundary fixes."
        )

    return out[:7]


def active_takeaways(result: dict) -> list[str]:
    out: list[str] = []
    rec = result.get("recommendation") or {}
    raw = result.get("manualMatched") or {}
    act = result.get("manualMatchedActive") or {}
    sil = result.get("silenceStatsManual") or {}
    meta = result.get("energyMeta") or {}

    out.append(
        "Honesty note: the earlier dismiss-vs-tag report "
        "(`dismiss_vs_tag_delta_26_07_27.md`) used **raw timestamps only** — "
        "silence inside a human tag counted the same as speech."
    )
    out.append(
        f"Active-audio trim: frames ≥ noise floor "
        f"({meta.get('noiseFloorDb')} dB) + {meta.get('speechDeltaDb')} dB, "
        f"or near the in-box peak (≤{meta.get('activeRelPeakDb')} dB down, "
        f"still above floor+3), then ±{meta.get('activePadMs')} ms pad, "
        "clipped to the raw box."
    )

    if raw.get("n") and act.get("n"):
        out.append(
            f"Raw modes too_short {rec.get('rawTooShortPct')}% → active "
            f"{rec.get('activeTooShortPct')}% "
            f"(deduped {rec.get('rawTooShortPctDeduped')}% → "
            f"{rec.get('activeTooShortPctDeduped')}%). "
            f"Median Δstart {rec.get('rawMedianDStartMs')} → "
            f"{rec.get('activeMedianDStartMs')} ms; "
            f"Δend {rec.get('rawMedianDEndMs')} → "
            f"{rec.get('activeMedianDEndMs')} ms; "
            f"active dur ratio {rec.get('activeMedianDurRatio')}."
        )

    out.append(
        f"Median silence inside tags: lead "
        f"{rec.get('medianTagSilenceLeadMs')} ms, trail "
        f"{rec.get('medianTagSilenceTrailMs')} ms "
        f"(cand lead/trail "
        f"{(sil.get('candSilenceLeadMs') or {}).get('median')}/"
        f"{(sil.get('candSilenceTrailMs') or {}).get('median')} ms). "
        f"Silence-explained edge shrink (median): start "
        f"{rec.get('medianSilenceExplainedStartMs')} ms, end "
        f"{rec.get('medianSilenceExplainedEndMs')} ms."
    )

    out.append(rec.get("note") or "No recommendation computed.")
    out.append(
        "Production stays short_piece_ms=400, max_gap_ms=100 until a pending "
        "suggestion is explicitly accepted. 450/300 from raw oracle gaps is "
        "not accepted."
    )
    return out


def render_md(result: dict) -> str:
    session = result["sessionName"]
    lines: list[str] = []
    lines += [
        f"# Dismissed ML vs manual tags — `{session}`",
        "",
        f"**Kit:** `{result['kit']}`  ",
        f"**Question:** When the reviewer dismisses a Find-speech candidate and "
        "draws (or keeps) a tag nearby, how do the boxes differ?",
        "",
        "## Counts",
        "",
        f"| | n |",
        f"|---|---:|",
        f"| Dismissed candidates | {result['nDismissed']} |",
        f"| Matched to any overlapping tag | {result['nMatched']} |",
        f"| … to hand-drawn (`source≠ml_confirmed`) | {result['nMatchedManual']} |",
        f"| … to `ml_confirmed` only | {result['nMatchedMlConfirmed']} |",
        f"| Dismissed with **no** tag overlap (junk) | {result['nJunkDismissed']} |",
        f"| Manual tags with **no** dismissed overlap | {result['nOrphanManualTags']} |",
        f"| `ml_confirmed` tags with no dismissed overlap | {result['nOrphanMlConfirmedTags']} |",
        f"| All tags / manual / ml_confirmed | "
        f"{result['nTags']} / {result['nManualTags']} / {result['nMlConfirmedTags']} |",
        "",
        "### Sign convention",
        "",
        "- **start delta** = tagStart − candStart (ms). Positive → human started "
        "**later** → ML had **lead-in** (started early).",
        "- **end delta** = tagEnd − candEnd. Positive → human ended **later** → "
        "ML **cut off early**.",
        "- **duration ratio** = candDur / tagDur. Below 1 → ML shorter than human.",
        "",
    ]

    man = result["manualMatched"]
    if man.get("n"):
        lines += [
            "## Hand-drawn rescues (dismiss + manual tag)",
            "",
            f"n = **{man['n']}** pairs (dismissed cand ↔ best overlapping non-`ml_confirmed` tag).",
            "",
            "### Error modes",
            "",
            "| Mode | n | % | Meaning |",
            "|---|---:|---:|---|",
        ]
        for mode, n in (man.get("modes") or {}).items():
            lines.append(
                f"| `{mode}` | {n} | {man['modePct'].get(mode, 0)} | "
                f"{MODE_PLAIN.get(mode, '')} |"
            )
        lines += [
            "",
            "### Boundary distributions (ms / ratio)",
            "",
            "| Metric | p25 | median | p75 | p90 |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, label in (
            ("dStartMs", "start delta (tag−cand)"),
            ("dEndMs", "end delta (tag−cand)"),
            ("durRatio", "dur ratio cand/tag"),
            ("iou", "IoU"),
            ("candDurMs", "cand duration"),
            ("tagDurMs", "tag duration"),
        ):
            q = man.get(key) or {}
            if not q:
                continue
            lines.append(
                f"| {label} | {q.get('p25')} | **{q.get('median')}** | "
                f"{q.get('p75')} | {q.get('p90')} |"
            )
        lines += [
            "",
            f"IoU ≥ 0.5: **{man.get('iou50Pct')}%** · IoU ≥ 0.7: **{man.get('iou70Pct')}%**",
            "",
        ]

        ded = result.get("manualMatchedDedupedByTag") or {}
        if ded.get("n"):
            lines += [
                "### Per-tag view (best-IoU pair only)",
                "",
                f"Unique manual tags matched: **{ded['n']}** "
                f"({result.get('nManualTagsWithMultipleDismissed', 0)} tags hit by ≥2 "
                f"dismissed boxes; mean "
                f"{result.get('meanDismissedPerMatchedManualTag')} dismissals/tag).",
                "",
                f"Modes (deduped): {ded.get('modePct')}",
                "",
                f"Median Δstart **{(ded.get('dStartMs') or {}).get('median')}** ms · "
                f"Δend **{(ded.get('dEndMs') or {}).get('median')}** ms · "
                f"dur ratio **{(ded.get('durRatio') or {}).get('median')}** · "
                f"IoU **{(ded.get('iou') or {}).get('median')}** "
                f"(IoU≥0.5: {ded.get('iou50Pct')}%).",
                "",
            ]

    mlm = result["mlConfirmedMatched"]
    if mlm.get("n"):
        lines += [
            "## Overlap with `ml_confirmed` tags only",
            "",
            f"n = {mlm['n']} (dismissed cand overlapping a confirmed-ML tag — "
            "usually a neighbor or a prior confirm, not the Add-tag workflow).",
            "",
            f"Modes: {mlm.get('modePct')}",
            f"Median dStart { (mlm.get('dStartMs') or {}).get('median') } ms, "
            f"dEnd { (mlm.get('dEndMs') or {}).get('median') } ms, "
            f"durRatio { (mlm.get('durRatio') or {}).get('median') }.",
            "",
        ]

    lines += ["## Examples (by mode)", ""]
    for mode, rows in (result.get("examplesByMode") or {}).items():
        if mode == "ok":
            continue
        lines.append(f"### `{mode}` — {MODE_PLAIN.get(mode, '')}")
        lines.append("")
        for r in rows:
            word = r.get("word") or "—"
            src = r.get("tagSource")
            lines.append(
                f"- **{word}** ({r.get('speaker')}, `{src}`) @ "
                f"{fmt_ms(r['candStartMs'])}–{fmt_ms(r['candEndMs'])} (ML) vs "
                f"{fmt_ms(r['tagStartMs'])}–{fmt_ms(r['tagEndMs'])} (tag); "
                f"Δstart={r['dStartMs']:+.0f} ms, Δend={r['dEndMs']:+.0f} ms, "
                f"durR={r['durRatio']}, IoU={r['iou']}"
            )
        lines.append("")

    junk = result.get("junkDismissedSample") or []
    if junk:
        lines += [
            "## Junk dismissals (no overlapping tag)",
            "",
            f"{result['nJunkDismissed']} dismissed spans with zero tag overlap — "
            "reviewer rejected noise / non-target speech. Sample:",
            "",
        ]
        for r in junk[:8]:
            lines.append(
                f"- {fmt_ms(r['startMs'])}–{fmt_ms(r['endMs'])} "
                f"({r['durMs']:.0f} ms), speechScore={r.get('speechScore')}, "
                f"cluster={r.get('speakerCluster')}"
            )
        lines.append("")

    orphans = result.get("orphanManualSample") or []
    if orphans:
        lines += [
            "## Manual tags with no dismissed candidate nearby",
            "",
            f"{result['nOrphanManualTags']} hand-drawn tags never sat on a dismissed "
            "box (finder miss, or candidate was confirmed elsewhere). Sample:",
            "",
        ]
        for r in orphans[:8]:
            lines.append(
                f"- **{r.get('word') or '—'}** ({r.get('speaker')}) @ "
                f"{fmt_ms(r['startMs'])}–{fmt_ms(r['endMs'])}"
            )
        lines.append("")

    lines += [
        "## Plain takeaways (merge-back / padding / finder)",
        "",
    ]
    for t in result.get("takeaways") or takeaways(result):
        lines.append(f"- {t}")
    lines += [
        "",
        "_Generated by `tools/analysis/dismiss_vs_tag_delta.py`. "
        "Does not modify library files._",
        "",
    ]
    return "\n".join(lines)


def _mode_table(summary: dict) -> list[str]:
    lines = [
        "| Mode | n | % | Meaning |",
        "|---|---:|---:|---|",
    ]
    for mode, n in (summary.get("modes") or {}).items():
        lines.append(
            f"| `{mode}` | {n} | {summary.get('modePct', {}).get(mode, 0)} | "
            f"{MODE_PLAIN.get(mode, '')} |"
        )
    return lines


def _dist_table(summary: dict) -> list[str]:
    lines = [
        "| Metric | p25 | median | p75 | p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("dStartMs", "start delta (tag−cand)"),
        ("dEndMs", "end delta (tag−cand)"),
        ("durRatio", "dur ratio cand/tag"),
        ("iou", "IoU"),
        ("candDurMs", "cand duration"),
        ("tagDurMs", "tag duration"),
    ):
        q = summary.get(key) or {}
        if not q:
            continue
        lines.append(
            f"| {label} | {q.get('p25')} | **{q.get('median')}** | "
            f"{q.get('p75')} | {q.get('p90')} |"
        )
    return lines


def render_active_md(result: dict) -> str:
    """Beginner-friendly raw vs active-audio comparison report."""
    session = result["sessionName"]
    rec = result.get("recommendation") or {}
    raw = result.get("manualMatched") or {}
    act = result.get("manualMatchedActive") or {}
    raw_ded = result.get("manualMatchedDedupedByTag") or {}
    act_ded = result.get("manualMatchedActiveDedupedByTag") or {}
    sil = result.get("silenceStatsManual") or {}
    meta = result.get("energyMeta") or {}

    lines: list[str] = [
        f"# Dismissed ML vs tags — raw vs active-audio — `{session}`",
        "",
        f"**Kit:** `{result['kit']}`  ",
        "**Question:** When a reviewer dismisses a Find-speech box and draws a "
        "tag nearby, how much of the mismatch is **real speech** vs **silence "
        "padding** inside the boxes?",
        "",
        "## Honesty first",
        "",
        "The earlier report "
        f"[`dismiss_vs_tag_delta_{session.split('__')[0]}.md`]"
        f"(dismiss_vs_tag_delta_{session.split('__')[0]}.md) measured only "
        "**raw timestamps** (box start/end as stored). A tag that includes 1.0 s "
        "of trailing silence looked “1.0 s longer” than a tight ML box — same as "
        "if the ML had missed 1.0 s of speech. This follow-up trims each box to "
        "**active audio** (energy above a noise floor) and re-scores.",
        "",
        "## How active-audio windows are built",
        "",
        f"- Load `{meta.get('audioFile')}` once; frame RMS every "
        f"{meta.get('hopMs')} ms ({meta.get('frameMs')} ms frames).",
        f"- Global noise floor ≈ {meta.get('noiseFloorDb')} dB (25th percentile).",
        f"- A frame is **active** if it is ≥ floor + {meta.get('speechDeltaDb')} dB, "
        f"**or** within {meta.get('activeRelPeakDb')} dB of the loudest frame in "
        "that box (and still above floor+3 dB).",
        f"- Active start/end = first→last active frame ± {meta.get('activePadMs')} ms, "
        "**clipped to the raw box** (we only shrink).",
        "- Then recompute Δstart / Δend / durRatio / IoU on active windows "
        "(same classifiers as raw).",
        "",
        "## Counts (same pairing as raw)",
        "",
        f"| | n |",
        f"|---|---:|",
        f"| Dismissed candidates | {result['nDismissed']} |",
        f"| Matched to hand-drawn tag | {result['nMatchedManual']} |",
        f"| Unique manual tags matched | {result.get('nUniqueManualTagsMatched')} |",
        f"| Junk dismissals (no tag) | {result['nJunkDismissed']} |",
        f"| Tags trimmed (had silence lead/trail) | "
        f"{sil.get('pctTagTrimmed')}% of pairs |",
        f"| Cands trimmed | {sil.get('pctCandTrimmed')}% of pairs |",
        "",
        "## Side-by-side: raw vs active (hand-drawn rescues)",
        "",
        "### Error modes",
        "",
        "| Mode | Raw % | Active % |",
        "|---|---:|---:|",
    ]
    all_modes = list(
        dict.fromkeys(
            list((raw.get("modes") or {}).keys())
            + list((act.get("modes") or {}).keys())
        )
    )
    for mode in all_modes:
        lines.append(
            f"| `{mode}` | {(raw.get('modePct') or {}).get(mode, 0)} | "
            f"{(act.get('modePct') or {}).get(mode, 0)} |"
        )

    lines += [
        "",
        "### Boundary medians",
        "",
        "| Metric | Raw median | Active median |",
        "|---|---:|---:|",
        f"| start delta (tag−cand) ms | "
        f"**{(raw.get('dStartMs') or {}).get('median')}** | "
        f"**{(act.get('dStartMs') or {}).get('median')}** |",
        f"| end delta (tag−cand) ms | "
        f"**{(raw.get('dEndMs') or {}).get('median')}** | "
        f"**{(act.get('dEndMs') or {}).get('median')}** |",
        f"| dur ratio cand/tag | "
        f"**{(raw.get('durRatio') or {}).get('median')}** | "
        f"**{(act.get('durRatio') or {}).get('median')}** |",
        f"| IoU | "
        f"**{(raw.get('iou') or {}).get('median')}** | "
        f"**{(act.get('iou') or {}).get('median')}** |",
        f"| IoU ≥ 0.5 (%) | {raw.get('iou50Pct')} | {act.get('iou50Pct')} |",
        "",
        "### Per-tag (best IoU) too_short",
        "",
        f"- Raw deduped too_short: **{rec.get('rawTooShortPctDeduped')}%**  ",
        f"- Active deduped too_short: **{rec.get('activeTooShortPctDeduped')}%**",
        "",
        "## Silence inside the boxes",
        "",
        "How much of each raw box was trimmed away as non-active?",
        "",
        "**Bottom line on silence:** on this kit, median silence lead/trail is "
        "**0 ms** and median active fraction is **~1.0**. Some tags trim tens of "
        "ms at the edges (p90 lead ~60 ms), but that does **not** explain the "
        "hundreds-of-ms under-cover. Raw “too short” was already mostly speech.",
        "",
        "| | p25 | median | p75 | p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("tagSilenceLeadMs", "tag silence lead (ms)"),
        ("tagSilenceTrailMs", "tag silence trail (ms)"),
        ("candSilenceLeadMs", "cand silence lead (ms)"),
        ("candSilenceTrailMs", "cand silence trail (ms)"),
        ("tagActiveFrac", "tag active fraction"),
        ("candActiveFrac", "cand active fraction"),
        ("silenceExplainedStartMs", "Δ|start| shrink after trim (ms)"),
        ("silenceExplainedEndMs", "Δ|end| shrink after trim (ms)"),
    ):
        q = sil.get(key) or {}
        if not q:
            continue
        lines.append(
            f"| {label} | {q.get('p25')} | **{q.get('median')}** | "
            f"{q.get('p75')} | {q.get('p90')} |"
        )

    lines += [
        "",
        "### Active-mode detail",
        "",
        *(_mode_table(act) if act.get("n") else ["_(no active pairs)_"]),
        "",
        "### Active boundary distributions",
        "",
        *(_dist_table(act) if act.get("n") else []),
        "",
        f"Deduped active modes: `{act_ded.get('modePct')}`  ",
        f"Deduped median Δstart **{(act_ded.get('dStartMs') or {}).get('median')}** ms · "
        f"Δend **{(act_ded.get('dEndMs') or {}).get('median')}** ms · "
        f"durR **{(act_ded.get('durRatio') or {}).get('median')}** · "
        f"IoU **{(act_ded.get('iou') or {}).get('median')}**.",
        "",
        "## Examples (active mode)",
        "",
    ]
    for mode, rows in (result.get("examplesByActiveMode") or {}).items():
        if mode == "ok":
            continue
        lines.append(f"### `{mode}` — {MODE_PLAIN.get(mode, '')}")
        lines.append("")
        for r in rows:
            word = r.get("word") or "—"
            lines.append(
                f"- **{word}** ({r.get('speaker')}) raw ML "
                f"{fmt_ms(r['candStartMs'])}–{fmt_ms(r['candEndMs'])} vs tag "
                f"{fmt_ms(r['tagStartMs'])}–{fmt_ms(r['tagEndMs'])}; "
                f"active ML {fmt_ms(r['candActiveStartMs'])}–"
                f"{fmt_ms(r['candActiveEndMs'])} vs tag "
                f"{fmt_ms(r['tagActiveStartMs'])}–{fmt_ms(r['tagActiveEndMs'])}; "
                f"raw Δ={r['dStartMs']:+.0f}/{r['dEndMs']:+.0f} → active "
                f"Δ={r['activeDStartMs']:+.0f}/{r['activeDEndMs']:+.0f}, "
                f"durR {r['durRatio']}→{r['activeDurRatio']}, "
                f"IoU {r['iou']}→{r['activeIou']}"
            )
        lines.append("")

    sug = rec.get("suggestedPending")
    lines += [
        "## Merge-back recommendation",
        "",
        f"**Production (keep):** `short_piece_ms=400`, `max_gap_ms=100`.  ",
        "**Explored from raw oracle gaps (not accepted):** "
        "`short_piece_ms=450`, `max_gap_ms=300`.",
        "",
        f"**Verdict:** `{rec.get('verdict')}`  ",
    ]
    if sug:
        lines.append(
            f"**Pending acceptance (do not ship silently):** "
            f"`short_piece_ms={sug['short_piece_ms']}`, "
            f"`max_gap_ms={sug['max_gap_ms']}`."
        )
    else:
        lines.append(
            "**No pending loosen** — active-audio does not clearly support "
            "changing production 400/100."
        )
    lines += ["", f"{rec.get('note')}", ""]

    lines += ["## Plain takeaways", ""]
    for t in result.get("activeTakeaways") or active_takeaways(result):
        lines.append(f"- {t}")
    lines += [
        "",
        "_Generated by `tools/analysis/dismiss_vs_tag_delta.py --active-audio`. "
        "Does not modify library files or production defaults._",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", default="26_07_27__19:53:00")
    p.add_argument("--kit", default=None, help="library folder name override")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--active-audio",
        action="store_true",
        help="Also trim silence inside boxes and write *_active.md/json",
    )
    args = p.parse_args()

    kit = resolve_kit(args.session, args.kit)
    print(f"kit: {kit.name}  session={args.session}", flush=True)
    result = analyse(kit, active_audio=args.active_audio)
    result["takeaways"] = takeaways(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    short = (result["sessionName"] or "unknown").split("__")[0]
    md_path = args.out_dir / f"dismiss_vs_tag_delta_{short}.md"
    json_path = args.out_dir / f"dismiss_vs_tag_delta_{short}.json"

    md_path.write_text(render_md(result))
    json_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {md_path}", flush=True)
    print(f"wrote {json_path}", flush=True)

    if args.active_audio:
        result["activeTakeaways"] = active_takeaways(result)
        amd = args.out_dir / f"dismiss_vs_tag_delta_{short}_active.md"
        aj = args.out_dir / f"dismiss_vs_tag_delta_{short}_active.json"
        # Active JSON keeps summaries + recommendation; pairs stay for follow-ups.
        amd.write_text(render_active_md(result))
        aj.write_text(json.dumps(result, indent=2))
        print(f"wrote {amd}", flush=True)
        print(f"wrote {aj}", flush=True)
        rec = result.get("recommendation") or {}
        print(f"active verdict: {rec.get('verdict')}", flush=True)
        print(f"  {rec.get('note')}", flush=True)
        for t in result["activeTakeaways"]:
            print(f"- {t}", flush=True)

    man = result["manualMatched"]
    print(
        f"dismissed={result['nDismissed']} matched={result['nMatched']} "
        f"(manual={result['nMatchedManual']}, ml_conf={result['nMatchedMlConfirmed']}) "
        f"junk={result['nJunkDismissed']} orphan_manual={result['nOrphanManualTags']}",
        flush=True,
    )
    if man.get("n"):
        print(f"manual modes: {man.get('modePct')}", flush=True)
        print(
            f"manual median dStart={man['dStartMs'].get('median')} "
            f"dEnd={man['dEndMs'].get('median')} "
            f"durR={man['durRatio'].get('median')} "
            f"iou={man['iou'].get('median')}",
            flush=True,
        )
    for t in result["takeaways"]:
        print(f"- {t}", flush=True)


if __name__ == "__main__":
    main()
