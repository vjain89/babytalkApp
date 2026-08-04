#!/usr/bin/env python3
"""Local snippet-finding lab (band-pass + VAD/merge-back knobs) — not production Review.

Fixed kit by default (sessionName). Does not write annotations.json / tags.json.
Does not share process or routes with review_server.py.

    tools/.venv/bin/python tools/snippet_explorer_server.py
    open http://127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR / "analysis"))

from babytalk_paths import LIBRARY_DIR  # noqa: E402
from resegment import (  # noqa: E402
    MERGE_BACK_MAX_GAP_MS,
    MERGE_BACK_SHORT_PIECE_MS,
    MergeBackParams,
)
from vad_segments import (  # noqa: E402
    SPEECH_DELTA_DB,
    build_candidates,
    frame_rms_db,
    noise_floor_db,
)
import ml_delta  # noqa: E402
from dismiss_vs_tag_delta import (  # noqa: E402
    MODE_PLAIN,
    SHORT_RATIO,
    active_bounds_ms,
    classify_mode,
)

DEFAULT_SESSION = "26_07_27__19:53:00"
DEFAULT_PORT = 8766

# Lab defaults for band-pass (toddler voice-ish band; tweak in UI).
DEFAULT_F_LOW = 300.0
DEFAULT_F_HIGH = 3000.0
DEFAULT_GAIN_DB = 0.0

POINT_TAG_SPAN_MS = 500.0
TOO_SHORT_RATIO = SHORT_RATIO  # dismiss_vs_tag_delta: durRatio < 0.75

# IoU pie bins for matched Baby-tag pairs (edges inclusive on the left except first).
# Buckets: [0, 0.3), [0.3, 0.5), [0.5, 0.7), [0.7, 1.0]
IOU_PIE_EDGES = (0.0, 0.3, 0.5, 0.7, 1.0)
IOU_PIE_LABELS = (
    "IoU < 0.3",
    "0.3 ≤ IoU < 0.5",
    "0.5 ≤ IoU < 0.7",
    "IoU ≥ 0.7",
)

_STATE_LOCK = threading.Lock()
_KIT: Path | None = None
_MANIFEST: dict = {}
_TAGS: list[dict] = []
_AUDIO: np.ndarray | None = None
_SR: int = 0
_AUDIO_PATH: Path | None = None
_ENERGY_TIMES: np.ndarray | None = None
_ENERGY_DBS: np.ndarray | None = None
_NOISE_FLOOR: float = 0.0
_LAST_FIND: dict | None = None
# Frozen production baseline (no band-pass). Recomputed only on kit load /
# explicit refresh — never when the user tweaks Find knobs.
_BASELINE: dict | None = None
_BASELINE_LOCK = threading.Lock()
_BASELINE_GEN = 0


def baseline_pipeline_params() -> dict:
    """Native prod VAD + merge-back defaults; no band-pass; lab diarization=none."""
    return {
        "bandPass": False,
        "fLow": None,
        "fHigh": None,
        "gainDb": 0.0,
        "speechDeltaDb": float(SPEECH_DELTA_DB),
        "shortPieceMs": float(MERGE_BACK_SHORT_PIECE_MS),
        "maxGapMs": float(MERGE_BACK_MAX_GAP_MS),
        "requireClearlyShort": True,
        "mergeBack": True,
        "diarization": "none",
    }


def resolve_kit_by_session(session: str) -> Path:
    hits: list[Path] = []
    for d in sorted(LIBRARY_DIR.iterdir()):
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
        raise FileNotFoundError(
            f"no kit with sessionName={session!r} under {LIBRARY_DIR}"
        )
    if len(hits) > 1:
        names = ", ".join(h.name for h in hits)
        raise FileNotFoundError(f"multiple kits for {session!r}: {names}")
    return hits[0]


def _mono(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    f_low: float,
    f_high: float,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Zero-phase SOS band-pass + optional linear gain."""
    from scipy.signal import butter, sosfiltfilt

    x = _mono(audio)
    nyq = float(sr) * 0.5
    lo = max(20.0, float(f_low))
    hi = min(float(f_high), nyq - 1.0)
    if lo >= hi:
        raise ValueError(f"invalid band: f_low={lo} f_high={hi} (nyquist={nyq})")
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    if gain_db:
        y = y * (10.0 ** (float(gain_db) / 20.0))
    return np.asarray(y, dtype=np.float64)


def _span(item: dict) -> tuple[float, float] | None:
    start = item.get("startMs")
    if start is None:
        start = item.get("tMs")
    if start is None:
        return None
    end = item.get("endMs")
    if end is None or end <= start:
        end = float(start) + POINT_TAG_SPAN_MS
    return float(start), float(end)


def _active_span_item(
    item: dict,
    times: np.ndarray,
    dbs: np.ndarray,
    floor: float,
    speech_delta_db: float,
) -> dict | None:
    sp = _span(item)
    if not sp:
        return None
    ab = active_bounds_ms(
        sp[0],
        sp[1],
        times,
        dbs,
        floor,
        speech_delta_db=speech_delta_db,
    )
    return {
        **{k: v for k, v in item.items() if k not in ("startMs", "endMs", "tMs")},
        "startMs": float(ab["startMs"]),
        "endMs": float(ab["endMs"]),
    }


def filter_tags_by_speaker(tags: list[dict], speaker: str | None) -> list[dict]:
    """If speaker is set (e.g. 'Baby'), keep tags whose speaker matches case-insensitively."""
    if not speaker:
        return tags
    want = speaker.strip().lower()
    return [t for t in tags if (t.get("speaker") or "").strip().lower() == want]


def filter_tags_to_window(
    tags: list[dict], start_ms: float | None, end_ms: float | None
) -> list[dict]:
    """Keep items whose span overlaps ``[start_ms, end_ms)`` (half-open overlap).

    Also used to slice cached baseline candidates for fair windowed scoring.
    """
    if start_ms is None or end_ms is None:
        return tags
    out = []
    for t in tags:
        sp = _span(t)
        if not sp:
            continue
        if sp[1] > start_ms and sp[0] < end_ms:
            out.append(t)
    return out


def _parse_window_ms(
    params: dict | None,
) -> tuple[float | None, float | None]:
    """Extract optional find-window bounds from a request/params dict."""
    if not params:
        return None, None
    win_start = params.get("windowStartMs")
    win_end = params.get("windowEndMs")
    win_start_ms = (
        float(win_start) if win_start is not None and win_start != "" else None
    )
    win_end_ms = (
        float(win_end) if win_end is not None and win_end != "" else None
    )
    if (
        win_start_ms is not None
        and win_end_ms is not None
        and win_end_ms > win_start_ms
    ):
        return win_start_ms, win_end_ms
    return None, None


def _window_label(win_start_ms: float | None, win_end_ms: float | None) -> str:
    if win_start_ms is None or win_end_ms is None:
        return "full file"
    dur_s = (win_end_ms - win_start_ms) / 1000.0
    return (
        f"find window {win_start_ms:.0f}–{win_end_ms:.0f} ms "
        f"({dur_s:.2f}s)"
    )


def baseline_scored_view(
    baseline: dict,
    *,
    win_start_ms: float | None = None,
    win_end_ms: float | None = None,
) -> dict:
    """Scorecard view of the frozen baseline, optionally window-scoped.

    Keeps the fullband candidate cache intact; when a find window is set,
    filters cached candidates + kit tags to that window and re-runs Baby/all
    FoM (IoU pies, error modes, distributions) so Baseline matches This run.
    """
    windowed = (
        win_start_ms is not None
        and win_end_ms is not None
        and win_end_ms > win_start_ms
    )
    params = baseline.get("params") or baseline_pipeline_params()
    n_full = int(baseline.get("nCandidates") or 0)
    scope = _window_label(win_start_ms, win_end_ms)

    if not windowed:
        return {
            "ok": True,
            "cached": True,
            "elapsedSec": baseline.get("elapsedSec"),
            "params": params,
            "nCandidates": n_full,
            "nCandidatesFull": n_full,
            "windowStartMs": None,
            "windowEndMs": None,
            "scoreScope": "full",
            "scoreScopeLabel": scope,
            "metricsBaby": baseline.get("metricsBaby"),
            "metricsAll": baseline.get("metricsAll"),
            "stats": baseline.get("stats"),
            "note": baseline.get("note"),
            "scoreNote": (
                "Baseline (no filter, prod defaults) — scored on full file."
            ),
        }

    with _STATE_LOCK:
        tags = list(_TAGS)
        times = _ENERGY_TIMES
        dbs = _ENERGY_DBS
        floor = _NOISE_FLOOR

    cands = list(baseline.get("candidates") or [])
    score_cands = filter_tags_to_window(cands, win_start_ms, win_end_ms)
    score_tags = filter_tags_to_window(tags, win_start_ms, win_end_ms)
    baby_tags = filter_tags_by_speaker(score_tags, "Baby")
    speech_delta = float(params.get("speechDeltaDb", SPEECH_DELTA_DB))

    metrics_baby = _score_bundle(
        baby_tags,
        score_cands,
        times=times,
        dbs=dbs,
        floor=floor,
        speech_delta_db=speech_delta,
        speaker_scope="Baby",
        label_raw="baseline",
    )
    metrics_all = _score_bundle(
        score_tags,
        score_cands,
        times=times,
        dbs=dbs,
        floor=floor,
        speech_delta_db=speech_delta,
        speaker_scope="all",
        label_raw="baseline",
    )

    return {
        "ok": True,
        "cached": True,
        "elapsedSec": baseline.get("elapsedSec"),
        "params": params,
        "nCandidates": len(score_cands),
        "nCandidatesFull": n_full,
        "windowStartMs": win_start_ms,
        "windowEndMs": win_end_ms,
        "scoreScope": "window",
        "scoreScopeLabel": scope,
        "metricsBaby": metrics_baby,
        "metricsAll": metrics_all,
        "stats": baseline.get("stats"),
        "note": baseline.get("note"),
        "scoreNote": (
            "Baseline (no filter, prod defaults) — scored on "
            f"{scope}. Cache still holds {n_full} fullband candidates; "
            f"{len(score_cands)} overlap the window."
        ),
    }


def _match_pairs(
    tags: list[dict], cands: list[dict]
) -> list[dict]:
    """Best-IoU overlapping match per tag (same pairing as ml_delta.compare)."""
    tspans = [(t, s) for t in tags if (s := _span(t))]
    cspans = [(c, s) for c in cands if (s := _span(c))]
    pairs: list[dict] = []
    for tag, ts in tspans:
        best = None
        for _cand, cs in cspans:
            inter = max(0.0, min(ts[1], cs[1]) - max(ts[0], cs[0]))
            if inter <= 0:
                continue
            union = (ts[1] - ts[0]) + (cs[1] - cs[0]) - inter
            score = inter / union if union > 0 else 0.0
            if best is None or score > best["iou"]:
                best = {"iou": score, "cspan": cs}
        if best is None:
            continue
        ts_dur = ts[1] - ts[0]
        cs = best["cspan"]
        cs_dur = cs[1] - cs[0]
        d_start = ts[0] - cs[0]
        d_end = ts[1] - cs[1]
        dur_ratio = cs_dur / ts_dur if ts_dur else 0.0
        mode = classify_mode(d_start, d_end, dur_ratio, best["iou"])
        pairs.append(
            {
                "iou": best["iou"],
                "dStart": d_start,
                "dEnd": d_end,
                "durRatio": dur_ratio,
                "mode": mode,
            }
        )
    return pairs


def _iou_pie_buckets(ious: list[float]) -> dict:
    """Readable IoU bins for pie chart: <0.3, [0.3,0.5), [0.5,0.7), ≥0.7."""
    edges = IOU_PIE_EDGES
    labels = IOU_PIE_LABELS
    n_bins = len(labels)
    counts = [0] * n_bins
    for v in ious:
        x = float(v)
        placed = False
        for i in range(n_bins - 1):
            if x < edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1  # ≥ last edge (0.7)
    n = len(ious)
    buckets = []
    for i, (label, count) in enumerate(zip(labels, counts)):
        buckets.append(
            {
                "label": label,
                "lo": edges[i],
                "hi": edges[i + 1],
                "n": count,
                "pct": round(100.0 * count / n, 1) if n else 0.0,
            }
        )
    return {
        "edges": list(edges),
        "nMatched": n,
        "buckets": buckets,
    }


def summarise_metrics(
    tags: list[dict], cands: list[dict], *, label: str, speakerScope: str
) -> dict:
    """FoM + dismiss_vs_tag_delta modes + boundary distributions for matched pairs."""
    full = ml_delta.compare(tags, cands, label=label)
    cov = full.get("coverage") or {}

    pairs = _match_pairs(tags, cands)
    n_matched = len(pairs)
    n_short = sum(1 for p in pairs if p["durRatio"] < TOO_SHORT_RATIO)
    too_short_pct = (
        round(100.0 * n_short / n_matched, 1) if n_matched else 0.0
    )

    mode_counts: dict[str, int] = {}
    for p in pairs:
        mode_counts[p["mode"]] = mode_counts.get(p["mode"], 0) + 1
    mode_breakdown = []
    for mode, n in sorted(mode_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        mode_breakdown.append(
            {
                "mode": mode,
                "n": n,
                "pct": round(100.0 * n / n_matched, 1) if n_matched else 0.0,
                "plain": MODE_PLAIN.get(mode, ""),
            }
        )

    d_starts = [p["dStart"] for p in pairs]
    d_ends = [p["dEnd"] for p in pairs]
    dur_ratios = [p["durRatio"] for p in pairs]
    ious = [p["iou"] for p in pairs]

    # Same quantile/histogram helpers + edges as ml_delta / dismiss reports.
    d_start_q = ml_delta.quantiles(d_starts)
    d_end_q = ml_delta.quantiles(d_ends)
    dur_q = ml_delta.quantiles(dur_ratios, 2)
    iou_q = ml_delta.quantiles(ious, 3)

    iou_pie = _iou_pie_buckets(ious)

    return {
        "label": label,
        "speakerScope": speakerScope,
        "nTags": int(full.get("nTags") or 0),
        "nCands": int(full.get("nCands") or 0),
        "nMatched": n_matched,
        "anyOverlapPct": cov.get("anyOverlapPct"),
        "iou50Pct": cov.get("iou50Pct"),
        "iou70Pct": cov.get("iou70Pct"),
        "medianDurRatio": dur_q.get("median"),
        "tooShortPct": too_short_pct,
        "tooShortN": n_short,
        "tooShortDef": (
            f"matched tags with cand/tag durRatio < {TOO_SHORT_RATIO} "
            "(same as dismiss_vs_tag_delta SHORT_RATIO)"
        ),
        "medianDStartMs": d_start_q.get("median"),
        "medianDEndMs": d_end_q.get("median"),
        "p10DStartMs": d_start_q.get("p10"),
        "p90DStartMs": d_start_q.get("p90"),
        "p10DEndMs": d_end_q.get("p10"),
        "p90DEndMs": d_end_q.get("p90"),
        "deltaSignNote": (
            "Δstart = tagStart−candStart (neg ⇒ ML late / missed onset); "
            "Δend = tagEnd−candEnd (pos ⇒ ML early cut)"
        ),
        "errorModes": mode_breakdown,
        "iouPie": iou_pie,
        "distributions": {
            "dStart": {
                "unit": "ms",
                "quantiles": d_start_q,
                "hist": ml_delta.histogram(
                    d_starts, [-2000, -1000, -500, -200, 0, 200, 500]
                ),
            },
            "dEnd": {
                "unit": "ms",
                "quantiles": d_end_q,
                "hist": ml_delta.histogram(
                    d_ends, [-500, -200, 0, 200, 500, 1000, 2000]
                ),
            },
            "durRatio": {
                "unit": "cand/tag",
                "quantiles": dur_q,
                "hist": ml_delta.histogram(
                    dur_ratios, [0.8, 1.0, 1.5, 2.0, 4.0, 8.0]
                ),
            },
            "iou": {
                "unit": "",
                "quantiles": iou_q,
                "hist": ml_delta.histogram(
                    ious, [0.1, 0.25, 0.5, 0.7, 0.9]
                ),
            },
        },
        "manual": (full.get("bySource") or {}).get("manual"),
    }


def _score_bundle(
    tags: list[dict],
    cands: list[dict],
    *,
    times: np.ndarray | None,
    dbs: np.ndarray | None,
    floor: float,
    speech_delta_db: float,
    speaker_scope: str,
    label_raw: str = "raw",
) -> dict:
    raw = summarise_metrics(tags, cands, label=label_raw, speakerScope=speaker_scope)
    active = None
    if times is not None and dbs is not None:
        act_tags = [
            item
            for t in tags
            if (item := _active_span_item(t, times, dbs, floor, speech_delta_db))
        ]
        act_cands = [
            item
            for c in cands
            if (item := _active_span_item(c, times, dbs, floor, speech_delta_db))
        ]
        active = summarise_metrics(
            act_tags,
            act_cands,
            label="activeAudio",
            speakerScope=speaker_scope,
        )
    return {"raw": raw, "active": active}


def _slim_candidates(anns: list[dict]) -> list[dict]:
    return [
        {
            "uuid": a.get("uuid"),
            "startMs": a.get("startMs"),
            "endMs": a.get("endMs"),
            "speaker": a.get("speaker"),
            "speechScore": a.get("speechScore"),
            "source": a.get("source"),
        }
        for a in anns
    ]


def ensure_baseline(*, force: bool = False) -> dict:
    """Run/cache fullband + production VAD/merge-back candidates once per kit.

    Frozen until ``force=True`` (Refresh baseline) or kit reload. Independent of
    the explorer band-pass / Find knobs.
    """
    global _BASELINE, _BASELINE_GEN

    with _BASELINE_LOCK:
        if _BASELINE is not None and not force:
            return _BASELINE
        _BASELINE_GEN += 1
        my_gen = _BASELINE_GEN
        if force:
            _BASELINE = None

    with _STATE_LOCK:
        if _AUDIO is None or _SR <= 0:
            raise RuntimeError("audio not loaded")
        audio = _AUDIO
        sr = _SR
        tags = list(_TAGS)
        times = _ENERGY_TIMES
        dbs = _ENERGY_DBS
        floor = _NOISE_FLOOR

    params = baseline_pipeline_params()
    mb = MergeBackParams(
        short_piece_ms=float(params["shortPieceMs"]),
        max_gap_ms=float(params["maxGapMs"]),
        require_clearly_short=bool(params["requireClearlyShort"]),
    )
    t0 = time.perf_counter()
    anns, stats = build_candidates(
        audio,
        sr,
        speech_delta_db=float(params["speechDeltaDb"]),
        merge_back=bool(params["mergeBack"]),
        merge_back_params=mb,
        diarization=str(params["diarization"]),
        resegment=True,
    )
    elapsed = time.perf_counter() - t0

    baby_tags = filter_tags_by_speaker(tags, "Baby")
    # Active-audio uses kit RMS + baseline speechDelta (production).
    metrics_baby = _score_bundle(
        baby_tags,
        anns,
        times=times,
        dbs=dbs,
        floor=floor,
        speech_delta_db=float(params["speechDeltaDb"]),
        speaker_scope="Baby",
        label_raw="baseline",
    )
    metrics_all = _score_bundle(
        tags,
        anns,
        times=times,
        dbs=dbs,
        floor=floor,
        speech_delta_db=float(params["speechDeltaDb"]),
        speaker_scope="all",
        label_raw="baseline",
    )

    result = {
        "ok": True,
        "cached": True,
        "elapsedSec": round(elapsed, 2),
        "params": params,
        "nCandidates": len(anns),
        "candidates": _slim_candidates(anns),
        "metricsBaby": metrics_baby,
        "metricsAll": metrics_all,
        "stats": {
            k: stats[k]
            for k in (
                "regions",
                "candidates",
                "segmentation",
                "resegSplits",
                "mergeBackMerges",
                "speechGateRejected",
            )
            if k in stats
        },
        "note": (
            "Baseline = no band-pass (fullband) + production VAD/merge-back "
            f"(speechDeltaDb={params['speechDeltaDb']}, "
            f"short_piece_ms={params['shortPieceMs']}, "
            f"max_gap_ms={params['maxGapMs']}, require_clearly_short=True, "
            "merge-back on, diarization=none). Frozen until Refresh baseline."
        ),
    }
    with _BASELINE_LOCK:
        if my_gen != _BASELINE_GEN:
            # A newer refresh superseded this run; prefer the newer cache.
            if _BASELINE is not None:
                return _BASELINE
        _BASELINE = result
        return _BASELINE


def run_find(params: dict) -> dict:
    global _LAST_FIND
    with _STATE_LOCK:
        if _AUDIO is None or _SR <= 0:
            raise RuntimeError("audio not loaded")
        audio = _AUDIO
        sr = _SR
        tags = list(_TAGS)
        times = _ENERGY_TIMES
        dbs = _ENERGY_DBS
        floor = _NOISE_FLOOR

    f_low = float(params.get("fLow", DEFAULT_F_LOW))
    f_high = float(params.get("fHigh", DEFAULT_F_HIGH))
    gain_db = float(params.get("gainDb", DEFAULT_GAIN_DB))
    speech_delta_db = float(params.get("speechDeltaDb", SPEECH_DELTA_DB))
    short_piece_ms = float(
        params.get("shortPieceMs", MERGE_BACK_SHORT_PIECE_MS)
    )
    max_gap_ms = float(params.get("maxGapMs", MERGE_BACK_MAX_GAP_MS))
    require_clearly_short = bool(params.get("requireClearlyShort", True))
    merge_back = bool(params.get("mergeBack", True))
    diarization = str(params.get("diarization", "none") or "none")
    win_start_ms, win_end_ms = _parse_window_ms(params)

    # Ensure frozen baseline exists (does not use Find knobs / band-pass).
    # Window only affects scoring slice below via baseline_scored_view.
    baseline_cache = ensure_baseline(force=False)

    t0 = time.perf_counter()
    filtered = bandpass_filter(audio, sr, f_low, f_high, gain_db)
    t_filt = time.perf_counter() - t0

    offset_ms = 0.0
    work = filtered
    if win_start_ms is not None and win_end_ms is not None:
        i0 = max(0, int(win_start_ms * sr / 1000.0))
        i1 = min(len(filtered), int(math.ceil(win_end_ms * sr / 1000.0)))
        work = filtered[i0:i1]
        offset_ms = win_start_ms
        if work.size < sr // 10:
            raise ValueError("window too short (<100ms of audio)")

    mb = MergeBackParams(
        short_piece_ms=short_piece_ms,
        max_gap_ms=max_gap_ms,
        require_clearly_short=require_clearly_short,
    )
    t1 = time.perf_counter()
    anns, stats = build_candidates(
        work,
        sr,
        speech_delta_db=speech_delta_db,
        merge_back=merge_back,
        merge_back_params=mb,
        diarization=diarization,
        resegment=True,
    )
    t_pipe = time.perf_counter() - t1

    if offset_ms:
        for a in anns:
            if a.get("startMs") is not None:
                a["startMs"] = float(a["startMs"]) + offset_ms
            if a.get("endMs") is not None:
                a["endMs"] = float(a["endMs"]) + offset_ms

    score_tags = filter_tags_to_window(tags, win_start_ms, win_end_ms)
    baby_tags = filter_tags_by_speaker(score_tags, "Baby")
    metrics_baby = _score_bundle(
        baby_tags,
        anns,
        times=times,
        dbs=dbs,
        floor=floor,
        speech_delta_db=speech_delta_db,
        speaker_scope="Baby",
        label_raw="thisRun",
    )
    metrics_all = _score_bundle(
        score_tags,
        anns,
        times=times,
        dbs=dbs,
        floor=floor,
        speech_delta_db=speech_delta_db,
        speaker_scope="all",
        label_raw="thisRun",
    )

    slim_cands = _slim_candidates(anns)

    # Baseline scorecard uses the same evaluation window as this run.
    baseline = baseline_scored_view(
        baseline_cache,
        win_start_ms=win_start_ms,
        win_end_ms=win_end_ms,
    )

    # Headline FoM = this run vs Baby tags (experiment).
    primary = metrics_baby["raw"]
    baseline_baby = (baseline.get("metricsBaby") or {}).get("raw") or {}
    scope_label = baseline.get("scoreScopeLabel") or "full file"
    result = {
        "ok": True,
        "elapsedSec": round(time.perf_counter() - t0, 2),
        "filterSec": round(t_filt, 2),
        "pipelineSec": round(t_pipe, 2),
        "params": {
            "fLow": f_low,
            "fHigh": f_high,
            "gainDb": gain_db,
            "speechDeltaDb": speech_delta_db,
            "shortPieceMs": short_piece_ms,
            "maxGapMs": max_gap_ms,
            "requireClearlyShort": require_clearly_short,
            "mergeBack": merge_back,
            "diarization": diarization,
            "windowStartMs": win_start_ms,
            "windowEndMs": win_end_ms,
        },
        "stats": {
            k: stats[k]
            for k in (
                "regions",
                "candidates",
                "segmentation",
                "resegSplits",
                "mergeBackMerges",
                "speechGateRejected",
            )
            if k in stats
        }
        | {"diarizationBackend": (stats.get("diarization") or {}).get("backend")},
        "nCandidates": len(slim_cands),
        "candidates": slim_cands,
        "metricsBaby": metrics_baby,
        "metricsAll": metrics_all,
        # Compat: "raw"/"active" aliases now mean THIS RUN (experiment).
        "metricsRaw": metrics_baby["raw"],
        "metricsActive": metrics_baby["active"],
        "baseline": baseline,
        "fom": {
            "speakerScope": "Baby",
            "nTags": primary.get("nTags"),
            "nCands": primary.get("nCands"),
            "anyOverlapPct": primary.get("anyOverlapPct"),
            "iou50Pct": primary.get("iou50Pct"),
            "tooShortPct": primary.get("tooShortPct"),
            "medianDurRatio": primary.get("medianDurRatio"),
            "medianDStartMs": primary.get("medianDStartMs"),
            "medianDEndMs": primary.get("medianDEndMs"),
            "baselineIou50Pct": baseline_baby.get("iou50Pct"),
            "baselineTooShortPct": baseline_baby.get("tooShortPct"),
            "scoreScope": baseline.get("scoreScope"),
            "scoreScopeLabel": scope_label,
        },
        "metricsNote": (
            "Baseline card = frozen fullband + production VAD/merge-back "
            "(no band-pass); pipeline knobs do not change it. "
            f"Both cards scored on {scope_label}. "
            "This-run card = candidates from the Find you just ran "
            "(band-pass + your knobs"
            + (
                "; VAD on windowed audio"
                if win_start_ms is not None
                else ""
            )
            + "). Baseline candidates = prod-default cache"
            + (
                " filtered to the same window"
                if win_start_ms is not None
                else " (full file)"
            )
            + ". Pairing = best IoU overlap "
            "(ml_delta.compare). too_short = durRatio cand/tag < "
            f"{TOO_SHORT_RATIO}. "
            "Δstart = tag−cand start (neg ⇒ ML late); "
            "Δend = tag−cand end (pos ⇒ ML early cut). "
            "Active-audio trims silence inside each box on original-kit RMS."
        ),
        "nTagsScored": len(baby_tags),
        "nTagsAllSpeakers": len(score_tags),
        "nTagsBaby": len(baby_tags),
    }
    with _STATE_LOCK:
        _LAST_FIND = result
    return result


def load_kit(session: str) -> None:
    global _KIT, _MANIFEST, _TAGS, _AUDIO, _SR, _AUDIO_PATH
    global _ENERGY_TIMES, _ENERGY_DBS, _NOISE_FLOOR, _BASELINE, _BASELINE_GEN, _LAST_FIND

    kit = resolve_kit_by_session(session)
    man = json.loads((kit / "manifest.json").read_text())
    audio_name = man.get("audioFile", "audio.wav")
    audio_path = kit / audio_name
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    tags_path = kit / "tags.json"
    tags = []
    if tags_path.exists():
        data = json.loads(tags_path.read_text())
        tags = data.get("tags", []) if isinstance(data, dict) else data

    print(f"Loading audio {audio_path} …", flush=True)
    audio, sr = sf.read(str(audio_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    mono = _mono(audio)
    times, dbs = frame_rms_db(mono, int(sr))
    floor = float(noise_floor_db(dbs))

    _KIT = kit
    _MANIFEST = man
    _TAGS = tags
    _AUDIO = mono
    _SR = int(sr)
    _AUDIO_PATH = audio_path
    _ENERGY_TIMES = times
    _ENERGY_DBS = dbs
    _NOISE_FLOOR = floor
    with _BASELINE_LOCK:
        _BASELINE = None
        _BASELINE_GEN += 1
    _LAST_FIND = None
    print(
        f"Ready: session={session!r} folder={kit.name} "
        f"sr={sr} dur={len(mono)/sr:.1f}s tags={len(tags)} "
        f"noiseFloor={floor:.1f} dB",
        flush=True,
    )
    # Warm baseline in background so the UI can show it without blocking boot.
    threading.Thread(
        target=_warm_baseline,
        name="baseline-warm",
        daemon=True,
    ).start()


def _warm_baseline() -> None:
    try:
        print("Computing production baseline (no band-pass)…", flush=True)
        bl = ensure_baseline(force=False)
        print(
            f"Baseline ready: {bl.get('nCandidates')} candidates "
            f"in {bl.get('elapsedSec')}s",
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        print("Baseline warm-up failed.", flush=True)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Snippet explorer lab</title>
<style>
  :root {
    --bg: #f6f3ee;
    --ink: #1c1916;
    --muted: #6b6560;
    --panel: #fffdf9;
    --line: #ddd5c8;
    --tag: rgba(40, 120, 90, 0.35);
    --tag-stroke: #2a7a58;
    --cand: rgba(200, 90, 40, 0.40);
    --cand-stroke: #c45a28;
    --sel: rgba(60, 100, 180, 0.18);
    --accent: #2f5d4a;
    --fom: #1e3d32;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font: 14px/1.4 "IBM Plex Sans", "Segoe UI", sans-serif;
    color: var(--ink); background: var(--bg);
  }
  header {
    padding: 12px 18px; border-bottom: 1px solid var(--line);
    background: var(--panel); display: flex; gap: 16px; align-items: baseline;
    flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 650; }
  header .meta { color: var(--muted); font-size: 12px; }
  main { display: grid; grid-template-columns: 300px 1fr; gap: 0; min-height: calc(100vh - 52px); }
  aside {
    padding: 14px; border-right: 1px solid var(--line); background: var(--panel);
    overflow: auto;
  }
  aside label { display: block; font-size: 11px; color: var(--muted); margin: 10px 0 3px; }
  aside input[type=number], aside select {
    width: 100%; padding: 6px 8px; border: 1px solid var(--line); border-radius: 4px;
    background: #fff; font: inherit;
  }
  aside .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  aside .check { display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size: 13px; }
  aside .check input { width: auto; }
  aside .help {
    font-size: 11px; color: var(--muted); line-height: 1.45;
    margin: 6px 0 4px; padding: 8px 9px;
    background: #f3efe8; border-radius: 5px; border: 1px solid var(--line);
  }
  aside .help strong { color: var(--ink); font-weight: 600; }
  button.primary {
    margin-top: 14px; width: 100%; padding: 10px; border: 0; border-radius: 6px;
    background: var(--accent); color: #fff; font: inherit; font-weight: 600; cursor: pointer;
  }
  button.primary:disabled { opacity: 0.55; cursor: wait; }
  button.ghost {
    margin-top: 8px; width: 100%; padding: 7px; border: 1px solid var(--line);
    border-radius: 6px; background: #fff; font: inherit; cursor: pointer;
  }
  .hint { font-size: 11px; color: var(--muted); margin-top: 10px; }
  section.work { padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .toolbar input { width: 90px; padding: 5px 7px; border: 1px solid var(--line); border-radius: 4px; }
  .toolbar button {
    padding: 5px 10px; border: 1px solid var(--line); border-radius: 4px;
    background: #fff; cursor: pointer; font: inherit;
  }
  .legend span { display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; font-size: 12px; color: var(--muted); }
  .swatch { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
  .swatch.tag { background: var(--tag); border: 1px solid var(--tag-stroke); }
  .swatch.cand { background: var(--cand); border: 1px solid var(--cand-stroke); }
  .wave-wrap {
    position: relative; height: 180px; background: var(--panel);
    border: 1px solid var(--line); border-radius: 6px; overflow: hidden;
  }
  canvas#wave { width: 100%; height: 100%; display: block; cursor: crosshair; }
  .wave-scroll {
    width: 100%; margin-top: 4px; height: 12px;
    accent-color: var(--accent);
  }
  .wave-scroll:disabled { opacity: 0.35; }
  .toolbar .check-inline {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 12px; color: var(--muted); margin: 0;
  }
  .toolbar .check-inline input { width: auto; margin: 0; }
  .toolbar button.playing { background: #e8f0ec; border-color: var(--accent); }
  .fom-banner {
    background: var(--fom); color: #f4faf7; border-radius: 8px;
    padding: 12px 14px; display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px 14px;
  }
  .fom-banner .fom-title {
    grid-column: 1 / -1; font-size: 12px; opacity: 0.85; margin: 0;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  }
  .fom-banner .fom-title strong { font-size: 14px; color: #fff; opacity: 1; }
  .fom-stat { min-width: 0; }
  .fom-stat .v {
    font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums;
    line-height: 1.1;
  }
  .fom-stat .k { font-size: 11px; opacity: 0.8; margin-top: 2px; }
  .metrics-tools { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  .metrics {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
  }
  .metrics.full { grid-template-columns: 1fr; }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 10px 12px;
  }
  .card h3 { margin: 0 0 8px; font-size: 13px; }
  .card h4 { margin: 12px 0 6px; font-size: 12px; color: var(--muted); font-weight: 600; }
  .card table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .card td { padding: 3px 0; vertical-align: top; }
  .card td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  .card td.plain { text-align: left; color: var(--muted); font-size: 11px; padding-left: 8px; }
  .card .mode-table td:nth-child(2),
  .card .mode-table td:nth-child(3) { text-align: right; width: 48px; }
  .hist {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px; line-height: 1.35; color: var(--ink);
    white-space: pre; overflow-x: auto; margin: 4px 0 0;
  }
  .qrow { font-size: 12px; font-variant-numeric: tabular-nums; }
  .qrow span { color: var(--muted); margin-right: 6px; }
  .iou-pie-wrap {
    display: flex; flex-wrap: wrap; align-items: center; gap: 14px 18px;
    margin: 6px 0 4px;
  }
  .iou-pie-wrap svg { flex: 0 0 auto; display: block; }
  .iou-pie-legend {
    list-style: none; margin: 0; padding: 0; font-size: 12px;
    font-variant-numeric: tabular-nums; min-width: 160px;
  }
  .iou-pie-legend li {
    display: flex; align-items: baseline; gap: 8px; margin: 3px 0;
  }
  .iou-pie-legend .sw {
    width: 10px; height: 10px; border-radius: 2px; flex: 0 0 auto;
    margin-top: 2px;
  }
  .iou-pie-legend .lbl { flex: 1 1 auto; color: var(--ink); }
  .iou-pie-legend .cnt { color: var(--muted); white-space: nowrap; }
  .iou-stack {
    display: flex; height: 10px; border-radius: 4px; overflow: hidden;
    background: #e8e6e1; margin-top: 6px; max-width: 280px;
  }
  .iou-stack span { display: block; height: 100%; min-width: 0; }
  .status { font-size: 12px; color: var(--muted); min-height: 1.2em; }
  .status.err { color: #a33; }
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
    .metrics { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>Snippet explorer lab</h1>
  <div class="meta" id="kitMeta">Loading…</div>
</header>
<main>
<aside>
  <div class="help">
    <strong>Goal:</strong> see whether band-pass and/or VAD/merge-back knobs improve
    <strong>tag-vs-candidate</strong> quality vs human tags (Baby speaker FoM).
  </div>

  <strong style="font-size:12px">Band-pass</strong>
  <div class="help">
    Applied to audio <strong>before</strong> snippet finding. VAD/energy detection,
    DJW, and merge-back all run on the <strong>filtered</strong> signal — not just for listening.
  </div>
  <div class="row">
    <div><label>f_low (Hz)</label><input type="number" id="fLow" value="300" step="10"/></div>
    <div><label>f_high (Hz)</label><input type="number" id="fHigh" value="3000" step="50"/></div>
  </div>
  <label>gain (dB)</label><input type="number" id="gainDb" value="0" step="1"/>

  <strong style="font-size:12px; display:block; margin-top:14px">VAD / merge-back</strong>
  <div class="help">
    Same roles as production Find speech segments
    (<code>speechDeltaDb</code>, <code>short_piece_ms</code>, <code>max_gap_ms</code>,
    <code>require_clearly_short</code>, merge on/off) — applied on each Find to the filtered audio.
  </div>
  <label>speechDeltaDb</label><input type="number" id="speechDeltaDb" value="6" step="0.5"/>
  <div class="row">
    <div><label>short_piece_ms</label><input type="number" id="shortPieceMs" value="400" step="10"/></div>
    <div><label>max_gap_ms</label><input type="number" id="maxGapMs" value="200" step="10"/></div>
  </div>
  <label class="check"><input type="checkbox" id="requireClearlyShort" checked/> require_clearly_short</label>
  <label class="check"><input type="checkbox" id="mergeBack" checked/> merge-back</label>
  <label>diarization</label>
  <select id="diarization">
    <option value="none" selected>none (fast lab)</option>
    <option value="auto">auto (production-ish)</option>
    <option value="vtc">vtc</option>
  </select>

  <strong style="font-size:12px; display:block; margin-top:14px">Optional find window (ms)</strong>
  <div class="row">
    <div><label>start</label><input type="number" id="windowStartMs" placeholder="full" step="1"/></div>
    <div><label>end</label><input type="number" id="windowEndMs" placeholder="full" step="1"/></div>
  </div>
  <button class="ghost" type="button" id="presetExample">Preset ~1003s example (±8s)</button>

  <button class="primary" type="button" id="findBtn">Find snippets</button>
  <p class="hint">Exploration only — does not write annotations.json. Review Browser stays on :8765.</p>
</aside>

<section class="work">
  <div class="toolbar">
    <button type="button" id="btnPlay">Play</button>
    <button type="button" id="btnPause">Pause</button>
    <label class="check-inline"><input type="checkbox" id="loopSel"/> Loop selection</label>
    <button type="button" id="btnClearSel">Clear sel</button>
    <button type="button" id="zoomOut">−</button>
    <button type="button" id="zoomIn">+</button>
    <button type="button" id="zoomSel">Zoom selection</button>
    <button type="button" id="zoomReset">Reset view</button>
    <label>Jump s <input type="number" id="jumpSec" step="0.1" value="1003"/></label>
    <button type="button" id="jumpBtn">Go</button>
    <span class="legend">
      <span><i class="swatch tag"></i> tags</span>
      <span><i class="swatch cand"></i> candidates</span>
    </span>
    <span id="zoomLabel" class="meta"></span>
  </div>
  <div class="wave-wrap"><canvas id="wave"></canvas></div>
  <input type="range" class="wave-scroll" id="waveScroll" min="0" max="0" step="0.01" value="0" disabled title="Pan when zoomed"/>
  <div class="status" id="status"></div>

  <div class="fom-banner" id="fomBanner">
    <p class="fom-title"><strong>FoM vs Baby tags — this run</strong> — run Find to score</p>
    <div class="fom-stat"><div class="v" id="fomAny">—</div><div class="k">any-overlap %</div></div>
    <div class="fom-stat"><div class="v" id="fomIou50">—</div><div class="k">IoU ≥ 0.5 %</div></div>
    <div class="fom-stat"><div class="v" id="fomTooShort">—</div><div class="k">too_short % (&lt;0.75)</div></div>
    <div class="fom-stat"><div class="v" id="fomDur">—</div><div class="k">median durRatio</div></div>
    <div class="fom-stat"><div class="v" id="fomDStart">—</div><div class="k">median Δstart ms</div></div>
    <div class="fom-stat"><div class="v" id="fomDEnd">—</div><div class="k">median Δend ms</div></div>
    <div class="fom-stat"><div class="v" id="fomN">—</div><div class="k">n tags / cands</div></div>
  </div>

  <div class="metrics-tools">
    <label class="check-inline">
      <input type="checkbox" id="scoreAllSpeakers"/> Show all-speaker scorecard (default: Baby)
    </label>
    <label class="check-inline">
      <input type="checkbox" id="showActive"/> Also show active-audio trim metrics
    </label>
    <button type="button" id="refreshBaselineBtn" class="ghost" style="padding:4px 10px;font:inherit;cursor:pointer;border:1px solid var(--line);border-radius:4px;background:#fff">Refresh baseline</button>
  </div>

  <div class="metrics" id="metricsGrid">
    <div class="card" id="cardBaseline">
      <h3 id="baselineTitle">Baseline (no filter, prod defaults)</h3>
      <p class="hint" id="baselineSub" style="margin:0 0 8px">Frozen until you click Refresh baseline. Fullband audio + prod VAD/merge-back; Find window scopes scoring only.</p>
      <div id="metricsBaseline">Computing baseline…</div>
      <div id="wrapBaselineActive" style="display:none">
        <h4>Active-audio (baseline)</h4>
        <div id="metricsBaselineActive"></div>
      </div>
    </div>
    <div class="card" id="cardRun">
      <h3 id="runTitle">This run (band-pass + your knobs)</h3>
      <p class="hint" id="runSub" style="margin:0 0 8px">Updates every Find with the explorer settings on the left.</p>
      <div id="metricsRun">Run Find to score.</div>
      <div id="wrapRunActive" style="display:none">
        <h4>Active-audio (this run)</h4>
        <div id="metricsRunActive"></div>
      </div>
    </div>
  </div>
  <p class="hint" id="metricsNote"></p>
</section>
</main>
<script>
const WAVE_SILENCE_PEAK = 0.004;
const MIN_VIEW_DUR = 0.25;
const PEAK_BUCKETS = 24000; // cached min/max envelope columns for full file
let kit = null;
let tags = [];
let candidates = [];
let lastFind = null;
let lastBaseline = null;
let audioBuf = null;
let audioCtx = null;
let durationSec = 0;
let viewStart = 0;
let viewDur = 30;
let selStart = null;
let selEnd = null;
let drag = null;
let panning = false;
let panAnchorX = 0;
let panAnchorViewStart = 0;
let playheadSec = 0;
let activeSource = null;
let playbackOriginCtx = 0;
let playbackOriginBuf = 0;
let playbackEndBuf = null;
let playbackParkBuf = 0;
let playbackLoop = false;
let rafId = null;
let syncingScroll = false;
let drawRaf = null;
let peakMins = null;  // Float32Array length PEAK_BUCKETS
let peakMaxs = null;
let peakN = 0;
let canvasCssW = 0;
let canvasCssH = 180;

function $(id) { return document.getElementById(id); }
function viewEnd() { return Math.min(durationSec, viewStart + viewDur); }
function isZoomed() { return durationSec > 0 && viewDur < durationSec - 1e-6; }
function hasSel() {
  return selStart != null && selEnd != null && Math.abs(selEnd - selStart) > 0.005;
}
function num(id, fallback) {
  const v = parseFloat($(id).value);
  return Number.isFinite(v) ? v : fallback;
}
function emptyOrNum(id) {
  const s = $(id).value.trim();
  if (!s) return null;
  const v = parseFloat(s);
  return Number.isFinite(v) ? v : null;
}
function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}
function fmt(v, digits) {
  if (v == null || !Number.isFinite(Number(v))) return '—';
  return Number(v).toFixed(digits == null ? 1 : digits);
}

function setStatus(msg, isErr) {
  const el = $('status');
  el.textContent = msg || '';
  el.className = 'status' + (isErr ? ' err' : '');
}

function barHist(hist, maxBars) {
  if (!hist || !hist.length) return '(no matched pairs)';
  const maxC = Math.max(1, ...hist.map(h => h.count || 0));
  const width = maxBars || 24;
  return hist.map(h => {
    const n = h.count || 0;
    const bars = Math.round((n / maxC) * width);
    const bar = '█'.repeat(bars) + '░'.repeat(Math.max(0, width - bars));
    return `${String(h.bucket).padEnd(12)} ${bar} ${n}`;
  }).join('\n');
}

function qLine(q, digits) {
  if (!q || q.n == null) return '—';
  const d = digits == null ? 1 : digits;
  return `p25 ${fmt(q.p25, d)} · med ${fmt(q.median, d)} · p75 ${fmt(q.p75, d)} · p90 ${fmt(q.p90, d)}`;
}

const IOU_PIE_COLORS = ['#c4b5a0', '#d4a017', '#3d7a5a', '#1a4d36'];

function renderIouPie(pie) {
  if (!pie || !pie.buckets || !pie.buckets.length) {
    return '<p class="hint" style="margin:4px 0">(no matched pairs)</p>';
  }
  const buckets = pie.buckets;
  const total = pie.nMatched || buckets.reduce((s, b) => s + (b.n || 0), 0);
  if (!total) {
    return '<p class="hint" style="margin:4px 0">(no matched pairs)</p>';
  }

  const R = 52;
  const CX = 60;
  const CY = 60;
  let angle = -Math.PI / 2;
  const slices = [];
  const nonZero = buckets.filter(b => (b.n || 0) > 0);

  if (nonZero.length === 1) {
    const bi = buckets.indexOf(nonZero[0]);
    slices.push(
      `<circle cx="${CX}" cy="${CY}" r="${R}" fill="${IOU_PIE_COLORS[bi]}"/>`
    );
  } else {
    buckets.forEach((b, i) => {
      const n = b.n || 0;
      if (!n) return;
      const sweep = (n / total) * Math.PI * 2;
      const a0 = angle;
      const a1 = angle + sweep;
      angle = a1;
      const x0 = CX + R * Math.cos(a0);
      const y0 = CY + R * Math.sin(a0);
      const x1 = CX + R * Math.cos(a1);
      const y1 = CY + R * Math.sin(a1);
      const large = sweep > Math.PI ? 1 : 0;
      slices.push(
        `<path d="M ${CX} ${CY} L ${x0.toFixed(2)} ${y0.toFixed(2)} ` +
        `A ${R} ${R} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)} Z" ` +
        `fill="${IOU_PIE_COLORS[i]}"/>`
      );
    });
  }

  const legend = buckets.map((b, i) =>
    `<li>` +
    `<span class="sw" style="background:${IOU_PIE_COLORS[i]}"></span>` +
    `<span class="lbl">${b.label}</span>` +
    `<span class="cnt">${b.n} · ${fmt(b.pct, 1)}%</span>` +
    `</li>`
  ).join('');

  const stack = buckets.map((b, i) => {
    const pct = total ? (100 * (b.n || 0) / total) : 0;
    if (pct <= 0) return '';
    return `<span style="width:${pct.toFixed(2)}%;background:${IOU_PIE_COLORS[i]}" title="${b.label}: ${b.n}"></span>`;
  }).join('');

  const edges = (pie.edges || []).join(', ');
  return `
    <div class="iou-pie-wrap">
      <svg width="120" height="120" viewBox="0 0 120 120" aria-label="IoU breakdown pie">
        ${slices.join('')}
        <circle cx="${CX}" cy="${CY}" r="22" fill="#f7f6f2"/>
        <text x="${CX}" y="${CY - 4}" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1a18">${total}</text>
        <text x="${CX}" y="${CY + 10}" text-anchor="middle" font-size="9" fill="#5c5a54">matched</text>
      </svg>
      <ul class="iou-pie-legend">${legend}</ul>
    </div>
    <div class="iou-stack" title="IoU share">${stack}</div>
    <p class="hint" style="margin:4px 0 0">bins: ${edges || '0, 0.3, 0.5, 0.7, 1'} · n=${total}</p>
  `;
}

function renderScorecard(el, m) {
  if (!m) { el.textContent = '—'; return; }
  const modes = m.errorModes || [];
  const modeRows = modes.length
    ? modes.map(r =>
        `<tr><td><code>${r.mode}</code></td><td>${r.n}</td><td>${fmt(r.pct, 1)}%</td>` +
        `<td class="plain">${r.plain || ''}</td></tr>`
      ).join('')
    : '<tr><td colspan="4">no matched pairs</td></tr>';
  const dist = m.distributions || {};
  const dS = dist.dStart || {};
  const dE = dist.dEnd || {};
  const dR = dist.durRatio || {};
  const dI = dist.iou || {};

  el.innerHTML = `
    <table>
      <tr><td>tags scored</td><td>${m.nTags}</td></tr>
      <tr><td>candidates</td><td>${m.nCands}</td></tr>
      <tr><td>matched (any overlap)</td><td>${m.nMatched ?? '—'}</td></tr>
      <tr><td>any-overlap %</td><td>${fmt(m.anyOverlapPct, 1)}</td></tr>
      <tr><td>IoU ≥ 0.5 %</td><td>${fmt(m.iou50Pct, 1)}</td></tr>
      <tr><td>too_short % (durRatio &lt; 0.75)</td><td><strong>${fmt(m.tooShortPct, 1)}</strong> (n=${m.tooShortN ?? 0})</td></tr>
      <tr><td>median durRatio</td><td>${fmt(m.medianDurRatio, 2)}</td></tr>
      <tr><td>median Δstart ms</td><td>${fmt(m.medianDStartMs, 1)}</td></tr>
      <tr><td>median Δend ms</td><td>${fmt(m.medianDEndMs, 1)}</td></tr>
    </table>
    <p class="hint" style="margin:6px 0 0">${m.deltaSignNote || ''}</p>
    <h4>IoU breakdown (matched pairs)</h4>
    ${renderIouPie(m.iouPie)}
    <h4>Error modes (matched pairs)</h4>
    <table class="mode-table">
      <tr><td><em>mode</em></td><td>n</td><td>%</td><td></td></tr>
      ${modeRows}
    </table>
    <h4>Boundary distributions</h4>
    <div class="qrow"><span>Δstart</span>${qLine(dS.quantiles, 1)}</div>
    <pre class="hist">${barHist(dS.hist)}</pre>
    <div class="qrow" style="margin-top:8px"><span>Δend</span>${qLine(dE.quantiles, 1)}</div>
    <pre class="hist">${barHist(dE.hist)}</pre>
    <div class="qrow" style="margin-top:8px"><span>durRatio</span>${qLine(dR.quantiles, 2)}</div>
    <pre class="hist">${barHist(dR.hist)}</pre>
    <div class="qrow" style="margin-top:8px"><span>IoU</span>${qLine(dI.quantiles, 3)}</div>
    <pre class="hist">${barHist(dI.hist)}</pre>
  `;
}

function updateFomBanner(m, baselineM) {
  if (!m) {
    $('fomAny').textContent = '—';
    $('fomIou50').textContent = '—';
    $('fomTooShort').textContent = '—';
    $('fomDur').textContent = '—';
    $('fomDStart').textContent = '—';
    $('fomDEnd').textContent = '—';
    $('fomN').textContent = '—';
    return;
  }
  $('fomAny').textContent = fmt(m.anyOverlapPct, 1);
  $('fomIou50').textContent = fmt(m.iou50Pct, 1);
  $('fomTooShort').textContent = fmt(m.tooShortPct, 1);
  $('fomDur').textContent = fmt(m.medianDurRatio, 2);
  $('fomDStart').textContent = fmt(m.medianDStartMs, 0);
  $('fomDEnd').textContent = fmt(m.medianDEndMs, 0);
  $('fomN').textContent = `${m.nTags ?? '—'} / ${m.nCands ?? '—'}`;
  const title = $('fomBanner').querySelector('.fom-title');
  if (title) {
    let extra = '';
    if (baselineM && baselineM.iou50Pct != null) {
      extra = ` · baseline IoU≥0.5 ${fmt(baselineM.iou50Pct, 1)}%`;
    }
    title.innerHTML =
      `<strong>FoM vs ${m.speakerScope || 'Baby'} tags — this run</strong>` +
      ` · matched ${m.nMatched ?? 0}` +
      ` · too_short = durRatio &lt; 0.75` + extra;
  }
}

function pickBundle(root, all) {
  if (!root) return null;
  if (all) return root.metricsAll || null;
  return root.metricsBaby || null;
}

function refreshMetricsPanels() {
  const all = $('scoreAllSpeakers').checked;
  const showActive = $('showActive').checked;
  const scope = all ? 'all speakers' : 'Baby tags';

  const blRoot = lastBaseline || (lastFind && lastFind.baseline) || null;
  const blBundle = pickBundle(blRoot, all);
  const scoreScope = (blRoot && blRoot.scoreScopeLabel)
    || (lastFind && lastFind.fom && lastFind.fom.scoreScopeLabel)
    || 'full file';
  const winBits = (blRoot && blRoot.windowStartMs != null && blRoot.windowEndMs != null)
    ? ` — scored on find window ${Math.round(blRoot.windowStartMs)}–${Math.round(blRoot.windowEndMs)} ms`
    : (scoreScope !== 'full file' ? ` — scored on ${scoreScope}` : ' — scored on full file');

  $('baselineTitle').textContent =
    `Baseline (no filter, prod defaults)${winBits} — ${scope}`;
  $('runTitle').textContent =
    `This run (band-pass + your knobs)${winBits} — ${scope}`;

  if (blBundle && blBundle.raw) {
    renderScorecard($('metricsBaseline'), blBundle.raw);
    if (showActive && blBundle.active) {
      $('wrapBaselineActive').style.display = '';
      renderScorecard($('metricsBaselineActive'), blBundle.active);
    } else {
      $('wrapBaselineActive').style.display = 'none';
      $('metricsBaselineActive').innerHTML = '';
    }
    const bp = (blRoot && blRoot.params) || {};
    const nC = blRoot && blRoot.nCandidates;
    const nFull = blRoot && blRoot.nCandidatesFull;
    let sub =
      `Frozen · fullband · Δ=${bp.speechDeltaDb ?? 6} · ` +
      `short=${bp.shortPieceMs ?? 400} · gap=${bp.maxGapMs ?? 200} · ` +
      `merge-back on · diarization=none`;
    if (nC != null) {
      sub += ` · ${nC} cands scored`;
      if (nFull != null && nFull !== nC) sub += ` (${nFull} in cache)`;
    }
    if (blRoot && blRoot.elapsedSec != null) sub += ` · cache ${blRoot.elapsedSec}s`;
    if (blRoot && blRoot.scoreNote) sub += ` · ${blRoot.scoreNote}`;
    $('baselineSub').textContent = sub;
  } else {
    $('metricsBaseline').textContent = 'Computing baseline…';
    $('wrapBaselineActive').style.display = 'none';
  }

  if (lastFind) {
    const runBundle = all ? lastFind.metricsAll : lastFind.metricsBaby;
    const runRaw = (runBundle && runBundle.raw) || lastFind.metricsRaw;
    const runActive = (runBundle && runBundle.active) || lastFind.metricsActive;
    renderScorecard($('metricsRun'), runRaw);
    if (showActive && runActive) {
      $('wrapRunActive').style.display = '';
      renderScorecard($('metricsRunActive'), runActive);
    } else {
      $('wrapRunActive').style.display = 'none';
      $('metricsRunActive').innerHTML = '';
    }
    updateFomBanner(
      runRaw,
      blBundle && blBundle.raw
    );
    $('metricsNote').textContent = lastFind.metricsNote || '';
    const rp = lastFind.params || {};
    if (rp.windowStartMs != null && rp.windowEndMs != null) {
      $('runSub').textContent =
        `Find window ${Math.round(rp.windowStartMs)}–${Math.round(rp.windowEndMs)} ms · ` +
        `band-pass + your knobs · updates every Find.`;
    } else {
      $('runSub').textContent =
        'Full file · band-pass + your knobs · updates every Find.';
    }
  } else {
    $('metricsRun').textContent = 'Run Find to score.';
    $('wrapRunActive').style.display = 'none';
    updateFomBanner(null);
    $('runSub').textContent =
      'Updates every Find with the explorer settings on the left.';
  }
}

async function fetchBaseline(force) {
  const btn = $('refreshBaselineBtn');
  if (btn) btn.disabled = true;
  $('metricsBaseline').textContent = force ? 'Refreshing baseline…' : 'Computing baseline…';
  const winStart = emptyOrNum('windowStartMs');
  const winEnd = emptyOrNum('windowEndMs');
  try {
    let res;
    if (force) {
      res = await fetch('/api/baseline/refresh', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          windowStartMs: winStart,
          windowEndMs: winEnd,
        }),
      });
    } else {
      const q = new URLSearchParams();
      if (winStart != null) q.set('windowStartMs', String(winStart));
      if (winEnd != null) q.set('windowEndMs', String(winEnd));
      const qs = q.toString();
      res = await fetch(qs ? `/api/baseline?${qs}` : '/api/baseline');
    }
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || res.statusText);
    lastBaseline = data;
    if (lastFind) lastFind.baseline = data;
    refreshMetricsPanels();
    const scope = data.scoreScopeLabel || 'full file';
    setStatus(
      `Baseline ready: ${data.nCandidates} candidates scored on ${scope}` +
      (data.nCandidatesFull != null && data.nCandidatesFull !== data.nCandidates
        ? ` (${data.nCandidatesFull} in fullband cache)`
        : '') +
      ` · cache built in ${data.elapsedSec}s (no band-pass, production VAD/merge-back).`
    );
  } catch (err) {
    $('metricsBaseline').textContent = 'Baseline failed: ' + err.message;
    setStatus('Baseline failed: ' + err.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function buildPeakEnvelope(channelData) {
  const n = channelData.length;
  const buckets = Math.min(PEAK_BUCKETS, Math.max(1, n));
  const mins = new Float32Array(buckets);
  const maxs = new Float32Array(buckets);
  const samplesPer = n / buckets;
  for (let b = 0; b < buckets; b++) {
    const s0 = Math.floor(b * samplesPer);
    const s1 = Math.min(n, Math.floor((b + 1) * samplesPer));
    let mn = 1, mx = -1;
    if (s1 <= s0) {
      const v = channelData[Math.min(s0, n - 1)] || 0;
      mn = mx = v;
    } else {
      for (let i = s0; i < s1; i++) {
        const v = channelData[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    mins[b] = mn;
    maxs[b] = mx;
  }
  peakMins = mins;
  peakMaxs = maxs;
  peakN = buckets;
}

function peakAbsFromCache(t0, t1) {
  if (!peakMins || !durationSec) return 0;
  const b0 = Math.max(0, Math.floor((t0 / durationSec) * peakN));
  const b1 = Math.min(peakN, Math.ceil((t1 / durationSec) * peakN));
  let peak = 0;
  for (let b = b0; b < b1; b++) {
    const a = Math.abs(peakMins[b]);
    const c = Math.abs(peakMaxs[b]);
    if (a > peak) peak = a;
    if (c > peak) peak = c;
  }
  return peak;
}

function waveAmpScale() {
  if (!hasSel()) {
    const viewPeak = peakAbsFromCache(viewStart, viewEnd());
    return viewPeak >= WAVE_SILENCE_PEAK ? 1 / viewPeak : 1;
  }
  const a = Math.min(selStart, selEnd);
  const b = Math.max(selStart, selEnd);
  const va = Math.max(a, viewStart);
  const vb = Math.min(b, viewEnd());
  if (vb > va) {
    const selPeak = peakAbsFromCache(va, vb);
    if (selPeak >= WAVE_SILENCE_PEAK) return 1 / selPeak;
  }
  const viewPeak = peakAbsFromCache(viewStart, viewEnd());
  return viewPeak >= WAVE_SILENCE_PEAK ? 1 / viewPeak : 1;
}

function xToSec(x, w) {
  return viewStart + (x / w) * viewDur;
}
function secToX(sec, w) {
  return ((sec - viewStart) / viewDur) * w;
}

function normalizeSel() {
  if (selStart == null || selEnd == null) return;
  if (selStart > selEnd) {
    const t = selStart; selStart = selEnd; selEnd = t;
  }
}

function setPlayhead(t) {
  playheadSec = Math.max(0, Math.min(durationSec || 0, t));
}

function isBufferPlaying() {
  return activeSource != null;
}

function stopBufferPlayback() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (activeSource) {
    try { activeSource.onended = null; activeSource.stop(); } catch (e) {}
    activeSource = null;
  }
  playbackEndBuf = null;
  playbackLoop = false;
  updatePlayButton();
}

function finishBufferPlayback(parkAt) {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (activeSource) {
    try { activeSource.onended = null; } catch (e) {}
    activeSource = null;
  }
  playbackEndBuf = null;
  const park = Number.isFinite(parkAt) ? parkAt : playbackParkBuf;
  playheadSec = Math.max(0, Math.min(durationSec || 0, park));
  playbackLoop = false;
  updatePlayButton();
  scheduleDraw();
}

function startBufferPlayback(fromSec, toSec, loop) {
  if (!audioBuf) return false;
  stopBufferPlayback();
  const ctx = getAudioCtx();
  if (ctx.state === 'suspended') ctx.resume();

  const start = Math.max(0, Math.min(fromSec, durationSec - 0.001));
  let end = toSec == null ? durationSec : toSec;
  end = Math.max(start + 0.001, Math.min(end, durationSec));
  const dur = end - start;

  const source = ctx.createBufferSource();
  source.buffer = audioBuf;
  source.connect(ctx.destination);
  playbackOriginCtx = ctx.currentTime;
  playbackOriginBuf = start;
  playbackEndBuf = end;
  playbackParkBuf = start;
  playbackLoop = !!loop;
  playheadSec = start;
  activeSource = source;

  source.onended = () => {
    if (activeSource !== source) return;
    const park = playbackParkBuf;
    const loop = playbackLoop && hasSel();
    const endAt = playbackEndBuf;
    activeSource = null;
    if (loop && endAt != null) {
      startBufferPlayback(park, endAt, true);
      return;
    }
    finishBufferPlayback(park);
  };

  source.start(0, start, dur);
  updatePlayButton();
  tickPlayhead();
  return true;
}

function tickPlayhead() {
  if (!activeSource) return;
  const ctx = getAudioCtx();
  playheadSec = playbackOriginBuf + (ctx.currentTime - playbackOriginCtx);

  if (playbackEndBuf != null && playheadSec >= playbackEndBuf - 0.001) {
    playheadSec = playbackEndBuf;
  }

  if (isZoomed()) {
    const margin = viewDur * 0.15;
    if (playheadSec > viewEnd() - margin || playheadSec < viewStart + margin) {
      viewStart = playheadSec - viewDur * 0.35;
      clampView();
    }
  }
  scheduleDraw();
  rafId = requestAnimationFrame(tickPlayhead);
}

function pauseIfPlaying() {
  if (!isBufferPlaying()) return;
  const ctx = getAudioCtx();
  playheadSec = playbackOriginBuf + (ctx.currentTime - playbackOriginCtx);
  if (playbackEndBuf != null) {
    playheadSec = Math.min(playheadSec, playbackEndBuf);
  }
  stopBufferPlayback();
  scheduleDraw();
}

function playToggle() {
  if (isBufferPlaying()) {
    pauseIfPlaying();
    return;
  }
  if (!audioBuf) return;
  const loop = $('loopSel').checked;
  if (hasSel()) {
    normalizeSel();
    startBufferPlayback(selStart, selEnd, loop);
  } else {
    startBufferPlayback(playheadSec, null, false);
  }
}

function updatePlayButton() {
  const btn = $('btnPlay');
  if (!btn) return;
  const playing = isBufferPlaying();
  btn.textContent = playing
    ? (hasSel() ? (playbackLoop ? 'Looping…' : 'Playing sel…') : 'Playing…')
    : 'Play';
  btn.classList.toggle('playing', playing);
}

function panBySeconds(dt) {
  if (!isZoomed()) return;
  viewStart += dt;
  clampView();
  syncWaveScroll();
  scheduleDraw();
}

function syncWaveScroll() {
  const el = $('waveScroll');
  if (!el || !durationSec) return;
  syncingScroll = true;
  const maxStart = Math.max(0, durationSec - viewDur);
  el.min = '0';
  el.max = String(maxStart);
  el.step = String(Math.max(0.01, viewDur / 200));
  el.value = String(viewStart);
  el.disabled = !isZoomed();
  syncingScroll = false;
}

function drawOverlays(ctx, w, h) {
  function paint(list, fill) {
    for (const item of list) {
      const a = (item.startMs ?? item.tMs ?? 0) / 1000;
      let b = item.endMs != null ? item.endMs / 1000 : a + 0.5;
      if (b <= viewStart || a >= viewEnd()) continue;
      const x0 = secToX(Math.max(a, viewStart), w);
      const x1 = secToX(Math.min(b, viewEnd()), w);
      ctx.fillStyle = fill;
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    }
  }
  paint(tags, 'rgba(40,120,90,0.28)');
  paint(candidates, 'rgba(200,90,40,0.32)');
  if (hasSel()) {
    const a = Math.min(selStart, selEnd);
    const b = Math.max(selStart, selEnd);
    const x0 = secToX(Math.max(a, viewStart), w);
    const x1 = secToX(Math.min(b, viewEnd()), w);
    ctx.fillStyle = 'rgba(60,100,180,0.18)';
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
  }
  if (playheadSec >= viewStart && playheadSec <= viewEnd()) {
    const x = secToX(playheadSec, w);
    ctx.strokeStyle = '#c62828';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}

function scheduleDraw() {
  if (drawRaf != null) return;
  drawRaf = requestAnimationFrame(() => {
    drawRaf = null;
    drawWaveNow();
  });
}

function drawWaveNow() {
  const canvas = $('wave');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.parentElement.clientWidth || 800;
  const h = 180;
  if (w !== canvasCssW || h !== canvasCssH || canvas.width !== Math.floor(w * dpr)) {
    canvasCssW = w;
    canvasCssH = h;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + 'px';
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fffdf9';
  ctx.fillRect(0, 0, w, h);
  if (!audioBuf || !peakMins) {
    ctx.fillStyle = '#999';
    ctx.fillText(audioBuf ? 'Building peaks…' : 'Loading waveform…', 16, h / 2);
    return;
  }
  const ampScale = waveAmpScale();
  const mid = h / 2;
  const yGain = mid * 0.92 * ampScale;
  const b0 = Math.max(0, Math.floor((viewStart / durationSec) * peakN));
  const b1 = Math.min(peakN, Math.ceil((viewEnd() / durationSec) * peakN));
  const spanB = Math.max(1, b1 - b0);
  ctx.strokeStyle = '#5c5c5c';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const i0 = b0 + Math.floor((x / w) * spanB);
    const i1 = b0 + Math.floor(((x + 1) / w) * spanB);
    let mn = 1, mx = -1;
    const hi = Math.max(i0 + 1, i1);
    for (let i = i0; i < hi && i < peakN; i++) {
      if (peakMins[i] < mn) mn = peakMins[i];
      if (peakMaxs[i] > mx) mx = peakMaxs[i];
    }
    ctx.moveTo(x + 0.5, mid + mn * yGain);
    ctx.lineTo(x + 0.5, mid + mx * yGain);
  }
  ctx.stroke();
  ctx.strokeStyle = '#ddd5c8';
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  drawOverlays(ctx, w, h);
  const zoomX = durationSec > 0 ? durationSec / viewDur : 1;
  $('zoomLabel').textContent =
    (zoomX <= 1.01
      ? `full ${durationSec.toFixed(1)}s`
      : `${zoomX.toFixed(1)}× · ${viewStart.toFixed(2)}–${viewEnd().toFixed(2)} s`) +
    ` · ▶ ${playheadSec.toFixed(2)}s` +
    `  (amp×${ampScale.toFixed(1)})`;
  syncWaveScroll();
}

function clampView() {
  viewDur = Math.max(MIN_VIEW_DUR, Math.min(viewDur, durationSec || 30));
  viewStart = Math.max(0, Math.min(viewStart, Math.max(0, durationSec - viewDur)));
}

async function loadKit() {
  const res = await fetch('/api/kit');
  kit = await res.json();
  tags = kit.tags || [];
  $('kitMeta').textContent =
    `${kit.sessionName} · ${kit.folder} · ${kit.nTags} tags · ${(kit.durationMs/1000).toFixed(1)}s · noiseFloor ${kit.noiseFloorDb} dB`;
  const ares = await fetch('/audio');
  const arr = await ares.arrayBuffer();
  const ctx = getAudioCtx();
  audioBuf = await ctx.decodeAudioData(arr.slice(0));
  durationSec = audioBuf.duration || (kit.durationMs || 0) / 1000;
  buildPeakEnvelope(audioBuf.getChannelData(0));
  viewStart = 0;
  viewDur = Math.min(30, durationSec);
  playheadSec = 0;
  scheduleDraw();
  setStatus(
    `Loaded ${tags.length} tags. Click = playhead · drag = select · ` +
    `trackpad swipe / shift+wheel / scrollbar = pan when zoomed · Space = play/pause.`
  );
  // Baseline is warmed server-side; poll/fetch so the left scorecard fills.
  fetchBaseline(false);
}

$('findBtn').onclick = async () => {
  const btn = $('findBtn');
  btn.disabled = true;
  setStatus('Finding snippets (band-pass → VAD/DJW/merge-back)…');
  const body = {
    fLow: num('fLow', 300),
    fHigh: num('fHigh', 3000),
    gainDb: num('gainDb', 0),
    speechDeltaDb: num('speechDeltaDb', 6),
    shortPieceMs: num('shortPieceMs', 400),
    maxGapMs: num('maxGapMs', 200),
    requireClearlyShort: $('requireClearlyShort').checked,
    mergeBack: $('mergeBack').checked,
    diarization: $('diarization').value,
    windowStartMs: emptyOrNum('windowStartMs'),
    windowEndMs: emptyOrNum('windowEndMs'),
  };
  try {
    const res = await fetch('/api/find', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || res.statusText);
    candidates = data.candidates || [];
    lastFind = data;
    if (data.baseline) lastBaseline = data.baseline;
    refreshMetricsPanels();
    setStatus(
      `Done: ${data.nCandidates} candidates in ${data.elapsedSec}s ` +
      `(filter ${data.filterSec}s + pipeline ${data.pipelineSec}s). ` +
      `Scored vs ${data.nTagsBaby} Baby / ${data.nTagsAllSpeakers} all tags. ` +
      `Baseline scored on ${(data.baseline && data.baseline.scoreScopeLabel) || 'full file'}` +
      ` (${(data.baseline && data.baseline.nCandidates) || '—'} cands` +
      (data.baseline && data.baseline.nCandidatesFull != null
        && data.baseline.nCandidatesFull !== data.baseline.nCandidates
        ? ` of ${data.baseline.nCandidatesFull} cached`
        : '') +
      `).`
    );
    scheduleDraw();
  } catch (err) {
    setStatus('Find failed: ' + err.message, true);
  } finally {
    btn.disabled = false;
  }
};

$('scoreAllSpeakers').onchange = () => refreshMetricsPanels();
$('showActive').onchange = () => refreshMetricsPanels();
$('refreshBaselineBtn').onclick = () => fetchBaseline(true);

$('presetExample').onclick = () => {
  $('windowStartMs').value = String(Math.round((1003 - 8) * 1000));
  $('windowEndMs').value = String(Math.round((1003 + 8) * 1000));
  $('jumpSec').value = '1003';
  viewStart = 1000;
  viewDur = 8;
  clampView();
  scheduleDraw();
};

$('btnPlay').onclick = () => playToggle();
$('btnPause').onclick = () => pauseIfPlaying();
$('btnClearSel').onclick = () => {
  pauseIfPlaying();
  selStart = selEnd = null;
  scheduleDraw();
};
$('loopSel').onchange = () => {
  if (isBufferPlaying() && hasSel()) {
    playbackLoop = $('loopSel').checked;
    updatePlayButton();
  }
};

$('zoomIn').onclick = () => { viewDur = Math.max(MIN_VIEW_DUR, viewDur / 1.6); clampView(); scheduleDraw(); };
$('zoomOut').onclick = () => { viewDur = Math.min(durationSec || 60, viewDur * 1.6); clampView(); scheduleDraw(); };
$('zoomReset').onclick = () => { viewStart = 0; viewDur = Math.min(30, durationSec || 30); scheduleDraw(); };
$('zoomSel').onclick = () => {
  if (!hasSel()) return;
  const a = Math.min(selStart, selEnd);
  const b = Math.max(selStart, selEnd);
  viewStart = Math.max(0, a - 0.1);
  viewDur = Math.max(MIN_VIEW_DUR, (b - a) * 1.2);
  clampView();
  scheduleDraw();
};
$('jumpBtn').onclick = () => {
  const t = parseFloat($('jumpSec').value);
  if (!Number.isFinite(t)) return;
  viewStart = Math.max(0, t - viewDur / 2);
  setPlayhead(t);
  clampView();
  scheduleDraw();
};

$('waveScroll').oninput = () => {
  if (syncingScroll || !isZoomed()) return;
  viewStart = parseFloat($('waveScroll').value) || 0;
  clampView();
  scheduleDraw();
};

(function wireWaveInteractions() {
  const canvas = $('wave');

  canvas.addEventListener('wheel', (e) => {
    if (!durationSec || !isZoomed()) {
      if (!e.shiftKey && Math.abs(e.deltaY) >= Math.abs(e.deltaX)) return;
    }
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || 1;
    const useX = Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey;
    if (useX && isZoomed()) {
      const dx = e.shiftKey && Math.abs(e.deltaX) < Math.abs(e.deltaY) ? e.deltaY : e.deltaX;
      const dt = (dx / w) * viewDur;
      panBySeconds(dt);
      return;
    }
    const center = xToSec(e.clientX - rect.left, w);
    const factor = e.deltaY > 0 ? 1.12 : 0.88;
    const rel = viewDur > 0 ? (center - viewStart) / viewDur : 0.5;
    viewDur = Math.min(durationSec, Math.max(MIN_VIEW_DUR, viewDur * factor));
    viewStart = center - rel * viewDur;
    clampView();
    scheduleDraw();
  }, { passive: false });

  canvas.addEventListener('mousedown', (e) => {
    if (!durationSec) return;
    if (e.shiftKey || e.altKey || e.button === 1) {
      if (!isZoomed()) return;
      panning = true;
      panAnchorX = e.clientX;
      panAnchorViewStart = viewStart;
      e.preventDefault();
      return;
    }
    pauseIfPlaying();
    const rect = canvas.getBoundingClientRect();
    const sec = xToSec(e.clientX - rect.left, rect.width);
    drag = { a: sec, moved: false };
    selStart = sec; selEnd = sec;
    setPlayhead(sec);
    scheduleDraw();
  });

  window.addEventListener('mousemove', (e) => {
    if (panning) {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width || 1;
      const dt = ((panAnchorX - e.clientX) / w) * viewDur;
      viewStart = panAnchorViewStart + dt;
      clampView();
      scheduleDraw();
      return;
    }
    if (!drag) return;
    const rect = canvas.getBoundingClientRect();
    selEnd = xToSec(e.clientX - rect.left, rect.width);
    if (Math.abs(selEnd - drag.a) > 0.01) drag.moved = true;
    scheduleDraw();
  });

  window.addEventListener('mouseup', () => {
    if (panning) {
      panning = false;
      return;
    }
    if (!drag) return;
    const clickThresh = Math.max(0.015, 0.02 * (viewDur / Math.max(durationSec, 0.001)));
    if (!drag.moved || Math.abs((selEnd ?? 0) - drag.a) < clickThresh) {
      setPlayhead(drag.a);
      selStart = selEnd = null;
    } else {
      normalizeSel();
      setPlayhead(Math.min(selStart, selEnd));
    }
    drag = null;
    scheduleDraw();
  });
})();

window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea, select')) return;
  if (e.code === 'Space') {
    e.preventDefault();
    playToggle();
  }
  if (e.key === 'Escape') {
    pauseIfPlaying();
    selStart = selEnd = null;
    scheduleDraw();
  }
});

window.addEventListener('resize', () => scheduleDraw());
loadKit().catch(err => setStatus('Load failed: ' + err.message, true));
</script>
</body>
</html>
"""



_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except _CLIENT_GONE:
            return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"ok": True})
            return
        if parsed.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/kit":
            with _STATE_LOCK:
                if _KIT is None:
                    self._send_json(503, {"ok": False, "error": "kit not loaded"})
                    return
                payload = {
                    "ok": True,
                    "sessionName": _MANIFEST.get("sessionName"),
                    "folder": _KIT.name,
                    "audioFile": _MANIFEST.get("audioFile", "audio.wav"),
                    "durationMs": _MANIFEST.get("durationMs")
                    or (
                        int(len(_AUDIO) / _SR * 1000)
                        if _AUDIO is not None and _SR
                        else None
                    ),
                    "sampleRate": _SR,
                    "nTags": len(_TAGS),
                    "noiseFloorDb": round(_NOISE_FLOOR, 2),
                    "defaults": {
                        "fLow": DEFAULT_F_LOW,
                        "fHigh": DEFAULT_F_HIGH,
                        "gainDb": DEFAULT_GAIN_DB,
                        "speechDeltaDb": SPEECH_DELTA_DB,
                        "shortPieceMs": MERGE_BACK_SHORT_PIECE_MS,
                        "maxGapMs": MERGE_BACK_MAX_GAP_MS,
                    },
                    "baselineParams": baseline_pipeline_params(),
                    "baselineReady": _BASELINE is not None,
                    "tags": [
                        {
                            "uuid": t.get("uuid"),
                            "startMs": t.get("startMs", t.get("tMs")),
                            "endMs": t.get("endMs"),
                            "category": t.get("category"),
                            "speaker": t.get("speaker"),
                            "word": t.get("word"),
                            "label": t.get("label"),
                            "source": t.get("source"),
                        }
                        for t in _TAGS
                    ],
                }
            self._send_json(200, payload)
            return
        if parsed.path == "/api/last":
            with _STATE_LOCK:
                payload = _LAST_FIND or {"ok": False, "error": "no find yet"}
            self._send_json(200, payload)
            return
        if parsed.path == "/api/baseline":
            try:
                qs = parse_qs(parsed.query)
                win_params = {
                    "windowStartMs": (qs.get("windowStartMs") or [None])[0],
                    "windowEndMs": (qs.get("windowEndMs") or [None])[0],
                }
                win_start_ms, win_end_ms = _parse_window_ms(win_params)
                payload = ensure_baseline(force=False)
                view = baseline_scored_view(
                    payload,
                    win_start_ms=win_start_ms,
                    win_end_ms=win_end_ms,
                )
                self._send_json(200, view)
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"ok": False, "error": str(exc)})
            return
        if parsed.path == "/audio":
            if _AUDIO_PATH is None or not _AUDIO_PATH.exists():
                self._send(404, b"missing audio", "text/plain")
                return
            data = _AUDIO_PATH.read_bytes()
            ctype = "audio/wav" if _AUDIO_PATH.suffix.lower() == ".wav" else "application/octet-stream"
            self._send(200, data, ctype)
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/baseline/refresh":
            try:
                body = self._read_json()
                win_start_ms, win_end_ms = _parse_window_ms(body)
                # Discard any in-flight warm result by forcing recompute.
                payload = ensure_baseline(force=True)
                view = baseline_scored_view(
                    payload,
                    win_start_ms=win_start_ms,
                    win_end_ms=win_end_ms,
                )
                self._send_json(200, view)
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"ok": False, "error": str(exc)})
            return
        if parsed.path != "/api/find":
            self._send(404, b"not found", "text/plain")
            return
        try:
            params = self._read_json()
            result = run_find(params)
            self._send_json(200, result)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": str(exc)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Snippet explorer lab (band-pass + find knobs)")
    ap.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=f"manifest.sessionName (default {DEFAULT_SESSION})",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"HTTP port (default {DEFAULT_PORT}; review uses 8765)",
    )
    ap.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1)",
    )
    args = ap.parse_args()

    load_kit(args.session)
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    url = f"http://{args.bind}:{args.port}/"
    print(f"Snippet explorer lab at {url}", flush=True)
    print("Ctrl+C to stop. Does not write library files.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
