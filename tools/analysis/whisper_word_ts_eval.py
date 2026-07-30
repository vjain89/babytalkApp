#!/usr/bin/env python3
"""Evaluate faster-whisper word timestamps as word-level candidate boxes.

Compares Whisper word intervals (independent of human tag edges) against
manual verbal tags to answer: are Whisper word boxes good enough for IoU,
or do we still need DJW syllable merge-back?

Method
------
1. Load manual tags with a word label (exclude non-verbal vocalization;
   exclude ``source == ml_confirmed``).
2. Cluster nearby tags into audio windows (gap ≤ ``--gap-ms``), pad each
   window, and run faster-whisper once per window with ``word_timestamps=True``.
3. Map Whisper word times to absolute ms; for each tag pick the best-
   overlapping Whisper word (and optionally a fuzzy text match).
4. Report coverage / IoU / error modes separately for overlap-only vs
   text-match subsets.

Model: ``BABYTALK_WHISPER_MODEL`` or asr_suggest default (``base``).

Usage
-----
    tools/.venv/bin/python tools/analysis/whisper_word_ts_eval.py
    tools/.venv/bin/python tools/analysis/whisper_word_ts_eval.py \\
        --sessions 26_07_27__19:53:00 26_07_05__00:00:00
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from asr_suggest import (  # noqa: E402
    DEFAULT_MODEL,
    LANGUAGE_HINTS,
    get_model,
)

LIBRARY = Path.home() / "Documents" / "BabyTalk" / "Library"
OUT_DIR = Path(__file__).resolve().parent / "out"
NONVERBAL = "non-verbal vocalization"
POINT_TAG_SPAN_MS = 500.0
DEFAULT_SESSIONS = ("26_07_27__19:53:00", "26_07_05__00:00:00")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def quantiles(values: list[float], nd: int = 3) -> dict:
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
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": round(vals[-1], nd),
        "mean": round(statistics.fmean(vals), nd),
    }


def overlap_ms(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = overlap_ms(a, b)
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def span_of(item: dict) -> tuple[float, float] | None:
    start = item.get("startMs")
    if start is None:
        start = item.get("tMs")
    if start is None:
        return None
    end = item.get("endMs")
    if end is None or end <= start:
        end = start + POINT_TAG_SPAN_MS
    return float(start), float(end)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_match(tag_word: str, asr_word: str, *, threshold: float = 0.72) -> bool:
    """Loose text match — Swiss German / baby speech often misspelled."""
    a = normalize_text(tag_word)
    b = normalize_text(asr_word)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        # avoid tiny substring traps (e.g. "a" in "teacher")
        if min(len(a), len(b)) >= 3 or a == b:
            return True
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio >= threshold:
        return True
    # token-wise: multi-word tags vs single whisper token
    a_toks = a.split()
    b_toks = b.split()
    if len(a_toks) > 1 or len(b_toks) > 1:
        for ta in a_toks:
            for tb in b_toks:
                if len(ta) >= 3 and (ta == tb or SequenceMatcher(None, ta, tb).ratio() >= threshold):
                    return True
    return False


def resolve_kit(lib: Path, session: str) -> Path:
    needle = session.strip()
    for kit in sorted(lib.iterdir()):
        man_path = kit / "manifest.json"
        if not man_path.exists():
            continue
        man = load_json(man_path)
        if man.get("sessionName") == needle:
            return kit
        if needle in kit.name or needle in str(man.get("sessionName") or ""):
            return kit
    raise FileNotFoundError(f"No kit with sessionName={needle!r} under {lib}")


def load_eval_tags(kit: Path) -> list[dict]:
    raw = load_json(kit / "tags.json")
    items = raw.get("tags", raw if isinstance(raw, list) else [])
    out = []
    for t in items:
        if t.get("source") == "ml_confirmed":
            continue
        cat = (t.get("category") or "").strip().lower()
        if cat == NONVERBAL:
            continue
        word = (t.get("word") or "").strip()
        if not word:
            continue
        # Prefer verbal; also keep other non-nonverbal with a word label
        sp = span_of(t)
        if not sp:
            continue
        out.append({**t, "_span": sp, "_word": word})
    out.sort(key=lambda t: t["_span"][0])
    return out


def cluster_windows(
    tags: list[dict],
    *,
    gap_ms: float,
    pad_ms: float,
    audio_dur_ms: float,
) -> list[dict]:
    """Merge nearby tags into padded transcription windows."""
    if not tags:
        return []
    clusters: list[list[dict]] = [[tags[0]]]
    for t in tags[1:]:
        prev = clusters[-1][-1]["_span"]
        cur = t["_span"]
        if cur[0] - prev[1] <= gap_ms:
            clusters[-1].append(t)
        else:
            clusters.append([t])

    windows = []
    for i, group in enumerate(clusters):
        lo = min(t["_span"][0] for t in group)
        hi = max(t["_span"][1] for t in group)
        start = max(0.0, lo - pad_ms)
        end = min(audio_dur_ms, hi + pad_ms)
        windows.append(
            {
                "id": i,
                "startMs": start,
                "endMs": end,
                "nTags": len(group),
                "tagUuids": [t.get("uuid") for t in group],
            }
        )
    return windows


def slice_audio_ms(
    audio: np.ndarray, sr: int, start_ms: float, end_ms: float
) -> np.ndarray:
    start = max(0, int(start_ms * sr / 1000))
    end = min(len(audio), int(end_ms * sr / 1000))
    if end <= start:
        raise ValueError("empty slice")
    chunk = np.asarray(audio[start:end], dtype=np.float32)
    peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
    if peak > 1.0:
        chunk = chunk / peak
    return chunk


def transcribe_window_words(
    model,
    audio: np.ndarray,
    sr: int,
    window: dict,
    *,
    language_hint: str | None,
    beam_size: int = 1,
) -> list[dict]:
    """Return absolute-ms Whisper words for one window."""
    chunk = slice_audio_ms(audio, sr, window["startMs"], window["endMs"])
    whisper_lang = None
    if language_hint:
        whisper_lang = LANGUAGE_HINTS.get(language_hint.strip()) or language_hint.strip()
        if len(whisper_lang) > 3:
            whisper_lang = LANGUAGE_HINTS.get(language_hint)

    # Write temp wav — same path as asr_suggest (reliable for faster-whisper).
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(name)
    try:
        sf.write(str(tmp), chunk, sr, subtype="PCM_16")
        segments, info = model.transcribe(
            str(tmp),
            language=whisper_lang,
            beam_size=beam_size,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        words: list[dict] = []
        for seg in segments:
            for w in seg.words or []:
                text = (w.word or "").strip()
                if not text:
                    continue
                # Whisper word times are relative to the slice start.
                abs_start = window["startMs"] + float(w.start) * 1000.0
                abs_end = window["startMs"] + float(w.end) * 1000.0
                if abs_end <= abs_start:
                    abs_end = abs_start + 50.0
                words.append(
                    {
                        "text": text,
                        "startMs": abs_start,
                        "endMs": abs_end,
                        "probability": float(getattr(w, "probability", 0.0) or 0.0),
                        "windowId": window["id"],
                        "detectedLanguage": getattr(info, "language", None),
                    }
                )
        return words
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def classify_error(
    tag: dict,
    overlapping: list[dict],
    best: dict | None,
    text_hit: dict | None,
) -> str:
    """Assign a primary error-mode label for one tag."""
    if not overlapping:
        return "no_speech"
    if best is None:
        return "no_speech"

    n_ov = len(overlapping)
    tag_span = tag["_span"]
    best_span = (best["startMs"], best["endMs"])
    score = iou(tag_span, best_span)
    tag_dur = tag_span[1] - tag_span[0]
    best_dur = best_span[1] - best_span[0]

    if text_hit is None:
        # Overlap but wrong lexeme
        if n_ov >= 2 and score < 0.5:
            return "word_split"
        if best_dur > 1.6 * tag_dur and score < 0.5:
            return "word_merge"
        if score < 0.3:
            return "big_boundary_error"
        return "wrong_word"

    # Text matched (or fuzzy)
    if n_ov >= 3 and score < 0.5:
        return "word_split"
    if best_dur > 1.8 * tag_dur and score < 0.5:
        return "word_merge"
    if score < 0.3:
        return "big_boundary_error"
    if score >= 0.5:
        return "ok"
    return "boundary_soft"  # text ok, IoU in (0.3, 0.5)


def match_tag_to_words(tag: dict, words: list[dict]) -> dict:
    ts = tag["_span"]
    overlapping = []
    for w in words:
        ws = (w["startMs"], w["endMs"])
        ov = overlap_ms(ts, ws)
        if ov <= 0:
            continue
        overlapping.append({**w, "_iou": iou(ts, ws), "_overlap": ov})

    overlapping.sort(key=lambda x: x["_iou"], reverse=True)
    best = overlapping[0] if overlapping else None

    text_candidates = [
        w for w in overlapping if fuzzy_match(tag["_word"], w["text"])
    ]
    # Also search nearby non-overlapping words (±150ms) for text match
    near = []
    for w in words:
        ws = (w["startMs"], w["endMs"])
        if overlap_ms(ts, ws) > 0:
            continue
        gap = max(ts[0] - ws[1], ws[0] - ts[1])
        if gap <= 150 and fuzzy_match(tag["_word"], w["text"]):
            near.append({**w, "_iou": iou(ts, ws), "_overlap": 0.0})
    text_hit = None
    if text_candidates:
        text_hit = max(text_candidates, key=lambda x: x["_iou"])
    elif near:
        text_hit = near[0]

    mode = classify_error(tag, overlapping, best, text_hit)

    return {
        "uuid": tag.get("uuid"),
        "word": tag["_word"],
        "category": tag.get("category"),
        "speaker": tag.get("speaker"),
        "source": tag.get("source"),
        "tagSpan": list(ts),
        "tagDurMs": round(ts[1] - ts[0], 1),
        "nOverlappingWords": len(overlapping),
        "bestOverlap": None
        if best is None
        else {
            "text": best["text"],
            "span": [round(best["startMs"], 1), round(best["endMs"], 1)],
            "iou": round(best["_iou"], 4),
            "probability": best.get("probability"),
        },
        "textMatch": None
        if text_hit is None
        else {
            "text": text_hit["text"],
            "span": [round(text_hit["startMs"], 1), round(text_hit["endMs"], 1)],
            "iou": round(text_hit["_iou"], 4),
            "probability": text_hit.get("probability"),
            "fuzzy": normalize_text(tag["_word"]) != normalize_text(text_hit["text"]),
        },
        "errorMode": mode,
        "overlappingTexts": [w["text"] for w in overlapping[:6]],
    }


def summarize_matches(matches: list[dict]) -> dict:
    n = len(matches)
    any_ov = [m for m in matches if m["nOverlappingWords"] > 0]
    text_m = [m for m in matches if m["textMatch"] is not None]
    # IoU on best-overlap word (boundary quality regardless of spelling)
    ious_ov = [m["bestOverlap"]["iou"] for m in any_ov if m["bestOverlap"]]
    ious_txt = [m["textMatch"]["iou"] for m in text_m if m["textMatch"] and m["textMatch"]["iou"] > 0]
    # For text matches with zero overlap (near-miss), exclude from IoU≥0.5 denom of "matched"
    text_with_ov = [m for m in text_m if m["textMatch"] and m["textMatch"]["iou"] > 0]

    modes = Counter(m["errorMode"] for m in matches)

    def iou50(rows: list[dict], key: str) -> tuple[int, float]:
        hits = 0
        for m in rows:
            block = m.get(key)
            if block and block["iou"] >= 0.5:
                hits += 1
        return hits, pct(hits, n)

    ov50_n, ov50_p = iou50(matches, "bestOverlap")
    tx50_n, tx50_p = iou50(matches, "textMatch")

    return {
        "nTags": n,
        "coverage": {
            "anyOverlapN": len(any_ov),
            "anyOverlapPct": pct(len(any_ov), n),
            "textMatchN": len(text_m),
            "textMatchPct": pct(len(text_m), n),
        },
        "iou": {
            "overlapOnly": {
                "nWithOverlap": len(ious_ov),
                "iou50N": ov50_n,
                "iou50PctOfAllTags": ov50_p,
                "iou50PctOfOverlapping": pct(ov50_n, len(any_ov)),
                "quantiles": quantiles(ious_ov),
            },
            "textMatch": {
                "nWithTextAndOverlap": len(text_with_ov),
                "iou50N": tx50_n,
                "iou50PctOfAllTags": tx50_p,
                "iou50PctOfTextMatches": pct(tx50_n, len(text_m)),
                "quantiles": quantiles(ious_txt),
            },
        },
        "errorModes": dict(modes.most_common()),
        "errorModePct": {k: pct(v, n) for k, v in modes.most_common()},
    }


def compare_archival_ml(kit: Path, tags: list[dict]) -> dict | None:
    """Optional: best-overlap IoU of archival annotations.json vs same tags."""
    ann_path = kit / "annotations.json"
    if not ann_path.exists():
        return None
    raw = load_json(ann_path)
    cands = raw.get("annotations", raw if isinstance(raw, list) else [])
    cspans = []
    for c in cands:
        sp = span_of(c)
        if sp:
            cspans.append(sp)
    if not cspans:
        return None

    ious = []
    any_ov = 0
    for t in tags:
        ts = t["_span"]
        best = 0.0
        hit = False
        for cs in cspans:
            if overlap_ms(ts, cs) > 0:
                hit = True
                best = max(best, iou(ts, cs))
        if hit:
            any_ov += 1
            ious.append(best)
    n = len(tags)
    return {
        "nCands": len(cspans),
        "anyOverlapPct": pct(any_ov, n),
        "iou50Pct": pct(sum(1 for x in ious if x >= 0.5), n),
        "iouQuantiles": quantiles(ious),
        "note": "archival annotations.json vs same manual verbal tags (DJW/ML boxes)",
    }


def evaluate_kit(
    kit: Path,
    *,
    model,
    model_name: str,
    gap_ms: float,
    pad_ms: float,
    language_hint: str | None,
    max_windows: int | None = None,
) -> dict:
    man = load_json(kit / "manifest.json")
    session = man.get("sessionName") or kit.name
    audio_path = kit / man.get("audioFile", "audio.wav")
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    sr = int(sr)
    dur_ms = 1000.0 * len(audio) / sr

    tags = load_eval_tags(kit)
    # Language: CLI hint > first tag language > Swiss German default for these kits
    lang = language_hint
    if not lang:
        for t in tags:
            if t.get("language"):
                lang = t["language"]
                break

    windows = cluster_windows(tags, gap_ms=gap_ms, pad_ms=pad_ms, audio_dur_ms=dur_ms)
    if max_windows is not None:
        windows = windows[:max_windows]

    print(
        f"\n=== {session} ===\n"
        f"  kit={kit.name}\n"
        f"  audio={dur_ms/1000:.1f}s  tags={len(tags)}  windows={len(windows)}  "
        f"lang_hint={lang!r}  model={model_name}",
        flush=True,
    )

    all_words: list[dict] = []
    t0 = time.time()
    for i, win in enumerate(windows):
        wlist = transcribe_window_words(
            model, audio, sr, win, language_hint=lang
        )
        all_words.extend(wlist)
        if (i + 1) % 5 == 0 or i + 1 == len(windows):
            print(
                f"  window {i+1}/{len(windows)}  "
                f"+{len(wlist)} words  total_words={len(all_words)}  "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    matches = [match_tag_to_words(t, all_words) for t in tags]
    summary = summarize_matches(matches)
    ml_cmp = compare_archival_ml(kit, tags)

    # Keep a compact sample of failures for the report
    samples = {
        "no_speech": [],
        "wrong_word": [],
        "word_split": [],
        "word_merge": [],
        "big_boundary_error": [],
        "ok": [],
    }
    for m in matches:
        mode = m["errorMode"]
        bucket = mode if mode in samples else None
        if bucket is None and mode == "boundary_soft":
            bucket = "big_boundary_error"
        if bucket and len(samples[bucket]) < 5:
            samples[bucket].append(
                {
                    "word": m["word"],
                    "tagSpan": m["tagSpan"],
                    "best": m["bestOverlap"],
                    "textMatch": m["textMatch"],
                    "overlappingTexts": m["overlappingTexts"],
                    "errorMode": m["errorMode"],
                }
            )

    return {
        "sessionName": session,
        "kitFolder": kit.name,
        "kitPath": str(kit),
        "audioDurSec": round(dur_ms / 1000.0, 2),
        "model": f"faster-whisper/{model_name}",
        "languageHint": lang,
        "method": {
            "description": (
                "Cluster nearby manual tags into padded windows; "
                "transcribe each with word_timestamps=True; align Whisper "
                "words to tags by overlap (and fuzzy text)."
            ),
            "gapMs": gap_ms,
            "padMs": pad_ms,
            "nWindows": len(windows),
            "nWhisperWords": len(all_words),
            "elapsedSec": round(time.time() - t0, 1),
        },
        "summary": summary,
        "archivalMlCompare": ml_cmp,
        "samples": samples,
        "matches": matches,
    }


def print_table(results: list[dict], pooled: dict) -> None:
    cols = [
        "kit",
        "n",
        "ov%",
        "txt%",
        "IoU50|ov",
        "IoU50|txt",
        "medIoU|ov",
        "no_sp%",
        "wrong%",
        "split%",
        "ML IoU50",
    ]
    print("\n" + " | ".join(cols))
    print("-+-".join("-" * len(c) for c in cols))

    def row(label: str, s: dict, ml: dict | None) -> None:
        iou_ov = s["iou"]["overlapOnly"]
        iou_tx = s["iou"]["textMatch"]
        modes = s["errorModePct"]
        med = (iou_ov["quantiles"] or {}).get("median", "—")
        ml50 = (ml or {}).get("iou50Pct", "—")
        cells = [
            label[:18],
            str(s["nTags"]),
            f"{s['coverage']['anyOverlapPct']}",
            f"{s['coverage']['textMatchPct']}",
            f"{iou_ov['iou50PctOfAllTags']}",
            f"{iou_tx['iou50PctOfAllTags']}",
            f"{med}",
            f"{modes.get('no_speech', 0)}",
            f"{modes.get('wrong_word', 0)}",
            f"{modes.get('word_split', 0)}",
            f"{ml50}",
        ]
        print(" | ".join(cells))

    for r in results:
        row(r["sessionName"], r["summary"], r.get("archivalMlCompare"))
    row("POOLED", pooled, None)
    print()


def write_markdown_fragment(path: Path, payload: dict) -> None:
    pooled = payload["pooled"]
    cov = pooled["coverage"]
    iou_ov = pooled["iou"]["overlapOnly"]
    iou_tx = pooled["iou"]["textMatch"]
    modes = pooled["errorModePct"]
    rec = payload["recommendation"]

    lines = [
        "## Whisper word timestamps vs syllable over-splits",
        "",
        "### Problem (plain language)",
        "",
        "The current ML path (VAD → ECAPA → pause-split → DJW resegment) often",
        "**cuts a single spoken word into syllable-sized boxes** (e.g. `tea|cher`).",
        "Those too-short children fail IoU≥0.5 against human word tags even when",
        "the sound was detected. IoU sweeps showed knob-tuning does not fix this —",
        "the lever is **merge-back** (and/or an external word-boundary signal).",
        "",
        "### What Whisper word timestamps are",
        "",
        "`faster-whisper` can emit not only a transcript string but **per-word",
        f"start/end times** (`word_timestamps=True`). Model used here:",
        f"**{payload['model']}** (override with `BABYTALK_WHISPER_MODEL`;",
        "same default as `tools/asr_suggest.py`).",
        "",
        "We clustered nearby manual tags into padded audio windows, transcribed",
        "each window once, then aligned Whisper word intervals to human tags —",
        "so the boxes are **not** forced to the human edges.",
        "",
        "### Why they could help",
        "",
        "If Whisper’s word intervals land near human word spans, we could use them",
        "as candidate boxes (or as merge targets for DJW children) instead of",
        "trusting syllable cuts alone — even when the **spelling** is wrong for",
        "Swiss German / baby speech.",
        "",
        "### Measured metrics",
        "",
        f"Kits: `{', '.join(payload['sessions'])}`  ",
        f"Manual verbal tags (excl. non-verbal / ml_confirmed): **n={pooled['nTags']}**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Any overlapping Whisper word | **{cov['anyOverlapPct']}%** |",
        f"| Fuzzy text match | **{cov['textMatchPct']}%** |",
        f"| IoU≥0.5 (best overlap word, ignore text) | **{iou_ov['iou50PctOfAllTags']}%** of all tags |",
        f"| IoU≥0.5 among overlapping only | {iou_ov['iou50PctOfOverlapping']}% |",
        f"| Median IoU (overlapping) | {(iou_ov['quantiles'] or {}).get('median', '—')} |",
        f"| IoU≥0.5 (text-matched word) | **{iou_tx['iou50PctOfAllTags']}%** of all tags |",
        f"| no_speech | {modes.get('no_speech', 0)}% |",
        f"| wrong_word | {modes.get('wrong_word', 0)}% |",
        f"| word_split | {modes.get('word_split', 0)}% |",
        f"| word_merge | {modes.get('word_merge', 0)}% |",
        f"| big_boundary_error | {modes.get('big_boundary_error', 0)}% |",
        "",
        "Per-kit detail is in `whisper_word_ts.json`. Archival ML IoU≥0.5 on the",
        "same tags (when annotations.json exists) is included there for context;",
        "DJW baseline from prior iou_sweep was ~53–57% manual IoU≥0.5 on dense kits.",
        "",
        "### Recommendation",
        "",
        f"**{rec['verdict']}**",
        "",
        rec["rationale"],
        "",
        f"_Generated {payload['generatedAt']}_",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def recommend(pooled: dict, per_kit: list[dict]) -> dict:
    cov = pooled["coverage"]
    iou_ov = pooled["iou"]["overlapOnly"]
    iou_tx = pooled["iou"]["textMatch"]
    modes = pooled["errorModePct"]
    ov50 = iou_ov["iou50PctOfAllTags"]
    txt_cov = cov["textMatchPct"]
    any_ov = cov["anyOverlapPct"]

    # Reference: archival / prior DJW ~50–57% IoU≥0.5 on manual tags
    archival_iou = []
    for r in per_kit:
        ml = r.get("archivalMlCompare") or {}
        if "iou50Pct" in ml:
            archival_iou.append(ml["iou50Pct"])
    arch_ref = statistics.fmean(archival_iou) if archival_iou else 55.0

    if ov50 >= 55 and any_ov >= 85 and ov50 >= arch_ref - 5:
        verdict = "Whisper sufficient for word boxes (with caveats)"
        rationale = (
            f"Overlap-based IoU≥0.5 is {ov50}% of tags (median IoU on hits "
            f"{(iou_ov['quantiles'] or {}).get('median')}), comparable to or "
            f"better than archival ML (~{arch_ref:.0f}%). Text match is only "
            f"{txt_cov}% — use Whisper for **boundaries**, not labels. "
            "Still filter empty/no_speech regions; do not replace DJW entirely "
            "without a speech-region proposer."
        )
    elif ov50 >= 40 and any_ov >= 70:
        verdict = "Hybrid: Whisper boundaries + DJW merge-back"
        rationale = (
            f"Whisper places *some* word near {any_ov}% of tags, but IoU≥0.5 "
            f"is only {ov50}% overall (text-match IoU≥0.5 {iou_tx['iou50PctOfAllTags']}%; "
            f"text coverage {txt_cov}%). That is not reliably better than DJW alone "
            f"(~{arch_ref:.0f}% archival IoU≥0.5). Best path: keep DJW children as "
            f"proposals, use Whisper word intervals as **merge targets** when they "
            f"overlap a run of too-short syllables; fall back to acoustic merge-back "
            f"when ASR is silent ({modes.get('no_speech', 0)}% no_speech) or wrong."
        )
    else:
        verdict = "Need DJW merge-back — Whisper not enough alone"
        rationale = (
            f"Coverage {any_ov}% / IoU≥0.5 {ov50}% is too weak to replace syllable "
            f"boxes. Prioritize acoustic merge-back of DJW over-splits; treat Whisper "
            f"as an optional weak prior only where it fires. Dominant modes: "
            f"no_speech={modes.get('no_speech', 0)}%, wrong_word={modes.get('wrong_word', 0)}%, "
            f"word_split={modes.get('word_split', 0)}%."
        )

    return {
        "verdict": verdict,
        "rationale": rationale,
        "thresholdsUsed": {
            "whisperAloneIfIou50Pct": 55,
            "hybridIfIou50Pct": 40,
            "archivalMlIou50Ref": round(arch_ref, 1),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sessions",
        nargs="+",
        default=list(DEFAULT_SESSIONS),
        help="manifest.sessionName values",
    )
    p.add_argument("--library", type=Path, default=LIBRARY)
    p.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model size")
    p.add_argument("--gap-ms", type=float, default=2500.0, help="cluster tags within this gap")
    p.add_argument("--pad-ms", type=float, default=400.0, help="pad each window")
    p.add_argument("--language", default=None, help="force language hint (e.g. de / Swiss German dialect)")
    p.add_argument("--max-windows", type=int, default=None, help="debug: limit windows per kit")
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "whisper_word_ts.json",
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=OUT_DIR / "whisper_word_ts_report_fragment.md",
    )
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_name = os.environ.get("BABYTALK_WHISPER_MODEL", args.model)
    print(f"Loading faster-whisper/{model_name} …", flush=True)
    model = get_model(model_name)

    results = []
    for session in args.sessions:
        kit = resolve_kit(args.library.expanduser(), session)
        results.append(
            evaluate_kit(
                kit,
                model=model,
                model_name=model_name,
                gap_ms=args.gap_ms,
                pad_ms=args.pad_ms,
                language_hint=args.language,
                max_windows=args.max_windows,
            )
        )

    # Pooled summary (recompute from concatenated matches)
    all_matches = []
    for r in results:
        all_matches.extend(r["matches"])
    pooled = summarize_matches(all_matches)

    rec = recommend(pooled, results)
    print_table(results, pooled)
    print(f"RECOMMENDATION: {rec['verdict']}")
    print(rec["rationale"])
    print()

    # Slim JSON for disk: drop full match lists' bulk if huge — keep them for analysis
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": f"faster-whisper/{model_name}",
        "modelDefaultSource": "BABYTALK_WHISPER_MODEL or asr_suggest.DEFAULT_MODEL (=base)",
        "sessions": list(args.sessions),
        "method": {
            "gapMs": args.gap_ms,
            "padMs": args.pad_ms,
            "wordTimestamps": True,
            "exclude": ["non-verbal vocalization", "source=ml_confirmed", "empty word"],
        },
        "problem": (
            "ML/DJW often emits too-short syllable boxes that miss IoU≥0.5 "
            "against human word tags (tea|cher class)."
        ),
        "whatWhisperWordTimestampsAre": (
            "Per-word start/end times from faster-whisper with word_timestamps=True, "
            "aligned from padded tag-cluster windows to absolute session time."
        ),
        "whyTheyCouldHelp": (
            "Independent word-boundary proposals that could replace or merge "
            "syllable over-splits, even when ASR spelling is wrong."
        ),
        "pooled": pooled,
        "recommendation": rec,
        "kits": [
            {
                **{k: v for k, v in r.items() if k != "matches"},
                "nMatches": len(r["matches"]),
                # keep matches for drill-down
                "matches": r["matches"],
            }
            for r in results
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown_fragment(args.md_out, payload)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
